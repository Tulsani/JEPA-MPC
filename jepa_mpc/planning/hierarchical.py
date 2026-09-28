"""Closed-loop two-clock planner with persistent latent subgoals.

Every call to :meth:`TwoClockPlanner.act` corresponds to one action block
(one fast-clock tick). The planner

1. advances each env's fast-clock hidden state with the block it just executed
   and the latent it now observes (so the gate sees real history);
2. decides per env whether the active subgoal is finished:
   * ``flat``:    no subgoals, the goal latent is the target;
   * ``learned``: the boundary gate fires on the real latent (our method);
   * ``fixed``:   every ``fixed_length`` blocks (FF-JEPA / Hi-LeWM style);
   * ``duration``: after the slow clock's predicted duration (HWM style);
   hierarchical modes also switch when the subgoal is reached or ``max_segment``
   blocks have elapsed;
3. for switching envs, plans macro-actions with CEM over the slow clock toward
   the goal and keeps the first predicted boundary latent as the new subgoal;
4. plans primitive blocks with CEM over the fast clock toward the target,
   scoring prefixes up to a horizon cap (the remaining subgoal duration);
5. returns the first block of the best sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jepa_mpc.models.two_clock import TwoClockWorldModel

from .cem import cem

PLANNER_MODES = ("flat", "learned", "fixed", "duration")


@dataclass
class PlannerConfig:
    mode: str = "learned"
    fixed_length: int = 5
    low_horizon: int = 5  # fast-clock planning horizon (blocks)
    high_jumps: int = 3  # slow-clock planning depth (subgoals)
    cost_mode: str = "prefix_min"  # "terminal" | "prefix_min"
    low_samples: int = 300
    low_iterations: int = 10
    low_elites: int = 30
    high_samples: int = 300
    high_iterations: int = 10
    high_elites: int = 30
    action_bound: float = 3.0  # in normalized (z-scored) action units
    macro_bound: float = 3.0  # macro prior is N(0, I)
    gate_threshold: float = 0.5
    reach_fraction: float = 0.1  # subgoal reached when error < fraction of initial error
    jump_penalty: float = 0.0  # per extra high-level jump, in goal-cost units

    def __post_init__(self) -> None:
        if self.mode not in PLANNER_MODES:
            raise ValueError(f"mode must be one of {PLANNER_MODES}")
        if self.cost_mode not in ("terminal", "prefix_min"):
            raise ValueError("cost_mode must be 'terminal' or 'prefix_min'")


def latent_cost(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean(dim=-1)


class TwoClockPlanner:
    def __init__(
        self,
        model: TwoClockWorldModel,
        config: PlannerConfig,
        num_envs: int,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        self.model = model.eval()
        self.config = config
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.num_envs = num_envs
        self.reset()

    # ------------------------------------------------------------ state
    def reset(self, env_ids: list[int] | None = None) -> None:
        model, n, device = self.model, self.num_envs, self.device
        if env_ids is None:
            self.hidden = torch.zeros(n, model.fast.hidden_dim, device=device)
            self.prev_latent = torch.zeros(n, model.repr_dim, device=device)
            self.prev_action = torch.zeros(n, model.action_dim, device=device)
            self.has_prev = torch.zeros(n, dtype=torch.bool, device=device)
            self.subgoal = torch.zeros(n, model.repr_dim, device=device)
            self.segment_start = torch.zeros(n, model.repr_dim, device=device)
            self.has_subgoal = torch.zeros(n, dtype=torch.bool, device=device)
            self.elapsed = torch.zeros(n, dtype=torch.long, device=device)
            self.duration = torch.ones(n, dtype=torch.long, device=device)
            self.warm_mean = torch.zeros(n, self.config.low_horizon, model.action_dim, device=device)
            return
        ids = torch.as_tensor(env_ids, device=device, dtype=torch.long)
        for tensor in (self.hidden, self.prev_latent, self.prev_action, self.subgoal,
                       self.segment_start, self.warm_mean):
            tensor[ids] = 0
        self.has_prev[ids] = False
        self.has_subgoal[ids] = False
        self.elapsed[ids] = 0
        self.duration[ids] = 1

    # ------------------------------------------------------------ planning
    @torch.no_grad()
    def _plan_subgoals(self, latent: torch.Tensor, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """High-level CEM: returns (first subgoal [e, D], its duration [e])."""
        config, model = self.config, self.model
        count = latent.shape[0]

        def cost_fn(macros: torch.Tensor) -> torch.Tensor:
            samples = macros.shape[1]
            start = latent[:, None].expand(count, samples, -1)
            predicted, _ = model.slow_rollout(start, macros)  # [e, N, K, D]
            costs = latent_cost(predicted, goal[:, None, None])
            costs = costs + config.jump_penalty * torch.arange(
                config.high_jumps, device=costs.device, dtype=costs.dtype
            )
            return costs.min(dim=-1).values  # reaching the goal early is allowed

        result = cem(
            cost_fn, count, config.high_jumps, model.macro_dim,
            num_samples=config.high_samples, iterations=config.high_iterations,
            elites=config.high_elites, bound=config.macro_bound, device=self.device,
            generator=self.generator,
        )
        subgoal, duration_logits = model.slow_jump(latent, result.best[:, 0])
        return subgoal, duration_logits.argmax(dim=-1) + 1

    @torch.no_grad()
    def _plan_blocks(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        target: torch.Tensor,
        cap: torch.Tensor,
        warm_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Low-level CEM: returns (best sequence [e, H, A], chosen horizon [e], mean)."""
        config, model = self.config, self.model
        count, horizon = latent.shape[0], config.low_horizon
        steps = torch.arange(1, horizon + 1, device=self.device)
        valid = steps[None, :] <= cap[:, None]  # [e, H]

        def prefix_costs(blocks: torch.Tensor) -> torch.Tensor:
            samples = blocks.shape[1]
            start = latent[:, None].expand(count, samples, -1)
            start_hidden = hidden[:, None].expand(count, samples, -1)
            rollout = model.fast_rollout(start, blocks, start_hidden).latents  # [e, N, H, D]
            return latent_cost(rollout, target[:, None, None])  # [e, N, H]

        def cost_fn(blocks: torch.Tensor) -> torch.Tensor:
            costs = prefix_costs(blocks)
            if config.cost_mode == "terminal":
                index = (cap - 1).clamp(0, horizon - 1)[:, None, None].expand(-1, costs.shape[1], 1)
                return costs.gather(-1, index).squeeze(-1)
            return costs.masked_fill(~valid[:, None], float("inf")).min(dim=-1).values

        result = cem(
            cost_fn, count, horizon, model.action_dim,
            num_samples=config.low_samples, iterations=config.low_iterations,
            elites=config.low_elites, init_mean=warm_mean, bound=config.action_bound,
            device=self.device, generator=self.generator,
        )
        best_costs = prefix_costs(result.best[:, None])[:, 0]  # [e, H]
        if config.cost_mode == "terminal":
            chosen = cap.clamp(1, horizon)
        else:
            chosen = best_costs.masked_fill(~valid, float("inf")).argmin(dim=-1) + 1
        return result.best, chosen, result.mean

    # ------------------------------------------------------------ main step
    @torch.no_grad()
    def act(
        self,
        latent: torch.Tensor,
        goal: torch.Tensor,
        env_ids: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One block for each env in ``env_ids``. Actions are normalized."""
        config, model = self.config, self.model
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else torch.as_tensor(
            env_ids, device=self.device, dtype=torch.long
        )
        latent, goal = latent.to(self.device), goal.to(self.device)

        # 1. advance the fast clock over the block that was just executed
        hidden = self.hidden[ids]
        has_prev = self.has_prev[ids]
        _, stepped = model.fast_step(self.prev_latent[ids], self.prev_action[ids], hidden)
        hidden = torch.where(has_prev[:, None], stepped, hidden)
        elapsed = self.elapsed[ids] + has_prev.long() * self.has_subgoal[ids].long()

        # 2. subgoal bookkeeping
        hazard = torch.full((len(ids),), float("nan"), device=self.device)
        if config.mode == "flat":
            switch = torch.zeros(len(ids), dtype=torch.bool, device=self.device)
            target = goal
            cap = torch.full((len(ids),), config.low_horizon, device=self.device, dtype=torch.long)
        else:
            has_subgoal = self.has_subgoal[ids]
            subgoal, start = self.subgoal[ids], self.segment_start[ids]
            initial_error = latent_cost(start, subgoal).clamp_min(1e-8)
            reached = latent_cost(latent, subgoal) < config.reach_fraction * initial_error
            timeout = elapsed >= model.max_segment
            if config.mode == "learned":
                hazard = torch.sigmoid(
                    model.boundary_logits(hidden, latent, start, elapsed.clamp(min=1))
                )
                finished = hazard > config.gate_threshold
            elif config.mode == "fixed":
                finished = elapsed >= config.fixed_length
            else:  # duration
                finished = elapsed >= self.duration[ids]
            switch = ~has_subgoal | ((finished | reached | timeout) & (elapsed > 0))
            if switch.any():
                new_goal, new_duration = self._plan_subgoals(latent[switch], goal[switch])
                subgoal = subgoal.clone()
                start = start.clone()
                subgoal[switch] = new_goal
                start[switch] = latent[switch]
                elapsed = torch.where(switch, torch.zeros_like(elapsed), elapsed)
                duration = self.duration[ids].clone()
                duration[switch] = new_duration
                self.duration[ids] = duration
                self.subgoal[ids] = subgoal
                self.segment_start[ids] = start
                self.has_subgoal[ids] = True
            target = subgoal
            remaining = self.duration[ids] if config.mode != "fixed" else torch.full_like(elapsed, config.fixed_length)
            cap = (remaining - elapsed).clamp(1, config.low_horizon)

        # 3. low-level planning toward the target
        best, chosen, mean = self._plan_blocks(latent, hidden, target, cap, self.warm_mean[ids])
        action = best[:, 0]

        # 4. commit state
        self.hidden[ids] = hidden
        self.prev_latent[ids] = latent
        self.prev_action[ids] = action
        self.has_prev[ids] = True
        self.elapsed[ids] = elapsed
        self.warm_mean[ids] = torch.cat((mean[:, 1:], torch.zeros_like(mean[:, :1])), dim=1)
        info = {
            "switched": switch,
            "elapsed": elapsed,
            "horizon_cap": cap,
            "chosen_horizon": chosen,
            "hazard": hazard,
            "target_distance": latent_cost(latent, target),
            "goal_distance": latent_cost(latent, goal),
        }
        return action, info
