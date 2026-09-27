from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from jepa_mpc.models.world_model import LatentWorldModel


@dataclass(frozen=True)
class PrefixSelection:
    candidate_index: torch.Tensor
    horizon: torch.Tensor
    cost: torch.Tensor
    action_sequence: torch.Tensor
    terminal_latent: torch.Tensor


class RolloutObjective(Protocol):
    def __call__(
        self,
        latents: torch.Tensor,
        goal: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor: ...


class GoalDistanceObjective:
    """Scores every rollout prefix using goal distance and optional risk."""

    def __init__(
        self,
        uncertainty_weight: float = 0.0,
        duration_weight: float = 0.0,
    ) -> None:
        self.uncertainty_weight = uncertainty_weight
        self.duration_weight = duration_weight

    def __call__(
        self,
        latents: torch.Tensor,
        goal: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # latents: [B, N, H, D], goal: [B, D] or [B, 1, D]
        if latents.ndim != 4:
            raise ValueError("latents must have shape [B, candidates, horizon, latent_dim]")
        if goal.ndim == 2:
            goal = goal[:, None, None, :]
        elif goal.ndim == 3:
            goal = goal[:, :, None, :]
        else:
            raise ValueError("goal must have shape [B, D] or [B, 1, D]")

        cost = torch.linalg.vector_norm(latents - goal, dim=-1)
        if uncertainty is not None:
            if uncertainty.shape != cost.shape:
                raise ValueError("uncertainty must have shape [B, candidates, horizon]")
            cost = cost + self.uncertainty_weight * uncertainty

        if self.duration_weight:
            duration = torch.arange(
                1,
                latents.shape[-2] + 1,
                device=latents.device,
                dtype=latents.dtype,
            )
            duration = duration / latents.shape[-2]
            cost = cost + self.duration_weight * duration
        return cost


def select_best_prefix(
    costs: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    min_horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Jointly select a candidate sequence and effective rollout horizon."""
    if costs.ndim != 3:
        raise ValueError("costs must have shape [B, candidates, horizon]")
    if not 1 <= min_horizon <= costs.shape[-1]:
        raise ValueError("min_horizon must lie inside the available rollout")

    selectable = costs.clone()
    selectable[..., : min_horizon - 1] = torch.inf
    if valid_mask is not None:
        if valid_mask.shape != costs.shape:
            raise ValueError("valid_mask must match costs")
        selectable = selectable.masked_fill(~valid_mask, torch.inf)

    flat_costs = selectable.flatten(start_dim=1)
    best_cost, flat_index = flat_costs.min(dim=1)
    if torch.isinf(best_cost).any():
        raise ValueError("at least one batch item has no valid rollout prefix")
    horizon_count = costs.shape[-1]
    candidate_index = torch.div(flat_index, horizon_count, rounding_mode="floor")
    horizon = flat_index.remainder(horizon_count) + 1
    return candidate_index, horizon, best_cost


class AdaptiveHorizonPlanner:
    """Evaluate sampled action sequences and choose the best prefix.

    Candidate generation (CEM, MPPI, or another sampler) remains separate from
    this class. This makes adaptive-horizon selection independently testable.
    """

    def __init__(
        self,
        world_model: LatentWorldModel,
        objective: RolloutObjective,
        min_horizon: int = 1,
    ) -> None:
        self.world_model = world_model
        self.objective = objective
        self.min_horizon = min_horizon

    def evaluate_candidates(
        self,
        initial_latent: torch.Tensor,
        candidate_actions: torch.Tensor,
        goal: torch.Tensor,
        uncertainty: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> PrefixSelection:
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must have shape [B, N, H, A]")
        batch_size, candidate_count, max_horizon, _ = candidate_actions.shape
        if initial_latent.shape[0] != batch_size:
            raise ValueError("initial_latent batch does not match candidates")

        expanded_initial = initial_latent[:, None, :].expand(
            batch_size, candidate_count, initial_latent.shape[-1]
        )
        rollout = self.world_model.rollout(expanded_initial, candidate_actions)
        costs = self.objective(rollout.latents, goal, uncertainty)
        candidate_index, horizon, best_cost = select_best_prefix(
            costs,
            valid_mask=valid_mask,
            min_horizon=self.min_horizon,
        )

        batch_index = torch.arange(batch_size, device=candidate_actions.device)
        terminal_latent = rollout.latents[
            batch_index,
            candidate_index,
            horizon - 1,
        ]
        selected_actions = candidate_actions[batch_index, candidate_index]
        prefix_mask = (
            torch.arange(max_horizon, device=candidate_actions.device)[None, :]
            < horizon[:, None]
        )
        selected_actions = selected_actions * prefix_mask[..., None]
        return PrefixSelection(
            candidate_index=candidate_index,
            horizon=horizon,
            cost=best_cost,
            action_sequence=selected_actions,
            terminal_latent=terminal_latent,
        )
