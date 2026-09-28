"""Losses for the two-clock world model.

Stage ``fast`` trains the fast clock alone. Stage ``slow`` freezes it and trains
the boundary gate, macro-action encoder, and slow transition. Baselines only
change the segment distribution ``p[s, j]`` (probability that a segment
starting at block ``s`` ends after ``j`` blocks):

* ``learned``: stick-breaking over the gate's hazards (our method);
* ``fixed``:   one-hot at ``j = K`` (fixed stride, FF-JEPA / Hi-LeWM style);
* ``random``:  uniform over ``1..J`` (random waypoints, HWM style).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from jepa_mpc.models.two_clock import TwoClockWorldModel

SEGMENT_MODES = ("learned", "fixed", "random")


def per_step_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).square().mean(dim=-1)


def fast_losses(
    model: TwoClockWorldModel,
    latents: torch.Tensor,
    actions: torch.Tensor,
    teacher_weight: float = 1.0,
    open_loop_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One-step teacher-forced loss plus open-loop multi-horizon loss."""
    targets = latents[:, 1:]
    filtered = model.fast_filter(latents, actions)
    teacher = per_step_mse(filtered.predictions, targets).mean()
    rollout = model.fast_rollout(latents[:, 0], actions).latents
    per_horizon = per_step_mse(rollout, targets).mean(dim=0)  # [T]
    open_loop = per_horizon.mean()
    loss = teacher_weight * teacher + open_loop_weight * open_loop
    metrics = {"fast/teacher": teacher.detach(), "fast/open_loop": open_loop.detach()}
    for step in (1, 2, 4, 8, 16):
        if step <= per_horizon.numel():
            metrics[f"fast/open_loop_h{step}"] = per_horizon[step - 1].detach()
    return loss, metrics


def stick_breaking(hazard_logits: torch.Tensor) -> torch.Tensor:
    """``[..., J]`` hazard logits → segment-end distribution; forced end at J."""
    hazard = torch.sigmoid(hazard_logits)
    hazard = torch.cat((hazard[..., :-1], torch.ones_like(hazard[..., -1:])), dim=-1)
    survive = torch.cumprod(1.0 - hazard, dim=-1)
    survive_before = torch.cat((torch.ones_like(survive[..., :1]), survive[..., :-1]), dim=-1)
    return hazard * survive_before


def segment_distribution(
    mode: str,
    hazard_logits: torch.Tensor,
    fixed_length: int | None = None,
) -> torch.Tensor:
    max_segment = hazard_logits.shape[-1]
    if mode == "learned":
        return stick_breaking(hazard_logits)
    if mode == "fixed":
        if fixed_length is None or not 1 <= fixed_length <= max_segment:
            raise ValueError("fixed mode needs 1 <= fixed_length <= max_segment")
        return F.one_hot(
            torch.full(hazard_logits.shape[:-1], fixed_length - 1, device=hazard_logits.device),
            max_segment,
        ).to(hazard_logits.dtype)
    if mode == "random":
        return torch.full_like(hazard_logits, 1.0 / max_segment)
    raise ValueError(f"unknown segment mode {mode!r}; expected one of {SEGMENT_MODES}")


def gate_objective(
    distribution: torch.Tensor, error: torch.Tensor, boundary_cost: float
) -> torch.Tensor:
    """Expected (normalized error − length bonus) under the stopping distribution.

    Errors are divided by their mean so ``boundary_cost`` is scale-free: the gate
    keeps extending a segment while the slow jump's error grows by less than
    ``boundary_cost`` mean-errors per ``max_segment`` blocks.
    """
    max_segment = error.shape[-1]
    normalized_error = error.detach() / error.detach().mean().clamp_min(1e-8)
    lengths = torch.arange(1, max_segment + 1, device=error.device, dtype=error.dtype)
    length_bonus = boundary_cost * lengths / max_segment
    return (distribution * (normalized_error - length_bonus)).sum(-1).mean()


@dataclass(frozen=True)
class SegmentBatch:
    """All (start s, length j) segment pairs inside a window."""

    start_latent: torch.Tensor  # [B, S, D]
    end_latent: torch.Tensor  # [B, S, J, D]
    end_hidden: torch.Tensor  # [B, S, J, H]
    actions: torch.Tensor  # [B, S, J, A]
    lengths: torch.Tensor  # [J] = 1..J


def gather_segments(
    latents: torch.Tensor, actions: torch.Tensor, hiddens: torch.Tensor, max_segment: int
) -> SegmentBatch:
    window = actions.shape[1]
    num_starts = window - max_segment + 1
    if num_starts < 1:
        raise ValueError("window must be at least max_segment blocks long")
    device = latents.device
    starts = torch.arange(num_starts, device=device)
    lengths = torch.arange(1, max_segment + 1, device=device)
    end_index = starts[:, None] + lengths[None, :]  # [S, J]
    action_index = starts[:, None] + lengths[None, :] - 1  # actions s..s+J-1
    return SegmentBatch(
        start_latent=latents[:, starts],
        end_latent=latents[:, end_index],
        end_hidden=hiddens[:, end_index],
        actions=actions[:, action_index],
        lengths=lengths,
    )


def slow_losses(
    model: TwoClockWorldModel,
    latents: torch.Tensor,
    actions: torch.Tensor,
    mode: str,
    fixed_length: int | None = None,
    boundary_cost: float = 0.5,
    uniform_weight: float = 0.1,
    macro_kl_weight: float = 1e-3,
    duration_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Gate + macro-encoder + slow-transition losses with a frozen fast clock."""
    max_segment = model.max_segment
    with torch.no_grad():
        hiddens = model.fast_filter(latents, actions).hiddens
    segments = gather_segments(latents, actions, hiddens, max_segment)
    batch, num_starts = segments.start_latent.shape[:2]

    start = segments.start_latent[:, :, None, :].expand_as(segments.end_latent)
    elapsed = segments.lengths.expand(batch, num_starts, max_segment)
    hazard_logits = model.boundary_logits(segments.end_hidden, segments.end_latent, start, elapsed)
    distribution = segment_distribution(mode, hazard_logits, fixed_length)  # [B, S, J]

    macro = model.encode_macro(segments.actions)  # prefix j encodes actions s..s+j-1
    predicted_end, duration_logits = model.slow_jump(start, macro.sample())
    error = per_step_mse(predicted_end, segments.end_latent)  # [B, S, J]

    # Slow model: focus on chosen segments but stay accurate for every length.
    weights = distribution.detach() + uniform_weight / max_segment
    weights = weights / weights.sum(dim=-1, keepdim=True)
    slow = (weights * error).sum(-1).mean()
    duration_target = (segments.lengths - 1).expand(batch, num_starts, max_segment)
    duration_ce = F.cross_entropy(
        duration_logits.reshape(-1, max_segment),
        duration_target.reshape(-1),
        reduction="none",
    ).reshape(batch, num_starts, max_segment)
    duration = (weights * duration_ce).sum(-1).mean()
    macro_kl = macro.kl_to_standard_normal().mean()

    loss = slow + duration_weight * duration + macro_kl_weight * macro_kl
    metrics: dict[str, torch.Tensor] = {
        "slow/error": slow.detach(),
        "slow/duration_ce": duration.detach(),
        "slow/macro_kl": macro_kl.detach(),
    }

    if mode == "learned":
        gate = gate_objective(distribution, error, boundary_cost)
        loss = loss + gate
        metrics["gate/objective"] = gate.detach()

    lengths = segments.lengths.to(error.dtype)
    metrics["segments/mean_length"] = (distribution.detach() * lengths).sum(-1).mean()
    metrics["segments/hazard"] = torch.sigmoid(hazard_logits.detach()).mean()
    per_length = error.detach().mean(dim=(0, 1))
    for length in (1, max_segment // 2, max_segment):
        metrics[f"slow/error_j{length}"] = per_length[length - 1]
    return loss, metrics
