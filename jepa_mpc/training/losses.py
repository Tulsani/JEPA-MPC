from __future__ import annotations

import torch
from torch.nn import functional as F


def multi_horizon_latent_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted latent regression across every recurrent rollout prefix."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    if predictions.ndim < 3:
        raise ValueError("expected [..., horizon, latent_dim] tensors")

    per_horizon = F.smooth_l1_loss(
        predictions,
        targets.detach(),
        reduction="none",
    ).mean(dim=-1)
    horizon_count = predictions.shape[-2]
    if horizon_weights is None:
        horizon_weights = torch.ones(
            horizon_count,
            device=predictions.device,
            dtype=predictions.dtype,
        )
    if horizon_weights.shape != (horizon_count,):
        raise ValueError("horizon_weights must have shape [horizon]")
    normalized_weights = horizon_weights / horizon_weights.sum().clamp_min(1e-8)
    return (per_horizon * normalized_weights).sum(dim=-1).mean()
