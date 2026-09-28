"""Batched cross-entropy method, independent of any world model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

CostFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CEMResult:
    best: torch.Tensor  # [B, H, A] lowest-cost sequence seen
    best_cost: torch.Tensor  # [B]
    mean: torch.Tensor  # [B, H, A] final sampling mean (for warm starts)
    cost_history: torch.Tensor  # [iterations, B] best cost per iteration


def cem(
    cost_fn: CostFn,
    batch: int,
    horizon: int,
    dim: int,
    num_samples: int = 300,
    iterations: int = 10,
    elites: int = 30,
    init_mean: torch.Tensor | None = None,
    init_std: float = 1.0,
    bound: float | None = 3.0,
    min_std: float = 0.05,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> CEMResult:
    """Minimize ``cost_fn([B, N, H, A]) -> [B, N]`` over action sequences.

    Samples live in a normalized space; ``bound`` clamps them (e.g. ±3σ of the
    training action distribution, which keeps candidates in-distribution).
    """
    if elites > num_samples:
        raise ValueError("elites cannot exceed num_samples")
    mean = (
        torch.zeros(batch, horizon, dim, device=device)
        if init_mean is None
        else init_mean.to(device).clone()
    )
    std = torch.full_like(mean, init_std)
    best = mean.clone()
    best_cost = torch.full((batch,), float("inf"), device=device)
    history = []
    batch_index = torch.arange(batch, device=device)[:, None]
    for _ in range(iterations):
        noise = torch.randn(batch, num_samples, horizon, dim, device=device, generator=generator)
        candidates = mean[:, None] + std[:, None] * noise
        candidates[:, 0] = mean  # always re-evaluate the current mean
        if bound is not None:
            candidates = candidates.clamp(-bound, bound)
        costs = cost_fn(candidates)
        if costs.shape != (batch, num_samples):
            raise ValueError(f"cost_fn must return [{batch}, {num_samples}], got {tuple(costs.shape)}")
        elite_cost, elite_index = costs.topk(elites, dim=1, largest=False)
        elite = candidates[batch_index, elite_index]
        mean = elite.mean(dim=1)
        std = elite.std(dim=1, unbiased=False).clamp_min(min_std)
        improved = elite_cost[:, 0] < best_cost
        best = torch.where(improved[:, None, None], elite[:, 0], best)
        best_cost = torch.minimum(best_cost, elite_cost[:, 0])
        history.append(best_cost.clone())
    return CEMResult(best=best, best_cost=best_cost, mean=mean, cost_history=torch.stack(history))
