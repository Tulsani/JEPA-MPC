from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dynamics import ActionConditionedTransition


@dataclass(frozen=True)
class LatentRollout:
    latents: torch.Tensor
    final_hidden: torch.Tensor


class LatentWorldModel(nn.Module):
    """Environment-independent encoder and action-conditioned latent dynamics."""

    def __init__(
        self,
        encoder: nn.Module,
        dynamics: ActionConditionedTransition,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.repr_dim = dynamics.repr_dim

    def encode(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(image, proprio)

    def rollout(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
        include_initial: bool = False,
    ) -> LatentRollout:
        """Roll latent dynamics across a complete candidate action sequence.

        Args:
            initial_latent: ``[..., D]`` latent state.
            actions: ``[..., H, A]`` action sequence with matching leading dims.
            hidden: optional ``[..., hidden_dim]`` recurrent state.
            include_initial: prepend the initial latent to the returned sequence.
        """
        if actions.ndim < 2:
            raise ValueError("actions must include horizon and action dimensions")
        if initial_latent.shape[:-1] != actions.shape[:-2]:
            raise ValueError("initial_latent and actions leading dimensions must match")

        current = initial_latent
        rollout_latents: list[torch.Tensor] = [current] if include_initial else []
        current_hidden = hidden
        for step in range(actions.shape[-2]):
            transition = self.dynamics(current, actions[..., step, :], current_hidden)
            current = transition.latent
            current_hidden = transition.hidden
            rollout_latents.append(current)

        if not rollout_latents:
            raise ValueError("cannot roll out an empty action sequence")
        if current_hidden is None:
            raise RuntimeError("dynamics did not produce a hidden state")
        return LatentRollout(
            latents=torch.stack(rollout_latents, dim=-2),
            final_hidden=current_hidden,
        )
