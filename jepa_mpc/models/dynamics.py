from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransitionState:
    latent: torch.Tensor
    hidden: torch.Tensor


class ActionConditionedTransition(nn.Module):
    """A recurrent latent transition with explicit hidden-state propagation.

    Passing hidden state explicitly makes the recurrence unambiguous and lets a
    planner carry separate recurrent state for every sampled action sequence.
    Set ``residual=True`` to predict a latent displacement instead of replacing
    the previous latent directly. Set ``output_norm=False`` when targets come
    from a frozen encoder whose latents are not layer-normalized (e.g. LeWM).
    """

    def __init__(
        self,
        repr_dim: int = 256,
        action_dim: int = 2,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
        residual: bool = True,
        output_norm: bool = True,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.residual = residual

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU(),
        )
        self.recurrent_cell = nn.GRUCell(repr_dim + action_embed_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, repr_dim),
        )
        self.output_norm: nn.Module = nn.LayerNorm(repr_dim) if output_norm else nn.Identity()

    def initial_hidden(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.new_zeros((*latent.shape[:-1], self.hidden_dim))

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> TransitionState:
        if latent.shape[-1] != self.repr_dim:
            raise ValueError(f"expected latent dimension {self.repr_dim}")
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"expected action dimension {self.action_dim}")
        if latent.shape[:-1] != action.shape[:-1]:
            raise ValueError("latent and action leading dimensions must match")

        if hidden is None:
            hidden = self.initial_hidden(latent)

        leading_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, self.repr_dim)
        flat_action = action.reshape(-1, self.action_dim)
        flat_hidden = hidden.reshape(-1, self.hidden_dim)

        action_embedding = self.action_encoder(flat_action)
        next_hidden = self.recurrent_cell(
            torch.cat((flat_latent, action_embedding), dim=-1),
            flat_hidden,
        )
        predicted = self.delta_head(next_hidden)
        if self.residual:
            predicted = flat_latent + predicted
        predicted = self.output_norm(predicted)

        return TransitionState(
            latent=predicted.reshape(*leading_shape, self.repr_dim),
            hidden=next_hidden.reshape(*leading_shape, self.hidden_dim),
        )
