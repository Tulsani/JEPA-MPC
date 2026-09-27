from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from jepa_mpc.training.ema import update_ema
from jepa_mpc.training.losses import multi_horizon_latent_loss

from .world_model import LatentWorldModel


@dataclass(frozen=True)
class JEPAPredictions:
    predicted: torch.Tensor
    target: torch.Tensor


class MultiHorizonJEPA(nn.Module):
    """Online world model paired with a frozen EMA target encoder."""

    def __init__(
        self,
        world_model: LatentWorldModel,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        self.target_encoder = deepcopy(world_model.encoder)
        self.ema_decay = ema_decay
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "MultiHorizonJEPA":
        super().train(mode)
        # The target encoder must never update running statistics from batches.
        self.target_encoder.eval()
        return self

    @staticmethod
    def _flatten_time(tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        batch_size, time_steps = tensor.shape[:2]
        return tensor.reshape(batch_size * time_steps, *tensor.shape[2:]), (
            batch_size,
            time_steps,
        )

    def forward(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> JEPAPredictions:
        """Predict each future latent from the first observation.

        Args:
            images: ``[B, H+1, C, height, width]``.
            actions: ``[B, H, action_dim]``.
            proprio: optional ``[B, H+1, proprio_dim]``.
        """
        if images.ndim != 5:
            raise ValueError("images must have shape [B, H+1, C, height, width]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [B, H, action_dim]")
        if images.shape[0] != actions.shape[0] or images.shape[1] != actions.shape[1] + 1:
            raise ValueError("images must contain exactly one more timestep than actions")
        if proprio is not None and proprio.shape[:2] != images.shape[:2]:
            raise ValueError("proprio must align with image timesteps")

        initial_proprio = None if proprio is None else proprio[:, 0]
        initial_latent = self.world_model.encode(images[:, 0], initial_proprio)
        predicted = self.world_model.rollout(initial_latent, actions).latents

        future_images, (batch_size, horizon) = self._flatten_time(images[:, 1:])
        future_proprio = None
        if proprio is not None:
            future_proprio, _ = self._flatten_time(proprio[:, 1:])
        with torch.no_grad():
            target = self.target_encoder(future_images, future_proprio)
        target = target.reshape(batch_size, horizon, -1)
        return JEPAPredictions(predicted=predicted, target=target)

    def loss(
        self,
        images: torch.Tensor,
        actions: torch.Tensor,
        proprio: torch.Tensor | None = None,
        horizon_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self(images, actions, proprio)
        return multi_horizon_latent_loss(
            outputs.predicted,
            outputs.target,
            horizon_weights=horizon_weights,
        )

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        update_ema(self.world_model.encoder, self.target_encoder, self.ema_decay)
