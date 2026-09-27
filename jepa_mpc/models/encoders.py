from __future__ import annotations

import torch
from torch import nn


class ConvObservationEncoder(nn.Module):
    """Resolution-independent image encoder with optional proprioception.

    This replaces the wall-specific two-stream encoder for new environments.
    Wall observations can still be passed as two-channel images, while Push-T
    can use RGB images plus the two-dimensional agent position.
    """

    def __init__(
        self,
        in_channels: int,
        repr_dim: int = 256,
        proprio_dim: int = 0,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if proprio_dim < 0:
            raise ValueError("proprio_dim cannot be negative")

        channels = [in_channels, base_channels, base_channels * 2, base_channels * 4]
        blocks: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1),
                    nn.GroupNorm(8, output_channels),
                    nn.SiLU(),
                ]
            )
        self.visual_encoder = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        visual_dim = channels[-1]
        self.proprio_dim = proprio_dim
        proprio_feature_dim = 0
        if proprio_dim:
            proprio_feature_dim = base_channels
            self.proprio_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(proprio_dim, base_channels),
                nn.LayerNorm(base_channels),
                nn.SiLU(),
            )
        else:
            self.proprio_encoder = None

        self.projector = nn.Sequential(
            nn.Linear(visual_dim + proprio_feature_dim, repr_dim),
            nn.LayerNorm(repr_dim),
        )
        self.repr_dim = repr_dim

    def forward(
        self,
        image: torch.Tensor,
        proprio: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"expected image [B, C, H, W], received {tuple(image.shape)}")
        visual = self.visual_encoder(image)

        if self.proprio_encoder is None:
            if proprio is not None:
                raise ValueError("proprio was provided to an encoder configured without it")
            features = visual
        else:
            if proprio is None:
                raise ValueError("proprio is required by this encoder")
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(
                    f"expected proprio dimension {self.proprio_dim}, received {proprio.shape[-1]}"
                )
            features = torch.cat((visual, self.proprio_encoder(proprio)), dim=-1)

        return self.projector(features)
