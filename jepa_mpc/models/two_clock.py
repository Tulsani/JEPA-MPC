"""Two-clock recurrent world model with a learned subgoal boundary gate.

Fast clock: ``ActionConditionedTransition`` steps once per action block.
Boundary gate: reads the fast clock's recurrent state and decides whether the
current segment (subgoal) is finished.
Slow clock: ticks only at boundaries. It jumps from the segment's start latent
directly to its end latent, conditioned on a low-dimensional macro-action that
summarizes the segment's primitive actions, and predicts the segment duration.

All modules operate on latent vectors, so the same model trains on cached
frozen-encoder embeddings or on an online encoder's output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .dynamics import ActionConditionedTransition


def _mlp(input_dim: int, width: int, output_dim: int, depth: int = 2) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(dim, width), nn.LayerNorm(width), nn.SiLU()])
        dim = width
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class FastFilterOutput:
    """Teacher-forced pass of the fast clock over an observed trajectory.

    ``predictions[:, t]`` predicts ``latents[:, t + 1]`` from observed history.
    ``hiddens[:, t]`` is the recurrent state *at* time ``t`` (zeros at t=0),
    i.e. after consuming ``(latents[:, t-1], actions[:, t-1])``.
    """

    predictions: torch.Tensor
    hiddens: torch.Tensor


@dataclass(frozen=True)
class FastRollout:
    latents: torch.Tensor
    hiddens: torch.Tensor


@dataclass(frozen=True)
class MacroAction:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        return self.mean + torch.randn_like(self.mean) * torch.exp(0.5 * self.logvar)

    def kl_to_standard_normal(self) -> torch.Tensor:
        return 0.5 * (self.mean.square() + self.logvar.exp() - 1.0 - self.logvar).sum(-1)


class BoundaryGate(nn.Module):
    """Hazard of ending the current segment after ``elapsed`` blocks."""

    def __init__(
        self,
        repr_dim: int,
        hidden_dim: int,
        max_segment: int,
        width: int = 256,
        init_rate: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0.0 < init_rate < 1.0:
            raise ValueError("init_rate must lie in (0, 1)")
        self.max_segment = max_segment
        self.elapsed_embedding = nn.Embedding(max_segment + 1, 32)
        self.net = _mlp(hidden_dim + 2 * repr_dim + 32, width, 1)
        # Start from a constant hazard so early training sees mid-length segments.
        final = self.net[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(init_rate / (1.0 - init_rate)))

    def forward(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        start_latent: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = elapsed.clamp(0, self.max_segment).long()
        features = torch.cat(
            (hidden, latent, latent - start_latent, self.elapsed_embedding(elapsed)),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class MacroActionEncoder(nn.Module):
    """Encodes every prefix of an action segment into a Gaussian macro-action."""

    def __init__(self, action_dim: int, macro_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.macro_dim = macro_dim
        self.input = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.SiLU())
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 2 * macro_dim)

    def forward(self, actions: torch.Tensor) -> MacroAction:
        """``actions``: ``[..., J, A]`` → per-prefix macro-actions ``[..., J, M]``."""
        leading = actions.shape[:-2]
        flat = actions.reshape(-1, *actions.shape[-2:])
        outputs, _ = self.rnn(self.input(flat))
        mean, logvar = self.head(outputs).chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 4.0)
        return MacroAction(
            mean=mean.reshape(*leading, *mean.shape[-2:]),
            logvar=logvar.reshape(*leading, *logvar.shape[-2:]),
        )


class SlowTransition(nn.Module):
    """Jump from a segment start latent to its end latent; predict duration."""

    def __init__(
        self,
        repr_dim: int,
        macro_dim: int,
        max_segment: int,
        width: int = 512,
    ) -> None:
        super().__init__()
        self.max_segment = max_segment
        self.delta = _mlp(repr_dim + macro_dim, width, repr_dim, depth=3)
        self.duration = _mlp(repr_dim + macro_dim, width // 2, max_segment)

    def forward(
        self, start_latent: torch.Tensor, macro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((start_latent, macro), dim=-1)
        return start_latent + self.delta(features), self.duration(features)


class TwoClockWorldModel(nn.Module):
    def __init__(
        self,
        repr_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        macro_dim: int = 8,
        max_segment: int = 10,
        gate_width: int = 256,
        slow_width: int = 512,
        init_boundary_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        self.max_segment = max_segment
        self.macro_dim = macro_dim
        self.fast = ActionConditionedTransition(
            repr_dim=repr_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            action_embed_dim=128,
            residual=True,
            output_norm=False,
        )
        self.gate = BoundaryGate(
            repr_dim, hidden_dim, max_segment, width=gate_width, init_rate=init_boundary_rate
        )
        self.macro_encoder = MacroActionEncoder(action_dim, macro_dim)
        self.slow = SlowTransition(repr_dim, macro_dim, max_segment, width=slow_width)

    # ------------------------------------------------------------------ fast
    def fast_step(
        self, latent: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        transition = self.fast(latent, action, hidden)
        return transition.latent, transition.hidden

    def fast_rollout(
        self,
        initial_latent: torch.Tensor,
        actions: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> FastRollout:
        """Open-loop rollout. ``actions``: ``[..., H, A]``."""
        latent = initial_latent
        latents, hiddens = [], []
        for step in range(actions.shape[-2]):
            latent, hidden = self.fast_step(latent, actions[..., step, :], hidden)
            latents.append(latent)
            hiddens.append(hidden)
        return FastRollout(torch.stack(latents, dim=-2), torch.stack(hiddens, dim=-2))

    def fast_filter(self, latents: torch.Tensor, actions: torch.Tensor) -> FastFilterOutput:
        """Teacher-forced pass. ``latents``: ``[B, T+1, D]``, ``actions``: ``[B, T, A]``."""
        if latents.shape[1] != actions.shape[1] + 1:
            raise ValueError("latents must contain one more step than actions")
        hidden = self.fast.initial_hidden(latents[:, 0])
        predictions, hiddens = [], [hidden]
        for step in range(actions.shape[1]):
            prediction, hidden = self.fast_step(latents[:, step], actions[:, step], hidden)
            predictions.append(prediction)
            hiddens.append(hidden)
        return FastFilterOutput(torch.stack(predictions, dim=1), torch.stack(hiddens, dim=1))

    # ------------------------------------------------------------------ gate
    def boundary_logits(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        start_latent: torch.Tensor,
        elapsed: torch.Tensor,
    ) -> torch.Tensor:
        return self.gate(hidden, latent, start_latent, elapsed)

    # ------------------------------------------------------------------ slow
    def encode_macro(self, actions: torch.Tensor) -> MacroAction:
        return self.macro_encoder(actions)

    def slow_jump(
        self, start_latent: torch.Tensor, macro: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (end latent, duration logits over 1..max_segment)."""
        return self.slow(start_latent, macro)

    def slow_rollout(
        self, start_latent: torch.Tensor, macros: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chain slow jumps. ``macros``: ``[..., K, M]`` → latents ``[..., K, D]``."""
        latent = start_latent
        latents, durations = [], []
        for step in range(macros.shape[-2]):
            latent, duration_logits = self.slow_jump(latent, macros[..., step, :])
            latents.append(latent)
            durations.append(duration_logits)
        return torch.stack(latents, dim=-2), torch.stack(durations, dim=-2)
