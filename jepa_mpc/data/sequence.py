from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class EpisodeSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-safe fixed-window sampling over array-like trajectory storage."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        episode_ends: np.ndarray,
        horizon: int,
        extras: Mapping[str, np.ndarray] | None = None,
        channel_last_images: bool = False,
        scale_uint_images: bool = True,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if len(observations) != len(actions):
            raise ValueError("observations and actions must have equal step counts")

        self.observations = observations
        self.actions = actions
        self.episode_ends = np.asarray(episode_ends, dtype=np.int64)
        self.horizon = horizon
        self.extras = dict(extras or {})
        self.channel_last_images = channel_last_images
        self.scale_uint_images = scale_uint_images

        if not len(self.episode_ends) or self.episode_ends[-1] != len(observations):
            raise ValueError("episode_ends must terminate at the number of observations")
        if np.any(np.diff(self.episode_ends) <= 0):
            raise ValueError("episode_ends must be strictly increasing")
        for name, values in self.extras.items():
            if len(values) != len(observations):
                raise ValueError(f"extra array {name!r} has the wrong length")

        starts: list[int] = []
        episode_start = 0
        for episode_end in self.episode_ends:
            # Need H actions at t:t+H and H+1 observations at t:t+H+1.
            starts.extend(range(episode_start, episode_end - horizon))
            episode_start = int(episode_end)
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    @staticmethod
    def _to_tensor(array: np.ndarray) -> torch.Tensor:
        # Copy prevents undefined behavior from read-only memory maps.
        return torch.from_numpy(np.asarray(array).copy())

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        observation_end = start + self.horizon + 1
        action_end = start + self.horizon

        observations = self._to_tensor(self.observations[start:observation_end])
        if observations.dtype == torch.uint8 and self.scale_uint_images:
            observations = observations.float().div_(255.0)
        else:
            observations = observations.float()
        if self.channel_last_images:
            observations = observations.movedim(-1, -3)

        sample = {
            "observations": observations,
            "actions": self._to_tensor(self.actions[start:action_end]).float(),
        }
        for name, values in self.extras.items():
            sample[name] = self._to_tensor(values[start:observation_end]).float()
        return sample
