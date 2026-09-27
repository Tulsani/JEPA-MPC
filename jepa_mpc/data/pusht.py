from __future__ import annotations

from pathlib import Path

import numpy as np

from .sequence import EpisodeSequenceDataset


class PushTSequenceDataset(EpisodeSequenceDataset):
    """Canonical Push-T Zarr adapter with episode-safe multi-step windows."""

    @classmethod
    def from_zarr(
        cls,
        path: str | Path,
        horizon: int = 16,
        image_key: str = "img",
        state_key: str = "state",
        action_key: str = "action",
    ) -> "PushTSequenceDataset":
        try:
            import zarr
        except ImportError as error:
            raise ImportError(
                "Push-T Zarr loading requires the optional dependency `zarr`."
            ) from error

        root = zarr.open(str(path), mode="r")
        data = root["data"]
        meta = root["meta"]
        # Keep large image and action arrays lazy; individual windows are read
        # by EpisodeSequenceDataset instead of materializing the full replay.
        images = data[image_key]
        states = data[state_key]
        actions = data[action_key]
        episode_ends = np.asarray(meta["episode_ends"])

        return cls(
            observations=images,
            actions=actions,
            episode_ends=episode_ends,
            horizon=horizon,
            extras={"agent_position": np.asarray(states[:, :2])},
            channel_last_images=True,
            scale_uint_images=True,
        )
