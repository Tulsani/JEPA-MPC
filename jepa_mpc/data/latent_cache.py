"""Block-level sequences over pre-encoded (frozen-encoder) latents.

``scripts/cache_latents.py`` writes one directory per dataset:

    emb.npy        [N, D]  float16  frozen-encoder embedding of every frame
    action.npy     [N, A]  float32  raw environment action taken at each frame
    state.npy      [N, S]  float32  simulator state (optional, diagnostics only)
    proprio.npy    [N, P]  float32  proprioception (optional)
    n_contacts.npy [N]     float32  simulator contact count (optional)
    ep_len.npy     [E]     int64
    ep_offset.npy  [E]     int64
    meta.json

One world-model step is an *action block* of ``frameskip`` environment steps,
matching LeWM: latents are taken every ``frameskip`` frames and the block action
is the concatenation of the ``frameskip`` normalized env actions in between.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

REQUIRED_FILES = ("emb.npy", "action.npy", "ep_len.npy", "ep_offset.npy")


@dataclass(frozen=True)
class ActionNormalizer:
    """Per-dimension z-score for raw env actions (fit on training episodes)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, actions: np.ndarray, eps: float = 1e-6) -> "ActionNormalizer":
        finite = actions[np.isfinite(actions).all(axis=1)]
        if not len(finite):
            raise ValueError("no finite actions to fit normalization")
        return cls(
            mean=finite.mean(axis=0).astype(np.float32),
            std=np.maximum(finite.std(axis=0), eps).astype(np.float32),
        )

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.mean) / self.std

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * self.std + self.mean

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "ActionNormalizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            std=np.asarray(state["std"], dtype=np.float32),
        )


@dataclass
class LatentCache:
    emb: np.ndarray
    action: np.ndarray
    ep_len: np.ndarray
    ep_offset: np.ndarray
    state: np.ndarray | None = None
    proprio: np.ndarray | None = None
    contacts: np.ndarray | None = None
    meta: dict | None = None

    def __post_init__(self) -> None:
        self.ep_len = np.asarray(self.ep_len, dtype=np.int64)
        self.ep_offset = np.asarray(self.ep_offset, dtype=np.int64)
        n = len(self.emb)
        if len(self.action) != n:
            raise ValueError("emb and action must have the same number of frames")
        if len(self.ep_len) != len(self.ep_offset):
            raise ValueError("ep_len and ep_offset must have equal length")
        if np.any(self.ep_len <= 0):
            raise ValueError("episode lengths must be positive")
        if np.any(self.ep_offset + self.ep_len > n) or np.any(self.ep_offset < 0):
            raise ValueError("episode offsets/lengths exceed the number of frames")
        for name in ("state", "proprio", "contacts"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{name} must have one row per frame")

    @property
    def num_episodes(self) -> int:
        return len(self.ep_len)

    @property
    def embed_dim(self) -> int:
        return int(self.emb.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action.shape[1])

    def episode_frames(self, episodes: np.ndarray) -> np.ndarray:
        """Concatenated frame indices of the given episodes."""
        return np.concatenate(
            [np.arange(self.ep_offset[e], self.ep_offset[e] + self.ep_len[e]) for e in episodes]
        )

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = True) -> "LatentCache":
        directory = Path(directory)
        missing = [name for name in REQUIRED_FILES if not (directory / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"latent cache {directory} is missing {missing}; run scripts/cache_latents.py"
            )
        mode = "r" if mmap else None

        def optional(name: str) -> np.ndarray | None:
            path = directory / f"{name}.npy"
            return np.load(path, mmap_mode=mode) if path.exists() else None

        meta_path = directory / "meta.json"
        return cls(
            emb=np.load(directory / "emb.npy", mmap_mode=mode),
            action=np.load(directory / "action.npy", mmap_mode=mode),
            ep_len=np.load(directory / "ep_len.npy"),
            ep_offset=np.load(directory / "ep_offset.npy"),
            state=optional("state"),
            proprio=optional("proprio"),
            contacts=optional("n_contacts"),
            meta=json.loads(meta_path.read_text()) if meta_path.exists() else None,
        )


def split_episodes(
    num_episodes: int, val_fraction: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint train/validation episode indices."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie in (0, 1)")
    if num_episodes < 2:
        raise ValueError("need at least two episodes to split")
    order = np.random.default_rng(seed).permutation(num_episodes)
    num_val = min(max(1, int(round(num_episodes * val_fraction))), num_episodes - 1)
    return np.sort(order[num_val:]), np.sort(order[:num_val])


class LatentBlockDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-safe windows of ``window`` action blocks over cached latents.

    Each item contains ``window + 1`` latents at block boundaries and
    ``window`` block actions, plus aligned state/proprio when available.
    Every frame offset is a valid window start (``start_stride`` subsamples).
    """

    def __init__(
        self,
        cache: LatentCache,
        episodes: np.ndarray,
        window: int,
        frameskip: int,
        normalizer: ActionNormalizer,
        start_stride: int = 1,
    ) -> None:
        if window < 1 or frameskip < 1 or start_stride < 1:
            raise ValueError("window, frameskip and start_stride must be positive")
        self.cache = cache
        self.episodes = np.asarray(episodes, dtype=np.int64)
        self.window = window
        self.frameskip = frameskip
        self.normalizer = normalizer

        span = window * frameskip  # frames between first and last latent
        starts: list[np.ndarray] = []
        episode_of_start: list[np.ndarray] = []
        for episode in self.episodes:
            offset, length = int(cache.ep_offset[episode]), int(cache.ep_len[episode])
            # Last latent index is start + span, which must stay inside the episode.
            count = length - span
            if count <= 0:
                continue
            local = np.arange(0, count, start_stride, dtype=np.int64)
            starts.append(offset + local)
            episode_of_start.append(np.full(len(local), episode, dtype=np.int64))
        if not starts:
            raise ValueError("no episode is long enough for the requested window")
        self.starts = np.concatenate(starts)
        self.episode_of_start = np.concatenate(episode_of_start)

    def __len__(self) -> int:
        return len(self.starts)

    def block_actions(self, start: int) -> np.ndarray:
        raw = np.asarray(self.cache.action[start : start + self.window * self.frameskip])
        normalized = np.nan_to_num(self.normalizer.normalize(raw), nan=0.0)
        return normalized.reshape(self.window, self.frameskip * self.cache.action_dim)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[index])
        frames = start + self.frameskip * np.arange(self.window + 1)
        item = {
            "latents": torch.from_numpy(np.asarray(self.cache.emb[frames], dtype=np.float32)),
            "actions": torch.from_numpy(self.block_actions(start).astype(np.float32)),
            "start_frame": torch.tensor(start, dtype=torch.int64),
            "episode": torch.tensor(self.episode_of_start[index], dtype=torch.int64),
        }
        for name in ("state", "proprio"):
            values = getattr(self.cache, name)
            if values is not None:
                item[name] = torch.from_numpy(np.asarray(values[frames], dtype=np.float32))
        return item
