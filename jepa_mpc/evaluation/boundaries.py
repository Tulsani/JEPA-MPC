"""Offline analysis for claims C1 (long-range prediction by slow jumps) and
C2 (alignment of learned boundaries with physical events)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from jepa_mpc.data.latent_cache import ActionNormalizer, LatentCache
from jepa_mpc.models.two_clock import TwoClockWorldModel


@dataclass(frozen=True)
class EpisodeBlocks:
    """One episode resampled at block boundaries (every ``frameskip`` frames)."""

    episode: int
    latents: np.ndarray  # [N+1, D]
    actions: np.ndarray  # [N, A_block] normalized
    state: np.ndarray | None  # [N+1, S]
    contacts: np.ndarray | None = None  # [N] max simulator contact count per block step


def episode_blocks(
    cache: LatentCache,
    episode: int,
    frameskip: int,
    normalizer: ActionNormalizer,
    phase: int = 0,
) -> EpisodeBlocks | None:
    offset, length = int(cache.ep_offset[episode]), int(cache.ep_len[episode])
    num_blocks = (length - 1 - phase) // frameskip
    if num_blocks < 1:
        return None
    start = offset + phase
    frames = start + frameskip * np.arange(num_blocks + 1)
    raw = np.asarray(cache.action[start : start + num_blocks * frameskip])
    actions = np.nan_to_num(normalizer.normalize(raw), nan=0.0).reshape(num_blocks, -1)
    contacts = None
    if cache.contacts is not None:
        raw = np.asarray(cache.contacts[start : start + num_blocks * frameskip], dtype=np.float32)
        contacts = raw.reshape(num_blocks, frameskip, -1).max(axis=(1, 2))
    return EpisodeBlocks(
        episode=episode,
        latents=np.asarray(cache.emb[frames], dtype=np.float32),
        actions=actions.astype(np.float32),
        state=None if cache.state is None else np.asarray(cache.state[frames], dtype=np.float32),
        contacts=contacts,
    )


# ------------------------------------------------------------ physical events
def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def block_motion(
    state: np.ndarray,
    block_xy: tuple[int, int] = (2, 3),
    block_angle: int = 4,
    position_threshold: float = 1.0,
    angle_threshold: float = 0.02,
) -> np.ndarray:
    """Boolean ``[N]``: whether the object moved during each block step.

    Push-T state is ``[agent_x, agent_y, block_x, block_y, block_angle, ...]``
    in simulator pixels/radians. Motion of the passive block is a proxy for
    agent–block contact.
    """
    displacement = np.linalg.norm(np.diff(state[:, list(block_xy)], axis=0), axis=-1)
    rotation = np.abs(wrap_angle(np.diff(state[:, block_angle])))
    return (displacement > position_threshold) | (rotation > angle_threshold)


def contact_during_blocks(contacts: np.ndarray) -> np.ndarray:
    """Boolean ``[N]``: any agent–block contact during each block step."""
    return np.asarray(contacts) > 0


def motion_events(moving: np.ndarray) -> np.ndarray:
    """Block indices (1..N-1) where motion starts or stops (contact onset/release)."""
    changes = np.flatnonzero(moving[1:] != moving[:-1]) + 1
    return changes.astype(np.int64)


# --------------------------------------------------------------- segmentation
@torch.no_grad()
def gate_hazards_along(
    model: TwoClockWorldModel,
    latents: torch.Tensor,
    hiddens: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Hazard of ending a segment that began at ``start`` at each t in (start, stop]."""
    times = torch.arange(start + 1, stop + 1, device=latents.device)
    elapsed = (times - start).clamp(max=model.max_segment)
    logits = model.boundary_logits(
        hiddens[times], latents[times], latents[start].expand(len(times), -1), elapsed
    )
    return torch.sigmoid(logits)


@torch.no_grad()
def segment_episode(
    model: TwoClockWorldModel,
    blocks: EpisodeBlocks,
    mode: str,
    fixed_length: int | None = None,
    threshold: float = 0.5,
    rng: np.random.Generator | None = None,
    device: torch.device | str = "cpu",
    start: int = 0,
    hiddens: torch.Tensor | None = None,
) -> tuple[list[int], np.ndarray]:
    """Greedy segmentation of an observed episode from block ``start`` onward.

    Returns boundary block indices in ``(start, N)`` and, for ``learned``, the
    hazard trace (hazard at each block given the segment active at that time).
    Pass precomputed fast-filter ``hiddens`` to avoid recomputing them.
    """
    num_blocks = len(blocks.actions)
    max_segment = model.max_segment
    hazard_trace = np.full(num_blocks + 1, np.nan, dtype=np.float32)
    boundaries: list[int] = []
    if mode == "fixed":
        if fixed_length is None:
            raise ValueError("fixed mode needs fixed_length")
        return list(range(start + fixed_length, num_blocks, fixed_length)), hazard_trace
    if mode == "random":
        rng = rng or np.random.default_rng(0)
        position = start + int(rng.integers(1, max_segment + 1))
        while position < num_blocks:
            boundaries.append(position)
            position += int(rng.integers(1, max_segment + 1))
        return boundaries, hazard_trace
    if mode != "learned":
        raise ValueError(f"unknown mode {mode!r}")

    latents = torch.from_numpy(blocks.latents).to(device)
    actions = torch.from_numpy(blocks.actions).to(device)
    if hiddens is None:
        hiddens = model.fast_filter(latents[None], actions[None]).hiddens[0]
    while start < num_blocks:
        stop = min(start + max_segment, num_blocks)
        hazards = gate_hazards_along(model, latents, hiddens, start, stop).cpu().numpy()
        hazard_trace[start + 1 : stop + 1] = hazards
        fired = np.flatnonzero(hazards > threshold)
        length = int(fired[0]) + 1 if len(fired) else stop - start
        start += length
        if start < num_blocks:
            boundaries.append(start)
    return boundaries, hazard_trace


# ---------------------------------------------------------------- alignment
def alignment_scores(
    boundaries: np.ndarray, events: np.ndarray, tolerance: int
) -> tuple[float, float]:
    """(precision, recall) of boundaries against events within ±tolerance blocks."""
    boundaries, events = np.asarray(boundaries), np.asarray(events)
    if len(boundaries) == 0 or len(events) == 0:
        return float("nan"), float("nan")
    distance = np.abs(boundaries[:, None] - events[None, :])
    precision = float((distance.min(axis=1) <= tolerance).mean())
    recall = float((distance.min(axis=0) <= tolerance).mean())
    return precision, recall


def random_alignment(
    num_boundaries: int,
    num_blocks: int,
    events: np.ndarray,
    tolerance: int,
    rng: np.random.Generator,
    draws: int = 50,
) -> tuple[float, float]:
    """Rate-matched chance level: same number of boundaries, uniform positions."""
    if num_boundaries == 0 or num_blocks < 2 or len(events) == 0:
        return float("nan"), float("nan")
    scores = []
    candidates = np.arange(1, num_blocks)
    count = min(num_boundaries, len(candidates))
    for _ in range(draws):
        sample = np.sort(rng.choice(candidates, count, replace=False))
        scores.append(alignment_scores(sample, events, tolerance))
    return tuple(np.nanmean(np.asarray(scores), axis=0).tolist())


# ------------------------------------------------------ chained slow prediction
@torch.no_grad()
def chained_slow_prediction(
    model: TwoClockWorldModel,
    blocks: EpisodeBlocks,
    boundaries: list[int],
    horizon: int,
    start: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray | int]:
    """Predict ``z[start + horizon]`` by chaining slow jumps along the segmentation.

    Segments are the given boundaries clipped to ``(start, start + horizon]``;
    the final segment ends exactly at the endpoint so every method is scored
    at the same future time. Also returns the flat fast-clock rollout.
    """
    end = start + horizon
    cuts = [b for b in boundaries if start < b < end] + [end]
    latents = torch.from_numpy(blocks.latents).to(device)
    actions = torch.from_numpy(blocks.actions).to(device)
    current, previous = latents[start], start
    jump_latents = []
    for cut in cuts:
        macro = model.encode_macro(actions[previous:cut][None]).mean[0, -1]
        current, _ = model.slow_jump(current, macro)
        jump_latents.append(current)
        previous = cut
    flat = model.fast_rollout(latents[start][None], actions[start:end][None]).latents[0]
    return {
        "slow_end": current.cpu().numpy(),
        "flat_end": flat[-1].cpu().numpy(),
        "true_end": blocks.latents[end],
        "num_jumps": len(cuts),
        "cuts": np.asarray(cuts),
        "jump_latents": torch.stack(jump_latents).cpu().numpy(),
    }


# -------------------------------------------------------------------- probes
@dataclass(frozen=True)
class RidgeProbe:
    x_mean: np.ndarray
    y_mean: np.ndarray
    weights: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> "RidgeProbe":
        x, y = x.astype(np.float64), y.astype(np.float64)
        x_mean, y_mean = x.mean(0), y.mean(0)
        xc = x - x_mean
        weights = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ (y - y_mean))
        return cls(x_mean, y_mean, weights)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) - self.x_mean) @ self.weights + self.y_mean


def block_pose_targets(state: np.ndarray, block_xy=(2, 3), block_angle: int = 4) -> np.ndarray:
    """``[x, y, cos θ, sin θ]`` so angle regression has no wrap-around."""
    angle = state[:, block_angle]
    return np.stack(
        (state[:, block_xy[0]], state[:, block_xy[1]], np.cos(angle), np.sin(angle)), axis=1
    )


def block_pose_errors(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position error (state units) and absolute angle error (radians)."""
    position = np.linalg.norm(predicted[:, :2] - target[:, :2], axis=1)
    angle = np.abs(
        wrap_angle(np.arctan2(predicted[:, 3], predicted[:, 2]) - np.arctan2(target[:, 3], target[:, 2]))
    )
    return position, angle


@torch.no_grad()
def macro_statistics(
    model: TwoClockWorldModel,
    cache: LatentCache,
    episodes: np.ndarray,
    frameskip: int,
    normalizer: ActionNormalizer,
    num_segments: int = 20_000,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Mean/std of encoded macro-actions over random training segments.

    Segment lengths are uniform on ``1..max_segment`` blocks. The planner
    samples high-level candidates from this distribution instead of N(0, I).
    """
    rng = np.random.default_rng(seed)
    max_segment = model.max_segment
    span = max_segment * frameskip
    eligible = np.asarray([e for e in episodes if cache.ep_len[e] > span])
    if not len(eligible):
        raise ValueError("no episode is long enough to sample macro segments")
    chosen = rng.choice(eligible, num_segments)
    starts = cache.ep_offset[chosen] + rng.integers(0, cache.ep_len[chosen] - span)
    lengths = rng.integers(1, max_segment + 1, size=num_segments)
    raw = np.stack([np.asarray(cache.action[s : s + span]) for s in starts])  # [n, span, a]
    blocks = np.nan_to_num(normalizer.normalize(raw), nan=0.0).reshape(num_segments, max_segment, -1)
    means = []
    for begin in range(0, num_segments, 4096):
        chunk = torch.from_numpy(blocks[begin : begin + 4096].astype(np.float32)).to(device)
        prefix_means = model.encode_macro(chunk).mean  # [n, J, M]
        index = torch.from_numpy(lengths[begin : begin + 4096] - 1).to(device)
        means.append(prefix_means[torch.arange(len(index), device=device), index].cpu().numpy())
    macros = np.concatenate(means)
    return macros.mean(axis=0), np.maximum(macros.std(axis=0), 1e-3)
