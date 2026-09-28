import numpy as np
import torch

from jepa_mpc.data.latent_cache import ActionNormalizer, LatentCache
from jepa_mpc.evaluation.boundaries import (
    RidgeProbe,
    alignment_scores,
    block_motion,
    block_pose_errors,
    chained_slow_prediction,
    episode_blocks,
    motion_events,
    segment_episode,
)
from jepa_mpc.models.two_clock import TwoClockWorldModel


def make_blocks(num_frames=41, frameskip=2):
    rng = np.random.default_rng(0)
    cache = LatentCache(
        emb=rng.normal(size=(num_frames, 6)).astype(np.float16),
        action=rng.normal(size=(num_frames, 2)).astype(np.float32),
        ep_len=np.array([num_frames]),
        ep_offset=np.array([0]),
        state=rng.normal(size=(num_frames, 5)).astype(np.float32),
    )
    normalizer = ActionNormalizer.fit(np.asarray(cache.action))
    return episode_blocks(cache, 0, frameskip, normalizer)


def make_model():
    torch.manual_seed(0)
    return TwoClockWorldModel(6, 4, hidden_dim=16, macro_dim=3, max_segment=4, gate_width=16, slow_width=16)


def test_episode_blocks_shapes():
    blocks = make_blocks()
    assert blocks.latents.shape == (21, 6)
    assert blocks.actions.shape == (20, 4)
    assert blocks.state.shape == (21, 5)


def test_motion_events_are_edges():
    moving = np.array([0, 0, 1, 1, 1, 0, 0, 1], dtype=bool)
    assert motion_events(moving).tolist() == [2, 5, 7]


def test_block_motion_uses_block_pose_only():
    state = np.zeros((4, 5))
    state[:, 0] = [0, 10, 20, 30]  # agent moves, block does not
    state[2:, 2] = 5.0  # block jumps between blocks 1 and 2
    assert block_motion(state).tolist() == [False, True, False]


def test_alignment_scores():
    precision, recall = alignment_scores(np.array([3, 10]), np.array([4, 20]), tolerance=1)
    assert precision == 0.5 and recall == 0.5
    assert np.isnan(alignment_scores(np.array([]), np.array([1]), 1)[0])


def test_fixed_and_learned_segmentation():
    model, blocks = make_model(), make_blocks()
    fixed, _ = segment_episode(model, blocks, "fixed", fixed_length=3)
    assert fixed == [3, 6, 9, 12, 15, 18]
    fixed_from, _ = segment_episode(model, blocks, "fixed", fixed_length=3, start=2)
    assert fixed_from[0] == 5
    learned, hazard = segment_episode(model, blocks, "learned")
    gaps = np.diff([0, *learned, 20])
    assert (gaps >= 1).all() and (gaps <= model.max_segment).all()
    assert np.isfinite(hazard[1:]).all()


def test_chained_prediction_hits_the_endpoint():
    model, blocks = make_model(), make_blocks()
    result = chained_slow_prediction(model, blocks, [2, 5, 9], horizon=7, start=1)
    assert result["cuts"].tolist() == [2, 5, 8]
    assert result["num_jumps"] == 3
    assert np.allclose(result["true_end"], blocks.latents[8])
    assert result["slow_end"].shape == (6,)


def test_ridge_probe_and_pose_errors():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 6))
    y = x @ rng.normal(size=(6, 4))
    probe = RidgeProbe.fit(x, y, alpha=1e-6)
    assert np.allclose(probe(x), y, atol=1e-3)
    target = np.array([[0.0, 0.0, 1.0, 0.0]])
    predicted = np.array([[3.0, 4.0, np.cos(3.1), np.sin(3.1)]])
    position, angle = block_pose_errors(predicted, target)
    assert np.isclose(position[0], 5.0) and np.isclose(angle[0], 3.1, atol=1e-6)
