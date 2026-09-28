import numpy as np
import pytest
import torch

from jepa_mpc.data.latent_cache import (
    ActionNormalizer,
    LatentBlockDataset,
    LatentCache,
    split_episodes,
)
from jepa_mpc.models.two_clock import TwoClockWorldModel
from jepa_mpc.training.two_clock_losses import (
    fast_losses,
    gate_objective,
    gather_segments,
    segment_distribution,
    slow_losses,
    stick_breaking,
)


def make_model(max_segment: int = 4) -> TwoClockWorldModel:
    torch.manual_seed(0)
    return TwoClockWorldModel(
        repr_dim=6,
        action_dim=4,
        hidden_dim=16,
        macro_dim=3,
        max_segment=max_segment,
        gate_width=16,
        slow_width=32,
    )


def make_cache(lengths=(12, 9, 15), embed_dim=6, action_dim=2) -> LatentCache:
    total = sum(lengths)
    offsets = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    frame = np.arange(total, dtype=np.float32)
    return LatentCache(
        emb=np.repeat(frame[:, None], embed_dim, axis=1).astype(np.float16),
        action=np.stack([frame, -frame], axis=1)[:, :action_dim],
        ep_len=np.asarray(lengths),
        ep_offset=offsets,
        state=np.stack([frame] * 5, axis=1),
    )


def test_stick_breaking_is_distribution_with_forced_end():
    logits = torch.randn(3, 5, 7)
    distribution = stick_breaking(logits)
    assert torch.allclose(distribution.sum(-1), torch.ones(3, 5))
    assert (distribution >= 0).all()
    # A certain boundary at j=2 puts all mass there.
    logits = torch.full((1, 4), -20.0)
    logits[0, 1] = 20.0
    assert stick_breaking(logits)[0, 1] > 0.999


def test_baseline_segment_distributions():
    logits = torch.zeros(2, 3, 5)
    fixed = segment_distribution("fixed", logits, fixed_length=3)
    assert torch.equal(fixed.argmax(-1), torch.full((2, 3), 2))
    random = segment_distribution("random", logits)
    assert torch.allclose(random, torch.full_like(logits, 0.2))
    with pytest.raises(ValueError):
        segment_distribution("fixed", logits, fixed_length=9)
    with pytest.raises(ValueError):
        segment_distribution("bogus", logits)


def test_gather_segments_indices():
    latents = torch.arange(9, dtype=torch.float32).view(1, 9, 1).expand(2, 9, 3)
    actions = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(2, 8, 2)
    hiddens = latents.clone()
    segments = gather_segments(latents, actions, hiddens, max_segment=3)
    num_starts = 8 - 3 + 1
    assert segments.start_latent.shape == (2, num_starts, 3)
    for s in range(num_starts):
        assert segments.start_latent[0, s, 0] == s
        for j in range(1, 4):
            assert segments.end_latent[0, s, j - 1, 0] == s + j
            assert segments.end_hidden[0, s, j - 1, 0] == s + j
            # prefix j ends with action s + j - 1
            assert segments.actions[0, s, j - 1, 0] == s + j - 1


def test_fast_filter_alignment_and_rollout_shapes():
    model = make_model()
    latents, actions = torch.randn(2, 6, 6), torch.randn(2, 5, 4)
    filtered = model.fast_filter(latents, actions)
    assert filtered.predictions.shape == (2, 5, 6)
    assert filtered.hiddens.shape == (2, 6, 16)
    assert torch.count_nonzero(filtered.hiddens[:, 0]) == 0
    # The first teacher-forced prediction equals a one-step open-loop rollout.
    rollout = model.fast_rollout(latents[:, 0], actions)
    assert torch.allclose(rollout.latents[:, 0], filtered.predictions[:, 0])
    assert rollout.latents.shape == (2, 5, 6)


def test_fast_actions_change_predictions():
    model = make_model()
    latent = torch.randn(1, 6)
    first = model.fast_rollout(latent, torch.zeros(1, 3, 4)).latents
    second = model.fast_rollout(latent, torch.ones(1, 3, 4)).latents
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("mode", ["learned", "fixed", "random"])
def test_slow_losses_train_slow_modules_only(mode):
    model = make_model(max_segment=4)
    latents, actions = torch.randn(3, 9, 6), torch.randn(3, 8, 4)
    loss, metrics = slow_losses(model, latents, actions, mode=mode, fixed_length=2)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.fast.parameters())
    assert any(p.grad is not None for p in model.slow.parameters())
    assert any(p.grad is not None for p in model.macro_encoder.parameters())
    gate_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.gate.parameters()
    )
    assert gate_has_grad == (mode == "learned")
    assert 1.0 <= metrics["segments/mean_length"] <= 4.0


def test_fast_loss_decreases_on_one_batch():
    model = make_model()
    latents = torch.randn(4, 7, 6)
    actions = torch.randn(4, 6, 4)
    optimizer = torch.optim.Adam(model.fast.parameters(), lr=3e-3)
    initial, _ = fast_losses(model, latents, actions)
    for _ in range(150):
        optimizer.zero_grad()
        loss, _ = fast_losses(model, latents, actions)
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.3 * initial.item()


def test_gate_objective_stops_where_error_jumps():
    # Error grows slowly until j=3, then jumps: the gate should end at j=3.
    error = torch.tensor([[0.10, 0.12, 0.14, 1.50, 1.60, 1.70]])
    logits = torch.zeros(1, 6, requires_grad=True)
    optimizer = torch.optim.Adam([logits], lr=0.1)
    for _ in range(400):
        optimizer.zero_grad()
        gate_objective(stick_breaking(logits), error, boundary_cost=0.5).backward()
        optimizer.step()
    assert stick_breaking(logits).argmax().item() == 2


def test_gate_objective_longer_segments_with_higher_cost():
    error = torch.linspace(0.1, 1.0, 6)[None]

    def optimal_length(cost: float) -> float:
        logits = torch.zeros(1, 6, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=0.1)
        for _ in range(400):
            optimizer.zero_grad()
            gate_objective(stick_breaking(logits), error, boundary_cost=cost).backward()
            optimizer.step()
        distribution = stick_breaking(logits).detach()
        return float((distribution * torch.arange(1, 7)).sum())

    assert optimal_length(0.1) < optimal_length(5.0)


def test_latent_block_dataset_is_episode_safe():
    cache = make_cache()
    normalizer = ActionNormalizer.fit(np.asarray(cache.action))
    dataset = LatentBlockDataset(cache, np.arange(3), window=2, frameskip=3, normalizer=normalizer)
    # Episode lengths 12, 9, 15 with span 6 → 6 + 3 + 9 starts.
    assert len(dataset) == 18
    for index in range(len(dataset)):
        item = dataset[index]
        frames = item["latents"][:, 0].numpy().astype(int)
        episode = int(item["episode"])
        low = cache.ep_offset[episode]
        high = low + cache.ep_len[episode]
        assert (frames >= low).all() and (frames < high).all()
        assert np.array_equal(np.diff(frames), [3, 3])
        assert item["actions"].shape == (2, 6)
        assert item["state"].shape == (3, 5)


def test_action_blocks_are_normalized_and_invertible():
    cache = make_cache()
    normalizer = ActionNormalizer.fit(np.asarray(cache.action))
    dataset = LatentBlockDataset(cache, np.arange(3), window=2, frameskip=3, normalizer=normalizer)
    item = dataset[0]
    raw = normalizer.denormalize(item["actions"].numpy().reshape(6, 2))
    assert np.allclose(raw, cache.action[0:6], atol=1e-3)
    restored = ActionNormalizer.from_state_dict(normalizer.state_dict())
    assert np.allclose(restored.mean, normalizer.mean)


def test_nan_actions_are_zeroed():
    cache = make_cache()
    cache.action[3] = np.nan
    normalizer = ActionNormalizer.fit(np.asarray(cache.action))
    dataset = LatentBlockDataset(cache, np.arange(3), window=2, frameskip=3, normalizer=normalizer)
    assert torch.isfinite(dataset[0]["actions"]).all()


def test_split_episodes_disjoint_and_complete():
    train, val = split_episodes(50, 0.1, seed=3)
    assert len(np.intersect1d(train, val)) == 0
    assert len(train) + len(val) == 50
    assert len(val) == 5


def test_cache_validation():
    with pytest.raises(ValueError):
        LatentCache(
            emb=np.zeros((5, 2)),
            action=np.zeros((5, 2)),
            ep_len=np.array([3, 3]),
            ep_offset=np.array([0, 3]),
        )
