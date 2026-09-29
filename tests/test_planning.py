import pytest
import torch

from jepa_mpc.models.two_clock import TwoClockWorldModel
from jepa_mpc.planning.cem import cem
from jepa_mpc.planning.hierarchical import PlannerConfig, TwoClockPlanner


def small_config(**overrides):
    base = dict(low_horizon=3, high_jumps=2, low_samples=32, low_iterations=3, low_elites=4,
                high_samples=32, high_iterations=3, high_elites=4)
    base.update(overrides)
    return PlannerConfig(**base)


def make_model():
    torch.manual_seed(0)
    return TwoClockWorldModel(6, 4, hidden_dim=16, macro_dim=3, max_segment=4, gate_width=16, slow_width=16)


def test_cem_minimizes_quadratic_within_bounds():
    target = torch.tensor([[[0.5, -1.0]] * 3, [[2.0, 0.0]] * 3])  # [B=2, H=3, A=2]

    def cost(candidates):
        return (candidates - target[:, None]).square().sum(dim=(-1, -2))

    result = cem(cost, batch=2, horizon=3, dim=2, num_samples=200, iterations=15, elites=20,
                 bound=1.5, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(result.best[0], target[0], atol=0.1)
    assert result.best.abs().max() <= 1.5 + 1e-6  # clamped: target 2.0 is out of bounds
    assert (result.cost_history[1:] <= result.cost_history[:-1] + 1e-6).all()


@pytest.mark.parametrize("mode", ["flat", "learned", "fixed", "duration"])
def test_planner_runs_in_every_mode(mode):
    planner = TwoClockPlanner(make_model(), small_config(mode=mode, fixed_length=2), num_envs=3)
    latent, goal = torch.randn(3, 6), torch.randn(3, 6)
    for _ in range(5):
        action, info = planner.act(latent + 0.1 * torch.randn(3, 6), goal)
        assert action.shape == (3, 4)
        assert action.abs().max() <= planner.config.action_bound + 1e-6
        assert (info["chosen_horizon"] >= 1).all()
        assert (info["chosen_horizon"] <= info["horizon_cap"]).all()
    if mode == "flat":
        assert not info["switched"].any()


def test_fixed_mode_switches_every_k_blocks():
    planner = TwoClockPlanner(make_model(), small_config(mode="fixed", fixed_length=2,
                                                         reach_fraction=0.0), num_envs=1)
    latent, goal = torch.randn(1, 6), torch.randn(1, 6)
    switches = []
    for _ in range(7):
        _, info = planner.act(latent, goal)
        switches.append(bool(info["switched"][0]))
    assert switches == [True, False, True, False, True, False, True]


def test_subgoal_persists_across_replanning_until_switch():
    planner = TwoClockPlanner(make_model(), small_config(mode="fixed", fixed_length=3,
                                                         reach_fraction=0.0), num_envs=2)
    latent, goal = torch.randn(2, 6), torch.randn(2, 6)
    planner.act(latent, goal)
    subgoal = planner.subgoal.clone()
    planner.act(latent, goal)
    planner.act(latent, goal)
    assert torch.equal(planner.subgoal, subgoal)
    planner.act(latent, goal)  # elapsed reaches 3 → new subgoal
    assert not torch.equal(planner.subgoal, subgoal)


def test_partial_env_updates_and_reset():
    planner = TwoClockPlanner(make_model(), small_config(mode="learned"), num_envs=3)
    latent, goal = torch.randn(3, 6), torch.randn(3, 6)
    planner.act(latent, goal)
    planner.act(latent[[1]], goal[[1]], env_ids=[1])
    assert planner.elapsed[1] == 1 and planner.elapsed[0] == 0
    planner.reset([1])
    assert not planner.has_subgoal[1] and planner.has_subgoal[0]


def test_macro_statistics_shift_high_level_samples():
    import numpy as np

    from jepa_mpc.data.latent_cache import ActionNormalizer, LatentCache
    from jepa_mpc.evaluation.boundaries import macro_statistics

    model = make_model()
    rng = np.random.default_rng(0)
    cache = LatentCache(
        emb=rng.normal(size=(200, 6)).astype(np.float16),
        action=rng.normal(size=(200, 2)).astype(np.float32),
        ep_len=np.array([100, 100]), ep_offset=np.array([0, 100]),
    )
    normalizer = ActionNormalizer.fit(np.asarray(cache.action))
    mean, std = macro_statistics(model, cache, np.array([0, 1]), 2, normalizer, num_segments=500)
    assert mean.shape == (3,) and std.shape == (3,) and (std > 0).all()

    config = small_config(mode="learned", macro_mean=mean.tolist(), macro_std=std.tolist())
    planner = TwoClockPlanner(model, config, num_envs=2)
    action, _ = planner.act(torch.randn(2, 6), torch.randn(2, 6))
    assert action.shape == (2, 4)
    with pytest.raises(ValueError):
        TwoClockPlanner(model, small_config(macro_mean=[0.0]), num_envs=1)
