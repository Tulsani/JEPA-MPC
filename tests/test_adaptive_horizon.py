import torch
from torch import nn

from jepa_mpc.models.world_model import LatentWorldModel
from jepa_mpc.planning import AdaptiveHorizonPlanner, GoalDistanceObjective


class AdditiveTransition(nn.Module):
    repr_dim = 2
    hidden_dim = 2

    def forward(self, latent, action, hidden=None):
        from jepa_mpc.models.dynamics import TransitionState

        next_latent = latent + action
        return TransitionState(latent=next_latent, hidden=next_latent)


def test_planner_jointly_selects_candidate_and_prefix_horizon():
    model = LatentWorldModel(nn.Identity(), AdditiveTransition())
    planner = AdaptiveHorizonPlanner(model, GoalDistanceObjective())
    initial = torch.tensor([[0.0, 0.0]])
    candidates = torch.tensor(
        [[
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]]
    )
    goal = torch.tensor([[0.0, 2.0]])

    result = planner.evaluate_candidates(initial, candidates, goal)

    assert result.candidate_index.item() == 1
    assert result.horizon.item() == 2
    torch.testing.assert_close(result.terminal_latent, goal)
    torch.testing.assert_close(
        result.action_sequence,
        torch.tensor([[[0.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]),
    )
