import torch
from torch import nn

from jepa_mpc.models.world_model import LatentWorldModel


class AdditiveTransition(nn.Module):
    repr_dim = 2
    hidden_dim = 2

    def forward(self, latent, action, hidden=None):
        from jepa_mpc.models.dynamics import TransitionState

        next_latent = latent + action
        return TransitionState(latent=next_latent, hidden=next_latent)


def test_world_model_recurrently_rolls_each_action_prefix():
    model = LatentWorldModel(nn.Identity(), AdditiveTransition())
    initial = torch.tensor([[0.0, 0.0]])
    actions = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [-0.5, 0.0]]])

    rollout = model.rollout(initial, actions)

    assert rollout.latents.shape == (1, 3, 2)
    torch.testing.assert_close(
        rollout.latents,
        torch.tensor([[[1.0, 0.0], [1.0, 2.0], [0.5, 2.0]]]),
    )
