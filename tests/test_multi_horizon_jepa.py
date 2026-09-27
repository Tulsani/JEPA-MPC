import torch

from jepa_mpc.models import (
    ActionConditionedTransition,
    ConvObservationEncoder,
    LatentWorldModel,
    MultiHorizonJEPA,
)


def test_multi_horizon_jepa_aligns_predictions_and_frozen_targets():
    encoder = ConvObservationEncoder(
        in_channels=3,
        repr_dim=16,
        proprio_dim=2,
        base_channels=8,
    )
    dynamics = ActionConditionedTransition(
        repr_dim=16,
        action_dim=2,
        hidden_dim=16,
        action_embed_dim=8,
    )
    model = MultiHorizonJEPA(LatentWorldModel(encoder, dynamics))
    images = torch.rand(2, 5, 3, 32, 32)
    actions = torch.rand(2, 4, 2)
    proprio = torch.rand(2, 5, 2)

    outputs = model(images, actions, proprio)
    loss = model.loss(images, actions, proprio)

    assert outputs.predicted.shape == (2, 4, 16)
    assert outputs.target.shape == (2, 4, 16)
    assert not outputs.target.requires_grad
    assert loss.ndim == 0
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
