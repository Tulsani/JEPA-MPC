import numpy as np

from jepa_mpc.data import EpisodeSequenceDataset


def test_sequence_windows_never_cross_episode_boundaries():
    observations = np.arange(8, dtype=np.float32)[:, None]
    actions = np.arange(8, dtype=np.float32)[:, None]
    dataset = EpisodeSequenceDataset(
        observations=observations,
        actions=actions,
        episode_ends=np.array([4, 8]),
        horizon=2,
    )

    assert len(dataset) == 4
    windows = [dataset[index]["observations"].squeeze(-1).tolist() for index in range(4)]
    assert windows == [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]]


def test_channel_last_uint_images_are_scaled_and_reordered():
    observations = np.full((4, 8, 8, 3), 255, dtype=np.uint8)
    actions = np.zeros((4, 2), dtype=np.float32)
    dataset = EpisodeSequenceDataset(
        observations=observations,
        actions=actions,
        episode_ends=np.array([4]),
        horizon=2,
        channel_last_images=True,
    )

    sample = dataset[0]
    assert sample["observations"].shape == (3, 3, 8, 8)
    assert sample["observations"].min().item() == 1.0
    assert sample["observations"].max().item() == 1.0
