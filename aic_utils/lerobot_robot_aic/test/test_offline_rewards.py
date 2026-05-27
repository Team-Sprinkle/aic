from __future__ import annotations

import pandas as pd
import pytest

from lerobot_robot_aic.offline_rewards import OfflineRewardConfig, dense_offline_rewards


def test_dense_offline_rewards_use_real_distance_progress() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "distance_to_target": [0.05, 0.03, 0.005],
            "orientation_error": [0.2, 0.1, 0.0],
        }
    )
    rewards = dense_offline_rewards(df, OfflineRewardConfig(objective="insertion"))

    assert rewards.shape == (3,)
    assert rewards[-1] > rewards[0]
    assert rewards[-1] > 1.0


def test_dense_offline_rewards_fail_without_geometry() -> None:
    df = pd.DataFrame({"episode_index": [0, 0, 0]})
    with pytest.raises(ValueError, match="missing required geometry"):
        dense_offline_rewards(df, OfflineRewardConfig(objective="insertion"))


def test_dense_offline_rewards_can_use_named_tcp_error_state() -> None:
    df = pd.DataFrame(
        {
            "episode_index": [0, 0, 0],
            "observation.state": [
                [0.05, 0.0, 0.0, 0.2, 0.0, 0.0],
                [0.03, 0.0, 0.0, 0.1, 0.0, 0.0],
                [0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        }
    )
    df.attrs["observation_state_names"] = [
        "tcp_error.x",
        "tcp_error.y",
        "tcp_error.z",
        "tcp_error.rx",
        "tcp_error.ry",
        "tcp_error.rz",
    ]

    rewards = dense_offline_rewards(df, OfflineRewardConfig(objective="insertion"))

    assert rewards.shape == (3,)
    assert rewards[-1] > rewards[0]
    assert rewards[-1] > 1.0
