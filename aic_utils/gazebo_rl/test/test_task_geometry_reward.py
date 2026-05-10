from __future__ import annotations

import math

import pytest

from gazebo_rl.task_geometry_reward import (
    TaskGeometryRewardConfig,
    dense_task_geometry_reward,
    plug_to_port_distance,
    plug_to_port_orientation_error,
)


def _obs(position, quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "oracle": {
            "plug_pose_base_link": {
                "position": list(position),
                "orientation_xyzw": list(quat),
            },
            "target_port_pose_base_link": {
                "position": [0.0, 0.0, 0.0],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
    }


def test_plug_to_port_distance_uses_oracle_poses():
    assert plug_to_port_distance(_obs([0.03, 0.04, 0.0])) == pytest.approx(0.05)


def test_plug_to_port_orientation_error_is_zero_for_aligned_quaternions():
    assert plug_to_port_orientation_error(_obs([0.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_dense_task_geometry_reward_is_maximal_at_target_alignment():
    reward, info = dense_task_geometry_reward(
        prev_obs=_obs([0.05, 0.0, 0.0]),
        obs=_obs([0.0, 0.0, 0.0]),
        config=TaskGeometryRewardConfig(terminal_score_weight=0.0),
    )
    assert info["task_geometry_reward_available"] == 1.0
    assert info["plug_to_port_distance_m"] == pytest.approx(0.0)
    assert info["plug_to_port_orientation_error_rad"] == pytest.approx(0.0)
    assert info["plug_to_port_progress_m"] == pytest.approx(0.05)
    assert reward == pytest.approx(0.05 + 0.5 + 0.3 + 0.05)


def test_dense_task_geometry_reward_penalizes_orientation_error():
    half_turn_z = (0.0, 0.0, 1.0, 0.0)
    reward, info = dense_task_geometry_reward(
        prev_obs=None,
        obs=_obs([0.0, 0.0, 0.0], quat=half_turn_z),
        config=TaskGeometryRewardConfig(terminal_score_weight=0.0),
    )
    assert info["plug_to_port_orientation_error_rad"] == pytest.approx(math.pi)
    assert info["plug_to_port_orientation_reward"] < 1.0e-10
    assert reward == pytest.approx(0.5 + 0.3, abs=1.0e-8)


def test_dense_task_geometry_reward_reports_unavailable_without_ground_truth():
    reward, info = dense_task_geometry_reward(prev_obs=None, obs={"oracle": {}})
    assert reward == 0.0
    assert info == {"task_geometry_reward_available": 0.0}
