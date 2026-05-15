from __future__ import annotations

import pytest

from gazebo_rl.insertion_reward import (
    GazeboRewardConfig,
    calculate_gazebo_insertion_reward,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)


def _episode() -> dict:
    return {
        "scene": {
            "target": {
                "target_pose_world": {"position": [0.0, 0.0, 0.010], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "entrance_pose_world": {"position": [0.0, 0.0, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
                "insertion_axis_world": [0.0, 0.0, 1.0],
            }
        }
    }


def _obs(position, quat_xyzw=(0.0, 0.0, 0.0, 1.0), force=(0.0, 0.0, 0.0)) -> dict:
    return {
        "oracle": {
            "plug_pose_base_link": {
                "position": list(position),
                "orientation_xyzw": list(quat_xyzw),
            }
        },
        "wrist_wrench": {"force": list(force), "torque": [0.0, 0.0, 0.0]},
    }


def test_quaternion_xyzw_wxyz_conversion_round_trips() -> None:
    assert xyzw_to_wxyz([0.0, 0.0, 0.0, 1.0]) == [1.0, 0.0, 0.0, 0.0]
    assert wxyz_to_xyzw([1.0, 0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0, 1.0]


def test_aligned_near_entrance_has_positive_hover_alignment_reward() -> None:
    cfg = GazeboRewardConfig(cheatcode_phase_weight=1.0, distance_weight=0.0, close_weight=0.0)
    reward, info = calculate_gazebo_insertion_reward(
        prev_obs=_obs([0.002, 0.0, -0.006]),
        obs=_obs([0.0, 0.0, -0.004]),
        episode_config=_episode(),
        config=cfg,
    )
    assert info["insertion_reward_available"] == 1.0
    assert info["cheatcode/lateral_progress"] > 0.0
    assert reward > -1.0


def test_forward_centered_insertion_gets_positive_axial_and_corridor_reward() -> None:
    cfg = GazeboRewardConfig(
        distance_weight=0.0,
        close_weight=0.0,
        axial_progress_weight=1.0,
        corridor_weight=1.0,
        lateral_gate_sigma=0.002,
    )
    reward, info = calculate_gazebo_insertion_reward(
        prev_obs=_obs([0.0, 0.0, 0.002]),
        obs=_obs([0.0, 0.0, 0.006]),
        episode_config=_episode(),
        config=cfg,
    )
    assert info["axial_progress"] > 0.0
    assert info["corridor"] > 0.0
    assert reward > 0.0


def test_forward_off_axis_insertion_is_penalized_by_bypass_signal() -> None:
    cfg = GazeboRewardConfig(
        distance_weight=0.0,
        close_weight=0.0,
        axial_progress_weight=1.0,
        corridor_weight=1.0,
        lateral_gate_sigma=0.001,
        bypass_penalty_scale=6.0,
    )
    reward, info = calculate_gazebo_insertion_reward(
        prev_obs=_obs([0.004, 0.0, 0.002]),
        obs=_obs([0.004, 0.0, 0.006]),
        episode_config=_episode(),
        config=cfg,
    )
    assert info["axial_progress"] < 0.0
    assert info["corridor"] < 0.0
    assert reward < 0.0


def test_retreat_inside_port_gets_negative_cheatcode_retreat() -> None:
    cfg = GazeboRewardConfig(cheatcode_phase_weight=1.0, distance_weight=0.0, close_weight=0.0)
    _, info = calculate_gazebo_insertion_reward(
        prev_obs=_obs([0.0, 0.0, 0.006]),
        obs=_obs([0.0, 0.0, 0.004]),
        episode_config=_episode(),
        config=cfg,
    )
    assert info["cheatcode/retreat"] < 0.0


def test_missing_geometry_reports_unavailable() -> None:
    _, info = calculate_gazebo_insertion_reward(prev_obs={}, obs={}, episode_config=None)
    assert info["insertion_reward_available"] == 0.0
