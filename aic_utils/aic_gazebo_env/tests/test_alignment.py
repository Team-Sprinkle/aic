"""Tests for alignment against documented evaluation assumptions."""

from __future__ import annotations

from aic_gazebo_env import (
    MinimalTaskEnv,
    build_alignment_report,
    check_alignment_invariants,
)


def test_known_configuration_produces_expected_state() -> None:
    env = MinimalTaskEnv()
    observation, info = env.reset(seed=0)
    privileged = info["privileged_observation"]

    assert observation["step_count"] == 0
    assert privileged["board_pose"]["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert privileged["target_pose"]["position"] == [0.3, 0.0, 0.55]
    assert privileged["object_pose"]["orientation"] == [0.0, 0.0, 0.0, 1.0]


def test_key_invariants_match_evaluation_expectations() -> None:
    report = build_alignment_report()
    invariants = check_alignment_invariants()

    assert report["world_frame"] == "world"
    assert report["robot_name"] == "ur5e"
    assert report["robot_base_frame"] == "base_link"
    assert report["tcp_frame"] == "gripper/tcp"
    assert invariants["world_frame_matches_eval"] is True
    assert invariants["robot_name_matches_eval"] is True
    assert invariants["tcp_frame_matches_eval_docs"] is True
    assert invariants["base_frame_matches_eval_docs"] is True
    assert invariants["action_semantics_documented_difference"] is True
    assert len(report["differences_vs_eval"]) >= 3
