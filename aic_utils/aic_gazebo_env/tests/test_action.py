"""Tests for action validation, clipping, and deterministic application."""

from __future__ import annotations

import pytest

from aic_gazebo_env import GazeboEnv, FakeRuntime, StubBackend


def test_zero_action_minimal_change() -> None:
    env = GazeboEnv(runtime=FakeRuntime(backend=StubBackend()))
    first_observation, _ = env.reset()

    second_observation, _, _, _, info = env.step(
        {"joint_position_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    )

    assert first_observation["joint_positions"] == second_observation["joint_positions"]
    assert second_observation["joint_velocities"] == [0.0] * 6
    assert second_observation["sim_time"] > first_observation["sim_time"]
    assert info["substeps"] == 4


def test_nonzero_action_changes_state() -> None:
    env = GazeboEnv(runtime=FakeRuntime(backend=StubBackend()))
    env.reset()

    observation, _, _, _, info = env.step(
        {"joint_position_delta": [0.1, -0.1, 0.2, 0.0, 0.0, 0.0]}
    )

    assert observation["joint_positions"] != [0.0] * 6
    assert observation["joint_velocities"] != [0.0] * 6
    assert observation["end_effector_pose"]["position"] != [0.0, 0.0, 0.5]
    assert info["applied_action"]["joint_position_delta"] == [0.1, -0.1, 0.2, 0.0, 0.0, 0.0]


def test_invalid_shape_raises_error() -> None:
    env = GazeboEnv(runtime=FakeRuntime(backend=StubBackend()))
    env.reset()

    with pytest.raises(ValueError, match="must have length 6"):
        env.step({"joint_position_delta": [0.1, 0.2]})


def test_repeated_steps_behave_consistently() -> None:
    env = GazeboEnv(runtime=FakeRuntime(backend=StubBackend()))
    env.reset()

    first_observation, _, _, _, first_info = env.step(
        {"joint_position_delta": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    )
    second_observation, _, _, _, second_info = env.step(
        {"joint_position_delta": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    )

    assert first_info["applied_action"]["joint_position_delta"][0] == 0.25
    assert second_info["applied_action"]["joint_position_delta"][0] == 0.25
    assert first_observation["joint_positions"][0] == 0.25
    assert second_observation["joint_positions"][0] == 0.5
    assert first_observation["sim_time"] == 0.2
    assert second_observation["sim_time"] == 0.4
