"""Tests for the observation pipeline."""

from __future__ import annotations

import pytest

from aic_gazebo_env import FakeRuntime, StubBackend


def test_observation_keys_exist_and_are_stable() -> None:
    runtime = FakeRuntime(backend=StubBackend())
    runtime.start()

    observation, _ = runtime.reset()

    assert list(observation.keys()) == [
        "step_count",
        "sim_time",
        "joint_positions",
        "joint_velocities",
        "end_effector_pose",
    ]
    assert list(observation["end_effector_pose"].keys()) == [
        "position",
        "orientation",
    ]


def test_observation_shapes_are_correct() -> None:
    runtime = FakeRuntime(backend=StubBackend(joint_count=6))
    runtime.start()

    observation, _ = runtime.reset()

    assert isinstance(observation["step_count"], int)
    assert isinstance(observation["sim_time"], float)
    assert len(observation["joint_positions"]) == 6
    assert len(observation["joint_velocities"]) == 6
    assert len(observation["end_effector_pose"]["position"]) == 3
    assert len(observation["end_effector_pose"]["orientation"]) == 4
    assert all(isinstance(value, float) for value in observation["joint_positions"])
    assert all(isinstance(value, float) for value in observation["joint_velocities"])


def test_observation_values_change_across_steps() -> None:
    runtime = FakeRuntime(backend=StubBackend())
    runtime.start()

    first_observation, _ = runtime.reset()
    second_observation, _, _, _, _ = runtime.step({"command": [0.1, 0.2]})
    third_observation, _ = runtime.get_observation()

    assert first_observation["step_count"] == 0
    assert second_observation["step_count"] == 1
    assert third_observation["step_count"] == 1
    assert first_observation["joint_positions"] != second_observation["joint_positions"]
    assert first_observation["joint_velocities"] != second_observation["joint_velocities"]
    assert first_observation["end_effector_pose"] != second_observation["end_effector_pose"]


def test_invalid_missing_entities_fail_clearly() -> None:
    backend = StubBackend(
        observation_mutator=lambda observation: {
            key: value for key, value in observation.items() if key != "joint_positions"
        }
    )
    runtime = FakeRuntime(backend=backend)
    runtime.start()

    with pytest.raises(ValueError, match="missing required key: 'joint_positions'"):
        runtime.reset()


def test_invalid_shapes_fail_clearly() -> None:
    backend = StubBackend(
        observation_mutator=lambda observation: {
            **observation,
            "end_effector_pose": {
                "position": [0.0, 1.0],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
    )
    runtime = FakeRuntime(backend=backend)
    runtime.start()

    with pytest.raises(ValueError, match="end_effector_pose.position"):
        runtime.reset()
