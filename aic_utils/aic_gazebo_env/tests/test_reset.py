"""Tests for deterministic and stable reset behavior."""

from __future__ import annotations

from aic_gazebo_env import FakeRuntime, StubBackend


def test_same_seed_gives_same_initial_state() -> None:
    runtime = FakeRuntime(backend=StubBackend())
    runtime.start()

    first_observation, first_info = runtime.reset(seed=123)
    second_observation, second_info = runtime.reset(seed=123)

    assert first_observation == second_observation
    assert first_info["task_object_pose"] == second_info["task_object_pose"]
    assert first_info["settling_steps"] == second_info["settling_steps"]


def test_reset_after_steps_returns_valid_state() -> None:
    runtime = FakeRuntime(backend=StubBackend())
    runtime.start()

    runtime.reset(seed=7)
    runtime.step({"joint_position_delta": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]})
    observation, info = runtime.reset(seed=7)

    assert observation["step_count"] == 0
    assert observation["joint_velocities"] == [0.0] * 6
    assert observation["sim_time"] == 0.1
    assert info["entity_count"] == 2
    assert info["task_object_pose"]["orientation"] == [0.0, 0.0, 0.0, 1.0]


def test_repeated_reset_does_not_leak_entities() -> None:
    backend = StubBackend()
    runtime = FakeRuntime(backend=backend)
    runtime.start()

    counts: list[int] = []
    for seed in (1, 2, 3):
        _, info = runtime.reset(seed=seed)
        counts.append(info["entity_count"])

    assert counts == [2, 2, 2]
    assert backend.entity_ids == ["robot", "task_object"]
