"""Tests for reward and termination behavior."""

from __future__ import annotations

from aic_gazebo_env import FakeRuntime, StubBackend, Termination


def test_success_triggers_termination() -> None:
    backend = StubBackend(
        termination_model=Termination(success_distance_threshold=10.0, max_episode_steps=25)
    )
    runtime = FakeRuntime(backend=backend)
    runtime.start()
    runtime.reset(seed=0)

    _, reward, terminated, truncated, info = runtime.step(
        {"joint_position_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    )

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "success"
    assert reward > 0.0


def test_timeout_works() -> None:
    backend = StubBackend(
        termination_model=Termination(
            success_distance_threshold=0.0,
            max_episode_steps=2,
            failure_distance_threshold=100.0,
        )
    )
    runtime = FakeRuntime(backend=backend)
    runtime.start()
    runtime.reset(seed=1)

    first = runtime.step({"joint_position_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})
    second = runtime.step({"joint_position_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    assert first[2] is False
    assert first[3] is False
    assert second[2] is False
    assert second[3] is True
    assert second[4]["termination_reason"] == "timeout"


def test_reward_decreases_as_distance_increases() -> None:
    runtime = FakeRuntime(backend=StubBackend())
    runtime.start()
    runtime.reset(seed=2)

    near_step = runtime.step({"joint_position_delta": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]})

    runtime.reset(seed=2)
    far_step = runtime.step({"joint_position_delta": [0.25, 0.25, 0.25, 0.25, 0.25, 0.25]})

    assert near_step[4]["distance_to_target"] < far_step[4]["distance_to_target"]
    assert near_step[1] > far_step[1]
