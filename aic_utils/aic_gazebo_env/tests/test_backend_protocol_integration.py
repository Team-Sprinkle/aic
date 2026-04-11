"""Integration tests for runtime-to-backend protocol roundtrips."""

from aic_gazebo_env import FakeRuntime, StubBackend


def test_python_can_send_request_and_receive_response() -> None:
    runtime = FakeRuntime(backend=StubBackend())

    runtime.start()
    observation, info = runtime.reset(seed=5, options={"mode": "integration"})

    assert observation["backend"] == "stub"
    assert observation["step_count"] == 0
    assert info["backend"] == "stub"
    assert info["seed"] == 5


def test_step_returns_incrementing_counter() -> None:
    runtime = FakeRuntime(backend=StubBackend())

    runtime.start()
    runtime.reset()
    first = runtime.step({"command": [1, 2, 3]})
    second = runtime.step({"command": [4, 5, 6]})

    first_observation, _, _, _, first_info = first
    second_observation, _, _, _, second_info = second
    assert first_observation["step_count"] == 1
    assert second_observation["step_count"] == 2
    assert first_info["step_count"] == 1
    assert second_info["step_count"] == 2


def test_multiple_calls_are_consistent() -> None:
    runtime = FakeRuntime(backend=StubBackend())

    runtime.start()
    runtime.reset(seed=9, options={"episode": "a"})
    runtime.step({"command": [0.1]})
    runtime.step({"command": [0.2]})
    observation, info = runtime.get_observation()

    assert observation["step_count"] == 2
    assert observation["last_action"] == {"command": [0.2]}
    assert info["backend"] == "stub"
