"""Tests for the runtime abstraction and fake runtime."""

from aic_gazebo_env import FakeRuntime, GazeboEnv


def test_env_works_end_to_end_with_fake_runtime() -> None:
    runtime = FakeRuntime()
    env = GazeboEnv(runtime=runtime)

    observation, info = env.reset(seed=7, options={"mode": "test"})
    step_result = env.step({"command": [1.0, 2.0]})
    env.close()

    assert observation["runtime"] == "fake"
    assert info["seed"] == 7
    assert len(step_result) == 5
    assert runtime.is_started is True
    assert runtime.is_stopped is True


def test_fake_runtime_lifecycle_and_step_behavior() -> None:
    runtime = FakeRuntime()

    runtime.start()
    observation, info = runtime.reset(seed=11, options={"episode": 1})
    step_observation, reward, terminated, truncated, step_info = runtime.step(
        {"command": [0.5, -0.5]}
    )
    runtime.stop()

    assert runtime.is_started is True
    assert observation["step_count"] == 0
    assert info["options"] == {"episode": 1}
    assert step_observation["last_action"] == [0.5, -0.5]
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert step_info["step_count"] == 1
