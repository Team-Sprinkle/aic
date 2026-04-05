"""Import tests for the aic_gazebo_env package skeleton."""

from aic_gazebo_env import (
    Action,
    FakeRuntime,
    GazeboEnv,
    Observation,
    Reward,
    Runtime,
    Termination,
)


def test_top_level_imports() -> None:
    assert GazeboEnv is not None
    assert Runtime is not None
    assert FakeRuntime is not None
    assert Observation is not None
    assert Action is not None
    assert Reward is not None
    assert Termination is not None


def test_module_imports() -> None:
    import aic_gazebo_env.action
    import aic_gazebo_env.env
    import aic_gazebo_env.observation
    import aic_gazebo_env.reward
    import aic_gazebo_env.runtime
    import aic_gazebo_env.termination

    assert aic_gazebo_env.env is not None
