"""End-to-end tests for the minimal training task."""

from __future__ import annotations

import random

from aic_gazebo_env import MinimalTaskEnv


def _scripted_action(privileged_observation: dict[str, object]) -> dict[str, list[float]]:
    object_position = privileged_observation["object_pose"]["position"]
    target_position = privileged_observation["target_pose"]["position"]
    grasped = privileged_observation["grasped"]
    goal = target_position if grasped else object_position

    x_delta = goal[0] / 2.0
    y_delta = goal[1] / 2.0
    z_delta = (goal[2] - 0.5) / 2.0
    return {
        "joint_position_delta": [
            x_delta,
            x_delta,
            y_delta,
            y_delta,
            z_delta,
            z_delta,
        ]
    }


def test_random_policy_runs_n_steps_without_crash() -> None:
    env = MinimalTaskEnv()
    _, info = env.reset(seed=3)
    assert "privileged_observation" in info

    rng = random.Random(3)
    for _ in range(20):
        observation, reward, terminated, truncated, info = env.step(
            {
                "joint_position_delta": [
                    rng.uniform(-0.1, 0.1) for _ in range(6)
                ]
            }
        )
        assert isinstance(observation, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "privileged_observation" in info
        if terminated or truncated:
            break


def test_scripted_policy_can_achieve_success_sometimes() -> None:
    successes = 0
    for seed in range(5):
        env = MinimalTaskEnv()
        _, info = env.reset(seed=seed)
        for _ in range(15):
            action = _scripted_action(info["privileged_observation"])
            _, _, terminated, truncated, info = env.step(action)
            if terminated and info["termination_reason"] == "success":
                successes += 1
                break
            if truncated:
                break
    assert successes >= 1


def test_env_reset_and_rollout_work() -> None:
    env = MinimalTaskEnv()

    first_observation, first_info = env.reset(seed=11)
    assert first_observation["step_count"] == 0
    assert "privileged_observation" in first_info

    for _ in range(5):
        observation, _, terminated, truncated, info = env.step(
            {"joint_position_delta": [0.05, 0.05, 0.0, 0.0, 0.0, 0.0]}
        )
        assert "privileged_observation" in info
        if terminated or truncated:
            break

    second_observation, second_info = env.reset(seed=11)
    assert second_observation["step_count"] == 0
    assert second_observation == first_observation
    assert second_info["privileged_observation"] == first_info["privileged_observation"]
