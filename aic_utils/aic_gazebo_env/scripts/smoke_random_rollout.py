#!/usr/bin/env python3
"""Manual smoke test for a short random rollout."""

from __future__ import annotations

import random

from aic_gazebo_env import MinimalTaskEnv


def main() -> None:
    rng = random.Random(7)
    env = MinimalTaskEnv()
    steps = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    try:
        _, info = env.reset(seed=7, options={"mode": "smoke"})
        if "privileged_observation" not in info:
            raise RuntimeError("Expected privileged_observation in reset info.")

        for _ in range(10):
            observation, reward, terminated, truncated, info = env.step(
                {
                    "joint_position_delta": [
                        rng.uniform(-0.05, 0.05) for _ in range(6)
                    ]
                }
            )
            steps += 1
            total_reward += reward
            if not isinstance(observation, dict):
                raise RuntimeError("Expected observation dict during rollout.")
            if terminated or truncated:
                break
    finally:
        env.close()

    print(
        "rollout:",
        {
            "steps": steps,
            "total_reward": round(total_reward, 6),
            "terminated": terminated,
            "truncated": truncated,
            "reason": info.get("termination_reason"),
        },
    )
    print("smoke_random_rollout: OK")


if __name__ == "__main__":
    main()
