#!/usr/bin/env python3
"""Manual smoke test for a constant small-action rollout."""

from __future__ import annotations

from aic_gazebo_env import MinimalTaskEnv


def main() -> None:
    env = MinimalTaskEnv()
    action = {"joint_position_delta": [0.02, 0.02, 0.0, 0.0, 0.0, 0.0]}
    try:
        observation, _ = env.reset(seed=21, options={"mode": "smoke"})
        start_x = observation["end_effector_pose"]["position"][0]

        final_observation = observation
        steps = 0
        for _ in range(5):
            final_observation, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break

        end_x = final_observation["end_effector_pose"]["position"][0]
        if end_x <= start_x:
            raise RuntimeError(
                f"Expected end-effector x to increase, got start={start_x} end={end_x}"
            )
        print(
            "constant_action:",
            {
                "steps": steps,
                "start_x": round(start_x, 6),
                "end_x": round(end_x, 6),
                "reward": round(reward, 6),
                "terminated": terminated,
                "truncated": truncated,
                "reason": info.get("termination_reason"),
            },
        )
    finally:
        env.close()

    print("smoke_constant_action: OK")


if __name__ == "__main__":
    main()
