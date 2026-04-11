#!/usr/bin/env python3
"""Manual smoke test for reset/step/close."""

from __future__ import annotations

from aic_gazebo_env import MinimalTaskEnv


def main() -> None:
    env = MinimalTaskEnv()
    try:
        observation, info = env.reset(seed=123, options={"mode": "smoke"})
        print(
            "reset:",
            {
                "step_count": observation["step_count"],
                "sim_time": observation["sim_time"],
                "info_keys": sorted(info.keys()),
            },
        )

        observation, reward, terminated, truncated, info = env.step(
            {"joint_position_delta": [0.05, 0.05, 0.0, 0.0, 0.0, 0.0]}
        )
        print(
            "step:",
            {
                "step_count": observation["step_count"],
                "reward": round(reward, 6),
                "terminated": terminated,
                "truncated": truncated,
                "termination_reason": info.get("termination_reason"),
            },
        )

        if not isinstance(observation, dict):
            raise RuntimeError("Expected observation dict from step().")
        if not isinstance(info, dict):
            raise RuntimeError("Expected info dict from step().")
    finally:
        env.close()

    print("smoke_reset_step: OK")


if __name__ == "__main__":
    main()
