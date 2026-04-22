"""Runs a simple shaped heuristic policy against the standalone env."""

from __future__ import annotations

import argparse

import numpy as np

from .env import make_default_env


def _heuristic_action(observation: dict[str, np.ndarray]) -> np.ndarray:
    plug_position = observation["plug_pose"][:3]
    target_position = observation["target_port_pose"][:3]
    entrance_position = observation["target_port_entrance_pose"][:3]
    insertion_progress = float(observation["score_geometry"]["insertion_progress"][0])
    target_point = entrance_position if insertion_progress < 0.2 else target_position
    direction = target_point - plug_position
    linear = np.clip(2.5 * direction, -0.05, 0.05)
    action = np.zeros(6, dtype=np.float32)
    action[:3] = linear.astype(np.float32)
    return action


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_default_env()
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(f"episode={episode} trial_id={info['trial_id']} sim_time={observation['sim_time']:.3f}")
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = _heuristic_action(observation)
                observation, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
            final_report = step_info["final_evaluation"]
            print(
                "  finished",
                f"steps={observation['step_count']}",
                f"rl_step_reward_total={total_reward:.3f}",
                f"distance={step_info['distance_to_target']:.4f}",
                f"gym_final_score={final_report['gym_final_score']:.2f}",
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
