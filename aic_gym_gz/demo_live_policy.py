"""Runs a random policy against the live Gazebo-backed env."""

from __future__ import annotations

import argparse

from .env import make_live_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--world-path", type=str, default=None)
    args = parser.parse_args()

    env = make_live_env(world_path=args.world_path)
    try:
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            print(
                f"episode={episode} trial_id={info['trial_id']} "
                f"sim_tick={observation['sim_tick']} sim_time={observation['sim_time']:.3f}"
            )
            terminated = truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, step_info = env.step(action)
                total_reward += reward
            print(
                "  finished",
                f"steps={observation['step_count']}",
                f"reward={total_reward:.3f}",
                f"distance={step_info['distance_to_target']:.4f}",
                f"evaluation={step_info['evaluation']['total_score']:.2f}",
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
