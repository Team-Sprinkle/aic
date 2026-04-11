"""Runs a random policy against the standalone env."""

from __future__ import annotations

import argparse

from .env import make_default_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_default_env()
    for episode in range(args.episodes):
        observation, info = env.reset(seed=args.seed + episode)
        print(f"episode={episode} trial_id={info['trial_id']} sim_time={observation['sim_time']:.3f}")
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
    env.close()


if __name__ == "__main__":
    main()
