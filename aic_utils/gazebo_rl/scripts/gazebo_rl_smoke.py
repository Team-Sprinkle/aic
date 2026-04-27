#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.score_parser import score_from_scoring_yaml


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one short Gazebo RL random-action rollout.")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument("--sim-distrobox", default=None)
    parser.add_argument("--ground-truth", type=_bool, default=True)
    parser.add_argument("--gazebo-gui", type=_bool, default=False)
    parser.add_argument("--launch-rviz", type=_bool, default=False)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=300.0)
    parser.add_argument("--results-dir", default="outputs/gazebo_rl/smoke_results")
    args = parser.parse_args()

    env = GazeboRLEnv(
        workspace_dir=args.workspace_dir,
        engine_config=args.engine_config,
        sim_distrobox=args.sim_distrobox,
        ground_truth=args.ground_truth,
        gazebo_gui=args.gazebo_gui,
        launch_rviz=args.launch_rviz,
        max_steps=args.max_steps,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        results_dir=args.results_dir,
    )
    try:
        obs, _ = env.reset()
        print(f"first observation keys: {sorted(obs.keys())}")
        for step in range(args.max_steps):
            action = [random.uniform(-0.002, 0.002) for _ in range(3)] + [
                random.uniform(-0.02, 0.02) for _ in range(3)
            ]
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"step {step} action: {action}")
            print(f"step {step} reward: {reward}")
            if terminated or truncated:
                print(f"rollout ended: terminated={terminated} truncated={truncated} info={info}")
                break
        print(f"final parsed score: {score_from_scoring_yaml(env.results_dir)}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
