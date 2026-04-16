"""Benchmark planner latency and rollout throughput for the teacher stack."""

from __future__ import annotations

import argparse
import json
import time

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher import AgentTeacherController, TeacherConfig, run_teacher_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    env = make_default_env(enable_randomization=True)
    controller = AgentTeacherController(
        planner=DeterministicMockPlannerBackend(),
        config=TeacherConfig(candidate_plan_count=3, segment_limit=4),
    )
    try:
        wall_start = time.perf_counter()
        step_counts: list[int] = []
        for seed in range(args.episodes):
            result = run_teacher_rollout(env=env, controller=controller, seed=100 + seed)
            step_counts.append(len(result.artifact.step_logs))
        wall_s = time.perf_counter() - wall_start
        print(
            json.dumps(
                {
                    "episodes": args.episodes,
                    "wall_s": wall_s,
                    "episodes_per_second": args.episodes / max(wall_s, 1e-9),
                    "mean_step_count": sum(step_counts) / max(len(step_counts), 1),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
