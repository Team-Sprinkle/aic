"""Run a mock planner teacher rollout and save a replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher import AgentTeacherController, TeacherConfig, run_teacher_rollout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_rollout.json")
    parser.add_argument("--planner-backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=float, default=0.1)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=8)
    parser.add_argument("--openai-cache-dir", default=None)
    args = parser.parse_args()

    env = (
        make_live_env(include_images=args.include_images, enable_randomization=False)
        if args.live
        else make_default_env(include_images=args.include_images, enable_randomization=True)
    )
    try:
        planner = (
            DeterministicMockPlannerBackend()
            if args.planner_backend == "mock"
            else OpenAIPlannerBackend(
                OpenAIPlannerConfig(
                    enabled=True,
                    model=args.openai_model,
                    temperature=args.openai_temperature,
                    timeout_s=args.openai_timeout,
                    max_retries=args.openai_max_retries,
                    max_calls_per_episode=args.openai_max_calls_per_episode,
                    cache_dir=args.openai_cache_dir,
                )
            )
        )
        controller = AgentTeacherController(
            planner=planner,
            config=TeacherConfig(candidate_plan_count=3),
        )
        result = run_teacher_rollout(
            env=env,
            controller=controller,
            seed=args.seed,
            trial_id=args.trial_id,
            output_path=Path(args.output),
        )
        print(
            json.dumps(
                {
                    "output": str(result.output_path),
                    "trial_id": result.artifact.metadata["trial_id"],
                    "task_id": result.artifact.metadata["task_id"],
                    "planner_backend": result.artifact.metadata["planner_backend"],
                    "segment_count": len(result.artifact.trajectory_segments),
                    "step_count": len(result.artifact.step_logs),
                    "probe_count": len(result.artifact.probe_results),
                    "final_distance_to_target": result.artifact.final_info.get("distance_to_target"),
                    "data_quality": result.artifact.metadata.get("data_quality"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
