"""Run teacher candidate search and ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher.search import TeacherCandidateSearch, TeacherSearchConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_search.json")
    parser.add_argument("--near-perfect-threshold", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--planner-backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=float, default=0.1)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=8)
    parser.add_argument("--openai-max-calls-per-search", type=int, default=64)
    parser.add_argument("--openai-cache-dir", default=None)
    args = parser.parse_args()

    factory = (
        (lambda: make_live_env(enable_randomization=False, include_images=False))
        if args.live
        else (lambda: make_default_env(enable_randomization=True, include_images=False))
    )
    planner_factory = (
        (lambda: DeterministicMockPlannerBackend())
        if args.planner_backend == "mock"
        else (
            lambda: OpenAIPlannerBackend(
                OpenAIPlannerConfig(
                    enabled=True,
                    model=args.openai_model,
                    temperature=args.openai_temperature,
                    timeout_s=args.openai_timeout,
                    max_retries=args.openai_max_retries,
                    max_calls_per_episode=args.openai_max_calls_per_episode,
                    max_calls_per_search=args.openai_max_calls_per_search,
                    cache_dir=args.openai_cache_dir,
                )
            )
        )
    )
    search = TeacherCandidateSearch(
        env_factory=factory,
        planner_factory=planner_factory,
        config=TeacherSearchConfig(
            near_perfect_threshold=args.near_perfect_threshold,
            top_k=args.top_k,
        ),
    )
    result = search.run(
        seed=args.seed,
        trial_id=args.trial_id,
        output_path=Path(args.output),
    )
    top = result.payload["top_candidates"][0] if result.payload["top_candidates"] else None
    print(
        json.dumps(
            {
                "output": str(result.output_path),
                "candidate_count": len(result.payload["ranked_candidates"]),
                "top_candidate": None
                if top is None
                else {
                    "name": top["candidate_spec"]["name"],
                    "score": top["teacher_official_style_score"]["total_score"],
                    "composite_score": top["ranking_metrics"]["composite_score"],
                    "near_perfect": top["near_perfect"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
