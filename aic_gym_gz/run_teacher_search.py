"""Run teacher candidate search and ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher.search import TeacherCandidateSearch, TeacherSearchConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_search.json")
    parser.add_argument("--near-perfect-threshold", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    factory = (
        (lambda: make_live_env(enable_randomization=False, include_images=False))
        if args.live
        else (lambda: make_default_env(enable_randomization=True, include_images=False))
    )
    search = TeacherCandidateSearch(
        env_factory=factory,
        planner_factory=lambda: DeterministicMockPlannerBackend(),
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
                    "score": top["official_style_score"]["total_score"],
                    "near_perfect": top["near_perfect"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
