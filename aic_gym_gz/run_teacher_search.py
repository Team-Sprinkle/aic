"""Run teacher candidate search and ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher.replay import TeacherReplayArtifact
from aic_gym_gz.teacher.search import TeacherCandidateSearch, TeacherSearchConfig
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import (
    HeadlessTrajectoryVideoRecorder,
    build_run_name,
    default_video_output_dir,
    record_teacher_artifact_replay,
)


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "auto"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_search.json")
    parser.add_argument("--near-perfect-threshold", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--planner-backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=_optional_float, default=0.1)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=8)
    parser.add_argument("--openai-max-calls-per-search", type=int, default=64)
    parser.add_argument("--openai-global-max-calls-per-episode", type=int, default=5)
    parser.add_argument("--openai-cache-dir", default=None)
    parser.add_argument("--enable-global-guidance", action="store_true")
    parser.add_argument(
        "--planner-output-mode",
        choices=("absolute_cartesian_waypoint", "delta_cartesian_waypoint", "native_6d_action"),
        default="absolute_cartesian_waypoint",
    )
    parser.add_argument("--prefer-live-scene-overview", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--video-dir", default=None)
    args = parser.parse_args()

    include_images = bool(args.include_images or not args.disable_video)
    if include_images and not args.live:
        raise RuntimeError(
            "Real images/video require --live. "
            "The default mock env does not provide real wrist-camera frames."
        )
    factory = (
        (lambda: make_live_env(enable_randomization=False, include_images=include_images))
        if args.live
        else (lambda: make_default_env(enable_randomization=True, include_images=include_images))
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
                    global_max_calls_per_episode=args.openai_global_max_calls_per_episode,
                    enable_global_guidance=args.enable_global_guidance,
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
            planner_output_mode=args.planner_output_mode,
            prefer_live_scene_overview=args.prefer_live_scene_overview,
        ),
    )
    result = search.run(
        seed=args.seed,
        trial_id=args.trial_id,
        output_path=Path(args.output),
    )
    top = result.payload["top_candidates"][0] if result.payload["top_candidates"] else None
    video_summary = None
    if top is not None and not args.disable_video:
        artifact_dict = top["artifact"]
        artifact = TeacherReplayArtifact(
            metadata=artifact_dict["metadata"],
            trajectory_segments=artifact_dict["trajectory_segments"],
            probe_results=artifact_dict.get("probe_results", []),
            planner_candidates=artifact_dict.get("planner_candidates", []),
            step_logs=artifact_dict.get("step_logs", []),
            final_info=artifact_dict.get("final_info", {}),
            limitations=artifact_dict.get("limitations", []),
        )
        run_name = build_run_name(prefix="teacher_search_top1", seed=args.seed, trial_id=args.trial_id)
        recorder = HeadlessTrajectoryVideoRecorder(
            output_dir=Path(args.video_dir) if args.video_dir else default_video_output_dir(run_name=run_name),
            enabled=True,
            require_real_wrist_images=True,
            require_live_overview=True,
        )
        env = factory()
        try:
            replay_final_info = record_teacher_artifact_replay(
                env=env,
                artifact=artifact,
                recorder=recorder,
                seed=args.seed,
                trial_id=args.trial_id,
            )
        finally:
            env.close()
        video_summary = recorder.close()
        video_summary["replay_final_info"] = replay_final_info
    print(
        json.dumps(
            to_jsonable(
                {
                    "output": str(result.output_path),
                    "video_output": video_summary,
                    "candidate_count": len(result.payload["ranked_candidates"]),
                    "top_candidate": None
                    if top is None
                    else {
                        "name": top["candidate_spec"]["name"],
                        "score": top["teacher_official_style_score"]["total_score"],
                        "composite_score": top["ranking_metrics"]["composite_score"],
                        "near_perfect": top["near_perfect"],
                    },
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
