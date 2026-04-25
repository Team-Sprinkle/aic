"""Run teacher candidate search and ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.randomizer import AicEnvRandomizer
from aic_gym_gz.run_cheatcode_gym_20hz_video import _start_official_bringup, _stop_process_tree
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


def _rollout_fps_from_ticks(ticks_per_step: int) -> float:
    return 1.0 / (max(int(ticks_per_step), 1) * 0.002)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prepare-official-scene", action="store_true")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_search.json")
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--live-timeout", type=float, default=20.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=90.0)
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="transport")
    parser.add_argument(
        "--image-observation-mode",
        choices=("artifact_validation", "async_training"),
        default="async_training",
    )
    parser.add_argument(
        "--state-observation-mode",
        choices=("honest_live", "synthetic_training"),
        default="synthetic_training",
    )
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--near-perfect-threshold", type=float, default=90.0)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--planner-candidate-count", type=int, default=4)
    parser.add_argument("--local-perturbation-count", type=int, default=4)
    parser.add_argument("--candidate-segment-limit", type=int, default=8)
    parser.add_argument("--refinement-segment-limit", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=512)
    parser.add_argument("--disable-probes", action="store_true")
    parser.add_argument("--planner-backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=_optional_float, default=None)
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
    image_shape = (int(args.image_height), int(args.image_width), 3)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    launch_process = None
    if args.prepare_official_scene:
        if not args.live:
            raise RuntimeError("--prepare-official-scene requires --live.")
        scenario = AicEnvRandomizer(enable_randomization=False).sample(
            seed=args.seed,
            trial_id=args.trial_id,
        )
        launch_process = _start_official_bringup(
            scenario,
            log_path=output_path.parent / "official_bringup.log",
        )

    factory = (
        (
            lambda: make_live_env(
                enable_randomization=False,
                include_images=include_images,
                ticks_per_step=int(args.ticks_per_step),
                attach_to_existing=bool(args.prepare_official_scene),
                transport_backend=args.transport_backend,
                timeout=float(args.live_timeout),
                attach_ready_timeout=float(args.attach_ready_timeout),
                image_shape=image_shape,
                image_observation_mode=args.image_observation_mode,
                state_observation_mode=args.state_observation_mode,
            )
        )
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
            planner_candidate_count=args.planner_candidate_count,
            local_perturbation_count=args.local_perturbation_count,
            candidate_segment_limit=args.candidate_segment_limit,
            refinement_segment_limit=args.refinement_segment_limit,
            enable_probes=not args.disable_probes,
            planner_output_mode=args.planner_output_mode,
            prefer_live_scene_overview=args.prefer_live_scene_overview,
            hold_ticks_per_action=int(args.ticks_per_step),
            max_env_steps=int(args.max_env_steps),
        ),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = search.run(
            seed=args.seed,
            trial_id=args.trial_id,
            output_path=output_path,
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
                fps=_rollout_fps_from_ticks(args.ticks_per_step),
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
    finally:
        _stop_process_tree(launch_process)
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
