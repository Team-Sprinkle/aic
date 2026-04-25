"""Run a mock planner teacher rollout and save a replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher import AgentTeacherController, TeacherConfig, run_teacher_rollout
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder, build_run_name, default_video_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--ticks-per-step", type=int, default=128)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--enable-randomization", action="store_true")
    parser.add_argument("--world-path", default=None)
    parser.add_argument(
        "--image-observation-mode",
        choices=("artifact_validation", "async_training"),
        default="async_training",
    )
    parser.add_argument(
        "--state-observation-mode",
        choices=("honest_live", "synthetic_training"),
        default="honest_live",
    )
    parser.add_argument(
        "--observation-transport-override",
        choices=("auto", "one_shot", "persistent"),
        default="persistent",
    )
    parser.add_argument("--output", default="aic_gym_gz/artifacts/teacher_rollout.json")
    parser.add_argument("--planner-backend", choices=("mock", "openai"), default="mock")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=float, default=0.1)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=2)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=8)
    parser.add_argument("--openai-global-max-calls-per-episode", type=int, default=5)
    parser.add_argument("--openai-cache-dir", default=None)
    parser.add_argument("--openai-trace-dir", default=None)
    parser.add_argument("--enable-global-guidance", action="store_true")
    parser.add_argument("--global-plan-interval-segments", type=int, default=2)
    parser.add_argument(
        "--planner-output-mode",
        choices=("absolute_cartesian_waypoint", "delta_cartesian_waypoint", "native_6d_action"),
        default="absolute_cartesian_waypoint",
    )
    parser.add_argument("--prefer-live-scene-overview", action="store_true")
    parser.add_argument("--max-planner-calls", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--video-dir", default=None)
    args = parser.parse_args()

    include_images = bool(args.include_images or not args.disable_video)
    image_shape = (int(args.image_height), int(args.image_width), 3)
    if include_images and not args.live:
        raise RuntimeError(
            "Real images/video require --live. "
            "The default mock env does not provide real wrist-camera frames."
        )
    env = (
        make_live_env(
            include_images=include_images,
            enable_randomization=bool(args.enable_randomization),
            ticks_per_step=args.ticks_per_step,
            world_path=args.world_path,
            image_shape=image_shape,
            image_observation_mode=args.image_observation_mode,
            state_observation_mode=args.state_observation_mode,
            observation_transport_override=args.observation_transport_override,
        )
        if args.live
        else make_default_env(
            include_images=include_images,
            enable_randomization=True,
            ticks_per_step=args.ticks_per_step,
            image_shape=image_shape,
        )
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
                    global_max_calls_per_episode=args.openai_global_max_calls_per_episode,
                    enable_global_guidance=args.enable_global_guidance,
                    cache_dir=args.openai_cache_dir,
                    trace_dir=args.openai_trace_dir,
                )
            )
        )
        controller = AgentTeacherController(
            planner=planner,
            config=TeacherConfig(
                candidate_plan_count=3,
                max_planner_calls_per_episode=args.max_planner_calls,
                max_env_steps=args.max_env_steps,
                run_until_env_done=True,
                hold_ticks_per_action=args.ticks_per_step,
                planner_output_mode=args.planner_output_mode,
                prefer_live_scene_overview=args.prefer_live_scene_overview,
                enable_global_guidance=args.enable_global_guidance,
                global_plan_interval_segments=args.global_plan_interval_segments,
            ),
        )
        run_name = build_run_name(prefix="teacher_rollout", seed=args.seed, trial_id=args.trial_id)
        video_recorder = HeadlessTrajectoryVideoRecorder(
            output_dir=Path(args.video_dir) if args.video_dir else default_video_output_dir(run_name=run_name),
            enabled=not args.disable_video,
            require_real_wrist_images=not args.disable_video,
            require_live_overview=not args.disable_video,
        )
        result = run_teacher_rollout(
            env=env,
            controller=controller,
            seed=args.seed,
            trial_id=args.trial_id,
            output_path=Path(args.output),
            trajectory_recorder=video_recorder,
        )
        video_summary = video_recorder.close()
        print(
            json.dumps(
                to_jsonable(
                    {
                        "output": str(result.output_path),
                        "video_output": video_summary,
                        "trial_id": result.artifact.metadata["trial_id"],
                        "task_id": result.artifact.metadata["task_id"],
                        "planner_backend": result.artifact.metadata["planner_backend"],
                        "segment_count": len(result.artifact.trajectory_segments),
                        "step_count": len(result.artifact.step_logs),
                        "probe_count": len(result.artifact.probe_results),
                        "final_distance_to_target": result.artifact.final_info.get("distance_to_target"),
                        "data_quality": result.artifact.metadata.get("data_quality"),
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
