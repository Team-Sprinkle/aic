"""Run a fully inspectable teacher rollout and save a manifest of all artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.io import RosCameraSubscriber
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher import AgentTeacherController, TeacherConfig, run_teacher_rollout
from aic_gym_gz.teacher.analysis import analyze_rollout_artifact
from aic_gym_gz.teacher.dataset_export import RolloutDatasetFrame, export_rollout_lerobot_dataset
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder, build_run_name, default_video_output_dir


def _optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "auto"}:
        return None
    return float(value)


def _rollout_fps_from_ticks(ticks_per_step: int) -> int:
    return max(1, int(round(1.0 / (max(int(ticks_per_step), 1) * 0.002))))


def _flatten_step_table(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    segment_step_ranges: list[tuple[int, int]] = []
    cursor = 0
    for segment_index, segment in enumerate(artifact.get("trajectory_segments", [])):
        point_count = len(segment.get("points", []))
        segment_step_ranges.append((segment_index, cursor + point_count))
        cursor += point_count
    for step_index, step in enumerate(artifact.get("step_logs", [])):
        segment_index = 0
        local_step_index = step_index
        running = 0
        for seg_index, segment in enumerate(artifact.get("trajectory_segments", [])):
            point_count = len(segment.get("points", []))
            if step_index < running + point_count:
                segment_index = seg_index
                local_step_index = step_index - running
                break
            running += point_count
        obs = dict(step.get("observation_summary", {}))
        traj = dict(step.get("trajectory_point", {}))
        flattened.append(
            {
                "step_index": step_index,
                "segment_index": segment_index,
                "segment_local_step_index": local_step_index,
                "phase": step.get("phase"),
                "sim_tick": step.get("sim_tick"),
                "sim_time": step.get("sim_time"),
                "reward": step.get("reward"),
                "terminated": step.get("terminated"),
                "truncated": step.get("truncated"),
                "planner_rationale": step.get("planner_rationale"),
                "action": traj.get("action"),
                "target_tcp_pose": traj.get("target_tcp_pose"),
                "target_dt": traj.get("dt"),
                "tcp_pose": obs.get("tcp_pose"),
                "tcp_velocity": obs.get("tcp_velocity"),
                "plug_pose": obs.get("plug_pose"),
                "target_port_pose": obs.get("target_port_pose"),
                "target_port_entrance_pose": obs.get("target_port_entrance_pose"),
                "wrench": obs.get("wrench"),
                "wrench_timestamp": obs.get("wrench_timestamp"),
                "plug_to_port_relative": obs.get("plug_to_port_relative"),
                "score_geometry": obs.get("score_geometry"),
                "controller_tcp_pose": obs.get("controller_tcp_pose"),
                "controller_reference_tcp_pose": obs.get("controller_reference_tcp_pose"),
                "controller_tcp_error": obs.get("controller_tcp_error"),
                "controller_target_mode": obs.get("controller_target_mode"),
                "data_quality": step.get("data_quality"),
            }
        )
    return flattened


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")


def _preflight_live_attach(
    *,
    timeout_s: float,
    require_images: bool,
    image_shape: tuple[int, int, int],
) -> dict[str, Any]:
    from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig
    from aic_gazebo_env.protocol import GetObservationRequest

    deadline = time.monotonic() + max(float(timeout_s), 1.0)
    last_error: Exception | None = None
    last_observation_summary: dict[str, Any] | None = None

    camera_subscriber = (
        RosCameraSubscriber(node_name="aic_gym_gz_camera_preflight", image_shape=image_shape)
        if require_images
        else None
    )
    if camera_subscriber is not None:
        camera_subscriber.start()
    try:
        while time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 1.0)
            client = GazeboCliClient(
                GazeboCliClientConfig(
                    executable="gz",
                    world_path="aic_description/world/aic.sdf",
                    timeout=min(max(remaining, 5.0), 20.0),
                    world_name="aic_world",
                    source_entity_name="ati/tool_link",
                    target_entity_name="tabletop",
                    transport_backend="cli",
                    observation_transport="one_shot",
                )
            )
            try:
                response = client.get_observation(GetObservationRequest())
                observation = dict(response.observation)
                entities_by_name = observation.get("entities_by_name") or {}
                source_ready = isinstance(entities_by_name.get("wrist_3_link"), dict) or isinstance(
                    entities_by_name.get("ati/tool_link"), dict
                )
                target_ready = isinstance(entities_by_name.get("tabletop"), dict)
                last_observation_summary = {
                    "entity_count": observation.get("entity_count"),
                    "joint_count": observation.get("joint_count"),
                    "step_count": observation.get("step_count"),
                    "entities": sorted(entities_by_name.keys()),
                    "source_ready": source_ready,
                    "target_ready": target_ready,
                }
                images_ready = True
                if camera_subscriber is not None:
                    wait_s = min(5.0, max(deadline - time.monotonic(), 0.1))
                    images_ready = camera_subscriber.wait_until_ready(timeout_s=wait_s)
                    if images_ready:
                        images, _, _ = camera_subscriber.latest_images()
                        images_ready = all(int(image.sum()) > 0 for image in images.values())
                if source_ready and target_ready and images_ready:
                    return {
                        "observation_summary": last_observation_summary,
                        "images_ready": images_ready,
                    }
            except Exception as exc:
                last_error = exc
            finally:
                client.close()
            time.sleep(0.5)
    finally:
        if camera_subscriber is not None:
            camera_subscriber.close()
    raise RuntimeError(
        "Live attach preflight failed before env.reset(). "
        f"last_error={last_error}, last_observation_summary={last_observation_summary}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trial-id", default=None)
    parser.add_argument("--ticks-per-step", type=int, default=128)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--attach-to-existing", action="store_true")
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="transport")
    parser.add_argument("--live-timeout", type=float, default=20.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=60.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--planner-backend", choices=("openai", "mock"), default="openai")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=_optional_float, default=None)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=1)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=96)
    parser.add_argument("--max-planner-calls", type=int, default=10)
    parser.add_argument("--enable-global-guidance", action="store_true")
    parser.add_argument("--candidate-plan-count", type=int, default=2)
    parser.add_argument("--segment-limit", type=int, default=96)
    parser.add_argument("--max-env-steps", type=int, default=256)
    parser.add_argument("--prefer-live-scene-overview", action="store_true")
    parser.add_argument("--planner-output-mode", choices=("absolute_cartesian_waypoint", "delta_cartesian_waypoint", "native_6d_action"), default="absolute_cartesian_waypoint")
    parser.add_argument("--dataset-repo-id", default="local/aic_gym_gz_teacher_rollout")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-no-videos", action="store_true")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not args.live:
        raise RuntimeError(
            "run_teacher_rollout_inspect requires --live so the artifact bundle contains real wrist and overview imagery."
        )
    run_name = args.run_name or build_run_name(prefix="teacher_rollout_inspect", seed=args.seed, trial_id=args.trial_id)
    output_dir = Path(args.output_dir) if args.output_dir else Path("aic_gym_gz/artifacts/inspect_runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "openai_traces"
    rollout_path = output_dir / "teacher_rollout_artifact.json"
    analysis_json_path = output_dir / "rollout_analysis.json"
    analysis_md_path = output_dir / "rollout_analysis.md"
    step_table_path = output_dir / "step_table.jsonl"
    manifest_path = output_dir / "manifest.json"
    video_dir = output_dir / "videos"
    dataset_root = Path(args.dataset_root) if args.dataset_root else output_dir / "lerobot_dataset"
    dataset_frames: list[RolloutDatasetFrame] = []

    include_images = True
    image_shape = (int(args.image_height), int(args.image_width), 3)
    preflight_summary = None
    if args.live and args.attach_to_existing and not args.skip_preflight:
        preflight_summary = _preflight_live_attach(
            timeout_s=args.attach_ready_timeout,
            require_images=include_images,
            image_shape=image_shape,
        )
    env = (
        make_live_env(
            include_images=include_images,
            enable_randomization=False,
            ticks_per_step=args.ticks_per_step,
            attach_to_existing=args.attach_to_existing,
            transport_backend=args.transport_backend,
            timeout=args.live_timeout,
            attach_ready_timeout=args.attach_ready_timeout,
            image_shape=image_shape,
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
                    max_calls_per_episode=min(args.openai_max_calls_per_episode, args.max_planner_calls),
                    enable_global_guidance=args.enable_global_guidance,
                    global_temperature=None,
                    use_cache=False,
                    trace_dir=str(trace_dir),
                )
            )
        )
        controller = AgentTeacherController(
            planner=planner,
            config=TeacherConfig(
                candidate_plan_count=args.candidate_plan_count,
                segment_limit=args.segment_limit,
                max_env_steps=args.max_env_steps,
                run_until_env_done=True,
                max_planner_calls_per_episode=args.max_planner_calls,
                hold_ticks_per_action=args.ticks_per_step,
                planner_output_mode=args.planner_output_mode,
                prefer_live_scene_overview=args.prefer_live_scene_overview,
                enable_global_guidance=args.enable_global_guidance,
            ),
        )
        recorder = HeadlessTrajectoryVideoRecorder(
            output_dir=video_dir,
            enabled=True,
            require_real_wrist_images=True,
            require_live_overview=True,
        )
        def _dataset_callback(
            *,
            kind: str,
            observation: dict[str, Any],
            reward: float,
            terminated: bool,
            truncated: bool,
            info: dict[str, Any],
            state: Any,
            action: Any,
            planner_rationale: str | None,
            phase: str | None,
        ) -> None:
            del state
            dataset_frames.append(
                RolloutDatasetFrame(
                    kind=kind,
                    observation=dict(observation),
                    action=[float(value) for value in list(action)],
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=dict(info),
                    planner_rationale=planner_rationale,
                    phase=phase,
                )
            )
        result = run_teacher_rollout(
            env=env,
            controller=controller,
            seed=args.seed,
            trial_id=args.trial_id,
            output_path=rollout_path,
            trajectory_recorder=recorder,
            step_callback=_dataset_callback,
        )
        video_summary = recorder.close()
    finally:
        env.close()

    artifact = result.artifact.to_dict()
    analysis = analyze_rollout_artifact(artifact)
    _write_json(analysis_json_path, analysis.summary)
    analysis_md_path.write_text(analysis.markdown, encoding="utf-8")
    step_table = _flatten_step_table(artifact)
    _write_jsonl(step_table_path, step_table)
    dataset_result = export_rollout_lerobot_dataset(
        dataset_frames,
        repo_id=args.dataset_repo_id,
        output_root=dataset_root,
        single_task="Insert cable into target port",
        fps=_rollout_fps_from_ticks(args.ticks_per_step),
        use_videos=not args.dataset_no_videos,
        metadata={
            "run_name": run_name,
            "trial_id": result.artifact.metadata.get("trial_id"),
            "task_id": result.artifact.metadata.get("task_id"),
            "planner_backend": result.artifact.metadata.get("planner_backend"),
            "ticks_per_step": int(args.ticks_per_step),
            "max_planner_calls": int(args.max_planner_calls),
            "max_env_steps": int(args.max_env_steps),
        },
    )

    trace_files = sorted(str(path) for path in trace_dir.glob("*.json"))
    manifest = {
        "run_name": run_name,
        "seed": args.seed,
        "trial_id": args.trial_id,
        "ticks_per_step": args.ticks_per_step,
        "planner_backend": args.planner_backend,
        "live_env_requested": args.live,
        "artifacts": {
            "rollout_artifact_json": str(rollout_path),
            "rollout_analysis_json": str(analysis_json_path),
            "rollout_analysis_markdown": str(analysis_md_path),
            "step_table_jsonl": str(step_table_path),
            "openai_trace_dir": str(trace_dir),
            "openai_trace_files": trace_files,
            "video_dir": str(video_dir),
            "video_summary": video_summary,
            "lerobot_dataset_dir": str(dataset_result.dataset_path),
            "lerobot_dataset_metadata": str(dataset_result.metadata_path),
        },
        "notes": {
            "rollout_artifact_json": "Full nested artifact including metadata, planner_candidates, trajectory_segments, step_logs, final_info.",
            "step_table_jsonl": "Flattened one-row-per-env.step table with action, target pose, current pose, wrench, controller fields, and score geometry.",
            "openai_trace_files": "One JSON file per OpenAI planner/global-guidance call including sanitized request payload, full response payload, parsed output, and response_text.",
            "lerobot_dataset_dir": "LeRobot-format rollout dataset aligned to the exp/data policy-recorder schema, with added sim_time/sim_tick and target/plug pose fields.",
        },
        "preflight_summary": preflight_summary,
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
