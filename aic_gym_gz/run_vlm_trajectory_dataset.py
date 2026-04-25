"""Generate VLM-planned trajectory artifacts in the organized inspect-run layout."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import signal
import shutil
from typing import Any

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.official_scene import export_training_world_for_scenario, resolve_official_setup_script
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.scenario import AicScenario, RailEntity, TaskBoardScenario, TaskDefinition, load_trials
from aic_gym_gz.teacher import AgentTeacherController, TeacherConfig, run_teacher_rollout
from aic_gym_gz.teacher.analysis import analyze_rollout_artifact
from aic_gym_gz.teacher.dataset_export import RolloutDatasetFrame, export_rollout_lerobot_dataset
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder
from aic_gym_gz.vlm_feedback import run_final_gpt5_feedback

from aic_gym_gz.run_teacher_rollout_inspect import (
    _build_performance_summary,
    _flatten_step_table,
    _rollout_fps_from_ticks,
    _write_json,
)


TASK_BOARD_REFERENCE_URL = "https://github.com/intrinsic-dev/aic/blob/main/docs/task_board_description.md"
CHEATCODE_DATASET_REFERENCE_URL = (
    "https://github.com/Team-Sprinkle/aic/blob/exp/data-organize/"
    "aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py"
)


@dataclass(frozen=True)
class FixedScenarioProvider:
    scenario: AicScenario

    def sample(self, *, seed: int | None = None, trial_id: str | None = None) -> AicScenario:
        del seed, trial_id
        return self.scenario


def _parse_optional_float(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized in {"none", "null", "auto"}:
        return None
    return float(value)


def _run_dir(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if args.task_family == "sfp_to_nic":
        group = f"nic_cards_{args.nic_cards}"
    else:
        group = f"sc_ports_{args.sc_ports}"
    return root / args.task_family / "vlm_planner" / group / f"n{args.trajectory_index}"


def _build_scenario(args: argparse.Namespace) -> tuple[AicScenario, dict[str, Any]]:
    if args.task_family == "sfp_to_nic":
        return _build_sfp_to_nic_scenario(args)
    return _build_sc_to_sc_scenario(args)


def _build_sfp_to_nic_scenario(args: argparse.Namespace) -> tuple[AicScenario, dict[str, Any]]:
    if not 1 <= args.nic_cards <= 5:
        raise ValueError("--nic-cards must be between 1 and 5.")
    if not 0 <= args.target_nic_index < args.nic_cards:
        raise ValueError("--target-nic-index must identify one of the inserted NIC cards.")
    if args.target_nic_port not in (0, 1):
        raise ValueError("--target-nic-port must be 0 or 1.")

    base = load_trials()["trial_1"]
    source_task = next(iter(base.tasks.values()))
    nic_rails = {
        key: RailEntity(
            present=index < args.nic_cards,
            name=f"nic_card_{index}" if index < args.nic_cards else None,
            translation=base.task_board.nic_rails[key].translation
            if base.task_board.nic_rails[key].present
            else 0.036,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )
        for index, key in enumerate(sorted(base.task_board.nic_rails))
    }
    task = TaskDefinition(
        task_id=source_task.task_id,
        cable_type=source_task.cable_type,
        cable_name=source_task.cable_name,
        plug_type="sfp",
        plug_name="sfp_tip",
        port_type="sfp",
        port_name=f"sfp_port_{args.target_nic_port}",
        target_module_name=f"nic_card_mount_{args.target_nic_index}",
        time_limit_s=source_task.time_limit_s,
    )
    scenario = AicScenario(
        trial_id=f"sfp_to_nic_nic_cards_{args.nic_cards}",
        task_board=TaskBoardScenario(
            pose_xyz_rpy=base.task_board.pose_xyz_rpy,
            nic_rails=nic_rails,
            sc_rails=base.task_board.sc_rails,
            mount_rails=base.task_board.mount_rails,
        ),
        cables=base.cables,
        tasks={task.task_id: task},
        metadata={
            **base.metadata,
            "organized_task_family": "sfp_to_nic",
            "dataset_layout": "inspect_runs/sfp_to_nic/vlm_planner/nic_cards_{count}/n{trajectory}",
            "task_board_reference_url": TASK_BOARD_REFERENCE_URL,
        },
    )
    task_metadata = {
        "task_family": "sfp_to_nic",
        "inserted_nic_card_count": int(args.nic_cards),
        "present_nic_cards": [f"nic_card_mount_{index}" for index in range(args.nic_cards)],
        "right_card_index": int(args.target_nic_index),
        "right_card": f"nic_card_mount_{args.target_nic_index}",
        "right_port_number": int(args.target_nic_port),
        "right_port_name": f"sfp_port_{args.target_nic_port}",
        "plug_name": "sfp_tip",
        "zone_description": "Task board Zone 1 NIC card mount; each NIC card has two SFP ports.",
    }
    return scenario, task_metadata


def _build_sc_to_sc_scenario(args: argparse.Namespace) -> tuple[AicScenario, dict[str, Any]]:
    if not 1 <= args.sc_ports <= 2:
        raise ValueError("--sc-ports must be between 1 and 2 for the current official launch/runtime mapping.")
    if not 0 <= args.target_sc_index < args.sc_ports:
        raise ValueError("--target-sc-index must identify one of the inserted SC ports.")

    base = load_trials()["trial_3"]
    source_task = next(iter(base.tasks.values()))
    sc_rails = {
        key: RailEntity(
            present=index < args.sc_ports,
            name=f"sc_mount_{index}" if index < args.sc_ports else None,
            translation=base.task_board.sc_rails[key].translation
            if base.task_board.sc_rails[key].present
            else 0.042,
            roll=0.0,
            pitch=0.0,
            yaw=base.task_board.sc_rails[key].yaw if base.task_board.sc_rails[key].present else 0.0,
        )
        for index, key in enumerate(sorted(base.task_board.sc_rails))
    }
    task = TaskDefinition(
        task_id=source_task.task_id,
        cable_type=source_task.cable_type,
        cable_name=source_task.cable_name,
        plug_type="sc",
        plug_name="sc_tip",
        port_type="sc",
        port_name="sc_port_base",
        target_module_name=f"sc_port_{args.target_sc_index}",
        time_limit_s=source_task.time_limit_s,
    )
    scenario = AicScenario(
        trial_id=f"sc_to_sc_sc_ports_{args.sc_ports}",
        task_board=TaskBoardScenario(
            pose_xyz_rpy=base.task_board.pose_xyz_rpy,
            nic_rails=base.task_board.nic_rails,
            sc_rails=sc_rails,
            mount_rails=base.task_board.mount_rails,
        ),
        cables=base.cables,
        tasks={task.task_id: task},
        metadata={
            **base.metadata,
            "organized_task_family": "sc_to_sc",
            "dataset_layout": "inspect_runs/sc_to_sc/vlm_planner/sc_ports_{count}/n{trajectory}",
            "task_board_reference_url": TASK_BOARD_REFERENCE_URL,
        },
    )
    task_metadata = {
        "task_family": "sc_to_sc",
        "inserted_sc_port_count": int(args.sc_ports),
        "present_sc_ports": [f"sc_port_{index}" for index in range(args.sc_ports)],
        "right_sc_port_index": int(args.target_sc_index),
        "right_sc_port": f"sc_port_{args.target_sc_index}",
        "right_port_name": "sc_port_base",
        "plug_name": "sc_tip",
        "zone_description": "Task board Zone 2 SC ports; current runtime/launch mapping exposes sc_port_0 and sc_port_1.",
    }
    return scenario, task_metadata


def _planner(args: argparse.Namespace, trace_dir: Path):
    if args.planner_backend == "mock":
        return DeterministicMockPlannerBackend()
    return OpenAIPlannerBackend(
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


def _dataset_callback_factory(dataset_frames: list[RolloutDatasetFrame]):
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

    return _dataset_callback


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(to_jsonable(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


@contextmanager
def _wall_clock_timeout(timeout_s: float | None, *, label: str):
    if timeout_s is None or timeout_s <= 0:
        yield
        return
    previous_handler = signal.getsignal(signal.SIGALRM)

    def _handle_timeout(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {timeout_s:.1f}s wall-clock timeout.")

    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


def _feedback_manifest_summary(feedback: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(feedback, dict):
        return None
    return {
        "path": feedback.get("path"),
        "model": feedback.get("model"),
        "parse_error": feedback.get("parse_error"),
        "error": feedback.get("error"),
        "request_frame_count": feedback.get("request_frame_count"),
        "feedback": feedback.get("feedback"),
    }


def _final_port_frame_metrics(artifact: dict[str, Any]) -> dict[str, Any]:
    step_logs = list(artifact.get("step_logs", []))
    if not step_logs:
        return {}
    final_obs = dict(step_logs[-1].get("observation_summary", {}))
    plug = final_obs.get("plug_pose")
    target = final_obs.get("target_port_pose")
    entrance = final_obs.get("target_port_entrance_pose")
    if plug is None or target is None or entrance is None:
        return {}
    import numpy as np

    plug_xyz = np.asarray(plug[:3], dtype=np.float64)
    target_xyz = np.asarray(target[:3], dtype=np.float64)
    entrance_xyz = np.asarray(entrance[:3], dtype=np.float64)
    axis = target_xyz - entrance_xyz
    required_depth = float(np.linalg.norm(axis))
    axis_unit = axis / required_depth if required_depth > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offset = plug_xyz - entrance_xyz
    signed_axial_depth = float(np.dot(offset, axis_unit))
    lateral_vector = offset - signed_axial_depth * axis_unit
    wrench = np.asarray(final_obs.get("wrench", [0.0] * 6), dtype=np.float64).reshape(-1)
    force_xyz = np.zeros(3, dtype=np.float64) if wrench.size < 3 else wrench[:3]
    return {
        "axis_unit_world": [float(value) for value in axis_unit.tolist()],
        "required_axial_depth_m": required_depth,
        "signed_axial_depth_m": signed_axial_depth,
        "axial_depth_residual_m": float(required_depth - signed_axial_depth),
        "lateral_error_m": float(np.linalg.norm(lateral_vector)),
        "lateral_error_vector_world_m": [float(value) for value in lateral_vector.tolist()],
        "axial_force_n": float(np.dot(force_xyz, axis_unit)),
        "force_norm_n": float(np.linalg.norm(force_xyz)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-family", choices=("sfp_to_nic", "sc_to_sc"), required=True)
    parser.add_argument("--output-root", default="aic_gym_gz/artifacts/inspect_runs")
    parser.add_argument("--trajectory-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ticks-per-step", type=int, default=25)
    parser.add_argument("--debug-mock-backend", action="store_true")
    parser.add_argument("--live-timeout", type=float, default=30.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=90.0)
    parser.add_argument("--rollout-wall-timeout", type=float, default=900.0)
    parser.add_argument("--transport-backend", choices=("auto", "transport", "cli"), default="auto")
    parser.add_argument(
        "--image-observation-mode",
        choices=("artifact_validation", "async_training"),
        default="async_training",
    )
    parser.add_argument(
        "--state-observation-mode",
        choices=("honest_live",),
        default="honest_live",
    )
    parser.add_argument("--nic-cards", type=int, default=2)
    parser.add_argument("--target-nic-index", type=int, default=1)
    parser.add_argument("--target-nic-port", type=int, default=0)
    parser.add_argument("--sc-ports", type=int, default=2)
    parser.add_argument("--target-sc-index", type=int, default=1)
    parser.add_argument("--planner-backend", choices=("openai", "mock"), default="openai")
    parser.add_argument("--openai-model", default="gpt-5.4-mini")
    parser.add_argument("--openai-temperature", type=_parse_optional_float, default=None)
    parser.add_argument("--openai-timeout", type=float, default=20.0)
    parser.add_argument("--openai-max-retries", type=int, default=1)
    parser.add_argument("--openai-max-calls-per-episode", type=int, default=96)
    parser.add_argument("--max-planner-calls", type=int, default=10)
    parser.add_argument("--candidate-plan-count", type=int, default=2)
    parser.add_argument("--segment-limit", type=int, default=96)
    parser.add_argument("--max-env-steps", type=int, default=512)
    parser.add_argument("--enable-global-guidance", action="store_true")
    parser.add_argument("--planner-output-mode", choices=("absolute_cartesian_waypoint",), default="absolute_cartesian_waypoint")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--overview-capture-stride", type=int, default=1)
    parser.add_argument("--enable-final-gpt5-feedback", action="store_true")
    parser.add_argument("--final-feedback-model", default="gpt-5")
    parser.add_argument("--final-feedback-timeout", type=float, default=120.0)
    parser.add_argument("--final-feedback-max-output-tokens", type=int, default=4000)
    parser.add_argument("--final-feedback-frames-per-angle", type=int, default=10)
    parser.add_argument("--dataset-repo-id", default="local/aic_gym_gz_vlm_trajectory")
    parser.add_argument("--dataset-no-videos", action="store_true")
    parser.add_argument("--skip-dataset-export", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = _run_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_dir = output_dir / "openai_traces"
    video_dir = output_dir / "videos"
    dataset_root = output_dir / "lerobot_dataset"
    rollout_path = output_dir / "teacher_rollout_artifact.json"
    analysis_json_path = output_dir / "rollout_analysis.json"
    analysis_md_path = output_dir / "rollout_analysis.md"
    step_table_path = output_dir / "step_table.jsonl"
    performance_summary_path = output_dir / "performance_summary.json"
    manifest_path = output_dir / "manifest.json"
    task_metadata_path = output_dir / "task_metadata.json"
    run_name = str(output_dir.relative_to(Path(args.output_root)))
    for stale_dir in (trace_dir, video_dir, dataset_root, output_dir / "vlm_feedback_frames"):
        if stale_dir.exists():
            shutil.rmtree(stale_dir)
    for stale_file in (
        rollout_path,
        analysis_json_path,
        analysis_md_path,
        step_table_path,
        performance_summary_path,
        manifest_path,
        output_dir / "final_gpt5_vlm_feedback.json",
        output_dir / "generated_world.sdf",
        output_dir / "generated_world.launch.log",
    ):
        if stale_file.exists():
            stale_file.unlink()

    scenario, task_metadata = _build_scenario(args)
    _write_json(task_metadata_path, task_metadata)

    image_shape = (int(args.image_height), int(args.image_width), 3)
    generated_world_metadata = None
    if args.debug_mock_backend:
        env = make_default_env(
            include_images=False,
            enable_randomization=False,
            ticks_per_step=args.ticks_per_step,
            image_shape=image_shape,
        )
        real_video_required = False
        runtime_backend = "mock_debug"
    else:
        generated_world_path = output_dir / "generated_world.sdf"
        generated_world_metadata = export_training_world_for_scenario(
            scenario,
            output_path=generated_world_path,
            setup_script=resolve_official_setup_script(),
            timeout_s=max(float(args.attach_ready_timeout), 120.0),
        )
        env = make_live_env(
            include_images=True,
            enable_randomization=False,
            ticks_per_step=args.ticks_per_step,
            world_path=str(generated_world_path),
            attach_to_existing=False,
            transport_backend=args.transport_backend,
            timeout=args.live_timeout,
            attach_ready_timeout=args.attach_ready_timeout,
            image_shape=image_shape,
            live_mode="gazebo_training_fast",
            image_observation_mode=args.image_observation_mode,
            observation_transport_override="persistent",
            state_observation_mode=args.state_observation_mode,
            allow_synthetic_tcp_pose=False,
            allow_synthetic_plug_pose=False,
        )
        real_video_required = True
        runtime_backend = "scenario_gym_gz"
    env.randomizer = FixedScenarioProvider(scenario)  # type: ignore[assignment]
    dataset_frames: list[RolloutDatasetFrame] = []
    recorder = HeadlessTrajectoryVideoRecorder(
        output_dir=video_dir,
        enabled=True,
        fps=_rollout_fps_from_ticks(args.ticks_per_step),
        require_real_wrist_images=real_video_required,
        require_live_overview=real_video_required,
        prefer_live_overview_camera=real_video_required,
        allow_direct_overview_fetch=not real_video_required,
        overview_capture_stride=max(int(args.overview_capture_stride), 1),
    )
    result = None
    video_summary: dict[str, Any] | None = None
    try:
        controller = AgentTeacherController(
            planner=_planner(args, trace_dir),
            config=TeacherConfig(
                candidate_plan_count=args.candidate_plan_count,
                segment_limit=args.segment_limit,
                max_env_steps=args.max_env_steps,
                run_until_env_done=True,
                max_planner_calls_per_episode=args.max_planner_calls,
                hold_ticks_per_action=args.ticks_per_step,
                planner_backend_name=args.planner_backend,
                planner_output_mode=args.planner_output_mode,
                prefer_live_scene_overview=False,
                require_live_scene_overview=False,
                enable_global_guidance=args.enable_global_guidance,
                enable_close_range_handoff=True,
            ),
        )
        with _wall_clock_timeout(args.rollout_wall_timeout, label="teacher rollout"):
            result = run_teacher_rollout(
                env=env,
                controller=controller,
                seed=args.seed,
                trial_id=scenario.trial_id,
                output_path=rollout_path,
                trajectory_recorder=recorder,
                step_callback=_dataset_callback_factory(dataset_frames),
            )
    finally:
        try:
            video_summary = recorder.close()
        finally:
            env.close()

    if result is None:
        raise RuntimeError("Rollout failed before producing an artifact.")
    if video_summary is None:
        raise RuntimeError("Video recorder did not produce a summary.")

    artifact = result.artifact.to_dict()
    analysis = analyze_rollout_artifact(artifact)
    final_vlm_feedback_summary = None
    if args.enable_final_gpt5_feedback:
        try:
            final_vlm_feedback_summary = run_final_gpt5_feedback(
                run_dir=output_dir,
                model=args.final_feedback_model,
                timeout_s=float(args.final_feedback_timeout),
                max_output_tokens=int(args.final_feedback_max_output_tokens),
                frames_per_angle=int(args.final_feedback_frames_per_angle),
            )
        except Exception as exc:
            final_vlm_feedback_summary = {
                "error": str(exc),
                "model": args.final_feedback_model,
            }
        analysis.summary["final_gpt5_vlm_feedback"] = final_vlm_feedback_summary

    _write_json(analysis_json_path, analysis.summary)
    analysis_md_path.write_text(analysis.markdown, encoding="utf-8")
    _write_jsonl(step_table_path, _flatten_step_table(artifact))

    dataset_result = None
    if args.skip_dataset_export:
        dataset_export_status = {"skipped": True, "reason": "skip_dataset_export_flag"}
    else:
        try:
            if dataset_root.exists():
                shutil.rmtree(dataset_root)
            dataset_result = export_rollout_lerobot_dataset(
                dataset_frames,
                repo_id=args.dataset_repo_id,
                output_root=dataset_root,
                single_task=f"{args.task_family}: Insert {task_metadata['plug_name']} into target port",
                fps=_rollout_fps_from_ticks(args.ticks_per_step),
                use_videos=not args.dataset_no_videos,
                metadata={
                    "organized_output_dir": str(output_dir),
                    "task_metadata": task_metadata,
                    "hybrid_policy": "VLM coarse Cartesian planner + MinimumJerkSmoother + close-range insertion cheatcode handoff",
                    "task_board_reference_url": TASK_BOARD_REFERENCE_URL,
                    "cheatcode_dataset_reference_url": CHEATCODE_DATASET_REFERENCE_URL,
                },
            )
            dataset_export_status = {
                "skipped": False,
                "dataset_path": str(dataset_result.dataset_path),
                "metadata_path": str(dataset_result.metadata_path),
                "format": dataset_result.format,
            }
        except ModuleNotFoundError as exc:
            dataset_export_status = {
                "skipped": True,
                "reason": "missing_dataset_dependency",
                "error": str(exc),
            }

    trace_files = sorted(str(path) for path in trace_dir.glob("*.json"))
    summary_args = argparse.Namespace(
        live=not args.debug_mock_backend,
        live_mode=runtime_backend,
        state_observation_mode=args.state_observation_mode if not args.debug_mock_backend else "mock_state",
        planner_backend=args.planner_backend,
        openai_model=args.openai_model,
    )
    performance_summary = _build_performance_summary(
        run_name=run_name,
        args=summary_args,
        artifact=artifact,
        analysis_summary=analysis.summary,
        video_summary=video_summary,
        final_vlm_feedback_summary=final_vlm_feedback_summary,
        trace_files=trace_files,
    )
    performance_summary["task_metadata"] = task_metadata
    performance_summary["hybrid_policy"] = {
        "vlm_role": "Sparse global/coarse planning in Cartesian coordinates.",
        "optimizer_role": "MinimumJerkSmoother densely interpolates smooth Cartesian motion between VLM waypoints.",
        "cheatcode_handoff": "CloseRangeInsertionPolicy takes over for guarded insertion near the target or after planner budget exhaustion.",
    }
    performance_summary["final_port_frame_metrics"] = _final_port_frame_metrics(artifact)
    performance_summary["reference_urls"] = {
        "task_board_description": TASK_BOARD_REFERENCE_URL,
        "cheatcode_dataset_script": CHEATCODE_DATASET_REFERENCE_URL,
    }
    _write_json(performance_summary_path, performance_summary)

    manifest = {
        "run_name": run_name,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "runtime_backend": runtime_backend,
        "task_metadata": task_metadata,
        "artifacts": {
            "task_metadata_json": str(task_metadata_path),
            "rollout_artifact_json": str(rollout_path),
            "rollout_analysis_json": str(analysis_json_path),
            "rollout_analysis_markdown": str(analysis_md_path),
            "step_table_jsonl": str(step_table_path),
            "performance_summary_json": str(performance_summary_path),
            "openai_trace_dir": str(trace_dir),
            "openai_trace_files": trace_files,
            "video_dir": str(video_dir),
            "video_summary": video_summary,
            "lerobot_dataset_dir": None if dataset_result is None else str(dataset_result.dataset_path),
            "lerobot_dataset_metadata": None if dataset_result is None else str(dataset_result.metadata_path),
            "dataset_export_status": dataset_export_status,
            "final_gpt5_vlm_feedback": _feedback_manifest_summary(final_vlm_feedback_summary),
            "generated_world": generated_world_metadata,
        },
        "notes": {
            "videos": (
                "Videos are always generated. Dataset-quality runs require real Gazebo wrist and overview streams; "
                "diagnostic schematic renders are only available with --debug-mock-backend."
            ),
            "dataset_layout": "Matches inspect_runs/{task_family}/vlm_planner/{count_group}/n{trajectory_index}.",
        },
    }
    _write_json(manifest_path, manifest)
    print(json.dumps(to_jsonable(manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
