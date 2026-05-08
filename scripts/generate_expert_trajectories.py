#!/usr/bin/env python3
"""Generate validated expert trajectories for AIC insertion.

This CLI is intentionally wired for the new architecture: GPT-5-mini may only
return strategy/cable-risk JSON, MoveIt is required for free-space planning, and
CheatCode-style geometry owns final insertion.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.expert_generator.dataset_writer import DatasetMetadataWriter, ExpertEpisodeMetadata  # noqa: E402
from aic_teacher_official.expert_generator.debug_artifacts import (  # noqa: E402
    DebugArtifactPaths,
    call_gpt5_failure_analysis,
    compute_phase_speed_metrics,
    load_lerobot_debug_rows,
    sample_lerobot_rows,
    write_debug_artifacts,
)
from aic_teacher_official.expert_generator.ft_guard import FTGuardConfig  # noqa: E402
from aic_teacher_official.expert_generator.generation_loop import GenerationConfig  # noqa: E402
from aic_teacher_official.expert_generator.moveit_planner import MoveItPlanner, MoveItUnavailableError  # noqa: E402
from aic_teacher_official.expert_generator.planner_runner import ExpertPlannerRecordingRunner, ExpertPlannerRunConfig  # noqa: E402
from aic_teacher_official.expert_generator.replay_runner import OfficialReplayConfig  # noqa: E402
from aic_teacher_official.expert_generator.replay_runner import OfficialRecordingReplayRunner  # noqa: E402
from aic_teacher_official.expert_generator.trajectory_repair import repair_precontact_approach  # noqa: E402
from aic_teacher_official.expert_generator.trajectory_validator import TrajectoryValidator, ValidationCriteria  # noqa: E402
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode  # noqa: E402
from aic_teacher_official.postprocess import postprocess_file  # noqa: E402


def str_bool(value: str) -> bool:
    lowered = str(value).lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--nominal", action="store_true", help="Generate clean nominal ACT-style expert trajectories.")
    mode_group.add_argument("--nominalrecovery", action="store_true", help="Generate nominal trajectories that may include labeled recovery segments.")
    mode_group.add_argument("--recovery", action="store_true", help="Generate recovery-focused trajectories and metadata.")
    mode_group.add_argument("--expert-mode", choices=[m.value for m in ExpertMode], help="Deprecated; prefer --nominal, --nominalrecovery, or --recovery.")
    parser.add_argument("--target-accepted-trajectories", type=int, default=100)
    parser.add_argument("--max-total-attempts", type=int, default=300)
    parser.add_argument("--candidates-per-scene", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=95.0)
    parser.add_argument("--allow-near-gate-acceptance", type=str_bool, default=False)
    parser.add_argument("--near-gate-max-lateral-error-m", type=float, default=0.003)
    parser.add_argument("--near-gate-max-axial-error-m", type=float, default=None)
    parser.add_argument("--near-gate-max-tcp-speed-mps", type=float, default=None)
    parser.add_argument("--near-gate-max-force-delta-n", type=float, default=None)
    parser.add_argument("--near-gate-max-force-n", type=float, default=None)
    parser.add_argument("--ft-threshold", type=float, default=None)
    parser.add_argument("--max-force-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-offlimit-contacts", type=int, default=0)
    parser.add_argument("--require-insertion-event", type=str_bool, default=True)
    parser.add_argument("--rerandomize-scene", type=str_bool, default=True)
    parser.add_argument("--respawn-assets", type=str_bool, default=False)
    parser.add_argument("--scene-randomization-config", default="")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--strategy-model", default="gpt-5-mini")
    parser.add_argument("--analysis-model", default="gpt-5")
    parser.add_argument("--use-gpt5-analysis", type=str_bool, default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--moveit-required", type=str_bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-soft-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--ft-hard-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--backup-distance-m", type=float, default=0.015)
    parser.add_argument(
        "--backoff-increment-m",
        type=float,
        default=None,
        help="Optional per-stage recovery backoff increment. Set equal to backup distance for single-step backoff.",
    )
    parser.add_argument(
        "--backoff-stage-sec",
        type=float,
        default=None,
        help="Optional duration for each recovery backoff stage.",
    )
    parser.add_argument("--min-backoff-distance-m", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--force-confirm-sec", type=float, default=0.0)
    parser.add_argument(
        "--recovery-release-force-threshold",
        type=float,
        default=None,
        help="Optional force-delta threshold used only to decide that recovery backoff released contact.",
    )
    parser.add_argument(
        "--cartesian-stiffness",
        default=None,
        help="Optional scalar or 6-value comma/space list for replay Cartesian stiffness.",
    )
    parser.add_argument(
        "--cartesian-damping",
        default=None,
        help="Optional scalar or 6-value comma/space list for replay Cartesian damping.",
    )
    parser.add_argument(
        "--recovery-cartesian-stiffness",
        default=None,
        help="Optional scalar or 6-value comma/space list used for recovery/backoff Cartesian commands.",
    )
    parser.add_argument(
        "--recovery-cartesian-damping",
        default=None,
        help="Optional scalar or 6-value comma/space list used for recovery/backoff Cartesian commands.",
    )
    parser.add_argument(
        "--joint-stiffness",
        default=None,
        help="Optional scalar or joint-count-value list for replay joint stiffness.",
    )
    parser.add_argument(
        "--joint-damping",
        default=None,
        help="Optional scalar or joint-count-value list for replay joint damping.",
    )
    parser.add_argument("--probe-pattern", choices=["small_cross", "small_spiral", "none"], default="small_cross")
    parser.add_argument("--gazebo-gui", type=str_bool, default=False)
    parser.add_argument("--launch-rviz", type=str_bool, default=False)
    parser.add_argument("--startup-delay-sec", type=int, default=8)
    parser.add_argument("--recorder-drain-sec", type=int, default=120)
    parser.add_argument("--planner-recorder-drain-sec", type=int, default=45)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=0)
    parser.add_argument("--sim-distrobox", default="")
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--compact-stalls", type=str_bool, default=False)
    parser.add_argument("--trajectory-speedup", type=float, default=2.0)
    parser.add_argument("--launch-moveit", type=str_bool, default=True)
    parser.add_argument("--moveit-launch-file", default="aic_moveit_config moveit.launch.py")
    parser.add_argument(
        "--dry-run-config",
        action="store_true",
        help="Validate CLI/config and write generation_config.json without running Gazebo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = _selected_mode(args)
    ft_threshold = args.ft_threshold if args.ft_threshold is not None else args.max_force_threshold
    if not args.moveit_required:
        raise ValueError("MoveIt is required for expert generation; --moveit-required false is not supported.")
    generation_config = GenerationConfig(
        expert_mode=mode,
        target_accepted_trajectories=args.target_accepted_trajectories,
        max_total_attempts=args.max_total_attempts,
        candidates_per_scene=args.candidates_per_scene,
        rerandomize_scene=args.rerandomize_scene,
        respawn_assets=args.respawn_assets,
    )
    ft_config = FTGuardConfig(
        soft_threshold_n=ft_threshold or args.ft_soft_threshold or 1.0,
        hard_threshold_n=args.ft_hard_threshold or ((ft_threshold * 3.0) if ft_threshold is not None else 3.0),
        backup_distance_m=args.backup_distance_m,
        max_retries=args.max_retries,
        probe_pattern=args.probe_pattern,
    )
    replay_config = OfficialReplayConfig(
        repo_root=REPO_ROOT,
        engine_config=Path(args.config),
        output_dir=output_dir / "replay_attempts",
        action_mode="joint_position_then_cheatcode",
        expert_mode=mode.value,
        ft_threshold_n=ft_threshold,
        gazebo_gui=args.gazebo_gui,
        launch_rviz=args.launch_rviz,
        startup_delay_sec=args.startup_delay_sec,
        recorder_drain_sec=args.recorder_drain_sec,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        sim_distrobox=args.sim_distrobox,
        recovery_backoff_distance_m=args.backup_distance_m,
        recovery_backoff_increment_m=args.backoff_increment_m,
        recovery_backoff_sec=args.backoff_stage_sec,
        recovery_min_backoff_distance_m=(
            args.backup_distance_m if args.min_backoff_distance_m is None else args.min_backoff_distance_m
        ),
        recovery_max_retries=args.max_retries,
        recovery_release_force_threshold_n=args.recovery_release_force_threshold,
        force_confirm_sec=args.force_confirm_sec,
        cartesian_stiffness=args.cartesian_stiffness,
        cartesian_damping=args.cartesian_damping,
        recovery_cartesian_stiffness=args.recovery_cartesian_stiffness,
        recovery_cartesian_damping=args.recovery_cartesian_damping,
        joint_stiffness=args.joint_stiffness,
        joint_damping=args.joint_damping,
    )
    planner_config = ExpertPlannerRunConfig(
        repo_root=REPO_ROOT,
        engine_config=Path(args.config),
        output_dir=output_dir / "planner_attempts",
        expert_mode=mode.value,
        strategy_model=args.strategy_model,
        candidates_per_scene=args.candidates_per_scene,
        gazebo_gui=args.gazebo_gui,
        launch_rviz=args.launch_rviz,
        startup_delay_sec=args.startup_delay_sec,
        recorder_drain_sec=args.planner_recorder_drain_sec,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        sim_distrobox=args.sim_distrobox,
        launch_moveit=args.launch_moveit,
        moveit_launch_file=args.moveit_launch_file,
        ft_threshold_n=ft_threshold,
    )
    payload = {
        "schema_version": "aic_expert_generation_config/v1",
        "args": vars(args),
        "generation": generation_config.to_dict(),
        "ft_guard": {
            "ft_threshold_n": ft_threshold,
            "soft_threshold_n": ft_config.soft_threshold_n,
            "hard_threshold_n": ft_config.hard_threshold_n,
            "backup_distance_m": ft_config.backup_distance_m,
            "max_retries": ft_config.max_retries,
            "recovery_release_force_threshold_n": args.recovery_release_force_threshold,
            "probe_pattern": ft_config.probe_pattern,
        },
        "replay": {
            "engine_config": str(replay_config.engine_config),
            "output_dir": str(replay_config.output_dir),
            "policy_class": replay_config.policy_class,
            "action_mode": replay_config.action_mode,
            "gazebo_gui": replay_config.gazebo_gui,
            "launch_rviz": replay_config.launch_rviz,
            "require_recorder_save_log": replay_config.require_recorder_save_log,
            "remove_bag_data": replay_config.remove_bag_data,
            "cartesian_stiffness": replay_config.cartesian_stiffness,
            "cartesian_damping": replay_config.cartesian_damping,
            "recovery_backoff_increment_m": replay_config.recovery_backoff_increment_m,
            "recovery_backoff_sec": replay_config.recovery_backoff_sec,
            "recovery_cartesian_stiffness": replay_config.recovery_cartesian_stiffness,
            "recovery_cartesian_damping": replay_config.recovery_cartesian_damping,
            "joint_stiffness": replay_config.joint_stiffness,
            "joint_damping": replay_config.joint_damping,
        },
        "planner_recording": {
            "engine_config": str(planner_config.engine_config),
            "output_dir": str(planner_config.output_dir),
            "policy_class": "aic_teacher_official.OfficialExpertGeneratorPlanner",
            "expert_mode": planner_config.expert_mode,
            "strategy_model": planner_config.strategy_model,
            "candidates_per_scene": planner_config.candidates_per_scene,
            "launch_moveit": planner_config.launch_moveit,
            "moveit_launch_file": planner_config.moveit_launch_file,
            "ft_threshold_n": planner_config.ft_threshold_n,
        },
        "notes": {
            "vlm_waypoints_allowed": False,
            "moveit_required": args.moveit_required,
            "geometric_fallback_available": False,
            "normal_generation_uses_gpt5_analysis": False,
            "gpt5_analysis_requested": args.use_gpt5_analysis,
            "debug_enabled": args.debug,
        },
    }
    (output_dir / "generation_config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.use_gpt5_analysis:
        args.debug = True
    if args.dry_run_config:
        return 0
    try:
        MoveItPlanner(required=args.moveit_required)
    except MoveItUnavailableError as ex:
        raise SystemExit(str(ex)) from ex
    summary = run_live_generation(
        args=args,
        mode=mode,
        planner_config=planner_config,
        replay_config=replay_config,
        output_dir=output_dir,
        ft_threshold=ft_threshold,
    )
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["accepted"] < args.target_accepted_trajectories:
        return 2
    return 0


def _selected_mode(args: argparse.Namespace) -> ExpertMode:
    if args.nominal:
        return ExpertMode.NOMINAL
    if args.nominalrecovery:
        return ExpertMode.NOMINAL_RECOVERY
    if args.recovery:
        return ExpertMode.RECOVERY
    if args.expert_mode:
        return ExpertMode(args.expert_mode)
    raise ValueError("One mode is required: --nominal, --nominalrecovery, --recovery, or deprecated --expert-mode")


def run_live_generation(
    *,
    args: argparse.Namespace,
    mode: ExpertMode,
    planner_config: ExpertPlannerRunConfig,
    replay_config: OfficialReplayConfig,
    output_dir: Path,
    ft_threshold: float | None,
) -> dict:
    planner_runner = ExpertPlannerRecordingRunner(planner_config)
    replay_runner = OfficialRecordingReplayRunner(replay_config)
    validator = TrajectoryValidator(
        ValidationCriteria(
            score_threshold=args.score_threshold,
            max_force_threshold=ft_threshold,
            max_offlimit_contacts=args.max_offlimit_contacts,
            require_insertion_event=args.require_insertion_event,
        )
    )
    metadata_writer = DatasetMetadataWriter(output_dir / "accepted_metadata")
    accepted = 0
    attempts = 0
    records = []
    while accepted < args.target_accepted_trajectories and attempts < args.max_total_attempts:
        candidate_index = attempts % args.candidates_per_scene
        attempts += 1
        planner_result = planner_runner.run_planner(
            attempt_index=attempts,
            candidate_index=candidate_index,
            seed=args.seed + attempts - 1,
        )
        record = {
            "attempt": attempts,
            "candidate_index": candidate_index,
            "planner": planner_result,
            "accepted": False,
        }
        piecewise_path = Path(planner_result["piecewise_path"])
        if not piecewise_path.exists():
            record["reason"] = "planner_did_not_write_piecewise"
            records.append(record)
            continue
        smooth_path = piecewise_path.with_name("smooth_trajectory.json")
        try:
            smooth = postprocess_file(
                piecewise_path,
                smooth_path,
                args.sample_dt,
                compact_stalls=args.compact_stalls,
                speedup=args.trajectory_speedup,
            )
        except Exception as ex:
            record["reason"] = "postprocess_failed"
            record["error"] = f"{type(ex).__name__}: {ex}"
            records.append(record)
            continue
        replay_metrics = replay_runner.replay_and_score(
            smooth,
            attempt_index=attempts,
            candidate_index=candidate_index,
        )
        debug_assessment = _live_debug_assessment(smooth, replay_metrics)
        replay_metrics.update(debug_assessment.get("metrics", {}))
        planning = smooth.metadata.planning
        strategy = planning.get("vlm_strategy", {})
        validation = validator.evaluate(
            {
                **replay_metrics,
                "mode": mode.value,
                "vlm_cable_risk": strategy.get("cable_risk"),
                "moveit_success": bool((planning.get("moveit") or {}).get("success", True)),
                "candidate_index": candidate_index,
                "scene_seed": args.seed + attempts - 1,
                "phase_labels": [wp.phase.value for wp in smooth.waypoints],
            }
        )
        validation = _augment_validation_with_debug(validation, debug_assessment, mode=mode)
        validation, near_gate_metadata = _maybe_accept_near_gate(
            validation,
            debug_assessment,
            args=args,
        )
        record["replay_metrics"] = replay_metrics
        record["validation"] = {**validation.to_dict(), **near_gate_metadata}
        record["debug_assessment"] = debug_assessment
        repair_metrics = None
        if mode == ExpertMode.NOMINAL and _should_attempt_nominal_repair(validation, debug_assessment):
            repaired, repair_metrics = repair_precontact_approach(smooth, sample_dt=args.sample_dt)
            repaired_path = smooth_path.with_name("smooth_trajectory_repaired.json")
            repaired.save_json(repaired_path)
            repaired_replay_metrics = replay_runner.replay_and_score(
                repaired,
                attempt_index=attempts,
                candidate_index=candidate_index,
                variant_label="repair_00",
            )
            repaired_assessment = _live_debug_assessment(repaired, repaired_replay_metrics)
            repaired_replay_metrics.update(repaired_assessment.get("metrics", {}))
            repaired_validation = validator.evaluate(
                {
                    **repaired_replay_metrics,
                    "mode": mode.value,
                    "vlm_cable_risk": strategy.get("cable_risk"),
                    "moveit_success": bool((planning.get("moveit") or {}).get("success", True)),
                    "candidate_index": candidate_index,
                    "scene_seed": args.seed + attempts - 1,
                    "phase_labels": [wp.phase.value for wp in repaired.waypoints],
                }
            )
            repaired_validation = _augment_validation_with_debug(repaired_validation, repaired_assessment, mode=mode)
            repaired_validation, repaired_near_gate_metadata = _maybe_accept_near_gate(
                repaired_validation,
                repaired_assessment,
                args=args,
            )
            if args.debug:
                repaired_debug_root = (
                    output_dir / f"accepted_dataset_{mode.value}"
                    if repaired_validation.accepted
                    else output_dir / "rejected_attempts" / f"attempt_{attempts:06d}_repair_00"
                )
                repaired_debug_paths = write_debug_artifacts(
                    repaired_debug_root,
                    trajectory=repaired,
                    replay_metrics={
                        **repaired_replay_metrics,
                        "precontact_repair": repair_metrics,
                    },
                    validation={
                        **repaired_validation.to_dict(),
                        **repaired_near_gate_metadata,
                        "precontact_repair": repair_metrics,
                        "debug_assessment": repaired_assessment,
                    },
                    camera_images=_images_from_planner_debug(planner_result),
                    lerobot_dataset_root=Path(repaired_replay_metrics["trajectory_path"]).parent / "dataset",
                )
                record["repaired_debug_dir"] = str(repaired_debug_paths.debug_dir)
                _maybe_run_gpt5_failure_analysis(
                    paths=repaired_debug_paths,
                    args=args,
                    record=record,
                    label="repaired_candidate",
                    should_run=not repaired_validation.accepted,
                )
            record["repair_attempted"] = True
            record["repair_metrics"] = repair_metrics
            record["repaired_replay_metrics"] = repaired_replay_metrics
            record["repaired_validation"] = {**repaired_validation.to_dict(), **repaired_near_gate_metadata}
            record["repaired_debug_assessment"] = repaired_assessment
            smooth = repaired
            smooth_path = repaired_path
            replay_metrics = repaired_replay_metrics
            validation = repaired_validation
            debug_assessment = repaired_assessment
            near_gate_metadata = repaired_near_gate_metadata
        else:
            record["repair_attempted"] = False
            if repair_metrics:
                record["repair_metrics"] = repair_metrics
        if args.debug:
            debug_root = (
                output_dir / f"accepted_dataset_{mode.value}"
                if validation.accepted
                else output_dir / "rejected_attempts" / f"attempt_{attempts:06d}"
            )
            debug_paths = write_debug_artifacts(
                debug_root,
                trajectory=smooth,
                replay_metrics=replay_metrics,
                validation={
                    **validation.to_dict(),
                    **near_gate_metadata,
                    "debug_assessment": debug_assessment,
                    "precontact_repair": repair_metrics,
                },
                camera_images=_images_from_planner_debug(planner_result),
                lerobot_dataset_root=Path(replay_metrics["trajectory_path"]).parent / "dataset",
            )
            record["debug_dir"] = str(debug_paths.debug_dir)
            _maybe_run_gpt5_failure_analysis(
                paths=debug_paths,
                args=args,
                record=record,
                label="candidate",
                should_run=not validation.accepted,
            )
        if validation.accepted:
            record["accepted"] = True
            metadata_writer.append_episode(
                ExpertEpisodeMetadata(
                    episode_index=accepted,
                    mode=mode.value,
                    scene_id=f"attempt_{attempts:06d}",
                    candidate_index=candidate_index,
                    trajectory_path=str(smooth_path),
                    validation={**validation.to_dict(), **near_gate_metadata},
                    vlm_strategy=strategy,
                    moveit=planning.get("moveit", {}),
                    phase_labels=[
                        {
                            "timestamp": wp.timestamp,
                            "phase": wp.phase.value,
                            "source": wp.source.value,
                        }
                        for wp in smooth.waypoints
                    ],
                    extra={
                        "planner_result": planner_result,
                        "candidate": planning.get("candidate", {}),
                    },
                )
            )
            accepted += 1
        records.append(record)
    return {
        "schema_version": "aic_expert_live_generation_summary/v1",
        "mode": mode.value,
        "target_accepted_trajectories": args.target_accepted_trajectories,
        "max_total_attempts": args.max_total_attempts,
        "accepted": accepted,
        "attempts": attempts,
        "stopped_reason": (
            "target_reached"
            if accepted >= args.target_accepted_trajectories
            else "max_attempts_exhausted"
        ),
        "records": records,
    }


def _maybe_run_gpt5_failure_analysis(
    *,
    paths: DebugArtifactPaths,
    args: argparse.Namespace,
    record: dict[str, Any],
    label: str,
    should_run: bool,
) -> None:
    if not args.use_gpt5_analysis or not should_run:
        return
    try:
        analysis = call_gpt5_failure_analysis(
            paths.gpt5_prompt.read_text(encoding="utf-8"),
            model=args.analysis_model,
        )
        paths.gpt5_analysis.write_text(analysis, encoding="utf-8")
        record.setdefault("gpt5_failure_analysis", {})[label] = {
            "status": "ok",
            "analysis": str(paths.gpt5_analysis),
            "prompt": str(paths.gpt5_prompt),
            "payload": str(paths.gpt5_payload),
        }
    except Exception as ex:
        error_path = paths.debug_dir / "gpt5_failure_analysis_error.json"
        error_path.write_text(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(ex).__name__}: {ex}",
                    "analysis_model": args.analysis_model,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        record.setdefault("gpt5_failure_analysis", {})[label] = {
            "status": "error",
            "error": f"{type(ex).__name__}: {ex}",
            "error_path": str(error_path),
        }


def _images_from_planner_debug(planner_result: dict) -> list[Path]:
    debug_dir = Path(str(planner_result.get("debug_dir", "")))
    if not debug_dir.exists():
        return []
    return sorted(path for path in debug_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _live_debug_assessment(smooth, replay_metrics: dict) -> dict:
    dataset_root = Path(replay_metrics["trajectory_path"]).parent / "dataset"
    rows = load_lerobot_debug_rows(dataset_root)
    samples = sample_lerobot_rows(rows, trajectory=smooth, sample_period_sec=0.25)
    phase_speed = compute_phase_speed_metrics(samples)
    tracking_values = [
        sample["tracking_error"].get("position_error_m")
        for sample in samples
        if sample.get("tracking_error", {}).get("position_error_m") is not None
    ]
    runtime_trace_path = Path(str(replay_metrics.get("runtime_trace_path", "")))
    runtime_events = []
    malformed_runtime_event_lines = 0
    if runtime_trace_path.exists():
        for line in runtime_trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                runtime_events.append(json.loads(line))
            except json.JSONDecodeError:
                malformed_runtime_event_lines += 1
    max_guarded_speed = phase_speed.get("max_guarded_insert_speed_mps")
    runtime_guarded_speeds = [
        float(event["actual_tcp_speed_mps"])
        for event in runtime_events
        if event.get("event") in {"guarded_insert_speed_gate_advance", "guarded_insert_speed_gate_hold"}
        and event.get("actual_tcp_speed_mps") is not None
    ]
    if runtime_guarded_speeds:
        max_guarded_speed = max(runtime_guarded_speeds)
    gate_events = [event for event in runtime_events if event.get("event") == "tracking_gate_checked"]
    guarded_start_time = None
    for event in runtime_events:
        if event.get("event") == "guarded_insert_started":
            guarded_start_time = event.get("time_sec")
            break
    preinsert_gate_events = [
        event
        for event in gate_events
        if guarded_start_time is None
        or event.get("time_sec") is None
        or float(event.get("time_sec")) <= float(guarded_start_time)
    ]
    preinsert_gate = preinsert_gate_events[-1] if preinsert_gate_events else (gate_events[-1] if gate_events else {})
    gate_repair_events = [
        event
        for event in runtime_events
        if event.get("event")
        in {
            "cheatcode_handoff_gate_repaired_with_live_z_offset",
            "cheatcode_handoff_skipped_live_z_offset",
        }
    ]
    guarded_insert_success = any(event.get("event") == "guarded_insert_success" for event in runtime_events)
    tracking_gate_passed = all(bool(event.get("tracking_gate_passed")) for event in gate_events) if gate_events else None
    if guarded_insert_success and gate_events and bool(gate_events[-1].get("tracking_gate_passed")):
        tracking_gate_passed = True
    if tracking_gate_passed is False and gate_repair_events and guarded_insert_success:
        tracking_gate_passed = True
    contact_events = [
        event
        for event in runtime_events
        if event.get("event") in {"contact_detected", "precontact_port_align_aborted_force"}
    ]
    backoff_events = [event for event in runtime_events if event.get("event") == "recovery_backoff_completed"]
    backoff_stage_events = [
        event for event in runtime_events if event.get("event") == "recovery_backoff_stage_completed"
    ]
    release_events = [event for event in runtime_events if event.get("event") == "recovery_force_release_wait"]
    return_gate_events = [
        event
        for event in runtime_events
        if event.get("event") == "recovery_return_to_preinsert_gate_checked"
    ]
    latest_backoff = backoff_events[-1] if backoff_events else {}
    latest_backoff_stage = backoff_stage_events[-1] if backoff_stage_events else {}
    measured_backoff_distance = latest_backoff.get("measured_backoff_distance_m")
    if measured_backoff_distance is None:
        measured_backoff_distance = latest_backoff_stage.get("measured_backoff_distance_m")
    measured_backoff_occurred = latest_backoff.get("measured_backoff_occurred")
    if measured_backoff_occurred is None and measured_backoff_distance is not None:
        required_measured_backoff = float(
            os.environ.get("AIC_OFFICIAL_TEACHER_RECOVERY_REQUIRED_MEASURED_BACKOFF_M", "0.002")
        )
        measured_backoff_occurred = float(measured_backoff_distance) >= required_measured_backoff
    force_release_before_realign = release_events[-1].get("force_released") if release_events else None
    if force_release_before_realign is None and return_gate_events:
        force_release_before_realign = any(bool(event.get("tracking_gate_passed")) for event in return_gate_events)
    max_tracking = max(tracking_values) if tracking_values else None
    return {
        "schema_version": "aic_expert_live_debug_assessment/v1",
        "phase_speed_metrics": phase_speed,
        "runtime_event_count": len(runtime_events),
        "malformed_runtime_event_lines": malformed_runtime_event_lines,
        "tracking_gate_checked": bool(gate_events),
        "tracking_gate_passed": tracking_gate_passed,
        "tracking_gate_repaired_with_live_z_offset": bool(gate_repair_events),
        "tracking_gate_final_error_m": gate_events[-1].get("final_tracking_error_m") if gate_events else None,
        "preinsert_tracking_gate_checked": bool(preinsert_gate),
        "preinsert_tracking_gate_passed": preinsert_gate.get("tracking_gate_passed") if preinsert_gate else None,
        "preinsert_tracking_gate_final_error_m": (
            preinsert_gate.get("final_tracking_error_m") if preinsert_gate else None
        ),
        "preinsert_tracking_gate_final_lateral_error_m": (
            preinsert_gate.get("final_lateral_error_m") if preinsert_gate else None
        ),
        "preinsert_tracking_gate_final_axial_error_m": (
            preinsert_gate.get("final_axial_error_m") if preinsert_gate else None
        ),
        "preinsert_tracking_gate_final_tcp_speed_mps": (
            preinsert_gate.get("final_tcp_speed_mps") if preinsert_gate else None
        ),
        "preinsert_tracking_gate_force_delta_n": preinsert_gate.get("force_delta_n") if preinsert_gate else None,
        "contact_detected": bool(contact_events),
        "backoff_occurred": bool(backoff_events or backoff_stage_events),
        "backoff_distance_achieved_m": latest_backoff.get("backoff_distance_achieved_m") if backoff_events else None,
        "measured_backoff_occurred": measured_backoff_occurred,
        "measured_backoff_distance_m": measured_backoff_distance,
        "force_release_before_realign": force_release_before_realign,
        "metrics": {
            "max_tracking_error_m": max_tracking,
            "max_guarded_insert_speed_mps": max_guarded_speed,
            "max_guarded_insert_speed_source": "runtime_trace" if runtime_guarded_speeds else "sampled_phase_metrics",
            "tracking_gate_passed": tracking_gate_passed,
            "tracking_gate_repaired_with_live_z_offset": bool(gate_repair_events),
            "preinsert_tracking_gate_checked": bool(preinsert_gate),
            "preinsert_tracking_gate_passed": preinsert_gate.get("tracking_gate_passed") if preinsert_gate else None,
            "preinsert_tracking_gate_final_error_m": (
                preinsert_gate.get("final_tracking_error_m") if preinsert_gate else None
            ),
            "preinsert_tracking_gate_final_lateral_error_m": (
                preinsert_gate.get("final_lateral_error_m") if preinsert_gate else None
            ),
            "preinsert_tracking_gate_final_axial_error_m": (
                preinsert_gate.get("final_axial_error_m") if preinsert_gate else None
            ),
            "preinsert_tracking_gate_final_tcp_speed_mps": (
                preinsert_gate.get("final_tcp_speed_mps") if preinsert_gate else None
            ),
            "preinsert_tracking_gate_force_delta_n": preinsert_gate.get("force_delta_n") if preinsert_gate else None,
            "contact_detected": bool(contact_events),
            "backoff_occurred": bool(backoff_events or backoff_stage_events),
            "backoff_distance_achieved_m": latest_backoff.get("backoff_distance_achieved_m") if backoff_events else None,
            "measured_backoff_occurred": measured_backoff_occurred,
            "measured_backoff_distance_m": measured_backoff_distance,
            "force_release_before_realign": force_release_before_realign,
        },
    }


def _augment_validation_with_debug(validation, assessment: dict, *, mode: ExpertMode):
    reasons = list(validation.reasons)
    metrics = assessment.get("metrics", {})
    guarded_speed = metrics.get("max_guarded_insert_speed_mps")
    guarded_speed_threshold = float(os.environ.get("AIC_EXPERT_VALIDATION_MAX_GUARDED_SPEED_MPS", "0.02"))
    if guarded_speed is not None and guarded_speed > guarded_speed_threshold:
        reasons.append("guarded_insert_speed_threshold")
    gate_passed = metrics.get("tracking_gate_passed")
    if gate_passed is False:
        nominal_gate_allowance_m = float(
            os.environ.get("AIC_EXPERT_VALIDATION_NOMINAL_GATE_MISS_ALLOWANCE_M", "0.004")
        )
        nominal_gate_min_score = float(
            os.environ.get("AIC_EXPERT_VALIDATION_NOMINAL_GATE_MISS_MIN_SCORE", "92.0")
        )
        gate_error = assessment.get("tracking_gate_final_error_m")
        score = validation.score
        if not (
            mode == ExpertMode.NOMINAL
            and score is not None
            and score >= nominal_gate_min_score
            and gate_error is not None
            and float(gate_error) <= nominal_gate_allowance_m
        ):
            reasons.append("tracking_gate_failed")
    if mode == ExpertMode.RECOVERY:
        if not metrics.get("contact_detected"):
            reasons.append("recovery_contact_trigger_missing")
        if not metrics.get("backoff_occurred"):
            reasons.append("recovery_backoff_missing")
        if metrics.get("measured_backoff_occurred") is not True:
            reasons.append("recovery_measured_backoff_missing")
        if metrics.get("force_release_before_realign") is not True:
            reasons.append("recovery_force_release_missing")
    if reasons == validation.reasons:
        return validation
    return replace(validation, accepted=False, reasons=reasons)


def _maybe_accept_near_gate(validation, assessment: dict, *, args: argparse.Namespace):
    if not args.allow_near_gate_acceptance:
        return validation, {}
    metrics = assessment.get("metrics", {})
    lateral_error = metrics.get("preinsert_tracking_gate_final_lateral_error_m")
    axial_error = metrics.get("preinsert_tracking_gate_final_axial_error_m")
    tcp_speed = metrics.get("preinsert_tracking_gate_final_tcp_speed_mps")
    force_delta = metrics.get("preinsert_tracking_gate_force_delta_n")
    max_force_n = validation.max_force_n
    checks = {
        "preinsert_tracking_gate_checked": bool(metrics.get("preinsert_tracking_gate_checked")),
        "preinsert_tracking_gate_passed": metrics.get("preinsert_tracking_gate_passed") is True,
        "lateral_error_within_threshold": (
            lateral_error is not None and float(lateral_error) <= args.near_gate_max_lateral_error_m
        ),
        "axial_error_within_threshold": (
            args.near_gate_max_axial_error_m is None
            or (axial_error is not None and float(axial_error) <= args.near_gate_max_axial_error_m)
        ),
        "tcp_speed_within_threshold": (
            args.near_gate_max_tcp_speed_mps is None
            or (tcp_speed is not None and float(tcp_speed) <= args.near_gate_max_tcp_speed_mps)
        ),
        "force_delta_within_threshold": (
            args.near_gate_max_force_delta_n is None
            or (force_delta is not None and float(force_delta) <= args.near_gate_max_force_delta_n)
        ),
        "max_force_within_threshold": (
            args.near_gate_max_force_n is None
            or (max_force_n is not None and float(max_force_n) <= args.near_gate_max_force_n)
        ),
        "offlimit_contact_count_ok": validation.offlimit_contact_count == 0,
        "moveit_success": validation.moveit_success,
    }
    accepted = all(checks.values())
    original_reasons = list(validation.reasons)
    if validation.accepted:
        original_reasons.append("standard_validation_would_accept")
    metadata = {
        "near_gate_acceptance": {
            "enabled": True,
            "accepted": accepted,
            "checks": checks,
            "thresholds": {
                "max_lateral_error_m": args.near_gate_max_lateral_error_m,
                "max_axial_error_m": args.near_gate_max_axial_error_m,
                "max_tcp_speed_mps": args.near_gate_max_tcp_speed_mps,
                "max_force_delta_n": args.near_gate_max_force_delta_n,
                "max_force_n": args.near_gate_max_force_n,
            },
            "metrics": {
                "preinsert_tracking_gate_final_lateral_error_m": lateral_error,
                "preinsert_tracking_gate_final_axial_error_m": axial_error,
                "preinsert_tracking_gate_final_tcp_speed_mps": tcp_speed,
                "preinsert_tracking_gate_force_delta_n": force_delta,
                "max_force_n": max_force_n,
                "preinsert_tracking_gate_final_error_m": metrics.get("preinsert_tracking_gate_final_error_m"),
            },
            "original_reasons": original_reasons,
            "original_score": validation.score,
        }
    }
    if not accepted:
        reasons = list(dict.fromkeys([*validation.reasons, "near_gate_acceptance_failed"]))
        return replace(validation, accepted=False, reasons=reasons), metadata
    metadata["acceptance_type"] = "near_gate"
    metadata["score"] = float(args.score_threshold)
    return replace(validation, accepted=True, reasons=[]), metadata


def _should_attempt_nominal_repair(validation, assessment: dict) -> bool:
    if validation.accepted:
        return False
    if assessment.get("contact_detected"):
        return False
    if "force_above_threshold_or_missing" in validation.reasons:
        return False
    if "offlimit_contact_threshold" in validation.reasons:
        return False
    if "tracking_gate_failed" in validation.reasons:
        return False
    phase_speed = assessment.get("phase_speed_metrics", {}).get("phases", {})
    local_align = phase_speed.get("local_preinsert_align", {})
    moveit = phase_speed.get("moveit_approach", {})
    local_speed = local_align.get("max_speed_mps")
    moveit_speed = moveit.get("max_speed_mps")
    return bool(
        (local_speed is not None and local_speed > 0.06)
        or (moveit_speed is not None and moveit_speed > 0.18)
    )


if __name__ == "__main__":
    raise SystemExit(main())
