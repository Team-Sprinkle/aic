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
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.expert_generator.dataset_writer import DatasetMetadataWriter, ExpertEpisodeMetadata  # noqa: E402
from aic_teacher_official.expert_generator.debug_artifacts import (  # noqa: E402
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
    parser.add_argument("--min-backoff-distance-m", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--force-confirm-sec", type=float, default=0.0)
    parser.add_argument(
        "--recovery-release-force-threshold",
        type=float,
        default=None,
        help="Optional force-delta threshold used only to decide that recovery backoff released contact.",
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
        recovery_min_backoff_distance_m=(
            args.backup_distance_m if args.min_backoff_distance_m is None else args.min_backoff_distance_m
        ),
        recovery_max_retries=args.max_retries,
        recovery_release_force_threshold_n=args.recovery_release_force_threshold,
        force_confirm_sec=args.force_confirm_sec,
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
        validation = _augment_validation_with_debug(validation, debug_assessment)
        record["replay_metrics"] = replay_metrics
        record["validation"] = validation.to_dict()
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
            repaired_validation = _augment_validation_with_debug(repaired_validation, repaired_assessment)
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
                        "precontact_repair": repair_metrics,
                        "debug_assessment": repaired_assessment,
                    },
                    camera_images=_images_from_planner_debug(planner_result),
                    lerobot_dataset_root=Path(repaired_replay_metrics["trajectory_path"]).parent / "dataset",
                )
                record["repaired_debug_dir"] = str(repaired_debug_paths.debug_dir)
            record["repair_attempted"] = True
            record["repair_metrics"] = repair_metrics
            record["repaired_replay_metrics"] = repaired_replay_metrics
            record["repaired_validation"] = repaired_validation.to_dict()
            record["repaired_debug_assessment"] = repaired_assessment
            smooth = repaired
            smooth_path = repaired_path
            replay_metrics = repaired_replay_metrics
            validation = repaired_validation
            debug_assessment = repaired_assessment
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
                    "debug_assessment": debug_assessment,
                    "precontact_repair": repair_metrics,
                },
                camera_images=_images_from_planner_debug(planner_result),
                lerobot_dataset_root=Path(replay_metrics["trajectory_path"]).parent / "dataset",
            )
            record["debug_dir"] = str(debug_paths.debug_dir)
        if validation.accepted:
            record["accepted"] = True
            metadata_writer.append_episode(
                ExpertEpisodeMetadata(
                    episode_index=accepted,
                    mode=mode.value,
                    scene_id=f"attempt_{attempts:06d}",
                    candidate_index=candidate_index,
                    trajectory_path=str(smooth_path),
                    validation=validation.to_dict(),
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


def _images_from_planner_debug(planner_result: dict) -> list[Path]:
    debug_dir = Path(str(planner_result.get("debug_dir", "")))
    if not debug_dir.exists():
        return []
    return sorted(path for path in debug_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def _live_debug_assessment(smooth, replay_metrics: dict) -> dict:
    dataset_root = Path(replay_metrics["trajectory_path"]).parent / "dataset"
    rows = load_lerobot_debug_rows(dataset_root)
    samples = sample_lerobot_rows(rows, trajectory=smooth, sample_period_sec=0.5)
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
    gate_events = [event for event in runtime_events if event.get("event") == "tracking_gate_checked"]
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
    if tracking_gate_passed is False and gate_repair_events and guarded_insert_success:
        tracking_gate_passed = True
    contact_events = [
        event
        for event in runtime_events
        if event.get("event") in {"contact_detected", "precontact_port_align_aborted_force"}
    ]
    backoff_events = [event for event in runtime_events if event.get("event") == "recovery_backoff_completed"]
    release_events = [event for event in runtime_events if event.get("event") == "recovery_force_release_wait"]
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
        "contact_detected": bool(contact_events),
        "backoff_occurred": bool(backoff_events),
        "backoff_distance_achieved_m": backoff_events[-1].get("backoff_distance_achieved_m") if backoff_events else None,
        "force_release_before_realign": release_events[-1].get("force_released") if release_events else None,
        "metrics": {
            "max_tracking_error_m": max_tracking,
            "max_guarded_insert_speed_mps": max_guarded_speed,
            "tracking_gate_passed": tracking_gate_passed,
            "tracking_gate_repaired_with_live_z_offset": bool(gate_repair_events),
            "contact_detected": bool(contact_events),
            "backoff_occurred": bool(backoff_events),
            "backoff_distance_achieved_m": backoff_events[-1].get("backoff_distance_achieved_m") if backoff_events else None,
            "force_release_before_realign": release_events[-1].get("force_released") if release_events else None,
        },
    }


def _augment_validation_with_debug(validation, assessment: dict):
    reasons = list(validation.reasons)
    guarded_speed = assessment.get("metrics", {}).get("max_guarded_insert_speed_mps")
    if guarded_speed is not None and guarded_speed > 0.02:
        reasons.append("guarded_insert_speed_threshold")
    gate_passed = assessment.get("metrics", {}).get("tracking_gate_passed")
    if gate_passed is False:
        reasons.append("tracking_gate_failed")
    if reasons == validation.reasons:
        return validation
    return replace(validation, accepted=False, reasons=reasons)


def _should_attempt_nominal_repair(validation, assessment: dict) -> bool:
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
