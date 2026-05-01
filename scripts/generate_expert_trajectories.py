#!/usr/bin/env python3
"""Generate validated expert trajectories for AIC insertion.

This CLI is intentionally wired for the new architecture: GPT-5-mini may only
return strategy/cable-risk JSON, MoveIt is required for free-space planning, and
CheatCode-style geometry owns final insertion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.expert_generator.dataset_writer import DatasetMetadataWriter, ExpertEpisodeMetadata  # noqa: E402
from aic_teacher_official.expert_generator.ft_guard import FTGuardConfig  # noqa: E402
from aic_teacher_official.expert_generator.generation_loop import GenerationConfig  # noqa: E402
from aic_teacher_official.expert_generator.moveit_planner import MoveItPlanner, MoveItUnavailableError  # noqa: E402
from aic_teacher_official.expert_generator.planner_runner import ExpertPlannerRecordingRunner, ExpertPlannerRunConfig  # noqa: E402
from aic_teacher_official.expert_generator.replay_runner import OfficialReplayConfig  # noqa: E402
from aic_teacher_official.expert_generator.replay_runner import OfficialRecordingReplayRunner  # noqa: E402
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-mode", choices=[m.value for m in ExpertMode], required=True)
    parser.add_argument("--target-accepted-trajectories", type=int, default=100)
    parser.add_argument("--max-total-attempts", type=int, default=300)
    parser.add_argument("--candidates-per-scene", type=int, default=5)
    parser.add_argument("--score-threshold", type=float, default=95.0)
    parser.add_argument("--max-force-threshold", type=float, default=None)
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
    parser.add_argument("--moveit-required", type=str_bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ft-soft-threshold", type=float, default=1.0)
    parser.add_argument("--ft-hard-threshold", type=float, default=3.0)
    parser.add_argument("--backup-distance-m", type=float, default=0.006)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--probe-pattern", choices=["small_cross", "small_spiral", "none"], default="small_cross")
    parser.add_argument("--gazebo-gui", type=str_bool, default=False)
    parser.add_argument("--launch-rviz", type=str_bool, default=False)
    parser.add_argument("--startup-delay-sec", type=int, default=8)
    parser.add_argument("--recorder-drain-sec", type=int, default=120)
    parser.add_argument("--planner-recorder-drain-sec", type=int, default=45)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=0)
    parser.add_argument("--sim-distrobox", default="")
    parser.add_argument("--sample-dt", type=float, default=0.05)
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
    mode = ExpertMode(args.expert_mode)
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
        soft_threshold_n=args.ft_soft_threshold,
        hard_threshold_n=args.ft_hard_threshold,
        backup_distance_m=args.backup_distance_m,
        max_retries=args.max_retries,
        probe_pattern=args.probe_pattern,
    )
    replay_config = OfficialReplayConfig(
        repo_root=REPO_ROOT,
        engine_config=Path(args.config),
        output_dir=output_dir / "replay_attempts",
        action_mode="joint_position_then_cheatcode",
        gazebo_gui=args.gazebo_gui,
        launch_rviz=args.launch_rviz,
        startup_delay_sec=args.startup_delay_sec,
        recorder_drain_sec=args.recorder_drain_sec,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        sim_distrobox=args.sim_distrobox,
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
    )
    payload = {
        "schema_version": "aic_expert_generation_config/v1",
        "args": vars(args),
        "generation": generation_config.to_dict(),
        "ft_guard": {
            "soft_threshold_n": ft_config.soft_threshold_n,
            "hard_threshold_n": ft_config.hard_threshold_n,
            "backup_distance_m": ft_config.backup_distance_m,
            "max_retries": ft_config.max_retries,
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
        },
        "notes": {
            "vlm_waypoints_allowed": False,
            "moveit_required": args.moveit_required,
            "geometric_fallback_available": False,
            "normal_generation_uses_gpt5_analysis": False,
            "gpt5_analysis_requested": args.use_gpt5_analysis,
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
    )
    (output_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if summary["accepted"] < args.target_accepted_trajectories:
        return 2
    return 0


def run_live_generation(
    *,
    args: argparse.Namespace,
    mode: ExpertMode,
    planner_config: ExpertPlannerRunConfig,
    replay_config: OfficialReplayConfig,
    output_dir: Path,
) -> dict:
    planner_runner = ExpertPlannerRecordingRunner(planner_config)
    replay_runner = OfficialRecordingReplayRunner(replay_config)
    validator = TrajectoryValidator(
        ValidationCriteria(
            score_threshold=args.score_threshold,
            max_force_threshold=args.max_force_threshold,
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
            smooth = postprocess_file(piecewise_path, smooth_path, args.sample_dt)
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
        record["replay_metrics"] = replay_metrics
        record["validation"] = validation.to_dict()
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


if __name__ == "__main__":
    raise SystemExit(main())
