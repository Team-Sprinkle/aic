#!/usr/bin/env python3
"""Build reproducible Isaac teacher-collection commands.

The command this script writes is intentionally a data-collection recipe, not a
success claim.  It executes the privileged target-tip guide, logs semantic
tip/module diagnostics, saves images/video/replay, and leaves strict success
evaluation unchanged.  The resulting replay should be filtered after the run
before any imitation-heavy policy update.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_COMMAND = (
    REPO_ROOT
    / "outputs"
    / "agentic_reward_curriculum_20260529"
    / "commands"
    / "train_v561_collect_nonexecuted_guide_replay_v556_220.txt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "agentic_reward_curriculum_20260529"


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    negated = f"--no-{flag[2:]}" if flag.startswith("--") else None
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == flag or (negated is not None and token == negated):
            idx += 1
            while idx < len(argv) and not argv[idx].startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def _set_flag(argv: list[str], flag: str, *values: Any) -> list[str]:
    argv = _strip_flag(argv, flag)
    return [*argv, flag, *[str(value) for value in values]]


def _enable_flag(argv: list[str], flag: str) -> list[str]:
    return [*_strip_flag(argv, flag), flag]


def _disable_flag(argv: list[str], flag: str) -> list[str]:
    if not flag.startswith("--"):
        raise ValueError(f"boolean flag must start with --: {flag}")
    return [*_strip_flag(argv, flag), f"--no-{flag[2:]}"]


def _repo_rel(path: Path) -> str:
    path = path.resolve()
    try:
        return f"aic/{path.relative_to(REPO_ROOT)}"
    except ValueError:
        return str(path)


def _container_path(path: Path) -> str:
    rel = _repo_rel(path)
    if rel.startswith("aic/"):
        return f"/workspace/isaaclab/{rel}"
    return f"/workspace/isaaclab/aic/{rel}"


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    argv = shlex.split(args.base_command.read_text(encoding="utf-8"))
    steps = int(args.steps)
    run_name = str(args.run_name)
    replay_path = args.output_root / "replay" / f"{run_name}.pt"

    one_value_flags = {
        "--run_name": run_name,
        "--episode_config_dir": _repo_rel(args.episode_config_dir),
        "--num_envs": int(args.num_envs),
        "--steps": steps,
        "--updates": steps,
        "--warmup_steps": 1000000,
        "--max_wall_time_minutes": int(args.max_wall_time_minutes),
        "--target_action_guide_collect_blend": 1.0,
        "--target_action_guide_collect_steps": steps,
        "--target_action_guide_weight": 0.0,
        "--target_action_guide_mode": "target_tip_stabilize",
        "--target_action_guide_target_tip_goal_depth_m": float(args.target_depth_m),
        "--target_action_guide_target_tip_axial_step_m": float(args.axial_step_m),
        "--target_action_guide_target_tip_axial_direction_sign": float(args.axial_direction_sign),
        "--target_action_guide_target_tip_lateral_step_m": float(args.lateral_step_m),
        "--target_action_guide_target_tip_secondary_lateral_step_m": float(args.lateral_step_m),
        "--target_action_guide_target_tip_axial_lateral_gate_m": float(args.lateral_gate_m),
        "--target_action_guide_target_tip_secondary_axial_lateral_gate_m": float(args.module_lateral_gate_m),
        "--target_action_guide_orientation_switch_rad": float(args.orientation_gate_rad),
        "--target_action_guide_final_orientation_threshold_rad": float(args.final_orientation_threshold_rad),
        "--target_action_guide_final_orientation_lateral_m": float(args.final_orientation_lateral_m),
        "--target_action_guide_final_orientation_depth_m": float(args.final_orientation_depth_m),
        "--target_action_guide_rotation_step_size": float(args.rotation_step_rad),
        "--target_action_guide_final_orientation_rotation_step_size": float(args.rotation_step_rad),
        "--target_action_guide_rotation_compensation_clip_m": float(args.rotation_compensation_clip_m),
        "--target_action_guide_target_tip_shallow_bypass_activation_depth_m": -0.020,
        "--target_action_guide_target_tip_shallow_bypass_module_gap_m": 0.006,
        "--target_action_guide_target_tip_shallow_bypass_backoff_m": float(args.backoff_m),
        "--target_action_guide_target_tip_realized_r_recovery_activation_depth_m": -0.010,
        "--target_action_guide_target_tip_realized_r_recovery_depth_progress_m": 0.000005,
        "--target_action_guide_target_tip_realized_r_recovery_backoff_m": float(args.backoff_m),
        "--target_action_guide_target_tip_realized_r_recovery_theta_gate_rad": float(args.orientation_gate_rad),
        "--target_action_guide_phase_lateral_gate_m": float(args.lateral_gate_m),
        "--target_action_guide_phase_module_lateral_gate_m": float(args.module_lateral_gate_m),
        "--target_action_guide_phase_theta_min_rad": 0.0,
        "--target_action_guide_phase_theta_max_rad": 0.070,
        "--target_action_guide_phase_min_s_m": -0.010,
        "--target_action_guide_phase_max_s_m": float(args.target_depth_m),
        "--target_action_guide_phase_weight": 1.0,
        "--save_replay_path": _repo_rel(replay_path),
        "--save_replay_filter": "all",
        "--diagnostics_every": int(args.diagnostics_every),
        "--log_every": int(args.diagnostics_every),
        "--image_log_every": int(args.image_log_every),
        "--max_logged_image_steps": steps,
        "--log_robot_state_every": int(args.log_robot_state_every),
        "--video_fps": int(args.video_fps),
        "--video_crf": int(args.video_crf),
        "--save_every_steps": 0,
        "--save_latest_every_steps": 0,
        "--actor_update_start_steps": steps + 1,
        "--actor_update_end_steps": steps + 1,
        "--actor_q_weight": 0.0,
        "--tcp_translation_action_clip": float(args.translation_clip_m),
        "--tcp_rotation_action_clip": float(args.rotation_clip_rad),
        "--action_clip": float(args.translation_clip_m),
        "--batch_size": max(1, int(args.num_envs)),
        "--output_dir": _repo_rel(args.output_root / "policy_train_runs"),
    }
    for flag, value in one_value_flags.items():
        argv = _set_flag(argv, flag, value)

    if args.checkpoint:
        argv = _set_flag(argv, "--checkpoint", _repo_rel(args.checkpoint))

    for flag in (
        "--target_action_guide_train_executed",
        "--target_action_guide_separate_rotation_compensation",
        "--target_action_guide_target_tip_clamp_positive_axial_when_gated",
        "--target_action_guide_target_tip_orientation_realized_reject",
        "--target_action_guide_orientation_probe_basis",
        "--target_action_guide_orientation_probe_strict_lateral_gate",
        "--target_action_guide_orientation_probe_strict_secondary_lateral_gate",
        "--save_step_images",
        "--save_videos",
        "--save_replay_at_end",
        "--log_policy_actions_by_env",
        "--freeze_act",
        "--debug_diagnostics",
        "--enable_contact_sensor",
    ):
        argv = _enable_flag(argv, flag)
    if bool(args.shallow_bypass_recovery):
        argv = _enable_flag(argv, "--target_action_guide_target_tip_shallow_bypass_recovery")
    else:
        argv = _disable_flag(argv, "--target_action_guide_target_tip_shallow_bypass_recovery")
    if bool(args.realized_r_recovery):
        argv = _enable_flag(argv, "--target_action_guide_target_tip_realized_r_recovery")
    else:
        argv = _disable_flag(argv, "--target_action_guide_target_tip_realized_r_recovery")

    for flag in (
        "--insertion_action_guard",
        "--insertion_action_guard_target_tip_servo",
        "--insertion_action_guard_final_two_stage_servo",
        "--insertion_action_guard_module_lateral_alignment",
        "--insertion_action_guard_module_lateral_alignment_rotation",
        "--insertion_action_guard_contact_force_recovery",
        "--insertion_action_guard_contact_force_retreat_state_machine",
        "--insertion_action_guard_reject_predicted_r_increase",
        "--insertion_action_guard_final_fixed_world_rotation",
        "--insertion_action_guard_final_axis_alignment_rotation",
        "--insertion_action_guard_prelip_lateral_clamp",
        "--insertion_action_guard_prelip_offgate_axial_lock",
    ):
        argv = _disable_flag(argv, flag)

    manifest = {
        "run_name": run_name,
        "base_command": str(args.base_command),
        "episode_config_dir": str(args.episode_config_dir),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
        "steps": steps,
        "num_envs": int(args.num_envs),
        "target_depth_m": float(args.target_depth_m),
        "axial_direction_sign": float(args.axial_direction_sign),
        "shallow_bypass_recovery": bool(args.shallow_bypass_recovery),
        "realized_r_recovery": bool(args.realized_r_recovery),
        "final_orientation_threshold_rad": float(args.final_orientation_threshold_rad),
        "purpose": (
            "Collect Isaac teacher/residual data from privileged target-tip guide. "
            "This is not a policy-only success claim."
        ),
        "strict_success_unchanged": True,
        "replay_path": str(replay_path),
        "docker_invocation": (
            "docker exec -w /workspace/isaaclab isaac-lab-base bash -lc "
            f"'set -e; ./isaaclab.sh -p $(cat {_container_path(args.output_command)})'"
        ),
    }
    return argv, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-command", type=Path, default=DEFAULT_BASE_COMMAND)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-command", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--episode-config-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-wall-time-minutes", type=int, default=90)
    parser.add_argument("--target-depth-m", type=float, default=0.046864)
    parser.add_argument("--axial-step-m", type=float, default=0.00006)
    parser.add_argument(
        "--axial-direction-sign",
        type=float,
        default=1.0,
        help="Forward to --target_action_guide_target_tip_axial_direction_sign for controller-frame sign ablations.",
    )
    parser.add_argument("--lateral-step-m", type=float, default=0.00035)
    parser.add_argument("--lateral-gate-m", type=float, default=0.0006)
    parser.add_argument("--module-lateral-gate-m", type=float, default=0.0012)
    parser.add_argument("--orientation-gate-rad", type=float, default=0.030)
    parser.add_argument(
        "--final-orientation-threshold-rad",
        type=float,
        default=0.030,
        help=(
            "Strict-ish target-tip final orientation trim threshold. This is intentionally separate "
            "from --orientation-gate-rad, which controls whether axial motion is allowed."
        ),
    )
    parser.add_argument("--final-orientation-lateral-m", type=float, default=0.0006)
    parser.add_argument("--final-orientation-depth-m", type=float, default=-0.010)
    parser.add_argument("--rotation-step-rad", type=float, default=0.0003)
    parser.add_argument("--rotation-compensation-clip-m", type=float, default=0.0010)
    parser.add_argument("--backoff-m", type=float, default=0.00008)
    parser.add_argument(
        "--shallow-bypass-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable target-tip shallow-bypass backoff during teacher collection. "
            "Default is off because prior smoke runs showed it can suppress all inward teacher motion."
        ),
    )
    parser.add_argument(
        "--realized-r-recovery",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable target-tip realized-r/theta recovery during teacher collection. "
            "Default is off for module-following teacher probes; use eval metrics to reject bad rollouts instead."
        ),
    )
    parser.add_argument("--translation-clip-m", type=float, default=0.00008)
    parser.add_argument("--rotation-clip-rad", type=float, default=0.0006)
    parser.add_argument("--diagnostics-every", type=int, default=5)
    parser.add_argument(
        "--image-log-every",
        type=int,
        default=1,
        help=(
            "Save every simulator step by default so --save-videos produces a true full-rate "
            "episode video. Increase only for snapshot-only diagnostics."
        ),
    )
    parser.add_argument("--log-robot-state-every", type=int, default=1)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-crf", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.base_command = args.base_command.resolve()
    args.output_root = args.output_root.resolve()
    args.output_command = args.output_command.resolve()
    args.episode_config_dir = args.episode_config_dir.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()

    argv, manifest = build_command(args)
    args.output_command.parent.mkdir(parents=True, exist_ok=True)
    args.output_command.write_text(shlex.join(argv) + "\n", encoding="utf-8")
    manifest_path = args.output_command.with_suffix(".json")
    manifest["command_file"] = str(args.output_command)
    manifest["git_status"] = _git(["status", "--short"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"command": str(args.output_command), "manifest": str(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
