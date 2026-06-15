#!/usr/bin/env python3
"""Validate Isaac SERL reset episodes after actual train.py settle steps.

This script is intentionally a thin wrapper around the runtime SERL entrypoint.
It catches cases where an episode YAML looks geometrically valid before physics
settles, but the train/eval path relaxes into lateral bypass, bad orientation,
or invalid module consistency after one or more zero-action steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACT = "aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "agentic_reward_curriculum_20260524_depth_correction" / "reset_settle_validation"


@dataclass
class EpisodeScore:
    env: int
    episode_file: str
    accepted: bool
    failure_label: str
    s_m: float | None
    r_m: float | None
    theta_rad: float | None
    target_depth_m: float | None
    consistency_gate: float | None
    consistency_final_axial_error_m: float | None
    strict_success: bool


def _rel_container_path(path: Path) -> str:
    path = path.resolve()
    try:
        return "aic/" + str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _float_at(values: Any, idx: int) -> float | None:
    if isinstance(values, list) and idx < len(values):
        try:
            return float(values[idx])
        except Exception:
            return None
    return None


def _bool_at(values: Any, idx: int) -> bool:
    if isinstance(values, list) and idx < len(values):
        return bool(values[idx])
    return False


def _stats(values: list[float | None]) -> dict[str, float | int] | None:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return None
    return {
        "min": min(finite),
        "mean": sum(finite) / len(finite),
        "max": max(finite),
        "count": len(finite),
    }


def _metric_row(row: dict[str, Any], env_idx: int, episode_file: str) -> dict[str, Any]:
    geom = row.get("post_step_insertion_geometry") or {}
    module = ((row.get("post_step_all_body_insertion_geometry") or {}).get("sfp_module_link") or {})
    return {
        "step": row.get("step"),
        "env": env_idx,
        "episode_file": episode_file,
        "s_m": _float_at(geom.get("signed_depth_m_by_env"), env_idx),
        "r_m": _float_at(geom.get("lateral_error_m_by_env"), env_idx),
        "theta_rad": _float_at(geom.get("orientation_error_rad_by_env"), env_idx),
        "module_s_m": _float_at(module.get("signed_depth_m_by_env"), env_idx),
        "module_r_m": _float_at(module.get("lateral_error_m_by_env"), env_idx),
        "consistency_final_axial_error_m": _float_at(
            geom.get("consistency_final_axial_error_m_by_env"), env_idx
        ),
        "strict_success": _bool_at(geom.get("strict_success_by_env"), env_idx),
        "terminated": _bool_at(row.get("terminated_by_env"), env_idx),
    }


def _write_reset_metrics(
    *,
    rows: list[dict[str, Any]],
    episodes: list[Path],
    num_envs: int,
    out_dir: Path,
) -> dict[str, Any]:
    selected_indices = [0]
    if len(rows) > 1:
        selected_indices.append(len(rows) - 1)
    selected_indices = sorted(set(selected_indices))

    metric_rows: list[dict[str, Any]] = []
    by_step: dict[str, dict[str, Any]] = {}
    for row_idx in selected_indices:
        row = rows[row_idx]
        key = "step1" if row_idx == 0 else f"step{row.get('step', row_idx + 1)}"
        current = [_metric_row(row, idx, episodes[idx].name) for idx in range(num_envs)]
        metric_rows.extend({"summary_step": key, **item} for item in current)
        by_step[key] = {
            "score_step": row.get("step"),
            "s": _stats([item["s_m"] for item in current]),
            "r": _stats([item["r_m"] for item in current]),
            "theta": _stats([item["theta_rad"] for item in current]),
            "module_s": _stats([item["module_s_m"] for item in current]),
            "module_r": _stats([item["module_r_m"] for item in current]),
            "consistency_final_axial_error": _stats(
                [item["consistency_final_axial_error_m"] for item in current]
            ),
            "strict_success_count": sum(1 for item in current if item["strict_success"]),
            "terminated_by_env": [bool(item["terminated"]) for item in current],
        }

    csv_path = out_dir / "post_step_reset_metrics.csv"
    fieldnames = [
        "summary_step",
        "step",
        "env",
        "episode_file",
        "s_m",
        "r_m",
        "theta_rad",
        "module_s_m",
        "module_r_m",
        "consistency_final_axial_error_m",
        "strict_success",
        "terminated",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)

    summary = {"steps": by_step, "csv": str(csv_path)}
    (out_dir / "post_step_reset_metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _episode_files(episodes_dir: Path) -> list[Path]:
    return sorted(p for p in episodes_dir.glob("*.yaml") if p.is_file())


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if "[AIC SERL] step=" in line or "Traceback" in line or "ValueError" in line:
                print(line.rstrip())
        return int(proc.wait())


def _score_env(
    geom: dict[str, Any],
    env_idx: int,
    episode_file: str,
    *,
    min_s_m: float,
    max_s_m: float,
    max_lateral_m: float,
    max_theta_rad: float,
    min_consistency_gate: float,
    max_consistency_final_axial_error_m: float,
    require_strict_success: bool,
) -> EpisodeScore:
    s_m = _float_at(geom.get("signed_depth_m_by_env"), env_idx)
    r_m = _float_at(geom.get("lateral_error_m_by_env"), env_idx)
    theta_rad = _float_at(geom.get("orientation_error_rad_by_env"), env_idx)
    target_depth_m = _float_at(geom.get("target_depth_m_by_env"), env_idx)
    consistency_gate = _float_at(geom.get("consistency_gate_by_env"), env_idx)
    consistency_final_axial_error_m = _float_at(
        geom.get("consistency_final_axial_error_m_by_env"), env_idx
    )
    if consistency_final_axial_error_m is None and env_idx == 0:
        raw_final = geom.get("consistency_final_axial_error_m_env0")
        if raw_final is not None:
            consistency_final_axial_error_m = float(raw_final)
    strict_success = _bool_at(geom.get("strict_success_by_env"), env_idx)

    failures: list[str] = []
    if s_m is None or r_m is None or theta_rad is None:
        failures.append("missing_geometry")
    else:
        if math.isfinite(min_s_m) and s_m < min_s_m:
            failures.append("depth_too_low")
        if math.isfinite(max_s_m) and s_m > max_s_m:
            failures.append("depth_too_high")
        if r_m > max_lateral_m:
            failures.append("lateral_error")
        if theta_rad > max_theta_rad:
            failures.append("orientation_error")
    if math.isfinite(min_consistency_gate):
        if consistency_gate is None or consistency_gate < min_consistency_gate:
            failures.append("module_consistency_gate")
    if math.isfinite(max_consistency_final_axial_error_m):
        if consistency_final_axial_error_m is None:
            failures.append("missing_final_consistency_error")
        elif abs(consistency_final_axial_error_m) > max_consistency_final_axial_error_m:
            failures.append("final_consistency_error")
    if require_strict_success and not strict_success:
        failures.append("not_strict_success")

    return EpisodeScore(
        env=env_idx,
        episode_file=episode_file,
        accepted=not failures,
        failure_label="accepted" if not failures else "+".join(failures),
        s_m=s_m,
        r_m=r_m,
        theta_rad=theta_rad,
        target_depth_m=target_depth_m,
        consistency_gate=consistency_gate,
        consistency_final_axial_error_m=consistency_final_axial_error_m,
        strict_success=strict_success,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-config-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=None,
        help="Where the underlying train.py probe writes metrics. Defaults to the historical depth-correction folder.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--docker-container", default="isaac-lab-base")
    parser.add_argument("--act-torchscript", default=DEFAULT_ACT)
    parser.add_argument("--num-envs", type=int, default=0, help="Defaults to the number of YAML episode files.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=0.0,
        help="Episode length for settle probe. Defaults to enough time for --steps at 20 Hz plus margin.",
    )
    parser.add_argument("--seed", type=int, default=55270)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-port-index", type=int, default=1)
    parser.add_argument("--target-card-index", type=int, default=0)
    parser.add_argument("--target-card-valid", type=int, default=1)
    parser.add_argument(
        "--episode-target-orientation-source",
        choices=["target_pose", "reference_reward_body_start", "auto_reference_then_target"],
        default="target_pose",
        help="Pass through train.py episode YAML orientation source for quaternion diagnostics.",
    )
    parser.add_argument(
        "--target-reward-consistency-gate-mode",
        choices=["target_depth", "current_depth"],
        default="target_depth",
        help=(
            "Pass through train.py consistency gate mode. Use current_depth for partial/shallow reset "
            "validation; strict success still uses the final seated consistency thresholds."
        ),
    )
    parser.add_argument("--max-wall-time-minutes", type=float, default=30.0)
    parser.add_argument("--near-gate-reset-max-iterations", type=int, default=200)
    parser.add_argument("--near-gate-reset-position-tolerance", type=float, default=0.0001)
    parser.add_argument("--near-gate-reset-orientation-tolerance", type=float, default=0.003)
    parser.add_argument(
        "--disable-collision-prim-regex",
        action="append",
        default=[],
        help="Pass through diagnostic collision-disabling regexes to serl/train.py.",
    )
    parser.add_argument(
        "--replace-sfp-body-sdf-collision-with-sdf-boxes",
        action="store_true",
        help="Pass through train.py diagnostic replacement of SFP body_sdf_collision with SDF box colliders.",
    )
    parser.add_argument(
        "--replace-sfp-module-sdf-collision-with-all-sdf-boxes",
        action="store_true",
        help="Pass through train.py diagnostic replacement of SFP body_sdf_collision with all authored SFP module SDF box colliders.",
    )
    parser.add_argument(
        "--replace-sfp-module-sdf-collision-with-active-sdf-boxes",
        action="store_true",
        help=(
            "Pass through train.py diagnostic replacement of SFP body_sdf_collision with Gazebo-active "
            "SFP module SDF box colliders."
        ),
    )
    parser.add_argument(
        "--replace-sfp-body-sdf-collision-with-shrunk-sdf-boxes",
        action="store_true",
        help="Pass through train.py replacement of SFP body_sdf_collision with shrunk SDF box colliders.",
    )
    parser.add_argument(
        "--sfp-shrunk-box-margin-m",
        type=float,
        nargs=3,
        default=(0.00015, 0.0, 0.00015),
        metavar=("X", "Y", "Z"),
        help="Margins for --replace-sfp-body-sdf-collision-with-shrunk-sdf-boxes.",
    )
    parser.add_argument(
        "--randomization-profile",
        choices=["none", "light", "heavy"],
        default=None,
        help="Optional AIC_ISAAC_RANDOMIZATION_PROFILE override for reset-settle ablations.",
    )
    parser.add_argument(
        "--ik-body-name",
        default=None,
        help=(
            "Optional AIC_ISAAC_IK_BODY_NAME override for reset/controller diagnostics. "
            "Leave unset to preserve the environment default."
        ),
    )
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment variable export for the underlying Isaac train.py probe.",
    )
    parser.add_argument(
        "--zero-action",
        action="store_true",
        help=(
            "Use train.py debug-audit constant zero TCP actions during settle. "
            "Without this, the ACT policy runs and post-step metrics include policy motion."
        ),
    )
    parser.add_argument("--settle-step", type=int, default=-1, help="Metrics row to score; negative selects the last row.")
    parser.add_argument("--min-s-m", type=float, default=float("-inf"))
    parser.add_argument("--max-s-m", type=float, default=float("inf"))
    parser.add_argument("--max-lateral-m", type=float, default=0.0005)
    parser.add_argument("--max-theta-rad", type=float, default=0.030)
    parser.add_argument("--min-consistency-gate", type=float, default=float("nan"))
    parser.add_argument("--max-consistency-final-axial-error-m", type=float, default=float("nan"))
    parser.add_argument("--require-strict-success", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    episodes_dir = args.episode_config_dir
    if (episodes_dir / "episodes").is_dir():
        episodes_dir = episodes_dir / "episodes"
    episodes_dir = episodes_dir.resolve()
    episodes = _episode_files(episodes_dir)
    if not episodes:
        raise FileNotFoundError(f"No YAML episodes found in {episodes_dir}")
    num_envs = int(args.num_envs) if int(args.num_envs) > 0 else len(episodes)
    num_envs = min(num_envs, len(episodes))

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.run_name or f"reset_settle_{episodes_dir.parent.name}_{num_envs}env"
    out_dir = (args.output_root / f"{timestamp}_{run_name}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_run_name = f"{run_name}_serl_probe"
    train_output_dir = (
        args.train_output_root.resolve()
        if args.train_output_root is not None
        else REPO_ROOT / "outputs" / "agentic_reward_curriculum_20260524_depth_correction" / "policy_train_runs"
    )
    episode_length_s = float(args.episode_length_s)
    if episode_length_s <= 0.0:
        episode_length_s = max(1.0, float(args.steps) / 20.0 + 0.5)
    randomization_export = (
        []
        if args.randomization_profile is None
        else [f"export AIC_ISAAC_RANDOMIZATION_PROFILE={args.randomization_profile};"]
    )
    ik_body_export = (
        []
        if args.ik_body_name is None or not str(args.ik_body_name).strip()
        else [f"export AIC_ISAAC_IK_BODY_NAME={shlex.quote(str(args.ik_body_name).strip())};"]
    )
    extra_env_export: list[str] = []
    for item in args.extra_env or []:
        if "=" not in str(item):
            raise ValueError(f"--extra-env expects KEY=VALUE, got {item!r}")
        key, value = str(item).split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            raise ValueError(f"Invalid --extra-env key {key!r}")
        extra_env_export.append(f"export {key}={shlex.quote(value)};")
    zero_action_args = (
        [
            f"--debug_audit_steps {int(args.steps)}",
            "--debug_audit_start_step 1",
            "--debug_audit_constant_action 0.0 0.0 0.0 0.0 0.0 0.0",
        ]
        if bool(args.zero_action)
        else []
    )
    collision_disable_args: list[str] = []
    for pattern in args.disable_collision_prim_regex or []:
        if str(pattern).strip():
            collision_disable_args.extend(["--disable_collision_prim_regex", shlex.quote(str(pattern))])
    collision_replacement_args = (
        ["--replace_sfp_body_sdf_collision_with_sdf_boxes"]
        if bool(args.replace_sfp_body_sdf_collision_with_sdf_boxes)
        else []
    )
    if bool(args.replace_sfp_module_sdf_collision_with_all_sdf_boxes):
        collision_replacement_args.append("--replace_sfp_module_sdf_collision_with_all_sdf_boxes")
    if bool(args.replace_sfp_module_sdf_collision_with_active_sdf_boxes):
        collision_replacement_args.append("--replace_sfp_module_sdf_collision_with_active_sdf_boxes")
    if bool(args.replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes):
        collision_replacement_args.extend(
            [
                "--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes",
                "--sfp_shrunk_box_margin_m",
                *(f"{float(v):g}" for v in args.sfp_shrunk_box_margin_m),
            ]
        )
    cmd = [
        "docker",
        "exec",
        "-w",
        "/workspace/isaaclab",
        args.docker_container,
        "bash",
        "-lc",
        " ".join(
            [
                "set -e;",
                "export AIC_ISAAC_SOLVER_POSITION_ITERATIONS=16;",
                "export AIC_ISAAC_SOLVER_VELOCITY_ITERATIONS=4;",
                *randomization_export,
                *ik_body_export,
                *extra_env_export,
                "./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py",
                "--task AIC-Task-v0",
                f"--num_envs {num_envs}",
                f"--seed {int(args.seed)}",
                f"--device {args.device}",
                "--headless --rendering_mode performance",
                f"--act_only --act_torchscript {args.act_torchscript}",
                f"--output_dir {_rel_container_path(train_output_dir)}",
                f"--run_name {train_run_name}",
                f"--steps {int(args.steps)} --updates 1 --update_every_steps 100000 --warmup_steps 100000 --actor_update_start_steps 100000",
                f"--batch_size 16 --replay_capacity 10000 --n_action_steps 1 --episode_length_s {episode_length_s} --policy_hz 20.0",
                f"--max_wall_time_minutes {float(args.max_wall_time_minutes)}",
                "--log_every 1 --save_every_steps 10000 --save_latest_every_steps 10000",
                "--tcp_action_frame gripper_tcp --tcp_translation_action_clip 0.0 --tcp_rotation_action_clip 0.0",
                "--reward_preset cheatcode_insertion_v1 --target_reward_body sfp_tip_link --target_reward_consistency_body sfp_module_link",
                "--target_reward_orientation_error_mode quat --target_reward_orientation_axis_local 0.0 0.0 1.0",
                f"--target_reward_consistency_gate_mode {args.target_reward_consistency_gate_mode}",
                f"--episode_target_orientation_source {args.episode_target_orientation_source}",
                "--target_success_axial_threshold 0.0005 --target_success_lateral_threshold 0.0005 --target_success_orientation_threshold 0.03",
                "--target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015 --terminate_on_target_success",
                f"--episode_config_dir {_rel_container_path(episodes_dir)}",
                f"--near_gate_reset_max_iterations {int(args.near_gate_reset_max_iterations)}",
                f"--near_gate_reset_position_tolerance {float(args.near_gate_reset_position_tolerance)}",
                f"--near_gate_reset_orientation_tolerance {float(args.near_gate_reset_orientation_tolerance)}",
                "--target_action_guide_weight 0.0 --target_action_guide_collect_blend 0.0 --target_action_guide_collect_steps 0 --no-target_action_guide_train_executed",
                "--no-insertion_action_guard --action_clip 0.0 --isaac_action_scale 1.0 --state_source lerobot_compatible",
                *collision_disable_args,
                *collision_replacement_args,
                (
                    "--task_family sfp_to_nic "
                    f"--target_port_index {int(args.target_port_index)} "
                    f"--target_card_index {int(args.target_card_index)} "
                    f"--target_card_valid {int(args.target_card_valid)} "
                    "--gripper_joint_position 0.0035405"
                ),
                "--freeze_act --debug_diagnostics --diagnostics_every 1 --debug_audit_steps 0 --debug_audit_start_step 1",
                *zero_action_args,
                "--disable_command_pose_rewards --enable_contact_sensor --fix_isaac_ik_xy_sign",
            ]
        ),
    ]
    (out_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    if args.dry_run:
        print(json.dumps({"run_dir": str(out_dir), "command": cmd}, indent=2))
        return 0

    code = _run_command(cmd, cwd=REPO_ROOT, log_path=out_dir / "stdout.log")
    latest_runs = sorted(train_output_dir.glob(f"*_{train_run_name}"))
    train_run_dir = latest_runs[-1] if latest_runs else None
    rows = _jsonl_rows(train_run_dir / "metrics.jsonl") if train_run_dir else []
    if code != 0 or not rows:
        summary = {
            "returncode": code,
            "train_run_dir": None if train_run_dir is None else str(train_run_dir),
            "accepted_count": 0,
            "episode_count": num_envs,
            "failure": "probe_failed_or_missing_metrics",
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    row_idx = int(args.settle_step)
    row = rows[row_idx] if row_idx < 0 else rows[min(row_idx, len(rows) - 1)]
    geom = row.get("post_step_insertion_geometry") or {}
    scores = [
        _score_env(
            geom,
            idx,
            episodes[idx].name,
            min_s_m=float(args.min_s_m),
            max_s_m=float(args.max_s_m),
            max_lateral_m=float(args.max_lateral_m),
            max_theta_rad=float(args.max_theta_rad),
            min_consistency_gate=float(args.min_consistency_gate),
            max_consistency_final_axial_error_m=float(args.max_consistency_final_axial_error_m),
            require_strict_success=bool(args.require_strict_success),
        )
        for idx in range(num_envs)
    ]
    reset_metrics_summary = _write_reset_metrics(
        rows=rows,
        episodes=episodes,
        num_envs=num_envs,
        out_dir=out_dir,
    )
    accepted_dir = out_dir / "accepted_episodes" / "episodes"
    accepted_dir.mkdir(parents=True, exist_ok=True)
    for out_idx, score in enumerate([s for s in scores if s.accepted], start=1):
        shutil.copy2(episodes[score.env], accepted_dir / f"episode_{out_idx:06d}.yaml")

    summary = {
        "returncode": code,
        "train_run_dir": str(train_run_dir),
        "score_step": row.get("step"),
        "episode_count": num_envs,
        "accepted_count": sum(1 for s in scores if s.accepted),
        "accepted_dir": str(accepted_dir.parent),
        "post_step_reset_metrics": reset_metrics_summary,
        "scores": [asdict(s) for s in scores],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "accepted_manifest.json").write_text(
        json.dumps([asdict(s) for s in scores if s.accepted], indent=2),
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("train_run_dir", "score_step", "episode_count", "accepted_count", "accepted_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
