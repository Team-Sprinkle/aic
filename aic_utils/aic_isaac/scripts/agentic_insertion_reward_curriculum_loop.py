#!/usr/bin/env python3
"""Bounded agentic reward/curriculum/guide tuning harness for Isaac insertion.

The harness is deliberately conservative: it ranks and promotes candidates from
post-step semantic insertion metrics, module-body consistency, and artifact
availability. Reward return alone is never a success criterion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML is present in repo envs, but keep JSON usable.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DATE = "20260523"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / f"agentic_reward_curriculum_{DATE}"
DEFAULT_ACT = "aic/outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt"
DEFAULT_EPISODES = "aic/outputs/analysis/isaac_near_gate_handoff_r105/episode_configs/episodes"
BEST_CHECKPOINT = "aic/outputs/one_day_insertion_pipeline/multgate_semantic_consistency_20260517/best_semantic_progress10_poststep_finalrot_latest_checkpoint.pt"


@dataclass
class StrictThresholds:
    min_depth_fraction: float = 0.90
    max_lateral_m: float = 0.0005
    max_theta_rad: float = 0.030
    min_module_consistency_gate: float = 0.80
    max_force_n: float = 35.0


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return _default_config()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs; use JSON or install yaml")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _default_config() -> dict[str, Any]:
    return {
        "experiment": "default_bounded_retention_orientation",
        "docker_container": "isaac-lab-base",
        "act_torchscript": DEFAULT_ACT,
        "episode_config_dir": DEFAULT_EPISODES,
        "checkpoint": BEST_CHECKPOINT,
        "num_envs": 2,
        "steps": 180,
        "updates": 180,
        "max_wall_time_minutes": 25,
        "candidates": [
            {
                "name": "retention_r05_ax5um_rot0015_rejectr",
                "hypothesis": "Tight retention with predicted-r rejection prevents final lateral drift without adding rotation authority.",
                "family": "retention_guard",
                "flags": {
                    "--insertion_action_guard_lateral_threshold_m": "0.0005",
                    "--insertion_action_guard_lateral_step_m": "0.00015",
                    "--insertion_action_guard_centered_axial_step_m": "0.000005",
                    "--target_action_guide_final_orientation_rotation_step_size": "0.0015",
                    "--target_action_guide_rotation_compensation_clip_m": "0.0005",
                    "--insertion_action_guard_retention_require_orientation_gate": None,
                    "--insertion_action_guard_reject_predicted_r_increase": None,
                    "--insertion_action_guard_predicted_r_increase_margin_m": "0.000025",
                    "--insertion_action_guard_predicted_r_reject_backoff_m": "0.00005",
                },
            },
            {
                "name": "retention_r075_ax10um_axisfinal",
                "hypothesis": "Axis-only only in the final window may reduce theta while preserving main full-quat approach.",
                "family": "final_orientation_refinement",
                "flags": {
                    "--target_action_guide_orientation_probe_basis": None,
                    "--target_action_guide_orientation_probe_lateral_penalty": "120.0",
                    "--target_action_guide_adaptive_orientation_sign": None,
                    "--target_action_guide_adaptive_orientation_flip_margin_rad": "0.00025",
                    "--insertion_action_guard_lateral_threshold_m": "0.00075",
                    "--insertion_action_guard_lateral_step_m": "0.00020",
                    "--insertion_action_guard_centered_axial_step_m": "0.000010",
                    "--target_action_guide_final_axis_only_orientation": None,
                    "--target_action_guide_final_orientation_rotation_step_size": "0.0020",
                    "--target_action_guide_rotation_compensation_clip_m": "0.0005",
                    "--insertion_action_guard_retention_require_orientation_gate": None,
                    "--insertion_action_guard_reject_predicted_r_increase": None,
                },
            },
            {
                "name": "retention_r10_ax20um_semantic_late",
                "hypothesis": "Slightly larger retention tube allows axial progress while depth-gated semantic consistency prevents tip-only seating.",
                "family": "semantic_consistency_curriculum",
                "flags": {
                    "--insertion_action_guard_lateral_threshold_m": "0.0010",
                    "--insertion_action_guard_lateral_step_m": "0.00025",
                    "--insertion_action_guard_centered_axial_step_m": "0.000020",
                    "--target_reward_cheatcode_semantic_progress_weight": "0.25",
                    "--target_reward_cheatcode_semantic_loss_weight": "0.35",
                    "--target_action_guide_final_orientation_rotation_step_size": "0.0015",
                    "--insertion_action_guard_retention_require_orientation_gate": None,
                    "--insertion_action_guard_reject_predicted_r_increase": None,
                },
            },
            {
                "name": "retention_r12_ax40um_no_finalrot",
                "hypothesis": "If orientation hold is stealing axial progress, removing final rotation should classify no-axial-progress vs orientation-blocked.",
                "family": "ablation",
                "flags": {
                    "--insertion_action_guard_lateral_threshold_m": "0.0012",
                    "--insertion_action_guard_lateral_step_m": "0.00025",
                    "--insertion_action_guard_centered_axial_step_m": "0.000040",
                    "--target_action_guide_final_orientation_depth_m": "nan",
                    "--target_action_guide_final_orientation_rotation_step_size": "0.0",
                    "--insertion_action_guard_retention_require_orientation_gate": None,
                    "--insertion_action_guard_reject_predicted_r_increase": None,
                },
            },
        ],
    }


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _by_env(metric: dict[str, Any], key: str) -> list[float]:
    val = metric.get(f"{key}_by_env")
    if isinstance(val, list):
        out = []
        for item in val:
            try:
                out.append(float(item))
            except Exception:
                pass
        return out
    val = metric.get(f"{key}_mean")
    if val is None:
        return []
    try:
        return [float(val)]
    except Exception:
        return []


def _orientation_by_env(geom: dict[str, Any]) -> list[float]:
    phase = geom.get("cheatcode_phase_reward") or {}
    theta = phase.get("theta_by_env") or phase.get("orientation_error_by_env")
    if isinstance(theta, list):
        return [float(x) for x in theta]
    for key in ("orientation_error_rad_by_env", "theta_rad_by_env"):
        val = geom.get(key)
        if isinstance(val, list):
            return [float(x) for x in val]
    mean = phase.get("theta_mean") or phase.get("orientation_error_mean") or geom.get("orientation_error_rad_mean")
    return [] if mean is None else [float(mean)]


def _module_consistency_by_env(row: dict[str, Any], all_body: dict[str, Any], tip_depths: list[float]) -> list[float]:
    geom = row.get("post_step_insertion_geometry") or {}
    phase = geom.get("cheatcode_phase_reward") or {}
    gate = phase.get("g_semantic_by_env")
    if isinstance(gate, list):
        return [float(x) for x in gate]
    module = all_body.get("sfp_module_link") or {}
    module_depths = _by_env(module, "signed_depth_m")
    module_lat = _by_env(module, "lateral_error_m")
    if not module_depths:
        return [0.0 for _ in tip_depths]
    target_depths = _by_env(geom, "target_depth_m") or _by_env(module, "target_depth_m")
    if not target_depths:
        target_depths = [0.008 for _ in module_depths]
    out: list[float] = []
    for idx, depth in enumerate(module_depths):
        tip = tip_depths[min(idx, len(tip_depths) - 1)] if tip_depths else 0.0
        target = target_depths[min(idx, len(target_depths) - 1)]
        lat = module_lat[min(idx, len(module_lat) - 1)] if module_lat else 0.0
        gap = tip - depth
        expected = target - gap
        axial_gate = math.exp(-((depth - expected) / 0.004) ** 2)
        lateral_gate = math.exp(-((lat / 0.0015) ** 2))
        out.append(float(axial_gate * lateral_gate))
    return out


def summarize_run(run_dir: Path, thresholds: StrictThresholds = StrictThresholds()) -> dict[str, Any]:
    rows = _jsonl_rows(run_dir / "metrics.jsonl")
    best: dict[str, Any] = {
        "run_dir": str(run_dir),
        "has_metrics": bool(rows),
        "strict_success": False,
        "failure_label": "insufficient_logs",
        "best_score": -1.0e9,
        "best_step": None,
        "best_env": None,
        "best_s_m": None,
        "best_r_m": None,
        "best_theta_rad": None,
        "best_depth_fraction": None,
        "best_module_consistency": None,
        "max_force_n": None,
        "visual_artifacts": _visual_artifacts(run_dir),
    }
    if not rows:
        return best
    max_force = max(float(row.get("force_norm_mean", 0.0) or 0.0) for row in rows)
    best["max_force_n"] = max_force
    for row in rows:
        geom = row.get("post_step_insertion_geometry") or row.get("pre_step_insertion_geometry") or {}
        all_body = row.get("post_step_all_body_insertion_geometry") or {}
        s_vals = _by_env(geom, "signed_depth_m")
        r_vals = _by_env(geom, "lateral_error_m")
        depth_fracs = _by_env(geom, "depth_fraction")
        theta_vals = _orientation_by_env(geom)
        module_vals = _module_consistency_by_env(row, all_body, s_vals)
        n = max(len(s_vals), len(r_vals), len(depth_fracs), len(theta_vals), len(module_vals))
        for env_id in range(n):
            s = s_vals[min(env_id, len(s_vals) - 1)] if s_vals else float("-inf")
            r = r_vals[min(env_id, len(r_vals) - 1)] if r_vals else float("inf")
            depth_frac = depth_fracs[min(env_id, len(depth_fracs) - 1)] if depth_fracs else 0.0
            theta = theta_vals[min(env_id, len(theta_vals) - 1)] if theta_vals else float("inf")
            module = module_vals[min(env_id, len(module_vals) - 1)] if module_vals else 0.0
            strict = (
                depth_frac >= thresholds.min_depth_fraction
                and r <= thresholds.max_lateral_m
                and theta <= thresholds.max_theta_rad
                and module >= thresholds.min_module_consistency_gate
                and max_force <= thresholds.max_force_n
            )
            score = (
                10.0 * min(depth_frac, 1.0)
                - 1000.0 * max(r - thresholds.max_lateral_m, 0.0)
                - 20.0 * max(theta - thresholds.max_theta_rad, 0.0)
                + module
            )
            if strict or score > float(best["best_score"]):
                best.update(
                    {
                        "strict_success": bool(strict),
                        "best_score": score,
                        "best_step": row.get("step"),
                        "best_env": env_id,
                        "best_s_m": s,
                        "best_r_m": r,
                        "best_theta_rad": theta,
                        "best_depth_fraction": depth_frac,
                        "best_module_consistency": module,
                    }
                )
    best["failure_label"] = "strict_success" if best["strict_success"] else classify_summary(best, rows)
    return best


def classify_summary(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "insufficient_logs"
    s = _num(summary.get("best_s_m"), -1.0)
    r = _num(summary.get("best_r_m"), 1.0)
    theta = _num(summary.get("best_theta_rad"), 1.0)
    depth = _num(summary.get("best_depth_fraction"), 0.0)
    module = _num(summary.get("best_module_consistency"), 0.0)
    max_force = _num(summary.get("max_force_n"), 0.0)
    final_step = max(int(row.get("step", 0) or 0) for row in rows)
    configured_steps = _configured_steps(summary["run_dir"])
    if max_force >= 34.0:
        return "reset_regression" if final_step <= 3 else "contact_spike"
    if depth >= 0.90 and r <= 0.0005 and theta <= 0.030 and module < 0.80:
        return "near_success_module_consistency_blocked"
    if depth >= 0.90 and r <= 0.0005 and module >= 0.80 and theta > 0.030:
        return "near_success_orientation_blocked"
    if s > 0.0 and module < 0.40:
        return "tip_depth_false_positive"
    if s > 0.0 and r > 0.0015:
        return "lateral_bypass"
    if _metric_max(rows, "insertion_action_guard_final_orientation_induced_tip_delta_m_mean") > 0.0005 and r > 0.001:
        return "rotation_induced_lateral_sweep"
    if depth < 0.20 and r <= 0.001 and theta <= 0.040:
        return "no_axial_progress"
    if configured_steps and final_step + 1 >= configured_steps and depth < 0.90:
        return "timeout_or_episode_too_short"
    if _metric_max(rows, "insertion_action_guard_correction_norm_mean") > 0.0015:
        return "controller_realization_mismatch"
    if _metric_max(rows, "adapter_clipped_fraction") > 0.5:
        return "unstable_learning_or_actor_drift"
    if r <= 0.0005 and 0.030 < theta < 0.065:
        return "orientation_plateau_env_or_card_dependent"
    return "insufficient_logs"


def _num(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _metric_max(rows: list[dict[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        try:
            vals.append(float(row.get(key, 0.0) or 0.0))
        except Exception:
            pass
    return max(vals) if vals else 0.0


def _configured_steps(run_dir: str) -> int | None:
    path = Path(run_dir) / "train_config.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for key in ("steps",):
        if key in data:
            return int(data[key])
    result = data.get("result") or {}
    if "steps_completed" in result:
        return int(result["steps_completed"])
    return None


def _visual_artifacts(run_dir: Path) -> dict[str, Any]:
    images = sorted(str(p) for p in (run_dir / "step_images").glob("step_*/*_camera.png"))[:18]
    videos = sorted(str(p) for p in (run_dir / "videos").glob("*.mp4"))
    return {"image_count": len(images), "video_count": len(videos), "sample_images": images[:6], "videos": videos[:6]}


def _base_isaac_args(config: dict[str, Any], candidate: dict[str, Any], output_root: Path) -> list[str]:
    run_name = str(candidate["name"])
    steps = int(config.get("steps", 180))
    updates = int(config.get("updates", steps))
    output_root_abs = output_root if output_root.is_absolute() else (REPO_ROOT / output_root)
    args = [
        "cd /workspace/isaaclab && ./isaaclab.sh -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py",
        "--headless --task AIC-Task-v0",
        f"--num_envs {int(config.get('num_envs', 2))}",
        "--seed 236 --device cuda:0 --enable_cameras --rendering_mode performance",
        f"--output_dir aic/{output_root_abs.relative_to(REPO_ROOT)}/runs",
        f"--run_name {shlex.quote(run_name)}",
        f"--steps {steps} --updates {updates}",
        f"--update_every_steps 100000 --warmup_steps {steps} --actor_update_start_steps 100000",
        "--batch_size 32 --replay_capacity 10000",
        "--act_only --act_only_actor_mode act_direct",
        f"--act_torchscript {shlex.quote(str(config.get('act_torchscript', DEFAULT_ACT)))}",
        "--n_action_steps 1 --tcp_action_frame root",
        "--tcp_translation_action_clip 0.00025 --tcp_rotation_action_clip 0.003",
        "--reward_preset cheatcode_insertion_v1 --disable_command_pose_rewards",
        "--target_reward_body sfp_tip_link",
        "--target_reward_orientation_error_mode axis --target_reward_orientation_axis_local 0.0 0.0 1.0",
        "--target_reward_cheatcode_action_axis_gate --target_reward_cheatcode_action_axis_source body_delta",
        "--target_reward_consistency_body auto --target_reward_consistency_axial_std 0.004 --target_reward_consistency_lateral_sigma 0.0015",
        "--target_success_consistency_axial_threshold 0.001 --target_success_consistency_lateral_threshold 0.0015",
        f"--episode_config_dir {shlex.quote(str(config.get('episode_config_dir', DEFAULT_EPISODES)))}",
        "--near_gate_reset_max_iterations 8 --near_gate_reset_position_tolerance 0.002 --near_gate_reset_orientation_tolerance 0.05",
        f"--target_action_guide_mode cheatcode_transform --target_action_guide_collect_blend 1.0 --target_action_guide_collect_steps {steps}",
        "--target_action_guide_step_size 0.00020 --target_action_guide_rotation_step_size 0.0015",
        "--target_action_guide_separate_rotation_compensation --target_action_guide_rotation_compensation_clip_m 0.0005",
        "--target_action_guide_orientation_switch_rad 0.035",
        "--target_action_guide_final_orientation_depth_m 0.005 --target_action_guide_final_orientation_lateral_m 0.0006",
        "--target_action_guide_final_orientation_threshold_rad 0.030 --target_action_guide_final_orientation_axial_step_size 0.0",
        "--insertion_action_guard --insertion_action_guard_zero_rotation_when_offcenter",
        "--insertion_action_guard_retention --insertion_action_guard_retention_entry_depth_m 0.0",
        "--insertion_action_guard_retention_lateral_threshold_m 0.0010 --insertion_action_guard_retention_min_axial_step_m 0.000005",
        "--debug_diagnostics --diagnostics_every 1 --save_step_images --image_log_every 20 --max_logged_image_steps 180 --debug_visual_overlays",
        f"--max_wall_time_minutes {float(config.get('max_wall_time_minutes', 25))}",
    ]
    checkpoint = config.get("checkpoint")
    if checkpoint:
        args.append(f"--checkpoint {shlex.quote(str(checkpoint))}")
    for flag, value in (candidate.get("flags") or {}).items():
        args.append(str(flag) if value is None else f"{flag} {shlex.quote(str(value))}")
    return args


def build_docker_command(config: dict[str, Any], candidate: dict[str, Any], output_root: Path) -> str:
    inner = " ".join(_base_isaac_args(config, candidate, output_root))
    container = str(config.get("docker_container", "isaac-lab-base"))
    num_envs = int(config.get("num_envs", 2))
    return (
        "LC_USER_ID=yoonjung docker exec "
        "-e AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1 "
        "-e AIC_ISAAC_RANDOMIZATION_PROFILE=none "
        f"-e CUDA_VISIBLE_DEVICES=0 -e DEVICE=cuda:0 -e NUM_ENVS={num_envs} -e RENDERING_MODE=performance "
        f"{shlex.quote(container)} bash -lc {shlex.quote(inner)}"
    )


def _write_iteration(output_root: Path, idx: int, candidate: dict[str, Any], command: str, previous_best: dict[str, Any] | None, new_metrics: dict[str, Any], decision: str) -> None:
    iter_dir = output_root / "iterations" / f"iter_{idx:03d}_{candidate['name']}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    decision_doc = {
        "hypothesis": candidate.get("hypothesis"),
        "family": candidate.get("family"),
        "config_changes": candidate.get("flags", {}),
        "expected_failure_mode_addressed": candidate.get("family"),
        "command": command,
        "previous_best_metrics": previous_best,
        "new_metrics": new_metrics,
        "decision": decision,
    }
    (iter_dir / "agent_decision.json").write_text(json.dumps(decision_doc, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# Iteration {idx:03d}: {candidate['name']}",
        "",
        f"Hypothesis: {candidate.get('hypothesis')}",
        f"Decision: `{decision}`",
        f"Failure label: `{new_metrics.get('failure_label')}`",
        "",
        "```bash",
        command,
        "```",
        "",
        "## Metrics",
        "",
        json.dumps(new_metrics, indent=2, sort_keys=True),
    ]
    (iter_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_repro(output_root: Path, config: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (output_root / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (output_root / "git_diff.patch").write_text(_run_git(["diff", "--"]), encoding="utf-8")


def _latest_run(output_root: Path, candidate_name: str) -> Path | None:
    runs = sorted((output_root / "runs").glob(f"*_{candidate_name}"), key=lambda p: p.stat().st_mtime)
    return runs[-1] if runs else None


def _execute(command: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(command, cwd=REPO_ROOT, shell=True, text=True, stdout=f, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def run_loop(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    output_root = args.output_root
    _copy_repro(output_root, config)
    thresholds = StrictThresholds()
    existing_runs = [Path(p) for p in config.get("existing_runs", [])]
    existing_runs += list((output_root / "runs").glob("*"))
    summaries = [summarize_run(path, thresholds) for path in existing_runs if path.is_dir()]
    previous_best = max(summaries, key=lambda x: float(x.get("best_score", -1.0e9)), default=None)
    all_rows: list[dict[str, Any]] = []
    for item in summaries:
        item["decision"] = "baseline"
        all_rows.append(item)

    for idx, candidate in enumerate(config.get("candidates", []), start=1):
        command = build_docker_command(config, candidate, output_root)
        if args.max_iterations and idx > args.max_iterations:
            break
        if args.execute:
            code = _execute(command, output_root / "iterations" / f"iter_{idx:03d}_{candidate['name']}" / "command.log")
            run_dir = _latest_run(output_root, str(candidate["name"]))
            metrics = {"returncode": code, "failure_label": "insufficient_logs", "strict_success": False}
            if run_dir is not None:
                metrics = summarize_run(run_dir, thresholds)
                metrics["returncode"] = code
        else:
            metrics = {"strict_success": False, "failure_label": "not_executed", "command_only": True}
        decision = _decision(previous_best, metrics)
        _write_iteration(output_root, idx, candidate, command, previous_best, metrics, decision)
        metrics["decision"] = decision
        metrics["candidate"] = candidate["name"]
        metrics["command"] = command
        all_rows.append(metrics)
        if decision == "promote":
            previous_best = metrics
        if metrics.get("strict_success") and args.stop_on_success:
            break

    _write_results(output_root, all_rows, previous_best)
    return 0


def _decision(previous_best: dict[str, Any] | None, metrics: dict[str, Any]) -> str:
    if metrics.get("strict_success"):
        return "promote"
    if metrics.get("command_only"):
        return "retry"
    if previous_best is None:
        return "promote"
    if float(metrics.get("best_score", -1.0e9)) > float(previous_best.get("best_score", -1.0e9)) + 0.25:
        return "promote"
    label = metrics.get("failure_label")
    if label in {"insufficient_logs", "reset_regression", "contact_spike"}:
        return "retry"
    return "reject"


def _write_results(output_root: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    (output_root / "agent_loop_results.json").write_text(json.dumps({"runs": rows, "best": best}, indent=2, sort_keys=True), encoding="utf-8")
    csv_path = output_root / "agent_loop_results.csv"
    fieldnames = [
        "candidate",
        "run_dir",
        "strict_success",
        "failure_label",
        "decision",
        "best_step",
        "best_env",
        "best_s_m",
        "best_r_m",
        "best_theta_rad",
        "best_depth_fraction",
        "best_module_consistency",
        "max_force_n",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--execute", action="store_true", help="Actually launch generated docker/Isaac commands.")
    parser.add_argument("--max-iterations", type=int, default=0, help="0 means all candidates.")
    parser.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--summarize-run", type=Path, default=None, help="Only summarize/classify one existing run folder.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.summarize_run is not None:
        print(json.dumps(summarize_run(args.summarize_run), indent=2, sort_keys=True))
        return 0
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
