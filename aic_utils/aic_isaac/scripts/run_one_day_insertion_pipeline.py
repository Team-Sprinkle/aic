#!/usr/bin/env python3
"""Autonomous independent-GPU near-gate insertion experiment pipeline."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
INNER_TRAIN = "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py"
AUDIT_SCRIPT = REPO / "aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py"
EPISODE_CONFIG_SCRIPT = REPO / "aic_utils/aic_isaac/scripts/isaac_episode_configs.py"
DEFAULT_ACT = REPO / "outputs/train/clean_sfp_sc/act/bc/20260510_clean_act_nact8_400k/act_policy_ts_175000_cuda0.pt"
DEFAULT_EPISODES = REPO / "outputs/analysis/isaac_near_gate_6mm_orientation_gate/episode_configs/episodes"


@dataclass
class Experiment:
    stage: str
    name: str
    steps: int
    updates: int
    args: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    parent: str | None = None


@dataclass
class Running:
    exp: Experiment
    gpu: str
    log_path: Path
    process: subprocess.Popen
    launched_at: float


@dataclass(frozen=True)
class CurriculumStage:
    key: str
    label: str
    axial_min_m: float
    axial_max_m: float
    lateral_min_m: float
    lateral_max_m: float
    reward_preset: str
    hover_weight: float
    hover_depth: float
    axial_weight: float
    corridor_weight: float
    inside_weight: float
    retreat_weight: float
    alignment_r_pass_m: float
    alignment_theta_pass_rad: float
    max_lateral_explosion_m: float


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rel_to_container(path: Path) -> str:
    path = path.resolve()
    try:
        return "aic/" + str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _checkpoint_for_row(row: dict[str, Any] | None) -> Path | None:
    if not row:
        return None
    run_dir_raw = row.get("run_dir")
    if not run_dir_raw:
        return None
    checkpoint = Path(str(run_dir_raw)) / "checkpoint_latest.pt"
    return checkpoint if checkpoint.exists() else None


def _repo_python() -> str:
    pixi_python = REPO / ".pixi/envs/default/bin/python"
    return str(pixi_python if pixi_python.exists() else sys.executable)


def _run_checked(cmd: list[str], *, cwd: Path = REPO, log: Path | None = None) -> int:
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=cwd, text=True)
    return int(proc.returncode)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _safe_dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False)


def _stage_slug(stage: CurriculumStage) -> str:
    return stage.label.replace(" ", "_").replace("/", "_")


def _linspace_samples(min_v: float, max_v: float, count: int) -> list[float]:
    if count <= 1 or abs(max_v - min_v) < 1.0e-12:
        return [float((min_v + max_v) * 0.5)]
    return [float(min_v + (max_v - min_v) * i / (count - 1)) for i in range(count)]


def _episode_signed_geometry(episode: dict[str, Any]) -> dict[str, float]:
    scene = episode.get("scene") or {}
    target = scene.get("target") or {}
    start = scene.get("start_near_gate") or {}
    entrance = (target.get("entrance_pose_world") or {}).get("position")
    tip = start.get("reference_tip_center_position_world") or start.get("reference_reward_body_start_position_world")
    axis = target.get("insertion_axis_world")
    seated = (target.get("target_pose_world") or {}).get("position")
    if not (isinstance(entrance, list) and isinstance(tip, list) and isinstance(axis, list) and isinstance(seated, list)):
        return {}
    delta = [float(tip[i]) - float(entrance[i]) for i in range(3)]
    axis_f = [float(v) for v in axis]
    signed_s = sum(delta[i] * axis_f[i] for i in range(3))
    lateral_sq = max(0.0, sum(v * v for v in delta) - signed_s * signed_s)
    target_delta = [float(seated[i]) - float(entrance[i]) for i in range(3)]
    target_depth = sum(target_delta[i] * axis_f[i] for i in range(3))
    return {
        "s_start_m": signed_s,
        "r_start_m": math.sqrt(lateral_sq),
        "target_depth_m": target_depth,
        "requested_axial_m": _float(start.get("axial_distance_m")) or float("nan"),
        "requested_lateral_m": _float(start.get("lateral_distance_m")) or float("nan"),
    }


def _curriculum_stage_specs(args: argparse.Namespace) -> list[CurriculumStage]:
    return [
        CurriculumStage(
            key="stage_a",
            label="stage_a_20mm_align",
            axial_min_m=float(args.stage_a_axial_m),
            axial_max_m=float(args.stage_a_axial_m),
            lateral_min_m=float(args.stage_a_lateral_min_m),
            lateral_max_m=float(args.stage_a_lateral_max_m),
            reward_preset="cheatcode_alignment_v1",
            hover_weight=0.0,
            hover_depth=-0.020,
            axial_weight=0.0,
            corridor_weight=0.0,
            inside_weight=0.0,
            retreat_weight=0.0,
            alignment_r_pass_m=0.0020,
            alignment_theta_pass_rad=0.08,
            max_lateral_explosion_m=0.030,
        ),
        CurriculumStage(
            key="stage_b",
            label="stage_b_12mm_approach",
            axial_min_m=float(args.stage_b_axial_m),
            axial_max_m=float(args.stage_b_axial_m),
            lateral_min_m=float(args.stage_b_lateral_min_m),
            lateral_max_m=float(args.stage_b_lateral_max_m),
            reward_preset="cheatcode_alignment_v1",
            hover_weight=0.05,
            hover_depth=-0.009,
            axial_weight=0.0,
            corridor_weight=0.0,
            inside_weight=0.0,
            retreat_weight=0.0,
            alignment_r_pass_m=0.0018,
            alignment_theta_pass_rad=0.08,
            max_lateral_explosion_m=0.030,
        ),
        CurriculumStage(
            key="stage_c",
            label="stage_c_6_8mm_entry",
            axial_min_m=float(args.stage_c_axial_min_m),
            axial_max_m=float(args.stage_c_axial_max_m),
            lateral_min_m=float(args.stage_c_lateral_min_m),
            lateral_max_m=float(args.stage_c_lateral_max_m),
            reward_preset="cheatcode_insertion_v1",
            hover_weight=0.10,
            hover_depth=-0.004,
            axial_weight=0.10,
            corridor_weight=0.25,
            inside_weight=0.10,
            retreat_weight=0.10,
            alignment_r_pass_m=0.0015,
            alignment_theta_pass_rad=0.06,
            max_lateral_explosion_m=0.010,
        ),
        CurriculumStage(
            key="stage_d",
            label="stage_d_3_6mm_final",
            axial_min_m=float(args.stage_d_axial_min_m),
            axial_max_m=float(args.stage_d_axial_max_m),
            lateral_min_m=float(args.stage_d_lateral_min_m),
            lateral_max_m=float(args.stage_d_lateral_max_m),
            reward_preset="cheatcode_insertion_v1",
            hover_weight=0.10,
            hover_depth=-0.003,
            axial_weight=0.25,
            corridor_weight=1.00,
            inside_weight=0.20,
            retreat_weight=0.20,
            alignment_r_pass_m=0.0010,
            alignment_theta_pass_rad=0.06,
            max_lateral_explosion_m=0.010,
        ),
    ]


def _stage_reward_args(stage: CurriculumStage) -> dict[str, Any]:
    return {
        "reward_preset": stage.reward_preset,
        "target_reward_cheatcode_hover_depth": stage.hover_depth,
        "target_reward_cheatcode_hover_weight": stage.hover_weight,
        "target_reward_cheatcode_axial_progress_weight": stage.axial_weight,
        "target_reward_cheatcode_corridor_weight": stage.corridor_weight,
        "target_reward_cheatcode_inside_alignment_weight": stage.inside_weight,
        "target_reward_cheatcode_retreat_weight": stage.retreat_weight,
        "target_reward_consistency_body": "auto",
        "target_reward_consistency_axial_std": 0.004,
        "target_reward_consistency_lateral_sigma": 0.003,
        "target_success_consistency_axial_threshold": 0.001,
        "target_success_consistency_lateral_threshold": 0.0015,
    }


def _make_stage_request(axial: float, lateral: float, seed: int) -> dict[str, Any]:
    return {
        "task_family": "sfp_to_nic",
        "generation": {"target_accepted_trajectories": 1, "seed": seed},
        "scene": {
            "target": {"entrance_axis_offset_m": 0.0, "seated_depth_m": 0.04872},
            "start_near_gate": {
                "axial_distance_m": round(float(axial), 6),
                "lateral_distance_m": round(float(lateral), 6),
                "min_clearance_m": 0.0,
                "reset_body_name": "gripper_tcp",
                "reset_body_offset_from_reference_world": [-0.007149, 0.002556, 0.059066],
                "reset_body_orientation_wxyz": [0.026548, 0.013188, 0.991236, 0.128732],
            },
            "nic_cards": {"count": 1, "target_card": 0, "target_port": "sfp_port_0"},
        },
    }


def _materialize_curriculum_stage(args: argparse.Namespace, stage: CurriculumStage) -> dict[str, Any]:
    base = REPO / f"outputs/analysis/curriculum_insertion_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
    output_dir = base / _stage_slug(stage) / "episode_configs"
    episodes_dir = output_dir / "episodes"
    if args.reuse_existing_stage_configs and episodes_dir.exists() and list(episodes_dir.glob("episode_*.yaml")):
        return _validate_stage_episodes(stage, episodes_dir, output_dir)

    requests_dir = output_dir / "requests"
    requests_dir.mkdir(parents=True, exist_ok=True)
    for old in requests_dir.glob("*.yaml"):
        old.unlink()

    count = max(1, int(args.episodes_per_stage))
    axial_values = _linspace_samples(stage.axial_min_m, stage.axial_max_m, count)
    lateral_values = _linspace_samples(stage.lateral_min_m, stage.lateral_max_m, count)
    rng = random.Random(1000 + sum(ord(c) for c in stage.key))
    request_paths: list[str] = []
    for idx in range(count):
        axial = axial_values[idx % len(axial_values)]
        lateral = lateral_values[idx % len(lateral_values)]
        if count > 2:
            # Shuffle lateral side direction through the generator RNG while keeping
            # requested magnitudes deterministic and easy to audit.
            seed = rng.randint(1, 10_000_000)
        else:
            seed = 1000 + idx
        req = _make_stage_request(axial, lateral, seed)
        path = requests_dir / f"{stage.key}_{idx + 1:03d}_a{round(axial * 1000):03d}_l{round(lateral * 1000):03d}.yaml"
        path.write_text(_safe_dump_yaml(req), encoding="utf-8")
        request_paths.append(str(path))

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        _repo_python(),
        str(EPISODE_CONFIG_SCRIPT),
        "--multi",
        "--output-dir", str(output_dir),
        "--max-gpus", str(max(1, len(args.gpus))),
    ]
    for path in request_paths:
        cmd.extend(["--input-yaml", path])
    rc = _run_checked(cmd, log=output_dir / "materialize.log")
    if rc != 0:
        raise RuntimeError(f"Failed to materialize {stage.key} episode configs; see {output_dir / 'materialize.log'}")
    return _validate_stage_episodes(stage, episodes_dir, output_dir)


def _validate_stage_episodes(stage: CurriculumStage, episodes_dir: Path, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for ep_path in sorted(episodes_dir.glob("episode_*.yaml")):
        episode = _load_yaml(ep_path)
        geom = _episode_signed_geometry(episode)
        if not geom:
            failures.append(f"{ep_path.name}: missing geometry")
            continue
        row = {"episode": ep_path.name, **geom}
        rows.append(row)
        s = geom["s_start_m"]
        r = geom["r_start_m"]
        if not (-stage.axial_max_m - 0.001 <= s <= -stage.axial_min_m + 0.001):
            failures.append(f"{ep_path.name}: s_start {s:.6f} outside expected negative axial range")
        if not (stage.lateral_min_m - 0.001 <= r <= stage.lateral_max_m + 0.001):
            failures.append(f"{ep_path.name}: r_start {r:.6f} outside expected lateral range")
        if not (0.003 <= geom["target_depth_m"] <= 0.060):
            failures.append(f"{ep_path.name}: target depth {geom['target_depth_m']:.6f} invalid")
    _write_csv(output_dir / "stage_geometry_validation.csv", rows)
    summary = {
        "stage": stage.key,
        "label": stage.label,
        "episodes_dir": str(episodes_dir),
        "episode_count": len(rows),
        "failures": failures,
        "s_start_min_m": min((r["s_start_m"] for r in rows), default=None),
        "s_start_max_m": max((r["s_start_m"] for r in rows), default=None),
        "r_start_min_m": min((r["r_start_m"] for r in rows), default=None),
        "r_start_max_m": max((r["r_start_m"] for r in rows), default=None),
        "target_depth_min_m": min((r["target_depth_m"] for r in rows), default=None),
        "target_depth_max_m": max((r["target_depth_m"] for r in rows), default=None),
    }
    (output_dir / "stage_geometry_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise RuntimeError(f"{stage.key} geometry validation failed: {failures[:3]}")
    return summary


def _iter_metrics(metrics_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not metrics_path.exists():
        return rows
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _find_run_dir(output_root: Path, run_name: str) -> Path | None:
    candidates = sorted(output_root.glob(f"*{run_name}*"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return candidates[-1] if candidates else None


def _metrics_for_run(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {"status": "missing_run_dir"}
    rows = _iter_metrics(run_dir / "metrics.jsonl")
    summary = _load_json(run_dir / "cheatcode_phase_summary.json")
    train_cfg = _load_json(run_dir / "train_config.json")
    if not rows:
        return {
            "status": "missing_metrics",
            "run_dir": str(run_dir),
            "summary": summary,
            "stop_reason": (train_cfg.get("result") or {}).get("stop_reason"),
        }

    s_vals: list[float] = []
    r_vals: list[float] = []
    r_worst_vals: list[float] = []
    theta_vals: list[float] = []
    theta_worst_vals: list[float] = []
    force_vals: list[float] = []
    reward_vals: list[float] = []
    guide_norms: list[float] = []
    action_norms: list[float] = []
    q_vals: list[float] = []
    success_count = 0
    terminated_count = 0
    consistency_axial_vals: list[float] = []
    consistency_lateral_vals: list[float] = []
    first = last = None

    def geom_scalar(geom: dict[str, Any], stem: str) -> float | None:
        # Prefer all-env mean metrics when num_envs > 1.  Fall back to env0 for
        # older logs that predate the mean/by-env diagnostics.
        return _float(geom.get(f"{stem}_mean", geom.get(f"{stem}_env0")))

    def geom_worst(geom: dict[str, Any], stem: str) -> float | None:
        vals = geom.get(f"{stem}_by_env")
        if isinstance(vals, list):
            parsed = [_float(v) for v in vals]
            finite = [v for v in parsed if v is not None]
            if finite:
                return max(finite)
        return geom_scalar(geom, stem)

    for row in rows:
        geom = row.get("pre_step_insertion_geometry") or {}
        phase = geom.get("cheatcode_phase_reward") or {}
        if not first and geom:
            first = geom
        if geom:
            last = geom
        for target, stem in (
            (s_vals, "signed_depth_m"),
            (r_vals, "lateral_error_m"),
            (theta_vals, "orientation_error_rad"),
            (consistency_axial_vals, "consistency_axial_error_m"),
            (consistency_lateral_vals, "consistency_lateral_error_m"),
        ):
            v = geom_scalar(geom, stem)
            if v is not None:
                target.append(v)
        r_worst = geom_worst(geom, "lateral_error_m")
        theta_worst = geom_worst(geom, "orientation_error_rad")
        if r_worst is not None:
            r_worst_vals.append(r_worst)
        if theta_worst is not None:
            theta_worst_vals.append(theta_worst)
        success_by_env = phase.get("success_candidate_by_env")
        if isinstance(success_by_env, list):
            if any((_float(v) or 0.0) >= 0.5 for v in success_by_env):
                success_count += 1
        elif _float(phase.get("success_candidate_env0")) and float(phase["success_candidate_env0"]) >= 0.5:
            success_count += 1
        for target, key in (
            (force_vals, "force_norm_mean"),
            (reward_vals, "reward_mean"),
            (guide_norms, "guide_action_norm_mean"),
            (action_norms, "executed_policy_tcp_action_norm_mean"),
            (q_vals, "q_mean"),
        ):
            v = _float(row.get(key))
            if v is not None:
                target.append(v)
        if row.get("terminated_mean", 0.0):
            terminated_count += 1

    def stat(vals: list[float], fn: str, default: float | None = None) -> float | None:
        if not vals:
            return default
        if fn == "min":
            return min(vals)
        if fn == "max":
            return max(vals)
        if fn == "mean":
            return sum(vals) / len(vals)
        return vals[-1]

    initial_r = geom_scalar(first or {}, "lateral_error_m")
    final_r = geom_scalar(last or {}, "lateral_error_m")
    initial_theta = geom_scalar(first or {}, "orientation_error_rad")
    final_theta = geom_scalar(last or {}, "orientation_error_rad")
    final_s = geom_scalar(last or {}, "signed_depth_m")
    best_r = stat(r_vals, "min")
    best_theta = stat(theta_vals, "min")
    final_r_worst = stat(r_worst_vals, "last")
    final_theta_worst = stat(theta_worst_vals, "last")
    best_r_worst = stat(r_worst_vals, "min")
    best_theta_worst = stat(theta_worst_vals, "min")
    max_s = stat(s_vals, "max")
    bad_axial_r = summary.get("fraction_delta_s_positive_and_r_gt_0p0015")
    bad_axial_theta = summary.get("fraction_delta_s_positive_and_theta_gt_0p06")
    axial_reward_bad_r = summary.get("fraction_axial_progress_positive_and_r_gt_0p0015")
    axial_reward_bad_theta = summary.get("fraction_axial_progress_positive_and_theta_gt_0p06")
    force_clipped_fraction = None
    if force_vals:
        force_clipped_fraction = sum(1 for v in force_vals if v >= 34.9) / len(force_vals)

    alignment_score = 0.0
    if final_r is not None:
        alignment_score -= final_r * 1000.0
    if final_theta is not None:
        alignment_score -= final_theta * 8.0
    if best_r is not None:
        alignment_score -= best_r * 200.0
    if final_r_worst is not None:
        alignment_score -= final_r_worst * 500.0
    if best_theta is not None:
        alignment_score -= best_theta * 2.0
    if final_theta_worst is not None:
        alignment_score -= final_theta_worst * 4.0
    if initial_r is not None and final_r is not None:
        alignment_score += max(0.0, initial_r - final_r) * 800.0
        alignment_score -= max(0.0, final_r - initial_r) * 1200.0
    if initial_theta is not None and final_theta is not None:
        alignment_score += max(0.0, initial_theta - final_theta) * 4.0
    alignment_score -= float(bad_axial_r or 0.0) * 5.0
    alignment_score -= float(bad_axial_theta or 0.0) * 3.0
    alignment_score -= float(force_clipped_fraction or 0.0) * 2.0

    insertion_score = alignment_score
    if max_s is not None:
        insertion_score += max_s * 500.0
    if final_s is not None and final_s > 0.0 and final_r is not None and final_r < 0.0015:
        insertion_score += 5.0
    if consistency_axial_vals:
        insertion_score -= stat(consistency_axial_vals, "last", 0.0) * 100.0  # type: ignore[operator]

    return {
        "status": "ok",
        "run_dir": str(run_dir),
        "steps_logged": len(rows),
        "stop_reason": (train_cfg.get("result") or {}).get("stop_reason"),
        "initial_s_m": s_vals[0] if s_vals else None,
        "final_s_m": final_s,
        "max_s_m": max_s,
        "initial_r_m": initial_r,
        "final_r_m": final_r,
        "best_r_m": best_r,
        "final_r_worst_m": final_r_worst,
        "best_r_worst_m": best_r_worst,
        "initial_theta_rad": initial_theta,
        "final_theta_rad": final_theta,
        "best_theta_rad": best_theta,
        "final_theta_worst_rad": final_theta_worst,
        "best_theta_worst_rad": best_theta_worst,
        "first_g_align_insert_gt_0p5": summary.get("first_step_g_align_insert_gt_0p5"),
        "first_s_gt_0": summary.get("first_step_s_gt_0"),
        "success_candidate_count": success_count,
        "terminated_count": terminated_count,
        "bad_delta_s_r_frac": bad_axial_r,
        "bad_delta_s_theta_frac": bad_axial_theta,
        "bad_axial_reward_r_frac": axial_reward_bad_r,
        "bad_axial_reward_theta_frac": axial_reward_bad_theta,
        "max_force_norm": stat(force_vals, "max"),
        "force_clipped_fraction": force_clipped_fraction,
        "final_consistency_axial_error_m": stat(consistency_axial_vals, "last"),
        "final_consistency_lateral_error_m": stat(consistency_lateral_vals, "last"),
        "mean_reward": stat(reward_vals, "mean"),
        "final_reward": stat(reward_vals, "last"),
        "max_abs_q": max((abs(v) for v in q_vals), default=None),
        "mean_guide_action_norm": stat(guide_norms, "mean"),
        "mean_executed_action_norm": stat(action_norms, "mean"),
        "alignment_score": alignment_score,
        "insertion_score": insertion_score,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _base_inner_args(args: argparse.Namespace, exp: Experiment, run_name: str, output_dir: Path) -> list[str]:
    episode_config_dir = Path(exp.args.get("episode_config_dir") or args.episode_config_dir)
    cmd = [
        INNER_TRAIN,
        "--headless",
        "--task", "AIC-Task-v0",
        "--num_envs", str(args.num_envs),
        "--seed", str(exp.args.get("seed", 1)),
        "--device", "cuda:0",
        "--enable_cameras",
        "--rendering_mode", "performance",
        "--output_dir", _rel_to_container(output_dir),
        "--run_name", run_name,
        "--steps", str(exp.steps),
        "--updates", str(exp.updates),
        "--update_every_steps", str(exp.args.get("update_every_steps", 100000)),
        "--warmup_steps", str(exp.args.get("warmup_steps", exp.steps)),
        "--actor_update_start_steps", str(exp.args.get("actor_update_start_steps", 100000)),
        "--batch_size", str(exp.args.get("batch_size", 32)),
        "--replay_capacity", str(exp.args.get("replay_capacity", 10000)),
        "--act_only",
        "--act_only_actor_mode", str(exp.args.get("act_only_actor_mode", "act_direct")),
        "--act_torchscript", _rel_to_container(args.act_torchscript),
        "--n_action_steps", "1",
        "--tcp_action_frame", str(exp.args.get("tcp_action_frame", "root")),
        "--fix_isaac_ik_xy_sign" if exp.args.get("fix_isaac_ik_xy_sign", True) else "--no-fix_isaac_ik_xy_sign",
        "--no-isaac_ik_xy_sign_by_target_card",
        "--adapter_lr", str(exp.args.get("adapter_lr", 1e-5)),
        "--adapter_delta_clip", str(exp.args.get("adapter_delta_clip", 0.005)),
        "--adapter_penalty_weight", str(exp.args.get("adapter_penalty_weight", 0.05)),
        "--act_preservation_weight", str(exp.args.get("act_preservation_weight", 0.5)),
        "--actor_q_weight", str(exp.args.get("actor_q_weight", 0.0)),
        "--tcp_translation_action_clip", str(exp.args.get("tcp_translation_action_clip", 0.0005)),
        "--tcp_rotation_action_clip", str(exp.args.get("tcp_rotation_action_clip", 0.005)),
        "--reward_preset", str(exp.args.get("reward_preset", "cheatcode_alignment_v1")),
        "--disable_command_pose_rewards",
        "--target_reward_body", str(exp.args.get("target_reward_body", "sfp_tip_link")),
        "--target_reward_orientation_error_mode", str(exp.args.get("target_reward_orientation_error_mode", "axis")),
        "--target_reward_orientation_axis_local", *[
            str(v) for v in exp.args.get("target_reward_orientation_axis_local", (0.0, 0.0, 1.0))
        ],
        "--episode_config_dir", _rel_to_container(episode_config_dir),
        "--near_gate_reset_max_iterations", str(exp.args.get("near_gate_reset_max_iterations", 8)),
        "--near_gate_reset_position_tolerance", str(exp.args.get("near_gate_reset_position_tolerance", 0.002)),
        "--near_gate_reset_orientation_tolerance", str(exp.args.get("near_gate_reset_orientation_tolerance", 0.05)),
        "--target_action_guide_weight", str(exp.args.get("target_action_guide_weight", 0.0)),
        "--target_action_guide_mode", str(exp.args.get("target_action_guide_mode", "cheatcode_transform")),
        "--target_action_guide_step_size", str(exp.args.get("target_action_guide_step_size", 0.0003)),
        "--target_action_guide_rotation_step_size", str(exp.args.get("target_action_guide_rotation_step_size", 0.005)),
        "--target_action_guide_rotation_sign", str(exp.args.get("target_action_guide_rotation_sign", 1.0)),
        "--target_action_guide_axial_step_size", str(exp.args.get("target_action_guide_axial_step_size", 0.0)),
        "--target_action_guide_lateral_switch_m", str(exp.args.get("target_action_guide_lateral_switch_m", 0.0015)),
        "--target_action_guide_axial_blend_lateral_m", str(exp.args.get("target_action_guide_axial_blend_lateral_m", 0.006)),
        "--target_action_guide_orientation_switch_rad", str(exp.args.get("target_action_guide_orientation_switch_rad", 0.06)),
        "--target_action_guide_rotate_while_lateral" if exp.args.get("target_action_guide_rotate_while_lateral", True)
        else "--no-target_action_guide_rotate_while_lateral",
        "--target_action_guide_preinsert_hover_depth", str(exp.args.get("target_action_guide_preinsert_hover_depth", "nan")),
        "--target_action_guide_collect_blend", str(exp.args.get("target_action_guide_collect_blend", 0.0)),
        "--target_action_guide_collect_steps", str(exp.args.get("target_action_guide_collect_steps", 0)),
        "--target_action_guide_collect_decay" if exp.args.get("target_action_guide_collect_decay", False)
        else "--no-target_action_guide_collect_decay",
        "--target_action_guide_prefix_decay" if exp.args.get("target_action_guide_prefix_decay", False)
        else "--no-target_action_guide_prefix_decay",
        "--target_action_guide_train_executed" if exp.args.get("target_action_guide_train_executed", False)
        else "--no-target_action_guide_train_executed",
        "--debug_diagnostics",
        "--diagnostics_every", str(exp.args.get("diagnostics_every", 10)),
        "--save_step_images" if exp.args.get("save_step_images", True) else "--no-save_step_images",
        "--image_log_every", str(exp.args.get("image_log_every", max(1, exp.steps // 3))),
        "--max_logged_image_steps", str(exp.args.get("max_logged_image_steps", exp.steps)),
        "--debug_visual_overlays",
        "--log_every", str(exp.args.get("log_every", 25)),
        "--max_wall_time_minutes", str(exp.args.get("max_wall_time_minutes", args.per_run_max_wall_time_minutes)),
    ]
    if exp.args.get("checkpoint"):
        cmd.extend(["--checkpoint", _rel_to_container(Path(str(exp.args["checkpoint"])))])
    if int(exp.args.get("save_every_steps", 0)) > 0:
        cmd.extend(["--save_every_steps", str(int(exp.args["save_every_steps"]))])
    if int(exp.args.get("save_latest_every_steps", 0)) > 0:
        cmd.extend(["--save_latest_every_steps", str(int(exp.args["save_latest_every_steps"]))])
    if exp.args.get("insertion_action_guard", False):
        cmd.append("--insertion_action_guard")
    else:
        cmd.append("--no-insertion_action_guard")
    for key, flag in [
        ("target_reward_cheatcode_lateral_progress_weight", "--target_reward_cheatcode_lateral_progress_weight"),
        ("target_reward_cheatcode_orientation_progress_weight", "--target_reward_cheatcode_orientation_progress_weight"),
        ("target_reward_cheatcode_near_misaligned_weight", "--target_reward_cheatcode_near_misaligned_weight"),
        ("target_reward_cheatcode_hover_weight", "--target_reward_cheatcode_hover_weight"),
        ("target_reward_cheatcode_hover_depth", "--target_reward_cheatcode_hover_depth"),
        ("target_reward_cheatcode_axial_progress_weight", "--target_reward_cheatcode_axial_progress_weight"),
        ("target_reward_cheatcode_corridor_weight", "--target_reward_cheatcode_corridor_weight"),
        ("target_reward_cheatcode_inside_alignment_weight", "--target_reward_cheatcode_inside_alignment_weight"),
        ("target_reward_cheatcode_retreat_weight", "--target_reward_cheatcode_retreat_weight"),
        ("target_reward_consistency_body", "--target_reward_consistency_body"),
        ("target_reward_consistency_axial_std", "--target_reward_consistency_axial_std"),
        ("target_reward_consistency_lateral_sigma", "--target_reward_consistency_lateral_sigma"),
        ("target_success_consistency_axial_threshold", "--target_success_consistency_axial_threshold"),
        ("target_success_consistency_lateral_threshold", "--target_success_consistency_lateral_threshold"),
    ]:
        if key in exp.args:
            cmd.extend([flag, str(exp.args[key])])
    return cmd


def _launch(args: argparse.Namespace, exp: Experiment, gpu: str, output_dir: Path, log_dir: Path) -> Running:
    run_name = f"{exp.stage}_{exp.name}_{_now()}_gpu{gpu}".replace("/", "_")
    exp.args["run_name"] = run_name
    inner = _base_inner_args(args, exp, run_name, output_dir)
    inner_shell = "cd /workspace/isaaclab && ./isaaclab.sh -p " + shlex.join(inner)
    docker = [
        "docker",
        "exec",
        "-e", "AIC_ISAAC_ALLOW_UNSUPPORTED_RTX_DRIVER=1",
        "-e", "AIC_ISAAC_RANDOMIZATION_PROFILE=none",
        "-e", f"CUDA_VISIBLE_DEVICES={gpu}",
        "-e", "DEVICE=cuda:0",
        "-e", f"NUM_ENVS={args.num_envs}",
        "-e", "RENDERING_MODE=performance",
        args.container_name,
        "bash",
        "-lc",
        inner_shell,
    ]
    shell_cmd = "LC_USER_ID=yoonjung " + shlex.join(docker)
    zsh_cmd = ["zsh", "-lc", shell_cmd]
    log_path = log_dir / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(zsh_cmd, cwd=REPO, stdout=log_file, stderr=subprocess.STDOUT, text=True)
    exp.args["command"] = shlex.join(zsh_cmd)
    return Running(exp=exp, gpu=gpu, log_path=log_path, process=proc, launched_at=time.monotonic())


def _terminate(running: Running) -> None:
    if running.process.poll() is not None:
        return
    running.process.send_signal(signal.SIGTERM)
    try:
        running.process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        running.process.kill()


def _generate_videos(run_dir: Path | None) -> list[str]:
    if run_dir is None:
        return []
    step_dir = run_dir / "step_images"
    if not step_dir.exists():
        return []
    videos_dir = run_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    outputs: list[str] = []
    for cam in ("center_camera", "left_camera", "right_camera"):
        pattern = str(step_dir / "step_*" / f"env_0000_{cam}.png")
        if not glob.glob(pattern):
            continue
        out = videos_dir / f"env0_{cam}_h264.mp4"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-framerate", "20", "-pattern_type", "glob", "-i", pattern,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
            ],
            cwd=REPO,
        )
        if out.exists():
            outputs.append(str(out))
    return outputs


def _run_wave(
    args: argparse.Namespace,
    experiments: list[Experiment],
    *,
    output_dir: Path,
    log_dir: Path,
    stage_deadline: float,
) -> list[dict[str, Any]]:
    pending = list(experiments)
    free_gpus = list(args.gpus)
    running: list[Running] = []
    results: list[dict[str, Any]] = []
    while pending or running:
        while pending and free_gpus and time.monotonic() < stage_deadline:
            exp = pending.pop(0)
            gpu = free_gpus.pop(0)
            print(f"[pipeline] launch {exp.stage}/{exp.name} on GPU {gpu}", flush=True)
            running.append(_launch(args, exp, gpu, output_dir, log_dir))
        if not running:
            break
        for item in list(running):
            code = item.process.poll()
            run_dir = _find_run_dir(output_dir, item.exp.args["run_name"])
            if code is None and run_dir is not None:
                metrics = _metrics_for_run(run_dir)
                if metrics.get("status") == "ok":
                    lateral_limit = float(item.exp.args.get("max_lateral_explosion_m", 0.12))
                    if metrics.get("max_abs_q") is not None and float(metrics["max_abs_q"]) > 100.0:
                        print(f"[pipeline] early-stop Q explosion: {item.exp.name}", flush=True)
                        _terminate(item)
                    if metrics.get("final_r_m") is not None and float(metrics["final_r_m"]) > lateral_limit:
                        print(f"[pipeline] early-stop lateral explosion: {item.exp.name}", flush=True)
                        _terminate(item)
                    if metrics.get("force_clipped_fraction") is not None and float(metrics["force_clipped_fraction"]) > 0.8:
                        print(f"[pipeline] early-stop sustained force clipping: {item.exp.name}", flush=True)
                        _terminate(item)
                    max_s_allowed = _float(item.exp.args.get("max_s_allowed_m"))
                    if (
                        max_s_allowed is not None
                        and metrics.get("max_s_m") is not None
                        and float(metrics["max_s_m"]) > max_s_allowed
                    ):
                        print(
                            f"[pipeline] early-stop stage axial guard: {item.exp.name} "
                            f"max_s={metrics['max_s_m']} limit={max_s_allowed}",
                            flush=True,
                        )
                        _terminate(item)
            if code is not None:
                running.remove(item)
                free_gpus.append(item.gpu)
                run_dir = _find_run_dir(output_dir, item.exp.args["run_name"])
                videos = _generate_videos(run_dir)
                metrics = _metrics_for_run(run_dir)
                metrics.update(
                    {
                        "stage": item.exp.stage,
                        "name": item.exp.name,
                        "gpu": item.gpu,
                        "returncode": code,
                        "log_path": str(item.log_path),
                        "run_name": item.exp.args["run_name"],
                        "episode_config_dir": str(item.exp.args.get("episode_config_dir", args.episode_config_dir)),
                        "videos": ";".join(videos),
                        "command": item.exp.args.get("command"),
                        "exp_args_json": json.dumps(
                            {k: v for k, v in item.exp.args.items() if k != "command"},
                            sort_keys=True,
                            default=str,
                        ),
                    }
                )
                results.append(metrics)
                print(
                    f"[pipeline] done {item.exp.stage}/{item.exp.name} rc={code} "
                    f"score={metrics.get('alignment_score')} r={metrics.get('final_r_m')} theta={metrics.get('final_theta_rad')}",
                    flush=True,
                )
        if time.monotonic() >= stage_deadline:
            for item in running:
                print(f"[pipeline] deadline stop {item.exp.stage}/{item.exp.name}", flush=True)
                _terminate(item)
        time.sleep(5.0)
    return results


def _guide_candidates(
    steps: int,
    *,
    stage_name: str = "stage2",
    episode_config_dir: Path | None = None,
    stage_args: dict[str, Any] | None = None,
) -> list[Experiment]:
    specs = [
        # step, rot_step, rot_sign, lateral_switch, orientation_switch, trans_clip,
        # rot_clip, rotate_while_lateral, fix_ik_xy_sign, insertion_guard, hover_depth
        (0.0002, 0.002, 1.0, 0.0010, 0.05, 0.0003, 0.004, True, False, False, -0.004),
        (0.0003, 0.003, 1.0, 0.0015, 0.06, 0.0005, 0.005, True, False, False, -0.004),
        (0.0003, 0.005, 1.0, 0.0015, 0.06, 0.0005, 0.005, False, False, False, -0.004),
        (0.0002, 0.005, 1.0, 0.0010, 0.05, 0.0003, 0.005, False, False, False, -0.004),
        (0.0003, 0.003, -1.0, 0.0015, 0.06, 0.0005, 0.005, True, False, False, -0.004),
        (0.0003, 0.005, -1.0, 0.0015, 0.06, 0.0005, 0.005, False, False, False, -0.004),
        (0.0002, 0.005, -1.0, 0.0010, 0.05, 0.0003, 0.005, False, False, False, -0.004),
        (0.0003, 0.003, 1.0, 0.0015, 0.06, 0.0005, 0.005, True, True, False, -0.004),
    ]
    out: list[Experiment] = []
    for idx, (step, rot, rot_sign, lat_switch, ori_switch, tclip, rclip, rotate, fix_sign, guard, hover_depth) in enumerate(specs, start=1):
        out.append(
            Experiment(
                stage=stage_name,
                name=f"guide_w1_{idx}",
                steps=steps,
                updates=steps,
                args={
                    **(stage_args or {}),
                    "seed": 100 + idx,
                    "episode_config_dir": episode_config_dir,
                    "reward_preset": (stage_args or {}).get("reward_preset", "cheatcode_alignment_v1"),
                    "act_only_actor_mode": "act_direct",
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": steps,
                    "target_action_guide_step_size": step,
                    "target_action_guide_rotation_step_size": rot,
                    "target_action_guide_rotation_sign": rot_sign,
                    "target_action_guide_lateral_switch_m": lat_switch,
                    "target_action_guide_orientation_switch_rad": ori_switch,
                    "target_action_guide_rotate_while_lateral": rotate,
                    "target_action_guide_preinsert_hover_depth": (stage_args or {}).get(
                        "target_action_guide_preinsert_hover_depth",
                        hover_depth,
                    ),
                    "fix_isaac_ik_xy_sign": fix_sign,
                    "insertion_action_guard": guard,
                    "tcp_translation_action_clip": tclip,
                    "tcp_rotation_action_clip": rclip,
                    "diagnostics_every": 5,
                },
            )
        )
    return out


def _alignment_imitation_candidates(parents: list[dict[str, Any]], steps: int) -> list[Experiment]:
    candidates: list[Experiment] = []
    variants = [
        (0.05, 500, False, 1e-5, 0.003, 0.0005, 0.005),
        (0.10, 500, False, 1e-5, 0.003, 0.0005, 0.010),
        (0.05, 800, True, 3e-6, 0.002, 0.0003, 0.005),
        (0.10, 800, True, 3e-6, 0.002, 0.0003, 0.010),
    ]
    for parent_idx, parent in enumerate(parents[:2], start=1):
        for var_idx, (guide_weight, collect_steps, decay, lr, dclip, tclip, rclip) in enumerate(variants, start=1):
            p_args = _args_from_parent(parent)
            candidates.append(
                Experiment(
                    stage="stage3",
                    name=f"imit_p{parent_idx}_{var_idx}",
                    steps=steps,
                    updates=steps,
                    parent=str(parent.get("run_dir")),
                    args={
                        **p_args,
                        "seed": 300 + parent_idx * 10 + var_idx,
                        "reward_preset": "cheatcode_alignment_v1",
                        "act_only_actor_mode": "act_direct",
                        "target_action_guide_weight": guide_weight,
                        "target_action_guide_collect_blend": 1.0,
                        "target_action_guide_collect_steps": min(collect_steps, steps),
                        "target_action_guide_collect_decay": decay,
                        "target_action_guide_train_executed": True,
                        "actor_q_weight": 0.0,
                        "update_every_steps": 4,
                        "warmup_steps": 50,
                        "actor_update_start_steps": 100,
                        "adapter_lr": lr,
                        "adapter_delta_clip": dclip,
                        "tcp_translation_action_clip": tclip,
                        "tcp_rotation_action_clip": rclip,
                        "diagnostics_every": 10,
                    },
                )
            )
    return candidates


def _weak_insertion_candidates(parents: list[dict[str, Any]], steps: int) -> list[Experiment]:
    out: list[Experiment] = []
    variants = [
        (0.02, 0.10, 0.25, 0.10, 0.10, 500),
        (0.05, 0.10, 0.25, 0.10, 0.10, 1000),
        (0.02, 0.20, 0.50, 0.10, 0.10, 1000),
        (0.05, 0.20, 0.50, 0.20, 0.20, 1000),
    ]
    for parent_idx, parent in enumerate(parents[:2], start=1):
        for var_idx, (q, axial, corridor, inside, retreat, collect_steps) in enumerate(variants, start=1):
            p_args = _args_from_parent(parent)
            out.append(
                Experiment(
                    stage="stage5",
                    name=f"weak_p{parent_idx}_{var_idx}",
                    steps=steps,
                    updates=steps,
                    parent=str(parent.get("run_dir")),
                    args={
                        **p_args,
                        "seed": 500 + parent_idx * 10 + var_idx,
                        "reward_preset": "cheatcode_insertion_v1",
                        "target_action_guide_weight": 0.05,
                        "target_action_guide_collect_blend": 1.0,
                        "target_action_guide_collect_steps": min(collect_steps, steps),
                        "target_action_guide_collect_decay": True,
                        "target_action_guide_train_executed": True,
                        "actor_q_weight": q,
                        "update_every_steps": 4,
                        "warmup_steps": 50,
                        "actor_update_start_steps": 200,
                        "target_reward_cheatcode_axial_progress_weight": axial,
                        "target_reward_cheatcode_corridor_weight": corridor,
                        "target_reward_cheatcode_inside_alignment_weight": inside,
                        "target_reward_cheatcode_retreat_weight": retreat,
                        "diagnostics_every": 10,
                    },
                )
            )
    return out


def _args_from_command(command: Any) -> dict[str, Any]:
    if not isinstance(command, str):
        return {}
    toks = shlex.split(command)
    mapping = {
        "--target_action_guide_step_size": ("target_action_guide_step_size", float),
        "--target_action_guide_rotation_step_size": ("target_action_guide_rotation_step_size", float),
        "--target_action_guide_rotation_sign": ("target_action_guide_rotation_sign", float),
        "--target_action_guide_lateral_switch_m": ("target_action_guide_lateral_switch_m", float),
        "--target_action_guide_orientation_switch_rad": ("target_action_guide_orientation_switch_rad", float),
        "--tcp_translation_action_clip": ("tcp_translation_action_clip", float),
        "--tcp_rotation_action_clip": ("tcp_rotation_action_clip", float),
        "--target_action_guide_preinsert_hover_depth": ("target_action_guide_preinsert_hover_depth", float),
        "--target_reward_orientation_error_mode": ("target_reward_orientation_error_mode", str),
    }
    out: dict[str, Any] = {}
    for idx, tok in enumerate(toks[:-1]):
        if tok in mapping:
            key, typ = mapping[tok]
            try:
                out[key] = typ(toks[idx + 1])
            except Exception:
                pass
    out["target_action_guide_rotate_while_lateral"] = "--target_action_guide_rotate_while_lateral" in toks
    out["fix_isaac_ik_xy_sign"] = "--fix_isaac_ik_xy_sign" in toks and "--no-fix_isaac_ik_xy_sign" not in toks
    out["insertion_action_guard"] = "--insertion_action_guard" in toks
    return out


def _args_from_parent(parent: dict[str, Any]) -> dict[str, Any]:
    raw = parent.get("exp_args_json")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return {
                k: v
                for k, v in parsed.items()
                if k not in {"run_name", "seed", "updates", "steps"}
            }
    return _args_from_command(parent.get("command"))


def _passed_alignment(row: dict[str, Any]) -> bool:
    best_r = _float(row.get("best_r_m"))
    best_theta = _float(row.get("best_theta_rad"))
    initial_r = _float(row.get("initial_r_m"))
    final_r = _float(row.get("final_r_m"))
    initial_theta = _float(row.get("initial_theta_rad"))
    final_theta = _float(row.get("final_theta_rad"))
    bad_r = _float(row.get("bad_delta_s_r_frac")) or 0.0
    force_clip = _float(row.get("force_clipped_fraction")) or 0.0
    max_s = _float(row.get("max_s_m"))
    return bool(
        best_r is not None
        and best_theta is not None
        and initial_r is not None
        and final_r is not None
        and initial_theta is not None
        and final_theta is not None
        and max_s is not None
        and best_r < 0.0015
        and best_theta < 0.06
        and final_r < initial_r
        and final_theta <= initial_theta + 0.02
        # Alignment-only guide rollouts intentionally move from the 6 mm start
        # to the safe pre-insertion hover near -4 mm.  The coarse bad-delta-s
        # fraction counts that as inward motion even though the tip remains
        # safely outside the entrance, so use max_s as the hard guard here.
        and (bad_r < 0.5 or max_s < -0.001)
        and force_clip < 0.5
    )


def _passed_curriculum_stage(row: dict[str, Any], stage: CurriculumStage) -> bool:
    if int(_float(row.get("returncode")) or 0) != 0:
        return False
    best_r = _float(row.get("best_r_m"))
    best_r_worst = _float(row.get("best_r_worst_m"))
    best_theta = _float(row.get("best_theta_rad"))
    best_theta_worst = _float(row.get("best_theta_worst_rad"))
    initial_r = _float(row.get("initial_r_m"))
    final_r = _float(row.get("final_r_m"))
    final_r_worst = _float(row.get("final_r_worst_m"))
    final_theta = _float(row.get("final_theta_rad"))
    final_theta_worst = _float(row.get("final_theta_worst_rad"))
    max_s = _float(row.get("max_s_m"))
    bad_r = _float(row.get("bad_delta_s_r_frac")) or 0.0
    force_clip = _float(row.get("force_clipped_fraction")) or 0.0
    if (
        best_r is None
        or best_theta is None
        or initial_r is None
        or final_r is None
        or final_theta is None
        or max_s is None
    ):
        return False
    best_r_guard = best_r_worst if best_r_worst is not None else best_r
    best_theta_guard = best_theta_worst if best_theta_worst is not None else best_theta
    final_r_guard = final_r_worst if final_r_worst is not None else final_r
    final_theta_guard = final_theta_worst if final_theta_worst is not None else final_theta
    r_tolerance = stage.alignment_r_pass_m * 1.25
    theta_tolerance = stage.alignment_theta_pass_rad + 0.01
    r_improved_or_started_centered = final_r <= initial_r or initial_r <= stage.alignment_r_pass_m * 0.5
    base_ok = (
        best_r <= stage.alignment_r_pass_m
        and best_r_guard <= r_tolerance
        and best_theta <= stage.alignment_theta_pass_rad
        and best_theta_guard <= theta_tolerance
        and final_r_guard <= r_tolerance
        and final_theta_guard <= theta_tolerance
        and r_improved_or_started_centered
        and force_clip < 0.5
    )
    if stage.key in {"stage_a", "stage_b"}:
        return bool(base_ok and max_s < -0.001 and bad_r < 0.75)
    if stage.key == "stage_c":
        return bool(base_ok and (bad_r < 0.20 or max_s < 0.0))
    success_count = int(_float(row.get("success_candidate_count")) or 0)
    return bool(base_ok and (success_count > 0 or (max_s > 0.0 and bad_r < 0.10)))


def _write_report(
    path: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    *,
    stage_summaries: list[dict[str, Any]] | None = None,
) -> None:
    by_stage: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_stage.setdefault(str(row.get("stage")), []).append(row)
    best_alignment = max(rows, key=lambda r: float(r.get("alignment_score") or -1e9), default=None)
    best_insertion = max(rows, key=lambda r: float(r.get("insertion_score") or -1e9), default=None)
    lines = [
        "# One-Day Insertion Pipeline - 2026-05-15",
        "",
        f"Output root: `{args.output_root}`",
        f"GPUs: `{','.join(args.gpus)}`",
        f"ACT: `{args.act_torchscript}`",
        f"Episodes: `{args.episode_config_dir}`",
        f"Curriculum mode: `{getattr(args, 'curriculum_mode', 'single')}`",
        "",
        "## Summary",
        "",
        f"- Runs completed: {len(rows)}",
        f"- Best alignment: `{(best_alignment or {}).get('run_dir')}`",
        f"- Best insertion: `{(best_insertion or {}).get('run_dir')}`",
        "",
        "## Metric Table",
        "",
        "| stage | name | score | final r mean mm | final r worst mm | best r mean mm | final theta mean | final theta worst | max s mm | bad inward r frac | force clip frac | run |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if stage_summaries:
        lines.extend(["", "## Curriculum Episode Configs", ""])
        for summary in stage_summaries:
            s_min = _float(summary.get("s_start_min_m"))
            s_max = _float(summary.get("s_start_max_m"))
            r_min = _float(summary.get("r_start_min_m"))
            r_max = _float(summary.get("r_start_max_m"))
            def fmt_mm(value: float | None) -> str:
                return "" if value is None else f"{value * 1000.0:.2f}"
            lines.append(
                f"- `{summary.get('stage')}` episodes: `{summary.get('episodes_dir')}` "
                f"({summary.get('episode_count')} eps, "
                f"s {fmt_mm(s_min)}..{fmt_mm(s_max)} mm, "
                f"r {fmt_mm(r_min)}..{fmt_mm(r_max)} mm)"
            )
        lines.append("")
    for row in sorted(rows, key=lambda r: (str(r.get("stage")), -float(r.get("alignment_score") or -1e9))):
        def mm(key: str) -> str:
            v = _float(row.get(key))
            return "" if v is None else f"{v * 1000.0:.2f}"

        def num(key: str, nd: int = 3) -> str:
            v = _float(row.get(key))
            return "" if v is None else f"{v:.{nd}f}"

        lines.append(
            f"| {row.get('stage')} | {row.get('name')} | {num('alignment_score', 2)} | "
            f"{mm('final_r_m')} | {mm('final_r_worst_m')} | {mm('best_r_m')} | "
            f"{num('final_theta_rad')} | {num('final_theta_worst_rad')} | {mm('max_s_m')} | "
            f"{num('bad_delta_s_r_frac')} | {num('force_clipped_fraction')} | `{row.get('run_dir')}` |"
        )
    lines.extend(["", "## Stage Conclusions", ""])
    for stage, stage_rows in sorted(by_stage.items()):
        best = max(stage_rows, key=lambda r: float(r.get("alignment_score") or -1e9))
        lines.append(f"- `{stage}` best: `{best.get('name')}` at `{best.get('run_dir')}`")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "Use the best guide-only/alignment-imitation command as the safe fallback unless a weak insertion run satisfies strict alignment during positive depth.",
            "Do not treat `sfp_tip_link` positive depth as success without lateral, orientation, and consistency-body checks.",
            "",
            "## Commands",
            "",
        ]
    )
    for row in rows:
        lines.extend([f"### {row.get('stage')} / {row.get('name')}", "", "```bash", str(row.get("command")), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _zero_action_experiments_for_stage(args: argparse.Namespace, stage: CurriculumStage, episodes_dir: Path) -> list[Experiment]:
    base = {
        **_stage_reward_args(stage),
        "episode_config_dir": episodes_dir,
        "max_lateral_explosion_m": stage.max_lateral_explosion_m,
        "act_only_actor_mode": "act_direct",
        "target_action_guide_collect_blend": 0.0,
        "target_action_guide_collect_steps": 0,
        "target_action_guide_weight": 0.0,
        "tcp_translation_action_clip": 0.0,
        "tcp_rotation_action_clip": 0.0,
        "diagnostics_every": 1,
        "image_log_every": 1,
        "max_logged_image_steps": min(int(args.stage1_steps), 50),
    }
    return [
        Experiment(stage.key, "zero_stability", args.stage1_steps, args.stage1_steps, {**base, "seed": 10}),
        Experiment(
            stage.key,
            "zero_tight_reset",
            args.stage1_steps,
            args.stage1_steps,
            {**base, "seed": 11, "near_gate_reset_position_tolerance": 0.0005},
        ),
    ]


def _stage_guided_args(stage: CurriculumStage, episodes_dir: Path) -> dict[str, Any]:
    out = {
        **_stage_reward_args(stage),
        "episode_config_dir": episodes_dir,
        "max_lateral_explosion_m": stage.max_lateral_explosion_m,
        "target_action_guide_rotation_sign": -1.0,
        "fix_isaac_ik_xy_sign": False,
    }
    if stage.key == "stage_a":
        out["target_action_guide_preinsert_hover_depth"] = "nan"
        out["max_s_allowed_m"] = -0.001
    elif stage.key == "stage_b":
        out["target_action_guide_preinsert_hover_depth"] = -0.009
        out["max_s_allowed_m"] = -0.0005
    elif stage.key == "stage_c":
        out["target_action_guide_preinsert_hover_depth"] = -0.004
    else:
        out["target_action_guide_preinsert_hover_depth"] = -0.003
    return out


def _smart4_candidates_for_stage(
    stage: CurriculumStage,
    episodes_dir: Path,
    *,
    parent: dict[str, Any] | None,
    initial_checkpoint: Path | None,
    steps: int,
) -> list[Experiment]:
    """Four hand-picked candidates per stage.

    These are intentionally not a Cartesian grid.  They cover distinct control
    hypotheses: conservative retention, balanced guide imitation, stronger
    orientation correction, and weak gated insertion where appropriate.
    """
    base_parent_args = _args_from_parent(parent or {})
    checkpoint = _checkpoint_for_row(parent) or initial_checkpoint
    base = {
        **_stage_guided_args(stage, episodes_dir),
        **base_parent_args,
        **_stage_reward_args(stage),
        "episode_config_dir": episodes_dir,
        "checkpoint": checkpoint,
        "save_latest_every_steps": max(50, steps // 4),
        "save_every_steps": 0,
        "act_only_actor_mode": "act_direct",
        "target_action_guide_mode": "cheatcode_transform",
        "target_action_guide_rotation_sign": 1.0,
        "target_action_guide_train_executed": True,
        "update_every_steps": 4,
        "warmup_steps": 50,
        "actor_update_start_steps": 100 if stage.key in {"stage_b", "stage_c"} else 150,
        "batch_size": 32,
        "replay_capacity": 20000,
        "diagnostics_every": 10,
        "image_log_every": max(1, steps // 4),
        "max_logged_image_steps": steps,
    }

    if stage.key == "stage_b":
        specs = [
            (
                "balanced_align",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 800),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0003,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0015,
                    "target_action_guide_orientation_switch_rad": 0.06,
                    "target_action_guide_rotate_while_lateral": False,
                    "target_action_guide_axial_step_size": 0.0,
                    "tcp_translation_action_clip": 0.0005,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 1e-5,
                    "adapter_delta_clip": 0.003,
                    "actor_q_weight": 0.0,
                },
            ),
            (
                "rotate_while_align",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 800),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0003,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0015,
                    "target_action_guide_orientation_switch_rad": 0.08,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.0,
                    "tcp_translation_action_clip": 0.0005,
                    "tcp_rotation_action_clip": 0.010,
                    "adapter_lr": 1e-5,
                    "adapter_delta_clip": 0.003,
                    "actor_q_weight": 0.0,
                },
            ),
            (
                "conservative_align",
                {
                    "target_action_guide_weight": 0.10,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1000),
                    "target_action_guide_collect_decay": False,
                    "target_action_guide_step_size": 0.0002,
                    "target_action_guide_rotation_step_size": 0.003,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.06,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.0,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.0,
                },
            ),
            (
                "weak_approach_hover",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 800),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0003,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0015,
                    "target_action_guide_orientation_switch_rad": 0.06,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00005,
                    "tcp_translation_action_clip": 0.0005,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 1e-5,
                    "adapter_delta_clip": 0.003,
                    "actor_q_weight": 0.02,
                    "actor_update_start_steps": 200,
                    "target_reward_cheatcode_axial_progress_weight": 0.02,
                },
            ),
        ]
    elif stage.key == "stage_c":
        specs = [
            (
                "weak_insert_conservative",
                {
                    "target_action_guide_weight": 0.08,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1000),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0002,
                    "target_action_guide_rotation_step_size": 0.004,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.06,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00005,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.02,
                    "target_reward_cheatcode_axial_progress_weight": 0.10,
                    "target_reward_cheatcode_corridor_weight": 0.25,
                },
            ),
            (
                "weak_insert_balanced",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 800),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0003,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0015,
                    "target_action_guide_orientation_switch_rad": 0.06,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00010,
                    "tcp_translation_action_clip": 0.0005,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 1e-5,
                    "adapter_delta_clip": 0.003,
                    "actor_q_weight": 0.02,
                    "target_reward_cheatcode_axial_progress_weight": 0.10,
                    "target_reward_cheatcode_corridor_weight": 0.25,
                },
            ),
            (
                "medium_corridor",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 800),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0003,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.05,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00010,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.05,
                    "target_reward_cheatcode_axial_progress_weight": 0.15,
                    "target_reward_cheatcode_corridor_weight": 0.50,
                },
            ),
            (
                "guarded_insert",
                {
                    "target_action_guide_weight": 0.08,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1000),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0002,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.05,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00005,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.02,
                    "insertion_action_guard": True,
                    "target_reward_cheatcode_axial_progress_weight": 0.10,
                    "target_reward_cheatcode_corridor_weight": 0.25,
                },
            ),
        ]
    else:
        specs = [
            (
                "final_slow_safe",
                {
                    "target_action_guide_weight": 0.08,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1200),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.00015,
                    "target_action_guide_rotation_step_size": 0.004,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.05,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00005,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.02,
                    "target_reward_cheatcode_axial_progress_weight": 0.20,
                    "target_reward_cheatcode_corridor_weight": 0.75,
                },
            ),
            (
                "final_balanced",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1000),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0002,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0010,
                    "target_action_guide_orientation_switch_rad": 0.05,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00010,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.05,
                    "target_reward_cheatcode_axial_progress_weight": 0.25,
                    "target_reward_cheatcode_corridor_weight": 1.00,
                },
            ),
            (
                "final_strong_corridor",
                {
                    "target_action_guide_weight": 0.05,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1000),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.0002,
                    "target_action_guide_rotation_step_size": 0.005,
                    "target_action_guide_lateral_switch_m": 0.0008,
                    "target_action_guide_orientation_switch_rad": 0.04,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00010,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.03,
                    "target_reward_cheatcode_axial_progress_weight": 0.25,
                    "target_reward_cheatcode_corridor_weight": 1.50,
                },
            ),
            (
                "final_guarded",
                {
                    "target_action_guide_weight": 0.10,
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": min(steps, 1500),
                    "target_action_guide_collect_decay": True,
                    "target_action_guide_step_size": 0.00015,
                    "target_action_guide_rotation_step_size": 0.004,
                    "target_action_guide_lateral_switch_m": 0.0008,
                    "target_action_guide_orientation_switch_rad": 0.04,
                    "target_action_guide_rotate_while_lateral": True,
                    "target_action_guide_axial_step_size": 0.00005,
                    "tcp_translation_action_clip": 0.0003,
                    "tcp_rotation_action_clip": 0.005,
                    "adapter_lr": 3e-6,
                    "adapter_delta_clip": 0.002,
                    "actor_q_weight": 0.02,
                    "insertion_action_guard": True,
                    "target_reward_cheatcode_axial_progress_weight": 0.20,
                    "target_reward_cheatcode_corridor_weight": 1.00,
                },
            ),
        ]

    out: list[Experiment] = []
    for idx, (name, overrides) in enumerate(specs, start=1):
        exp_args = {
            **base,
            **overrides,
            "seed": 700 + {"stage_b": 0, "stage_c": 100, "stage_d": 200}.get(stage.key, 300) + idx,
            "target_reward_cheatcode_inside_alignment_weight": overrides.get(
                "target_reward_cheatcode_inside_alignment_weight",
                stage.inside_weight,
            ),
            "target_reward_cheatcode_retreat_weight": overrides.get(
                "target_reward_cheatcode_retreat_weight",
                stage.retreat_weight,
            ),
        }
        out.append(Experiment(stage.key, f"smart4_{idx}_{name}", steps, steps, exp_args, parent=str((parent or {}).get("run_dir") or "")))
    return out


def _run_staged_curriculum(
    args: argparse.Namespace,
    *,
    run_output: Path,
    log_dir: Path,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage_summaries: list[dict[str, Any]] = []
    stage_dirs: dict[str, Path] = {}
    for stage in _curriculum_stage_specs(args):
        summary = _materialize_curriculum_stage(args, stage)
        stage_summaries.append(summary)
        stage_dirs[stage.key] = Path(str(summary["episodes_dir"]))
        print(
            f"[pipeline] {stage.key} episodes {summary['episodes_dir']} "
            f"s={summary.get('s_start_min_m')}..{summary.get('s_start_max_m')} "
            f"r={summary.get('r_start_min_m')}..{summary.get('r_start_max_m')}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    promoted: list[dict[str, Any]] = []
    for stage in _curriculum_stage_specs(args):
        if time.monotonic() >= deadline:
            break
        episodes_dir = stage_dirs[stage.key]
        stage_args = _stage_guided_args(stage, episodes_dir)

        print(f"[pipeline] {stage.key} zero-action stability", flush=True)
        zero_rows = _run_wave(
            args,
            _zero_action_experiments_for_stage(args, stage, episodes_dir),
            output_dir=run_output,
            log_dir=log_dir,
            stage_deadline=min(deadline, time.monotonic() + 30 * 60),
        )
        rows.extend(zero_rows)
        _write_csv(args.output_root / f"metrics_after_{stage.key}_zero.csv", rows)
        stable = sorted(
            zero_rows,
            key=lambda r: (
                float(r.get("force_clipped_fraction") or 1.0),
                float(r.get("final_r_m") or 1.0),
            ),
        )
        if stable:
            best_zero = stable[0]
            force_clip = _float(best_zero.get("force_clipped_fraction")) or 0.0
            final_r = _float(best_zero.get("final_r_m")) or 0.0
            if force_clip > 0.75 or final_r > stage.max_lateral_explosion_m:
                print(
                    f"[pipeline] {stage.key} reset is unstable; continuing with least-bad config "
                    f"force_clip={force_clip} final_r={final_r}",
                    flush=True,
                )

        print(f"[pipeline] {stage.key} guide sweep", flush=True)
        wave1 = _guide_candidates(
            args.stage2_wave1_steps,
            stage_name=stage.key,
            episode_config_dir=episodes_dir,
            stage_args=stage_args,
        )
        wave1_rows = _run_wave(
            args,
            wave1,
            output_dir=run_output,
            log_dir=log_dir,
            stage_deadline=min(deadline, time.monotonic() + 60 * 60),
        )
        rows.extend(wave1_rows)
        _write_csv(args.output_root / f"metrics_after_{stage.key}_wave1.csv", rows)
        ranked = sorted(wave1_rows, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)

        wave2: list[Experiment] = []
        for idx, parent in enumerate(ranked[:4], start=1):
            p_args = {**stage_args, **_args_from_parent(parent)}
            p_args["episode_config_dir"] = episodes_dir
            p_args["max_lateral_explosion_m"] = stage.max_lateral_explosion_m
            wave2.append(
                Experiment(
                    stage.key,
                    f"guide_w2_from_{idx}",
                    args.stage2_wave2_steps,
                    args.stage2_wave2_steps,
                    {
                        **p_args,
                        "seed": 200 + idx,
                        "reward_preset": stage.reward_preset,
                        "act_only_actor_mode": "act_direct",
                        "target_action_guide_collect_blend": 1.0,
                        "target_action_guide_collect_steps": args.stage2_wave2_steps,
                        "diagnostics_every": 10,
                    },
                    parent=str(parent.get("run_dir")),
                )
            )
        wave2_rows = _run_wave(
            args,
            wave2,
            output_dir=run_output,
            log_dir=log_dir,
            stage_deadline=min(deadline, time.monotonic() + 60 * 60),
        )
        rows.extend(wave2_rows)
        _write_csv(args.output_root / f"metrics_after_{stage.key}_wave2.csv", rows)
        ranked = sorted(wave2_rows or wave1_rows, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)
        passed_ranked = [r for r in ranked if _passed_curriculum_stage(r, stage)]
        if not passed_ranked:
            print(f"[pipeline] {stage.key} did not pass promotion criteria; stopping staged curriculum", flush=True)
            break
        promoted = passed_ranked[:2]

        if stage.key in {"stage_a", "stage_b"} and time.monotonic() < deadline:
            stage3 = _alignment_imitation_candidates(promoted, args.stage3_steps)
            for exp in stage3:
                exp.stage = stage.key
                exp.args.update({"episode_config_dir": episodes_dir, **_stage_reward_args(stage)})
                exp.args["max_lateral_explosion_m"] = stage.max_lateral_explosion_m
            stage3_rows = _run_wave(
                args,
                stage3,
                output_dir=run_output,
                log_dir=log_dir,
                stage_deadline=min(deadline, time.monotonic() + 75 * 60),
            )
            rows.extend(stage3_rows)
            _write_csv(args.output_root / f"metrics_after_{stage.key}_imitation.csv", rows)
            stage3_ranked = sorted(stage3_rows or promoted, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)
            stage3_passed = [r for r in stage3_ranked if _passed_curriculum_stage(r, stage)]
            if stage3_passed:
                promoted = stage3_passed[:2]
        elif stage.key in {"stage_c", "stage_d"} and time.monotonic() < deadline:
            stage5 = _weak_insertion_candidates(promoted, args.stage5_steps)
            for exp in stage5:
                exp.stage = stage.key
                exp.args.update({"episode_config_dir": episodes_dir, **_stage_reward_args(stage)})
                exp.args["max_lateral_explosion_m"] = stage.max_lateral_explosion_m
            stage5_rows = _run_wave(
                args,
                stage5,
                output_dir=run_output,
                log_dir=log_dir,
                stage_deadline=min(deadline, time.monotonic() + 90 * 60),
            )
            rows.extend(stage5_rows)
            _write_csv(args.output_root / f"metrics_after_{stage.key}_insertion.csv", rows)

    return rows, stage_summaries


def _run_smart4_curriculum(
    args: argparse.Namespace,
    *,
    run_output: Path,
    log_dir: Path,
    deadline: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stages = _curriculum_stage_specs(args)
    start_idx = next((i for i, s in enumerate(stages) if s.key == args.start_stage), 0)
    stop_idx = next((i for i, s in enumerate(stages) if s.key == args.stop_after_stage), len(stages) - 1)
    selected_stages = stages[start_idx : stop_idx + 1]

    stage_summaries: list[dict[str, Any]] = []
    stage_dirs: dict[str, Path] = {}
    for stage in selected_stages:
        summary = _materialize_curriculum_stage(args, stage)
        stage_summaries.append(summary)
        stage_dirs[stage.key] = Path(str(summary["episodes_dir"]))
        print(
            f"[pipeline] {stage.key} episodes {summary['episodes_dir']} "
            f"s={summary.get('s_start_min_m')}..{summary.get('s_start_max_m')} "
            f"r={summary.get('r_start_min_m')}..{summary.get('r_start_max_m')}",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    parent: dict[str, Any] | None = None
    initial_checkpoint = args.initial_checkpoint.resolve() if args.initial_checkpoint else None
    for stage in selected_stages:
        if time.monotonic() >= deadline:
            break
        episodes_dir = stage_dirs[stage.key]
        print(f"[pipeline] {stage.key} smart4 hyperparameter candidates", flush=True)
        candidates = _smart4_candidates_for_stage(
            stage,
            episodes_dir,
            parent=parent,
            initial_checkpoint=initial_checkpoint,
            steps=int(args.smart4_steps),
        )
        stage_rows = _run_wave(
            args,
            candidates,
            output_dir=run_output,
            log_dir=log_dir,
            stage_deadline=min(deadline, time.monotonic() + (float(args.per_run_max_wall_time_minutes) + 5.0) * 60.0),
        )
        rows.extend(stage_rows)
        _write_csv(args.output_root / f"metrics_after_{stage.key}_smart4.csv", rows)
        ranked = sorted(stage_rows, key=lambda r: float(r.get("insertion_score" if stage.key in {"stage_c", "stage_d"} else "alignment_score") or -1e9), reverse=True)
        passed = [r for r in ranked if _passed_curriculum_stage(r, stage)]
        parent = (passed or ranked or [None])[0]
        if parent:
            checkpoint = _checkpoint_for_row(parent)
            print(
                f"[pipeline] selected {stage.key}/{parent.get('name')} "
                f"score={parent.get('alignment_score')} r={parent.get('final_r_m')} "
                f"theta={parent.get('final_theta_rad')} checkpoint={checkpoint}",
                flush=True,
            )
            initial_checkpoint = checkpoint
        if not parent or not initial_checkpoint:
            print(f"[pipeline] no checkpoint available after {stage.key}; stopping smart4 curriculum", flush=True)
            break

    return rows, stage_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--act-torchscript", type=Path, default=DEFAULT_ACT)
    parser.add_argument("--episode-config-dir", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output-root", type=Path, default=REPO / f"outputs/one_day_insertion_pipeline/one_day_{_now()}")
    parser.add_argument("--max-wall-time-minutes", type=float, default=240.0)
    parser.add_argument("--per-run-max-wall-time-minutes", type=float, default=20.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--stage1-steps", type=int, default=40)
    parser.add_argument("--stage2-wave1-steps", type=int, default=160)
    parser.add_argument("--stage2-wave2-steps", type=int, default=400)
    parser.add_argument("--stage3-steps", type=int, default=700)
    parser.add_argument("--stage5-steps", type=int, default=1200)
    parser.add_argument("--container-name", default="isaac-lab-base")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--curriculum-mode", choices=["single", "staged"], default="single")
    parser.add_argument("--curriculum-strategy", choices=["successive_halving", "smart4"], default="successive_halving")
    parser.add_argument("--start-stage", choices=["stage_a", "stage_b", "stage_c", "stage_d"], default="stage_a")
    parser.add_argument("--stop-after-stage", choices=["stage_a", "stage_b", "stage_c", "stage_d"], default="stage_d")
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--smart4-steps", type=int, default=1500)
    parser.add_argument("--stage-a-axial-m", type=float, default=0.020)
    parser.add_argument("--stage-a-lateral-min-m", type=float, default=0.004)
    parser.add_argument("--stage-a-lateral-max-m", type=float, default=0.008)
    parser.add_argument("--stage-b-axial-m", type=float, default=0.012)
    parser.add_argument("--stage-b-lateral-min-m", type=float, default=0.002)
    parser.add_argument("--stage-b-lateral-max-m", type=float, default=0.006)
    parser.add_argument("--stage-c-axial-min-m", type=float, default=0.006)
    parser.add_argument("--stage-c-axial-max-m", type=float, default=0.008)
    parser.add_argument("--stage-c-lateral-min-m", type=float, default=0.000)
    parser.add_argument("--stage-c-lateral-max-m", type=float, default=0.003)
    parser.add_argument("--stage-d-axial-min-m", type=float, default=0.003)
    parser.add_argument("--stage-d-axial-max-m", type=float, default=0.006)
    parser.add_argument("--stage-d-lateral-min-m", type=float, default=0.000)
    parser.add_argument("--stage-d-lateral-max-m", type=float, default=0.0015)
    parser.add_argument("--episodes-per-stage", type=int, default=8)
    parser.add_argument("--reuse-existing-stage-configs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    return args


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    run_output = args.output_root / "runs"
    log_dir = args.output_root / "logs"
    report_path = (
        REPO / "docs/curriculum_insertion_pipeline_20260515.md"
        if args.curriculum_mode == "staged"
        else REPO / "docs/one_day_insertion_pipeline_20260515.md"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "pipeline_args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8")

    deadline = time.monotonic() + float(args.max_wall_time_minutes) * 60.0
    rows: list[dict[str, Any]] = []

    if not args.skip_preflight:
        print("[pipeline] stage0 preflight", flush=True)
        stage0 = args.output_root / "stage0"
        stage0.mkdir(exist_ok=True)
        tests_rc = _run_checked(
            [_repo_python(), "-m", "pytest", "aic_utils/aic_isaac/test/test_insertion_reward_geometry.py", "-q"],
            log=stage0 / "pytest_insertion_reward_geometry.log",
        )
        audit_rc = _run_checked(
            [
                _repo_python(), str(AUDIT_SCRIPT), "--mode", "cheatcode",
                "--run-name", "one_day_stage0_cheatcode_audit",
                "--output-root", str(args.output_root / "reward_audits"),
                "--target-depth-m", "0.010", "--sigma-r", "0.0015", "--bypass-penalty-scale", "6.0", "--grid", "121",
            ],
            log=stage0 / "reward_audit.log",
        )
        if tests_rc != 0 or audit_rc != 0:
            raise SystemExit(f"Stage0 failed: tests_rc={tests_rc} audit_rc={audit_rc}")

    if args.curriculum_mode == "staged":
        if args.dry_run:
            summaries = [_materialize_curriculum_stage(args, stage) for stage in _curriculum_stage_specs(args)]
            print(json.dumps(summaries, indent=2, default=str))
            return 0
        if args.curriculum_strategy == "smart4":
            rows, stage_summaries = _run_smart4_curriculum(args, run_output=run_output, log_dir=log_dir, deadline=deadline)
        else:
            rows, stage_summaries = _run_staged_curriculum(args, run_output=run_output, log_dir=log_dir, deadline=deadline)
        _write_csv(args.output_root / "metrics_all.csv", rows)
        _write_report(report_path, args, rows, stage_summaries=stage_summaries)
        print(f"[pipeline] wrote {report_path}", flush=True)
        return 0

    stage1 = [
        Experiment("stage1", "zero_act_direct", args.stage1_steps, args.stage1_steps, {"seed": 11, "act_only_actor_mode": "act_direct", "reward_preset": "cheatcode_alignment_v1", "tcp_translation_action_clip": 0.0, "tcp_rotation_action_clip": 0.0, "diagnostics_every": 1}),
        Experiment("stage1", "zero_tight_reset", args.stage1_steps, args.stage1_steps, {"seed": 12, "act_only_actor_mode": "act_direct", "near_gate_reset_position_tolerance": 0.0005, "tcp_translation_action_clip": 0.0, "tcp_rotation_action_clip": 0.0, "diagnostics_every": 1}),
        Experiment("stage1", "act_adapter_no_update", args.stage1_steps, args.stage1_steps, {"seed": 13, "act_only_actor_mode": "act_adapter", "tcp_translation_action_clip": 0.0005, "tcp_rotation_action_clip": 0.005, "diagnostics_every": 1}),
        Experiment("stage1", "zero_with_guard", args.stage1_steps, args.stage1_steps, {"seed": 14, "act_only_actor_mode": "act_direct", "insertion_action_guard": True, "tcp_translation_action_clip": 0.0, "tcp_rotation_action_clip": 0.0, "diagnostics_every": 1}),
    ]
    if args.dry_run:
        print(json.dumps([e.__dict__ for e in stage1 + _guide_candidates(args.stage2_wave1_steps)], indent=2, default=str))
        return 0

    rows.extend(_run_wave(args, stage1, output_dir=run_output, log_dir=log_dir, stage_deadline=min(deadline, time.monotonic() + 30 * 60)))
    _write_csv(args.output_root / "metrics_stage1.csv", rows)

    wave1 = _guide_candidates(args.stage2_wave1_steps)
    wave1_results = _run_wave(args, wave1, output_dir=run_output, log_dir=log_dir, stage_deadline=min(deadline, time.monotonic() + 60 * 60))
    rows.extend(wave1_results)
    _write_csv(args.output_root / "metrics_after_stage2_wave1.csv", rows)
    guide_ranked = sorted(wave1_results, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)
    wave2 = []
    for idx, parent in enumerate(guide_ranked[:4], start=1):
        p_args = _args_from_parent(parent)
        wave2.append(
            Experiment(
                "stage2",
                f"guide_w2_from_{idx}",
                args.stage2_wave2_steps,
                args.stage2_wave2_steps,
                {
                    **p_args,
                    "seed": 200 + idx,
                    "reward_preset": "cheatcode_alignment_v1",
                    "act_only_actor_mode": "act_direct",
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": args.stage2_wave2_steps,
                    "diagnostics_every": 10,
                },
                parent=str(parent.get("run_dir")),
            )
        )
    wave2_results = _run_wave(args, wave2, output_dir=run_output, log_dir=log_dir, stage_deadline=min(deadline, time.monotonic() + 60 * 60))
    rows.extend(wave2_results)
    _write_csv(args.output_root / "metrics_after_stage2_wave2.csv", rows)
    guide_ranked = sorted(wave2_results or wave1_results, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)

    if any(_passed_alignment(r) for r in guide_ranked):
        stage3 = _alignment_imitation_candidates(guide_ranked, args.stage3_steps)
        stage3_results = _run_wave(args, stage3, output_dir=run_output, log_dir=log_dir, stage_deadline=min(deadline, time.monotonic() + 90 * 60))
        rows.extend(stage3_results)
        _write_csv(args.output_root / "metrics_after_stage3.csv", rows)
        stage3_ranked = sorted(stage3_results, key=lambda r: float(r.get("alignment_score") or -1e9), reverse=True)
        if any(_passed_alignment(r) for r in stage3_ranked):
            stage5 = _weak_insertion_candidates(stage3_ranked, args.stage5_steps)
            rows.extend(_run_wave(args, stage5, output_dir=run_output, log_dir=log_dir, stage_deadline=deadline))
    else:
        print("[pipeline] guide-only did not pass strict alignment; stopping before learning stages", flush=True)

    _write_csv(args.output_root / "metrics_all.csv", rows)
    _write_report(report_path, args, rows)
    print(f"[pipeline] wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
