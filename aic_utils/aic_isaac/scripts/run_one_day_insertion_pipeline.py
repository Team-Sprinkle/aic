#!/usr/bin/env python3
"""Autonomous independent-GPU near-gate insertion experiment pipeline."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
INNER_TRAIN = "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py"
AUDIT_SCRIPT = REPO / "aic_utils/aic_isaac/scripts/audit_insertion_reward_geometry.py"
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


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _rel_to_container(path: Path) -> str:
    path = path.resolve()
    try:
        return "aic/" + str(path.relative_to(REPO))
    except ValueError:
        return str(path)


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
    theta_vals: list[float] = []
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
    for row in rows:
        geom = row.get("pre_step_insertion_geometry") or {}
        phase = geom.get("cheatcode_phase_reward") or {}
        if not first and geom:
            first = geom
        if geom:
            last = geom
        for target, key in (
            (s_vals, "signed_depth_m_env0"),
            (r_vals, "lateral_error_m_env0"),
            (theta_vals, "orientation_error_rad_env0"),
            (consistency_axial_vals, "consistency_axial_error_m_env0"),
            (consistency_lateral_vals, "consistency_lateral_error_m_env0"),
        ):
            v = _float(geom.get(key))
            if v is not None:
                target.append(v)
        if _float(phase.get("success_candidate_env0")) and float(phase["success_candidate_env0"]) >= 0.5:
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

    initial_r = _float((first or {}).get("lateral_error_m_env0"))
    final_r = _float((last or {}).get("lateral_error_m_env0"))
    initial_theta = _float((first or {}).get("orientation_error_rad_env0"))
    final_theta = _float((last or {}).get("orientation_error_rad_env0"))
    final_s = _float((last or {}).get("signed_depth_m_env0"))
    best_r = stat(r_vals, "min")
    best_theta = stat(theta_vals, "min")
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
    if best_theta is not None:
        alignment_score -= best_theta * 2.0
    if initial_r is not None and final_r is not None:
        alignment_score += max(0.0, initial_r - final_r) * 800.0
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
        "initial_theta_rad": initial_theta,
        "final_theta_rad": final_theta,
        "best_theta_rad": best_theta,
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
        "--episode_config_dir", _rel_to_container(args.episode_config_dir),
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
    if exp.args.get("insertion_action_guard", False):
        cmd.append("--insertion_action_guard")
    else:
        cmd.append("--no-insertion_action_guard")
    for key, flag in [
        ("target_reward_cheatcode_axial_progress_weight", "--target_reward_cheatcode_axial_progress_weight"),
        ("target_reward_cheatcode_corridor_weight", "--target_reward_cheatcode_corridor_weight"),
        ("target_reward_cheatcode_inside_alignment_weight", "--target_reward_cheatcode_inside_alignment_weight"),
        ("target_reward_cheatcode_retreat_weight", "--target_reward_cheatcode_retreat_weight"),
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
                    if metrics.get("max_abs_q") is not None and float(metrics["max_abs_q"]) > 100.0:
                        print(f"[pipeline] early-stop Q explosion: {item.exp.name}", flush=True)
                        _terminate(item)
                    if metrics.get("final_r_m") is not None and float(metrics["final_r_m"]) > 0.12:
                        print(f"[pipeline] early-stop lateral explosion: {item.exp.name}", flush=True)
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


def _guide_candidates(steps: int) -> list[Experiment]:
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
                stage="stage2",
                name=f"guide_w1_{idx}",
                steps=steps,
                updates=steps,
                args={
                    "seed": 100 + idx,
                    "reward_preset": "cheatcode_alignment_v1",
                    "act_only_actor_mode": "act_direct",
                    "target_action_guide_collect_blend": 1.0,
                    "target_action_guide_collect_steps": steps,
                    "target_action_guide_step_size": step,
                    "target_action_guide_rotation_step_size": rot,
                    "target_action_guide_rotation_sign": rot_sign,
                    "target_action_guide_lateral_switch_m": lat_switch,
                    "target_action_guide_orientation_switch_rad": ori_switch,
                    "target_action_guide_rotate_while_lateral": rotate,
                    "target_action_guide_preinsert_hover_depth": hover_depth,
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


def _write_report(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
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
        "",
        "## Summary",
        "",
        f"- Runs completed: {len(rows)}",
        f"- Best alignment: `{(best_alignment or {}).get('run_dir')}`",
        f"- Best insertion: `{(best_insertion or {}).get('run_dir')}`",
        "",
        "## Metric Table",
        "",
        "| stage | name | score | final r mm | best r mm | final theta | max s mm | bad inward r frac | force clip frac | run |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in sorted(rows, key=lambda r: (str(r.get("stage")), -float(r.get("alignment_score") or -1e9))):
        def mm(key: str) -> str:
            v = _float(row.get(key))
            return "" if v is None else f"{v * 1000.0:.2f}"

        def num(key: str, nd: int = 3) -> str:
            v = _float(row.get(key))
            return "" if v is None else f"{v:.{nd}f}"

        lines.append(
            f"| {row.get('stage')} | {row.get('name')} | {num('alignment_score', 2)} | "
            f"{mm('final_r_m')} | {mm('best_r_m')} | {num('final_theta_rad')} | {mm('max_s_m')} | "
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpus = [x.strip() for x in str(args.gpus).split(",") if x.strip()]
    return args


def main() -> int:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    run_output = args.output_root / "runs"
    log_dir = args.output_root / "logs"
    report_path = REPO / "docs/one_day_insertion_pipeline_20260515.md"
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
