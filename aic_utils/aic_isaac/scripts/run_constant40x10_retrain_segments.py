#!/usr/bin/env python3
"""Run segmented constant-distance 40x10 SERL retraining with periodic evals."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


def _replace_flag(argv: list[str], flag: str, values: list[str] | None) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    if values is not None:
        out.extend([flag, *values])
    return out


def _remove_bool(argv: list[str], flag: str) -> list[str]:
    return [x for x in argv if x != flag]


def _add_bool(argv: list[str], flag: str) -> list[str]:
    argv = _remove_bool(argv, flag)
    argv.append(flag)
    return argv


def _load_base_argv(config_path: Path) -> list[str]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    argv = list(cfg["argv"])
    if not argv:
        raise ValueError(f"{config_path} has empty argv")
    argv[0] = "aic/aic_utils/aic_isaac/aic_isaaclab/scripts/serl/train.py"
    return argv


def _common_argv(base: list[str], *, output_dir: str, episode_config_dir: str, seed: int, num_envs: int) -> list[str]:
    argv = list(base)
    argv = _replace_flag(argv, "--output_dir", [output_dir])
    argv = _replace_flag(argv, "--episode_config_dir", [episode_config_dir])
    argv = _replace_flag(argv, "--seed", [str(seed)])
    argv = _replace_flag(argv, "--num_envs", [str(num_envs)])
    argv = _replace_flag(argv, "--load_replay_path", None)
    argv = _replace_flag(argv, "--load_replay_max_transitions", None)
    argv = _replace_flag(argv, "--max_logged_image_steps", ["0"])
    argv = _replace_flag(argv, "--diagnostics_every", ["20"])
    argv = _replace_flag(argv, "--save_every_steps", ["0"])
    argv = _replace_flag(argv, "--save_latest_every_steps", ["0"])
    argv = _remove_bool(argv, "--save_step_images")
    argv = _remove_bool(argv, "--save_videos")
    argv = _add_bool(argv, "--no-save_step_images")
    argv = _add_bool(argv, "--no-save_videos")
    argv = _add_bool(argv, "--no-save_replay_at_end")
    return argv


def _train_argv(
    base: list[str],
    *,
    checkpoint: str,
    output_dir: str,
    episode_config_dir: str,
    run_prefix: str,
    segment: int,
    seed: int,
    num_envs: int,
    minutes: float,
    adapter_lr: float,
    actor_mode: str,
    tcp_translation_action_clip: float,
    tcp_rotation_action_clip: float,
    adapter_penalty_weight: float,
    act_preservation_weight: float,
    guide_weight: float,
    guide_preset: str,
    gradient_updates_per_step: int,
    batch_size: int,
    actor_update_start_steps: int,
    warmup_steps: int,
    save_latest_every_steps: int,
    resume_online_state: bool,
) -> list[str]:
    argv = _common_argv(base, output_dir=output_dir, episode_config_dir=episode_config_dir, seed=seed, num_envs=num_envs)
    argv = _replace_flag(argv, "--checkpoint", [checkpoint])
    argv = _replace_flag(argv, "--run_name", [f"2026-06-04_train_{run_prefix}_constant40x10_seg{segment:02d}"])
    argv = _replace_flag(argv, "--steps", ["1000000"])
    argv = _replace_flag(argv, "--updates", ["1000000"])
    argv = _replace_flag(argv, "--max_wall_time_minutes", [f"{minutes:g}"])
    argv = _replace_flag(argv, "--adapter_lr", [f"{adapter_lr:g}"])
    argv = _replace_flag(argv, "--act_only_actor_mode", [actor_mode])
    argv = _replace_flag(argv, "--tcp_translation_action_clip", [f"{tcp_translation_action_clip:g}"])
    argv = _replace_flag(argv, "--tcp_rotation_action_clip", [f"{tcp_rotation_action_clip:g}"])
    argv = _replace_flag(argv, "--adapter_penalty_weight", [f"{adapter_penalty_weight:g}"])
    argv = _replace_flag(argv, "--act_preservation_weight", [f"{act_preservation_weight:g}"])
    argv = _replace_flag(argv, "--gradient_updates_per_step", [str(gradient_updates_per_step)])
    argv = _replace_flag(argv, "--batch_size", [str(batch_size)])
    argv = _replace_flag(argv, "--actor_update_start_steps", [str(actor_update_start_steps)])
    argv = _replace_flag(argv, "--warmup_steps", [str(warmup_steps)])
    argv = _replace_flag(argv, "--actor_update_end_steps", ["0"])
    argv = _replace_flag(argv, "--save_latest_every_steps", [str(save_latest_every_steps)])
    argv = _replace_flag(argv, "--target_action_guide_collect_steps", ["1000000"])
    argv = _replace_flag(argv, "--target_action_guide_collect_blend", ["1.0"])
    argv = _replace_flag(argv, "--target_action_guide_weight", [f"{guide_weight:g}"])
    if guide_preset == "target-tip":
        argv = _replace_flag(argv, "--target_action_guide_mode", ["target_tip_stabilize"])
        argv = _replace_flag(argv, "--target_action_guide_lateral_direction_sign", ["-1"])
        argv = _replace_flag(argv, "--target_action_guide_step_size", ["0.00008"])
        argv = _replace_flag(argv, "--target_action_guide_axial_step_size", ["0.00018"])
        argv = _replace_flag(argv, "--target_action_guide_rotation_step_size", ["0.00025"])
        argv = _replace_flag(argv, "--target_action_guide_orientation_switch_rad", ["0.030"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_goal_depth_m", ["0.0458"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_axial_step_m", ["0.00018"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_lateral_step_m", ["0.00008"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_axial_lateral_gate_m", ["0.00075"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_body_name", ["sfp_tip_link"])
        argv = _replace_flag(argv, "--target_action_guide_target_tip_orientation_body_name", ["sfp_tip_link"])
        argv = _replace_flag(argv, "--target_action_guide_final_orientation_depth_m", ["0.030"])
        argv = _replace_flag(argv, "--target_action_guide_final_orientation_lateral_m", ["0.0007"])
        argv = _replace_flag(argv, "--target_action_guide_final_orientation_threshold_rad", ["0.030"])
        argv = _replace_flag(argv, "--target_action_guide_final_orientation_rotation_step_size", ["0.00025"])
        argv = _replace_flag(argv, "--target_action_guide_final_orientation_axial_step_size", ["0.0"])
        argv = _replace_flag(argv, "--target_action_guide_orientation_probe_lateral_penalty", ["50.0"])
        argv = _replace_flag(argv, "--target_action_guide_orientation_probe_lateral_margin_m", ["0.00005"])
        argv = _replace_flag(argv, "--tcp_translation_action_clip", ["0.0003"])
        argv = _replace_flag(argv, "--tcp_rotation_action_clip", ["0.005"])
        for flag in (
            "--target_action_guide_axis_only_orientation",
            "--target_action_guide_rotate_while_lateral",
            "--target_action_guide_final_axis_only_orientation",
            "--target_action_guide_orientation_probe_basis",
            "--target_action_guide_orientation_probe_strict_lateral_gate",
        ):
            argv = _add_bool(argv, flag)
    elif guide_preset != "base":
        raise ValueError(f"unsupported guide preset: {guide_preset}")
    if resume_online_state:
        argv = _add_bool(argv, "--resume_online_state")
    else:
        argv = _remove_bool(argv, "--resume_online_state")
    argv = _add_bool(argv, "--save_final_checkpoint")
    return argv


def _eval_argv(
    base: list[str],
    *,
    checkpoint: str,
    output_dir: str,
    episode_config_dir: str,
    run_prefix: str,
    segment: int,
    seed: int,
    num_envs: int,
    steps: int,
    actor_mode: str,
    tcp_translation_action_clip: float,
    tcp_rotation_action_clip: float,
) -> list[str]:
    argv = _common_argv(base, output_dir=output_dir, episode_config_dir=episode_config_dir, seed=seed, num_envs=num_envs)
    argv = _replace_flag(argv, "--checkpoint", [checkpoint])
    argv = _replace_flag(argv, "--run_name", [f"2026-06-04_eval_{run_prefix}_constant40x10_seg{segment:02d}_actoronly"])
    argv = _replace_flag(argv, "--steps", [str(steps)])
    argv = _replace_flag(argv, "--act_only_actor_mode", [actor_mode])
    argv = _replace_flag(argv, "--tcp_translation_action_clip", [f"{tcp_translation_action_clip:g}"])
    argv = _replace_flag(argv, "--tcp_rotation_action_clip", [f"{tcp_rotation_action_clip:g}"])
    argv = _replace_flag(argv, "--updates", ["1000000"])
    argv = _replace_flag(argv, "--update_every_steps", ["1000000"])
    argv = _replace_flag(argv, "--warmup_steps", ["1000000"])
    argv = _replace_flag(argv, "--actor_update_start_steps", ["1000000"])
    argv = _replace_flag(argv, "--actor_update_end_steps", ["0"])
    argv = _replace_flag(argv, "--max_wall_time_minutes", ["0"])
    argv = _replace_flag(argv, "--target_action_guide_collect_blend", ["0.0"])
    argv = _replace_flag(argv, "--target_action_guide_weight", ["0.0"])
    argv = _replace_flag(argv, "--target_action_guide_collect_steps", ["0"])
    argv = _remove_bool(argv, "--resume_online_state")
    argv = _add_bool(argv, "--no-save_final_checkpoint")
    return argv


def _run(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
    print("+ " + " ".join(shlex.quote(x) for x in cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return proc.wait()


def _latest_run(output_dir: Path, run_name: str) -> Path:
    candidates = sorted(output_dir.glob(f"*_{run_name}"))
    if not candidates:
        raise FileNotFoundError(f"no run found for {run_name} under {output_dir}")
    return candidates[-1]


def _summarize_eval(run_dir: Path) -> dict[str, Any]:
    metrics = run_dir / "metrics.jsonl"
    strict_by_env: dict[int, list[int]] = {}
    best_by_env: dict[int, dict[str, Any]] = {}
    rows = 0
    if not metrics.exists():
        return {"run_dir": str(run_dir), "error": "missing metrics.jsonl"}
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows += 1
        step = int(rec.get("step", rows))
        geom = rec.get("post_step_insertion_geometry") or {}
        strict = geom.get("strict_success_by_env") or []
        s_vals = geom.get("signed_depth_m_by_env") or []
        r_vals = geom.get("lateral_error_m_by_env") or []
        theta_vals = geom.get("orientation_error_rad_by_env") or []
        for env, value in enumerate(strict):
            if bool(value):
                strict_by_env.setdefault(env, []).append(step)
        for env in range(min(len(s_vals), len(r_vals), len(theta_vals))):
            score = (float(s_vals[env]), -float(r_vals[env]), -float(theta_vals[env]))
            old = best_by_env.get(env)
            if old is None or score > old["score"]:
                best_by_env[env] = {
                    "score": score,
                    "step": step,
                    "s_m": float(s_vals[env]),
                    "r_m": float(r_vals[env]),
                    "theta_rad": float(theta_vals[env]),
                }
    return {
        "run_dir": str(run_dir),
        "rows": rows,
        "strict_success_env_count": len(strict_by_env),
        "strict_success_by_env": {str(k): v for k, v in sorted(strict_by_env.items())},
        "best_by_env": {
            str(k): {kk: vv for kk, vv in v.items() if kk != "score"}
            for k, v in sorted(best_by_env.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--start-checkpoint", required=True)
    parser.add_argument("--episode-config-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segments", type=int, default=8)
    parser.add_argument("--segment-minutes", type=float, default=30.0)
    parser.add_argument("--train-num-envs", type=int, default=4)
    parser.add_argument("--eval-num-envs", type=int, default=8)
    parser.add_argument("--eval-steps", type=int, default=420)
    parser.add_argument("--seed", type=int, default=1222)
    parser.add_argument("--run-prefix", default="v1222")
    parser.add_argument("--train-adapter-lr", type=float, default=5e-8)
    parser.add_argument("--train-actor-mode", choices=["act_adapter", "act_direct"], default="act_adapter")
    parser.add_argument("--train-tcp-translation-action-clip", type=float, default=0.0)
    parser.add_argument("--train-tcp-rotation-action-clip", type=float, default=0.0)
    parser.add_argument("--train-adapter-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--train-act-preservation-weight", type=float, default=1e-2)
    parser.add_argument("--train-guide-weight", type=float, default=0.35)
    parser.add_argument("--train-guide-preset", choices=["base", "target-tip"], default="base")
    parser.add_argument("--train-gradient-updates-per-step", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-actor-update-start-steps", type=int, default=20)
    parser.add_argument("--train-warmup-steps", type=int, default=20)
    parser.add_argument("--train-save-latest-every-steps", type=int, default=100)
    parser.add_argument("--train-resume-online-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--isaaclab-root", type=Path, default=Path("/workspace/isaaclab"))
    args = parser.parse_args()

    base = _load_base_argv(args.base_config)
    output_dir = Path(args.output_dir)
    checkpoint = str(args.start_checkpoint)
    summaries: list[dict[str, Any]] = []
    previous_train_checkpoints: list[Path] = []
    for segment in range(1, int(args.segments) + 1):
        train_name = f"2026-06-04_train_{args.run_prefix}_constant40x10_seg{segment:02d}"
        train_cmd = [
            str(args.isaaclab_root / "isaaclab.sh"),
            "-p",
            *_train_argv(
                base,
                checkpoint=checkpoint,
                output_dir=str(output_dir),
                episode_config_dir=str(args.episode_config_dir),
                run_prefix=str(args.run_prefix),
                segment=segment,
                seed=int(args.seed) + segment,
                num_envs=int(args.train_num_envs),
                minutes=float(args.segment_minutes),
                adapter_lr=float(args.train_adapter_lr),
                actor_mode=str(args.train_actor_mode),
                tcp_translation_action_clip=float(args.train_tcp_translation_action_clip),
                tcp_rotation_action_clip=float(args.train_tcp_rotation_action_clip),
                adapter_penalty_weight=float(args.train_adapter_penalty_weight),
                act_preservation_weight=float(args.train_act_preservation_weight),
                guide_weight=float(args.train_guide_weight),
                guide_preset=str(args.train_guide_preset),
                gradient_updates_per_step=int(args.train_gradient_updates_per_step),
                batch_size=int(args.train_batch_size),
                actor_update_start_steps=int(args.train_actor_update_start_steps),
                warmup_steps=int(args.train_warmup_steps),
                save_latest_every_steps=int(args.train_save_latest_every_steps),
                resume_online_state=bool(args.train_resume_online_state),
            ),
        ]
        rc = _run(train_cmd, cwd=args.isaaclab_root, log_path=output_dir / "segment_logs" / f"{train_name}.log")
        if rc != 0:
            return rc
        train_run = _latest_run(output_dir, train_name)
        checkpoint = str(train_run / "checkpoint_latest.pt")
        for old_checkpoint in previous_train_checkpoints:
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                print(f"[v1222] removed superseded checkpoint {old_checkpoint}", flush=True)
        previous_train_checkpoints = [Path(checkpoint)]

        eval_name = f"2026-06-04_eval_{args.run_prefix}_constant40x10_seg{segment:02d}_actoronly"
        eval_cmd = [
            str(args.isaaclab_root / "isaaclab.sh"),
            "-p",
            *_eval_argv(
                base,
                checkpoint=checkpoint,
                output_dir=str(output_dir),
                episode_config_dir=str(args.episode_config_dir),
                run_prefix=str(args.run_prefix),
                segment=segment,
                seed=int(args.seed) + 1000 + segment,
                num_envs=int(args.eval_num_envs),
                steps=int(args.eval_steps),
                actor_mode=str(args.train_actor_mode),
                tcp_translation_action_clip=float(args.train_tcp_translation_action_clip),
                tcp_rotation_action_clip=float(args.train_tcp_rotation_action_clip),
            ),
        ]
        rc = _run(eval_cmd, cwd=args.isaaclab_root, log_path=output_dir / "segment_logs" / f"{eval_name}.log")
        if rc != 0:
            return rc
        eval_run = _latest_run(output_dir, eval_name)
        summary = _summarize_eval(eval_run)
        summaries.append({"segment": segment, "train_run": str(train_run), "checkpoint": checkpoint, "eval": summary})
        summary_path = output_dir / f"{args.run_prefix}_segment_eval_summary.json"
        summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
        print("[v1222] eval_summary " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
