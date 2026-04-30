#!/usr/bin/env python3
"""Run an offline SERL checkpoint through the Gazebo RL bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.score_parser import score_from_scoring_yaml
from gazebo_rl.serl_policy import OfflineSERLGazeboPolicy
from gazebo_rl.train import add_recording_args


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def classify_rollout(score: dict, *, success_threshold: float) -> str:
    total = score.get("total")
    if total is None:
        return "no_score"
    if float(total) >= success_threshold:
        return "success"
    return "transfer_failure"


def run_validation(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output_dir = Path(args.output_dir).resolve()
    results_dir = output_dir / "results"
    policy = OfflineSERLGazeboPolicy(args.checkpoint, device=args.device)
    env = GazeboRLEnv(
        workspace_dir=args.workspace_dir,
        engine_config=args.engine_config,
        sim_distrobox=args.sim_distrobox,
        ground_truth=args.ground_truth,
        gazebo_gui=args.gazebo_gui,
        launch_rviz=args.launch_rviz,
        max_steps=args.max_steps,
        per_trial_timeout_sec=args.per_trial_timeout_sec,
        results_dir=results_dir,
        record_lerobot=args.record_lerobot,
        record_root=args.record_root,
        record_repo_id=args.record_repo_id,
        record_single_task=args.record_single_task,
        record_video=args.record_video,
        record_fps=args.record_fps,
        record_resume=args.record_resume,
        record_drain_sec=args.record_drain_sec,
        record_image_writer_processes=args.record_image_writer_processes,
        record_image_writer_threads_per_camera=args.record_image_writer_threads_per_camera,
        record_video_encoding_batch_size=args.record_video_encoding_batch_size,
    )
    real_steps = 0
    total_reward = 0.0
    terminal_info = {}
    try:
        obs, reset_info = env.reset()
        terminal_info["reset"] = reset_info
        for _ in range(args.max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            real_steps += 1
            terminal_info = info
            if terminated or truncated:
                break
    finally:
        env.close()

    score = score_from_scoring_yaml(results_dir)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "elapsed_sec": time.monotonic() - started,
        "real_steps": real_steps,
        "total_reward": total_reward,
        "results_dir": str(results_dir),
        "score": score,
        "classification": classify_rollout(score, success_threshold=args.success_threshold),
        "terminal_info": terminal_info,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "transfer_validation_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument("--sim-distrobox", default=None)
    parser.add_argument("--ground-truth", type=_bool, default=True)
    parser.add_argument("--gazebo-gui", type=_bool, default=False)
    parser.add_argument("--launch-rviz", type=_bool, default=False)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=900.0)
    parser.add_argument("--success-threshold", type=float, default=90.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/gazebo_rl/serl_transfer_validation/latest")
    add_recording_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    print(json.dumps(run_validation(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
