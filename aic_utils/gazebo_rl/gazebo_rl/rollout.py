from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from gazebo_rl.gym_env import GazeboRLEnv
from gazebo_rl.score_parser import score_from_scoring_yaml
from gazebo_rl.train import TinyPolicy, add_recording_args


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def run_rollout(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    results_dir = output_dir / "results"
    record_root = args.record_root
    if args.record_lerobot and record_root is None:
        record_root = str(output_dir / "lerobot_dataset")

    policy = TinyPolicy(seed=args.seed)
    policy.load(checkpoint_path)

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
        record_root=record_root,
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
    total_reward = 0.0
    real_steps = 0
    terminal_info = {}
    try:
        obs, reset_info = env.reset()
        for _ in range(args.max_steps):
            action = policy.act(obs, explore=args.explore)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            real_steps += 1
            terminal_info = info
            if terminated or truncated:
                break
    finally:
        env.close()

    summary_path = output_dir / "rollout_summary.json"
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "elapsed_sec": time.monotonic() - started,
        "real_steps": real_steps,
        "total_reward": total_reward,
        "results_dir": str(results_dir),
        "record_lerobot": bool(args.record_lerobot),
        "record_root": str(record_root) if record_root is not None else None,
        "terminal_info": terminal_info,
        "score": score_from_scoring_yaml(results_dir),
        "summary_path": str(summary_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Roll out a saved Gazebo RL policy checkpoint.")
    parser.add_argument("--checkpoint", default="outputs/gazebo_rl/checkpoints/gazebo_rl_short.pt")
    parser.add_argument("--workspace-dir", default=".")
    parser.add_argument("--engine-config", default=None)
    parser.add_argument(
        "--sim-distrobox",
        default=None,
        help=(
            "Optional user-created distrobox container name for the evaluation "
            "environment. Omit to use the local pixi launch path."
        ),
    )
    parser.add_argument("--ground-truth", type=_bool, default=True)
    parser.add_argument("--gazebo-gui", type=_bool, default=False)
    parser.add_argument("--launch-rviz", type=_bool, default=False)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--per-trial-timeout-sec", type=float, default=300.0)
    parser.add_argument("--output-dir", default="outputs/gazebo_rl/rollouts/latest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--explore", action="store_true", help="Add exploration noise during rollout.")
    add_recording_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    summary = run_rollout(args)
    print(json.dumps(summary, indent=2))
