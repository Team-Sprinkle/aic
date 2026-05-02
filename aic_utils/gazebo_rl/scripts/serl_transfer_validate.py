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
from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy, OfflineSERLGazeboPolicy, task_vector_from_context
from gazebo_rl.train import add_recording_args


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def classify_rollout(score: dict, *, success_threshold: float) -> str:
    total = score.get("total_score", score.get("total"))
    if total is None:
        return "no_score"
    if float(total) >= success_threshold:
        return "success"
    return "transfer_failure"


def run_validation(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    output_dir = Path(args.output_dir).resolve()
    results_dir = output_dir / "results"
    task_vector = task_vector_from_context(
        task_family=args.task_family,
        target_port_index=args.target_port_index,
        target_card_index=args.target_card_index,
        target_card_valid=args.target_card_valid,
        task_context_json=args.task_context_json,
    )
    if args.policy_kind == "lowdim_serl":
        policy = OfflineSERLGazeboPolicy(args.checkpoint, device=args.device, task_vector=task_vector)
    elif args.policy_kind == "act_adapter_serl":
        if args.act_torchscript is None:
            raise ValueError("--act-torchscript is required for --policy-kind act_adapter_serl")
        policy = ACTAdapterSERLGazeboPolicy(
            args.checkpoint,
            act_torchscript=args.act_torchscript,
            device=args.device,
            allow_zero_images=args.allow_zero_images,
            adapter_delta_clip=args.adapter_delta_clip,
            action_clip=args.action_clip,
            task_vector=task_vector,
        )
    else:
        raise ValueError(f"Unsupported policy kind: {args.policy_kind}")
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
        include_images=args.include_images,
    )
    real_steps = 0
    total_reward = 0.0
    terminal_info = {}
    action_norms: list[float] = []
    delta_norms: list[float] = []
    raw_delta_norms: list[float] = []
    try:
        obs, reset_info = env.reset()
        terminal_info["reset"] = reset_info
        for _ in range(args.max_steps):
            action = policy.act(obs)
            action_norms.append(sum(float(x) * float(x) for x in action) ** 0.5)
            components = getattr(policy, "last_action_components", {})
            if "delta_action_norm" in components:
                delta_norms.append(float(components["delta_action_norm"]))
            if "raw_delta_action_norm" in components:
                raw_delta_norms.append(float(components["raw_delta_action_norm"]))
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
        "policy_kind": args.policy_kind,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "act_torchscript": str(Path(args.act_torchscript).resolve()) if args.act_torchscript else None,
        "allow_zero_images": bool(args.allow_zero_images),
        "include_images": bool(args.include_images),
        "task_vector": None if task_vector is None else task_vector.astype(float).tolist(),
        "elapsed_sec": time.monotonic() - started,
        "real_steps": real_steps,
        "total_reward": total_reward,
        "action_norm_mean": sum(action_norms) / len(action_norms) if action_norms else None,
        "action_norm_max": max(action_norms) if action_norms else None,
        "adapter_delta_norm_mean": sum(delta_norms) / len(delta_norms) if delta_norms else None,
        "raw_adapter_delta_norm_mean": sum(raw_delta_norms) / len(raw_delta_norms) if raw_delta_norms else None,
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
    parser.add_argument("--policy-kind", choices=["lowdim_serl", "act_adapter_serl"], default="lowdim_serl")
    parser.add_argument("--act-torchscript", default=None)
    parser.add_argument(
        "--include-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Send live RGB images through the Gazebo bridge IPC. Defaults to true "
            "for ACT-adapter SERL unless --allow-zero-images is used, otherwise false."
        ),
    )
    parser.add_argument(
        "--allow-zero-images",
        action="store_true",
        help=(
            "Allow ACT-adapter validation with zero RGB images when the Gazebo IPC observation is lowdim-only. "
            "Use only for interface validation, not real transfer scoring."
        ),
    )
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
    parser.add_argument("--adapter-delta-clip", type=float, default=0.05)
    parser.add_argument("--action-clip", type=float, default=0.05)
    parser.add_argument("--task-family", choices=["sfp_to_nic", "sc_to_sc"], default=None)
    parser.add_argument("--target-port-index", type=int, default=None)
    parser.add_argument("--target-card-index", type=int, default=None)
    parser.add_argument("--target-card-valid", type=int, default=None)
    parser.add_argument("--task-context-json", default=None)
    parser.add_argument("--output-dir", default="outputs/gazebo_rl/serl_transfer_validation/latest")
    add_recording_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.include_images is None:
        args.include_images = args.policy_kind == "act_adapter_serl" and not args.allow_zero_images
    print(json.dumps(run_validation(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
