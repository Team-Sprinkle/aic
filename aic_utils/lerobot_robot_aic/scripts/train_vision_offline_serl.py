#!/usr/bin/env python3
"""Vision offline SERL pretraining using a LeRobot ACT checkpoint as actor."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn
from torch.utils.data import DataLoader

from lerobot_robot_aic.act_warmstart import inspect_act_checkpoint
from lerobot_robot_aic.vision_offline_serl import (
    VisionOfflineSERLConfig,
    VisionOfflineSERLDataset,
    VisionOfflineSERLTrainer,
    load_act_actor,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="vision_offline_serl_smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=0.0)
    parser.add_argument("--adapter-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--act-preservation-weight", type=float, default=1e-2)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--actor-mode", choices=["act_direct", "act_adapter"], default="act_adapter")
    parser.add_argument("--freeze-act", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-num-layers", type=int, default=2)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-delta-clip", type=float, default=None)
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["dataset", "final_success", "zero"],
        default="dataset",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help="LeRobot visual observation keys. Defaults to ACT checkpoint camera keys.",
    )
    parser.add_argument("--dataset-video-backend", default="pyav")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _select_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Retry with --device cpu.")
    return requested


def _camera_keys(args: argparse.Namespace) -> list[str]:
    if args.camera_keys:
        return list(args.camera_keys)
    metadata = inspect_act_checkpoint(args.act_checkpoint)
    keys = metadata.get("camera_keys") or []
    if not keys:
        raise ValueError("No camera keys were found in ACT checkpoint metadata; pass --camera-keys explicitly.")
    return list(keys)


def _train_config(args: argparse.Namespace, dataset_summary: dict[str, Any], warmstart: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_root": str(args.dataset_root),
        "act_checkpoint": str(args.act_checkpoint),
        "output_dir": str(args.output_dir),
        "job_name": args.job_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "lr": args.lr,
        "adapter_lr": args.adapter_lr,
        "act_lr": args.act_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "bc_weight": args.bc_weight,
        "cql_weight": args.cql_weight,
        "adapter_penalty_weight": args.adapter_penalty_weight,
        "act_preservation_weight": args.act_preservation_weight,
        "smoothness_weight": args.smoothness_weight,
        "action_horizon": args.action_horizon,
        "actor_mode": args.actor_mode,
        "freeze_act": args.freeze_act,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "adapter_num_layers": args.adapter_num_layers,
        "adapter_scale": args.adapter_scale,
        "adapter_delta_clip": args.adapter_delta_clip,
        "action_clip": args.action_clip,
        "reward_mode": args.reward_mode,
        "camera_keys": dataset_summary["camera_keys"],
        "dataset_video_backend": args.dataset_video_backend,
        "save_every": args.save_every,
        "dataset_summary": dataset_summary,
        "act_warmstart": warmstart,
    }


def _model_summary(trainer: VisionOfflineSERLTrainer) -> dict[str, int]:
    adapter_params = int(sum(p.numel() for p in getattr(trainer.actor, "adapter", nn.Module()).parameters()))
    return {
        "actor_parameters": int(sum(p.numel() for p in trainer.actor.parameters())),
        "actor_trainable_parameters": int(sum(p.numel() for p in trainer.actor.parameters() if p.requires_grad)),
        "act_trainable_parameters": int(sum(p.numel() for p in trainer.actor.act_policy.parameters() if p.requires_grad)),
        "adapter_parameters": adapter_params,
        "adapter_trainable_parameters": int(
            sum(p.numel() for p in getattr(trainer.actor, "adapter", nn.Module()).parameters() if p.requires_grad)
        ),
        "critic1_parameters": int(sum(p.numel() for p in trainer.critic1.parameters())),
        "critic2_parameters": int(sum(p.numel() for p in trainer.critic2.parameters())),
    }


def main() -> int:
    args = parse_args()
    device = _select_device(args.device)
    camera_keys = _camera_keys(args)

    dataset = VisionOfflineSERLDataset(
        args.dataset_root,
        camera_keys=camera_keys,
        action_horizon=args.action_horizon,
        reward_mode=args.reward_mode,
        video_backend=args.dataset_video_backend,
    )
    dataset_summary = asdict(dataset.summary)
    actor, warmstart = load_act_actor(
        args.act_checkpoint,
        action_horizon=args.action_horizon,
        device=device,
        actor_mode=args.actor_mode,
        state_dim=dataset.state_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_num_layers=args.adapter_num_layers,
        adapter_scale=args.adapter_scale,
        freeze_act=args.freeze_act,
        adapter_delta_clip=args.adapter_delta_clip,
        action_clip=args.action_clip,
    )
    config = VisionOfflineSERLConfig(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        action_horizon=args.action_horizon,
        camera_keys=camera_keys,
        gamma=args.gamma,
        tau=args.tau,
        bc_weight=args.bc_weight,
        cql_weight=args.cql_weight,
        adapter_penalty_weight=args.adapter_penalty_weight,
        act_preservation_weight=args.act_preservation_weight,
        smoothness_weight=args.smoothness_weight,
        adapter_delta_clip=args.adapter_delta_clip,
        action_clip=args.action_clip,
        lr=args.lr,
        adapter_lr=args.adapter_lr,
        act_lr=args.act_lr,
        critic_lr=args.critic_lr,
        actor_mode=args.actor_mode,
        freeze_act=args.freeze_act,
    )
    trainer = VisionOfflineSERLTrainer(config=config, actor=actor, device=device)
    first_batch = next(iter(DataLoader(dataset, batch_size=min(args.batch_size, 2), shuffle=False, num_workers=0)))
    first_obs = trainer._obs_to_device(first_batch["obs"])
    with torch.no_grad():
        initial_components = trainer.actor.action_components(first_obs)
        warmstart["initial_delta_norm"] = float(initial_components["delta_action"].norm(dim=-1).mean().detach().cpu())
        warmstart["initial_final_minus_act_norm"] = float(
            (initial_components["final_action"] - initial_components["base_action"]).norm(dim=-1).mean().detach().cpu()
        )
    train_config = _train_config(args, dataset_summary, warmstart)
    summary = {
        "training_config": train_config,
        "model_summary": _model_summary(trainer),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "train_config.json", train_config)
    write_json(run_dir / "warmstart_report.json", warmstart)
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    step = 0
    while step < args.steps:
        for batch in loader:
            step += 1
            metrics = trainer.train_step(batch)
            metrics["step"] = step
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, sort_keys=True) + "\n")
            if args.save_every > 0 and step % args.save_every == 0:
                trainer.save_checkpoint(
                    run_dir / f"checkpoint_{step:06d}.pt",
                    train_config=train_config,
                    dataset_summary=dataset_summary,
                    warmstart_report=warmstart,
                    step=step,
                )
            if step >= args.steps:
                break

    latest = run_dir / "checkpoint_latest.pt"
    trainer.save_checkpoint(
        latest,
        train_config=train_config,
        dataset_summary=dataset_summary,
        warmstart_report=warmstart,
        step=step,
    )
    print(f"Wrote vision offline SERL checkpoint: {latest}")
    print(f"Wrote warm-start report: {run_dir / 'warmstart_report.json'}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
