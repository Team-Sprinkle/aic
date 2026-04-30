#!/usr/bin/env python3
"""Minimal offline SERL-style pretraining on AIC LeRobot expert data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from lerobot_robot_aic.offline_rl_dataset import (
    OfflineRLTransitionDataset,
    load_lerobot_transitions,
)
from lerobot_robot_aic.act_warmstart import inspect_act_checkpoint, load_act_action_head_bias
from lerobot_robot_aic.offline_serl import OfflineSERLConfig, OfflineSERLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="offline_serl_smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of future action steps to flatten into each actor target.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--reward-mode",
        choices=["dataset", "final_success", "zero"],
        default="dataset",
    )
    parser.add_argument("--obs-mode", choices=["lowdim"], default="lowdim")
    parser.add_argument("--act-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--act-warmstart-mode",
        choices=["metadata", "action_head_bias"],
        default="action_head_bias",
        help=(
            "How to consume --act-checkpoint. 'action_head_bias' transfers the "
            "single-step ACT output bias as a repeated chunk prior; 'metadata' "
            "only validates and records checkpoint metadata."
        ),
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_dict(args: argparse.Namespace, dataset: OfflineRLTransitionDataset) -> dict[str, Any]:
    return {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "job_name": args.job_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "lr": args.lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "bc_weight": args.bc_weight,
        "cql_weight": args.cql_weight,
        "action_horizon": args.action_horizon,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "reward_mode": args.reward_mode,
        "obs_mode": args.obs_mode,
        "act_checkpoint": str(args.act_checkpoint) if args.act_checkpoint else None,
        "save_every": args.save_every,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "num_transitions": len(dataset),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    act_warmstart: dict[str, Any] | None = None
    if args.act_checkpoint is not None:
        act_warmstart = inspect_act_checkpoint(args.act_checkpoint)
        act_warmstart["mode"] = args.act_warmstart_mode

    arrays, schema = load_lerobot_transitions(
        args.dataset_root,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        action_horizon=args.action_horizon,
    )
    dataset = OfflineRLTransitionDataset(arrays, normalize=True)
    cfg = config_dict(args, dataset)

    summary = {
        "dataset_schema": schema.as_dict(),
        "training_config": cfg,
        "normalization_stats": dataset.stats.as_dict(),
        "reward_mean": float(arrays.reward.mean()),
        "done_count": int(arrays.done.sum()),
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "train_config.json", summary)

    trainer = OfflineSERLTrainer(
        OfflineSERLConfig(
            obs_dim=dataset.obs_dim,
            action_dim=dataset.action_dim,
            gamma=args.gamma,
            tau=args.tau,
            bc_weight=args.bc_weight,
            cql_weight=args.cql_weight,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            action_horizon=args.action_horizon,
        ),
        device=args.device,
    )
    if args.act_checkpoint is not None and args.act_warmstart_mode == "action_head_bias":
        bias = load_act_action_head_bias(args.act_checkpoint)
        transfer_report = trainer.warm_start_actor_bias_from_action_head(bias)
        act_warmstart = {**(act_warmstart or {}), **transfer_report}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    metrics_path = run_dir / "metrics.jsonl"
    latest_path = run_dir / "checkpoint_latest.pt"
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
                    train_config=cfg,
                    schema_summary=schema.as_dict(),
                    normalization_stats=dataset.stats.as_dict(),
                    step=step,
                    warmstart_metadata=act_warmstart,
                )
            if step >= args.steps:
                break

    trainer.save_checkpoint(
        latest_path,
        train_config=cfg,
        schema_summary=schema.as_dict(),
        normalization_stats=dataset.stats.as_dict(),
        step=step,
        warmstart_metadata=act_warmstart,
    )
    print(f"Wrote offline SERL checkpoint: {latest_path}")
    print(f"Wrote metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
