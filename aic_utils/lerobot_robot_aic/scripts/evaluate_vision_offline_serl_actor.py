#!/usr/bin/env python3
"""Evaluate a vision offline SERL actor checkpoint on a held-out LeRobot split."""

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
from torch.utils.data import DataLoader, Subset

from lerobot_robot_aic.vision_offline_serl import VisionOfflineSERLDataset, load_act_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _split_by_episode(dataset: VisionOfflineSERLDataset, val_fraction: float) -> list[int]:
    if val_fraction <= 0.0 or val_fraction >= 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    episodes = sorted(set(int(v) for v in dataset.episodes))
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_count = min(val_count, len(episodes) - 1)
    val_episodes = set(episodes[-val_count:])
    return [idx for idx, episode in enumerate(dataset.episodes) if int(episode) in val_episodes]


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row[key] for row in rows) / max(len(rows), 1))


@torch.no_grad()
def main() -> int:
    args = parse_args()
    payload: dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_config = payload.get("train_config") or {}
    dataset_summary = payload.get("dataset_summary") or {}
    serl_config = payload.get("vision_offline_serl_config") or {}
    val_fraction = float(args.val_fraction if args.val_fraction is not None else train_config.get("val_fraction", 0.05))
    action_horizon = int(dataset_summary.get("action_horizon") or train_config.get("action_horizon") or serl_config["action_horizon"])
    camera_keys = list(dataset_summary.get("camera_keys") or train_config.get("camera_keys") or serl_config["camera_keys"])
    dataset = VisionOfflineSERLDataset(
        args.dataset_root,
        camera_keys=camera_keys,
        action_horizon=action_horizon,
        reward_mode=train_config.get("reward_mode", "dataset"),
        video_backend=train_config.get("dataset_video_backend", "pyav"),
    )
    val_indices = _split_by_episode(dataset, val_fraction)
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    actor, _ = load_act_actor(
        Path(train_config["act_checkpoint"]),
        action_horizon=action_horizon,
        device=args.device,
        actor_mode=train_config.get("actor_mode", "act_adapter"),
        state_dim=int(dataset_summary.get("state_dim") or serl_config["state_dim"]),
        adapter_hidden_dim=int(train_config.get("adapter_hidden_dim", 256)),
        adapter_num_layers=int(train_config.get("adapter_num_layers", 2)),
        adapter_arch=str(train_config.get("adapter_arch", "mlp")),
        adapter_layer_norm=bool(train_config.get("adapter_layer_norm", False)),
        adapter_activation=str(train_config.get("adapter_activation", serl_config.get("adapter_activation", "relu"))),
        state_encoding=str(train_config.get("state_encoding", serl_config.get("state_encoding", "none"))),
        state_encoding_indices=tuple(
            int(i) for i in train_config.get("state_encoding_indices", serl_config.get("state_encoding_indices", []))
        ),
        state_encoding_num_bands=int(
            train_config.get("state_encoding_num_bands", serl_config.get("state_encoding_num_bands", 4))
        ),
        state_encoding_max_freq=float(
            train_config.get("state_encoding_max_freq", serl_config.get("state_encoding_max_freq", 8.0))
        ),
        state_encoding_scale=float(
            train_config.get("state_encoding_scale", serl_config.get("state_encoding_scale", 1.0))
        ),
        adapter_scale=float(train_config.get("adapter_scale", 1.0)),
        freeze_act=bool(train_config.get("freeze_act", True)),
        adapter_delta_clip=train_config.get("adapter_delta_clip"),
        action_clip=train_config.get("action_clip"),
    )
    missing, unexpected = actor.load_state_dict(payload["actor"], strict=False)
    actor.to(args.device)
    actor.eval()

    rows: list[dict[str, float]] = []
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        obs = {
            "state": batch["obs"]["state"].to(args.device),
            "images": {key: value.to(args.device) for key, value in batch["obs"]["images"].items()},
        }
        target = batch["action"].to(args.device)
        components = actor.action_components(obs)
        final_action = components["final_action"]
        base_action = components["base_action"]
        rows.append(
            {
                "frames": float(target.shape[0]),
                "final_l1": float((final_action - target).abs().mean().detach().cpu()),
                "base_l1": float((base_action - target).abs().mean().detach().cpu()),
                "delta_l1": float((final_action - base_action).abs().mean().detach().cpu()),
                "final_mse": float((final_action - target).square().mean().detach().cpu()),
                "base_mse": float((base_action - target).square().mean().detach().cpu()),
                "delta_norm": float((final_action - base_action).norm(dim=-1).mean().detach().cpu()),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": args.device,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "evaluated_batches": len(rows),
        "evaluated_frames": int(sum(row["frames"] for row in rows)),
        "val_fraction": val_fraction,
        "val_frame_count_total": len(val_indices),
        "final_l1": _mean(rows, "final_l1"),
        "base_l1": _mean(rows, "base_l1"),
        "delta_l1": _mean(rows, "delta_l1"),
        "final_mse": _mean(rows, "final_mse"),
        "base_mse": _mean(rows, "base_mse"),
        "delta_norm": _mean(rows, "delta_norm"),
        "missing_actor_keys": sorted(missing),
        "unexpected_actor_keys": sorted(unexpected),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
