#!/usr/bin/env python3
"""Evaluate ACT supervised loss on a fixed held-out slice of a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_episode_indices(dataset_root: Path) -> np.ndarray:
    files = sorted((dataset_root / "data").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")
    df = pd.concat((pd.read_parquet(path, columns=["episode_index"]) for path in files), ignore_index=True)
    return df["episode_index"].to_numpy(dtype=np.int64)


def _tail_episode_holdout_indices(dataset_root: Path, val_fraction: float) -> list[int]:
    episodes_by_frame = _read_episode_indices(dataset_root)
    episodes = sorted(set(int(v) for v in episodes_by_frame))
    if val_fraction <= 0.0:
        return list(range(len(episodes_by_frame)))
    if val_fraction >= 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_count = min(val_count, len(episodes) - 1)
    val_episodes = set(episodes[-val_count:])
    return [idx for idx, episode in enumerate(episodes_by_frame) if int(episode) in val_episodes]


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    checkpoint = args.act_checkpoint
    dataset_root = args.dataset_root
    policy = ACTPolicy.from_pretrained(checkpoint, local_files_only=True)
    policy.to(device)

    ds_meta = LeRobotDatasetMetadata(f"local/{dataset_root.name}", root=dataset_root)
    dataset = LeRobotDataset(
        f"local/{dataset_root.name}",
        root=dataset_root,
        delta_timestamps=resolve_delta_timestamps(policy.config, ds_meta),
        video_backend=args.video_backend,
    )
    val_indices = _tail_episode_holdout_indices(dataset_root, args.val_fraction)
    if not val_indices:
        raise ValueError("Validation split produced no frames")
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=checkpoint)

    # ACT's VAE training loss only materializes mu/log_sigma in train mode. We
    # also compute inference-mode chunk L1 because rollout uses the latent-zero
    # path, not teacher-forced VAE reconstruction.
    rows: list[dict[str, float]] = []
    frames = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            batch = _to_device(preprocessor(batch), device)
            policy.eval()
            actions_hat = policy.predict_action_chunk(batch)
            action_target = batch["action"]
            action_is_pad = batch["action_is_pad"].unsqueeze(-1)
            inference_l1 = (
                torch.nn.functional.l1_loss(action_target, actions_hat, reduction="none") * ~action_is_pad
            ).mean()
            n_action_steps = int(getattr(policy.config, "n_action_steps", actions_hat.shape[1]))
            inference_l1_n_action_steps = (
                torch.nn.functional.l1_loss(
                    action_target[:, :n_action_steps],
                    actions_hat[:, :n_action_steps],
                    reduction="none",
                )
                * ~action_is_pad[:, :n_action_steps]
            ).mean()
            policy.train()
            loss, loss_dict = policy(batch)
            batch_size = int(batch["action"].shape[0])
            frames += batch_size
            rows.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "inference_l1": float(inference_l1.detach().cpu()),
                    "inference_l1_n_action_steps": float(inference_l1_n_action_steps.detach().cpu()),
                    **{key: float(value) for key, value in loss_dict.items()},
                    "batch_size": float(batch_size),
                }
            )
            if args.max_batches > 0 and batch_index >= args.max_batches:
                break

    weights = np.asarray([row["batch_size"] for row in rows], dtype=np.float64)
    summary = {
        "dataset_root": str(dataset_root),
        "act_checkpoint": str(checkpoint),
        "device": str(device),
        "val_fraction": args.val_fraction,
        "val_frame_count_total": len(val_indices),
        "evaluated_frames": frames,
        "evaluated_batches": len(rows),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "state_shape": list(policy.config.input_features["observation.state"].shape),
        "action_shape": list(policy.config.output_features["action"].shape),
    }
    for key in ("loss", "l1_loss", "kld_loss", "inference_l1", "inference_l1_n_action_steps"):
        values = np.asarray([row[key] for row in rows if key in row], dtype=np.float64)
        if len(values):
            summary[key] = float(np.average(values, weights=weights[: len(values)]))

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
