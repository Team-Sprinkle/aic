#!/usr/bin/env python3
"""Compare ACT base and ACT-adapter SERL actions on offline LeRobot frames."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils" / "lerobot_robot_aic"))

import torch
from torch.utils.data import DataLoader, Subset

from lerobot_robot_aic.vision_offline_serl import VisionOfflineSERLDataset, load_act_actor


def _split_by_episode(dataset: VisionOfflineSERLDataset, fraction: float, split: str) -> list[int]:
    episodes = sorted(set(int(v) for v in dataset.episodes))
    if not episodes:
        return []
    if split == "all":
        selected = set(episodes)
    else:
        val_count = max(1, int(round(len(episodes) * fraction)))
        val_count = min(val_count, max(1, len(episodes) - 1))
        selected = set(episodes[-val_count:] if split == "val" else episodes[:-val_count])
    return [idx for idx, episode in enumerate(dataset.episodes) if int(episode) in selected]


def _stats(t: torch.Tensor) -> dict[str, Any]:
    value = t.detach().float().cpu()
    finite = torch.isfinite(value)
    out: dict[str, Any] = {
        "shape": list(value.shape),
        "finite": bool(finite.all()),
        "nan_count": int(torch.isnan(value).sum()),
        "inf_count": int(torch.isinf(value).sum()),
    }
    if finite.any():
        f = value[finite]
        out.update(
            {
                "mean": float(f.mean()),
                "std": float(f.std(unbiased=False)) if f.numel() > 1 else 0.0,
                "min": float(f.min()),
                "max": float(f.max()),
                "abs_max": float(f.abs().max()),
                "norm_mean": float(value.flatten(start_dim=1).norm(dim=1).mean()) if value.ndim > 1 else float(value.norm()),
                "norm_max": float(value.flatten(start_dim=1).norm(dim=1).max()) if value.ndim > 1 else float(value.norm()),
            }
        )
    return out


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return float(sum(row[key] for row in rows) / max(1, len(rows)))


def _load_actor(args: argparse.Namespace, payload: dict[str, Any], dataset_summary: dict[str, Any], serl_config: dict[str, Any]):
    train_config = payload.get("train_config") or {}
    action_horizon = int(dataset_summary.get("action_horizon") or train_config.get("action_horizon") or serl_config["action_horizon"])
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
        state_encoding_indices=tuple(int(i) for i in train_config.get("state_encoding_indices", serl_config.get("state_encoding_indices", []))),
        state_encoding_num_bands=int(train_config.get("state_encoding_num_bands", serl_config.get("state_encoding_num_bands", 4))),
        state_encoding_max_freq=float(train_config.get("state_encoding_max_freq", serl_config.get("state_encoding_max_freq", 8.0))),
        state_encoding_scale=float(train_config.get("state_encoding_scale", serl_config.get("state_encoding_scale", 1.0))),
        adapter_scale=float(train_config.get("adapter_scale", 1.0)),
        freeze_act=bool(train_config.get("freeze_act", True)),
        adapter_delta_clip=train_config.get("adapter_delta_clip"),
        action_clip=train_config.get("action_clip"),
    )
    missing, unexpected = actor.load_state_dict(payload["actor"], strict=False)
    actor.to(args.device).eval()
    return actor, action_horizon, sorted(missing), sorted(unexpected)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    parser.add_argument("--val-fraction", type=float, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_config = payload.get("train_config") or {}
    dataset_summary = payload.get("dataset_summary") or {}
    serl_config = payload.get("vision_offline_serl_config") or {}
    val_fraction = float(args.val_fraction if args.val_fraction is not None else train_config.get("val_fraction", 0.05))
    camera_keys = list(dataset_summary.get("camera_keys") or train_config.get("camera_keys") or serl_config["camera_keys"])
    actor, action_horizon, missing, unexpected = _load_actor(args, payload, dataset_summary, serl_config)
    dataset = VisionOfflineSERLDataset(
        args.dataset_root,
        camera_keys=camera_keys,
        action_horizon=action_horizon,
        reward_mode=train_config.get("reward_mode", "dataset"),
        video_backend=train_config.get("dataset_video_backend", "pyav"),
        swap_rgb_channels=train_config.get("swap_rgb_channels"),
    )
    indices = _split_by_episode(dataset, val_fraction, args.split)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    rows: list[dict[str, float]] = []
    delta_sum = None
    delta_sq_sum = None
    delta_abs_max = None
    clipped_count = 0
    total_values = 0
    nonfinite = {"base": 0, "raw_delta": 0, "delta": 0, "final": 0, "target": 0}
    sample_rows: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(loader):
        if args.max_batches > 0 and batch_idx >= args.max_batches:
            break
        obs = {
            "state": batch["obs"]["state"].to(args.device),
            "images": {key: value.to(args.device) for key, value in batch["obs"]["images"].items()},
        }
        target = batch["action"].to(args.device)
        components = actor.action_components(obs)
        base = components["base_action"]
        raw_delta = components["raw_delta_action"]
        delta = components["delta_action"]
        final = components["final_action"]
        final_minus_act = final - base
        for key, value in (("base", base), ("raw_delta", raw_delta), ("delta", delta), ("final", final), ("target", target)):
            nonfinite[key] += int((~torch.isfinite(value)).sum().detach().cpu())
        clipped_count += int((raw_delta != delta).sum().detach().cpu())
        total_values += int(delta.numel())
        batch_delta = final_minus_act.detach().float().cpu()
        flat = batch_delta.reshape(-1, batch_delta.shape[-1])
        delta_sum = flat.sum(dim=0) if delta_sum is None else delta_sum + flat.sum(dim=0)
        delta_sq_sum = (flat * flat).sum(dim=0) if delta_sq_sum is None else delta_sq_sum + (flat * flat).sum(dim=0)
        cur_abs = flat.abs().max(dim=0).values
        delta_abs_max = cur_abs if delta_abs_max is None else torch.maximum(delta_abs_max, cur_abs)
        rows.append(
            {
                "batch": float(batch_idx),
                "frames": float(target.shape[0]),
                "act_bc_mse": float((base - target).square().mean().detach().cpu()),
                "serl_bc_mse": float((final - target).square().mean().detach().cpu()),
                "act_bc_l1": float((base - target).abs().mean().detach().cpu()),
                "serl_bc_l1": float((final - target).abs().mean().detach().cpu()),
                "final_minus_act_norm": float(final_minus_act.norm(dim=-1).mean().detach().cpu()),
                "raw_adapter_delta_norm": float(raw_delta.norm(dim=-1).mean().detach().cpu()),
                "adapter_delta_norm": float(delta.norm(dim=-1).mean().detach().cpu()),
                "base_action_norm": float(base.norm(dim=-1).mean().detach().cpu()),
                "final_action_norm": float(final.norm(dim=-1).mean().detach().cpu()),
            }
        )
        if len(sample_rows) < 32:
            sample_count = min(target.shape[0], 32 - len(sample_rows))
            for i in range(sample_count):
                sample_rows.append(
                    {
                        "batch": batch_idx,
                        "row": i,
                        "act_first6": [float(v) for v in base[i, :6].detach().cpu().tolist()],
                        "serl_first6": [float(v) for v in final[i, :6].detach().cpu().tolist()],
                        "expert_first6": [float(v) for v in target[i, :6].detach().cpu().tolist()],
                        "delta_first6": [float(v) for v in final_minus_act[i, :6].detach().cpu().tolist()],
                    }
                )

    count = int(sum(row["frames"] for row in rows))
    if delta_sum is None or delta_sq_sum is None or delta_abs_max is None:
        raise RuntimeError("No diagnostic batches were evaluated")
    per_dim_mean = delta_sum / max(count, 1)
    per_dim_var = (delta_sq_sum / max(count, 1)) - per_dim_mean.square()
    per_dim_std = per_dim_var.clamp(min=0.0).sqrt()
    per_dim_path = args.output_dir / "per_dim_delta.csv"
    with per_dim_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "mean", "std", "abs_max"])
        writer.writeheader()
        for idx in range(per_dim_mean.numel()):
            writer.writerow(
                {
                    "dim": idx,
                    "mean": float(per_dim_mean[idx]),
                    "std": float(per_dim_std[idx]),
                    "abs_max": float(delta_abs_max[idx]),
                }
            )
    samples_path = args.output_dir / "sample_actions.json"
    samples_path.write_text(json.dumps(sample_rows, indent=2), encoding="utf-8")
    batch_path = args.output_dir / "batch_metrics.csv"
    with batch_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "device": args.device,
        "split": args.split,
        "val_fraction": val_fraction,
        "dataset_len": len(dataset),
        "candidate_frame_count": len(indices),
        "evaluated_batches": len(rows),
        "evaluated_frames": count,
        "action_horizon": action_horizon,
        "camera_keys": camera_keys,
        "missing_actor_keys": missing,
        "unexpected_actor_keys": unexpected,
        "act_bc_mse": _mean(rows, "act_bc_mse"),
        "serl_bc_mse": _mean(rows, "serl_bc_mse"),
        "bc_mse_change_serl_minus_act": _mean(rows, "serl_bc_mse") - _mean(rows, "act_bc_mse"),
        "act_bc_l1": _mean(rows, "act_bc_l1"),
        "serl_bc_l1": _mean(rows, "serl_bc_l1"),
        "bc_l1_change_serl_minus_act": _mean(rows, "serl_bc_l1") - _mean(rows, "act_bc_l1"),
        "final_minus_act_norm": _mean(rows, "final_minus_act_norm"),
        "raw_adapter_delta_norm": _mean(rows, "raw_adapter_delta_norm"),
        "adapter_delta_norm": _mean(rows, "adapter_delta_norm"),
        "base_action_norm": _mean(rows, "base_action_norm"),
        "final_action_norm": _mean(rows, "final_action_norm"),
        "clipped_fraction": float(clipped_count / max(total_values, 1)),
        "nonfinite_counts": nonfinite,
        "per_dim_delta_csv": str(per_dim_path),
        "batch_metrics_csv": str(batch_path),
        "sample_actions_json": str(samples_path),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
