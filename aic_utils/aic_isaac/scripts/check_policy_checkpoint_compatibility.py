#!/usr/bin/env python3
"""Check whether a warm-start checkpoint shape can plausibly initialize Isaac PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-obs-dim", type=int, default=None)
    parser.add_argument("--expected-action-dim", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("offline_serl_config", {})
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_type": "offline_serl" if "offline_serl_config" in ckpt else "unknown",
        "obs_dim": cfg.get("obs_dim"),
        "action_dim": cfg.get("action_dim"),
        "action_horizon": cfg.get("action_horizon"),
        "architecture": "OfflineSERL MLP actor-critic",
        "isaac_target": "RSL-RL PPO actor-critic",
        "direct_weight_transfer_supported": False,
        "reason": (
            "Offline SERL currently saves a standalone MLP actor-critic, while "
            "Stage 5 uses RSL-RL's PPO actor-critic. Shape-compatible dims are "
            "necessary but not sufficient for direct weight loading."
        ),
    }
    if args.expected_obs_dim is not None:
        report["obs_dim_matches_expected"] = report["obs_dim"] == args.expected_obs_dim
    if args.expected_action_dim is not None:
        report["action_dim_matches_expected"] = (
            report["action_dim"] == args.expected_action_dim
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        for key, value in report.items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
