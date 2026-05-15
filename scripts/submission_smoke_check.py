#!/usr/bin/env python3
"""Load the submission policy artifacts and run one deterministic dummy inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy


def _dummy_observation() -> dict:
    image = np.zeros((256, 288, 3), dtype=np.uint8)
    return {
        "timestamp": 0.0,
        "controller": {
            "current_tcp_pose": {
                "position": [0.24, 0.24, 0.15],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "tcp_velocity": {"linear": [0.0, 0.0, 0.0], "angular": [0.0, 0.0, 0.0]},
            "tcp_error": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "joints": {"position": [0.0, -1.57, 1.57, -1.57, -1.57, 0.0, 0.0035405]},
        "wrist_wrench": {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0]},
        "images": {
            "observation.images.center_camera": image,
            "observation.images.left_camera": image,
            "observation.images.right_camera": image,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--act-torchscript", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-action-steps", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    act_torchscript = Path(args.act_torchscript)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not act_torchscript.exists():
        raise FileNotFoundError(act_torchscript)
    if not act_torchscript.with_suffix(".json").exists():
        raise FileNotFoundError(act_torchscript.with_suffix(".json"))

    policy = ACTAdapterSERLGazeboPolicy(
        checkpoint,
        act_torchscript=act_torchscript,
        device=args.device,
        allow_zero_images=False,
        adapter_delta_clip=None,
        action_clip=None,
    )
    policy.feature_assembler.reset([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    chunk = policy.act_chunk(_dummy_observation(), n_action_steps=args.n_action_steps)
    if chunk.shape != (args.n_action_steps, 6):
        raise RuntimeError(f"unexpected action chunk shape {chunk.shape}")
    if not np.isfinite(chunk).all():
        raise RuntimeError("policy produced non-finite action values")
    summary = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "act_torchscript": str(act_torchscript),
        "state_dim": policy.state_dim,
        "action_horizon": policy.action_horizon,
        "n_action_steps": args.n_action_steps,
        "action_shape": list(chunk.shape),
        "action_abs_max": float(np.abs(chunk).max()),
        "action_norm_mean": float(np.linalg.norm(chunk, axis=1).mean()),
        "last_action_components": policy.last_action_components,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
