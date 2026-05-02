#!/usr/bin/env python3
"""Inspect canonical AIC hybrid-training obs/action metadata."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_aic.hybrid_schema import hybrid_schema_json, inspect_hybrid_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Action chunk horizon used by ACT/offline SERL consumers.",
    )
    parser.add_argument(
        "--simulator-source",
        choices=["gazebo", "isaac", "unknown"],
        default=None,
        help="Override simulator source when it cannot be inferred.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = inspect_hybrid_schema(
        args.dataset_root,
        action_horizon=args.action_horizon,
        simulator_source=args.simulator_source,
    )
    data = summary.as_dict()
    if args.json:
        print(hybrid_schema_json(summary))
        return 0

    print(f"dataset_root: {data['dataset_root']}")
    print(f"task_family: {data['task_family']}")
    print(f"simulator_source: {data['simulator_source']}")
    print(f"action_mode: {data['action_mode']}")
    print(f"action_dim: {data['action_dim']}")
    print(f"action_horizon: {data['action_horizon']}")
    print(f"obs_mode: {data['obs_mode']}")
    print(f"obs_dim: {data['obs_dim']}")
    print("camera_keys:")
    for key in data["camera_keys"]:
        print(f"  - {key}")
    print("lowdim_keys:")
    for key in data["lowdim_keys"]:
        print(f"  - {key}")
    print("validation:")
    for key, value in data["validation"].items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
