#!/usr/bin/env python3
"""Inspect an AIC LeRobot dataset schema and infer its action mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_robot_aic.dataset_schema import summarize_dataset_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_dataset_schema(args.dataset_root)
    data = summary.as_dict()

    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
        return 0

    print(f"dataset_root: {data['dataset_root']}")
    print(f"fps: {data['fps']}")
    print(f"robot_type: {data['robot_type']}")
    print(f"action_mode: {data['action_mode']}")
    print("action_keys:")
    for key in data["action_keys"]:
        print(f"  - {key}")
    print("action_names:")
    for name in data["action_names"]:
        print(f"  - {name}")
    print("observation_keys:")
    for key in data["observation_keys"]:
        print(f"  - {key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
