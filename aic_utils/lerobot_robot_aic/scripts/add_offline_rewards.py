#!/usr/bin/env python3
"""Add dense offline SERL rewards to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.offline_rewards import OfflineRewardConfig, dense_offline_rewards  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--objective", choices=["insertion", "near_gate"], default="insertion")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validation-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    shutil.copytree(args.dataset_root, args.output_root)
    config = OfflineRewardConfig(objective=args.objective)

    data_files = sorted((args.output_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.output_root / 'data'}")
    rewards = []
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        df["reward"] = dense_offline_rewards(df, config)
        rewards.extend(float(value) for value in df["reward"].to_numpy())
        df.to_parquet(data_file, index=False)

    info_path = args.output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["aic_offline_rewards"] = {
        "schema_version": "aic_offline_rewards/v1",
        "method": "task_geometry_dense_reward",
        "config": config.as_dict(),
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "config": config.as_dict(),
        "reward_count": len(rewards),
        "reward_min": min(rewards) if rewards else None,
        "reward_max": max(rewards) if rewards else None,
        "reward_mean": sum(rewards) / len(rewards) if rewards else None,
    }
    report_path = args.validation_report or (args.output_root / "meta" / "offline_reward_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
