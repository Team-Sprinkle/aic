#!/usr/bin/env python3
"""Add dense offline SERL rewards to a LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.offline_rewards import (  # noqa: E402
    OfflineRewardConfig,
    dense_offline_reward_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--objective", choices=["insertion", "near_gate"], default="insertion")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--link-files",
        action="store_true",
        help=(
            "Materialize output with hardlinks before rewriting parquet/meta files. "
            "This avoids copying or re-encoding large videos while keeping the source dataset unchanged."
        ),
    )
    parser.add_argument(
        "--swap-rgb-channels",
        action="store_true",
        help=(
            "Mark the output dataset as requiring RGB channel reversal at load time. "
            "This is a temporary no-reencode patch for datasets whose recorded videos have red/blue swapped."
        ),
    )
    parser.add_argument("--validation-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    copy_function = os.link if args.link_files else shutil.copy2
    shutil.copytree(args.dataset_root, args.output_root, copy_function=copy_function)
    config = OfflineRewardConfig(objective=args.objective)

    data_files = sorted((args.output_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.output_root / 'data'}")

    info_path = args.output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    observation_state_names = (
        info.get("features", {}).get("observation.state", {}).get("names")
        if isinstance(info.get("features"), dict)
        else None
    )
    if observation_state_names is not None and not isinstance(observation_state_names, list):
        observation_state_names = None

    component_values: dict[str, list[float]] = {}
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if observation_state_names is not None:
            df.attrs["observation_state_names"] = observation_state_names
        components = dense_offline_reward_components(df, config)
        df["reward"] = components["reward"]
        for name, values in components.items():
            component_values.setdefault(name, []).extend(float(value) for value in values)
        df.to_parquet(data_file, index=False)

    info["aic_offline_rewards"] = {
        "schema_version": "aic_offline_rewards/v2_dense_parity_isaac",
        "method": "task_geometry_dense_reward",
        "config": config.as_dict(),
    }
    info["aic_rgb_patch"] = {
        "swap_rgb_channels": bool(args.swap_rgb_channels),
        "method": "loader_time_channel_reverse_no_video_reencode" if args.swap_rgb_channels else "none",
    }
    info.setdefault("features", {})["reward"] = {
        "dtype": "float32",
        "shape": [1],
        "names": None,
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    rewards = component_values.get("reward", [])
    component_stats = {}
    for name, values in component_values.items():
        component_stats[name] = {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }
    report = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "config": config.as_dict(),
        "link_files": bool(args.link_files),
        "swap_rgb_channels": bool(args.swap_rgb_channels),
        "component_stats": component_stats,
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
