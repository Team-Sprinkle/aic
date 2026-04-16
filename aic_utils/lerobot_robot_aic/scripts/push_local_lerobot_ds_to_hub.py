#!/usr/bin/env python3

"""Push a local LeRobot dataset directory to the Hub via LeRobotDataset API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local LeRobot dataset directory with LeRobotDataset and "
            "push it to the Hugging Face Hub."
        )
    )
    parser.add_argument(
        "--local-dataset-dir",
        required=True,
        type=Path,
        help="Path to local LeRobot dataset root (must contain meta/info.json).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hub repo id, e.g. '<user_or_org>/<dataset_name>'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/push repository as private.",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated dataset tags to attach on push.",
    )
    return parser.parse_args()


def parse_tags(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]


def validate_local_dataset_dir(dataset_dir: Path) -> None:
    info_path = dataset_dir / "meta" / "info.json"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Local dataset directory not found: {dataset_dir}")
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Not a LeRobot dataset root (missing {info_path})."
        )
    try:
        json.loads(info_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid JSON in {info_path}: {exc}") from exc


def main() -> int:
    args = parse_args()
    local_dataset_dir = args.local_dataset_dir.resolve()
    validate_local_dataset_dir(local_dataset_dir)
    tags = parse_tags(args.tags)

    dataset_name = local_dataset_dir.name

    dataset = LeRobotDataset(args.repo_id, root=local_dataset_dir)

    print(f"[load] local_dataset_dir={local_dataset_dir}")
    print(f"[load] dataset_name={dataset_name}")
    print(f"[push] repo_id={args.repo_id}")
    print(f"[push] private={args.private}")
    print(f"[push] tags={tags}")

    dataset.push_to_hub(tags=tags, private=args.private)

    print("[done] Push complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
