#!/usr/bin/env python3
"""Append offline contact/recovery features to every LeRobot dataset in a tree."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/trajectory_datasets/clean"),
        help="Root containing raw LeRobot datasets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/trajectory_datasets/clean_with_contact_features"),
        help="Root where derived datasets with appended contact/recovery features are written.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip output roots that already contain contact/recovery feature metadata.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def dataset_roots(input_root: Path) -> list[Path]:
    roots = []
    for info_path in sorted(input_root.rglob("meta/info.json")):
        root = info_path.parent.parent
        if (root / "data").is_dir():
            roots.append(root)
    return roots


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    script = (
        Path(__file__).resolve().parents[1]
        / "aic_utils"
        / "lerobot_robot_aic"
        / "scripts"
        / "add_contact_recovery_features.py"
    )
    roots = dataset_roots(input_root)
    if args.limit is not None:
        roots = roots[: args.limit]
    if not roots:
        raise FileNotFoundError(f"No LeRobot dataset roots found under {input_root}")

    for index, dataset_root in enumerate(roots, start=1):
        rel = dataset_root.relative_to(input_root)
        out = output_root / rel
        if args.skip_existing and (out / "meta" / "contact_recovery_feature_report.json").exists():
            print(f"[{index}/{len(roots)}] {rel} (skip existing)", flush=True)
            continue
        cmd = [
            sys.executable,
            str(script),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(out),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        print(f"[{index}/{len(roots)}] {rel}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
