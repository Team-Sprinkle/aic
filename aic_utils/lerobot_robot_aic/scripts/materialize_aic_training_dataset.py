#!/usr/bin/env python3
"""Materialize the canonical AIC training dataset schema.

The output observation.state order is:
  32D base state + 40D contact/recovery features + 10D task vector
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.task_metadata import append_task_vectors_to_observation_state_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--task-metadata", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reward-objective",
        choices=["none", "insertion", "near_gate"],
        default="none",
        help="Materialize dense offline SERL reward column when objective geometry columns are available.",
    )
    parser.add_argument("--missing-task-vector", choices=["error", "zeros"], default="error")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    args.output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="aic_training_dataset_") as tmp:
        tmp_root = Path(tmp)
        task_root = tmp_root / "task_conditioned"
        contact_root = args.output_root if args.reward_objective == "none" else tmp_root / "contact_features"
        append_task_vectors_to_observation_state_dataset(
            dataset_root=args.dataset_root,
            manifest_path=args.task_metadata,
            output_root=task_root,
            missing=args.missing_task_vector,
            overwrite=True,
        )
        _run(
            [
                sys.executable,
                str(PACKAGE_DIR / "scripts" / "add_contact_recovery_features.py"),
                "--dataset-root",
                str(task_root),
                "--output-root",
                str(contact_root),
                "--overwrite",
            ]
        )
        if args.reward_objective != "none":
            _run(
                [
                    sys.executable,
                    str(PACKAGE_DIR / "scripts" / "add_offline_rewards.py"),
                    "--dataset-root",
                    str(contact_root),
                    "--output-root",
                    str(args.output_root),
                    "--objective",
                    str(args.reward_objective),
                    "--overwrite",
                ]
            )

    manifest = {
        "schema_version": "aic_training_dataset_materialization/v1",
        "dataset_root": str(args.dataset_root),
        "task_metadata": str(args.task_metadata),
        "output_root": str(args.output_root),
        "state_schema": "base32_contact40_task10",
        "reward_objective": args.reward_objective,
    }
    report_path = args.output_root / "meta" / "aic_training_dataset_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
