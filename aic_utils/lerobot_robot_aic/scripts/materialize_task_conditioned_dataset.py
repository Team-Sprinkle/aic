#!/usr/bin/env python3
"""Create a LeRobot dataset with task vectors appended to observation.state."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_aic.task_metadata import (  # noqa: E402
    append_task_vectors_to_observation_state_dataset,
    infer_task_metadata_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--task-metadata", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--missing-task-vector", choices=["error", "zeros"], default="error")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = args.task_metadata.resolve() if args.task_metadata else infer_task_metadata_path(dataset_root)
    if manifest_path is None and args.missing_task_vector == "error":
        raise FileNotFoundError(
            "No task metadata was provided or found at <dataset_parent>/manifests/accepted.csv"
        )

    summary = {
        "dataset_root": str(dataset_root),
        "task_metadata": str(manifest_path) if manifest_path else None,
        "output_root": str(output_root),
        "missing_task_vector": args.missing_task_vector,
        "overwrite": bool(args.overwrite),
        "method": "append_task_vector_to_observation.state",
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    result = append_task_vectors_to_observation_state_dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_root=output_root,
        missing=args.missing_task_vector,
        overwrite=args.overwrite,
    )
    summary["materialized_dataset_root"] = str(result)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
