#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Validate compatibility between two LeRobot datasets before merge/resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _normalize_feature(ft: dict[str, Any]) -> dict[str, Any]:
    """Normalize feature metadata for deterministic compatibility checks.

    The LeRobot `features` entries can contain dynamic `info` fields (video codec,
    dimensions, etc.) that are expected to vary across recordings. We exclude them
    for schema checks.
    """
    out = dict(ft)
    out.pop("info", None)

    if "shape" in out and isinstance(out["shape"], list):
        out["shape"] = tuple(out["shape"])

    if "names" in out and isinstance(out["names"], tuple):
        out["names"] = list(out["names"])

    return out


def _load_dataset(repo_id: str, root: Path | None) -> LeRobotDataset:
    return LeRobotDataset(repo_id, root=root)


def _is_action_key(key: str) -> bool:
    return key == "action" or key.startswith("action.")


def _find_mismatches(
    base: LeRobotDataset,
    candidate: LeRobotDataset,
    enforce_name_order: bool,
    ignore_robot_type: bool,
) -> list[str]:
    errors: list[str] = []

    if not ignore_robot_type and base.meta.robot_type != candidate.meta.robot_type:
        errors.append(
            f"robot_type mismatch: base={base.meta.robot_type} candidate={candidate.meta.robot_type}"
        )

    if base.fps != candidate.fps:
        errors.append(f"fps mismatch: base={base.fps} candidate={candidate.fps}")

    base_keys = set(base.features.keys())
    cand_keys = set(candidate.features.keys())
    if base_keys != cand_keys:
        missing_in_candidate = sorted(base_keys - cand_keys)
        extra_in_candidate = sorted(cand_keys - base_keys)
        if missing_in_candidate:
            errors.append(
                "missing feature keys in candidate: " + ", ".join(missing_in_candidate)
            )
        if extra_in_candidate:
            errors.append(
                "extra feature keys in candidate: " + ", ".join(extra_in_candidate)
            )

    shared_keys = sorted(base_keys & cand_keys)
    for key in shared_keys:
        base_ft = _normalize_feature(base.features[key])
        cand_ft = _normalize_feature(candidate.features[key])

        if not enforce_name_order and "names" in base_ft and "names" in cand_ft:
            base_ft["names"] = sorted(base_ft["names"])
            cand_ft["names"] = sorted(cand_ft["names"])

        if base_ft != cand_ft:
            errors.append(
                "feature mismatch for key "
                f"'{key}':\n  base={json.dumps(base_ft, sort_keys=True)}"
                f"\n  candidate={json.dumps(cand_ft, sort_keys=True)}"
            )

    base_action_keys = sorted([k for k in base.features if _is_action_key(k)])
    cand_action_keys = sorted([k for k in candidate.features if _is_action_key(k)])
    if base_action_keys != cand_action_keys:
        errors.append(
            f"action feature keys mismatch: base={base_action_keys} candidate={cand_action_keys}"
        )

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether two LeRobot datasets are merge-compatible.",
    )

    parser.add_argument(
        "--base.repo_id",
        dest="base_repo_id",
        required=True,
        help="Reference dataset repo_id (typically teleop dataset).",
    )
    parser.add_argument(
        "--candidate.repo_id",
        dest="candidate_repo_id",
        required=True,
        help="Candidate dataset repo_id to append/merge (typically policy dataset).",
    )
    parser.add_argument(
        "--base.root",
        dest="base_root",
        type=Path,
        default=None,
        help="Optional local root override for base dataset.",
    )
    parser.add_argument(
        "--candidate.root",
        dest="candidate_root",
        type=Path,
        default=None,
        help="Optional local root override for candidate dataset.",
    )
    parser.add_argument(
        "--ignore_robot_type",
        action="store_true",
        help="Allow different robot_type metadata.",
    )
    parser.add_argument(
        "--allow_reordered_names",
        action="store_true",
        help="Allow vector feature name reordering if names are otherwise identical.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON summary.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base = _load_dataset(args.base_repo_id, args.base_root)
    candidate = _load_dataset(args.candidate_repo_id, args.candidate_root)

    mismatches = _find_mismatches(
        base=base,
        candidate=candidate,
        enforce_name_order=not args.allow_reordered_names,
        ignore_robot_type=args.ignore_robot_type,
    )

    result = {
        "compatible": len(mismatches) == 0,
        "base": {
            "repo_id": args.base_repo_id,
            "root": str(args.base_root) if args.base_root else None,
            "robot_type": base.meta.robot_type,
            "fps": base.fps,
            "total_episodes": base.num_episodes,
        },
        "candidate": {
            "repo_id": args.candidate_repo_id,
            "root": str(args.candidate_root) if args.candidate_root else None,
            "robot_type": candidate.meta.robot_type,
            "fps": candidate.fps,
            "total_episodes": candidate.num_episodes,
        },
        "mismatches": mismatches,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if result["compatible"]:
            print("COMPATIBLE: datasets can be merged/resumed safely.")
        else:
            print("INCOMPATIBLE: dataset schemas/metadata differ.")
            for idx, mismatch in enumerate(mismatches, start=1):
                print(f"{idx}. {mismatch}")

    raise SystemExit(0 if result["compatible"] else 1)


if __name__ == "__main__":
    main()
