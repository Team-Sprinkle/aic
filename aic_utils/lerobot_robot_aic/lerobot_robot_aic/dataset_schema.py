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

"""Training-side helpers for inspecting AIC LeRobot dataset schemas.

These helpers intentionally do not alter the runtime policy/control interfaces.
They document and validate the action schema present in a recorded dataset so
training scripts can make conservative choices for Cartesian or joint data.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any

from .types import JointMotionUpdateActionDict, MotionUpdateActionDict


CARTESIAN_ACTION_NAMES = set(MotionUpdateActionDict.__annotations__)
JOINT_ACTION_NAMES = set(JointMotionUpdateActionDict.__annotations__)


@dataclass(frozen=True)
class DatasetSchemaSummary:
    dataset_root: Path
    fps: int | None
    robot_type: str | None
    feature_keys: list[str]
    action_keys: list[str]
    observation_keys: list[str]
    action_names: list[str]
    action_shape: list[int] | None
    action_mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.dataset_root),
            "fps": self.fps,
            "robot_type": self.robot_type,
            "feature_keys": self.feature_keys,
            "action_keys": self.action_keys,
            "observation_keys": self.observation_keys,
            "action_names": self.action_names,
            "action_shape": self.action_shape,
            "action_mode": self.action_mode,
        }


def load_lerobot_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing LeRobot dataset metadata: {info_path}")
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    if not isinstance(info, dict):
        raise ValueError(f"LeRobot info.json must contain a JSON object: {info_path}")
    return info


def _feature_names(feature: dict[str, Any]) -> list[str]:
    names = feature.get("names", [])
    if isinstance(names, dict):
        # LeRobot vector features may store names as {"motors": [...]}.
        flattened: list[str] = []
        for value in names.values():
            if isinstance(value, list):
                flattened.extend(str(item) for item in value)
            else:
                flattened.append(str(value))
        return flattened
    if isinstance(names, list):
        return [str(item) for item in names]
    return []


def _feature_shape(feature: dict[str, Any]) -> list[int] | None:
    shape = feature.get("shape")
    if isinstance(shape, list):
        return [int(x) for x in shape]
    if isinstance(shape, tuple):
        return [int(x) for x in shape]
    return None


def _feature_keys_by_prefix(features: dict[str, Any], prefix: str) -> list[str]:
    return sorted(k for k in features if k == prefix or k.startswith(f"{prefix}."))


def action_names_from_features(features: dict[str, Any]) -> list[str]:
    names: list[str] = []
    action_feature = features.get("action")
    if isinstance(action_feature, dict):
        names.extend(_feature_names(action_feature))

    for key in _feature_keys_by_prefix(features, "action"):
        if key == "action":
            continue
        names.append(key.removeprefix("action."))

    return sorted(dict.fromkeys(names))


def detect_action_mode(features: dict[str, Any]) -> str:
    names = set(action_names_from_features(features))
    if not names:
        return "unknown"
    if CARTESIAN_ACTION_NAMES.issubset(names):
        return "cartesian"
    if JOINT_ACTION_NAMES.issubset(names):
        return "joint"
    return "unknown"


def summarize_dataset_schema(dataset_root: Path) -> DatasetSchemaSummary:
    info = load_lerobot_info(dataset_root)
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("LeRobot info.json field 'features' must be a map")

    action_feature = features.get("action", {})
    return DatasetSchemaSummary(
        dataset_root=dataset_root,
        fps=info.get("fps"),
        robot_type=info.get("robot_type"),
        feature_keys=sorted(features),
        action_keys=_feature_keys_by_prefix(features, "action"),
        observation_keys=_feature_keys_by_prefix(features, "observation"),
        action_names=action_names_from_features(features),
        action_shape=_feature_shape(action_feature) if isinstance(action_feature, dict) else None,
        action_mode=detect_action_mode(features),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an AIC LeRobot dataset schema and infer its action mode.",
    )
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

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
