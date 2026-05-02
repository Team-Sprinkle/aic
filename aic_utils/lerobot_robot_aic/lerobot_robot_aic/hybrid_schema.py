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

"""Canonical hybrid-training schema metadata for AIC datasets.

This module describes the dataset/training interface shared by Gazebo expert
data, ACT, offline SERL, and future Isaac/Gazebo adapters. It intentionally does
not modify runtime policy classes.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import prod
from pathlib import Path
from typing import Any

from .dataset_schema import load_lerobot_info, summarize_dataset_schema

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is available in the repo env.
    yaml = None


SIMULATOR_SOURCES = {"gazebo", "isaac", "unknown"}
IMAGE_DTYPES = {"image", "video"}


@dataclass(frozen=True)
class HybridSchemaSummary:
    task_family: str
    action_mode: str
    action_dim: int | None
    action_horizon: int
    obs_mode: str
    obs_dim: int | None
    camera_keys: list[str]
    lowdim_keys: list[str]
    dataset_root: Path
    simulator_source: str
    action_keys: list[str]
    observation_keys: list[str]
    feature_keys: list[str]
    validation: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "action_mode": self.action_mode,
            "action_dim": self.action_dim,
            "action_horizon": self.action_horizon,
            "obs_mode": self.obs_mode,
            "obs_dim": self.obs_dim,
            "camera_keys": self.camera_keys,
            "lowdim_keys": self.lowdim_keys,
            "dataset_root": str(self.dataset_root),
            "simulator_source": self.simulator_source,
            "action_keys": self.action_keys,
            "observation_keys": self.observation_keys,
            "feature_keys": self.feature_keys,
            "validation": self.validation,
        }


def _shape(feature: dict[str, Any]) -> list[int] | None:
    raw = feature.get("shape")
    if isinstance(raw, (list, tuple)):
        return [int(item) for item in raw]
    return None


def _dim_from_shape(shape: list[int] | None) -> int | None:
    if not shape:
        return None
    return int(prod(shape))


def _is_camera_feature(key: str, feature: dict[str, Any]) -> bool:
    dtype = str(feature.get("dtype", "")).lower()
    if dtype in IMAGE_DTYPES:
        return True
    if key.startswith("observation.images.") or key.startswith("observation.image."):
        return True
    shape = _shape(feature)
    return bool(key.startswith("observation.") and shape and len(shape) == 3)


def _is_lowdim_observation(key: str, feature: dict[str, Any]) -> bool:
    if not key.startswith("observation"):
        return False
    if _is_camera_feature(key, feature):
        return False
    shape = _shape(feature)
    return bool(shape and len(shape) == 1)


def _find_request_yaml(dataset_root: Path) -> Path | None:
    candidates = [
        dataset_root / "request.yaml",
        dataset_root.parent / "request.yaml",
        dataset_root.parent.parent / "request.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_request(dataset_root: Path) -> dict[str, Any]:
    request_path = _find_request_yaml(dataset_root)
    if request_path is None or yaml is None:
        return {}
    with request_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def infer_task_family(dataset_root: Path, info: dict[str, Any], request: dict[str, Any]) -> str:
    for source in (request, info):
        value = source.get("task_family")
        if isinstance(value, str) and value:
            return value
    lowered_parts = [part.lower() for part in dataset_root.parts]
    for family in ("sfp_to_nic", "sc_to_sc"):
        if family in lowered_parts:
            return family
    return "unknown"


def infer_simulator_source(dataset_root: Path, info: dict[str, Any], request: dict[str, Any]) -> str:
    for source in (request.get("generation", {}), request, info):
        value = source.get("simulator_source")
        if isinstance(value, str) and value.lower() in SIMULATOR_SOURCES:
            return value.lower()
    text = "/".join(part.lower() for part in dataset_root.parts)
    if "isaac" in text:
        return "isaac"
    if "gazebo" in text or "trajectory_datasets" in text:
        return "gazebo"
    return "unknown"


def infer_obs_mode(camera_keys: list[str], lowdim_keys: list[str]) -> str:
    if camera_keys and lowdim_keys:
        return "image_lowdim"
    if camera_keys:
        return "image"
    if lowdim_keys:
        return "lowdim"
    return "unknown"


def inspect_hybrid_schema(
    dataset_root: Path,
    *,
    action_horizon: int = 1,
    simulator_source: str | None = None,
) -> HybridSchemaSummary:
    if action_horizon < 1:
        raise ValueError("action_horizon must be >= 1")

    info = load_lerobot_info(dataset_root)
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("LeRobot info.json field 'features' must be a map")
    request = _load_request(dataset_root)
    dataset_schema = summarize_dataset_schema(dataset_root)

    camera_keys: list[str] = []
    lowdim_keys: list[str] = []
    obs_dim = 0
    for key, feature in features.items():
        if not isinstance(feature, dict):
            continue
        if _is_camera_feature(key, feature):
            camera_keys.append(key)
            continue
        if _is_lowdim_observation(key, feature):
            lowdim_keys.append(key)
            dim = _dim_from_shape(_shape(feature))
            if dim is not None:
                obs_dim += dim

    action_feature = features.get("action", {})
    action_dim = _dim_from_shape(_shape(action_feature)) if isinstance(action_feature, dict) else None
    source = simulator_source or infer_simulator_source(dataset_root, info, request)
    if source not in SIMULATOR_SOURCES:
        raise ValueError(f"simulator_source must be one of {sorted(SIMULATOR_SOURCES)}")

    validation = {
        "has_meta_info": (dataset_root / "meta" / "info.json").exists(),
        "has_data_dir": (dataset_root / "data").exists(),
        "has_videos_dir": (dataset_root / "videos").exists(),
        "has_images_dir": (dataset_root / "images").exists(),
        "has_request_yaml": _find_request_yaml(dataset_root) is not None,
    }
    return HybridSchemaSummary(
        task_family=infer_task_family(dataset_root, info, request),
        action_mode=dataset_schema.action_mode,
        action_dim=action_dim,
        action_horizon=action_horizon,
        obs_mode=infer_obs_mode(sorted(camera_keys), sorted(lowdim_keys)),
        obs_dim=obs_dim if lowdim_keys else None,
        camera_keys=sorted(camera_keys),
        lowdim_keys=sorted(lowdim_keys),
        dataset_root=dataset_root,
        simulator_source=source,
        action_keys=dataset_schema.action_keys,
        observation_keys=dataset_schema.observation_keys,
        feature_keys=dataset_schema.feature_keys,
        validation=validation,
    )


def hybrid_schema_json(summary: HybridSchemaSummary) -> str:
    return json.dumps(summary.as_dict(), indent=2, sort_keys=True)
