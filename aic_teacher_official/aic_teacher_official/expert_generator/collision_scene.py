"""Best-effort rigid planning-scene extraction from AIC engine YAML."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aic_teacher_official.expert_generator.scene_snapshot import ObjectGeometry, SerializablePose


BOARD_BOX_DIMENSIONS_M = [0.75, 0.38, 0.035]
NIC_BOX_DIMENSIONS_M = [0.12, 0.06, 0.03]
SC_MOUNT_BOX_DIMENSIONS_M = [0.08, 0.05, 0.04]
FIXTURE_BOX_DIMENSIONS_M = [0.06, 0.04, 0.035]


def load_engine_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Engine config must be a YAML map: {config_path}")
    return data


def first_trial_id(config: dict[str, Any]) -> str:
    trials = config.get("trials")
    if not isinstance(trials, dict) or not trials:
        raise ValueError("Engine config must contain a non-empty 'trials' map")
    return next(iter(trials.keys()))


def trial_config(config: dict[str, Any], trial_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved = trial_id or first_trial_id(config)
    trials = config.get("trials")
    if not isinstance(trials, dict) or resolved not in trials:
        raise ValueError(f"Trial {resolved!r} not found in engine config")
    trial = trials[resolved]
    if not isinstance(trial, dict):
        raise ValueError(f"Trial {resolved!r} must be a YAML map")
    return resolved, trial


def object_geometries_from_engine_config(
    config_or_path: dict[str, Any] | str | Path,
    *,
    trial_id: str | None = None,
) -> list[ObjectGeometry]:
    config = (
        load_engine_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    resolved_trial_id, trial = trial_config(config, trial_id)
    scene = trial.get("scene", {}) if isinstance(trial.get("scene", {}), dict) else {}
    board = scene.get("task_board", {}) if isinstance(scene.get("task_board", {}), dict) else {}
    board_pose = board.get("pose", {}) if isinstance(board.get("pose", {}), dict) else {}
    objects: list[ObjectGeometry] = []
    objects.append(
        ObjectGeometry(
            name=f"{resolved_trial_id}_task_board_base",
            pose=SerializablePose(
                position=[
                    float(board_pose.get("x", 0.0)),
                    float(board_pose.get("y", 0.0)),
                    float(board_pose.get("z", 0.0)),
                ],
                orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                frame_id="world",
            ),
            shape="box",
            dimensions=BOARD_BOX_DIMENSIONS_M,
            role="task_board",
            metadata={"source": "engine_config", "yaw_rad": float(board_pose.get("yaw", 0.0))},
        )
    )
    for rail_name, rail in board.items():
        if not isinstance(rail, dict) or not rail.get("entity_present"):
            continue
        if not rail_name.endswith("_rail_0") and "_rail_" not in rail_name:
            continue
        pose = rail.get("entity_pose", {}) if isinstance(rail.get("entity_pose", {}), dict) else {}
        entity_name = str(rail.get("entity_name", rail_name))
        role, dims = _role_and_dimensions(rail_name)
        objects.append(
            ObjectGeometry(
                name=f"{resolved_trial_id}_{entity_name}",
                pose=SerializablePose(
                    position=[
                        float(board_pose.get("x", 0.0)),
                        float(board_pose.get("y", 0.0)) + float(pose.get("translation", 0.0)),
                        float(board_pose.get("z", 0.0)),
                    ],
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    frame_id="world",
                ),
                shape="box",
                dimensions=dims,
                role=role,
                metadata={
                    "source": "engine_config",
                    "rail": rail_name,
                    "translation_along_rail_m": float(pose.get("translation", 0.0)),
                    "yaw_rad": float(pose.get("yaw", 0.0)),
                },
            )
        )
    return objects


def _role_and_dimensions(rail_name: str) -> tuple[str, list[float]]:
    if rail_name.startswith("nic_rail"):
        return "nic_card", NIC_BOX_DIMENSIONS_M
    if rail_name.startswith("sc_rail"):
        return "sc_port", SC_MOUNT_BOX_DIMENSIONS_M
    return "fixture_mount", FIXTURE_BOX_DIMENSIONS_M
