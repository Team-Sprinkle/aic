"""Official-config-aligned scenario dataclasses and loaders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .official_refs import OFFICIAL_SAMPLE_CONFIG


@dataclass(frozen=True)
class RailEntity:
    present: bool
    name: str | None = None
    translation: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class TaskBoardScenario:
    pose_xyz_rpy: tuple[float, float, float, float, float, float]
    nic_rails: dict[str, RailEntity]
    sc_rails: dict[str, RailEntity]
    mount_rails: dict[str, RailEntity]


@dataclass(frozen=True)
class CableScenario:
    cable_name: str
    cable_type: str
    attach_to_gripper: bool
    gripper_offset_xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    spawn_pose_xyz: tuple[float, float, float] = (-0.35, 0.4, 1.15)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    cable_type: str
    cable_name: str
    plug_type: str
    plug_name: str
    port_type: str
    port_name: str
    target_module_name: str
    time_limit_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_task_msg_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "cable_type": self.cable_type,
            "cable_name": self.cable_name,
            "plug_type": self.plug_type,
            "plug_name": self.plug_name,
            "port_type": self.port_type,
            "port_name": self.port_name,
            "target_module_name": self.target_module_name,
            "time_limit": int(round(self.time_limit_s)),
        }


@dataclass(frozen=True)
class AicScenario:
    trial_id: str
    task_board: TaskBoardScenario
    cables: dict[str, CableScenario]
    tasks: dict[str, TaskDefinition]
    metadata: dict[str, Any] = field(default_factory=dict)


def load_trials(config_path: Path | str = OFFICIAL_SAMPLE_CONFIG) -> dict[str, AicScenario]:
    """Load AIC trial scenarios from the official engine YAML schema."""
    with Path(config_path).open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    trials: dict[str, AicScenario] = {}
    for trial_id, trial_cfg in config["trials"].items():
        scene = trial_cfg["scene"]
        board = scene["task_board"]
        task_board = TaskBoardScenario(
            pose_xyz_rpy=(
                float(board["pose"]["x"]),
                float(board["pose"]["y"]),
                float(board["pose"]["z"]),
                float(board["pose"]["roll"]),
                float(board["pose"]["pitch"]),
                float(board["pose"]["yaw"]),
            ),
            nic_rails={
                key: _parse_entity(board[key])
                for key in sorted(k for k in board if k.startswith("nic_rail_"))
            },
            sc_rails={
                key: _parse_entity(board[key])
                for key in sorted(k for k in board if k.startswith("sc_rail_"))
            },
            mount_rails={
                key: _parse_entity(board[key])
                for key in sorted(
                    k
                    for k in board
                    if any(
                        k.startswith(prefix)
                        for prefix in ("lc_mount_rail_", "sfp_mount_rail_", "sc_mount_rail_")
                    )
                )
            },
        )
        cables = {
            cable_name: CableScenario(
                cable_name=cable_name,
                cable_type=str(cable_cfg["cable_type"]),
                attach_to_gripper=bool(cable_cfg["attach_cable_to_gripper"]),
                spawn_pose_xyz=(
                    float(cable_cfg.get("pose", {}).get("x", -0.35)),
                    float(cable_cfg.get("pose", {}).get("y", 0.4)),
                    float(cable_cfg.get("pose", {}).get("z", 1.15)),
                ),
                gripper_offset_xyz=(
                    float(cable_cfg["pose"]["gripper_offset"]["x"]),
                    float(cable_cfg["pose"]["gripper_offset"]["y"]),
                    float(cable_cfg["pose"]["gripper_offset"]["z"]),
                ),
                rpy=(
                    float(cable_cfg["pose"]["roll"]),
                    float(cable_cfg["pose"]["pitch"]),
                    float(cable_cfg["pose"]["yaw"]),
                ),
            )
            for cable_name, cable_cfg in scene["cables"].items()
        }
        tasks = {
            task_id: TaskDefinition(
                task_id=task_id,
                cable_type=str(task_cfg["cable_type"]),
                cable_name=str(task_cfg["cable_name"]),
                plug_type=str(task_cfg["plug_type"]),
                plug_name=str(task_cfg["plug_name"]),
                port_type=str(task_cfg["port_type"]),
                port_name=str(task_cfg["port_name"]),
                target_module_name=str(task_cfg["target_module_name"]),
                time_limit_s=float(task_cfg["time_limit"]),
            )
            for task_id, task_cfg in trial_cfg["tasks"].items()
        }
        trials[trial_id] = AicScenario(
            trial_id=trial_id,
            task_board=task_board,
            cables=cables,
            tasks=tasks,
            metadata={"source_config": str(config_path)},
        )
    return trials


def _parse_entity(entity_cfg: dict[str, Any]) -> RailEntity:
    present = bool(entity_cfg["entity_present"])
    pose_cfg = entity_cfg.get("entity_pose", {})
    return RailEntity(
        present=present,
        name=str(entity_cfg["entity_name"]) if present and entity_cfg.get("entity_name") else None,
        translation=float(pose_cfg.get("translation", 0.0)),
        roll=float(pose_cfg.get("roll", 0.0)),
        pitch=float(pose_cfg.get("pitch", 0.0)),
        yaw=float(pose_cfg.get("yaw", 0.0)),
    )
