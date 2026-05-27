"""Serializable scene state used by expert trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SerializablePose:
    """Frame-qualified pose with xyzw quaternion ordering."""

    position: list[float]
    orientation_xyzw: list[float]
    frame_id: str = "base_link"

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("SerializablePose.position must contain [x, y, z]")
        if len(self.orientation_xyzw) != 4:
            raise ValueError("SerializablePose.orientation_xyzw must contain [x, y, z, w]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerializablePose":
        return cls(
            position=[float(v) for v in data["position"]],
            orientation_xyzw=[float(v) for v in data["orientation_xyzw"]],
            frame_id=str(data.get("frame_id", "base_link")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": [float(v) for v in self.position],
            "orientation_xyzw": [float(v) for v in self.orientation_xyzw],
            "frame_id": self.frame_id,
        }


@dataclass(frozen=True)
class ObjectGeometry:
    """Rigid geometry or keep-out object for MoveIt planning-scene setup."""

    name: str
    pose: SerializablePose
    shape: str
    dimensions: list[float]
    role: str = "obstacle"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pose": self.pose.to_dict(),
            "shape": self.shape,
            "dimensions": [float(v) for v in self.dimensions],
            "role": self.role,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectGeometry":
        return cls(
            name=str(data["name"]),
            pose=SerializablePose.from_dict(data["pose"]),
            shape=str(data["shape"]),
            dimensions=[float(v) for v in data.get("dimensions", [])],
            role=str(data.get("role", "obstacle")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class SceneSnapshot:
    """Complete generation-time context.

    Missing fields are represented as ``None`` plus an entry in
    ``unavailable_reasons`` instead of raising. This makes debug metadata stable
    while allowing live ROS adapters to fill fields opportunistically.
    """

    run_id: str
    seed: int | None
    scene_id: str
    mode: str
    joint_state: dict[str, list[float]] | None = None
    tcp_pose: SerializablePose | None = None
    target_port_pose: SerializablePose | None = None
    plug_pose: SerializablePose | None = None
    plug_tip_pose: SerializablePose | None = None
    tf_frames: dict[str, str] = field(default_factory=dict)
    camera_images: list[str] = field(default_factory=list)
    ft_baseline: dict[str, Any] | None = None
    task_config: dict[str, Any] = field(default_factory=dict)
    collision_objects: list[ObjectGeometry] = field(default_factory=list)
    unavailable_reasons: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic_scene_snapshot/v1",
            "run_id": self.run_id,
            "seed": self.seed,
            "scene_id": self.scene_id,
            "mode": self.mode,
            "joint_state": self.joint_state,
            "tcp_pose": self.tcp_pose.to_dict() if self.tcp_pose else None,
            "target_port_pose": self.target_port_pose.to_dict() if self.target_port_pose else None,
            "plug_pose": self.plug_pose.to_dict() if self.plug_pose else None,
            "plug_tip_pose": self.plug_tip_pose.to_dict() if self.plug_tip_pose else None,
            "tf_frames": dict(self.tf_frames),
            "camera_images": list(self.camera_images),
            "ft_baseline": self.ft_baseline,
            "task_config": dict(self.task_config),
            "collision_objects": [obj.to_dict() for obj in self.collision_objects],
            "unavailable_reasons": dict(self.unavailable_reasons),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneSnapshot":
        return cls(
            run_id=str(data["run_id"]),
            seed=data.get("seed"),
            scene_id=str(data["scene_id"]),
            mode=str(data["mode"]),
            joint_state=data.get("joint_state"),
            tcp_pose=(
                SerializablePose.from_dict(data["tcp_pose"])
                if data.get("tcp_pose") is not None
                else None
            ),
            target_port_pose=(
                SerializablePose.from_dict(data["target_port_pose"])
                if data.get("target_port_pose") is not None
                else None
            ),
            plug_pose=(
                SerializablePose.from_dict(data["plug_pose"])
                if data.get("plug_pose") is not None
                else None
            ),
            plug_tip_pose=(
                SerializablePose.from_dict(data["plug_tip_pose"])
                if data.get("plug_tip_pose") is not None
                else None
            ),
            tf_frames=dict(data.get("tf_frames", {})),
            camera_images=list(data.get("camera_images", [])),
            ft_baseline=data.get("ft_baseline"),
            task_config=dict(data.get("task_config", {})),
            collision_objects=[
                ObjectGeometry.from_dict(obj) for obj in data.get("collision_objects", [])
            ],
            unavailable_reasons=dict(data.get("unavailable_reasons", {})),
            metadata=dict(data.get("metadata", {})),
        )
