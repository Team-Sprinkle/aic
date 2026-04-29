"""JSON data model for official teacher trajectories.

The model is intentionally ROS-free so planning, smoothing, and tests can run
outside the official container. The replay policy converts these records to ROS
messages only at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import json
from pathlib import Path
from typing import Any


class PhaseLabel(StrEnum):
    APPROACH = "approach"
    ALIGNMENT = "alignment"
    OBSTACLE_AVOIDANCE = "obstacle_avoidance"
    PRE_INSERTION = "pre_insertion"
    FINAL_INSERTION = "final_insertion"
    HOLD = "hold"


class SourceLabel(StrEnum):
    VLM = "vlm"
    PLACEHOLDER_VLM = "placeholder_vlm"
    OPTIMIZER = "optimizer"
    PLACEHOLDER_OPTIMIZER = "placeholder_optimizer"
    CHEATCODE = "cheatcode"
    POSTPROCESSOR = "postprocessor"
    HUMAN = "human"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TCPPose:
    position: list[float]
    orientation_xyzw: list[float]

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("TCPPose.position must contain [x, y, z]")
        if len(self.orientation_xyzw) != 4:
            raise ValueError("TCPPose.orientation_xyzw must contain [x, y, z, w]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TCPPose":
        orientation = data.get("orientation_xyzw", data.get("orientation"))
        if orientation is None:
            raise ValueError("TCP pose requires orientation_xyzw")
        return cls(
            position=[float(v) for v in data["position"]],
            orientation_xyzw=[float(v) for v in orientation],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position": [float(v) for v in self.position],
            "orientation_xyzw": [float(v) for v in self.orientation_xyzw],
        }


@dataclass(frozen=True)
class TrajectoryWaypoint:
    timestamp: float
    tcp_pose: TCPPose
    phase: PhaseLabel
    source: SourceLabel
    tcp_velocity: list[float] | None = None
    gripper_state: dict[str, Any] | None = None
    cable_state: dict[str, Any] | None = None
    port_state: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.tcp_velocity is not None and len(self.tcp_velocity) != 3:
            raise ValueError("tcp_velocity must contain [vx, vy, vz]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryWaypoint":
        return cls(
            timestamp=float(data["timestamp"]),
            tcp_pose=TCPPose.from_dict(data["tcp_pose"]),
            tcp_velocity=(
                [float(v) for v in data["tcp_velocity"]]
                if data.get("tcp_velocity") is not None
                else None
            ),
            gripper_state=data.get("gripper_state"),
            cable_state=data.get("cable_state"),
            port_state=data.get("port_state"),
            phase=PhaseLabel(data.get("phase", PhaseLabel.APPROACH)),
            source=SourceLabel(data.get("source", SourceLabel.UNKNOWN)),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "timestamp": float(self.timestamp),
            "tcp_pose": self.tcp_pose.to_dict(),
            "phase": self.phase.value,
            "source": self.source.value,
            "diagnostics": dict(self.diagnostics),
        }
        if self.tcp_velocity is not None:
            data["tcp_velocity"] = [float(v) for v in self.tcp_velocity]
        if self.gripper_state is not None:
            data["gripper_state"] = self.gripper_state
        if self.cable_state is not None:
            data["cable_state"] = self.cable_state
        if self.port_state is not None:
            data["port_state"] = self.port_state
        return data


@dataclass(frozen=True)
class TrajectoryMetadata:
    schema_version: str = "official_teacher_trajectory/v0"
    task: dict[str, Any] = field(default_factory=dict)
    planning: dict[str, Any] = field(default_factory=dict)
    postprocessing: dict[str, Any] = field(default_factory=dict)
    recording: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TrajectoryMetadata":
        if not data:
            return cls()
        return cls(
            schema_version=str(data.get("schema_version", cls.schema_version)),
            task=dict(data.get("task", {})),
            planning=dict(data.get("planning", {})),
            postprocessing=dict(data.get("postprocessing", {})),
            recording=dict(data.get("recording", {})),
            diagnostics=dict(data.get("diagnostics", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task": dict(self.task),
            "planning": dict(self.planning),
            "postprocessing": dict(self.postprocessing),
            "recording": dict(self.recording),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class PiecewiseTrajectory:
    waypoints: list[TrajectoryWaypoint]
    metadata: TrajectoryMetadata = field(default_factory=TrajectoryMetadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PiecewiseTrajectory":
        waypoints = [TrajectoryWaypoint.from_dict(w) for w in data["waypoints"]]
        if len(waypoints) < 2:
            raise ValueError("PiecewiseTrajectory requires at least two waypoints")
        return cls(
            waypoints=waypoints,
            metadata=TrajectoryMetadata.from_dict(data.get("metadata")),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "PiecewiseTrajectory":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "waypoints": [w.to_dict() for w in self.waypoints],
        }


@dataclass(frozen=True)
class SmoothTrajectory:
    waypoints: list[TrajectoryWaypoint]
    metadata: TrajectoryMetadata = field(default_factory=TrajectoryMetadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SmoothTrajectory":
        waypoints = [TrajectoryWaypoint.from_dict(w) for w in data["waypoints"]]
        if len(waypoints) < 2:
            raise ValueError("SmoothTrajectory requires at least two waypoints")
        return cls(
            waypoints=waypoints,
            metadata=TrajectoryMetadata.from_dict(data.get("metadata")),
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SmoothTrajectory":
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "waypoints": [w.to_dict() for w in self.waypoints],
        }


def assert_monotonic_timestamps(waypoints: list[TrajectoryWaypoint]) -> None:
    for prev, curr in zip(waypoints, waypoints[1:]):
        if curr.timestamp <= prev.timestamp:
            raise ValueError(
                "Trajectory timestamps must be strictly increasing: "
                f"{prev.timestamp} then {curr.timestamp}"
            )
