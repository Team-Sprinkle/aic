"""Observation schema and validation for the training-only Gazebo environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ObservationDict = dict[str, Any]


@dataclass(frozen=True)
class Observation:
    """Validated observation returned by the backend/runtime pipeline.

    The public environment API still returns plain dicts. This schema exists to
    define the stable observation contract and to validate raw backend payloads.
    """

    step_count: int
    sim_time: float
    joint_positions: list[float]
    joint_velocities: list[float]
    end_effector_pose: dict[str, list[float]]

    def to_dict(self) -> ObservationDict:
        """Convert the validated observation to the public dict form."""
        return {
            "step_count": self.step_count,
            "sim_time": self.sim_time,
            "joint_positions": list(self.joint_positions),
            "joint_velocities": list(self.joint_velocities),
            "end_effector_pose": {
                "position": list(self.end_effector_pose["position"]),
                "orientation": list(self.end_effector_pose["orientation"]),
            },
        }

    @classmethod
    def from_dict(cls, payload: ObservationDict) -> "Observation":
        """Validate and normalize a raw backend observation payload."""
        required_keys = (
            "sim_time",
            "joint_positions",
            "joint_velocities",
            "end_effector_pose",
        )
        for key in required_keys:
            if key not in payload:
                raise ValueError(f"Observation is missing required key: '{key}'.")

        step_count = payload.get("step_count", 0)
        if not isinstance(step_count, int):
            raise ValueError("Observation 'step_count' must be an int.")

        sim_time_value = payload["sim_time"]
        if not isinstance(sim_time_value, (int, float)):
            raise ValueError("Observation 'sim_time' must be a float-like number.")
        sim_time = float(sim_time_value)

        joint_positions = _validate_numeric_vector(
            payload["joint_positions"],
            key="joint_positions",
        )
        joint_velocities = _validate_numeric_vector(
            payload["joint_velocities"],
            key="joint_velocities",
        )
        if len(joint_positions) != len(joint_velocities):
            raise ValueError(
                "Observation joint_positions and joint_velocities must have the same length."
            )

        end_effector_pose = payload["end_effector_pose"]
        if not isinstance(end_effector_pose, dict):
            raise ValueError("Observation 'end_effector_pose' must be a dict.")
        if "position" not in end_effector_pose or "orientation" not in end_effector_pose:
            raise ValueError(
                "Observation 'end_effector_pose' must contain 'position' and 'orientation'."
            )
        position = _validate_numeric_vector(
            end_effector_pose["position"],
            key="end_effector_pose.position",
            expected_length=3,
        )
        orientation = _validate_numeric_vector(
            end_effector_pose["orientation"],
            key="end_effector_pose.orientation",
            expected_length=4,
        )
        return cls(
            step_count=step_count,
            sim_time=sim_time,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            end_effector_pose={
                "position": position,
                "orientation": orientation,
            },
        )


def _validate_numeric_vector(
    value: Any,
    *,
    key: str,
    expected_length: int | None = None,
) -> list[float]:
    """Validate a numeric vector and normalize it to a float list."""
    if not isinstance(value, list):
        raise ValueError(f"Observation '{key}' must be a list of numbers.")
    normalized: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Observation '{key}' must contain only numeric values.")
        normalized.append(float(item))
    if expected_length is not None and len(normalized) != expected_length:
        raise ValueError(
            f"Observation '{key}' must have length {expected_length}."
        )
    return normalized
