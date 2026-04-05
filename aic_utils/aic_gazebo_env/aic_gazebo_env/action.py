"""Action schema and validation for the training-only Gazebo environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ActionDict = dict[str, Any]


@dataclass(frozen=True)
class Action:
    """Validated joint position delta action.

    The canonical action space is `joint_position_delta`, represented as one
    delta per controlled joint. Values are clipped by the backend before being
    applied over a fixed number of substeps.
    """

    joint_position_delta: list[float]

    def to_dict(self) -> ActionDict:
        """Convert the action to the public dict form."""
        return {"joint_position_delta": list(self.joint_position_delta)}

    @classmethod
    def from_dict(
        cls,
        payload: ActionDict,
        *,
        expected_length: int,
    ) -> "Action":
        """Validate and normalize an action payload.

        For compatibility with earlier milestones, `command` is accepted as an
        alias of `joint_position_delta`.
        """
        key = "joint_position_delta"
        if key not in payload and "command" in payload:
            key = "command"
        if key not in payload:
            raise ValueError(
                "Action must contain 'joint_position_delta' or 'command' with one value per joint."
            )
        value = payload[key]
        if not isinstance(value, list):
            raise ValueError(
                f"Action '{key}' must be a list of numbers."
            )
        if key == "command" and len(value) > expected_length:
            raise ValueError(
                f"Action 'command' must have length at most {expected_length}."
            )
        if key == "joint_position_delta" and len(value) != expected_length:
            raise ValueError(
                f"Action 'joint_position_delta' must have length {expected_length}."
            )
        normalized: list[float] = []
        for item in value:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    f"Action '{key}' must contain only numeric values."
                )
            normalized.append(float(item))
        if key == "command" and len(normalized) < expected_length:
            normalized.extend([0.0] * (expected_length - len(normalized)))
        return cls(joint_position_delta=normalized)

    def clipped(self, *, max_abs_delta: float) -> "Action":
        """Return a clipped copy of the action."""
        return Action(
            joint_position_delta=[
                max(-max_abs_delta, min(max_abs_delta, value))
                for value in self.joint_position_delta
            ]
        )
