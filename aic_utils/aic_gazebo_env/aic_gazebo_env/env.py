"""Gymnasium-like public environment API for training-only Gazebo integration."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .runtime import FakeRuntime, Runtime


class GazeboEnv:
    """Stub Gazebo environment with a Gymnasium-like public API.

    The implementation is intentionally minimal for early milestones. It does
    not launch or talk to a simulator yet. Instead it depends only on the
    runtime abstraction so callers can integrate against the public Python API
    before backend work begins.
    """

    _PREFERRED_ACTION_SPEC: dict[str, Any] = {
        "format": "tracked_source_delta",
        "robot_adapter": {
            "format": "ee_delta_action",
            "field": "ee_delta_action",
            "required_any_of": ["position_delta", "orientation_delta"],
            "optional_fields": {
                "position_delta": {
                    "type": "list[float]",
                    "length": 3,
                    "description": "End-effector/world-frame translation delta [dx, dy, dz].",
                },
                "orientation_delta": {
                    "type": "list[float]",
                    "length": 4,
                    "description": "Quaternion pose delta [x, y, z, w] composed against the current tracked-source orientation.",
                },
                "frame": {
                    "type": "str",
                    "default": "world",
                    "allowed_values": ["world", "local"],
                    "description": "Reference frame for the translation delta. 'local' rotates the delta by the current tracked-source orientation before bridge handoff.",
                },
                "max_position_delta_norm": {
                    "type": "float",
                    "minimum": 0.0,
                    "description": "Optional world/local translation-delta norm cap applied before bridge handoff.",
                },
            },
            "normalizes_to": "tracked_source_delta",
        },
        "required_any_of": ["position_delta", "orientation_delta"],
        "optional_fields": {
            "position_delta": {
                "type": "list[float]",
                "length": 3,
                "description": "Tracked-source Cartesian translation delta [dx, dy, dz].",
            },
            "orientation_delta": {
                "type": "list[float]",
                "length": 4,
                "description": "Tracked-source quaternion delta [x, y, z, w].",
            },
            "multi_step": {
                "type": "int",
                "default": 1,
                "minimum": 1,
                "description": "Number of world-advance steps to apply after the action.",
            },
        },
        "normalizes_to": {
            "policy_action": {
                "position_delta": "<position_delta if provided>",
                "orientation_delta": "<orientation_delta if provided>",
            },
            "multi_step": "<multi_step defaulting to 1>",
        },
    }
    _STABLE_OBSERVATION_SPEC: dict[str, Any] = {
        "format": "structured_real_observation",
        "flattened_view": {
            "format": "tracked_pair_numeric_vector",
            "length": 18,
            "field_order": [
                "step_count",
                "entity_count",
                "tracked_entity_pair.relative_position[0]",
                "tracked_entity_pair.relative_position[1]",
                "tracked_entity_pair.relative_position[2]",
                "tracked_entity_pair.distance",
                "tracked_entity_pair.source_orientation[0]",
                "tracked_entity_pair.source_orientation[1]",
                "tracked_entity_pair.source_orientation[2]",
                "tracked_entity_pair.source_orientation[3]",
                "tracked_entity_pair.target_orientation[0]",
                "tracked_entity_pair.target_orientation[1]",
                "tracked_entity_pair.target_orientation[2]",
                "tracked_entity_pair.target_orientation[3]",
                "tracked_entity_pair.orientation_error",
                "tracked_entity_pair.distance_success",
                "tracked_entity_pair.orientation_success",
                "tracked_entity_pair.success",
            ],
            "bool_encoding": "0.0_or_1.0",
        },
        "stable_top_level_fields": {
            "world_name": {
                "type": "str",
                "description": "Gazebo world name for the current runtime.",
            },
            "step_count": {
                "type": "int",
                "minimum": 0,
                "description": "Observed world step count.",
            },
            "entity_count": {
                "type": "int",
                "minimum": 0,
                "description": "Number of parsed entities in the observation.",
            },
            "entity_names": {
                "type": "list[str]",
                "description": "Parsed entity names when available.",
            },
            "task_geometry": {
                "type": "dict",
                "description": "Derived tracked-pair geometry and success fields.",
            },
        },
        "stable_nested_fields": {
            "entities_by_name.<source_entity>.pose.position": {
                "type": "list[float]",
                "length": 3,
                "description": "Tracked source position when the named entity is present.",
            },
            "entities_by_name.<source_entity>.pose.orientation": {
                "type": "list[float]",
                "length": 4,
                "description": "Tracked source orientation when the named entity is present.",
            },
            "entities_by_name.<target_entity>.pose.position": {
                "type": "list[float]",
                "length": 3,
                "description": "Tracked target position when the named entity is present.",
            },
            "entities_by_name.<target_entity>.pose.orientation": {
                "type": "list[float]",
                "length": 4,
                "description": "Tracked target orientation when the named entity is present.",
            },
            "task_geometry.tracked_entity_pair.relative_position": {
                "type": "list[float]",
                "length": 3,
                "description": "Target minus source translation.",
            },
            "task_geometry.tracked_entity_pair.distance": {
                "type": "float",
                "minimum": 0.0,
                "description": "Euclidean distance between tracked source and target.",
            },
            "task_geometry.tracked_entity_pair.source_orientation": {
                "type": "list[float]",
                "length": 4,
                "description": "Tracked source quaternion.",
            },
            "task_geometry.tracked_entity_pair.target_orientation": {
                "type": "list[float]",
                "length": 4,
                "description": "Tracked target quaternion.",
            },
            "task_geometry.tracked_entity_pair.orientation_error": {
                "type": "float",
                "minimum": 0.0,
                "description": "Quaternion angular distance in radians.",
            },
            "task_geometry.tracked_entity_pair.distance_success": {
                "type": "bool",
                "description": "Whether the tracked distance threshold is satisfied.",
            },
            "task_geometry.tracked_entity_pair.orientation_success": {
                "type": "bool",
                "description": "Whether the tracked orientation threshold is satisfied or disabled.",
            },
            "task_geometry.tracked_entity_pair.success": {
                "type": "bool",
                "description": "Combined tracked-pair success flag.",
            },
        },
        "gymnasium_space_fields": {
            "step_count": {
                "type": "int",
                "shape": (1,),
            },
            "entity_count": {
                "type": "int",
                "shape": (1,),
            },
            "tracked_entity_pair.relative_position": {
                "type": "float",
                "shape": (3,),
            },
            "tracked_entity_pair.distance": {
                "type": "float",
                "shape": (1,),
            },
            "tracked_entity_pair.source_orientation": {
                "type": "float",
                "shape": (4,),
            },
            "tracked_entity_pair.target_orientation": {
                "type": "float",
                "shape": (4,),
            },
            "tracked_entity_pair.orientation_error": {
                "type": "float",
                "shape": (1,),
            },
            "tracked_entity_pair.distance_success": {
                "type": "bool",
                "shape": (1,),
            },
            "tracked_entity_pair.orientation_success": {
                "type": "bool",
                "shape": (1,),
            },
            "tracked_entity_pair.success": {
                "type": "bool",
                "shape": (1,),
            },
        },
    }

    def __init__(self, runtime: Runtime | None = None) -> None:
        """Initialize the environment with an optional runtime implementation."""
        self._runtime: Runtime = runtime or FakeRuntime()
        self._runtime.start()

    @property
    def action_spec(self) -> dict[str, Any]:
        """Return the preferred env-level tracked-source action contract."""
        return deepcopy(self._PREFERRED_ACTION_SPEC)

    @property
    def observation_spec(self) -> dict[str, Any]:
        """Return the stable structured observation contract for RL-style use."""
        return deepcopy(self._STABLE_OBSERVATION_SPEC)

    def flatten_observation(self, observation: dict[str, Any]) -> list[float]:
        """Flatten the stable tracked-pair observation subset into a numeric vector."""
        tracked_pair = observation["task_geometry"]["tracked_entity_pair"]
        relative_position = tracked_pair["relative_position"]
        source_orientation = tracked_pair["source_orientation"]
        target_orientation = tracked_pair["target_orientation"]
        return [
            float(observation["step_count"]),
            float(observation["entity_count"]),
            float(relative_position[0]),
            float(relative_position[1]),
            float(relative_position[2]),
            float(tracked_pair["distance"]),
            float(source_orientation[0]),
            float(source_orientation[1]),
            float(source_orientation[2]),
            float(source_orientation[3]),
            float(target_orientation[0]),
            float(target_orientation[1]),
            float(target_orientation[2]),
            float(target_orientation[3]),
            float(tracked_pair["orientation_error"]),
            1.0 if tracked_pair["distance_success"] else 0.0,
            1.0 if tracked_pair["orientation_success"] else 0.0,
            1.0 if tracked_pair["success"] else 0.0,
        ]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and return `(observation, info)`.

        Args:
            seed: Optional deterministic seed for future backends.
            options: Optional reset parameters for future backends.

        Returns:
            A tuple containing a dict observation and a dict info payload.
        """
        return self._runtime.reset(seed=seed, options=options)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action and return Gymnasium-style step outputs.

        Args:
            action: Env-facing action payload.
                Preferred real-runtime format:
                `{"position_delta": [dx, dy, dz],`
                ` "orientation_delta": [x, y, z, w], "multi_step": N}`.
                Robot-relevant adapter format also supported:
                `{"ee_delta_action": {"position_delta": [dx, dy, dz],`
                ` "orientation_delta": [x, y, z, w], "frame": "world|local"},`
                ` "multi_step": N}`.
                At least one of `position_delta` or `orientation_delta` must
                be present. `multi_step` is optional and defaults to `1`.
                This cleaner tracked-source action is normalized internally to
                `{"policy_action": ...}` before runtime handoff.
                The canonical bridge-level policy form
                `{"policy_action": {...}, "multi_step": N}` remains supported.
                Older bridge-level action payloads are still forwarded for
                backward compatibility.

        Returns:
            `(observation, reward, terminated, truncated, info)`.
        """
        normalized_action = self._normalize_action(action)
        if isinstance(self._runtime, FakeRuntime) and "command" in action:
            command = action["command"]
            if not isinstance(command, list):
                raise ValueError("Action 'command' must be a list of numbers.")
            if any(not isinstance(item, (int, float)) for item in command):
                raise ValueError("Action 'command' must contain only numeric values.")
            self._runtime.last_options["_last_action_cache"] = [float(item) for item in command]
        return self._runtime.step(normalized_action)

    def _normalize_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validate the public env-facing action surface before runtime handoff."""
        if not isinstance(action, dict):
            raise ValueError("Action must be a dict.")

        normalized_action = dict(action)
        ee_delta_action = normalized_action.get("ee_delta_action")
        joint_position_delta = normalized_action.get("joint_position_delta")
        has_position_delta = "position_delta" in normalized_action
        has_orientation_delta = "orientation_delta" in normalized_action
        policy_action = normalized_action.get("policy_action")
        if policy_action is not None and not isinstance(policy_action, dict):
            raise ValueError("Action 'policy_action' must be a dict.")
        if ee_delta_action is not None and not isinstance(ee_delta_action, dict):
            raise ValueError("Action 'ee_delta_action' must be a dict.")
        if joint_position_delta is not None and not isinstance(joint_position_delta, list):
            raise ValueError("Action 'joint_position_delta' must be a list of numbers.")
        if ee_delta_action is not None and (policy_action is not None or has_position_delta or has_orientation_delta):
            raise ValueError(
                "Action must use either 'ee_delta_action', top-level tracked-source deltas, or 'policy_action', not multiple action forms."
            )
        if joint_position_delta is not None and (
            ee_delta_action is not None
            or policy_action is not None
            or has_position_delta
            or has_orientation_delta
        ):
            raise ValueError(
                "Action must use either 'joint_position_delta', 'ee_delta_action', top-level tracked-source deltas, or 'policy_action', not multiple action forms."
            )
        if ee_delta_action is not None:
            normalized_action.pop("ee_delta_action")
            ee_frame = ee_delta_action.get("frame", "world")
            if ee_frame not in {"world", "local"}:
                raise ValueError("Action 'ee_delta_action.frame' must be 'world' or 'local'.")
            if "position_delta" in ee_delta_action:
                position_delta = self._validate_numeric_vector(
                    name="ee_delta_action.position_delta",
                    value=ee_delta_action["position_delta"],
                    expected_length=3,
                )
                max_position_delta_norm = ee_delta_action.get("max_position_delta_norm")
                if max_position_delta_norm is not None:
                    if not isinstance(max_position_delta_norm, (int, float)) or max_position_delta_norm < 0.0:
                        raise ValueError(
                            "Action 'ee_delta_action.max_position_delta_norm' must be a non-negative number."
                        )
                    position_delta = self._clip_vector_norm(
                        value=position_delta,
                        max_norm=float(max_position_delta_norm),
                    )
                if ee_frame == "local":
                    position_delta = self._rotate_local_position_delta_to_world(position_delta)
                normalized_action["position_delta"] = position_delta
            if "orientation_delta" in ee_delta_action:
                normalized_action["orientation_delta"] = self._normalize_quaternion_delta(
                    self._validate_numeric_vector(
                        name="ee_delta_action.orientation_delta",
                        value=ee_delta_action["orientation_delta"],
                        expected_length=4,
                    )
                )
            if "position_delta" not in ee_delta_action and "orientation_delta" not in ee_delta_action:
                raise ValueError(
                    "Action 'ee_delta_action' must include at least one of 'position_delta' or 'orientation_delta'."
                )
            has_position_delta = "position_delta" in normalized_action
            has_orientation_delta = "orientation_delta" in normalized_action
        if joint_position_delta is not None:
            normalized_action["joint_position_delta"] = self._validate_numeric_vector(
                name="joint_position_delta",
                value=joint_position_delta,
                expected_length=6,
            )
            multi_step = normalized_action.get("multi_step", 1)
            if not isinstance(multi_step, int) or multi_step < 1:
                raise ValueError("Action 'multi_step' must be a positive integer.")
            normalized_action["multi_step"] = multi_step
            return normalized_action
        if policy_action is not None and (has_position_delta or has_orientation_delta):
            raise ValueError(
                "Action must use either top-level tracked-source deltas or 'policy_action', not both."
            )
        if has_position_delta or has_orientation_delta:
            cleaner_policy_action: dict[str, Any] = {}
            if has_position_delta:
                position_delta = normalized_action.pop("position_delta")
                cleaner_policy_action["position_delta"] = self._validate_numeric_vector(
                    name="position_delta",
                    value=position_delta,
                    expected_length=3,
                )
            if has_orientation_delta:
                orientation_delta = normalized_action.pop("orientation_delta")
                if ee_delta_action is None:
                    cleaner_policy_action["orientation_delta"] = self._validate_numeric_vector(
                        name="orientation_delta",
                        value=orientation_delta,
                        expected_length=4,
                    )
                else:
                    cleaner_policy_action["orientation_delta"] = orientation_delta
            multi_step = normalized_action.get("multi_step", 1)
            if not isinstance(multi_step, int) or multi_step < 1:
                raise ValueError("Action 'multi_step' must be a positive integer.")
            normalized_action["multi_step"] = multi_step
            normalized_action["policy_action"] = cleaner_policy_action

        return normalized_action

    def _validate_numeric_vector(
        self,
        *,
        name: str,
        value: Any,
        expected_length: int,
    ) -> list[float]:
        """Validate a fixed-length numeric vector for the preferred action path."""
        if not isinstance(value, list):
            raise ValueError(f"Action '{name}' must be a list of numbers.")
        if len(value) != expected_length:
            raise ValueError(
                f"Action '{name}' must have length {expected_length} (exactly {expected_length} values) and contain only numeric values."
            )
        if any(not isinstance(item, (int, float)) for item in value):
            raise ValueError(f"Action '{name}' must contain only numeric values.")
        return [float(item) for item in value]

    def _normalize_quaternion_delta(self, value: list[float]) -> list[float]:
        """Normalize a quaternion delta for robot-style pose-delta actions."""
        magnitude = sum(item * item for item in value) ** 0.5
        if magnitude == 0.0:
            raise ValueError("Action 'ee_delta_action.orientation_delta' must not be the zero quaternion.")
        return [item / magnitude for item in value]

    def _clip_vector_norm(self, *, value: list[float], max_norm: float) -> list[float]:
        """Clip a vector to a maximum Euclidean norm while preserving direction."""
        magnitude = math.sqrt(sum(item * item for item in value))
        if magnitude == 0.0 or magnitude <= max_norm:
            return [self._canonicalize_float(item) for item in value]
        scale = max_norm / magnitude
        return [self._canonicalize_float(item * scale) for item in value]

    def _rotate_local_position_delta_to_world(self, value: list[float]) -> list[float]:
        """Rotate a local-frame tracked-source delta into the world frame."""
        source_orientation = self._get_tracked_source_orientation()
        return self._rotate_vector_by_quaternion(value, source_orientation)

    def _get_tracked_source_orientation(self) -> list[float]:
        """Read the current tracked-source orientation from the runtime observation."""
        observation, _ = self._runtime.get_observation()
        task_geometry = observation.get("task_geometry")
        if isinstance(task_geometry, dict):
            tracked_pair = task_geometry.get("tracked_entity_pair")
            if isinstance(tracked_pair, dict):
                orientation = tracked_pair.get("source_orientation")
                if isinstance(orientation, list) and len(orientation) == 4:
                    return self._normalize_quaternion_delta(
                        self._validate_numeric_vector(
                            name="tracked_entity_pair.source_orientation",
                            value=orientation,
                            expected_length=4,
                        )
                    )
        raise ValueError(
            "Action 'ee_delta_action.frame=local' requires the runtime observation to expose 'task_geometry.tracked_entity_pair.source_orientation'."
        )

    def _rotate_vector_by_quaternion(self, value: list[float], quaternion: list[float]) -> list[float]:
        """Rotate a 3D vector by a unit quaternion."""
        vector_quaternion = [value[0], value[1], value[2], 0.0]
        rotated = self._quaternion_multiply(
            self._quaternion_multiply(quaternion, vector_quaternion),
            self._quaternion_conjugate(quaternion),
        )
        return [self._canonicalize_float(rotated[0]), self._canonicalize_float(rotated[1]), self._canonicalize_float(rotated[2])]

    def _quaternion_multiply(self, left: list[float], right: list[float]) -> list[float]:
        """Multiply two quaternions in `[x, y, z, w]` format."""
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]

    def _quaternion_conjugate(self, quaternion: list[float]) -> list[float]:
        """Return the conjugate of a quaternion in `[x, y, z, w]` format."""
        return [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]]

    def _canonicalize_float(self, value: float) -> float:
        """Round away tiny floating-point noise for stable action normalization."""
        if abs(value) < 1e-12:
            return 0.0
        nearest_integer = round(value)
        if abs(value - nearest_integer) < 1e-12:
            return float(nearest_integer)
        return value

    def close(self) -> None:
        """Close the environment and stop the runtime."""
        self._runtime.stop()


class MinimalTaskEnv(GazeboEnv):
    """Convenience env for the built-in minimal single-object task."""

    def __init__(self, runtime: Runtime | None = None) -> None:
        """Initialize the minimal task environment."""
        super().__init__(runtime=runtime)
