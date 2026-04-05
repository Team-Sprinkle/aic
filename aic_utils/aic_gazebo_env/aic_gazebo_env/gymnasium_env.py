"""Thin Gymnasium-compatible wrapper around the public Gazebo env."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .env import GazeboEnv
from .runtime import Runtime

try:
    import gymnasium as gym
    from gymnasium import spaces
    import numpy as np

    GYMNASIUM_AVAILABLE = True
    GymEnvBase = gym.Env[Any, Any]
except ImportError:  # pragma: no cover
    GYMNASIUM_AVAILABLE = False

    class GymEnvBase:
        """Fallback base class when Gymnasium is not installed."""

        metadata: dict[str, Any] = {}

    @dataclass
    class _FallbackBox:
        low: Any
        high: Any
        shape: tuple[int, ...]
        dtype: Any

    @dataclass
    class _FallbackDict:
        spaces: dict[str, Any]

    class spaces:  # type: ignore[no-redef]
        """Minimal fallback spaces namespace."""

        Box = _FallbackBox
        Dict = _FallbackDict


class GymnasiumGazeboEnv(GymEnvBase):
    """Thin wrapper exposing the current Gazebo env through a Gymnasium-like API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: GazeboEnv | None = None,
        *,
        runtime: Runtime | None = None,
        flatten_observation: bool = False,
    ) -> None:
        self._env = env or GazeboEnv(runtime=runtime)
        self._flatten_observation = flatten_observation
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    @property
    def unwrapped_env(self) -> GazeboEnv:
        """Return the underlying Gazebo env."""
        return self._env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Delegate reset to the underlying env."""
        observation, info = self._env.reset(seed=seed, options=options)
        if not self._flatten_observation:
            return observation, info
        flattened_observation = self._env.flatten_observation(observation)
        flattened_info = dict(info)
        flattened_info["raw_observation"] = observation
        return flattened_observation, flattened_info

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Delegate step to the underlying env."""
        observation, reward, terminated, truncated, info = self._env.step(action)
        if not self._flatten_observation:
            return observation, reward, terminated, truncated, info
        flattened_observation = self._env.flatten_observation(observation)
        flattened_info = dict(info)
        flattened_info["raw_observation"] = observation
        return flattened_observation, reward, terminated, truncated, flattened_info

    def close(self) -> None:
        """Close the underlying env."""
        self._env.close()

    def _build_action_space(self) -> Any:
        """Build an action-space descriptor from the preferred env contract."""
        action_spec = self._env.action_spec
        position_spec = action_spec["optional_fields"]["position_delta"]
        orientation_spec = action_spec["optional_fields"]["orientation_delta"]
        multi_step_spec = action_spec["optional_fields"]["multi_step"]
        return spaces.Dict(
            {
                "position_delta": spaces.Box(
                    low=-math.inf,
                    high=math.inf,
                    shape=(position_spec["length"],),
                    dtype="float32",
                ),
                "orientation_delta": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(orientation_spec["length"],),
                    dtype="float32",
                ),
                "multi_step": spaces.Box(
                    low=(multi_step_spec["minimum"],),
                    high=(2147483647,),
                    shape=(1,),
                    dtype="int32",
                ),
            }
        )

    def _build_observation_space(self) -> Any:
        """Build a partial observation-space descriptor for stable numeric fields."""
        if self._flatten_observation:
            flattened_spec = self._env.observation_spec["flattened_view"]
            return spaces.Box(
                low=-math.inf,
                high=math.inf,
                shape=(flattened_spec["length"],),
                dtype="float32",
            )
        observation_spec = self._env.observation_spec
        gym_fields = observation_spec["gymnasium_space_fields"]
        return spaces.Dict(
            {
                "step_count": spaces.Box(
                    low=(0,),
                    high=(2147483647,),
                    shape=gym_fields["step_count"]["shape"],
                    dtype="int32",
                ),
                "entity_count": spaces.Box(
                    low=(0,),
                    high=(2147483647,),
                    shape=gym_fields["entity_count"]["shape"],
                    dtype="int32",
                ),
                "tracked_entity_pair": spaces.Dict(
                    {
                        "relative_position": spaces.Box(
                            low=-math.inf,
                            high=math.inf,
                            shape=gym_fields["tracked_entity_pair.relative_position"]["shape"],
                            dtype="float32",
                        ),
                        "distance": spaces.Box(
                            low=(0.0,),
                            high=(math.inf,),
                            shape=gym_fields["tracked_entity_pair.distance"]["shape"],
                            dtype="float32",
                        ),
                        "source_orientation": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=gym_fields["tracked_entity_pair.source_orientation"]["shape"],
                            dtype="float32",
                        ),
                        "target_orientation": spaces.Box(
                            low=-1.0,
                            high=1.0,
                            shape=gym_fields["tracked_entity_pair.target_orientation"]["shape"],
                            dtype="float32",
                        ),
                        "orientation_error": spaces.Box(
                            low=(0.0,),
                            high=(math.pi,),
                            shape=gym_fields["tracked_entity_pair.orientation_error"]["shape"],
                            dtype="float32",
                        ),
                        "distance_success": spaces.Box(
                            low=(0,),
                            high=(1,),
                            shape=gym_fields["tracked_entity_pair.distance_success"]["shape"],
                            dtype="int32",
                        ),
                        "orientation_success": spaces.Box(
                            low=(0,),
                            high=(1,),
                            shape=gym_fields["tracked_entity_pair.orientation_success"]["shape"],
                            dtype="int32",
                        ),
                        "success": spaces.Box(
                            low=(0,),
                            high=(1,),
                            shape=gym_fields["tracked_entity_pair.success"]["shape"],
                            dtype="int32",
                        ),
                    }
                ),
            }
        )
