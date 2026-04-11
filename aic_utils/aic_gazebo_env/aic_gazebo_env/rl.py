"""Minimal RL-facing wrappers and helpers for the Gazebo training env."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from .env import GazeboEnv
from .gymnasium_env import GYMNASIUM_AVAILABLE, GymEnvBase, spaces
from .runtime import Runtime

try:  # pragma: no cover - optional dependency
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]
    NUMPY_AVAILABLE = False


@dataclass(frozen=True)
class StableRLEnvConfig:
    """Frozen baseline RL contract for this branch."""

    action_mode: str = "position_delta_3d"
    observation_mode: str = "flat_tracked_pair_v1"
    action_clip: float = 1.0
    max_position_delta: float = 0.01
    multi_step: int = 1
    episode_step_limit: int | None = None


@dataclass
class EpisodeMonitor:
    """Track episode statistics for training/eval logs."""

    episode_index: int = 0
    episode_reward: float = 0.0
    episode_length: int = 0
    episode_success: bool = False
    start_time_s: float = field(default_factory=perf_counter)

    def step(self, *, reward: float, success: bool) -> None:
        self.episode_reward += reward
        self.episode_length += 1
        self.episode_success = self.episode_success or success

    def finish(self) -> dict[str, Any]:
        elapsed_s = perf_counter() - self.start_time_s
        summary = {
            "episode_index": self.episode_index,
            "episode_reward": self.episode_reward,
            "episode_length": self.episode_length,
            "episode_success": self.episode_success,
            "elapsed_s": elapsed_s,
            "steps_per_second": (
                self.episode_length / elapsed_s if elapsed_s > 0.0 else None
            ),
        }
        self.episode_index += 1
        self.episode_reward = 0.0
        self.episode_length = 0
        self.episode_success = False
        self.start_time_s = perf_counter()
        return summary


class StableRLGazeboEnv(GymEnvBase):
    """Single-action-mode, flat-observation wrapper suitable for PPO baselines."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env: GazeboEnv | None = None,
        runtime: Runtime | None = None,
        config: StableRLEnvConfig | None = None,
    ) -> None:
        self._env = env or GazeboEnv(runtime=runtime)
        self._config = config or StableRLEnvConfig()
        self._episode_steps = 0
        self.monitor = EpisodeMonitor()
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

    @property
    def rl_config(self) -> StableRLEnvConfig:
        return self._config

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[list[float], dict[str, Any]]:
        self._episode_steps = 0
        observation, info = self._env.reset(seed=seed, options=options)
        flattened = self._coerce_observation(self._env.flatten_observation(observation))
        return flattened, {
            **dict(info),
            "raw_observation": observation,
            "rl_api": self._config.action_mode,
        }

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        translated_action = self._translate_action(action)
        observation, reward, terminated, truncated, info = self._env.step(translated_action)
        self._episode_steps += 1
        tracked_success = bool(
            observation.get("task_geometry", {})
            .get("tracked_entity_pair", {})
            .get("success", False)
        )
        self.monitor.step(reward=reward, success=tracked_success)
        if (
            self._config.episode_step_limit is not None
            and self._episode_steps >= self._config.episode_step_limit
        ):
            truncated = True
            info = dict(info)
            info["time_limit_reached"] = True
        flattened = self._coerce_observation(self._env.flatten_observation(observation))
        return flattened, reward, terminated, truncated, {
            **dict(info),
            "raw_observation": observation,
            "translated_action": translated_action,
            "rl_api": self._config.action_mode,
        }

    def close(self) -> None:
        self._env.close()

    def _translate_action(self, action: Any) -> dict[str, Any]:
        if isinstance(action, tuple):
            values = list(action)
        elif isinstance(action, list):
            values = list(action)
        elif NUMPY_AVAILABLE and isinstance(action, np.ndarray):
            values = action.astype(float).tolist()
        else:
            raise ValueError("StableRLGazeboEnv action must be a 3-item numeric vector.")
        if len(values) != 3 or any(not isinstance(value, (int, float)) for value in values):
            raise ValueError("StableRLGazeboEnv action must contain exactly 3 numeric values.")
        clipped = [
            max(-self._config.action_clip, min(self._config.action_clip, float(value)))
            for value in values
        ]
        scaled = [value * self._config.max_position_delta for value in clipped]
        return {
            "position_delta": scaled,
            "multi_step": self._config.multi_step,
        }

    def _build_action_space(self) -> Any:
        return spaces.Box(
            low=(-self._config.action_clip,) * 3,
            high=(self._config.action_clip,) * 3,
            shape=(3,),
            dtype=np.float32 if NUMPY_AVAILABLE else "float32",
        )

    def _build_observation_space(self) -> Any:
        flattened_spec = self._env.observation_spec["flattened_view"]
        return spaces.Box(
            low=-1.0e9,
            high=1.0e9,
            shape=(flattened_spec["length"],),
            dtype=np.float32 if NUMPY_AVAILABLE else "float32",
        )

    def _coerce_observation(self, flattened: list[float]) -> Any:
        if NUMPY_AVAILABLE:
            return np.asarray(flattened, dtype=np.float32)
        return flattened


def training_api_report(config: StableRLEnvConfig) -> dict[str, Any]:
    """Return the frozen baseline RL contract for docs and saved runs."""
    return {
        "action_mode": config.action_mode,
        "observation_mode": config.observation_mode,
        "action_shape": (3,),
        "action_range": [-config.action_clip, config.action_clip],
        "max_position_delta": config.max_position_delta,
        "multi_step": config.multi_step,
        "episode_step_limit": config.episode_step_limit,
        "flattened_observation_length": 18,
        "gymnasium_available": GYMNASIUM_AVAILABLE,
        "numpy_available": NUMPY_AVAILABLE,
    }
