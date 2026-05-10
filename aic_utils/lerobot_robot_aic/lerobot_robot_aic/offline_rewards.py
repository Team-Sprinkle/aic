"""Offline dense reward shaping for AIC LeRobot datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


OfflineRewardObjective = Literal["insertion", "near_gate"]


@dataclass(frozen=True)
class OfflineRewardConfig:
    objective: OfflineRewardObjective = "insertion"
    distance_std_m: float = 0.05
    close_sigma_m: float = 0.01
    orientation_std_rad: float = 0.25
    progress_weight: float = 1.0
    distance_weight: float = 0.5
    close_weight: float = 0.3
    orientation_weight: float = 0.05
    force_delta_penalty_weight: float = 0.01
    terminal_bonus: float = 1.0
    near_gate_distance_threshold_m: float = 0.006
    insertion_success_distance_m: float = 0.01

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_geometry_value(value: Any, key: str) -> float | None:
    if not isinstance(value, dict) or key not in value:
        return None
    raw = value[key]
    if isinstance(raw, (list, tuple, np.ndarray)):
        if len(raw) == 0:
            return None
        raw = raw[0]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _numeric_column(df: pd.DataFrame, *names: str) -> np.ndarray | None:
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=np.float32)
    if "score_geometry" in df.columns:
        for name in names:
            values = [_score_geometry_value(value, name) for value in df["score_geometry"]]
            if any(value is not None for value in values):
                filled = [np.nan if value is None else value for value in values]
                return np.asarray(filled, dtype=np.float32)
    return None


def _finite_or_raise(values: np.ndarray | None, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(
            f"Cannot materialize offline rewards: missing required geometry field {name!r}. "
            "Use datasets with score_geometry/distance columns or add a geometry export step."
        )
    if np.isnan(values).any():
        raise ValueError(f"Geometry field {name!r} contains missing values")
    return values.astype(np.float32)


def _terminal_success(distance: np.ndarray, episode: np.ndarray, threshold: float) -> np.ndarray:
    success = np.zeros_like(distance, dtype=np.float32)
    last_indices = pd.Series(np.arange(len(distance))).groupby(episode, sort=False).tail(1).to_numpy()
    success[last_indices] = (distance[last_indices] <= threshold).astype(np.float32)
    return success


def dense_offline_rewards(df: pd.DataFrame, config: OfflineRewardConfig | None = None) -> np.ndarray:
    """Compute dense rewards from task-geometry columns already recorded in a dataset.

    This intentionally does not use Isaac's random command pose reward. Required
    distance fields must describe the real insertion/near-gate objective.
    """
    cfg = config or OfflineRewardConfig()
    if "episode_index" not in df.columns:
        raise ValueError("offline reward materialization requires episode_index")
    episode = df["episode_index"].to_numpy(dtype=np.int64)
    if cfg.objective == "near_gate":
        distance = _finite_or_raise(
            _numeric_column(df, "distance_to_entrance", "near_gate_distance_m", "distance_to_target"),
            "near_gate distance",
        )
        success_threshold = cfg.near_gate_distance_threshold_m
    elif cfg.objective == "insertion":
        distance = _finite_or_raise(
            _numeric_column(df, "distance_to_target", "insertion_distance_m"),
            "distance_to_target",
        )
        success_threshold = cfg.insertion_success_distance_m
    else:
        raise ValueError(f"Unsupported reward objective: {cfg.objective!r}")

    next_distance = distance.copy()
    same_episode_next = episode[:-1] == episode[1:]
    next_indices = np.where(same_episode_next)[0]
    next_distance[next_indices] = distance[next_indices + 1]
    progress = np.clip(distance - next_distance, -0.05, 0.05)
    distance_reward = 1.0 - np.tanh(distance / cfg.distance_std_m)
    close_reward = np.exp(-(distance**2) / (cfg.close_sigma_m**2))

    orientation = _numeric_column(df, "orientation_error", "orientation_error_rad")
    if orientation is None:
        orientation_reward = np.zeros_like(distance, dtype=np.float32)
    else:
        orientation_reward = 1.0 - np.tanh(orientation.astype(np.float32) / cfg.orientation_std_rad)

    force_delta = _numeric_column(df, "force_delta_norm", "wrist_force_delta_norm")
    force_penalty = np.zeros_like(distance, dtype=np.float32) if force_delta is None else -np.maximum(force_delta, 0.0)
    terminal = _terminal_success(distance, episode, success_threshold)

    reward = (
        cfg.progress_weight * progress
        + cfg.distance_weight * distance_reward
        + cfg.close_weight * close_reward
        + cfg.orientation_weight * orientation_reward
        + cfg.force_delta_penalty_weight * force_penalty
        + cfg.terminal_bonus * terminal
    )
    return reward.astype(np.float32)
