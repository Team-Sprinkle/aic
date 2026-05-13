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
    # Keep these defaults in parity with Isaac online SERL's dense target reward.
    # Wider distance kernels make the reward useful before the plug reaches the gate.
    distance_std_m: float = 0.08
    close_sigma_m: float = 0.02
    orientation_std_rad: float = 0.03
    orientation_gate_sigma_m: float = 0.012
    progress_scale_m: float = 0.01
    progress_weight: float = 0.60
    distance_weight: float = 0.60
    close_weight: float = 0.25
    orientation_weight: float = 0.10
    force_delta_penalty_weight: float = 0.25
    force_delta_threshold_n: float = 3.0
    force_delta_reference_n: float = 20.0
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


def _state_vectors(df: pd.DataFrame) -> np.ndarray | None:
    if "observation.state" not in df.columns:
        return None
    values = df["observation.state"].to_numpy()
    if len(values) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(values).astype(np.float32)


def _state_name_indices(df: pd.DataFrame) -> dict[str, int]:
    names = df.attrs.get("observation_state_names")
    if not names:
        return {}
    return {str(name): index for index, name in enumerate(names)}


def _state_tcp_error_norm(df: pd.DataFrame, suffixes: tuple[str, ...]) -> np.ndarray | None:
    indices_by_name = _state_name_indices(df)
    if not indices_by_name:
        return None
    indices = []
    for suffix in suffixes:
        index = indices_by_name.get(f"tcp_error.{suffix}")
        if index is None:
            return None
        indices.append(index)
    state = _state_vectors(df)
    if state is None:
        return None
    return np.linalg.norm(state[:, indices], axis=1).astype(np.float32)


def _state_wrist_force(df: pd.DataFrame) -> np.ndarray | None:
    indices_by_name = _state_name_indices(df)
    if not indices_by_name:
        return None
    indices = []
    for suffix in ("x", "y", "z"):
        index = indices_by_name.get(f"wrist_wrench.force.{suffix}")
        if index is None:
            return None
        indices.append(index)
    state = _state_vectors(df)
    if state is None:
        return None
    return state[:, indices].astype(np.float32)


def _force_delta_norm(df: pd.DataFrame, episode: np.ndarray) -> np.ndarray | None:
    explicit = _numeric_column(df, "force_delta_norm", "wrist_force_delta_norm")
    if explicit is not None:
        return explicit.astype(np.float32)
    force = _numeric_column(df, "wrist_force_norm", "force_norm")
    if force is not None:
        force = force.astype(np.float32)
        delta = np.zeros_like(force, dtype=np.float32)
        same_episode = episode[:-1] == episode[1:]
        indices = np.where(same_episode)[0] + 1
        delta[indices] = np.abs(force[indices] - force[indices - 1])
        return delta
    force_vec = _state_wrist_force(df)
    if force_vec is None:
        return None
    delta_vec = np.zeros_like(force_vec, dtype=np.float32)
    same_episode = episode[:-1] == episode[1:]
    indices = np.where(same_episode)[0] + 1
    delta_vec[indices] = force_vec[indices] - force_vec[indices - 1]
    return np.linalg.norm(delta_vec, axis=1).astype(np.float32)


def _force_delta_event_penalty(
    df: pd.DataFrame, episode: np.ndarray, cfg: OfflineRewardConfig
) -> tuple[np.ndarray, np.ndarray]:
    force_delta_norm = _force_delta_norm(df, episode)
    if force_delta_norm is None:
        zeros = np.zeros_like(episode, dtype=np.float32)
        return zeros, zeros
    threshold = float(cfg.force_delta_threshold_n)
    reference = max(float(cfg.force_delta_reference_n), threshold + 1e-6)
    excess = np.maximum(force_delta_norm.astype(np.float32) - threshold, 0.0) / (reference - threshold)
    penalty = -np.clip(excess, 0.0, 1.0).astype(np.float32)
    return penalty, force_delta_norm.astype(np.float32)


def _finite_or_raise(values: np.ndarray | None, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(
            f"Cannot materialize offline rewards: missing required geometry field {name!r}. "
            "Use datasets with score_geometry/distance columns, or pass observation_state_names "
            "so tcp_error distance/orientation can be derived from observation.state."
        )
    if np.isnan(values).any():
        raise ValueError(f"Geometry field {name!r} contains missing values")
    return values.astype(np.float32)


def _terminal_success(distance: np.ndarray, episode: np.ndarray, threshold: float) -> np.ndarray:
    success = np.zeros_like(distance, dtype=np.float32)
    last_indices = pd.Series(np.arange(len(distance))).groupby(episode, sort=False).tail(1).to_numpy()
    success[last_indices] = (distance[last_indices] <= threshold).astype(np.float32)
    return success


def dense_offline_reward_components(
    df: pd.DataFrame, config: OfflineRewardConfig | None = None
) -> dict[str, np.ndarray]:
    """Compute dense rewards from task-geometry columns already recorded in a dataset.

    This intentionally does not use Isaac's random command pose reward. Required
    distance fields must describe the real insertion/near-gate objective.
    """
    cfg = config or OfflineRewardConfig()
    if "episode_index" not in df.columns:
        raise ValueError("offline reward materialization requires episode_index")
    episode = df["episode_index"].to_numpy(dtype=np.int64)
    if cfg.objective == "near_gate":
        distance = _numeric_column(df, "distance_to_entrance", "near_gate_distance_m", "distance_to_target")
        if distance is None:
            distance = _state_tcp_error_norm(df, ("x", "y", "z"))
        distance = _finite_or_raise(distance, "near_gate distance")
        success_threshold = cfg.near_gate_distance_threshold_m
    elif cfg.objective == "insertion":
        distance = _numeric_column(df, "distance_to_target", "insertion_distance_m")
        if distance is None:
            distance = _state_tcp_error_norm(df, ("x", "y", "z"))
        distance = _finite_or_raise(distance, "distance_to_target")
        success_threshold = cfg.insertion_success_distance_m
    else:
        raise ValueError(f"Unsupported reward objective: {cfg.objective!r}")

    next_distance = distance.copy()
    same_episode_next = episode[:-1] == episode[1:]
    next_indices = np.where(same_episode_next)[0]
    next_distance[next_indices] = distance[next_indices + 1]
    progress = np.clip((distance - next_distance) / cfg.progress_scale_m, -1.0, 1.0)
    distance_reward = 1.0 - np.tanh(distance / cfg.distance_std_m)
    close_reward = np.exp(-(distance**2) / (cfg.close_sigma_m**2))

    orientation = _numeric_column(df, "orientation_error", "orientation_error_rad")
    if orientation is None:
        orientation = _state_tcp_error_norm(df, ("rx", "ry", "rz"))
    if orientation is None:
        orientation_reward = np.zeros_like(distance, dtype=np.float32)
    else:
        orientation_alignment = np.exp(-((orientation.astype(np.float32) / cfg.orientation_std_rad) ** 2))
        orientation_gate = np.exp(-(distance**2) / (cfg.orientation_gate_sigma_m**2))
        orientation_reward = orientation_alignment * orientation_gate

    force_penalty, force_delta_norm = _force_delta_event_penalty(df, episode, cfg)
    terminal = _terminal_success(distance, episode, success_threshold)

    components = {
        "progress": cfg.progress_weight * progress,
        "distance": cfg.distance_weight * distance_reward,
        "close": cfg.close_weight * close_reward,
        "orientation": cfg.orientation_weight * orientation_reward,
        "force_delta_penalty": cfg.force_delta_penalty_weight * force_penalty,
        "terminal": cfg.terminal_bonus * terminal,
        "force_delta_norm": force_delta_norm,
    }
    reward_component_names = (
        "progress",
        "distance",
        "close",
        "orientation",
        "force_delta_penalty",
        "terminal",
    )
    components["reward"] = sum(components[name] for name in reward_component_names)
    return {name: values.astype(np.float32) for name, values in components.items()}


def dense_offline_rewards(df: pd.DataFrame, config: OfflineRewardConfig | None = None) -> np.ndarray:
    return dense_offline_reward_components(df, config)["reward"]
