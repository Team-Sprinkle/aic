"""Causal force-threshold memory features shared by datasets and policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


FORCE_DELTA_THRESHOLDS_N = (1.0, 3.0, 5.0, 7.0)


def _threshold_label(threshold: float) -> str:
    if float(threshold).is_integer():
        return str(int(threshold))
    return str(threshold).replace(".", "p")


CONTACT_RECOVERY_FEATURE_NAMES = [
    name
    for threshold in FORCE_DELTA_THRESHOLDS_N
    for prefix in (f"force_thresh_{_threshold_label(threshold)}",)
    for name in (
        f"{prefix}.time_since_first_sec",
        f"{prefix}.first_delta.x",
        f"{prefix}.first_delta.y",
        f"{prefix}.first_delta.z",
        f"{prefix}.first_delta_norm",
        f"{prefix}.time_since_latest_sec",
        f"{prefix}.latest_delta.x",
        f"{prefix}.latest_delta.y",
        f"{prefix}.latest_delta.z",
        f"{prefix}.latest_delta_norm",
    )
]

CONTACT_RECOVERY_FEATURE_DIM = len(CONTACT_RECOVERY_FEATURE_NAMES)


@dataclass(frozen=True)
class ContactRecoveryFeatureConfig:
    force_delta_thresholds_n: tuple[float, ...] = FORCE_DELTA_THRESHOLDS_N

    def __post_init__(self) -> None:
        thresholds = tuple(float(value) for value in self.force_delta_thresholds_n)
        if not thresholds:
            raise ValueError("force_delta_thresholds_n must not be empty")
        for threshold in thresholds:
            if threshold <= 0.0 or not math.isfinite(threshold):
                raise ValueError(f"force delta thresholds must be finite positive values, got {threshold}")
        if thresholds != FORCE_DELTA_THRESHOLDS_N:
            raise ValueError(
                "force_delta_thresholds_n must match the fixed feature schema "
                f"{FORCE_DELTA_THRESHOLDS_N}, got {thresholds}"
            )
        object.__setattr__(self, "force_delta_thresholds_n", thresholds)


def _as_vec3(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"expected 3-vector, got shape {arr.shape}")
    return arr


class ContactRecoveryFeatureComputer:
    """Online-causal one-step force-delta threshold memory."""

    def __init__(self, config: ContactRecoveryFeatureConfig | None = None):
        self.config = config or ContactRecoveryFeatureConfig()
        self.reset()

    def reset(self) -> None:
        self._previous_force: np.ndarray | None = None
        self._first_time = {threshold: None for threshold in self.config.force_delta_thresholds_n}
        self._first_delta = {
            threshold: np.zeros(3, dtype=np.float64) for threshold in self.config.force_delta_thresholds_n
        }
        self._first_norm = {threshold: 0.0 for threshold in self.config.force_delta_thresholds_n}
        self._latest_time = {threshold: None for threshold in self.config.force_delta_thresholds_n}
        self._latest_delta = {
            threshold: np.zeros(3, dtype=np.float64) for threshold in self.config.force_delta_thresholds_n
        }
        self._latest_norm = {threshold: 0.0 for threshold in self.config.force_delta_thresholds_n}

    def update(
        self,
        *,
        time_sec: float,
        tcp_position_base: Iterable[float],
        tcp_orientation_xyzw: Iterable[float],
        force: Iterable[float],
        torque: Iterable[float],
    ) -> np.ndarray:
        del tcp_position_base, tcp_orientation_xyzw, torque
        force_vec = _as_vec3(force)
        if self._previous_force is None:
            force_delta = np.zeros(3, dtype=np.float64)
        else:
            force_delta = force_vec - self._previous_force
        self._previous_force = force_vec.copy()

        force_delta_norm = float(np.linalg.norm(force_delta))
        current_time_sec = float(time_sec)
        values: list[float] = []
        for threshold in self.config.force_delta_thresholds_n:
            if force_delta_norm >= threshold:
                if self._first_time[threshold] is None:
                    self._first_time[threshold] = current_time_sec
                    self._first_delta[threshold] = force_delta.copy()
                    self._first_norm[threshold] = force_delta_norm
                self._latest_time[threshold] = current_time_sec
                self._latest_delta[threshold] = force_delta.copy()
                self._latest_norm[threshold] = force_delta_norm

            first_time = self._first_time[threshold]
            latest_time = self._latest_time[threshold]
            values.extend(
                [
                    -1.0 if first_time is None else current_time_sec - first_time,
                    *self._first_delta[threshold].tolist(),
                    self._first_norm[threshold],
                    -1.0 if latest_time is None else current_time_sec - latest_time,
                    *self._latest_delta[threshold].tolist(),
                    self._latest_norm[threshold],
                ]
            )

        return np.asarray(values, dtype=np.float32)


def _parse_thresholds(value: object) -> tuple[float, ...]:
    if value is None:
        return FORCE_DELTA_THRESHOLDS_N
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if isinstance(value, Iterable):
        return tuple(float(part) for part in value)
    raise TypeError(f"expected thresholds as comma-separated string or iterable, got {type(value)!r}")


def config_from_mapping(values: dict[str, float | int | str] | None = None) -> ContactRecoveryFeatureConfig:
    values = values or {}
    return ContactRecoveryFeatureConfig(
        force_delta_thresholds_n=_parse_thresholds(
            values.get("force_delta_thresholds_n", values.get("force_thresholds_n"))
        )
    )
