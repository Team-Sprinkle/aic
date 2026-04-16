"""Rolling temporal context for teacher planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..runtime import RuntimeState
from .types import DynamicsSummary


@dataclass(frozen=True)
class TemporalFrame:
    sim_tick: int
    sim_time: float
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    plug_pose: np.ndarray
    wrench: np.ndarray
    off_limit_contact: bool
    action: np.ndarray
    image_refs: tuple[str, ...]
    image_timestamps: dict[str, float]
    image_summaries: dict[str, dict[str, Any]]


class TemporalObservationBuffer:
    """Stores a short observation/action history and computes cable-dynamics hints."""

    def __init__(
        self,
        *,
        max_frames: int = 16,
        significant_motion_threshold: float = 0.002,
        quasi_static_velocity_threshold: float = 0.01,
    ) -> None:
        self._frames: deque[TemporalFrame] = deque(maxlen=max_frames)
        self._significant_motion_threshold = float(significant_motion_threshold)
        self._quasi_static_velocity_threshold = float(quasi_static_velocity_threshold)

    def append(
        self,
        *,
        state: RuntimeState,
        action: np.ndarray,
        images: dict[str, Any] | None = None,
        image_timestamps: dict[str, float] | None = None,
        image_summaries: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        image_refs = tuple(sorted((images or {}).keys()))
        self._frames.append(
            TemporalFrame(
                sim_tick=int(state.sim_tick),
                sim_time=float(state.sim_time),
                tcp_pose=np.asarray(state.tcp_pose, dtype=np.float64).copy(),
                tcp_velocity=np.asarray(state.tcp_velocity, dtype=np.float64).copy(),
                plug_pose=np.asarray(state.plug_pose, dtype=np.float64).copy(),
                wrench=np.asarray(state.wrench, dtype=np.float64).copy(),
                off_limit_contact=bool(state.off_limit_contact),
                action=np.asarray(action, dtype=np.float64).copy(),
                image_refs=image_refs,
                image_timestamps={key: float(value) for key, value in (image_timestamps or {}).items()},
                image_summaries={key: dict(value) for key, value in (image_summaries or {}).items()},
            )
        )

    def clear(self) -> None:
        self._frames.clear()

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def frames(self) -> tuple[TemporalFrame, ...]:
        return tuple(self._frames)

    def latest(self) -> TemporalFrame:
        if not self._frames:
            raise RuntimeError("TemporalObservationBuffer is empty.")
        return self._frames[-1]

    def dynamics_summary(self) -> DynamicsSummary:
        if len(self._frames) < 2:
            return DynamicsSummary(
                plug_oscillation_magnitude=0.0,
                cable_settling_score=1.0,
                recent_motion_energy=0.0,
                quasi_static=True,
                time_since_last_significant_cable_motion=0.0,
                wrench_energy=0.0,
            )
        plug_positions = np.stack([frame.plug_pose[:3] for frame in self._frames], axis=0)
        tcp_positions = np.stack([frame.tcp_pose[:3] for frame in self._frames], axis=0)
        wrenches = np.stack([frame.wrench[:3] for frame in self._frames], axis=0)
        sim_times = np.asarray([frame.sim_time for frame in self._frames], dtype=np.float64)
        plug_deltas = np.linalg.norm(np.diff(plug_positions, axis=0), axis=1)
        tcp_deltas = np.linalg.norm(np.diff(tcp_positions, axis=0), axis=1)
        relative_displacement = plug_positions - tcp_positions
        centered_relative = relative_displacement - relative_displacement.mean(axis=0, keepdims=True)
        oscillation = float(np.sqrt(np.mean(np.sum(centered_relative * centered_relative, axis=1))))
        motion_energy = float(np.mean(plug_deltas * plug_deltas))
        wrench_energy = float(np.mean(np.sum(wrenches * wrenches, axis=1)))
        last_sig_index = -1
        for index, delta in enumerate(plug_deltas):
            if delta >= self._significant_motion_threshold:
                last_sig_index = index
        if last_sig_index < 0:
            time_since_sig = float(sim_times[-1] - sim_times[0])
        else:
            time_since_sig = float(sim_times[-1] - sim_times[last_sig_index + 1])
        recent_velocity = plug_deltas[-3:] / np.maximum(np.diff(sim_times)[-3:], 1e-6)
        quasi_static = bool(
            np.max(recent_velocity) <= self._quasi_static_velocity_threshold
            and np.max(tcp_deltas[-3:]) <= self._significant_motion_threshold
        )
        settling_score = 1.0 / (1.0 + 40.0 * motion_energy + 0.01 * wrench_energy + 20.0 * oscillation)
        return DynamicsSummary(
            plug_oscillation_magnitude=oscillation,
            cable_settling_score=float(np.clip(settling_score, 0.0, 1.0)),
            recent_motion_energy=motion_energy,
            quasi_static=quasi_static,
            time_since_last_significant_cable_motion=time_since_sig,
            wrench_energy=wrench_energy,
        )

    def compact_state(self) -> dict[str, Any]:
        latest = self.latest()
        summary = self.dynamics_summary()
        return {
            "window_size": len(self._frames),
            "latest_sim_tick": latest.sim_tick,
            "latest_sim_time": latest.sim_time,
            "latest_image_refs": list(latest.image_refs),
            "latest_image_timestamps": dict(latest.image_timestamps),
            "latest_image_summaries": dict(latest.image_summaries),
            "dynamics_summary": summary.to_dict(),
        }
