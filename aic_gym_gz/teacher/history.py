"""Rolling temporal context for teacher planning."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..runtime import RuntimeState
from .quality import (
    auxiliary_summary_quality,
    controller_state_summary,
    normalize_auxiliary_force_contact_summary,
    serialize_nested,
    summarize_auxiliary_force_contact_history,
    summarize_auxiliary_force_contact_summary,
)
from .types import DynamicsSummary


@dataclass(frozen=True)
class TemporalFrame:
    sim_tick: int
    sim_time: float
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    plug_pose: np.ndarray
    target_port_pose: np.ndarray
    wrench: np.ndarray
    wrench_timestamp: float
    off_limit_contact: bool
    action: np.ndarray
    controller_state: dict[str, Any]
    world_entities_summary: dict[str, Any]
    images: dict[str, np.ndarray]
    image_refs: tuple[str, ...]
    image_timestamps: dict[str, float]
    image_summaries: dict[str, dict[str, Any]]
    camera_info: dict[str, Any]
    signal_quality: dict[str, Any]
    score_geometry: dict[str, Any]
    auxiliary_force_contact_summary: dict[str, Any]
    auxiliary_summary_available: bool
    auxiliary_summary_status: str
    auxiliary_contact_metrics: dict[str, Any]


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
        camera_info: dict[str, Any] | None = None,
        signal_quality: dict[str, Any] | None = None,
        auxiliary_force_contact_summary: dict[str, Any] | None = None,
        auxiliary_summary_available: bool = False,
    ) -> None:
        image_refs = tuple(sorted((images or {}).keys()))
        auxiliary_summary = normalize_auxiliary_force_contact_summary(auxiliary_force_contact_summary)
        auxiliary_metrics = summarize_auxiliary_force_contact_summary(
            auxiliary_force_contact_summary=auxiliary_summary,
            auxiliary_summary_available=auxiliary_summary_available,
            current_wrench=state.wrench,
        )
        auxiliary_quality = auxiliary_summary_quality(
            auxiliary_force_contact_summary=auxiliary_summary,
            available=auxiliary_summary_available,
        )
        self._frames.append(
            TemporalFrame(
                sim_tick=int(state.sim_tick),
                sim_time=float(state.sim_time),
                tcp_pose=np.asarray(state.tcp_pose, dtype=np.float64).copy(),
                tcp_velocity=np.asarray(state.tcp_velocity, dtype=np.float64).copy(),
                plug_pose=np.asarray(state.plug_pose, dtype=np.float64).copy(),
                target_port_pose=np.asarray(state.target_port_pose, dtype=np.float64).copy(),
                wrench=np.asarray(state.wrench, dtype=np.float64).copy(),
                wrench_timestamp=float(state.wrench_timestamp),
                off_limit_contact=bool(state.off_limit_contact),
                action=np.asarray(action, dtype=np.float64).copy(),
                controller_state=_copy_mapping(state.controller_state),
                world_entities_summary=_copy_mapping(state.world_entities_summary),
                images={
                    name: np.asarray(image, dtype=np.uint8).copy()
                    for name, image in (images or {}).items()
                },
                image_refs=image_refs,
                image_timestamps={key: float(value) for key, value in (image_timestamps or {}).items()},
                image_summaries={key: dict(value) for key, value in (image_summaries or {}).items()},
                camera_info=_copy_mapping(camera_info or {}),
                signal_quality=_copy_mapping(signal_quality or {}),
                score_geometry=_copy_mapping(state.score_geometry),
                auxiliary_force_contact_summary=_copy_mapping(auxiliary_summary),
                auxiliary_summary_available=bool(auxiliary_summary_available),
                auxiliary_summary_status=str(auxiliary_quality["status"]),
                auxiliary_contact_metrics=_copy_mapping(auxiliary_metrics),
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
        return self.teacher_memory_summary()

    def current_observation_view(self) -> dict[str, Any]:
        latest = self.latest()
        return {
            "sim_tick": latest.sim_tick,
            "sim_time": latest.sim_time,
            "tcp_pose": latest.tcp_pose.astype(float).tolist(),
            "tcp_velocity": latest.tcp_velocity.astype(float).tolist(),
            "plug_pose": latest.plug_pose.astype(float).tolist(),
            "wrench": latest.wrench.astype(float).tolist(),
            "wrench_timestamp": latest.wrench_timestamp,
            "off_limit_contact": latest.off_limit_contact,
            "controller_state": controller_state_summary(latest.controller_state),
            "world_entities_summary": dict(latest.world_entities_summary),
            "images_available": bool(latest.images),
            "image_refs": list(latest.image_refs),
            "image_timestamps": dict(latest.image_timestamps),
            "image_summaries": dict(latest.image_summaries),
            "camera_info": serialize_nested(latest.camera_info),
            "signal_quality": dict(latest.signal_quality),
        }

    def teacher_memory_summary(self, *, max_items: int = 4) -> dict[str, Any]:
        latest = self.latest()
        summary = self.dynamics_summary()
        frames = list(self._frames)[-max_items:]
        official_history_summary = {
            "history_items": len(frames),
            "wrench_history": [frame.wrench.astype(float).tolist() for frame in frames],
            "wrench_timestamp_history": [float(frame.wrench_timestamp) for frame in frames],
            "tcp_velocity_history": [frame.tcp_velocity.astype(float).tolist() for frame in frames],
            "controller_state_history": [
                controller_state_summary(frame.controller_state) for frame in frames
            ],
        }
        auxiliary_history_frames = [dict(frame.auxiliary_contact_metrics) for frame in frames]
        auxiliary_history_summary = summarize_auxiliary_force_contact_history(auxiliary_history_frames)
        geometry_progress_summary = _geometry_progress_summary(frames)
        wrench_contact_trend_summary = _wrench_contact_trend_summary(frames)
        compact_signal_samples = _compact_signal_samples(frames)
        return {
            "window_size": len(self._frames),
            "history_items": len(frames),
            "latest_sim_tick": latest.sim_tick,
            "latest_sim_time": latest.sim_time,
            "latest_signal_quality": dict(latest.signal_quality),
            "action_history": [frame.action.astype(float).tolist() for frame in frames],
            "wrench_history": official_history_summary["wrench_history"],
            "wrench_timestamp_history": official_history_summary["wrench_timestamp_history"],
            "tcp_velocity_history": official_history_summary["tcp_velocity_history"],
            "controller_state_history": official_history_summary["controller_state_history"],
            "world_entities_summary": dict(latest.world_entities_summary),
            "image_ref_history": [list(frame.image_refs) for frame in frames],
            "image_timestamp_history": [dict(frame.image_timestamps) for frame in frames],
            "latest_image_summaries": dict(latest.image_summaries),
            "latest_camera_info": serialize_nested(latest.camera_info),
            "official_history_summary": official_history_summary,
            "auxiliary_force_contact_history": [
                dict(frame.auxiliary_force_contact_summary) for frame in frames
            ],
            "auxiliary_contact_metrics_history": auxiliary_history_frames,
            "auxiliary_history_summary": auxiliary_history_summary,
            "geometry_progress_summary": geometry_progress_summary,
            "wrench_contact_trend_summary": wrench_contact_trend_summary,
            "compact_signal_samples": compact_signal_samples,
            "dynamics_summary": summary.to_dict(),
        }

    def auxiliary_history_summary(self, *, max_items: int = 4) -> dict[str, Any]:
        frames = list(self._frames)[-max_items:]
        return summarize_auxiliary_force_contact_history(
            [dict(frame.auxiliary_contact_metrics) for frame in frames]
        )

    def recent_visual_frames(self, *, max_frames: int = 2) -> list[dict[str, Any]]:
        frames = [frame for frame in list(self._frames)[-max_frames:] if frame.images]
        latest_sim_time = frames[-1].sim_time if frames else None
        visual_frames: list[dict[str, Any]] = []
        for frame_index, frame in enumerate(frames):
            visual_frames.append(
                {
                    "frame_index": frame_index,
                    "sim_tick": frame.sim_tick,
                    "sim_time": frame.sim_time,
                    "age_from_latest_s": (
                        None if latest_sim_time is None else float(latest_sim_time - frame.sim_time)
                    ),
                    "age_from_latest_steps": (
                        None if not frames else int(frames[-1].sim_tick - frame.sim_tick)
                    ),
                    "images": {
                        name: image.copy()
                        for name, image in frame.images.items()
                    },
                    "image_timestamps": dict(frame.image_timestamps),
                    "image_summaries": dict(frame.image_summaries),
                }
            )
        return visual_frames


def _copy_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in mapping.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, dict):
            copied[key] = _copy_mapping(value)
        elif isinstance(value, list):
            copied[key] = list(value)
        elif isinstance(value, tuple):
            copied[key] = tuple(value)
        else:
            copied[key] = value
    return copied


def _geometry_progress_summary(frames: list[TemporalFrame]) -> dict[str, Any]:
    if not frames:
        return {"history_items": 0, "net_distance_to_target_progress": 0.0, "net_distance_to_entrance_progress": 0.0}
    distance_to_target_history = [
        float(np.linalg.norm(frame.plug_pose[:3] - frame.target_port_pose[:3])) for frame in frames
    ]
    distance_to_entrance_history = [
        float(frame.score_geometry.get("distance_to_entrance", distance_to_target_history[index]))
        for index, frame in enumerate(frames)
    ]
    insertion_progress_history = [
        float(frame.score_geometry.get("insertion_progress", 0.0))
        for frame in frames
    ]
    return {
        "history_items": len(frames),
        "distance_to_target_history": distance_to_target_history,
        "distance_to_entrance_history": distance_to_entrance_history,
        "insertion_progress_history": insertion_progress_history,
        "net_distance_to_target_progress": float(distance_to_target_history[0] - distance_to_target_history[-1]),
        "net_distance_to_entrance_progress": float(distance_to_entrance_history[0] - distance_to_entrance_history[-1]),
        "net_insertion_progress": float(insertion_progress_history[-1] - insertion_progress_history[0]),
    }


def _wrench_contact_trend_summary(frames: list[TemporalFrame]) -> dict[str, Any]:
    if not frames:
        return {
            "history_items": 0,
            "current_force_l2": 0.0,
            "max_force_l2_recent": 0.0,
            "mean_force_l2_recent": 0.0,
            "current_torque_l2": 0.0,
            "max_torque_l2_recent": 0.0,
            "mean_torque_l2_recent": 0.0,
            "contact_count_recent": 0,
            "contact_fraction_recent": 0.0,
            "force_increasing_recent": False,
            "last_wrench_timestamp": 0.0,
        }
    wrench_history = np.stack([frame.wrench for frame in frames], axis=0)
    force_l2 = np.linalg.norm(wrench_history[:, :3], axis=1)
    torque_l2 = np.linalg.norm(wrench_history[:, 3:], axis=1)
    contacts = np.asarray([float(frame.off_limit_contact) for frame in frames], dtype=np.float64)
    return {
        "history_items": len(frames),
        "current_force_l2": float(force_l2[-1]),
        "max_force_l2_recent": float(force_l2.max(initial=0.0)),
        "mean_force_l2_recent": float(force_l2.mean()),
        "current_torque_l2": float(torque_l2[-1]),
        "max_torque_l2_recent": float(torque_l2.max(initial=0.0)),
        "mean_torque_l2_recent": float(torque_l2.mean()),
        "contact_count_recent": int(contacts.sum()),
        "contact_fraction_recent": float(contacts.mean()),
        "force_increasing_recent": bool(force_l2[-1] > force_l2[0] + 1e-6),
        "last_wrench_timestamp": float(frames[-1].wrench_timestamp),
    }


def _compact_signal_samples(frames: list[TemporalFrame], *, max_items: int = 3) -> dict[str, Any]:
    selected = frames[-max_items:]
    return {
        "history_items": len(selected),
        "sim_time_samples": [float(frame.sim_time) for frame in selected],
        "wrench_samples": [frame.wrench.astype(float).tolist() for frame in selected],
        "wrench_timestamps": [float(frame.wrench_timestamp) for frame in selected],
        "tcp_velocity_samples": [frame.tcp_velocity.astype(float).tolist() for frame in selected],
        "action_samples": [frame.action.astype(float).tolist() for frame in selected],
        "off_limit_contact_samples": [bool(frame.off_limit_contact) for frame in selected],
    }
