"""Teacher-layer signal quality helpers.

These helpers keep all "is this signal real?" decisions explicit and
conservative. Missing or synthetic data is surfaced as metadata instead of being
quietly treated as official-quality observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..runtime import AuxiliaryForceContactSummary, RuntimeState


@dataclass(frozen=True)
class SignalQuality:
    signal_name: str
    available: bool
    is_real: bool
    is_synthetic: bool
    is_missing: bool
    source: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_signal_quality_snapshot(
    state: RuntimeState,
    *,
    include_images: bool,
    camera_info: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    camera_info = camera_info or {}
    wrench_norm = float(np.linalg.norm(np.asarray(state.wrench, dtype=np.float64)))
    wrench_real = bool(state.wrench_timestamp > 0.0)
    wrench_available = wrench_real or wrench_norm > 1e-9
    controller_available = bool(state.controller_state)
    controller_real = controller_available
    camera_info_available = bool(include_images and camera_info)
    camera_info_real = bool(
        include_images and camera_info and any(_camera_info_is_real(info) for info in camera_info.values())
    )
    entrance_pose_available = state.target_port_entrance_pose is not None

    return {
        "wrench": SignalQuality(
            signal_name="wrench",
            available=wrench_available,
            is_real=wrench_real,
            is_synthetic=bool(not wrench_real and wrench_available),
            is_missing=bool(not wrench_available),
            source="runtime_wrench_topic" if wrench_real else "teacher_conservative_fallback",
            note=(
                "Runtime reported a timestamped wrench sample."
                if wrench_real
                else "Treat wrench as approximate unless the live runtime bridge exposes real samples."
            ),
        ).to_dict(),
        "controller_state": SignalQuality(
            signal_name="controller_state",
            available=controller_available,
            is_real=controller_real,
            is_synthetic=False,
            is_missing=bool(not controller_available),
            source="runtime_controller_state_topic" if controller_real else "missing",
            note=(
                "Controller state was captured from the runtime ROS observer."
                if controller_real
                else "Controller-derived fields are unavailable in this rollout."
            ),
        ).to_dict(),
        "camera_info": SignalQuality(
            signal_name="camera_info",
            available=camera_info_available,
            is_real=camera_info_real,
            is_synthetic=bool(camera_info_available and not camera_info_real),
            is_missing=bool(include_images and not camera_info_available),
            source=(
                "ros_camera_sidecar"
                if camera_info_real
                else ("image_disabled" if not include_images else "missing_or_placeholder")
            ),
            note=(
                "CameraInfo came from the live ROS camera sidecar."
                if camera_info_real
                else "CameraInfo is unavailable, disabled, or placeholder-valued."
            ),
        ).to_dict(),
        "target_port_entrance_pose": SignalQuality(
            signal_name="target_port_entrance_pose",
            available=entrance_pose_available,
            is_real=entrance_pose_available,
            is_synthetic=False,
            is_missing=bool(not entrance_pose_available),
            source="runtime_geometry" if entrance_pose_available else "missing",
            note=(
                "Entrance pose is present in the runtime geometry."
                if entrance_pose_available
                else "Partial-insertion depth must remain approximate without entrance pose."
            ),
        ).to_dict(),
        "partial_insertion_depth": SignalQuality(
            signal_name="partial_insertion_depth",
            available=bool(entrance_pose_available),
            is_real=False,
            is_synthetic=False,
            is_missing=bool(not entrance_pose_available),
            source="local_teacher_scoring",
            note=(
                "Teacher scoring still uses a local approximation for partial insertion depth."
            ),
        ).to_dict(),
        "tier1_validity": SignalQuality(
            signal_name="tier1_validity",
            available=True,
            is_real=False,
            is_synthetic=False,
            is_missing=False,
            source="local_teacher_assumption",
            note=(
                "Tier 1 validity remains approximate unless the trajectory is executed through official aic_model."
            ),
        ).to_dict(),
    }


def serialize_nested(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, dict):
        return {str(key): serialize_nested(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_nested(item) for item in value]
    return value


def summarize_camera_info(camera_info: dict[str, Any] | None) -> dict[str, Any]:
    if not camera_info:
        return {}
    return {
        name: {
            "size": serialize_nested(info.get("size", [])),
            "k": serialize_nested(info.get("k", [])),
            "p": serialize_nested(info.get("p", [])),
            "is_real": _camera_info_is_real(info),
        }
        for name, info in camera_info.items()
    }


def controller_state_summary(controller_state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: serialize_nested(value)
        for key, value in controller_state.items()
    }


def ranking_quality_adjustment(
    data_quality: dict[str, dict[str, Any]],
    *,
    include_camera_penalty: bool = True,
) -> tuple[float, dict[str, float]]:
    penalties: dict[str, float] = {
        "wrench_missing_or_synthetic": 0.0,
        "controller_state_missing": 0.0,
        "camera_info_missing_or_synthetic": 0.0,
        "partial_insertion_depth_approximate": 0.0,
        "tier1_validity_approximate": 0.0,
    }
    wrench = data_quality.get("wrench", {})
    controller = data_quality.get("controller_state", {})
    camera = data_quality.get("camera_info", {})
    partial_depth = data_quality.get("partial_insertion_depth", {})
    tier1 = data_quality.get("tier1_validity", {})

    if not bool(wrench.get("is_real", False)):
        penalties["wrench_missing_or_synthetic"] = -6.0 if bool(wrench.get("is_missing", False)) else -3.0
    if not bool(controller.get("is_real", False)):
        penalties["controller_state_missing"] = -4.0
    if include_camera_penalty and bool(camera.get("available", False)) and not bool(camera.get("is_real", False)):
        penalties["camera_info_missing_or_synthetic"] = -2.0
    if not bool(partial_depth.get("is_real", False)):
        penalties["partial_insertion_depth_approximate"] = -1.5
    if not bool(tier1.get("is_real", False)):
        penalties["tier1_validity_approximate"] = -1.0

    total = float(sum(penalties.values()))
    return total, penalties


def normalize_auxiliary_force_contact_summary(
    payload: dict[str, Any] | AuxiliaryForceContactSummary | None,
) -> dict[str, Any]:
    if isinstance(payload, AuxiliaryForceContactSummary):
        payload = payload.to_dict()
    payload = dict(payload or {})
    return {
        "is_official_observation": bool(payload.get("is_official_observation", False)),
        "source": str(payload.get("source", "missing")),
        "substep_tick_count": int(payload.get("substep_tick_count", 0)),
        "sample_count": int(payload.get("sample_count", 0)),
        "wrench_current": _vector6(payload.get("wrench_current")),
        "wrench_max_abs_recent": _vector6(payload.get("wrench_max_abs_recent")),
        "wrench_mean_recent": _vector6(payload.get("wrench_mean_recent")),
        "wrench_max_force_abs_recent": float(payload.get("wrench_max_force_abs_recent", 0.0)),
        "wrench_max_torque_abs_recent": float(payload.get("wrench_max_torque_abs_recent", 0.0)),
        "had_contact_recent": bool(payload.get("had_contact_recent", False)),
        "max_contact_indicator_recent": float(payload.get("max_contact_indicator_recent", 0.0)),
        "first_wrench_recent": _vector6(payload.get("first_wrench_recent")),
        "last_wrench_recent": _vector6(payload.get("last_wrench_recent")),
        "time_of_peak_within_step": (
            None
            if payload.get("time_of_peak_within_step") is None
            else float(payload["time_of_peak_within_step"])
        ),
        "limitations": [str(item) for item in payload.get("limitations", [])],
    }


def synthetic_auxiliary_force_contact_summary(state: RuntimeState) -> dict[str, Any]:
    summary = normalize_auxiliary_force_contact_summary(state.auxiliary_force_contact_summary)
    summary["source"] = "teacher_synthetic_current_sample"
    limitations = list(summary.get("limitations", []))
    limitations.append(
        "Teacher synthesized this auxiliary summary from the current runtime state because step_info did not expose one."
    )
    summary["limitations"] = limitations
    return summary


def auxiliary_summary_quality(
    *,
    auxiliary_force_contact_summary: dict[str, Any] | AuxiliaryForceContactSummary | None,
    available: bool,
) -> dict[str, Any]:
    summary = normalize_auxiliary_force_contact_summary(auxiliary_force_contact_summary)
    is_synthetic = bool(summary.get("source") == "teacher_synthetic_current_sample")
    is_missing = bool(not available and not is_synthetic)
    return {
        "available": bool(available),
        "is_real": bool(available and not is_synthetic),
        "is_synthetic": is_synthetic,
        "is_missing": is_missing,
        "status": "missing" if is_missing else ("synthetic" if is_synthetic else "real"),
        "source": summary.get("source", "missing"),
        "sample_count": int(summary.get("sample_count", 0)),
        "substep_tick_count": int(summary.get("substep_tick_count", 0)),
    }


def summarize_auxiliary_force_contact_summary(
    *,
    auxiliary_force_contact_summary: dict[str, Any] | AuxiliaryForceContactSummary | None,
    auxiliary_summary_available: bool,
    current_wrench: Any,
) -> dict[str, Any]:
    summary = normalize_auxiliary_force_contact_summary(auxiliary_force_contact_summary)
    quality = auxiliary_summary_quality(
        auxiliary_force_contact_summary=summary,
        available=auxiliary_summary_available,
    )
    current_force = float(np.linalg.norm(np.asarray(current_wrench, dtype=np.float64).reshape(-1)[:3]))
    auxiliary_force_max = float(summary.get("wrench_max_force_abs_recent", 0.0))
    had_contact_recent = bool(summary.get("had_contact_recent", False))
    quiet_final_sample = bool(current_force <= max(0.5, 0.2 * auxiliary_force_max))
    hidden_contact_recent = bool(
        quality["available"]
        and had_contact_recent
        and auxiliary_force_max > current_force + 1e-6
        and quiet_final_sample
    )
    return {
        "auxiliary_summary_available": quality["available"],
        "auxiliary_summary_status": quality["status"],
        "auxiliary_force_summary_available": quality["available"],
        "auxiliary_summary_source": quality["source"],
        "auxiliary_summary_sample_count": quality["sample_count"],
        "auxiliary_summary_substep_tick_count": quality["substep_tick_count"],
        "current_wrench_force_l2_norm": current_force,
        "auxiliary_wrench_max_recent": auxiliary_force_max,
        "auxiliary_wrench_max_force_recent": auxiliary_force_max,
        "auxiliary_wrench_max_torque_recent": float(summary.get("wrench_max_torque_abs_recent", 0.0)),
        "had_contact_recent": had_contact_recent,
        "hidden_contact_recent": hidden_contact_recent,
        "max_contact_indicator_recent": float(summary.get("max_contact_indicator_recent", 0.0)),
    }


def summarize_auxiliary_force_contact_history(
    frames: list[dict[str, Any]],
) -> dict[str, Any]:
    if not frames:
        return {
            "window_size": 0,
            "history_items": 0,
            "auxiliary_summary_available": False,
            "available_count": 0,
            "real_count": 0,
            "synthetic_count": 0,
            "missing_count": 0,
            "had_contact_recent": False,
            "hidden_contact_recent": False,
            "hidden_contact_event_count_recent": 0,
            "repeated_contact_rich_steps": 0,
            "auxiliary_wrench_max_recent": 0.0,
            "current_wrench_force_l2_norm": 0.0,
            "latest_auxiliary_summary_status": "missing",
        }
    available_count = sum(int(frame.get("auxiliary_summary_available", False)) for frame in frames)
    synthetic_count = sum(int(frame.get("auxiliary_summary_status") == "synthetic") for frame in frames)
    real_count = sum(int(frame.get("auxiliary_summary_status") == "real") for frame in frames)
    missing_count = sum(int(frame.get("auxiliary_summary_status") == "missing") for frame in frames)
    hidden_count = sum(int(frame.get("hidden_contact_recent", False)) for frame in frames)
    contact_count = sum(int(frame.get("had_contact_recent", False)) for frame in frames)
    latest = frames[-1]
    return {
        "window_size": len(frames),
        "history_items": len(frames),
        "auxiliary_summary_available": bool(available_count > 0),
        "available_count": available_count,
        "real_count": real_count,
        "synthetic_count": synthetic_count,
        "missing_count": missing_count,
        "had_contact_recent": bool(latest.get("had_contact_recent", False)),
        "hidden_contact_recent": bool(latest.get("hidden_contact_recent", False)),
        "hidden_contact_event_count_recent": hidden_count,
        "repeated_contact_rich_steps": contact_count,
        "auxiliary_wrench_max_recent": max(
            (float(frame.get("auxiliary_wrench_max_recent", 0.0)) for frame in frames),
            default=0.0,
        ),
        "current_wrench_force_l2_norm": float(latest.get("current_wrench_force_l2_norm", 0.0)),
        "latest_auxiliary_summary_status": str(latest.get("auxiliary_summary_status", "missing")),
        "latest_auxiliary_summary_source": str(latest.get("auxiliary_summary_source", "missing")),
    }


def ranking_auxiliary_adjustment(
    auxiliary_metrics: dict[str, Any],
    *,
    hidden_contact_event_weight: float = 0.35,
    quiet_hidden_force_weight: float = 0.15,
    repeated_contact_rich_weight: float = 0.05,
) -> tuple[float, dict[str, float]]:
    penalties = {
        "auxiliary_hidden_transient_contact_count": 0.0,
        "auxiliary_quiet_final_sample_force_gap": 0.0,
        "auxiliary_repeated_contact_rich_steps": 0.0,
    }
    if not bool(auxiliary_metrics.get("auxiliary_summary_available", False)):
        return 0.0, penalties
    hidden_count = int(auxiliary_metrics.get("hidden_contact_event_count_recent", 0))
    repeated_contact_steps = int(auxiliary_metrics.get("repeated_contact_rich_steps", 0))
    current_force = float(auxiliary_metrics.get("current_wrench_force_l2_norm", 0.0))
    auxiliary_force = float(auxiliary_metrics.get("auxiliary_wrench_max_recent", 0.0))
    quiet_force_gap = max(auxiliary_force - current_force, 0.0)
    penalties["auxiliary_hidden_transient_contact_count"] = -hidden_contact_event_weight * float(hidden_count)
    penalties["auxiliary_quiet_final_sample_force_gap"] = -quiet_hidden_force_weight * min(quiet_force_gap / 10.0, 1.0)
    penalties["auxiliary_repeated_contact_rich_steps"] = -repeated_contact_rich_weight * float(
        max(repeated_contact_steps - 1, 0)
    )
    total = float(sum(penalties.values()))
    return total, penalties


def _camera_info_is_real(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("size", "k", "p"):
        value = info.get(key)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.size > 0 and not np.allclose(array, 0.0):
            return True
    return False


def _vector6(value: Any) -> list[float]:
    array = np.asarray(value if value is not None else np.zeros(6), dtype=np.float64).reshape(-1)
    if array.size < 6:
        array = np.pad(array, (0, 6 - array.size))
    return [float(item) for item in array[:6]]
