"""Teacher-layer signal quality helpers.

These helpers keep all "is this signal real?" decisions explicit and
conservative. Missing or synthetic data is surfaced as metadata instead of being
quietly treated as official-quality observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from ..runtime import RuntimeState


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
