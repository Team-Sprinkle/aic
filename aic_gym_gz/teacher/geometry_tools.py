"""Teacher-side geometry helper outputs for planner context."""

from __future__ import annotations

from typing import Any

import numpy as np


def frame_transform_query(
    *,
    source_frame: str,
    target_frame: str,
    pose: list[float] | np.ndarray | None = None,
    point: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Return an explicit transform result or an honest unknown status."""

    if pose is not None and point is not None:
        raise ValueError("Provide either pose or point, not both.")
    if pose is None and point is None:
        raise ValueError("Expected either pose or point.")

    if source_frame == target_frame:
        result: dict[str, Any] = {
            "ok": True,
            "source_frame": source_frame,
            "target_frame": target_frame,
            "transform_available": True,
            "transform_type": "identity",
            "translation_xyz": [0.0, 0.0, 0.0],
            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            "frame_chain": [source_frame, target_frame],
            "notes": [
                "Source and target frame are identical.",
            ],
        }
        if pose is not None:
            result["transformed_pose"] = np.asarray(pose, dtype=np.float64).astype(float).tolist()
        if point is not None:
            result["transformed_point"] = np.asarray(point, dtype=np.float64).astype(float).tolist()
        return result

    result = {
        "ok": False,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "transform_available": False,
        "transform_type": "unknown",
        "translation_xyz": None,
        "rotation_quat_xyzw": None,
        "frame_chain": [],
        "notes": [
            "No explicit transform graph is available on the teacher path for this frame pair.",
            "Do not numerically compare poses across these frames without a verified transform.",
        ],
    }
    if pose is not None:
        result["input_pose"] = np.asarray(pose, dtype=np.float64).astype(float).tolist()
    if point is not None:
        result["input_point"] = np.asarray(point, dtype=np.float64).astype(float).tolist()
    return result


def distance_and_alignment_query(
    *,
    actor_pose: list[float] | np.ndarray,
    target_pose: list[float] | np.ndarray,
    entrance_pose: list[float] | np.ndarray | None,
    actor_name: str = "plug",
    target_name: str = "target_port",
) -> dict[str, Any]:
    """Compute insertion-axis distances/alignment in the runtime pose frame."""

    actor = np.asarray(actor_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    entrance = target if entrance_pose is None else np.asarray(entrance_pose, dtype=np.float64)

    actor_xyz = actor[:3]
    target_xyz = target[:3]
    entrance_xyz = entrance[:3]
    insertion_axis = target_xyz - entrance_xyz
    insertion_axis_length = float(np.linalg.norm(insertion_axis))
    insertion_axis_unit = (
        insertion_axis / insertion_axis_length
        if insertion_axis_length > 1e-8
        else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    )

    actor_to_target = target_xyz - actor_xyz
    actor_to_entrance = entrance_xyz - actor_xyz
    offset_from_entrance = actor_xyz - entrance_xyz
    axial_depth = float(np.dot(offset_from_entrance, insertion_axis_unit))
    clipped_axial_depth = float(np.clip(axial_depth, 0.0, insertion_axis_length))
    lateral_vector = offset_from_entrance - axial_depth * insertion_axis_unit
    lateral_offset = float(np.linalg.norm(lateral_vector))
    pre_insertion_standoff_m = 0.025
    pre_insertion_waypoint = entrance_xyz - pre_insertion_standoff_m * insertion_axis_unit
    guarded_entry_waypoint = entrance_xyz - 0.005 * insertion_axis_unit

    return {
        "actor_name": actor_name,
        "target_name": target_name,
        "distance_to_target_m": float(np.linalg.norm(actor_to_target)),
        "distance_to_entrance_m": float(np.linalg.norm(actor_to_entrance)),
        "offset_to_target_xyz_m": actor_to_target.astype(float).tolist(),
        "offset_to_entrance_xyz_m": actor_to_entrance.astype(float).tolist(),
        "insertion_axis_world_xyz": insertion_axis_unit.astype(float).tolist(),
        "insertion_axis_length_m": insertion_axis_length,
        "lateral_offset_m": lateral_offset,
        "lateral_offset_vector_m": lateral_vector.astype(float).tolist(),
        "axial_depth_m": axial_depth,
        "signed_distance_to_entrance_plane_m": -axial_depth,
        "port_frame_error": {
            "coordinate_frame": "port_entrance_frame",
            "origin_world_xyz": entrance_xyz.astype(float).tolist(),
            "axis_positive_direction": "from port entrance toward fully inserted target",
            "axis_unit_world_xyz": insertion_axis_unit.astype(float).tolist(),
            "axial_depth_m": axial_depth,
            "signed_distance_before_entrance_plane_m": -axial_depth,
            "lateral_offset_norm_m": lateral_offset,
            "lateral_offset_vector_world_m": lateral_vector.astype(float).tolist(),
            "insertion_axis_length_m": insertion_axis_length,
            "pre_insertion_standoff_m": pre_insertion_standoff_m,
            "pre_insertion_waypoint_world_xyz": pre_insertion_waypoint.astype(float).tolist(),
            "guarded_entry_waypoint_world_xyz": guarded_entry_waypoint.astype(float).tolist(),
        },
        "insertion_progress": (
            0.0 if insertion_axis_length <= 1e-8 else clipped_axial_depth / insertion_axis_length
        ),
    }


def clearance_distance_query(
    *,
    actor_pose: list[float] | np.ndarray,
    target_pose: list[float] | np.ndarray | None = None,
    obstacle_points: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    actor_name: str = "plug",
) -> dict[str, Any]:
    """Approximate nearest-obstacle clearance for an actor and optional approach segment."""

    actor_xyz = np.asarray(actor_pose, dtype=np.float64)[:3]
    target_xyz = None if target_pose is None else np.asarray(target_pose, dtype=np.float64)[:3]
    min_actor_distance = float("inf")
    nearest_actor_name = None
    min_segment_distance = float("inf")
    nearest_segment_name = None

    for obstacle in obstacle_points:
        point = np.asarray(obstacle.get("approximate_world_xyz", [0.0, 0.0, 0.0]), dtype=np.float64)
        obstacle_name = str(obstacle.get("name", "obstacle"))
        actor_distance = float(np.linalg.norm(actor_xyz - point))
        if actor_distance < min_actor_distance:
            min_actor_distance = actor_distance
            nearest_actor_name = obstacle_name
        if target_xyz is not None:
            segment_distance = _point_to_segment_distance(point=point, start=actor_xyz, end=target_xyz)
            if segment_distance < min_segment_distance:
                min_segment_distance = segment_distance
                nearest_segment_name = obstacle_name

    return {
        "actor_name": actor_name,
        "nearest_obstacle_name": nearest_actor_name,
        "nearest_obstacle_distance_m": None
        if nearest_actor_name is None
        else float(min_actor_distance),
        "approach_segment_min_clearance_m": None
        if target_xyz is None or nearest_segment_name is None
        else float(min_segment_distance),
        "approach_segment_nearest_obstacle_name": nearest_segment_name,
        "obstacle_count_considered": int(len(obstacle_points)),
    }


def signal_reliability_summary(*, data_quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compact summary of which signals are real, approximate, or missing."""

    summary = {
        "real_signals": [],
        "approximate_signals": [],
        "missing_signals": [],
        "signals": {},
    }
    for signal_name, quality in sorted(data_quality.items()):
        status = {
            "is_real": bool(quality.get("is_real")),
            "is_missing": bool(quality.get("is_missing")),
            "source": quality.get("source"),
            "note": quality.get("note"),
        }
        summary["signals"][signal_name] = status
        if status["is_missing"]:
            summary["missing_signals"].append(signal_name)
        elif status["is_real"]:
            summary["real_signals"].append(signal_name)
        else:
            summary["approximate_signals"].append(signal_name)
    return summary


def build_overlay_metadata(
    *,
    tcp_pose: list[float] | np.ndarray,
    plug_pose: list[float] | np.ndarray,
    target_port_pose: list[float] | np.ndarray,
    target_port_entrance_pose: list[float] | np.ndarray | None,
    runtime_pose_frame: str,
) -> dict[str, Any]:
    """Return overlay descriptors for external visualization tooling."""

    tcp_xyz = np.asarray(tcp_pose, dtype=np.float64)[:3]
    plug_xyz = np.asarray(plug_pose, dtype=np.float64)[:3]
    target_xyz = np.asarray(target_port_pose, dtype=np.float64)[:3]
    entrance_xyz = target_xyz if target_port_entrance_pose is None else np.asarray(
        target_port_entrance_pose, dtype=np.float64
    )[:3]
    crop_center = 0.5 * (plug_xyz + entrance_xyz)
    crop_radius = max(0.03, float(np.linalg.norm(plug_xyz - entrance_xyz)) * 0.75)
    axis = target_xyz - entrance_xyz
    axis_norm = float(np.linalg.norm(axis))
    axis_unit = axis / axis_norm if axis_norm > 1e-8 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    pre_insertion_waypoint = entrance_xyz - 0.025 * axis_unit

    return {
        "runtime_pose_frame": runtime_pose_frame,
        "xyz_axis_overlay_targets": [
            {"name": "tcp", "pose": np.asarray(tcp_pose, dtype=np.float64).astype(float).tolist()},
            {"name": "plug", "pose": np.asarray(plug_pose, dtype=np.float64).astype(float).tolist()},
            {
                "name": "target_port",
                "pose": np.asarray(target_port_pose, dtype=np.float64).astype(float).tolist(),
            },
        ],
        "insertion_axis_overlay": {
            "start_xyz": entrance_xyz.astype(float).tolist(),
            "end_xyz": target_xyz.astype(float).tolist(),
            "label": "port_insertion_axis",
            "coordinate_frame": runtime_pose_frame,
            "axis_positive_direction": "from entrance toward fully inserted target",
        },
        "pre_insertion_waypoint_overlay": {
            "world_xyz": pre_insertion_waypoint.astype(float).tolist(),
            "label": "pre_insert_standoff_25mm",
            "purpose": "First align plug here, then move along the port insertion axis.",
        },
        "zoomed_interaction_crop": {
            "center_xyz": crop_center.astype(float).tolist(),
            "radius_m": crop_radius,
            "focus_objects": ["plug", "target_port_entrance"],
        },
        "recommended_additional_views": ["top_down_xy", "oblique_xy", "side_yz"],
    }


def _point_to_segment_distance(*, point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq <= 1e-12:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(np.dot(point - start, segment) / segment_norm_sq, 0.0, 1.0))
    projection = start + t * segment
    return float(np.linalg.norm(point - projection))
