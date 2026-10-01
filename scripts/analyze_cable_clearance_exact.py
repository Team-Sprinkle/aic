#!/usr/bin/env python3
"""Conservative sampled cable-cylinder clearance to Gazebo NIC collision solids."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
import yaml
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def matrix(tf):
    result = np.eye(4)
    q, p = tf.rotation, tf.translation
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [p.x, p.y, p.z]
    return result


def main(bag: Path, scene_path: Path, config_path: Path, output: Path):
    config = yaml.safe_load(config_path.read_text())
    trial = next(iter(config["trials"].values()))
    pose = trial["scene"]["task_board"]["pose"]
    board_position = np.asarray([pose[axis] for axis in ("x", "y", "z")], dtype=float)
    objects = [x for x in json.loads(scene_path.read_text()) if x["role"] == "nic_card"]
    centers = np.asarray([x["pose"]["position"] for x in objects], dtype=float)
    rotations = np.asarray([Rotation.from_quat(x["pose"]["orientation_xyzw"]).as_matrix().T for x in objects])
    # Cylinder AABBs conservatively enclose their true round cross-sections.
    sizes = np.asarray([x["dimensions"] if x["shape"] == "box"
                        else [2*x["dimensions"][1], 2*x["dimensions"][1], x["dimensions"][0]]
                        for x in objects], dtype=float)
    half = sizes / 2
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    tf = {}
    root_key = ("aic_world", "cable_1")
    board_key = ("aic_world", "task_board")
    link_keys = [("cable_1", f"cable_1/link_{i}") for i in range(1, 21)]
    times = []
    samples = []
    t0 = None
    next_t = 0.0
    board_error_mm = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if t0 is None:
            t0 = stamp
        if topic != "/scoring/tf":
            continue
        elapsed = (stamp - t0) * 1e-9
        msg = deserialize_message(raw, types[topic])
        for item in msg.transforms:
            key = (item.header.frame_id, item.child_frame_id)
            if key in link_keys or key in (root_key, board_key):
                tf[key] = matrix(item.transform)
        if elapsed < next_t or any(key not in tf for key in [root_key, board_key, *link_keys]):
            continue
        if board_error_mm is None:
            # Exact expanded scene is in the same Gazebo world coordinates.
            # Board center can be inferred from the configured task-board pose.
            board_error_mm = float(np.linalg.norm(tf[board_key][:3, 3] - board_position) * 1000)
            if board_error_mm > 2:
                raise RuntimeError(f"Gazebo board/world frame mismatch: {board_error_mm:.2f} mm")
        cable_root = tf[root_key]
        minimum = (float("inf"), -1, "", float("inf"), "")
        for index, key in enumerate(link_keys, 1):
            link = cable_root @ tf[key]
            length = 0.036 if index == 1 else 0.048
            shift = 0.006 if index == 1 else 0.0
            offsets = np.linspace(-length/2, length/2, 9) + shift
            points = link[:3, 3][None, :] + offsets[:, None] * link[:3, 2][None, :]
            delta = points[:, None, :] - centers[None, :, :]
            local = np.einsum("nij,pnj->pni", rotations, delta)
            gap = np.linalg.norm(np.maximum(np.abs(local) - half[None, :, :], 0), axis=2) - 0.002
            point_index, obstacle_index = np.unravel_index(np.argmin(gap), gap.shape)
            # Distance to a closed solid is 1-Lipschitz along the cable axis.
            # Subtract half the sampling pitch to get a lower bound along this segment.
            bound = float(gap[point_index, obstacle_index] - length / 16)
            if bound < minimum[0]:
                minimum = (bound, index, objects[obstacle_index]["name"],
                           float(gap[point_index, obstacle_index]), objects[obstacle_index]["shape"])
        times.append(elapsed)
        samples.append({"time_s": elapsed, "lower_bound_clearance_mm": minimum[0]*1000,
                        "sampled_surface_clearance_mm": minimum[3]*1000,
                        "cable_link": minimum[1], "nic_collision": minimum[2], "nic_shape": minimum[4]})
        next_t = elapsed + 0.05
    if not samples:
        raise RuntimeError("No complete scoring TF samples for cable/board")
    worst = min(samples, key=lambda x: x["lower_bound_clearance_mm"])
    summary = {
        "schema": "aic_exact_nic_cable_clearance/v1",
        "sample_count": len(samples),
        "max_sample_gap_s": float(max(np.diff(times))) if len(times) > 1 else None,
        "board_world_position_error_mm": board_error_mm,
        "cable_radius_mm": 2.0,
        "collision_shape_count": len(objects),
        "worst": worst,
        "coverage_limit": "Conservative along each sampled link segment, but unobserved between TF samples; only NIC collision shapes, not all fixtures or plug geometry.",
    }
    output.write_text(json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
