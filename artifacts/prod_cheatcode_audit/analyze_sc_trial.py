#!/usr/bin/env python3
"""Extract plug-to-port motion and force from an official scoring MCAP bag."""

import json
import math
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def matrix(transform):
    q = transform.rotation
    out = np.eye(4)
    out[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    p = transform.translation
    out[:3, 3] = [p.x, p.y, p.z]
    return out


def main():
    bag = Path(sys.argv[1])
    output = Path(sys.argv[2])
    port_index = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    port_frame = f"task_board/sc_port_{port_index}"
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topic_types = {x.name: x.type for x in reader.get_all_topics_and_types()}
    message_types = {k: get_message(v) for k, v in topic_types.items()}
    transforms = {}
    command = None
    measured_tcp = None
    samples = []
    forces = []
    insertions = 0
    t0 = None
    next_sample = 0.0
    required = {
        ("aic_world", "cable_1"),
        ("cable_1", "cable_1/sc_tip_link"),
        ("aic_world", "task_board"),
        ("task_board", port_frame),
        (port_frame, f"{port_frame}/sc_port_base_link"),
        (port_frame, f"{port_frame}/sc_port_base_link_entrance"),
    }

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if t0 is None:
            t0 = timestamp
        elapsed = (timestamp - t0) / 1e9
        if topic == "/scoring/tf":
            msg = deserialize_message(data, message_types[topic])
            for item in msg.transforms:
                key = (item.header.frame_id, item.child_frame_id)
                if key in required:
                    transforms[key] = matrix(item.transform)
            if required <= transforms.keys() and elapsed >= next_sample:
                world_plug = (transforms[("aic_world", "cable_1")] @
                              transforms[("cable_1", "cable_1/sc_tip_link")])
                world_port_root = (transforms[("aic_world", "task_board")] @
                                   transforms[("task_board", port_frame)])
                for suffix, field in (("sc_port_base_link", "link"),
                                      ("sc_port_base_link_entrance", "entrance")):
                    world_port = world_port_root @ transforms[(
                        port_frame, f"{port_frame}/{suffix}")]
                    relative = np.linalg.inv(world_port) @ world_plug
                    translation = relative[:3, 3] * 1000.0
                    angle = Rotation.from_matrix(relative[:3, :3]).magnitude() * 180 / math.pi
                    if field == "link":
                        row = {"time_s": elapsed}
                    row[f"{field}_relative_xyz_mm"] = translation.tolist()
                    row[f"{field}_distance_mm"] = float(np.linalg.norm(translation))
                    row[f"{field}_orientation_deg"] = float(angle)
                if measured_tcp is not None:
                    row["measured_tcp_xyz_m"] = measured_tcp.tolist()
                if command is not None:
                    row["commanded_tcp_xyz_m"] = command.tolist()
                    if measured_tcp is not None:
                        row["tcp_tracking_error_mm"] = float(
                            np.linalg.norm(command - measured_tcp) * 1000.0)
                samples.append(row)
                next_sample += 0.1
        elif topic == "/fts_broadcaster/wrench":
            msg = deserialize_message(data, message_types[topic])
            f = msg.wrench.force
            forces.append((elapsed, float(np.linalg.norm([f.x, f.y, f.z]))))
        elif topic == "/scoring/insertion_event":
            insertions += 1
        elif topic == "/aic_controller/pose_commands":
            msg = deserialize_message(data, message_types[topic])
            command = np.array([msg.pose.position.x, msg.pose.position.y,
                                msg.pose.position.z])
        elif topic == "/aic_controller/controller_state":
            msg = deserialize_message(data, message_types[topic])
            measured_tcp = np.array([msg.tcp_pose.position.x,
                                     msg.tcp_pose.position.y,
                                     msg.tcp_pose.position.z])

    force_values = np.array([x[1] for x in forces])
    summary = {
        "bag": str(bag),
        "insertion_event_count": insertions,
        "sample_count": len(samples),
        "force_sample_count": len(forces),
        "force_norm_n": {
            "max": float(force_values.max()) if len(force_values) else None,
            "p95": float(np.percentile(force_values, 95)) if len(force_values) else None,
            "p99": float(np.percentile(force_values, 99)) if len(force_values) else None,
        },
        "minimum_link_distance_mm": min(x["link_distance_mm"] for x in samples),
        "minimum_entrance_distance_mm": min(x["entrance_distance_mm"] for x in samples),
        "final": samples[-1],
        "samples": samples,
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
