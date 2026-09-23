#!/usr/bin/env python3
"""Summarize one Gazebo scoring bag for the ordinary broad CheatCode audit.

Run inside the official container after sourcing /ws_aic/install/setup.bash.
The tier-3 score remains the authoritative insertion outcome.
"""

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
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    p = transform.translation
    result[:3, 3] = [p.x, p.y, p.z]
    return result


def main(bag, output, family, target):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    transforms = {}
    command = measured = None
    samples = []
    forces = []
    events = 0
    t0 = None
    next_sample = 0.0
    cable = "cable_1" if family == "sc_to_sc" else "cable_0"
    plug_link = "sc_tip_link" if family == "sc_to_sc" else "sfp_tip_link"
    if family == "sc_to_sc":
        port_root = f"task_board/sc_port_{target}"
        entrance = f"{port_root}/sc_port_base_link_entrance"
    else:
        port_root = f"task_board/nic_card_mount_{target[0]}"
        entrance = f"{port_root}/sfp_port_{target[1]}_link_entrance"
    required = [
        ("aic_world", cable), (cable, f"{cable}/{plug_link}"),
        ("aic_world", "task_board"), ("task_board", port_root),
        (port_root, entrance),
    ]

    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if t0 is None:
            t0 = stamp
        elapsed = (stamp - t0) / 1e9
        if topic == "/scoring/tf":
            msg = deserialize_message(raw, types[topic])
            for item in msg.transforms:
                transforms[(item.header.frame_id, item.child_frame_id)] = matrix(item.transform)
            if elapsed < next_sample or any(key not in transforms for key in required):
                continue
            world_plug = transforms[required[0]] @ transforms[required[1]]
            world_port = transforms[required[2]] @ transforms[required[3]] @ transforms[required[4]]
            relative = np.linalg.inv(world_port) @ world_plug
            xyz = relative[:3, 3] * 1000
            row = {
                "time_s": round(elapsed, 3),
                "entrance_relative_xyz_mm": xyz.tolist(),
                "lateral_mm": float(np.linalg.norm(xyz[:2])),
                "axial_mm": float(xyz[2]),
                "orientation_deg": float(Rotation.from_matrix(relative[:3, :3]).magnitude() * 180 / math.pi),
                "measured_tcp_xyz_m": measured.tolist() if measured is not None else None,
                "commanded_tcp_xyz_m": command.tolist() if command is not None else None,
                "tcp_command_error_mm": float(np.linalg.norm(command - measured) * 1000) if command is not None and measured is not None else None,
            }
            samples.append(row)
            next_sample += .2
        elif topic == "/fts_broadcaster/wrench":
            msg = deserialize_message(raw, types[topic])
            f = msg.wrench.force
            forces.append((elapsed, float(np.linalg.norm([f.x, f.y, f.z]))))
        elif topic == "/scoring/insertion_event":
            events += 1
        elif topic == "/aic_controller/pose_commands":
            msg = deserialize_message(raw, types[topic])
            p = msg.pose.position
            command = np.array([p.x, p.y, p.z])
        elif topic == "/aic_controller/controller_state":
            msg = deserialize_message(raw, types[topic])
            p = msg.tcp_pose.position
            measured = np.array([p.x, p.y, p.z])

    peak = max(forces, key=lambda row: row[1]) if forces else None
    near_peak = min(samples, key=lambda row: abs(row["time_s"] - peak[0])) if peak and samples else None
    f = np.array([x[1] for x in forces])
    summary = {
        "schema": "aic_ordinary_broad_trial/v1",
        "bag": str(bag), "family": family, "target": target,
        "duration_s": round((samples[-1]["time_s"] if samples else 0), 3),
        "sample_count": len(samples), "raw_insertion_event_count": events,
        "force_max_n": float(f.max()) if len(f) else None,
        "force_p95_n": float(np.percentile(f, 95)) if len(f) else None,
        "peak_force_time_s": round(peak[0], 3) if peak else None,
        "state_near_peak_force": near_peak,
        "final": samples[-1] if samples else None,
        "samples": samples,
        "caveat": "No collider-specific cable/card contact topic is in this bag. Force and proximity cannot identify a cable snag by themselves.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output, len(samples), round(summary["force_max_n"] or 0, 1))


if __name__ == "__main__":
    raw_target = sys.argv[4]
    target = int(raw_target) if sys.argv[3] == "sc_to_sc" else tuple(map(int, raw_target.split(",")))
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], target)
