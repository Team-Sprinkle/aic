#!/usr/bin/env python3
"""Compare planned route segments with measured TCP and plug-tip motion."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def pose_matrix(t):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat([
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
    ]).as_matrix()
    m[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return m


def main(bag: Path, trajectory_path: Path, output: Path) -> None:
    trajectory = json.loads(trajectory_path.read_text())
    segments = {}
    for item in trajectory["waypoints"]:
        name = item["diagnostics"]["segment"]
        bounds = segments.setdefault(name, [item["timestamp"], item["timestamp"]])
        bounds[1] = item["timestamp"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()
             if item.name in {"/scoring/tf", "/aic_controller/joint_commands",
                              "/aic_controller/pose_commands", "/aic_controller/controller_state"}}
    start_ns = None
    last_joint_ns = None
    first_pose_ns = None
    latest_sim_ns = None
    tf = {}
    tcp = None
    rows = []
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in types:
            continue
        if topic == "/aic_controller/joint_commands":
            if latest_sim_ns is not None:
                if start_ns is None:
                    start_ns = latest_sim_ns
                last_joint_ns = latest_sim_ns
        elif topic == "/aic_controller/pose_commands":
            if first_pose_ns is None and latest_sim_ns is not None:
                first_pose_ns = latest_sim_ns
        elif topic == "/aic_controller/controller_state":
            msg = deserialize_message(raw, types[topic])
            p = msg.tcp_pose.position
            tcp = np.array([p.x, p.y, p.z], dtype=float)
            latest_sim_ns = msg.header.stamp.sec*1_000_000_000 + msg.header.stamp.nanosec
        elif topic == "/scoring/tf" and start_ns is not None:
            msg = deserialize_message(raw, types[topic])
            sim_ns = None
            for item in msg.transforms:
                sim_ns = item.header.stamp.sec*1_000_000_000 + item.header.stamp.nanosec
                key = (item.header.frame_id, item.child_frame_id)
                if key in {("aic_world", "cable_1"),
                           ("cable_1", "cable_1/sc_tip_link")}:
                    tf[key] = pose_matrix(item.transform)
            if len(tf) == 2 and tcp is not None and sim_ns is not None:
                tip = (tf[("aic_world", "cable_1")] @
                       tf[("cable_1", "cable_1/sc_tip_link")])[:3, 3]
                rows.append({"time_from_first_joint_s": (sim_ns - start_ns) * 1e-9,
                             "sim_ns": sim_ns, "tcp_base_m": tcp.tolist(),
                             "plug_tip_world_m": tip.tolist()})
    def nearest(second):
        return min(rows, key=lambda r: abs(r["time_from_first_joint_s"] - second))
    per_segment = {}
    for name, (start, end) in segments.items():
        a, b = nearest(start), nearest(end)
        per_segment[name] = {
            "planned_start_end_s": [start, end],
            "measured_start_s": a["time_from_first_joint_s"],
            "measured_end_s": b["time_from_first_joint_s"],
            "tcp_displacement_mm": float(1000*np.linalg.norm(np.asarray(b["tcp_base_m"])-a["tcp_base_m"])),
            "plug_tip_displacement_mm": float(1000*np.linalg.norm(np.asarray(b["plug_tip_world_m"])-a["plug_tip_world_m"])),
            "plug_tip_delta_world_mm": (1000*(np.asarray(b["plug_tip_world_m"])-a["plug_tip_world_m"])).tolist(),
            "plug_tip_start_world_m": a["plug_tip_world_m"],
            "plug_tip_end_world_m": b["plug_tip_world_m"],
        }
    plan_end = trajectory["waypoints"][-1]["tcp_pose"]["position"]
    handoff = min(rows, key=lambda r: abs(r["sim_ns"] - first_pose_ns)) if first_pose_ns else None
    result = {"schema": "aic_moveit_tilt_motion/v2", "first_joint_sim_ns": start_ns,
              "last_joint_s": (last_joint_ns-start_ns)*1e-9 if last_joint_ns else None,
              "first_pose_command_s": (first_pose_ns-start_ns)*1e-9 if first_pose_ns else None,
              "segments": per_segment,
              "handoff_observation": handoff,
              "handoff_tcp_vs_plan_mm": (float(1000*np.linalg.norm(np.asarray(handoff["tcp_base_m"])-plan_end))
                                         if handoff else None),
              "sample_count": len(rows)}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"segments": per_segment,
                      "handoff_tcp_vs_plan_mm": result["handoff_tcp_vs_plan_mm"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
