#!/usr/bin/env python3
"""Causal joint-command and measured endpoint audit for a scored MoveIt replay bag."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def main(bag: Path, trajectory_path: Path, output: Path):
    trajectory = json.loads(trajectory_path.read_text())
    final_waypoint = trajectory["waypoints"][-1]
    plan_joint_names = final_waypoint["joint_names"]
    plan_joint_positions = np.asarray(final_waypoint["joint_positions"], dtype=float)
    plan_tcp = np.asarray(final_waypoint["tcp_pose"]["position"], dtype=float)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    joint_commands = []
    pose_commands = 0
    last_joint_state = None
    last_tcp = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic not in {"/aic_controller/joint_commands", "/aic_controller/pose_commands",
                         "/joint_states", "/aic_controller/controller_state"}:
            continue
        msg = deserialize_message(raw, types[topic])
        if topic == "/aic_controller/joint_commands":
            joint_commands.append({"stamp_ns": stamp, "positions": list(msg.target_state.positions)})
        elif topic == "/aic_controller/pose_commands":
            pose_commands += 1
        elif topic == "/joint_states":
            last_joint_state = {name: float(position) for name, position in zip(msg.name, msg.position)}
        else:
            pose = msg.tcp_pose.position
            last_tcp = np.asarray([pose.x, pose.y, pose.z], dtype=float)
    if not joint_commands or last_joint_state is None or last_tcp is None:
        raise RuntimeError("Missing joint command, measured joint state, or TCP state")
    measured = np.asarray([last_joint_state[name] for name in plan_joint_names], dtype=float)
    last_command = np.asarray(joint_commands[-1]["positions"], dtype=float)
    summary = {
        "schema": "aic_moveit_route_replay_audit/v1",
        "joint_command_count": len(joint_commands),
        "pose_command_count": pose_commands,
        "last_command_vs_plan_joint_max_abs_rad": float(np.max(np.abs(last_command - plan_joint_positions))),
        "measured_vs_plan_final_joint_max_abs_rad": float(np.max(np.abs(measured - plan_joint_positions))),
        "measured_vs_plan_final_tcp_mm": float(np.linalg.norm(last_tcp - plan_tcp) * 1000),
        "measured_final_tcp_base_m": last_tcp.tolist(),
        "planned_final_tcp_base_m": plan_tcp.tolist(),
        "trajectory_waypoints": len(trajectory["waypoints"]),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
