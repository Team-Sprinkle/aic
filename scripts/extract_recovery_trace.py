#!/usr/bin/env python3
"""Extract observation-side force, command, and TCP trace from a scoring MCAP.

Run inside the pinned official evaluation image with ROS sourced. This does
not infer contacts or use scoring geometry as a policy input.
"""

from __future__ import annotations

import json
import math
import argparse
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def xyz(point):
    return [float(point.x), float(point.y), float(point.z)]


def quaternion(value):
    return [float(value.w), float(value.x), float(value.y), float(value.z)]


def norm(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main(bag: Path, destination: Path, rate_hz: float):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    required = {"/aic_controller/controller_state", "/aic_controller/pose_commands", "/fts_broadcaster/wrench"}
    absent = sorted(required - types.keys())
    if absent:
        raise RuntimeError(f"bag lacks required topics: {absent}")
    t0 = None
    command = previous_command = None
    command_change = 0.0
    latest_force = None
    rows = []
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if t0 is None:
            t0 = stamp
        if topic == "/aic_controller/pose_commands":
            msg = deserialize_message(raw, types[topic])
            command = xyz(msg.pose.position)
            if previous_command is not None:
                command_change += norm(command, previous_command)
            previous_command = command
        elif topic == "/fts_broadcaster/wrench":
            force = deserialize_message(raw, types[topic]).wrench.force
            latest_force = xyz(force)
        elif topic == "/aic_controller/controller_state":
            msg = deserialize_message(raw, types[topic])
            pose = msg.tcp_pose
            measured = xyz(pose.position)
            rows.append({
                "time_s": round((stamp - t0) / 1e9, 5),
                "measured_tcp_base_m": measured,
                "measured_tcp_quat_wxyz": quaternion(pose.orientation),
                "commanded_tcp_base_m": command,
                "command_motion_since_last_sample_m": command_change,
                "tcp_tracking_error_m": norm(command, measured) if command else None,
                "wrist_force_xyz_n": latest_force,
                "wrist_force_norm_n": math.sqrt(sum(value**2 for value in latest_force)) if latest_force else None,
            })
            command_change = 0.0
    if rate_hz > 0:
        reduced = []
        active_bucket = None
        bucket_command = 0.0
        bucket_force = 0.0
        peak_force_xyz = None
        last = None
        for row in rows:
            bucket = int(row["time_s"] * rate_hz)
            if active_bucket is not None and bucket != active_bucket and last is not None:
                last["command_motion_since_last_sample_m"] = bucket_command
                last["wrist_force_norm_n"] = bucket_force
                last["wrist_force_xyz_n"] = peak_force_xyz
                reduced.append(last)
                bucket_command = bucket_force = 0.0
                peak_force_xyz = None
            active_bucket = bucket
            bucket_command += row["command_motion_since_last_sample_m"]
            if (row["wrist_force_norm_n"] or 0.0) >= bucket_force:
                bucket_force = row["wrist_force_norm_n"] or 0.0
                peak_force_xyz = row["wrist_force_xyz_n"]
            last = row
        if last is not None:
            last["command_motion_since_last_sample_m"] = bucket_command
            last["wrist_force_norm_n"] = bucket_force
            last["wrist_force_xyz_n"] = peak_force_xyz
            reduced.append(last)
        rows = reduced
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(destination),
                      "force_max_n": max((row["wrist_force_norm_n"] or 0 for row in rows), default=0)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    arguments = parser.parse_args()
    main(arguments.bag, arguments.output, arguments.rate_hz)
