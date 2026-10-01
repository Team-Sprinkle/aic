#!/usr/bin/env python3
"""Sample scored-bag robot joints near trial startup for LLB start audit."""

import json
from pathlib import Path
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def main(bag, output):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {entry.name: get_message(entry.type) for entry in reader.get_all_topics_and_types()}
    rows = []
    first_stamp = None
    next_second = 0.0
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic != "/joint_states":
            continue
        if first_stamp is None:
            first_stamp = stamp
        seconds = (stamp - first_stamp) / 1e9
        if seconds < next_second:
            continue
        message = deserialize_message(raw, types[topic])
        by_name = dict(zip(message.name, message.position))
        if all(joint in by_name for joint in JOINTS):
            rows.append({"elapsed_wall_s": seconds,
                         "message_sim_stamp_s": message.header.stamp.sec + message.header.stamp.nanosec / 1e9,
                         "positions": {joint: float(by_name[joint]) for joint in JOINTS}})
            next_second += 2.0
        if seconds > 20:
            break
    output.write_text(json.dumps({"bag": str(bag), "samples": rows}, indent=2) + "\n")
    print(json.dumps({"samples": len(rows), "first": rows[0] if rows else None,
                      "last": rows[-1] if rows else None}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
