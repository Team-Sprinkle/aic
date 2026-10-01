#!/usr/bin/env python3
"""Save the first measured six-DOF arm state from a scored ROS bag."""

import json
from pathlib import Path
import sys

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")


def main(bag: Path, output: Path):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if topic != "/joint_states":
            continue
        message = deserialize_message(raw, types[topic])
        by_name = dict(zip(message.name, message.position))
        if all(name in by_name for name in JOINTS):
            row = {"name": list(JOINTS), "position": [float(by_name[name]) for name in JOINTS],
                   "source_scored_bag": str(bag), "source_stamp_ns": stamp}
            output.write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps(row))
            return
    raise RuntimeError("Scored bag contains no complete arm joint state")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
