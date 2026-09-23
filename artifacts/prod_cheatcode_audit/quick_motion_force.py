#!/usr/bin/env python3
"""Read only force, TCP state, and Cartesian commands from a scoring MCAP."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = ["/fts_broadcaster/wrench", "/aic_controller/controller_state",
          "/aic_controller/pose_commands"]


def main(bag, out):
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=TOPICS))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()
             if item.name in TOPICS}
    t0 = None
    forces = []
    states = []
    commands = []
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if t0 is None:
            t0 = stamp
        t = (stamp - t0) / 1e9
        msg = deserialize_message(raw, types[topic])
        if topic == TOPICS[0]:
            f = msg.wrench.force
            forces.append([t, float(np.linalg.norm([f.x, f.y, f.z]))])
        elif topic == TOPICS[1]:
            p = msg.tcp_pose.position
            states.append([t, p.x, p.y, p.z])
        elif topic == TOPICS[2]:
            p = msg.pose.position
            commands.append([t, p.x, p.y, p.z])
    summary = {"bag": str(bag), "force": forces, "measured_tcp": states,
               "commanded_tcp": commands,
               "peak_force_n": max((row[1] for row in forces), default=None),
               "counts": {"force": len(forces), "state": len(states), "command": len(commands)}}
    out.write_text(json.dumps(summary) + "\n")
    print(out, summary["counts"], summary["peak_force_n"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
