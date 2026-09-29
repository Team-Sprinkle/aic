#!/usr/bin/env python3
"""List SC-port TF edges in a scored Gazebo MCAP for label reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def extract_edges(bag: Path, first_seconds: float, include_camera_chain: bool = False) -> list[dict]:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {topic.name: get_message(topic.type) for topic in reader.get_all_topics_and_types()}
    found = {}
    first_stamp = None
    while reader.has_next():
        topic, raw, stamp = reader.read_next()
        if first_stamp is None:
            first_stamp = stamp
        if stamp > first_stamp + first_seconds * 1e9:
            break
        if topic not in {"/tf", "/tf_static", "/scoring/tf"}:
            continue
        message = deserialize_message(raw, types[topic])
        for edge in message.transforms:
            camera_edge = include_camera_chain and any(
                name in (edge.child_frame_id + " " + edge.header.frame_id)
                for name in ("cam_mount", "camera/", "ati/", "gripper/", "tool0")
            )
            if (not camera_edge and "sc_port" not in edge.child_frame_id and "sc_port" not in edge.header.frame_id
                    and edge.child_frame_id not in {"task_board", "base_link", "tabletop"}
                    and edge.header.frame_id not in {"task_board", "base_link", "tabletop"}):
                continue
            key = (topic, edge.header.frame_id, edge.child_frame_id)
            if key not in found:
                tr = edge.transform
                found[key] = {
                    "topic": topic, "parent": edge.header.frame_id, "child": edge.child_frame_id,
                    "xyz_m": [tr.translation.x, tr.translation.y, tr.translation.z],
                    "quat_xyzw": [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w],
                }
    return list(found.values())


def main(bag: Path, first_seconds: float, include_camera_chain: bool = False) -> None:
    print(json.dumps({"bag": str(bag), "edges": extract_edges(bag, first_seconds, include_camera_chain)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--first-seconds", type=float, default=5)
    parser.add_argument("--include-camera-chain", action="store_true")
    args = parser.parse_args()
    main(args.bag, args.first_seconds, args.include_camera_chain)
