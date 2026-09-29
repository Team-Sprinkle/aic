#!/usr/bin/env python3
"""Sample the scored Gazebo cable TF without loading image or action topics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage


def pose(edge) -> dict:
    tr = edge.transform
    return {
        "parent": edge.header.frame_id,
        "xyz_m": [float(tr.translation.x), float(tr.translation.y), float(tr.translation.z)],
        "quat_xyzw": [float(tr.rotation.x), float(tr.rotation.y), float(tr.rotation.z), float(tr.rotation.w)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--times", type=float, nargs="+", default=[1, 5, 20, 40, 60, 70])
    args = parser.parse_args()
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/scoring/tf"]))
    initial_stamp = None
    latest = {}
    samples = []
    targets = iter(sorted(set(args.times)))
    target = next(targets, None)
    while reader.has_next() and target is not None:
        _, raw, stamp = reader.read_next()
        if initial_stamp is None:
            initial_stamp = stamp
        message = deserialize_message(raw, TFMessage)
        for edge in message.transforms:
            child = edge.child_frame_id
            if child == "cable_1" or child.startswith("cable_1/"):
                latest[child] = pose(edge)
        elapsed = (stamp - initial_stamp) * 1e-9
        while target is not None and elapsed >= target:
            samples.append({"requested_elapsed_s": target,
                            "actual_elapsed_s": elapsed,
                            "timestamp_ns": int(stamp),
                            "transforms": dict(latest)})
            target = next(targets, None)
    result = {"schema": "aic_sc_gazebo_cable_shape/v1", "bag": str(args.bag),
              "samples": samples, "links_expected": list(range(1, 21))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"samples": len(samples), "link_counts": [
        sum(f"cable_1/link_{i}" in row["transforms"] for i in range(1, 21))
        for row in samples]}))


if __name__ == "__main__":
    main()
