#!/usr/bin/env python3
"""Extract robot arm joint postures around a scored Gazebo insertion event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/joint_states", "/scoring/insertion_event"]))
    events: list[dict] = []
    joint_rows: list[dict] = []
    while reader.has_next():
        topic, raw, timestamp_ns = reader.read_next()
        if topic == "/scoring/insertion_event":
            msg = deserialize_message(raw, String)
            events.append({"time_ns": int(timestamp_ns), "value": msg.data})
        elif topic == "/joint_states":
            msg = deserialize_message(raw, JointState)
            joints = dict(zip(msg.name, msg.position))
            if all(name in joints for name in ARM_JOINTS):
                joint_rows.append({
                    "time_ns": int(timestamp_ns),
                    "arm_joint_positions_rad": [float(joints[name]) for name in ARM_JOINTS],
                })
    samples: list[dict] = []
    for event in events:
        preceding = [row for row in joint_rows if row["time_ns"] <= event["time_ns"]]
        if preceding:
            samples.append({"event": event, "nearest_preceding_joint_state": preceding[-1]})
    nearest_transforms: dict[str, dict] = {}
    if events:
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="mcap"),
            rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        reader.set_filter(rosbag2_py.StorageFilter(topics=["/tf", "/tf_static"]))
        event_time = events[0]["time_ns"]
        while reader.has_next():
            topic, raw, timestamp_ns = reader.read_next()
            if timestamp_ns > event_time and topic != "/tf_static":
                break
            msg = deserialize_message(raw, TFMessage)
            for tf in msg.transforms:
                child = tf.child_frame_id
                nearest_transforms[child] = {
                    "time_ns": int(timestamp_ns),
                    "parent": tf.header.frame_id,
                    "xyz_m": [float(tf.transform.translation.x), float(tf.transform.translation.y), float(tf.transform.translation.z)],
                    "quat_xyzw": [float(tf.transform.rotation.x), float(tf.transform.rotation.y), float(tf.transform.rotation.z), float(tf.transform.rotation.w)],
                }
    result = {
        "bag": str(args.bag),
        "joint_names": list(ARM_JOINTS),
        "joint_state_count": len(joint_rows),
        "events": events,
        "event_joint_samples": samples,
        "transforms_at_first_event": nearest_transforms,
        "first_joint_state": joint_rows[0] if joint_rows else None,
        "last_joint_state": joint_rows[-1] if joint_rows else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
