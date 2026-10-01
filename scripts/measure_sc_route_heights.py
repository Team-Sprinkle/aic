#!/usr/bin/env python3
"""Measure the SC plug and NIC-card height envelope from scored Gazebo TF."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation


def matrix(transform):
    result = np.eye(4)
    p, q = transform.translation, transform.rotation
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [p.x, p.y, p.z]
    return result


def main(bag, scene_path, output):
    objects = json.loads(scene_path.read_text())
    cards = [item for item in objects if item["role"] == "nic_card"]
    tops = []
    for item in cards:
        transform = np.eye(4)
        pose = item["pose"]
        transform[:3, :3] = Rotation.from_quat(pose["orientation_xyzw"]).as_matrix()
        transform[:3, 3] = pose["position"]
        if item["shape"] == "box":
            half = np.asarray(item["dimensions"]) / 2
            local = np.array([[a, b, c] for a in (-half[0], half[0])
                              for b in (-half[1], half[1]) for c in (-half[2], half[2])])
            top = float(np.max((transform[:3, :3] @ local.T).T[:, 2] + transform[2, 3]))
        else:
            # Cylinder axis is local Z.
            radius = item["dimensions"][1]
            half_height = item["dimensions"][0] / 2
            top = float(transform[2, 3] + abs(transform[2, 2])*half_height
                        + np.linalg.norm(transform[2, :2])*radius)
        tops.append((top, item["name"]))
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    tf = {}
    rows = []
    seen_frames = set()
    first_sim_stamp = None
    for _ in range(100000000):
        if not reader.has_next():
            break
        topic, raw, stamp = reader.read_next()
        if topic != "/scoring/tf":
            continue
        sim_stamp = None
        for item in deserialize_message(raw, types[topic]).transforms:
            sim_stamp = item.header.stamp.sec * 1_000_000_000 + item.header.stamp.nanosec
            key = (item.header.frame_id, item.child_frame_id)
            if any(term in key[1] for term in ("base_link", "sc_tip_link", "cable_1", "nic_card_mount_4")):
                seen_frames.add(key)
            tf[key] = matrix(item.transform)
        root = ("aic_world", "cable_1")
        tip = ("cable_1", "cable_1/sc_tip_link")
        if root in tf and tip in tf and sim_stamp is not None:
            if first_sim_stamp is None:
                first_sim_stamp = sim_stamp
            xyz = (tf[root] @ tf[tip])[:3, 3]
            rows.append({"time_s": (sim_stamp-first_sim_stamp)*1e-9, "plug_tip_world_xyz_m": xyz.tolist(),
                         "tip_minus_highest_card_top_mm": float((xyz[2]-max(tops)[0])*1000)})
    first_below = next((row for row in rows if row["tip_minus_highest_card_top_mm"] < 0), None)
    by_time = [min(rows, key=lambda row: abs(row["time_s"] - second))
               for second in (0, 5, 10, 15, 20, 25, 30, 35, 40) if rows]
    result = {"highest_card_top_world_m": max(tops),
              "card_top_world_m_range": [min(x[0] for x in tops), max(x[0] for x in tops)],
              "first_plug_tip": rows[0] if rows else None,
              "last_plug_tip": rows[-1] if rows else None,
              "first_plug_tip_below_card_top": first_below,
              "sampled_timepoints": by_time,
              "min_tip_minus_top_mm": min((x["tip_minus_highest_card_top_mm"] for x in rows), default=None),
              "frame_examples": [list(x) for x in sorted(seen_frames)[:100]],
              "sample_count": len(rows)}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "frame_examples"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
