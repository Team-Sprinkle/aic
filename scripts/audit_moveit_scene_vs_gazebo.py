#!/usr/bin/env python3
"""Compare expanded MoveIt rigid model frames with measured Gazebo scoring TF."""

import json
from pathlib import Path
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation
import yaml


def matrix(tf):
    result = np.eye(4)
    q, p = tf.rotation, tf.translation
    result[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    result[:3, 3] = [p.x, p.y, p.z]
    return result


def main(bag: Path, scene_json: Path, config_path: Path, output: Path):
    objects = json.loads(scene_json.read_text())
    expected_models = {}
    for obj in objects:
        meta = obj.get("metadata", {})
        if "gazebo_model_name" in meta:
            expected_models[meta["gazebo_model_name"]] = meta["model_pose_world"]
    config = yaml.safe_load(config_path.read_text())
    trial = next(iter(config["trials"].values()))
    pose = trial["scene"]["task_board"]["pose"]
    expected_board = np.eye(4)
    expected_board[:3, :3] = Rotation.from_euler("xyz", [pose.get(k, 0.0) for k in ("roll", "pitch", "yaw")]).as_matrix()
    expected_board[:3, 3] = [pose[k] for k in ("x", "y", "z")]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {item.name: get_message(item.type) for item in reader.get_all_topics_and_types()}
    tf = {}
    required = {("aic_world", "task_board")}
    required.update(("task_board", f"task_board/{name}") for name in expected_models)
    while reader.has_next() and not required.issubset(tf):
        topic, raw, _ = reader.read_next()
        if topic == "/scoring/tf":
            for item in deserialize_message(raw, types[topic]).transforms:
                key = (item.header.frame_id, item.child_frame_id)
                if key in required:
                    tf[key] = matrix(item.transform)
    if not required.issubset(tf):
        raise RuntimeError(f"Scoring bag lacks {required - tf.keys()}")
    board = tf[("aic_world", "task_board")]
    rows = []
    for name, expected in expected_models.items():
        model_world = board @ tf[("task_board", f"task_board/{name}")]
        expected_world = np.eye(4)
        expected_world[:3, :3] = Rotation.from_quat(expected["orientation_xyzw"]).as_matrix()
        expected_world[:3, 3] = expected["position"]
        rows.append({"model": name,
                     "translation_error_mm": float(np.linalg.norm(model_world[:3, 3] - expected_world[:3, 3]) * 1000),
                     "rotation_error_deg": float(Rotation.from_matrix(model_world[:3, :3].T @ expected_world[:3, :3]).magnitude()*180/np.pi)})
    summary = {
        "schema": "aic_moveit_scene_tf_audit/v1",
        "board_translation_error_mm": float(np.linalg.norm(board[:3, 3] - expected_board[:3, 3]) * 1000),
        "board_rotation_error_deg": float(Rotation.from_matrix(board[:3, :3].T @ expected_board[:3, :3]).magnitude()*180/np.pi),
        "model_count": len(rows), "models": rows,
        "max_model_translation_error_mm": max(row["translation_error_mm"] for row in rows),
        "max_model_rotation_error_deg": max(row["rotation_error_deg"] for row in rows),
    }
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
