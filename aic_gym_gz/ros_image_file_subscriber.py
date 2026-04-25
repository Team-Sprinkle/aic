"""System-Python ROS image subscriber that writes latest frames to files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _image_to_array(message: Any, *, shape: tuple[int, int, int]) -> np.ndarray:
    height = int(message.height)
    width = int(message.width)
    channels = int(shape[2])
    data = np.frombuffer(message.data, dtype=np.uint8)
    image = data.reshape(height, width, channels)
    if str(message.encoding).lower() == "bgr8":
        image = image[:, :, ::-1]
    elif str(message.encoding).lower() != "rgb8":
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    out_h, out_w, _ = shape
    if (height, width) != (out_h, out_w):
        rows = np.linspace(0, height - 1, out_h, dtype=np.int64)
        cols = np.linspace(0, width - 1, out_w, dtype=np.int64)
        image = image[rows][:, cols]
    return np.asarray(image, dtype=np.uint8)


def _atomic_write_frame(path: Path, frame: np.ndarray) -> None:
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("wb") as handle:
            np.save(handle, frame)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image

    topics = json.loads(args.topics_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shape = (int(args.height), int(args.width), 3)

    rclpy.init()
    node = Node("aic_gym_gz_ros_image_file_subscriber")
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)

    def make_callback(name: str):
        def callback(message: Any) -> None:
            try:
                frame = _image_to_array(message, shape=shape)
                timestamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
                _atomic_write_frame(output_dir / f"{name}.npy", frame)
                _atomic_write_json(
                    output_dir / f"{name}.json",
                    {"timestamp": timestamp, "sum": int(frame.sum())},
                )
            except Exception as exc:
                _atomic_write_json(output_dir / f"{name}.error.json", {"error": str(exc)})

        return callback

    for name, topic in topics.items():
        node.create_subscription(Image, str(topic), make_callback(str(name)), qos)

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
