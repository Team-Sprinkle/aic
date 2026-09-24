#!/usr/bin/env python3
"""Capture native-rate wrist and visual-only wide audit cameras."""

from __future__ import annotations

import csv
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class Capture(Node):
    def __init__(self):
        super().__init__("smooth_wide_audit_capture")
        self.bridge = CvBridge()
        self.directory = Path("/audit/frames")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.csv_file = (Path("/audit") / "frame_manifest.csv").open("w", newline="", buffering=1)
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(("camera", "wall_ns", "sim_ns", "file"))
        self.last_sim_ns = {}
        self._audit_subscriptions = []
        for name in ("left", "center", "right", "overhead", "side"):
            topic = f"/{name}_camera/image" if name in {"left", "center", "right"} else f"/audit/{name}/image"
            self._audit_subscriptions.append(self.create_subscription(
                Image, topic, lambda msg, camera=name: self.save(camera, msg),
                qos_profile_sensor_data))

    def save(self, camera, msg):
        sim_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if sim_ns <= self.last_sim_ns.get(camera, -1):
            return
        self.last_sim_ns[camera] = sim_ns
        wall_ns = time.time_ns()
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        filename = f"{wall_ns}_{sim_ns}_{camera}.jpg"
        if not cv2.imwrite(str(self.directory / filename), image,
                           [cv2.IMWRITE_JPEG_QUALITY, 86]):
            raise RuntimeError(f"Could not save {filename}")
        self.writer.writerow((camera, wall_ns, sim_ns, filename))


def main():
    rclpy.init()
    node = Capture()
    try:
        rclpy.spin(node)
    finally:
        node.csv_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
