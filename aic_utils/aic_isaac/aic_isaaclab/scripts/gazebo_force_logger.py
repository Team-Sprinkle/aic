#!/usr/bin/env python3
"""Log Gazebo observation wrench/TCP data to CSV for force parity checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import rclpy
from aic_model_interfaces.msg import Observation
from rclpy.node import Node


class GazeboForceLogger(Node):
    def __init__(self, out_csv: Path, duration_s: float) -> None:
        super().__init__("gazebo_force_logger")
        self._out_csv = out_csv
        self._duration_s = duration_s
        self._start_time = None
        self._rows: list[tuple[float, float, float, float, float]] = []
        self._sub = self.create_subscription(
            Observation, "/observations", self._on_observation, 50
        )
        self.get_logger().info(
            f"Logging /observations to {self._out_csv} for up to {self._duration_s:.2f}s"
        )

    def _on_observation(self, msg: Observation) -> None:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        if self._start_time is None:
            self._start_time = now_s
        t = now_s - self._start_time
        force_z = float(msg.wrist_wrench.wrench.force.z)
        tcp = msg.controller_state.tcp_pose.position
        self._rows.append((t, force_z, float(tcp.x), float(tcp.y), float(tcp.z)))
        if t >= self._duration_s:
            self._write_and_shutdown()

    def _write_and_shutdown(self) -> None:
        self._out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self._out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "force_z_n", "ee_x_m", "ee_y_m", "ee_z_m"])
            for row in self._rows:
                writer.writerow(
                    [
                        f"{row[0]:.6f}",
                        f"{row[1]:.8f}",
                        f"{row[2]:.8f}",
                        f"{row[3]:.8f}",
                        f"{row[4]:.8f}",
                    ]
                )
        self.get_logger().info(
            f"Wrote {len(self._rows)} samples to {self._out_csv}; shutting down."
        )
        rclpy.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--duration-s", type=float, default=20.0)
    args = parser.parse_args()

    rclpy.init()
    node = GazeboForceLogger(Path(args.out), args.duration_s)
    rclpy.spin(node)
    node.destroy_node()


if __name__ == "__main__":
    main()

