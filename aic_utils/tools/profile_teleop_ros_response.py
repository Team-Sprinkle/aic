#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import time

import rclpy
from aic_control_interfaces.msg import ControllerState, MotionUpdate
from rclpy.node import Node
from sensor_msgs.msg import JointState


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def summarize_intervals(name: str, stamps_ns: list[int]) -> None:
    if len(stamps_ns) < 2:
        print(f"{name}: not enough samples")
        return
    intervals_ms = [
        (b - a) / 1_000_000.0 for a, b in zip(stamps_ns, stamps_ns[1:]) if b >= a
    ]
    if not intervals_ms:
        print(f"{name}: no valid intervals")
        return
    duration_sec = sum(intervals_ms) / 1000.0
    hz = len(intervals_ms) / duration_sec if duration_sec > 0.0 else 0.0
    print(
        f"{name}: count={len(stamps_ns)} hz={hz:.3f} "
        f"interval_mean={statistics.fmean(intervals_ms):.3f}ms "
        f"p95={percentile(intervals_ms, 95):.3f}ms "
        f"p99={percentile(intervals_ms, 99):.3f}ms "
        f"max={max(intervals_ms):.3f}ms"
    )


class TeleopRosResponseProfiler(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("teleop_ros_response_profiler")
        self.args = args
        self.start_monotonic = time.monotonic()
        self.command_stamps_ns: list[int] = []
        self.state_stamps_ns: list[int] = []
        self.latest_command: MotionUpdate | None = None
        self.latest_command_recv_ns = 0
        self.latest_joint_state: JointState | None = None

        self.csv_file = args.out.open("w", newline="")
        self.writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "event",
                "wall_time_ns",
                "age_since_last_command_ns",
                "cmd_linear_x",
                "cmd_linear_y",
                "cmd_linear_z",
                "cmd_angular_x",
                "cmd_angular_y",
                "cmd_angular_z",
                "state_tcp_linear_x",
                "state_tcp_linear_y",
                "state_tcp_linear_z",
                "state_tcp_angular_x",
                "state_tcp_angular_y",
                "state_tcp_angular_z",
                "tcp_error_x",
                "tcp_error_y",
                "tcp_error_z",
                "tcp_error_rx",
                "tcp_error_ry",
                "tcp_error_rz",
                "joint_shoulder_pan",
                "joint_shoulder_lift",
                "joint_elbow",
                "joint_wrist_1",
                "joint_wrist_2",
                "joint_wrist_3",
                "elbow_abs_rad",
                "shoulder_elbow_wrist1_sum",
                "cmd_to_state_linear_x_ratio",
                "cmd_to_state_linear_y_ratio",
                "cmd_to_state_linear_z_ratio",
                "cmd_to_state_angular_x_ratio",
                "cmd_to_state_angular_y_ratio",
                "cmd_to_state_angular_z_ratio",
            ],
        )
        self.writer.writeheader()

        self.create_subscription(
            MotionUpdate, args.command_topic, self.command_callback, 10
        )
        self.create_subscription(
            ControllerState, args.controller_state_topic, self.state_callback, 10
        )
        self.create_subscription(JointState, args.joint_states_topic, self.joint_callback, 10)

    def command_callback(self, msg: MotionUpdate) -> None:
        now_ns = time.time_ns()
        self.latest_command = msg
        self.latest_command_recv_ns = now_ns
        self.command_stamps_ns.append(now_ns)
        self.writer.writerow(
            {
                "event": "command",
                "wall_time_ns": now_ns,
                "age_since_last_command_ns": 0,
                "cmd_linear_x": msg.velocity.linear.x,
                "cmd_linear_y": msg.velocity.linear.y,
                "cmd_linear_z": msg.velocity.linear.z,
                "cmd_angular_x": msg.velocity.angular.x,
                "cmd_angular_y": msg.velocity.angular.y,
                "cmd_angular_z": msg.velocity.angular.z,
            }
        )

    def state_callback(self, msg: ControllerState) -> None:
        now_ns = time.time_ns()
        self.state_stamps_ns.append(now_ns)
        age_ns = now_ns - self.latest_command_recv_ns if self.latest_command else -1
        tcp_error = list(getattr(msg, "tcp_error", []))
        tcp_error.extend([0.0] * (6 - len(tcp_error)))
        joints = self._joint_positions_by_name()
        latest_command = self.latest_command
        self.writer.writerow(
            {
                "event": "controller_state",
                "wall_time_ns": now_ns,
                "age_since_last_command_ns": age_ns,
                "state_tcp_linear_x": msg.tcp_velocity.linear.x,
                "state_tcp_linear_y": msg.tcp_velocity.linear.y,
                "state_tcp_linear_z": msg.tcp_velocity.linear.z,
                "state_tcp_angular_x": msg.tcp_velocity.angular.x,
                "state_tcp_angular_y": msg.tcp_velocity.angular.y,
                "state_tcp_angular_z": msg.tcp_velocity.angular.z,
                "tcp_error_x": tcp_error[0],
                "tcp_error_y": tcp_error[1],
                "tcp_error_z": tcp_error[2],
                "tcp_error_rx": tcp_error[3],
                "tcp_error_ry": tcp_error[4],
                "tcp_error_rz": tcp_error[5],
                "joint_shoulder_pan": joints.get("shoulder_pan_joint", ""),
                "joint_shoulder_lift": joints.get("shoulder_lift_joint", ""),
                "joint_elbow": joints.get("elbow_joint", ""),
                "joint_wrist_1": joints.get("wrist_1_joint", ""),
                "joint_wrist_2": joints.get("wrist_2_joint", ""),
                "joint_wrist_3": joints.get("wrist_3_joint", ""),
                "elbow_abs_rad": abs(joints.get("elbow_joint", 0.0)),
                "shoulder_elbow_wrist1_sum": (
                    joints.get("shoulder_lift_joint", 0.0)
                    + joints.get("elbow_joint", 0.0)
                    + joints.get("wrist_1_joint", 0.0)
                ),
                "cmd_to_state_linear_x_ratio": self._ratio(
                    msg.tcp_velocity.linear.x,
                    latest_command.velocity.linear.x if latest_command else 0.0,
                ),
                "cmd_to_state_linear_y_ratio": self._ratio(
                    msg.tcp_velocity.linear.y,
                    latest_command.velocity.linear.y if latest_command else 0.0,
                ),
                "cmd_to_state_linear_z_ratio": self._ratio(
                    msg.tcp_velocity.linear.z,
                    latest_command.velocity.linear.z if latest_command else 0.0,
                ),
                "cmd_to_state_angular_x_ratio": self._ratio(
                    msg.tcp_velocity.angular.x,
                    latest_command.velocity.angular.x if latest_command else 0.0,
                ),
                "cmd_to_state_angular_y_ratio": self._ratio(
                    msg.tcp_velocity.angular.y,
                    latest_command.velocity.angular.y if latest_command else 0.0,
                ),
                "cmd_to_state_angular_z_ratio": self._ratio(
                    msg.tcp_velocity.angular.z,
                    latest_command.velocity.angular.z if latest_command else 0.0,
                ),
            }
        )

    def joint_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _joint_positions_by_name(self) -> dict[str, float]:
        if self.latest_joint_state is None:
            return {}
        return {
            name: float(position)
            for name, position in zip(
                self.latest_joint_state.name, self.latest_joint_state.position
            )
        }

    @staticmethod
    def _ratio(actual: float, commanded: float) -> float | str:
        if abs(commanded) < 1e-6:
            return ""
        return float(actual) / float(commanded)

    def done(self) -> bool:
        return (time.monotonic() - self.start_monotonic) >= self.args.duration_sec

    def close(self) -> None:
        self.csv_file.flush()
        self.csv_file.close()
        summarize_intervals("pose_commands", self.command_stamps_ns)
        summarize_intervals("controller_state", self.state_stamps_ns)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile teleop ROS command and controller-state timing."
    )
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--command-topic", default="/aic_controller/pose_commands")
    parser.add_argument(
        "--controller-state-topic", default="/aic_controller/controller_state"
    )
    parser.add_argument("--joint-states-topic", default="/joint_states")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    node = TeleopRosResponseProfiler(args)
    try:
        while rclpy.ok() and not node.done():
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
