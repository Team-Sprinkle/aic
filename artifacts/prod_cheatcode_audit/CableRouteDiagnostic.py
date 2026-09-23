"""Privileged, fixed-scene cable-route diagnostic; never use for evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.time import Time

from aic_example_policies.ros.CheatCode import CheatCode


class CableRouteDiagnostic(CheatCode):
    """Drive a deliberately long path, then hand off to stock CheatCode.

    Card positions and target geometry come from simulator TF and are only used
    in this diagnostic. The actor being trained never receives these labels.
    """

    def _pose(self, xyz: np.ndarray, orientation: Quaternion) -> Pose:
        return Pose(position=Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])),
                    orientation=orientation)

    def _tcp(self) -> Pose:
        tr = self._parent_node._tf_buffer.lookup_transform("base_link", "gripper/tcp", Time()).transform
        return Pose(position=Point(x=tr.translation.x, y=tr.translation.y, z=tr.translation.z),
                    orientation=tr.rotation)

    def _card_xyz(self, task) -> list[list[float]]:
        result = []
        for index in range(5):
            frame = f"task_board/nic_card_mount_{index}/nic_card_link"
            if not self._wait_for_tf("base_link", frame, timeout_sec=2.0):
                continue
            tr = self._parent_node._tf_buffer.lookup_transform("base_link", frame, Time()).transform
            result.append([tr.translation.x, tr.translation.y, tr.translation.z])
        return result

    def _drive(self, move_robot, waypoint: np.ndarray, duration: float) -> None:
        start_pose = self._tcp()
        start = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z], dtype=float)
        steps = max(1, round(duration / 0.05))
        for step in range(steps + 1):
            u = step / steps
            smooth = 10 * u**3 - 15 * u**4 + 6 * u**5
            target = start + smooth * (waypoint - start)
            self.set_pose_target(move_robot, self._pose(target, start_pose.orientation))
            self.sleep_for(0.05)
        for _ in range(10):
            self.set_pose_target(move_robot, self._pose(waypoint, start_pose.orientation))
            self.sleep_for(0.05)

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self._task = task
        variant = os.environ.get("AIC_CABLE_ROUTE_VARIANT", "across_cards")
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        if not self._wait_for_tf("base_link", port_frame):
            return False
        port = self._parent_node._tf_buffer.lookup_transform("base_link", port_frame, Time()).transform
        start_pose = self._tcp()
        start = np.array([start_pose.position.x, start_pose.position.y, start_pose.position.z], dtype=float)
        cards = np.asarray(self._card_xyz(task), dtype=float)
        if len(cards) != 5:
            self.get_logger().error(f"Expected five visible card frames, got {len(cards)}")
            return False
        card_min_y, card_max_y = float(cards[:, 1].min()), float(cards[:, 1].max())
        if variant == "across_cards":
            lane_x = float(np.median(cards[:, 0]))
        elif variant == "outside_left":
            lane_x = float(cards[:, 0].min() - 0.08)
        else:
            raise ValueError(f"Unknown diagnostic route variant {variant}")
        high_z = max(float(start[2]) + 0.02, float(cards[:, 2].max()) + 0.21)
        low_offset = float(os.environ.get("AIC_CABLE_ROUTE_LOW_OFFSET_M", "0.105"))
        low_z = float(cards[:, 2].max()) + low_offset
        port_xyz = np.array([port.translation.x, port.translation.y, port.translation.z], dtype=float)
        waypoints = [
            ("lift", np.array([start[0], start[1], high_z]), 2.0),
            ("return_behind_cards", np.array([lane_x, card_min_y - 0.035, high_z]), 4.5),
            ("lower_along_lane", np.array([lane_x, card_min_y - 0.035, low_z]), 4.0),
            ("traverse_card_row", np.array([lane_x, card_max_y + 0.045, low_z]), 8.0),
            ("lift_after_row", np.array([lane_x, card_max_y + 0.045, high_z]), 3.0),
            ("above_port", np.array([port_xyz[0], port_xyz[1], port_xyz[2] + 0.20]), 4.0),
        ]
        plan = {"schema": "aic_cable_route_diagnostic/v1", "variant": variant,
                "start_tcp_base_m": start.tolist(), "port_link_base_m": port_xyz.tolist(),
                "nic_card_centers_base_m": cards.tolist(), "lane_x_m": lane_x,
                "low_offset_m": low_offset,
                "waypoints": [{"name": name, "tcp_base_m": xyz.tolist(), "duration_s": sec}
                              for name, xyz, sec in waypoints]}
        Path("/audit/route_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        self.get_logger().info(f"Diagnostic cable route: {variant}, lane x={lane_x:.3f} m")
        for name, xyz, seconds in waypoints:
            self.get_logger().info(f"Diagnostic waypoint {name}: {xyz.tolist()}")
            self._drive(move_robot, xyz, seconds)
        self.get_logger().info("Diagnostic route complete; handing off to stock CheatCode insertion")
        return super().insert_cable(task, get_observation, move_robot, send_feedback)
