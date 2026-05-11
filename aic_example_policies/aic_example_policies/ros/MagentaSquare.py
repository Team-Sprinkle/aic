#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

import cv2
import numpy as np
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import Point, Pose, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import TransformException


@dataclass
class BoardEdgePixels:
    p0_px: np.ndarray
    p1_px: np.ndarray
    inward_px: np.ndarray
    score: float


@dataclass
class BoardEdgeCandidate:
    camera_name: str
    p0_base: np.ndarray
    p1_base: np.ndarray
    center_base: np.ndarray
    width_base: float
    score: float


@dataclass
class BoardEdgeObservation:
    camera_name: str
    p0_base: np.ndarray
    p1_base: np.ndarray
    inward_base: np.ndarray
    width_base: float
    endpoint_margin_px: float
    score: float


@dataclass
class FusedBoardEdge:
    center_base: np.ndarray
    board_x_axis: np.ndarray
    board_y_axis: np.ndarray
    reconstructed_p0_base: np.ndarray
    reconstructed_p1_base: np.ndarray
    observed_width_base: float


@dataclass
class MagentaMarkerObservation:
    camera_name: str
    marker_center_base: np.ndarray
    board_origin_base: np.ndarray
    board_x_axis: np.ndarray
    board_y_axis: np.ndarray
    gap_edge_center_base: np.ndarray
    plus_y_edge_center_base: np.ndarray
    score: float
    area_px: float


@dataclass
class MagentaRoiObservation:
    camera_name: str
    marker_center_base: np.ndarray
    marker_center_px: np.ndarray
    score: float
    area_px: float


class MagentaSquare(Policy):
    """Transport the TCP using a board pose estimated from the magenta marker.

    The policy uses only camera observations and fixed task geometry. It assumes
    the task board is flat on a known plane in base_link (`board_plane_z`). The
    primary board pose cue is the magenta square marker in board-frame -x/+y.
    If marker detection fails, the policy falls back to the existing short-edge
    board estimator.

    Runtime flow:
    1. Move the TCP up to a high z view while preserving x/y and orientation.
    2. Detect the magenta square marker in the left, center, and right camera
       images and project marker points onto the marker plane.
    3. Recover the board pose from the marker geometry. The disconnected marker
       edge is the square -y side; the opposite edge is used to locate the board
       +y boundary.
    4. If marker detection fails, try the existing multi-camera short-edge fit.
    5. Move to the NIC rail target in the recovered board frame.
    6. Optionally tilt the TCP in place by the minimal rotation needed to align
       the estimated SFP tip +z axis with base_link -z for downward insertion.
    7. Descend straight down in base_link at a fixed rate, stopping early if
       the scoring insertion event is observed.
    """

    BOARD_SIZE_X = 0.30
    BOARD_SIZE_Y = 0.425
    NIC_RAIL_X = -0.081418
    NIC_RAIL_Y0 = -0.1745
    NIC_RAIL_Y_SPACING = 0.04
    NIC_RAIL_APPROACH_Z = 0.133476
    SFP_PORT_X_OFFSETS = {
        0: 0.01295,
        1: -0.01025,
    }

    MAGENTA_SQUARE_SIZE = 0.095
    MAGENTA_CENTER_BOARD = np.array([-0.075, 0.150, 0.011], dtype=np.float64)
    MAGENTA_GAP_EDGE_BOARD_Y = 0.1025
    MAGENTA_PLUS_Y_EDGE_BOARD_Y = 0.1975
    MAGENTA_PLUS_Y_BOUNDARY_OFFSET = 0.015

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._latest_insertion_event_namespace = ""
        self._insertion_event_sub = self._parent_node.create_subscription(
            String,
            "/scoring/insertion_event",
            self._insertion_event_callback,
            10,
        )
        self.z_offset = float(os.getenv("AIC_TRANSPORT_MEAN_Z_OFFSET", "0.12"))
        self.duration_sec = float(os.getenv("AIC_TRANSPORT_MEAN_DURATION_SEC", "3"))
        self.dt = float(os.getenv("AIC_TRANSPORT_MEAN_DT", "0.05"))
        self.hold_sec = float(os.getenv("AIC_TRANSPORT_MEAN_HOLD_SEC", "10.0"))
        self.min_target_z = float(os.getenv("AIC_TRANSPORT_MIN_TARGET_Z", "0.25"))
        self.xy_then_z_transport = self._env_flag(
            "AIC_TRANSPORT_XY_THEN_Z_TRANSPORT",
            default=True,
        )
        self.xy_transport_duration_sec = float(
            os.getenv("AIC_TRANSPORT_XY_DURATION_SEC", str(self.duration_sec))
        )
        self.z_transport_duration_sec = float(
            os.getenv("AIC_TRANSPORT_Z_DURATION_SEC", str(self.duration_sec))
        )
        self.detect_board = self._env_flag("AIC_TRANSPORT_DETECT_BOARD", default=True)
        self.detect_only = self._env_flag("AIC_TRANSPORT_DETECT_ONLY", default=False)
        self.move_to_view_first = self._env_flag(
            "AIC_TRANSPORT_MOVE_TO_VIEW_FIRST",
            default=True,
        )
        self.magenta_view_z = float(os.getenv("AIC_MAGENTA_VIEW_Z", "0.45"))
        self.magenta_min_area_px = float(os.getenv("AIC_MAGENTA_MIN_AREA_PX", "120.0"))
        self.magenta_max_area_px = float(os.getenv("AIC_MAGENTA_MAX_AREA_PX", "80000.0"))
        self.magenta_edge_support_tolerance_m = float(
            os.getenv("AIC_MAGENTA_EDGE_SUPPORT_TOLERANCE_M", "0.008")
        )
        self.magenta_square_size_tolerance_m = float(
            os.getenv("AIC_MAGENTA_SQUARE_SIZE_TOLERANCE_M", "0.04")
        )
        self.magenta_min_side_ratio = float(
            os.getenv("AIC_MAGENTA_MIN_SIDE_RATIO", "0.75")
        )
        self.magenta_consensus_distance_m = float(
            os.getenv("AIC_MAGENTA_CONSENSUS_DISTANCE_M", "0.08")
        )
        self.magenta_edge_width_tolerance_m = float(
            os.getenv("AIC_MAGENTA_BOARD_EDGE_WIDTH_TOLERANCE_M", "0.07")
        )
        self.magenta_edge_marker_distance_tolerance_m = float(
            os.getenv("AIC_MAGENTA_BOARD_EDGE_MARKER_DISTANCE_TOLERANCE_M", "0.12")
        )
        self.magenta_linear_view_enabled = self._env_flag(
            "AIC_MAGENTA_LINEAR_VIEW_ENABLED",
            default=True,
        )
        self.magenta_linear_view_y_distance_m = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_Y_DISTANCE_M", "0.7")
        )
        self.magenta_linear_view_steps = int(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_STEPS", "7")
        )
        self.magenta_linear_view_move_duration_sec = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_MOVE_DURATION_SEC", "0.45")
        )
        self.magenta_linear_view_hold_sec = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_HOLD_SEC", "0.15")
        )
        self.extend_visible_edge_toward_top = self._env_flag(
            "AIC_TRANSPORT_EXTEND_EDGE_TOWARD_TOP",
            default=True,
        )
        self.sort_short_edge_left_to_right = self._env_flag(
            "AIC_TRANSPORT_SORT_SHORT_EDGE_LEFT_TO_RIGHT",
            default=True,
        )
        self.view_z = float(os.getenv("AIC_TRANSPORT_VIEW_Z", "0.0"))
        self.board_plane_z = float(os.getenv("AIC_TRANSPORT_BOARD_PLANE_Z", "0.0"))
        self.magenta_marker_plane_z = float(
            os.getenv(
                "AIC_MAGENTA_MARKER_PLANE_Z",
                str(self.board_plane_z + self.MAGENTA_CENTER_BOARD[2]),
            )
        )
        self.visible_short_edge_board_y = float(
            os.getenv(
                "AIC_TRANSPORT_VISIBLE_SHORT_EDGE_BOARD_Y",
                str(-0.5 * self.BOARD_SIZE_Y),
            )
        )
        self.flip_board_x_axis = self._env_flag(
            "AIC_TRANSPORT_FLIP_BOARD_X_AXIS",
            default=False,
        )
        self.board_pose_target_mode = os.getenv(
            "AIC_TRANSPORT_BOARD_POSE_TARGET_MODE",
            "nic_rail",
        ).strip().lower()
        self.nic_rail_approach_z = float(
            os.getenv(
                "AIC_TRANSPORT_NIC_RAIL_APPROACH_Z",
                str(self.NIC_RAIL_APPROACH_Z),
            )
        )
        self.nic_rail_x_offset = float(os.getenv("AIC_TRANSPORT_NIC_RAIL_X_OFFSET", "0.0"))
        self.nic_rail_y_offset = float(os.getenv("AIC_TRANSPORT_NIC_RAIL_Y_OFFSET", "0.0"))
        self.nic_rail_use_port_x_offset = self._env_flag(
            "AIC_TRANSPORT_NIC_RAIL_USE_PORT_X_OFFSET",
            default=False,
        )
        self.short_edge_scan_y_step = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_Y_STEP", "-0.04")
        )
        self.short_edge_scan_max_steps = int(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_MAX_STEPS", "4")
        )
        self.short_edge_scan_duration_sec = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_DURATION_SEC", "0.8")
        )
        self.short_edge_scan_hold_sec = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_HOLD_SEC", "0.2")
        )
        self.short_edge_scan_z_offset = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_Z_OFFSET", "0.03")
        )
        self.short_edge_min_width = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_MIN_WIDTH", "0.24")
        )
        self.short_edge_width_tolerance = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_WIDTH_TOLERANCE", "0.08")
        )
        self.short_edge_min_endpoint_margin_px = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_MIN_ENDPOINT_MARGIN_PX", "20.0")
        )
        self.short_edge_min_observations = int(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_MIN_OBSERVATIONS", "1")
        )
        self.reconstruct_short_edge_width = self._env_flag(
            "AIC_TRANSPORT_RECONSTRUCT_SHORT_EDGE_WIDTH",
            default=True,
        )
        self.tilt_tip_down_after_transport = self._env_flag(
            "AIC_TRANSPORT_TILT_TIP_DOWN_AFTER_TRANSPORT",
            default=True,
        )
        self.tilt_tip_duration_sec = float(
            os.getenv("AIC_TRANSPORT_TILT_TIP_DURATION_SEC", "1.0")
        )
        self.tip_axis_tcp = self._normalized_vector(
            np.array(
                [
                    float(os.getenv("AIC_TRANSPORT_TIP_AXIS_TCP_X", "0.000136")),
                    float(os.getenv("AIC_TRANSPORT_TIP_AXIS_TCP_Y", "-0.350201")),
                    float(os.getenv("AIC_TRANSPORT_TIP_AXIS_TCP_Z", "0.936674")),
                ],
                dtype=np.float64,
            )
        )
        if self.tip_axis_tcp is None:
            self.tip_axis_tcp = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.descend_after_transport = self._env_flag(
            "AIC_TRANSPORT_DESCEND_AFTER_TRANSPORT",
            default=True,
        )
        self.descend_step_m = float(os.getenv("AIC_TRANSPORT_DESCEND_STEP_M", "0.0005"))
        self.descend_dt_sec = float(os.getenv("AIC_TRANSPORT_DESCEND_DT_SEC", "0.05"))
        self.descend_max_distance_m = float(
            os.getenv(
                "AIC_TRANSPORT_DESCEND_MAX_DISTANCE_M",
                str(max(self.z_offset + 0.015, 0.0)),
            )
        )
        self.descend_wait_for_insertion_sec = float(
            os.getenv("AIC_TRANSPORT_DESCEND_WAIT_FOR_INSERTION_SEC", "5.0")
        )
        self.trial_config_path = os.getenv("AIC_TRANSPORT_TRIAL_CONFIG", "").strip()
        self.trial_config_search_glob = os.getenv(
            "AIC_TRANSPORT_TRIAL_CONFIG_GLOB",
            "outputs/configs/trial_*.yaml",
        ).strip()
        self.robot_world_x = float(os.getenv("AIC_TRANSPORT_ROBOT_WORLD_X", "-0.2"))
        self.robot_world_y = float(os.getenv("AIC_TRANSPORT_ROBOT_WORLD_Y", "0.2"))
        self.robot_world_z = float(os.getenv("AIC_TRANSPORT_ROBOT_WORLD_Z", "1.14"))
        self.robot_world_yaw = float(
            os.getenv("AIC_TRANSPORT_ROBOT_WORLD_YAW", str(-np.pi))
        )

    @staticmethod
    def _env_flag(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    def _insertion_event_callback(self, msg: String) -> None:
        self._latest_insertion_event_namespace = msg.data.strip().strip("/")
        self.get_logger().info(
            "TransportToMean received insertion event for namespace: "
            f"'{self._latest_insertion_event_namespace}'"
        )

    def _task_completed_in_simulation(self, task: Task) -> bool:
        namespace = self._latest_insertion_event_namespace
        if not namespace:
            return False
        tokens = [token for token in namespace.split("/") if token]
        if len(tokens) < 2:
            return False
        return tokens[-2] == task.target_module_name and tokens[-1] == task.port_name

    @staticmethod
    def _parse_index_from_suffix(value: str, prefix: str) -> int:
        if not value.startswith(prefix):
            raise ValueError(f"Expected {value!r} to start with {prefix!r}")
        return int(value.removeprefix(prefix))

    def _task_indices(self, task: Task) -> tuple[int, int]:
        card_index = self._parse_index_from_suffix(
            task.target_module_name,
            "nic_card_mount_",
        )
        port_index = self._parse_index_from_suffix(task.port_name, "sfp_port_")
        return card_index, port_index

    def _nic_rail_xyz_board(self, task: Task) -> tuple[float, float, float]:
        card_index, port_index = self._task_indices(task)
        if not 0 <= card_index <= 4 or port_index not in self.SFP_PORT_X_OFFSETS:
            raise ValueError(
                "Unsupported NIC rail target "
                f"card={card_index}, port={port_index}; expected card 0..4 and port 0..1"
            )

        port_x_offset = (
            self.SFP_PORT_X_OFFSETS[port_index]
            if self.nic_rail_use_port_x_offset
            else 0.0
        )
        return (
            self.NIC_RAIL_X + self.nic_rail_x_offset + port_x_offset,
            self.NIC_RAIL_Y0 + self.NIC_RAIL_Y_SPACING * card_index + self.nic_rail_y_offset,
            self.nic_rail_approach_z,
        )

    def _board_pose_target_xyz_board(self, task: Task) -> tuple[float, float, float]:
        if self.board_pose_target_mode in ("nic_rail", "rail"):
            return self._nic_rail_xyz_board(task)
        raise ValueError(
            "Unsupported AIC_TRANSPORT_BOARD_POSE_TARGET_MODE "
            f"{self.board_pose_target_mode!r}; expected 'nic_rail'"
        )

    @staticmethod
    def _tcp_pose_from_observation(get_observation: GetObservationCallback) -> Pose:
        tcp_pose = get_observation().controller_state.tcp_pose
        return Pose(
            position=Point(
                x=tcp_pose.position.x,
                y=tcp_pose.position.y,
                z=tcp_pose.position.z,
            ),
            orientation=Quaternion(
                x=tcp_pose.orientation.x,
                y=tcp_pose.orientation.y,
                z=tcp_pose.orientation.z,
                w=tcp_pose.orientation.w,
            ),
        )

    def _interpolated_pose(
        self,
        start_pose: Pose,
        target_xyz: tuple[float, float, float],
        fraction: float,
        target_orientation: Quaternion | None = None,
    ) -> Pose:
        if target_orientation is None:
            orientation = start_pose.orientation
        else:
            q_start = np.array(
                [
                    start_pose.orientation.x,
                    start_pose.orientation.y,
                    start_pose.orientation.z,
                    start_pose.orientation.w,
                ],
                dtype=np.float64,
            )
            q_target = np.array(
                [
                    target_orientation.x,
                    target_orientation.y,
                    target_orientation.z,
                    target_orientation.w,
                ],
                dtype=np.float64,
            )
            q_interp = self._quat_slerp(q_start, q_target, fraction)
            orientation = Quaternion(
                x=float(q_interp[0]),
                y=float(q_interp[1]),
                z=float(q_interp[2]),
                w=float(q_interp[3]),
            )
        return Pose(
            position=Point(
                x=start_pose.position.x
                + fraction * (target_xyz[0] - start_pose.position.x),
                y=start_pose.position.y
                + fraction * (target_xyz[1] - start_pose.position.y),
                z=start_pose.position.z
                + fraction * (target_xyz[2] - start_pose.position.z),
            ),
            orientation=orientation,
        )

    def _move_to_pose(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        target_xyz: tuple[float, float, float],
        *,
        duration_sec: float,
        target_orientation: Quaternion | None = None,
    ) -> Pose:
        steps = max(1, int(duration_sec / self.dt))
        for step in range(steps + 1):
            fraction = step / steps
            pose = self._interpolated_pose(
                start_pose,
                target_xyz,
                fraction,
                target_orientation,
            )
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(self.dt)
        return self._interpolated_pose(start_pose, target_xyz, 1.0, target_orientation)

    def _move_to_transport_target_pose(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        target_xyz: tuple[float, float, float],
    ) -> Pose:
        if not self.xy_then_z_transport:
            return self._move_to_pose(
                move_robot,
                start_pose,
                target_xyz,
                duration_sec=self.duration_sec,
            )

        xy_target_xyz = (
            target_xyz[0],
            target_xyz[1],
            start_pose.position.z,
        )
        xy_distance = float(
            np.linalg.norm(
                np.array(
                    [
                        target_xyz[0] - start_pose.position.x,
                        target_xyz[1] - start_pose.position.y,
                    ],
                    dtype=np.float64,
                )
            )
        )
        z_distance = abs(float(target_xyz[2] - start_pose.position.z))
        self.get_logger().info(
            "TransportToMean moving to transport target as XY-then-Z path: "
            f"xy_target={[round(value, 5) for value in xy_target_xyz]}, "
            f"final_target={[round(value, 5) for value in target_xyz]}, "
            f"xy_distance={xy_distance:.5f} m, z_distance={z_distance:.5f} m"
        )

        current_pose = start_pose
        if xy_distance > 1e-6:
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                xy_target_xyz,
                duration_sec=self.xy_transport_duration_sec,
            )
        if z_distance > 1e-6:
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                target_xyz,
                duration_sec=self.z_transport_duration_sec,
            )
        return current_pose

    def _move_to_tip_down_pose(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
    ) -> Pose:
        q_tcp = self._quat_normalized(
            np.array(
                [
                    start_pose.orientation.x,
                    start_pose.orientation.y,
                    start_pose.orientation.z,
                    start_pose.orientation.w,
                ],
                dtype=np.float64,
            )
        )
        tcp_rot_base = self._quat_xyzw_to_rot(q_tcp)
        tip_axis_base = self._normalized_vector(tcp_rot_base @ self.tip_axis_tcp)
        target_axis_base = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        if tip_axis_base is None:
            self.get_logger().warn(
                "TransportToMean tip-down tilt skipped: configured tip axis is invalid."
            )
            return start_pose

        q_delta = self._quat_from_two_vectors(tip_axis_base, target_axis_base)
        q_target = self._quat_multiply(q_delta, q_tcp)
        correction_angle = float(
            np.arccos(np.clip(np.dot(tip_axis_base, target_axis_base), -1.0, 1.0))
        )
        target_orientation = Quaternion(
            x=float(q_target[0]),
            y=float(q_target[1]),
            z=float(q_target[2]),
            w=float(q_target[3]),
        )
        self.get_logger().info(
            "TransportToMean tilting TCP to align cable tip with base -z: "
            f"tip_axis_tcp={np.round(self.tip_axis_tcp, 5).tolist()}, "
            f"tip_axis_base={np.round(tip_axis_base, 5).tolist()}, "
            f"correction_angle={correction_angle:.5f} rad "
            f"({np.degrees(correction_angle):.2f} deg), "
            f"target_orientation_xyzw={np.round(q_target, 5).tolist()}"
        )
        return self._move_to_pose(
            move_robot,
            start_pose,
            (
                start_pose.position.x,
                start_pose.position.y,
                start_pose.position.z,
            ),
            duration_sec=self.tilt_tip_duration_sec,
            target_orientation=target_orientation,
        )

    def _descend_until_inserted(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
    ) -> Pose:
        step_m = abs(self.descend_step_m)
        if step_m <= 0.0 or self.descend_dt_sec <= 0.0 or self.descend_max_distance_m <= 0.0:
            self.get_logger().info(
                "TransportToMean descent skipped because descent step, dt, or max "
                "distance is non-positive."
            )
            return start_pose

        max_steps = int(np.ceil(self.descend_max_distance_m / step_m))
        current_pose = start_pose
        self.get_logger().info(
            "TransportToMean descending TCP in base -z: "
            f"step={step_m:.5f} m, dt={self.descend_dt_sec:.3f} s, "
            f"max_distance={self.descend_max_distance_m:.5f} m, "
            f"rate={step_m / self.descend_dt_sec:.5f} m/s"
        )
        for step in range(1, max_steps + 1):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "TransportToMean descent early exit: simulation reported "
                    "task completion."
                )
                return current_pose

            distance = min(step * step_m, self.descend_max_distance_m)
            current_pose = Pose(
                position=Point(
                    x=start_pose.position.x,
                    y=start_pose.position.y,
                    z=start_pose.position.z - distance,
                ),
                orientation=start_pose.orientation,
            )
            self.set_pose_target(move_robot=move_robot, pose=current_pose)
            if step == 1 or step % 20 == 0 or step == max_steps:
                self.get_logger().info(
                    "TransportToMean descent: "
                    f"step={step}/{max_steps}, distance={distance:.5f} m, "
                    f"target_z={current_pose.position.z:.5f}"
                )
            self.sleep_for(self.descend_dt_sec)

        if self.descend_wait_for_insertion_sec > 0.0:
            self.get_logger().info("TransportToMean waiting briefly for insertion event.")
            wait_started = self.time_now()
            wait_timeout = Duration(seconds=self.descend_wait_for_insertion_sec)
            while (self.time_now() - wait_started) < wait_timeout:
                if self._task_completed_in_simulation(task):
                    self.get_logger().info(
                        "TransportToMean insertion event observed before timeout."
                    )
                    break
                self.sleep_for(min(self.descend_dt_sec, 0.05))

        return current_pose

    def _move_to_view_pose(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> Pose:
        start_pose = self._tcp_pose_from_observation(get_observation)
        target_xyz = (
            start_pose.position.x,
            start_pose.position.y,
            self.magenta_view_z,
        )
        self.get_logger().info(
            "MagentaSquare moving to high marker view pose: "
            f"start_z={start_pose.position.z:.4f}, target_z={self.magenta_view_z:.4f}"
        )
        return self._move_to_pose(
            move_robot,
            start_pose,
            target_xyz,
            duration_sec=self.duration_sec,
        )

    def _short_edge_visibility(
        self,
        obs,
    ) -> tuple[list[BoardEdgeObservation], int, float, float]:
        observations = self._multicamera_board_edge_observations(obs)
        if not observations:
            return [], 0, 0.0, 0.0

        clear_observations = [
            observation
            for observation in observations
            if observation.width_base >= self.short_edge_min_width
            and abs(observation.width_base - self.BOARD_SIZE_X)
            <= self.short_edge_width_tolerance
            and observation.endpoint_margin_px >= self.short_edge_min_endpoint_margin_px
        ]
        widest = max(observation.width_base for observation in observations)
        best_margin = max(observation.endpoint_margin_px for observation in observations)
        return observations, len(clear_observations), widest, best_margin

    def _move_to_short_edge_view(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> None:
        start_pose = self._tcp_pose_from_observation(get_observation)
        scan_z = max(start_pose.position.z + self.short_edge_scan_z_offset, self.view_z)
        current_pose = start_pose
        self.get_logger().info(
            "TransportToMean scanning for full short-edge view "
            f"start_y={start_pose.position.y:.4f}, y_step={self.short_edge_scan_y_step:.4f}, "
            f"max_steps={self.short_edge_scan_max_steps}, z={scan_z:.4f}, "
            f"z_offset={self.short_edge_scan_z_offset:.4f}, "
            f"min_width={self.short_edge_min_width:.4f}, "
            f"min_endpoint_margin_px={self.short_edge_min_endpoint_margin_px:.1f}"
        )

        for step in range(self.short_edge_scan_max_steps + 1):
            obs = get_observation()
            observations, clear_count, widest, best_margin = self._short_edge_visibility(obs)
            camera_names = [observation.camera_name for observation in observations]
            self.get_logger().info(
                "TransportToMean short-edge scan observation: "
                f"step={step}, y={current_pose.position.y:.4f}, "
                f"cameras={camera_names}, clear_observations={clear_count}, "
                f"widest_edge={widest:.4f}, best_endpoint_margin_px={best_margin:.1f}"
            )
            if clear_count >= self.short_edge_min_observations:
                self.get_logger().info(
                    "TransportToMean short-edge scan accepted current view."
                )
                return

            if step == self.short_edge_scan_max_steps:
                break

            target_xyz = (
                current_pose.position.x,
                start_pose.position.y + self.short_edge_scan_y_step * float(step + 1),
                scan_z,
            )
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                target_xyz,
                duration_sec=self.short_edge_scan_duration_sec,
            )
            hold_steps = max(0, int(self.short_edge_scan_hold_sec / self.dt))
            for _ in range(hold_steps):
                self.set_pose_target(move_robot=move_robot, pose=current_pose)
                self.sleep_for(self.dt)

        self.get_logger().warn(
            "TransportToMean short-edge scan did not get a fully clear edge; "
            "continuing with the best available view."
        )

    def _detect_target_from_linear_view_search(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        center_pose: Pose,
    ) -> tuple[tuple[float, float, float] | None, str]:
        if (
            not self.magenta_linear_view_enabled
            or self.magenta_linear_view_steps <= 0
            or self.magenta_linear_view_y_distance_m <= 0.0
        ):
            obs = get_observation()
            target_xyz = self._multicamera_magenta_target_xyz(task, obs)
            if target_xyz is not None:
                return target_xyz, "magenta ROI + board +y edge fit"
            target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
            if target_xyz is not None:
                return target_xyz, "multi-camera short-edge fit"
            return None, ""

        fallback_target = None
        fallback_source = ""
        current_pose = center_pose
        obs = get_observation()
        target_xyz = self._multicamera_magenta_target_xyz(task, obs)
        if target_xyz is not None:
            self.get_logger().info(
                "MagentaSquare accepted magenta-guided edge target at high view pose; "
                "skipping +y linear search."
            )
            return target_xyz, "magenta ROI + board +y edge fit"

        candidate = self._multicamera_short_edge_target_xyz(task, obs)
        if candidate is not None:
            fallback_target = candidate
            fallback_source = "multi-camera short-edge fit"

        step_count = max(1, self.magenta_linear_view_steps)
        self.get_logger().info(
            "MagentaSquare sampling linear +y view search at fixed z/orientation: "
            f"start_xyz={[round(center_pose.position.x, 5), round(center_pose.position.y, 5), round(center_pose.position.z, 5)]}, "
            f"y_distance={self.magenta_linear_view_y_distance_m:.4f}, "
            f"samples={step_count + 1}, "
            f"move_duration={self.magenta_linear_view_move_duration_sec:.3f}, "
            f"hold_sec={self.magenta_linear_view_hold_sec:.3f}"
        )
        for step_index in range(1, step_count + 1):
            fraction = float(step_index) / float(step_count)
            target_xyz = (
                center_pose.position.x,
                center_pose.position.y + self.magenta_linear_view_y_distance_m * fraction,
                center_pose.position.z,
            )
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                target_xyz,
                duration_sec=self.magenta_linear_view_move_duration_sec,
            )
            hold_until = self.time_now() + Duration(
                seconds=max(self.magenta_linear_view_hold_sec, 0.0)
            )
            while self.time_now() < hold_until:
                self.set_pose_target(move_robot=move_robot, pose=current_pose)
                self.sleep_for(min(self.dt, self.magenta_linear_view_hold_sec))

            obs = get_observation()
            target_xyz = self._multicamera_magenta_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare linear +y view sample accepted magenta-guided edge target: "
                    f"sample={step_index}/{step_count}, "
                    f"y_offset={self.magenta_linear_view_y_distance_m * fraction:.4f}"
                )
                return target_xyz, "magenta ROI + board +y edge fit"

            if fallback_target is None:
                candidate = self._multicamera_short_edge_target_xyz(task, obs)
                if candidate is not None:
                    fallback_target = candidate
                    fallback_source = "multi-camera short-edge fit"

        current_pose = self._move_to_pose(
            move_robot,
            current_pose,
            (
                center_pose.position.x,
                center_pose.position.y,
                center_pose.position.z,
            ),
            duration_sec=self.magenta_linear_view_move_duration_sec,
        )
        if fallback_target is not None:
            self.get_logger().info(
                "MagentaSquare linear +y view search using short-edge fallback target."
            )
        return fallback_target, fallback_source

    @staticmethod
    def _image_msg_to_rgb(raw_img) -> np.ndarray:
        channels = 3
        return np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height,
            raw_img.width,
            channels,
        )

    @staticmethod
    def _quat_xyzw_to_rot(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return np.eye(3, dtype=np.float64)
        x, y, z, w = quat / norm
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalized_vector(vector: np.ndarray) -> np.ndarray | None:
        vector = np.asarray(vector, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9:
            return None
        return vector / norm

    @staticmethod
    def _quat_normalized(quat_xyzw: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    @staticmethod
    def _quat_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
        q1 = TransportToMean._quat_normalized(q1_xyzw)
        q2 = TransportToMean._quat_normalized(q2_xyzw)
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return TransportToMean._quat_normalized(
            np.array(
                [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ],
                dtype=np.float64,
            )
        )

    @staticmethod
    def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        axis = axis / norm
        half_angle = 0.5 * float(angle)
        return TransportToMean._quat_normalized(
            np.array(
                [
                    axis[0] * np.sin(half_angle),
                    axis[1] * np.sin(half_angle),
                    axis[2] * np.sin(half_angle),
                    np.cos(half_angle),
                ],
                dtype=np.float64,
            )
        )

    @staticmethod
    def _quat_from_two_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_vec = TransportToMean._normalized_vector(source)
        target_vec = TransportToMean._normalized_vector(target)
        if source_vec is None or target_vec is None:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        dot = float(np.clip(np.dot(source_vec, target_vec), -1.0, 1.0))
        if dot > 0.999999:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if dot < -0.999999:
            axis = np.cross(source_vec, np.array([1.0, 0.0, 0.0], dtype=np.float64))
            if float(np.linalg.norm(axis)) <= 1e-9:
                axis = np.cross(
                    source_vec,
                    np.array([0.0, 1.0, 0.0], dtype=np.float64),
                )
            return TransportToMean._quat_from_axis_angle(axis, np.pi)

        axis = np.cross(source_vec, target_vec)
        return TransportToMean._quat_from_axis_angle(axis, float(np.arccos(dot)))

    @staticmethod
    def _quat_slerp(
        q0_xyzw: np.ndarray,
        q1_xyzw: np.ndarray,
        fraction: float,
    ) -> np.ndarray:
        q0 = TransportToMean._quat_normalized(q0_xyzw)
        q1 = TransportToMean._quat_normalized(q1_xyzw)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return TransportToMean._quat_normalized(q0 + fraction * (q1 - q0))

        theta_0 = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        theta = theta_0 * float(fraction)
        sin_theta = float(np.sin(theta))
        sin_theta_0 = float(np.sin(theta_0))
        s0 = float(np.cos(theta)) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return TransportToMean._quat_normalized(s0 * q0 + s1 * q1)

    @staticmethod
    def _normalized_xy(vector: np.ndarray) -> np.ndarray | None:
        xy = np.asarray(vector, dtype=np.float64).reshape(-1)[:2]
        norm = float(np.linalg.norm(xy))
        if norm <= 1e-9:
            return None
        return np.array([xy[0] / norm, xy[1] / norm, 0.0], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _camera_to_base_link_transform(
        self,
        camera_frame: str,
        *,
        warn: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                camera_frame,
                Time(),
            )
        except TransformException as ex:
            if warn:
                self.get_logger().warn(
                    f"TransportToMean could not transform {camera_frame} -> base_link: {ex}"
                )
            return None

        transform = tf_stamped.transform
        rotation = self._quat_xyzw_to_rot(
            np.array(
                [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ],
                dtype=np.float64,
            )
        )
        translation = np.array(
            [
                transform.translation.x,
                transform.translation.y,
                transform.translation.z,
            ],
            dtype=np.float64,
        )
        return rotation, translation

    def _world_point_to_base_link(self, point_world: np.ndarray) -> np.ndarray:
        transform = self._camera_to_base_link_transform("world", warn=False)
        if transform is not None:
            rotation, translation = transform
            return rotation @ np.asarray(point_world, dtype=np.float64).reshape(3) + translation

        point_world = np.asarray(point_world, dtype=np.float64).reshape(3)
        translated = point_world - np.array(
            [self.robot_world_x, self.robot_world_y, self.robot_world_z],
            dtype=np.float64,
        )
        yaw = self.robot_world_yaw
        cos_yaw = float(np.cos(-yaw))
        sin_yaw = float(np.sin(-yaw))
        world_to_base_rot = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return world_to_base_rot @ translated

    def _world_vector_to_base_link(self, vector_world: np.ndarray) -> np.ndarray:
        transform = self._camera_to_base_link_transform("world", warn=False)
        if transform is not None:
            rotation, _ = transform
            return rotation @ np.asarray(vector_world, dtype=np.float64).reshape(3)

        yaw = self.robot_world_yaw
        cos_yaw = float(np.cos(-yaw))
        sin_yaw = float(np.sin(-yaw))
        world_to_base_rot = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return world_to_base_rot @ np.asarray(vector_world, dtype=np.float64).reshape(3)

    def _pixel_to_base_plane(
        self,
        pixel: np.ndarray,
        camera_info,
        plane_z: float | None = None,
    ) -> np.ndarray | None:
        camera_frame = camera_info.header.frame_id
        if not camera_frame:
            return None

        camera_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(camera_matrix).all() or camera_matrix[0, 0] <= 0.0:
            return None

        dist_coeffs = np.asarray(camera_info.d, dtype=np.float64)
        if dist_coeffs.size == 0:
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        undistorted = cv2.undistortPoints(
            np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
            camera_matrix,
            dist_coeffs,
        ).reshape(2)
        ray_camera = np.array([undistorted[0], undistorted[1], 1.0], dtype=np.float64)

        transform = self._camera_to_base_link_transform(camera_frame)
        if transform is None:
            return None
        rotation, origin_base = transform
        ray_base = rotation @ ray_camera
        if abs(float(ray_base[2])) <= 1e-9:
            return None

        target_plane_z = self.board_plane_z if plane_z is None else float(plane_z)
        scale = (target_plane_z - origin_base[2]) / ray_base[2]
        if scale <= 0.0:
            return None
        return origin_base + scale * ray_base

    def _candidate_trial_config_paths(self) -> list[Path]:
        if self.trial_config_path:
            return [Path(self.trial_config_path).expanduser()]

        if not self.trial_config_search_glob:
            return []

        search_glob = Path(self.trial_config_search_glob).expanduser()
        if search_glob.is_absolute():
            return sorted(search_glob.parent.glob(search_glob.name))

        roots = [
            Path.cwd(),
            Path(__file__).resolve().parents[3],
        ]
        candidates: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            for candidate in sorted(root.glob(str(search_glob))):
                resolved = candidate.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(candidate)
        return candidates

    @staticmethod
    def _task_matches_config_task(task: Task, config_task: dict) -> bool:
        return (
            config_task.get("target_module_name") == task.target_module_name
            and config_task.get("port_name") == task.port_name
        )

    def _matching_trial_entries(self, task: Task) -> list[tuple[Path, str, dict]]:
        matches = []
        for path in self._candidate_trial_config_paths():
            if not path.exists():
                self.get_logger().warn(
                    f"TransportToMean trial debug config does not exist: {path}"
                )
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError) as ex:
                self.get_logger().warn(
                    f"TransportToMean could not read trial debug config {path}: {ex}"
                )
                continue

            trials = config.get("trials", {})
            if not isinstance(trials, dict):
                continue
            for trial_id, trial_config in trials.items():
                tasks = trial_config.get("tasks", {})
                if not isinstance(tasks, dict):
                    continue
                if any(
                    self._task_matches_config_task(task, config_task)
                    for config_task in tasks.values()
                    if isinstance(config_task, dict)
                ):
                    matches.append((path, str(trial_id), trial_config))

        return matches

    def _true_target_xy_from_trial_config(
        self,
        task: Task,
    ) -> tuple[np.ndarray, Path, str] | None:
        true_board_pose = self._true_board_pose_from_trial_config(task)
        if true_board_pose is None:
            return None

        true_origin, true_x_axis, true_y_axis, _, path, trial_id = true_board_pose
        try:
            target_board = np.asarray(
                self._board_pose_target_xyz_board(task),
                dtype=np.float64,
            )
        except ValueError as ex:
            self.get_logger().warn(
                f"TransportToMean trial debug config {path}:{trial_id} "
                f"target lookup failed: {ex}"
            )
            return None

        target_base = (
            true_origin
            + target_board[0] * true_x_axis
            + target_board[1] * true_y_axis
        )
        target_base[2] = true_origin[2] + target_board[2]
        return target_base, path, trial_id

    def _true_board_pose_from_trial_config(
        self,
        task: Task,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Path, str] | None:
        matches = self._matching_trial_entries(task)
        if not matches:
            return None
        if len(matches) > 1 and not self.trial_config_path:
            self.get_logger().warn(
                "TransportToMean found multiple matching trial configs for "
                f"{task.target_module_name}/{task.port_name}; set "
                "AIC_TRANSPORT_TRIAL_CONFIG to the active trial YAML for exact debug error."
            )
            return None

        path, trial_id, trial_config = matches[0]
        try:
            board_pose = trial_config["scene"]["task_board"]["pose"]
            board_x = float(board_pose["x"])
            board_y = float(board_pose["y"])
            board_z = float(board_pose["z"])
            board_yaw = float(board_pose["yaw"])
        except (KeyError, TypeError, ValueError) as ex:
            self.get_logger().warn(
                f"TransportToMean trial debug config {path}:{trial_id} is incomplete: {ex}"
            )
            return None

        cos_yaw = float(np.cos(board_yaw))
        sin_yaw = float(np.sin(board_yaw))
        board_to_world_rot = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        origin_world = np.array([board_x, board_y, board_z], dtype=np.float64)
        origin_base = self._world_point_to_base_link(origin_world)
        board_x_axis = self._world_vector_to_base_link(board_to_world_rot[:, 0])
        board_y_axis = self._world_vector_to_base_link(board_to_world_rot[:, 1])
        board_x_axis = board_x_axis / max(float(np.linalg.norm(board_x_axis)), 1e-9)
        board_y_axis = board_y_axis / max(float(np.linalg.norm(board_y_axis)), 1e-9)
        yaw_base = float(np.arctan2(board_x_axis[1], board_x_axis[0]))
        return origin_base, board_x_axis, board_y_axis, yaw_base, path, trial_id

    def _log_trial_config_board_pose_error(
        self,
        task: Task,
        estimated_origin: np.ndarray,
        estimated_x_axis: np.ndarray,
        estimated_y_axis: np.ndarray,
        estimated_yaw: float,
        source: str,
    ) -> None:
        expected = self._true_board_pose_from_trial_config(task)
        if expected is None:
            return

        true_origin, true_x_axis, true_y_axis, true_yaw, path, trial_id = expected
        estimated_origin = np.asarray(estimated_origin, dtype=np.float64).reshape(3)
        estimated_x_axis = np.asarray(estimated_x_axis, dtype=np.float64).reshape(3)
        estimated_y_axis = np.asarray(estimated_y_axis, dtype=np.float64).reshape(3)
        xy_error = estimated_origin[:2] - true_origin[:2]
        yaw_error = self._wrap_angle(float(estimated_yaw) - true_yaw)
        self.get_logger().info(
            "TransportToMean trial-config board pose check: "
            f"source={source}, trial={trial_id}, config={path}, "
            f"estimated_origin={np.round(estimated_origin, 5).tolist()}, "
            f"true_origin={np.round(true_origin, 5).tolist()}, "
            f"origin_error_xy={np.round(xy_error, 5).tolist()}, "
            f"origin_error_norm={float(np.linalg.norm(xy_error)):.5f} m, "
            f"estimated_yaw={float(estimated_yaw):.5f}, "
            f"true_yaw={true_yaw:.5f}, yaw_error={yaw_error:.5f}, "
            f"estimated_x_axis={np.round(estimated_x_axis, 5).tolist()}, "
            f"true_x_axis={np.round(true_x_axis, 5).tolist()}, "
            f"estimated_y_axis={np.round(estimated_y_axis, 5).tolist()}, "
            f"true_y_axis={np.round(true_y_axis, 5).tolist()}"
        )

    def _log_trial_config_target_error(
        self,
        task: Task,
        estimated_target_xyz: tuple[float, float, float],
        target_source: str,
    ) -> None:
        expected = self._true_target_xy_from_trial_config(task)
        if expected is None:
            return

        expected_base, path, trial_id = expected
        estimated = np.asarray(estimated_target_xyz, dtype=np.float64)
        xy_error = estimated[:2] - expected_base[:2]
        xy_error_norm = float(np.linalg.norm(xy_error))
        self.get_logger().info(
            "TransportToMean trial-config target check: "
            f"source={target_source}, trial={trial_id}, config={path}, "
            f"estimated_xy={np.round(estimated[:2], 5).tolist()}, "
            f"true_target_xy={np.round(expected_base[:2], 5).tolist()}, "
            f"error_xy={np.round(xy_error, 5).tolist()}, "
            f"error_norm={xy_error_norm:.5f} m"
        )

    def _log_trial_config_final_pose_error(
        self,
        task: Task,
        commanded_target_xyz: tuple[float, float, float],
        final_tcp_pose: Pose,
        target_source: str,
    ) -> None:
        expected = self._true_target_xy_from_trial_config(task)
        if expected is None:
            return

        expected_base, path, trial_id = expected
        commanded = np.asarray(commanded_target_xyz, dtype=np.float64)
        actual = np.array(
            [
                final_tcp_pose.position.x,
                final_tcp_pose.position.y,
                final_tcp_pose.position.z,
            ],
            dtype=np.float64,
        )
        actual_minus_commanded = actual - commanded
        actual_minus_true_xy = actual[:2] - expected_base[:2]
        commanded_minus_true_xy = commanded[:2] - expected_base[:2]
        self.get_logger().info(
            "TransportToMean final pose trial-config check: "
            f"source={target_source}, trial={trial_id}, config={path}, "
            f"actual_tcp_xyz={np.round(actual, 5).tolist()}, "
            f"commanded_target_xyz={np.round(commanded, 5).tolist()}, "
            f"true_target_xyz={np.round(expected_base, 5).tolist()}, "
            f"actual_minus_commanded={np.round(actual_minus_commanded, 5).tolist()}, "
            f"actual_minus_commanded_norm={float(np.linalg.norm(actual_minus_commanded)):.5f} m, "
            f"commanded_minus_true_xy={np.round(commanded_minus_true_xy, 5).tolist()}, "
            f"commanded_minus_true_xy_norm={float(np.linalg.norm(commanded_minus_true_xy)):.5f} m, "
            f"actual_minus_true_xy={np.round(actual_minus_true_xy, 5).tolist()}, "
            f"actual_minus_true_xy_norm={float(np.linalg.norm(actual_minus_true_xy)):.5f} m"
        )

    def _detect_visible_short_edge_px(
        self,
        rgb: np.ndarray,
        dark_mask: np.ndarray,
    ) -> BoardEdgePixels | None:
        edges = cv2.Canny(dark_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=80,
            maxLineGap=30,
        )
        if lines is None:
            return None

        image_h, image_w = rgb.shape[:2]
        best: BoardEdgePixels | None = None

        for raw_line in lines.reshape(-1, 4):
            p0 = raw_line[:2].astype(np.float64)
            p1 = raw_line[2:].astype(np.float64)
            if self.sort_short_edge_left_to_right:
                if (p0[0] > p1[0]) or (p0[0] == p1[0] and p0[1] > p1[1]):
                    p0, p1 = p1, p0

            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 80.0:
                continue

            u = edge_vec / edge_len
            normals = (
                np.array([-u[1], u[0]], dtype=np.float64),
                np.array([u[1], -u[0]], dtype=np.float64),
            )

            if self.extend_visible_edge_toward_top:
                inward = min(normals, key=lambda n: n[1])
                dark_fraction = 1.0
            else:
                support: list[tuple[float, int, int, np.ndarray]] = []
                for normal in normals:
                    sample_count = 0
                    dark_count = 0
                    for alpha in np.linspace(0.1, 0.9, 9):
                        point = p0 + alpha * edge_vec
                        for distance_px in (12.0, 24.0, 36.0, 48.0, 72.0, 96.0):
                            sample = np.round(point + normal * distance_px).astype(int)
                            if 0 <= sample[0] < image_w and 0 <= sample[1] < image_h:
                                sample_count += 1
                                if dark_mask[sample[1], sample[0]] > 0:
                                    dark_count += 1
                    dark_fraction = dark_count / sample_count if sample_count else 0.0
                    support.append((dark_fraction, dark_count, sample_count, normal))

                support.sort(key=lambda item: (item[0], item[1]), reverse=True)
                dark_fraction, _, _, inward = support[0]
                if dark_fraction < 0.25:
                    continue

            endpoint_margin = min(
                float(np.min([p0[0], p1[0], image_w - p0[0], image_w - p1[0]])),
                float(np.min([p0[1], p1[1], image_h - p0[1], image_h - p1[1]])),
            )
            visibility_bonus = min(
                max(endpoint_margin, 0.0)
                / max(self.short_edge_min_endpoint_margin_px, 1.0),
                1.0,
            )
            score = edge_len * (0.5 + dark_fraction) * (0.75 + 0.25 * visibility_bonus)
            if best is None or score > best.score:
                best = BoardEdgePixels(
                    p0_px=p0,
                    p1_px=p1,
                    inward_px=inward,
                    score=score,
                )

        return best

    def _detect_board_edge_pixel_candidates(
        self,
        rgb: np.ndarray,
        dark_mask: np.ndarray,
    ) -> list[BoardEdgePixels]:
        edges = cv2.Canny(dark_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=35,
            minLineLength=80,
            maxLineGap=35,
        )
        if lines is None:
            return []

        image_h, image_w = rgb.shape[:2]
        candidates = []
        for raw_line in lines.reshape(-1, 4):
            p0 = raw_line[:2].astype(np.float64)
            p1 = raw_line[2:].astype(np.float64)
            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len < 80.0:
                continue
            endpoint_margin = min(
                float(np.min([p0[0], p1[0], image_w - p0[0], image_w - p1[0]])),
                float(np.min([p0[1], p1[1], image_h - p0[1], image_h - p1[1]])),
            )
            visibility_bonus = min(
                max(endpoint_margin, 0.0)
                / max(self.short_edge_min_endpoint_margin_px, 1.0),
                1.0,
            )
            candidates.append(
                BoardEdgePixels(
                    p0_px=p0,
                    p1_px=p1,
                    inward_px=np.zeros(2, dtype=np.float64),
                    score=edge_len * (0.75 + 0.25 * visibility_bonus),
                )
            )
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates[:12]

    @staticmethod
    def _dark_board_mask(rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 180, 115]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    @staticmethod
    def _magenta_mask(rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hsv_mask = cv2.inRange(
            hsv,
            np.array([135, 35, 18], dtype=np.uint8),
            np.array([178, 255, 180], dtype=np.uint8),
        )
        red = rgb[:, :, 0].astype(np.int16)
        green = rgb[:, :, 1].astype(np.int16)
        blue = rgb[:, :, 2].astype(np.int16)
        rb_min = np.minimum(red, blue)
        rb_max = np.maximum(red, blue)
        rgb_mask = (
            (rb_min >= 20)
            & (green <= np.maximum(35, (0.75 * rb_min).astype(np.int16)))
            & ((red + blue) >= 55)
            & (np.abs(red - blue) <= 85)
            & (rb_max <= 190)
        ).astype(np.uint8) * 255
        mask = cv2.bitwise_or(hsv_mask, rgb_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    @staticmethod
    def _ordered_box_points(rect) -> np.ndarray:
        points = cv2.boxPoints(rect).astype(np.float64)
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        return points[np.argsort(angles)]

    def _magenta_edge_gap_center(
        self,
        points_xy: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len <= 1e-9:
            return 0.0, 0.5 * (p0 + p1)

        edge_dir = edge_vec / edge_len
        normal = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float64)
        rel = points_xy - p0
        along = rel @ edge_dir
        dist = np.abs(rel @ normal)
        on_edge = along[
            (dist <= self.magenta_edge_support_tolerance_m)
            & (along >= -self.magenta_edge_support_tolerance_m)
            & (along <= edge_len + self.magenta_edge_support_tolerance_m)
        ]
        if on_edge.size < 2:
            return 0.0, 0.5 * (p0 + p1)

        coords = np.sort(np.clip(on_edge, 0.0, edge_len))
        coords = np.concatenate(([0.0], coords, [edge_len]))
        gaps = np.diff(coords)
        gap_index = int(np.argmax(gaps))
        gap_size = float(gaps[gap_index])
        gap_coord = 0.5 * (coords[gap_index] + coords[gap_index + 1])
        return gap_size, p0 + gap_coord * edge_dir

    def _board_pose_from_magenta_points(
        self,
        camera_name: str,
        points_base: np.ndarray,
        *,
        area_px: float,
    ) -> MagentaMarkerObservation | None:
        if points_base.shape[0] < 8:
            return None

        points_xy = points_base[:, :2].astype(np.float64)
        rect = cv2.minAreaRect(points_xy.astype(np.float32))
        box_xy = self._ordered_box_points(rect)
        side_lengths = [
            float(np.linalg.norm(box_xy[(idx + 1) % 4] - box_xy[idx]))
            for idx in range(4)
        ]
        long_side = max(side_lengths)
        short_side = min(side_lengths)
        if short_side <= 1e-6:
            return None
        side_ratio = short_side / long_side
        if side_ratio < self.magenta_min_side_ratio:
            self.get_logger().info(
                "MagentaSquare ignoring marker candidate with non-square projection: "
                f"camera={camera_name}, sides={np.round(side_lengths, 4).tolist()}, "
                f"side_ratio={side_ratio:.3f}, min_ratio={self.magenta_min_side_ratio:.3f}"
            )
            return None
        if abs(long_side - self.MAGENTA_SQUARE_SIZE) > self.magenta_square_size_tolerance_m:
            self.get_logger().info(
                "MagentaSquare ignoring marker candidate with unexpected size: "
                f"camera={camera_name}, sides={np.round(side_lengths, 4).tolist()}"
            )
            return None

        edge_support = []
        edge_gap_centers = []
        for idx in range(4):
            p0 = box_xy[idx]
            p1 = box_xy[(idx + 1) % 4]
            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-9:
                edge_support.append(0)
                edge_gap_centers.append(0.5 * (p0 + p1))
                continue
            edge_dir = edge_vec / edge_len
            normal = np.array([-edge_dir[1], edge_dir[0]], dtype=np.float64)
            rel = points_xy - p0
            along = rel @ edge_dir
            dist = np.abs(rel @ normal)
            support_count = int(
                np.count_nonzero(
                    (dist <= self.magenta_edge_support_tolerance_m)
                    & (along >= -self.magenta_edge_support_tolerance_m)
                    & (along <= edge_len + self.magenta_edge_support_tolerance_m)
                )
            )
            gap_size, gap_center = self._magenta_edge_gap_center(points_xy, p0, p1)
            gap_fraction = gap_size / max(edge_len, 1e-9)
            edge_support.append(support_count / max(edge_len, 1e-9) - 200.0 * gap_fraction)
            edge_gap_centers.append(gap_center)

        gap_edge_idx = int(np.argmin(edge_support))
        plus_y_edge_idx = (gap_edge_idx + 2) % 4
        gap_edge_center_xy = 0.5 * (
            box_xy[gap_edge_idx] + box_xy[(gap_edge_idx + 1) % 4]
        )
        plus_y_edge_center_xy = 0.5 * (
            box_xy[plus_y_edge_idx] + box_xy[(plus_y_edge_idx + 1) % 4]
        )
        board_y_axis = self._normalized_xy(plus_y_edge_center_xy - gap_edge_center_xy)
        if board_y_axis is None:
            return None
        board_x_axis = np.array([board_y_axis[1], -board_y_axis[0], 0.0], dtype=np.float64)
        gap_center_xy = edge_gap_centers[gap_edge_idx]
        marker_center_base = np.array(
            [
                float(np.mean(box_xy[:, 0])),
                float(np.mean(box_xy[:, 1])),
                self.magenta_marker_plane_z,
            ],
            dtype=np.float64,
        )
        gap_offset = np.array(
            [
                gap_center_xy[0] - marker_center_base[0],
                gap_center_xy[1] - marker_center_base[1],
                0.0,
            ],
            dtype=np.float64,
        )
        gap_offset = gap_offset - np.dot(gap_offset, board_y_axis) * board_y_axis
        gap_based_x_axis = self._normalized_xy(-gap_offset)
        if gap_based_x_axis is not None:
            board_x_axis = gap_based_x_axis
            board_y_axis = np.array(
                [-board_x_axis[1], board_x_axis[0], 0.0],
                dtype=np.float64,
            )

        plus_y_edge_center = np.array(
            [
                plus_y_edge_center_xy[0],
                plus_y_edge_center_xy[1],
                self.magenta_marker_plane_z,
            ],
            dtype=np.float64,
        )
        plus_y_boundary_center = (
            plus_y_edge_center + self.MAGENTA_PLUS_Y_BOUNDARY_OFFSET * board_y_axis
        )
        board_origin = (
            plus_y_boundary_center
            - self.MAGENTA_CENTER_BOARD[0] * board_x_axis
            - (0.5 * self.BOARD_SIZE_Y) * board_y_axis
        )
        board_origin[2] = self.board_plane_z
        gap_edge_center = np.array(
            [gap_edge_center_xy[0], gap_edge_center_xy[1], self.magenta_marker_plane_z],
            dtype=np.float64,
        )
        score = float(area_px) / (1.0 + abs(long_side - self.MAGENTA_SQUARE_SIZE))
        self.get_logger().info(
            "MagentaSquare marker pose candidate: "
            f"camera={camera_name}, area_px={area_px:.1f}, "
            f"center={np.round(marker_center_base, 4).tolist()}, "
            f"sides={np.round(side_lengths, 4).tolist()}, "
            f"edge_support={np.round(edge_support, 2).tolist()}, "
            f"gap_edge={gap_edge_idx}, "
            f"board_origin={np.round(board_origin, 4).tolist()}, "
            f"board_x_axis={np.round(board_x_axis, 4).tolist()}, "
            f"board_y_axis={np.round(board_y_axis, 4).tolist()}"
        )
        return MagentaMarkerObservation(
            camera_name=camera_name,
            marker_center_base=marker_center_base,
            board_origin_base=board_origin,
            board_x_axis=board_x_axis,
            board_y_axis=board_y_axis,
            gap_edge_center_base=gap_edge_center,
            plus_y_edge_center_base=plus_y_edge_center,
            score=score,
            area_px=float(area_px),
        )

    def _detect_magenta_marker_observation(
        self,
        camera_name: str,
        image_msg,
        camera_info,
    ) -> MagentaMarkerObservation | None:
        if not camera_info.header.frame_id:
            self.get_logger().warn(f"MagentaSquare {camera_name} CameraInfo has no frame_id.")
            return None

        rgb = self._image_msg_to_rgb(image_msg)
        mask = self._magenta_mask(rgb)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        candidates = sorted(contours, key=cv2.contourArea, reverse=True)
        for contour in candidates[:5]:
            area_px = float(cv2.contourArea(contour))
            if area_px < self.magenta_min_area_px or area_px > self.magenta_max_area_px:
                continue

            contour_region = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_region, [contour], -1, 255, thickness=-1)
            component_mask = cv2.bitwise_and(mask, contour_region)
            ys, xs = np.nonzero(component_mask)
            if xs.size < 8:
                continue

            stride = max(1, xs.size // 800)
            pixels = np.column_stack([xs[::stride], ys[::stride]]).astype(np.float64)
            projected = []
            for pixel in pixels:
                point = self._pixel_to_base_plane(
                    pixel,
                    camera_info,
                    plane_z=self.magenta_marker_plane_z,
                )
                if point is not None:
                    projected.append(point)
            if len(projected) < 8:
                continue

            observation = self._board_pose_from_magenta_points(
                camera_name,
                np.vstack(projected),
                area_px=area_px,
            )
            if observation is not None:
                return observation
        return None

    def _multicamera_magenta_observations(self, obs) -> list[MagentaMarkerObservation]:
        camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        observations = []
        for camera_name, image_msg, camera_info in camera_inputs:
            detection = self._detect_magenta_marker_observation(
                camera_name,
                image_msg,
                camera_info,
            )
            if detection is not None:
                observations.append(detection)
        return observations

    def _detect_magenta_roi_observation(
        self,
        camera_name: str,
        image_msg,
        camera_info,
    ) -> MagentaMarkerObservation | MagentaRoiObservation | None:
        if not camera_info.header.frame_id:
            self.get_logger().warn(f"MagentaSquare {camera_name} CameraInfo has no frame_id.")
            return None

        rgb = self._image_msg_to_rgb(image_msg)
        mask = self._magenta_mask(rgb)
        cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cleanup_kernel, iterations=1)

        # The marker is a hollow border, often split into several visible pieces.
        # Dilate only for grouping; measure the ROI using the original magenta pixels.
        grouping_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        grouping_mask = cv2.dilate(clean_mask, grouping_kernel, iterations=1)
        group_count, group_labels, group_stats, _ = cv2.connectedComponentsWithStats(
            grouping_mask,
            connectivity=8,
        )
        if group_count <= 1:
            return None

        min_area_px = max(30.0, 0.25 * self.magenta_min_area_px)
        group_indices = sorted(
            range(1, group_count),
            key=lambda idx: group_stats[idx, cv2.CC_STAT_AREA],
            reverse=True,
        )
        for idx in group_indices[:8]:
            grouped_region = (group_labels == idx).astype(np.uint8) * 255
            component_mask = cv2.bitwise_and(clean_mask, grouped_region)
            ys, xs = np.nonzero(component_mask)
            area_px = float(xs.size)
            if area_px < min_area_px or area_px > self.magenta_max_area_px:
                continue

            x_min = float(np.min(xs))
            y_min = float(np.min(ys))
            x_max = float(np.max(xs))
            y_max = float(np.max(ys))
            width = x_max - x_min + 1.0
            height = y_max - y_min + 1.0
            if width <= 1.0 or height <= 1.0:
                continue
            fill_ratio = area_px / max(width * height, 1.0)
            if fill_ratio < 0.025:
                continue

            center_px = np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)
            stride = max(1, xs.size // 1000)
            projected = []
            for pixel in np.column_stack([xs[::stride], ys[::stride]]).astype(np.float64):
                point = self._pixel_to_base_plane(
                    pixel,
                    camera_info,
                    plane_z=self.magenta_marker_plane_z,
                )
                if point is not None:
                    projected.append(point)

            if len(projected) >= 8:
                marker_observation = self._board_pose_from_magenta_points(
                    camera_name,
                    np.vstack(projected),
                    area_px=area_px,
                )
                if marker_observation is not None:
                    self.get_logger().info(
                        "MagentaSquare accepted geometry-fit magenta marker prior: "
                        f"camera={camera_name}, area_px={area_px:.1f}, "
                        f"bbox={[round(x_min, 1), round(y_min, 1), round(width, 1), round(height, 1)]}, "
                        f"fill_ratio={fill_ratio:.3f}, "
                        f"group_count={group_count - 1}, "
                        f"center_px={np.round(center_px, 1).tolist()}, "
                        f"center_base={np.round(marker_observation.marker_center_base, 4).tolist()}"
                    )
                    return marker_observation

            center_base = self._pixel_to_base_plane(
                center_px,
                camera_info,
                plane_z=self.magenta_marker_plane_z,
            )
            if center_base is None:
                continue

            score = area_px * min(fill_ratio, 1.0)
            self.get_logger().info(
                "MagentaSquare rough magenta ROI candidate: "
                f"camera={camera_name}, area_px={area_px:.1f}, "
                f"bbox={[round(x_min, 1), round(y_min, 1), round(width, 1), round(height, 1)]}, "
                f"fill_ratio={fill_ratio:.3f}, "
                f"group_count={group_count - 1}, "
                f"center_px={np.round(center_px, 1).tolist()}, "
                f"center_base={np.round(center_base, 4).tolist()}"
            )
            return MagentaRoiObservation(
                camera_name=camera_name,
                marker_center_base=center_base,
                marker_center_px=center_px,
                score=score,
                area_px=area_px,
            )
        return None

    def _multicamera_magenta_roi_observations(
        self,
        obs,
    ) -> list[MagentaMarkerObservation | MagentaRoiObservation]:
        camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        observations = []
        for camera_name, image_msg, camera_info in camera_inputs:
            detection = self._detect_magenta_roi_observation(
                camera_name,
                image_msg,
                camera_info,
            )
            if detection is not None:
                observations.append(detection)
        return observations

    def _magenta_roi_consensus(
        self,
        observations: list[MagentaMarkerObservation | MagentaRoiObservation],
    ) -> list[MagentaMarkerObservation | MagentaRoiObservation]:
        if len(observations) <= 1:
            return observations

        best_cluster = []
        for candidate in observations:
            cluster = [
                observation
                for observation in observations
                if np.linalg.norm(
                    observation.marker_center_base[:2] - candidate.marker_center_base[:2]
                )
                <= self.magenta_consensus_distance_m
            ]
            if len(cluster) > len(best_cluster) or (
                len(cluster) == len(best_cluster)
                and sum(observation.score for observation in cluster)
                > sum(observation.score for observation in best_cluster)
            ):
                best_cluster = cluster

        if len(best_cluster) >= 2:
            if len(best_cluster) < len(observations):
                best_cluster_ids = {id(observation) for observation in best_cluster}
                rejected = [
                    observation.camera_name
                    for observation in observations
                    if id(observation) not in best_cluster_ids
                ]
                self.get_logger().info(
                    "MagentaSquare rejecting rough magenta ROI outliers: "
                    f"rejected_cameras={rejected}"
                )
            return best_cluster

        best = max(observations, key=lambda observation: observation.score)
        self.get_logger().warn(
            "MagentaSquare rough magenta ROI observations disagree; "
            "using highest-score single-camera prior for board-edge search: "
            f"camera={best.camera_name}, "
            f"centers={[np.round(ob.marker_center_base, 4).tolist() for ob in observations]}, "
            f"consensus_distance={self.magenta_consensus_distance_m:.3f} m"
        )
        return [best]

    def _detect_board_edge_candidates(
        self,
        camera_name: str,
        image_msg,
        camera_info,
    ) -> list[BoardEdgeCandidate]:
        if not camera_info.header.frame_id:
            return []

        rgb = self._image_msg_to_rgb(image_msg)
        edge_pixels = self._detect_board_edge_pixel_candidates(rgb, self._dark_board_mask(rgb))
        candidates = []
        for edge in edge_pixels:
            p0_base = self._pixel_to_base_plane(edge.p0_px, camera_info)
            p1_base = self._pixel_to_base_plane(edge.p1_px, camera_info)
            if p0_base is None or p1_base is None:
                continue
            width_base = float(np.linalg.norm(p1_base - p0_base))
            if width_base <= 1e-6:
                continue
            center_base = 0.5 * (p0_base + p1_base)
            center_base[2] = self.board_plane_z
            candidates.append(
                BoardEdgeCandidate(
                    camera_name=camera_name,
                    p0_base=p0_base,
                    p1_base=p1_base,
                    center_base=center_base,
                    width_base=width_base,
                    score=edge.score,
                )
            )
        return candidates

    def _multicamera_board_edge_candidates(self, obs) -> list[BoardEdgeCandidate]:
        camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        candidates = []
        for camera_name, image_msg, camera_info in camera_inputs:
            candidates.extend(
                self._detect_board_edge_candidates(camera_name, image_msg, camera_info)
            )
        return candidates

    def _plus_y_edge_target_from_magenta_prior(
        self,
        task: Task,
        obs,
        marker_observations: list[MagentaMarkerObservation | MagentaRoiObservation],
    ) -> tuple[float, float, float] | None:
        weights = np.asarray(
            [max(marker.score, 1.0) for marker in marker_observations],
            dtype=np.float64,
        )
        marker_center = np.average(
            np.vstack([marker.marker_center_base for marker in marker_observations]),
            axis=0,
            weights=weights,
        )
        marker_center[2] = self.magenta_marker_plane_z
        marker_y_axis = None
        marker_y_axis_sources = [
            marker.board_y_axis
            for marker in marker_observations
            if isinstance(marker, MagentaMarkerObservation)
        ]
        if marker_y_axis_sources:
            reference_y = marker_y_axis_sources[0]
            aligned_y_axes = []
            for y_axis in marker_y_axis_sources:
                if np.dot(y_axis, reference_y) < 0.0:
                    y_axis = -y_axis
                aligned_y_axes.append(y_axis)
            marker_y_axis = self._normalized_xy(
                np.average(np.vstack(aligned_y_axes), axis=0)
            )

        edge_candidates = self._multicamera_board_edge_candidates(obs)
        if not edge_candidates:
            self.get_logger().warn(
                "MagentaSquare found magenta marker but no dark board-edge candidates."
            )
            return None

        expected_marker_to_plus_y_edge = float(
            np.hypot(
                -self.MAGENTA_CENTER_BOARD[0],
                0.5 * self.BOARD_SIZE_Y - self.MAGENTA_CENTER_BOARD[1],
            )
        )
        scored = []
        for candidate in edge_candidates:
            edge_dir = self._normalized_xy(candidate.p1_base - candidate.p0_base)
            if edge_dir is None:
                continue
            width_error = abs(candidate.width_base - self.BOARD_SIZE_X)
            if width_error > self.magenta_edge_width_tolerance_m:
                continue
            marker_to_edge = candidate.center_base - marker_center
            marker_to_edge[2] = 0.0
            distance = float(np.linalg.norm(marker_to_edge[:2]))
            distance_error = abs(distance - expected_marker_to_plus_y_edge)
            if distance_error > self.magenta_edge_marker_distance_tolerance_m:
                continue
            board_y_axis = marker_to_edge - np.dot(marker_to_edge, edge_dir) * edge_dir
            board_y_axis = self._normalized_xy(board_y_axis)
            if board_y_axis is None:
                continue
            board_x_axis = edge_dir
            if np.dot(marker_center[:2] - candidate.center_base[:2], board_x_axis[:2]) > 0.0:
                board_x_axis = -board_x_axis
            width_score = 1.0 / (1.0 + (width_error / 0.02) ** 2)
            distance_score = 1.0 / (1.0 + (distance_error / 0.04) ** 2)
            marker_axis_score = 0.0
            if marker_y_axis is not None:
                marker_axis_score = max(
                    0.0,
                    float(np.dot(board_y_axis[:2], marker_y_axis[:2])),
                )
            score = (
                2.0 * width_score
                + distance_score
                + marker_axis_score
                + min(candidate.score / 400.0, 2.0)
            )
            scored.append(
                (
                    score,
                    candidate,
                    board_x_axis,
                    board_y_axis,
                    width_error,
                    distance_error,
                    marker_axis_score,
                )
            )

        if not scored:
            self.get_logger().warn(
                "MagentaSquare found no board +y edge consistent with marker prior: "
                f"marker_center={np.round(marker_center, 4).tolist()}, "
                f"candidate_count={len(edge_candidates)}"
            )
            return None

        scored.sort(key=lambda item: item[0], reverse=True)
        (
            best_score,
            best_candidate,
            board_x_axis,
            board_y_axis,
            width_error,
            distance_error,
            marker_axis_score,
        ) = scored[0]
        plus_y_edge_center = best_candidate.center_base.copy()
        plus_y_edge_center[2] = self.board_plane_z
        board_origin = plus_y_edge_center - 0.5 * self.BOARD_SIZE_Y * board_y_axis
        board_origin[2] = self.board_plane_z

        edge_yaw = float(np.arctan2(board_x_axis[1], board_x_axis[0]))
        self._log_trial_config_board_pose_error(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            edge_yaw,
            source="magenta-guided +y short-edge fit",
        )
        target_base = self._target_xyz_from_board_pose(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            source="magenta-guided +y short-edge fit",
        )
        if target_base is None:
            return None

        self._log_estimated_final_board_pose(
            source="magenta-guided +y short-edge fit",
            board_origin=board_origin,
            board_x_axis=board_x_axis,
            board_y_axis=board_y_axis,
            target_base=target_base,
            task=task,
        )
        self.get_logger().info(
            "MagentaSquare magenta-guided board edge pose: "
            f"marker_center={np.round(marker_center, 4).tolist()}, "
            f"edge_camera={best_candidate.camera_name}, "
            f"edge_center={np.round(plus_y_edge_center, 4).tolist()}, "
            f"edge_p0={np.round(best_candidate.p0_base, 4).tolist()}, "
            f"edge_p1={np.round(best_candidate.p1_base, 4).tolist()}, "
            f"edge_width={best_candidate.width_base:.4f}, "
            f"width_error={width_error:.4f}, "
            f"distance_error={distance_error:.4f}, "
            f"marker_axis_score={marker_axis_score:.3f}, "
            f"score={best_score:.3f}, "
            f"candidate_count={len(edge_candidates)}, "
            f"board_origin={np.round(board_origin, 4).tolist()}, "
            f"board_x_axis={np.round(board_x_axis, 4).tolist()}, "
            f"board_y_axis={np.round(board_y_axis, 4).tolist()}, "
            f"target_base={np.round(np.asarray(target_base), 4).tolist()}"
        )
        return target_base

    def _multicamera_magenta_target_xyz(
        self,
        task: Task,
        obs,
    ) -> tuple[float, float, float] | None:
        observations = self._multicamera_magenta_roi_observations(obs)
        if not observations:
            self.get_logger().warn(
                "MagentaSquare multi-camera detector found no rough magenta ROI."
            )
            return None

        observations = self._magenta_roi_consensus(observations)

        edge_target = self._plus_y_edge_target_from_magenta_prior(task, obs, observations)
        if edge_target is not None:
            return edge_target
        self.get_logger().warn(
            "MagentaSquare found rough magenta ROI but could not fit the board +y edge; "
            "continuing to another XY view sample or short-edge fallback."
        )
        return None

    def _target_xyz_from_board_pose(
        self,
        task: Task,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
        *,
        source: str,
    ) -> tuple[float, float, float] | None:
        try:
            target_board = np.asarray(
                self._board_pose_target_xyz_board(task),
                dtype=np.float64,
            )
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return None

        target_base = (
            np.asarray(board_origin, dtype=np.float64).reshape(3)
            + target_board[0] * np.asarray(board_x_axis, dtype=np.float64).reshape(3)
            + target_board[1] * np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        )
        target_base[2] = max(
            self.board_plane_z + target_board[2] + self.z_offset,
            self.min_target_z,
        )
        self.get_logger().info(
            "TransportToMean board-pose target: "
            f"source={source}, "
            f"mode={self.board_pose_target_mode}, "
            f"board_target={np.round(target_board, 4).tolist()}, "
            f"target_base={np.round(target_base, 4).tolist()}"
        )
        return tuple(float(v) for v in target_base)

    def _log_estimated_final_board_pose(
        self,
        *,
        source: str,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
        target_base: tuple[float, float, float],
        task: Task,
    ) -> None:
        try:
            target_board = np.asarray(
                self._board_pose_target_xyz_board(task),
                dtype=np.float64,
            )
        except ValueError:
            target_board = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        board_origin = np.asarray(board_origin, dtype=np.float64).reshape(3)
        board_x_axis = np.asarray(board_x_axis, dtype=np.float64).reshape(3)
        board_y_axis = np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        target_base_np = np.asarray(target_base, dtype=np.float64).reshape(3)
        yaw_x = float(np.arctan2(board_x_axis[1], board_x_axis[0]))
        yaw_y = float(np.arctan2(board_y_axis[1], board_y_axis[0]))
        x_contribution = target_board[0] * board_x_axis
        y_contribution = target_board[1] * board_y_axis
        self.get_logger().info(
            "MagentaSquare estimated final board pose: "
            f"source={source}, "
            f"board_origin={np.round(board_origin, 5).tolist()}, "
            f"board_x_axis={np.round(board_x_axis, 5).tolist()}, "
            f"board_y_axis={np.round(board_y_axis, 5).tolist()}, "
            f"board_x_yaw={yaw_x:.5f} rad, "
            f"board_y_yaw={yaw_y:.5f} rad, "
            f"target_board={np.round(target_board, 5).tolist()}, "
            f"target_x_contribution={np.round(x_contribution, 5).tolist()}, "
            f"target_y_contribution={np.round(y_contribution, 5).tolist()}, "
            f"target_base={np.round(target_base_np, 5).tolist()}"
        )

    def _detect_board_edge_observation(
        self,
        camera_name: str,
        image_msg,
        camera_info,
    ) -> BoardEdgeObservation | None:
        camera_frame = camera_info.header.frame_id
        if not camera_frame:
            self.get_logger().warn(
                f"TransportToMean {camera_name} CameraInfo has no frame_id."
            )
            return None

        rgb = self._image_msg_to_rgb(image_msg)
        edge = self._detect_visible_short_edge_px(rgb, self._dark_board_mask(rgb))
        if edge is None:
            return None

        p0_base = self._pixel_to_base_plane(edge.p0_px, camera_info)
        p1_base = self._pixel_to_base_plane(edge.p1_px, camera_info)
        midpoint_px = 0.5 * (edge.p0_px + edge.p1_px)
        inward_sample_px = midpoint_px + 80.0 * edge.inward_px
        midpoint_base = self._pixel_to_base_plane(midpoint_px, camera_info)
        inward_sample_base = self._pixel_to_base_plane(inward_sample_px, camera_info)
        if (
            p0_base is None
            or p1_base is None
            or midpoint_base is None
            or inward_sample_base is None
        ):
            self.get_logger().warn(
                f"TransportToMean could not project {camera_name} board edge to plane."
            )
            return None

        inward_base = self._normalized_xy(inward_sample_base - midpoint_base)
        short_dir = self._normalized_xy(p1_base - p0_base)
        if inward_base is None or short_dir is None:
            return None

        # Keep the inferred board-long direction perpendicular to the observed short edge.
        inward_base = inward_base - np.dot(inward_base, short_dir) * short_dir
        inward_base = self._normalized_xy(inward_base)
        if inward_base is None:
            return None

        image_h, image_w = rgb.shape[:2]
        endpoint_margin_px = min(
            float(np.min([edge.p0_px[0], edge.p1_px[0], image_w - edge.p0_px[0], image_w - edge.p1_px[0]])),
            float(np.min([edge.p0_px[1], edge.p1_px[1], image_h - edge.p0_px[1], image_h - edge.p1_px[1]])),
        )
        width_base = float(np.linalg.norm(p1_base - p0_base))
        self.get_logger().info(
            "TransportToMean detected visible board edge: "
            f"camera={camera_name}, frame={camera_frame}, "
            f"p0_px={np.round(edge.p0_px, 1).tolist()}, "
            f"p1_px={np.round(edge.p1_px, 1).tolist()}, "
            f"p0_base={np.round(p0_base, 4).tolist()}, "
            f"p1_base={np.round(p1_base, 4).tolist()}, "
            f"inward_base={np.round(inward_base, 4).tolist()}, "
            f"width_base={width_base:.4f}, "
            f"endpoint_margin_px={endpoint_margin_px:.1f}"
        )
        return BoardEdgeObservation(
            camera_name=camera_name,
            p0_base=p0_base,
            p1_base=p1_base,
            inward_base=inward_base,
            width_base=width_base,
            endpoint_margin_px=endpoint_margin_px,
            score=edge.score,
        )

    def _multicamera_board_edge_observations(self, obs) -> list[BoardEdgeObservation]:
        camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        observations = []
        for camera_name, image_msg, camera_info in camera_inputs:
            detection = self._detect_board_edge_observation(
                camera_name,
                image_msg,
                camera_info,
            )
            if detection is not None:
                observations.append(detection)
        return observations

    def _fused_board_axes(
        self,
        observations: list[BoardEdgeObservation],
    ) -> FusedBoardEdge | None:
        """Fuse per-camera short-edge observations into a planar board pose cue.

        Each observation contributes a projected edge midpoint, an edge direction,
        and an inward direction. Width-inconsistent observations are dropped, then
        the remaining observations are averaged with a weight that favors projected
        widths close to the known physical short edge. The final endpoints are
        reconstructed from the known board width instead of copied from vision.
        """
        if not observations:
            return None

        short_dirs = []
        inward_dirs = []
        weights = []
        edge_midpoints = []
        for observation in observations:
            width_error = abs(observation.width_base - self.BOARD_SIZE_X)
            if (
                self.reconstruct_short_edge_width
                and width_error > self.short_edge_width_tolerance
            ):
                self.get_logger().info(
                    "TransportToMean ignoring short-edge observation with "
                    f"inconsistent width: camera={observation.camera_name}, "
                    f"width={observation.width_base:.4f}, expected={self.BOARD_SIZE_X:.4f}"
                )
                continue
            short_dir = self._normalized_xy(observation.p1_base - observation.p0_base)
            if short_dir is None:
                continue
            if short_dirs and np.dot(short_dir, short_dirs[0]) < 0.0:
                short_dir = -short_dir
            inward_dir = observation.inward_base
            if inward_dirs and np.dot(inward_dir, inward_dirs[0]) < 0.0:
                inward_dir = -inward_dir
            short_dirs.append(short_dir)
            inward_dirs.append(inward_dir)
            width_weight = 1.0 / (1.0 + (width_error / 0.02) ** 2)
            weights.append(max(float(observation.score), 1.0) * width_weight)
            edge_midpoints.append(0.5 * (observation.p0_base + observation.p1_base))

        if not short_dirs:
            return None

        weights_np = np.asarray(weights, dtype=np.float64)
        short_dir = np.average(np.vstack(short_dirs), axis=0, weights=weights_np)
        short_dir = self._normalized_xy(short_dir)
        if short_dir is None:
            return None
        if self.flip_board_x_axis:
            short_dir = -short_dir

        inward_dir = np.average(np.vstack(inward_dirs), axis=0, weights=weights_np)
        inward_dir = inward_dir - np.dot(inward_dir, short_dir) * short_dir
        inward_dir = self._normalized_xy(inward_dir)
        if inward_dir is None:
            return None

        edge_midpoint = np.average(np.vstack(edge_midpoints), axis=0, weights=weights_np)
        edge_points = np.vstack(
            [point for obs in observations for point in (obs.p0_base, obs.p1_base)]
        )
        observed_width = float(
            np.percentile(edge_points @ short_dir, 98.0)
            - np.percentile(edge_points @ short_dir, 2.0)
        )
        if self.reconstruct_short_edge_width:
            edge_center = edge_midpoint
        else:
            short_coords = edge_points @ short_dir
            min_short = float(np.min(short_coords))
            max_short = float(np.max(short_coords))
            center_short = 0.5 * (min_short + max_short)
            edge_center = edge_midpoint + (
                center_short - float(edge_midpoint @ short_dir)
            ) * short_dir

        edge_center[2] = self.board_plane_z
        half_width = 0.5 * self.BOARD_SIZE_X
        reconstructed_p0 = edge_center - half_width * short_dir
        reconstructed_p1 = edge_center + half_width * short_dir
        reconstructed_p0[2] = self.board_plane_z
        reconstructed_p1[2] = self.board_plane_z
        return FusedBoardEdge(
            center_base=edge_center,
            board_x_axis=short_dir,
            board_y_axis=inward_dir,
            reconstructed_p0_base=reconstructed_p0,
            reconstructed_p1_base=reconstructed_p1,
            observed_width_base=observed_width,
        )

    def _multicamera_short_edge_target_xyz(
        self,
        task: Task,
        obs,
    ) -> tuple[float, float, float] | None:
        observations = self._multicamera_board_edge_observations(obs)
        if not observations:
            self.get_logger().warn(
                "TransportToMean multi-camera detector found no visible board edge."
            )
            return None

        fused = self._fused_board_axes(observations)
        if fused is None:
            self.get_logger().warn("TransportToMean could not fuse board edge observations.")
            return None

        visible_edge_center = fused.center_base
        board_x_axis = fused.board_x_axis
        board_y_axis = fused.board_y_axis
        board_origin = visible_edge_center - self.visible_short_edge_board_y * board_y_axis
        board_origin[2] = self.board_plane_z
        edge_yaw = float(np.arctan2(board_x_axis[1], board_x_axis[0]))
        self._log_trial_config_board_pose_error(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            edge_yaw,
            source="multi-camera short-edge fit",
        )
        target_base = self._target_xyz_from_board_pose(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            source="multi-camera short-edge fit",
        )
        if target_base is None:
            return None

        self._log_estimated_final_board_pose(
            source="multi-camera short-edge fit",
            board_origin=board_origin,
            board_x_axis=board_x_axis,
            board_y_axis=board_y_axis,
            target_base=target_base,
            task=task,
        )
        self.get_logger().info(
            "TransportToMean multi-camera board pose: "
            f"cameras={[obs.camera_name for obs in observations]}, "
            f"visible_edge_center={np.round(visible_edge_center, 4).tolist()}, "
            f"reconstructed_edge_p0={np.round(fused.reconstructed_p0_base, 4).tolist()}, "
            f"reconstructed_edge_p1={np.round(fused.reconstructed_p1_base, 4).tolist()}, "
            f"board_origin={np.round(board_origin, 4).tolist()}, "
            f"board_x_axis={np.round(board_x_axis, 4).tolist()}, "
            f"board_y_axis={np.round(board_y_axis, 4).tolist()}, "
            f"observed_edge_width={fused.observed_width_base:.4f}, "
            f"reconstructed_edge_width={self.BOARD_SIZE_X:.4f}, "
            f"reconstruct_width={self.reconstruct_short_edge_width}, "
            f"target_base={np.round(np.asarray(target_base), 4).tolist()}"
        )
        return target_base

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"MagentaSquare.insert_cable() task: {task}")
        self._latest_insertion_event_namespace = ""
        try:
            card_index, port_index = self._task_indices(task)
            self.get_logger().info(
                "MagentaSquare parsed task target: "
                f"card={card_index}, port={port_index}, "
                f"target_mode={self.board_pose_target_mode}, "
                f"rail_target={self._nic_rail_xyz_board(task)}"
            )
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return False

        target_xyz = None
        target_source = ""
        if self.detect_board:
            view_pose = self._tcp_pose_from_observation(get_observation)
            if self.move_to_view_first:
                view_pose = self._move_to_view_pose(get_observation, move_robot)

            target_xyz, target_source = self._detect_target_from_linear_view_search(
                task,
                get_observation,
                move_robot,
                view_pose,
            )
            detected_board = target_xyz is not None
            if not detected_board:
                send_feedback(
                    "Linear +y view-search board detection failed; trying one final short-edge board fit."
                )
                obs = get_observation()
                target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
                detected_board = target_xyz is not None
                if detected_board:
                    target_source = "multi-camera short-edge fit"

            if not detected_board:
                send_feedback("Task board detection failed; no fixed mean fallback.")
            if self.detect_only:
                self.get_logger().info("MagentaSquare detect-only mode exiting.")
                return detected_board

        if target_xyz is None:
            self.get_logger().error(
                "MagentaSquare could not localize the task board from magenta marker "
                "or short-edge fallback."
            )
            return False

        start_pose = self._tcp_pose_from_observation(get_observation)
        send_feedback(
            f"Transporting TCP to {target_source} target position "
            f"x={target_xyz[0]:.4f}, y={target_xyz[1]:.4f}, z={target_xyz[2]:.4f}"
        )
        self._log_trial_config_target_error(task, target_xyz, target_source)

        target_pose = self._move_to_transport_target_pose(
            move_robot,
            start_pose,
            target_xyz,
        )
        if self.tilt_tip_down_after_transport:
            send_feedback("Tilting TCP to align cable tip with downward insertion axis.")
            target_pose = self._move_to_tip_down_pose(
                move_robot,
                self._tcp_pose_from_observation(get_observation),
            )
        if self.descend_after_transport:
            send_feedback("Descending TCP for insertion.")
            target_pose = self._descend_until_inserted(
                task,
                move_robot,
                self._tcp_pose_from_observation(get_observation),
            )
        final_tcp_pose = self._tcp_pose_from_observation(get_observation)
        self.get_logger().info(
            "MagentaSquare final TCP pose after transport/alignment/descent: "
            f"xyz={[round(final_tcp_pose.position.x, 5), round(final_tcp_pose.position.y, 5), round(final_tcp_pose.position.z, 5)]}, "
            f"orientation_xyzw={[round(final_tcp_pose.orientation.x, 5), round(final_tcp_pose.orientation.y, 5), round(final_tcp_pose.orientation.z, 5), round(final_tcp_pose.orientation.w, 5)]}"
        )
        self._log_trial_config_final_pose_error(
            task,
            target_xyz,
            final_tcp_pose,
            target_source,
        )

        if self.hold_sec > 0.0:
            hold_steps = max(1, int(self.hold_sec / self.dt))
            for _ in range(hold_steps):
                self.set_pose_target(move_robot=move_robot, pose=target_pose)
                self.sleep_for(self.dt)

        self.get_logger().info("MagentaSquare.insert_cable() exiting successfully.")
        return True


# Keep legacy static-helper references working after splitting this file from
# TransportToMean.py. The aic_model loader will use the MagentaSquare class.
TransportToMean = MagentaSquare
