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
from rclpy.time import Time
from tf2_ros import TransformException


@dataclass
class BoardEdgePixels:
    p0_px: np.ndarray
    p1_px: np.ndarray
    inward_px: np.ndarray
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


class TransportToMean(Policy):
    """Transport the TCP using a board pose estimated from the visible short edge.

    The policy uses only camera observations and fixed task geometry. It assumes
    the task board is flat on a known plane in base_link (`board_plane_z`) and
    that the visible lower board boundary is the short edge at board-frame
    y = -BOARD_SIZE_Y / 2.

    Runtime flow:
    1. Optionally slide the TCP in base -y until at least one camera sees the
       short edge clearly.
    2. Detect the best dark line segment in each of the left, center, and right
       camera images.
    3. Project each segment endpoint onto the fixed board plane using camera
       intrinsics and TF.
    4. Fuse the projected segments into a board x-axis, board y-axis, and edge
       center. The physical short-edge width is reconstructed as exactly
       BOARD_SIZE_X to reduce jitter from partial line detections.
    5. Recover the board origin from the known edge offset and move to either
       the NIC rail target or the legacy mean-port target in that board frame.
       The TCP orientation is preserved exactly from the pre-transport pose.
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

    MEAN_SFP_PORT_POSITIONS_BOARD = {
        (0, 0): (-0.069518, -0.187365, 0.133476),
        (0, 1): (-0.092718, -0.187365, 0.133476),
        (1, 0): (-0.069518, -0.147365, 0.133476),
        (1, 1): (-0.092718, -0.147365, 0.133476),
        (2, 0): (-0.069518, -0.107365, 0.133476),
        (2, 1): (-0.092718, -0.107365, 0.133476),
        (3, 0): (-0.069518, -0.067365, 0.133476),
        (3, 1): (-0.092718, -0.067365, 0.133476),
        (4, 0): (-0.069518, -0.027365, 0.133476),
        (4, 1): (-0.092718, -0.027365, 0.133476),
    }

    # Mean port centers in robot base_link coordinates. These are the world-frame
    # task-board means transformed through the default robot spawn pose
    # (x=-0.2, y=0.2, z=1.14, yaw=-pi).
    MEAN_SFP_PORT_POSITIONS_BASE_LINK = {
        (0, 0): (-0.431619, 0.198151, 0.133476),
        (0, 1): (-0.454707, 0.199690, 0.133476),
        (1, 0): (-0.428964, 0.237958, 0.133476),
        (1, 1): (-0.452052, 0.239498, 0.133476),
        (2, 0): (-0.426309, 0.277766, 0.133476),
        (2, 1): (-0.449398, 0.279305, 0.133476),
        (3, 0): (-0.423654, 0.317573, 0.133476),
        (3, 1): (-0.446743, 0.319113, 0.133476),
        (4, 0): (-0.421000, 0.357380, 0.133476),
        (4, 1): (-0.444088, 0.358920, 0.133476),
    }

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.z_offset = float(os.getenv("AIC_TRANSPORT_MEAN_Z_OFFSET", "0.12"))
        self.duration_sec = float(os.getenv("AIC_TRANSPORT_MEAN_DURATION_SEC", "3"))
        self.dt = float(os.getenv("AIC_TRANSPORT_MEAN_DT", "0.05"))
        self.hold_sec = float(os.getenv("AIC_TRANSPORT_MEAN_HOLD_SEC", "10.0"))
        self.min_target_z = float(os.getenv("AIC_TRANSPORT_MIN_TARGET_Z", "0.25"))
        self.detect_board = self._env_flag("AIC_TRANSPORT_DETECT_BOARD", default=True)
        self.detect_only = self._env_flag("AIC_TRANSPORT_DETECT_ONLY", default=False)
        self.move_to_view_first = self._env_flag(
            "AIC_TRANSPORT_MOVE_TO_VIEW_FIRST",
            default=True,
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
        self.constrain_board_pose_bounds = self._env_flag(
            "AIC_TRANSPORT_CONSTRAIN_BOARD_POSE_BOUNDS",
            default=True,
        )
        self.board_origin_base_x_min = float(
            os.getenv("AIC_TRANSPORT_BOARD_ORIGIN_BASE_X_MIN", "-0.39")
        )
        self.board_origin_base_x_max = float(
            os.getenv("AIC_TRANSPORT_BOARD_ORIGIN_BASE_X_MAX", "-0.31")
        )
        self.board_origin_base_y_min = float(
            os.getenv("AIC_TRANSPORT_BOARD_ORIGIN_BASE_Y_MIN", "0.30")
        )
        self.board_origin_base_y_max = float(
            os.getenv("AIC_TRANSPORT_BOARD_ORIGIN_BASE_Y_MAX", "0.46")
        )
        self.board_yaw_base_min = float(
            os.getenv("AIC_TRANSPORT_BOARD_YAW_BASE_MIN", "-0.25")
        )
        self.board_yaw_base_max = float(
            os.getenv("AIC_TRANSPORT_BOARD_YAW_BASE_MAX", "0.12")
        )
        self.trial_config_path = os.getenv("AIC_TRANSPORT_TRIAL_CONFIG", "/home/jk/ws_aic/src/aic/outputs/configs/trial_000004.yaml").strip()
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

    def _mean_target_xyz(self, task: Task) -> tuple[float, float, float]:
        x, y, z = self._mean_port_xyz_base_link(task)
        return x, y, max(z + self.z_offset, self.min_target_z)

    def _mean_port_xyz_board(self, task: Task) -> tuple[float, float, float]:
        card_index, port_index = self._task_indices(task)
        if (card_index, port_index) not in self.MEAN_SFP_PORT_POSITIONS_BOARD:
            raise ValueError(
                "Unsupported SFP target "
                f"card={card_index}, port={port_index}; expected card 0..4 and port 0..1"
            )
        return self.MEAN_SFP_PORT_POSITIONS_BOARD[(card_index, port_index)]

    def _mean_port_xyz_base_link(self, task: Task) -> tuple[float, float, float]:
        card_index, port_index = self._task_indices(task)
        if (card_index, port_index) not in self.MEAN_SFP_PORT_POSITIONS_BASE_LINK:
            raise ValueError(
                "Unsupported SFP target "
                f"card={card_index}, port={port_index}; expected card 0..4 and port 0..1"
            )
        return self.MEAN_SFP_PORT_POSITIONS_BASE_LINK[(card_index, port_index)]

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
        if self.board_pose_target_mode in ("mean_port", "port", "mean"):
            return self._mean_port_xyz_board(task)
        raise ValueError(
            "Unsupported AIC_TRANSPORT_BOARD_POSE_TARGET_MODE "
            f"{self.board_pose_target_mode!r}; expected 'nic_rail' or 'mean_port'"
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
    ) -> Pose:
        return Pose(
            position=Point(
                x=start_pose.position.x
                + fraction * (target_xyz[0] - start_pose.position.x),
                y=start_pose.position.y
                + fraction * (target_xyz[1] - start_pose.position.y),
                z=start_pose.position.z
                + fraction * (target_xyz[2] - start_pose.position.z),
            ),
            orientation=start_pose.orientation,
        )

    def _move_to_pose(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        target_xyz: tuple[float, float, float],
        *,
        duration_sec: float,
    ) -> Pose:
        steps = max(1, int(duration_sec / self.dt))
        for step in range(steps + 1):
            fraction = step / steps
            pose = self._interpolated_pose(start_pose, target_xyz, fraction)
            self.set_pose_target(move_robot=move_robot, pose=pose)
            self.sleep_for(self.dt)
        return self._interpolated_pose(start_pose, target_xyz, 1.0)

    def _move_to_view_pose(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> None:
        self._move_to_short_edge_view(get_observation, move_robot)

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
        scan_z = max(start_pose.position.z, self.view_z)
        current_pose = start_pose
        self.get_logger().info(
            "TransportToMean scanning for full short-edge view "
            f"start_y={start_pose.position.y:.4f}, y_step={self.short_edge_scan_y_step:.4f}, "
            f"max_steps={self.short_edge_scan_max_steps}, z={scan_z:.4f}, "
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
    def _normalized_xy(vector: np.ndarray) -> np.ndarray | None:
        xy = np.asarray(vector, dtype=np.float64).reshape(-1)[:2]
        norm = float(np.linalg.norm(xy))
        if norm <= 1e-9:
            return None
        return np.array([xy[0] / norm, xy[1] / norm, 0.0], dtype=np.float64)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _angle_in_board_yaw_bounds(self, yaw: float) -> bool:
        yaw = self._wrap_angle(yaw)
        yaw_min = self._wrap_angle(self.board_yaw_base_min)
        yaw_max = self._wrap_angle(self.board_yaw_base_max)
        if yaw_min <= yaw_max:
            return yaw_min <= yaw <= yaw_max
        return yaw >= yaw_min or yaw <= yaw_max

    def _board_origin_in_bounds(self, origin_xy: np.ndarray) -> bool:
        if not self.constrain_board_pose_bounds:
            return True
        x, y = np.asarray(origin_xy, dtype=np.float64).reshape(2)
        return (
            self.board_origin_base_x_min <= x <= self.board_origin_base_x_max
            and self.board_origin_base_y_min <= y <= self.board_origin_base_y_max
        )

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

        scale = (self.board_plane_z - origin_base[2]) / ray_base[2]
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

    def _true_mean_target_xy_from_trial_config(
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
        expected = self._true_mean_target_xy_from_trial_config(task)
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
        expected = self._true_mean_target_xy_from_trial_config(task)
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

    @staticmethod
    def _dark_board_mask(rgb: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 180, 115]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel, iterations=1)
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
        if self.constrain_board_pose_bounds and (
            not self._board_origin_in_bounds(board_origin[:2])
            or not self._angle_in_board_yaw_bounds(edge_yaw)
        ):
            self.get_logger().warn(
                "TransportToMean rejected short-edge board pose outside eval-like bounds: "
                f"origin={np.round(board_origin, 4).tolist()}, yaw={edge_yaw:.4f}"
            )
            return None
        target_base = self._target_xyz_from_board_pose(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            source="multi-camera short-edge fit",
        )
        if target_base is None:
            return None

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
        self.get_logger().info(f"TransportToMean.insert_cable() task: {task}")
        try:
            card_index, port_index = self._task_indices(task)
            self.get_logger().info(
                "TransportToMean parsed task target: "
                f"card={card_index}, port={port_index}, "
                f"target_mode={self.board_pose_target_mode}, "
                f"rail_target={self._nic_rail_xyz_board(task)}, "
                f"board_mean={self.MEAN_SFP_PORT_POSITIONS_BOARD.get((card_index, port_index))}, "
                f"base_fallback_mean={self.MEAN_SFP_PORT_POSITIONS_BASE_LINK.get((card_index, port_index))}"
            )
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return False

        target_xyz = None
        target_source = "fixed mean"
        if self.detect_board:
            if self.move_to_view_first:
                self._move_to_view_pose(get_observation, move_robot)

            obs = get_observation()
            target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
            detected_board = target_xyz is not None
            if detected_board:
                target_source = "multi-camera short-edge fit"

            if not detected_board:
                send_feedback("Task board detection failed; falling back to fixed mean.")
            if self.detect_only:
                self.get_logger().info("TransportToMean detect-only mode exiting.")
                return detected_board

        if target_xyz is None:
            try:
                target_xyz = self._mean_target_xyz(task)
            except ValueError as ex:
                self.get_logger().error(str(ex))
                return False

        start_pose = self._tcp_pose_from_observation(get_observation)
        send_feedback(
            f"Transporting TCP to {target_source} target position "
            f"x={target_xyz[0]:.4f}, y={target_xyz[1]:.4f}, z={target_xyz[2]:.4f}"
        )
        self._log_trial_config_target_error(task, target_xyz, target_source)

        target_pose = self._move_to_pose(
            move_robot,
            start_pose,
            target_xyz,
            duration_sec=self.duration_sec,
        )
        final_tcp_pose = self._tcp_pose_from_observation(get_observation)
        self.get_logger().info(
            "TransportToMean final TCP pose after transport: "
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

        self.get_logger().info("TransportToMean.insert_cable() exiting successfully.")
        return True
