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


@dataclass
class LocalizedTransportTarget:
    target_xyz: tuple[float, float, float]
    board_origin: np.ndarray
    board_x_axis: np.ndarray
    board_y_axis: np.ndarray
    target_board_xyz: np.ndarray
    source: str


@dataclass
class SfpNicPortPose:
    entrance_board: np.ndarray
    port_center_board: np.ndarray
    port_axis_board: np.ndarray
    width_axis_board: np.ndarray
    height_axis_board: np.ndarray


@dataclass
class SfpNicPortLocalization:
    entrance_base: np.ndarray
    port_center_base: np.ndarray
    port_axis_base: np.ndarray
    width_axis_base: np.ndarray
    height_axis_base: np.ndarray
    rail_translation_m: float
    rail_yaw_rad: float
    score: float
    confidence: float
    observed_camera_count: int
    source: str
    accepted: bool


@dataclass
class SfpNicPortImageScore:
    score: float
    rectangular_edge_score: float
    axis_edge_score: float
    rail_parallel_score: float
    rectangle_dark_score: float
    center_dark_score: float
    scale_score: float


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
    5. Move near the target rail/port in the recovered board frame.
    6. For SFP-to-NIC tasks, run a close SDF-informed port-entrance localizer,
       align the estimated SFP tip axis to the detected port axis, and descend
       along that port axis.
    7. For SC tasks, keep the existing tip-down/base-z descent behavior,
       stopping early if the scoring insertion event is observed.
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
    SFP_NIC_CARD_LINK_TRANSLATION_MOUNT = np.array(
        [-0.002, -0.01785, 0.0899],
        dtype=np.float64,
    )
    SFP_NIC_CARD_LINK_ROLL = -1.57
    SFP_NIC_PORT_TRANSLATION_CARD = {
        0: np.array([0.01295, -0.031572, 0.00501], dtype=np.float64),
        1: np.array([-0.01025, -0.031572, 0.00501], dtype=np.float64),
    }
    SFP_NIC_PORT_ROLL = 4.69895
    SFP_NIC_PORT_ENTRANCE_OFFSET_PORT = np.array(
        [0.0, 0.0, -0.0458],
        dtype=np.float64,
    )
    SFP_NIC_PORT_OPENING_HALF_WIDTH = 0.008112
    SFP_NIC_PORT_OPENING_HALF_HEIGHT = 0.0056
    SFP_NIC_MAX_RECOVERY_RETRIES = 3
    SFP_NIC_FRONT_RIDGE_LINES_CARD = (
        (
            np.array([-0.06435, -0.074646, 0.01352], dtype=np.float64),
            np.array([0.04905, -0.074646, 0.01352], dtype=np.float64),
        ),
        (
            np.array([-0.06435, -0.074646, -0.000751], dtype=np.float64),
            np.array([0.04905, -0.074646, -0.000751], dtype=np.float64),
        ),
        (
            np.array([-0.04485, -0.073499, 0.015586], dtype=np.float64),
            np.array([0.03215, -0.073499, 0.015586], dtype=np.float64),
        ),
    )
    SFP_NIC_CAGE_CENTER_X_CARD = {
        0: 0.012963,
        1: -0.010237,
    }
    SFP_NIC_CAGE_SIDE_HALF_WIDTH_M = 0.007806
    SFP_NIC_CAGE_Y_BACK = -0.077314
    SFP_NIC_CAGE_Y_FRONT = -0.028738
    SFP_NIC_CAGE_Z_MID = 0.00547
    SFP_NIC_CAGE_Z_TOP = 0.010484
    SFP_NIC_CAGE_Z_BOTTOM = 0.000046
    SFP_NIC_CIRCULAR_FEATURES_CARD = (
        (np.array([0.03175, -0.063, -0.000988], dtype=np.float64), 0.00268),
        (np.array([-0.02225, -0.0675, -0.000986], dtype=np.float64), 0.00268),
    )
    SC_RAIL_X = -0.075
    SC_RAIL_Y = {
        0: 0.0295,
        1: 0.0705,
    }
    SC_RAIL_APPROACH_Z = 0.133476
    SC_PORT_ENTRANCE_Z = 0.03014

    MAGENTA_SQUARE_SIZE = 0.095
    MAGENTA_CENTER_BOARD = np.array([-0.075, 0.150, 0.011], dtype=np.float64)
    MAGENTA_GAP_EDGE_BOARD_Y = 0.1025
    MAGENTA_PLUS_Y_EDGE_BOARD_Y = 0.1975
    MAGENTA_PLUS_Y_BOUNDARY_OFFSET = 0.015

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._last_localized_target: LocalizedTransportTarget | None = None
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
        self.magenta_edge_thickness_tolerance_m = float(
            os.getenv(
                "AIC_MAGENTA_EDGE_THICKNESS_TOLERANCE_M",
                str(2.5 * self.magenta_edge_support_tolerance_m),
            )
        )
        self.magenta_gap_fraction_weight = float(
            os.getenv("AIC_MAGENTA_GAP_FRACTION_WEIGHT", "3.0")
        )
        self.magenta_gap_thickness_weight = float(
            os.getenv("AIC_MAGENTA_GAP_THICKNESS_WEIGHT", "1.0")
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
        self.magenta_require_geometry_fit_for_target = self._env_flag(
            "AIC_MAGENTA_REQUIRE_GEOMETRY_FIT_FOR_TARGET",
            default=True,
        )
        self.magenta_spawn_require_consensus = self._env_flag(
            "AIC_MAGENTA_SPAWN_REQUIRE_CONSENSUS",
            default=True,
        )
        self.magenta_edge_width_tolerance_m = float(
            os.getenv("AIC_MAGENTA_BOARD_EDGE_WIDTH_TOLERANCE_M", "0.07")
        )
        self.magenta_edge_marker_distance_tolerance_m = float(
            os.getenv("AIC_MAGENTA_BOARD_EDGE_MARKER_DISTANCE_TOLERANCE_M", "0.12")
        )
        self.magenta_edge_parallel_min_dot = float(
            os.getenv("AIC_MAGENTA_BOARD_EDGE_PARALLEL_MIN_DOT", "0.85")
        )
        self.magenta_linear_view_enabled = self._env_flag(
            "AIC_MAGENTA_LINEAR_VIEW_ENABLED",
            default=True,
        )
        self.magenta_linear_view_y_distance_m = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_Y_DISTANCE_M", "0.8")
        )
        self.magenta_linear_view_steps = int(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_STEPS", "8")
        )
        self.magenta_linear_view_move_duration_sec = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_MOVE_DURATION_SEC", "0.45")
        )
        self.magenta_linear_view_hold_sec = float(
            os.getenv("AIC_MAGENTA_LINEAR_VIEW_HOLD_SEC", "0.15")
        )
        self.magenta_sc_circular_view_enabled = self._env_flag(
            "AIC_MAGENTA_SC_CIRCULAR_VIEW_ENABLED",
            default=True,
        )
        self.magenta_sc_circular_view_radius_m = float(
            os.getenv("AIC_MAGENTA_SC_CIRCULAR_VIEW_RADIUS_M", "0.15")
        )
        self.magenta_sc_circular_view_steps = int(
            os.getenv("AIC_MAGENTA_SC_CIRCULAR_VIEW_STEPS", "12")
        )
        self.magenta_sc_circular_view_move_duration_sec = float(
            os.getenv(
                "AIC_MAGENTA_SC_CIRCULAR_VIEW_MOVE_DURATION_SEC",
                str(self.magenta_linear_view_move_duration_sec),
            )
        )
        self.magenta_sc_circular_view_hold_sec = float(
            os.getenv(
                "AIC_MAGENTA_SC_CIRCULAR_VIEW_HOLD_SEC",
                str(self.magenta_linear_view_hold_sec),
            )
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
        self.sc_rail_approach_z = float(
            os.getenv(
                "AIC_TRANSPORT_SC_RAIL_APPROACH_Z",
                str(self.SC_RAIL_APPROACH_Z),
            )
        )
        self.sc_rail_x_offset = float(os.getenv("AIC_TRANSPORT_SC_RAIL_X_OFFSET", "0.0"))
        self.sc_rail_y_offset = float(os.getenv("AIC_TRANSPORT_SC_RAIL_Y_OFFSET", "0.0"))
        self.short_edge_scan_y_step = float(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_Y_STEP", "-0.04")
        )
        self.short_edge_scan_max_steps = int(
            os.getenv("AIC_TRANSPORT_SHORT_EDGE_SCAN_MAX_STEPS", "10")
        )
        self.sfp_nic_short_edge_scan_max_distance_m = float(
            os.getenv("AIC_SFP_NIC_SHORT_EDGE_SCAN_MAX_DISTANCE_M", "0.3")
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
        self.tip_position_tcp = self._configured_tip_position_tcp()
        self.tip_width_axis_tcp = self._normalized_vector(
            np.array(
                [
                    float(os.getenv("AIC_TRANSPORT_TIP_WIDTH_AXIS_TCP_X", "1.0")),
                    float(os.getenv("AIC_TRANSPORT_TIP_WIDTH_AXIS_TCP_Y", "0.0")),
                    float(os.getenv("AIC_TRANSPORT_TIP_WIDTH_AXIS_TCP_Z", "0.0")),
                ],
                dtype=np.float64,
            )
        )
        if self.tip_width_axis_tcp is None:
            self.tip_width_axis_tcp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.descend_after_transport = self._env_flag(
            "AIC_TRANSPORT_DESCEND_AFTER_TRANSPORT",
            default=True,
        )
        self.descend_step_m = float(os.getenv("AIC_TRANSPORT_DESCEND_STEP_M", "0.0005"))
        self.descend_dt_sec = float(os.getenv("AIC_TRANSPORT_DESCEND_DT_SEC", "0.05"))
        self.sc_descend_speed_multiplier = float(
            os.getenv("AIC_TRANSPORT_SC_DESCEND_SPEED_MULTIPLIER", "2.0")
        )
        self.descend_max_distance_m = float(
            os.getenv(
                "AIC_TRANSPORT_DESCEND_MAX_DISTANCE_M",
                str(max(self.z_offset + 0.015, 0.0)),
            )
        )
        self.sc_descend_max_distance_m = float(
            os.getenv(
                "AIC_TRANSPORT_SC_DESCEND_MAX_DISTANCE_M",
                str(max(self.z_offset + self.sc_rail_approach_z + 0.02, 0.0)),
            )
        )
        self.sc_descend_stop_z = float(
            os.getenv("AIC_TRANSPORT_SC_DESCEND_STOP_Z", str(self.SC_PORT_ENTRANCE_Z))
        )
        self.sc_descend_stop_z_margin = float(
            os.getenv("AIC_TRANSPORT_SC_DESCEND_STOP_Z_MARGIN", "0.01")
        )
        self.descend_wait_for_insertion_sec = float(
            os.getenv("AIC_TRANSPORT_DESCEND_WAIT_FOR_INSERTION_SEC", "5.0")
        )
        self.sfp_nic_insertion_enabled = self._env_flag(
            "AIC_SFP_NIC_INSERTION_ENABLED",
            default=True,
        )
        self.sfp_nic_localizer_enabled = self._env_flag(
            "AIC_SFP_NIC_LOCALIZER_ENABLED",
            default=True,
        )
        self.sfp_nic_pair_localizer_enabled = self._env_flag(
            "AIC_SFP_NIC_PAIR_LOCALIZER_ENABLED",
            default=True,
        )
        self.sfp_nic_front_geometry_localizer_enabled = self._env_flag(
            "AIC_SFP_NIC_FRONT_GEOMETRY_LOCALIZER_ENABLED",
            default=True,
        )
        self.sfp_nic_front_geometry_score_weight = float(
            os.getenv("AIC_SFP_NIC_FRONT_GEOMETRY_SCORE_WEIGHT", "0.50")
        )
        self.sfp_nic_rail_parallel_score_enabled = self._env_flag(
            "AIC_SFP_NIC_RAIL_PARALLEL_SCORE_ENABLED",
            default=True,
        )
        self.sfp_nic_rail_parallel_port_weight = float(
            os.getenv("AIC_SFP_NIC_RAIL_PARALLEL_PORT_WEIGHT", "0.14")
        )
        self.sfp_nic_rail_parallel_pair_weight = float(
            os.getenv("AIC_SFP_NIC_RAIL_PARALLEL_PAIR_WEIGHT", "0.12")
        )
        self.sfp_nic_rail_parallel_angle_window_rad = float(
            os.getenv(
                "AIC_SFP_NIC_RAIL_PARALLEL_ANGLE_WINDOW_RAD",
                str(np.deg2rad(35.0)),
            )
        )
        self.sfp_nic_rail_parallel_tolerance_rad = float(
            os.getenv(
                "AIC_SFP_NIC_RAIL_PARALLEL_TOLERANCE_RAD",
                str(np.deg2rad(12.0)),
            )
        )
        self.sfp_nic_debug_overlay_enabled = self._env_flag(
            "AIC_SFP_NIC_DEBUG_OVERLAY_ENABLED",
            default=False,
        )
        self.sfp_nic_debug_overlay_save = self._env_flag(
            "AIC_SFP_NIC_DEBUG_OVERLAY_SAVE",
            default=True,
        )
        self.sfp_nic_debug_overlay_stream = self._env_flag(
            "AIC_SFP_NIC_DEBUG_OVERLAY_STREAM",
            default=False,
        )
        self.sfp_nic_debug_overlay_dir = Path(
            os.getenv(
                "AIC_SFP_NIC_DEBUG_OVERLAY_DIR",
                "outputs/debug/sfp_nic_port_pair",
            )
        )
        self.sfp_nic_debug_overlay_topic = os.getenv(
            "AIC_SFP_NIC_DEBUG_OVERLAY_TOPIC",
            "/debug/sfp_nic_port_pair_overlay",
        )
        self._sfp_nic_debug_overlay_seq = 0
        self._sfp_nic_debug_overlay_pub = None
        self._sfp_nic_debug_image_msg_type = None
        if self.sfp_nic_debug_overlay_enabled and self.sfp_nic_debug_overlay_stream:
            self._setup_sfp_nic_debug_overlay_stream()
        self.sfp_nic_allow_prior_fallback = self._env_flag(
            "AIC_SFP_NIC_ALLOW_PRIOR_FALLBACK",
            default=True,
        )
        self.sfp_nic_align_to_port = self._env_flag(
            "AIC_SFP_NIC_ALIGN_TO_PORT",
            default=True,
        )
        self.sfp_nic_rail_translation_min = float(
            os.getenv("AIC_SFP_NIC_RAIL_TRANSLATION_MIN", "-0.0215")
        )
        self.sfp_nic_rail_translation_max = float(
            os.getenv("AIC_SFP_NIC_RAIL_TRANSLATION_MAX", "0.0234")
        )
        self.sfp_nic_rail_yaw_min = float(
            os.getenv("AIC_SFP_NIC_RAIL_YAW_MIN", str(-np.deg2rad(10.0)))
        )
        self.sfp_nic_rail_yaw_max = float(
            os.getenv("AIC_SFP_NIC_RAIL_YAW_MAX", str(np.deg2rad(10.0)))
        )
        self.sfp_nic_translation_grid_steps = int(
            os.getenv("AIC_SFP_NIC_TRANSLATION_GRID_STEPS", "9")
        )
        self.sfp_nic_yaw_grid_steps = int(
            os.getenv("AIC_SFP_NIC_YAW_GRID_STEPS", "9")
        )
        self.sfp_nic_localizer_min_confidence = float(
            os.getenv("AIC_SFP_NIC_LOCALIZER_MIN_CONFIDENCE", "0.18")
        )
        self.sfp_nic_localizer_prior_weight = float(
            os.getenv("AIC_SFP_NIC_LOCALIZER_PRIOR_WEIGHT", "0.025")
        )
        self.sfp_nic_view_search_enabled = self._env_flag(
            "AIC_SFP_NIC_VIEW_SEARCH_ENABLED",
            default=True,
        )
        self.sfp_nic_view_search_offset_m = float(
            os.getenv("AIC_SFP_NIC_VIEW_SEARCH_OFFSET_M", "0.010")
        )
        self.sfp_nic_view_search_duration_sec = float(
            os.getenv("AIC_SFP_NIC_VIEW_SEARCH_DURATION_SEC", "0.35")
        )
        self.sfp_nic_preinsert_clearance_m = float(
            os.getenv("AIC_SFP_NIC_PREINSERT_CLEARANCE_M", "0.012")
        )
        self.sfp_nic_descent_distance_m = float(
            os.getenv("AIC_SFP_NIC_DESCENT_DISTANCE_M", "0.065")
        )
        self.sfp_nic_force_stop_n = float(os.getenv("AIC_SFP_NIC_FORCE_STOP_N", "30.0"))
        self.sfp_nic_axial_force_delta_stop_n = float(
            os.getenv("AIC_SFP_NIC_AXIAL_FORCE_DELTA_STOP_N", "2.0")
        )
        self.sfp_nic_axial_force_delta_min_distance_m = float(
            os.getenv("AIC_SFP_NIC_AXIAL_FORCE_DELTA_MIN_DISTANCE_M", "0.001")
        )
        self.sfp_nic_offset_exploration_enabled = self._env_flag(
            "AIC_SFP_NIC_OFFSET_EXPLORATION_ENABLED",
            default=True,
        )
        self.sfp_nic_offset_exploration_steps_x = int(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_STEPS_X", "1")
        )
        self.sfp_nic_offset_exploration_steps_y = int(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_STEPS_Y", "1")
        )
        self.sfp_nic_offset_exploration_max_probes = int(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_MAX_PROBES", "3")
        )
        self.sfp_nic_offset_exploration_max_x_m = float(
            os.getenv(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_MAX_X_M",
                str(self.SFP_NIC_PORT_OPENING_HALF_WIDTH),
            )
        )
        self.sfp_nic_offset_exploration_max_y_m = float(
            os.getenv(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_MAX_Y_M",
                str(self.SFP_NIC_PORT_OPENING_HALF_HEIGHT),
            )
        )
        self.sfp_nic_offset_exploration_backoff_m = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_BACKOFF_M", "0.0005")
        )
        self.sfp_nic_offset_exploration_descent_m = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_DESCENT_M", "0.030")
        )
        self.sfp_nic_offset_exploration_insertion_check_depth_m = float(
            os.getenv(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_INSERTION_CHECK_DEPTH_M",
                "0.004",
            )
        )
        self.sfp_nic_offset_exploration_slide_in_enabled = self._env_flag(
            "AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_IN_ENABLED",
            default=True,
        )
        self.sfp_nic_offset_exploration_slide_force_min_n = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_FORCE_MIN_N", "0.10")
        )
        self.sfp_nic_offset_exploration_slide_gain_m_per_n = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_GAIN_M_PER_N", "0.00015")
        )
        self.sfp_nic_offset_exploration_slide_step_max_m = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_STEP_MAX_M", "0.00075")
        )
        self.sfp_nic_offset_exploration_slide_total_max_m = float(
            os.getenv("AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_TOTAL_MAX_M", "0.00400")
        )
        self.sfp_nic_offset_exploration_slide_direction_tracking_enabled = (
            self._env_flag(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_DIRECTION_TRACKING_ENABLED",
                default=True,
            )
        )
        self.sfp_nic_offset_exploration_slide_direction_memory_alpha = float(
            os.getenv(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_DIRECTION_MEMORY_ALPHA",
                "0.95",
            )
        )
        self.sfp_nic_offset_exploration_slide_direction_step_m = float(
            os.getenv(
                "AIC_SFP_NIC_OFFSET_EXPLORATION_SLIDE_DIRECTION_STEP_M",
                "0.00075",
            )
        )
        self.sfp_nic_recovery_enabled = self._env_flag(
            "AIC_SFP_NIC_RECOVERY_ENABLED",
            default=True,
        )
        self.sfp_nic_recovery_radius_m = float(
            os.getenv("AIC_SFP_NIC_RECOVERY_RADIUS_M", "0.0015")
        )
        self.sfp_nic_recovery_attempts = min(
            max(int(os.getenv("AIC_SFP_NIC_RECOVERY_ATTEMPTS", "3")), 0),
            self.SFP_NIC_MAX_RECOVERY_RETRIES,
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

    def _setup_sfp_nic_debug_overlay_stream(self) -> None:
        try:
            from sensor_msgs.msg import Image
        except ImportError as ex:
            self.get_logger().warn(
                "SFP/NIC debug overlay streaming disabled because sensor_msgs "
                f"Image is unavailable: {ex}"
            )
            return

        try:
            self._sfp_nic_debug_image_msg_type = Image
            self._sfp_nic_debug_overlay_pub = self._parent_node.create_publisher(
                Image,
                self.sfp_nic_debug_overlay_topic,
                10,
            )
        except Exception as ex:
            self._sfp_nic_debug_image_msg_type = None
            self._sfp_nic_debug_overlay_pub = None
            self.get_logger().warn(
                "SFP/NIC debug overlay streaming disabled because publisher "
                f"creation failed: {ex}"
            )

    def _configured_tip_position_tcp(self) -> np.ndarray:
        keys = (
            "AIC_TRANSPORT_TIP_POSITION_TCP_X",
            "AIC_TRANSPORT_TIP_POSITION_TCP_Y",
            "AIC_TRANSPORT_TIP_POSITION_TCP_Z",
        )
        if any(os.getenv(key) is not None for key in keys):
            return np.array(
                [
                    float(os.getenv(keys[0], "0.0")),
                    float(os.getenv(keys[1], "0.0")),
                    float(os.getenv(keys[2], "0.0")),
                ],
                dtype=np.float64,
            )

        offset_along_tip_axis = float(
            os.getenv("AIC_TRANSPORT_TIP_OFFSET_ALONG_AXIS_M", "0.070")
        )
        return offset_along_tip_axis * np.asarray(self.tip_axis_tcp, dtype=np.float64)

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

    def _sc_task_index(self, task: Task) -> int:
        if task.port_name != "sc_port_base":
            raise ValueError(
                f"Unsupported SC port name {task.port_name!r}; expected 'sc_port_base'"
            )
        return self._parse_index_from_suffix(task.target_module_name, "sc_port_")

    def _task_family(self, task: Task) -> str:
        if task.target_module_name.startswith("nic_card_mount_"):
            return "sfp_to_nic"
        if task.target_module_name.startswith("sc_port_"):
            return "sc_to_sc"
        raise ValueError(
            "Unsupported target module "
            f"{task.target_module_name!r}; expected nic_card_mount_* or sc_port_*"
        )

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

    def _sc_rail_xyz_board(self, task: Task) -> tuple[float, float, float]:
        port_index = self._sc_task_index(task)
        if port_index not in self.SC_RAIL_Y:
            raise ValueError(
                "Unsupported SC rail target "
                f"port={port_index}; expected port 0..1"
            )
        return (
            self.SC_RAIL_X + self.sc_rail_x_offset,
            self.SC_RAIL_Y[port_index] + self.sc_rail_y_offset,
            self.sc_rail_approach_z,
        )

    def _board_pose_target_xyz_board(self, task: Task) -> tuple[float, float, float]:
        task_family = self._task_family(task)
        if task_family == "sc_to_sc":
            return self._sc_rail_xyz_board(task)
        if self.board_pose_target_mode in ("nic_rail", "rail"):
            return self._nic_rail_xyz_board(task)
        raise ValueError(
            "Unsupported AIC_TRANSPORT_BOARD_POSE_TARGET_MODE "
            f"{self.board_pose_target_mode!r}; expected 'nic_rail' for SFP-to-NIC"
        )

    def _task_target_summary(self, task: Task) -> str:
        task_family = self._task_family(task)
        if task_family == "sc_to_sc":
            port_index = self._sc_task_index(task)
            return (
                f"family=sc_to_sc, sc_port={port_index}, "
                f"target_mode=sc_rail, rail_target={self._sc_rail_xyz_board(task)}"
            )
        card_index, port_index = self._task_indices(task)
        return (
            f"family=sfp_to_nic, card={card_index}, port={port_index}, "
            f"target_mode={self.board_pose_target_mode}, "
            f"rail_target={self._nic_rail_xyz_board(task)}"
        )

    def _sfp_nic_port_pose_board(
        self,
        *,
        card_index: int,
        port_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
    ) -> SfpNicPortPose:
        if not 0 <= card_index <= 4:
            raise ValueError(f"Unsupported NIC card index {card_index}; expected 0..4")
        if port_index not in self.SFP_NIC_PORT_TRANSLATION_CARD:
            raise ValueError(f"Unsupported SFP port index {port_index}; expected 0 or 1")

        mount_translation_board = np.array(
            [
                self.NIC_RAIL_X + float(rail_translation_m),
                self.NIC_RAIL_Y0 + self.NIC_RAIL_Y_SPACING * card_index,
                0.012,
            ],
            dtype=np.float64,
        )
        mount_rot_board = self._rot_z(float(rail_yaw_rad))
        card_rot_mount = self._rot_x(self.SFP_NIC_CARD_LINK_ROLL)
        port_rot_card = self._rot_x(self.SFP_NIC_PORT_ROLL)
        card_translation_mount = self.SFP_NIC_CARD_LINK_TRANSLATION_MOUNT
        port_translation_card = self.SFP_NIC_PORT_TRANSLATION_CARD[port_index]
        port_rot_mount = card_rot_mount @ port_rot_card
        port_center_mount = card_translation_mount + card_rot_mount @ port_translation_card
        entrance_mount = card_translation_mount + card_rot_mount @ (
            port_translation_card
            + port_rot_card @ self.SFP_NIC_PORT_ENTRANCE_OFFSET_PORT
        )
        port_center_board = mount_translation_board + mount_rot_board @ port_center_mount
        entrance_board = mount_translation_board + mount_rot_board @ entrance_mount
        port_axis_board = self._normalized_vector(mount_rot_board @ port_rot_mount[:, 2])
        width_axis_board = self._normalized_vector(mount_rot_board @ port_rot_mount[:, 0])
        height_axis_board = self._normalized_vector(mount_rot_board @ port_rot_mount[:, 1])
        if port_axis_board is None or width_axis_board is None or height_axis_board is None:
            raise ValueError("Invalid SFP/NIC port geometry basis")
        return SfpNicPortPose(
            entrance_board=entrance_board,
            port_center_board=port_center_board,
            port_axis_board=port_axis_board,
            width_axis_board=width_axis_board,
            height_axis_board=height_axis_board,
        )

    def _sfp_nic_card_points_to_board(
        self,
        points_card: np.ndarray,
        *,
        card_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
    ) -> np.ndarray:
        points_card = np.asarray(points_card, dtype=np.float64).reshape(-1, 3)
        mount_translation_board = np.array(
            [
                self.NIC_RAIL_X + float(rail_translation_m),
                self.NIC_RAIL_Y0 + self.NIC_RAIL_Y_SPACING * card_index,
                0.012,
            ],
            dtype=np.float64,
        )
        mount_rot_board = self._rot_z(float(rail_yaw_rad))
        card_rot_mount = self._rot_x(self.SFP_NIC_CARD_LINK_ROLL)
        points_mount = (
            self.SFP_NIC_CARD_LINK_TRANSLATION_MOUNT.reshape(1, 3)
            + (card_rot_mount @ points_card.T).T
        )
        return mount_translation_board.reshape(1, 3) + (mount_rot_board @ points_mount.T).T

    def _sfp_nic_card_points_to_base(
        self,
        points_card: np.ndarray,
        *,
        card_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
        localized_target: LocalizedTransportTarget,
    ) -> np.ndarray:
        points_board = self._sfp_nic_card_points_to_board(
            points_card,
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
        )
        return np.asarray(
            [
                self._board_xyz_to_base_xyz(
                    point_board,
                    localized_target.board_origin,
                    localized_target.board_x_axis,
                    localized_target.board_y_axis,
                )
                for point_board in points_board
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _line_segments_to_model_points(
        line_segments: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        points: list[np.ndarray] = []
        edge_indices: list[tuple[int, int]] = []
        for p0, p1 in line_segments:
            edge_start = len(points)
            points.extend([p0, p1])
            edge_indices.append((edge_start, edge_start + 1))
        return np.asarray(points, dtype=np.float64), edge_indices

    def _sfp_nic_front_ridge_lines_card(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            (p0.copy(), p1.copy())
            for p0, p1 in self.SFP_NIC_FRONT_RIDGE_LINES_CARD
        ]

    def _sfp_nic_cage_lines_card(self) -> list[tuple[np.ndarray, np.ndarray]]:
        line_segments: list[tuple[np.ndarray, np.ndarray]] = []
        for center_x in self.SFP_NIC_CAGE_CENTER_X_CARD.values():
            x_left = center_x - self.SFP_NIC_CAGE_SIDE_HALF_WIDTH_M
            x_right = center_x + self.SFP_NIC_CAGE_SIDE_HALF_WIDTH_M
            line_segments.extend(
                [
                    (
                        np.array(
                            [
                                x_left,
                                self.SFP_NIC_CAGE_Y_BACK,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                x_left,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                    ),
                    (
                        np.array(
                            [
                                x_right,
                                self.SFP_NIC_CAGE_Y_BACK,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                x_right,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                    ),
                    (
                        np.array(
                            [
                                center_x,
                                self.SFP_NIC_CAGE_Y_BACK,
                                self.SFP_NIC_CAGE_Z_TOP,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                center_x,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_TOP,
                            ],
                            dtype=np.float64,
                        ),
                    ),
                    (
                        np.array(
                            [
                                center_x,
                                self.SFP_NIC_CAGE_Y_BACK,
                                self.SFP_NIC_CAGE_Z_BOTTOM,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                center_x,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_BOTTOM,
                            ],
                            dtype=np.float64,
                        ),
                    ),
                    (
                        np.array(
                            [
                                x_left,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                        np.array(
                            [
                                x_right,
                                self.SFP_NIC_CAGE_Y_FRONT,
                                self.SFP_NIC_CAGE_Z_MID,
                            ],
                            dtype=np.float64,
                        ),
                    ),
                ]
            )
        return line_segments

    def _sfp_nic_circular_feature_lines_card(self) -> list[tuple[np.ndarray, np.ndarray]]:
        line_segments: list[tuple[np.ndarray, np.ndarray]] = []
        ring_samples = 12
        for center, radius in self.SFP_NIC_CIRCULAR_FEATURES_CARD:
            ring_points = [
                center
                + radius
                * np.array(
                    [
                        float(np.cos(theta)),
                        float(np.sin(theta)),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                for theta in np.linspace(0.0, 2.0 * np.pi, ring_samples, endpoint=False)
            ]
            for index, point in enumerate(ring_points):
                line_segments.append((point, ring_points[(index + 1) % ring_samples]))
        return line_segments

    def _sfp_nic_front_geometry_lines_card(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return (
            self._sfp_nic_front_ridge_lines_card()
            + self._sfp_nic_cage_lines_card()
            + self._sfp_nic_circular_feature_lines_card()
        )

    def _sfp_nic_port_localization_for_indices(
        self,
        localized_target: LocalizedTransportTarget,
        *,
        card_index: int,
        port_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
        score: float,
        confidence: float,
        observed_camera_count: int,
        source: str,
        accepted: bool,
    ) -> SfpNicPortLocalization:
        port_pose_board = self._sfp_nic_port_pose_board(
            card_index=card_index,
            port_index=port_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
        )
        entrance_base = np.array(
            self._board_xyz_to_base_xyz(
                port_pose_board.entrance_board,
                localized_target.board_origin,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            ),
            dtype=np.float64,
        )
        port_center_base = np.array(
            self._board_xyz_to_base_xyz(
                port_pose_board.port_center_board,
                localized_target.board_origin,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            ),
            dtype=np.float64,
        )
        port_axis_base = self._normalized_vector(
            self._board_vector_to_base_vector(
                port_pose_board.port_axis_board,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
        )
        width_axis_base = self._normalized_vector(
            self._board_vector_to_base_vector(
                port_pose_board.width_axis_board,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
        )
        height_axis_base = self._normalized_vector(
            self._board_vector_to_base_vector(
                port_pose_board.height_axis_board,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
        )
        if port_axis_base is None or width_axis_base is None or height_axis_base is None:
            raise ValueError("Invalid localized SFP/NIC port basis")
        return SfpNicPortLocalization(
            entrance_base=entrance_base,
            port_center_base=port_center_base,
            port_axis_base=port_axis_base,
            width_axis_base=width_axis_base,
            height_axis_base=height_axis_base,
            rail_translation_m=float(rail_translation_m),
            rail_yaw_rad=float(rail_yaw_rad),
            score=float(score),
            confidence=float(confidence),
            observed_camera_count=int(observed_camera_count),
            source=source,
            accepted=accepted,
        )

    def _sfp_nic_port_localization_from_board_pose(
        self,
        task: Task,
        localized_target: LocalizedTransportTarget,
        *,
        rail_translation_m: float,
        rail_yaw_rad: float,
        score: float,
        confidence: float,
        observed_camera_count: int,
        source: str,
        accepted: bool,
    ) -> SfpNicPortLocalization:
        card_index, port_index = self._task_indices(task)
        return self._sfp_nic_port_localization_for_indices(
            localized_target,
            card_index=card_index,
            port_index=port_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            score=score,
            confidence=confidence,
            observed_camera_count=observed_camera_count,
            source=source,
            accepted=accepted,
        )

    def _sfp_nic_label_pair_by_board_x_zero_line(
        self,
        localizations: dict[int, SfpNicPortLocalization],
        localized_target: LocalizedTransportTarget,
    ) -> dict[int, SfpNicPortLocalization]:
        if len(localizations) != 2:
            return localizations

        def board_x_zero_line_distance(localization: SfpNicPortLocalization) -> float:
            board_xy = self._base_point_to_board_xy(
                localization.entrance_base,
                localized_target.board_origin,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
            return abs(float(board_xy[0]))

        ordered_localizations = sorted(
            localizations.values(),
            key=board_x_zero_line_distance,
        )
        return {
            0: ordered_localizations[0],
            1: ordered_localizations[1],
        }

    def _sfp_nic_port_pair_localizations_from_board_pose(
        self,
        task: Task,
        localized_target: LocalizedTransportTarget,
        *,
        rail_translation_m: float,
        rail_yaw_rad: float,
        score: float,
        confidence: float,
        observed_camera_count: int,
        source: str,
        accepted: bool,
    ) -> tuple[int, dict[int, SfpNicPortLocalization]]:
        card_index, target_port_index = self._task_indices(task)
        raw_localizations = {
            port_index: self._sfp_nic_port_localization_for_indices(
                localized_target,
                card_index=card_index,
                port_index=port_index,
                rail_translation_m=rail_translation_m,
                rail_yaw_rad=rail_yaw_rad,
                score=score,
                confidence=confidence,
                observed_camera_count=observed_camera_count,
                source=source,
                accepted=accepted,
            )
            for port_index in sorted(self.SFP_NIC_PORT_TRANSLATION_CARD)
        }
        localizations = self._sfp_nic_label_pair_by_board_x_zero_line(
            raw_localizations,
            localized_target,
        )
        return target_port_index, localizations

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
            self.get_logger().info(
                "MagentaSquare movement method: direct Cartesian transport to localized target; "
                f"target={[round(value, 5) for value in target_xyz]}, "
                f"duration={self.duration_sec:.3f}s"
            )
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
            "MagentaSquare movement method: XY-then-Z Cartesian transport to localized target; "
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

    def _sfp_nic_target_orientation(
        self,
        start_pose: Pose,
        localization: SfpNicPortLocalization,
    ) -> Quaternion:
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
        target_axis_base = self._normalized_vector(localization.port_axis_base)
        if tip_axis_base is None or target_axis_base is None:
            return start_pose.orientation

        q_delta = self._quat_from_two_vectors(tip_axis_base, target_axis_base)
        q_aligned = self._quat_multiply(q_delta, q_tcp)
        aligned_rot = self._quat_xyzw_to_rot(q_aligned)
        width_axis_tcp = self.tip_width_axis_tcp - (
            np.dot(self.tip_width_axis_tcp, self.tip_axis_tcp) * self.tip_axis_tcp
        )
        width_axis_tcp = self._normalized_vector(width_axis_tcp)
        target_width = localization.width_axis_base - (
            np.dot(localization.width_axis_base, target_axis_base) * target_axis_base
        )
        target_width = self._normalized_vector(target_width)
        if width_axis_tcp is not None and target_width is not None:
            width_after_axis_alignment = self._normalized_vector(
                aligned_rot @ width_axis_tcp
            )
            if width_after_axis_alignment is not None:
                width_after_axis_alignment = self._normalized_vector(
                    width_after_axis_alignment
                    - np.dot(width_after_axis_alignment, target_axis_base)
                    * target_axis_base
                )
            if width_after_axis_alignment is not None:
                sin_angle = float(
                    np.dot(
                        target_axis_base,
                        np.cross(width_after_axis_alignment, target_width),
                    )
                )
                cos_angle = float(
                    np.clip(np.dot(width_after_axis_alignment, target_width), -1.0, 1.0)
                )
                roll_angle = float(np.arctan2(sin_angle, cos_angle))
                q_roll = self._quat_from_axis_angle(target_axis_base, roll_angle)
                q_aligned = self._quat_multiply(q_roll, q_aligned)

        correction_angle = float(
            np.arccos(np.clip(np.dot(tip_axis_base, target_axis_base), -1.0, 1.0))
        )
        self.get_logger().info(
            "SFP/NIC aligning TCP to detected port axis: "
            f"tip_axis_base={np.round(tip_axis_base, 5).tolist()}, "
            f"port_axis_base={np.round(target_axis_base, 5).tolist()}, "
            f"port_width_axis_base={np.round(localization.width_axis_base, 5).tolist()}, "
            f"correction_angle={correction_angle:.5f} rad "
            f"({np.degrees(correction_angle):.2f} deg), "
            f"target_orientation_xyzw={np.round(q_aligned, 5).tolist()}"
        )
        return Quaternion(
            x=float(q_aligned[0]),
            y=float(q_aligned[1]),
            z=float(q_aligned[2]),
            w=float(q_aligned[3]),
        )

    def _pose_for_sfp_tip_target(
        self,
        desired_tip_base: np.ndarray,
        target_orientation: Quaternion,
    ) -> Pose:
        q_target = self._quat_normalized(
            np.array(
                [
                    target_orientation.x,
                    target_orientation.y,
                    target_orientation.z,
                    target_orientation.w,
                ],
                dtype=np.float64,
            )
        )
        target_rot = self._quat_xyzw_to_rot(q_target)
        tcp_base = np.asarray(desired_tip_base, dtype=np.float64).reshape(3) - (
            target_rot @ self.tip_position_tcp
        )
        return Pose(
            position=Point(
                x=float(tcp_base[0]),
                y=float(tcp_base[1]),
                z=float(tcp_base[2]),
            ),
            orientation=target_orientation,
        )

    def _move_to_sfp_nic_preinsert_pose(
        self,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        localization: SfpNicPortLocalization,
    ) -> Pose:
        target_orientation = (
            self._sfp_nic_target_orientation(start_pose, localization)
            if self.sfp_nic_align_to_port
            else start_pose.orientation
        )
        desired_tip_base = (
            localization.entrance_base
            - self.sfp_nic_preinsert_clearance_m * localization.port_axis_base
        )
        target_pose = self._pose_for_sfp_tip_target(
            desired_tip_base,
            target_orientation,
        )
        self.get_logger().info(
            "SFP/NIC moving to pre-insertion standoff: "
            f"desired_tip={np.round(desired_tip_base, 5).tolist()}, "
            f"tcp_target={[round(target_pose.position.x, 5), round(target_pose.position.y, 5), round(target_pose.position.z, 5)]}, "
            f"clearance={self.sfp_nic_preinsert_clearance_m:.5f}, "
            f"tip_position_tcp={np.round(self.tip_position_tcp, 5).tolist()}"
        )
        return self._move_to_pose(
            move_robot,
            start_pose,
            (
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
            ),
            duration_sec=self.tilt_tip_duration_sec,
            target_orientation=target_pose.orientation,
        )

    @staticmethod
    def _wrist_force_vector(obs) -> np.ndarray:
        try:
            force = obs.wrist_wrench.wrench.force
        except AttributeError:
            return np.zeros(3, dtype=np.float64)
        return np.array([force.x, force.y, force.z], dtype=np.float64)

    @staticmethod
    def _wrist_force_norm(obs) -> float:
        return float(np.linalg.norm(MagentaSquare._wrist_force_vector(obs)))

    def _sfp_nic_lateral_force_component(
        self,
        force_delta_base: np.ndarray,
        localization: SfpNicPortLocalization,
    ) -> np.ndarray:
        force_delta = np.asarray(force_delta_base, dtype=np.float64).reshape(3)
        axis = self._normalized_vector(localization.port_axis_base)
        if axis is None:
            return force_delta
        return force_delta - float(np.dot(force_delta, axis)) * axis

    def _record_sfp_nic_contact_force(
        self,
        force_delta_base: np.ndarray,
        localization: SfpNicPortLocalization,
        observed_distance_m: float,
    ) -> None:
        force_delta = np.asarray(force_delta_base, dtype=np.float64).reshape(3)
        lateral_force = self._sfp_nic_lateral_force_component(force_delta, localization)
        self._last_sfp_nic_force_delta_base = force_delta
        self._last_sfp_nic_lateral_force_base = lateral_force
        self._last_sfp_nic_contact_distance_m = float(observed_distance_m)

    @staticmethod
    def _append_unique_offsets(
        offsets: list[np.ndarray],
        candidate: np.ndarray | None,
        *,
        min_separation_m: float = 1e-6,
    ) -> None:
        if candidate is None:
            return
        candidate = np.asarray(candidate, dtype=np.float64).reshape(3)
        if any(np.linalg.norm(candidate - offset) < min_separation_m for offset in offsets):
            return
        offsets.append(candidate)

    def _sfp_nic_board_offset_exploration_offsets(
        self,
        localized_target: LocalizedTransportTarget,
    ) -> list[np.ndarray]:
        board_x_axis = self._normalized_vector(localized_target.board_x_axis)
        board_y_axis = self._normalized_vector(localized_target.board_y_axis)
        if board_x_axis is None or board_y_axis is None:
            return []

        max_x_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_max_x_m",
                    self.SFP_NIC_PORT_OPENING_HALF_WIDTH,
                )
            ),
            0.0,
        )
        max_y_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_max_y_m",
                    self.SFP_NIC_PORT_OPENING_HALF_HEIGHT,
                )
            ),
            0.0,
        )
        steps_x = max(
            int(getattr(self, "sfp_nic_offset_exploration_steps_x", 0)),
            0,
        )
        steps_y = max(
            int(getattr(self, "sfp_nic_offset_exploration_steps_y", 0)),
            0,
        )
        if (max_x_m <= 0.0 or steps_x <= 0) and (max_y_m <= 0.0 or steps_y <= 0):
            return []

        offsets: list[np.ndarray] = []
        x_indices = range(steps_x + 1) if max_x_m > 0.0 and steps_x > 0 else range(1)
        y_indices = range(steps_y + 1) if max_y_m > 0.0 and steps_y > 0 else range(1)
        for ix in x_indices:
            for iy in y_indices:
                if ix == 0 and iy == 0:
                    continue
                dx = max_x_m * ix / steps_x if steps_x > 0 else 0.0
                dy = max_y_m * iy / steps_y if steps_y > 0 else 0.0
                offset = dx * board_x_axis - dy * board_y_axis
                self._append_unique_offsets(offsets, offset)

        return sorted(
            offsets,
            key=lambda offset: (
                float(np.linalg.norm(offset)),
                -float(np.dot(offset, board_x_axis)),
                float(np.dot(offset, board_y_axis)),
            ),
        )

    def _sfp_nic_clamped_board_exploration_offset(
        self,
        offset_base: np.ndarray,
        localized_target: LocalizedTransportTarget,
        *,
        max_x_m: float,
        max_y_m: float,
    ) -> np.ndarray:
        board_x_axis = self._normalized_vector(localized_target.board_x_axis)
        board_y_axis = self._normalized_vector(localized_target.board_y_axis)
        if board_x_axis is None or board_y_axis is None:
            return np.asarray(offset_base, dtype=np.float64).reshape(3)

        offset = np.asarray(offset_base, dtype=np.float64).reshape(3)
        board_x_offset_m = float(np.dot(offset, board_x_axis))
        board_y_offset_m = float(np.dot(offset, board_y_axis))
        board_x_offset_m = min(max(board_x_offset_m, 0.0), max(max_x_m, 0.0))
        board_y_offset_m = min(max(board_y_offset_m, -max(max_y_m, 0.0)), 0.0)
        return board_x_offset_m * board_x_axis + board_y_offset_m * board_y_axis

    def _sfp_nic_slide_in_step_from_force_delta(
        self,
        force_delta_base: np.ndarray,
        localization: SfpNicPortLocalization,
    ) -> np.ndarray | None:
        if not getattr(self, "sfp_nic_offset_exploration_slide_in_enabled", True):
            return None

        lateral_force = self._sfp_nic_lateral_force_component(
            force_delta_base,
            localization,
        )
        lateral_force_norm = float(np.linalg.norm(lateral_force))
        min_force_n = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_force_min_n",
                    0.25,
                )
            ),
            0.0,
        )
        if lateral_force_norm < min_force_n:
            return None

        gain_m_per_n = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_gain_m_per_n",
                    0.0001,
                )
            ),
            0.0,
        )
        step_max_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_step_max_m",
                    0.0005,
                )
            ),
            0.0,
        )
        if gain_m_per_n <= 0.0 or step_max_m <= 0.0:
            return None

        slide_step = -gain_m_per_n * lateral_force
        slide_step_norm = float(np.linalg.norm(slide_step))
        if slide_step_norm <= 1e-12:
            return None
        if slide_step_norm > step_max_m:
            slide_step *= step_max_m / slide_step_norm
        return slide_step

    def _run_sfp_nic_board_offset_exploration(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        current_pose: Pose,
        localization: SfpNicPortLocalization,
        q_target: Quaternion,
        *,
        localized_target: LocalizedTransportTarget | None,
    ) -> tuple[Pose, bool, bool]:
        if not getattr(self, "sfp_nic_offset_exploration_enabled", True):
            return current_pose, False, True
        if localized_target is None:
            self.get_logger().warn(
                "SFP/NIC offset exploration skipped because no board pose was provided."
            )
            return current_pose, False, True

        step_m = abs(self.descend_step_m)
        if step_m <= 0.0 or self.descend_dt_sec <= 0.0:
            return current_pose, False, True

        exploration_offsets = self._sfp_nic_board_offset_exploration_offsets(
            localized_target,
        )
        max_probes = max(
            int(getattr(self, "sfp_nic_offset_exploration_max_probes", 0)),
            0,
        )
        if max_probes > 0:
            exploration_offsets = exploration_offsets[:max_probes]
        if not exploration_offsets:
            return current_pose, False, True

        exploration_backoff_m = max(
            float(getattr(self, "sfp_nic_offset_exploration_backoff_m", 0.0)),
            0.0,
        )
        exploration_descent_m = max(
            float(getattr(self, "sfp_nic_offset_exploration_descent_m", 0.0)),
            0.0,
        )
        insertion_check_depth_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_insertion_check_depth_m",
                    0.004,
                )
            ),
            0.0,
        )
        exploration_max_x_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_max_x_m",
                    self.SFP_NIC_PORT_OPENING_HALF_WIDTH,
                )
            ),
            0.0,
        )
        exploration_max_y_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_max_y_m",
                    self.SFP_NIC_PORT_OPENING_HALF_HEIGHT,
                )
            ),
            0.0,
        )
        slide_total_max_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_total_max_m",
                    0.002,
                )
            ),
            0.0,
        )
        slide_direction_tracking_enabled = getattr(
            self,
            "sfp_nic_offset_exploration_slide_direction_tracking_enabled",
            True,
        )
        slide_direction_memory_alpha = min(
            max(
                float(
                    getattr(
                        self,
                        "sfp_nic_offset_exploration_slide_direction_memory_alpha",
                        0.80,
                    )
                ),
                0.0,
            ),
            1.0,
        )
        slide_direction_step_max_m = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_direction_step_m",
                    0.00050,
                )
            ),
            0.0,
        )
        slide_gain_m_per_n = max(
            float(
                getattr(
                    self,
                    "sfp_nic_offset_exploration_slide_gain_m_per_n",
                    0.0001,
                )
            ),
            0.0,
        )
        if exploration_backoff_m <= 0.0 or exploration_descent_m <= 0.0:
            return current_pose, False, True
        probe_distance_limit_m = max(
            exploration_descent_m,
            exploration_backoff_m + insertion_check_depth_m,
        )

        self.get_logger().info(
            "SFP/NIC running board-frame offset exploration after contact: "
            f"attempts={len(exploration_offsets)}, shift_candidates="
            f"{[np.round(offset, 5).tolist() for offset in exploration_offsets]}, "
            f"backoff={exploration_backoff_m:.5f} m, "
            f"probe_descent={probe_distance_limit_m:.5f} m, "
            f"insertion_check_depth={insertion_check_depth_m:.5f} m, "
            f"slide_in={getattr(self, 'sfp_nic_offset_exploration_slide_in_enabled', True)}, "
            f"slide_total_max={slide_total_max_m:.5f} m, "
            f"slide_direction_tracking={slide_direction_tracking_enabled}"
        )

        axial_delta_stop_n = max(self.sfp_nic_axial_force_delta_stop_n, 0.0)
        insertion_check_tcp_z = (
            float(localization.entrance_base[2]) - insertion_check_depth_m
        )
        tracked_slide_direction_base: np.ndarray | None = None
        tracked_slide_force_n = 0.0
        for attempt, exploration_offset in enumerate(exploration_offsets, start=1):
            if self._task_completed_in_simulation(task):
                return current_pose, True, False

            start_tip_base = (
                localization.entrance_base
                - exploration_backoff_m * localization.port_axis_base
                + exploration_offset
            )
            start_probe_pose = self._pose_for_sfp_tip_target(start_tip_base, q_target)
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                (
                    start_probe_pose.position.x,
                    start_probe_pose.position.y,
                    start_probe_pose.position.z,
                ),
                duration_sec=self.sfp_nic_view_search_duration_sec,
                target_orientation=q_target,
            )

            max_probe_steps = int(np.ceil(probe_distance_limit_m / step_m))
            force_baseline_base: np.ndarray | None = None
            probe_force_stop = False
            reached_insertion_check_depth = False
            slide_offset_base = np.zeros(3, dtype=np.float64)
            for step in range(1, max_probe_steps + 1):
                if self._task_completed_in_simulation(task):
                    return current_pose, True, False

                obs = get_observation()
                force_base = self._wrist_force_vector(obs)
                if force_baseline_base is None:
                    force_baseline_base = force_base.copy()
                force_norm = float(np.linalg.norm(force_base))
                force_delta = force_base - force_baseline_base
                axial_force_delta = float(
                    np.dot(force_delta, localization.port_axis_base)
                )
                axial_force_delta_abs = abs(axial_force_delta)
                lateral_force = self._sfp_nic_lateral_force_component(
                    force_delta,
                    localization,
                )
                lateral_force_norm = float(np.linalg.norm(lateral_force))
                probe_distance = min(step * step_m, probe_distance_limit_m)
                slide_step_applied = False
                force_slide_step = self._sfp_nic_slide_in_step_from_force_delta(
                    force_delta,
                    localization,
                )
                if force_slide_step is not None and slide_direction_tracking_enabled:
                    force_slide_direction = self._normalized_vector(force_slide_step)
                    if force_slide_direction is not None:
                        if tracked_slide_direction_base is None:
                            tracked_slide_direction_base = force_slide_direction
                            tracked_slide_force_n = lateral_force_norm
                        else:
                            blended_direction = (
                                slide_direction_memory_alpha
                                * tracked_slide_direction_base
                                + (1.0 - slide_direction_memory_alpha)
                                * force_slide_direction
                            )
                            blended_direction = self._normalized_vector(
                                blended_direction
                            )
                            tracked_slide_direction_base = (
                                force_slide_direction
                                if blended_direction is None
                                else blended_direction
                            )
                            tracked_slide_force_n = (
                                slide_direction_memory_alpha * tracked_slide_force_n
                                + (1.0 - slide_direction_memory_alpha)
                                * lateral_force_norm
                            )
                elif slide_direction_tracking_enabled and tracked_slide_force_n > 0.0:
                    tracked_slide_force_n *= slide_direction_memory_alpha

                slide_step = force_slide_step
                if (
                    slide_step is None
                    and slide_direction_tracking_enabled
                    and tracked_slide_direction_base is not None
                    and slide_direction_step_max_m > 0.0
                    and slide_gain_m_per_n > 0.0
                    and tracked_slide_force_n > 0.0
                ):
                    remembered_step_m = min(
                        slide_gain_m_per_n * tracked_slide_force_n,
                        slide_direction_step_max_m,
                    )
                    slide_step = remembered_step_m * tracked_slide_direction_base

                if slide_step is not None:
                    proposed_slide_offset = slide_offset_base + slide_step
                    proposed_slide_norm = float(np.linalg.norm(proposed_slide_offset))
                    if (
                        slide_total_max_m > 0.0
                        and proposed_slide_norm > slide_total_max_m
                    ):
                        proposed_slide_offset *= (
                            slide_total_max_m / proposed_slide_norm
                        )
                    proposed_total_offset = exploration_offset + proposed_slide_offset
                    clamped_total_offset = self._sfp_nic_clamped_board_exploration_offset(
                        proposed_total_offset,
                        localized_target,
                        max_x_m=exploration_max_x_m,
                        max_y_m=exploration_max_y_m,
                    )
                    applied_slide_offset = clamped_total_offset - exploration_offset
                    applied_delta_norm = float(
                        np.linalg.norm(applied_slide_offset - slide_offset_base)
                    )
                    if applied_delta_norm > 1e-9:
                        slide_offset_base = applied_slide_offset
                        slide_step_applied = True

                axial_delta_stop = (
                    axial_delta_stop_n > 0.0
                    and axial_force_delta_abs > axial_delta_stop_n
                )
                if force_norm > self.sfp_nic_force_stop_n or (
                    axial_delta_stop and not slide_step_applied
                ):
                    self._record_sfp_nic_contact_force(
                        force_delta,
                        localization,
                        probe_distance,
                    )
                    self.get_logger().warn(
                        "SFP/NIC offset exploration probe hit force threshold: "
                        f"attempt={attempt}/{len(exploration_offsets)}, "
                        f"force={force_norm:.2f} N, axial_delta={axial_force_delta:.2f} N, "
                        f"offset={np.round(exploration_offset, 5).tolist()}, "
                        f"slide_offset={np.round(slide_offset_base, 5).tolist()}"
                    )
                    probe_force_stop = True
                    break

                desired_tip_base = (
                    start_tip_base
                    + probe_distance * localization.port_axis_base
                    + slide_offset_base
                )
                current_pose = self._pose_for_sfp_tip_target(
                    desired_tip_base,
                    q_target,
                )
                if slide_step_applied and (step == 1 or step % 10 == 0):
                    self.get_logger().info(
                        "SFP/NIC offset exploration slide-in correction: "
                        f"attempt={attempt}/{len(exploration_offsets)}, "
                        f"step={step}/{max_probe_steps}, "
                        f"force_delta={np.round(force_delta, 3).tolist()}, "
                        f"tracked_force={tracked_slide_force_n:.3f} N, "
                        f"slide_offset={np.round(slide_offset_base, 5).tolist()}, "
                        f"tracked_direction="
                        f"{None if tracked_slide_direction_base is None else np.round(tracked_slide_direction_base, 4).tolist()}"
                    )
                self.set_pose_target(move_robot=move_robot, pose=current_pose)
                self.sleep_for(self.descend_dt_sec)
                if self._task_completed_in_simulation(task):
                    return current_pose, True, False
                if float(current_pose.position.z) < insertion_check_tcp_z:
                    self.get_logger().info(
                        "SFP/NIC offset exploration reached insertion-check depth "
                        "without insertion; trying next grid point: "
                        f"attempt={attempt}/{len(exploration_offsets)}, "
                        f"tcp_z={current_pose.position.z:.5f}, "
                        f"check_z={insertion_check_tcp_z:.5f}, "
                        f"offset={np.round(exploration_offset, 5).tolist()}, "
                        f"slide_offset={np.round(slide_offset_base, 5).tolist()}"
                    )
                    reached_insertion_check_depth = True
                    break

            if probe_force_stop:
                continue
            if reached_insertion_check_depth:
                continue

            self.get_logger().info(
                "SFP/NIC offset exploration probe finished without insertion: "
                f"attempt={attempt}/{len(exploration_offsets)}, "
                f"offset={np.round(exploration_offset, 5).tolist()}"
            )

        return current_pose, self._task_completed_in_simulation(task), True

    def _descend_sfp_nic_along_port_axis(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
        localization: SfpNicPortLocalization,
        *,
        lateral_offset_base: np.ndarray | None = None,
        descent_distance_m: float | None = None,
        localized_target: LocalizedTransportTarget | None = None,
        allow_offset_exploration: bool = True,
    ) -> tuple[Pose, bool, bool]:
        step_m = abs(self.descend_step_m)
        max_distance_m = (
            self.sfp_nic_descent_distance_m
            if descent_distance_m is None
            else float(descent_distance_m)
        )
        if step_m <= 0.0 or self.descend_dt_sec <= 0.0 or max_distance_m <= 0.0:
            self.get_logger().info(
                "SFP/NIC axis descent skipped because step, dt, or distance is non-positive."
            )
            return start_pose, False, False

        q_target = Quaternion(
            x=start_pose.orientation.x,
            y=start_pose.orientation.y,
            z=start_pose.orientation.z,
            w=start_pose.orientation.w,
        )
        lateral_offset = (
            np.zeros(3, dtype=np.float64)
            if lateral_offset_base is None
            else np.asarray(lateral_offset_base, dtype=np.float64).reshape(3)
        )
        start_tip_base = (
            localization.entrance_base
            - self.sfp_nic_preinsert_clearance_m * localization.port_axis_base
            + lateral_offset
        )
        max_steps = int(np.ceil(max_distance_m / step_m))
        current_pose = start_pose
        force_stop = False
        contact_force_delta_base: np.ndarray | None = None
        force_baseline_base: np.ndarray | None = None
        self._last_sfp_nic_force_delta_base = None
        self._last_sfp_nic_lateral_force_base = None
        self._last_sfp_nic_contact_distance_m = None
        axial_delta_stop_n = max(self.sfp_nic_axial_force_delta_stop_n, 0.0)
        axial_delta_min_distance_m = max(
            self.sfp_nic_axial_force_delta_min_distance_m,
            0.0,
        )
        self.get_logger().info(
            "SFP/NIC descending along detected port axis: "
            f"step={step_m:.5f} m, dt={self.descend_dt_sec:.3f} s, "
            f"max_distance={max_distance_m:.5f} m, "
            f"axis={np.round(localization.port_axis_base, 5).tolist()}, "
            f"lateral_offset={np.round(lateral_offset, 5).tolist()}, "
            f"axial_force_delta_stop={axial_delta_stop_n:.2f} N"
        )
        for step in range(1, max_steps + 1):
            if self._task_completed_in_simulation(task):
                return current_pose, True, force_stop

            obs = get_observation()
            force_base = self._wrist_force_vector(obs)
            if force_baseline_base is None:
                force_baseline_base = force_base.copy()
            force_norm = float(np.linalg.norm(force_base))
            observed_distance = min((step - 1) * step_m, max_distance_m)
            force_delta_base = force_base - force_baseline_base
            axial_force_delta = float(
                np.dot(force_delta_base, localization.port_axis_base)
            )
            axial_force_delta_abs = abs(axial_force_delta)
            if force_norm > self.sfp_nic_force_stop_n:
                self.get_logger().warn(
                    "SFP/NIC axis descent stopped on force threshold: "
                    f"force={force_norm:.2f} N, threshold={self.sfp_nic_force_stop_n:.2f} N"
                )
                contact_force_delta_base = force_delta_base
                self._record_sfp_nic_contact_force(
                    force_delta_base,
                    localization,
                    observed_distance,
                )
                force_stop = True
                break
            if (
                axial_delta_stop_n > 0.0
                and observed_distance >= axial_delta_min_distance_m
                and axial_force_delta_abs > axial_delta_stop_n
            ):
                self.get_logger().warn(
                    "SFP/NIC axis descent stopped on axial force delta spike: "
                    f"axial_delta={axial_force_delta:.2f} N, "
                    f"abs_axial_delta={axial_force_delta_abs:.2f} N, "
                    f"threshold={axial_delta_stop_n:.2f} N, "
                    f"observed_distance={observed_distance:.5f} m"
                )
                contact_force_delta_base = force_delta_base
                self._record_sfp_nic_contact_force(
                    force_delta_base,
                    localization,
                    observed_distance,
                )
                force_stop = True
                break

            distance = min(step * step_m, max_distance_m)
            desired_tip_base = start_tip_base + distance * localization.port_axis_base
            current_pose = self._pose_for_sfp_tip_target(desired_tip_base, q_target)
            self.set_pose_target(move_robot=move_robot, pose=current_pose)
            if step == 1 or step % 20 == 0 or step == max_steps:
                self.get_logger().info(
                    "SFP/NIC axis descent: "
                    f"step={step}/{max_steps}, distance={distance:.5f} m, "
                    f"desired_tip={np.round(desired_tip_base, 5).tolist()}, "
                    f"tcp_target={[round(current_pose.position.x, 5), round(current_pose.position.y, 5), round(current_pose.position.z, 5)]}, "
                    f"force={force_norm:.2f} N, "
                    f"axial_delta={axial_force_delta:.2f} N"
                )
            self.sleep_for(self.descend_dt_sec)

        if force_stop:
            if allow_offset_exploration and contact_force_delta_base is not None:
                exploration_pose, exploration_inserted, exploration_force_stop = (
                    self._run_sfp_nic_board_offset_exploration(
                        task,
                        get_observation,
                        move_robot,
                        current_pose,
                        localization,
                        q_target,
                        localized_target=localized_target,
                    )
                )
                current_pose = exploration_pose
                if exploration_inserted:
                    return exploration_pose, True, exploration_force_stop
            return current_pose, self._task_completed_in_simulation(task), force_stop

        if self.descend_wait_for_insertion_sec > 0.0:
            wait_started = self.time_now()
            wait_timeout = Duration(seconds=self.descend_wait_for_insertion_sec)
            while (self.time_now() - wait_started) < wait_timeout:
                if self._task_completed_in_simulation(task):
                    return current_pose, True, force_stop
                self.sleep_for(min(self.descend_dt_sec, 0.05))
        return current_pose, self._task_completed_in_simulation(task), force_stop

    def _sfp_nic_recovery_offsets(
        self,
        localization: SfpNicPortLocalization,
    ) -> list[np.ndarray]:
        radius = max(self.sfp_nic_recovery_radius_m, 0.0)
        recovery_attempts = min(
            max(self.sfp_nic_recovery_attempts, 0),
            self.SFP_NIC_MAX_RECOVERY_RETRIES,
        )
        if radius <= 0.0 or recovery_attempts <= 0:
            return []
        candidates: list[np.ndarray] = []
        for candidate in [
            radius * localization.width_axis_base,
            -radius * localization.width_axis_base,
            radius * localization.height_axis_base,
            -radius * localization.height_axis_base,
            radius * (localization.width_axis_base + localization.height_axis_base) / np.sqrt(2.0),
            radius * (-localization.width_axis_base + localization.height_axis_base) / np.sqrt(2.0),
        ]:
            self._append_unique_offsets(candidates, candidate)
        return candidates[:recovery_attempts]

    def _run_sfp_nic_insertion(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        localized_target: LocalizedTransportTarget,
        start_pose: Pose,
    ) -> Pose:
        localization, search_pose = self._localize_sfp_nic_port_entrance_with_view_search(
            task,
            get_observation,
            move_robot,
            localized_target,
            start_pose,
        )
        if localization is None:
            self.get_logger().warn(
                "SFP/NIC insertion could not localize port entrance; falling back to "
                "legacy tip-down/base-z descent."
            )
            aligned_pose = self._move_to_tip_down_pose(move_robot, search_pose)
            if self.descend_after_transport:
                return self._descend_until_inserted(task, move_robot, aligned_pose)
            return aligned_pose

        current_pose = self._move_to_sfp_nic_preinsert_pose(
            move_robot,
            self._tcp_pose_from_observation(get_observation),
            localization,
        )
        if not self.descend_after_transport:
            return current_pose

        current_pose, inserted, force_stop = self._descend_sfp_nic_along_port_axis(
            task,
            get_observation,
            move_robot,
            current_pose,
            localization,
            localized_target=localized_target,
        )
        if inserted or not self.sfp_nic_recovery_enabled:
            return current_pose

        recovery_attempts = min(
            len(self._sfp_nic_recovery_offsets(localization)),
            self.SFP_NIC_MAX_RECOVERY_RETRIES,
        )
        if recovery_attempts <= 0:
            return current_pose

        self.get_logger().info(
            "SFP/NIC insertion did not complete; running bounded backoff/redetect "
            f"recovery: attempts={recovery_attempts}, force_stop={force_stop}"
        )
        for attempt in range(1, recovery_attempts + 1):
            if self._task_completed_in_simulation(task):
                return current_pose
            recovery_offsets = self._sfp_nic_recovery_offsets(localization)
            if not recovery_offsets:
                return current_pose
            lateral_offset = recovery_offsets[min(attempt - 1, len(recovery_offsets) - 1)]
            desired_tip_base = (
                localization.entrance_base
                - self.sfp_nic_preinsert_clearance_m * localization.port_axis_base
                + lateral_offset
            )
            preinsert_pose = self._pose_for_sfp_tip_target(
                desired_tip_base,
                current_pose.orientation,
            )
            current_pose = self._move_to_pose(
                move_robot,
                self._tcp_pose_from_observation(get_observation),
                (
                    preinsert_pose.position.x,
                    preinsert_pose.position.y,
                    preinsert_pose.position.z,
                ),
                duration_sec=self.tilt_tip_duration_sec,
                target_orientation=preinsert_pose.orientation,
            )
            self.get_logger().info(
                "SFP/NIC recovery backed off for redetection: "
                f"attempt={attempt}/{recovery_attempts}, "
                f"backoff_tip={np.round(desired_tip_base, 5).tolist()}, "
                f"lateral_offset={np.round(lateral_offset, 5).tolist()}"
            )

            refreshed_localization, current_pose = (
                self._localize_sfp_nic_port_entrance_with_view_search(
                    task,
                    get_observation,
                    move_robot,
                    localized_target,
                    current_pose,
                )
            )
            if refreshed_localization is None:
                self.get_logger().warn(
                    "SFP/NIC recovery redetection failed; skipping this descent attempt."
                )
                continue

            previous_entrance = localization.entrance_base
            localization = refreshed_localization
            self.get_logger().info(
                "SFP/NIC recovery redetected port entrance: "
                f"attempt={attempt}/{recovery_attempts}, "
                f"source={localization.source}, accepted={localization.accepted}, "
                f"confidence={localization.confidence:.3f}, "
                f"previous_entrance={np.round(previous_entrance, 5).tolist()}, "
                f"new_entrance={np.round(localization.entrance_base, 5).tolist()}"
            )
            current_pose = self._move_to_sfp_nic_preinsert_pose(
                move_robot,
                self._tcp_pose_from_observation(get_observation),
                localization,
            )
            current_pose, inserted, force_stop = self._descend_sfp_nic_along_port_axis(
                task,
                get_observation,
                move_robot,
                current_pose,
                localization,
                descent_distance_m=min(self.sfp_nic_descent_distance_m, 0.030),
                localized_target=localized_target,
            )
            if inserted:
                return current_pose
            if force_stop:
                if attempt < recovery_attempts:
                    self.get_logger().warn(
                        "SFP/NIC recovery descent stopped on force threshold; backing off "
                        "and redetecting again if attempts remain."
                    )
                else:
                    self.get_logger().warn(
                        "SFP/NIC recovery descent stopped on force threshold; retry "
                        "limit reached, stopping at final pose."
                    )
        return current_pose

    def _descend_until_inserted(
        self,
        task: Task,
        move_robot: MoveRobotCallback,
        start_pose: Pose,
    ) -> Pose:
        step_m = abs(self.descend_step_m)
        try:
            task_family = self._task_family(task)
        except ValueError:
            task_family = "unknown"
        speed_multiplier = (
            self.sc_descend_speed_multiplier if task_family == "sc_to_sc" else 1.0
        )
        step_m *= max(speed_multiplier, 0.0)
        max_distance_m = (
            self.sc_descend_max_distance_m
            if task_family == "sc_to_sc"
            else self.descend_max_distance_m
        )
        stop_z = None
        if task_family == "sc_to_sc":
            stop_z = (
                self.board_plane_z
                + self.sc_descend_stop_z
                + self.sc_descend_stop_z_margin
            )
            distance_to_stop_z = max(float(start_pose.position.z) - stop_z, 0.0)
            if distance_to_stop_z <= 0.0:
                self.get_logger().info(
                    "TransportToMean SC descent skipped because TCP is already at or below "
                    f"the SC port stop z: current_z={start_pose.position.z:.5f}, "
                    f"stop_z={stop_z:.5f}"
                )
                return start_pose
            max_distance_m = min(max_distance_m, distance_to_stop_z)
        if step_m <= 0.0 or self.descend_dt_sec <= 0.0 or max_distance_m <= 0.0:
            self.get_logger().info(
                "TransportToMean descent skipped because descent step, dt, or max "
                "distance is non-positive."
            )
            return start_pose

        max_steps = int(np.ceil(max_distance_m / step_m))
        current_pose = start_pose
        self.get_logger().info(
            "TransportToMean descending TCP in base -z: "
            f"task_family={task_family}, "
            f"speed_multiplier={speed_multiplier:.3f}, "
            f"step={step_m:.5f} m, dt={self.descend_dt_sec:.3f} s, "
            f"max_distance={max_distance_m:.5f} m, "
            f"stop_z={stop_z if stop_z is not None else 'none'}, "
            f"sc_stop_margin={self.sc_descend_stop_z_margin:.5f} m, "
            f"rate={step_m / self.descend_dt_sec:.5f} m/s"
        )
        for step in range(1, max_steps + 1):
            if self._task_completed_in_simulation(task):
                self.get_logger().info(
                    "TransportToMean descent early exit: simulation reported "
                    "task completion."
                )
                return current_pose

            distance = min(step * step_m, max_distance_m)
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
        *,
        max_scan_distance_m: float | None = None,
        return_to_start_on_failure: bool = False,
    ) -> bool:
        start_pose = self._tcp_pose_from_observation(get_observation)
        scan_z = max(start_pose.position.z + self.short_edge_scan_z_offset, self.view_z)
        current_pose = start_pose
        y_step = float(self.short_edge_scan_y_step)
        max_steps = max(int(self.short_edge_scan_max_steps), 0)
        max_travel_m = abs(y_step) * float(max_steps)
        if max_scan_distance_m is not None:
            max_travel_m = min(max_travel_m, max(float(max_scan_distance_m), 0.0))
        scan_moves = (
            0
            if abs(y_step) <= 1e-9 or max_travel_m <= 0.0
            else int(np.ceil(max_travel_m / abs(y_step)))
        )
        self.get_logger().info(
            "TransportToMean scanning for full short-edge view "
            f"start_y={start_pose.position.y:.4f}, y_step={y_step:.4f}, "
            f"max_steps={scan_moves}, "
            f"total_y={np.sign(y_step) * max_travel_m:.4f}, "
            f"max_scan_distance={max_scan_distance_m}, "
            f"return_to_start_on_failure={return_to_start_on_failure}, "
            f"z={scan_z:.4f}, "
            f"z_offset={self.short_edge_scan_z_offset:.4f}, "
            f"min_width={self.short_edge_min_width:.4f}, "
            f"min_endpoint_margin_px={self.short_edge_min_endpoint_margin_px:.1f}"
        )

        for step in range(scan_moves + 1):
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
                return True

            if step == scan_moves:
                break

            travel_m = min(abs(y_step) * float(step + 1), max_travel_m)
            target_xyz = (
                current_pose.position.x,
                start_pose.position.y + np.sign(y_step) * travel_m,
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
            + (
                "returning to the scan start pose."
                if return_to_start_on_failure
                else "continuing with the best available view."
            )
        )
        if return_to_start_on_failure:
            self.get_logger().info(
                "TransportToMean returning to original short-edge scan start pose "
                "before switching detection strategies."
            )
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                (
                    start_pose.position.x,
                    start_pose.position.y,
                    start_pose.position.z,
                ),
                duration_sec=self.short_edge_scan_duration_sec,
                target_orientation=start_pose.orientation,
            )
            hold_steps = max(0, int(self.short_edge_scan_hold_sec / self.dt))
            for _ in range(hold_steps):
                self.set_pose_target(move_robot=move_robot, pose=current_pose)
                self.sleep_for(self.dt)
        return False

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
            self.get_logger().info(
                "MagentaSquare detection method: single-view magenta attempt; "
                "linear +y magenta search disabled, falling back to short-edge if needed."
            )
            obs = get_observation()
            target_xyz = self._multicamera_magenta_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare detection selected: magenta ROI + board +y edge fit "
                    "from current view."
                )
                return target_xyz, "magenta ROI + board +y edge fit"
            target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare detection selected: short-edge fallback from current view."
                )
                return target_xyz, "multi-camera short-edge fit"
            return None, ""

        fallback_target = None
        fallback_source = ""
        current_pose = center_pose
        obs = get_observation()
        target_xyz = self._multicamera_magenta_target_xyz(task, obs)
        if target_xyz is not None:
            self.get_logger().info(
                "MagentaSquare detection selected: magenta ROI + board +y edge fit "
                "at high view pose; "
                "skipping +y linear search."
            )
            return target_xyz, "magenta ROI + board +y edge fit"

        candidate = self._multicamera_short_edge_target_xyz(task, obs)
        if candidate is not None:
            fallback_target = candidate
            fallback_source = "multi-camera short-edge fit"
            self.get_logger().info(
                "MagentaSquare detection fallback candidate stored: short-edge fit "
                "from high view pose; continuing magenta +y search before using it."
            )

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
                    self.get_logger().info(
                        "MagentaSquare detection fallback candidate stored: "
                        "short-edge fit during linear +y search; continuing magenta search."
                    )

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
                "MagentaSquare detection selected: short-edge fallback after linear +y "
                "magenta search did not produce a valid geometry-fit marker target."
            )
        return fallback_target, fallback_source

    def _detect_target_from_circular_view_search(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        center_pose: Pose,
    ) -> tuple[tuple[float, float, float] | None, str]:
        if (
            not self.magenta_sc_circular_view_enabled
            or self.magenta_sc_circular_view_steps <= 0
            or self.magenta_sc_circular_view_radius_m <= 0.0
        ):
            self.get_logger().info(
                "MagentaSquare detection method: high-view single-view magenta attempt; "
                "circular XY search disabled, falling back to short-edge if needed."
            )
            obs = get_observation()
            target_xyz = self._multicamera_magenta_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare detection selected: magenta ROI + board +y edge fit "
                    "from high view."
                )
                return target_xyz, "magenta ROI + board +y edge fit"
            target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare detection selected: short-edge fallback from high view."
                )
                return target_xyz, "multi-camera short-edge fit"
            return None, ""

        fallback_target = None
        fallback_source = ""
        current_pose = center_pose
        obs = get_observation()
        target_xyz = self._multicamera_magenta_target_xyz(task, obs)
        if target_xyz is not None:
            self.get_logger().info(
                "MagentaSquare detection selected: magenta ROI + board +y edge fit "
                "at high view center; skipping circular XY search."
            )
            return target_xyz, "magenta ROI + board +y edge fit"

        candidate = self._multicamera_short_edge_target_xyz(task, obs)
        if candidate is not None:
            fallback_target = candidate
            fallback_source = "multi-camera short-edge fit"
            self.get_logger().info(
                "MagentaSquare detection fallback candidate stored: short-edge fit "
                "from high view center; continuing circular magenta search."
            )

        step_count = max(1, self.magenta_sc_circular_view_steps)
        radius_m = self.magenta_sc_circular_view_radius_m
        self.get_logger().info(
            "MagentaSquare sampling circular XY view search at fixed z/orientation: "
            f"center_xyz={[round(center_pose.position.x, 5), round(center_pose.position.y, 5), round(center_pose.position.z, 5)]}, "
            f"radius={radius_m:.4f}, "
            f"samples={step_count}, "
            f"move_duration={self.magenta_sc_circular_view_move_duration_sec:.3f}, "
            f"hold_sec={self.magenta_sc_circular_view_hold_sec:.3f}"
        )
        for step_index in range(step_count):
            theta = 2.0 * np.pi * float(step_index) / float(step_count)
            target_xyz = (
                center_pose.position.x + radius_m * float(np.cos(theta)),
                center_pose.position.y + radius_m * float(np.sin(theta)),
                center_pose.position.z,
            )
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                target_xyz,
                duration_sec=self.magenta_sc_circular_view_move_duration_sec,
            )
            hold_until = self.time_now() + Duration(
                seconds=max(self.magenta_sc_circular_view_hold_sec, 0.0)
            )
            while self.time_now() < hold_until:
                self.set_pose_target(move_robot=move_robot, pose=current_pose)
                self.sleep_for(min(self.dt, self.magenta_sc_circular_view_hold_sec))

            obs = get_observation()
            target_xyz = self._multicamera_magenta_target_xyz(task, obs)
            if target_xyz is not None:
                self.get_logger().info(
                    "MagentaSquare circular view sample accepted magenta-guided edge target: "
                    f"sample={step_index + 1}/{step_count}, "
                    f"theta={theta:.3f} rad, "
                    f"xy_offset={[round(radius_m * float(np.cos(theta)), 5), round(radius_m * float(np.sin(theta)), 5)]}"
                )
                return target_xyz, "magenta ROI + board +y edge fit"

            if fallback_target is None:
                candidate = self._multicamera_short_edge_target_xyz(task, obs)
                if candidate is not None:
                    fallback_target = candidate
                    fallback_source = "multi-camera short-edge fit"
                    self.get_logger().info(
                        "MagentaSquare detection fallback candidate stored: "
                        "short-edge fit during circular XY search; continuing magenta search."
                    )

        current_pose = self._move_to_pose(
            move_robot,
            current_pose,
            (
                center_pose.position.x,
                center_pose.position.y,
                center_pose.position.z,
            ),
            duration_sec=self.magenta_sc_circular_view_move_duration_sec,
        )
        if fallback_target is not None:
            self.get_logger().info(
                "MagentaSquare detection selected: short-edge fallback after circular XY "
                "magenta search did not produce a valid geometry-fit marker target."
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
    def _quat_xyzw_from_rot(rot: np.ndarray) -> np.ndarray:
        rot = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = float(np.sqrt(trace + 1.0) * 2.0)
            quat = np.array(
                [
                    (rot[2, 1] - rot[1, 2]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                    (rot[1, 0] - rot[0, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
        else:
            diag_index = int(np.argmax(np.diag(rot)))
            if diag_index == 0:
                s = float(np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0)
                quat = np.array(
                    [
                        0.25 * s,
                        (rot[0, 1] + rot[1, 0]) / s,
                        (rot[0, 2] + rot[2, 0]) / s,
                        (rot[2, 1] - rot[1, 2]) / s,
                    ],
                    dtype=np.float64,
                )
            elif diag_index == 1:
                s = float(np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0)
                quat = np.array(
                    [
                        (rot[0, 1] + rot[1, 0]) / s,
                        0.25 * s,
                        (rot[1, 2] + rot[2, 1]) / s,
                        (rot[0, 2] - rot[2, 0]) / s,
                    ],
                    dtype=np.float64,
                )
            else:
                s = float(np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0)
                quat = np.array(
                    [
                        (rot[0, 2] + rot[2, 0]) / s,
                        (rot[1, 2] + rot[2, 1]) / s,
                        0.25 * s,
                        (rot[1, 0] - rot[0, 1]) / s,
                    ],
                    dtype=np.float64,
                )
        return TransportToMean._quat_normalized(quat)

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

    @staticmethod
    def _rot_x(angle: float) -> np.ndarray:
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_a, -sin_a],
                [0.0, sin_a, cos_a],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rot_z(angle: float) -> np.ndarray:
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        return np.array(
            [
                [cos_a, -sin_a, 0.0],
                [sin_a, cos_a, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _board_xyz_to_base_xyz(
        point_board: np.ndarray,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
    ) -> tuple[float, float, float]:
        point_board = np.asarray(point_board, dtype=np.float64).reshape(3)
        board_origin = np.asarray(board_origin, dtype=np.float64).reshape(3)
        board_x_axis = np.asarray(board_x_axis, dtype=np.float64).reshape(3)
        board_y_axis = np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        point_base = (
            board_origin
            + point_board[0] * board_x_axis
            + point_board[1] * board_y_axis
        )
        point_base[2] = board_origin[2] + point_board[2]
        return tuple(float(v) for v in point_base)

    @staticmethod
    def _board_vector_to_base_vector(
        vector_board: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
    ) -> np.ndarray:
        vector_board = np.asarray(vector_board, dtype=np.float64).reshape(3)
        board_x_axis = np.asarray(board_x_axis, dtype=np.float64).reshape(3)
        board_y_axis = np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        return (
            vector_board[0] * board_x_axis
            + vector_board[1] * board_y_axis
            + np.array([0.0, 0.0, vector_board[2]], dtype=np.float64)
        )

    @staticmethod
    def _base_point_to_board_xy(
        point_base: np.ndarray,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
    ) -> np.ndarray:
        delta = np.asarray(point_base, dtype=np.float64).reshape(3) - np.asarray(
            board_origin,
            dtype=np.float64,
        ).reshape(3)
        board_x_axis = np.asarray(board_x_axis, dtype=np.float64).reshape(3)
        board_y_axis = np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        return np.array(
            [
                float(np.dot(delta, board_x_axis)),
                float(np.dot(delta, board_y_axis)),
            ],
            dtype=np.float64,
        )

    def _sc_nic_zone_rect(self, *, inflated: bool) -> tuple[float, float, float, float]:
        margin = float(getattr(self, "sc_nic_avoid_margin_m", 0.0)) if inflated else 0.0
        half_x = 0.075
        half_y = 0.022
        min_x = self.NIC_RAIL_X - half_x - margin
        max_x = self.NIC_RAIL_X + half_x + margin
        min_y = self.NIC_RAIL_Y0 - half_y - margin
        max_y = (
            self.NIC_RAIL_Y0
            + self.NIC_RAIL_Y_SPACING * 4
            + half_y
            + margin
        )
        return min_x, max_x, min_y, max_y

    @staticmethod
    def _point_inside_rect(
        point_xy: np.ndarray,
        rect: tuple[float, float, float, float],
    ) -> bool:
        x, y = np.asarray(point_xy, dtype=np.float64).reshape(-1)[:2]
        min_x, max_x, min_y, max_y = rect
        return bool(min_x <= x <= max_x and min_y <= y <= max_y)

    @staticmethod
    def _segment_intersects_rect(
        p0_xy: np.ndarray,
        p1_xy: np.ndarray,
        rect: tuple[float, float, float, float],
    ) -> bool:
        p0_xy = np.asarray(p0_xy, dtype=np.float64).reshape(2)
        p1_xy = np.asarray(p1_xy, dtype=np.float64).reshape(2)
        if MagentaSquare._point_inside_rect(p0_xy, rect) or MagentaSquare._point_inside_rect(
            p1_xy,
            rect,
        ):
            return True
        min_x, max_x, min_y, max_y = rect
        for fraction in np.linspace(0.0, 1.0, 41):
            point = p0_xy + fraction * (p1_xy - p0_xy)
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                return True
        return False

    def _planned_sc_transport_waypoints(
        self,
        start_pose: Pose,
        localized_target: LocalizedTransportTarget,
    ) -> list[tuple[float, float, float]] | None:
        if not bool(getattr(self, "sc_avoid_nic_zone", False)):
            return None
        start_board_xy = self._base_point_to_board_xy(
            np.array(
                [
                    start_pose.position.x,
                    start_pose.position.y,
                    start_pose.position.z,
                ],
                dtype=np.float64,
            ),
            localized_target.board_origin,
            localized_target.board_x_axis,
            localized_target.board_y_axis,
        )
        target_board_xy = localized_target.target_board_xyz[:2]
        rect = self._sc_nic_zone_rect(inflated=True)
        if not self._segment_intersects_rect(start_board_xy, target_board_xy, rect):
            return None
        clearance = float(getattr(self, "sc_nic_avoid_clearance_m", 0.03))
        _, max_x, _, max_y = rect
        waypoint_z = localized_target.target_xyz[2]
        dogleg_board = [
            np.array([max_x + clearance, start_board_xy[1], waypoint_z], dtype=np.float64),
            np.array([max_x + clearance, max_y + clearance, waypoint_z], dtype=np.float64),
            np.array([target_board_xy[0], max_y + clearance, waypoint_z], dtype=np.float64),
        ]
        waypoints = [
            self._board_xyz_to_base_xyz(
                waypoint,
                localized_target.board_origin,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
            for waypoint in dogleg_board
        ]
        self.get_logger().info(
            "MagentaSquare planned SC dogleg around NIC zone: "
            f"waypoints={[np.round(np.asarray(waypoint), 5).tolist() for waypoint in waypoints]}"
        )
        return waypoints

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

    def _project_base_points_to_pixels(
        self,
        points_base: np.ndarray,
        camera_info,
    ) -> np.ndarray | None:
        camera_frame = camera_info.header.frame_id
        if not camera_frame:
            return None
        transform = self._camera_to_base_link_transform(camera_frame, warn=False)
        if transform is None:
            return None
        camera_matrix = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(camera_matrix).all() or camera_matrix[0, 0] <= 0.0:
            return None
        camera_to_base_rot, camera_origin_base = transform
        points_base = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
        points_camera = (camera_to_base_rot.T @ (points_base - camera_origin_base).T).T
        if np.any(points_camera[:, 2] <= 1e-6):
            return None
        pixels = np.empty((points_camera.shape[0], 2), dtype=np.float64)
        pixels[:, 0] = (
            camera_matrix[0, 0] * points_camera[:, 0] / points_camera[:, 2]
            + camera_matrix[0, 2]
        )
        pixels[:, 1] = (
            camera_matrix[1, 1] * points_camera[:, 1] / points_camera[:, 2]
            + camera_matrix[1, 2]
        )
        return pixels

    @staticmethod
    def _sample_mask_line(
        mask: np.ndarray,
        p0_px: np.ndarray,
        p1_px: np.ndarray,
        *,
        samples: int = 24,
    ) -> float:
        p0_px = np.asarray(p0_px, dtype=np.float64).reshape(2)
        p1_px = np.asarray(p1_px, dtype=np.float64).reshape(2)
        height, width = mask.shape[:2]
        hits = 0
        total = 0
        for fraction in np.linspace(0.0, 1.0, max(samples, 2)):
            pixel = p0_px + fraction * (p1_px - p0_px)
            x = int(round(float(pixel[0])))
            y = int(round(float(pixel[1])))
            if 0 <= x < width and 0 <= y < height:
                total += 1
                if mask[y, x] > 0:
                    hits += 1
        if total == 0:
            return 0.0
        return float(hits) / float(total)

    @staticmethod
    def _sample_mask_polygon(mask: np.ndarray, polygon_px: np.ndarray) -> float:
        polygon_px = np.asarray(polygon_px, dtype=np.float64).reshape(-1, 2)
        if polygon_px.shape[0] < 3:
            return 0.0
        height, width = mask.shape[:2]
        if (
            np.max(polygon_px[:, 0]) < 0
            or np.max(polygon_px[:, 1]) < 0
            or np.min(polygon_px[:, 0]) >= width
            or np.min(polygon_px[:, 1]) >= height
        ):
            return 0.0
        polygon_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        clipped = polygon_px.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, width - 1.0)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, height - 1.0)
        cv2.fillConvexPoly(polygon_mask, np.round(clipped).astype(np.int32), 255)
        area = int(np.count_nonzero(polygon_mask))
        if area <= 0:
            return 0.0
        return float(np.count_nonzero(cv2.bitwise_and(mask, polygon_mask))) / float(area)

    @staticmethod
    def _nic_port_detection_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dark_mask = cv2.inRange(gray, 0, 85)
        dark_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, dark_kernel, iterations=1)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, dark_kernel, iterations=1)
        edges = cv2.Canny(gray, 45, 140)
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)
        return dark_mask, edge_mask

    def _sfp_nic_candidate_model_points(
        self,
        localization: SfpNicPortLocalization,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        half_w = self.SFP_NIC_PORT_OPENING_HALF_WIDTH
        half_h = self.SFP_NIC_PORT_OPENING_HALF_HEIGHT
        entrance = localization.entrance_base
        width = localization.width_axis_base
        height = localization.height_axis_base
        axis = localization.port_axis_base
        corners = np.vstack(
            [
                entrance - half_w * width - half_h * height,
                entrance + half_w * width - half_h * height,
                entrance + half_w * width + half_h * height,
                entrance - half_w * width + half_h * height,
            ]
        )
        axis_end = entrance + 0.025 * axis
        points = np.vstack([corners, entrance, axis_end])
        edge_indices = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)]
        return points, corners, edge_indices

    @staticmethod
    def _sfp_nic_rectangular_edge_score(rectangle_edge_scores: list[float]) -> float:
        if len(rectangle_edge_scores) != 4:
            return 0.0
        scores = np.clip(np.asarray(rectangle_edge_scores, dtype=np.float64), 0.0, 1.0)
        mean_score = float(np.mean(scores))
        second_lowest_score = float(np.sort(scores)[1])
        paired_closure_score = float(
            min(min(scores[0], scores[2]), min(scores[1], scores[3]))
        )
        return (
            0.30 * mean_score
            + 0.45 * second_lowest_score
            + 0.25 * paired_closure_score
        )

    @staticmethod
    def _undirected_angle_diff_rad(angle_a: float, angle_b: float) -> float:
        diff = (float(angle_a) - float(angle_b) + 0.5 * np.pi) % np.pi - 0.5 * np.pi
        return abs(float(diff))

    @staticmethod
    def _line_angle_rad(p0_px: np.ndarray, p1_px: np.ndarray) -> float | None:
        vector = np.asarray(p1_px, dtype=np.float64) - np.asarray(
            p0_px,
            dtype=np.float64,
        )
        if float(np.linalg.norm(vector)) <= 1e-6:
            return None
        return float(np.arctan2(vector[1], vector[0]))

    def _sfp_nic_candidate_long_edge_angle_rad(
        self,
        rectangle_pixels: np.ndarray,
    ) -> float | None:
        rectangle_pixels = np.asarray(rectangle_pixels, dtype=np.float64).reshape(4, 2)
        long_edge_angles = [
            self._line_angle_rad(rectangle_pixels[0], rectangle_pixels[1]),
            self._line_angle_rad(rectangle_pixels[3], rectangle_pixels[2]),
        ]
        valid_angles = [angle for angle in long_edge_angles if angle is not None]
        if not valid_angles:
            return None
        if len(valid_angles) == 1:
            return valid_angles[0]
        sin_sum = float(sum(np.sin(2.0 * angle) for angle in valid_angles))
        cos_sum = float(sum(np.cos(2.0 * angle) for angle in valid_angles))
        return 0.5 * float(np.arctan2(sin_sum, cos_sum))

    @staticmethod
    def _sfp_nic_image_line_segments(
        edge_mask: np.ndarray,
        image_shape: tuple[int, int],
    ) -> list[tuple[float, float]]:
        image_h, image_w = image_shape
        min_line_length = max(28, int(min(image_h, image_w) * 0.035))
        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1.0,
            theta=np.pi / 180.0,
            threshold=35,
            minLineLength=min_line_length,
            maxLineGap=10,
        )
        if lines is None:
            return []

        segments: list[tuple[float, float]] = []
        for line in lines.reshape(-1, 4):
            x0, y0, x1, y1 = [float(value) for value in line]
            dx = x1 - x0
            dy = y1 - y0
            length = float(np.hypot(dx, dy))
            if length < min_line_length:
                continue
            segments.append((float(np.arctan2(dy, dx)), length))
        return segments

    def _sfp_nic_rail_parallel_score_for_angle(
        self,
        long_edge_angle_rad: float | None,
        image_line_segments: list[tuple[float, float]],
    ) -> float:
        if long_edge_angle_rad is None or not image_line_segments:
            return 0.0

        angle_window = max(
            float(
                getattr(
                    self,
                    "sfp_nic_rail_parallel_angle_window_rad",
                    np.deg2rad(35.0),
                )
            ),
            1e-6,
        )
        nearby_segments = [
            (angle, length)
            for angle, length in image_line_segments
            if self._undirected_angle_diff_rad(angle, long_edge_angle_rad) <= angle_window
        ]
        if not nearby_segments:
            return 0.0

        weights = np.asarray([length for _, length in nearby_segments], dtype=np.float64)
        angles = np.asarray([angle for angle, _ in nearby_segments], dtype=np.float64)
        sin_sum = float(np.sum(weights * np.sin(2.0 * angles)))
        cos_sum = float(np.sum(weights * np.cos(2.0 * angles)))
        dominant_angle = 0.5 * float(np.arctan2(sin_sum, cos_sum))
        angle_error = self._undirected_angle_diff_rad(
            long_edge_angle_rad,
            dominant_angle,
        )
        tolerance = max(
            float(
                getattr(
                    self,
                    "sfp_nic_rail_parallel_tolerance_rad",
                    np.deg2rad(12.0),
                )
            ),
            1e-6,
        )
        return float(np.exp(-0.5 * (angle_error / tolerance) ** 2))

    def _sfp_nic_rail_parallel_score_from_pixels(
        self,
        rectangle_pixels: np.ndarray,
        image_line_segments: list[tuple[float, float]],
    ) -> float:
        return self._sfp_nic_rail_parallel_score_for_angle(
            self._sfp_nic_candidate_long_edge_angle_rad(rectangle_pixels),
            image_line_segments,
        )

    def _sfp_nic_localization_image_score_from_masks(
        self,
        localization: SfpNicPortLocalization,
        image_shape: tuple[int, int],
        dark_mask: np.ndarray,
        edge_mask: np.ndarray,
        camera_info,
        image_line_segments: list[tuple[float, float]] | None = None,
    ) -> SfpNicPortImageScore | None:
        image_h, image_w = image_shape
        model_points, _, edge_indices = self._sfp_nic_candidate_model_points(
            localization
        )
        pixels = self._project_base_points_to_pixels(model_points, camera_info)
        if pixels is None:
            return None
        center_px = pixels[4]
        if (
            center_px[0] < -0.1 * image_w
            or center_px[0] > 1.1 * image_w
            or center_px[1] < -0.1 * image_h
            or center_px[1] > 1.1 * image_h
        ):
            return None
        edge_scores = [
            self._sample_mask_line(edge_mask, pixels[i0], pixels[i1])
            for i0, i1 in edge_indices
        ]
        rectangle_edge_scores = edge_scores[:4]
        axis_edge_score = edge_scores[4] if len(edge_scores) > 4 else 0.0
        rectangular_edge_score = self._sfp_nic_rectangular_edge_score(
            rectangle_edge_scores
        )
        rail_parallel_score = 0.0
        if getattr(self, "sfp_nic_rail_parallel_score_enabled", True):
            if image_line_segments is None:
                image_line_segments = self._sfp_nic_image_line_segments(
                    edge_mask,
                    image_shape,
                )
            rail_parallel_score = self._sfp_nic_rail_parallel_score_from_pixels(
                pixels[:4],
                image_line_segments,
            )
        rectangle_dark_score = self._sample_mask_polygon(dark_mask, pixels[:4])
        center_dark_score = self._sample_mask_line(
            dark_mask,
            center_px - np.array([2.0, 0.0], dtype=np.float64),
            center_px + np.array([2.0, 0.0], dtype=np.float64),
            samples=5,
        )
        projected_width_px = float(np.linalg.norm(pixels[1] - pixels[0]))
        projected_height_px = float(np.linalg.norm(pixels[2] - pixels[1]))
        scale_score = min(max(min(projected_width_px, projected_height_px) / 6.0, 0.0), 1.0)
        rectangularity_gate = min(max(rectangular_edge_score / 0.20, 0.0), 1.0)
        rail_parallel_weight = min(
            max(float(getattr(self, "sfp_nic_rail_parallel_port_weight", 0.14)), 0.0),
            0.30,
        )
        rectangular_weight = 0.82 - rail_parallel_weight
        score = (
            rectangular_weight * rectangular_edge_score
            + 0.08 * axis_edge_score
            + rail_parallel_weight * rail_parallel_score
            + rectangularity_gate
            * (
                0.06 * rectangle_dark_score
                + 0.02 * center_dark_score
                + 0.02 * scale_score
            )
        )
        return SfpNicPortImageScore(
            score=score,
            rectangular_edge_score=rectangular_edge_score,
            axis_edge_score=axis_edge_score,
            rail_parallel_score=rail_parallel_score,
            rectangle_dark_score=rectangle_dark_score,
            center_dark_score=center_dark_score,
            scale_score=scale_score,
        )

    def _score_sfp_nic_localization_in_camera(
        self,
        localization: SfpNicPortLocalization,
        image_msg,
        camera_info,
    ) -> float | None:
        if not camera_info.header.frame_id:
            return None
        rgb = self._image_msg_to_rgb(image_msg)
        dark_mask, edge_mask = self._nic_port_detection_masks(rgb)
        image_line_segments = (
            self._sfp_nic_image_line_segments(edge_mask, rgb.shape[:2])
            if getattr(self, "sfp_nic_rail_parallel_score_enabled", True)
            else []
        )
        image_score = self._sfp_nic_localization_image_score_from_masks(
            localization,
            rgb.shape[:2],
            dark_mask,
            edge_mask,
            camera_info,
            image_line_segments=image_line_segments,
        )
        if image_score is None:
            return None
        return image_score.score

    def _score_sfp_nic_port_pair_localization_in_camera(
        self,
        target_port_index: int,
        localizations: dict[int, SfpNicPortLocalization],
        image_msg,
        camera_info,
    ) -> float | None:
        if not camera_info.header.frame_id:
            return None
        rgb = self._image_msg_to_rgb(image_msg)
        dark_mask, edge_mask = self._nic_port_detection_masks(rgb)
        return self._score_sfp_nic_port_pair_localization_from_masks(
            target_port_index,
            localizations,
            rgb.shape[:2],
            dark_mask,
            edge_mask,
            camera_info,
        )

    def _score_sfp_nic_port_pair_localization_from_masks(
        self,
        target_port_index: int,
        localizations: dict[int, SfpNicPortLocalization],
        image_shape: tuple[int, int],
        dark_mask: np.ndarray,
        edge_mask: np.ndarray,
        camera_info,
        image_line_segments: list[tuple[float, float]] | None = None,
    ) -> float | None:
        if 0 not in localizations or 1 not in localizations:
            return None
        if image_line_segments is None:
            image_line_segments = (
                self._sfp_nic_image_line_segments(edge_mask, image_shape)
                if getattr(self, "sfp_nic_rail_parallel_score_enabled", True)
                else []
            )
        image_scores: dict[int, SfpNicPortImageScore] = {}
        for port_index, localization in localizations.items():
            image_score = self._sfp_nic_localization_image_score_from_masks(
                localization,
                image_shape,
                dark_mask,
                edge_mask,
                camera_info,
                image_line_segments=image_line_segments,
            )
            if image_score is None:
                return None
            image_scores[port_index] = image_score

        target_score = image_scores[target_port_index]
        peer_score = image_scores[1 - target_port_index]
        rectangle_scores = np.array(
            [
                target_score.rectangular_edge_score,
                peer_score.rectangular_edge_score,
            ],
            dtype=np.float64,
        )
        port_scores = np.array(
            [target_score.score, peer_score.score],
            dtype=np.float64,
        )
        pair_rectangular_score = (
            0.45 * float(np.mean(rectangle_scores))
            + 0.35 * float(np.min(rectangle_scores))
            + 0.20 * target_score.rectangular_edge_score
        )
        port_score_support = (
            0.50 * float(np.mean(port_scores))
            + 0.30 * float(np.min(port_scores))
            + 0.20 * target_score.score
        )
        pair_axis_score = float(
            np.mean([target_score.axis_edge_score, peer_score.axis_edge_score])
        )
        pair_rail_parallel_score = (
            0.50
            * float(
                np.mean(
                    [target_score.rail_parallel_score, peer_score.rail_parallel_score]
                )
            )
            + 0.30
            * float(min(target_score.rail_parallel_score, peer_score.rail_parallel_score))
            + 0.20 * target_score.rail_parallel_score
        )
        pair_scale_score = float(
            np.mean([target_score.scale_score, peer_score.scale_score])
        )
        rail_parallel_weight = min(
            max(float(getattr(self, "sfp_nic_rail_parallel_pair_weight", 0.12)), 0.0),
            0.30,
        )
        rectangular_weight = 0.72 - rail_parallel_weight
        return (
            rectangular_weight * pair_rectangular_score
            + 0.18 * port_score_support
            + rail_parallel_weight * pair_rail_parallel_score
            + 0.07 * pair_axis_score
            + 0.03 * pair_scale_score
        )

    @staticmethod
    def _projected_line_overlaps_image(
        p0_px: np.ndarray,
        p1_px: np.ndarray,
        image_shape: tuple[int, int],
        *,
        margin_px: float = 8.0,
    ) -> bool:
        image_h, image_w = image_shape
        p0_px = np.asarray(p0_px, dtype=np.float64).reshape(2)
        p1_px = np.asarray(p1_px, dtype=np.float64).reshape(2)
        return not (
            max(p0_px[0], p1_px[0]) < -margin_px
            or min(p0_px[0], p1_px[0]) > image_w - 1.0 + margin_px
            or max(p0_px[1], p1_px[1]) < -margin_px
            or min(p0_px[1], p1_px[1]) > image_h - 1.0 + margin_px
        )

    def _score_projected_model_lines_from_masks(
        self,
        points_base: np.ndarray,
        edge_indices: list[tuple[int, int]],
        image_shape: tuple[int, int],
        edge_mask: np.ndarray,
        camera_info,
        *,
        min_visible_lines: int = 1,
    ) -> float | None:
        pixels = self._project_base_points_to_pixels(points_base, camera_info)
        if pixels is None:
            return None

        line_scores: list[float] = []
        for i0, i1 in edge_indices:
            p0_px = pixels[i0]
            p1_px = pixels[i1]
            if not self._projected_line_overlaps_image(p0_px, p1_px, image_shape):
                continue
            projected_length_px = float(np.linalg.norm(p1_px - p0_px))
            samples = max(8, min(96, int(projected_length_px / 3.0) + 1))
            line_scores.append(
                self._sample_mask_line(
                    edge_mask,
                    p0_px,
                    p1_px,
                    samples=samples,
                )
            )

        if len(line_scores) < min_visible_lines:
            return None

        scores = np.clip(np.asarray(line_scores, dtype=np.float64), 0.0, 1.0)
        return float(
            0.45 * np.mean(scores)
            + 0.35 * np.median(scores)
            + 0.20 * np.max(scores)
        )

    def _score_sfp_nic_card_line_group_from_masks(
        self,
        line_segments_card: list[tuple[np.ndarray, np.ndarray]],
        *,
        card_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
        localized_target: LocalizedTransportTarget,
        image_shape: tuple[int, int],
        edge_mask: np.ndarray,
        camera_info,
        min_visible_lines: int = 1,
    ) -> float | None:
        if not line_segments_card:
            return None
        model_points_card, edge_indices = self._line_segments_to_model_points(
            line_segments_card
        )
        model_points_base = self._sfp_nic_card_points_to_base(
            model_points_card,
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            localized_target=localized_target,
        )
        return self._score_projected_model_lines_from_masks(
            model_points_base,
            edge_indices,
            image_shape,
            edge_mask,
            camera_info,
            min_visible_lines=min_visible_lines,
        )

    def _score_sfp_nic_front_geometry_from_masks(
        self,
        *,
        card_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
        localized_target: LocalizedTransportTarget,
        image_shape: tuple[int, int],
        edge_mask: np.ndarray,
        camera_info,
    ) -> float | None:
        component_scores: list[tuple[float, float]] = []

        ridge_score = self._score_sfp_nic_card_line_group_from_masks(
            self._sfp_nic_front_ridge_lines_card(),
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            localized_target=localized_target,
            image_shape=image_shape,
            edge_mask=edge_mask,
            camera_info=camera_info,
            min_visible_lines=1,
        )
        if ridge_score is not None:
            component_scores.append((0.50, ridge_score))

        cage_score = self._score_sfp_nic_card_line_group_from_masks(
            self._sfp_nic_cage_lines_card(),
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            localized_target=localized_target,
            image_shape=image_shape,
            edge_mask=edge_mask,
            camera_info=camera_info,
            min_visible_lines=2,
        )
        if cage_score is not None:
            component_scores.append((0.50, cage_score))

        circular_feature_score = self._score_sfp_nic_card_line_group_from_masks(
            self._sfp_nic_circular_feature_lines_card(),
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            localized_target=localized_target,
            image_shape=image_shape,
            edge_mask=edge_mask,
            camera_info=camera_info,
            min_visible_lines=3,
        )
        if circular_feature_score is not None:
            component_scores.append((0.0, circular_feature_score))

        if not component_scores:
            return None
        total_weight = sum(weight for weight, _ in component_scores)
        if total_weight <= 0.0:
            return None
        return float(
            sum(weight * score for weight, score in component_scores) / total_weight
        )

    def _score_sfp_nic_front_geometry_in_camera(
        self,
        *,
        card_index: int,
        rail_translation_m: float,
        rail_yaw_rad: float,
        localized_target: LocalizedTransportTarget,
        image_msg,
        camera_info,
    ) -> float | None:
        if not camera_info.header.frame_id:
            return None
        rgb = self._image_msg_to_rgb(image_msg)
        _, edge_mask = self._nic_port_detection_masks(rgb)
        return self._score_sfp_nic_front_geometry_from_masks(
            card_index=card_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            localized_target=localized_target,
            image_shape=rgb.shape[:2],
            edge_mask=edge_mask,
            camera_info=camera_info,
        )

    @staticmethod
    def _combine_sfp_nic_visual_scores(
        port_score: float | None,
        front_geometry_score: float | None,
        front_geometry_weight: float,
    ) -> float | None:
        if port_score is None:
            return front_geometry_score
        if front_geometry_score is None:
            return port_score
        weight = min(max(float(front_geometry_weight), 0.0), 0.95)
        return float((1.0 - weight) * port_score + weight * front_geometry_score)

    @staticmethod
    def _debug_overlay_safe_token(value: str) -> str:
        token = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in str(value)
        ).strip("_")
        return token or "unknown"

    @staticmethod
    def _pixel_tuple(pixel: np.ndarray) -> tuple[int, int] | None:
        pixel = np.asarray(pixel, dtype=np.float64).reshape(2)
        if not np.isfinite(pixel).all():
            return None
        return (
            int(round(float(np.clip(pixel[0], -1.0e6, 1.0e6)))),
            int(round(float(np.clip(pixel[1], -1.0e6, 1.0e6)))),
        )

    def _draw_debug_line(
        self,
        overlay_rgb: np.ndarray,
        p0_px: np.ndarray,
        p1_px: np.ndarray,
        color_rgb: tuple[int, int, int],
        *,
        thickness: int = 2,
    ) -> None:
        p0 = self._pixel_tuple(p0_px)
        p1 = self._pixel_tuple(p1_px)
        if p0 is None or p1 is None:
            return
        cv2.line(overlay_rgb, p0, p1, color_rgb, thickness, lineType=cv2.LINE_AA)

    def _draw_debug_text_block(
        self,
        overlay_rgb: np.ndarray,
        lines: list[str],
        *,
        origin_px: tuple[int, int] = (10, 22),
    ) -> None:
        x, y = origin_px
        line_height = 18
        for index, line in enumerate(lines):
            text_origin = (x, y + index * line_height)
            cv2.putText(
                overlay_rgb,
                line,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_rgb,
                line,
                text_origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_sfp_nic_port_overlay(
        self,
        overlay_rgb: np.ndarray,
        camera_info,
        port_index: int,
        localization: SfpNicPortLocalization,
        *,
        target_port_index: int,
    ) -> None:
        image_shape = overlay_rgb.shape[:2]
        model_points, _, edge_indices = self._sfp_nic_candidate_model_points(
            localization
        )
        pixels = self._project_base_points_to_pixels(model_points, camera_info)
        if pixels is None:
            return

        is_target = port_index == target_port_index
        color = (0, 220, 255) if port_index == 0 else (255, 170, 0)
        thickness = 3 if is_target else 2
        for i0, i1 in edge_indices[:4]:
            if self._projected_line_overlaps_image(
                pixels[i0],
                pixels[i1],
                image_shape,
                margin_px=24.0,
            ):
                self._draw_debug_line(
                    overlay_rgb,
                    pixels[i0],
                    pixels[i1],
                    color,
                    thickness=thickness,
                )

        label_point = self._pixel_tuple(pixels[0] + np.array([3.0, -4.0]))
        if label_point is not None:
            cv2.putText(
                overlay_rgb,
                f"P{port_index}",
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        if not is_target:
            return

        entrance_px = self._pixel_tuple(pixels[4])
        if entrance_px is not None:
            cv2.drawMarker(
                overlay_rgb,
                entrance_px,
                (0, 255, 0),
                cv2.MARKER_CROSS,
                18,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlay_rgb, entrance_px, 5, (0, 255, 0), 2, cv2.LINE_AA)

        axis_start = self._pixel_tuple(pixels[4])
        axis_end = self._pixel_tuple(pixels[5])
        if axis_start is not None and axis_end is not None:
            cv2.arrowedLine(
                overlay_rgb,
                axis_start,
                axis_end,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

    def _draw_sfp_nic_ground_truth_port_overlay(
        self,
        overlay_rgb: np.ndarray,
        camera_info,
        port_index: int,
        localization: SfpNicPortLocalization,
    ) -> None:
        image_shape = overlay_rgb.shape[:2]
        model_points, _, edge_indices = self._sfp_nic_candidate_model_points(
            localization
        )
        pixels = self._project_base_points_to_pixels(model_points, camera_info)
        if pixels is None:
            return

        color = (255, 40, 40)
        for i0, i1 in edge_indices[:4]:
            if self._projected_line_overlaps_image(
                pixels[i0],
                pixels[i1],
                image_shape,
                margin_px=24.0,
            ):
                self._draw_debug_line(
                    overlay_rgb,
                    pixels[i0],
                    pixels[i1],
                    color,
                    thickness=2,
                )

        entrance_px = self._pixel_tuple(pixels[4])
        if entrance_px is not None:
            cv2.drawMarker(
                overlay_rgb,
                entrance_px,
                color,
                cv2.MARKER_TILTED_CROSS,
                20,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(overlay_rgb, entrance_px, 6, color, 2, cv2.LINE_AA)

        axis_start = self._pixel_tuple(pixels[4])
        axis_end = self._pixel_tuple(pixels[5])
        if axis_start is not None and axis_end is not None:
            cv2.arrowedLine(
                overlay_rgb,
                axis_start,
                axis_end,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
                tipLength=0.25,
            )

        label_point = self._pixel_tuple(pixels[2] + np.array([3.0, -4.0]))
        if label_point is not None:
            cv2.putText(
                overlay_rgb,
                f"GT P{port_index}",
                label_point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _publish_sfp_nic_debug_overlay(
        self,
        overlay_rgb: np.ndarray,
        camera_info,
    ) -> None:
        publisher = getattr(self, "_sfp_nic_debug_overlay_pub", None)
        image_msg_type = getattr(self, "_sfp_nic_debug_image_msg_type", None)
        if publisher is None or image_msg_type is None:
            return

        overlay_rgb = np.ascontiguousarray(overlay_rgb, dtype=np.uint8)
        msg = image_msg_type()
        msg.height = int(overlay_rgb.shape[0])
        msg.width = int(overlay_rgb.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(overlay_rgb.shape[1] * 3)
        msg.data = overlay_rgb.tobytes()
        try:
            msg.header.frame_id = camera_info.header.frame_id
            msg.header.stamp = self._parent_node.get_clock().now().to_msg()
        except AttributeError:
            pass
        publisher.publish(msg)

    def _save_sfp_nic_debug_overlay(
        self,
        overlay_rgb: np.ndarray,
        *,
        camera_name: str,
        task: Task,
        used_visual_detection: bool,
    ) -> None:
        if not getattr(self, "sfp_nic_debug_overlay_save", False):
            return

        debug_dir = Path(
            getattr(
                self,
                "sfp_nic_debug_overlay_dir",
                Path("outputs/debug/sfp_nic_port_pair"),
            )
        )
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
        except OSError as ex:
            self.get_logger().warn(
                f"SFP/NIC debug overlay directory could not be created: {ex}"
            )
            return

        seq = int(getattr(self, "_sfp_nic_debug_overlay_seq", 0))
        mode = "visual" if used_visual_detection else "geometry_fallback"
        filename = (
            f"{seq:05d}_"
            f"{self._debug_overlay_safe_token(task.target_module_name)}_"
            f"{self._debug_overlay_safe_token(task.port_name)}_"
            f"{self._debug_overlay_safe_token(camera_name)}_"
            f"{mode}.png"
        )
        path = debug_dir / filename
        bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(path), bgr):
            self.get_logger().warn(f"SFP/NIC debug overlay write failed: {path}")
            return
        self.get_logger().info(f"SFP/NIC debug overlay saved: {path}")

    def _sfp_nic_pair_localizations_for_overlay(
        self,
        task: Task,
        localized_target: LocalizedTransportTarget,
        localization: SfpNicPortLocalization,
    ) -> tuple[int, dict[int, SfpNicPortLocalization]]:
        return self._sfp_nic_port_pair_localizations_from_board_pose(
            task,
            localized_target,
            rail_translation_m=localization.rail_translation_m,
            rail_yaw_rad=localization.rail_yaw_rad,
            score=localization.score,
            confidence=localization.confidence,
            observed_camera_count=localization.observed_camera_count,
            source=localization.source,
            accepted=localization.accepted,
        )

    def _save_sfp_nic_port_pair_debug_overlays(
        self,
        task: Task,
        obs,
        localized_target: LocalizedTransportTarget,
        localization: SfpNicPortLocalization,
        *,
        localizer_mode: str,
        used_visual_detection: bool,
        fallback_reason: str = "",
        visual_confidence: float | None = None,
        camera_scores: list[float] | None = None,
        port_scores: list[float] | None = None,
        front_geometry_scores: list[float] | None = None,
    ) -> None:
        if not getattr(self, "sfp_nic_debug_overlay_enabled", False):
            return
        if not (
            getattr(self, "sfp_nic_debug_overlay_save", False)
            or getattr(self, "sfp_nic_debug_overlay_stream", False)
        ):
            return

        target_port_index, localizations = self._sfp_nic_pair_localizations_for_overlay(
            task,
            localized_target,
            localization,
        )
        selected = localizations.get(target_port_index, localization)
        ground_truth = self._true_sfp_nic_port_localization_from_trial_config(task)
        ground_truth_localization = None
        ground_truth_source = ""
        if ground_truth is not None:
            ground_truth_localization, path, trial_id = ground_truth
            ground_truth_source = f"{path.name}:{trial_id}"
        camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        scores_text = (
            f"scores all={np.round(camera_scores or [], 3).tolist()} "
            f"port={np.round(port_scores or [], 3).tolist()} "
            f"front={np.round(front_geometry_scores or [], 3).tolist()}"
        )
        confidence_text = f"conf={localization.confidence:.3f}"
        if visual_confidence is not None:
            confidence_text += f" visual_conf={visual_confidence:.3f}"

        for camera_name, image_msg, camera_info in camera_inputs:
            if not getattr(camera_info.header, "frame_id", ""):
                continue
            try:
                rgb = self._image_msg_to_rgb(image_msg)
            except Exception as ex:
                self.get_logger().warn(
                    f"SFP/NIC debug overlay skipped {camera_name}: image decode failed: {ex}"
                )
                continue

            overlay = np.ascontiguousarray(rgb.copy())
            for port_index in sorted(localizations):
                self._draw_sfp_nic_port_overlay(
                    overlay,
                    camera_info,
                    port_index,
                    localizations[port_index],
                    target_port_index=target_port_index,
                )

            gt_error_text = ""
            if ground_truth_localization is not None:
                self._draw_sfp_nic_ground_truth_port_overlay(
                    overlay,
                    camera_info,
                    target_port_index,
                    ground_truth_localization,
                )
                det_minus_gt = selected.entrance_base - ground_truth_localization.entrance_base
                gt_error_text = (
                    f"gt={ground_truth_source} "
                    f"det_minus_gt={np.round(det_minus_gt, 5).tolist()} "
                    f"|err|={float(np.linalg.norm(det_minus_gt)):.5f}m"
                )

            status = "visual detection" if used_visual_detection else "geometry fallback"
            text_lines = [
                f"SFP/NIC {status} | {localizer_mode} | target P{target_port_index}",
                confidence_text,
                f"source={localization.source}",
                f"rail_t={localization.rail_translation_m:.5f} yaw={np.degrees(localization.rail_yaw_rad):.2f}deg",
                f"entrance={np.round(selected.entrance_base, 5).tolist()}",
                f"axis={np.round(selected.port_axis_base, 4).tolist()}",
                scores_text,
            ]
            if gt_error_text:
                text_lines.insert(6, gt_error_text)
            if fallback_reason:
                text_lines.insert(2, f"fallback={fallback_reason}")
            self._draw_debug_text_block(overlay, text_lines)

            self._save_sfp_nic_debug_overlay(
                overlay,
                camera_name=camera_name,
                task=task,
                used_visual_detection=used_visual_detection,
            )
            if getattr(self, "sfp_nic_debug_overlay_stream", False):
                self._publish_sfp_nic_debug_overlay(overlay, camera_info)

        self._sfp_nic_debug_overlay_seq = int(
            getattr(self, "_sfp_nic_debug_overlay_seq", 0)
        ) + 1

    @staticmethod
    def _grid_with_zero(min_value: float, max_value: float, steps: int) -> np.ndarray:
        steps = max(int(steps), 1)
        values = np.linspace(float(min_value), float(max_value), steps)
        if min_value <= 0.0 <= max_value and not np.any(np.isclose(values, 0.0)):
            values = np.sort(np.append(values, 0.0))
        return values

    def _prior_sfp_nic_port_localization(
        self,
        task: Task,
        localized_target: LocalizedTransportTarget,
        *,
        source: str,
    ) -> SfpNicPortLocalization:
        return self._sfp_nic_port_localization_from_board_pose(
            task,
            localized_target,
            rail_translation_m=0.0,
            rail_yaw_rad=0.0,
            score=0.0,
            confidence=0.0,
            observed_camera_count=0,
            source=source,
            accepted=False,
        )

    def _localize_sfp_nic_port_entrance(
        self,
        task: Task,
        obs,
        localized_target: LocalizedTransportTarget,
    ) -> SfpNicPortLocalization | None:
        if not self.sfp_nic_localizer_enabled:
            prior = self._prior_sfp_nic_port_localization(
                task,
                localized_target,
                source="SFP/NIC geometry prior (localizer disabled)",
            )
            self._save_sfp_nic_port_pair_debug_overlays(
                task,
                obs,
                localized_target,
                prior,
                localizer_mode="disabled",
                used_visual_detection=False,
                fallback_reason="localizer disabled",
            )
            return prior

        translations = self._grid_with_zero(
            self.sfp_nic_rail_translation_min,
            self.sfp_nic_rail_translation_max,
            self.sfp_nic_translation_grid_steps,
        )
        yaws = self._grid_with_zero(
            self.sfp_nic_rail_yaw_min,
            self.sfp_nic_rail_yaw_max,
            self.sfp_nic_yaw_grid_steps,
        )
        raw_camera_inputs = (
            ("left", obs.left_image, obs.left_camera_info),
            ("center", obs.center_image, obs.center_camera_info),
            ("right", obs.right_image, obs.right_camera_info),
        )
        camera_inputs = []
        for camera_name, image_msg, camera_info in raw_camera_inputs:
            if not camera_info.header.frame_id:
                continue
            rgb = self._image_msg_to_rgb(image_msg)
            dark_mask, edge_mask = self._nic_port_detection_masks(rgb)
            image_shape = rgb.shape[:2]
            image_line_segments = (
                self._sfp_nic_image_line_segments(edge_mask, image_shape)
                if getattr(self, "sfp_nic_rail_parallel_score_enabled", True)
                else []
            )
            camera_inputs.append(
                (
                    camera_name,
                    image_shape,
                    dark_mask,
                    edge_mask,
                    camera_info,
                    image_line_segments,
                )
            )
        best: SfpNicPortLocalization | None = None
        best_camera_scores: list[float] = []
        best_port_camera_scores: list[float] = []
        best_front_geometry_camera_scores: list[float] = []
        translation_scale = max(
            abs(self.sfp_nic_rail_translation_min),
            abs(self.sfp_nic_rail_translation_max),
            1e-6,
        )
        yaw_scale = max(
            abs(self.sfp_nic_rail_yaw_min),
            abs(self.sfp_nic_rail_yaw_max),
            1e-6,
        )
        pair_localizer_enabled = getattr(
            self,
            "sfp_nic_pair_localizer_enabled",
            True,
        )
        front_geometry_enabled = getattr(
            self,
            "sfp_nic_front_geometry_localizer_enabled",
            True,
        )
        front_geometry_weight = getattr(
            self,
            "sfp_nic_front_geometry_score_weight",
            0.40,
        )
        card_index, _ = self._task_indices(task)
        for rail_translation_m in translations:
            for rail_yaw_rad in yaws:
                if pair_localizer_enabled:
                    source = (
                        "SFP/NIC projected port-pair + front-geometry localizer"
                        if front_geometry_enabled
                        else "SFP/NIC projected port-pair localizer"
                    )
                    target_port_index, port_localizations = (
                        self._sfp_nic_port_pair_localizations_from_board_pose(
                            task,
                            localized_target,
                            rail_translation_m=float(rail_translation_m),
                            rail_yaw_rad=float(rail_yaw_rad),
                            score=0.0,
                            confidence=0.0,
                            observed_camera_count=0,
                            source=source,
                            accepted=False,
                        )
                    )
                    localization = port_localizations[target_port_index]
                else:
                    source = (
                        "SFP/NIC projected port + front-geometry localizer"
                        if front_geometry_enabled
                        else "SFP/NIC projected port localizer"
                    )
                    target_port_index = -1
                    port_localizations = {}
                    localization = self._sfp_nic_port_localization_from_board_pose(
                        task,
                        localized_target,
                        rail_translation_m=float(rail_translation_m),
                        rail_yaw_rad=float(rail_yaw_rad),
                        score=0.0,
                        confidence=0.0,
                        observed_camera_count=0,
                        source=source,
                        accepted=False,
                    )
                camera_scores = []
                port_camera_scores = []
                front_geometry_camera_scores = []
                for (
                    _,
                    image_shape,
                    dark_mask,
                    edge_mask,
                    camera_info,
                    image_line_segments,
                ) in camera_inputs:
                    if pair_localizer_enabled:
                        port_score = (
                            self._score_sfp_nic_port_pair_localization_from_masks(
                                target_port_index,
                                port_localizations,
                                image_shape,
                                dark_mask,
                                edge_mask,
                                camera_info,
                                image_line_segments=image_line_segments,
                            )
                        )
                    else:
                        image_score = self._sfp_nic_localization_image_score_from_masks(
                            localization,
                            image_shape,
                            dark_mask,
                            edge_mask,
                            camera_info,
                            image_line_segments=image_line_segments,
                        )
                        port_score = None if image_score is None else image_score.score

                    front_geometry_score = None
                    if front_geometry_enabled:
                        front_geometry_score = (
                            self._score_sfp_nic_front_geometry_from_masks(
                                card_index=card_index,
                                rail_translation_m=float(rail_translation_m),
                                rail_yaw_rad=float(rail_yaw_rad),
                                localized_target=localized_target,
                                image_shape=image_shape,
                                edge_mask=edge_mask,
                                camera_info=camera_info,
                            )
                        )

                    score = self._combine_sfp_nic_visual_scores(
                        port_score,
                        front_geometry_score,
                        front_geometry_weight,
                    )
                    if score is not None:
                        camera_scores.append(score)
                    if port_score is not None:
                        port_camera_scores.append(port_score)
                    if front_geometry_score is not None:
                        front_geometry_camera_scores.append(front_geometry_score)
                if not camera_scores:
                    continue
                image_score = float(np.mean(camera_scores))
                prior_penalty = self.sfp_nic_localizer_prior_weight * (
                    (float(rail_translation_m) / translation_scale) ** 2
                    + (float(rail_yaw_rad) / yaw_scale) ** 2
                )
                total_score = image_score - prior_penalty
                localization.score = total_score
                localization.confidence = max(0.0, min(1.0, image_score))
                localization.observed_camera_count = len(camera_scores)
                if best is None or total_score > best.score:
                    best = localization
                    best_camera_scores = camera_scores
                    best_port_camera_scores = port_camera_scores
                    best_front_geometry_camera_scores = front_geometry_camera_scores

        if best is None:
            self.get_logger().warn(
                "SFP/NIC port localizer could not project the target entrance into "
                "any wrist camera; using prior fallback."
            )
            if self.sfp_nic_allow_prior_fallback:
                prior = self._prior_sfp_nic_port_localization(
                    task,
                    localized_target,
                    source="SFP/NIC geometry prior (no projected camera evidence)",
                )
                self._save_sfp_nic_port_pair_debug_overlays(
                    task,
                    obs,
                    localized_target,
                    prior,
                    localizer_mode="no_projection",
                    used_visual_detection=False,
                    fallback_reason="no projected camera evidence",
                )
                return prior
            return None

        best.accepted = best.confidence >= self.sfp_nic_localizer_min_confidence
        localizer_mode = "pair" if pair_localizer_enabled else "single"
        if front_geometry_enabled:
            localizer_mode += "+front_geometry"
        self.get_logger().info(
            "SFP/NIC port localizer result: "
            f"mode={localizer_mode}, "
            f"accepted={best.accepted}, "
            f"confidence={best.confidence:.3f}, score={best.score:.3f}, "
            f"camera_scores={[round(score, 3) for score in best_camera_scores]}, "
            f"port_scores={[round(score, 3) for score in best_port_camera_scores]}, "
            f"front_geometry_scores="
            f"{[round(score, 3) for score in best_front_geometry_camera_scores]}, "
            f"observed_cameras={best.observed_camera_count}, "
            f"rail_translation={best.rail_translation_m:.5f}, "
            f"rail_yaw={best.rail_yaw_rad:.5f}, "
            f"entrance_base={np.round(best.entrance_base, 5).tolist()}, "
            f"axis_base={np.round(best.port_axis_base, 5).tolist()}"
        )
        if best.accepted:
            self._save_sfp_nic_port_pair_debug_overlays(
                task,
                obs,
                localized_target,
                best,
                localizer_mode=localizer_mode,
                used_visual_detection=True,
                camera_scores=best_camera_scores,
                port_scores=best_port_camera_scores,
                front_geometry_scores=best_front_geometry_camera_scores,
            )
            return best
        if self.sfp_nic_allow_prior_fallback:
            self.get_logger().warn(
                "SFP/NIC port localizer confidence is below threshold; using the "
                "SDF-derived geometry prior for this insertion attempt."
            )
            prior = self._prior_sfp_nic_port_localization(
                task,
                localized_target,
                source="SFP/NIC geometry prior (weak visual evidence)",
            )
            self._save_sfp_nic_port_pair_debug_overlays(
                task,
                obs,
                localized_target,
                prior,
                localizer_mode=localizer_mode,
                used_visual_detection=False,
                fallback_reason="weak visual evidence",
                visual_confidence=best.confidence,
                camera_scores=best_camera_scores,
                port_scores=best_port_camera_scores,
                front_geometry_scores=best_front_geometry_camera_scores,
            )
            return prior
        self.get_logger().warn(
            "SFP/NIC port localizer rejected weak visual evidence and prior fallback "
            "is disabled."
        )
        self._save_sfp_nic_port_pair_debug_overlays(
            task,
            obs,
            localized_target,
            best,
            localizer_mode=localizer_mode,
            used_visual_detection=False,
            fallback_reason="weak visual evidence rejected; prior fallback disabled",
            visual_confidence=best.confidence,
            camera_scores=best_camera_scores,
            port_scores=best_port_camera_scores,
            front_geometry_scores=best_front_geometry_camera_scores,
        )
        return None

    def _localize_sfp_nic_port_entrance_with_view_search(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        localized_target: LocalizedTransportTarget,
        start_pose: Pose,
    ) -> tuple[SfpNicPortLocalization | None, Pose]:
        obs = get_observation()
        localization = self._localize_sfp_nic_port_entrance(task, obs, localized_target)
        if (
            not self.sfp_nic_localizer_enabled
            or (
                localization is not None
                and localization.accepted
            )
            or not self.sfp_nic_view_search_enabled
            or self.sfp_nic_view_search_offset_m <= 0.0
        ):
            return localization, start_pose

        offset = self.sfp_nic_view_search_offset_m
        board_offsets = [
            np.array([offset, 0.0, 0.0], dtype=np.float64),
            np.array([-offset, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, offset, 0.0], dtype=np.float64),
            np.array([0.0, -offset, 0.0], dtype=np.float64),
        ]
        best = localization
        current_pose = start_pose
        self.get_logger().info(
            "SFP/NIC localizer running small close-view search: "
            f"offset={offset:.4f} m, samples={len(board_offsets)}"
        )
        for board_offset in board_offsets:
            offset_base = self._board_vector_to_base_vector(
                board_offset,
                localized_target.board_x_axis,
                localized_target.board_y_axis,
            )
            target_xyz = (
                current_pose.position.x + float(offset_base[0]),
                current_pose.position.y + float(offset_base[1]),
                current_pose.position.z,
            )
            current_pose = self._move_to_pose(
                move_robot,
                current_pose,
                target_xyz,
                duration_sec=self.sfp_nic_view_search_duration_sec,
            )
            candidate = self._localize_sfp_nic_port_entrance(
                task,
                get_observation(),
                localized_target,
            )
            if candidate is not None and (
                best is None or candidate.confidence > best.confidence
            ):
                best = candidate
            if candidate is not None and candidate.accepted:
                return candidate, current_pose

        return best, current_pose

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

    def _matching_trial_entry(self, task: Task) -> tuple[Path, str, dict] | None:
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
        return matches[0]

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
        matching_entry = self._matching_trial_entry(task)
        if matching_entry is None:
            return None
        path, trial_id, trial_config = matching_entry
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

    def _true_sfp_nic_port_localization_from_trial_config(
        self,
        task: Task,
    ) -> tuple[SfpNicPortLocalization, Path, str] | None:
        try:
            card_index, port_index = self._task_indices(task)
        except ValueError:
            return None

        matching_entry = self._matching_trial_entry(task)
        true_board_pose = self._true_board_pose_from_trial_config(task)
        if matching_entry is None or true_board_pose is None:
            return None

        path, trial_id, trial_config = matching_entry
        true_origin, true_x_axis, true_y_axis, _, _, _ = true_board_pose
        rail_key = f"nic_rail_{card_index}"
        try:
            rail_config = trial_config["scene"]["task_board"][rail_key]
            if not bool(rail_config.get("entity_present", False)):
                self.get_logger().warn(
                    f"TransportToMean trial debug config {path}:{trial_id} has "
                    f"{rail_key} marked absent; cannot draw SFP/NIC ground truth."
                )
                return None
            entity_pose = rail_config["entity_pose"]
            rail_translation_m = float(entity_pose.get("translation", 0.0))
            rail_yaw_rad = float(entity_pose.get("yaw", 0.0))
        except (KeyError, TypeError, ValueError) as ex:
            self.get_logger().warn(
                f"TransportToMean trial debug config {path}:{trial_id} is missing "
                f"{rail_key} ground-truth pose: {ex}"
            )
            return None

        localized_target = LocalizedTransportTarget(
            target_xyz=(0.0, 0.0, 0.0),
            board_origin=true_origin,
            board_x_axis=true_x_axis,
            board_y_axis=true_y_axis,
            target_board_xyz=np.zeros(3, dtype=np.float64),
            source=f"trial-config ground truth {path}:{trial_id}",
        )
        localization = self._sfp_nic_port_localization_for_indices(
            localized_target,
            card_index=card_index,
            port_index=port_index,
            rail_translation_m=rail_translation_m,
            rail_yaw_rad=rail_yaw_rad,
            score=1.0,
            confidence=1.0,
            observed_camera_count=0,
            source=f"SFP/NIC trial-config ground truth {path}:{trial_id}",
            accepted=True,
        )
        return localization, path, trial_id

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
        edge_gap_fractions = []
        edge_thickness_scores = []
        edge_gap_scores = []
        for idx in range(4):
            p0 = box_xy[idx]
            p1 = box_xy[(idx + 1) % 4]
            edge_vec = p1 - p0
            edge_len = float(np.linalg.norm(edge_vec))
            if edge_len <= 1e-9:
                edge_support.append(0)
                edge_gap_centers.append(0.5 * (p0 + p1))
                edge_gap_fractions.append(0.0)
                edge_thickness_scores.append(0.0)
                edge_gap_scores.append(0.0)
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
            thick_mask = (
                (dist <= self.magenta_edge_thickness_tolerance_m)
                & (along >= -self.magenta_edge_support_tolerance_m)
                & (along <= edge_len + self.magenta_edge_support_tolerance_m)
            )
            if np.count_nonzero(thick_mask) > 0:
                thickness_score = float(
                    np.percentile(dist[thick_mask], 90)
                    / max(self.magenta_edge_thickness_tolerance_m, 1e-9)
                )
            else:
                thickness_score = 0.0
            gap_score = (
                self.magenta_gap_fraction_weight * gap_fraction
                + self.magenta_gap_thickness_weight * thickness_score
            )
            edge_support.append(support_count / max(edge_len, 1e-9) - 200.0 * gap_fraction)
            edge_gap_centers.append(gap_center)
            edge_gap_fractions.append(float(gap_fraction))
            edge_thickness_scores.append(thickness_score)
            edge_gap_scores.append(gap_score)

        gap_edge_idx = int(np.argmax(edge_gap_scores))
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
            f"edge_gap_fraction={np.round(edge_gap_fractions, 3).tolist()}, "
            f"edge_thickness_score={np.round(edge_thickness_scores, 3).tolist()}, "
            f"edge_gap_score={np.round(edge_gap_scores, 3).tolist()}, "
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

    def _spawn_magenta_observations(
        self,
        obs,
    ) -> list[MagentaMarkerObservation | MagentaRoiObservation]:
        observations = self._multicamera_magenta_roi_observations(obs)
        if not observations:
            return []

        geometry_fit_observations = [
            observation
            for observation in observations
            if isinstance(observation, MagentaMarkerObservation)
        ]
        if geometry_fit_observations:
            return geometry_fit_observations

        if not self.magenta_spawn_require_consensus:
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
            return best_cluster

        self.get_logger().info(
            "MagentaSquare rejected weak spawn magenta evidence: "
            f"cameras={[observation.camera_name for observation in observations]}, "
            f"centers={[np.round(ob.marker_center_base, 4).tolist() for ob in observations]}, "
            f"require_consensus={self.magenta_spawn_require_consensus}, "
            f"consensus_distance={self.magenta_consensus_distance_m:.3f} m"
        )
        return []

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
        marker_edge_axis = None
        if marker_y_axis is not None:
            marker_edge_axis = self._normalized_xy(
                np.array([marker_y_axis[1], -marker_y_axis[0], 0.0], dtype=np.float64)
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
        parallel_reject_count = 0
        for candidate in edge_candidates:
            edge_dir = self._normalized_xy(candidate.p1_base - candidate.p0_base)
            if edge_dir is None:
                continue
            parallel_score = 0.0
            if marker_edge_axis is not None:
                parallel_score = abs(float(np.dot(edge_dir[:2], marker_edge_axis[:2])))
                if parallel_score < self.magenta_edge_parallel_min_dot:
                    parallel_reject_count += 1
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
                + parallel_score
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
                    parallel_score,
                )
            )

        if not scored:
            self.get_logger().warn(
                "MagentaSquare found no board +y edge consistent with marker prior: "
                f"marker_center={np.round(marker_center, 4).tolist()}, "
                f"candidate_count={len(edge_candidates)}, "
                f"parallel_reject_count={parallel_reject_count}, "
                f"parallel_min_dot={self.magenta_edge_parallel_min_dot:.3f}"
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
            parallel_score,
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
            f"parallel_score={parallel_score:.3f}, "
            f"parallel_min_dot={self.magenta_edge_parallel_min_dot:.3f}, "
            f"marker_axis_score={marker_axis_score:.3f}, "
            f"score={best_score:.3f}, "
            f"candidate_count={len(edge_candidates)}, "
            f"parallel_reject_count={parallel_reject_count}, "
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
        if self.magenta_require_geometry_fit_for_target:
            geometry_fit_observations = [
                observation
                for observation in observations
                if isinstance(observation, MagentaMarkerObservation)
            ]
            if not geometry_fit_observations:
                self.get_logger().warn(
                    "MagentaSquare found only partial/rough magenta ROI evidence; "
                    "requiring a geometry-fit hollow square before accepting a "
                    "magenta-guided transport target."
                )
                return None
            observations = geometry_fit_observations

        edge_target = self._plus_y_edge_target_from_magenta_prior(task, obs, observations)
        if edge_target is not None:
            return edge_target
        self.get_logger().warn(
            "MagentaSquare found rough magenta ROI but could not fit the board +y edge; "
            "continuing to another XY view sample or short-edge fallback."
        )
        return None

    def _localized_target_from_board_pose(
        self,
        task: Task,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
        *,
        source: str,
    ) -> LocalizedTransportTarget | None:
        try:
            target_board = np.asarray(
                self._board_pose_target_xyz_board(task),
                dtype=np.float64,
            )
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return None

        board_origin = np.asarray(board_origin, dtype=np.float64).reshape(3)
        board_x_axis = np.asarray(board_x_axis, dtype=np.float64).reshape(3)
        board_y_axis = np.asarray(board_y_axis, dtype=np.float64).reshape(3)
        target_base = np.array(
            self._board_xyz_to_base_xyz(
                target_board,
                board_origin,
                board_x_axis,
                board_y_axis,
            ),
            dtype=np.float64,
        )
        target_base[2] = max(
            self.board_plane_z + target_board[2] + self.z_offset,
            self.min_target_z,
        )
        target_mode = (
            "sc_rail"
            if task.target_module_name.startswith("sc_port_")
            else self.board_pose_target_mode
        )
        self.get_logger().info(
            "TransportToMean board-pose target: "
            f"source={source}, "
            f"mode={target_mode}, "
            f"board_target={np.round(target_board, 4).tolist()}, "
            f"target_base={np.round(target_base, 4).tolist()}"
        )
        localized_target = LocalizedTransportTarget(
            target_xyz=tuple(float(v) for v in target_base),
            board_origin=board_origin,
            board_x_axis=board_x_axis,
            board_y_axis=board_y_axis,
            target_board_xyz=target_board,
            source=source,
        )
        self._last_localized_target = localized_target
        return localized_target

    def _target_xyz_from_board_pose(
        self,
        task: Task,
        board_origin: np.ndarray,
        board_x_axis: np.ndarray,
        board_y_axis: np.ndarray,
        *,
        source: str,
    ) -> tuple[float, float, float] | None:
        localized_target = self._localized_target_from_board_pose(
            task,
            board_origin,
            board_x_axis,
            board_y_axis,
            source=source,
        )
        if localized_target is None:
            return None
        return localized_target.target_xyz

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
        self._last_localized_target = None
        try:
            task_family = self._task_family(task)
            self.get_logger().info(
                "MagentaSquare parsed task target: "
                f"{self._task_target_summary(task)}"
            )
        except ValueError as ex:
            self.get_logger().error(str(ex))
            return False

        target_xyz = None
        target_source = ""
        if self.detect_board:
            view_pose = self._tcp_pose_from_observation(get_observation)
            short_edge_scan_done = False
            detected_board = False
            if task_family == "sc_to_sc":
                self.get_logger().info(
                    "MagentaSquare detection route selected: SC-to-SC high aerial "
                    "circular magenta-guided board search; bypassing spawn magenta gate."
                )
                if self.move_to_view_first:
                    view_pose = self._move_to_view_pose(get_observation, move_robot)
                else:
                    self.get_logger().info(
                        "MagentaSquare movement method: high aerial view move disabled; "
                        "running SC circular magenta-guided search from current pose."
                    )

                target_xyz, target_source = self._detect_target_from_circular_view_search(
                    task,
                    get_observation,
                    move_robot,
                    view_pose,
                )
                detected_board = target_xyz is not None
            else:
                spawn_obs = get_observation()
                spawn_magenta_observations = self._spawn_magenta_observations(spawn_obs)
                if not spawn_magenta_observations:
                    self.get_logger().info(
                        "MagentaSquare detection route selected: spawn magenta gate failed; "
                        "running capped -y short-edge scan before aerial circular search."
                    )
                    short_edge_found = self._move_to_short_edge_view(
                        get_observation,
                        move_robot,
                        max_scan_distance_m=self.sfp_nic_short_edge_scan_max_distance_m,
                        return_to_start_on_failure=True,
                    )
                    short_edge_scan_done = True
                    if short_edge_found:
                        obs = get_observation()
                        target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
                        detected_board = target_xyz is not None
                        if detected_board:
                            target_source = "multi-camera short-edge fit"
                            self.get_logger().info(
                                "MagentaSquare detection selected: short-edge fit after "
                                "spawn-gated capped -y scan."
                            )
                    if not detected_board:
                        self.get_logger().info(
                            "MagentaSquare capped -y short-edge scan did not localize the "
                            "board; moving to aerial view for circular search."
                        )
                        view_pose = self._move_to_view_pose(get_observation, move_robot)
                        target_xyz, target_source = (
                            self._detect_target_from_circular_view_search(
                                task,
                                get_observation,
                                move_robot,
                                view_pose,
                            )
                        )
                        detected_board = target_xyz is not None
                else:
                    self.get_logger().info(
                        "MagentaSquare detection route selected: spawn magenta gate passed; "
                        "using aerial magenta-guided board search: "
                        f"cameras={[observation.camera_name for observation in spawn_magenta_observations]}"
                    )
                    if self.move_to_view_first:
                        view_pose = self._move_to_view_pose(get_observation, move_robot)
                    else:
                        self.get_logger().info(
                            "MagentaSquare movement method: high aerial view move disabled; "
                            "running magenta-guided search from current pose."
                        )

                    target_xyz, target_source = self._detect_target_from_linear_view_search(
                        task,
                        get_observation,
                        move_robot,
                        view_pose,
                    )
                    detected_board = target_xyz is not None

            if not detected_board:
                send_feedback(
                    "Board detection failed; trying one final short-edge board fit "
                    "from the current view."
                )
                if not short_edge_scan_done:
                    self.get_logger().info(
                        "MagentaSquare detection fallback route: running -y short-edge scan "
                        "before final board fit."
                    )
                    short_edge_found = self._move_to_short_edge_view(
                        get_observation,
                        move_robot,
                        max_scan_distance_m=(
                            self.sfp_nic_short_edge_scan_max_distance_m
                            if task_family == "sfp_to_nic"
                            else None
                        ),
                        return_to_start_on_failure=task_family == "sfp_to_nic",
                    )
                    short_edge_scan_done = True
                    if task_family == "sfp_to_nic" and not short_edge_found:
                        self.get_logger().info(
                            "MagentaSquare final capped -y short-edge scan did not "
                            "find a clear edge; moving to aerial view for circular search."
                        )
                        view_pose = self._move_to_view_pose(get_observation, move_robot)
                        target_xyz, target_source = (
                            self._detect_target_from_circular_view_search(
                                task,
                                get_observation,
                                move_robot,
                                view_pose,
                            )
                        )
                        detected_board = target_xyz is not None
                obs = get_observation()
                if not detected_board:
                    target_xyz = self._multicamera_short_edge_target_xyz(task, obs)
                    detected_board = target_xyz is not None
                if detected_board and not target_source:
                    target_source = "multi-camera short-edge fit"
                    self.get_logger().info(
                        "MagentaSquare detection selected: final short-edge fallback."
                    )

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
        self.get_logger().info(
            "MagentaSquare final localization choice: "
            f"source={target_source}, target_xyz={np.round(np.asarray(target_xyz), 5).tolist()}"
        )
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
        localized_target = self._last_localized_target
        if (
            localized_target is not None
            and np.linalg.norm(
                np.asarray(localized_target.target_xyz, dtype=np.float64)
                - np.asarray(target_xyz, dtype=np.float64)
            )
            > 1e-4
        ):
            self.get_logger().warn(
                "MagentaSquare localized board target and selected XYZ disagree; "
                "skipping SFP/NIC close localizer for this run."
            )
            localized_target = None

        if (
            task_family == "sfp_to_nic"
            and self.sfp_nic_insertion_enabled
            and localized_target is not None
        ):
            self.get_logger().info(
                "MagentaSquare movement stage: SFP/NIC entrance localization, "
                "port-axis alignment, and axis descent."
            )
            send_feedback("Localizing SFP port entrance and aligning to port axis.")
            target_pose = self._run_sfp_nic_insertion(
                task,
                get_observation,
                move_robot,
                localized_target,
                self._tcp_pose_from_observation(get_observation),
            )
        else:
            if (
                task_family == "sfp_to_nic"
                and self.sfp_nic_insertion_enabled
                and localized_target is None
            ):
                self.get_logger().warn(
                    "SFP/NIC close insertion path unavailable because board pose details "
                    "were not retained; using legacy base-z descent."
                )
            if self.tilt_tip_down_after_transport:
                self.get_logger().info(
                    "MagentaSquare movement stage: tilt TCP to align estimated cable tip downward."
                )
                send_feedback("Tilting TCP to align cable tip with downward insertion axis.")
                target_pose = self._move_to_tip_down_pose(
                    move_robot,
                    self._tcp_pose_from_observation(get_observation),
                )
            else:
                self.get_logger().info(
                    "MagentaSquare movement stage skipped: tip-down tilt disabled."
                )
            if self.descend_after_transport:
                self.get_logger().info(
                    "MagentaSquare movement stage: descend in base -z for insertion."
                )
                send_feedback("Descending TCP for insertion.")
                target_pose = self._descend_until_inserted(
                    task,
                    move_robot,
                    self._tcp_pose_from_observation(get_observation),
                )
            else:
                self.get_logger().info("MagentaSquare movement stage skipped: descent disabled.")
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
