"""Privileged training-data collection with physically perturbed observations.

This is an expert data collector, not a learned policy evaluation. Images and
labels use the observation's actual simulation stamp. Store the clean teacher
target while applying occasional bounded perturbations to the executed target.
Accept/reject complete episodes afterward using the official insertion score.
"""
from __future__ import annotations

import copy
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from aic_model.policy import compute_delta_pose, quaternion_xyzw_to_rotation_vector
from .CheatCode import CheatCode

PACKAGE = Path(__file__).resolve().parents[3] / "aic_utils/lerobot_robot_aic"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))
from lerobot_robot_aic.runtime_features import base_state_from_ros_observation


class CollectCorrectiveCheatCode(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        cv2.setNumThreads(1)
        self.data_root = Path(os.environ["AIC_CORRECTIVE_DATA_DIR"])
        self.seed = int(os.environ.get("AIC_CORRECTIVE_SEED", "202609171"))
        self.perturbation_scale = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_SCALE", "1"))
        self.perturb_start_sec = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_START_SEC", "4"))
        self.perturb_end_sec = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_END_SEC", "20"))
        self.perturb_pulse_sec = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_PULSE_SEC", ".6"))
        self.perturb_xy_std_m = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_XY_STD_M", ".004"))
        self.perturb_xy_clip_m = float(os.environ.get("AIC_CORRECTIVE_PERTURBATION_XY_CLIP_M", ".008"))
        self.execution_frame = os.environ.get("AIC_CORRECTIVE_EXECUTION_FRAME", "gripper/tcp")
        self.sc_bias_limit_m = float(os.environ.get("AIC_CORRECTIVE_SC_BIAS_LIMIT_M", ".045"))
        self.sc_near_bias_limit_m = float(os.environ.get(
            "AIC_CORRECTIVE_SC_NEAR_BIAS_LIMIT_M", str(self.sc_bias_limit_m)))
        self.sc_preroute_lift_m = float(os.environ.get("AIC_CORRECTIVE_SC_PREROUTE_LIFT_M", "0"))
        self.sc_bypass_side = os.environ.get("AIC_CORRECTIVE_SC_BYPASS_SIDE", "off")
        self.sc_bypass_route = os.environ.get("AIC_CORRECTIVE_SC_BYPASS_ROUTE", "legacy")
        self.sc_bypass_behind_extra_m = float(os.environ.get(
            "AIC_CORRECTIVE_SC_BYPASS_BEHIND_EXTRA_M", "0"))
        self.sc_bypass_forward_extra_m = float(os.environ.get(
            "AIC_CORRECTIVE_SC_BYPASS_FORWARD_EXTRA_M", "0"))
        self.sc_bypass_lane_margin_m = float(os.environ.get(
            "AIC_CORRECTIVE_SC_BYPASS_LANE_MARGIN_M", ".085"))
        if self.sc_bypass_side not in {"off", "left", "right"}:
            raise ValueError("SC bypass side must be off, left, or right")
        if self.sc_bypass_route not in {"legacy", "clear_return", "low_cross_return"}:
            raise ValueError("SC bypass route must be legacy, clear_return, or low_cross_return")
        if not .06 <= self.sc_bypass_lane_margin_m <= .14:
            raise ValueError("SC bypass lane margin must be 0.06–0.14 m")
        if not -.03 <= self.sc_bypass_forward_extra_m <= .10:
            raise ValueError("SC bypass forward extra must be in [-.03, .10] m")
        if abs(self.sc_bypass_behind_extra_m) > .03:
            raise ValueError("SC bypass behind extra must be within 30 mm")
        if self.sc_bypass_side != "off" and self.sc_preroute_lift_m:
            raise ValueError("SC lateral bypass and pre-route lift are separate probes")
        self.sc_bypass_plan = None
        if not np.isfinite(self.sc_preroute_lift_m) or not 0 <= self.sc_preroute_lift_m <= .12:
            raise ValueError("AIC_CORRECTIVE_SC_PREROUTE_LIFT_M must be in [0, .12] meters")
        self.native_image_every_n_frames = int(os.environ.get("AIC_CORRECTIVE_NATIVE_IMAGE_EVERY_N_FRAMES", "0"))
        if self.native_image_every_n_frames < 0:
            raise ValueError("AIC_CORRECTIVE_NATIVE_IMAGE_EVERY_N_FRAMES must be nonnegative")
        if not np.isfinite(self.sc_bias_limit_m) or not 0 <= self.sc_bias_limit_m <= .07:
            raise ValueError("SC expert lateral compensation limit must be in [0, .07] meters")
        if not np.isfinite(self.sc_near_bias_limit_m) or not 0 <= self.sc_near_bias_limit_m <= self.sc_bias_limit_m:
            raise ValueError("SC near-port compensation limit must be in [0, SC bias limit]")
        if not 0 <= self.perturbation_scale <= 1 or self.execution_frame not in {"base_link", "gripper/tcp"}:
            raise ValueError("Invalid expert-collection perturbation scale or execution frame")
        if not (0 <= self.perturb_start_sec < self.perturb_end_sec <= 180 and
                0 < self.perturb_pulse_sec <= 3 and
                0 <= self.perturb_xy_std_m <= self.perturb_xy_clip_m <= .012):
            raise ValueError("Invalid bounded expert-collection perturbation schedule")
        self.episode_counter = 0
        self.rng = np.random.default_rng(self.seed)
        self.student_probability = float(os.environ.get("AIC_CORRECTIVE_STUDENT_PROBABILITY", "0"))
        allowed_episodes = os.environ.get("AIC_CORRECTIVE_STUDENT_EPISODES", "")
        self.student_episodes = ({int(x) for x in allowed_episodes.split(",") if x.strip()}
                                 if allowed_episodes else None)
        if self.student_episodes is not None and (not self.student_episodes or min(self.student_episodes) < 1):
            raise ValueError("Student episode indices must be positive")
        self.student_kind = os.environ.get("AIC_CORRECTIVE_STUDENT_KIND", "act")
        self.student_position_clip_m = float(os.environ.get(
            "AIC_CORRECTIVE_STUDENT_POSITION_CLIP_M", ".03"))
        self.student_rotation_clip_rad = float(os.environ.get(
            "AIC_CORRECTIVE_STUDENT_ROTATION_CLIP_RAD", ".08"))
        self.student_unbounded_takeover = os.environ.get(
            'AIC_CORRECTIVE_STUDENT_UNBOUNDED_TAKEOVER', '0') == '1'
        self.student_start_sec = float(os.environ.get("AIC_CORRECTIVE_STUDENT_START_SEC", "3"))
        self.student_end_sec = float(os.environ.get("AIC_CORRECTIVE_STUDENT_END_SEC", "35"))
        self.student_cycle_sec = float(os.environ.get("AIC_CORRECTIVE_STUDENT_CYCLE_SEC", "4"))
        self.student_window_sec = float(os.environ.get("AIC_CORRECTIVE_STUDENT_WINDOW_SEC", "2.5"))
        self.student = None
        self.student_lineage = None
        if not 0 <= self.student_probability <= 1:
            raise ValueError("Student intervention probability must be in [0, 1]")
        if (self.student_kind not in {"act", "shared_port_tcp"} or
                not 0 < self.student_position_clip_m <= .03 or
                not 0 < self.student_rotation_clip_rad <= .08):
            raise ValueError("Invalid bounded student intervention settings")
        if not (0 <= self.student_start_sec < self.student_end_sec <= 180 and
                0 < self.student_window_sec <= self.student_cycle_sec <= 180):
            raise ValueError("Invalid bounded student intervention schedule")
        if self.student_probability:
            if self.execution_frame != "base_link":
                raise ValueError("Student correction collection requires absolute base_link execution")
            if self.student_kind == "act":
                from .RunACTTorchScript import RunACTTorchScript
                self.student = RunACTTorchScript(parent_node)
                if self.student.action_representation != "absolute_pose":
                    raise ValueError("Student correction collection requires an absolute-pose ACT")
                student_path = self.student.torchscript_path
            else:
                from .RunSharedPortTCP import RunSharedPortTCP
                student_path = Path(os.environ["AIC_CORRECTIVE_STUDENT_CHECKPOINT"])
                self.student = RunSharedPortTCP(parent_node, checkpoint_path=student_path)
                student_cap = os.environ.get('AIC_CORRECTIVE_STUDENT_ACTION_CAP_M', '')
                if student_cap:
                    self.student.max_translation = float(student_cap)
                    if self.student.max_translation <= 0:
                        raise ValueError('Shared student action cap must be positive')
            self.student_lineage = {"kind": self.student_kind,
                "checkpoint": str(student_path),
                "sha256": hashlib.sha256(student_path.read_bytes()).hexdigest(),
                "probability_per_cycle": self.student_probability,
                "cycle_sec": self.student_cycle_sec,
                "student_window_sec": self.student_window_sec,
                "student_action_cap_m": (self.student.max_translation
                                         if self.student_kind == 'shared_port_tcp' else None),
                "unbounded_student_takeover": self.student_unbounded_takeover,
                "active_nominal_time_sec": [self.student_start_sec, self.student_end_sec],
                "allowed_episode_indices": sorted(self.student_episodes) if self.student_episodes else None,
                "max_teacher_disagreement_m": self.student_position_clip_m,
                "max_teacher_disagreement_rad": self.student_rotation_clip_rad}
        self.get_logger().warn("PRIVILEGED CORRECTIVE DATA COLLECTION: exclude all scores from learned-policy evaluation")

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.episode_counter += 1
        self._current_task_is_sc = task.target_module_name.startswith("sc_port_")
        self.episode_dir = self.data_root / f"episode_{self.episode_counter:04d}_{time.time_ns()}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.frames_log = (self.episode_dir / "frames.jsonl").open("w")
        self.observe = get_observation
        self.first_time = self.last_recorded_time = None
        self.frame_count = 0
        self.native_frame_count = 0
        self.native_image_shape_hwc = None
        self.command_count = 0
        self.student_cycle = None
        self.student_cycle_selected = False
        self.student_frames = 0
        self._sc_alignment_gate_passed = False
        self._sc_last_z_offset = None
        self._sc_near_bias_active = False
        self.sc_bypass_plan = None
        if self.student is not None:
            self.student.reset_for_task(task)
        self.image_executor = ThreadPoolExecutor(max_workers=2)
        self.pending_images = deque()
        self.cycle = None
        self.offset = np.zeros(5)
        result = None
        try:
            if self._current_task_is_sc and self.sc_bypass_side != "off":
                self._run_sc_lateral_bypass(task, move_robot)
            if self._current_task_is_sc and self.sc_preroute_lift_m:
                # Optional scored-data route probe: lift the grasped connector
                # before the standard straight interpolation across the card
                # row. It is disabled by default and uses only observed TCP.
                from geometry_msgs.msg import Point, Pose, Quaternion
                from rclpy.time import Time
                self._task = task
                if not self._wait_for_tf("base_link", "gripper/tcp"):
                    raise RuntimeError("No TCP TF for SC pre-route lift")
                initial = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", "gripper/tcp", Time()).transform
                steps = 100
                for step in range(steps + 1):
                    fraction = step / steps
                    blend = fraction * fraction * (3. - 2. * fraction)
                    target = Pose(
                        position=Point(x=initial.translation.x,
                                       y=initial.translation.y,
                                       z=initial.translation.z + self.sc_preroute_lift_m * blend),
                        orientation=Quaternion(x=initial.rotation.x, y=initial.rotation.y,
                                               z=initial.rotation.z, w=initial.rotation.w),
                    )
                    self._send_delta_pose_target(move_robot, target)
                    self.sleep_for(.05)
                self.get_logger().info(f"SC pre-route lift target: {self.sc_preroute_lift_m:.3f} m")
            result = super().insert_cable(task, get_observation, move_robot, send_feedback)
            if (result and task.target_module_name.startswith("sc_port_")
                    and self._sc_alignment_gate_passed and not self._task_completed_in_simulation(task)):
                from rclpy.time import Time
                port = self._parent_node._tf_buffer.lookup_transform(
                    "base_link", f"task_board/{task.target_module_name}/{task.port_name}_link", Time()).transform
                self.get_logger().info("SC final seating: bounded extra depth, guarded by lateral alignment")
                for _ in range(400):
                    if self._task_completed_in_simulation(task):
                        break
                    desired_offset = max(-.03, self._sc_last_z_offset - .00005)
                    self._send_delta_pose_target(move_robot, self.calc_gripper_pose(port, z_offset=desired_offset))
                    self.sleep_for(.05)
            return result
        finally:
            self.frames_log.close()
            self.image_executor.shutdown(wait=True)
            for future in self.pending_images:
                future.result()
            metadata = {"kind": "privileged_aligned_demonstration" if self.perturbation_scale == 0 and self.student is None else "privileged_corrective_demonstration", "frames": self.frame_count,
                        "seed": self.seed, "episode_counter": self.episode_counter,
                        "task": {key: str(getattr(task, key)) for key in
                                 ("id", "cable_name", "plug_name", "port_name", "target_module_name")},
                        "image_channel_order": "rgb", "image_shape_hwc": [256, 288, 3],
                        "native_image_every_n_frames": self.native_image_every_n_frames,
                        "native_image_shape_hwc": self.native_image_shape_hwc,
                        "native_recorded_frames": self.native_frame_count,
                        "timestamp_source": "center camera simulation timestamp; not frame_index/fps",
                        "label": "Clean teacher absolute target, converted to full TCP-relative command using this recorded observation",
                        "executed_action": ("Intermittent bounded student target with teacher correction labels" if self.student else
                                            "Teacher target plus bounded intermittent XY and rotation perturbations"),
                        "student_correction": self.student_lineage, "student_recorded_frames": self.student_frames,
                        "execution_frame": self.execution_frame,
                        "teacher_geometry": ("full_rigid_tcp_to_tip_offset" if task.target_module_name.startswith("sc_port_")
                                             else "legacy_cheatcode_geometry"),
                        "sc_teacher_control": ({"alignment_gate_height_m": .05, "gate_lateral_error_m": .0015,
                                                "gate_rotation_error_rad": .03, "descent_pause_error_m": .002,
                                                "preroute_lift_m": self.sc_preroute_lift_m,
                                                "lateral_bypass": self.sc_bypass_plan,
                                                "integral_bias_limit_m": self.sc_bias_limit_m, "max_extra_seating_steps": 400,
                                                "near_port_integral_bias_limit_m": self.sc_near_bias_limit_m,
                                                "extra_seating_step_m": .00005, "min_tip_target_height_m": -.03,
                                                "alignment_gate_passed": self._sc_alignment_gate_passed}
                                               if task.target_module_name.startswith("sc_port_") else None),
                        "perturbation_config": {"xy_std_m": self.perturb_xy_std_m, "xy_clip_m": self.perturb_xy_clip_m, "rotation_std_rad": .025,
                                                "rotation_clip_rad": .05, "period_sec": 3.,
                                                "active_window_sec": [self.perturb_start_sec, self.perturb_end_sec],
                                                "pulse_sec": self.perturb_pulse_sec,
                                                "schedule": "command_index * 0.05",
                                                "scale": self.perturbation_scale},
                        "policy_returned": result, "official_success": None,
                        "acceptance": "Require separate official Tier 3 = 75 score before training"}
            (self.episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def _run_sc_lateral_bypass(self, task, move_robot):
        """Privileged collection route around the card row, before insertion.

        Go behind the row on the starting side first, cross behind it to an
        outside lane, traverse beside the cards, then sweep toward the port
        only after passing the row. The waypoint phase is recorded through
        the normal causal image/action collector.
        """
        from geometry_msgs.msg import Point, Pose
        from rclpy.time import Time

        self._task = task
        buffer = self._parent_node._tf_buffer
        if not self._wait_for_tf("base_link", "gripper/tcp"):
            raise RuntimeError("No TCP TF for SC lateral bypass")
        def xyz(frame):
            transform = buffer.lookup_transform("base_link", frame, Time()).transform
            return np.array([transform.translation.x, transform.translation.y,
                             transform.translation.z], dtype=float)
        tcp_tf = buffer.lookup_transform("base_link", "gripper/tcp", Time()).transform
        start = xyz("gripper/tcp")
        cards = []
        for index in range(5):
            frame = f"task_board/nic_card_mount_{index}/nic_card_link"
            if self._wait_for_tf("base_link", frame, timeout_sec=1.):
                cards.append(xyz(frame))
        if len(cards) < 2:
            raise RuntimeError(f"SC lateral bypass requires >=2 card TFs, got {len(cards)}")
        cards = np.stack(cards)
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        if not self._wait_for_tf("base_link", port_frame):
            raise RuntimeError("No target port TF for SC lateral bypass")
        port = xyz(port_frame)
        sign = 1. if port[1] > np.median(cards[:, 1]) else -1.
        clear_route = self.sc_bypass_route in {"clear_return", "low_cross_return"}
        behind_offset = (.125 if clear_route else .045) + self.sc_bypass_behind_extra_m
        beyond_offset = (.125 if clear_route else .055) + self.sc_bypass_forward_extra_m
        behind_y = float(cards[:, 1].min() - behind_offset if sign > 0 else cards[:, 1].max() + behind_offset)
        beyond_y = float(cards[:, 1].max() + beyond_offset if sign > 0 else cards[:, 1].min() - beyond_offset)
        lane_margin = self.sc_bypass_lane_margin_m + (.005 if clear_route else 0.)
        lane_x = float(cards[:, 0].min() - lane_margin
                       if self.sc_bypass_side == "left" else
                       cards[:, 0].max() + lane_margin)
        high_z = max(float(start[2]) + (.02 if clear_route else 0.), float(cards[:, 2].max() + .21))
        lane_z = float(cards[:, 2].max() + .105)
        start_gap = float(np.min(np.abs(start[0] - cards[:, 0])))
        egress = []
        if start_gap < .055:
            # Move farther away on the starting side before traveling behind
            # the row. Never cross the card x corridor at its current y.
            if start[0] > float(cards[:, 0].max() + .02):
                start_side_x = float(cards[:, 0].max() + .085)
            elif start[0] < float(cards[:, 0].min() - .02):
                start_side_x = float(cards[:, 0].min() - .085)
            else:
                raise RuntimeError(f"TCP starts within card x corridor: {start_gap:.3f} m")
            egress = [("egress_away_from_cards", [start_side_x, start[1], high_z], 3.0)]
        else:
            start_side_x = float(start[0])
        waypoints = egress + [("clear_start_side", [start_side_x, behind_y, high_z], 3.0)]
        if self.sc_bypass_route == "low_cross_return":
            waypoints += [
                ("lower_on_start_side_behind_row", [start_side_x, behind_y, lane_z], 3.0),
                ("shift_behind_row_low", [lane_x, behind_y, lane_z], 3.5),
            ]
        else:
            waypoints += [
                ("shift_behind_row", [lane_x, behind_y, high_z], 3.5),
                ("lower_outside_lane", [lane_x, behind_y, lane_z], 3.0),
            ]
        waypoints += [
            ("traverse_outside_row", [lane_x, beyond_y, lane_z], 9.5 if clear_route else 8.0),
            ("sweep_after_row", [port[0], beyond_y, lane_z], 3.5),
        ]
        if clear_route:
            waypoints.append(("return_toward_port", [port[0], port[1], lane_z], 2.5))
        waypoints.append(("above_port", [port[0], port[1], port[2] + .20], 3.5))
        self.sc_bypass_plan = {"side": self.sc_bypass_side, "route": self.sc_bypass_route,
                               "lane_margin_m": self.sc_bypass_lane_margin_m,
                               "effective_lane_margin_m": lane_margin,
                               "behind_offset_m": behind_offset,
                               "beyond_offset_m": beyond_offset,
                               "behind_extra_m": self.sc_bypass_behind_extra_m,
                               "forward_extra_m": self.sc_bypass_forward_extra_m,
                               "start_tcp_base_m": start.tolist(),
                               "card_centers_base_m": cards.tolist(), "port_base_m": port.tolist(),
                               "start_card_x_gap_m": start_gap, "lane_x_m": lane_x,
                               "waypoints": [{"name": name, "tcp_base_m": list(map(float, point)),
                                              "duration_s": duration}
                                             for name, point, duration in waypoints]}
        (self.episode_dir / "lateral_bypass_plan.json").write_text(
            json.dumps(self.sc_bypass_plan, indent=2) + "\n")
        orientation = tcp_tf.rotation
        for name, target_xyz, duration in waypoints:
            self.get_logger().info(f"SC lateral bypass {name}: {target_xyz}")
            initial = xyz("gripper/tcp")
            target_xyz = np.asarray(target_xyz)
            steps = max(1, round(duration / .05))
            for step in range(steps + 1):
                u = step / steps
                smooth = 10*u**3 - 15*u**4 + 6*u**5
                point = initial + smooth * (target_xyz - initial)
                pose = Pose(position=Point(x=float(point[0]), y=float(point[1]), z=float(point[2])),
                            orientation=orientation)
                self._send_delta_pose_target(move_robot, pose)
                self.sleep_for(.05)
            if clear_route:
                for _ in range(10):
                    self._send_delta_pose_target(move_robot, pose)
                    self.sleep_for(.05)

    @staticmethod
    def _pose_values(pose):
        return [pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]

    @staticmethod
    def _tcp_position_for_tip_target(tcp_pose, tip_position, target_quaternion, target_tip_position):
        """Account for the lateral SC tip offset under the desired TCP rotation."""
        from scipy.spatial.transform import Rotation
        tcp_pose = np.asarray(tcp_pose)
        offset_local = Rotation.from_quat(tcp_pose[3:]).apply(
            np.asarray(tip_position) - tcp_pose[:3], inverse=True)
        return np.asarray(target_tip_position) - Rotation.from_quat(target_quaternion).apply(offset_local)

    def calc_gripper_pose(self, port_transform, slerp_fraction=1.0, position_fraction=1.0,
                          z_offset=0.1, reset_xy_integrator=False):
        if not self._task.target_module_name.startswith("sc_port_"):
            return super().calc_gripper_pose(port_transform, slerp_fraction, position_fraction,
                                              z_offset, reset_xy_integrator)
        from rclpy.time import Time
        from scipy.spatial.transform import Rotation
        buffer = self._parent_node._tf_buffer
        tcp = buffer.lookup_transform("base_link", "gripper/tcp", Time()).transform
        tip = buffer.lookup_transform("base_link", f"{self._task.cable_name}/{self._task.plug_name}_link", Time()).transform
        tcp_values = [tcp.translation.x, tcp.translation.y, tcp.translation.z,
                      tcp.rotation.x, tcp.rotation.y, tcp.rotation.z, tcp.rotation.w]
        tip_position = [tip.translation.x, tip.translation.y, tip.translation.z]
        lateral_error = np.linalg.norm(np.asarray(tip_position[:2]) - [port_transform.translation.x,
                                                                      port_transform.translation.y])
        actual_z = tip.translation.z - port_transform.translation.z
        tip_rotation = Rotation.from_quat([tip.rotation.x, tip.rotation.y, tip.rotation.z, tip.rotation.w])
        port_rotation = Rotation.from_quat([port_transform.rotation.x, port_transform.rotation.y,
                                           port_transform.rotation.z, port_transform.rotation.w])
        rotation_error = (tip_rotation.inv() * port_rotation).magnitude()
        if not self._sc_alignment_gate_passed and z_offset <= .05:
            if lateral_error < .0015 and rotation_error < .03 and abs(actual_z - .05) < .015:
                self._sc_alignment_gate_passed = True
                self.get_logger().info(f"SC alignment gate passed: lateral={lateral_error:.6f} rotation={rotation_error:.6f}")
            else:
                z_offset = max(z_offset, .05)
        if (self._sc_alignment_gate_passed and z_offset <= .05
                and self.sc_near_bias_limit_m < self.sc_bias_limit_m
                and not self._sc_near_bias_active):
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
            self._sc_near_bias_active = True
            self.get_logger().info("SC near-port integral bias reset for bounded seating probe")
        if self._sc_last_z_offset is not None and self._sc_alignment_gate_passed:
            # Advance at most 7 mm/s at the 20 Hz expert command rate; pause
            # descent when cable forces push the tip out of alignment.
            step = .00035 if lateral_error < .002 else 0.
            z_offset = max(z_offset, self._sc_last_z_offset - step)
        self._sc_last_z_offset = z_offset
        old_limit = self._max_integrator_windup
        self._max_integrator_windup = (self.sc_near_bias_limit_m if self._sc_near_bias_active
                                       else self.sc_bias_limit_m) / .15
        try:
            # SC cable forces create a persistent tracking error before contact.
            # Accumulate compensation during the full-pose approach as well.
            pose = super().calc_gripper_pose(port_transform, slerp_fraction, position_fraction,
                                            z_offset, position_fraction < 1. or slerp_fraction < 1.)
        finally:
            self._max_integrator_windup = old_limit
        target_tip = [port_transform.translation.x + .15 * self._tip_x_error_integrator,
                      port_transform.translation.y + .15 * self._tip_y_error_integrator,
                      port_transform.translation.z + z_offset]
        target_tcp = self._tcp_position_for_tip_target(tcp_values, tip_position,
                                                       self._pose_values(pose)[3:], target_tip)
        blended = position_fraction * target_tcp + (1.0 - position_fraction) * np.asarray(tcp_values[:3])
        pose.position.x, pose.position.y, pose.position.z = map(float, blended)
        return pose

    @staticmethod
    def _save_camera(msg, path, *, native=False):
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding.lower())
        if channels is None:
            raise ValueError(f"Unsupported corrective-data camera encoding: {msg.encoding}")
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        array = array[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
        # OpenCV's file writer expects BGR; the resulting JPEG decodes as RGB.
        if msg.encoding.lower().startswith("rgb"):
            array = array[..., ::-1]
        if not native:
            array = cv2.resize(array, (288, 256), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(path), array, [cv2.IMWRITE_JPEG_QUALITY, 90 if native else 95]):
            raise RuntimeError(f"Failed to write corrective image {path}")

    def _send_delta_pose_target(self, move_robot, target_pose):
        from scipy.spatial.transform import Rotation

        observation = self.observe()
        if observation is None:
            return self._execute_target(move_robot, target_pose)
        stamp = observation.center_image.header.stamp
        now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if self.first_time is None:
            self.first_time = now
            if self.student is not None:
                self.student._episode_first_image_time = now
                if hasattr(self.student, "first_sim_time"):
                    self.student.first_sim_time = now
        elapsed = now - self.first_time
        nominal_time = self.command_count * .05
        self.command_count += 1
        cycle = int(nominal_time // 3.)
        if cycle != self.cycle:
            self.cycle = cycle
            self.offset = np.concatenate([np.clip(self.rng.normal(0, self.perturb_xy_std_m, 2),
                                                  -self.perturb_xy_clip_m, self.perturb_xy_clip_m),
                                          np.clip(self.rng.normal(0, .025, 3), -.05, .05)])
        phase = nominal_time % 3.
        envelope = (math.sin(math.pi * phase / self.perturb_pulse_sec)
                    if self.perturb_start_sec <= nominal_time <= self.perturb_end_sec
                    and phase < self.perturb_pulse_sec else 0.)
        perturbation = self.offset * envelope * self.perturbation_scale
        executed = copy.deepcopy(target_pose)
        executed.position.x += float(perturbation[0])
        executed.position.y += float(perturbation[1])
        quat = (Rotation.from_quat(self._pose_values(target_pose)[3:]) * Rotation.from_rotvec(perturbation[2:])).as_quat()
        executed.orientation.x, executed.orientation.y, executed.orientation.z, executed.orientation.w = map(float, quat)

        student_active = False
        student_target = None
        if self.student is not None:
            student_cycle = int(nominal_time // self.student_cycle_sec)
            if student_cycle != self.student_cycle:
                self.student_cycle = student_cycle
                self.student_cycle_selected = self.rng.random() < self.student_probability
            student_active = (self.student_cycle_selected and
                              (self.student_episodes is None or self.episode_counter in self.student_episodes) and
                              self.student_start_sec <= nominal_time <= self.student_end_sec and
                              nominal_time % self.student_cycle_sec < self.student_window_sec)
            if student_active:
                # The student sees only ordinary observations. Privileged expert
                # targets supply labels and bound the collection intervention.
                raw = (self.student.select_delta_action(observation, nominal_time_s=nominal_time)
                       if self.student_kind == 'shared_port_tcp'
                       else self.student.select_delta_action(observation))
                if not np.isfinite(raw).all():
                    raise RuntimeError("Nonfinite ACT target in corrective collection")
                student_target = raw.tolist()
                teacher_values = np.asarray(self._pose_values(target_pose))
                if self.student_unbounded_takeover:
                    executed.position.x, executed.position.y, executed.position.z = map(float, raw[:3])
                    q = Rotation.from_rotvec(raw[3:6]).as_quat()
                else:
                    position_delta = raw[:3] - teacher_values[:3]
                    position_delta *= min(1., self.student_position_clip_m /
                                          max(np.linalg.norm(position_delta), 1e-12))
                    teacher_rotation = Rotation.from_quat(teacher_values[3:])
                    rotation_delta = (teacher_rotation.inv() * Rotation.from_rotvec(raw[3:6])).as_rotvec()
                    rotation_delta *= min(1., self.student_rotation_clip_rad /
                                          max(np.linalg.norm(rotation_delta), 1e-12))
                    executed.position.x, executed.position.y, executed.position.z = map(float, teacher_values[:3] + position_delta)
                    q = (teacher_rotation * Rotation.from_rotvec(rotation_delta)).as_quat()
                executed.orientation.x, executed.orientation.y, executed.orientation.z, executed.orientation.w = map(float, q)
            else:
                # Do not reuse future actions predicted before an expert segment.
                if hasattr(self.student, "clear_pending_actions"):
                    self.student.clear_pending_actions()
                else:
                    self.student._action_queue.clear()
                    self.student._ensemble_predictions.clear()

        teacher_values = np.asarray(self._pose_values(target_pose))
        executed_values = np.asarray(self._pose_values(executed))
        applied_delta = np.concatenate([executed_values[:3] - teacher_values[:3],
            (Rotation.from_quat(teacher_values[3:]).inv() * Rotation.from_quat(executed_values[3:])).as_rotvec()])

        if self.last_recorded_time is None or now - self.last_recorded_time >= .05 - 1e-6:
            delta = compute_delta_pose(observation.controller_state.tcp_pose, target_pose)
            action = self._pose_values(delta)[:3] + quaternion_xyzw_to_rotation_vector(np.asarray(self._pose_values(delta)[3:])).tolist()
            state = base_state_from_ros_observation(observation)
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise RuntimeError("Nonfinite corrective-data state or teacher label")
            images = {}
            native_images = {}
            save_native = (self.native_image_every_n_frames > 0
                           and self._current_task_is_sc
                           and self.frame_count % self.native_image_every_n_frames == 0)
            for camera in ("center", "left", "right"):
                name = f"{camera}_{self.frame_count:06d}.jpg"
                camera_msg = getattr(observation, camera + "_image")
                self.pending_images.append(self.image_executor.submit(
                    self._save_camera, camera_msg, self.episode_dir / name))
                images[camera] = name
                if save_native:
                    native_name = f"native_{camera}_{self.frame_count:06d}.jpg"
                    self.pending_images.append(self.image_executor.submit(
                        self._save_camera, camera_msg, self.episode_dir / native_name, native=True))
                    native_images[camera] = native_name
                    if self.native_image_shape_hwc is None:
                        self.native_image_shape_hwc = [int(camera_msg.height), int(camera_msg.width), 3]
            self.native_frame_count += int(save_native)
            while self.pending_images and (self.pending_images[0].done() or len(self.pending_images) >= 24):
                self.pending_images.popleft().result()
            row = {"frame": self.frame_count, "sim_time": now, "elapsed_sim_time": elapsed,
                   "wall_time": time.time(), "images": images, "state": state.tolist(), "action": action,
                   "nominal_expert_time": nominal_time, "command_index": self.command_count - 1,
                   "teacher_target_pose": self._pose_values(target_pose), "executed_target_pose": self._pose_values(executed),
                   "perturbation_xy_rotvec": applied_delta[[0, 1, 3, 4, 5]].tolist(),
                   "perturbation_xyz_rotvec": applied_delta.tolist(),
                   "student_active": bool(student_active), "student_absolute_action": student_target}
            if native_images:
                row["native_images"] = native_images
            self.frames_log.write(json.dumps(row) + "\n")
            self.frames_log.flush()
            self.frame_count += 1
            self.student_frames += int(student_active)
            self.last_recorded_time = now
        return self._execute_target(move_robot, executed)

    def _execute_target(self, move_robot, pose):
        if self.execution_frame == "base_link":
            return self.set_pose_target(move_robot=move_robot, pose=pose, frame_id="base_link")
        return super()._send_delta_pose_target(move_robot, pose)
