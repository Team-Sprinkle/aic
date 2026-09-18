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
        self.execution_frame = os.environ.get("AIC_CORRECTIVE_EXECUTION_FRAME", "gripper/tcp")
        self.sc_bias_limit_m = float(os.environ.get("AIC_CORRECTIVE_SC_BIAS_LIMIT_M", ".045"))
        if not np.isfinite(self.sc_bias_limit_m) or not 0 <= self.sc_bias_limit_m <= .07:
            raise ValueError("SC expert lateral compensation limit must be in [0, .07] meters")
        if not 0 <= self.perturbation_scale <= 1 or self.execution_frame not in {"base_link", "gripper/tcp"}:
            raise ValueError("Invalid expert-collection perturbation scale or execution frame")
        self.episode_counter = 0
        self.rng = np.random.default_rng(self.seed)
        self.student_probability = float(os.environ.get("AIC_CORRECTIVE_STUDENT_PROBABILITY", "0"))
        self.student = None
        self.student_lineage = None
        if not 0 <= self.student_probability <= 1:
            raise ValueError("Student intervention probability must be in [0, 1]")
        if self.student_probability:
            if self.execution_frame != "base_link":
                raise ValueError("Student correction collection requires absolute base_link execution")
            from .RunACTTorchScript import RunACTTorchScript
            self.student = RunACTTorchScript(parent_node)
            if self.student.action_representation != "absolute_pose":
                raise ValueError("Student correction collection requires an absolute-pose ACT")
            self.student_lineage = {"torchscript": str(self.student.torchscript_path),
                "sha256": hashlib.sha256(self.student.torchscript_path.read_bytes()).hexdigest(),
                "probability_per_cycle": self.student_probability,
                "cycle_sec": 4., "student_window_sec": 2.5,
                "active_nominal_time_sec": [3., 35.],
                "max_teacher_disagreement_m": .03, "max_teacher_disagreement_rad": .08}
        self.get_logger().warn("PRIVILEGED CORRECTIVE DATA COLLECTION: exclude all scores from learned-policy evaluation")

    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        self.episode_counter += 1
        self.episode_dir = self.data_root / f"episode_{self.episode_counter:04d}_{time.time_ns()}"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.frames_log = (self.episode_dir / "frames.jsonl").open("w")
        self.observe = get_observation
        self.first_time = self.last_recorded_time = None
        self.frame_count = 0
        self.command_count = 0
        self.student_cycle = None
        self.student_cycle_selected = False
        self.student_frames = 0
        self._sc_alignment_gate_passed = False
        self._sc_last_z_offset = None
        if self.student is not None:
            self.student.reset_for_task(task)
        self.image_executor = ThreadPoolExecutor(max_workers=2)
        self.pending_images = deque()
        self.cycle = None
        self.offset = np.zeros(5)
        result = None
        try:
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
                        "timestamp_source": "center camera simulation timestamp; not frame_index/fps",
                        "label": "Clean teacher absolute target, converted to full TCP-relative command using this recorded observation",
                        "executed_action": ("Intermittent bounded ACT target with clean expert correction labels" if self.student else
                                            "Teacher target plus bounded intermittent XY and rotation perturbations"),
                        "student_correction": self.student_lineage, "student_recorded_frames": self.student_frames,
                        "execution_frame": self.execution_frame,
                        "teacher_geometry": ("full_rigid_tcp_to_tip_offset" if task.target_module_name.startswith("sc_port_")
                                             else "legacy_cheatcode_geometry"),
                        "sc_teacher_control": ({"alignment_gate_height_m": .05, "gate_lateral_error_m": .0015,
                                                "gate_rotation_error_rad": .03, "descent_pause_error_m": .002,
                                                "integral_bias_limit_m": self.sc_bias_limit_m, "max_extra_seating_steps": 400,
                                                "extra_seating_step_m": .00005, "min_tip_target_height_m": -.03,
                                                "alignment_gate_passed": self._sc_alignment_gate_passed}
                                               if task.target_module_name.startswith("sc_port_") else None),
                        "perturbation_config": {"xy_std_m": .004, "xy_clip_m": .008, "rotation_std_rad": .025,
                                                "rotation_clip_rad": .05, "period_sec": 3., "pulse_sec": .6,
                                                "active_window_sec": [4., 20.], "schedule": "command_index * 0.05",
                                                "scale": self.perturbation_scale},
                        "policy_returned": result, "official_success": None,
                        "acceptance": "Require separate official Tier 3 = 75 score before training"}
            (self.episode_dir / "episode.json").write_text(json.dumps(metadata, indent=2) + "\n")

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
        if self._sc_last_z_offset is not None and self._sc_alignment_gate_passed:
            # Advance at most 7 mm/s at the 20 Hz expert command rate; pause
            # descent when cable forces push the tip out of alignment.
            step = .00035 if lateral_error < .002 else 0.
            z_offset = max(z_offset, self._sc_last_z_offset - step)
        self._sc_last_z_offset = z_offset
        old_limit = self._max_integrator_windup
        self._max_integrator_windup = self.sc_bias_limit_m / .15
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
    def _save_camera(msg, path):
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding.lower())
        if channels is None:
            raise ValueError(f"Unsupported corrective-data camera encoding: {msg.encoding}")
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        array = array[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
        # OpenCV's file writer expects BGR; the resulting JPEG decodes as RGB.
        if msg.encoding.lower().startswith("rgb"):
            array = array[..., ::-1]
        array = cv2.resize(array, (288, 256), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(path), array, [cv2.IMWRITE_JPEG_QUALITY, 95]):
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
        elapsed = now - self.first_time
        nominal_time = self.command_count * .05
        self.command_count += 1
        cycle = int(nominal_time // 3.)
        if cycle != self.cycle:
            self.cycle = cycle
            self.offset = np.concatenate([np.clip(self.rng.normal(0, .004, 2), -.008, .008),
                                          np.clip(self.rng.normal(0, .025, 3), -.05, .05)])
        phase = nominal_time % 3.
        envelope = math.sin(math.pi * phase / .6) if 4. <= nominal_time <= 20. and phase < .6 else 0.
        perturbation = self.offset * envelope * self.perturbation_scale
        executed = copy.deepcopy(target_pose)
        executed.position.x += float(perturbation[0])
        executed.position.y += float(perturbation[1])
        quat = (Rotation.from_quat(self._pose_values(target_pose)[3:]) * Rotation.from_rotvec(perturbation[2:])).as_quat()
        executed.orientation.x, executed.orientation.y, executed.orientation.z, executed.orientation.w = map(float, quat)

        student_active = False
        student_target = None
        if self.student is not None:
            student_cycle = int(nominal_time // 4.)
            if student_cycle != self.student_cycle:
                self.student_cycle = student_cycle
                self.student_cycle_selected = self.rng.random() < self.student_probability
            student_active = self.student_cycle_selected and 3. <= nominal_time <= 35. and nominal_time % 4. < 2.5
            if student_active:
                # The student sees only ordinary observations. Privileged expert
                # targets supply labels and bound the collection intervention.
                raw = self.student.select_delta_action(observation)
                if not np.isfinite(raw).all():
                    raise RuntimeError("Nonfinite ACT target in corrective collection")
                student_target = raw.tolist()
                teacher_values = np.asarray(self._pose_values(target_pose))
                position_delta = raw[:3] - teacher_values[:3]
                position_delta *= min(1., .03 / max(np.linalg.norm(position_delta), 1e-12))
                teacher_rotation = Rotation.from_quat(teacher_values[3:])
                rotation_delta = (teacher_rotation.inv() * Rotation.from_rotvec(raw[3:6])).as_rotvec()
                rotation_delta *= min(1., .08 / max(np.linalg.norm(rotation_delta), 1e-12))
                executed.position.x, executed.position.y, executed.position.z = map(float, teacher_values[:3] + position_delta)
                q = (teacher_rotation * Rotation.from_rotvec(rotation_delta)).as_quat()
                executed.orientation.x, executed.orientation.y, executed.orientation.z, executed.orientation.w = map(float, q)
            else:
                # Do not reuse future actions predicted before an expert segment.
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
            for camera in ("center", "left", "right"):
                name = f"{camera}_{self.frame_count:06d}.jpg"
                self.pending_images.append(self.image_executor.submit(
                    self._save_camera, getattr(observation, camera + "_image"), self.episode_dir / name))
                images[camera] = name
            while self.pending_images and (self.pending_images[0].done() or len(self.pending_images) >= 24):
                self.pending_images.popleft().result()
            row = {"frame": self.frame_count, "sim_time": now, "elapsed_sim_time": elapsed,
                   "wall_time": time.time(), "images": images, "state": state.tolist(), "action": action,
                   "nominal_expert_time": nominal_time, "command_index": self.command_count - 1,
                   "teacher_target_pose": self._pose_values(target_pose), "executed_target_pose": self._pose_values(executed),
                   "perturbation_xy_rotvec": applied_delta[[0, 1, 3, 4, 5]].tolist(),
                   "perturbation_xyz_rotvec": applied_delta.tolist(),
                   "student_active": bool(student_active), "student_absolute_action": student_target}
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
