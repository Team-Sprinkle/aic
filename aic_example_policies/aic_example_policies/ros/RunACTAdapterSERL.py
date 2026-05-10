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

"""Official-runtime policy wrapper for ACT-adapter SERL checkpoints.

This is the policy entry point for evaluating offline/Isaac/Gazebo ACT-adapter
SERL checkpoints through the original AIC toolkit flow:

  /entrypoint.sh ... start_aic_engine:=true
  pixi run ros2 run aic_model aic_model --ros-args \
    -p policy:=aic_example_policies.ros.RunACTAdapterSERL

Set:
  AIC_SERL_CHECKPOINT
  AIC_SERL_ACT_TORCHSCRIPT

The model receives the runtime state dimension expected by the checkpoint:
32D base observation, optionally plus 40D contact/force recovery features and
the canonical 10D task vector derived from the official Task message.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from geometry_msgs.msg import Twist, Vector3, Wrench
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
    clamp_delta_pose_components,
    pose_to_position_motion_update,
    build_pose_from_vectors,
    rotation_vector_to_quaternion_xyzw,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


REPO_ROOT = Path(__file__).resolve().parents[3]
GAZEBO_RL_PACKAGE_DIR = REPO_ROOT / "aic_utils" / "gazebo_rl"
if GAZEBO_RL_PACKAGE_DIR.exists() and str(GAZEBO_RL_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(GAZEBO_RL_PACKAGE_DIR))

from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy  # noqa: E402


DEFAULT_CONTROL_HZ = 20.0
DEFAULT_MAX_RUNTIME_SEC = 30.0


class RunACTAdapterSERL(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device(
            os.environ.get("AIC_SERL_DEVICE")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_path = self._required_path("AIC_SERL_CHECKPOINT")
        self.act_torchscript = self._required_path("AIC_SERL_ACT_TORCHSCRIPT")
        self.control_hz = float(os.environ.get("AIC_SERL_CONTROL_HZ", DEFAULT_CONTROL_HZ))
        self.max_runtime_sec = float(
            os.environ.get("AIC_SERL_MAX_RUNTIME_SEC", DEFAULT_MAX_RUNTIME_SEC)
        )
        self.start_delay_sec = float(os.environ.get("AIC_SERL_START_DELAY_SEC", 0.0))
        self.command_mode = os.environ.get("AIC_SERL_COMMAND_MODE", "delta_pose")
        self.command_frame = os.environ.get("AIC_SERL_COMMAND_FRAME", "gripper/tcp")
        self.max_translation_delta = float(os.environ.get("AIC_SERL_MAX_TRANSLATION_DELTA", 0.02))
        self.max_rotation_delta = float(os.environ.get("AIC_SERL_MAX_ROTATION_DELTA", 0.2))
        self.translation_deadband = float(os.environ.get("AIC_SERL_TRANSLATION_DEADBAND", 5e-4))
        self.rotation_deadband = float(os.environ.get("AIC_SERL_ROTATION_DEADBAND", 1e-3))
        self.adapter_delta_clip = self._optional_float("AIC_SERL_ADAPTER_DELTA_CLIP", 0.02)
        self.action_clip = self._optional_float("AIC_SERL_ACTION_CLIP", 0.05)
        self.allow_zero_images = self._bool_env("AIC_SERL_ALLOW_ZERO_IMAGES", False)

        self._current_task: Task | None = None
        self.policy = ACTAdapterSERLGazeboPolicy(
            self.checkpoint_path,
            act_torchscript=self.act_torchscript,
            device=str(self.device),
            allow_zero_images=self.allow_zero_images,
            adapter_delta_clip=self.adapter_delta_clip,
            action_clip=self.action_clip,
        )
        self.get_logger().info(
            "ACT-adapter SERL policy loaded from "
            f"{self.checkpoint_path} with ACT base {self.act_torchscript} on {self.device}; "
            f"state_dim={self.policy.state_dim}, action_dim={self.policy.action_dim}, "
            f"action_horizon={self.policy.action_horizon}, control_hz={self.control_hz}, "
            f"command_mode={self.command_mode}, command_frame={self.command_frame}, "
            f"adapter_delta_clip={self.adapter_delta_clip}, action_clip={self.action_clip}"
        )

    @staticmethod
    def _required_path(env_name: str) -> Path:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"{env_name} must point to a checkpoint/artifact path")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        return path

    @staticmethod
    def _optional_float(env_name: str, default: float | None) -> float | None:
        value = os.environ.get(env_name)
        if value is None or value == "":
            return default
        if value.lower() in {"none", "null"}:
            return None
        return float(value)

    @staticmethod
    def _bool_env(env_name: str, default: bool) -> bool:
        value = os.environ.get(env_name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _parse_index_from_suffix(value: str, prefix: str) -> int:
        if not value.startswith(prefix):
            raise ValueError(f"Expected {value!r} to start with {prefix!r}")
        return int(value.removeprefix(prefix))

    def _task_vector(self, task: Task) -> list[float]:
        if task.target_module_name.startswith("nic_card_mount_"):
            card_index = self._parse_index_from_suffix(task.target_module_name, "nic_card_mount_")
            port_index = self._parse_index_from_suffix(task.port_name, "sfp_port_")
            if not 0 <= card_index <= 4:
                raise ValueError(f"target card index must be in [0, 4], got {card_index}")
            if port_index not in (0, 1):
                raise ValueError(f"target port index must be 0 or 1, got {port_index}")
            card = [0.0] * 5
            card[card_index] = 1.0
            return [1.0, 0.0, float(port_index == 0), float(port_index == 1), *card, 1.0]
        if task.target_module_name.startswith("sc_port_"):
            port_index = self._parse_index_from_suffix(task.target_module_name, "sc_port_")
            if port_index not in (0, 1):
                raise ValueError(f"target SC port index must be 0 or 1, got {port_index}")
            return [0.0, 1.0, float(port_index == 0), float(port_index == 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raise ValueError(f"Cannot infer task vector from target_module_name={task.target_module_name!r}")

    @staticmethod
    def _image_msg_to_rgb(raw_img: Any) -> np.ndarray:
        image = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height,
            raw_img.width,
            -1,
        )
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def _observation_dict(self, obs_msg: Observation) -> dict[str, Any]:
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity
        if self._current_task is None:
            raise ValueError("No active task is set for SERL task-vector inference")
        return {
            "controller": {
                "current_tcp_pose": {
                    "position": [tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z],
                    "orientation_xyzw": [
                        tcp_pose.orientation.x,
                        tcp_pose.orientation.y,
                        tcp_pose.orientation.z,
                        tcp_pose.orientation.w,
                    ],
                },
                "tcp_velocity": {
                    "linear": [tcp_vel.linear.x, tcp_vel.linear.y, tcp_vel.linear.z],
                    "angular": [tcp_vel.angular.x, tcp_vel.angular.y, tcp_vel.angular.z],
                },
                "tcp_error": list(obs_msg.controller_state.tcp_error),
            },
            "joints": {"position": list(obs_msg.joint_states.position[:7])},
            "wrist_wrench": {
                "force": [
                    obs_msg.wrist_wrench.wrench.force.x,
                    obs_msg.wrist_wrench.wrench.force.y,
                    obs_msg.wrist_wrench.wrench.force.z,
                ],
                "torque": [
                    obs_msg.wrist_wrench.wrench.torque.x,
                    obs_msg.wrist_wrench.wrench.torque.y,
                    obs_msg.wrist_wrench.wrench.torque.z,
                ],
            },
            "images": {
                "observation.images.left_camera": self._image_msg_to_rgb(obs_msg.left_image),
                "observation.images.center_camera": self._image_msg_to_rgb(obs_msg.center_image),
                "observation.images.right_camera": self._image_msg_to_rgb(obs_msg.right_image),
            },
        }

    def select_delta_action(self, obs_msg: Observation) -> np.ndarray:
        if self._current_task is None:
            raise ValueError("No active task is set")
        self.policy.task_vector = torch.as_tensor(
            self._task_vector(self._current_task),
            dtype=torch.float32,
            device=self.policy.device,
        ).reshape(1, -1)
        self.policy.feature_assembler.task_vector = self.policy.task_vector.squeeze(0).detach().cpu().numpy()
        action = self.policy.act(self._observation_dict(obs_msg), explore=False)
        return np.asarray(action[:6], dtype=np.float32)

    def _send_velocity_target(
        self,
        move_robot: MoveRobotCallback,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> None:
        position, rotation = self._clamp_action(delta_position_xyz, delta_rotation_xyz)
        twist = Twist(
            linear=Vector3(
                x=float(position[0] * self.control_hz),
                y=float(position[1] * self.control_hz),
                z=float(position[2] * self.control_hz),
            ),
            angular=Vector3(
                x=float(rotation[0] * self.control_hz),
                y=float(rotation[1] * self.control_hz),
                z=float(rotation[2] * self.control_hz),
            ),
        )
        motion_update = MotionUpdate()
        motion_update.velocity = twist
        motion_update.header.frame_id = self.command_frame
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        motion_update.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        move_robot(motion_update=motion_update)

    def _send_delta_pose_target(
        self,
        move_robot: MoveRobotCallback,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> None:
        position, rotation = self._clamp_action(delta_position_xyz, delta_rotation_xyz)
        delta_pose = build_pose_from_vectors(
            position,
            rotation_vector_to_quaternion_xyzw(rotation),
        )
        motion_update = pose_to_position_motion_update(
            delta_pose,
            stamp=self.get_clock().now().to_msg(),
            frame_id=self.command_frame,
            stiffness=[90.0, 90.0, 90.0, 50.0, 50.0, 50.0],
            damping=[50.0, 50.0, 50.0, 20.0, 20.0, 20.0],
        )
        move_robot(motion_update=motion_update)

    def _clamp_action(
        self,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return clamp_delta_pose_components(
            delta_position_xyz,
            delta_rotation_xyz,
            max_translation=self.max_translation_delta,
            max_rotation=self.max_rotation_delta,
            deadband_translation=self.translation_deadband,
            deadband_rotation=self.rotation_deadband,
        )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self._current_task = task
        if self.policy.feature_assembler.uses_task_vector:
            self.policy.feature_assembler.reset(self._task_vector(task))
        else:
            self.policy.feature_assembler.reset()
        self.get_logger().info(f"RunACTAdapterSERL.insert_cable() enter. Task: {task}")
        if self.start_delay_sec > 0.0:
            self.get_logger().info(f"Waiting {self.start_delay_sec:.2f}s before first SERL command.")
            time.sleep(self.start_delay_sec)

        period_sec = 1.0 / self.control_hz
        start_time = time.monotonic()
        command_count = 0
        while time.monotonic() - start_time < self.max_runtime_sec:
            loop_start = time.monotonic()
            observation_msg = get_observation()
            if observation_msg is None:
                self.get_logger().info("No observation received.")
                time.sleep(period_sec)
                continue

            action = self.select_delta_action(observation_msg)
            if hasattr(self._parent_node, "_target_mode"):
                self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
            if self.command_mode == "none":
                pass
            elif self.command_mode == "velocity":
                self._send_velocity_target(move_robot, action[:3], action[3:6])
            elif self.command_mode == "delta_pose":
                self._send_delta_pose_target(move_robot, action[:3], action[3:6])
            else:
                raise ValueError(f"Unsupported AIC_SERL_COMMAND_MODE={self.command_mode!r}")

            command_count += 1
            if command_count % max(1, int(self.control_hz)) == 0:
                stats = self.policy.last_action_components
                send_feedback(
                    "in progress; "
                    f"commands={command_count}; "
                    f"base_norm={stats.get('base_action_norm', 0.0):.4f}; "
                    f"delta_norm={stats.get('delta_action_norm', 0.0):.4f}; "
                    f"final_norm={stats.get('final_action_norm', 0.0):.4f}"
                )

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period_sec - elapsed))

        self.get_logger().info("RunACTAdapterSERL.insert_cable() exiting...")
        return True
