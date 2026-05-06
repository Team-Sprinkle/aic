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

"""SmolVLA policy runner for AIC LeRobot policy-recorder datasets."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import cv2
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TargetMode, TrajectoryGenerationMode
from geometry_msgs.msg import Twist, Vector3, Wrench
from huggingface_hub import snapshot_download
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import (  # noqa: F401
    SmolVLANewLineProcessor,
)


DEFAULT_CONTROL_HZ = 4.0
DEFAULT_MAX_RUNTIME_SEC = 120.0
TRAINING_DATASET_TASK = "Insert cable into target port"
ACTION_NAMES = (
    "delta_position.x",
    "delta_position.y",
    "delta_position.z",
    "delta_rotation.x",
    "delta_rotation.y",
    "delta_rotation.z",
)


def _fixed_len(values: Any, length: int) -> list[float]:
    out = [float(v) for v in list(values)[:length]]
    if len(out) < length:
        out.extend([0.0] * (length - len(out)))
    return out


class RunSmolVLA(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        requested_device = os.environ.get("AIC_SMOLVLA_DEVICE")
        self.device = torch.device(
            requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.control_hz = float(
            os.environ.get("AIC_SMOLVLA_CONTROL_HZ", DEFAULT_CONTROL_HZ)
        )
        self.max_runtime_sec = float(
            os.environ.get("AIC_SMOLVLA_MAX_RUNTIME_SEC", DEFAULT_MAX_RUNTIME_SEC)
        )
        self.start_delay_sec = float(os.environ.get("AIC_SMOLVLA_START_DELAY_SEC", 0.0))
        self.command_mode = os.environ.get("AIC_SMOLVLA_COMMAND_MODE", "delta_pose")
        self.command_frame = os.environ.get("AIC_SMOLVLA_COMMAND_FRAME", "gripper/tcp")
        self.robot_type = os.environ.get("AIC_SMOLVLA_ROBOT_TYPE", "aic")
        self.max_translation_delta = float(
            os.environ.get("AIC_SMOLVLA_MAX_TRANSLATION_DELTA", 0.02)
        )
        self.max_rotation_delta = float(
            os.environ.get("AIC_SMOLVLA_MAX_ROTATION_DELTA", 0.2)
        )
        self.translation_deadband = float(
            os.environ.get("AIC_SMOLVLA_TRANSLATION_DEADBAND", 5e-4)
        )
        self.rotation_deadband = float(
            os.environ.get("AIC_SMOLVLA_ROTATION_DEADBAND", 1e-3)
        )

        self.policy_path = self._resolve_policy_path()
        config = PreTrainedConfig.from_pretrained(self.policy_path)
        if not isinstance(config, SmolVLAConfig):
            raise TypeError(
                f"Expected a SmolVLA checkpoint config, got {type(config).__name__}"
            )
        config.device = str(self.device)

        self.policy = SmolVLAPolicy.from_pretrained(self.policy_path, config=config)
        self.policy.eval()
        self.policy.to(self.device)
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            str(self.policy_path),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.camera_shapes = self._camera_shapes()
        self.state_dim = 32
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        if self.action_dim < len(ACTION_NAMES):
            raise ValueError(
                f"RunSmolVLA requires at least {len(ACTION_NAMES)} action dimensions, "
                f"got {self.action_dim}"
            )

        self.get_logger().info(
            "SmolVLA policy loaded from "
            f"{self.policy_path} on {self.device}; "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"control_hz={self.control_hz}, start_delay_sec={self.start_delay_sec}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"action_frame=gripper/tcp, command_mode={self.command_mode}, "
            f"command_frame={self.command_frame}"
        )

    def _resolve_policy_path(self) -> Path:
        policy_path = os.environ.get(
            "AIC_SMOLVLA_POLICY_PATH",
            "/home/jk/ws_aic/src/aic/outputs/train/smolvla-sfp2nic_card2_port1/checkpoints/050000/pretrained_model"
        )
        if policy_path:
            path = Path(policy_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"AIC_SMOLVLA_POLICY_PATH does not exist: {path}")
            return path

        repo_id = os.environ.get("AIC_SMOLVLA_POLICY_REPO_ID")
        if not repo_id:
            raise ValueError(
                "Set AIC_SMOLVLA_POLICY_PATH to a local checkpoint directory or "
                "AIC_SMOLVLA_POLICY_REPO_ID to a Hugging Face policy repo."
            )
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor.json",
                    "policy_postprocessor.json",
                    "*.safetensors",
                ],
            )
        )

    def _camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        rename_map = self._observation_rename_map()
        source_key_by_policy_key = {dst: src for src, dst in rename_map.items()}
        shapes = {}
        for key, feature in self.policy.config.input_features.items():
            if not key.startswith("observation.images."):
                continue
            shape = tuple(int(v) for v in feature.shape)
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"expected CHW camera feature for {key}, got {shape}")
            shapes[source_key_by_policy_key.get(key, key)] = shape
        if not shapes:
            raise ValueError("SmolVLA checkpoint has no observation.images.* input features")
        return shapes

    def _observation_rename_map(self) -> dict[str, str]:
        for step in self.preprocess.steps:
            rename_map = getattr(step, "rename_map", None)
            if rename_map:
                return dict(rename_map)
        return {}

    @staticmethod
    def _image_to_bgr(image_msg: Any) -> np.ndarray:
        h = int(image_msg.height)
        w = int(image_msg.width)
        step = int(image_msg.step)
        if h <= 0 or w <= 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        row_data = np.frombuffer(image_msg.data, dtype=np.uint8)
        try:
            row_data = row_data.reshape(h, step)
        except ValueError:
            return np.zeros((h, w, 3), dtype=np.uint8)

        encoding = str(getattr(image_msg, "encoding", "")).lower()
        if encoding == "bgr8":
            return row_data[:, : w * 3].reshape(h, w, 3)
        if encoding == "rgb8":
            rgb = row_data[:, : w * 3].reshape(h, w, 3)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if encoding == "mono8":
            return cv2.cvtColor(row_data[:, :w], cv2.COLOR_GRAY2BGR)
        if encoding == "bgra8":
            bgra = row_data[:, : w * 4].reshape(h, w, 4)
            return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        if encoding == "rgba8":
            rgba = row_data[:, : w * 4].reshape(h, w, 4)
            return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        return np.zeros((h, w, 3), dtype=np.uint8)

    @staticmethod
    def _image_tensor(image_msg: Any, shape: tuple[int, int, int]) -> torch.Tensor:
        channels, height, width = shape
        if channels != 3:
            raise ValueError(f"expected 3-channel image feature, got {shape}")
        image = RunSmolVLA._image_to_bgr(image_msg)
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        return (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div(255.0)
            .unsqueeze(0)
        )

    def _state_vector(self, obs_msg: Observation) -> np.ndarray:
        tcp_pose = obs_msg.controller_state.tcp_pose
        tcp_vel = obs_msg.controller_state.tcp_velocity
        values = [
            tcp_pose.position.x,
            tcp_pose.position.y,
            tcp_pose.position.z,
            tcp_pose.orientation.x,
            tcp_pose.orientation.y,
            tcp_pose.orientation.z,
            tcp_pose.orientation.w,
            tcp_vel.linear.x,
            tcp_vel.linear.y,
            tcp_vel.linear.z,
            tcp_vel.angular.x,
            tcp_vel.angular.y,
            tcp_vel.angular.z,
            *_fixed_len(obs_msg.controller_state.tcp_error, 6),
            *_fixed_len(obs_msg.joint_states.position, 7),
            obs_msg.wrist_wrench.wrench.force.x,
            obs_msg.wrist_wrench.wrench.force.y,
            obs_msg.wrist_wrench.wrench.force.z,
            obs_msg.wrist_wrench.wrench.torque.x,
            obs_msg.wrist_wrench.wrench.torque.y,
            obs_msg.wrist_wrench.wrench.torque.z,
        ]
        if self.state_dim == 26:
            values = values[:26]
        if len(values) != self.state_dim:
            raise ValueError(
                f"checkpoint expects observation.state dim {self.state_dim}, "
                f"but RunSmolVLA built {len(values)}"
            )
        return np.asarray(values, dtype=np.float32)

    def prepare_observations(
        self, obs_msg: Observation, task: Task
    ) -> dict[str, torch.Tensor]:
        del task
        observation: dict[str, Any] = {}
        camera_msg_by_key = {
            "observation.images.left_camera": obs_msg.left_image,
            "observation.images.center_camera": obs_msg.center_image,
            "observation.images.right_camera": obs_msg.right_image,
        }
        for key, shape in self.camera_shapes.items():
            observation[key] = self._image_tensor(camera_msg_by_key[key], shape).to(
                self.device
            )

        observation["observation.state"] = (
            torch.from_numpy(self._state_vector(obs_msg)).unsqueeze(0).to(self.device)
        )
        observation["task"] = TRAINING_DATASET_TASK
        observation["robot_type"] = self.robot_type
        return self.preprocess(observation)

    def select_delta_action(self, obs_msg: Observation, task: Task) -> np.ndarray:
        batch = self.prepare_observations(obs_msg, task)
        with torch.inference_mode():
            normalized_action = self.policy.select_action(batch)
            action = self.postprocess(normalized_action)
        if isinstance(action, dict):
            return np.asarray(
                [float(action[name]) for name in ACTION_NAMES], dtype=np.float64
            )
        return action[0, : len(ACTION_NAMES)].detach().cpu().numpy().astype(np.float64)

    def _set_velocity_target(
        self,
        move_robot: MoveRobotCallback,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> bool:
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
        motion_update.target_stiffness = np.diag(
            [100.0, 100.0, 100.0, 50.0, 50.0, 50.0]
        ).flatten()
        motion_update.target_damping = np.diag(
            [40.0, 40.0, 40.0, 15.0, 15.0, 15.0]
        ).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(
            force=Vector3(x=0.0, y=0.0, z=0.0),
            torque=Vector3(x=0.0, y=0.0, z=0.0),
        )
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_VELOCITY
        )
        return bool(move_robot(motion_update=motion_update))

    def _clamp_action(
        self,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        from aic_model.policy import clamp_delta_pose_components

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
        del kwargs
        self.policy.reset()
        self.get_logger().info(f"RunSmolVLA.insert_cable() enter. Task: {task}")
        if self.start_delay_sec > 0.0:
            self.get_logger().info(
                f"Waiting {self.start_delay_sec:.2f}s before first SmolVLA command."
            )
            time.sleep(self.start_delay_sec)

        period_sec = 1.0 / self.control_hz
        start_time = time.monotonic()
        command_count = 0
        while time.monotonic() - start_time < self.max_runtime_sec:
            loop_start = time.monotonic()
            observation_msg = get_observation()
            w = observation_msg.wrist_wrench.wrench
            self.get_logger().info(
                f"FT force norm={np.sqrt(w.force.x**2+w.force.y**2+w.force.z**2):.3f} | "
                f"torque norm  ={np.sqrt(w.torque.x**2+w.torque.y**2+w.torque.z**2):.3f}"
#                f"FT force norm=({w.force.x:.3f}, {w.force.y:.3f}, {w.force.z:.3f}) "
#                f"torque norm=({w.torque.x:.3f}, {w.torque.y:.3f}, {w.torque.z:.3f})"
            )

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                time.sleep(period_sec)
                continue

            action = self.select_delta_action(observation_msg, task)
            self.get_logger().info(f"SmolVLA action: {action}")
            if hasattr(self._parent_node, "_target_mode"):
                self._parent_node._target_mode = TargetMode.MODE_CARTESIAN

            if self.command_mode == "velocity":
                self._set_velocity_target(move_robot, action[:3], action[3:6])
            elif self.command_mode == "delta_pose":
                self.set_delta_pose_target_from_components(
                    move_robot=move_robot,
                    delta_position_xyz=action[:3],
                    delta_rotation_xyz=action[3:6],
                    max_translation=self.max_translation_delta,
                    max_rotation=self.max_rotation_delta,
                    deadband_translation=self.translation_deadband,
                    deadband_rotation=self.rotation_deadband,
                )
            else:
                raise ValueError(
                    f"Unsupported AIC_SMOLVLA_COMMAND_MODE={self.command_mode!r}"
                )

            command_count += 1
            if command_count % max(1, int(self.control_hz)) == 0:
                send_feedback(f"in progress; commands={command_count}")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period_sec - elapsed))

        self.get_logger().info("RunSmolVLA.insert_cable() exiting successfully.")
        return True
