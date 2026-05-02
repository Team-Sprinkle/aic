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

"""ACT policy runner for current AIC LeRobot datasets.

This runner is intentionally matched to the current 20 Hz AIC observation stream
and the 6D delta-pose action schema used by the converted/expert datasets:

  action = [delta_position.xyz, delta_rotation.xyz]
  action frame = gripper/tcp

Set AIC_ACT_POLICY_PATH to a local LeRobot pretrained_model directory. If it is
unset, AIC_ACT_POLICY_REPO_ID is downloaded from Hugging Face.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import cv2
import draccus
import numpy as np
import torch
from huggingface_hub import snapshot_download
from rclpy.node import Node
from safetensors.torch import load_file

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy


DEFAULT_POLICY_REPO_ID = "grkw/aic_act_policy"
DEFAULT_CONTROL_HZ = 20.0
DEFAULT_MAX_RUNTIME_SEC = 30.0


class RunACT(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        requested_device = os.environ.get("AIC_ACT_DEVICE")
        self.device = torch.device(
            requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.control_hz = float(os.environ.get("AIC_ACT_CONTROL_HZ", DEFAULT_CONTROL_HZ))
        self.max_runtime_sec = float(
            os.environ.get("AIC_ACT_MAX_RUNTIME_SEC", DEFAULT_MAX_RUNTIME_SEC)
        )
        self.max_translation_delta = float(os.environ.get("AIC_ACT_MAX_TRANSLATION_DELTA", 0.02))
        self.max_rotation_delta = float(os.environ.get("AIC_ACT_MAX_ROTATION_DELTA", 0.2))
        self.translation_deadband = float(os.environ.get("AIC_ACT_TRANSLATION_DEADBAND", 5e-4))
        self.rotation_deadband = float(os.environ.get("AIC_ACT_ROTATION_DEADBAND", 1e-3))

        self.policy_path = self._resolve_policy_path()
        config_dict = self._load_config_dict(self.policy_path)
        config = draccus.decode(ACTConfig, config_dict)
        config.device = str(self.device)

        self.policy = ACTPolicy(config)
        self.policy.load_state_dict(load_file(self.policy_path / "model.safetensors"))
        self.policy.eval()
        self.policy.to(self.device)

        self.input_features = dict(config.input_features)
        self.output_features = dict(config.output_features)
        self.camera_shapes = self._camera_shapes(config_dict)
        self.state_dim = int(config_dict["input_features"]["observation.state"]["shape"][0])
        self.action_dim = int(config_dict["output_features"]["action"]["shape"][0])
        if self.action_dim < 6:
            raise ValueError(f"RunACT requires at least 6 action dimensions, got {self.action_dim}")

        stats = load_file(
            self.policy_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        )
        self.img_stats = {
            key: {
                "mean": self._stat(stats, f"{key}.mean", (1, 3, 1, 1)),
                "std": self._stat(stats, f"{key}.std", (1, 3, 1, 1)),
            }
            for key in self.camera_shapes
        }
        self.state_mean = self._stat(stats, "observation.state.mean", (1, -1))
        self.state_std = self._stat(stats, "observation.state.std", (1, -1))
        self.action_mean = self._stat(stats, "action.mean", (1, -1))
        self.action_std = self._stat(stats, "action.std", (1, -1))

        self.get_logger().info(
            "ACT policy loaded from "
            f"{self.policy_path} on {self.device}; "
            f"state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"control_hz={self.control_hz}, "
            f"chunk_size={self.policy.config.chunk_size}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            "action_mode=delta_pose, action_frame=gripper/tcp"
        )

    def _resolve_policy_path(self) -> Path:
        policy_path = os.environ.get("AIC_ACT_POLICY_PATH")
        if policy_path:
            path = Path(policy_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"AIC_ACT_POLICY_PATH does not exist: {path}")
            return path

        repo_id = os.environ.get("AIC_ACT_POLICY_REPO_ID", DEFAULT_POLICY_REPO_ID)
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "policy_preprocessor_step_3_normalizer_processor.safetensors",
                ],
            )
        )

    @staticmethod
    def _load_config_dict(policy_path: Path) -> dict[str, Any]:
        with (policy_path / "config.json").open("r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dict.pop("type", None)
        return config_dict

    def _stat(self, stats: dict[str, torch.Tensor], key: str, shape: tuple[int, ...]) -> torch.Tensor:
        if key not in stats:
            raise KeyError(f"policy normalizer is missing required statistic {key!r}")
        return stats[key].to(self.device).view(*shape)

    @staticmethod
    def _camera_shapes(config_dict: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
        shapes = {}
        for key, feature in config_dict["input_features"].items():
            if not key.startswith("observation.images."):
                continue
            shape = tuple(int(v) for v in feature["shape"])
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(f"expected CHW RGB camera feature for {key}, got {shape}")
            shapes[key] = shape
        return shapes

    @staticmethod
    def _image_msg_to_chw_float(raw_img: Any, shape: tuple[int, int, int]) -> torch.Tensor:
        channels, height, width = shape
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height,
            raw_img.width,
            channels,
        )
        if raw_img.height != height or raw_img.width != width:
            img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    def _normalized_image(self, key: str, raw_img: Any) -> torch.Tensor:
        tensor = self._image_msg_to_chw_float(raw_img, self.camera_shapes[key]).to(self.device)
        return (tensor - self.img_stats[key]["mean"]) / self.img_stats[key]["std"]

    def prepare_observations(self, obs_msg: Observation) -> dict[str, torch.Tensor]:
        obs = {}
        camera_msg_by_key = {
            "observation.images.left_camera": obs_msg.left_image,
            "observation.images.center_camera": obs_msg.center_image,
            "observation.images.right_camera": obs_msg.right_image,
        }
        for key in self.camera_shapes:
            obs[key] = self._normalized_image(key, camera_msg_by_key[key])

        state_np = self._state_vector(obs_msg)
        if state_np.shape[0] != self.state_dim:
            raise ValueError(
                f"checkpoint expects observation.state dim {self.state_dim}, "
                f"but RunACT built {state_np.shape[0]}"
            )
        raw_state = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        obs["observation.state"] = (raw_state - self.state_mean) / self.state_std
        return obs

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
            *obs_msg.controller_state.tcp_error,
            *obs_msg.joint_states.position[:7],
        ]
        if self.state_dim >= 32:
            values.extend(
                [
                    obs_msg.wrist_wrench.wrench.force.x,
                    obs_msg.wrist_wrench.wrench.force.y,
                    obs_msg.wrist_wrench.wrench.force.z,
                    obs_msg.wrist_wrench.wrench.torque.x,
                    obs_msg.wrist_wrench.wrench.torque.y,
                    obs_msg.wrist_wrench.wrench.torque.z,
                ]
            )
        return np.array(values, dtype=np.float32)

    def select_delta_action(self, obs_msg: Observation) -> np.ndarray:
        obs_tensors = self.prepare_observations(obs_msg)
        with torch.inference_mode():
            normalized_action = self.policy.select_action(obs_tensors)
        raw_action = (normalized_action * self.action_std) + self.action_mean
        return raw_action[0, :6].detach().cpu().numpy()

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.policy.reset()
        self.get_logger().info(f"RunACT.insert_cable() enter. Task: {task}")

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
            self.set_delta_pose_target_from_components(
                move_robot=move_robot,
                delta_position_xyz=action[:3],
                delta_rotation_xyz=action[3:6],
                max_translation=self.max_translation_delta,
                max_rotation=self.max_rotation_delta,
                deadband_translation=self.translation_deadband,
                deadband_rotation=self.rotation_deadband,
            )
            command_count += 1
            if command_count % max(1, int(self.control_hz)) == 0:
                send_feedback(f"in progress; commands={command_count}")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period_sec - elapsed))

        self.get_logger().info("RunACT.insert_cable() exiting...")
        return True
