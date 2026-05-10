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

"""TorchScript ACT policy runner for the official AIC runtime."""

from __future__ import annotations

import json
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
from safetensors.torch import load_file

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task

REPO_ROOT = Path(__file__).resolve().parents[3]
LEROBOT_AIC_PACKAGE_DIR = REPO_ROOT / "aic_utils" / "lerobot_robot_aic"
if LEROBOT_AIC_PACKAGE_DIR.exists() and str(LEROBOT_AIC_PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(LEROBOT_AIC_PACKAGE_DIR))

from lerobot_robot_aic.runtime_features import AICRuntimeFeatureAssembler  # noqa: E402


class RunACTTorchScript(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        requested_device = os.environ.get("AIC_ACT_DEVICE")
        self.device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.control_hz = float(os.environ.get("AIC_ACT_CONTROL_HZ", 20.0))
        self.max_runtime_sec = float(os.environ.get("AIC_ACT_MAX_RUNTIME_SEC", 30.0))
        self.start_delay_sec = float(os.environ.get("AIC_ACT_START_DELAY_SEC", 0.0))
        self.command_mode = os.environ.get("AIC_ACT_RUNTIME_COMMAND_MODE", os.environ.get("AIC_ACT_COMMAND_MODE", "delta_pose"))
        self.command_frame = os.environ.get("AIC_ACT_COMMAND_FRAME", "gripper/tcp")
        self.max_translation_delta = float(os.environ.get("AIC_ACT_MAX_TRANSLATION_DELTA", 0.02))
        self.max_rotation_delta = float(os.environ.get("AIC_ACT_MAX_ROTATION_DELTA", 0.2))
        self.translation_deadband = float(os.environ.get("AIC_ACT_TRANSLATION_DEADBAND", 5e-4))
        self.rotation_deadband = float(os.environ.get("AIC_ACT_ROTATION_DEADBAND", 1e-3))
        self.log_every_n_commands = max(1, int(os.environ.get("AIC_ACT_LOG_EVERY_N_COMMANDS", 20)))

        self.torchscript_path = self._required_path("AIC_ACT_TORCHSCRIPT")
        self.metadata = self._load_metadata(self.torchscript_path)
        self.model = torch.jit.load(str(self.torchscript_path), map_location=self.device).eval()
        self.state_dim = int((self.metadata.get("state_shape") or [42])[0])
        self.action_dim = int((self.metadata.get("action_shape") or [6])[0])
        self.camera_shapes = {
            "observation.images.center_camera": (3, 256, 288),
            "observation.images.left_camera": (3, 256, 288),
            "observation.images.right_camera": (3, 256, 288),
        }

        stats_path = self._stats_path()
        stats = load_file(stats_path)
        self.img_stats = {
            key: {
                "mean": self._stat(stats, f"{key}.mean", (1, 3, 1, 1)),
                "std": self._stat(stats, f"{key}.std", (1, 3, 1, 1)),
            }
            for key in self.camera_shapes
        }
        self.state_mean = self._stat(stats, "observation.state.mean", (1, -1))
        self.state_std = self._stat(stats, "observation.state.std", (1, -1))
        self._apply_task_vector_identity_normalization()
        self.action_mean = self._stat(stats, "action.mean", (1, -1))
        self.action_std = self._stat(stats, "action.std", (1, -1))
        self._current_task: Task | None = None
        self.feature_assembler = AICRuntimeFeatureAssembler(self.state_dim, fps=self.control_hz)

        self.get_logger().info(
            "TorchScript ACT policy loaded from "
            f"{self.torchscript_path} on {self.device}; state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, control_hz={self.control_hz}, "
            f"start_delay_sec={self.start_delay_sec}, command_mode={self.command_mode}, "
            f"command_frame={self.command_frame}"
        )

    @staticmethod
    def _required_path(env_name: str) -> Path:
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"{env_name} is required")
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} does not exist: {path}")
        return path

    @staticmethod
    def _load_metadata(torchscript_path: Path) -> dict[str, Any]:
        meta_path = torchscript_path.with_suffix(".json")
        if not meta_path.exists():
            return {}
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _stats_path(self) -> Path:
        checkpoint_dir = self.metadata.get("checkpoint_dir")
        if not checkpoint_dir:
            raise ValueError("TorchScript metadata must include checkpoint_dir to load normalizer stats")
        stats_path = Path(checkpoint_dir) / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        if not stats_path.exists():
            raise FileNotFoundError(f"ACT normalizer stats not found: {stats_path}")
        return stats_path

    def _stat(self, stats: dict[str, torch.Tensor], key: str, shape: tuple[int, ...]) -> torch.Tensor:
        if key not in stats:
            raise KeyError(f"policy normalizer is missing required statistic {key!r}")
        tensor = stats[key].to(self.device).view(*shape)
        if key.endswith(".std"):
            tensor = torch.where(torch.abs(tensor) < 1e-8, torch.ones_like(tensor), tensor)
        return tensor

    def _apply_task_vector_identity_normalization(self) -> None:
        if self.state_dim not in (42, 82):
            return
        self.state_mean[:, -10:] = 0.0
        self.state_std[:, -10:] = 1.0

    @staticmethod
    def _image_msg_to_chw_float(raw_img: Any, shape: tuple[int, int, int]) -> torch.Tensor:
        channels, height, width = shape
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(raw_img.height, raw_img.width, channels)
        if raw_img.height != height or raw_img.width != width:
            img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img_np).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    def _normalized_image(self, key: str, raw_img: Any) -> torch.Tensor:
        tensor = self._image_msg_to_chw_float(raw_img, self.camera_shapes[key]).to(self.device)
        return (tensor - self.img_stats[key]["mean"]) / self.img_stats[key]["std"]

    @staticmethod
    def _parse_index_from_suffix(value: str, prefix: str) -> int:
        if not value.startswith(prefix):
            raise ValueError(f"Expected {value!r} to start with {prefix!r}")
        return int(value.removeprefix(prefix))

    def _task_vector(self, task: Task) -> list[float]:
        if task.target_module_name.startswith("nic_card_mount_"):
            card_index = self._parse_index_from_suffix(task.target_module_name, "nic_card_mount_")
            port_index = self._parse_index_from_suffix(task.port_name, "sfp_port_")
            card = [0.0] * 5
            card[card_index] = 1.0
            return [1.0, 0.0, float(port_index == 0), float(port_index == 1), *card, 1.0]
        if task.target_module_name.startswith("sc_port_"):
            port_index = self._parse_index_from_suffix(task.target_module_name, "sc_port_")
            return [0.0, 1.0, float(port_index == 0), float(port_index == 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        raise ValueError(f"Cannot infer task vector from target_module_name={task.target_module_name!r}")

    def _state_vector(self, obs_msg: Observation) -> np.ndarray:
        if self.feature_assembler.uses_task_vector:
            if self._current_task is None:
                raise ValueError(f"checkpoint expects task-conditioned {self.state_dim}D state, but no active task is set")
            self.feature_assembler.task_vector = np.asarray(self._task_vector(self._current_task), dtype=np.float32)
        return self.feature_assembler.assemble_ros(obs_msg)

    def prepare_observations(self, obs_msg: Observation) -> dict[str, torch.Tensor]:
        camera_msg_by_key = {
            "observation.images.center_camera": obs_msg.center_image,
            "observation.images.left_camera": obs_msg.left_image,
            "observation.images.right_camera": obs_msg.right_image,
        }
        raw_state = torch.from_numpy(self._state_vector(obs_msg)).float().unsqueeze(0).to(self.device)
        if raw_state.shape[-1] != self.state_dim:
            raise ValueError(f"checkpoint expects observation.state dim {self.state_dim}, got {raw_state.shape[-1]}")
        return {
            "state": (raw_state - self.state_mean) / self.state_std,
            "observation.images.center_camera": self._normalized_image(
                "observation.images.center_camera", camera_msg_by_key["observation.images.center_camera"]
            ),
            "observation.images.left_camera": self._normalized_image(
                "observation.images.left_camera", camera_msg_by_key["observation.images.left_camera"]
            ),
            "observation.images.right_camera": self._normalized_image(
                "observation.images.right_camera", camera_msg_by_key["observation.images.right_camera"]
            ),
        }

    def select_delta_action(self, obs_msg: Observation) -> np.ndarray:
        obs = self.prepare_observations(obs_msg)
        with torch.no_grad():
            chunk = self.model(
                obs["state"],
                obs["observation.images.center_camera"],
                obs["observation.images.left_camera"],
                obs["observation.images.right_camera"],
            )
        normalized_action = chunk[:, 0, : self.action_dim]
        raw_action = normalized_action * self.action_std[:, : self.action_dim] + self.action_mean[:, : self.action_dim]
        return raw_action[0, :6].detach().cpu().numpy()

    def _finite_action_or_none(self, action: np.ndarray, command_count: int) -> np.ndarray | None:
        if np.all(np.isfinite(action[:6])):
            return action
        self.get_logger().error(
            "ACT produced non-finite action; skipping robot command "
            f"at command_index={command_count + 1}: {np.array2string(action[:6], precision=5)}"
        )
        return None

    def _send_velocity_target(
        self,
        move_robot: MoveRobotCallback,
        delta_position_xyz: np.ndarray,
        delta_rotation_xyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position, rotation = self._clamp_action(delta_position_xyz, delta_rotation_xyz)
        twist = Twist(
            linear=Vector3(x=float(position[0] * self.control_hz), y=float(position[1] * self.control_hz), z=float(position[2] * self.control_hz)),
            angular=Vector3(x=float(rotation[0] * self.control_hz), y=float(rotation[1] * self.control_hz), z=float(rotation[2] * self.control_hz)),
        )
        motion_update = MotionUpdate()
        motion_update.velocity = twist
        motion_update.header.frame_id = self.command_frame
        motion_update.header.stamp = self.get_clock().now().to_msg()
        motion_update.target_stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 50.0]).flatten()
        motion_update.target_damping = np.diag([40.0, 40.0, 40.0, 15.0, 15.0, 15.0]).flatten()
        motion_update.feedforward_wrench_at_tip = Wrench(force=Vector3(x=0.0, y=0.0, z=0.0), torque=Vector3(x=0.0, y=0.0, z=0.0))
        motion_update.wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
        motion_update.trajectory_generation_mode.mode = TrajectoryGenerationMode.MODE_VELOCITY
        move_robot(motion_update=motion_update)
        return position, rotation

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
        self._current_task = task
        self.feature_assembler.reset(self._task_vector(task) if self.feature_assembler.uses_task_vector else None)
        self.get_logger().info(f"RunACTTorchScript.insert_cable() enter. Task: {task}")
        if self.start_delay_sec > 0.0:
            self.get_logger().info(f"Waiting {self.start_delay_sec:.2f}s before first ACT TorchScript command.")
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
            action = self._finite_action_or_none(action, command_count)
            if action is None:
                time.sleep(period_sec)
                continue
            if hasattr(self._parent_node, "_target_mode"):
                self._parent_node._target_mode = TargetMode.MODE_CARTESIAN
            if self.command_mode == "none":
                position, rotation = self._clamp_action(action[:3], action[3:6])
            elif self.command_mode == "velocity":
                position, rotation = self._send_velocity_target(move_robot, action[:3], action[3:6])
            elif self.command_mode == "delta_pose":
                position, rotation = self._clamp_action(action[:3], action[3:6])
                self.set_delta_pose_target_from_components(
                    move_robot=move_robot,
                    delta_position_xyz=position,
                    delta_rotation_xyz=rotation,
                    max_translation=self.max_translation_delta,
                    max_rotation=self.max_rotation_delta,
                )
            else:
                raise ValueError(f"Unsupported AIC_ACT_RUNTIME_COMMAND_MODE={self.command_mode!r}")
            command_count += 1
            if command_count == 1 or command_count % self.log_every_n_commands == 0:
                self.get_logger().info(
                    "ACT command "
                    f"{command_count}: mode={self.command_mode}, raw={np.array2string(action[:6], precision=5)}, "
                    f"clamped_position={np.array2string(position, precision=5)}, "
                    f"clamped_rotation={np.array2string(rotation, precision=5)}"
                )
            if command_count % max(1, int(self.control_hz)) == 0:
                send_feedback(f"in progress; commands={command_count}")

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period_sec - elapsed))

        self.get_logger().info("RunACTTorchScript.insert_cable() exiting...")
        return True
