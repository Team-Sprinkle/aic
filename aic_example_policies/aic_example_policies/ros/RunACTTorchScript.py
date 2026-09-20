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
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
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
from lerobot_robot_aic.act_state_contract import validate_act_state_contract  # noqa: E402
from lerobot_robot_aic.task_encoding import encode_task_vector  # noqa: E402


class RunACTTorchScript(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        torch.set_num_threads(int(os.environ.get("AIC_ACT_TORCH_THREADS", "4")))
        requested_device = os.environ.get("AIC_ACT_DEVICE")
        self.device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.control_hz = float(os.environ.get("AIC_ACT_CONTROL_HZ", 20.0))
        self.control_clock = os.environ.get("AIC_ACT_CONTROL_CLOCK", "wall")
        if self.control_clock not in {"wall", "simulation"} or self.control_hz <= 0:
            raise ValueError("ACT needs a positive control rate and a wall or simulation control clock")
        self.max_runtime_sec = float(os.environ.get("AIC_ACT_MAX_RUNTIME_SEC", 30.0))
        simulation_limit = os.environ.get("AIC_ACT_MAX_SIMULATION_SEC", "")
        self.max_simulation_sec = float(simulation_limit) if simulation_limit else None
        if not np.isfinite(self.max_runtime_sec) or self.max_runtime_sec <= 0:
            raise ValueError("ACT wall watchdog must be finite and positive")
        if self.max_simulation_sec is not None and (
                not np.isfinite(self.max_simulation_sec) or self.max_simulation_sec <= 0):
            raise ValueError("ACT simulation duration must be finite and positive when enabled")
        self.start_delay_sec = float(os.environ.get("AIC_ACT_START_DELAY_SEC", 0.0))
        self.command_mode = os.environ.get("AIC_ACT_RUNTIME_COMMAND_MODE", os.environ.get("AIC_ACT_COMMAND_MODE", "delta_pose"))
        self.command_frame = os.environ.get("AIC_ACT_COMMAND_FRAME", "gripper/tcp")
        self.delta_pose_reference = os.environ.get("AIC_ACT_DELTA_POSE_REFERENCE", "controller")
        if self.delta_pose_reference not in {"controller", "observation"}:
            raise ValueError("Delta pose reference must be controller or observation")
        if self.delta_pose_reference == "observation" and self.command_mode != "delta_pose":
            raise ValueError("Observation-referenced execution requires delta_pose model actions")
        self.max_translation_delta = float(os.environ.get("AIC_ACT_MAX_TRANSLATION_DELTA", 0.02))
        self.translation_limit_mode = os.environ.get("AIC_ACT_TRANSLATION_LIMIT_MODE", "component")
        if self.translation_limit_mode not in {"component", "norm"}:
            raise ValueError("Translation limit mode must be component or norm")
        self.max_rotation_delta = float(os.environ.get("AIC_ACT_MAX_ROTATION_DELTA", 0.2))
        self.translation_deadband = float(os.environ.get("AIC_ACT_TRANSLATION_DEADBAND", 5e-4))
        self.rotation_deadband = float(os.environ.get("AIC_ACT_ROTATION_DEADBAND", 1e-3))
        self.log_every_n_commands = max(1, int(os.environ.get("AIC_ACT_LOG_EVERY_N_COMMANDS", 20)))
        self.n_action_steps = int(os.environ.get("AIC_ACT_N_ACTION_STEPS", "4"))
        ensemble_coefficient = os.environ.get("AIC_ACT_TEMPORAL_ENSEMBLE_COEFF", "")
        self.temporal_ensemble_coefficient = float(ensemble_coefficient) if ensemble_coefficient else None
        if self.temporal_ensemble_coefficient is not None and (
            not np.isfinite(self.temporal_ensemble_coefficient) or self.temporal_ensemble_coefficient < 0
            or self.n_action_steps != 1 or self.command_mode != "absolute_pose"
        ):
            raise ValueError("Temporal ensembling requires a finite nonnegative coefficient, absolute poses and n_action_steps=1")
        if self.n_action_steps < 1:
            raise ValueError(f"AIC_ACT_N_ACTION_STEPS must be >= 1, got {self.n_action_steps}")

        self.torchscript_path = self._required_path("AIC_ACT_TORCHSCRIPT")
        self.metadata = self._load_metadata(self.torchscript_path)
        self.metadata.update(validate_act_state_contract(self.metadata))
        self.image_channel_order = os.environ.get("AIC_ACT_IMAGE_CHANNEL_ORDER", self.metadata.get("image_channel_order", "rgb"))
        if self.image_channel_order not in {"rgb", "bgr"}:
            raise ValueError("ACT image channel order must be rgb or bgr")
        self.action_representation = self.metadata.get("action_representation", "delta_pose")
        if self.command_mode != "none" and (
            (self.action_representation == "absolute_pose") != (self.command_mode == "absolute_pose")
        ):
            raise ValueError("ACT action representation and command mode disagree")
        if self.command_mode == "delta_pose" and self.command_frame != "gripper/tcp":
            raise ValueError("TCP delta ACT commands require gripper/tcp command frame")
        if self.metadata.get("action_frame") is not None and self.metadata["action_frame"] != self.command_frame:
            raise ValueError("ACT checkpoint action frame and runtime command frame disagree")
        if (self.metadata.get("delta_pose_reference") is not None
                and self.metadata["delta_pose_reference"] != self.delta_pose_reference):
            raise ValueError("ACT checkpoint delta reference and runtime reference disagree")
        self.model = torch.jit.load(str(self.torchscript_path), map_location=self.device).eval()
        self.state_dim = int(self.metadata["state_shape"][0])
        self.include_elapsed_sim_time = self.metadata["include_elapsed_sim_time"]
        self.base_state_dim = self.metadata["base_state_dim"]
        self.task_vector_indices = self.metadata["task_vector_indices"]
        self.time_clip_sec = float(self.metadata.get("time_clip_sec", 40.))
        self.quaternion_sign = self.metadata.get("quaternion_sign", "w")
        if self.quaternion_sign not in {"w", "x"}:
            raise ValueError("Unsupported ACT quaternion convention")
        self._episode_first_image_time = None
        if self.state_dim != self.base_state_dim + int(self.include_elapsed_sim_time):
            raise ValueError("ACT state metadata does not match exported feature count")
        self.action_dim = int((self.metadata.get("action_shape") or [6])[0])
        self.chunk_size = int(self.metadata.get("chunk_size") or 1)
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"AIC_ACT_N_ACTION_STEPS={self.n_action_steps} exceeds TorchScript chunk_size={self.chunk_size}"
            )
        self.camera_shapes = {
            "observation.images.center_camera": (3, 256, 288),
            "observation.images.left_camera": (3, 256, 288),
            "observation.images.right_camera": (3, 256, 288),
        }
        # Compile/initialize GPU kernels before the engine begins a scored
        # trial. The first inference otherwise consumes a large part of the
        # teacher's initial motion phase.
        warmup_start = time.monotonic()
        with torch.inference_mode():
            warmup = [torch.zeros(1, self.state_dim, device=self.device)]
            warmup.extend(torch.zeros(1, *shape, device=self.device) for shape in self.camera_shapes.values())
            for _ in range(2):
                self.model(*warmup)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.get_logger().info(f"ACT inference warmup completed in {time.monotonic() - warmup_start:.3f}s before trial start")

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
        self.feature_assembler = AICRuntimeFeatureAssembler(self.base_state_dim, fps=self.control_hz)
        self._action_queue: list[np.ndarray] = []
        self._ensemble_predictions: list[tuple[int, np.ndarray]] = []

        self.get_logger().info(
            "TorchScript ACT policy loaded from "
            f"{self.torchscript_path} on {self.device}; state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, control_hz={self.control_hz}, "
            f"control_clock={self.control_clock}, "
            f"image_channel_order={self.image_channel_order}, "
            f"chunk_size={self.chunk_size}, n_action_steps={self.n_action_steps}, "
            f"temporal_ensemble_coefficient={self.temporal_ensemble_coefficient}, "
            f"translation_limit_mode={self.translation_limit_mode}, max_translation_delta={self.max_translation_delta}, "
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
        override = os.environ.get("AIC_ACT_NORMALIZER_PATH")
        if override:
            return self._required_path("AIC_ACT_NORMALIZER_PATH")
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
        if self.state_mean.shape[-1] != self.state_dim or self.state_std.shape[-1] != self.state_dim:
            raise ValueError("ACT normalizer state statistics disagree with the exported state shape")
        if self.task_vector_indices is None:
            return
        start, stop = self.task_vector_indices
        self.state_mean[:, start:stop] = 0.0
        self.state_std[:, start:stop] = 1.0

    @staticmethod
    def _image_msg_to_chw_float(raw_img: Any, shape: tuple[int, int, int]) -> torch.Tensor:
        channels, height, width = shape
        encoding = raw_img.encoding.lower()
        source_channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}.get(encoding)
        if source_channels is None or channels != 3:
            raise ValueError(f"Unsupported ACT camera encoding or output shape: {encoding}, {shape}")
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(raw_img.height, raw_img.step)
        img_np = img_np[:, :raw_img.width * source_channels].reshape(raw_img.height, raw_img.width, source_channels)
        if source_channels == 1:
            img_np = np.repeat(img_np, 3, axis=-1)
        else:
            img_np = img_np[..., :3]
            if encoding.startswith("bgr"):
                img_np = img_np[..., ::-1]
        if raw_img.height != height or raw_img.width != width:
            img_np = cv2.resize(img_np, (width, height), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(np.ascontiguousarray(img_np)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)

    def _normalized_image(self, key: str, raw_img: Any) -> torch.Tensor:
        tensor = self._image_msg_to_chw_float(raw_img, self.camera_shapes[key]).to(self.device)
        if self.image_channel_order == "bgr":
            tensor = tensor[:, [2, 1, 0]]
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
            return encode_task_vector(task_family="sfp_to_nic", target_port_index=port_index,
                                      target_card_index=card_index, target_card_valid=1).tolist()
        if task.target_module_name.startswith("sc_port_"):
            port_index = self._parse_index_from_suffix(task.target_module_name, "sc_port_")
            return encode_task_vector(task_family="sc_to_sc", target_port_index=port_index,
                                      target_card_index=-1, target_card_valid=0).tolist()
        raise ValueError(f"Cannot infer task vector from target_module_name={task.target_module_name!r}")

    def _state_vector(self, obs_msg: Observation) -> np.ndarray:
        if self.feature_assembler.uses_task_vector:
            if self._current_task is None:
                raise ValueError(f"checkpoint expects task-conditioned {self.state_dim}D state, but no active task is set")
            self.feature_assembler.task_vector = np.asarray(self._task_vector(self._current_task), dtype=np.float32)
        state = self.feature_assembler.assemble_ros(obs_msg)
        if self.quaternion_sign == "x" and state[3] < 0:
            state[3:7] *= -1
        if self.include_elapsed_sim_time:
            stamp = obs_msg.center_image.header.stamp
            now = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if self._episode_first_image_time is None:
                self._episode_first_image_time = now
            elapsed = np.clip(now - self._episode_first_image_time, 0., self.time_clip_sec)
            state = np.concatenate([state, np.asarray([elapsed], dtype=np.float32)])
        return state

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

    def _predict_action_chunk(self, obs_msg: Observation) -> np.ndarray:
        obs = self.prepare_observations(obs_msg)
        with torch.no_grad():
            chunk = self.model(
                obs["state"],
                obs["observation.images.center_camera"],
                obs["observation.images.left_camera"],
                obs["observation.images.right_camera"],
            )
        if chunk.ndim != 3 or chunk.shape[1] < self.n_action_steps or chunk.shape[2] < self.action_dim:
            raise ValueError(
                "TorchScript ACT returned incompatible chunk shape "
                f"{tuple(chunk.shape)} for n_action_steps={self.n_action_steps}, action_dim={self.action_dim}"
            )
        horizon = self.chunk_size if self.temporal_ensemble_coefficient is not None else self.n_action_steps
        normalized = chunk[:, :horizon, : self.action_dim]
        raw = normalized * self.action_std[:, : self.action_dim].unsqueeze(1) + self.action_mean[
            :, : self.action_dim
        ].unsqueeze(1)
        return raw[0, :, :6].detach().cpu().numpy()

    def select_delta_action(self, obs_msg: Observation) -> np.ndarray:
        if self.temporal_ensemble_coefficient is not None:
            return self._ensemble_action(self._predict_action_chunk(obs_msg))
        if not self._action_queue:
            self._action_queue = [action for action in self._predict_action_chunk(obs_msg)]
        return self._action_queue.pop(0)

    def _ensemble_action(self, chunk: np.ndarray) -> np.ndarray:
        # Each prediction's first remaining element targets the current command
        # step. Age zero is the newest observation; expired predictions vanish.
        if not np.isfinite(chunk).all():
            self._ensemble_predictions.clear()
            return np.full(6, np.nan)
        self._ensemble_predictions.append((0, chunk))
        weights = np.asarray([np.exp(-self.temporal_ensemble_coefficient * age)
                              for age, _ in self._ensemble_predictions])
        action = np.average(np.stack([remaining[0] for _, remaining in self._ensemble_predictions]),
                            axis=0, weights=weights)
        self._ensemble_predictions = [(age + 1, remaining[1:]) for age, remaining in self._ensemble_predictions
                                      if len(remaining) > 1]
        return action

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

        position, rotation = clamp_delta_pose_components(
            delta_position_xyz,
            delta_rotation_xyz,
            max_translation=self.max_translation_delta if self.translation_limit_mode == "component" else None,
            max_rotation=self.max_rotation_delta,
            deadband_translation=self.translation_deadband,
            deadband_rotation=self.rotation_deadband,
        )
        if self.translation_limit_mode == "norm":
            position = self._limit_translation_norm(position, self.max_translation_delta)
        return position, rotation

    @staticmethod
    def _limit_translation_norm(position: np.ndarray, limit: float) -> np.ndarray:
        norm = np.linalg.norm(position)
        return position * min(1., limit / max(norm, 1e-12))

    @staticmethod
    def _observed_delta_target_values(current_pose, position, rotation):
        """Anchor a camera-observation-relative label before transport latency."""
        from scipy.spatial.transform import Rotation
        q = current_pose.orientation
        current_rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
        p = current_pose.position
        target_position = np.asarray([p.x, p.y, p.z]) + current_rotation.apply(position)
        target_quaternion = (current_rotation * Rotation.from_rotvec(rotation)).as_quat()
        return target_position, target_quaternion

    def _send_observed_delta_target(self, move_robot, current_pose, position, rotation):
        from aic_model.policy import build_pose_from_vectors
        target_position, target_quaternion = self._observed_delta_target_values(current_pose, position, rotation)
        self.set_pose_target(move_robot=move_robot,
                             pose=build_pose_from_vectors(target_position, target_quaternion), frame_id="base_link")

    def reset_for_task(self, task):
        self._current_task = task
        self._episode_first_image_time = None
        self.feature_assembler.reset(self._task_vector(task) if self.feature_assembler.uses_task_vector else None)
        self._action_queue.clear()
        self._ensemble_predictions.clear()

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ) -> bool:
        self.reset_for_task(task)
        self.get_logger().info(f"RunACTTorchScript.insert_cable() enter. Task: {task}")
        if self.start_delay_sec > 0.0:
            self.get_logger().info(f"Waiting {self.start_delay_sec:.2f}s before first ACT TorchScript command.")
            time.sleep(self.start_delay_sec)

        period_sec = 1.0 / self.control_hz
        start_time = time.monotonic()
        command_count = 0
        last_control_sim_time = None
        first_observation_sim_time = None
        simulation_elapsed_sec = None
        stop_reason = "wall_watchdog"
        runtime_identity = {
            "runtime_episode_id": uuid.uuid4().hex, "task_id": task.id,
            "target_module_name": task.target_module_name, "port_name": task.port_name,
            "plug_name": task.plug_name, "plug_type": task.plug_type,
        }
        self.get_logger().info("ACT_RUNTIME_START " + json.dumps(runtime_identity, sort_keys=True))
        while time.monotonic() - start_time < self.max_runtime_sec:
            loop_start = time.monotonic()
            observation_msg = get_observation()
            if observation_msg is None:
                self.get_logger().info("No observation received.")
                time.sleep(period_sec)
                continue

            stamp = observation_msg.center_image.header.stamp
            sim_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
            if first_observation_sim_time is None:
                first_observation_sim_time = sim_time
            simulation_elapsed_sec = max(simulation_elapsed_sec or 0., sim_time - first_observation_sim_time)
            if self.max_simulation_sec is not None and simulation_elapsed_sec >= self.max_simulation_sec - 1e-6:
                stop_reason = "simulation_limit"
                break
            if self.control_clock == "simulation":
                if (last_control_sim_time is not None
                        and 0 <= sim_time - last_control_sim_time < period_sec - 1e-6):
                    time.sleep(min(.005, period_sec))
                    continue
                last_control_sim_time = sim_time

            action = self.select_delta_action(observation_msg)
            action = self._finite_action_or_none(action, command_count)
            if action is None:
                time.sleep(period_sec)
                continue
            if self.command_mode == "none":
                position, rotation = self._clamp_action(action[:3], action[3:6])
            elif self.command_mode == "velocity":
                position, rotation = self._send_velocity_target(move_robot, action[:3], action[3:6])
            elif self.command_mode == "delta_pose":
                position, rotation = self._clamp_action(action[:3], action[3:6])
                if self.delta_pose_reference == "observation":
                    self._send_observed_delta_target(move_robot, observation_msg.controller_state.tcp_pose, position, rotation)
                else:
                    self.set_delta_pose_target_from_components(
                        move_robot=move_robot,
                        delta_position_xyz=position,
                        delta_rotation_xyz=rotation,
                        max_translation=self.max_translation_delta,
                        max_rotation=self.max_rotation_delta,
                    )
            elif self.command_mode == "absolute_pose":
                from aic_model.policy import (
                    build_pose_from_vectors, compute_delta_pose,
                    quaternion_xyzw_to_rotation_vector, rotation_vector_to_quaternion_xyzw,
                    quaternion_xyzw_to_rotation_matrix, quaternion_multiply_xyzw,
                )
                target = build_pose_from_vectors(action[:3], rotation_vector_to_quaternion_xyzw(action[3:6]))
                delta = compute_delta_pose(observation_msg.controller_state.tcp_pose, target)
                position = np.asarray([delta.position.x, delta.position.y, delta.position.z])
                rotation = quaternion_xyzw_to_rotation_vector(np.asarray([
                    delta.orientation.x, delta.orientation.y, delta.orientation.z, delta.orientation.w,
                ]))
                position, rotation = self._clamp_action(position, rotation)
                current = observation_msg.controller_state.tcp_pose
                current_quat = np.asarray([current.orientation.x, current.orientation.y,
                                           current.orientation.z, current.orientation.w])
                bounded_target = build_pose_from_vectors(
                    np.asarray([current.position.x, current.position.y, current.position.z])
                    + quaternion_xyzw_to_rotation_matrix(current_quat) @ position,
                    quaternion_multiply_xyzw(current_quat, rotation_vector_to_quaternion_xyzw(rotation)),
                )
                # Send the absolute target directly. Re-issuing it as a relative
                # command would add motion since this camera observation to the
                # target when the controller receives it.
                self.set_pose_target(
                    move_robot=move_robot, pose=bounded_target, frame_id="base_link",
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
            time.sleep(max(0.0, period_sec - elapsed) if self.control_clock == "wall" else .001)

        budget_reached = (self.max_simulation_sec is not None and simulation_elapsed_sec is not None
                          and simulation_elapsed_sec >= self.max_simulation_sec - 1e-6)
        self.get_logger().info("ACT_RUNTIME_STOP " + json.dumps({
            **runtime_identity,
            "reason": stop_reason, "wall_elapsed_sec": time.monotonic() - start_time,
            "simulation_elapsed_sec": simulation_elapsed_sec, "simulation_limit_sec": self.max_simulation_sec,
            "wall_watchdog_sec": self.max_runtime_sec, "commands": command_count,
            "simulation_budget_reached": budget_reached,
            "wall_watchdog_shortened": self.max_simulation_sec is not None and not budget_reached,
        }, sort_keys=True))
        self.get_logger().info("RunACTTorchScript.insert_cable() exiting...")
        return True
