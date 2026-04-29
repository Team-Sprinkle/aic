import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict

import cv2
import numpy as np
import torch
import yaml
from rclpy.node import Node

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task


class RunMIP(Policy):
    """Run a MIP checkpoint trained in much-ado-about-noising within AIC ROS policy loop."""

    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths for repo-local MIP artifacts.
        self._assets_root = self._resolve_assets_root()
        self._checkpoint_path = self._assets_root / "model_100000.pt"
        self._task_cfg_path = (
            self._assets_root / "configs" / "task" / "aic_lerobot_image_state.yaml"
        )
        self._net_base_cfg_path = (
            self._assets_root / "configs" / "network" / "_base.yaml"
        )
        self._net_cfg_path = (
            self._assets_root / "configs" / "network" / "mlp.yaml"
        )
        self._opt_cfg_path = (
            self._assets_root / "configs" / "optimization" / "default.yaml"
        )
        self._log_cfg_path = (
            self._assets_root / "configs" / "log" / "default.yaml"
        )

        from mip.agent import TrainingAgent
        from mip.config import Config, LogConfig, NetworkConfig, OptimizationConfig, TaskConfig
        from mip.datasets.lerobot_dataset import make_dataset

        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"MIP checkpoint not found: {self._checkpoint_path}")

        config = self._build_mip_config(
            Config,
            TaskConfig,
            NetworkConfig,
            OptimizationConfig,
            LogConfig,
        )
        self.config = config

        self.get_logger().info(f"MIP configuration: {self.config}")

        # Dataset is loaded to recover the same normalizers used in train/eval.
        self.dataset = make_dataset(self.config.task)
        self.obs_normalizers = self.dataset.normalizer["obs"]
        self.action_normalizer = self.dataset.normalizer["action"]

        self.policy = TrainingAgent(self.config)
        self.policy.load(str(self._checkpoint_path), load_optimizer=False)
        self.policy.eval()

        self.obs_steps = int(self.config.task.obs_steps)
        self.horizon = int(self.config.task.horizon)
        self.act_steps = int(self.config.task.act_steps)
        self.act_dim = int(self.config.task.act_dim)

        self.left_hist: Deque[np.ndarray] = deque(maxlen=self.obs_steps)
        self.center_hist: Deque[np.ndarray] = deque(maxlen=self.obs_steps)
        self.right_hist: Deque[np.ndarray] = deque(maxlen=self.obs_steps)
        self.state_hist: Deque[np.ndarray] = deque(maxlen=self.obs_steps)
        self.pending_actions: Deque[np.ndarray] = deque()

        self.camera_scale = 0.25
        self.max_translation_delta = 0.02
        self.max_rotation_delta = 0.2
        self.translation_deadband = 5e-4
        self.rotation_deadband = 1e-3

        self.get_logger().info(
            f"Loaded MIP policy on {self.device} from {self._checkpoint_path} "
            f"(obs_steps={self.obs_steps}, act_steps={self.act_steps}, "
            f"horizon={self.horizon}, act_dim={self.act_dim})."
        )

    def _resolve_assets_root(self) -> Path:
        env_override = os.getenv("AIC_MIP_ASSETS_DIR")
        candidates = []
        if env_override:
            candidates.append(Path(env_override).expanduser())

        # Installed package path (site-packages) or source-tree path.
        pkg_candidate = Path(__file__).resolve().parent.parent / "assets" / "mip"
        candidates.append(pkg_candidate)

        # TODO fix this to smarter Common workspace layouts when running from source checkout.
        cwd = Path.cwd()
        candidates.extend(
            [
                cwd / "aic_example_policies" / "aic_example_policies" / "assets" / "mip",
                cwd
                / "src"
                / "aic"
                / "aic_example_policies"
                / "aic_example_policies"
                / "assets"
                / "mip",
                Path("/home/jk/ws_aic")
                / "src"
                / "aic"
                / "aic_example_policies"
                / "aic_example_policies"
                / "assets"
                / "mip",
            ]
        )

        for candidate in candidates:
            if (candidate / "model_latest.pt").exists():
                self.get_logger().info(f"Using MIP assets root: {candidate}")
                return candidate

        raise FileNotFoundError(
            "Could not locate MIP assets directory containing model_latest.pt. "
            "Set AIC_MIP_ASSETS_DIR or place artifacts under "
            "aic_example_policies/aic_example_policies/assets/mip."
        )

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r") as f:
            raw = yaml.safe_load(f) or {}
        raw.pop("defaults", None)
        return raw

    def _merge_dicts(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = self._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _build_mip_config(
        self,
        Config,
        TaskConfig,
        NetworkConfig,
        OptimizationConfig,
        LogConfig,
    ):
        task_cfg = self._load_yaml(self._task_cfg_path)
        network_cfg = self._merge_dicts(
            self._load_yaml(self._net_base_cfg_path),
            self._load_yaml(self._net_cfg_path),
        )
        optimization_cfg = self._load_yaml(self._opt_cfg_path)
        log_cfg = self._load_yaml(self._log_cfg_path)

        # YAML keeps scientific notation strings quoted in this config; coerce for optimizer.
        for float_key in ("lr", "weight_decay"):
            if float_key in optimization_cfg:
                optimization_cfg[float_key] = float(optimization_cfg[float_key])

        dataset_path = Path(task_cfg["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_root_env = os.getenv("MIP_DATASET_ROOT")
            candidate_roots = []
            if dataset_root_env:
                candidate_roots.append(Path(dataset_root_env).expanduser())
            candidate_roots.extend(
                [
                    self._assets_root,
                    Path("/home/jk/much-ado-about-noising"),
                ]
            )

            resolved_path = None
            for root in candidate_roots:
                candidate = root / dataset_path
                if candidate.exists():
                    resolved_path = candidate
                    break

            if resolved_path is None:
                resolved_path = self._assets_root / dataset_path
                self.get_logger().warn(
                    "Could not find relative dataset path under known roots. "
                    f"Using fallback path: {resolved_path}"
                )
            dataset_path = resolved_path
        task_cfg["dataset_path"] = str(dataset_path)

        # Runtime-safe inference settings.
        optimization_cfg["device"] = str(self.device)
        optimization_cfg["use_compile"] = False
        optimization_cfg["use_cudagraphs"] = False
        optimization_cfg["auto_resume"] = False

        return Config(
            optimization=OptimizationConfig(**optimization_cfg),
            network=NetworkConfig(**network_cfg),
            task=TaskConfig(**task_cfg),
            log=LogConfig(**log_cfg),
            mode="eval",
        )

    @staticmethod
    def _image_to_chw_float(raw_img, scale: float) -> np.ndarray:
        img_np = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(
            raw_img.height, raw_img.width, 3
        )

        if scale != 1.0:
            img_np = cv2.resize(
                img_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        chw = np.moveaxis(img_np, -1, 0).astype(np.float32) / 255.0
        return chw

    @staticmethod
    def _state_vector_32(obs_msg: Observation) -> np.ndarray:
        ctrl = obs_msg.controller_state
        wrench = obs_msg.wrist_wrench.wrench

        joints = list(obs_msg.joint_states.position[:7])
        if len(joints) < 7:
            joints.extend([0.0] * (7 - len(joints)))

        return np.array(
            [
                ctrl.tcp_pose.position.x,
                ctrl.tcp_pose.position.y,
                ctrl.tcp_pose.position.z,
                ctrl.tcp_pose.orientation.x,
                ctrl.tcp_pose.orientation.y,
                ctrl.tcp_pose.orientation.z,
                ctrl.tcp_pose.orientation.w,
                ctrl.tcp_velocity.linear.x,
                ctrl.tcp_velocity.linear.y,
                ctrl.tcp_velocity.linear.z,
                ctrl.tcp_velocity.angular.x,
                ctrl.tcp_velocity.angular.y,
                ctrl.tcp_velocity.angular.z,
                ctrl.tcp_error[0],
                ctrl.tcp_error[1],
                ctrl.tcp_error[2],
                ctrl.tcp_error[3],
                ctrl.tcp_error[4],
                ctrl.tcp_error[5],
                joints[0],
                joints[1],
                joints[2],
                joints[3],
                joints[4],
                joints[5],
                joints[6],
                wrench.force.x,
                wrench.force.y,
                wrench.force.z,
                wrench.torque.x,
                wrench.torque.y,
                wrench.torque.z,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _stack_history(hist: Deque[np.ndarray], obs_steps: int) -> np.ndarray:
        if not hist:
            raise RuntimeError("Observation history is empty; cannot stack history.")

        values = list(hist)
        if len(values) < obs_steps:
            pad = [values[0]] * (obs_steps - len(values))
            values = pad + values

        return np.stack(values[-obs_steps:], axis=0)

    def _update_observation_history(self, obs_msg: Observation) -> None:
        left = self._image_to_chw_float(obs_msg.left_image, self.camera_scale)
        center = self._image_to_chw_float(obs_msg.center_image, self.camera_scale)
        right = self._image_to_chw_float(obs_msg.right_image, self.camera_scale)
        state = self._state_vector_32(obs_msg)

        self.left_hist.append(left)
        self.center_hist.append(center)
        self.right_hist.append(right)
        self.state_hist.append(state)

    def _build_model_obs(self) -> Dict[str, torch.Tensor]:
        left = self._stack_history(self.left_hist, self.obs_steps)
        center = self._stack_history(self.center_hist, self.obs_steps)
        right = self._stack_history(self.right_hist, self.obs_steps)
        state = self._stack_history(self.state_hist, self.obs_steps)

        left = self.obs_normalizers["left_camera"].normalize(left)
        center = self.obs_normalizers["center_camera"].normalize(center)
        right = self.obs_normalizers["right_camera"].normalize(right)
        state = self.obs_normalizers["state"].normalize(state)

        obs = {
            "left_camera": torch.tensor(
                left, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "center_camera": torch.tensor(
                center, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "right_camera": torch.tensor(
                right, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
            "state": torch.tensor(
                state, device=self.device, dtype=torch.float32
            ).unsqueeze(0),
        }
        return obs

    def _sample_action_chunk(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        act_0 = torch.randn(
            (1, self.horizon, self.act_dim),
            device=self.device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            act_normed = self.policy.sample(
                act_0=act_0,
                obs=obs_dict,
                num_steps=int(self.config.optimization.num_steps),
                use_ema=True,
            )

        act_normed_np = act_normed.detach().cpu().numpy()
        act = self.action_normalizer.unnormalize(act_normed_np)

        start = self.obs_steps - 1
        end = start + self.act_steps
        return act[0, start:end, :]

    def _next_action(self, obs_dict: Dict[str, torch.Tensor]) -> np.ndarray:
        if not self.pending_actions:
            chunk = self._sample_action_chunk(obs_dict)
            for row in chunk:
                self.pending_actions.append(row.astype(np.float32, copy=False))

        return self.pending_actions.popleft()

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
        **kwargs,
    ):
        self.get_logger().info(f"RunMIP.insert_cable() enter. Task: {task}")

        self.left_hist.clear()
        self.center_hist.clear()
        self.right_hist.clear()
        self.state_hist.clear()
        self.pending_actions.clear()

        start_time = time.time()

        while time.time() - start_time < 120.0:
            loop_start = time.time()
            observation_msg = get_observation()

            if observation_msg is None:
                self.get_logger().info("No observation received.")
                continue

            self._update_observation_history(observation_msg)
            obs_dict = self._build_model_obs()
            action = self._next_action(obs_dict)

            if action.shape[0] < 6:
                self.get_logger().error(
                    f"Expected at least 6 action dimensions, got {action.shape[0]}"
                )
                return False

            self.get_logger().info(f"Action: {action}")
            self.set_delta_pose_target_from_components(
                move_robot=move_robot,
                delta_position_xyz=action[:3],
                delta_rotation_xyz=action[3:6],
                max_translation=self.max_translation_delta,
                max_rotation=self.max_rotation_delta,
                deadband_translation=self.translation_deadband,
                deadband_rotation=self.rotation_deadband,
            )
            send_feedback("in progress...")

            elapsed = time.time() - loop_start
            time.sleep(max(0, 0.25 - elapsed))

        self.get_logger().info("RunMIP.insert_cable() exiting...")
        return True
