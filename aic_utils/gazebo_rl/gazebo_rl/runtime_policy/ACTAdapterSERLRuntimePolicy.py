from __future__ import annotations

import json
import os
from typing import Any

from aic_model.policy import (
    DEFAULT_CARTESIAN_DAMPING,
    DEFAULT_CARTESIAN_STIFFNESS,
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
    build_pose_from_vectors,
    pose_to_position_motion_update,
)

from gazebo_rl.action import delta_tcp_action_from_array
from gazebo_rl.observation import observation_to_dict
from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy, task_vector_from_context


class ACTAdapterSERLRuntimePolicy(Policy):
    """AIC runtime policy for ACT-adapter SERL checkpoints without trainer IPC."""

    def __init__(self, parent_node):
        self._policy: ACTAdapterSERLGazeboPolicy | None = None
        self._step_count = 0
        super().__init__(parent_node)

    def _param(self, name: str, default: Any) -> Any:
        self._parent_node.declare_parameter(name, default)
        return self._parent_node.get_parameter(name).value

    def _bool_param_env(self, param_name: str, env_name: str, default: bool) -> bool:
        raw = os.environ.get(env_name)
        if raw is not None:
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(self._param(param_name, default))

    def _float_param_env(self, param_name: str, env_name: str, default: float) -> float:
        raw = os.environ.get(env_name)
        return float(raw) if raw is not None else float(self._param(param_name, default))

    def _str_param_env(self, param_name: str, env_name: str, default: str = "") -> str:
        raw = os.environ.get(env_name)
        return raw if raw is not None else str(self._param(param_name, default))

    def _load_policy(self, task: Any) -> ACTAdapterSERLGazeboPolicy:
        if self._policy is not None:
            return self._policy
        checkpoint = self._str_param_env("serl_checkpoint", "AIC_SERL_CHECKPOINT")
        act_torchscript = self._str_param_env("act_torchscript", "AIC_ACT_TORCHSCRIPT")
        if not checkpoint:
            raise RuntimeError("Missing SERL checkpoint. Set ROS param serl_checkpoint or env AIC_SERL_CHECKPOINT.")
        if not act_torchscript:
            raise RuntimeError("Missing ACT TorchScript. Set ROS param act_torchscript or env AIC_ACT_TORCHSCRIPT.")
        task_context_json = self._str_param_env("task_context_json", "AIC_TASK_CONTEXT_JSON", "")
        task_context: dict[str, Any] | None = json.loads(task_context_json) if task_context_json else None
        if task_context is None:
            task_context = _task_context_from_msg(task)
        task_vector = None
        if task_context:
            task_vector = task_vector_from_context(task_context_json=task_context)
        self._policy = ACTAdapterSERLGazeboPolicy(
            checkpoint,
            act_torchscript=act_torchscript,
            device=self._str_param_env("device", "AIC_SERL_DEVICE", "cpu"),
            allow_zero_images=self._bool_param_env("allow_zero_images", "AIC_SERL_ALLOW_ZERO_IMAGES", False),
            adapter_delta_clip=self._float_param_env("adapter_delta_clip", "AIC_SERL_ADAPTER_DELTA_CLIP", 0.05),
            action_clip=self._float_param_env("action_clip", "AIC_SERL_ACTION_CLIP", 0.05),
            task_vector=task_vector,
        )
        self.get_logger().info(
            "Loaded ACT-adapter SERL runtime policy "
            f"checkpoint={checkpoint} act_torchscript={act_torchscript}"
        )
        return self._policy

    def _send_delta_action(self, move_robot: MoveRobotCallback, raw_action: Any) -> None:
        delta = delta_tcp_action_from_array(raw_action)
        motion_update = pose_to_position_motion_update(
            build_pose_from_vectors(delta.delta_position_xyz, delta.delta_quaternion_xyzw),
            stamp=self._parent_node.get_clock().now().to_msg(),
            frame_id="gripper/tcp",
            stiffness=DEFAULT_CARTESIAN_STIFFNESS,
            damping=DEFAULT_CARTESIAN_DAMPING,
        )
        result = move_robot(motion_update=motion_update)
        if result is False:
            raise RuntimeError("move_robot rejected the ACT-adapter SERL motion update")

    def insert_cable(
        self,
        task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        policy = self._load_policy(task)
        include_images = self._bool_param_env("include_images", "AIC_SERL_INCLUDE_IMAGES", True)
        ground_truth = self._bool_param_env("ground_truth", "AIC_SERL_GROUND_TRUTH", False)
        max_steps = int(self._float_param_env("max_steps", "AIC_SERL_MAX_STEPS", 600))
        command_dt_sec = self._float_param_env("command_dt_sec", "AIC_SERL_COMMAND_DT_SEC", 0.05)
        self._step_count = 0
        while self._step_count < max_steps:
            try:
                obs_msg = get_observation()
                obs = observation_to_dict(
                    obs_msg,
                    task=task,
                    step_count=self._step_count,
                    tf_buffer=getattr(self._parent_node, "_tf_buffer", None),
                    ground_truth=ground_truth,
                    include_images=include_images,
                    logger=lambda msg: self.get_logger().warn(msg),
                )
                action = policy.act(obs)
                self._send_delta_action(move_robot, action)
                if getattr(policy, "last_action_components", None):
                    self.get_logger().debug(f"ACT-adapter SERL action metrics: {policy.last_action_components}")
            except Exception as ex:
                send_feedback(f"ACT-adapter SERL runtime policy failed at step {self._step_count}: {ex}")
                self.get_logger().error(str(ex))
                return False
            self._step_count += 1
            self.sleep_for(command_dt_sec)
        return True


def _task_context_from_msg(task: Any) -> dict[str, Any]:
    port_name = str(getattr(task, "port_name", "") or "")
    target_module = str(getattr(task, "target_module_name", "") or "")
    family = "sfp_to_nic" if "nic" in target_module or "sfp" in port_name else "sc_to_sc"
    return {
        "task_family": family,
        "target_port_index": _trailing_int(port_name, default=0),
        "target_card_index": _trailing_int(target_module, default=0 if family == "sfp_to_nic" else -1),
        "target_card_valid": 1 if family == "sfp_to_nic" else 0,
    }


def _trailing_int(value: str, *, default: int) -> int:
    digits = []
    for ch in reversed(value):
        if ch.isdigit():
            digits.append(ch)
        elif digits:
            break
    return int("".join(reversed(digits))) if digits else int(default)
