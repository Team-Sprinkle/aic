#!/usr/bin/env python3
"""Probe AIC Isaac target-geometry rewards with a privileged oracle controller."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--task_family", choices=["sfp_to_nic", "sc_to_sc"], default="sfp_to_nic")
parser.add_argument("--target_port_index", type=int, default=0)
parser.add_argument("--target_body", default="sfp_tip_link")
parser.add_argument("--max_delta", type=float, default=0.02)
parser.add_argument("--max_rotation", type=float, default=0.08)
parser.add_argument("--descent_start_step", type=int, default=90)
parser.add_argument("--approach_offset", type=float, nargs=3, default=None)
parser.add_argument("--insert_offset", type=float, nargs=3, default=None)
parser.add_argument("--record_cameras", action="store_true")
parser.add_argument("--video_dir", type=Path)
parser.add_argument("--video_fps", type=float, default=20.0)
parser.add_argument("--distance_std", type=float, default=0.02)
parser.add_argument("--close_sigma", type=float, default=0.01)
parser.add_argument("--reaching_threshold", type=float, default=0.01)
parser.add_argument("--orientation_weight", type=float, default=0.0)
parser.add_argument(
    "--disable_semantic_orientation_offsets",
    action="store_true",
    help="Use root/body quaternions directly while still using semantic position offsets.",
)
parser.add_argument(
    "--use_semantic_orientation_offsets",
    action="store_true",
    help="Apply the Gazebo SDF port/tip orientation frames in addition to semantic position offsets.",
)
parser.add_argument(
    "--target_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Target-local XYZ offset from target rigid-object root to reward/controller target. "
        "Defaults to (0,0,0) for sfp_to_nic/nic_card and calibrated SC-port offset for sc_to_sc."
    ),
)
parser.add_argument(
    "--body_position_offset",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help=(
        "Body-local XYZ offset from target_body origin to the point measured against the target. "
        "Defaults to the SFP module tip for sfp_to_nic and body origin for sc_to_sc."
    ),
)
parser.add_argument(
    "--controller",
    choices=("target_body_world", "cheatcode_tcp"),
    default="cheatcode_tcp",
    help=(
        "target_body_world is the old direct world-frame body delta probe. "
        "cheatcode_tcp computes a wrist target that aligns the target body to the port, "
        "then uses the same TCP-to-Isaac action conversion as online SERL."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils import math as math_utils

import aic_task.tasks  # noqa: F401

CAMERA_NAMES = ("center_camera", "left_camera", "right_camera")
SFP_PORT_LOCAL = {
    0: (0.01295, -0.031572, 0.00501),
    1: (-0.01025, -0.031572, 0.00501),
}
SFP_PORT_RPY = (4.69895, 0.0, 0.0)
SFP_PORT_ENTRANCE_LOCAL = (0.0, 0.0, -0.0458)
SFP_TIP_LOCAL = (0.0, -0.02365, 0.0)
SFP_TIP_RPY = (1.5708, 0.0, 0.0)


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matvec(mat: tuple[tuple[float, float, float], ...], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(mat[row][col] * vec[col] for col in range(3)) for row in range(3))


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        cy * cr * cp + sy * sr * sp,
        cy * sr * cp - sy * cr * sp,
        cy * cr * sp + sy * sr * cp,
        sy * cr * cp - cy * sr * sp,
    )


def _offset_from_port_frame(offset: tuple[float, float, float]) -> tuple[float, float, float]:
    port = SFP_PORT_LOCAL[int(args_cli.target_port_index)]
    rotated = _matvec(_rpy_matrix(*SFP_PORT_RPY), offset)
    return tuple(port[i] + rotated[i] for i in range(3))


def _target_scene_name(index: int) -> str:
    if args_cli.task_family == "sfp_to_nic":
        return "nic_card"
    if args_cli.task_family != "sc_to_sc":
        raise ValueError(f"Unsupported task_family: {args_cli.task_family}")
    if index == 0:
        return "sc_port"
    if index == 1:
        return "sc_port_2"
    raise ValueError("target_port_index must be 0 or 1")


def _target_position_offset() -> tuple[float, float, float]:
    if args_cli.target_position_offset is not None:
        return tuple(float(v) for v in args_cli.target_position_offset)
    if args_cli.task_family == "sfp_to_nic":
        return _offset_from_port_frame(SFP_PORT_ENTRANCE_LOCAL)
    return (0.093, 0.140, 0.020)


def _controller_approach_offset() -> tuple[float, float, float]:
    if args_cli.approach_offset is not None:
        return tuple(float(v) for v in args_cli.approach_offset)
    if args_cli.task_family == "sfp_to_nic":
        return _offset_from_port_frame(SFP_PORT_ENTRANCE_LOCAL)
    return _target_position_offset()


def _controller_insert_offset() -> tuple[float, float, float]:
    if args_cli.insert_offset is not None:
        return tuple(float(v) for v in args_cli.insert_offset)
    return _target_position_offset()


def _body_position_offset() -> tuple[float, float, float]:
    if args_cli.body_position_offset is not None:
        return tuple(float(v) for v in args_cli.body_position_offset)
    if args_cli.task_family == "sfp_to_nic":
        if args_cli.target_body == "sfp_tip_link":
            return (0.0, 0.0, 0.0)
        return SFP_TIP_LOCAL
    return (0.0, 0.0, 0.0)


def _target_orientation_offset() -> tuple[float, float, float, float] | None:
    if args_cli.disable_semantic_orientation_offsets:
        return None
    if args_cli.task_family == "sfp_to_nic":
        return _quat_from_rpy(*SFP_PORT_RPY)
    return None


def _body_orientation_offset() -> tuple[float, float, float, float] | None:
    if args_cli.disable_semantic_orientation_offsets:
        return None
    if args_cli.task_family == "sfp_to_nic":
        if args_cli.target_body == "sfp_tip_link":
            return _quat_from_rpy(0.0, math.pi, 0.0)
        return _quat_from_rpy(*SFP_TIP_RPY)
    return None


def _configure_rewards(env_cfg) -> dict[str, object]:
    rewards = env_cfg.rewards
    for name in (
        "end_effector_position_tracking",
        "end_effector_position_tracking_fine_grained",
        "end_effector_position_tracking_exp",
        "end_effector_orientation_tracking",
        "end_effector_orientation_tracking_fine_grained",
        "reaching_bonus",
    ):
        getattr(rewards, name).weight = 0.0

    target_scene = _target_scene_name(args_cli.target_port_index)
    target_offset = _target_position_offset()
    body_offset = _body_position_offset()
    target_orientation_offset = _target_orientation_offset()
    body_orientation_offset = _body_orientation_offset()
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_orientation_tanh",
        "target_reaching_bonus",
        "target_lateral_error",
    ):
        term = getattr(rewards, name)
        term.params["body_cfg"].body_names = [args_cli.target_body]
        term.params["target_cfg"].name = target_scene
        if "target_position_offset" in term.params:
            term.params["target_position_offset"] = target_offset
        if "body_position_offset" in term.params:
            term.params["body_position_offset"] = body_offset
        if "target_orientation_offset" in term.params:
            term.params["target_orientation_offset"] = target_orientation_offset
        if "body_orientation_offset" in term.params:
            term.params["body_orientation_offset"] = body_orientation_offset

    rewards.target_distance_tanh.weight = 0.5
    rewards.target_distance_exp.weight = 0.3
    rewards.target_orientation_tanh.weight = float(args_cli.orientation_weight)
    rewards.target_reaching_bonus.weight = 1.0
    rewards.target_lateral_error.weight = 0.0
    rewards.target_distance_tanh.params["std"] = float(args_cli.distance_std)
    rewards.target_distance_exp.params["sigma"] = float(args_cli.close_sigma)
    rewards.target_reaching_bonus.params["threshold"] = float(args_cli.reaching_threshold)
    return {
        "target_scene": target_scene,
        "target_body": args_cli.target_body,
        "target_position_offset": [float(v) for v in target_offset],
        "body_position_offset": [float(v) for v in body_offset],
        "target_orientation_offset": None
        if target_orientation_offset is None
        else [float(v) for v in target_orientation_offset],
        "body_orientation_offset": None
        if body_orientation_offset is None
        else [float(v) for v in body_orientation_offset],
        "distance_std": float(args_cli.distance_std),
        "close_sigma": float(args_cli.close_sigma),
        "reaching_threshold": float(args_cli.reaching_threshold),
        "distance_weight": 0.5,
        "close_weight": 0.3,
        "orientation_weight": float(args_cli.orientation_weight),
        "reaching_weight": 1.0,
        "lateral_weight": 0.0,
        "command_pose_rewards_disabled": True,
    }


def _named_index(names: list[str], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"{name!r} not found in available names: {names}") from exc


def _tcp_delta_action_to_isaac_base_action(env, tcp_action: torch.Tensor) -> torch.Tensor:
    """Convert Gazebo/LeRobot gripper-tcp delta actions to Isaac IK root-frame deltas."""
    robot = env.unwrapped.scene["robot"]
    body_names = list(getattr(robot, "body_names", []))
    wrist_index = _named_index(body_names, "wrist_3_link")
    data = robot.data
    _, tcp_quat_b = math_utils.subtract_frame_transforms(
        data.root_pos_w,
        data.root_quat_w,
        data.body_pos_w[:, wrist_index],
        data.body_quat_w[:, wrist_index],
    )
    delta_pos_b = math_utils.quat_apply(tcp_quat_b, tcp_action[:, :3])
    delta_rot_b = math_utils.quat_apply(tcp_quat_b, tcp_action[:, 3:6])
    return torch.cat([delta_pos_b, delta_rot_b], dim=-1)


def _clip_by_norm(vec: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vec)
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp_min(1.0e-9)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return vec * scale


def _offset_quat_tensor(
    offset: tuple[float, float, float, float] | None,
    *,
    like: torch.Tensor,
) -> torch.Tensor:
    if offset is None:
        out = torch.zeros_like(like)
        out[:, 0] = 1.0
        return out
    return torch.tensor(offset, dtype=like.dtype, device=like.device).reshape(1, 4).expand_as(like)


def _smoothstep(fraction: float) -> float:
    value = max(0.0, min(1.0, float(fraction)))
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _tensor_xyz(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)[:3].detach().cpu()]


def _tensor_quat(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.reshape(-1)[:4].detach().cpu()]


def _lerp_offset(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    return tuple(float(a + (b - a) * fraction) for a, b in zip(start, end))


class VideoRecorder:
    def __init__(self, output_dir: Path, fps: float):
        self.output_dir = output_dir
        self.fps = float(fps)
        self._writers: dict[str, object] = {}
        self._paths: dict[str, Path] = {}

    def add_scene(self, env) -> None:
        sensors = env.unwrapped.scene.sensors
        for camera_name in CAMERA_NAMES:
            if camera_name not in sensors:
                continue
            output = sensors[camera_name].data.output
            if "rgb" not in output:
                continue
            self.add_frame(camera_name, output["rgb"])

    def add_frame(self, camera_name: str, image: torch.Tensor) -> None:
        import cv2

        frame = image.detach().cpu()
        if frame.ndim == 4:
            frame = frame[0]
        array = frame.numpy()
        if array.ndim == 3 and array.shape[0] in (3, 4):
            array = array[:3].transpose(1, 2, 0)
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            array = array[..., :3]
        else:
            raise ValueError(f"Expected RGB camera frame for {camera_name}, got {tuple(frame.shape)}")
        if array.dtype != "uint8":
            if array.max() <= 2.0:
                array = array * 255.0
            array = array.clip(0, 255).astype("uint8")
        height, width = array.shape[:2]
        writer = self._writers.get(camera_name)
        if writer is None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"isaac_{camera_name}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open video writer for {path}")
            self._writers[camera_name] = writer
            self._paths[camera_name] = path
        writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def close(self) -> dict[str, str]:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        return {name: str(path) for name, path in sorted(self._paths.items())}


def _cheatcode_tcp_action(
    *,
    robot,
    target,
    wrist_index: int,
    body_index: int,
    controller_target_offset: tuple[float, float, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a TCP-frame delta that moves the wrist to align body with target.

    This mirrors the ROS CheatCode geometry: compute the rigid transform that
    would align the plug/body frame to the target-port frame, apply that same
    transform to the controlled wrist/TCP frame, then command the relative TCP
    delta. The returned action still needs Isaac's TCP-to-base conversion.
    """
    body_pos_w = robot.data.body_pos_w[:, body_index]
    body_quat_w = robot.data.body_quat_w[:, body_index]
    wrist_pos_w = robot.data.body_pos_w[:, wrist_index]
    wrist_quat_w = robot.data.body_quat_w[:, wrist_index]
    target_pos_w = target.data.root_pos_w
    target_quat_w = target.data.root_quat_w
    body_frame_quat_w = math_utils.quat_mul(
        body_quat_w,
        _offset_quat_tensor(_body_orientation_offset(), like=body_quat_w),
    )
    target_frame_quat_w = math_utils.quat_mul(
        target_quat_w,
        _offset_quat_tensor(_target_orientation_offset(), like=target_quat_w),
    )
    target_offset = torch.tensor(
        controller_target_offset,
        dtype=target_pos_w.dtype,
        device=target_pos_w.device,
    ).reshape(1, 3)
    body_offset = torch.tensor(
        _body_position_offset(),
        dtype=body_pos_w.dtype,
        device=body_pos_w.device,
    ).reshape(1, 3)
    body_point_w = body_pos_w + math_utils.quat_apply(body_quat_w, body_offset.expand_as(body_pos_w))
    target_pos_w = target_pos_w + math_utils.quat_apply(target_quat_w, target_offset.expand_as(target_pos_w))

    if args_cli.max_rotation <= 0.0:
        q_diff = torch.zeros_like(body_frame_quat_w)
        q_diff[:, 0] = 1.0
    else:
        q_body_inv = math_utils.quat_inv(body_frame_quat_w)
        q_diff = math_utils.quat_mul(target_frame_quat_w, q_body_inv)
    target_wrist_quat_w = math_utils.quat_mul(q_diff, wrist_quat_w)
    target_wrist_pos_w = target_pos_w + (wrist_pos_w - body_point_w)

    delta_pos_tcp = math_utils.quat_apply_inverse(wrist_quat_w, target_wrist_pos_w - wrist_pos_w)
    delta_quat_tcp = math_utils.quat_mul(math_utils.quat_inv(wrist_quat_w), target_wrist_quat_w)
    delta_rot_tcp = math_utils.axis_angle_from_quat(delta_quat_tcp)

    clipped_pos_tcp = _clip_by_norm(delta_pos_tcp, args_cli.max_delta)
    clipped_rot_tcp = _clip_by_norm(delta_rot_tcp, args_cli.max_rotation)
    tcp_action = torch.cat([clipped_pos_tcp, clipped_rot_tcp], dim=-1)
    diagnostics = {
        "tcp_delta_pos_norm_m": float(torch.linalg.norm(delta_pos_tcp, dim=-1)[0].detach().cpu()),
        "tcp_delta_rot_norm_rad": float(torch.linalg.norm(delta_rot_tcp, dim=-1)[0].detach().cpu()),
        "cmd_tcp_pos_norm_m": float(torch.linalg.norm(clipped_pos_tcp, dim=-1)[0].detach().cpu()),
        "cmd_tcp_rot_norm_rad": float(torch.linalg.norm(clipped_rot_tcp, dim=-1)[0].detach().cpu()),
        "controller_target_offset_x": float(controller_target_offset[0]),
        "controller_target_offset_y": float(controller_target_offset[1]),
        "controller_target_offset_z": float(controller_target_offset[2]),
    }
    return tcp_action, diagnostics


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.center_rgb = None
    env_cfg.observations.policy.left_rgb = None
    env_cfg.observations.policy.right_rgb = None
    env_cfg.actions.arm_action.scale = 1.0
    reward_config = _configure_rewards(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs
    video_recorder = (
        VideoRecorder(args_cli.video_dir or args_cli.output.parent / "isaac_camera_videos", args_cli.video_fps)
        if args_cli.record_cameras
        else None
    )
    video_paths: dict[str, str] = {}
    if video_recorder is not None:
        video_recorder.add_scene(env)
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    target = unwrapped.scene[reward_config["target_scene"]]
    body_names = list(robot.body_names)
    body_index = _named_index(body_names, args_cli.target_body)
    wrist_index = _named_index(body_names, "wrist_3_link")

    def metrics(step: int, reward_value: torch.Tensor | None = None) -> dict[str, float | int | None]:
        body_pos = robot.data.body_pos_w[:, body_index]
        body_quat = robot.data.body_quat_w[:, body_index]
        target_pos = target.data.root_pos_w
        target_quat = target.data.root_quat_w
        body_frame_quat = math_utils.quat_mul(
            body_quat,
            _offset_quat_tensor(_body_orientation_offset(), like=body_quat),
        )
        target_frame_quat = math_utils.quat_mul(
            target_quat,
            _offset_quat_tensor(_target_orientation_offset(), like=target_quat),
        )
        target_offset = torch.tensor(
            _target_position_offset(),
            dtype=target_pos.dtype,
            device=target_pos.device,
        ).reshape(1, 3)
        body_offset = torch.tensor(
            _body_position_offset(),
            dtype=body_pos.dtype,
            device=body_pos.device,
        ).reshape(1, 3)
        body_point = body_pos + math_utils.quat_apply(body_quat, body_offset.expand_as(body_pos))
        target_pos = target_pos + math_utils.quat_apply(target_quat, target_offset.expand_as(target_pos))
        distance = torch.linalg.norm(body_point - target_pos, dim=1)
        body_minus_target_local = math_utils.quat_apply_inverse(target_quat, body_point - target_pos)
        dot = torch.abs(torch.sum(body_frame_quat * target_frame_quat, dim=1)).clamp(0.0, 1.0)
        orientation = 2.0 * torch.acos(dot)
        distance_reward = 1.0 - torch.tanh(distance / float(args_cli.distance_std))
        close_reward = torch.exp(-(distance * distance) / (float(args_cli.close_sigma) ** 2))
        orientation_reward = 1.0 - torch.tanh(orientation / 0.25)
        expected = (
            0.5 * distance_reward
            + 0.3 * close_reward
            + float(args_cli.orientation_weight) * orientation_reward
            + (distance < float(args_cli.reaching_threshold)).float()
        )
        return {
            "step": step,
            "reward": None if reward_value is None else float(reward_value.reshape(-1)[0].detach().cpu()),
            "distance_m": float(distance[0].detach().cpu()),
            "orientation_error_rad": float(orientation[0].detach().cpu()),
            "distance_reward": float(distance_reward[0].detach().cpu()),
            "close_reward": float(close_reward[0].detach().cpu()),
            "orientation_reward": float(orientation_reward[0].detach().cpu()),
            "expected_target_reward_before_safety": float(expected[0].detach().cpu()),
            "body_minus_target_local_x": float(body_minus_target_local[0, 0].detach().cpu()),
            "body_minus_target_local_y": float(body_minus_target_local[0, 1].detach().cpu()),
            "body_minus_target_local_z": float(body_minus_target_local[0, 2].detach().cpu()),
            "target_root_pos_w": _tensor_xyz(target.data.root_pos_w[0]),
            "target_root_quat_w": _tensor_quat(target.data.root_quat_w[0]),
            "reward_target_pos_w": _tensor_xyz(target_pos[0]),
            "body_root_pos_w": _tensor_xyz(body_pos[0]),
            "body_root_quat_w": _tensor_quat(body_quat[0]),
            "reward_body_pos_w": _tensor_xyz(body_point[0]),
        }

    rows = [metrics(0)]
    rows[-1]["controller"] = args_cli.controller
    print(json.dumps(rows[-1]), flush=True)
    for step in range(1, args_cli.steps + 1):
        diagnostics: dict[str, float] = {}
        if args_cli.controller == "target_body_world":
            target_pos = target.data.root_pos_w[:, :3]
            target_offset = torch.tensor(
                _target_position_offset(),
                dtype=target_pos.dtype,
                device=target_pos.device,
            ).reshape(1, 3)
            target_pos = target_pos + math_utils.quat_apply(
                target.data.root_quat_w, target_offset.expand_as(target_pos)
            )
            body_pos = robot.data.body_pos_w[:, body_index, :3]
            body_quat = robot.data.body_quat_w[:, body_index]
            body_offset = torch.tensor(
                _body_position_offset(),
                dtype=body_pos.dtype,
                device=body_pos.device,
            ).reshape(1, 3)
            body_point = body_pos + math_utils.quat_apply(body_quat, body_offset.expand_as(body_pos))
            delta = target_pos - body_point
            action = torch.zeros((args_cli.num_envs, 6), device=unwrapped.device)
            action[:, :3] = _clip_by_norm(delta, args_cli.max_delta)
            diagnostics = {
                "world_delta_norm_m": float(torch.linalg.norm(delta, dim=-1)[0].detach().cpu()),
                "cmd_world_pos_norm_m": float(torch.linalg.norm(action[:, :3], dim=-1)[0].detach().cpu()),
            }
        else:
            if step < args_cli.descent_start_step:
                controller_target_offset = _controller_approach_offset()
            else:
                denom = max(1, args_cli.steps - args_cli.descent_start_step)
                controller_target_offset = _lerp_offset(
                    _controller_approach_offset(),
                    _controller_insert_offset(),
                    _smoothstep((step - args_cli.descent_start_step) / denom),
                )
            tcp_action, diagnostics = _cheatcode_tcp_action(
                robot=robot,
                target=target,
                wrist_index=wrist_index,
                body_index=body_index,
                controller_target_offset=controller_target_offset,
            )
            action = _tcp_delta_action_to_isaac_base_action(env, tcp_action)
        _, reward, terminated, truncated, _ = env.step(action)
        if video_recorder is not None:
            video_recorder.add_scene(env)
        row = metrics(step, reward)
        row["controller"] = args_cli.controller
        row.update(diagnostics)
        rows.append(row)
        print(json.dumps(row), flush=True)
        if bool(torch.logical_or(terminated, truncated).any()):
            break

    exact = {
        "distance_m": 0.0,
        "orientation_error_rad": 0.0,
        "distance_reward": 1.0,
        "close_reward": 1.0,
        "orientation_reward": 1.0,
        "expected_target_reward_before_safety": 1.8 + float(args_cli.orientation_weight),
    }
    summary = {
        "reward_config": reward_config,
        "rows": rows,
        "best": max(rows, key=lambda row: float(row["expected_target_reward_before_safety"])),
        "exact_alignment_expected": exact,
        "video_paths": video_paths,
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if video_recorder is not None:
        video_paths = video_recorder.close()
        summary["video_paths"] = video_paths
        args_cli.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    env.close()
    print("SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    status = 0
    try:
        main()
    except BaseException:
        status = 1
        raise
    finally:
        simulation_app.close()
    raise SystemExit(status)
