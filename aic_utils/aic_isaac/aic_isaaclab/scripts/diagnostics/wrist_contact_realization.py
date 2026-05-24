#!/usr/bin/env python3
"""Standalone wrist IK contact-realization diagnostic for SFP insertion.

This bypasses ACT/SERL completely.  It creates the Isaac Lab AIC task, uses the
configured Differential IK wrist action path, drives the semantic tip to a
shallow positive insertion depth, then applies direct wrist translation or
rotation probes while logging realized ``sfp_tip_link`` and ``sfp_module_link``
motion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=2)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="aic/outputs/agentic_reward_curriculum_20260524_wrist_contact/runs")
parser.add_argument("--run_name", default="wrist_contact_realization")
parser.add_argument("--episode_config_dir", type=str, default=None)
parser.add_argument(
    "--override_start_signed_depth_m",
    type=float,
    default=float("nan"),
    help=(
        "If finite, generate a temporary episode-config copy whose semantic tip reset starts "
        "at this signed depth relative to the entrance. Positive values start inside the cage."
    ),
)
parser.add_argument("--override_start_lateral_m", type=float, default=0.0003)
parser.add_argument(
    "--override_start_orientation_wxyz",
    type=float,
    nargs=4,
    default=None,
    help="Optional reset-body orientation to write into generated shallow/depth override episode configs.",
)
parser.add_argument(
    "--override_start_orientation_rotvec_world",
    type=float,
    nargs=3,
    default=None,
    help="Optional world-frame rotation vector composed onto the generated reset-body orientation.",
)
parser.add_argument(
    "--override_start_tip_orientation_wxyz",
    type=float,
    nargs=4,
    default=None,
    help=(
        "Desired semantic tip orientation for generated resets. With "
        "--derive_reset_orientation_from_tip, the reset-body orientation is derived by preserving "
        "the source episode's gripper-to-tip orientation relationship."
    ),
)
parser.add_argument(
    "--derive_reset_orientation_from_tip",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument(
    "--derive_reset_position_from_orientation",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "When generating an override reset with a changed reset-body orientation, recompute the "
        "reset-body position from the source local reset-body-to-tip offset instead of reusing "
        "the source world offset. This prevents rotation-induced semantic tip lateral sweep."
    ),
)
parser.add_argument("--episode_length_s", type=float, default=12.0)
parser.add_argument("--near_gate_reset_max_iterations", type=int, default=0)
parser.add_argument("--near_gate_reset_position_tolerance", type=float, default=0.0)
parser.add_argument("--near_gate_reset_orientation_tolerance", type=float, default=0.0)
parser.add_argument("--isaac_action_scale", type=float, default=1.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--target_reward_body", default="sfp_tip_link")
parser.add_argument("--target_reward_consistency_body", default="sfp_module_link")
parser.add_argument("--target_reward_orientation_axis_local", type=float, nargs=3, default=(0.0, 0.0, 1.0))
parser.add_argument("--target_reward_consistency_axial_std", type=float, default=0.0025)
parser.add_argument("--target_reward_consistency_lateral_sigma", type=float, default=0.0020)
parser.add_argument("--approach_steps", type=int, default=95)
parser.add_argument("--probe_steps", type=int, default=30)
parser.add_argument("--shallow_depth_m", type=float, default=0.0050)
parser.add_argument("--approach_lateral_step_m", type=float, default=0.00018)
parser.add_argument("--approach_axial_step_m", type=float, default=0.00012)
parser.add_argument("--approach_lateral_gate_m", type=float, default=0.00035)
parser.add_argument("--approach_lateral_sign", type=float, default=1.0)
parser.add_argument("--approach_axial_sign", type=float, default=1.0)
parser.add_argument(
    "--probe",
    choices=[
        "axis_backout",
        "axis_forward",
        "rotation_axes",
        "rotation_axis",
        "orientation_servo_best",
        "pose_hold",
        "pose_hold_orientation_servo_best",
        "pose_hold_rotation_axis",
        "target_tip_stabilize",
        "target_tip_fixed_rotation_axis",
    ],
    default="axis_backout",
)
parser.add_argument("--probe_translation_step_m", type=float, default=0.00020)
parser.add_argument("--probe_rotation_step_rad", type=float, default=0.004)
parser.add_argument("--probe_rotation_axis", choices=["x", "y", "z"], default="y")
parser.add_argument("--probe_rotation_sign", type=float, default=1.0)
parser.add_argument("--probe_lateral_correction_step_m", type=float, default=0.0)
parser.add_argument("--probe_axial_step_m", type=float, default=0.0)
parser.add_argument("--probe_axial_lateral_gate_m", type=float, default=0.0005)
parser.add_argument(
    "--target_tip_stabilize_axial_step_m",
    type=float,
    default=float("nan"),
    help="Axial correction clip for target_tip_stabilize; defaults to --probe_translation_step_m.",
)
parser.add_argument(
    "--target_tip_stabilize_lateral_step_m",
    type=float,
    default=float("nan"),
    help="Lateral correction clip for target_tip_stabilize; defaults to --probe_translation_step_m.",
)
parser.add_argument(
    "--target_tip_stabilize_rotation_compensation_clip_m",
    type=float,
    default=0.0010,
    help="Clip for translation compensation that counteracts rotation-induced semantic tip sweep.",
)
parser.add_argument(
    "--target_tip_stabilize_orientation_gate_lateral_m",
    type=float,
    default=0.0007,
    help="Only apply semantic orientation correction when current lateral error is below this threshold.",
)
parser.add_argument(
    "--target_tip_stabilize_orientation_gate_depth_m",
    type=float,
    default=0.0060,
    help="Only apply semantic orientation correction when current signed depth is above this threshold.",
)
parser.add_argument(
    "--target_tip_stabilize_inward_bias_m",
    type=float,
    default=0.0,
    help="Small positive axis bias added while near the target; useful to preload contact at reset.",
)
parser.add_argument(
    "--target_tip_stabilize_goal_depth_m",
    type=float,
    default=float("nan"),
    help=(
        "If finite, target the semantic tip to entrance + this signed depth along the insertion "
        "axis instead of holding the post-reset tip position. Use 0.008 for the strict seated depth."
    ),
)
parser.add_argument(
    "--probe_orientation_lateral_penalty",
    type=float,
    default=0.0,
    help="Tie-break penalty on predicted lateral sweep for orientation_servo_best.",
)
parser.add_argument(
    "--pose_hold_body",
    default="gripper_tcp",
    help="Robot body whose post-reset world pose is held by the pose_hold probe.",
)
parser.add_argument(
    "--pose_hold_position_gain",
    type=float,
    default=1.0,
    help="Gain applied to pose_hold translational error before clipping by --probe_translation_step_m.",
)
parser.add_argument(
    "--pose_hold_rotation_gain",
    type=float,
    default=1.0,
    help="Gain applied to pose_hold rotational error before clipping by --probe_rotation_step_rad.",
)
parser.add_argument("--pose_hold_orientation_activation_depth_m", type=float, default=0.0075)
parser.add_argument("--pose_hold_orientation_activation_lateral_m", type=float, default=0.0005)
parser.add_argument("--pose_hold_orientation_activation_consistency", type=float, default=0.80)
parser.add_argument(
    "--pose_hold_orientation_step_rad",
    type=float,
    default=float("nan"),
    help="Semantic orientation trim step for pose_hold_orientation_servo_best; defaults to --probe_rotation_step_rad.",
)
parser.add_argument(
    "--pose_hold_fixed_rotation_step_rad",
    type=float,
    default=float("nan"),
    help="Fixed world-axis trim step for pose_hold_rotation_axis; defaults to --probe_rotation_step_rad.",
)
parser.add_argument(
    "--pose_hold_orientation_start_probe_step",
    type=int,
    default=1,
    help="Earliest zero-based probe step where pose_hold_orientation_servo_best may apply semantic orientation trim.",
)
parser.add_argument("--fix_isaac_ik_xy_sign", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--image_log_every", type=int, default=10)
parser.add_argument("--max_logged_image_steps", type=int, default=140)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def _quat_normalize_list(q: list[float] | tuple[float, float, float, float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    return [float(v) / norm for v in q]


def _quat_conj_list(q: list[float] | tuple[float, float, float, float]) -> list[float]:
    qn = _quat_normalize_list(q)
    return [qn[0], -qn[1], -qn[2], -qn[3]]


def _quat_mul_list(
    lhs: list[float] | tuple[float, float, float, float],
    rhs: list[float] | tuple[float, float, float, float],
) -> list[float]:
    lw, lx, ly, lz = _quat_normalize_list(lhs)
    rw, rx, ry, rz = _quat_normalize_list(rhs)
    return _quat_normalize_list(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_from_rotvec_list(rotvec: list[float] | tuple[float, float, float]) -> list[float]:
    angle = math.sqrt(sum(float(v) * float(v) for v in rotvec))
    if angle < 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    axis = [float(v) / angle for v in rotvec]
    half = 0.5 * angle
    return _quat_normalize_list([math.cos(half), *(math.sin(half) * v for v in axis)])


def _quat_apply_list(
    quat: list[float] | tuple[float, float, float, float],
    vec: list[float] | tuple[float, float, float],
) -> list[float]:
    def raw_mul(lhs: list[float], rhs: list[float]) -> list[float]:
        lw, lx, ly, lz = lhs
        rw, rx, ry, rz = rhs
        return [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]

    qn = _quat_normalize_list(quat)
    rotated = raw_mul(raw_mul(qn, [0.0, *[float(v) for v in vec]]), _quat_conj_list(qn))
    return [float(v) for v in rotated[1:]]


def _prepare_episode_config_dir() -> str | None:
    if not args_cli.episode_config_dir:
        return None
    source = Path(args_cli.episode_config_dir)
    if not math.isfinite(float(args_cli.override_start_signed_depth_m)):
        return str(source)
    source_episodes = source / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(f"episode_config_dir has no episodes directory: {source_episodes}")
    out_root = (
        Path("aic/outputs/agentic_reward_curriculum_20260524_wrist_contact/generated_episode_configs")
        / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_s{int(round(float(args_cli.override_start_signed_depth_m) * 1000.0))}mm"
    )
    episodes_out = out_root / "episodes"
    episodes_out.mkdir(parents=True, exist_ok=True)
    for idx, episode_path in enumerate(sorted(source_episodes.glob("episode_*.yaml"))[: max(int(args_cli.num_envs), 1)], start=1):
        data = yaml.safe_load(episode_path.read_text(encoding="utf-8"))
        scene = data.get("scene") or {}
        target = scene.get("target") or {}
        start = target.get("start_near_gate")
        if not isinstance(start, dict):
            start = scene.get("start_near_gate")
        if not isinstance(start, dict):
            start = data.setdefault("start_near_gate", {})
        entrance = target.get("entrance_pose_world", {}).get("position")
        axis = target.get("insertion_axis_world")
        lateral_dir = start.get("lateral_direction_world")
        offset = start.get("reset_body_offset_from_reference_world")
        source_body_start = start.get("body_start_position_world") or start.get("tcp_start_position_world")
        source_reference = (
            start.get("reference_reward_body_start_position_world")
            or start.get("reference_tip_center_position_world")
            or start.get("reference_body_position")
            or start.get("reference_tcp_position")
        )
        source_reset_orientation = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
        source_tip_orientation = start.get("reference_reward_body_start_orientation_wxyz") or target.get(
            "body_start_orientation_wxyz"
        )
        desired_tip_orientation = (
            [float(x) for x in args_cli.override_start_tip_orientation_wxyz]
            if args_cli.override_start_tip_orientation_wxyz is not None
            else source_tip_orientation
        )
        if bool(args_cli.derive_reset_orientation_from_tip):
            if source_reset_orientation is None or desired_tip_orientation is None or source_tip_orientation is None:
                raise ValueError(
                    f"{episode_path} cannot derive reset orientation without source reset/tip orientations"
                )
            tip_to_reset = _quat_mul_list(_quat_conj_list(source_tip_orientation), source_reset_orientation)
            orientation = _quat_mul_list(desired_tip_orientation, tip_to_reset)
        else:
            orientation = (
                [float(x) for x in args_cli.override_start_orientation_wxyz]
                if args_cli.override_start_orientation_wxyz is not None
                else source_reset_orientation or target.get("body_start_orientation_wxyz")
            )
        if orientation is not None and args_cli.override_start_orientation_rotvec_world is not None:
            orientation = _quat_mul_list(
                _quat_from_rotvec_list([float(x) for x in args_cli.override_start_orientation_rotvec_world]),
                orientation,
            )
        if entrance is None or axis is None or lateral_dir is None:
            raise ValueError(f"{episode_path} is missing entrance/axis/lateral fields")
        if (
            isinstance(source_body_start, (list, tuple))
            and len(source_body_start) == 3
            and isinstance(source_reference, (list, tuple))
            and len(source_reference) == 3
        ):
            # Some generated episode configs carry stale reset_body_offset_from_reference_world.
            # The actual body/reference poses are authoritative for preserving semantic tip placement.
            offset = [float(source_body_start[i]) - float(source_reference[i]) for i in range(3)]
        elif offset is None:
            raise ValueError(f"{episode_path} is missing reset body/reference pose fields")
        signed_depth = float(args_cli.override_start_signed_depth_m)
        lateral_m = float(args_cli.override_start_lateral_m)
        reference = [
            float(entrance[i]) + signed_depth * float(axis[i]) + lateral_m * float(lateral_dir[i])
            for i in range(3)
        ]
        if bool(args_cli.derive_reset_position_from_orientation):
            if source_reset_orientation is None or orientation is None:
                raise ValueError(f"{episode_path} cannot derive reset position without source/current orientation")
            local_reset_to_tip = _quat_apply_list(
                _quat_conj_list(source_reset_orientation),
                [-float(offset[i]) for i in range(3)],
            )
            rotated_reset_to_tip = _quat_apply_list(orientation, local_reset_to_tip)
            body_start = [reference[i] - rotated_reset_to_tip[i] for i in range(3)]
        else:
            body_start = [reference[i] + float(offset[i]) for i in range(3)]
        start["axial_distance_m"] = -signed_depth
        start["achieved_axial_distance_m"] = -signed_depth
        start["lateral_distance_m"] = lateral_m
        start["achieved_lateral_distance_m"] = lateral_m
        start["reference_tip_center_position_world"] = reference
        start["reference_reward_body_start_position_world"] = reference
        start["reference_body_position"] = reference
        start["reference_tcp_position"] = reference
        start["body_start_position_world"] = body_start
        start["tcp_start_position_world"] = body_start
        if orientation is not None:
            orientation = [round(float(v), 6) for v in _quat_normalize_list(orientation)]
            start["body_start_orientation_wxyz"] = orientation
            start["reset_body_orientation_wxyz"] = orientation
            start["tcp_start_orientation_world"] = orientation
        if desired_tip_orientation is not None:
            start["reference_reward_body_start_orientation_wxyz"] = [
                round(float(v), 6) for v in _quat_normalize_list(desired_tip_orientation)
            ]
        data["episode_id"] = f"episode_{idx:06d}"
        data["episode_index"] = idx
        (episodes_out / f"episode_{idx:06d}.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False),
            encoding="utf-8",
        )
    (out_root / "source.txt").write_text(str(source) + "\n", encoding="utf-8")
    return str(out_root)


prepared_episode_config_dir = _prepare_episode_config_dir()
if prepared_episode_config_dir:
    os.environ["AIC_ISAAC_EPISODE_CONFIG_DIR"] = prepared_episode_config_dir
os.environ["AIC_ISAAC_ENABLE_CONTACT_SENSOR"] = "1"
os.environ["AIC_ISAAC_POLICY_HZ"] = os.environ.get("AIC_ISAAC_POLICY_HZ", "20.0")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402

import aic_task.tasks  # noqa: F401,E402
from aic_task.tasks.manager_based.aic_task.mdp.insertion_geometry import compute_insertion_geometry  # noqa: E402


STRICT = {
    "min_depth_m": 0.008,
    "max_lateral_m": 0.0005,
    "max_theta_rad": 0.030,
    "min_module_consistency": 0.80,
}


def _run_git(args: list[str]) -> str:
    cwd = _repo_root()
    try:
        return subprocess.run(
            ["git", *args],
            cwd=None if cwd is None else cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _tensor_list(tensor: torch.Tensor | None) -> list[float] | None:
    if tensor is None:
        return None
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_rows(tensor: torch.Tensor | None) -> list[list[float]] | None:
    if tensor is None:
        return None
    return [[float(v) for v in row] for row in tensor.detach().cpu().reshape(tensor.shape[0], -1).tolist()]


def _refresh_cameras(env) -> None:
    sim = getattr(env.unwrapped, "sim", None)
    if sim is not None and hasattr(sim, "render"):
        sim.render()
    step_dt = float(getattr(env.unwrapped, "step_dt", 0.0) or 0.0)
    for sensor_name in ("center_camera", "left_camera", "right_camera"):
        sensor = env.unwrapped.scene.sensors.get(sensor_name)
        if sensor is None or not hasattr(sensor, "update"):
            continue
        try:
            sensor.update(step_dt, force_recompute=True)
        except TypeError:
            sensor.update(step_dt)


def _camera_rgb_uint8(env, sensor_name: str) -> torch.Tensor:
    sensor = env.unwrapped.scene.sensors.get(sensor_name)
    if sensor is None:
        raise RuntimeError(f"Camera sensor {sensor_name!r} is not present in the scene")
    output = sensor.data.output
    if "rgb" not in output:
        raise RuntimeError(f"Camera sensor {sensor_name!r} does not expose rgb output keys: {sorted(output)}")
    image = output["rgb"].detach()
    if image.ndim != 4:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.shape[-1] in (3, 4):
        image = image[..., :3]
    elif image.shape[1] in (3, 4):
        image = image[:, :3].permute(0, 2, 3, 1).contiguous()
    else:
        raise RuntimeError(f"Camera sensor {sensor_name!r} rgb output has unexpected shape: {tuple(image.shape)}")
    if image.dtype == torch.uint8:
        return image.cpu()
    image_f = image.float()
    if image_f.numel() and float(image_f.max().detach().cpu()) <= 2.0:
        image_f = image_f * 255.0
    return image_f.clamp(0.0, 255.0).to(torch.uint8).cpu()


def _save_step_images(env, *, run_dir: Path, step: int, record: dict[str, Any]) -> list[str]:
    from PIL import Image, ImageDraw

    _refresh_cameras(env)
    image_dir = run_dir / "step_images" / f"step_{step:06d}"
    image_dir.mkdir(parents=True, exist_ok=True)
    geom = record.get("post_step_insertion_geometry") or {}
    module_geom = record.get("post_step_module_geometry") or {}
    saved: list[str] = []

    def value(values: list[Any], env_idx: int, default: float = float("nan")) -> float:
        return float(values[env_idx]) if env_idx < len(values) else default

    for camera in ("center_camera", "left_camera", "right_camera"):
        images = _camera_rgb_uint8(env, camera)
        for env_idx in range(images.shape[0]):
            image = Image.fromarray(images[env_idx].numpy())
            draw = ImageDraw.Draw(image)
            s_vals = geom.get("signed_depth_m_by_env") or []
            r_vals = geom.get("lateral_error_m_by_env") or []
            theta_vals = geom.get("orientation_error_rad_by_env") or []
            strict_vals = geom.get("strict_success_by_env") or []
            cons_vals = geom.get("consistency_gate_by_env") or module_geom.get("consistency_gate_by_env") or []
            label = (
                f"step={step} env={env_idx} {camera} "
                f"s={value(s_vals, env_idx) * 1000.0:+.3f}mm "
                f"r={value(r_vals, env_idx) * 1000.0:.3f}mm "
                f"theta={value(theta_vals, env_idx):.5f} "
                f"cons={value(cons_vals, env_idx):.3f} "
                f"strict={bool(strict_vals[env_idx]) if env_idx < len(strict_vals) else False}"
            )
            draw.rectangle([0, 0, min(image.size[0], 900), 24], fill=(255, 255, 255))
            draw.text((4, 4), label, fill=(0, 0, 0))
            out_path = image_dir / f"env_{env_idx:04d}_{camera}.png"
            image.save(out_path)
            saved.append(str(out_path))
    return saved


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(rotvec, dim=1, keepdim=True)
    half_angle = 0.5 * angle
    axis = rotvec / angle.clamp(min=1.0e-9)
    quat = torch.cat([torch.cos(half_angle), axis * torch.sin(half_angle)], dim=1)
    small_quat = torch.cat([torch.ones_like(angle), 0.5 * rotvec], dim=1)
    quat = torch.where((angle < 1.0e-8).expand_as(quat), small_quat, quat)
    return quat / torch.linalg.norm(quat, dim=1, keepdim=True).clamp(min=1.0e-9)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 0:1], -quat[:, 1:4]], dim=1)


def _current_episode_by_env(env) -> dict[int, dict[str, Any]]:
    return dict(getattr(env.unwrapped, "_aic_current_episode_by_env", {}) or {})


def _episode_target_position(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        pos = ((target.get("target_pose_world") or {}).get("position"))
        if pos is None:
            return None
        rows.append(torch.tensor(pos, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_entrance_position(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        pos = ((target.get("entrance_pose_world") or {}).get("position"))
        if pos is None:
            return None
        rows.append(torch.tensor(pos, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_axis(env) -> torch.Tensor | None:
    episodes = _current_episode_by_env(env)
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        axis = target.get("insertion_axis_world")
        if axis is None:
            return None
        axis_t = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
        rows.append(axis_t / torch.linalg.norm(axis_t).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _body_index(env, body_name: str) -> int | None:
    names = list(getattr(env.unwrapped.scene["robot"], "body_names", []))
    return names.index(body_name) if body_name in names else None


def _body_position(env, body_name: str) -> torch.Tensor | None:
    idx = _body_index(env, body_name)
    if idx is None:
        return None
    return env.unwrapped.scene["robot"].data.body_pos_w[:, idx]


def _body_orientation(env, body_name: str) -> torch.Tensor | None:
    idx = _body_index(env, body_name)
    if idx is None:
        return None
    return env.unwrapped.scene["robot"].data.body_quat_w[:, idx]


def _orientation_error(env, body_name: str, axis_w: torch.Tensor) -> torch.Tensor | None:
    quat = _body_orientation(env, body_name)
    if quat is None:
        return None
    local_axis = torch.tensor(
        args_cli.target_reward_orientation_axis_local,
        dtype=quat.dtype,
        device=quat.device,
    ).view(1, 3)
    if body_name == "sfp_tip_link":
        # Match the existing strict/evaluation semantics: sfp_tip_link uses a
        # pi body-orientation offset, which flips the local semantic tip axis.
        local_axis = -local_axis
    local_axis = local_axis / torch.linalg.norm(local_axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    body_axis = math_utils.quat_apply(quat, local_axis.expand(quat.shape[0], -1))
    return torch.acos(torch.sum(body_axis * axis_w.to(device=quat.device, dtype=quat.dtype), dim=1).clamp(-1.0, 1.0))


def _semantic_body_axis(env, body_name: str, axis_w: torch.Tensor) -> torch.Tensor | None:
    quat = _body_orientation(env, body_name)
    if quat is None:
        return None
    local_axis = torch.tensor(
        args_cli.target_reward_orientation_axis_local,
        dtype=quat.dtype,
        device=quat.device,
    ).view(1, 3)
    if body_name == "sfp_tip_link":
        local_axis = -local_axis
    local_axis = local_axis / torch.linalg.norm(local_axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    body_axis = math_utils.quat_apply(quat, local_axis.expand(quat.shape[0], -1))
    return body_axis / torch.linalg.norm(body_axis, dim=1, keepdim=True).clamp(min=1.0e-9)


def _geometry(env, body_name: str = "sfp_tip_link") -> dict[str, Any]:
    target = _episode_target_position(env)
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    body = _body_position(env, body_name)
    out: dict[str, Any] = {
        "body_name": body_name,
        "has_target": target is not None,
        "has_entrance": entrance is not None,
        "has_axis": axis is not None,
        "has_body": body is not None,
    }
    if target is None or entrance is None or axis is None or body is None:
        return out
    geom = compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    theta = _orientation_error(env, body_name, geom.axis)
    consistency = _module_consistency(env, geom, entrance, target, axis)
    strict = (
        (geom.axial_depth >= STRICT["min_depth_m"])
        & (geom.lateral_error <= STRICT["max_lateral_m"])
        & (torch.zeros_like(geom.axial_depth, dtype=torch.bool) if theta is None else theta <= STRICT["max_theta_rad"])
        & (
            torch.zeros_like(geom.axial_depth, dtype=torch.bool)
            if consistency is None
            else consistency >= STRICT["min_module_consistency"]
        )
    )
    out.update(
        {
            "signed_depth_m_by_env": _tensor_list(geom.axial_depth),
            "lateral_error_m_by_env": _tensor_list(geom.lateral_error),
            "target_depth_m_by_env": _tensor_list(geom.target_depth),
            "depth_fraction_by_env": _tensor_list(geom.depth_fraction),
            "orientation_error_rad_by_env": _tensor_list(theta),
            "consistency_gate_by_env": _tensor_list(consistency),
            "strict_success_by_env": [bool(x) for x in strict.detach().cpu().tolist()],
            "axis_world_env0": _tensor_list(axis[0]),
            "entrance_world_env0": _tensor_list(entrance[0]),
            "body_world_env0": _tensor_list(body[0]),
        }
    )
    return out


def _module_consistency(
    env,
    tip_geom,
    entrance: torch.Tensor,
    target: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor | None:
    module = _body_position(env, str(args_cli.target_reward_consistency_body))
    if module is None:
        return None
    module_geom = compute_insertion_geometry(
        body_pos_w=module,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=max(float(args_cli.target_reward_consistency_lateral_sigma), 1.0e-9),
    )
    gap = tip_geom.axial_depth - module_geom.axial_depth
    attr = "_aic_wrist_contact_reference_gap"
    reference = getattr(env.unwrapped, attr, None)
    if reference is None or reference.shape != gap.shape:
        reference = gap.detach().clone()
        setattr(env.unwrapped, attr, reference)
    expected_module_depth = tip_geom.target_depth - reference.to(gap.device)
    axial_gate = torch.exp(
        -torch.square(
            (module_geom.axial_depth - expected_module_depth)
            / max(float(args_cli.target_reward_consistency_axial_std), 1.0e-9)
        )
    )
    return axial_gate * module_geom.lateral_gate


def _contact_summary(env) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sensor = getattr(env.unwrapped.scene, "sensors", {}).get("contact_forces")
    if sensor is None:
        return {"available": False}
    net = getattr(sensor.data, "net_forces_w", None)
    names = list(getattr(sensor, "body_names", []) or [])
    out["available"] = net is not None
    out["body_names"] = names
    if net is not None and net.numel() > 0:
        norms = torch.linalg.norm(net[:, :, :3], dim=-1)
        out["force_norm_mean_by_env"] = _tensor_list(norms.mean(dim=1))
        out["force_norm_max_by_env"] = _tensor_list(norms.max(dim=1).values)
        if names:
            out["env0_by_body"] = {name: float(norms[0, idx].detach().cpu()) for idx, name in enumerate(names)}
    return out


def _root_action_from_world_delta(env, world_delta: torch.Tensor, rotvec_world: torch.Tensor | None = None) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    root_quat = robot.data.root_quat_w.to(device=world_delta.device, dtype=world_delta.dtype)
    root_delta = math_utils.quat_apply_inverse(root_quat, world_delta)
    if bool(args_cli.fix_isaac_ik_xy_sign):
        root_delta = root_delta.clone()
        root_delta[:, 0:2] = -root_delta[:, 0:2]
    if rotvec_world is None:
        root_rot = torch.zeros_like(root_delta)
    else:
        root_rot = math_utils.quat_apply_inverse(root_quat, rotvec_world.to(device=world_delta.device, dtype=world_delta.dtype))
    return torch.cat([root_delta, root_rot], dim=1)


def _zero_action(env) -> torch.Tensor:
    return torch.zeros(env.action_space.shape, device=env.unwrapped.device, dtype=torch.float32)


def _approach_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    if tip is None or entrance is None or axis is None:
        return _zero_action(env), {"reason": "missing_geometry"}
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.approach_lateral_step_m), 0.0)),
    )
    lateral_dir = lateral_vec / lateral_error.clamp(min=1.0e-9)
    lateral_delta = lateral_dir * lateral_step * (1.0 if float(args_cli.approach_lateral_sign) >= 0.0 else -1.0)
    can_advance = lateral_error <= max(float(args_cli.approach_lateral_gate_m), 0.0)
    below_shallow = depth < float(args_cli.shallow_depth_m)
    axial_step = torch.full_like(depth, max(float(args_cli.approach_axial_step_m), 0.0))
    axial_delta = torch.where(
        can_advance & below_shallow,
        axial_step * axis * (1.0 if float(args_cli.approach_axial_sign) >= 0.0 else -1.0),
        torch.zeros_like(lateral_delta),
    )
    world_delta = lateral_delta + axial_delta
    return _root_action_from_world_delta(env, world_delta), {
        "depth_m_by_env": _tensor_list(depth),
        "lateral_error_m_by_env": _tensor_list(lateral_error),
        "advanced_fraction": float((can_advance & below_shallow).float().mean().detach().cpu()),
    }


def _probe_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    axis = _episode_axis(env)
    if axis is None:
        return _zero_action(env), {"reason": "missing_axis"}
    if args_cli.probe == "axis_backout":
        world_delta = -axis * float(args_cli.probe_translation_step_m)
        return _root_action_from_world_delta(env, world_delta), {"probe_label": "axis_backout"}
    if args_cli.probe == "axis_forward":
        world_delta = axis * float(args_cli.probe_translation_step_m)
        return _root_action_from_world_delta(env, world_delta), {"probe_label": "axis_forward"}
    if args_cli.probe == "rotation_axis":
        axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
        rot = torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device)
        sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
        rot[:, axis_idx] = sign * float(args_cli.probe_rotation_step_rad)
        world_delta, correction_info = _probe_retention_delta(env)
        return _root_action_from_world_delta(env, world_delta, rotvec_world=rot), {
            "probe_label": f"{'+' if sign >= 0.0 else '-'}r{args_cli.probe_rotation_axis}",
            **correction_info,
        }
    if args_cli.probe == "orientation_servo_best":
        return _orientation_servo_best_action(env)
    if args_cli.probe == "pose_hold":
        return _pose_hold_action(env)
    if args_cli.probe == "pose_hold_orientation_servo_best":
        return _pose_hold_orientation_servo_best_action(env, probe_step)
    if args_cli.probe == "pose_hold_rotation_axis":
        return _pose_hold_rotation_axis_action(env, probe_step)
    if args_cli.probe == "target_tip_stabilize":
        return _target_tip_stabilize_action(env)
    if args_cli.probe == "target_tip_fixed_rotation_axis":
        return _target_tip_fixed_rotation_axis_action(env)
    labels = ("+rx", "+ry", "+rz", "-rx", "-ry", "-rz")
    idx = probe_step % len(labels)
    sign = 1.0 if idx < 3 else -1.0
    axis_idx = idx % 3
    rot = torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device)
    rot[:, axis_idx] = sign * float(args_cli.probe_rotation_step_rad)
    world_delta = torch.zeros_like(rot)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rot), {"probe_label": labels[idx]}


def _clip_vector_norm(vector: torch.Tensor, max_norm: float) -> torch.Tensor:
    if max_norm <= 0.0:
        return torch.zeros_like(vector)
    norm = torch.linalg.norm(vector, dim=1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), torch.full_like(norm, max_norm) / norm.clamp(min=1.0e-9))
    return vector * scale


def _clip_scalar(value: torch.Tensor, max_abs: float) -> torch.Tensor:
    if max_abs <= 0.0:
        return torch.zeros_like(value)
    return value.clamp(min=-max_abs, max=max_abs)


def _target_tip_stabilize_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    stabilized = _target_tip_stabilize_delta(env)
    if stabilized is None:
        return _zero_action(env), {"probe_label": "target_tip_stabilize", "reason": "missing_target_tip_geometry"}
    world_delta, geom, info = stabilized

    rotvec, orient_info = _best_semantic_orientation_rotvec(env, world_delta)
    if rotvec is None:
        rotvec = torch.zeros_like(world_delta)
        orientation_active = torch.zeros((world_delta.shape[0], 1), dtype=torch.bool, device=world_delta.device)
    else:
        orientation_active = (
            (geom.lateral_error.view(-1, 1) <= float(args_cli.target_tip_stabilize_orientation_gate_lateral_m))
            & (geom.axial_depth.view(-1, 1) >= float(args_cli.target_tip_stabilize_orientation_gate_depth_m))
        )
        rotvec = torch.where(orientation_active.expand_as(rotvec), rotvec, torch.zeros_like(rotvec))

    compensation = _rotation_tip_sweep_compensation(env, rotvec)
    compensation = _clip_vector_norm(compensation, max(float(args_cli.target_tip_stabilize_rotation_compensation_clip_m), 0.0))
    world_delta = world_delta + compensation
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "target_tip_stabilize",
        "target_tip_rotation_compensation_m_by_env": _tensor_rows(compensation),
        "target_tip_orientation_active_by_env": [bool(x) for x in orientation_active.detach().cpu().reshape(-1).tolist()],
        **info,
        **orient_info,
    }


def _target_tip_fixed_rotation_axis_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    stabilized = _target_tip_stabilize_delta(env)
    if stabilized is None:
        return _zero_action(env), {
            "probe_label": "target_tip_fixed_rotation_axis",
            "reason": "missing_target_tip_geometry",
        }
    world_delta, geom, info = stabilized
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
    sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
    rotvec = torch.zeros_like(world_delta)
    rotvec[:, axis_idx] = sign * max(float(args_cli.probe_rotation_step_rad), 0.0)
    orientation_active = (
        (geom.lateral_error.view(-1, 1) <= float(args_cli.target_tip_stabilize_orientation_gate_lateral_m))
        & (geom.axial_depth.view(-1, 1) >= float(args_cli.target_tip_stabilize_orientation_gate_depth_m))
    )
    rotvec = torch.where(orientation_active.expand_as(rotvec), rotvec, torch.zeros_like(rotvec))
    compensation = _rotation_tip_sweep_compensation(env, rotvec)
    compensation = _clip_vector_norm(compensation, max(float(args_cli.target_tip_stabilize_rotation_compensation_clip_m), 0.0))
    world_delta = world_delta + compensation
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "target_tip_fixed_rotation_axis",
        "target_tip_fixed_rotation_axis": str(args_cli.probe_rotation_axis),
        "target_tip_fixed_rotation_sign": sign,
        "target_tip_fixed_rotation_active_by_env": [
            bool(x) for x in orientation_active.detach().cpu().reshape(-1).tolist()
        ],
        "target_tip_rotation_compensation_m_by_env": _tensor_rows(compensation),
        **info,
    }


def _target_tip_stabilize_delta(env) -> tuple[torch.Tensor, Any, dict[str, Any]] | None:
    target_tip_pos = getattr(env.unwrapped, "_aic_target_tip_stabilize_pos_w", None)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    target = _episode_target_position(env)
    axis = _episode_axis(env)
    if target_tip_pos is None or tip is None or entrance is None or target is None or axis is None:
        return None
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    if math.isfinite(float(args_cli.target_tip_stabilize_goal_depth_m)):
        target_tip_pos = entrance + float(args_cli.target_tip_stabilize_goal_depth_m) * axis
    else:
        target_tip_pos = target_tip_pos.to(device=tip.device, dtype=tip.dtype)

    error = target_tip_pos - tip
    axial_error = torch.sum(error * axis, dim=1, keepdim=True)
    lateral_error_vec = error - axial_error * axis
    axial_clip = (
        float(args_cli.target_tip_stabilize_axial_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_axial_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    lateral_clip = (
        float(args_cli.target_tip_stabilize_lateral_step_m)
        if math.isfinite(float(args_cli.target_tip_stabilize_lateral_step_m))
        else float(args_cli.probe_translation_step_m)
    )
    axial_delta = _clip_scalar(axial_error, max(axial_clip, 0.0)) * axis
    lateral_delta = _clip_vector_norm(lateral_error_vec, max(lateral_clip, 0.0))

    geom = compute_insertion_geometry(
        body_pos_w=tip,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    near_target = (
        (torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True) <= max(lateral_clip, 1.0e-9))
        & (torch.abs(axial_error) <= max(axial_clip, 1.0e-9))
    )
    inward_bias = torch.where(
        near_target,
        torch.full_like(axial_error, max(float(args_cli.target_tip_stabilize_inward_bias_m), 0.0)),
        torch.zeros_like(axial_error),
    ) * axis
    world_delta = lateral_delta + axial_delta + inward_bias

    return world_delta, geom, {
        "target_tip_position_error_m_by_env": _tensor_list(torch.linalg.norm(error, dim=1, keepdim=True)),
        "target_tip_axial_error_m_by_env": _tensor_list(axial_error),
        "target_tip_lateral_error_m_by_env": _tensor_list(torch.linalg.norm(lateral_error_vec, dim=1, keepdim=True)),
        "target_tip_world_delta_m_by_env": _tensor_rows(world_delta),
        "target_tip_pre_step_s_m_by_env": _tensor_list(geom.axial_depth),
        "target_tip_pre_step_r_m_by_env": _tensor_list(geom.lateral_error),
    }


def _rotation_tip_sweep_compensation(env, rotvec_world: torch.Tensor) -> torch.Tensor:
    tip = _body_position(env, "sfp_tip_link")
    wrist = _body_position(env, "wrist_3_link")
    if tip is None or wrist is None:
        return torch.zeros_like(rotvec_world)
    q_step = _quat_from_rotvec(rotvec_world)
    lever = tip - wrist
    predicted_tip = wrist + math_utils.quat_apply(q_step, lever)
    return tip - predicted_tip


def _pose_hold_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold", "reason": "missing_pose_hold_target"}
    world_delta, rotvec_world, info = hold
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec_world), {
        "probe_label": "pose_hold",
        **info,
    }


def _pose_hold_delta(env) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]] | None:
    body_name = str(args_cli.pose_hold_body)
    target_pos = getattr(env.unwrapped, "_aic_pose_hold_target_pos_w", None)
    target_quat = getattr(env.unwrapped, "_aic_pose_hold_target_quat_w", None)
    current_pos = _body_position(env, body_name)
    current_quat = _body_orientation(env, body_name)
    if target_pos is None or target_quat is None or current_pos is None or current_quat is None:
        return None
    target_pos = target_pos.to(device=current_pos.device, dtype=current_pos.dtype)
    target_quat = target_quat.to(device=current_quat.device, dtype=current_quat.dtype)
    pos_error = (target_pos - current_pos) * float(args_cli.pose_hold_position_gain)
    world_delta = _clip_vector_norm(pos_error, max(float(args_cli.probe_translation_step_m), 0.0))
    delta_quat_w = math_utils.quat_mul(target_quat, math_utils.quat_inv(current_quat))
    rotvec_world = math_utils.axis_angle_from_quat(delta_quat_w) * float(args_cli.pose_hold_rotation_gain)
    rotvec_world = _clip_vector_norm(rotvec_world, max(float(args_cli.probe_rotation_step_rad), 0.0))
    return world_delta, rotvec_world, {
        "pose_hold_body": body_name,
        "pose_hold_position_error_m_by_env": _tensor_list(torch.linalg.norm(pos_error, dim=1, keepdim=True)),
        "pose_hold_rotation_error_rad_by_env": _tensor_list(
            torch.linalg.norm(math_utils.axis_angle_from_quat(delta_quat_w), dim=1, keepdim=True)
        ),
        "pose_hold_world_delta_m_by_env": _tensor_rows(world_delta),
        "pose_hold_rotvec_world_rad_by_env": _tensor_rows(rotvec_world),
    }


def _pose_hold_orientation_servo_best_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold_orientation_servo_best", "reason": "missing_pose_hold_target"}
    world_delta, hold_rotvec, hold_info = hold
    rot_step = (
        float(args_cli.pose_hold_orientation_step_rad)
        if math.isfinite(float(args_cli.pose_hold_orientation_step_rad))
        else float(args_cli.probe_rotation_step_rad)
    )
    rotvec, orient_info = _best_semantic_orientation_rotvec(env, world_delta, rotation_step_rad=rot_step)
    if rotvec is None:
        rotvec = hold_rotvec
        active = torch.zeros((world_delta.shape[0], 1), dtype=torch.bool, device=world_delta.device)
    else:
        active = _pose_hold_orientation_active(env)
        if probe_step < int(args_cli.pose_hold_orientation_start_probe_step):
            active = torch.zeros_like(active)
        rotvec = torch.where(active.expand_as(rotvec), rotvec, hold_rotvec)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "pose_hold_orientation_servo_best",
        "orientation_servo_active_by_env": [bool(x) for x in active.detach().cpu().reshape(-1).tolist()],
        **hold_info,
        **orient_info,
    }


def _pose_hold_rotation_axis_action(env, probe_step: int) -> tuple[torch.Tensor, dict[str, Any]]:
    hold = _pose_hold_delta(env)
    if hold is None:
        return _zero_action(env), {"probe_label": "pose_hold_rotation_axis", "reason": "missing_pose_hold_target"}
    world_delta, hold_rotvec, hold_info = hold
    axis_idx = {"x": 0, "y": 1, "z": 2}[str(args_cli.probe_rotation_axis)]
    sign = 1.0 if float(args_cli.probe_rotation_sign) >= 0.0 else -1.0
    fixed_step = (
        float(args_cli.pose_hold_fixed_rotation_step_rad)
        if math.isfinite(float(args_cli.pose_hold_fixed_rotation_step_rad))
        else float(args_cli.probe_rotation_step_rad)
    )
    fixed_rot = torch.zeros_like(hold_rotvec)
    fixed_rot[:, axis_idx] = sign * max(fixed_step, 0.0)
    active = _pose_hold_orientation_active(env)
    if probe_step < int(args_cli.pose_hold_orientation_start_probe_step):
        active = torch.zeros_like(active)
    rotvec = torch.where(active.expand_as(fixed_rot), fixed_rot, hold_rotvec)
    return _root_action_from_world_delta(env, world_delta, rotvec_world=rotvec), {
        "probe_label": "pose_hold_rotation_axis",
        "pose_hold_fixed_rotation_active_by_env": [bool(x) for x in active.detach().cpu().reshape(-1).tolist()],
        "pose_hold_fixed_rotation_axis": str(args_cli.probe_rotation_axis),
        "pose_hold_fixed_rotation_sign": sign,
        **hold_info,
    }


def _pose_hold_orientation_active(env) -> torch.Tensor:
    target = _episode_target_position(env)
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    if target is None or entrance is None or axis is None or tip is None:
        return torch.zeros((env.unwrapped.num_envs, 1), dtype=torch.bool, device=env.unwrapped.device)
    geom = compute_insertion_geometry(
        body_pos_w=tip,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0025,
    )
    consistency = _module_consistency(env, geom, entrance, target, axis)
    if consistency is None:
        consistency = torch.zeros_like(geom.axial_depth)
    return (
        (geom.axial_depth.view(-1, 1) >= float(args_cli.pose_hold_orientation_activation_depth_m))
        & (geom.lateral_error.view(-1, 1) <= float(args_cli.pose_hold_orientation_activation_lateral_m))
        & (consistency.view(-1, 1) >= float(args_cli.pose_hold_orientation_activation_consistency))
    )


def _best_semantic_orientation_rotvec(
    env,
    world_delta: torch.Tensor,
    *,
    rotation_step_rad: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    body_axis = None if axis is None else _semantic_body_axis(env, "sfp_tip_link", axis)
    if axis is None or tip is None or entrance is None or body_axis is None:
        return None, {"orientation_servo_reason": "missing_orientation_servo_geometry"}
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    current_theta = torch.acos(torch.sum(body_axis * axis, dim=1, keepdim=True).clamp(-1.0, 1.0))
    candidate_axes = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=body_axis.dtype,
        device=body_axis.device,
    )
    rot_step = max(float(args_cli.probe_rotation_step_rad if rotation_step_rad is None else rotation_step_rad), 0.0)
    candidate_rotvec = candidate_axes.view(1, 6, 3).expand(body_axis.shape[0], -1, -1) * rot_step
    q_candidate = _quat_from_rotvec(candidate_rotvec.reshape(-1, 3))
    predicted_axis = math_utils.quat_apply(
        q_candidate,
        body_axis[:, None, :].expand(-1, 6, -1).reshape(-1, 3),
    ).reshape(body_axis.shape[0], 6, 3)
    predicted_theta = torch.acos(torch.sum(predicted_axis * axis[:, None, :], dim=2).clamp(-1.0, 1.0))
    if float(args_cli.probe_orientation_lateral_penalty) > 0.0:
        wrist = _body_position(env, "wrist_3_link")
        if wrist is not None:
            lever = tip - wrist
            predicted_tip = (
                math_utils.quat_apply(q_candidate, lever[:, None, :].expand(-1, 6, -1).reshape(-1, 3))
                .reshape(tip.shape[0], 6, 3)
                + wrist[:, None, :]
                + world_delta[:, None, :]
            )
            pred_rel = predicted_tip - entrance[:, None, :]
            pred_depth = torch.sum(pred_rel * axis[:, None, :], dim=2, keepdim=True)
            pred_lateral = torch.linalg.norm(pred_rel - pred_depth * axis[:, None, :], dim=2)
            predicted_theta = predicted_theta + float(args_cli.probe_orientation_lateral_penalty) * pred_lateral
    _, best_idx = torch.min(predicted_theta, dim=1, keepdim=True)
    selected_rotvec = torch.gather(candidate_rotvec, 1, best_idx[:, :, None].expand(-1, -1, 3)).squeeze(1)
    predicted_raw_theta = torch.acos(
        torch.sum(math_utils.quat_apply(_quat_from_rotvec(selected_rotvec), body_axis) * axis, dim=1, keepdim=True).clamp(
            -1.0, 1.0
        )
    )
    improves = predicted_raw_theta < current_theta
    selected_rotvec = torch.where(improves.expand_as(selected_rotvec), selected_rotvec, torch.zeros_like(selected_rotvec))
    labels = ["+rx", "-rx", "+ry", "-ry", "+rz", "-rz"]
    return selected_rotvec, {
        "selected_axis_by_env": [labels[int(i)] for i in best_idx.detach().cpu().reshape(-1).tolist()],
        "current_theta_rad_by_env": _tensor_list(current_theta),
        "predicted_theta_rad_by_env": _tensor_list(predicted_raw_theta),
        "improves_by_env": [bool(x) for x in improves.detach().cpu().reshape(-1).tolist()],
    }


def _orientation_servo_best_action(env) -> tuple[torch.Tensor, dict[str, Any]]:
    axis = _episode_axis(env)
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    body_axis = None if axis is None else _semantic_body_axis(env, "sfp_tip_link", axis)
    if axis is None or tip is None or entrance is None or body_axis is None:
        return _zero_action(env), {"reason": "missing_orientation_servo_geometry"}
    axis = axis / torch.linalg.norm(axis, dim=1, keepdim=True).clamp(min=1.0e-9)
    current_theta = torch.acos(torch.sum(body_axis * axis, dim=1, keepdim=True).clamp(-1.0, 1.0))
    candidate_axes = torch.tensor(
        (
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ),
        dtype=body_axis.dtype,
        device=body_axis.device,
    )
    rot_step = max(float(args_cli.probe_rotation_step_rad), 0.0)
    candidate_rotvec = candidate_axes.view(1, 6, 3).expand(body_axis.shape[0], -1, -1) * rot_step
    q_candidate = _quat_from_rotvec(candidate_rotvec.reshape(-1, 3))
    predicted_axis = math_utils.quat_apply(
        q_candidate,
        body_axis[:, None, :].expand(-1, 6, -1).reshape(-1, 3),
    ).reshape(body_axis.shape[0], 6, 3)
    predicted_theta = torch.acos(
        torch.sum(predicted_axis * axis[:, None, :], dim=2).clamp(-1.0, 1.0)
    )
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.probe_lateral_correction_step_m), 0.0)),
    )
    lateral_delta = lateral_vec / lateral_error.clamp(min=1.0e-9) * lateral_step
    axial_step = torch.full_like(depth, float(args_cli.probe_axial_step_m))
    can_advance = lateral_error <= max(float(args_cli.probe_axial_lateral_gate_m), 0.0)
    axial_delta = torch.where(can_advance, axial_step * axis, torch.zeros_like(lateral_delta))
    world_delta = lateral_delta + axial_delta
    if float(args_cli.probe_orientation_lateral_penalty) > 0.0:
        # Approximate one-step sweep from rotating the semantic tip around the wrist/root action frame.
        wrist = _body_position(env, "wrist_3_link")
        if wrist is not None:
            lever = tip - wrist
            predicted_tip = (
                math_utils.quat_apply(q_candidate, lever[:, None, :].expand(-1, 6, -1).reshape(-1, 3))
                .reshape(tip.shape[0], 6, 3)
                + wrist[:, None, :]
                + world_delta[:, None, :]
            )
            pred_rel = predicted_tip - entrance[:, None, :]
            pred_depth = torch.sum(pred_rel * axis[:, None, :], dim=2, keepdim=True)
            pred_lateral = torch.linalg.norm(pred_rel - pred_depth * axis[:, None, :], dim=2)
            predicted_theta = predicted_theta + float(args_cli.probe_orientation_lateral_penalty) * pred_lateral
    best_score, best_idx = torch.min(predicted_theta, dim=1, keepdim=True)
    selected_rotvec = torch.gather(
        candidate_rotvec,
        1,
        best_idx[:, :, None].expand(-1, -1, 3),
    ).squeeze(1)
    predicted_raw_theta = torch.acos(
        torch.sum(
            math_utils.quat_apply(_quat_from_rotvec(selected_rotvec), body_axis) * axis,
            dim=1,
            keepdim=True,
        ).clamp(-1.0, 1.0)
    )
    improves = predicted_raw_theta < current_theta
    selected_rotvec = torch.where(improves.expand_as(selected_rotvec), selected_rotvec, torch.zeros_like(selected_rotvec))
    labels = ["+rx", "-rx", "+ry", "-ry", "+rz", "-rz"]
    return _root_action_from_world_delta(env, world_delta, rotvec_world=selected_rotvec), {
        "probe_label": "orientation_servo_best",
        "selected_axis_by_env": [labels[int(i)] for i in best_idx.detach().cpu().reshape(-1).tolist()],
        "current_theta_rad_by_env": _tensor_list(current_theta),
        "predicted_theta_rad_by_env": _tensor_list(predicted_raw_theta),
        "improves_by_env": [bool(x) for x in improves.detach().cpu().reshape(-1).tolist()],
        "retention_lateral_error_m_by_env": _tensor_list(lateral_error),
        "retention_depth_m_by_env": _tensor_list(depth),
        "retention_advanced_fraction": float(can_advance.float().mean().detach().cpu()),
    }


def _probe_retention_delta(env) -> tuple[torch.Tensor, dict[str, Any]]:
    tip = _body_position(env, "sfp_tip_link")
    entrance = _episode_entrance_position(env)
    axis = _episode_axis(env)
    if tip is None or entrance is None or axis is None:
        return torch.zeros((env.unwrapped.num_envs, 3), dtype=torch.float32, device=env.unwrapped.device), {
            "retention_reason": "missing_geometry"
        }
    rel = tip - entrance
    depth = torch.sum(rel * axis, dim=1, keepdim=True)
    centerline = entrance + depth * axis
    lateral_vec = centerline - tip
    lateral_error = torch.linalg.norm(lateral_vec, dim=1, keepdim=True)
    lateral_step = torch.minimum(
        lateral_error,
        torch.full_like(lateral_error, max(float(args_cli.probe_lateral_correction_step_m), 0.0)),
    )
    lateral_delta = lateral_vec / lateral_error.clamp(min=1.0e-9) * lateral_step
    axial_step = torch.full_like(depth, float(args_cli.probe_axial_step_m))
    can_advance = lateral_error <= max(float(args_cli.probe_axial_lateral_gate_m), 0.0)
    axial_delta = torch.where(can_advance, axial_step * axis, torch.zeros_like(lateral_delta))
    return lateral_delta + axial_delta, {
        "retention_lateral_error_m_by_env": _tensor_list(lateral_error),
        "retention_depth_m_by_env": _tensor_list(depth),
        "retention_advanced_fraction": float(can_advance.float().mean().detach().cpu()),
    }


def _body_positions(env) -> dict[str, list[list[float]] | None]:
    out: dict[str, list[list[float]] | None] = {}
    for name in ("wrist_3_link", "gripper_tcp", "sfp_module_link", "sfp_tip_link"):
        pos = _body_position(env, name)
        out[name] = None if pos is None else [[float(v) for v in row] for row in pos.detach().cpu().tolist()]
    return out


def _body_orientations(env) -> dict[str, list[list[float]] | None]:
    out: dict[str, list[list[float]] | None] = {}
    for name in ("wrist_3_link", "gripper_tcp", "sfp_module_link", "sfp_tip_link"):
        quat = _body_orientation(env, name)
        out[name] = None if quat is None else [[float(v) for v in row] for row in quat.detach().cpu().tolist()]
    return out


def _relative_body_transforms(env) -> dict[str, Any]:
    pairs = (
        ("gripper_tcp", "sfp_module_link"),
        ("gripper_tcp", "sfp_tip_link"),
        ("sfp_module_link", "sfp_tip_link"),
    )
    out: dict[str, Any] = {}
    for parent, child in pairs:
        parent_pos = _body_position(env, parent)
        child_pos = _body_position(env, child)
        parent_quat = _body_orientation(env, parent)
        child_quat = _body_orientation(env, child)
        key = f"{parent}_to_{child}"
        if parent_pos is None or child_pos is None or parent_quat is None or child_quat is None:
            out[key] = None
            continue
        parent_inv = _quat_conjugate(parent_quat)
        rel_pos = math_utils.quat_apply(parent_inv, child_pos - parent_pos)
        rel_quat = math_utils.quat_mul(parent_inv, child_quat)
        out[key] = {
            "position_parent_frame_m_by_env": _tensor_rows(rel_pos),
            "orientation_parent_child_wxyz_by_env": _tensor_rows(rel_quat),
        }
    return out


def _reset_diagnostic(env) -> dict[str, Any]:
    episodes = _current_episode_by_env(env)
    axis = _episode_axis(env)
    out: dict[str, Any] = {
        "episode_start_by_env": {},
        "actual_body_position_world_by_env": _body_positions(env),
        "actual_body_orientation_wxyz_by_env": _body_orientations(env),
        "relative_body_transforms": _relative_body_transforms(env),
        "tip_geometry": _geometry(env, "sfp_tip_link"),
        "module_geometry": _geometry(env, "sfp_module_link"),
        "contact": _contact_summary(env),
    }
    if axis is not None:
        out["axis_world_by_env"] = [[float(v) for v in row] for row in axis.detach().cpu().tolist()]
    for env_id, episode in episodes.items():
        scene = (episode or {}).get("scene") or {}
        target = scene.get("target") or {}
        start = target.get("start_near_gate")
        if not isinstance(start, dict):
            start = scene.get("start_near_gate")
        if not isinstance(start, dict):
            start = (episode or {}).get("start_near_gate") or {}
        out["episode_start_by_env"][str(env_id)] = {
            "body_start_position_world": start.get("body_start_position_world"),
            "body_start_orientation_wxyz": start.get("body_start_orientation_wxyz")
            or start.get("reset_body_orientation_wxyz"),
            "reference_reward_body_start_position_world": start.get("reference_reward_body_start_position_world")
            or start.get("reference_tip_center_position_world"),
            "reference_reward_body_start_orientation_wxyz": start.get("reference_reward_body_start_orientation_wxyz")
            or target.get("body_start_orientation_wxyz"),
            "axial_distance_m": start.get("axial_distance_m"),
            "lateral_distance_m": start.get("lateral_distance_m"),
        }
    return out


def _step_record(
    env,
    *,
    step: int,
    phase: str,
    action: torch.Tensor,
    action_info: dict[str, Any],
    before_positions: dict[str, Any],
) -> dict[str, Any]:
    after_positions = _body_positions(env)
    geom = _geometry(env, "sfp_tip_link")
    module_geom = _geometry(env, "sfp_module_link")
    realized: dict[str, Any] = {}
    for name, after in after_positions.items():
        before = before_positions.get(name)
        if after is None or before is None:
            realized[name] = None
            continue
        delta = torch.tensor(after, dtype=torch.float32) - torch.tensor(before, dtype=torch.float32)
        realized[name] = {
            "delta_norm_m_by_env": _tensor_list(torch.linalg.norm(delta, dim=1)),
            "delta_world_by_env": [[float(v) for v in row] for row in delta.tolist()],
        }
    return {
        "step": step,
        "phase": phase,
        "action_info": action_info,
        "action_env0": _tensor_list(action[0]),
        "post_step_insertion_geometry": geom,
        "post_step_module_geometry": module_geom,
        "contact": _contact_summary(env),
        "realized_body_motion": realized,
        "relative_body_transforms": _relative_body_transforms(env),
    }


def _strict_any(row: dict[str, Any]) -> bool:
    geom = row.get("post_step_insertion_geometry") or {}
    return any(bool(x) for x in geom.get("strict_success_by_env") or [])


def _configure_semantic_reward_terms(env_cfg) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    body_names = [str(args_cli.target_reward_body)]
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_distance_progress",
        "target_orientation_tanh",
        "target_orientation_gated_exp",
        "target_reaching_bonus",
        "target_success_once_bonus",
        "target_lateral_error",
        "target_motion_projection",
        "target_lateral_progress",
        "target_axial_progress",
        "target_insertion_corridor",
        "target_cheatcode_phase_reward",
    ):
        term = getattr(rewards, name, None)
        params = None if term is None else getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = "axis"
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)
        if "consistency_axial_std" in params:
            params["consistency_axial_std"] = float(args_cli.target_reward_consistency_axial_std)
        if "consistency_lateral_sigma" in params:
            params["consistency_lateral_sigma"] = float(args_cli.target_reward_consistency_lateral_sigma)
    terminations = getattr(env_cfg, "terminations", None)
    target_success = None if terminations is None else getattr(terminations, "target_success", None)
    params = None if target_success is None else getattr(target_success, "params", None)
    if isinstance(params, dict):
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = "axis"
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)


def _configure_near_gate_reset(env_cfg) -> dict[str, Any]:
    events = getattr(env_cfg, "events", None)
    term = None if events is None else getattr(events, "reset_robot_tcp_to_episode_start", None)
    params = None if term is None else getattr(term, "params", None)
    if not isinstance(params, dict):
        return {"configured": False}
    before = dict(params)
    if int(args_cli.near_gate_reset_max_iterations) > 0:
        params["max_iterations"] = int(args_cli.near_gate_reset_max_iterations)
    if float(args_cli.near_gate_reset_position_tolerance) > 0.0:
        params["position_tolerance"] = float(args_cli.near_gate_reset_position_tolerance)
    if float(args_cli.near_gate_reset_orientation_tolerance) > 0.0:
        params["orientation_tolerance"] = float(args_cli.near_gate_reset_orientation_tolerance)
    return {"configured": True, "before": before, "after": dict(params)}


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    best_score = -1.0e9
    for row in rows:
        geom = row.get("post_step_insertion_geometry") or {}
        s_vals = geom.get("signed_depth_m_by_env") or []
        r_vals = geom.get("lateral_error_m_by_env") or []
        theta_vals = geom.get("orientation_error_rad_by_env") or []
        cons_vals = geom.get("consistency_gate_by_env") or []
        for idx in range(max(len(s_vals), len(r_vals), len(theta_vals), len(cons_vals), 0)):
            s = float(s_vals[min(idx, len(s_vals) - 1)]) if s_vals else -1.0
            r = float(r_vals[min(idx, len(r_vals) - 1)]) if r_vals else 1.0
            theta = float(theta_vals[min(idx, len(theta_vals) - 1)]) if theta_vals else 1.0
            cons = float(cons_vals[min(idx, len(cons_vals) - 1)]) if cons_vals else 0.0
            score = 10.0 * min(max(s, 0.0) / STRICT["min_depth_m"], 1.0) - 1000.0 * max(r - 0.0005, 0.0) - 20.0 * max(theta - 0.030, 0.0) + cons
            if score > best_score:
                best_score = score
                best = {"step": row["step"], "env": idx, "s_m": s, "r_m": r, "theta_rad": theta, "module_consistency": cons, "score": score}
    return best


def main() -> int:
    torch.manual_seed(int(args_cli.seed))
    output_root = Path(args_cli.output_dir)
    run_dir = output_root / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps({"stage": "created_run_dir"}, indent=2) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(str(x)) for x in sys.argv) + "\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(_run_git(["diff", "--", "."]), encoding="utf-8")
    run_config = dict(vars(args_cli))
    run_config["prepared_episode_config_dir"] = prepared_episode_config_dir
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[wrist-contact] run_dir={run_dir}", flush=True)

    env = None
    try:
        status_path.write_text(json.dumps({"stage": "parse_env_cfg"}, indent=2) + "\n", encoding="utf-8")
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg.seed = int(args_cli.seed)
        if float(args_cli.episode_length_s) > 0.0:
            env_cfg.episode_length_s = float(args_cli.episode_length_s)
        if hasattr(env_cfg.observations, "policy"):
            env_cfg.observations.policy.center_rgb = None
            env_cfg.observations.policy.left_rgb = None
            env_cfg.observations.policy.right_rgb = None
        env_cfg.actions.arm_action.scale = float(args_cli.isaac_action_scale)
        _configure_semantic_reward_terms(env_cfg)
        reset_config = _configure_near_gate_reset(env_cfg)

        status_path.write_text(json.dumps({"stage": "gym_make"}, indent=2) + "\n", encoding="utf-8")
        env = gym.make(args_cli.task, cfg=env_cfg)
        rows: list[dict[str, Any]] = []
        status_path.write_text(json.dumps({"stage": "reset"}, indent=2) + "\n", encoding="utf-8")
        env.reset(seed=int(args_cli.seed))
        pose_hold_pos = _body_position(env, str(args_cli.pose_hold_body))
        pose_hold_quat = _body_orientation(env, str(args_cli.pose_hold_body))
        if pose_hold_pos is not None and pose_hold_quat is not None:
            setattr(env.unwrapped, "_aic_pose_hold_target_pos_w", pose_hold_pos.detach().clone())
            setattr(env.unwrapped, "_aic_pose_hold_target_quat_w", pose_hold_quat.detach().clone())
        target_tip_pos = _body_position(env, "sfp_tip_link")
        if target_tip_pos is not None:
            setattr(env.unwrapped, "_aic_target_tip_stabilize_pos_w", target_tip_pos.detach().clone())
        reset_diagnostic = _reset_diagnostic(env)
        (run_dir / "reset_diagnostic.json").write_text(
            json.dumps(_jsonable(reset_diagnostic), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metrics_path = run_dir / "metrics.jsonl"
        total_steps = int(args_cli.approach_steps) + int(args_cli.probe_steps)
        for step in range(1, total_steps + 1):
            status_path.write_text(
                json.dumps({"stage": "step", "step": step, "total_steps": total_steps}, indent=2) + "\n",
                encoding="utf-8",
            )
            phase = "approach" if step <= int(args_cli.approach_steps) else "probe"
            probe_step = max(0, step - int(args_cli.approach_steps) - 1)
            before_positions = _body_positions(env)
            if phase == "approach":
                action, action_info = _approach_action(env)
            else:
                action, action_info = _probe_action(env, probe_step)
            env.step(action)
            record = _step_record(
                env,
                step=step,
                phase=phase,
                action=action,
                action_info=action_info,
                before_positions=before_positions,
            )
            rows.append(record)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")
            if args_cli.save_images and step % max(int(args_cli.image_log_every), 1) == 0 and step <= int(args_cli.max_logged_image_steps):
                try:
                    record["saved_images"] = _save_step_images(env, run_dir=run_dir, step=step, record=record)
                except Exception as exc:
                    record["image_save_error"] = f"{type(exc).__name__}: {exc}"
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(_jsonable({"step": step, "image_save_error": record["image_save_error"]}), sort_keys=True) + "\n")
        summary = {
            "run_dir": str(run_dir),
            "strict_success": any(_strict_any(row) for row in rows),
            "best_row": _best_row(rows),
            "final_row": rows[-1] if rows else None,
            "reset_diagnostic": reset_diagnostic,
            "near_gate_reset_config": reset_config,
            "phase": "complete",
            "interpretation": "direct_wrist_ik_contact_realization_probe",
        }
        (run_dir / "wrist_contact_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary_md(run_dir, summary)
        status_path.write_text(json.dumps({"stage": "complete"}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "strict_success": summary["strict_success"], "best_row": summary["best_row"]}, indent=2))
    except Exception as exc:
        error = {
            "stage": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        status_path.write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        (run_dir / "error.json").write_text(json.dumps(error, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(error, indent=2), flush=True)
        return 1
    finally:
        if env is not None:
            env.close()
    return 0


def _write_summary_md(run_dir: Path, summary: dict[str, Any]) -> None:
    best = summary.get("best_row") or {}
    lines = [
        "# Wrist/contact realization diagnostic",
        "",
        f"Run: `{run_dir}`",
        f"Strict success: `{str(bool(summary.get('strict_success'))).lower()}`",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| best step | {best.get('step')} |",
        f"| best env | {best.get('env')} |",
        f"| best s mm | {1000.0 * float(best.get('s_m', float('nan'))):.3f} |",
        f"| best r mm | {1000.0 * float(best.get('r_m', float('nan'))):.3f} |",
        f"| best theta rad | {float(best.get('theta_rad', float('nan'))):.5f} |",
        f"| best consistency | {float(best.get('module_consistency', float('nan'))):.3f} |",
        "",
        "This run bypasses ACT/SERL and sends scripted wrist IK commands through `env.step`.",
    ]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
