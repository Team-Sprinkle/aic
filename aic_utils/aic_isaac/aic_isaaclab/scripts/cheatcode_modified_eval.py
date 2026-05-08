#!/usr/bin/env python3
"""Run a CheatCodeModified-style evaluation in Isaac Lab and write output.csv.

This script reproduces the intended "misalign then insert" behavior used for
force-parity checks and logs trajectory + wrench traces in one CSV.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import logging
import math
import re
import traceback
from pathlib import Path
from typing import Any

import yaml

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="CheatCodeModified-style policy runner for Isaac force parity."
)
parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments. Force-parity mode supports only 1.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--target_module_name", type=str, default="", help="Task metadata label.")
parser.add_argument("--port_name", type=str, default="", help="Task metadata label.")
parser.add_argument("--plug_name", type=str, default="", help="Task metadata label.")
parser.add_argument("--cable_name", type=str, default="", help="Task metadata label.")
parser.add_argument("--cable_type", type=str, default="", help="Task metadata label.")
parser.add_argument(
    "--out",
    type=str,
    default="aic/outputs/force_parity/output.csv",
    help="Output CSV path.",
)
parser.add_argument(
    "--force_log_body",
    type=str,
    default="wrist_3_link",
    help="Body name suffix used for force logging.",
)
parser.add_argument(
    "--misalign_x_m",
    type=float,
    default=0.004,
    help="Intentional x misalignment in meters.",
)
parser.add_argument(
    "--misalign_y_m",
    type=float,
    default=0.0,
    help="Intentional y misalignment in meters.",
)
parser.add_argument(
    "--align_seconds",
    type=float,
    default=1.0,
    help="Duration to apply lateral misalignment motion.",
)
parser.add_argument(
    "--initial_z_offset_m",
    type=float,
    default=0.2,
    help="CheatCode-style initial height offset used for handoff distance.",
)
parser.add_argument(
    "--start_descent_z_offset_m",
    type=float,
    default=0.005,
    help="CheatCode-style start insertion height offset.",
)
parser.add_argument(
    "--end_z_offset_m",
    type=float,
    default=-0.015,
    help="CheatCode-style insertion end height offset.",
)
parser.add_argument(
    "--handoff_min_seconds",
    type=float,
    default=2.0,
    help="Minimum duration of the pre-insertion handoff phase.",
)
parser.add_argument(
    "--handoff_speed_mps",
    type=float,
    default=0.02,
    help="Descent speed during handoff phase (m/s).",
)
parser.add_argument(
    "--settle_seconds",
    type=float,
    default=1.0,
    help="Settle duration before insertion.",
)
parser.add_argument(
    "--insertion_speed_mps",
    type=float,
    default=0.0009,
    help="Insertion speed in m/s (minimum-jerk schedule).",
)
parser.add_argument(
    "--hold_seconds",
    type=float,
    default=5.0,
    help="Hold duration after insertion motion.",
)
parser.add_argument(
    "--force_backoff_threshold_n",
    type=float,
    default=1.0e9,
    help="Absolute delta-from-baseline Fz threshold to trigger immediate backoff.",
)
parser.add_argument(
    "--backoff_m",
    type=float,
    default=0.015,
    help="Backoff distance in +Z after force trigger.",
)
parser.add_argument(
    "--backoff_seconds",
    type=float,
    default=0.7,
    help="Backoff command duration.",
)
parser.add_argument(
    "--gazebo_trial_config",
    type=str,
    default="",
    help="Optional Gazebo/aic_engine YAML config path for pose parity conversion.",
)
parser.add_argument(
    "--gazebo_trial_id",
    type=str,
    default="trial_1",
    help="Trial id under 'trials' in the Gazebo/aic_engine config.",
)
parser.add_argument(
    "--enable_gazebo_parity",
    action="store_true",
    default=False,
    help="Apply Gazebo->Isaac frame/pose conversion from the trial config.",
)
parser.add_argument(
    "--gazebo_to_isaac_offset_x",
    type=float,
    default=0.0,
    help="Gazebo->Isaac world transform translation x (meters).",
)
parser.add_argument(
    "--gazebo_to_isaac_offset_y",
    type=float,
    default=0.0,
    help="Gazebo->Isaac world transform translation y (meters).",
)
parser.add_argument(
    "--gazebo_to_isaac_offset_z",
    type=float,
    default=-1.14,
    help="Gazebo->Isaac world transform translation z (meters).",
)
parser.add_argument(
    "--gazebo_to_isaac_yaw_offset",
    type=float,
    default=math.pi / 2.0,
    help="Gazebo->Isaac world transform yaw offset (radians).",
)
parser.add_argument(
    "--gazebo_to_isaac_board_yaw_extra",
    type=float,
    default=math.pi,
    help="Additional board-only yaw rotation (radians) after Gazebo->Isaac transform.",
)
parser.add_argument("--robot_offset_x", type=float, default=0.08, help="Robot world-frame x offset (m).")
parser.add_argument("--robot_offset_y", type=float, default=0.08, help="Robot world-frame y offset (m).")
parser.add_argument("--robot_offset_yaw", type=float, default=0.0, help="Robot world-frame yaw offset (rad).")
parser.add_argument("--task_board_offset_x", type=float, default=0.0, help="Task board world-frame x offset (m).")
parser.add_argument("--task_board_offset_y", type=float, default=0.12, help="Task board world-frame y offset (m).")
parser.add_argument(
    "--task_board_offset_yaw", type=float, default=0.0, help="Task board world-frame yaw offset (rad)."
)
parser.add_argument("--sc_port_offset_x", type=float, default=0.0, help="sc_port world-frame x offset (m).")
parser.add_argument("--sc_port_offset_y", type=float, default=0.0, help="sc_port world-frame y offset (m).")
parser.add_argument("--sc_port_offset_yaw", type=float, default=0.0, help="sc_port world-frame yaw offset (rad).")
parser.add_argument("--sc_port_2_offset_x", type=float, default=0.0, help="sc_port_2 world-frame x offset (m).")
parser.add_argument("--sc_port_2_offset_y", type=float, default=0.0, help="sc_port_2 world-frame y offset (m).")
parser.add_argument(
    "--sc_port_2_offset_yaw", type=float, default=0.0, help="sc_port_2 world-frame yaw offset (rad)."
)
parser.add_argument("--nic_card_offset_x", type=float, default=0.0, help="nic_card world-frame x offset (m).")
parser.add_argument("--nic_card_offset_y", type=float, default=0.0, help="nic_card world-frame y offset (m).")
parser.add_argument("--nic_card_offset_yaw", type=float, default=0.0, help="nic_card world-frame yaw offset (rad).")
parser.add_argument(
    "--no_match_gazebo_physics",
    action="store_true",
    default=False,
    help="Do not align Isaac physics step/solver defaults to Gazebo world settings.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

if args_cli.num_envs != 1:
    raise ValueError("--num_envs must be 1 for force-parity policy evaluation.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401

LOGGER = logging.getLogger("aic.cheatcode_eval")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_wrist_index(robot: Any, body_name_suffix: str) -> int:
    body_names = list(robot.body_names)
    for i, body_name in enumerate(body_names):
        if body_name.endswith(body_name_suffix):
            return i
    raise RuntimeError(
        f"Could not find body suffix '{body_name_suffix}' in robot.body_names: {body_names}"
    )


def _clear_scene_randomization(env_cfg: Any) -> None:
    if not hasattr(env_cfg, "events"):
        return
    if not hasattr(env_cfg.events, "randomize_board_and_parts"):
        return
    event_term = env_cfg.events.randomize_board_and_parts
    if event_term is None or not hasattr(event_term, "params"):
        return
    params = event_term.params
    params["board_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    parts = params.get("parts", [])
    for part_cfg in parts:
        if not isinstance(part_cfg, dict):
            continue
        part_cfg["pose_range"] = {}
        if "snap_step" in part_cfg:
            part_cfg.pop("snap_step")


def _min_jerk_fraction(s: float) -> float:
    return 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5


def _quat_conjugate(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_mul(
    q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n <= 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return _quat_normalize(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _quat_apply(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    qv = (0.0, v[0], v[1], v[2])
    r = _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))
    return (r[1], r[2], r[3])


def _transform_gazebo_world_to_isaac(
    pos_gz: tuple[float, float, float],
    rpy_gz: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    c = math.cos(args_cli.gazebo_to_isaac_yaw_offset)
    s = math.sin(args_cli.gazebo_to_isaac_yaw_offset)
    x_rot = c * pos_gz[0] - s * pos_gz[1]
    y_rot = s * pos_gz[0] + c * pos_gz[1]
    pos_isaac = (
        x_rot + args_cli.gazebo_to_isaac_offset_x,
        y_rot + args_cli.gazebo_to_isaac_offset_y,
        pos_gz[2] + args_cli.gazebo_to_isaac_offset_z,
    )
    # Keep world mapping strictly planar: preserve only yaw in the world transform.
    # This prevents accidental roll/pitch "tilt" from leaking into Isaac world poses.
    rpy_isaac = (
        0.0,
        0.0,
        rpy_gz[2] + args_cli.gazebo_to_isaac_yaw_offset,
    )
    return pos_isaac, rpy_isaac


def _compose_local_pose(
    parent_world_pos: tuple[float, float, float],
    parent_world_quat: tuple[float, float, float, float],
    child_local_pos: tuple[float, float, float],
    child_local_rpy: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    child_local_quat = _quat_from_rpy(*child_local_rpy)
    child_world_quat = _quat_normalize(_quat_mul(parent_world_quat, child_local_quat))
    rotated = _quat_apply(parent_world_quat, child_local_pos)
    child_world_pos = (
        parent_world_pos[0] + rotated[0],
        parent_world_pos[1] + rotated[1],
        parent_world_pos[2] + rotated[2],
    )
    return child_world_pos, child_world_quat


def _apply_world_xy_yaw_offset(
    pos_w: tuple[float, float, float],
    quat_w: tuple[float, float, float, float],
    dx: float,
    dy: float,
    dyaw: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    out_pos = (pos_w[0] + dx, pos_w[1] + dy, pos_w[2])
    if abs(dyaw) <= 1.0e-12:
        return out_pos, quat_w
    q_delta = _quat_from_rpy(0.0, 0.0, dyaw)
    out_quat = _quat_normalize(_quat_mul(q_delta, quat_w))
    return out_pos, out_quat


def _first_present_rail(task_board_cfg: dict[str, Any], rail_prefix: str, count: int) -> int | None:
    for i in range(count):
        rail_key = f"{rail_prefix}_{i}"
        rail_cfg = task_board_cfg.get(rail_key, {})
        if bool(rail_cfg.get("entity_present", False)):
            return i
    return None


def _select_target_nic_rail_idx(trial_cfg: dict[str, Any], task_board_cfg: dict[str, Any]) -> int | None:
    target_name = ""
    try:
        target_name = str(trial_cfg["tasks"]["task_1"]["target_module_name"])
    except Exception:
        target_name = ""
    if target_name.startswith("nic_card_mount_"):
        suffix = target_name.removeprefix("nic_card_mount_")
        if suffix.isdigit():
            idx = int(suffix)
            rail_key = f"nic_rail_{idx}"
            rail_cfg = task_board_cfg.get(rail_key, {})
            if bool(rail_cfg.get("entity_present", False)):
                return idx
    return _first_present_rail(task_board_cfg, "nic_rail", 5)


def _as_pose_tuple(pose_cfg: dict[str, Any]) -> tuple[float, float, float]:
    return (float(pose_cfg.get("x", 0.0)), float(pose_cfg.get("y", 0.0)), float(pose_cfg.get("z", 0.0)))


def _as_rpy_tuple(pose_cfg: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(pose_cfg.get("roll", 0.0)),
        float(pose_cfg.get("pitch", 0.0)),
        float(pose_cfg.get("yaw", 0.0)),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def _measure_nic_rails_from_xacro() -> tuple[float, list[float], float] | None:
    """Read NIC rail geometry from Gazebo task_board xacro.

    Returns:
        (spacing_m, y_by_idx_gz, x_base_gz) or None on failure.
    """
    xacro_path = (
        Path(__file__).resolve().parents[4] / "aic_description" / "urdf" / "task_board.urdf.xacro"
    )
    if not xacro_path.is_file():
        return None
    text = xacro_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'nic_card_mount_(\d+)" pose="\$\{([-0-9.]+) \+ \$\(arg nic_card_mount_\d+_translation\)\} ([-0-9.]+) '
    )
    matches = pattern.findall(text)
    if len(matches) < 5:
        return None
    data = sorted((int(idx), float(xbase), float(yval)) for idx, xbase, yval in matches[:5])
    y_by_idx = [y for _, _, y in data]
    x_base = float(data[0][1])
    diffs = [abs(y_by_idx[i + 1] - y_by_idx[i]) for i in range(len(y_by_idx) - 1)]
    spacing = sum(diffs) / len(diffs) if diffs else 0.04
    return spacing, y_by_idx, x_base


def _apply_gazebo_trial_parity(env_cfg: Any) -> None:
    if not args_cli.gazebo_trial_config:
        raise ValueError("--enable_gazebo_parity requires --gazebo_trial_config")
    hidden_pos = (10.0, 10.0, -10.0)
    identity_quat = (1.0, 0.0, 0.0, 0.0)

    config_path = Path(args_cli.gazebo_trial_config).expanduser()
    config_root = _load_yaml(config_path)
    trials = config_root.get("trials", {})
    if args_cli.gazebo_trial_id not in trials:
        raise KeyError(
            f"Trial '{args_cli.gazebo_trial_id}' not found in {config_path}. "
            f"Available: {list(trials.keys())}"
        )
    trial_cfg = trials[args_cli.gazebo_trial_id]
    scene_cfg = trial_cfg["scene"]
    task_board_cfg = scene_cfg["task_board"]
    board_pose_cfg = task_board_cfg["pose"]

    board_pos_gz = _as_pose_tuple(board_pose_cfg)
    board_rpy_gz = _as_rpy_tuple(board_pose_cfg)
    board_pos_is, board_rpy_is = _transform_gazebo_world_to_isaac(board_pos_gz, board_rpy_gz)
    board_rpy_is = (
        board_rpy_is[0],
        board_rpy_is[1],
        board_rpy_is[2] + args_cli.gazebo_to_isaac_board_yaw_extra,
    )
    board_quat_is = _quat_from_rpy(*board_rpy_is)
    board_pos_is, board_quat_is = _apply_world_xy_yaw_offset(
        board_pos_is,
        board_quat_is,
        args_cli.task_board_offset_x,
        args_cli.task_board_offset_y,
        args_cli.task_board_offset_yaw,
    )

    env_cfg.scene.task_board.init_state.pos = board_pos_is
    env_cfg.scene.task_board.init_state.rot = board_quat_is

    # SC ports: use Isaac-local rail anchors and apply Gazebo translation along
    # the SC rail direction (x-axis in Isaac-local board frame).
    sc_anchor_local = {
        0: (0.0067, -0.0362, 0.005),
        1: (0.0076, -0.0783, 0.005),
    }
    sc_base_local_quat = {
        0: _quat_normalize(tuple(float(v) for v in env_cfg.scene.sc_port.init_state.rot)),
        1: _quat_normalize(tuple(float(v) for v in env_cfg.scene.sc_port_2.init_state.rot)),
    }
    for idx, scene_name in ((0, "sc_port"), (1, "sc_port_2")):
        rail_key = f"sc_rail_{idx}"
        rail_cfg = task_board_cfg.get(rail_key, {})
        if not bool(rail_cfg.get("entity_present", False)):
            LOGGER.info("Gazebo parity: %s not present; hiding %s.", rail_key, scene_name)
            getattr(env_cfg.scene, scene_name).init_state.pos = hidden_pos
            getattr(env_cfg.scene, scene_name).init_state.rot = identity_quat
            continue
        pose = rail_cfg.get("entity_pose", {})
        trans = -float(pose.get("translation", 0.0))
        roll = float(pose.get("roll", 0.0))
        pitch = float(pose.get("pitch", 0.0))
        yaw = float(pose.get("yaw", 0.0))
        ax, ay, az = sc_anchor_local[idx]
        local_pos = (ax + trans, ay, az)
        local_delta_quat = _quat_from_rpy(roll, pitch, yaw)
        local_quat = _quat_normalize(_quat_mul(sc_base_local_quat[idx], local_delta_quat))
        part_pos, part_quat = _compose_local_pose(board_pos_is, board_quat_is, local_pos, (0.0, 0.0, 0.0))
        part_quat = _quat_normalize(_quat_mul(board_quat_is, local_quat))
        if scene_name == "sc_port":
            part_pos, part_quat = _apply_world_xy_yaw_offset(
                part_pos,
                part_quat,
                args_cli.sc_port_offset_x,
                args_cli.sc_port_offset_y,
                args_cli.sc_port_offset_yaw,
            )
        else:
            part_pos, part_quat = _apply_world_xy_yaw_offset(
                part_pos,
                part_quat,
                args_cli.sc_port_2_offset_x,
                args_cli.sc_port_2_offset_y,
                args_cli.sc_port_2_offset_yaw,
            )
        getattr(env_cfg.scene, scene_name).init_state.pos = part_pos
        getattr(env_cfg.scene, scene_name).init_state.rot = part_quat
        LOGGER.info(
            "Gazebo parity: %s -> %s world_pos=%s world_quat=%s",
            rail_key,
            scene_name,
            part_pos,
            part_quat,
        )

    # NIC card mount pose in Gazebo is rail-relative.
    nic_idx = _select_target_nic_rail_idx(trial_cfg, task_board_cfg)
    if nic_idx is not None:
        rail_key = f"nic_rail_{nic_idx}"
        rail_cfg = task_board_cfg.get(rail_key, {})
        pose = rail_cfg.get("entity_pose", {})
        trans = -float(pose.get("translation", 0.0))
        roll = float(pose.get("roll", 0.0))
        pitch = float(pose.get("pitch", 0.0))
        yaw = float(pose.get("yaw", 0.0))
        # Isaac-local rail model:
        # - slot index chooses local y row
        # - Gazebo NIC rail translation maps to local x (as in task_board xacro)
        # - x correction is measured from repository xacro vs Isaac center anchor.
        nic_anchor_local_center = (-0.03235, 0.02329, 0.0743)
        nic_base_local_quat = _quat_normalize(tuple(float(v) for v in env_cfg.scene.nic_card.init_state.rot))
        rail_info = _measure_nic_rails_from_xacro()
        if rail_info is not None:
            spacing_m, y_by_idx_gz, x_base_gz = rail_info
            y_by_idx = [
                nic_anchor_local_center[1] + (2 - i) * spacing_m for i in range(5)
            ]
            # Gazebo NIC mount x-base vs Isaac NIC center-anchor x.
            nic_local_x_correction = x_base_gz - nic_anchor_local_center[0]
            LOGGER.info(
                "Gazebo parity: measured NIC rails from xacro spacing=%.5f y_gz=%s x_base=%.6f -> local_x_correction=%.6f",
                spacing_m,
                y_by_idx_gz,
                x_base_gz,
                nic_local_x_correction,
            )
        else:
            y_by_idx = [0.10329, 0.06329, 0.02329, -0.01671, -0.05671]
            nic_local_x_correction = 0.0
            spacing_m = 0.04
            LOGGER.warning(
                "Gazebo parity: could not measure NIC rails from xacro; using fallback y rails and x correction 0.0"
            )
        nic_pos_y_shift = 2.0 * spacing_m
        local_pos = (
            nic_anchor_local_center[0] + nic_local_x_correction + trans,
            y_by_idx[nic_idx] + nic_pos_y_shift,
            nic_anchor_local_center[2],
        )
        local_delta_quat = _quat_from_rpy(roll, pitch, yaw)
        local_quat = _quat_normalize(_quat_mul(nic_base_local_quat, local_delta_quat))
        nic_pos, nic_quat = _compose_local_pose(board_pos_is, board_quat_is, local_pos, (0.0, 0.0, 0.0))
        nic_quat = _quat_normalize(_quat_mul(board_quat_is, local_quat))
        nic_pos, nic_quat = _apply_world_xy_yaw_offset(
            nic_pos,
            nic_quat,
            args_cli.nic_card_offset_x,
            args_cli.nic_card_offset_y,
            args_cli.nic_card_offset_yaw,
        )
        env_cfg.scene.nic_card.init_state.pos = nic_pos
        env_cfg.scene.nic_card.init_state.rot = nic_quat
        LOGGER.info(
            "Gazebo parity: nic_rail_%d -> nic_card world_pos=%s world_quat=%s",
            nic_idx,
            nic_pos,
            nic_quat,
        )
    else:
        LOGGER.info("Gazebo parity: no present nic_rail_* found for trial; hiding nic_card.")
        env_cfg.scene.nic_card.init_state.pos = hidden_pos
        env_cfg.scene.nic_card.init_state.rot = identity_quat

    # Use official robot home joint positions when provided.
    robot_cfg = config_root.get("robot", {})
    # Match Gazebo robot spawn arguments from aic_gz_bringup.launch.py, then
    # map through Gazebo->Isaac world transform so robot remains above workcell.
    robot_pos_gz = (-0.2, 0.2, 1.14)
    robot_rpy_gz = (0.0, 0.0, -3.141)
    robot_pos_is, robot_rpy_is = _transform_gazebo_world_to_isaac(robot_pos_gz, robot_rpy_gz)
    robot_quat_is = _quat_from_rpy(*robot_rpy_is)
    robot_pos_is, robot_quat_is = _apply_world_xy_yaw_offset(
        robot_pos_is,
        robot_quat_is,
        args_cli.robot_offset_x,
        args_cli.robot_offset_y,
        args_cli.robot_offset_yaw,
    )
    env_cfg.scene.robot.init_state.pos = robot_pos_is
    env_cfg.scene.robot.init_state.rot = robot_quat_is
    LOGGER.info(
        "Gazebo parity: robot spawn gz_pos=%s gz_rpy=%s -> isaac_pos=%s isaac_rpy=%s",
        robot_pos_gz,
        robot_rpy_gz,
        robot_pos_is,
        robot_rpy_is,
    )
    home_joint_positions = robot_cfg.get("home_joint_positions")
    if isinstance(home_joint_positions, dict):
        env_cfg.scene.robot.init_state.joint_pos = {
            str(k): float(v) for k, v in home_joint_positions.items()
        }

    LOGGER.info(
        "Applied Gazebo parity from %s (%s): board_pos_gz=%s board_pos_isaac=%s",
        config_path,
        args_cli.gazebo_trial_id,
        board_pos_gz,
        board_pos_is,
    )


def _apply_gazebo_physics_match(env_cfg: Any) -> None:
    # Gazebo world uses 2 ms Bullet Featherstone physics steps (500 Hz).
    env_cfg.sim.dt = 0.002
    env_cfg.decimation = 1
    env_cfg.sim.render_interval = 1
    if hasattr(env_cfg.sim, "gravity"):
        env_cfg.sim.gravity = (0.0, 0.0, -9.8)
    physx_cfg = getattr(env_cfg.sim, "physx", None)
    if physx_cfg is not None and hasattr(physx_cfg, "solver_type"):
        # PGS is closer in behavior to non-TGS schemes than PhysX TGS.
        physx_cfg.solver_type = 0
    LOGGER.info(
        "Applied Gazebo physics match: dt=%s decimation=%s render_interval=%s gravity=%s physx.solver_type=%s",
        getattr(env_cfg.sim, "dt", None),
        getattr(env_cfg, "decimation", None),
        getattr(env_cfg.sim, "render_interval", None),
        getattr(env_cfg.sim, "gravity", None),
        getattr(getattr(env_cfg.sim, "physx", None), "solver_type", None),
    )


def _to_xyz_tuple(value: Any) -> tuple[float, float, float] | str:
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return str(value)


def _log_scene_profile_snapshot(env_cfg: Any, env: Any) -> None:
    for key in ("task_board", "sc_port", "sc_port_2", "nic_card"):
        cfg_pos = None
        cfg_rot = None
        try:
            cfg_obj = getattr(env_cfg.scene, key)
            cfg_pos = _to_xyz_tuple(cfg_obj.init_state.pos)
            cfg_rot = tuple(float(v) for v in cfg_obj.init_state.rot)
        except Exception:
            cfg_pos = "unavailable"
            cfg_rot = "unavailable"

        live_pos = None
        live_rot = None
        try:
            obj = env.scene[key]
            live_pos = _to_xyz_tuple(obj.data.root_pos_w[0, :3])
            live_rot = tuple(float(v) for v in obj.data.root_quat_w[0, :4].tolist())
        except Exception:
            live_pos = "unavailable"
            live_rot = "unavailable"

        LOGGER.info(
            "Scene profile '%s': cfg_init_pos=%s cfg_init_rot=%s live_root_pos=%s live_root_rot=%s",
            key,
            cfg_pos,
            cfg_rot,
            live_pos,
            live_rot,
        )


def main() -> None:
    _setup_logging()
    LOGGER.info("Starting cheatcode_modified_eval.")
    LOGGER.info("CLI args: %s", vars(args_cli))
    env = None
    try:
        task_meta = {
            "target_module_name": str(args_cli.target_module_name),
            "port_name": str(args_cli.port_name),
            "plug_name": str(args_cli.plug_name),
            "cable_name": str(args_cli.cable_name),
            "cable_type": str(args_cli.cable_type),
        }
        LOGGER.info("Task metadata: %s", task_meta)
        LOGGER.info(
            "Launcher flags: headless=%s, enable_cameras=%s, device=%s",
            getattr(args_cli, "headless", None),
            getattr(args_cli, "enable_cameras", None),
            getattr(args_cli, "device", None),
        )

        LOGGER.info("Parsing environment config for task '%s'...", args_cli.task)
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        if not args_cli.no_match_gazebo_physics:
            _apply_gazebo_physics_match(env_cfg)
        if args_cli.enable_gazebo_parity:
            _apply_gazebo_trial_parity(env_cfg)
        LOGGER.info(
            "Parsed env cfg: decimation=%s, sim.dt=%s, render_interval=%s",
            getattr(env_cfg, "decimation", None),
            getattr(getattr(env_cfg, "sim", None), "dt", None),
            getattr(env_cfg, "render_interval", None),
        )
        LOGGER.info(
            "Env cfg class: %s.%s (file=%s)",
            env_cfg.__class__.__module__,
            env_cfg.__class__.__name__,
            inspect.getsourcefile(env_cfg.__class__),
        )
        LOGGER.info("Action term arm_action: %s", getattr(env_cfg.actions, "arm_action", None))
        LOGGER.info(
            "Event terms: reset_robot_joints=%s randomize_light=%s randomize_board_and_parts=%s",
            getattr(env_cfg.events, "reset_robot_joints", None),
            getattr(env_cfg.events, "randomize_light", None),
            getattr(env_cfg.events, "randomize_board_and_parts", None),
        )
        _clear_scene_randomization(env_cfg)
        LOGGER.info("Creating gym environment...")
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        LOGGER.info("Environment created. Calling reset()...")
        env.reset()
        LOGGER.info("Environment reset complete.")
        _log_scene_profile_snapshot(env_cfg, env)

        robot = env.scene["robot"]
        LOGGER.info("Robot body names: %s", list(robot.body_names))
        wrist_idx = _resolve_wrist_index(robot, args_cli.force_log_body)
        step_dt = float(getattr(env, "step_dt", 1.0 / 60.0))
        action_shape = env.action_space.shape
        if len(action_shape) != 2:
            raise RuntimeError(f"Expected vectorized action space shape (N, A), got {action_shape}")
        action_dim = int(action_shape[1])

        if action_dim < 3:
            raise RuntimeError(
                f"Expected at least 3 action dims (xyz delta), got {action_dim}"
            )

        align_steps = max(1, int(round(args_cli.align_seconds / step_dt)))
        handoff_distance = max(
            0.0, args_cli.initial_z_offset_m - args_cli.start_descent_z_offset_m
        )
        handoff_seconds = max(
            args_cli.handoff_min_seconds,
            handoff_distance / max(args_cli.handoff_speed_mps, 1.0e-9),
        )
        handoff_steps = max(1, int(round(handoff_seconds / step_dt)))
        settle_steps = max(1, int(round(args_cli.settle_seconds / step_dt)))
        insertion_distance = max(
            0.0, args_cli.start_descent_z_offset_m - args_cli.end_z_offset_m
        )
        insertion_seconds = max(
            step_dt, insertion_distance / max(args_cli.insertion_speed_mps, 1.0e-9)
        )
        insertion_steps = max(1, int(round(insertion_seconds / step_dt)))
        hold_steps = max(1, int(round(args_cli.hold_seconds / step_dt)))
        backoff_steps = max(1, int(round(args_cli.backoff_seconds / step_dt)))

        print(
            "CheatCodeModified-style schedule "
            f"(dt={step_dt:.6f}s): align={align_steps}, handoff={handoff_steps}, "
            f"settle={settle_steps}, insertion={insertion_steps}, hold={hold_steps}"
        )
        print(
            "Task metadata labels: "
            f"target_module={task_meta['target_module_name']}, "
            f"port={task_meta['port_name']}, plug={task_meta['plug_name']}"
        )

        rows: list[list[str]] = []
        baseline_fz: float | None = None
        backoff_remaining = 0
        insertion_started = False
        insertion_triggered = False

        scale_m_per_action = 0.05
        cumulative_insertion = 0.0
        total_steps = align_steps + handoff_steps + settle_steps + insertion_steps + hold_steps
        LOGGER.info(
            "Computed schedule: total_steps=%d (align=%d handoff=%d settle=%d insertion=%d hold=%d backoff=%d)",
            total_steps,
            align_steps,
            handoff_steps,
            settle_steps,
            insertion_steps,
            hold_steps,
            backoff_steps,
        )

        for step in range(total_steps):
            if not simulation_app.is_running():
                LOGGER.warning(
                    "Simulation app stopped before step %d/%d. "
                    "No further env.step() calls will run.",
                    step,
                    total_steps,
                )
                break

            phase = "hold"
            cmd_x_m = 0.0
            cmd_y_m = 0.0
            cmd_z_m = 0.0

            if backoff_remaining > 0:
                phase = "backoff"
                cmd_z_m = args_cli.backoff_m / backoff_steps
                backoff_remaining -= 1
            else:
                if step < align_steps:
                    phase = "align"
                    cmd_x_m = args_cli.misalign_x_m / align_steps
                    cmd_y_m = args_cli.misalign_y_m / align_steps
                elif step < align_steps + handoff_steps:
                    phase = "handoff"
                    cmd_z_m = -handoff_distance / handoff_steps
                elif step < align_steps + handoff_steps + settle_steps:
                    phase = "settle"
                elif step < align_steps + handoff_steps + settle_steps + insertion_steps:
                    phase = "insertion"
                    insertion_started = True
                    insertion_step = step - (align_steps + handoff_steps + settle_steps) + 1
                    frac = _min_jerk_fraction(insertion_step / insertion_steps)
                    target_cumulative = frac * insertion_distance
                    delta = target_cumulative - cumulative_insertion
                    cumulative_insertion = target_cumulative
                    cmd_z_m = -delta
                else:
                    phase = "hold"

            cmd_x = cmd_x_m / scale_m_per_action
            cmd_y = cmd_y_m / scale_m_per_action
            cmd_z = cmd_z_m / scale_m_per_action

            with torch.inference_mode():
                actions = torch.zeros(action_shape, device=env.device)
                actions[0, 0] = cmd_x
                actions[0, 1] = cmd_y
                actions[0, 2] = cmd_z
                env.step(actions)
                if step == 0:
                    LOGGER.info("First env.step() completed successfully.")

                if hasattr(robot.data, "body_incoming_wrench_b"):
                    wrench_b = robot.data.body_incoming_wrench_b[0, wrist_idx]
                elif hasattr(robot.data, "body_incoming_joint_wrench_b"):
                    wrench_b = robot.data.body_incoming_joint_wrench_b[0, wrist_idx]
                else:
                    raise AttributeError(
                        "Robot data has neither body_incoming_wrench_b nor "
                        "body_incoming_joint_wrench_b."
                    )

                ee_pos_w = robot.data.body_pos_w[0, wrist_idx]
                if hasattr(robot.data, "body_quat_w"):
                    ee_quat_w = robot.data.body_quat_w[0, wrist_idx]
                    qw = float(ee_quat_w[0].item())
                    qx = float(ee_quat_w[1].item())
                    qy = float(ee_quat_w[2].item())
                    qz = float(ee_quat_w[3].item())
                else:
                    qw = math.nan
                    qx = math.nan
                    qy = math.nan
                    qz = math.nan

                fx = float(wrench_b[0].item())
                fy = float(wrench_b[1].item())
                fz = float(wrench_b[2].item())
                tx = float(wrench_b[3].item())
                ty = float(wrench_b[4].item())
                tz = float(wrench_b[5].item())
                ee_x = float(ee_pos_w[0].item())
                ee_y = float(ee_pos_w[1].item())
                ee_z = float(ee_pos_w[2].item())

                if insertion_started and baseline_fz is None:
                    baseline_fz = fz
                if (
                    insertion_started
                    and baseline_fz is not None
                    and not insertion_triggered
                    and abs(fz - baseline_fz) >= args_cli.force_backoff_threshold_n
                ):
                    insertion_triggered = True
                    backoff_remaining = backoff_steps

                rows.append(
                    [
                        f"{step * step_dt:.6f}",
                        str(step),
                        phase,
                        f"{cmd_x:.8f}",
                        f"{cmd_y:.8f}",
                        f"{cmd_z:.8f}",
                        f"{cmd_x_m:.8f}",
                        f"{cmd_y_m:.8f}",
                        f"{cmd_z_m:.8f}",
                        f"{fx:.8f}",
                        f"{fy:.8f}",
                        f"{fz:.8f}",
                        f"{tx:.8f}",
                        f"{ty:.8f}",
                        f"{tz:.8f}",
                        f"{ee_x:.8f}",
                        f"{ee_y:.8f}",
                        f"{ee_z:.8f}",
                        f"{qw:.8f}",
                        f"{qx:.8f}",
                        f"{qy:.8f}",
                        f"{qz:.8f}",
                        task_meta["target_module_name"],
                        task_meta["port_name"],
                        task_meta["plug_name"],
                        task_meta["cable_name"],
                        task_meta["cable_type"],
                    ]
                )

        out_path = Path(args_cli.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "time_s",
                    "step",
                    "phase",
                    "cmd_x_norm",
                    "cmd_y_norm",
                    "cmd_z_norm",
                    "cmd_x_m",
                    "cmd_y_m",
                    "cmd_z_m",
                    "force_x_n",
                    "force_y_n",
                    "force_z_n",
                    "torque_x_nm",
                    "torque_y_nm",
                    "torque_z_nm",
                    "ee_x_m",
                    "ee_y_m",
                    "ee_z_m",
                    "ee_qw",
                    "ee_qx",
                    "ee_qy",
                    "ee_qz",
                    "target_module_name",
                    "port_name",
                    "plug_name",
                    "cable_name",
                    "cable_type",
                ]
            )
            writer.writerows(rows)

        if rows:
            fz_values = [float(row[11]) for row in rows]
            peak_abs_fz = max(abs(v) for v in fz_values)
            print(
                f"Wrote {len(rows)} samples to {out_path} "
                f"(peak |Fz|={peak_abs_fz:.6f} N, triggered_backoff={insertion_triggered})"
            )
        else:
            LOGGER.warning(
                "No samples were recorded. This usually means the app was no longer running "
                "before the first simulation step."
            )
            print(f"No samples were recorded. Output CSV still created at {out_path}.")
    except Exception:
        LOGGER.error("Fatal exception during evaluation.")
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
    hidden_pos = (10.0, 10.0, -10.0)
    identity_quat = (1.0, 0.0, 0.0, 0.0)
