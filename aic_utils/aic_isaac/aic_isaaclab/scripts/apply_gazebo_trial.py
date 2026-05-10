#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Apply a Gazebo trials YAML scene to the AIC Isaac Lab environment.

Usage:
  # Calculate poses without simulation
  python scripts/apply_gazebo_trial.py --trials_yaml /path/to/trials.yaml --trial_id trial_1

  # Launch Isaac Sim for visualization
  python scripts/apply_gazebo_trial.py --trials_yaml /path/to/trials.yaml --trial_id trial_1 --sim
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import yaml


# --- Pose Constants & Utilities ---
GAZEBO_TO_ISAAC_YAW_OFFSET = 0.5 * math.pi
GAZEBO_TO_ISAAC_Z_OFFSET = -1.14
GAZEBO_TO_ISAAC_BOARD_YAW_EXTRA = math.pi
ISAAC_ROBOT_DEFAULT_YAW = -0.5 * math.pi

# Local asset anchors used by the current Isaac AIC scene. These are offsets
# from the task-board root to the existing Isaac scene assets.
ISAAC_SC_PORT_ANCHORS = {0: (0.0067, -0.0362, 0.005), 1: (0.0067, -0.083, 0.005)}
ISAAC_SC_PORT_LOCAL_QUAT = (0.73136, 0.0, 0.0, -0.682)
ISAAC_SC_PORT_2_LOCAL_QUAT = (0.999391, 0.0, 0.0, 0.034903)
ISAAC_SC_RAIL_X_LIMITS = (-0.06, 0.055)
ISAAC_NIC_ANCHOR_CENTER = (-0.03235, 0.02329, 0.0743)
ISAAC_NIC_LOCAL_QUAT = (0.0, 0.0, -0.7068252, 0.7073883)
ISAAC_NIC_RAIL_Y_BY_INDEX = (0.18329, 0.14329, 0.10329, 0.06329, 0.02329)
ISAAC_SFP_PORT_LOCAL_BY_NAME = {
    "sfp_port_0": (0.01295, -0.031572, 0.00501),
    "sfp_port_1": (-0.01025, -0.031572, 0.00501),
}
ROBOT_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def _trial_key(trial_id: str) -> str:
    trial_id = str(trial_id)
    return trial_id if trial_id.startswith("trial_") else f"trial_{trial_id}"


def load_gazebo_trial(path: str | Path, trial_id: str) -> dict[str, Any]:
    """Load one trial node from a Gazebo AIC trials YAML file."""
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream)
    key = _trial_key(trial_id)
    return root["trials"][key]


def load_gazebo_config(path: str | Path) -> dict[str, Any]:
    """Load full Gazebo trial config YAML."""
    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sc_rail_x_limits_from_config(config: dict[str, Any]) -> tuple[float, float]:
    sc_rail_limits = config.get("task_board_limits", {}).get("sc_rail", {})
    return (
        float(sc_rail_limits.get("min_translation", ISAAC_SC_RAIL_X_LIMITS[0])),
        float(sc_rail_limits.get("max_translation", ISAAC_SC_RAIL_X_LIMITS[1])),
    )


def _home_joint_positions_from_config(config: dict[str, Any]) -> dict[str, float]:
    joint_config = config.get("robot", {}).get("home_joint_positions", {})
    return {
        joint_name: float(joint_config[joint_name])
        for joint_name in ROBOT_JOINT_NAMES
        if joint_name in joint_config
    }


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    quat = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )
    norm = math.sqrt(sum(v * v for v in quat))
    return (quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm) if norm > 1e-12 else (1, 0, 0, 0)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _normalize_quat(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(v * v for v in quat))
    return (quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm) if norm > 1e-12 else (1, 0, 0, 0)


def _quat_multiply(
    lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return _normalize_quat(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _quat_rotate(
    quat: tuple[float, float, float, float], vec: tuple[float, float, float]
) -> tuple[float, float, float]:
    qw, qx, qy, qz = _normalize_quat(quat)
    vx, vy, vz = vec
    # Equivalent to q * (0, v) * conjugate(q), expanded to avoid temporary tuples.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _compose_pose(
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    local_pos: tuple[float, float, float],
    local_quat: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    rotated = _quat_rotate(parent_quat, local_pos)
    return (
        (
            parent_pos[0] + rotated[0],
            parent_pos[1] + rotated[1],
            parent_pos[2] + rotated[2],
        ),
        _quat_multiply(parent_quat, local_quat),
    )


def gazebo_world_to_isaac_pose(
    pose_gz: dict[str, float], *, board: bool = False
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert Gazebo world xyz/rpy pose into the current Isaac AIC frame."""
    x, y, z = float(pose_gz["x"]), float(pose_gz["y"]), float(pose_gz["z"])
    roll, pitch, yaw = float(pose_gz["roll"]), float(pose_gz["pitch"]), float(pose_gz["yaw"])
    c, s = math.cos(GAZEBO_TO_ISAAC_YAW_OFFSET), math.sin(GAZEBO_TO_ISAAC_YAW_OFFSET)
    pos = (c * x - s * y, s * x + c * y, z + GAZEBO_TO_ISAAC_Z_OFFSET)
    roll_isaac = roll
    yaw_isaac = yaw + GAZEBO_TO_ISAAC_YAW_OFFSET
    if board:
        yaw_isaac += GAZEBO_TO_ISAAC_BOARD_YAW_EXTRA
    return pos, _quat_wxyz_from_rpy(roll_isaac, pitch, yaw_isaac)


def calculate_all_poses(
    trial: dict[str, Any],
    robot_args: dict,
    *,
    sc_rail_x_limits: tuple[float, float] = ISAAC_SC_RAIL_X_LIMITS,
) -> dict[str, dict[str, tuple]]:
    """Calculate all poses without requiring a running simulation."""
    task_board = trial["scene"]["task_board"]
    board_pos, board_quat = gazebo_world_to_isaac_pose(task_board["pose"], board=True)

    # Calculate Robot pose
    robot_quat = _quat_wxyz_from_rpy(robot_args["roll"], robot_args["pitch"], robot_args["yaw"])
    robot_pos = (robot_args["x"], robot_args["y"], robot_args["z"])

    poses = {
        "robot": {"pos": robot_pos, "quat": robot_quat},
        "task_board": {"pos": board_pos, "quat": board_quat},
    }

    # Add SC ports
    for sc_index, asset_name in ((0, "sc_port"), (1, "sc_port_2")):
        rail = task_board.get(f"sc_rail_{sc_index}", {})
        if not rail.get("entity_present", False):
            continue
        translation = float(rail.get("entity_pose", {}).get("translation", 0.0))
        anchor = ISAAC_SC_PORT_ANCHORS[sc_index]
        sc_local_x = _clamp(anchor[0] - translation, *sc_rail_x_limits)
        sc_local_quat = ISAAC_SC_PORT_2_LOCAL_QUAT if sc_index == 1 else ISAAC_SC_PORT_LOCAL_QUAT
        sc_pos, sc_quat = _compose_pose(
            board_pos,
            board_quat,
            (sc_local_x, anchor[1], anchor[2]),
            sc_local_quat,
        )
        poses[asset_name] = {
            "pos": sc_pos,
            "quat": sc_quat,
        }

    # Add NIC card and derived SFP ports
    nic_index = None
    for task in trial.get("tasks", {}).values():
        target = str(task.get("target_module_name", ""))
        if target.startswith("nic_card_mount_"):
            nic_index = int(target.removeprefix("nic_card_mount_"))
            break

    if nic_index is None:
        for index in range(len(ISAAC_NIC_RAIL_Y_BY_INDEX)):
            rail = task_board.get(f"nic_rail_{index}", {})
            if rail.get("entity_present", False):
                nic_index = index
                break

    if nic_index is not None:
        rail = task_board.get(f"nic_rail_{nic_index}", {})
        if rail.get("entity_present", False):
            translation = float(rail.get("entity_pose", {}).get("translation", 0.0))
            nic_local_pos = (
                ISAAC_NIC_ANCHOR_CENTER[0] - translation,
                ISAAC_NIC_RAIL_Y_BY_INDEX[nic_index],
                ISAAC_NIC_ANCHOR_CENTER[2],
            )
            nic_pos, nic_quat = _compose_pose(
                board_pos,
                board_quat,
                nic_local_pos,
                ISAAC_NIC_LOCAL_QUAT,
            )
            poses["nic_card"] = {
                "pos": nic_pos,
                "quat": nic_quat,
            }

            for sfp_name, sfp_local in ISAAC_SFP_PORT_LOCAL_BY_NAME.items():
                sfp_board_pos = (
                    nic_local_pos[0] + sfp_local[0],
                    nic_local_pos[1] + sfp_local[1],
                    nic_local_pos[2] + sfp_local[2],
                )
                sfp_pos, sfp_quat = _compose_pose(
                    board_pos,
                    board_quat,
                    sfp_board_pos,
                    ISAAC_NIC_LOCAL_QUAT,
                )
                poses[sfp_name] = {
                    "pos": sfp_pos,
                    "quat": sfp_quat,
                }

    return poses


def build_gazebo_aligned_env_cfg(
    results: dict[str, dict[str, tuple]],
    device: str,
    num_envs: int,
    *,
    home_joint_positions: dict[str, float] | None = None,
    use_fabric: bool = True,
):
    """Build the AIC task env config with converted Gazebo poses baked in."""
    from aic_task.tasks.manager_based.aic_task.aic_task_env_cfg import AICTaskEnvCfg

    env_cfg = AICTaskEnvCfg()
    env_cfg.sim.device = device
    env_cfg.sim.use_fabric = use_fabric
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.robot.init_state.pos = results["robot"]["pos"]
    env_cfg.scene.robot.init_state.rot = results["robot"]["quat"]
    if home_joint_positions:
        env_cfg.scene.robot.init_state.joint_pos.update(home_joint_positions)
    env_cfg.scene.task_board.init_state.pos = results["task_board"]["pos"]
    env_cfg.scene.task_board.init_state.rot = results["task_board"]["quat"]
    if "sc_port" in results:
        env_cfg.scene.sc_port.init_state.pos = results["sc_port"]["pos"]
        env_cfg.scene.sc_port.init_state.rot = results["sc_port"]["quat"]
    if "sc_port_2" in results:
        env_cfg.scene.sc_port_2.init_state.pos = results["sc_port_2"]["pos"]
        env_cfg.scene.sc_port_2.init_state.rot = results["sc_port_2"]["quat"]
    if "nic_card" in results:
        env_cfg.scene.nic_card.init_state.pos = results["nic_card"]["pos"]
        env_cfg.scene.nic_card.init_state.rot = results["nic_card"]["quat"]

    return env_cfg


def apply_robot_home_joints(env, home_joint_positions: dict[str, float]) -> None:
    """Apply home joints from the Gazebo/aic_engine config after reset."""
    if not home_joint_positions:
        return

    import torch

    env = env.unwrapped
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    for joint_name in ROBOT_JOINT_NAMES:
        if joint_name not in home_joint_positions:
            continue
        try:
            joint_ids, _ = robot.find_joints(joint_name)
        except ValueError:
            continue
        if joint_ids:
            joint_pos[:, joint_ids[0]] = home_joint_positions[joint_name]

    robot.write_joint_state_to_sim(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
    robot.data.default_joint_pos[env_ids] = joint_pos
    robot.data.default_joint_vel[env_ids] = joint_vel
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials_yaml", required=True, help="Path to Gazebo trials YAML.")
    parser.add_argument(
        "--trial_id",
        default="trial_1",
        help="Trial id to apply, e.g. 'trial_1'. Integers are accepted as shorthand.",
    )
    parser.add_argument("--sim", action="store_true", help="Launch Isaac Sim for debugging.")
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable fabric and use USD I/O operations.",
    )
    # Robot Pos Arguments
    parser.add_argument("--robot_x", type=float, default=-0.18, help="Robot spawn X position.")
    parser.add_argument("--robot_y", type=float, default=-0.122, help="Robot spawn Y position.")
    parser.add_argument("--robot_z", type=float, default=0.0, help="Robot spawn Z position.")
    parser.add_argument(
        "--robot_roll", type=float, default=0.0, help="Robot spawn roll orientation (radians)."
    )
    parser.add_argument(
        "--robot_pitch", type=float, default=0.0, help="Robot spawn pitch orientation (radians)."
    )
    parser.add_argument(
        "--robot_yaw",
        type=float,
        default=ISAAC_ROBOT_DEFAULT_YAW,
        help="Robot spawn yaw orientation (radians).",
    )

    sim_preparser = argparse.ArgumentParser(add_help=False)
    sim_preparser.add_argument("--sim", action="store_true")
    sim_args, _ = sim_preparser.parse_known_args()
    if sim_args.sim:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)

    args_cli = parser.parse_args()

    # 1. Load Data
    config_root = load_gazebo_config(args_cli.trials_yaml)
    trial = config_root["trials"][_trial_key(args_cli.trial_id)]
    home_joint_positions = _home_joint_positions_from_config(config_root)

    # 2. Calculate Poses
    robot_params = {
        "x": args_cli.robot_x,
        "y": args_cli.robot_y,
        "z": args_cli.robot_z,
        "roll": args_cli.robot_roll,
        "pitch": args_cli.robot_pitch,
        "yaw": args_cli.robot_yaw,
    }
    results = calculate_all_poses(
        trial,
        robot_params,
        sc_rail_x_limits=_sc_rail_x_limits_from_config(config_root),
    )

    # 3. Output ENV variables for AICTaskSceneCfg
    print("\n" + "=" * 30)
    print("COPY INTO YOUR CONFIG CLASS")
    print("=" * 30)
    for asset, data in results.items():
        pos = [round(v, 4) for v in data["pos"]]
        quat = [round(v, 4) for v in data["quat"]]
        print(f"# {asset.upper()} CONFIG")
        print(f"pos={tuple(pos)}")
        print(f"rot={tuple(quat)}")
        print("-" * 10)
    if home_joint_positions:
        print("# ROBOT JOINT CONFIG")
        for joint_name in ROBOT_JOINT_NAMES:
            if joint_name in home_joint_positions:
                print(f"{joint_name}={round(home_joint_positions[joint_name], 4)}")
        print("-" * 10)

    # 4. Conditional Sim Launch
    if args_cli.sim:
        args_cli.enable_cameras = True
        print("[INFO] Launching Simulation App...")
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app

        import gymnasium as gym
        import torch

        import aic_task.tasks  # noqa: F401

        os.environ.setdefault("AIC_ISAAC_RANDOMIZATION_PROFILE", "none")

        env_cfg = build_gazebo_aligned_env_cfg(
            results,
            args_cli.device,
            num_envs=1,
            home_joint_positions=home_joint_positions,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg.events.reset_robot_joints = None
        env_cfg.events.randomize_board_and_parts = None

        env = gym.make("AIC-Task-v0", cfg=env_cfg)
        env.reset()
        apply_robot_home_joints(env, home_joint_positions)

        while simulation_app.is_running():
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
