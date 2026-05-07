# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teleoperation script for Isaac Lab environments (with aic_task support)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Teleoperation for Isaac Lab environments."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Teleop device. Built-ins: keyboard, spacemouse, gamepad.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--sensitivity", type=float, default=1.0, help="Sensitivity factor."
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
parser.add_argument(
    "--force_log_csv",
    type=str,
    default="aic/outputs/force_parity/teleop_force_log.csv",
    help="Output CSV path for wrist force logging.",
)
parser.add_argument(
    "--force_log_body",
    type=str,
    default="wrist_3_link",
    help="Body name suffix to use for force logging.",
)
parser.add_argument(
    "--save_log_key",
    type=str,
    default="L",
    help="Keyboard/gamepad callback key to save force log CSV on demand.",
)
parser.add_argument(
    "--save_on_reset_button",
    action="store_true",
    default=False,
    help="For SpaceMouse, also save CSV when pressing the right/reset button.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    app_launcher_args["xr"] = True

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging

import gymnasium as gym
import torch

from isaaclab.devices import (
    Se3Gamepad,
    Se3GamepadCfg,
    Se3Keyboard,
    Se3KeyboardCfg,
    Se3SpaceMouse,
    Se3SpaceMouseCfg,
)
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

logger = logging.getLogger(__name__)


def main() -> None:
    def write_force_log_csv(rows: list[tuple[float, float, float, float, float]]) -> None:
        out = Path(args_cli.force_log_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", "force_z_n", "ee_x_m", "ee_y_m", "ee_z_m"])
            for t, fz, x, y, z in rows:
                writer.writerow([f"{t:.6f}", f"{fz:.8f}", f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"])
        print(f"Force log written: {out} ({len(rows)} samples)")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # env_cfg.terminations.time_out = None  # disabled: causes simulation view crash with some robots
    if "Lift" in args_cli.task:
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        env_cfg.terminations.object_reached_goal = DoneTerm(
            func=mdp.object_reached_goal
        )

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    should_reset = False
    teleoperation_active = True
    should_save_log = False

    def reset_recording_instance() -> None:
        nonlocal should_reset
        should_reset = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    def save_force_log_instance() -> None:
        nonlocal should_save_log
        should_save_log = True
        print("Force-log save requested (callback)")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
        args_cli.save_log_key.upper(): save_force_log_instance,
        args_cli.save_log_key.lower(): save_force_log_instance,
    }
    # Accept both uppercase/lowercase reset from keyboard paths.
    teleoperation_callbacks["r"] = reset_recording_instance

    if args_cli.xr:
        teleoperation_active = False

    teleop_interface = None
    try:
        if (
            hasattr(env_cfg, "teleop_devices")
            and args_cli.teleop_device in env_cfg.teleop_devices.devices
        ):
            teleop_interface = create_teleop_device(
                args_cli.teleop_device,
                env_cfg.teleop_devices.devices,
                teleoperation_callbacks,
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(
                        pos_sensitivity=0.025 * sensitivity,
                        rot_sensitivity=0.025 * sensitivity,
                    )
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(
                        pos_sensitivity=0.025 * sensitivity,
                        rot_sensitivity=0.025 * sensitivity,
                    )
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(
                        pos_sensitivity=0.05 * sensitivity,
                        rot_sensitivity=0.05 * sensitivity,
                    )
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                env.close()
                simulation_app.close()
                return

            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")

            # SpaceMouse implementation only emits "L"/"R" callbacks from its side buttons.
            # Bind save/reset explicitly for that path.
            if args_cli.teleop_device.lower() == "spacemouse":
                print("SpaceMouse mode: binding side buttons (L=save, R=reset)")
                try:
                    teleop_interface.add_callback("L", save_force_log_instance)
                except Exception as e:
                    logger.warning(f"Failed to bind SpaceMouse L callback: {e}")
                try:
                    if args_cli.save_on_reset_button:
                        def _save_and_reset():
                            save_force_log_instance()
                            reset_recording_instance()
                        teleop_interface.add_callback("R", _save_and_reset)
                        print("SpaceMouse mode: R button will save + reset")
                    else:
                        teleop_interface.add_callback("R", reset_recording_instance)
                except Exception as e:
                    logger.warning(f"Failed to bind SpaceMouse R callback: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")

    env.reset()
    teleop_interface.reset()

    print(
        f"Teleoperation started. Press 'R' to reset the environment, "
        f"'{args_cli.save_log_key}' to save force log."
    )
    step_dt = float(getattr(env, "step_dt", 1.0 / 60.0))
    rows: list[tuple[float, float, float, float, float]] = []
    step_count = 0
    wrist_idx = None
    body_names = None
    try:
        robot = env.scene["robot"]
        body_names = list(robot.body_names)
        wrist_idx = next(
            i for i, n in enumerate(body_names) if n.endswith(args_cli.force_log_body)
        )
        print(f"Force logger attached to body: {body_names[wrist_idx]}")
    except Exception:
        print(
            f"Warning: could not resolve body '{args_cli.force_log_body}' for force logging. "
            "CSV will only include rows if resolved later."
        )

    try:
        while simulation_app.is_running():
            try:
                with torch.inference_mode():
                    action = teleop_interface.advance()

                    if teleoperation_active:
                        actions = action.repeat(env.num_envs, 1)
                        env.step(actions)
                        if wrist_idx is not None:
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
                            rows.append(
                                (
                                    step_count * step_dt,
                                    float(wrench_b[2].item()),
                                    float(ee_pos_w[0].item()),
                                    float(ee_pos_w[1].item()),
                                    float(ee_pos_w[2].item()),
                                )
                            )
                        step_count += 1
                    else:
                        env.sim.render()

                    if should_reset:
                        env.reset()
                        teleop_interface.reset()
                        should_reset = False
                        print("Environment reset complete")
                    if should_save_log:
                        write_force_log_csv(rows)
                        should_save_log = False
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break
    finally:
        write_force_log_csv(rows)
        env.close()
        print("Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
