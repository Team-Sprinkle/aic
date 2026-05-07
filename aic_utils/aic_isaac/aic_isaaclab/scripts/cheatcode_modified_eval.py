#!/usr/bin/env python3
"""Run a CheatCodeModified-style evaluation in Isaac Lab and write output.csv.

This script reproduces the intended "misalign then insert" behavior used for
force-parity checks and logs trajectory + wrench traces in one CSV.

It supports loading the same Gazebo trial YAML used for evaluation by setting:
  - AIC_GAZEBO_TRIAL_CONFIG_PATH
  - AIC_GAZEBO_TRIAL_NAME
for the Isaac task env config at runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Any

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
parser.add_argument(
    "--gazebo_config",
    type=str,
    default="aic/outputs/configs/fixed_1_trials_sfp2nic.yaml",
    help="Gazebo trial YAML to mirror in Isaac (same file used in Gazebo eval).",
)
parser.add_argument(
    "--trial_name",
    type=str,
    default="trial_1",
    help="Trial key inside the Gazebo YAML.",
)
parser.add_argument(
    "--task_name",
    type=str,
    default="task_1",
    help="Task key inside the selected trial.",
)
parser.add_argument(
    "--apply_board_rpy",
    action="store_true",
    default=False,
    help="Also apply board roll/pitch/yaw from Gazebo YAML.",
)
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    raise ValueError("--num_envs must be 1 for force-parity policy evaluation.")

if args_cli.gazebo_config:
    os.environ["AIC_GAZEBO_TRIAL_CONFIG_PATH"] = str(
        Path(args_cli.gazebo_config).expanduser().resolve()
    )
    os.environ["AIC_GAZEBO_TRIAL_NAME"] = args_cli.trial_name
    os.environ["AIC_GAZEBO_APPLY_BOARD_RPY"] = "1" if args_cli.apply_board_rpy else "0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import aic_task.tasks  # noqa: F401


def _load_task_meta(
    config_path: str | None,
    trial_name: str,
    task_name: str,
) -> dict[str, str]:
    if not config_path:
        return {
            "target_module_name": "",
            "port_name": "",
            "plug_name": "",
            "cable_name": "",
            "cable_type": "",
        }
    try:
        import yaml
    except ImportError:
        return {
            "target_module_name": "",
            "port_name": "",
            "plug_name": "",
            "cable_name": "",
            "cable_type": "",
        }
    try:
        with open(Path(config_path).expanduser(), "r", encoding="utf-8") as f:
            full_cfg = yaml.safe_load(f) or {}
    except Exception:
        return {
            "target_module_name": "",
            "port_name": "",
            "plug_name": "",
            "cable_name": "",
            "cable_type": "",
        }
    trial_cfg = full_cfg.get("trials", {}).get(trial_name, {})
    task_cfg = trial_cfg.get("tasks", {}).get(task_name, {})
    if not isinstance(task_cfg, dict):
        task_cfg = {}
    return {
        "target_module_name": str(task_cfg.get("target_module_name", "")),
        "port_name": str(task_cfg.get("port_name", "")),
        "plug_name": str(task_cfg.get("plug_name", "")),
        "cable_name": str(task_cfg.get("cable_name", "")),
        "cable_type": str(task_cfg.get("cable_type", "")),
    }


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
    params = env_cfg.events.randomize_board_and_parts.params
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


def main() -> None:
    env = None
    try:
        task_meta = _load_task_meta(
            args_cli.gazebo_config, args_cli.trial_name, args_cli.task_name
        )

        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        _clear_scene_randomization(env_cfg)
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.reset()

        robot = env.scene["robot"]
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
            "Task metadata from YAML: "
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

        for step in range(total_steps):
            if not simulation_app.is_running():
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
            print(f"No samples were recorded. Output CSV still created at {out_path}.")
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
