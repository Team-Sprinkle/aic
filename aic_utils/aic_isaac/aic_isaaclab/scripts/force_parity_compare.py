#!/usr/bin/env python3
"""Force parity tooling for Gazebo (CheatCodeModified) vs Isaac Lab.

Usage:
  1) Log Isaac force trace:
     isaaclab -p aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py \
       isaac-log --task AIC-Task-v0 --out /tmp/isaac_force.csv

  2) Compare against a Gazebo force CSV (time_s, force_z_n):
     python aic/aic_utils/aic_isaac/aic_isaaclab/scripts/force_parity_compare.py \
       compare --gazebo /tmp/gazebo_force.csv --isaac /tmp/isaac_force.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _read_force_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    fz: list[float] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"time_s", "force_z_n"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns: time_s, force_z_n")
        for row in reader:
            times.append(float(row["time_s"]))
            fz.append(float(row["force_z_n"]))
    if not times:
        raise ValueError(f"{path} is empty")
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(fz, dtype=np.float64)
    t = t - t[0]
    return t, y


@dataclass
class ParityMetrics:
    n_samples: int
    overlap_s: float
    rmse_n: float
    mae_n: float
    max_abs_err_n: float
    corr: float
    gazebo_peak_abs_n: float
    isaac_peak_abs_n: float
    gazebo_baseline_n: float
    isaac_baseline_n: float
    baseline_offset_n: float
    gazebo_collision_peak_delta_n: float
    isaac_collision_peak_delta_n: float
    peak_delta_error_n: float


def _compare_series(
    t_gz: np.ndarray,
    fz_gz: np.ndarray,
    t_isaac: np.ndarray,
    fz_isaac: np.ndarray,
    dt_s: float,
) -> tuple[ParityMetrics, np.ndarray, np.ndarray, np.ndarray]:
    t0 = max(float(t_gz[0]), float(t_isaac[0]))
    t1 = min(float(t_gz[-1]), float(t_isaac[-1]))
    if t1 <= t0:
        raise ValueError("No overlapping time window between Gazebo and Isaac traces")
    grid = np.arange(t0, t1 + 0.5 * dt_s, dt_s, dtype=np.float64)
    gz_i = np.interp(grid, t_gz, fz_gz)
    isaac_i = np.interp(grid, t_isaac, fz_isaac)
    err = isaac_i - gz_i
    corr = float(np.corrcoef(gz_i, isaac_i)[0, 1]) if grid.size > 1 else math.nan
    baseline_window = max(1, int(round(0.5 / dt_s)))
    gz_baseline = float(np.mean(gz_i[:baseline_window]))
    isaac_baseline = float(np.mean(isaac_i[:baseline_window]))
    gz_peak = float(np.max(np.abs(gz_i)))
    isaac_peak = float(np.max(np.abs(isaac_i)))
    metrics = ParityMetrics(
        n_samples=int(grid.size),
        overlap_s=float(t1 - t0),
        rmse_n=float(np.sqrt(np.mean(err**2))),
        mae_n=float(np.mean(np.abs(err))),
        max_abs_err_n=float(np.max(np.abs(err))),
        corr=corr,
        gazebo_peak_abs_n=gz_peak,
        isaac_peak_abs_n=isaac_peak,
        gazebo_baseline_n=gz_baseline,
        isaac_baseline_n=isaac_baseline,
        baseline_offset_n=isaac_baseline - gz_baseline,
        gazebo_collision_peak_delta_n=gz_peak - abs(gz_baseline),
        isaac_collision_peak_delta_n=isaac_peak - abs(isaac_baseline),
        peak_delta_error_n=(isaac_peak - abs(isaac_baseline)) - (gz_peak - abs(gz_baseline)),
    )
    return metrics, grid - t0, gz_i, isaac_i


def _write_aligned_csv(path: Path, t: np.ndarray, gz: np.ndarray, isaac: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "gazebo_force_z_n", "isaac_force_z_n", "error_n"])
        for ti, g, i in zip(t, gz, isaac):
            writer.writerow([f"{ti:.6f}", f"{g:.8f}", f"{i:.8f}", f"{(i-g):.8f}"])


def run_compare(args: argparse.Namespace) -> None:
    t_gz, fz_gz = _read_force_csv(Path(args.gazebo))
    t_isaac, fz_isaac = _read_force_csv(Path(args.isaac))
    metrics, t, gz_i, isaac_i = _compare_series(
        t_gz=t_gz,
        fz_gz=fz_gz,
        t_isaac=t_isaac,
        fz_isaac=fz_isaac,
        dt_s=args.dt,
    )
    out = Path(args.out)
    _write_aligned_csv(out, t, gz_i, isaac_i)
    print("Force parity metrics (Isaac - Gazebo, Fz in N)")
    print(f"- samples: {metrics.n_samples}")
    print(f"- overlap_s: {metrics.overlap_s:.4f}")
    print(f"- rmse_n: {metrics.rmse_n:.6f}")
    print(f"- mae_n: {metrics.mae_n:.6f}")
    print(f"- max_abs_err_n: {metrics.max_abs_err_n:.6f}")
    print(f"- corr: {metrics.corr:.6f}")
    print(f"- gazebo_peak_abs_n: {metrics.gazebo_peak_abs_n:.6f}")
    print(f"- isaac_peak_abs_n: {metrics.isaac_peak_abs_n:.6f}")
    print(f"- gazebo_baseline_n: {metrics.gazebo_baseline_n:.6f}")
    print(f"- isaac_baseline_n: {metrics.isaac_baseline_n:.6f}")
    print(f"- baseline_offset_n (isaac-gazebo): {metrics.baseline_offset_n:.6f}")
    print(
        f"- gazebo_collision_peak_delta_n: {metrics.gazebo_collision_peak_delta_n:.6f}"
    )
    print(
        f"- isaac_collision_peak_delta_n: {metrics.isaac_collision_peak_delta_n:.6f}"
    )
    print(f"- peak_delta_error_n (isaac-gazebo): {metrics.peak_delta_error_n:.6f}")
    print(f"- aligned_csv: {out}")


def run_isaac_log(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    from isaaclab.app import AppLauncher
    from isaaclab_tasks.utils import parse_env_cfg
    import isaaclab_tasks  # noqa: F401
    import aic_task.tasks  # noqa: F401

    sim_args = argparse.Namespace(
        headless=args.headless,
        device=args.device,
        enable_cameras=False,
        disable_fabric=args.disable_fabric,
    )
    app_launcher = AppLauncher(sim_args)
    simulation_app = app_launcher.app

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=1,
        use_fabric=not args.disable_fabric,
    )
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    body_names = list(robot.body_names)
    try:
        wrist_idx = next(i for i, n in enumerate(body_names) if n.endswith("wrist_3_link"))
    except StopIteration:
        raise RuntimeError(f"Could not find wrist_3_link in robot.body_names: {body_names}")

    step_dt = float(getattr(env, "step_dt", 1.0 / 60.0))
    action_shape = env.action_space.shape
    if len(action_shape) != 2:
        raise RuntimeError(f"Expected vectorized action shape (N, A), got {action_shape}")
    action_dim = int(action_shape[1])

    rows: list[tuple[float, float, float, float, float]] = []
    step = 0
    total_steps = int(args.seconds / step_dt)
    settle_steps = int(args.settle_seconds / step_dt)
    align_steps = int(args.align_seconds / step_dt)
    descend_steps = int(args.descend_seconds / step_dt)
    hold_steps = int(args.hold_seconds / step_dt)
    descend_z_per_step = args.descend_speed_mps * step_dt

    while simulation_app.is_running() and step < total_steps:
        with torch.inference_mode():
            actions = torch.zeros(action_shape, device=env.device)
            # For Differential IK relative-mode action:
            # cmd_delta ~= action[:3] * action_scale (0.05 m from env cfg) per step.
            # We synthesize phase motion:
            # 1) settle, 2) lateral misalign, 3) descend at requested speed, 4) hold.
            if action_dim >= 3:
                if settle_steps <= step < settle_steps + align_steps:
                    if align_steps > 0:
                        lateral_step_m = args.misalign_y_m / align_steps
                        actions[0, 1] = lateral_step_m / 0.05
                elif settle_steps + align_steps <= step < settle_steps + align_steps + descend_steps:
                    actions[0, 2] = -descend_z_per_step / 0.05
                elif (
                    settle_steps + align_steps + descend_steps
                    <= step
                    < settle_steps + align_steps + descend_steps + hold_steps
                ):
                    actions[0, 2] = 0.0
            env.step(actions)

            wrench_b = robot.data.body_incoming_wrench_b[0, wrist_idx]  # [Fx,Fy,Fz,Tx,Ty,Tz]
            ee_pos_w = robot.data.body_pos_w[0, wrist_idx]
            force_z_n = float(wrench_b[2].item())
            rows.append(
                (
                    step * step_dt,
                    force_z_n,
                    float(ee_pos_w[0].item()),
                    float(ee_pos_w[1].item()),
                    float(ee_pos_w[2].item()),
                )
            )
            step += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "force_z_n", "ee_x_m", "ee_y_m", "ee_z_m"])
        for t, fz, x, y, z in rows:
            writer.writerow([f"{t:.6f}", f"{fz:.8f}", f"{x:.8f}", f"{y:.8f}", f"{z:.8f}"])

    env.close()
    simulation_app.close()
    print(f"Wrote Isaac force log: {out} ({len(rows)} samples, dt={step_dt:.6f}s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gazebo vs Isaac force parity tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="Compare Gazebo and Isaac force CSV files")
    p_compare.add_argument("--gazebo", required=True, help="Gazebo CSV with time_s,force_z_n")
    p_compare.add_argument("--isaac", required=True, help="Isaac CSV with time_s,force_z_n")
    p_compare.add_argument("--dt", type=float, default=0.01, help="Resample dt in seconds")
    p_compare.add_argument(
        "--out",
        default="/tmp/force_parity_aligned.csv",
        help="Output aligned CSV path",
    )

    p_log = sub.add_parser("isaac-log", help="Run Isaac task and log wrist Fz")
    p_log.add_argument("--task", default="AIC-Task-v0")
    p_log.add_argument("--out", required=True)
    p_log.add_argument("--seconds", type=float, default=12.0)
    p_log.add_argument("--settle-seconds", type=float, default=2.0)
    p_log.add_argument("--align-seconds", type=float, default=1.0)
    p_log.add_argument("--descend-seconds", type=float, default=6.0)
    p_log.add_argument("--hold-seconds", type=float, default=2.0)
    p_log.add_argument("--misalign-y-m", type=float, default=0.004)
    p_log.add_argument(
        "--descend-speed-mps",
        type=float,
        default=0.0009,
        help="Target EE descent speed to match CheatCode insertion speed.",
    )
    p_log.add_argument("--disable-fabric", action="store_true", default=False)
    p_log.add_argument("--headless", action="store_true", default=False)
    p_log.add_argument("--device", default="cuda:0")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "compare":
        run_compare(args)
    elif args.cmd == "isaac-log":
        run_isaac_log(args)
    else:
        raise RuntimeError(f"Unsupported command: {args.cmd}")


if __name__ == "__main__":
    main()
