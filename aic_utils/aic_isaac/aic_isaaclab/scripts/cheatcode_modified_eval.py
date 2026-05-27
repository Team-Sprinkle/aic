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
from collections import deque
from pathlib import Path
from typing import Any, Tuple, List, Optional

import yaml
import numpy as np
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

from cheatcode_eval_helpers import FrameDebugHelper, TrajectorySchedule, obs_term_slices

LOGGER = logging.getLogger("aic.cheatcode_eval")


# =============================================================================
# CLI PARSER SETUP
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CheatCodeModified-style policy runner for Isaac force parity.")
    
    # Environment & Hardware
    parser.add_argument("--task", type=str, default="AIC-Task-v0", help="Task name.")
    parser.add_argument("--num_envs", type=int, default=1, help="Force-parity mode supports only 1.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
    
    # Task Metadata
    parser.add_argument("--target_module_name", type=str, default="")
    parser.add_argument("--port_name", type=str, default="")
    parser.add_argument("--plug_name", type=str, default="")
    parser.add_argument("--cable_name", type=str, default="")
    parser.add_argument("--cable_type", type=str, default="")
    parser.add_argument("--out", type=str, default="aic/outputs/force_parity/output.csv")
    parser.add_argument("--force_log_body", type=str, default="wrist_3_link")
    parser.add_argument("--tcp_frame_name", type=str, default="gripper_tcp")
    parser.add_argument("--allow_tcp_fallback_to_wrist", action="store_true", default=False)
    parser.add_argument("--disable_cable_dynamics", action="store_true", default=False)

    # Trajectory Schedule
    parser.add_argument("--misalign_x_m", type=float, default=0.004)
    parser.add_argument("--misalign_y_m", type=float, default=0.0)
    parser.add_argument("--align_seconds", type=float, default=1.0)
    parser.add_argument("--initial_z_offset_m", type=float, default=0.2)
    parser.add_argument("--start_descent_z_offset_m", type=float, default=0.005)
    parser.add_argument("--end_z_offset_m", type=float, default=-0.015)
    parser.add_argument("--handoff_min_seconds", type=float, default=2.0)
    parser.add_argument("--handoff_speed_mps", type=float, default=0.02)
    parser.add_argument("--settle_seconds", type=float, default=1.0)
    parser.add_argument("--insertion_speed_mps", type=float, default=0.0009)
    parser.add_argument("--hold_seconds", type=float, default=5.0)

    # Force Guard Parameters
    parser.add_argument("--force_backoff_threshold_n", type=float, default=1.0e9)
    parser.add_argument("--force_l2_drop_threshold_n", type=float, default=1.7)
    parser.add_argument("--force_l2_horizon_sec", type=float, default=0.15)
    parser.add_argument("--force_median_samples", type=int, default=3)
    parser.add_argument("--force_z_delta_threshold_n", type=float, default=1.0e9)
    parser.add_argument("--stall_window_sec", type=float, default=0.25)
    parser.add_argument("--stall_max_descent_m", type=float, default=0.00015)
    parser.add_argument("--stall_force_rise_n", type=float, default=1.0e9)
    parser.add_argument("--force_direction_xy_scale", type=float, default=0.3)
    parser.add_argument("--unrecoverable_force_n", type=float, default=50.0)
    parser.add_argument("--backoff_m", type=float, default=0.015)
    parser.add_argument("--backoff_seconds", type=float, default=0.7)

    # Gazebo Parity Configs
    parser.add_argument("--gazebo_trial_config", type=str, default="")
    parser.add_argument("--gazebo_trial_id", type=str, default="trial_1")
    parser.add_argument("--enable_gazebo_parity", action="store_true", default=False)
    parser.add_argument("--gazebo_to_isaac_offset_x", type=float, default=0.0)
    parser.add_argument("--gazebo_to_isaac_offset_y", type=float, default=0.0)
    parser.add_argument("--gazebo_to_isaac_offset_z", type=float, default=-1.14)
    parser.add_argument("--gazebo_to_isaac_yaw_offset", type=float, default=math.pi / 2.0)
    parser.add_argument("--gazebo_to_isaac_board_yaw_extra", type=float, default=math.pi)
    
    # Asset Offsets
    for asset in ["robot", "task_board", "sc_port", "sc_port_2", "nic_card"]:
        parser.add_argument(f"--{asset}_offset_x", type=float, default=0.0 if asset != "robot" else 0.08)
        parser.add_argument(f"--{asset}_offset_y", type=float, default=0.12 if asset == "task_board" else (0.08 if asset == "robot" else 0.0))
        parser.add_argument(f"--{asset}_offset_yaw", type=float, default=0.0)

    # Debug & Misc
    parser.add_argument("--validate_frame_transformer_only", action="store_true", default=False)
    parser.add_argument("--transformer_validate_steps", type=int, default=300)
    parser.add_argument("--debug_frame_log_every", type=int, default=50)
    parser.add_argument("--enable_force_analysis_png", action="store_true", default=True)
    parser.add_argument("--force_analysis_png_path", type=str, default="")
    parser.add_argument("--no_match_gazebo_physics", action="store_true", default=False)
    parser.add_argument("--enable_orientation_align", action="store_true", default=True)
    parser.add_argument("--rot_scale_rad_per_action", type=float, default=0.2)
    parser.add_argument("--approach_in_port_local_frame", action="store_true", default=True)
    parser.add_argument("--tcp_bias_x_m", type=float, default=0.0)
    parser.add_argument("--tcp_bias_y_m", type=float, default=0.0)
    parser.add_argument("--tcp_bias_z_m", type=float, default=0.0)
    parser.add_argument("--tcp_bias_roll_deg", type=float, default=0.0)
    parser.add_argument("--tcp_bias_pitch_deg", type=float, default=0.0)
    parser.add_argument("--tcp_bias_yaw_deg", type=float, default=0.0)
    parser.add_argument("--tcp_to_port_roll_offset_deg", type=float, default=0.0)
    parser.add_argument("--tcp_to_port_pitch_offset_deg", type=float, default=0.0)
    parser.add_argument("--tcp_to_port_yaw_offset_deg", type=float, default=0.0)
    # Backward compatibility aliases.
    parser.add_argument("--tcp_target_roll_offset_deg", type=float, default=0.0)
    parser.add_argument("--tcp_target_pitch_offset_deg", type=float, default=0.0)
    parser.add_argument("--tcp_target_yaw_offset_deg", type=float, default=0.0)

    AppLauncher.add_app_launcher_args(parser)
    return parser


# =============================================================================
# MATH UTILITIES
# =============================================================================
class QuatMath:
    @staticmethod
    def conjugate(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        return (q[0], -q[1], -q[2], -q[3])

    @staticmethod
    def mul(q1: Tuple[float, float, float, float], q2: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

    @staticmethod
    def normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        n = math.sqrt(sum(x*x for x in q))
        return (1.0, 0.0, 0.0, 0.0) if n <= 1.0e-12 else (q[0]/n, q[1]/n, q[2]/n, q[3]/n)

    @staticmethod
    def from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return QuatMath.normalize((
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ))

    @staticmethod
    def apply(q: Tuple[float, float, float, float], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        r = QuatMath.mul(QuatMath.mul(q, (0.0, v[0], v[1], v[2])), QuatMath.conjugate(q))
        return (r[1], r[2], r[3])


# =============================================================================
# GAZEBO PARITY MANAGER
# =============================================================================
class GazeboParityManager:
    def __init__(self, args, env_cfg):
        self.args = args
        self.env_cfg = env_cfg
        self.hidden_pos = (10.0, 10.0, -10.0)
        self.identity_quat = (1.0, 0.0, 0.0, 0.0)

    def apply(self):
        if not self.args.gazebo_trial_config:
            raise ValueError("--enable_gazebo_parity requires --gazebo_trial_config")
        
        config_root = self._load_yaml(Path(self.args.gazebo_trial_config).expanduser())
        trial_cfg = config_root.get("trials", {}).get(self.args.gazebo_trial_id)
        if not trial_cfg:
            raise KeyError(f"Trial '{self.args.gazebo_trial_id}' not found.")

        task_board_cfg = trial_cfg["scene"]["task_board"]
        board_pos_is, board_quat_is = self._setup_task_board(task_board_cfg)
        self._setup_sc_ports(task_board_cfg, board_pos_is, board_quat_is)
        self._setup_nic_card(trial_cfg, task_board_cfg, board_pos_is, board_quat_is)
        self._setup_robot(config_root.get("robot", {}))

    def _load_yaml(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _transform_world(self, pos_gz, rpy_gz):
        c, s = math.cos(self.args.gazebo_to_isaac_yaw_offset), math.sin(self.args.gazebo_to_isaac_yaw_offset)
        pos_isaac = (
            c * pos_gz[0] - s * pos_gz[1] + self.args.gazebo_to_isaac_offset_x,
            s * pos_gz[0] + c * pos_gz[1] + self.args.gazebo_to_isaac_offset_y,
            pos_gz[2] + self.args.gazebo_to_isaac_offset_z,
        )
        rpy_isaac = (0.0, 0.0, rpy_gz[2] + self.args.gazebo_to_isaac_yaw_offset)
        return pos_isaac, rpy_isaac

    def _apply_offset(self, pos_w, quat_w, dx, dy, dyaw):
        out_pos = (pos_w[0] + dx, pos_w[1] + dy, pos_w[2])
        if abs(dyaw) <= 1.0e-12: return out_pos, quat_w
        return out_pos, QuatMath.normalize(QuatMath.mul(QuatMath.from_rpy(0.0, 0.0, dyaw), quat_w))

    def _compose_pose(self, p_pos, p_quat, c_pos, c_rpy):
        c_quat = QuatMath.from_rpy(*c_rpy)
        c_world_quat = QuatMath.normalize(QuatMath.mul(p_quat, c_quat))
        rot = QuatMath.apply(p_quat, c_pos)
        c_world_pos = (p_pos[0] + rot[0], p_pos[1] + rot[1], p_pos[2] + rot[2])
        return c_world_pos, c_world_quat

    def _setup_task_board(self, task_board_cfg):
        cfg = task_board_cfg["pose"]
        pos_gz = (cfg.get("x", 0.0), cfg.get("y", 0.0), cfg.get("z", 0.0))
        rpy_gz = (cfg.get("roll", 0.0), cfg.get("pitch", 0.0), cfg.get("yaw", 0.0))
        
        pos_is, rpy_is = self._transform_world(pos_gz, rpy_gz)
        rpy_is = (rpy_is[0], rpy_is[1], rpy_is[2] + self.args.gazebo_to_isaac_board_yaw_extra)
        
        board_pos, board_quat = self._apply_offset(
            pos_is, QuatMath.from_rpy(*rpy_is), 
            self.args.task_board_offset_x, self.args.task_board_offset_y, self.args.task_board_offset_yaw
        )
        self.env_cfg.scene.task_board.init_state.pos = board_pos
        self.env_cfg.scene.task_board.init_state.rot = board_quat
        return board_pos, board_quat

    def _setup_sc_ports(self, task_board_cfg, board_pos_is, board_quat_is):
        sc_anchors = {0: (0.0067, -0.0362, 0.005), 1: (0.0076, -0.0783, 0.005)}
        sc_quats = {
            0: QuatMath.normalize(tuple(self.env_cfg.scene.sc_port.init_state.rot)),
            1: QuatMath.normalize(tuple(self.env_cfg.scene.sc_port_2.init_state.rot)),
        }

        for idx, scene_name in ((0, "sc_port"), (1, "sc_port_2")):
            rail_cfg = task_board_cfg.get(f"sc_rail_{idx}", {})
            if not rail_cfg.get("entity_present", False):
                getattr(self.env_cfg.scene, scene_name).init_state.pos = self.hidden_pos
                getattr(self.env_cfg.scene, scene_name).init_state.rot = self.identity_quat
                continue

            pose = rail_cfg.get("entity_pose", {})
            local_pos = (sc_anchors[idx][0] - pose.get("translation", 0.0), sc_anchors[idx][1], sc_anchors[idx][2])
            local_quat = QuatMath.normalize(QuatMath.mul(sc_quats[idx], QuatMath.from_rpy(pose.get("roll", 0.0), pose.get("pitch", 0.0), pose.get("yaw", 0.0))))
            
            part_pos, _ = self._compose_pose(board_pos_is, board_quat_is, local_pos, (0.0, 0.0, 0.0))
            part_quat = QuatMath.normalize(QuatMath.mul(board_quat_is, local_quat))
            
            dx, dy, dyaw = (self.args.sc_port_offset_x, self.args.sc_port_offset_y, self.args.sc_port_offset_yaw) if idx == 0 else \
                           (self.args.sc_port_2_offset_x, self.args.sc_port_2_offset_y, self.args.sc_port_2_offset_yaw)
            
            part_pos, part_quat = self._apply_offset(part_pos, part_quat, dx, dy, dyaw)
            getattr(self.env_cfg.scene, scene_name).init_state.pos = part_pos
            getattr(self.env_cfg.scene, scene_name).init_state.rot = part_quat

    def _setup_nic_card(self, trial_cfg, task_board_cfg, board_pos_is, board_quat_is):
        nic_idx = self._get_nic_idx(trial_cfg, task_board_cfg)
        if nic_idx is None:
            self.env_cfg.scene.nic_card.init_state.pos = self.hidden_pos
            self.env_cfg.scene.nic_card.init_state.rot = self.identity_quat
            return

        rail_cfg = task_board_cfg.get(f"nic_rail_{nic_idx}", {})
        pose = rail_cfg.get("entity_pose", {})
        
        # Hardcoded fallback logic wrapped cleanly
        nic_anchor_center = (-0.03235, 0.02329, 0.0743)
        nic_base_quat = QuatMath.normalize(tuple(self.env_cfg.scene.nic_card.init_state.rot))
        spacing_m, nic_local_x_correction = 0.04, 0.0
        y_by_idx = [0.10329, 0.06329, 0.02329, -0.01671, -0.05671]

        local_pos = (
            nic_anchor_center[0] + nic_local_x_correction - pose.get("translation", 0.0),
            y_by_idx[nic_idx] + (2.0 * spacing_m),
            nic_anchor_center[2],
        )
        local_delta_quat = QuatMath.from_rpy(pose.get("roll", 0.0), pose.get("pitch", 0.0), pose.get("yaw", 0.0))
        local_quat = QuatMath.normalize(QuatMath.mul(nic_base_quat, local_delta_quat))
        
        nic_pos, _ = self._compose_pose(board_pos_is, board_quat_is, local_pos, (0.0, 0.0, 0.0))
        nic_quat = QuatMath.normalize(QuatMath.mul(board_quat_is, local_quat))
        nic_pos, nic_quat = self._apply_offset(nic_pos, nic_quat, self.args.nic_card_offset_x, self.args.nic_card_offset_y, self.args.nic_card_offset_yaw)

        self.env_cfg.scene.nic_card.init_state.pos = nic_pos
        self.env_cfg.scene.nic_card.init_state.rot = nic_quat

    def _get_nic_idx(self, trial_cfg, task_board_cfg) -> Optional[int]:
        target_name = str(trial_cfg.get("tasks", {}).get("task_1", {}).get("target_module_name", ""))
        if target_name.startswith("nic_card_mount_"):
            idx = int(target_name.replace("nic_card_mount_", ""))
            if task_board_cfg.get(f"nic_rail_{idx}", {}).get("entity_present", False):
                return idx
        for i in range(5):
            if task_board_cfg.get(f"nic_rail_{i}", {}).get("entity_present", False):
                return i
        return None

    def _setup_robot(self, robot_cfg):
        pos_is, rpy_is = self._transform_world((-0.2, 0.2, 1.14), (0.0, 0.0, -3.141))
        robot_pos, robot_quat = self._apply_offset(pos_is, QuatMath.from_rpy(*rpy_is), self.args.robot_offset_x, self.args.robot_offset_y, self.args.robot_offset_yaw)
        self.env_cfg.scene.robot.init_state.pos = robot_pos
        self.env_cfg.scene.robot.init_state.rot = robot_quat

        home_joints = robot_cfg.get("home_joint_positions")
        if isinstance(home_joints, dict):
            self.env_cfg.scene.robot.init_state.joint_pos = {str(k): float(v) for k, v in home_joints.items()}


# =============================================================================
# GUARDED INSERTION MONITOR
# =============================================================================
class GuardedInsertionMonitor:
    def __init__(self, args):
        self.args = args
        self.history = deque(maxlen=max(16, int(math.ceil(max(args.force_l2_horizon_sec, args.stall_window_sec) / 0.016)) + 8))
        self.baseline_fz = None
        self.first_time = None
        self.triggered = False
        self.stop_reason = ""
        self.delta_force_out = None
        # Conservative defaults for penalty-avoidance behavior (CheatCodeModified-style).
        self.sustained_force_n = 20.0
        self.sustained_force_sec = 0.10
        self.emergency_force_n = 60.0
        self.release_force_n = 12.0
        self.release_dwell_sec = 0.25
        self.guard_warmup_sec = 0.40
        self.release_ready_time = None

    def update(self, time_s: float, force_xyz: np.ndarray, tcp_z: float):
        if self.first_time is None:
            self.first_time = time_s
        if self.baseline_fz is None:
            self.baseline_fz = float(force_xyz[2])
        self.history.append((time_s, force_xyz, tcp_z))
        self._check_guard()

    def should_release_descent_lock(self) -> bool:
        if len(self.history) == 0:
            return False
        latest_time, latest_force, _ = self.history[-1]
        fnorm = float(np.linalg.norm(latest_force))
        if fnorm <= self.release_force_n:
            if self.release_ready_time is None:
                self.release_ready_time = latest_time + self.release_dwell_sec
            return latest_time >= self.release_ready_time
        self.release_ready_time = None
        return False

    def _median_force(self, samples) -> Optional[np.ndarray]:
        vectors = [s[1] for s in samples if s[1] is not None]
        return np.median(np.stack(vectors[-self.args.force_median_samples:]), axis=0) if len(vectors) >= self.args.force_median_samples else None

    def _check_guard(self):
        if len(self.history) < 2 or self.triggered: return
        latest_time, latest_force, latest_z = self.history[-1]
        fz = float(latest_force[2])
        if self.first_time is not None and (latest_time - self.first_time) < self.guard_warmup_sec:
            return

        # 0) Conservative collision detection.
        fnorm = float(np.linalg.norm(latest_force))
        if fnorm >= self.emergency_force_n:
            self._trigger("force_emergency_norm", np.asarray(latest_force, dtype=np.float64))
            return
        # Conservative component-wise check for eval penalty rule (|F| >= 20N).
        comp_window = [s for s in self.history if latest_time - s[0] <= self.sustained_force_sec]
        if comp_window and all(float(np.max(np.abs(s[1]))) >= self.sustained_force_n for s in comp_window):
            if (latest_time - comp_window[0][0]) >= self.sustained_force_sec:
                self._trigger("force_sustained_20n_component", np.asarray(latest_force, dtype=np.float64))
                return
        sustained_window = [s for s in self.history if latest_time - s[0] <= self.sustained_force_sec]
        if sustained_window and all(float(np.linalg.norm(s[1])) >= self.sustained_force_n for s in sustained_window):
            if (latest_time - sustained_window[0][0]) >= self.sustained_force_sec:
                self._trigger("force_sustained_20n", np.asarray(latest_force, dtype=np.float64))
                return

        # 1. Absolute Threshold
        if abs(fz - self.baseline_fz) >= self.args.force_backoff_threshold_n:
            self._trigger("force_abs_delta", np.array([0.0, 0.0, max(0.0, fz - self.baseline_fz)], dtype=np.float64))
            return

        # 2. L2 Drop Threshold
        if len(self.history) >= 2 * self.args.force_median_samples:
            current = self._median_force(list(self.history))
            prev_candidates = [s for s in self.history if s[0] <= latest_time - self.args.force_l2_horizon_sec]
            previous = self._median_force(prev_candidates)
            if current is not None and previous is not None:
                delta = previous - current
                if float(np.linalg.norm(delta)) > self.args.force_l2_drop_threshold_n:
                    self._trigger("force_l2_drop", delta)
                    return

        # 3. Z Delta Threshold
        if abs(fz - self.baseline_fz) >= self.args.force_z_delta_threshold_n:
            self._trigger("force_z_delta", np.array([0.0, 0.0, max(0.0, fz - self.baseline_fz)], dtype=np.float64))
            return

        # 4. Stall Detection
        window = [s for s in self.history if latest_time - s[0] <= self.args.stall_window_sec]
        if len(window) >= 2 and latest_time > window[0][0]:
            z_descent = float(window[0][2] - latest_z)
            force_rise = fz - min(float(s[1][2]) for s in window)
            if z_descent <= self.args.stall_max_descent_m and force_rise >= self.args.stall_force_rise_n:
                self._trigger("stall", np.array([0.0, 0.0, 1.0], dtype=np.float64))

    def _trigger(self, reason: str, delta: np.ndarray):
        self.triggered = True
        self.stop_reason = reason
        self.delta_force_out = delta

    def get_backoff_cmd(self, steps: int) -> Tuple[float, float, float]:
        direction = np.asarray(self.delta_force_out, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        direction = np.array([0.0, 0.0, 1.0]) if norm <= 1.0e-9 else direction / (norm + 1.0e-6)
        
        direction[0] *= self.args.force_direction_xy_scale
        direction[1] *= self.args.force_direction_xy_scale
        if direction[2] < 0.0: direction[2] = abs(direction[2])
        
        norm = float(np.linalg.norm(direction))
        if norm > 1.0e-9: direction /= norm
        
        return (
            self.args.backoff_m * float(direction[0]) / steps,
            self.args.backoff_m * float(direction[1]) / steps,
            self.args.backoff_m * float(direction[2]) / steps
        )


# =============================================================================
# DATA LOGGER
# =============================================================================
class TrajectoryLogger:
    def __init__(
        self,
        out_path: str,
        task_meta: dict,
        target_port_root: str,
        target_port_rigid: str,
        initial_tcp_pose_b: np.ndarray,
        initial_gripper_pose_b: np.ndarray,
    ):
        self.out_path = Path(out_path).expanduser()
        self.task_meta = task_meta
        self.target_port_root = target_port_root
        self.target_port_rigid = target_port_rigid
        self.initial_tcp_pose_b = initial_tcp_pose_b
        self.initial_gripper_pose_b = initial_gripper_pose_b
        self.rows = []
        self.headers = [
            "time_s", "step", "phase", "cmd_x_norm", "cmd_y_norm", "cmd_z_norm",
            "cmd_x_m", "cmd_y_m", "cmd_z_m", "force_x_n", "force_y_n", "force_z_n",
            "torque_x_nm", "torque_y_nm", "torque_z_nm", "ee_x_m", "ee_y_m", "ee_z_m",
            "ee_qw", "ee_qx", "ee_qy", "ee_qz", "target_module_name", "port_name",
            "plug_name", "cable_name", "cable_type", "guarded_stop_reason",
            "port_x_in_base_m", "port_y_in_base_m", "port_z_in_base_m", "tcp_z_in_base_m",
            "target_port_x_b_m", "target_port_y_b_m", "target_port_z_b_m",
            "target_port_root_prim", "target_port_rigid_prim",
            "initial_tcp_x_b_m", "initial_tcp_y_b_m", "initial_tcp_z_b_m", "initial_tcp_qw_b", "initial_tcp_qx_b", "initial_tcp_qy_b", "initial_tcp_qz_b",
            "initial_gripper_x_b_m", "initial_gripper_y_b_m", "initial_gripper_z_b_m", "initial_gripper_qw_b", "initial_gripper_qx_b", "initial_gripper_qy_b", "initial_gripper_qz_b",
            "target_x_b_m", "target_y_b_m", "target_z_b_m", "target_qw_b", "target_qx_b", "target_qy_b", "target_qz_b",
            "tcp_x_b_m", "tcp_y_b_m", "tcp_z_b_m", "tcp_qw_b", "tcp_qx_b", "tcp_qy_b", "tcp_qz_b",
            "tcp_target_rot_err_x_rad", "tcp_target_rot_err_y_rad", "tcp_target_rot_err_z_rad", "tcp_target_rot_err_angle_rad",
            "gripper_x_b_m", "gripper_y_b_m", "gripper_z_b_m", "gripper_qw_b", "gripper_qx_b", "gripper_qy_b", "gripper_qz_b",
            "plug_x_b_m", "plug_y_b_m", "plug_z_b_m", "plug_qw_b", "plug_qx_b", "plug_qy_b", "plug_qz_b",
        ]
        # Write-through mode: ensure file exists even on abrupt interruptions.
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def record(
        self,
        step_dt,
        step,
        phase,
        cmds,
        cmds_m,
        wrench,
        ee_pose,
        reason,
        port_rel,
        tcp_z,
        target_pose_b,
        tcp_pose_b,
        tcp_target_rot_err_b,
        gripper_pose_b,
        plug_pose_b,
    ):
        row = [
            f"{step * step_dt:.6f}", str(step), phase,
            f"{cmds[0]:.8f}", f"{cmds[1]:.8f}", f"{cmds[2]:.8f}",
            f"{cmds_m[0]:.8f}", f"{cmds_m[1]:.8f}", f"{cmds_m[2]:.8f}",
            f"{wrench[0]:.8f}", f"{wrench[1]:.8f}", f"{wrench[2]:.8f}",
            f"{wrench[3]:.8f}", f"{wrench[4]:.8f}", f"{wrench[5]:.8f}",
            f"{ee_pose[0]:.8f}", f"{ee_pose[1]:.8f}", f"{ee_pose[2]:.8f}",
            f"{ee_pose[3]:.8f}", f"{ee_pose[4]:.8f}", f"{ee_pose[5]:.8f}", f"{ee_pose[6]:.8f}",
            self.task_meta["target_module_name"], self.task_meta["port_name"],
            self.task_meta["plug_name"], self.task_meta["cable_name"], self.task_meta["cable_type"],
            reason, f"{port_rel[0]:.8f}", f"{port_rel[1]:.8f}", f"{port_rel[2]:.8f}", f"{tcp_z:.8f}",
            f"{port_rel[0]:.8f}", f"{port_rel[1]:.8f}", f"{port_rel[2]:.8f}",
            self.target_port_root, self.target_port_rigid,
            f"{self.initial_tcp_pose_b[0]:.8f}", f"{self.initial_tcp_pose_b[1]:.8f}", f"{self.initial_tcp_pose_b[2]:.8f}",
            f"{self.initial_tcp_pose_b[3]:.8f}", f"{self.initial_tcp_pose_b[4]:.8f}", f"{self.initial_tcp_pose_b[5]:.8f}", f"{self.initial_tcp_pose_b[6]:.8f}",
            f"{self.initial_gripper_pose_b[0]:.8f}", f"{self.initial_gripper_pose_b[1]:.8f}", f"{self.initial_gripper_pose_b[2]:.8f}",
            f"{self.initial_gripper_pose_b[3]:.8f}", f"{self.initial_gripper_pose_b[4]:.8f}", f"{self.initial_gripper_pose_b[5]:.8f}", f"{self.initial_gripper_pose_b[6]:.8f}",
            f"{target_pose_b[0]:.8f}", f"{target_pose_b[1]:.8f}", f"{target_pose_b[2]:.8f}",
            f"{target_pose_b[3]:.8f}", f"{target_pose_b[4]:.8f}", f"{target_pose_b[5]:.8f}", f"{target_pose_b[6]:.8f}",
            f"{tcp_pose_b[0]:.8f}", f"{tcp_pose_b[1]:.8f}", f"{tcp_pose_b[2]:.8f}",
            f"{tcp_pose_b[3]:.8f}", f"{tcp_pose_b[4]:.8f}", f"{tcp_pose_b[5]:.8f}", f"{tcp_pose_b[6]:.8f}",
            f"{tcp_target_rot_err_b[0]:.8f}", f"{tcp_target_rot_err_b[1]:.8f}", f"{tcp_target_rot_err_b[2]:.8f}",
            f"{float(np.linalg.norm(tcp_target_rot_err_b)):.8f}",
            f"{gripper_pose_b[0]:.8f}", f"{gripper_pose_b[1]:.8f}", f"{gripper_pose_b[2]:.8f}",
            f"{gripper_pose_b[3]:.8f}", f"{gripper_pose_b[4]:.8f}", f"{gripper_pose_b[5]:.8f}", f"{gripper_pose_b[6]:.8f}",
            f"{plug_pose_b[0]:.8f}", f"{plug_pose_b[1]:.8f}", f"{plug_pose_b[2]:.8f}",
            f"{plug_pose_b[3]:.8f}", f"{plug_pose_b[4]:.8f}", f"{plug_pose_b[5]:.8f}", f"{plug_pose_b[6]:.8f}",
        ]
        self.rows.append(row)
        with self.out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def save(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.rows)

    def save_force_analysis_png(self, png_path: Optional[str] = None):
        if len(self.rows) == 0:
            return None
        try:
            import matplotlib.pyplot as plt
        except Exception:
            LOGGER.warning("matplotlib is not available; skipping force analysis PNG generation.")
            return None

        col = {name: idx for idx, name in enumerate(self.headers)}
        t = [float(r[col["time_s"]]) for r in self.rows]
        fx = [float(r[col["force_x_n"]]) for r in self.rows]
        fy = [float(r[col["force_y_n"]]) for r in self.rows]
        fz = [float(r[col["force_z_n"]]) for r in self.rows]
        tx = [float(r[col["torque_x_nm"]]) for r in self.rows]
        ty = [float(r[col["torque_y_nm"]]) for r in self.rows]
        tz = [float(r[col["torque_z_nm"]]) for r in self.rows]

        out_path = Path(png_path).expanduser() if png_path else self.out_path.with_suffix(".force_analysis.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(t, fx, label="Fx [N]", linewidth=1.0)
        axes[0].plot(t, fy, label="Fy [N]", linewidth=1.0)
        axes[0].plot(t, fz, label="Fz [N]", linewidth=1.0)
        axes[0].set_ylabel("Force [N]")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="upper right")

        axes[1].plot(t, tx, label="Tx [Nm]", linewidth=1.0)
        axes[1].plot(t, ty, label="Ty [Nm]", linewidth=1.0)
        axes[1].plot(t, tz, label="Tz [Nm]", linewidth=1.0)
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Torque [Nm]")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper right")

        fig.suptitle("Force / Torque vs Time")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path


# =============================================================================
# ENVIRONMENT HELPERS
# =============================================================================
def configure_physics(env_cfg, args):
    if args.no_match_gazebo_physics: return
    env_cfg.sim.dt = 0.002
    env_cfg.decimation = 1
    env_cfg.sim.render_interval = 1
    if hasattr(env_cfg.sim, "gravity"): env_cfg.sim.gravity = (0.0, 0.0, -9.8)
    if hasattr(env_cfg.sim, "physx") and hasattr(env_cfg.sim.physx, "solver_type"):
        env_cfg.sim.physx.solver_type = 0

def clear_randomization(env_cfg):
    term = getattr(getattr(env_cfg, "events", None), "randomize_board_and_parts", None)
    if term and hasattr(term, "params"):
        term.params["board_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        for p in term.params.get("parts", []):
            if isinstance(p, dict):
                p["pose_range"] = {}
                p.pop("snap_step", None)


def _quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[0], -q[1], -q[2], -q[3]))


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def _quat_norm_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q), min=1.0e-9)


def _quat_to_rotvec_wxyz(q: torch.Tensor) -> torch.Tensor:
    qn = _quat_norm_wxyz(q)
    if qn[0] < 0.0:
        qn = -qn
    w = torch.clamp(qn[0], -1.0, 1.0)
    angle = 2.0 * torch.acos(w)
    s = torch.sqrt(torch.clamp(1.0 - w * w, min=0.0))
    if float(s.item()) < 1.0e-6:
        return torch.zeros(3, device=qn.device, dtype=qn.dtype)
    axis = qn[1:4] / s
    return axis * angle


def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    v_quat = torch.stack((torch.tensor(0.0, device=v.device, dtype=v.dtype), v[0], v[1], v[2]))
    out = _quat_mul_wxyz(_quat_mul_wxyz(q, v_quat), _quat_conj_wxyz(q))
    return out[1:4]


def _get_prim_pose_world_wxyz(stage, prim_path: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim does not exist: {prim_path}")
    xf = UsdGeom.Xformable(prim)
    m = xf.ComputeLocalToWorldTransform(0.0)
    t = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    # pxr returns imaginary + real; convert to wxyz
    pos = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float64)
    quat = np.array([float(q.GetReal()), float(q.GetImaginary()[0]), float(q.GetImaginary()[1]), float(q.GetImaginary()[2])], dtype=np.float64)
    return pos, quat


def _world_pose_to_base_pose(
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    base_pos_w: torch.Tensor,
    base_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_quat_conj = _quat_conj_wxyz(base_quat_w)
    pos_rel_w = target_pos_w - base_pos_w
    pos_rel_b = _quat_apply_wxyz(base_quat_conj, pos_rel_w)
    quat_rel_b = _quat_norm_wxyz(_quat_mul_wxyz(base_quat_conj, target_quat_w))
    return pos_rel_b, quat_rel_b


def _disable_cable_dynamics_on_stage():
    from omni.usd import get_context

    stage = get_context().get_stage()
    cable_root_path = "/World/envs/env_0/Robot/cable"
    cable_root = stage.GetPrimAtPath(cable_root_path)
    if not cable_root.IsValid():
        LOGGER.warning("Cable root prim not found at '%s'; cannot disable cable dynamics.", cable_root_path)
        return

    rigid_count = 0
    collision_count = 0
    queue = [cable_root]
    while queue:
        prim = queue.pop(0)
        rb_attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if rb_attr.IsValid():
            rb_attr.Set(False)
            rigid_count += 1
        kin_attr = prim.GetAttribute("physics:kinematicEnabled")
        if kin_attr.IsValid():
            kin_attr.Set(True)
        col_attr = prim.GetAttribute("physics:collisionEnabled")
        if col_attr.IsValid():
            col_attr.Set(False)
            collision_count += 1
        queue.extend(list(prim.GetChildren()))
    LOGGER.info(
        "Disabled cable dynamics on '%s': rigid_body_attrs=%d collision_attrs=%d",
        cable_root_path,
        rigid_count,
        collision_count,
    )

def init_transformer(task_meta: dict, args):
    from isaaclab.sensors import FrameTransformer
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
    from omni.usd import get_context

    stage = get_context().get_stage()

    def resolve_first_named(base_path: str, name: str) -> str:
        base = stage.GetPrimAtPath(base_path)
        if not base.IsValid():
            raise RuntimeError(f"Prim does not exist: {base_path}")
        queue = [base]
        while queue:
            prim = queue.pop(0)
            if prim.GetName() == name:
                return str(prim.GetPath())
            queue.extend(list(prim.GetChildren()))
        raise RuntimeError(f"Could not resolve prim '{name}' under {base_path}")

    def resolve_first_rigid(base_path: str) -> str:
        base = stage.GetPrimAtPath(base_path)
        if not base.IsValid():
            raise RuntimeError(f"Prim does not exist: {base_path}")
        queue = [base]
        while queue:
            prim = queue.pop(0)
            # avoid pxr dependency: rigid bodies expose this schema attr when authored
            attr = prim.GetAttribute("physics:rigidBodyEnabled")
            if attr.IsValid():
                return str(prim.GetPath())
            queue.extend(list(prim.GetChildren()))
        raise RuntimeError(f"Could not resolve rigid body under {base_path}")

    def resolve_plug(robot_root: str) -> str:
        hint = str(task_meta.get("plug_name", "")).lower()
        candidates = ["sc_plug_link", "lc_plug_link"] if "sc" in hint else ["lc_plug_link", "sc_plug_link"]
        for c in candidates:
            try:
                return resolve_first_named(robot_root, c)
            except RuntimeError:
                continue
        raise RuntimeError(f"Could not resolve plug link under {robot_root}")

    def resolve_named_prim_under(base_path: str, name_hint: str) -> Optional[str]:
        base = stage.GetPrimAtPath(base_path)
        if not base.IsValid():
            return None
        name_hint = name_hint.lower()
        exact_match = None
        contains_match = None
        queue = [base]
        while queue:
            prim = queue.pop(0)
            prim_name = prim.GetName().lower()
            if prim_name == name_hint:
                exact_match = str(prim.GetPath())
                break
            if contains_match is None and name_hint in prim_name:
                contains_match = str(prim.GetPath())
            queue.extend(list(prim.GetChildren()))
        return exact_match or contains_match

    def resolve_port_root() -> str:
        target_module = str(task_meta.get("target_module_name", "")).lower()
        port_name = str(task_meta.get("port_name", "")).lower()
        if "nic_card" in target_module or "sfp" in port_name:
            nic_root = "/World/envs/env_0/nic_card"
            matched = resolve_named_prim_under(nic_root, port_name)
            if matched is not None:
                LOGGER.info("Resolved NIC target port by name: port_name='%s' -> '%s'", port_name, matched)
                return matched
            LOGGER.warning(
                "Could not resolve port_name='%s' under '%s'. Falling back to NIC root; targeting may be wrong.",
                port_name,
                nic_root,
            )
            return nic_root
        if "sc" in port_name:
            return "/World/envs/env_0/sc_port"
        if "lc" in port_name:
            return "/World/envs/env_0/lc_port"
        raise ValueError(f"Unknown target port mapping for {target_module}/{port_name}")

    robot_root = "/World/envs/env_0/Robot"
    base_link = resolve_first_named(robot_root, "base_link")
    tcp_candidates = [args.tcp_frame_name]
    if args.allow_tcp_fallback_to_wrist and args.tcp_frame_name != "wrist_3_link":
        tcp_candidates.append("wrist_3_link")
    tcp_link = None
    for candidate in tcp_candidates:
        try:
            tcp_link = resolve_first_named(robot_root, candidate)
            break
        except RuntimeError:
            continue
    if tcp_link is None:
        raise RuntimeError(
            f"Could not resolve TCP frame under {robot_root}. Tried: {tcp_candidates}"
        )
    ee_gripper = resolve_first_named(robot_root, "gripper_tcp")
    plug_link = resolve_plug(robot_root)
    port_root = resolve_port_root()
    port_rigid = None
    use_usd_port_pose = False
    try:
        port_rigid = resolve_first_rigid(port_root)
    except RuntimeError:
        use_usd_port_pose = True
        LOGGER.warning(
            "Resolved port prim '%s' is non-rigid. Using USD world pose lookup for port target frame.",
            port_root,
        )
    nic_rigid = resolve_first_rigid("/World/envs/env_0/nic_card")

    target_frames = [
        FrameTransformerCfg.FrameCfg(name="base_self", prim_path=base_link),
        FrameTransformerCfg.FrameCfg(name="tcp", prim_path=tcp_link),
        FrameTransformerCfg.FrameCfg(name="ee_gripper", prim_path=ee_gripper),
        FrameTransformerCfg.FrameCfg(name="plug", prim_path=plug_link),
        FrameTransformerCfg.FrameCfg(name="nic_card_port", prim_path=nic_rigid),
    ]
    if port_rigid is not None:
        target_frames.append(FrameTransformerCfg.FrameCfg(name="port", prim_path=port_rigid))

    cfg = FrameTransformerCfg(
        prim_path=base_link,
        target_frames=target_frames,
    )
    transformer = FrameTransformer(cfg)
    if not transformer.is_initialized: transformer._initialize_impl()
    LOGGER.info(
        "FrameTransformer init: base=%s tcp=%s ee_gripper=%s plug=%s port_root=%s port_rigid=%s use_usd_port_pose=%s",
        base_link,
        tcp_link,
        ee_gripper,
        plug_link,
        port_root,
        port_rigid,
        use_usd_port_pose,
    )
    return transformer, stage, port_root, port_rigid, tcp_link, use_usd_port_pose


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================
def main():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args = build_parser().parse_args()
    if hasattr(args, "enable_cameras"): args.enable_cameras = True
    
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    import isaaclab_tasks  # noqa: F401
    import aic_task.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    task_meta = {k: getattr(args, k) for k in ["target_module_name", "port_name", "plug_name", "cable_name", "cable_type"]}
    env = None
    logger = None
    
    try:
        # 1. Setup Environment
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs, use_fabric=not args.disable_fabric)
        configure_physics(env_cfg, args)
        if args.enable_gazebo_parity: GazeboParityManager(args, env_cfg).apply()
        clear_randomization(env_cfg)
        
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        if args.disable_cable_dynamics:
            _disable_cable_dynamics_on_stage()
            # one extra reset to ensure modified physics attrs are reflected in state init
            env.reset()

        step_dt = float(getattr(env, "step_dt", 1.0 / 60.0))
        action_shape = env.action_space.shape
        scale_m_per_action = 0.05
        
        # 2. Setup Tools
        schedule = TrajectorySchedule(
            align_steps=max(1, int(round(args.align_seconds / step_dt))),
            handoff_steps=max(1, int(round(max(args.handoff_min_seconds, max(0.0, args.initial_z_offset_m - args.start_descent_z_offset_m) / args.handoff_speed_mps) / step_dt))),
            settle_steps=max(1, int(round(args.settle_seconds / step_dt))),
            insertion_steps=max(1, int(round(max(step_dt, max(0.0, args.start_descent_z_offset_m - args.end_z_offset_m) / args.insertion_speed_mps) / step_dt))),
            handoff_distance=max(0.0, args.initial_z_offset_m - args.start_descent_z_offset_m),
            insertion_distance=max(0.0, args.start_descent_z_offset_m - args.end_z_offset_m),
            misalign_x_m=args.misalign_x_m,
            misalign_y_m=args.misalign_y_m,
        )
        
        total_steps = schedule.align_steps + schedule.handoff_steps + schedule.settle_steps + schedule.insertion_steps + max(1, int(round(args.hold_seconds / step_dt)))
        if args.validate_frame_transformer_only:
            total_steps = int(max(1, args.transformer_validate_steps))
        backoff_steps = max(1, int(round(args.backoff_seconds / step_dt)))
        
        frame_transformer, stage, target_port_root, target_port_rigid, resolved_tcp_frame, use_usd_port_pose = init_transformer(task_meta, args)
        LOGGER.info(
            "Target port mapping resolved: root='%s' rigid='%s' for target_module='%s' port='%s' tcp_frame='%s'",
            target_port_root,
            target_port_rigid,
            task_meta["target_module_name"],
            task_meta["port_name"],
            resolved_tcp_frame,
        )
        frame_names = list(frame_transformer.data.target_frame_names)
        base_required = ["tcp", "ee_gripper", "plug"]
        if not use_usd_port_pose:
            base_required.append("port")
        frame_idx = {n: frame_names.index(n) for n in base_required}
        frame_transformer.update(step_dt)
        init_frames = FrameDebugHelper.sample(frame_transformer, frame_idx)
        initial_tcp_pose_b = np.concatenate(
            [init_frames["tcp_pos_b"].detach().cpu().numpy(), init_frames["tcp_quat_b"].detach().cpu().numpy()]
        )
        initial_gripper_pose_b = np.concatenate(
            [init_frames["ee_gripper_pos_b"].detach().cpu().numpy(), init_frames["ee_gripper_quat_b"].detach().cpu().numpy()]
        )
        roll_deg = args.tcp_to_port_roll_offset_deg if abs(args.tcp_to_port_roll_offset_deg) > 1.0e-12 else args.tcp_target_roll_offset_deg
        pitch_deg = args.tcp_to_port_pitch_offset_deg if abs(args.tcp_to_port_pitch_offset_deg) > 1.0e-12 else args.tcp_target_pitch_offset_deg
        yaw_deg = args.tcp_to_port_yaw_offset_deg if abs(args.tcp_to_port_yaw_offset_deg) > 1.0e-12 else args.tcp_target_yaw_offset_deg
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        quat_offset_np = QuatMath.from_rpy(roll, pitch, yaw)  # wxyz
        quat_offset_t = torch.tensor(quat_offset_np, device=env.device, dtype=torch.float32)
        bias_roll = math.radians(args.tcp_bias_roll_deg)
        bias_pitch = math.radians(args.tcp_bias_pitch_deg)
        bias_yaw = math.radians(args.tcp_bias_yaw_deg)
        quat_bias_np = QuatMath.from_rpy(bias_roll, bias_pitch, bias_yaw)  # wxyz
        quat_bias_t = torch.tensor(quat_bias_np, device=env.device, dtype=torch.float32)
        pos_bias_t = torch.tensor([args.tcp_bias_x_m, args.tcp_bias_y_m, args.tcp_bias_z_m], device=env.device, dtype=torch.float32)
        
        monitor = GuardedInsertionMonitor(args)
        logger = TrajectoryLogger(
            args.out,
            task_meta,
            target_port_root,
            target_port_rigid,
            initial_tcp_pose_b,
            initial_gripper_pose_b,
        )
        
        obs_slices = obs_term_slices(env, "policy")
        cumulative_insertion = 0.0
        backoff_remaining = 0
        insertion_started = False
        backoff_cmd_xyz = (0.0, 0.0, 0.0)
        running_offset_x = 0.0
        running_offset_y = 0.0
        running_offset_z = 0.0
        descent_lock = False
        # Gentle collision handling: stop briefly, then retreat +5mm upward continuously.
        collision_hold_steps = max(1, int(round(0.10 / step_dt)))
        gentle_backoff_total_m = 0.005
        gentle_backoff_speed_mps = 0.002
        gentle_backoff_steps = max(1, int(round(gentle_backoff_total_m / (gentle_backoff_speed_mps * step_dt))))
        gentle_backoff_step_m = gentle_backoff_total_m / gentle_backoff_steps
        collision_hold_remaining = 0
        collision_backoff_remaining = 0
        # 3. Main Rollout Loop
        for step in range(total_steps):
            if not simulation_app.is_running(): break
            
            if args.validate_frame_transformer_only:
                phase = "validate"
                cmd_dx_m = 0.0
                cmd_dy_m = 0.0
                cmd_dz_m = 0.0
                started_insert = False
            else:
                phase, cmd_dx_m, cmd_dy_m, cmd_dz_m, cumulative_insertion, started_insert, backoff_remaining = schedule.command_for_step(
                    step,
                    cumulative_insertion=cumulative_insertion,
                    backoff_remaining=backoff_remaining,
                    backoff_cmd_xyz_m=backoff_cmd_xyz,
                )
            if collision_hold_remaining > 0:
                phase = "collision_stop"
                cmd_dx_m = 0.0
                cmd_dy_m = 0.0
                cmd_dz_m = 0.0
                collision_hold_remaining -= 1
            elif collision_backoff_remaining > 0:
                phase = "collision_backoff"
                cmd_dx_m = 0.0
                cmd_dy_m = 0.0
                cmd_dz_m = gentle_backoff_step_m
                collision_backoff_remaining -= 1
            if descent_lock and cmd_dz_m < 0.0:
                cmd_dz_m = 0.0
            insertion_started = insertion_started or started_insert
            with torch.inference_mode():
                # Apply Actions
                tcp_pos_base_pre = frame_transformer.data.target_pos_source[0, frame_idx["tcp"]]
                if use_usd_port_pose:
                    base_pos_w = frame_transformer.data.source_pos_w[0]
                    base_quat_w = frame_transformer.data.source_quat_w[0]
                    port_pos_w_np, port_quat_w_np = _get_prim_pose_world_wxyz(stage, target_port_root)
                    port_pos_w_t = torch.tensor(port_pos_w_np, device=env.device, dtype=base_pos_w.dtype)
                    port_quat_w_t = torch.tensor(port_quat_w_np, device=env.device, dtype=base_quat_w.dtype)
                    port_pos_base, port_quat_base = _world_pose_to_base_pose(port_pos_w_t, port_quat_w_t, base_pos_w, base_quat_w)
                else:
                    port_pos_base = frame_transformer.data.target_pos_source[0, frame_idx["port"]]
                    port_quat_base = frame_transformer.data.target_quat_source[0, frame_idx["port"]]
                desired_quat_b = _quat_norm_wxyz(_quat_mul_wxyz(_quat_mul_wxyz(port_quat_base, quat_offset_t), quat_bias_t.to(port_quat_base.dtype)))
                running_offset_x += cmd_dx_m
                running_offset_y += cmd_dy_m
                running_offset_z += cmd_dz_m
                if args.approach_in_port_local_frame:
                    local_offset = torch.tensor(
                        [running_offset_x, running_offset_y, args.initial_z_offset_m + running_offset_z],
                        device=env.device,
                        dtype=port_pos_base.dtype,
                    )
                    desired_pos_b = port_pos_base + _quat_apply_wxyz(port_quat_base, local_offset + pos_bias_t.to(port_pos_base.dtype))
                else:
                    desired_pos_b = torch.tensor(
                        [
                            port_pos_base[0] + running_offset_x + float(pos_bias_t[0].item()),
                            port_pos_base[1] + running_offset_y + float(pos_bias_t[1].item()),
                            port_pos_base[2] + args.initial_z_offset_m + running_offset_z + float(pos_bias_t[2].item()),
                        ],
                        device=env.device,
                        dtype=port_pos_base.dtype,
                    )
                tcp_quat_base_pre = frame_transformer.data.target_quat_source[0, frame_idx["tcp"]]
                quat_err = _quat_mul_wxyz(desired_quat_b, _quat_conj_wxyz(tcp_quat_base_pre))
                rotvec_err = _quat_to_rotvec_wxyz(quat_err)

                pos_err_b = desired_pos_b - tcp_pos_base_pre
                actions = torch.zeros(action_shape, device=env.device)
                if not args.validate_frame_transformer_only:
                    actions[0, 0] = torch.clamp(pos_err_b[0] / scale_m_per_action, -1.0, 1.0)
                    actions[0, 1] = torch.clamp(pos_err_b[1] / scale_m_per_action, -1.0, 1.0)
                    actions[0, 2] = torch.clamp(pos_err_b[2] / scale_m_per_action, -1.0, 1.0)
                    if args.enable_orientation_align:
                        rot_scale = max(args.rot_scale_rad_per_action, 1.0e-4)
                        actions[0, 3] = torch.clamp(rotvec_err[0] / rot_scale, -1.0, 1.0)
                        actions[0, 4] = torch.clamp(rotvec_err[1] / rot_scale, -1.0, 1.0)
                        actions[0, 5] = torch.clamp(rotvec_err[2] / rot_scale, -1.0, 1.0)
                    else:
                        actions[0, 3:6] = 0.0
                env.step(actions)
                frame_transformer.update(step_dt)
                
                # Read Observations
                eef = env.obs_buf["policy"][0, obs_slices["eef_pose"][0]:obs_slices["eef_pose"][1]].cpu().numpy()
                wrench_scaled = env.obs_buf["policy"][0, obs_slices["body_forces"][0]+36:obs_slices["body_forces"][0]+42].cpu().numpy()
                wrench = wrench_scaled / 0.1
                frames = FrameDebugHelper.sample(frame_transformer, frame_idx)
                tcp_pos_base = frames["tcp_pos_b"]
                tcp_quat_base = frames["tcp_quat_b"]
                gripper_pos_b = frames["ee_gripper_pos_b"]
                gripper_quat_b = frames["ee_gripper_quat_b"]
                plug_pos_b = frames["plug_pos_b"]
                plug_quat_b = frames["plug_quat_b"]
                tcp_target_quat_err = _quat_mul_wxyz(desired_quat_b, _quat_conj_wxyz(tcp_quat_base))
                tcp_target_rotvec_err = _quat_to_rotvec_wxyz(tcp_target_quat_err)
                
                # Check Guard (conservative: active in all rollout phases except explicit validation mode).
                if not args.validate_frame_transformer_only:
                    monitor.update(step * step_dt, wrench[:3], float(tcp_pos_base[2]))
                    if monitor.triggered and collision_hold_remaining == 0 and collision_backoff_remaining == 0:
                        descent_lock = True
                        collision_hold_remaining = collision_hold_steps
                        collision_backoff_remaining = gentle_backoff_steps
                    if descent_lock and collision_hold_remaining == 0 and collision_backoff_remaining == 0 and monitor.should_release_descent_lock():
                        # Keep descent locked; only clear trigger state after safe-force dwell.
                        monitor.triggered = False
                        monitor.stop_reason = ""
                        monitor.delta_force_out = None
                    
                    if abs(wrench[2]) >= args.unrecoverable_force_n:
                        env.reset()
                        monitor = GuardedInsertionMonitor(args) # Reset monitor
                        descent_lock = False
                        collision_hold_remaining = 0
                        collision_backoff_remaining = 0
                        continue

                # Log
                cmds = (cmd_dx_m/scale_m_per_action, cmd_dy_m/scale_m_per_action, cmd_dz_m/scale_m_per_action)
                target_pose_b = np.concatenate([desired_pos_b.detach().cpu().numpy(), desired_quat_b.detach().cpu().numpy()])
                tcp_pose_b = np.concatenate([tcp_pos_base.detach().cpu().numpy(), tcp_quat_base.detach().cpu().numpy()])
                gripper_pose_b = np.concatenate([gripper_pos_b.detach().cpu().numpy(), gripper_quat_b.detach().cpu().numpy()])
                plug_pose_b = np.concatenate([plug_pos_b.detach().cpu().numpy(), plug_quat_b.detach().cpu().numpy()])
                logger.record(
                    step_dt, step, phase, cmds, (cmd_dx_m, cmd_dy_m, cmd_dz_m), wrench, eef, monitor.stop_reason,
                    port_pos_base.detach().cpu().numpy(), float(tcp_pos_base[2].item()),
                    target_pose_b,
                    tcp_pose_b,
                    tcp_target_rotvec_err.detach().cpu().numpy(),
                    gripper_pose_b,
                    plug_pose_b,
                )

        # 4. Normal exit save
        logger.save()

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Saving partial CSV...")
    except Exception:
        traceback.print_exc()
    finally:
        if logger is not None:
            try:
                logger.save()
                print(f"CSV saved to: {logger.out_path}")
                if args.enable_force_analysis_png:
                    png_out = logger.save_force_analysis_png(args.force_analysis_png_path if args.force_analysis_png_path else None)
                    if png_out is not None:
                        print(f"Force analysis PNG saved to: {png_out}")
            except Exception:
                traceback.print_exc()
        if env: env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
