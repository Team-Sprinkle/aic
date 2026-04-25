"""Deterministic close-range controller used after coarse planning."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass
class CloseRangeInsertionPolicy:
    """CheatCode-like close-range controller for approach/alignment/insertion.

    This policy is intentionally deterministic and local. It is used either as
    the standalone CheatCode gym adapter or as the near-target handoff
    controller after the VLM has brought the robot close enough to the port.
    """

    handoff_distance_to_entrance_m: float = 0.055
    handoff_lateral_misalignment_m: float = 0.010
    handoff_orientation_error_rad: float = 0.12
    handoff_distance_to_target_m: float = 0.060
    final_approach_distance_m: float = 0.015
    final_approach_progress: float = 0.75
    axial_force_soft_limit_n: float = 25.0
    axial_force_hard_limit_n: float = 40.0
    force_relief_speed_mps: float = 0.003
    phase: str = "approach_entrance"
    _insert_latched: bool = False
    _handoff_latched: bool = False

    def action(self, observation: dict[str, Any]) -> np.ndarray:
        plug = np.asarray(observation["plug_pose"][:3], dtype=np.float64)
        target = np.asarray(observation["target_port_pose"][:3], dtype=np.float64)
        entrance = np.asarray(observation["target_port_entrance_pose"][:3], dtype=np.float64)
        score_geometry = observation.get("score_geometry", {})
        distance_to_entrance = float(score_geometry["distance_to_entrance"][0])
        distance_to_target = float(score_geometry["distance_to_target"][0])
        lateral_misalignment = float(score_geometry["lateral_misalignment"][0])
        insertion_progress = float(score_geometry.get("insertion_progress", [0.0])[0])
        wrench = np.asarray(observation.get("wrench", np.zeros(6, dtype=np.float64)), dtype=np.float64).reshape(-1)
        corridor_axis_value = score_geometry.get("corridor_axis")
        corridor_axis = np.asarray(
            [0.0, 0.0, 0.0] if corridor_axis_value is None else corridor_axis_value,
            dtype=np.float64,
        ).reshape(-1)
        if corridor_axis.size < 3 or float(np.linalg.norm(corridor_axis[:3])) <= 1e-8:
            inferred_axis = target - entrance
            inferred_norm = float(np.linalg.norm(inferred_axis))
            corridor_axis = (
                inferred_axis / inferred_norm
                if inferred_norm > 1e-8
                else np.array([0.0, 0.0, 1.0], dtype=np.float64)
            )
        else:
            corridor_axis = corridor_axis[:3] / float(np.linalg.norm(corridor_axis[:3]))
        action = np.zeros(6, dtype=np.float32)
        offset_from_entrance = plug - entrance
        axial_depth = float(np.dot(offset_from_entrance, corridor_axis))
        lateral_vector = offset_from_entrance - axial_depth * corridor_axis
        lateral_norm = float(np.linalg.norm(lateral_vector))
        force_xyz = np.zeros(3, dtype=np.float64) if wrench.size < 3 else wrench[:3]
        axial_force = abs(float(np.dot(force_xyz, corridor_axis)))
        force_norm = float(np.linalg.norm(force_xyz))
        current_yaw = float(np.asarray(observation["plug_pose"], dtype=np.float64)[5])
        target_yaw = float(np.asarray(observation["target_port_pose"], dtype=np.float64)[5])
        yaw_error = _wrap_to_pi(target_yaw - current_yaw)
        if distance_to_entrance > 0.05 and insertion_progress <= 0.02:
            self._insert_latched = False
        if (
            distance_to_entrance <= 0.025
            and lateral_norm <= 0.008
            and -0.020 <= axial_depth <= 0.006
        ):
            self._insert_latched = True
        if (
            insertion_progress >= 0.60
            and distance_to_target <= 0.030
            and lateral_norm <= 0.002
            and abs(yaw_error) <= 0.10
        ):
            self._insert_latched = True

        if not self._insert_latched:
            self.phase = "port_frame_pre_insert_align"
            if lateral_norm <= 0.002 and abs(yaw_error) <= 0.10:
                desired_axial_depth = max(axial_depth + 0.012, 0.70 * float(np.linalg.norm(target - entrance)))
            else:
                desired_axial_depth = -0.025 if lateral_norm > 0.012 else -0.006
            axial_delta = desired_axial_depth - axial_depth
            lateral_command = -2.0 * lateral_vector
            if lateral_norm <= 0.002 and abs(yaw_error) <= 0.10:
                lateral_command *= 0.25
            axial_command = 1.4 * axial_delta * corridor_axis
            max_lateral_speed = 0.025 if distance_to_entrance > 0.04 else 0.014
            max_axial_speed = 0.030 if distance_to_entrance > 0.04 else 0.018
            lateral_speed = float(np.linalg.norm(lateral_command))
            if lateral_speed > max_lateral_speed:
                lateral_command *= max_lateral_speed / max(lateral_speed, 1e-9)
            axial_speed = float(np.dot(axial_command, corridor_axis))
            axial_command = float(np.clip(axial_speed, -max_axial_speed, max_axial_speed)) * corridor_axis
            linear = lateral_command + axial_command
        else:
            self.phase = "insert"
            desired_axial_depth = min(float(np.linalg.norm(target - entrance)) + 0.004, axial_depth + 0.018)
            axial_delta = desired_axial_depth - axial_depth
            lateral_command = -0.8 * lateral_vector
            if lateral_norm <= 0.0015 and abs(yaw_error) <= 0.08:
                lateral_command *= 0.0
            lateral_speed = float(np.linalg.norm(lateral_command))
            if lateral_speed > 0.008:
                lateral_command *= 0.008 / max(lateral_speed, 1e-9)
            near_final_depth = bool(
                distance_to_target <= self.final_approach_distance_m
                or insertion_progress >= self.final_approach_progress
            )
            max_forward_axial_speed = 0.025
            if near_final_depth:
                max_forward_axial_speed = 0.006
            if axial_force >= self.axial_force_soft_limit_n or force_norm >= self.axial_force_soft_limit_n:
                max_forward_axial_speed = min(max_forward_axial_speed, 0.002)
            if axial_force >= self.axial_force_hard_limit_n or force_norm >= self.axial_force_hard_limit_n:
                self.phase = "insert_force_relief"
                axial_command = -self.force_relief_speed_mps * corridor_axis
            else:
                axial_command = (
                    float(np.clip(1.0 * axial_delta, -0.004, max_forward_axial_speed)) * corridor_axis
                )
            linear = lateral_command + axial_command
        if abs(yaw_error) > 0.25:
            linear *= 0.25
        if abs(yaw_error) > 0.75 and self._insert_latched:
            linear *= 0.0
        total_speed = float(np.linalg.norm(linear))
        max_total_speed = 0.018 if (
            self._insert_latched
            and (
                distance_to_target <= self.final_approach_distance_m
                or insertion_progress >= self.final_approach_progress
                or axial_force >= self.axial_force_soft_limit_n
                or force_norm >= self.axial_force_soft_limit_n
            )
        ) else 0.035
        if total_speed > max_total_speed:
            linear *= max_total_speed / max(total_speed, 1e-9)
        action[:3] = linear.astype(np.float32)
        yaw_limit = 0.5 if abs(yaw_error) > 0.25 else (0.2 if self._insert_latched else 0.5)
        action[5] = float(np.clip(1.0 * yaw_error, -yaw_limit, yaw_limit))
        return action

    def should_handoff(self, observation: dict[str, Any]) -> bool:
        if self._handoff_latched:
            return True
        score_geometry = observation.get("score_geometry", {})
        distance_to_entrance = float(score_geometry.get("distance_to_entrance", [1.0])[0])
        distance_to_target = float(score_geometry.get("distance_to_target", [1.0])[0])
        lateral_misalignment = float(score_geometry.get("lateral_misalignment", [1.0])[0])
        orientation_error = float(score_geometry.get("orientation_error", [1.0])[0])
        ready = bool(
            distance_to_entrance <= self.handoff_distance_to_entrance_m
            and lateral_misalignment <= self.handoff_lateral_misalignment_m
            and orientation_error <= self.handoff_orientation_error_rad
        ) or bool(
            distance_to_target <= self.handoff_distance_to_target_m
            and lateral_misalignment <= self.handoff_lateral_misalignment_m
            and orientation_error <= self.handoff_orientation_error_rad
        )
        if ready:
            self._handoff_latched = True
        return self._handoff_latched

    def force_handoff(self) -> None:
        self._handoff_latched = True


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)
