"""Deterministic close-range controller used after coarse planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CloseRangeInsertionPolicy:
    """CheatCode-like close-range controller for approach/alignment/insertion.

    This policy is intentionally deterministic and local. It is used either as
    the standalone CheatCode gym adapter or as the near-target handoff
    controller after the VLM has brought the robot close enough to the port.
    """

    handoff_distance_to_entrance_m: float = 0.10
    handoff_lateral_misalignment_m: float = 0.05
    handoff_distance_to_target_m: float = 0.12
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
        action = np.zeros(6, dtype=np.float32)
        pre_entrance = entrance.copy()
        pre_entrance[2] += 0.03
        if distance_to_entrance > 0.05:
            self._insert_latched = False
        if distance_to_entrance <= 0.005 and lateral_misalignment <= 0.005:
            self._insert_latched = True

        if distance_to_entrance > 0.08 and not self._insert_latched:
            self.phase = "approach_pre_entrance"
            desired = pre_entrance
            gains = np.array([8.0, 8.0, 10.0], dtype=np.float64)
            clip_xy = 0.25
            clip_z = 0.25
        elif not self._insert_latched and (distance_to_entrance > 0.015 or lateral_misalignment > 0.01):
            self.phase = "approach_entrance"
            desired = entrance
            gains = np.array([3.0, 3.0, 4.0], dtype=np.float64)
            clip_xy = 0.08
            clip_z = 0.12
        else:
            self.phase = "insert"
            desired = target
            gains = np.array([2.0, 2.0, 5.0], dtype=np.float64)
            clip_xy = 0.03
            clip_z = 0.10
            if distance_to_target < 0.03:
                gains = np.array([1.0, 1.0, 4.0], dtype=np.float64)
                clip_xy = 0.015
                clip_z = 0.06
        delta = desired - plug
        linear = gains * delta
        linear[:2] = np.clip(linear[:2], -clip_xy, clip_xy)
        linear[2] = float(np.clip(linear[2], -clip_z, clip_z))
        action[:3] = linear.astype(np.float32)
        return action

    def should_handoff(self, observation: dict[str, Any]) -> bool:
        if self._handoff_latched:
            return True
        score_geometry = observation.get("score_geometry", {})
        distance_to_entrance = float(score_geometry.get("distance_to_entrance", [1.0])[0])
        distance_to_target = float(score_geometry.get("distance_to_target", [1.0])[0])
        lateral_misalignment = float(score_geometry.get("lateral_misalignment", [1.0])[0])
        ready = bool(
            distance_to_entrance <= self.handoff_distance_to_entrance_m
            and lateral_misalignment <= self.handoff_lateral_misalignment_m
        ) or bool(distance_to_target <= self.handoff_distance_to_target_m)
        if ready:
            self._handoff_latched = True
        return self._handoff_latched

    def force_handoff(self) -> None:
        self._handoff_latched = True
