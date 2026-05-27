#!/usr/bin/env python3
"""Helper utilities for cheatcode_modified_eval rollout and debugging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def obs_term_slices(env: Any, group_name: str = "policy") -> dict[str, tuple[int, int]]:
    """Return [start, end) column slices for each observation term in a group."""
    names = list(env.observation_manager.active_terms[group_name])
    dims = list(env.observation_manager.group_obs_term_dim[group_name])
    out: dict[str, tuple[int, int]] = {}
    cursor = 0
    for n, d in zip(names, dims, strict=True):
        width = int(np.prod(d))
        out[n] = (cursor, cursor + width)
        cursor += width
    return out


@dataclass
class TrajectorySchedule:
    align_steps: int
    handoff_steps: int
    settle_steps: int
    insertion_steps: int
    handoff_distance: float
    insertion_distance: float
    misalign_x_m: float
    misalign_y_m: float

    def command_for_step(
        self,
        step: int,
        *,
        cumulative_insertion: float,
        backoff_remaining: int,
        backoff_cmd_xyz_m: tuple[float, float, float],
    ) -> tuple[str, float, float, float, float, bool, int]:
        if backoff_remaining > 0:
            return "backoff", backoff_cmd_xyz_m[0], backoff_cmd_xyz_m[1], backoff_cmd_xyz_m[2], cumulative_insertion, False, backoff_remaining - 1

        if step < self.align_steps:
            return "align", self.misalign_x_m / self.align_steps, self.misalign_y_m / self.align_steps, 0.0, cumulative_insertion, False, backoff_remaining
        if step < self.align_steps + self.handoff_steps:
            return "handoff", 0.0, 0.0, -self.handoff_distance / self.handoff_steps, cumulative_insertion, False, backoff_remaining
        if step < self.align_steps + self.handoff_steps + self.settle_steps:
            return "settle", 0.0, 0.0, 0.0, cumulative_insertion, False, backoff_remaining
        if step < self.align_steps + self.handoff_steps + self.settle_steps + self.insertion_steps:
            insertion_step = step - (self.align_steps + self.handoff_steps + self.settle_steps) + 1
            frac = 10.0 * (insertion_step / self.insertion_steps) ** 3 - 15.0 * (insertion_step / self.insertion_steps) ** 4 + 6.0 * (insertion_step / self.insertion_steps) ** 5
            target_cumulative = frac * self.insertion_distance
            delta = target_cumulative - cumulative_insertion
            return "insertion", 0.0, 0.0, -delta, target_cumulative, True, backoff_remaining
        return "hold", 0.0, 0.0, 0.0, cumulative_insertion, False, backoff_remaining


class FrameDebugHelper:
    @staticmethod
    def sample(frame_transformer: Any, idx: dict[str, int]) -> dict[str, torch.Tensor]:
        data = frame_transformer.data
        out: dict[str, torch.Tensor] = {
            "source_pos_w": data.source_pos_w[0],
            "source_quat_w": data.source_quat_w[0],
        }
        for key, frame_idx in idx.items():
            out[f"{key}_pos_b"] = data.target_pos_source[0, frame_idx]
            out[f"{key}_quat_b"] = data.target_quat_source[0, frame_idx]
            out[f"{key}_pos_w"] = data.target_pos_w[0, frame_idx]
            out[f"{key}_quat_w"] = data.target_quat_w[0, frame_idx]
        return out

    @staticmethod
    def log_validation_step(logger: Any, step: int, frames: dict[str, torch.Tensor]) -> None:
        logger.info(
            "Transformer check step=%d source_pos_w=(%.6f,%.6f,%.6f) "
            "source_quat_w=(%.6f,%.6f,%.6f,%.6f) "
            "nic_card_port_in_source=(%.6f,%.6f,%.6f) task_port_in_source=(%.6f,%.6f,%.6f) "
            "plug_in_source=(%.6f,%.6f,%.6f) ee_gripper_in_source=(%.6f,%.6f,%.6f) "
            "ee_gripper_quat_in_source=(%.6f,%.6f,%.6f,%.6f)",
            step,
            float(frames["source_pos_w"][0].item()),
            float(frames["source_pos_w"][1].item()),
            float(frames["source_pos_w"][2].item()),
            float(frames["source_quat_w"][0].item()),
            float(frames["source_quat_w"][1].item()),
            float(frames["source_quat_w"][2].item()),
            float(frames["source_quat_w"][3].item()),
            float(frames["nic_card_port_pos_b"][0].item()),
            float(frames["nic_card_port_pos_b"][1].item()),
            float(frames["nic_card_port_pos_b"][2].item()),
            float(frames["port_pos_b"][0].item()),
            float(frames["port_pos_b"][1].item()),
            float(frames["port_pos_b"][2].item()),
            float(frames["plug_pos_b"][0].item()),
            float(frames["plug_pos_b"][1].item()),
            float(frames["plug_pos_b"][2].item()),
            float(frames["ee_gripper_pos_b"][0].item()),
            float(frames["ee_gripper_pos_b"][1].item()),
            float(frames["ee_gripper_pos_b"][2].item()),
            float(frames["ee_gripper_quat_b"][0].item()),
            float(frames["ee_gripper_quat_b"][1].item()),
            float(frames["ee_gripper_quat_b"][2].item()),
            float(frames["ee_gripper_quat_b"][3].item()),
        )

    @staticmethod
    def log_rollout_step(
        logger: Any,
        *,
        step: int,
        phase: str,
        actions: torch.Tensor,
        cmd_xyz_m: tuple[float, float, float],
        eef_pose: torch.Tensor,
        target_pose: torch.Tensor,
        pos_err_b: torch.Tensor,
        frames: dict[str, torch.Tensor],
    ) -> None:
        logger.info(
            "Step=%d phase=%s action=(%.4f,%.4f,%.4f) cmd_m=(%.5f,%.5f,%.5f) "
            "ee_b=(%.5f,%.5f,%.5f|%.5f,%.5f,%.5f,%.5f) target_b=(%.5f,%.5f,%.5f|%.5f,%.5f,%.5f,%.5f) err_b=(%.5f,%.5f,%.5f)",
            step,
            phase,
            float(actions[0, 0].item()),
            float(actions[0, 1].item()),
            float(actions[0, 2].item()),
            cmd_xyz_m[0],
            cmd_xyz_m[1],
            cmd_xyz_m[2],
            *[float(v.item()) for v in eef_pose[:7]],
            *[float(v.item()) for v in target_pose[:7]],
            float(pos_err_b[0].item()),
            float(pos_err_b[1].item()),
            float(pos_err_b[2].item()),
        )
        logger.info(
            "Frames source_w=(%.5f,%.5f,%.5f|%.5f,%.5f,%.5f,%.5f) "
            "tcp_b=(%.5f,%.5f,%.5f) port_b=(%.5f,%.5f,%.5f) plug_b=(%.5f,%.5f,%.5f) ee_gripper_b=(%.5f,%.5f,%.5f) nic_b=(%.5f,%.5f,%.5f)",
            *[float(v.item()) for v in frames["source_pos_w"]],
            *[float(v.item()) for v in frames["source_quat_w"]],
            *[float(v.item()) for v in frames["tcp_pos_b"]],
            *[float(v.item()) for v in frames["port_pos_b"]],
            *[float(v.item()) for v in frames["plug_pos_b"]],
            *[float(v.item()) for v in frames["ee_gripper_pos_b"]],
            *[float(v.item()) for v in frames["nic_card_port_pos_b"]],
        )
