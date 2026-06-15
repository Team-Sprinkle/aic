#!/usr/bin/env python3
"""Audit the axial-first inserted-start stateful reward cases."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MDP_DIR = (
    REPO_ROOT
    / "aic_utils"
    / "aic_isaac"
    / "aic_isaaclab"
    / "source"
    / "aic_task"
    / "aic_task"
    / "tasks"
    / "manager_based"
    / "aic_task"
    / "mdp"
)
if str(MDP_DIR) not in sys.path:
    sys.path.insert(0, str(MDP_DIR))

from insertion_geometry import cheatcode_insertion_phase_reward, compute_insertion_geometry


@dataclass(frozen=True)
class Case:
    name: str
    s: float
    r: float
    theta: float
    prev_s: float
    prev_r: float
    prev_theta: float
    action_axial: float
    action_lateral: float = 0.0
    action_side: float = 0.0
    action_rotation: float = 0.0


def _geometry(s: torch.Tensor, r: torch.Tensor):
    body = torch.stack([s, r, torch.zeros_like(s)], dim=1)
    entrance = torch.zeros_like(body)
    target = torch.zeros_like(body)
    target[:, 0] = 0.0458
    axis = torch.zeros_like(body)
    axis[:, 0] = 1.0
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0015,
    )


def _eval(case: Case) -> dict[str, Any]:
    s = torch.tensor([case.s], dtype=torch.float32)
    r = torch.tensor([case.r], dtype=torch.float32)
    theta = torch.tensor([case.theta], dtype=torch.float32)
    comp = cheatcode_insertion_phase_reward(
        geometry=_geometry(s, r),
        previous_depth=torch.tensor([case.prev_s], dtype=torch.float32),
        previous_lateral_error=torch.tensor([case.prev_r], dtype=torch.float32),
        orientation_error=theta,
        previous_orientation_error=torch.tensor([case.prev_theta], dtype=torch.float32),
        sigma_lat_pre=0.0025,
        sigma_lat_insert=0.0015,
        schedule_lateral_radius=True,
        sigma_lat_pre_far=0.004,
        sigma_lat_insert_far=0.004,
        sigma_theta_pre=0.10,
        sigma_theta_insert=0.060,
        schedule_orientation_tolerance=True,
        sigma_theta_pre_far=0.12,
        sigma_theta_insert_far=0.10,
        lateral_progress_weight=8.0,
        orientation_progress_weight=8.0,
        orientation_progress_scale=0.005,
        near_misaligned_weight=0.5,
        near_misaligned_max=25.0,
        lateral_funnel_weight=0.0,
        lateral_funnel_scale=0.010,
        lateral_funnel_max=4.0,
        hover_weight=0.0,
        axial_progress_weight=8.0,
        preinsert_aligned_axial_weight=8.0,
        lateral_alignment_action_weight=6.0,
        lateral_alignment_action_scale=0.00025,
        lateral_alignment_require_axial_quiet=True,
        lateral_alignment_axial_quiet_scale=0.0005,
        off_axis_axial_action_penalty_weight=64.0,
        off_axis_axial_action_scale=0.001,
        off_axis_axial_action_penalty_max=4.0,
        lateral_error_state_penalty_weight=1.0,
        lateral_error_state_penalty_scale=0.010,
        lateral_error_state_penalty_max=25.0,
        phase_gate_insertion_credit=True,
        stateful_phase_mode=True,
        stateful_lateral_enter_threshold=0.001,
        stateful_orientation_enter_threshold=0.040,
        stateful_axial_action_quiet_scale=0.0005,
        stateful_axial_lateral_action_penalty_weight=24.0,
        stateful_axial_lateral_action_scale=0.00010,
        stateful_axial_lateral_action_penalty_max=50.0,
        stateful_axial_alignment_loss_penalty_weight=12.0,
        stateful_axial_alignment_loss_lateral_scale=0.0005,
        stateful_axial_alignment_loss_orientation_scale=0.020,
        stateful_axial_alignment_loss_penalty_max=16.0,
        stateful_axial_pure_action_weight=16.0,
        stateful_axial_impure_action_penalty_weight=32.0,
        stateful_axial_impure_action_penalty_max=20.0,
        stateful_axial_rotation_action_penalty_weight=32.0,
        stateful_axial_rotation_action_scale=0.00010,
        stateful_axial_rotation_action_penalty_max=50.0,
        stateful_axial_forward_action_penalty_weight=16.0,
        stateful_axial_forward_action_scale=0.00020,
        stateful_axial_forward_action_penalty_max=8.0,
        corridor_weight=10.0,
        inside_alignment_weight=0.5,
        inside_alignment_max=25.0,
        retreat_weight=0.10,
        bypass_penalty_scale=6.0,
        bypass_gate_tolerance=0.0,
        success_candidate_weight=32.0,
        action_delta_w=torch.tensor(
            [[case.action_axial, case.action_lateral, case.action_side]],
            dtype=torch.float32,
        ),
        action_rotation_norm=torch.tensor([case.action_rotation], dtype=torch.float32),
        action_axis_gate=True,
        action_lateral_sigma=0.00005,
        action_lateral_sigma_far=0.00030,
        action_forward_scale=0.00020,
        semantic_gate=torch.tensor([1.0], dtype=torch.float32),
        previous_semantic_gate=torch.tensor([1.0], dtype=torch.float32),
    )
    return {
        "case": case.name,
        "s": case.s,
        "r": case.r,
        "theta": case.theta,
        "action_axial": case.action_axial,
        "action_lateral": case.action_lateral,
        "total": float(comp.total[0]),
        "axial_progress": float(comp.axial_progress[0]),
        "preinsert_aligned_axial": float(comp.preinsert_aligned_axial[0]),
        "corridor": float(comp.corridor[0]),
        "inside_alignment": float(comp.inside_alignment[0]),
        "success_candidate": float(comp.success_candidate[0]),
        "g_action_axis": float(comp.g_action_axis[0]),
        "g_insert_combined": float(comp.g_insert_combined[0]),
        "lateral_progress": float(comp.lateral_progress[0]),
        "orientation_progress": float(comp.orientation_progress[0]),
    }


def main() -> int:
    cases = [
        Case("partial_aligned_forward", 0.012, 0.0, 0.020, 0.011, 0.0, 0.020, 0.001),
        Case("partial_aligned_hold", 0.012, 0.0, 0.020, 0.012, 0.0, 0.020, 0.0),
        Case("partial_aligned_retreat", 0.011, 0.0, 0.020, 0.012, 0.0, 0.020, -0.001),
        Case("partial_aligned_lateral_action", 0.012, 0.0, 0.020, 0.012, 0.0, 0.020, 0.0, 0.001),
        Case("partial_aligned_diagonal_forward", 0.013, 0.0, 0.020, 0.012, 0.0, 0.020, 0.001, 0.001),
        Case("partial_aligned_forward_with_rotation", 0.013, 0.0, 0.020, 0.012, 0.0, 0.020, 0.001, 0.0, 0.0, 0.001),
        Case("partial_aligned_then_lost_alignment_forward", 0.021, 0.010, 0.036, 0.012, 0.0, 0.020, 0.001),
        Case("partial_off_axis_forward", 0.012, 0.004, 0.020, 0.011, 0.004, 0.020, 0.001),
        Case("partial_orientation_bad_forward", 0.012, 0.0, 0.090, 0.011, 0.0, 0.090, 0.001),
        Case("orientation_phase_quiet_improve", 0.0452, 0.0, 0.041, 0.0452, 0.0, 0.045, 0.0),
        Case(
            "orientation_phase_translate_while_improving",
            0.0550,
            0.007,
            0.041,
            0.0452,
            0.0,
            0.045,
            0.001,
            0.001,
        ),
        Case("near_full_aligned_forward", 0.040, 0.0, 0.020, 0.039, 0.0, 0.020, 0.001),
        Case("full_success_candidate", 0.0458, 0.0, 0.020, 0.0448, 0.0, 0.020, 0.001),
    ]
    rows = [_eval(case) for case in cases]
    rng = random.Random(20260613)
    random_rows = []
    for i in range(64):
        s = rng.uniform(-0.010, 0.046)
        r = rng.uniform(0.0, 0.008)
        theta = rng.uniform(0.0, 0.12)
        action = rng.uniform(-0.001, 0.001)
        random_rows.append(_eval(Case(f"random_{i:03d}", s, r, theta, s - action, r, theta, action)))
    by_name = {row["case"]: row for row in rows}
    failures: list[str] = []
    if not by_name["partial_aligned_forward"]["total"] > by_name["partial_aligned_hold"]["total"] + 4.0:
        failures.append("partial aligned axial-forward must beat hold by a clear margin")
    if not by_name["partial_aligned_retreat"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("retreat must be worse than hold")
    if not by_name["partial_aligned_lateral_action"]["total"] < by_name["partial_aligned_forward"]["total"]:
        failures.append("lateral action must be worse than clean axial-forward")
    if not by_name["partial_aligned_lateral_action"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("aligned lateral action must be worse than hold")
    if not by_name["partial_aligned_diagonal_forward"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("aligned diagonal forward action must be worse than hold")
    if not by_name["partial_aligned_diagonal_forward"]["g_action_axis"] < 0.05:
        failures.append("aligned diagonal forward action must be rejected by the action-axis gate")
    if not by_name["partial_aligned_forward_with_rotation"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("aligned forward with rotational action must be worse than hold")
    if not by_name["partial_aligned_forward_with_rotation"]["g_action_axis"] < 0.05:
        failures.append("aligned forward with rotational action must be rejected by the action-axis gate")
    if not by_name["partial_aligned_then_lost_alignment_forward"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("axial-forward that loses alignment must be worse than aligned hold")
    if not by_name["partial_off_axis_forward"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("off-axis axial-forward must not beat aligned hold")
    if not by_name["partial_orientation_bad_forward"]["total"] < by_name["partial_aligned_forward"]["total"]:
        failures.append("orientation-misaligned axial-forward must be gated down")
    if not by_name["orientation_phase_translate_while_improving"]["total"] < by_name["orientation_phase_quiet_improve"]["total"]:
        failures.append("orientation improvement with translation must be worse than quiet orientation improvement")
    if not by_name["orientation_phase_translate_while_improving"]["total"] < by_name["partial_aligned_hold"]["total"]:
        failures.append("orientation improvement with axial/lateral translation must be worse than aligned hold")
    if not by_name["near_full_aligned_forward"]["total"] > by_name["partial_aligned_forward"]["total"]:
        failures.append("near-full aligned forward should beat shallower aligned forward")
    if not by_name["full_success_candidate"]["total"] == max(row["total"] for row in rows):
        failures.append("full strict-success candidate must be highest among named cases")
    aligned_forward_random = [
        row
        for row in random_rows
        if row["r"] <= 0.001 and row["theta"] <= 0.040 and row["action_axial"] > 0.0
    ]
    off_axis_forward_random = [
        row for row in random_rows if row["r"] > 0.003 and row["action_axial"] > 0.0
    ]
    output = {"cases": rows, "random_cases": random_rows, "failures": failures}
    print(json.dumps(output, indent=2, sort_keys=True))
    if aligned_forward_random and off_axis_forward_random:
        if max(row["total"] for row in off_axis_forward_random) >= max(row["total"] for row in aligned_forward_random):
            failures.append("random sanity: off-axis forward outscored aligned forward")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
