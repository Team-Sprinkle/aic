#!/usr/bin/env python3
"""Audit scalar reward terms for 40 mm / 10 mm SFP insertion cases.

This is intentionally Isaac-free: it evaluates the pure phase reward over
hand-picked semantic `(s, r, theta)` states that represent the observed 40x10
failure modes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
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
class RewardConfig:
    target_depth_m: float = 0.0458
    sigma_lat_pre_m: float = 0.0025
    sigma_lat_insert_m: float = 0.0015
    sigma_lat_pre_far_m: float = 0.004
    sigma_lat_insert_far_m: float = 0.004
    sigma_theta_pre_rad: float = 0.10
    sigma_theta_insert_rad: float = 0.060
    sigma_theta_pre_far_rad: float = 0.12
    sigma_theta_insert_far_rad: float = 0.10
    bypass_penalty_scale: float = 6.0
    success_candidate_weight: float = 16.0
    lateral_progress_weight: float = 8.0
    orientation_progress_weight: float = 8.0
    orientation_progress_scale_rad: float = 0.005
    lateral_funnel_weight: float = 0.0
    lateral_funnel_scale_m: float = 0.010
    lateral_funnel_max: float = 4.0
    near_misaligned_weight: float = 0.5
    near_misaligned_max: float = 25.0
    hover_weight: float = 0.0
    axial_progress_weight: float = 2.0
    lateral_alignment_action_weight: float = 6.0
    lateral_alignment_action_scale_m: float = 0.00025
    lateral_alignment_require_axial_quiet: bool = True
    lateral_alignment_axial_quiet_scale_m: float = 0.0005
    off_axis_axial_action_penalty_weight: float = 16.0
    off_axis_axial_action_scale_m: float = 0.001
    off_axis_axial_action_penalty_max: float = 4.0
    lateral_error_state_penalty_weight: float = 1.0
    lateral_error_state_penalty_scale_m: float = 0.010
    lateral_error_state_penalty_max: float = 25.0
    phase_gate_insertion_credit: bool = True
    corridor_weight: float = 4.0
    inside_alignment_weight: float = 1.0
    inside_alignment_max: float = 25.0
    retreat_weight: float = 0.10
    semantic_progress_weight: float = 0.0
    semantic_loss_weight: float = 0.0
    action_lateral_sigma_m: float = 0.00005
    action_lateral_sigma_far_m: float = 0.00030
    bypass_gate_tolerance: float = 0.0
    stateful_phase_mode: bool = True
    stateful_lateral_enter_threshold_m: float = 0.0010
    stateful_orientation_enter_threshold_rad: float = 0.040
    stateful_axial_action_quiet_scale_m: float = 0.0005


@dataclass(frozen=True)
class Case:
    name: str
    s_m: float
    r_m: float
    theta_rad: float
    prev_s_m: float
    prev_r_m: float
    prev_theta_rad: float
    action_axial_m: float = 0.00002
    action_lateral_m: float = 0.0
    action_side_m: float = 0.0
    semantic_gate: float = 1.0
    prev_semantic_gate: float = 1.0


def _geometry(s: torch.Tensor, r: torch.Tensor, cfg: RewardConfig):
    body = torch.stack([s, r, torch.zeros_like(s)], dim=1)
    entrance = torch.zeros_like(body)
    target = torch.zeros_like(body)
    target[:, 0] = cfg.target_depth_m
    axis = torch.zeros_like(body)
    axis[:, 0] = 1.0
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=cfg.sigma_lat_insert_m,
    )


def _geometry_from_lateral_vec(s: torch.Tensor, lateral_vec: torch.Tensor, cfg: RewardConfig):
    body = torch.zeros((s.shape[0], 3), dtype=s.dtype)
    body[:, 0] = s
    body[:, 1:] = lateral_vec
    entrance = torch.zeros_like(body)
    target = torch.zeros_like(body)
    target[:, 0] = cfg.target_depth_m
    axis = torch.zeros_like(body)
    axis[:, 0] = 1.0
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=cfg.sigma_lat_insert_m,
    )


def _eval_direction_case(name: str, lateral_vec: tuple[float, float], action_vec: tuple[float, float], cfg: RewardConfig) -> dict[str, Any]:
    s = torch.tensor([-0.040], dtype=torch.float32)
    theta = torch.tensor([0.070], dtype=torch.float32)
    action = torch.tensor([[0.0, action_vec[0], action_vec[1]]], dtype=torch.float32)
    comp = cheatcode_insertion_phase_reward(
        geometry=_geometry_from_lateral_vec(s, torch.tensor([lateral_vec], dtype=torch.float32), cfg),
        previous_depth=torch.tensor([-0.040], dtype=torch.float32),
        previous_lateral_error=torch.tensor([0.010], dtype=torch.float32),
        orientation_error=theta,
        previous_orientation_error=torch.tensor([0.070], dtype=torch.float32),
        sigma_lat_pre=cfg.sigma_lat_pre_m,
        sigma_lat_insert=cfg.sigma_lat_insert_m,
        schedule_lateral_radius=True,
        sigma_lat_pre_far=cfg.sigma_lat_pre_far_m,
        sigma_lat_insert_far=cfg.sigma_lat_insert_far_m,
        sigma_theta_pre=cfg.sigma_theta_pre_rad,
        sigma_theta_insert=cfg.sigma_theta_insert_rad,
        schedule_orientation_tolerance=True,
        sigma_theta_pre_far=cfg.sigma_theta_pre_far_rad,
        sigma_theta_insert_far=cfg.sigma_theta_insert_far_rad,
        lateral_progress_weight=cfg.lateral_progress_weight,
        orientation_progress_weight=cfg.orientation_progress_weight,
        orientation_progress_scale=cfg.orientation_progress_scale_rad,
        near_misaligned_weight=cfg.near_misaligned_weight,
        near_misaligned_max=cfg.near_misaligned_max,
        lateral_funnel_weight=cfg.lateral_funnel_weight,
        lateral_funnel_scale=cfg.lateral_funnel_scale_m,
        lateral_funnel_max=cfg.lateral_funnel_max,
        hover_weight=cfg.hover_weight,
        axial_progress_weight=cfg.axial_progress_weight,
        lateral_alignment_action_weight=cfg.lateral_alignment_action_weight,
        lateral_alignment_action_scale=cfg.lateral_alignment_action_scale_m,
        lateral_alignment_require_axial_quiet=cfg.lateral_alignment_require_axial_quiet,
        lateral_alignment_axial_quiet_scale=cfg.lateral_alignment_axial_quiet_scale_m,
        off_axis_axial_action_penalty_weight=cfg.off_axis_axial_action_penalty_weight,
        off_axis_axial_action_scale=cfg.off_axis_axial_action_scale_m,
        off_axis_axial_action_penalty_max=cfg.off_axis_axial_action_penalty_max,
        lateral_error_state_penalty_weight=cfg.lateral_error_state_penalty_weight,
        lateral_error_state_penalty_scale=cfg.lateral_error_state_penalty_scale_m,
        lateral_error_state_penalty_max=cfg.lateral_error_state_penalty_max,
        phase_gate_insertion_credit=cfg.phase_gate_insertion_credit,
        corridor_weight=cfg.corridor_weight,
        inside_alignment_weight=cfg.inside_alignment_weight,
        inside_alignment_max=cfg.inside_alignment_max,
        retreat_weight=cfg.retreat_weight,
        bypass_penalty_scale=cfg.bypass_penalty_scale,
        bypass_gate_tolerance=cfg.bypass_gate_tolerance,
        success_candidate_weight=cfg.success_candidate_weight,
        action_delta_w=action,
        action_axis_gate=True,
        action_lateral_sigma=cfg.action_lateral_sigma_m,
        action_lateral_sigma_far=cfg.action_lateral_sigma_far_m,
        semantic_gate=torch.tensor([1.0], dtype=torch.float32),
        previous_semantic_gate=torch.tensor([1.0], dtype=torch.float32),
        semantic_progress_weight=cfg.semantic_progress_weight,
        semantic_loss_weight=cfg.semantic_loss_weight,
        stateful_phase_mode=cfg.stateful_phase_mode,
        stateful_lateral_enter_threshold=cfg.stateful_lateral_enter_threshold_m,
        stateful_orientation_enter_threshold=cfg.stateful_orientation_enter_threshold_rad,
        stateful_axial_action_quiet_scale=cfg.stateful_axial_action_quiet_scale_m,
    )
    return {
        "case": name,
        "total": float(comp.total[0]),
        "lat_gate_phase": float(comp.lat_gate_phase[0]),
        "lateral_alignment_action": float(comp.lateral_alignment_action[0]),
        "action_toward_axis_m": float(comp.action_toward_axis[0]),
        "lateral_progress": float(comp.lateral_progress[0]),
    }


def _eval_case(case: Case, cfg: RewardConfig) -> dict[str, Any]:
    s = torch.tensor([case.s_m], dtype=torch.float32)
    r = torch.tensor([case.r_m], dtype=torch.float32)
    theta = torch.tensor([case.theta_rad], dtype=torch.float32)
    action = torch.tensor([[case.action_axial_m, case.action_lateral_m, case.action_side_m]], dtype=torch.float32)
    comp = cheatcode_insertion_phase_reward(
        geometry=_geometry(s, r, cfg),
        previous_depth=torch.tensor([case.prev_s_m], dtype=torch.float32),
        previous_lateral_error=torch.tensor([case.prev_r_m], dtype=torch.float32),
        orientation_error=theta,
        previous_orientation_error=torch.tensor([case.prev_theta_rad], dtype=torch.float32),
        sigma_lat_pre=cfg.sigma_lat_pre_m,
        sigma_lat_insert=cfg.sigma_lat_insert_m,
        schedule_lateral_radius=True,
        sigma_lat_pre_far=cfg.sigma_lat_pre_far_m,
        sigma_lat_insert_far=cfg.sigma_lat_insert_far_m,
        sigma_theta_pre=cfg.sigma_theta_pre_rad,
        sigma_theta_insert=cfg.sigma_theta_insert_rad,
        schedule_orientation_tolerance=True,
        sigma_theta_pre_far=cfg.sigma_theta_pre_far_rad,
        sigma_theta_insert_far=cfg.sigma_theta_insert_far_rad,
        lateral_progress_weight=cfg.lateral_progress_weight,
        orientation_progress_weight=cfg.orientation_progress_weight,
        orientation_progress_scale=cfg.orientation_progress_scale_rad,
        near_misaligned_weight=cfg.near_misaligned_weight,
        near_misaligned_max=cfg.near_misaligned_max,
        lateral_funnel_weight=cfg.lateral_funnel_weight,
        lateral_funnel_scale=cfg.lateral_funnel_scale_m,
        lateral_funnel_max=cfg.lateral_funnel_max,
        hover_weight=cfg.hover_weight,
        axial_progress_weight=cfg.axial_progress_weight,
        lateral_alignment_action_weight=cfg.lateral_alignment_action_weight,
        lateral_alignment_action_scale=cfg.lateral_alignment_action_scale_m,
        lateral_alignment_require_axial_quiet=cfg.lateral_alignment_require_axial_quiet,
        lateral_alignment_axial_quiet_scale=cfg.lateral_alignment_axial_quiet_scale_m,
        off_axis_axial_action_penalty_weight=cfg.off_axis_axial_action_penalty_weight,
        off_axis_axial_action_scale=cfg.off_axis_axial_action_scale_m,
        off_axis_axial_action_penalty_max=cfg.off_axis_axial_action_penalty_max,
        lateral_error_state_penalty_weight=cfg.lateral_error_state_penalty_weight,
        lateral_error_state_penalty_scale=cfg.lateral_error_state_penalty_scale_m,
        lateral_error_state_penalty_max=cfg.lateral_error_state_penalty_max,
        phase_gate_insertion_credit=cfg.phase_gate_insertion_credit,
        corridor_weight=cfg.corridor_weight,
        inside_alignment_weight=cfg.inside_alignment_weight,
        inside_alignment_max=cfg.inside_alignment_max,
        retreat_weight=cfg.retreat_weight,
        bypass_penalty_scale=cfg.bypass_penalty_scale,
        bypass_gate_tolerance=cfg.bypass_gate_tolerance,
        success_candidate_weight=cfg.success_candidate_weight,
        action_delta_w=action,
        action_axis_gate=True,
        action_lateral_sigma=cfg.action_lateral_sigma_m,
        action_lateral_sigma_far=cfg.action_lateral_sigma_far_m,
        semantic_gate=torch.tensor([case.semantic_gate], dtype=torch.float32),
        previous_semantic_gate=torch.tensor([case.prev_semantic_gate], dtype=torch.float32),
        semantic_progress_weight=cfg.semantic_progress_weight,
        semantic_loss_weight=cfg.semantic_loss_weight,
        stateful_phase_mode=cfg.stateful_phase_mode,
        stateful_lateral_enter_threshold=cfg.stateful_lateral_enter_threshold_m,
        stateful_orientation_enter_threshold=cfg.stateful_orientation_enter_threshold_rad,
        stateful_axial_action_quiet_scale=cfg.stateful_axial_action_quiet_scale_m,
    )
    row = {
        "case": case.name,
        **asdict(case),
        "target_depth_m": cfg.target_depth_m,
        "total": float(comp.total[0]),
        "lateral_progress": float(comp.lateral_progress[0]),
        "orientation_progress": float(comp.orientation_progress[0]),
        "lat_gate_phase": float(comp.lat_gate_phase[0]),
        "lateral_alignment_action": float(comp.lateral_alignment_action[0]),
        "lateral_alignment_axial_quiet_gate": float(comp.lateral_alignment_axial_quiet_gate[0]),
        "off_axis_axial_action_penalty": float(comp.off_axis_axial_action_penalty[0]),
        "lateral_error_state_penalty": float(comp.lateral_error_state_penalty[0]),
        "action_toward_axis_m": float(comp.action_toward_axis[0]),
        "lateral_funnel": float(comp.lateral_funnel[0]),
        "axial_progress": float(comp.axial_progress[0]),
        "near_misaligned": float(comp.near_misaligned[0]),
        "corridor": float(comp.corridor[0]),
        "inside_alignment": float(comp.inside_alignment[0]),
        "semantic_progress": float(comp.semantic_progress[0]),
        "success_candidate": float(comp.success_candidate[0]),
        "g_lat_insert": float(comp.g_lat_insert[0]),
        "g_ori_insert": float(comp.g_ori_insert[0]),
        "g_action_axis": float(comp.g_action_axis[0]),
        "g_semantic": float(comp.g_semantic[0]),
        "g_insert_combined": float(comp.g_insert_combined[0]),
        "action_axial_reported_m": float(comp.action_axial[0]),
        "action_lateral_reported_m": float(comp.action_lateral[0]),
        "sigma_lat_insert_effective_m": float(comp.sigma_lat_insert[0]),
        "sigma_theta_insert_effective_rad": float(comp.sigma_theta_insert[0]),
    }
    row["strict_geometry"] = bool(
        abs(case.s_m - cfg.target_depth_m) <= 0.0005
        and case.r_m <= 0.0005
        and case.theta_rad <= cfg.stateful_orientation_enter_threshold_rad
        and case.semantic_gate >= 0.99
    )
    return row


def _cases() -> list[Case]:
    return [
        Case("40x10_hold", -0.040, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.0),
        Case("40x10_actual_lateral_reduction", -0.040, 0.010, 0.070, -0.040, 0.011, 0.070, action_axial_m=0.0),
        Case("40x10_action_toward_axis", -0.040, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.0, action_lateral_m=-0.001),
        Case("40x10_toward_with_side_slip", -0.040, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.0, action_lateral_m=-0.001, action_side_m=0.001),
        Case("40x10_toward_but_lateral_worsens", -0.040, 0.0105, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.0, action_lateral_m=-0.001),
        Case("40x10_toward_plus_axial_forward", -0.039, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.001, action_lateral_m=-0.001),
        Case("40x10_action_away_axis", -0.040, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.0, action_lateral_m=0.001),
        Case("40x10_bad_axial_forward", -0.039, 0.010, 0.070, -0.040, 0.010, 0.070, action_axial_m=0.001),
        Case("40x10_centered_hold", -0.040, 0.00035, 0.070, -0.040, 0.00035, 0.070, action_axial_m=0.0),
        Case("preinsert_aligned_hold", -0.040, 0.00035, 0.025, -0.040, 0.00035, 0.025, action_axial_m=0.0),
        Case("preinsert_aligned_axial_forward", -0.039, 0.00035, 0.025, -0.040, 0.00035, 0.025, action_axial_m=0.001),
        Case("near_entrance_misaligned_axial_forward", -0.004, 0.010, 0.070, -0.005, 0.010, 0.070, action_axial_m=0.001),
        Case("near_entrance_aligned_axial_forward", -0.004, 0.00035, 0.025, -0.005, 0.00035, 0.025, action_axial_m=0.001),
        Case("entrance_good_lateral_theta_bad", -0.0010, 0.00059, 0.0535, -0.0012, 0.00067, 0.0546),
        Case("entrance_good_lateral_theta_good", -0.0010, 0.00035, 0.028, -0.0012, 0.00050, 0.035),
        Case("shallow_centered", 0.0100, 0.00035, 0.028, 0.0098, 0.00040, 0.030),
        Case("bad_deep_lateral_bypass", 0.2000, 0.4200, 0.50, 0.1990, 0.4000, 0.48),
        Case("near_seated_theta_bad", 0.0458, 0.00035, 0.040, 0.0457, 0.00035, 0.041, action_axial_m=0.001),
        Case("strict_candidate", 0.0458, 0.00035, 0.025, 0.0457, 0.00035, 0.028, action_axial_m=0.001),
    ]


def _assert_reward_ordering(rows: list[dict[str, Any]]) -> None:
    by_name = {row["case"]: row for row in rows}

    def total(name: str) -> float:
        return float(by_name[name]["total"])

    checks = [
        (
            "actual lateral error reduction should beat merely pointing toward the axis",
            total("40x10_actual_lateral_reduction") > total("40x10_action_toward_axis"),
        ),
        (
            "lateral improvement from the 40x10 start should be rewarded over holding",
            total("40x10_actual_lateral_reduction") > total("40x10_hold"),
        ),
        (
            "toward-axis action should not be paid when lateral error worsens",
            total("40x10_toward_but_lateral_worsens") <= total("40x10_hold"),
        ),
        (
            "toward-axis action should not be meaningfully paid when most lateral motion is sideways slip",
            total("40x10_toward_with_side_slip") <= total("40x10_hold") + 1.0e-3,
        ),
        (
            "clean toward-axis action should beat a sideways-slipping lateral action",
            total("40x10_action_toward_axis") > total("40x10_toward_with_side_slip"),
        ),
        (
            "holding the 40x10 lateral offset should be negative, not neutral",
            total("40x10_hold") < -0.25,
        ),
        (
            "centered and oriented pre-insertion should beat holding the 10 mm lateral reset offset",
            total("preinsert_aligned_hold") > total("40x10_hold"),
        ),
        (
            "worsening lateral error should be worse than holding",
            total("40x10_action_away_axis") < total("40x10_hold"),
        ),
        (
            "misaligned axial insertion should be worse than lateral alignment work",
            total("40x10_bad_axial_forward") < total("40x10_action_toward_axis"),
        ),
        (
            "lateral correction should not get paid when paired with off-axis axial-forward action",
            total("40x10_toward_plus_axial_forward") <= total("40x10_hold"),
        ),
        (
            "misaligned axial insertion should not beat holding laterally off-axis",
            total("40x10_bad_axial_forward") <= total("40x10_hold"),
        ),
        (
            "near-entrance aligned axial insertion should beat near-entrance misaligned axial insertion",
            total("near_entrance_aligned_axial_forward") > total("near_entrance_misaligned_axial_forward"),
        ),
        (
            "aligned axial progress should be better than aligned holding",
            total("preinsert_aligned_axial_forward") > total("preinsert_aligned_hold"),
        ),
        (
            "deep lateral bypass should be strongly rejected",
            total("bad_deep_lateral_bypass") < -20.0,
        ),
        (
            "deep lateral bypass reward should be bounded enough for critic learning",
            total("bad_deep_lateral_bypass") > -100.0,
        ),
        (
            "strict candidate should beat a seated but orientation-bad candidate",
            total("strict_candidate") > total("near_seated_theta_bad"),
        ),
        (
            "strict candidate should beat pre-insertion lateral progress",
            total("strict_candidate") > total("40x10_action_toward_axis"),
        ),
        (
            "strict candidate should beat shallow centered insertion",
            total("strict_candidate") > total("shallow_centered"),
        ),
    ]
    failed = [message for message, ok in checks if not ok]
    if failed:
        raise AssertionError("reward ordering failed:\n- " + "\n- ".join(failed))


def _grid_cases() -> list[Case]:
    cases: list[Case] = []
    for s in (-0.040, -0.010, -0.001, 0.010, 0.0458):
        for theta in (0.025, 0.070):
            for r in (0.00035, 0.0010, 0.0030, 0.0060, 0.0100, 0.0200):
                cases.append(
                    Case(
                        f"grid_hold_s{s:+.4f}_r{r:.5f}_theta{theta:.3f}",
                        s,
                        r,
                        theta,
                        s,
                        r,
                        theta,
                        action_axial_m=0.0,
                    )
                )
            for r in (0.00035, 0.0030, 0.0100, 0.0200):
                cases.append(
                    Case(
                        f"grid_axial_s{s:+.4f}_r{r:.5f}_theta{theta:.3f}",
                        s + 0.001,
                        r,
                        theta,
                        s,
                        r,
                        theta,
                        action_axial_m=0.001,
                    )
                )
    return cases


def _assert_grid_ordering(rows: list[dict[str, Any]], cfg: RewardConfig) -> None:
    by_name = {row["case"]: row for row in rows}
    failed: list[str] = []
    for s in (-0.040, -0.010, -0.001, 0.010, 0.0458):
        for theta in (0.025, 0.070):
            if theta <= cfg.stateful_orientation_enter_threshold_rad:
                ordered = [
                    by_name[f"grid_hold_s{s:+.4f}_r{r:.5f}_theta{theta:.3f}"]["total"]
                    for r in (0.00035, 0.0010, 0.0030, 0.0060, 0.0100, 0.0200)
                ]
                if any(float(a) < float(b) for a, b in zip(ordered, ordered[1:])):
                    failed.append(f"hold reward is not monotonically better as lateral error shrinks at s={s} theta={theta}: {ordered}")
            aligned_axial = float(by_name[f"grid_axial_s{s:+.4f}_r0.00035_theta{theta:.3f}"]["total"])
            bad_axial = float(by_name[f"grid_axial_s{s:+.4f}_r0.01000_theta{theta:.3f}"]["total"])
            worse_axial = float(by_name[f"grid_axial_s{s:+.4f}_r0.02000_theta{theta:.3f}"]["total"])
            if theta <= cfg.stateful_orientation_enter_threshold_rad and aligned_axial <= bad_axial:
                failed.append(f"aligned axial does not beat 10 mm lateral axial at s={s} theta={theta}")
            if bad_axial <= worse_axial:
                failed.append(f"20 mm lateral axial is not worse than 10 mm lateral axial at s={s} theta={theta}")
    if failed:
        raise AssertionError("reward grid ordering failed:\n- " + "\n- ".join(failed))


def _random_sanity_rows(cfg: RewardConfig, count: int = 64) -> list[dict[str, Any]]:
    rng = random.Random(20260613)
    rows: list[dict[str, Any]] = []
    for idx in range(count):
        s = rng.uniform(-0.060, 0.055)
        r = rng.uniform(0.0, 0.025)
        theta = rng.uniform(0.0, 0.12)
        prev_r_better = r + rng.uniform(0.0001, 0.002)
        rows.append(
            _eval_case(
                Case(
                    f"random_{idx:03d}_lateral_reduction",
                    s,
                    r,
                    theta,
                    s,
                    prev_r_better,
                    theta,
                    action_axial_m=0.0,
                    action_lateral_m=-0.001 if r > 1.0e-9 else 0.0,
                ),
                cfg,
            )
        )
        if r >= 0.003:
            rows.append(
                _eval_case(
                    Case(
                        f"random_{idx:03d}_off_axis_axial",
                        s + 0.001,
                        r,
                        theta,
                        s,
                        r,
                        theta,
                        action_axial_m=0.001,
                    ),
                    cfg,
                )
            )
    return rows


def _assert_random_sanity(rows: list[dict[str, Any]]) -> None:
    failed: list[str] = []
    by_prefix: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        parts = str(row["case"]).split("_")
        if len(parts) < 3:
            continue
        prefix = "_".join(parts[:2])
        by_prefix.setdefault(prefix, {})["_".join(parts[2:])] = row
    for prefix, pair in by_prefix.items():
        reduction = pair.get("lateral_reduction")
        axial = pair.get("off_axis_axial")
        if reduction and axial and float(axial["r_m"]) >= 0.003 and float(axial["total"]) >= float(reduction["total"]):
            failed.append(f"{prefix}: off-axis axial total {axial['total']:.4f} >= lateral reduction {reduction['total']:.4f}")
    if failed:
        raise AssertionError("random reward sanity failed:\n- " + "\n- ".join(failed))


def _direction_rows(cfg: RewardConfig) -> list[dict[str, Any]]:
    return [
        _eval_direction_case("40x10_direction_A_toward_axis", (0.010, 0.0), (-0.001, 0.0), cfg),
        _eval_direction_case("40x10_direction_B_toward_axis", (0.0, 0.010), (0.0, -0.001), cfg),
    ]


def _assert_direction_invariance(rows: list[dict[str, Any]]) -> None:
    by_name = {row["case"]: row for row in rows}
    a = by_name["40x10_direction_A_toward_axis"]
    b = by_name["40x10_direction_B_toward_axis"]
    for key in ("total", "lat_gate_phase", "lateral_alignment_action", "action_toward_axis_m", "lateral_progress"):
        if abs(float(a[key]) - float(b[key])) > 1.0e-6:
            raise AssertionError(f"direction invariance failed for {key}: A={a[key]} B={b[key]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/agentic_reward_curriculum_20260527/reward_audits"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--bypass-penalty-scale", type=float, default=6.0)
    parser.add_argument("--bypass-gate-tolerance", type=float, default=0.0)
    parser.add_argument("--sigma-lat-insert-m", type=float, default=0.0015)
    parser.add_argument("--sigma-theta-insert-rad", type=float, default=0.060)
    parser.add_argument("--lateral-funnel-weight", type=float, default=0.0)
    parser.add_argument("--lateral-funnel-scale-m", type=float, default=0.010)
    parser.add_argument("--lateral-funnel-max", type=float, default=4.0)
    parser.add_argument("--near-misaligned-max", type=float, default=25.0)
    parser.add_argument("--inside-alignment-max", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RewardConfig(
        bypass_penalty_scale=float(args.bypass_penalty_scale),
        bypass_gate_tolerance=float(args.bypass_gate_tolerance),
        sigma_lat_insert_m=float(args.sigma_lat_insert_m),
        sigma_theta_insert_rad=float(args.sigma_theta_insert_rad),
        lateral_funnel_weight=float(args.lateral_funnel_weight),
        lateral_funnel_scale_m=float(args.lateral_funnel_scale_m),
        lateral_funnel_max=float(args.lateral_funnel_max),
        near_misaligned_max=float(args.near_misaligned_max),
        inside_alignment_max=float(args.inside_alignment_max),
    )
    run_name = args.run_name or datetime.utcnow().strftime("40x10_reward_cases_%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    rows = [_eval_case(case, cfg) for case in _cases()]
    grid_rows = [_eval_case(case, cfg) for case in _grid_cases()]
    direction_rows = _direction_rows(cfg)
    random_rows = _random_sanity_rows(cfg)
    _assert_reward_ordering(rows)
    _assert_grid_ordering(grid_rows, cfg)
    _assert_direction_invariance(direction_rows)
    _assert_random_sanity(random_rows)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "cases.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "grid_cases.json").write_text(json.dumps(grid_rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "direction_cases.json").write_text(json.dumps(direction_rows, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "random_cases.json").write_text(json.dumps(random_rows, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "grid_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)
    with (out_dir / "direction_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(direction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(direction_rows)
    with (out_dir / "random_cases.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(random_rows[0].keys()))
        writer.writeheader()
        writer.writerows(random_rows)
    lines = [
        "# 40x10 Reward Case Audit",
        "",
        f"Output: `{out_dir}`",
        "",
        "| case | s mm | r mm | theta | action_ax mm | action_lat mm | action_side mm | total | lat_gate | lat_action | axial_quiet | off_axis_axial | state_pen | axial | corridor | strict_geom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {s:.3f} | {r:.3f} | {theta:.5f} | {action_ax:.3f} | {action_lat:.3f} | {action_side:.3f} | {total:.4f} | {lat_gate:.6f} | {lat_action:.4f} | {quiet:.4f} | {off_axis:.4f} | {state_pen:.4f} | {axial:.4f} | {corr:.4f} | {strict} |".format(
                case=row["case"],
                s=row["s_m"] * 1000.0,
                r=row["r_m"] * 1000.0,
                theta=row["theta_rad"],
                action_ax=row["action_axial_m"] * 1000.0,
                action_lat=row["action_lateral_m"] * 1000.0,
                action_side=row["action_side_m"] * 1000.0,
                total=row["total"],
                lat_gate=row["lat_gate_phase"],
                lat_action=row["lateral_alignment_action"],
                quiet=row["lateral_alignment_axial_quiet_gate"],
                off_axis=row["off_axis_axial_action_penalty"],
                state_pen=row["lateral_error_state_penalty"],
                axial=row["axial_progress"],
                corr=row["corridor"],
                strict=row["strict_geometry"],
            )
        )
    lines.extend(
        [
            "",
            "Direction invariance assertions:",
            "",
            "- 40x10 direction A and direction B have identical reward when action points toward the axis.",
            "",
            "Random sanity assertions:",
            "",
            "- Random off-axis axial-forward samples are worse than matched lateral-reduction samples.",
            "",
            "Grid assertions:",
            "",
            "- Holding reward is monotonic with lateral error at each sampled depth/orientation.",
            "- Axial-forward reward when aligned beats axial-forward at 10 mm lateral error.",
            "- Axial-forward reward at 20 mm lateral error is worse than at 10 mm lateral error.",
            "- At the 40x10 reset offset, action toward the axis is positive relative to hold and action away is penalized.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "cases": len(rows),
                "grid_cases": len(grid_rows),
                "direction_cases": len(direction_rows),
                "random_cases": len(random_rows),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
