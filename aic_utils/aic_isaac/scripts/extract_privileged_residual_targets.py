#!/usr/bin/env python3
"""Extract controller-aware residual targets from Isaac insertion metrics.

The SERL replay buffer is not persisted, so historical metrics cannot become a
full imitation dataset by themselves.  This script still turns privileged
post-step geometry into a reproducible residual-target audit: it labels centered
high-theta states, pre-jump states, lateral bypasses, and strict-like safe
states, then writes per-step command targets that can drive the next guarded
residual experiment design.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _finite(value: Any, default: float = float("nan")) -> float:
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _vec(value: Any, n: int = 3) -> list[float]:
    if not isinstance(value, list):
        return [0.0] * n
    if value and isinstance(value[0], list):
        value = value[0]
    out = []
    for item in value[:n]:
        out.append(_finite(item, 0.0))
    while len(out) < n:
        out.append(0.0)
    return out


def _tip_geom(row: dict[str, Any]) -> dict[str, Any]:
    all_body = row.get("post_step_all_body_insertion_geometry")
    if isinstance(all_body, dict) and isinstance(all_body.get("sfp_tip_link"), dict):
        return all_body["sfp_tip_link"]
    geom = row.get("post_step_insertion_geometry")
    return geom if isinstance(geom, dict) else {}


def _module_geom(row: dict[str, Any]) -> dict[str, Any]:
    all_body = row.get("post_step_all_body_insertion_geometry")
    if isinstance(all_body, dict) and isinstance(all_body.get("sfp_module_link"), dict):
        return all_body["sfp_module_link"]
    geom = row.get("post_step_module_geometry")
    if isinstance(geom, dict):
        return geom
    return {}


@dataclass
class ResidualTargetRow:
    run: str
    step: int
    env_index: int
    label: str
    s_m: float
    r_m: float
    theta_rad: float
    module_s_m: float
    module_r_m: float
    depth_gap_to_full_m: float
    module_gap_to_entrance_m: float
    next_s_delta_m: float
    next_module_s_delta_m: float
    consistency_gate: float
    force_n: float
    guarded_dx_m: float
    guarded_dy_m: float
    guarded_dz_m: float
    target_dx_m: float
    target_dy_m: float
    target_dz_m: float
    target_rx_rad: float
    target_ry_rad: float
    target_rz_rad: float
    guide_weight: float
    teacher_required: bool
    unsafe_reason: str


def _label(
    *,
    row: dict[str, Any],
    next_row: dict[str, Any] | None,
    s: float,
    r: float,
    theta: float,
    module_s: float,
    module_r: float,
    consistency_gate: float,
    force_n: float,
    lateral_gate_m: float,
    theta_gate_rad: float,
    module_lateral_gate_m: float,
    target_depth_m: float,
    positive_depth_m: float,
    module_depth_gate_m: float,
    force_gate_n: float,
    prejump_r_increase_m: float,
    expected_tip_module_gap_m: float,
    consistency_axial_gate_m: float,
) -> tuple[str, str]:
    next_r = float("nan")
    next_s = float("nan")
    if next_row is not None:
        g = _tip_geom(next_row)
        next_r = _finite(g.get("lateral_error_m_by_env") or g.get("lateral_error_m"))
        next_s = _finite(g.get("signed_depth_m_by_env") or g.get("signed_depth_m"))
    module_depth_blocked = s >= positive_depth_m and module_s < module_depth_gate_m
    centered = r <= lateral_gate_m
    module_laterally_near = module_r <= module_lateral_gate_m
    depth_gap_m = target_depth_m - s
    consistency_axial_error = abs((s - module_s) - expected_tip_module_gap_m) if math.isfinite(module_s) else float("nan")
    near_full_consistent = (
        depth_gap_m <= 0.002
        and centered
        and module_laterally_near
        and math.isfinite(consistency_axial_error)
        and consistency_axial_error <= consistency_axial_gate_m
    )
    if near_full_consistent and theta > theta_gate_rad:
        if force_n >= force_gate_n:
            return (
                "near_full_orientation_contact_blocked",
                f"gap_to_full {depth_gap_m*1000.0:.1f} mm, theta {theta:.4f}, force {force_n:.2f}",
            )
        return (
            "near_full_orientation_blocked",
            f"gap_to_full {depth_gap_m*1000.0:.1f} mm, theta {theta:.4f}",
        )
    if force_n >= force_gate_n:
        return "contact_spike", f"force {force_n:.2f} >= {force_gate_n:.2f}"
    if math.isfinite(next_r) and math.isfinite(r) and next_r - r >= prejump_r_increase_m:
        return "prejump_realization_mismatch", f"next_r-r {(next_r-r)*1000.0:.2f} mm"
    if r > 0.003:
        return "lateral_bypass", f"r {r*1000.0:.2f} mm"
    if s > 0.0 and consistency_gate < 0.4:
        return "tip_depth_false_positive", f"s positive with consistency {consistency_gate:.3f}"
    if centered and module_depth_blocked and theta > theta_gate_rad:
        return (
            "module_depth_blocked_centered_high_theta",
            f"gap_to_full {depth_gap_m*1000.0:.1f} mm, module_s {module_s*1000.0:.1f} mm, theta {theta:.4f}",
        )
    if centered and module_depth_blocked and theta <= theta_gate_rad:
        return (
            "module_depth_blocked_centered_strict_theta",
            f"gap_to_full {depth_gap_m*1000.0:.1f} mm, module_s {module_s*1000.0:.1f} mm",
        )
    if centered and theta > theta_gate_rad:
        if module_laterally_near:
            return "centered_high_theta_module_near", f"theta {theta:.4f} > {theta_gate_rad:.4f}"
        return "centered_high_theta_module_offset", (
            f"theta {theta:.4f}, module_r {module_r*1000.0:.2f} mm"
        )
    if centered and theta <= theta_gate_rad and module_laterally_near:
        if math.isfinite(next_s) and next_s >= s - 1.0e-5:
            return "safe_centered_progress", "strict gates except depth"
        return "safe_centered_hold", "strict gates but no axial progress"
    if centered:
        return "centered_other", "centered but not strict-like"
    return "approach_or_realign", "outside centered gate"


def _target_for_label(
    *,
    label: str,
    guarded: list[float],
    s: float,
    r: float,
    theta: float,
    max_lateral_step_m: float,
    backoff_step_m: float,
    orientation_step_rad: float,
) -> tuple[list[float], float]:
    target = [guarded[0], guarded[1], guarded[2], 0.0, 0.0, 0.0]
    weight = 1.0
    if label in {"prejump_realization_mismatch", "contact_spike", "lateral_bypass", "tip_depth_false_positive"}:
        target[:3] = [0.0, 0.0, max(backoff_step_m, 0.0)]
        weight = 2.0
    elif label.startswith("module_depth_blocked"):
        # These rows are useful negative evidence, not demonstrations.  A
        # deeper module-following teacher/residual source is required before
        # they should receive imitation weight.
        target = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        weight = 0.0
    elif label.startswith("near_full_orientation") or label.startswith("centered_high_theta"):
        # Hold translation inside the narrow corridor and request a tiny bounded
        # orientation residual.  The sign remains an experiment parameter because
        # prior command-sign probes showed frame-dependent behavior.
        target[:3] = [
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[0])),
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[1])),
            0.0,
        ]
        target[3] = min(max(theta, 0.0), orientation_step_rad)
        weight = 2.5
    elif label == "safe_centered_progress":
        target[:3] = [
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[0])),
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[1])),
            min(guarded[2], 5.0e-6),
        ]
        weight = 1.5
    elif label == "safe_centered_hold":
        target[2] = 0.0
        weight = 1.25
    else:
        target[:3] = [
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[0])),
            max(-max_lateral_step_m, min(max_lateral_step_m, guarded[1])),
            0.0 if r > 0.001 or theta > 0.05 else min(guarded[2], 5.0e-6),
        ]
        weight = 0.75
    return target, weight


def extract_run(
    run_dir: Path,
    *,
    lateral_gate_m: float,
    theta_gate_rad: float,
    module_lateral_gate_m: float,
    target_depth_m: float,
    positive_depth_m: float,
    module_depth_gate_m: float,
    force_gate_n: float,
    prejump_r_increase_m: float,
    expected_tip_module_gap_m: float,
    consistency_axial_gate_m: float,
    max_lateral_step_m: float,
    backoff_step_m: float,
    orientation_step_rad: float,
) -> list[ResidualTargetRow]:
    metrics_path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: list[ResidualTargetRow] = []
    for idx, row in enumerate(rows):
        post_geom = row.get("post_step_insertion_geometry") or {}
        tip = _tip_geom(row)
        module = _module_geom(row)
        sample_count = max(
            len(tip.get("signed_depth_m_by_env") or []),
            len(tip.get("lateral_error_m_by_env") or []),
            len(tip.get("orientation_error_rad_by_env") or []),
            len(module.get("signed_depth_m_by_env") or []),
            1,
        )
        for env_index in range(sample_count):
            s = _finite((tip.get("signed_depth_m_by_env") or [tip.get("signed_depth_m")])[env_index:env_index + 1])
            r = _finite((tip.get("lateral_error_m_by_env") or [tip.get("lateral_error_m")])[env_index:env_index + 1])
            theta_source = (
                post_geom.get("orientation_error_rad_by_env")
                or tip.get("orientation_error_rad_by_env")
                or [tip.get("orientation_error_rad")]
            )
            if env_index == 0:
                theta_source = (
                    post_geom.get("orientation_error_rad_env0")
                    or theta_source
                )
            theta = _finite(theta_source[env_index:env_index + 1] if isinstance(theta_source, list) else theta_source)
            if not (math.isfinite(s) and math.isfinite(r) and math.isfinite(theta)):
                continue
            module_s = _finite((module.get("signed_depth_m_by_env") or [module.get("signed_depth_m")])[env_index:env_index + 1])
            module_r = _finite((module.get("lateral_error_m_by_env") or [module.get("lateral_error_m")])[env_index:env_index + 1])
            next_s = float("nan")
            next_module_s = float("nan")
            if idx + 1 < len(rows):
                next_tip = _tip_geom(rows[idx + 1])
                next_module = _module_geom(rows[idx + 1])
                next_s_values = next_tip.get("signed_depth_m_by_env") or [next_tip.get("signed_depth_m")]
                next_module_s_values = next_module.get("signed_depth_m_by_env") or [next_module.get("signed_depth_m")]
                if env_index < len(next_s_values):
                    next_s = _finite(next_s_values[env_index:env_index + 1])
                if env_index < len(next_module_s_values):
                    next_module_s = _finite(next_module_s_values[env_index:env_index + 1])
            consistency_values = (
                post_geom.get("consistency_gate_by_env")
                or tip.get("consistency_gate_by_env")
                or [tip.get("consistency_gate")]
            )
            consistency_gate = _finite(
                consistency_values[env_index:env_index + 1]
                if isinstance(consistency_values, list) and env_index < len(consistency_values)
                else row.get("insertion_action_guard_module_consistency_gate_mean"),
                0.0,
            )
            force_n = _finite(row.get("force_norm_mean"), 0.0)
            guarded_values = row.get("insertion_action_guard_guarded_world_delta_by_env")
            guarded = _vec(
                guarded_values[env_index]
                if isinstance(guarded_values, list)
                and env_index < len(guarded_values)
                and isinstance(guarded_values[env_index], list)
                else guarded_values,
                3,
            )
            label, reason = _label(
                row=row,
                next_row=rows[idx + 1] if idx + 1 < len(rows) else None,
                s=s,
                r=r,
                theta=theta,
                module_s=module_s,
                module_r=module_r,
                consistency_gate=consistency_gate,
                force_n=force_n,
                lateral_gate_m=lateral_gate_m,
                theta_gate_rad=theta_gate_rad,
                module_lateral_gate_m=module_lateral_gate_m,
                target_depth_m=target_depth_m,
                positive_depth_m=positive_depth_m,
                module_depth_gate_m=module_depth_gate_m,
                force_gate_n=force_gate_n,
                prejump_r_increase_m=prejump_r_increase_m,
                expected_tip_module_gap_m=expected_tip_module_gap_m,
                consistency_axial_gate_m=consistency_axial_gate_m,
            )
            target, weight = _target_for_label(
                label=label,
                guarded=guarded,
                s=s,
                r=r,
                theta=theta,
                max_lateral_step_m=max_lateral_step_m,
                backoff_step_m=backoff_step_m,
                orientation_step_rad=orientation_step_rad,
            )
            out.append(
                ResidualTargetRow(
                    run=str(run_dir),
                    step=int(row.get("step", idx + 1)),
                    env_index=env_index,
                    label=label,
                    s_m=s,
                    r_m=r,
                    theta_rad=theta,
                    module_s_m=module_s,
                    module_r_m=module_r,
                    depth_gap_to_full_m=target_depth_m - s,
                    module_gap_to_entrance_m=max(0.0, -module_s) if math.isfinite(module_s) else float("nan"),
                    next_s_delta_m=next_s - s if math.isfinite(next_s) and math.isfinite(s) else float("nan"),
                    next_module_s_delta_m=next_module_s - module_s
                    if math.isfinite(next_module_s) and math.isfinite(module_s)
                    else float("nan"),
                    consistency_gate=consistency_gate,
                    force_n=force_n,
                    guarded_dx_m=guarded[0],
                    guarded_dy_m=guarded[1],
                    guarded_dz_m=guarded[2],
                    target_dx_m=target[0],
                    target_dy_m=target[1],
                    target_dz_m=target[2],
                    target_rx_rad=target[3],
                    target_ry_rad=target[4],
                    target_rz_rad=target[5],
                    guide_weight=weight,
                    teacher_required=label.startswith("module_depth_blocked"),
                    unsafe_reason=reason,
                )
            )
    return out


def summarize(rows: list[ResidualTargetRow]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    for row in rows:
        labels[row.label] = labels.get(row.label, 0) + 1
    centered_high_theta = [row for row in rows if row.label.startswith("centered_high_theta")]
    module_depth_blocked = [row for row in rows if row.label.startswith("module_depth_blocked")]
    near_full_blocked = [row for row in rows if row.label.startswith("near_full_orientation")]
    prejump = [row for row in rows if row.label == "prejump_realization_mismatch"]
    safe = [row for row in rows if row.label.startswith("safe_centered")]
    best_centered = None
    centered = [row for row in rows if row.r_m <= 0.0007]
    if centered:
        best_centered = max(centered, key=lambda item: item.s_m)
    return {
        "rows": len(rows),
        "label_counts": dict(sorted(labels.items())),
        "centered_high_theta_rows": len(centered_high_theta),
        "module_depth_blocked_rows": len(module_depth_blocked),
        "near_full_orientation_blocked_rows": len(near_full_blocked),
        "prejump_rows": len(prejump),
        "safe_centered_rows": len(safe),
        "best_centered": None if best_centered is None else asdict(best_centered),
        "recommendation": recommendation(labels, best_centered),
    }


def recommendation(labels: dict[str, int], best_centered: ResidualTargetRow | None) -> dict[str, Any]:
    if labels.get("prejump_realization_mismatch", 0) > 0:
        primary = "enable pre-step realized-r rejection after all recovery/rotation overrides"
    elif (
        labels.get("near_full_orientation_contact_blocked", 0)
        + labels.get("near_full_orientation_blocked", 0)
        > 0
    ):
        primary = "test bounded final-window orientation/contact micro-recovery before more reward-only training"
    elif (
        labels.get("module_depth_blocked_centered_high_theta", 0)
        + labels.get("module_depth_blocked_centered_strict_theta", 0)
        > 0
    ):
        primary = "collect deeper module-following teacher trajectories before more reward-only training"
    elif labels.get("centered_high_theta_module_near", 0) + labels.get("centered_high_theta_module_offset", 0) > 0:
        primary = "train privileged residual orientation hold only inside centered corridor"
    else:
        primary = "collect more privileged servo data; current rows lack centered orientation-blocked examples"
    return {
        "primary_next_change": primary,
        "suggested_flags": {
            "actor_state_history_steps": 4,
            "target_action_guide_rotation_loss_weight": 0.1,
            "target_action_guide_train_executed": False,
            "residual_target_source": "privileged_residual_targets.csv",
            "orientation_residual_step_rad": 0.00025,
            "orientation_residual_lateral_gate_m": 0.0005,
            "orientation_residual_theta_gate_rad": 0.030,
            "module_depth_teacher_required": True,
            "teacher_target_depth_m": 0.046864,
            "teacher_min_module_s_m": 0.0,
            "block_axial_when_orientation_bad": True,
            "backoff_on_prejump": True,
        },
        "best_centered_gap": None
        if best_centered is None
        else {
            "s_gap_to_full_mm": round(best_centered.depth_gap_to_full_m * 1000.0, 3),
            "module_gap_to_entrance_mm": round(best_centered.module_gap_to_entrance_m * 1000.0, 3),
            "r_mm": round(best_centered.r_m * 1000.0, 3),
            "theta_rad": round(best_centered.theta_rad, 5),
            "module_r_mm": round(best_centered.module_r_m * 1000.0, 3),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lateral-gate-m", type=float, default=0.0007)
    parser.add_argument("--theta-gate-rad", type=float, default=0.030)
    parser.add_argument("--module-lateral-gate-m", type=float, default=0.0015)
    parser.add_argument("--target-depth-m", type=float, default=0.046864)
    parser.add_argument("--positive-depth-m", type=float, default=0.001)
    parser.add_argument("--module-depth-gate-m", type=float, default=0.0)
    parser.add_argument("--force-gate-n", type=float, default=20.0)
    parser.add_argument("--prejump-r-increase-m", type=float, default=0.003)
    parser.add_argument("--expected-tip-module-gap-m", type=float, default=0.02365)
    parser.add_argument("--consistency-axial-gate-m", type=float, default=0.001)
    parser.add_argument("--max-lateral-step-m", type=float, default=0.00008)
    parser.add_argument("--backoff-step-m", type=float, default=0.00008)
    parser.add_argument("--orientation-step-rad", type=float, default=0.00025)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[ResidualTargetRow] = []
    for run_dir in args.runs:
        all_rows.extend(
            extract_run(
                run_dir,
                lateral_gate_m=float(args.lateral_gate_m),
                theta_gate_rad=float(args.theta_gate_rad),
                module_lateral_gate_m=float(args.module_lateral_gate_m),
                target_depth_m=float(args.target_depth_m),
                positive_depth_m=float(args.positive_depth_m),
                module_depth_gate_m=float(args.module_depth_gate_m),
                force_gate_n=float(args.force_gate_n),
                prejump_r_increase_m=float(args.prejump_r_increase_m),
                expected_tip_module_gap_m=float(args.expected_tip_module_gap_m),
                consistency_axial_gate_m=float(args.consistency_axial_gate_m),
                max_lateral_step_m=float(args.max_lateral_step_m),
                backoff_step_m=float(args.backoff_step_m),
                orientation_step_rad=float(args.orientation_step_rad),
            )
        )
    csv_path = args.output_dir / "privileged_residual_targets.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ResidualTargetRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    summary = {
        "inputs": [str(path) for path in args.runs],
        "thresholds": {
            "lateral_gate_m": float(args.lateral_gate_m),
            "theta_gate_rad": float(args.theta_gate_rad),
            "module_lateral_gate_m": float(args.module_lateral_gate_m),
            "target_depth_m": float(args.target_depth_m),
            "positive_depth_m": float(args.positive_depth_m),
            "module_depth_gate_m": float(args.module_depth_gate_m),
            "force_gate_n": float(args.force_gate_n),
            "prejump_r_increase_m": float(args.prejump_r_increase_m),
            "expected_tip_module_gap_m": float(args.expected_tip_module_gap_m),
            "consistency_axial_gate_m": float(args.consistency_axial_gate_m),
        },
        **summarize(all_rows),
    }
    (args.output_dir / "privileged_residual_targets_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
