#!/usr/bin/env python3
"""Run a reproducible privileged insertion-servo sweep without Isaac startup.

This script is intentionally geometry-level.  It exercises the same semantic
state variables used by the Isaac rewards and action guards (`s`, `r`, `theta`,
trailing-body consistency, force/contact proxy, commanded vs realized motion)
so controller logic can be audited before spending Isaac/GPU time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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


@dataclass
class ServoConfig:
    target_depth_m: float = 0.046864
    max_steps: int = 1000
    lateral_gate_m: float = 0.0006
    orientation_gate_rad: float = 0.030
    lateral_servo_gain: float = 0.65
    lateral_step_limit_m: float = 0.00035
    orientation_servo_gain: float = 0.45
    orientation_step_limit_rad: float = 0.0030
    axial_step_m: float = 0.00006
    near_gate_depth_m: float = -0.004
    force_limit_n: float = 8.0
    contact_lateral_m: float = 0.0012
    backoff_step_m: float = 0.00030
    backoff_lateral_m: float = 0.00015
    max_retries: int = 4
    module_consistency_threshold: float = 0.80
    controller_lateral_realization: float = 0.92
    controller_axial_realization: float = 0.88
    controller_orientation_realization: float = 0.70
    rotation_sweep_m_per_rad: float = 0.075
    force_base_n: float = 0.6
    force_bad_gate_n: float = 13.0
    force_activation_depth_m: float = -0.001
    seed: int = 7


ORIENTATION_STARTS = {
    "small": 0.035,
    "medium": 0.070,
    "hard": 0.120,
}


def _parse_mm_list(raw: str) -> list[float]:
    return [float(item.strip()) * 0.001 for item in raw.split(",") if item.strip()]


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).stdout
    except Exception as exc:
        return f"<git command failed: {exc}>"


def _geometry(s: float, r: float, cfg: ServoConfig):
    body = torch.tensor([[s, r, 0.0]], dtype=torch.float32)
    entrance = torch.zeros_like(body)
    target = torch.tensor([[cfg.target_depth_m, 0.0, 0.0]], dtype=torch.float32)
    axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    return compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=cfg.lateral_gate_m,
    )


def _strict_success(s: float, r: float, theta: float, module_gate: float, max_force: float, cfg: ServoConfig) -> bool:
    return (
        s >= cfg.target_depth_m - 0.0005
        and r <= cfg.lateral_gate_m
        and theta <= cfg.orientation_gate_rad
        and module_gate >= cfg.module_consistency_threshold
        and max_force <= cfg.force_limit_n
    )


def _failure_mode(
    *,
    s: float,
    r: float,
    theta: float,
    module_gate: float,
    max_force: float,
    axial_progress_m: float,
    realization_mismatch_m: float,
    cfg: ServoConfig,
) -> str:
    if max_force > cfg.force_limit_n:
        return "contact_spike"
    if module_gate < cfg.module_consistency_threshold:
        return "module_consistency_failure"
    if r > max(cfg.lateral_gate_m, 0.0015):
        return "lateral_bypass"
    if theta > cfg.orientation_gate_rad:
        return "orientation_residual"
    if realization_mismatch_m > 0.001:
        return "controller_realization_mismatch"
    if axial_progress_m < 0.001:
        return "no_axial_progress"
    return "near_success_not_strict"


def _simulate_case(case_id: str, lateral_start_m: float, axial_start_m: float, theta_start: float, theta_label: str, cfg: ServoConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(cfg.seed + abs(hash(case_id)) % 100000)
    s = -float(axial_start_m)
    r = float(lateral_start_m)
    theta = float(theta_start)
    previous_s = s
    previous_r = r
    previous_theta = theta
    retries = 0
    max_force = 0.0
    max_realization_mismatch = 0.0
    contact_events = 0
    rows: list[dict[str, Any]] = []
    initial_s = s

    for step in range(int(cfg.max_steps)):
        phase = "align"
        cmd_lateral = 0.0
        cmd_theta = 0.0
        cmd_axial = 0.0
        if s < cfg.near_gate_depth_m:
            phase = "near_gate"
            cmd_axial = min(cfg.axial_step_m, cfg.near_gate_depth_m - s)
        if r > cfg.lateral_gate_m:
            phase = "lateral_correct"
            cmd_lateral = -min(cfg.lateral_step_limit_m, cfg.lateral_servo_gain * r)
            cmd_axial = min(cmd_axial, 0.0)
        elif theta > cfg.orientation_gate_rad:
            phase = "orientation_trim"
            cmd_theta = -min(cfg.orientation_step_limit_rad, cfg.orientation_servo_gain * theta)
            cmd_axial = 0.0
        else:
            phase = "insert"
            cmd_lateral = -min(cfg.lateral_step_limit_m, cfg.lateral_servo_gain * r)
            cmd_theta = -min(cfg.orientation_step_limit_rad, cfg.orientation_servo_gain * theta)
            cmd_axial = cfg.axial_step_m

        rotation_sweep = abs(cmd_theta) * cfg.rotation_sweep_m_per_rad * (1.0 + 1.5 * max(r - cfg.lateral_gate_m, 0.0) / 0.010)
        realized_lateral = cfg.controller_lateral_realization * cmd_lateral + rotation_sweep
        realized_axial = cfg.controller_axial_realization * cmd_axial
        realized_theta = cfg.controller_orientation_realization * cmd_theta
        if phase == "insert" and r < cfg.lateral_gate_m and theta < cfg.orientation_gate_rad:
            realized_lateral += float(rng.normal(0.0, 0.000015))
            realized_axial += float(rng.normal(0.0, 0.00001))

        next_s = s + realized_axial
        next_r = max(0.0, r + realized_lateral)
        next_theta = max(0.0, theta + realized_theta)
        bad_forward = cmd_axial > 0.0 and (r > cfg.lateral_gate_m or theta > cfg.orientation_gate_rad)
        contact = next_s > -0.001 and (next_r > cfg.contact_lateral_m or next_theta > 2.0 * cfg.orientation_gate_rad)
        force_depth_gate = 1.0 if next_s >= cfg.force_activation_depth_m else 0.0
        force = cfg.force_base_n + force_depth_gate * (
            2400.0 * max(next_r - cfg.lateral_gate_m, 0.0)
            + 35.0 * max(next_theta - cfg.orientation_gate_rad, 0.0)
        )
        if bad_forward:
            force += force_depth_gate * cfg.force_bad_gate_n
        if contact:
            force += 4.0
            contact_events += 1
        max_force = max(max_force, force)

        if force > cfg.force_limit_n and retries < cfg.max_retries:
            phase = "recover"
            retries += 1
            next_s = s - cfg.backoff_step_m
            next_r = max(0.0, r - cfg.backoff_lateral_m)
            next_theta = max(0.0, theta - 0.5 * cfg.orientation_step_limit_rad)

        module_lag_m = 0.02365
        module_depth = next_s - module_lag_m
        expected_module_depth = cfg.target_depth_m - module_lag_m
        module_axial_error = abs(module_depth - expected_module_depth)
        module_gate = math.exp(-((module_axial_error / 0.0035) ** 2)) * math.exp(-((next_r / 0.0012) ** 2))
        commanded = np.array([cmd_axial, cmd_lateral, cmd_theta], dtype=float)
        realized = np.array([next_s - s, next_r - r, next_theta - theta], dtype=float)
        realization_mismatch = float(np.linalg.norm(commanded - realized))
        max_realization_mismatch = max(max_realization_mismatch, realization_mismatch)
        reward_comp = cheatcode_insertion_phase_reward(
            geometry=_geometry(next_s, next_r, cfg),
            previous_depth=torch.tensor([previous_s], dtype=torch.float32),
            previous_lateral_error=torch.tensor([previous_r], dtype=torch.float32),
            orientation_error=torch.tensor([next_theta], dtype=torch.float32),
            previous_orientation_error=torch.tensor([previous_theta], dtype=torch.float32),
            action_delta_w=torch.tensor([[cmd_axial, cmd_lateral, 0.0]], dtype=torch.float32),
            action_axis_gate=True,
            semantic_gate=torch.tensor([module_gate], dtype=torch.float32),
        )
        row = {
            "case_id": case_id,
            "step": step,
            "phase": phase,
            "s_m": next_s,
            "r_m": next_r,
            "theta_rad": next_theta,
            "module_consistency": module_gate,
            "force_n": force,
            "contact": bool(contact),
            "cmd_axial_m": cmd_axial,
            "cmd_lateral_m": cmd_lateral,
            "cmd_theta_rad": cmd_theta,
            "realized_axial_m": next_s - s,
            "realized_lateral_m": next_r - r,
            "realized_theta_rad": next_theta - theta,
            "realization_mismatch": realization_mismatch,
            "reward_total": float(reward_comp.total[0]),
            "reward_insert_gate": float(reward_comp.g_insert_combined[0]),
            "strict_success": _strict_success(next_s, next_r, next_theta, module_gate, max_force, cfg),
            "retries": retries,
        }
        rows.append(row)
        previous_s, previous_r, previous_theta = s, r, theta
        s, r, theta = next_s, next_r, next_theta
        if row["strict_success"]:
            break

    final = rows[-1]
    axial_progress = final["s_m"] - initial_s
    summary = {
        "case_id": case_id,
        "lateral_start_m": lateral_start_m,
        "axial_start_m": axial_start_m,
        "orientation_label": theta_label,
        "theta_start_rad": theta_start,
        "target_depth_m": cfg.target_depth_m,
        "steps": len(rows),
        "final_s_m": final["s_m"],
        "final_r_m": final["r_m"],
        "final_theta_rad": final["theta_rad"],
        "final_module_consistency": final["module_consistency"],
        "max_force_n": max_force,
        "contact_events": contact_events,
        "max_realization_mismatch_m": max_realization_mismatch,
        "strict_success": bool(final["strict_success"]),
        "failure_mode": "strict_success"
        if final["strict_success"]
        else _failure_mode(
            s=final["s_m"],
            r=final["r_m"],
            theta=final["theta_rad"],
            module_gate=final["module_consistency"],
            max_force=max_force,
            axial_progress_m=axial_progress,
            realization_mismatch_m=max_realization_mismatch,
            cfg=cfg,
        ),
    }
    return summary, rows


def _write_plots(run_dir: Path, all_rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots: dict[str, str] = {}
    fig, ax = plt.subplots(figsize=(8, 6))
    for case in summaries[: min(12, len(summaries))]:
        rows = [row for row in all_rows if row["case_id"] == case["case_id"]]
        ax.plot([r["s_m"] * 1000.0 for r in rows], [r["r_m"] * 1000.0 for r in rows], linewidth=1.0, label=case["case_id"])
    ax.axvline(0.0, color="black", linewidth=0.8)
    target_depth_m = summaries[0].get("target_depth_m", 0.0458) if summaries else 0.0458
    ax.axvline(target_depth_m * 1000.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("s axial depth (mm)")
    ax.set_ylabel("r lateral error (mm)")
    ax.set_title("Privileged Servo Trajectories")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    path = run_dir / "trajectory_snapshot.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots["trajectory_snapshot"] = str(path)

    fig, ax = plt.subplots(figsize=(8, 5))
    modes = sorted({str(item["failure_mode"]) for item in summaries})
    counts = [sum(1 for item in summaries if item["failure_mode"] == mode) for mode in modes]
    ax.bar(modes, counts)
    ax.set_ylabel("cases")
    ax.set_title("Strict Success / Failure Classification")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    path = run_dir / "failure_modes.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    plots["failure_modes"] = str(path)
    return plots


def _write_summary(run_dir: Path, summaries: list[dict[str, Any]], plots: dict[str, str]) -> None:
    success_count = sum(1 for row in summaries if row["strict_success"])
    lines = [
        "# Privileged Insertion Servo Sweep",
        "",
        f"Run folder: `{run_dir}`",
        f"Cases: {len(summaries)}",
        f"Strict successes: {success_count}/{len(summaries)}",
        "",
        "Strict success requires axial depth, lateral error, tip orientation, module/body consistency, and force/contact sanity. Tip-depth-only positive `s` is not counted.",
        "",
        "## Failure Modes",
        "",
        "| mode | count |",
        "| --- | ---: |",
    ]
    for mode in sorted({str(item["failure_mode"]) for item in summaries}):
        lines.append(f"| {mode} | {sum(1 for item in summaries if item['failure_mode'] == mode)} |")
    lines += [
        "",
        "## Case Table",
        "",
        "| case | strict | failure mode | final s mm | final r mm | theta rad | module | max force N |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            "| {case_id} | {strict_success} | {failure_mode} | {s:.2f} | {r:.2f} | {theta:.3f} | {module:.3f} | {force:.2f} |".format(
                case_id=item["case_id"],
                strict_success=str(item["strict_success"]).lower(),
                failure_mode=item["failure_mode"],
                s=item["final_s_m"] * 1000.0,
                r=item["final_r_m"] * 1000.0,
                theta=item["final_theta_rad"],
                module=item["final_module_consistency"],
                force=item["max_force_n"],
            )
        )
    lines += ["", "## Snapshots", ""]
    for name, path in plots.items():
        lines.append(f"- `{name}`: `{path}`")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_teacher_rows(run_dir: Path, summaries: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful_cases = {str(row["case_id"]) for row in summaries if row.get("strict_success")}
    teacher_rows = [row for row in all_rows if str(row.get("case_id")) in successful_cases]
    if teacher_rows:
        with (run_dir / "successful_teacher_trajectories.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(teacher_rows[0].keys()))
            writer.writeheader()
            writer.writerows(teacher_rows)
    summary = {
        "successful_cases": sorted(successful_cases),
        "successful_case_count": len(successful_cases),
        "teacher_rows": len(teacher_rows),
        "teacher_csv": str(run_dir / "successful_teacher_trajectories.csv") if teacher_rows else None,
    }
    (run_dir / "teacher_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/agent_reward_funnel/servo_sweeps"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--lateral-starts-mm", default="1,2,4,6,10")
    parser.add_argument("--axial-starts-mm", default="3,6,10,20")
    parser.add_argument("--orientation-starts", default="small,medium,hard")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--target-depth-m", type=float, default=0.046864)
    parser.add_argument("--axial-step-m", type=float, default=0.00006)
    parser.add_argument("--lateral-gate-m", type=float, default=0.0006)
    parser.add_argument("--orientation-gate-rad", type=float, default=0.030)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ServoConfig(
        max_steps=int(args.max_steps),
        target_depth_m=float(args.target_depth_m),
        axial_step_m=float(args.axial_step_m),
        lateral_gate_m=float(args.lateral_gate_m),
        orientation_gate_rad=float(args.orientation_gate_rad),
    )
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S_privileged_servo")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    lateral_starts = _parse_mm_list(args.lateral_starts_mm)
    axial_starts = _parse_mm_list(args.axial_starts_mm)
    orientation_names = [item.strip() for item in args.orientation_starts.split(",") if item.strip()]
    config = {
        "servo_config": asdict(cfg),
        "lateral_starts_m": lateral_starts,
        "axial_starts_m": axial_starts,
        "orientation_starts": {name: ORIENTATION_STARTS[name] for name in orientation_names},
        "script": str(Path(__file__).resolve()),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(_run_git(["diff", "--"]), encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for lateral in lateral_starts:
        for axial in axial_starts:
            for label in orientation_names:
                theta = ORIENTATION_STARTS[label]
                case_id = f"lat{lateral*1000:.0f}_ax{axial*1000:.0f}_{label}"
                summary, rows = _simulate_case(case_id, lateral, axial, theta, label, cfg)
                summaries.append(summary)
                all_rows.extend(rows)

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"summaries": summaries, "rows": all_rows}, f, indent=2, sort_keys=True)
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (run_dir / "case_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    teacher_summary = _write_teacher_rows(run_dir, summaries, all_rows)
    plots = _write_plots(run_dir, all_rows, summaries)
    _write_summary(run_dir, summaries, plots)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "successes": sum(1 for row in summaries if row["strict_success"]),
                "cases": len(summaries),
                "teacher_rows": teacher_summary["teacher_rows"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
