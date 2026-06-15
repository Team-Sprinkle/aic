#!/usr/bin/env python3
"""Summarize staged final-seat contact command diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _geom(row: dict[str, Any]) -> dict[str, Any]:
    geom = row.get("post_step_insertion_geometry")
    if isinstance(geom, dict):
        return geom
    return {}


def _metric(row: dict[str, Any], key: str) -> float | None:
    geom = _geom(row)
    value = geom.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit_started(row: dict[str, Any]) -> bool:
    reps = row.get("action_representations")
    if not isinstance(reps, dict):
        return False
    return bool(reps.get("debug_audit_started"))


def _audit_start_step(run_dir: Path) -> int | None:
    cfg_path = run_dir / "train_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    args = cfg.get("args") if isinstance(cfg.get("args"), dict) else {}
    value = args.get("debug_audit_start_step")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _audit_action_label(step: int, audit_start_step: int | None) -> str | None:
    if audit_start_step is None or step < audit_start_step:
        return None
    axis_index = (step - audit_start_step) % 6
    sign = "+" if axis_index < 3 else "-"
    axis = ("rx", "ry", "rz")[axis_index % 3]
    return f"{sign}{axis}"


def _summary(run_dir: Path) -> dict[str, Any]:
    rows = _rows(run_dir / "metrics.jsonl")
    audit_start = _audit_start_step(run_dir)
    started = [row for row in rows if _audit_started(row) or (audit_start is not None and int(row.get("step", 0)) >= audit_start)]
    baseline = rows[max(0, rows.index(started[0]) - 1)] if started else rows[0]
    final = rows[-1]
    best_theta_row = min(
        (row for row in started or rows if _metric(row, "orientation_error_rad_env0") is not None),
        key=lambda row: _metric(row, "orientation_error_rad_env0") or float("inf"),
    )
    best_score_row = max(
        (
            row
            for row in started or rows
            if _metric(row, "signed_depth_m_env0") is not None
            and _metric(row, "lateral_error_m_env0") is not None
            and _metric(row, "orientation_error_rad_env0") is not None
            and _metric(row, "consistency_gate_env0") is not None
        ),
        key=lambda row: (
            min((_metric(row, "signed_depth_m_env0") or 0.0) / 0.0458, 1.0) * 10.0
            - (_metric(row, "lateral_error_m_env0") or 1.0) * 1000.0
            - (_metric(row, "orientation_error_rad_env0") or 1.0) * 10.0
            + (_metric(row, "consistency_gate_env0") or 0.0)
        ),
    )

    def pack(row: dict[str, Any]) -> dict[str, Any]:
        step = int(row.get("step", 0))
        return {
            "step": step,
            "audit_action": _audit_action_label(step, audit_start),
            "s_m": _metric(row, "signed_depth_m_env0"),
            "r_m": _metric(row, "lateral_error_m_env0"),
            "theta_rad": _metric(row, "orientation_error_rad_env0"),
            "module_consistency": _metric(row, "consistency_gate_env0"),
            "force_n": row.get("force_norm_mean"),
            "audit_started": bool(_audit_started(row) or (audit_start is not None and step >= audit_start)),
        }

    base = pack(baseline)
    end = pack(final)
    by_action: dict[str, dict[str, Any]] = {}
    for row in started:
        packed = pack(row)
        label = packed["audit_action"]
        if label is None:
            continue
        previous = by_action.get(label)
        if previous is None or (
            packed["theta_rad"] is not None
            and (previous["theta_rad"] is None or packed["theta_rad"] < previous["theta_rad"])
        ):
            by_action[label] = packed
    return {
        "run_dir": str(run_dir),
        "audit_start_step": (pack(started[0])["step"] if started else None),
        "baseline": base,
        "final": end,
        "best_theta": pack(best_theta_row),
        "best_score": pack(best_score_row),
        "best_by_audit_action": by_action,
        "delta_final_minus_baseline": {
            "s_m": None if base["s_m"] is None or end["s_m"] is None else end["s_m"] - base["s_m"],
            "r_m": None if base["r_m"] is None or end["r_m"] is None else end["r_m"] - base["r_m"],
            "theta_rad": (
                None
                if base["theta_rad"] is None or end["theta_rad"] is None
                else end["theta_rad"] - base["theta_rad"]
            ),
            "module_consistency": (
                None
                if base["module_consistency"] is None or end["module_consistency"] is None
                else end["module_consistency"] - base["module_consistency"]
            ),
        },
        "strict_success": bool(
            end["s_m"] is not None
            and end["s_m"] >= 0.0072
            and end["r_m"] is not None
            and end["r_m"] <= 0.0005
            and end["theta_rad"] is not None
            and end["theta_rad"] <= 0.030
            and end["module_consistency"] is not None
            and end["module_consistency"] >= 0.80
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    summary = _summary(args.run_dir)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
