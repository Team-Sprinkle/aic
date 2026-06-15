#!/usr/bin/env python3
"""Compare successful Gazebo teacher insertion traces with Isaac insertion diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_env(value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return None
        if isinstance(value[0], list):
            return value[0]
        return value[0]
    return value


def _vec_norm(values: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in values))


def _quat_angle_xyzw(q0: list[float], q1: list[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(q0, q1))
    dot = max(-1.0, min(1.0, abs(dot)))
    return 2.0 * math.acos(dot)


def _summarize_gazebo_attempt(attempt_dir: Path) -> dict[str, Any]:
    runtime = _read_jsonl(attempt_dir / "runtime_trace.jsonl")
    smooth_path = attempt_dir / "smooth_trajectory.json"
    command_rows: list[dict[str, Any]] = []
    if smooth_path.exists():
        smooth = json.loads(smooth_path.read_text(encoding="utf-8"))
        command_rows = smooth.get("waypoints") or []
    replay_command_path = attempt_dir / "replay_command_trace.jsonl"
    if not replay_command_path.exists():
        replay_command_path = (
            attempt_dir.parents[1] / "accepted_dataset_nominal" / "debug" / "replay_command_trace.jsonl"
        )
    replay_commands = _read_jsonl(replay_command_path)
    final_commands = [
        row
        for row in replay_commands
        if row.get("phase") == "final_insertion" or row.get("command_source") == "guarded_insert"
    ]
    target_positions = [
        (row.get("target_tcp_pose") or {}).get("position")
        for row in final_commands
        if isinstance((row.get("target_tcp_pose") or {}).get("position"), list)
    ]
    target_quats = [
        (row.get("target_tcp_pose") or {}).get("orientation_xyzw")
        for row in final_commands
        if isinstance((row.get("target_tcp_pose") or {}).get("orientation_xyzw"), list)
    ]
    guarded_start = next((row for row in runtime if row.get("event") == "guarded_insert_started"), {})
    success_events = [row for row in runtime if "success" in str(row.get("event", ""))]
    z_values = [float(pos[2]) for pos in target_positions]
    xy_values = [[float(pos[0]), float(pos[1])] for pos in target_positions]
    xy_ref = xy_values[0] if xy_values else [float("nan"), float("nan")]
    xy_drift = [_vec_norm([xy[0] - xy_ref[0], xy[1] - xy_ref[1]]) for xy in xy_values]
    z_steps = [abs(z_values[i + 1] - z_values[i]) for i in range(len(z_values) - 1)]
    quat_drift = [_quat_angle_xyzw(target_quats[0], quat) for quat in target_quats[1:]] if target_quats else []

    videos = {}
    for camera in ("center", "left", "right"):
        path = attempt_dir / "dataset" / "videos" / f"observation.images.{camera}_camera" / "chunk-000" / "file-000.mp4"
        if path.exists():
            videos[camera] = str(path)

    return {
        "attempt_dir": str(attempt_dir),
        "runtime_trace": str(attempt_dir / "runtime_trace.jsonl"),
        "replay_command_trace": str(replay_command_path),
        "videos": videos,
        "guarded_insert_started": guarded_start,
        "success_events": success_events,
        "final_command_count": len(final_commands),
        "final_command_time_start_s": final_commands[0].get("timestamp") if final_commands else None,
        "final_command_time_end_s": final_commands[-1].get("timestamp") if final_commands else None,
        "final_target_z_start_m": z_values[0] if z_values else None,
        "final_target_z_end_m": z_values[-1] if z_values else None,
        "final_target_z_travel_m": (z_values[0] - z_values[-1]) if len(z_values) >= 2 else None,
        "final_target_z_step_mean_m": statistics.mean(z_steps) if z_steps else None,
        "final_target_z_step_max_m": max(z_steps) if z_steps else None,
        "final_target_xy_drift_max_m": max(xy_drift) if xy_drift else None,
        "final_target_orientation_drift_max_rad": max(quat_drift) if quat_drift else 0.0,
        "smooth_waypoint_count": len(command_rows),
    }


def _isaac_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    tip = row.get("post_step_insertion_geometry") or {}
    mod = row.get("post_step_module_geometry") or row.get("post_step_all_body_insertion_geometry", {}).get(
        "sfp_module_link", {}
    )
    contact = row.get("contact") or {}
    action = row.get("action_env0") or []
    realized = row.get("realized_body_motion") or {}
    tip_motion = realized.get("sfp_tip_link") or {}
    module_motion = realized.get("sfp_module_link") or {}
    return {
        "step": row.get("step"),
        "tip_s_m": _first_env(tip.get("signed_depth_m_by_env")),
        "tip_r_m": _first_env(tip.get("lateral_error_m_by_env")),
        "tip_theta_rad": _first_env(tip.get("orientation_error_rad_by_env")),
        "module_s_m": _first_env(mod.get("signed_depth_m_by_env")),
        "module_r_m": _first_env(mod.get("lateral_error_m_by_env")),
        "module_theta_rad": _first_env(mod.get("orientation_error_rad_by_env")),
        "consistency_gate": _first_env(tip.get("consistency_gate_by_env")),
        "target_depth_m": _first_env(tip.get("target_depth_m_by_env")),
        "strict_success": bool(_first_env(tip.get("strict_success_by_env"))),
        "force_proxy_norm": _first_env(contact.get("force_norm_max_by_env")),
        "action_norm": _vec_norm([float(x) for x in action]) if action else None,
        "realized_tip_delta_norm_m": _first_env(tip_motion.get("delta_norm_m_by_env")),
        "realized_module_delta_norm_m": _first_env(module_motion.get("delta_norm_m_by_env")),
    }


def _summarize_isaac_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = _read_jsonl(run_dir / "metrics.jsonl")
    metrics = [_isaac_row_metrics(row) for row in rows]
    metrics = [row for row in metrics if isinstance(row.get("tip_s_m"), (int, float))]
    if not metrics:
        return {"run_dir": str(run_dir), "n_rows": 0}, []
    best_tip = max(metrics, key=lambda row: float(row["tip_s_m"]))
    best_module = max(metrics, key=lambda row: float(row["module_s_m"] or -999.0))
    final = metrics[-1]
    target_depth = best_tip.get("target_depth_m")
    ceiling_rows = [
        row
        for row in metrics
        if isinstance(row.get("tip_s_m"), (int, float)) and 0.020 <= float(row["tip_s_m"]) <= 0.030
    ]
    return (
        {
            "run_dir": str(run_dir),
            "n_rows": len(metrics),
            "strict_success_any": any(bool(row.get("strict_success")) for row in metrics),
            "best_tip_row": best_tip,
            "best_module_row": best_module,
            "final_row": final,
            "remaining_depth_error_m": (
                float(target_depth) - float(best_tip["tip_s_m"])
                if isinstance(target_depth, (int, float))
                else None
            ),
            "mid_depth_row_count_20_30mm": len(ceiling_rows),
            "mean_module_s_at_20_30mm_m": (
                statistics.mean(float(row["module_s_m"]) for row in ceiling_rows if row.get("module_s_m") is not None)
                if ceiling_rows
                else None
            ),
        },
        metrics,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gazebo-attempt-dir", type=Path, required=True)
    parser.add_argument("--isaac-run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/aic_agentic_reward_curriculum_20260529/cheatcode_trace_compare"))
    parser.add_argument("--run-name", default="gazebo_isaac_mid_insertion_compare")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = args.output_dir / f"{timestamp}_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (out_dir / "git_status.txt").write_text(_run_git(["status", "--short"]), encoding="utf-8")
    (out_dir / "git_diff.patch").write_text(_run_git(["diff", "--", "."]), encoding="utf-8")

    gazebo_summary = _summarize_gazebo_attempt(args.gazebo_attempt_dir)
    _write_json(out_dir / "gazebo_summary.json", gazebo_summary)

    isaac_summaries = []
    for run_dir in args.isaac_run_dir:
        summary, metrics = _summarize_isaac_run(run_dir)
        isaac_summaries.append(summary)
        safe_name = run_dir.name
        _write_json(out_dir / f"isaac_summary_{safe_name}.json", summary)
        _write_csv(out_dir / f"isaac_metrics_{safe_name}.csv", metrics)

    comparison = {
        "gazebo": gazebo_summary,
        "isaac": isaac_summaries,
        "interpretation": {
            "gazebo_final_command": (
                "Successful nominal Gazebo replay commands an absolute Cartesian final insertion with "
                "nearly pinned XY/orientation and monotonic target-z descent."
            ),
            "isaac_failure": (
                "Isaac diagnostics reach mid positive tip depth but module depth remains near or behind the entrance; "
                "strict success remains false."
            ),
        },
    }
    _write_json(out_dir / "comparison_summary.json", comparison)

    lines = [
        "# Gazebo vs Isaac Mid-Insertion Comparison",
        "",
        f"Output folder: `{out_dir}`",
        "",
        "## Gazebo Successful Replay",
        "",
        f"- Attempt: `{args.gazebo_attempt_dir}`",
        f"- Final insertion commands: `{gazebo_summary['final_command_count']}`",
        f"- Target Z travel: `{(gazebo_summary['final_target_z_travel_m'] or 0.0) * 1000.0:.3f} mm`",
        f"- Mean target Z step: `{(gazebo_summary['final_target_z_step_mean_m'] or 0.0) * 1000.0:.3f} mm`",
        f"- Max XY drift during final insertion: `{(gazebo_summary['final_target_xy_drift_max_m'] or 0.0) * 1000.0:.3f} mm`",
        f"- Max orientation target drift: `{gazebo_summary['final_target_orientation_drift_max_rad']:.6f} rad`",
        "",
        "## Isaac Diagnostics",
        "",
        "| run | best tip s | best r | theta | module s | remaining depth | final tip s | strict |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for summary in isaac_summaries:
        best = summary.get("best_tip_row") or {}
        final = summary.get("final_row") or {}
        lines.append(
            "| `{}` | `{:.3f} mm` | `{:.3f} mm` | `{:.4f}` | `{:.3f} mm` | `{:.3f} mm` | `{:.3f} mm` | `{}` |".format(
                Path(str(summary.get("run_dir"))).name,
                float(best.get("tip_s_m") or 0.0) * 1000.0,
                float(best.get("tip_r_m") or 0.0) * 1000.0,
                float(best.get("tip_theta_rad") or 0.0),
                float(best.get("module_s_m") or 0.0) * 1000.0,
                float(summary.get("remaining_depth_error_m") or 0.0) * 1000.0,
                float(final.get("tip_s_m") or 0.0) * 1000.0,
                str(summary.get("strict_success_any")),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Gazebo's successful final insertion is an absolute Cartesian target sequence with pinned lateral target and "
            "orientation, while the current Isaac teacher diagnostics fail to make the module body follow the tip through "
            "the 20-30 mm mid-depth region. The next Isaac test should port the Gazebo-style pinned final target and "
            "lateral/z-gate behavior into the Isaac guide only if semantic module metrics improve; otherwise the blocker "
            "is likely contact/asset/controller realization rather than reward or speed.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(out_dir)


if __name__ == "__main__":
    main()
