#!/usr/bin/env python3
"""Reconstruct effective run settings from expert matrix outputs.

Older matrix runs stored the command and generation config, but not the
matrix-runner environment override table. This script collects the immutable
parts that were saved and extracts effective runtime parameters from trace
events so a setting/mode/repeat can be rerun or compared later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CONFIG_EVENT_KEYS = {
    "tracking_gate_checked": (
        "threshold_m",
        "nominal_controller_threshold_m",
        "gate_source",
        "timeout_sec",
        "speed_threshold_mps",
        "max_lateral_error_m",
        "ft_threshold_n",
        "force_gate_threshold_n",
        "force_gate_fraction",
        "servo_compensation_enabled",
    ),
    "local_preinsert_align_started": (
        "requested_duration_sec",
        "duration_sec",
        "max_speed_mps",
        "z_offset",
        "vertical_bias_m",
        "preserve_current_z",
        "lateral_offset_base",
    ),
    "precontact_port_align_started": (
        "duration_sec",
        "max_offset_m",
        "residual_threshold_m",
        "speed_threshold_mps",
        "force_abort_threshold_n",
        "gain",
        "frame",
        "representation",
    ),
    "cheatcode_handoff_started": (
        "from_z_offset",
        "to_z_offset",
        "vertical_bias_m",
        "duration_sec",
        "handoff_speed_mps",
        "profile",
        "command_mode",
        "lateral_offset_base",
    ),
    "guarded_insert_started": (
        "insertion_speed_mps",
        "ft_threshold_n",
        "descent_profile",
        "cheatcode_z_mode",
        "insertion_command_mode",
        "insertion_start_z_offset",
        "vertical_bias_m",
        "insertion_depth_m",
        "fallback_insertion_depth_m",
        "insertion_duration_sec",
        "insertion_steps",
        "exact_position_step_m",
        "exact_position_settle_sec",
        "min_success_z_offset",
        "target_policy",
    ),
    "force_trigger_confirmed_checked": (
        "window_samples",
        "center_gap_samples",
        "center_gap_sec",
        "instant_threshold_n",
        "median_threshold_n",
        "median_rise_threshold_n",
        "sustained_threshold_n",
        "instant_trigger_enabled",
    ),
    "recovery_backoff_started": (
        "retry_count",
        "backoff_distance_m",
        "min_backoff_distance_m",
        "mode",
        "duration_sec",
    ),
    "recovery_backoff_stage_completed": (
        "retry_count",
        "stage_index",
        "requested_distance_m",
        "measured_distance_m",
    ),
    "recovery_return_to_preinsert_gate_checked": (
        "retry_count",
        "tracking_gate_passed",
        "threshold_m",
        "max_lateral_error_m",
    ),
}


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _extract_cmd_value(cmd: list[Any], flag: str) -> str | None:
    items = [str(item) for item in cmd]
    try:
        index = items.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(items):
        return None
    return items[index + 1]


def _runtime_config_events(summary: dict[str, Any]) -> dict[str, Any]:
    records = summary.get("records") if isinstance(summary, dict) else None
    if not isinstance(records, list) or not records:
        return {}
    replay_metrics = records[0].get("replay_metrics") or {}
    trace_path = replay_metrics.get("runtime_trace_path")
    if not trace_path:
        return {}
    trace = _read_jsonl(Path(trace_path))
    extracted: dict[str, list[dict[str, Any]]] = {}
    for event in trace:
        name = event.get("event")
        keys = CONFIG_EVENT_KEYS.get(str(name))
        if not keys:
            continue
        payload = {key: event.get(key) for key in keys if key in event}
        if payload:
            extracted.setdefault(str(name), []).append(payload)
    return {
        name: {
            "first": values[0],
            "last": values[-1],
            "count": len(values),
        }
        for name, values in extracted.items()
    }


def _row_reconstruction(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result") or {}
    summary_path = Path(result.get("summary", ""))
    summary = _read_json(summary_path) or {}
    generation_config = _read_json(summary_path.parent / "generation_config.json") or {}
    cmd = result.get("cmd") or []
    return {
        "schema_version": "aic_expert_matrix_attempt_reconstruction/v1",
        "output_root": str(root),
        "setting": row.get("setting"),
        "mode": row.get("mode"),
        "repeat": row.get("repeat"),
        "passed": row.get("passed"),
        "score_threshold": row.get("score_threshold"),
        "scores": result.get("scores"),
        "accepted": result.get("accepted"),
        "returncode": result.get("returncode"),
        "seed": row.get("seed") or _extract_cmd_value(cmd, "--seed"),
        "cmd": cmd,
        "cmd_args": {
            "config": _extract_cmd_value(cmd, "--config"),
            "output_dir": _extract_cmd_value(cmd, "--output-dir"),
            "score_threshold": _extract_cmd_value(cmd, "--score-threshold"),
            "ft_threshold": _extract_cmd_value(cmd, "--ft-threshold"),
        },
        "embedded_run_config": row.get("run_config"),
        "embedded_mode_env": result.get("mode_env"),
        "embedded_mode_args": result.get("mode_args"),
        "generation_config_args": generation_config.get("args"),
        "generation_ft_guard": generation_config.get("ft_guard"),
        "generation_replay": generation_config.get("replay"),
        "generation_planner_recording": generation_config.get("planner_recording"),
        "runtime_effective_params": _runtime_config_events(summary),
        "summary_path": str(summary_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for root in args.roots:
            matrix_log = root / "matrix_results.jsonl"
            for row in _read_jsonl(matrix_log):
                handle.write(json.dumps(_row_reconstruction(root, row), sort_keys=True) + "\n")
                count += 1
    print(f"wrote {count} reconstruction row(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
