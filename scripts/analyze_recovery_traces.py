#!/usr/bin/env python3
"""Read-only diagnostic for retained Gazebo TCP/force traces.

This identifies candidate *obstruction* intervals, not the contact pair or a
deployable trigger. The thresholds are exploratory and must be validated on
episode-grouped scenes before driving the robot.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
from pathlib import Path


def analyze(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    times = [row["time_s"] for row in rows]
    baseline_rows = [row["wrist_force_norm_n"] for row in rows if row["time_s"] < 5]
    if not rows or not baseline_rows:
        raise ValueError(f"Trace lacks rows or first-five-second force baseline: {path}")
    baseline = statistics.median(baseline_rows)
    vector_baseline_rows = [row["wrist_force_xyz_n"] for row in rows
                            if row["time_s"] < 5 and row.get("wrist_force_xyz_n") is not None]
    vector_baseline = ([statistics.median(row[axis] for row in vector_baseline_rows) for axis in range(3)]
                       if vector_baseline_rows else None)
    first = None
    first_norm_subtraction = None
    first_strict = None
    strict_count = 0
    for i, row in enumerate(rows):
        if i == 0:
            continue
        previous = rows[i - 1]
        realized_step = math.dist(row["measured_tcp_base_m"], previous["measured_tcp_base_m"])
        vector_tared_force = (math.dist(row["wrist_force_xyz_n"], vector_baseline)
                              if vector_baseline and row.get("wrist_force_xyz_n") is not None else 0.0)
        high_force = vector_tared_force >= 8.0
        demand = max(row["command_motion_since_last_sample_m"], row["tcp_tracking_error_m"] or 0) >= 0.00005
        strict_count = strict_count + 1 if high_force and demand and realized_step <= 0.00015 else 0
        if strict_count >= 2 and first_strict is None:
            first_strict = row["time_s"]
        j = bisect.bisect_left(times, row["time_s"] - 2.0)
        k = bisect.bisect_left(times, row["time_s"] - 10.0)
        if j >= i:
            continue
        recent_force_excess = max((math.dist(r["wrist_force_xyz_n"], vector_baseline)
                                   if vector_baseline and r.get("wrist_force_xyz_n") is not None else 0.0)
                                  for r in rows[k : i + 1])
        recent_norm_subtraction = max(r["wrist_force_norm_n"] - baseline for r in rows[k : i + 1])
        displacement_mm = 1000 * math.dist(row["measured_tcp_base_m"], rows[j]["measured_tcp_base_m"])
        error_mm = 1000 * (row["tcp_tracking_error_m"] or 0)
        error_growth_mm = error_mm - 1000 * (rows[j]["tcp_tracking_error_m"] or 0)
        if first is None and recent_force_excess > 8 and displacement_mm < 2 and error_mm > 40 and error_growth_mm > 8:
            first = {
                "time_s": row["time_s"],
                "force_excess_recent_10s_n": recent_force_excess,
                "tcp_displacement_recent_2s_mm": displacement_mm,
                "tracking_error_mm": error_mm,
                "tracking_error_growth_recent_2s_mm": error_growth_mm,
            }
        if first_norm_subtraction is None and recent_norm_subtraction > 8 and displacement_mm < 2 and error_mm > 40 and error_growth_mm > 8:
            first_norm_subtraction = row["time_s"]
    return {
        "trace": str(path),
        "samples": len(rows),
        "force_norm_baseline_first_5s_n": baseline,
        "force_vector_baseline_first_5s_n": vector_baseline,
        "peak_vector_tared_force_n": (max(math.dist(row["wrist_force_xyz_n"], vector_baseline)
                                            for row in rows if row.get("wrist_force_xyz_n") is not None)
                                       if vector_baseline else None),
        "peak_force_norm_n": max(row["wrist_force_norm_n"] for row in rows),
        "strict_two_sample_trigger_time_s": first_strict,
        "exploratory_vector_tared_force_latch_progress_trigger": first,
        "incorrect_norm_subtraction_trigger_time_s_for_comparison_only": first_norm_subtraction,
        "warning": "All thresholds are exploratory. Tared vector force is physically meaningful; subtracting norms can miss directional changes. This is observation-side diagnosis, not contact identification or a live controller.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = {"schema": "aic_recovery_trace_diagnostic/v1", "traces": [analyze(path) for path in args.traces]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
