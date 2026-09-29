#!/usr/bin/env python3
"""Summarize retained Gazebo trace outcomes without labeling contact cause."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def summary(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    vectors = [row["wrist_force_xyz_n"] for row in rows if row["time_s"] < 5
               and row.get("wrist_force_xyz_n") is not None]
    if not vectors:
        raise ValueError(f"No initial force-vector baseline in {path}")
    baseline = [statistics.median(vector[axis] for vector in vectors) for axis in range(3)]
    tared = [(row["time_s"], math.dist(row["wrist_force_xyz_n"], baseline))
             for row in rows if row.get("wrist_force_xyz_n") is not None]
    first_80 = next((row["time_s"] for row in rows
                     if (row["tcp_tracking_error_m"] or 0) >= 0.080), None)
    return {
        "samples": len(rows),
        "peak_tracking_error_mm": 1000 * max((row["tcp_tracking_error_m"] or 0) for row in rows),
        "peak_vector_tared_force_n": max(force for _, force in tared),
        "first_80mm_tracking_error_s": first_80,
        "recent_10s_tared_force_at_80mm_n": (max(force for time, force in tared
                                                  if first_80 - 10 <= time <= first_80)
                                              if first_80 is not None else None),
        "end_time_s": rows[-1]["time_s"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text())
    outcomes = {row["id"]: row for row in index["gazebo_incidents"]}
    result = []
    for number in range(2, 9):
        incident_id = f"wide_across_repeat_{number:02}"
        path = args.trace_dir / f"wide_repeat_{number:02}_trace.jsonl"
        result.append({"incident_id": incident_id, "outcome": outcomes[incident_id]["outcome"],
                       "scene_group": outcomes[incident_id]["scene_group"], **summary(path)})
    for number in range(1, 11):
        prefix = f"targeted_trial_{number:02}_"
        incident_id = next(name for name in outcomes if name.startswith(prefix))
        path = args.trace_dir / f"targeted_{number:02}_trace.jsonl"
        result.append({"incident_id": incident_id, "outcome": outcomes[incident_id]["outcome"],
                       "scene_group": outcomes[incident_id]["scene_group"], **summary(path)})
    report = {
        "schema": "aic_recovery_trace_cohorts/v1",
        "rows": result,
        "warning": "Thresholds were inspected after outcomes and are diagnostic only. Repeats within a scene group are correlated. High tracking error and force do not identify a cable contact.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"episodes": len(result), "groups": len(set(row["scene_group"] for row in result)),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
