#!/usr/bin/env python3
"""Insert explicit joint-target holds after selected MoveIt route segments."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--hold", action="append", required=True,
                        help="segment_name:seconds; may be supplied more than once")
    args = parser.parse_args()
    holds = {}
    for spec in args.hold:
        name, seconds = spec.rsplit(":", 1)
        duration = float(seconds)
        if not name or duration <= 0 or name in holds:
            parser.error(f"Invalid or duplicate hold {spec!r}")
        holds[name] = duration
    source_path = args.source / "moveit_joint_trajectory.json"
    trajectory = json.loads(source_path.read_text())
    original = trajectory["waypoints"]
    assert original and all(item["joint_positions"] for item in original)
    segments = {item["diagnostics"]["segment"] for item in original}
    if missing := holds.keys() - segments:
        parser.error(f"Missing trajectory segments: {sorted(missing)}")
    result = []
    added = 0.0
    for index, item in enumerate(original):
        copied = deepcopy(item)
        copied["timestamp"] = float(item["timestamp"]) + added
        result.append(copied)
        name = item["diagnostics"]["segment"]
        at_end = index + 1 == len(original) or original[index + 1]["diagnostics"]["segment"] != name
        if at_end and name in holds:
            hold = deepcopy(copied)
            hold["timestamp"] += holds[name]
            hold["joint_velocities"] = [0.0] * len(hold["joint_positions"])
            hold["diagnostics"]["replay_modifier"] = "endpoint_joint_hold"
            hold["diagnostics"]["hold_duration_s"] = holds[name]
            result.append(hold)
            added += holds[name]
    assert all(b["timestamp"] > a["timestamp"] for a, b in zip(result, result[1:]))
    trajectory["waypoints"] = result
    summary = json.loads((args.source / "plan_summary.json").read_text())
    summary["postplan_joint_holds"] = {
        "source_plan": str(args.source),
        "source_trajectory_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "hold_seconds_by_segment": holds,
        "added_duration_s": added,
        "description": "Repeats each MoveIt endpoint joint target; source path geometry is unchanged.",
    }
    summary["trajectory_waypoints"] = len(result)
    args.destination.mkdir(parents=True, exist_ok=False)
    (args.destination / "moveit_joint_trajectory.json").write_text(json.dumps(trajectory, indent=2) + "\n")
    (args.destination / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"source_waypoints": len(original), "output_waypoints": len(result),
                      "added_duration_s": added, "holds": holds}))


if __name__ == "__main__":
    main()
