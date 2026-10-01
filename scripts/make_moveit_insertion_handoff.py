#!/usr/bin/env python3
"""Append a no-motion phase marker for a privileged CheatCode insertion test."""

from dataclasses import replace
import json
from pathlib import Path
import sys

from aic_teacher_official.trajectory import PhaseLabel, SmoothTrajectory


def main(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / "plan_summary.json").read_text())
    trajectory = SmoothTrajectory.load_json(source / "moveit_joint_trajectory.json")
    last = trajectory.waypoints[-1]
    assert last.joint_positions and last.phase == PhaseLabel.OBSTACLE_AVOIDANCE
    trigger = replace(last, timestamp=last.timestamp + 0.1,
                      phase=PhaseLabel.FINAL_INSERTION,
                      diagnostics={**last.diagnostics, "handoff": "privileged_online_cheatcode"})
    trajectory.waypoints.append(trigger)
    trajectory.save_json(destination / "moveit_joint_trajectory.json")
    summary["teacher_insertion_handoff"] = {
        "enabled": True, "phase_marker_time_s": trigger.timestamp,
        "marker_has_no_robot_motion": True,
        "privileged": True,
    }
    summary["trajectory_waypoints"] = len(trajectory.waypoints)
    (destination / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"trajectory_waypoints": len(trajectory.waypoints),
                      "handoff_time_s": trigger.timestamp}))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
