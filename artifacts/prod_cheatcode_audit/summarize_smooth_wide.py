#!/usr/bin/env python3
"""Summarize fixed-scene, native-rate, five-camera cable-route repeats."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(source_root: Path, compact_root: Path) -> None:
    records = []
    for run in sorted(source_root.glob("across_repeat_*")):
        score = yaml.safe_load((run / "results/scoring.yaml").read_text())
        tier = score["trial_000001"]["tier_3"]
        bag = next((run / "results").glob("bag_trial_*"))
        metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
        start = metadata["starting_time"]["nanoseconds_since_epoch"]
        end = start + metadata["duration"]["nanoseconds"]
        per_camera = {}
        with (run / "frame_manifest.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                per_camera.setdefault(row["camera"], []).append(int(row["wall_ns"]))
        overhead = per_camera["overhead"]
        outcome = ("none" if "No insertion" in tier["message"] else
                   "partial" if "Partial" in tier["message"] else "full")
        row = {
            "run": run.name, "bulk_root": str(run), "outcome": outcome,
            "tier3_score": tier["score"], "tier3_message": tier["message"],
            "total_score": score["total"],
            "camera_frame_counts": {key: len(value) for key, value in per_camera.items()},
            "overhead_capture_start_relative_to_bag_s": (overhead[0] - start) / 1e9,
            "overhead_capture_end_relative_to_bag_s": (overhead[-1] - end) / 1e9,
            "complete_task_camera_coverage": all(
                frames[0] <= start and frames[-1] >= end
                for frames in per_camera.values()),
            "bag": str(bag),
        }
        if (run / "geometry/cable_geometry_summary.json").exists():
            geometry = json.loads((run / "geometry/cable_geometry_summary.json").read_text())
            row["minimum_cable_center_main_pcb_gap_mm"] = geometry["minimum_main_pcb_center_gap_mm"]
            row["link5_motion_60_to_80_wall_s_mm"] = geometry["cable_link5_motion_60_to_80_s_mm"]
        if (run / "trial_analysis.json").exists():
            analysis = json.loads((run / "trial_analysis.json").read_text())
            row["terminal_axial_mm"] = analysis["final"]["axial_mm"]
            row["terminal_lateral_mm"] = analysis["final"]["lateral_mm"]
            row["terminal_orientation_deg"] = analysis["final"]["orientation_deg"]
            row["terminal_tcp_command_error_mm"] = analysis["final"]["tcp_command_error_mm"]
            row["peak_wrist_force_n"] = analysis["force_max_n"]
            row["insertion_event_count"] = analysis["raw_insertion_event_count"]
        packaged = compact_root / run.name
        if packaged.exists():
            row["compact_video_paths"] = {
                name: str(packaged / "visuals" / name)
                for name in ("all_views_20fps.mp4", "overhead_20fps.mp4",
                             "side_20fps.mp4", "wrist_triptych_20fps.mp4")}
        records.append(row)
    baseline = source_root / "across_repeat_02"
    summary = {
        "schema": "aic_cable_route_smooth_wide/v1", "date": "2026-09-23",
        "source_root": str(source_root),
        "scene_config_sha256": sha256(baseline / "eval_config.yaml"),
        "wide_world_sha256": sha256(baseline / "world_audit.sdf"),
        "camera_modification": "Two static visual-only Gazebo cameras without collision or inertia; the saved task scene, cable/robot assets, route policy and physics settings were otherwise unchanged.",
        "controller": "Privileged diagnostic waypoints followed by installed stock CheatCode; no trained actor and no training.",
        "rate": "The five image topics publish at 20 Hz simulation time; the MP4s are H.264/AVC Constrained Baseline at 20 fps with no duplicated one-Hz frames.",
        "records": records,
        "complete_repeats": [row["run"] for row in records if row["complete_task_camera_coverage"]],
        "complete_outcome_counts": {
            name: sum(row["complete_task_camera_coverage"] and row["outcome"] == name for row in records)
            for name in ("full", "partial", "none")},
        "interpretation_caveat": "Run 06 is a new no-insertion example with cable visible across the card row, but the bag lacks named cable/card contact. Its final lateral error and link-5 motion differ from the earlier across_1 cable-trap candidate; this new video does not prove a cable snag caused its failure.",
    }
    (compact_root / "smooth_wide_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["complete_outcome_counts"]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
