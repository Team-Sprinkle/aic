#!/usr/bin/env python3
"""Summarize the fixed-scene September 29 low-cross route comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import yaml


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    scene_hashes = {name: set() for name in (
        "eval_config.yaml", "world_audit.sdf", "bridge_audit.yaml", "CableRouteDiagnostic.py")}
    start_poses = []
    port_poses = []
    for run in sorted(args.root.iterdir()):
        if not (run / "summary.json").is_file():
            continue
        score = read_json(run / "summary.json")
        variant = read_json(run / "variation.json")
        geometry = read_json(run / "geometry/cable_geometry_summary.json")
        port_trace = read_json(run / "port_trace.json")
        route = read_json(run / "route_plan.json")
        for name, hashes in scene_hashes.items():
            hashes.add(sha256(run / name))
        start_poses.append(route["start_tcp_base_m"])
        port_poses.append(route["port_link_base_m"])
        crossing = next((sample for sample in port_trace["samples"]
                         if sample["entrance_relative_xyz_mm"][2] >= 0), None)
        final = port_trace["final"]

        def lateral(sample: dict | None) -> float | None:
            if sample is None:
                return None
            return math.hypot(*sample["entrance_relative_xyz_mm"][:2])

        bags = list((run / "results").glob("bag_*/metadata.yaml"))
        if len(bags) != 1:
            raise ValueError(f"Expected one scored bag in {run}: {bags}")
        rows.append({
            "case": run.name,
            "route_family": variant["case"],
            "behind_extra_m": variant["behind_extra_m"],
            "lane_extra_m": variant["lane_extra_m"],
            "forward_extra_m": variant["forward_extra_m"],
            "full_insertion": score["tier3"] == 75,
            "tier3": score["tier3"],
            "insertion_event_count": port_trace["insertion_event_count"],
            "entrance_crossing_lateral_mm": lateral(crossing),
            "entrance_crossing_orientation_deg": (
                crossing["entrance_orientation_deg"] if crossing else None),
            "final_entrance_xyz_mm": final["entrance_relative_xyz_mm"],
            "final_lateral_mm": lateral(final),
            "final_orientation_deg": final["entrance_orientation_deg"],
            "minimum_sampled_cable_center_gap_mm": geometry["minimum_main_pcb_center_gap_mm"],
            "peak_wrist_force_n": geometry["peak_force_n"],
            "scored_bag": str(bags[0].parent),
            "route_plan": str(run / "route_plan.json"),
            "score": str(run / "results/scoring.yaml"),
            "port_trace": str(run / "port_trace.json"),
            "cable_geometry": str(run / "geometry/cable_geometry_summary.json"),
        })

    if any(len(hashes) != 1 for hashes in scene_hashes.values()):
        raise ValueError(f"Scene or policy code changed across cases: {scene_hashes}")
    original = yaml.safe_load((args.historical / "results/scoring.yaml").read_text())
    groups = {}
    for family in sorted({row["route_family"] for row in rows}):
        group = [row for row in rows if row["route_family"] == family]
        gaps = [row["minimum_sampled_cable_center_gap_mm"] for row in group]
        groups[family] = {
            "new_runs": len(group),
            "new_full_insertions": sum(row["full_insertion"] for row in group),
            "median_sampled_cable_center_gap_mm": statistics.median(gaps),
            "minimum_sampled_cable_center_gap_mm": min(gaps),
            "maximum_sampled_cable_center_gap_mm": max(gaps),
        }
    groups["repeat_failure"]["historical_partial_included"] = True
    groups["repeat_failure"]["all_runs"] = groups["repeat_failure"]["new_runs"] + 1
    groups["repeat_failure"]["all_full_insertions"] = groups["repeat_failure"]["new_full_insertions"]
    spans = lambda values: [1000 * (max(row[i] for row in values) -
                                   min(row[i] for row in values)) for i in range(3)]
    result = {
        "schema": "sc_low_cross_2cm_rescue/v1",
        "scene": "five NIC cards at leftmost allowed positions; SC port 0; fixed development seed 51500",
        "simulator": "Gazebo official evaluation image, privileged CableRouteDiagnostic teacher",
        "model_training": False,
        "final_scenes_opened": False,
        "historical_case": {
            "path": str(args.historical),
            "route": "outside_left_low_cross_return; forward_extra_m=-0.01; other extras=0",
            "tier3": float(original["trial_000001"]["tier_3"]["score"]),
            "full_insertion": False,
        },
        "scene_and_policy_sha256": {
            name: next(iter(hashes)) for name, hashes in scene_hashes.items()},
        "start_tcp_span_xyz_mm": spans(start_poses),
        "port_link_span_xyz_mm": spans(port_poses),
        "groups": groups,
        "runs": rows,
        "cable_gap_caveat": "Sampled link-center to simplified main PCB; excludes cable radius, card protrusions and named contacts.",
        "diagnosis_caveat": "Port trace uses scoring TF and sc_tip_link proxy. It does not identify named physical contact or prove the cause of a stall.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"groups": groups, "output": str(args.output)}))


if __name__ == "__main__":
    main()
