#!/usr/bin/env python3
"""Package the fixed five-card route probe without the large scoring bags."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import yaml

from transcode_compat_mp4 import transcode


RUNS = {
    "across_1": "across_cards_v3",
    "across_2": "across_cards_rep2",
    "across_3": "across_cards_rep3",
    "outside_1": "outside_left",
    "outside_2": "outside_left_rep2",
    "outside_3": "outside_left_rep3",
    "low_clearance": "across_cards_low",
}
KEEP = (
    "eval_config.yaml", "route_plan.json", "results/scoring.yaml",
    "trial_analysis.json", "geometry/cable_geometry_summary.json",
    "geometry/cable_samples.json", "geometry/cable_overhead_side_force.png",
    "visuals/all_cameras_1hz.mp4", "visuals/five_timepoints.jpg",
)


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main(source_root: Path, stock_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for label, dirname in RUNS.items():
        source = source_root / dirname
        for relative in KEEP:
            if relative == "visuals/all_cameras_1hz.mp4":
                transcode(source / relative, target_root / label / relative)
            else:
                copy(source / relative, target_root / label / relative)
        score = yaml.safe_load((source / "results/scoring.yaml").read_text())
        tier = score["trial_000001"]["tier_3"]
        analysis = json.loads((source / "trial_analysis.json").read_text())
        geometry = json.loads((source / "geometry/cable_geometry_summary.json").read_text())
        plan = json.loads((source / "route_plan.json").read_text())
        result = ("full" if "successful" in tier["message"].lower()
                  else "partial" if "partial" in tier["message"].lower()
                  else "none")
        physics_errors = (source / "engine.log").read_text(errors="replace").count("physics entity ptr")
        policy_log = (source / "policy.log").read_text(errors="replace")
        record = {
            "label": label, "bulk_root": str(source), "variant": plan["variant"],
            "low_offset_m": plan.get("low_offset_m", 0.105), "outcome": result,
            "tier3_score": tier["score"], "tier3_message": tier["message"],
            "total_score": score["total"], "tier1_validation": score["trial_000001"]["tier_1"],
            "physics_entity_errors": physics_errors,
            "route_completed": "Diagnostic route complete" in policy_log,
            "policy_exception": "Traceback" in policy_log,
            "wrist_force_peak_n": analysis["force_max_n"],
            "scorer_insertion_force_message": score["trial_000001"]["tier_2"]["categories"]["insertion force"]["message"],
            "terminal_lateral_mm": analysis["final"]["lateral_mm"],
            "terminal_axial_mm": analysis["final"]["axial_mm"],
            "terminal_orientation_deg": analysis["final"]["orientation_deg"],
            "terminal_tcp_command_error_mm": analysis["final"]["tcp_command_error_mm"],
            "minimum_main_pcb_center_gap_mm": geometry["minimum_main_pcb_center_gap_mm"],
            "minimum_gap_time_s": geometry["minimum_gap_time_s"],
            "wrist_force_at_minimum_gap_n": geometry["wrist_force_at_minimum_gap_n"],
            "cable_link5_motion_60_to_80_s_mm": geometry["cable_link5_motion_60_to_80_s_mm"],
            "video": f"{label}/visuals/all_cameras_1hz.mp4",
            "geometry_plot": f"{label}/geometry/cable_overhead_side_force.png",
        }
        assert record["physics_entity_errors"] == 0 and record["route_completed"] and not record["policy_exception"], label
        records.append(record)
    for relative in ("results/scoring.yaml", "geometry/cable_geometry_summary.json",
                     "geometry/cable_overhead_side_force.png"):
        copy(stock_root / relative, target_root / "stock_control" / relative)
    summary = {
        "schema": "aic_fixed_five_card_route_probe/v1", "date": "2026-09-23",
        "image": "ghcr.io/intrinsic-dev/aic/aic_eval@sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923",
        "scene": "The same generated SC-to-SC, one-port, five-NIC-card scene from seed 51500 was used in every run; the exact eval_config.yaml is saved per run.",
        "actor": "Privileged diagnostic waypoint route from Gazebo TF, then installed stock CheatCode for insertion. No learned policy and no training.",
        "review_video_encoding": "H.264/AVC Constrained Baseline, yuv420p, faststart; images sampled at 1 Hz and repeated at 10 fps without changing timeline duration",
        "stock_control": {"bulk_root": str(stock_root), "total_score": 54.50136311050045,
                          "tier3_score": 39.671110596431397,
                          "outcome": "partial", "minimum_main_pcb_center_gap_mm":
                          json.loads((stock_root / "geometry/cable_geometry_summary.json").read_text())["minimum_main_pcb_center_gap_mm"]},
        "records": records,
        "ordinary_height_counts": {
            "across_cards": {outcome: sum(r["outcome"] == outcome for r in records if r["label"].startswith("across_")) for outcome in ("full", "partial", "none")},
            "outside_left": {outcome: sum(r["outcome"] == outcome for r in records if r["label"].startswith("outside_")) for outcome in ("full", "partial", "none")},
        },
        "s3_audit": {
            "bucket": "aic-team-sprinkle",
            "checked_clean_prefix": "s3://aic-team-sprinkle/datasets/clean/sc_to_sc/agent/sc_ports_1/n100__sc_ports1_nic5_n100/",
            "clean_prefix_contains": "accepted_dataset/ and agent_generation/accepted_metadata/ plus generation_config.json; no failed replay_attempts or original failed seed-51500 video",
            "additional_prefixes_checked": ["s3://aic-team-sprinkle/datasets/dev/", "s3://aic-team-sprinkle/ec2_transfer/"],
            "downloaded_video_key": "s3://aic-team-sprinkle/datasets/clean/sc_to_sc/agent/sc_ports_1/n100__sc_ports1_nic5_n100/accepted_dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4",
            "downloaded_video_sha256": "9fb99750d4d5fc3c7fdf3a9342520eab2d972db2aad2c68cfb4f5efa17583430",
            "downloaded_video_matches_local_accepted": True,
            "downloaded_selection_report_sha256": "279f9ecc853181d2ca4152b1f58a96865cd9182621d51147f6fa4a5b87b7a860",
            "downloaded_selection_report_matches_local": True,
            "accepted_trial_ids": ["trial_000027", "trial_000028"],
            "caveat": "The clean video contains accepted attempts 27 and 28, not the historical failed seed-51500 route. Absence is limited to the prefixes inventoried here.",
        },
        "interpretation": "One normal-height across-card run had a cable segment center within 0.9 mm of a main PCB collider and that segment later barely moved while the plug stalled. Repeats of the same planned route inserted fully or partially with larger cable/card gap. This is a route-sensitive cable-trap candidate, not a named cable/card contact or proven causal snag. The lower pass stranded the robot far from the port with the cable far from cards and is a clearance failure confound.",
    }
    (target_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(target_root / "summary.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
