#!/usr/bin/env python3
"""Freeze all completed and interrupted fixed-scene MoveIt/Gazebo route attempts."""

import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path("/var/tmp/chmin_aic_20260918_act")
CASES = {
    "sparse_baseline": ("moveit_sc_corridor_baseline_20260929", "baseline_video"),
    "sparse_lane_plus20": ("moveit_sc_corridor_lane20_20260929", "lane20_video"),
    "strict_return_interrupted": ("moveit_sc_corridor_strict_return_20260929", None),
    "strict_return_retry1": ("moveit_sc_corridor_strict_return_retry1_20260929", "strict_return_video"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main(output: Path):
    root = Path(__file__).resolve().parents[1]
    video_root = root / "outputs/experiments/2026-09-29_moveit_gazebo_pilot"
    rows = []
    for label, (directory, video_name) in CASES.items():
        run = ROOT / directory
        score_path = run / "results/scoring.yaml"
        row = {
            "label": label, "run_dir": str(run),
            "config_sha256": sha(run / "eval_config.yaml"),
            "world_sha256": sha(run / "world_audit.sdf"),
            "bridge_sha256": sha(run / "bridge_audit.yaml"),
            "plan_sha256": sha(run / "plan_summary.json"),
            "trajectory_sha256": sha(run / "moveit_joint_trajectory.json"),
            "scored": score_path.is_file(),
        }
        if score_path.is_file():
            score = yaml.safe_load(score_path.read_text())
            row["score"] = {"total": score["total"],
                            "tier3": score["trial_000001"]["tier_3"],
                            "contact": score["trial_000001"]["tier_2"]["categories"]["contacts"]}
            row["scoring_sha256"] = sha(score_path)
            for name in ("route_replay_audit", "exact_cable_clearance", "port_trace"):
                path = run / (name + ".json")
                if path.is_file():
                    data = json.loads(path.read_text())
                    if name == "exact_cable_clearance":
                        data = data["summary"]
                    elif name == "port_trace":
                        data = {"insertion_event_count": data["insertion_event_count"],
                                "final_entrance_distance_mm": data["final"]["entrance_distance_mm"]}
                    row[name] = data
                    row[name + "_sha256"] = sha(path)
            bags = list((run / "results").glob("bag_trial_*/*.mcap"))
            row["scored_mcap"] = [{"path": str(b), "bytes": b.stat().st_size, "sha256": sha(b)} for b in bags]
        if video_name:
            video = video_root / video_name / "all_views_20fps.mp4"
            row["review_video"] = {"path": str(video.relative_to(root)), "bytes": video.stat().st_size,
                                   "sha256": sha(video)} if video.is_file() else None
        rows.append(row)
    summary = {
        "schema": "aic_moveit_sc_pilot_summary/v1",
        "fixed_scene": "five NIC cards, all-left, SC port 0, seed 51500",
        "rootless_docker": True, "gazebo_gpu": 3,
        "teacher_only": True, "insertion_handoff_enabled": False,
        "route_quality_gate": "failed_or_indeterminate_cable_clearance",
        "bc_admitted": False, "final_scenes_opened": False,
        "runs": rows,
        "code_sha256": {name: sha(root / name) for name in (
            "aic_teacher_official/aic_teacher_official/expert_generator/collision_scene.py",
            "aic_teacher_official/aic_teacher_official/expert_generator/moveit_py_backend.py",
            "scripts/plan_moveit_sc_fixed_route.py",
            "scripts/run_moveit_sc_route_pilot.sh",
            "scripts/analyze_cable_clearance_exact.py",
        )},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"runs": len(rows), "scored": sum(row["scored"] for row in rows),
                      "gate": summary["route_quality_gate"]}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
