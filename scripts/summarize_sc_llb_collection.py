#!/usr/bin/env python3
"""Summarize scored LLB attempts and their scene coverage without admitting BC data."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import yaml


def read_collection(root):
    manifest = json.loads((root / "manifest.json").read_text())
    progress_path = root / "collection_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"attempts": {}}
    rows = []
    for case in manifest["cases"]:
        name = case["name"]
        outcome = progress["attempts"].get(name)
        if outcome is None:
            continue
        audit = root / "runs" / name
        score_path = audit / "results/scoring.yaml"
        actual_score = None
        model_validation_score = None
        if score_path.exists():
            score = yaml.safe_load(score_path.read_text())
            trial = next((value for key, value in score.items() if key.startswith("trial_")), None)
            actual_score = float(trial["tier_3"]["score"]) if trial else None
            model_validation_score = float(trial["tier_1"]["score"]) if trial else None
        bag_metadata = sorted((audit / "results").glob("bag_*/metadata.yaml"))
        if outcome.get("score_available") and actual_score != outcome.get("tier3"):
            raise AssertionError(f"Score mismatch for {name}")
        rows.append({
            "case": name, "selected_port": case["selected_port"],
            "card_count": case["card_count"],
            "board_pose": case["board_pose"],
            "robot_home_joint_positions": case["robot_home_joint_positions"],
            "scene_sha256": case["scene_sha256"],
            "moveit_plan_exists": (root / "plans" / (name + "_insert") / "moveit_joint_trajectory.json").exists(),
            "scored": actual_score is not None,
            "has_scored_bag": bool(bag_metadata),
            "model_validation_passed": model_validation_score is not None and model_validation_score > 0,
            "tier3": actual_score,
            "full_insertion": actual_score is not None and actual_score >= 75,
            "audit": str(audit),
            "archive": outcome.get("archive"),
            "reason": outcome.get("reason"),
        })
    return manifest, progress, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    roots = [root.resolve() for root in args.roots]
    collections = [read_collection(root) for root in roots]
    cases = [case for manifest, _, _ in collections for case in manifest["cases"]]
    rows = [row for _, _, group in collections for row in group]
    scene_hashes = [case["scene_sha256"] for case in cases]
    if len(set(scene_hashes)) != len(scene_hashes):
        raise AssertionError("Duplicate scene YAML across requested settings")
    by_count = defaultdict(Counter)
    by_port = defaultdict(Counter)
    for row in rows:
        by_count[row["card_count"]]["attempted"] += 1
        by_count[row["card_count"]]["full_insertion"] += int(row["full_insertion"])
        by_port[row["selected_port"]]["attempted"] += 1
        by_port[row["selected_port"]]["full_insertion"] += int(row["full_insertion"])
    output = {
        "schema": "aic_sc_llb_collection_summary/v1",
        "route": "Low Lateral Bypass (LLB)",
        "requested": len(cases), "attempted": len(rows),
        "scored": sum(row["scored"] for row in rows),
        "scored_bags": sum(row["has_scored_bag"] for row in rows),
        "model_validation_passed": sum(row["model_validation_passed"] for row in rows),
        "full_insertions": sum(row["full_insertion"] for row in rows),
        "planned": sum(row["moveit_plan_exists"] for row in rows),
        "by_card_count": {str(key): dict(value) for key, value in sorted(by_count.items())},
        "by_selected_port": {str(key): dict(value) for key, value in sorted(by_port.items())},
        "unique_requested_scene_configs": len(set(scene_hashes)),
        "pilot_gate": collections[0][1].get("pilot_gate"),
        "final_scenes_opened": False,
        "bc_training_started": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in
                      ("requested", "attempted", "scored", "full_insertions", "planned", "pilot_gate")}))


if __name__ == "__main__":
    main()
