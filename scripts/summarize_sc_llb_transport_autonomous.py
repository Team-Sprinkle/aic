#!/usr/bin/env python3
"""Summarize six fixed-scene observation-only SC transport comparisons."""

import argparse
import hashlib
import json
from pathlib import Path


ARMS = ("clean", "warm", "motion_safe", "elapsed")
CASES = ("port0_two", "port1_three")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--tag", default="", help="Optional name component between arm and case")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = []
    for case in CASES:
        controls = []
        for arm in ARMS:
            root = a.root / (f"{arm}_{a.tag}_{case}" if a.tag else f"{arm}_{case}")
            summary = json.loads((root / "autonomous_summary.json").read_text())
            launch = json.loads((root / "eval_manifest.json").read_text())
            if (summary["geometry_live_input"] or summary["teacher_or_planner_live"] or
                    summary["final_scene_opened"] or launch["diagnostic_teacher_or_geometry"]):
                raise ValueError(f"Privileged or final rollout: {root}")
            scene_sha = digest(root / "world_audit.sdf")
            config_sha = digest(root / "eval_config.yaml")
            bridge_sha = digest(root / "bridge_audit.yaml")
            controls.append((scene_sha, config_sha, bridge_sha, launch["stiffness"], launch["damping"],
                             launch["simulation_budget_s"], launch["max_translation_delta_m"]))
            rows.append({"arm": arm, "case": case, "audit": str(root.resolve()),
                         "max_translation_delta_m": launch["max_translation_delta_m"],
                         "checkpoint": launch["checkpoint"],
                         "checkpoint_sha256": digest(Path(launch["checkpoint"])),
                         "tier3_score": summary["official_tier3"]["score"],
                         "official_total": summary["official_total"],
                         "insertion_detected": summary["official_tier3"]["score"] >= 75,
                         "final_standoff_error_mm": summary["final_planned_standoff_error_mm"],
                         "minimum_standoff_error_mm": summary["min_planned_standoff_error_mm"],
                         "final_standoff_orientation_deg": summary["final_standoff_orientation_error_deg"],
                         "raw_wrist_force_max_n": summary["raw_wrist_force_max_n"],
                         "inference_ms": summary["inference_ms"],
                         "video": str((root / "video/all_views_20fps.mp4").resolve())
                         if (root / "video/all_views_20fps.mp4").exists() else None})
        if controls[1:] != controls[:-1]:
            raise ValueError(f"Scene or controller differs between arms for {case}")
    report = {"schema": "aic_sc_llb_autonomous_transport_comparison/v1", "rollouts": rows,
              "same_scene_controller_and_budget_per_case": True,
              "observation_only": True, "reserved_final_opened": False,
              "note": "Minimum standoff is posthoc geometry analysis, not the actor's input or insertion success."}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"rollouts": len(rows), "insertions": sum(x["insertion_detected"] for x in rows)}))


if __name__ == "__main__":
    main()
