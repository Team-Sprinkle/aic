#!/usr/bin/env python3
"""Correct unplanned LLB MoveIt starts using scored per-joint xacro response."""

import argparse
import json
from pathlib import Path

from prepare_sc_llb_start_variants import BASE_CONFIG, BASE_MEASURED, JOINTS, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--min-index", type=int, required=True)
    args = parser.parse_args()
    root = args.collection.resolve()
    manifest_path = root / "start_variant_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_name = {case["name"]: case for case in json.loads((root / "manifest.json").read_text())["cases"]}
    repaired = []
    for row in manifest["cases"]:
        if row["index"] < args.min_index:
            continue
        name = row["name"]
        if any((root / "plans" / (name + suffix)).exists() for suffix in ("", "_held", "_insert")) or (root / "runs" / name).exists():
            raise RuntimeError(f"Refuse to change planned/replayed start: {name}")
        scene = root / "scenes" / name
        configured = by_name[name]["robot_home_joint_positions"]
        predicted = {joint: BASE_MEASURED[joint] + configured[joint] - BASE_CONFIG[joint]
                     for joint in JOINTS}
        joint_state = {"name": list(JOINTS), "position": [predicted[joint] for joint in JOINTS],
                       "source": "scored-bag calibrated direct per-joint xacro offsets from llb_0030/0031; audit every scored bag"}
        path = scene / "measured_start_joint_state.json"
        path.write_text(json.dumps(joint_state, indent=2) + "\n")
        row["predicted_measured_start_joints"] = predicted
        row["planner_start_sha256"] = sha(path)
        row["prediction_calibration"] = "direct per-joint offset from scored llb_0030 and llb_0031 bags"
        repaired.append(name)
    manifest["prediction_calibration"] = "direct per-joint offset from scored llb_0030 and llb_0031 bags"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = {"schema": "aic_sc_llb_start_repair/v1", "repaired": len(repaired),
              "first": repaired[0] if repaired else None, "last": repaired[-1] if repaired else None,
              "source_bags": ["llb_0030", "llb_0031"], "active_jobs_interrupted": False}
    (root / "start_prediction_repair.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
