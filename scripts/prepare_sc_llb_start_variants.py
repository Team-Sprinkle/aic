#!/usr/bin/env python3
"""Make requested robot-start perturbations effective in Gazebo.

The official one-trial engine reads robot.home_joint_positions but only calls
ResetJoints after the trial. The first trial instead uses fixed xacro control
initial positions. This helper creates a per-case xacro and a matching MoveIt
planning start. It refuses to touch any case with an existing plan or replay.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "aic_description/urdf/ur_gz.urdf.xacro"
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
BASE_CONFIG = dict(shoulder_pan_joint=-0.1597, shoulder_lift_joint=-1.3542,
                   elbow_joint=-1.6648, wrist_1_joint=-1.6933,
                   wrist_2_joint=1.5710, wrist_3_joint=1.4110)
# First scored reference measurement. Its elbow/wrist-1 numbers differ from
# the xacro dictionary, but scored varied-start bags show each joint follows
# its own xacro offset. This prediction must still be audited per scored bag.
BASE_MEASURED = dict(shoulder_pan_joint=-0.15803, shoulder_lift_joint=-1.35146,
                     elbow_joint=-1.69620, wrist_1_joint=-1.66472,
                     wrist_2_joint=1.57073, wrist_3_joint=1.41262)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--min-index", type=int, default=0)
    args = parser.parse_args()
    root = args.collection.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    source = SOURCE.read_text()
    rows = []
    for case in manifest["cases"]:
        if case["index"] < args.min_index:
            continue
        name = case["name"]
        scene = root / "scenes" / name
        if any((root / "plans" / (name + suffix)).exists() for suffix in ("", "_held", "_insert")) or (root / "runs" / name).exists():
            raise RuntimeError(f"Refuse to alter already planned or replayed case: {name}")
        configured = case["robot_home_joint_positions"]
        contents = source
        for joint in JOINTS:
            pattern = rf"({re.escape(joint)}=)(-?[0-9]+\.[0-9]+)"
            contents, count = re.subn(pattern, rf"\g<1>{configured[joint]:.5f}", contents, count=1)
            if count != 1:
                raise RuntimeError(f"Missing xacro initial position for {joint}")
        xacro = scene / "ur_llb_start.xacro"
        xacro.write_text(contents)
        predicted = dict(BASE_MEASURED)
        for joint in JOINTS:
            predicted[joint] += configured[joint] - BASE_CONFIG[joint]
        joint_state = {"name": list(JOINTS),
                       "position": [predicted[joint] for joint in JOINTS],
                       "source": "predicted Gazebo first-trial start from scored baseline plus per-case xacro offsets; audit measured bag",}
        (scene / "measured_start_joint_state.json").write_text(json.dumps(joint_state, indent=2) + "\n")
        row = {"name": name, "index": case["index"], "configured_xacro_joints": configured,
               "predicted_measured_start_joints": predicted, "xacro": str(xacro),
               "xacro_sha256": sha(xacro), "planner_start_sha256": sha(scene / "measured_start_joint_state.json"),
               "measured_validation": "pending_scored_bag"}
        rows.append(row)
    (root / "start_variant_manifest.json").write_text(json.dumps({
        "schema": "aic_sc_llb_start_variants/v1", "source_xacro": str(SOURCE),
        "reason": "first-trial start is fixed by robot description, not robot.home_joint_positions",
        "cases": rows, "scored_start_audit_required": True,
    }, indent=2) + "\n")
    print(json.dumps({"prepared": len(rows), "manifest": str(root / "start_variant_manifest.json")}))


if __name__ == "__main__":
    main()
