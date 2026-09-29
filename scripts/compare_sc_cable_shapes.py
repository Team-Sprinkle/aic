#!/usr/bin/env python3
"""Compare cable centerline geometry in scored Gazebo bags and Isaac probes.

The source episodes are not time/action matched. These are shape diagnostics,
not a simulator-fidelity score or a policy evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

def norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def metrics(points: list[list[float]]) -> dict:
    segments = [[b - a for a, b in zip(p, q)] for p, q in zip(points, points[1:])]
    lengths = [norm(v) for v in segments]
    turns = []
    for a, b, la, lb in zip(segments, segments[1:], lengths, lengths[1:]):
        cosine = sum(x * y for x, y in zip(a, b)) / max(la * lb, 1e-12)
        turns.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    chord = norm([b - a for a, b in zip(points[0], points[-1])])
    return {
        "median_segment_mm": statistics.median(lengths) * 1000,
        "min_segment_mm": min(lengths) * 1000,
        "max_segment_mm": max(lengths) * 1000,
        "max_turn_deg": max(turns),
        "turns_over_90_deg": sum(x > 90 for x in turns),
        "arc_to_chord": sum(lengths) / chord if chord > 1e-9 else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gazebo", type=Path, required=True)
    p.add_argument("--isaac", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    gaz = json.loads(a.gazebo.read_text())
    isa = json.loads(a.isaac.read_text())
    result = {
        "caveat": "Different scene geometry, times, and commands; compare only structural plausibility.",
        "gazebo_source": str(a.gazebo),
        "isaac_source": str(a.isaac),
        "gazebo": [],
        "isaac": [],
    }
    for sample in gaz["samples"]:
        tf = sample["transforms"]
        xyz = [tf[f"cable_1/link_{i}"]["xyz_m"] for i in range(1, 21)]
        result["gazebo"].append({"elapsed_s": sample["actual_elapsed_s"], **metrics(xyz)})
    for row in isa["rows"]:
        poses = row.get("debug_rope_pose_world")
        if not poses:
            continue
        xyz = [poses[f"link_{i}"]["xyz_m"] for i in range(1, 21)]
        result["isaac"].append({"step": row["step"], **metrics(xyz)})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.output)


if __name__ == "__main__":
    main()
