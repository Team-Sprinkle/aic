#!/usr/bin/env python3
"""Create a compact summary for the targeted SC failure reproduction suite."""

import json
import math
from pathlib import Path

import yaml


ROOT = Path("/audit")


def outcome(tier3):
    score = float(tier3["score"])
    message = tier3["message"].lower()
    if "successful" in message:
        return "full"
    if "partial" in message:
        return "partial"
    return "none"


def main():
    scoring = yaml.safe_load((ROOT / "results/scoring.yaml").read_text())
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = []
    for item in manifest["trials"]:
        name = item["trial"]
        result = scoring[name]
        matches = sorted((ROOT / "analysis").glob(f"bag_{name}_*.json"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one analysis for {name}, got {matches}")
        analysis = json.loads(matches[0].read_text())
        final = analysis["final"]
        xyz = final["entrance_relative_xyz_mm"]
        row = dict(item)
        row.update({
            "total_score": sum(float(v["score"]) for v in result.values()),
            "tier3_score": float(result["tier_3"]["score"]),
            "tier3_message": result["tier_3"]["message"],
            "outcome": outcome(result["tier_3"]),
            # Event counts can include transient-local history from earlier
            # trials. The official tier-3 scorer remains the outcome label.
            "raw_insertion_event_count": analysis["insertion_event_count"],
            "terminal_entrance_relative_xyz_mm": xyz,
            "terminal_lateral_mm": math.hypot(xyz[0], xyz[1]),
            "terminal_axial_mm": xyz[2],
            "terminal_orientation_deg": final["entrance_orientation_deg"],
            "terminal_tcp_tracking_error_mm": final.get("tcp_tracking_error_mm"),
            "force_norm_n": analysis["force_norm_n"],
            "minimum_entrance_distance_mm": analysis["minimum_entrance_distance_mm"],
            "bulk_analysis_file": str(matches[0]),
        })
        records.append(row)

    counts = {key: sum(r["outcome"] == key for r in records)
              for key in ("full", "partial", "none")}
    (ROOT / "summary.json").write_text(json.dumps({
        "classification": manifest["classification"],
        "outcome_label": "official tier-3 scorer",
        "insertion_event_count_caveat": (
            "Later bags can replay transient-local insertion history; do not use "
            "raw event counts as the trial outcome."
        ),
        "visual_review": {
            "five_card_sc1": "Cable remained visibly clear of the NIC card field; axial partial insertion, exact microscopic cause unresolved.",
            "five_card_sc0": "Cable remained visibly clear of the NIC card field; large TCP tracking failure, specific rigid contact versus workspace cause unresolved.",
            "confirmed_cable_snag_count": 0,
        },
        "counts": counts,
        "records": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
