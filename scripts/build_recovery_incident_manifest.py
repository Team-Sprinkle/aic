#!/usr/bin/env python3
"""Index retained Gazebo recovery incidents without inferring contact identity."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "artifacts/prod_cheatcode_audit/ordinary_broad_followup/cable_route_probe"
TARGETED = ROOT / "artifacts/prod_cheatcode_audit/targeted_failure_reproduction"
TARGETED_BAGS = Path("/var/tmp/chmin_aic_targeted_failure_20260923/results")
EXPERT = ROOT / "outputs/trajectory_datasets/expert_verified/manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(output: Path) -> None:
    sources = {
        "route": ROUTE / "summary.json",
        "wide": ROUTE / "smooth_wide_summary.json",
        "targeted": TARGETED / "summary.json",
        "experts": EXPERT,
    }
    data = {name: json.loads(path.read_text()) for name, path in sources.items()}
    incidents = []
    for row in data["route"]["records"]:
        if row["label"] == "stock_control":
            group = "fixed_seed_51500_stock"
        else:
            group = "fixed_seed_51500_routed"
        incidents.append({
            "id": "route_" + row["label"],
            "scene_group": group,
            "source": "fixed_five_card_route_probe",
            "outcome": row["outcome"],
            "official_tier3": row["tier3_score"],
            "peak_wrist_force_n": row.get("wrist_force_peak_n"),
            "terminal_axial_mm": row.get("terminal_axial_mm"),
            "terminal_lateral_mm": row.get("terminal_lateral_mm"),
            "terminal_tcp_tracking_error_mm": row.get("terminal_tcp_command_error_mm"),
            "bulk_root": row["bulk_root"],
            "compact_root": str(ROUTE / row["label"]),
            "failure_class": "ambiguous_approach_obstruction" if row["outcome"] == "none" else "control_or_partial",
            "named_contact_pair_observed": False,
        })
    for row in data["wide"]["records"]:
        if not row.get("complete_task_camera_coverage"):
            continue
        name = row["run"]
        incidents.append({
            "id": "wide_" + name,
            "scene_group": "fixed_seed_51500_routed",
            "source": "fixed_five_card_20hz_repeat",
            "outcome": row["outcome"],
            "official_tier3": row["tier3_score"],
            "peak_wrist_force_n": row.get("peak_wrist_force_n"),
            "terminal_axial_mm": row.get("terminal_axial_mm"),
            "terminal_lateral_mm": row.get("terminal_lateral_mm"),
            "terminal_tcp_tracking_error_mm": row.get("terminal_tcp_command_error_mm"),
            "bulk_root": row["bulk_root"],
            "bag": row.get("bag"),
            "compact_root": str(ROUTE / ("smooth_wide_repeat_" + name.rsplit("_", 1)[-1])),
            "failure_class": "ambiguous_approach_obstruction" if row["outcome"] == "none" else "control_or_partial",
            "named_contact_pair_observed": False,
        })
    for row in data["targeted"]["records"]:
        bags = sorted(TARGETED_BAGS.glob("bag_" + row["trial"] + "_*"))
        incidents.append({
            "id": "targeted_" + row["trial"],
            "scene_group": "targeted_" + row["source"] + "_" + str(row.get("card_rails")),
            "source": "targeted_postfix_gazebo",
            "outcome": row["outcome"],
            "official_tier3": row["tier3_score"],
            "terminal_axial_mm": row.get("terminal_axial_mm"),
            "terminal_lateral_mm": row.get("terminal_lateral_mm"),
            "terminal_tcp_tracking_error_mm": row.get("terminal_tcp_tracking_error_mm"),
            "bulk_analysis_file": row["bulk_analysis_file"],
            "bag": str(bags[0]) if len(bags) == 1 else None,
            "failure_class": (
                "large_approach_blockage" if row["outcome"] == "none"
                else "local_axial_partial" if row["outcome"] == "partial"
                else "control_full"
            ),
            "named_contact_pair_observed": False,
        })
    sc = [row for row in data["experts"]["episodes"] if
          (row.get("task") or {}).get("task_family") == "sc_to_sc"]
    report = {
        "schema": "aic_recovery_incident_index/v1",
        "source_sha256": {name: digest(path) for name, path in sources.items()},
        "important_limit": "Diagnostic teacher episodes are indexed, not converted into causal actor replay. Ground-truth route and contact identity are not deployment inputs.",
        "official_image": data["route"]["image"],
        "gazebo_incidents": incidents,
        "counts_by_outcome": dict(collections.Counter(row["outcome"] for row in incidents)),
        "canonical_sc_experts": {
            "episodes": len(sc),
            "split": dict(collections.Counter(row["split"] for row in sc)),
            "nic_count": dict(collections.Counter(row.get("nic_count") for row in sc)),
            "official_tier3": dict(collections.Counter(str(row.get("official_tier3")) for row in sc)),
            "label_status": list(sorted(set(row.get("label_status") for row in sc))),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "counts": report["counts_by_outcome"],
                      "sc_experts": len(sc)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
