#!/usr/bin/env python3
"""Locate scored bags for canonical SC experts and retain their causal port TF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_sc_port_tf import extract_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-prefix", required=True, type=Path,
                        help="Host prefix in the manifest, replaced by --mounted-prefix inside Docker")
    parser.add_argument("--mounted-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text())
    reports = []
    for row in payload["episodes"]:
        if (row.get("task") or {}).get("task_family") != "sc_to_sc":
            continue
        score_host = Path(row["official_score_path"])
        score_mounted = args.mounted_prefix / score_host.relative_to(args.source_prefix)
        bag_candidates = sorted(score_mounted.parent.glob("bag_" + row["trial_id"] + "_*"))
        if len(bag_candidates) != 1:
            raise RuntimeError(f"Expected one scored bag for episode {row['episode_index']}: {bag_candidates}")
        bag = bag_candidates[0]
        edges = extract_edges(bag, first_seconds=5)
        needed = {"tabletop", "base_link", "task_board",
                  f"task_board/sc_port_{row['task']['target_port_index']}",
                  f"task_board/sc_port_{row['task']['target_port_index']}/sc_port_base_link"}
        available = {edge["child"] for edge in edges}
        if not needed <= available:
            raise RuntimeError(f"Episode {row['episode_index']} missing TF edges: {sorted(needed - available)}")
        reports.append({
            "episode_index": row["episode_index"], "trial_id": row["trial_id"],
            "split": row["split"], "nic_count": row["nic_count"],
            "target_port_index": row["task"]["target_port_index"],
            "scene_sha256": row["scene_sha256"],
            "bag_host": str(args.source_prefix / bag.relative_to(args.mounted_prefix)),
            "edges": [edge for edge in edges if edge["child"] in needed],
        })
        print(json.dumps({"episode": row["episode_index"], "bag": str(bag)}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "aic_sc_expert_port_edges/v1", "episodes": reports}, indent=2) + "\n")
    print(json.dumps({"episodes": len(reports), "output": str(args.output)}))


if __name__ == "__main__":
    main()
