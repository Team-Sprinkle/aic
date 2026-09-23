#!/usr/bin/env python3
"""Separate verified noninsertions from unavailable historical SC lineage.

Also triage mid-episode wrist-force peaks. A force peak alone is not a cable
snag label because the archive has no retained collider-specific contact trace.
"""

import collections
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
ARCHIVE = REPO / "outputs/trajectory_datasets/clean_including_no_insert_trajs"
EXCLUDED = REPO / "outputs/trajectory_datasets/expert_verified/excluded_episodes.json"


def main(out):
    rows = json.loads(EXCLUDED.read_text())
    reason_counts = dict(collections.Counter(row["exclusion_reason"] for row in rows))
    sources = []
    for source in sorted({row["source"] for row in rows}):
        subset = [row for row in rows if row["source"] == source]
        sources.append({
            "source": source,
            "count": len(subset),
            "reason_counts": dict(collections.Counter(row["exclusion_reason"] for row in subset)),
            "official_scoring_available": sum(bool(row["official_scores"]) for row in subset),
            "raw_dataset_available": sum(bool(row["raw_available"]) for row in subset),
            "near_gate_collection": "stop_near_gate" in source,
        })

    force_groups = []
    for card_count in (1, 2, 3):
        source = f"sc_to_sc/agent/sc_ports_1/n100__sc_ports1_nic{card_count}_stop_near_gate_05mm_n100"
        data = ARCHIVE / source / "accepted_dataset/data/chunk-000/file-000.parquet"
        columns = pq.read_table(data, columns=["episode_index", "observation.state", "timestamp"]).to_pydict()
        episodes = np.asarray(columns["episode_index"])
        states = np.asarray(columns["observation.state"])
        times = np.asarray(columns["timestamp"])
        force = np.linalg.norm(states[:, 26:29], axis=1)
        peaks = []
        for episode in np.unique(episodes):
            mask = episodes == episode
            values = force[mask]
            lo, hi = int(.1 * len(values)), int(.8 * len(values))
            index = lo + int(np.argmax(values[lo:hi]))
            peaks.append({"episode_index": int(episode), "peak_n": round(float(values[index]), 3),
                          "time_s": round(float(times[mask][index]), 3)})
        force_groups.append({
            "card_count": card_count,
            "episode_count": len(peaks),
            "middle_10_to_80pct_over_30n": sum(row["peak_n"] > 30 for row in peaks),
            "middle_10_to_80pct_over_40n": sum(row["peak_n"] > 40 for row in peaks),
            "median_peak_n": round(float(np.median([row["peak_n"] for row in peaks])), 3),
            "top_peaks": sorted(peaks, key=lambda row: row["peak_n"], reverse=True)[:5],
        })

    summary = {
        "schema": "aic_historical_sc_cable_audit/v1",
        "excluded_total": len(rows),
        "excluded_manifest_sha256": hashlib.sha256(EXCLUDED.read_bytes()).hexdigest(),
        "reason_counts": reason_counts,
        "scored_noninsertions_were_near_gate_collection": all(
            "stop_near_gate" in row["source"] for row in rows
            if row["exclusion_reason"] == "scored_noninsertion"
        ),
        "sources": sources,
        "force_groups": force_groups,
        "interpretation": "Force peaks and cable/card proximity are candidates only; no collider-specific contact trace or stock-CheatCode outcome exists for these archived episodes.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(out, reason_counts, [(r["card_count"], r["middle_10_to_80pct_over_30n"]) for r in force_groups])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
