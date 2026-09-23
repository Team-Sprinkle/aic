#!/usr/bin/env python3
"""Inventory retained full-insertion SC agent/VLM replays and force stalls."""

import collections
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
import yaml


def main(root, output):
    records = []
    for score_file in root.glob("**/replay_attempts/**/scoring.yaml"):
        if "stop_near_gate" in str(score_file):
            continue
        attempt = score_file.parents[2]
        scoring = yaml.safe_load(score_file.read_text())
        trial = next(value for key, value in scoring.items() if key != "total")
        data = attempt / "dataset/data/chunk-000/file-000.parquet"
        row = {
            "attempt": str(attempt), "run": attempt.parents[2].name,
            "score": float(scoring["total"]),
            "tier3_message": trial["tier_3"]["message"],
            "off_limit_contact": trial["tier_2"]["categories"]["contacts"]["message"],
            "has_video": (attempt / "dataset/videos/observation.images.center_camera/chunk-000/file-000.mp4").exists(),
            "has_state_data": data.exists(),
            "force_peak_n": None, "one_second_mid_stall_candidates": [],
        }
        if data.exists():
            table = pq.read_table(data, columns=["observation.state", "timestamp"])
            state = np.asarray(table["observation.state"].to_pylist(), dtype=float)
            times = np.asarray(table["timestamp"].to_pylist(), dtype=float)
            force = np.linalg.norm(state[:, 26:29], axis=1)
            position = state[:, :3]
            row["force_peak_n"] = float(force.max())
            for start in np.arange(0.1 * times[-1], 0.8 * times[-1] - 1, 0.5):
                i = np.searchsorted(times, start)
                j = np.searchsorted(times, start + 1)
                if j >= len(times):
                    break
                peak = float(force[i:j].max())
                motion_mm = float(np.linalg.norm(position[j] - position[i]) * 1000)
                if peak > 30 and motion_mm < 3:
                    row["one_second_mid_stall_candidates"].append({
                        "start_s": round(float(start), 2),
                        "force_peak_n": round(peak, 2),
                        "measured_tcp_displacement_mm": round(motion_mm, 2),
                    })
        records.append(row)
    records.sort(key=lambda row: row["attempt"])
    summary = {
        "schema": "aic_sc_agent_replay_failure_scan/v1",
        "scope": "retained SC agent/VLM replay_attempts with full-insertion collection intent; explicit stop-near-gate runs excluded",
        "criterion": "A screen, not a snag label: score <= 1; in the middle 10-80% of a recorded episode, a one-second window has peak force >30 N and measured TCP displacement <3 mm. Overlapping windows sampled every 0.5 s.",
        "attempt_count": len(records),
        "low_score_count": sum(row["score"] <= 1 for row in records),
        "low_score_stall_candidate_count": sum(row["score"] <= 1 and bool(row["one_second_mid_stall_candidates"]) for row in records),
        "contact_messages": dict(collections.Counter(row["off_limit_contact"] for row in records if "Contacts detected" in row["off_limit_contact"])),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output, summary["attempt_count"], summary["low_score_count"], summary["low_score_stall_candidate_count"])


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
