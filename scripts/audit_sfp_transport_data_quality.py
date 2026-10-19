#!/usr/bin/env python3
"""Audit recorded SFP transport targets and measured motion by source cohort.

Read-only: does not change labels, timestamps, images, or split membership.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


def cohort(row):
    source = row.get("source_scored_stream", "")
    if not source:
        return "pre_oct6"
    for key, name in (("divergent_pairs_round2", "paired_round2"),
                      ("divergent_pairs_v1", "paired_round1"),
                      ("diverse_teacher", "diverse"),
                      ("diverse_retry", "diverse_retry")):
        if key in source:
            return name
    return "other_scored"


def quantiles(values):
    return {str(k): float(np.quantile(values, k)) for k in (.5, .9, .95, .99)}


def longest(mask, times):
    best = 0.
    start = None
    for i, on in enumerate(mask):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(mask) - 1):
            end = i if not on else i + 1
            best = max(best, times[min(end, len(times)-1)] - times[start])
            start = None
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve earlier audit")
    episodes = defaultdict(list)
    with a.labels.open() as f:
        for line in f:
            row = json.loads(line)
            observed = row["observed_tcp_pose_port_frame"][:3]
            target = row["teacher_tcp_target_poses_port_frame"][0][:3]
            episodes[(cohort(row), row["split"], int(row["episode_index"]))].append(
                (float(row["timestamp_s"]), row["state"][:3],
                 float(np.linalg.norm(np.asarray(target) - observed) * 1000),
                 row["scene_sha256"], tuple(row["task_vector"])))
    details = []
    for (source, split, episode), rows in episodes.items():
        t = np.asarray([r[0] for r in rows]); xyz = np.asarray([r[1] for r in rows])
        target_mm = np.asarray([r[2] for r in rows])
        if len(set(r[3] for r in rows)) != 1 or len(set(r[4] for r in rows)) != 1:
            raise ValueError(f"Scene/task changed within episode {episode}")
        dt = np.diff(t)
        motion_mm = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000
        low_both = (motion_mm < .1) & (target_mm[:-1] < .5)
        high_command_low_motion = (motion_mm < .1) & (target_mm[:-1] > 5.)
        details.append(dict(source=source, split=split, episode=episode,
                            scene=rows[0][3], rows=len(rows), duration_s=float(t[-1]-t[0]),
                            nonmonotonic_time=bool(np.any(dt <= 0)),
                            time_gap_over_120ms=int(np.sum(dt > .12)),
                            first_target_near_zero_fraction=float(np.mean(target_mm < .5)),
                            first_target_gt_30mm_fraction=float(np.mean(target_mm > 30)),
                            low_motion_low_command_fraction=float(np.mean(low_both)),
                            high_command_low_motion_fraction=float(np.mean(high_command_low_motion)),
                            longest_low_motion_low_command_s=float(longest(low_both, t)),
                            longest_high_command_low_motion_s=float(longest(high_command_low_motion, t)),
                            target_mm_p95=float(np.quantile(target_mm, .95))))
    grouped = defaultdict(list)
    for r in details:
        grouped[(r["source"], r["split"])].append(r)
    summary = {}
    for key, group in sorted(grouped.items()):
        summary[str(key)] = dict(episodes=len(group), rows=sum(r["rows"] for r in group),
            mean_near_zero_target_fraction=float(np.mean([r["first_target_near_zero_fraction"] for r in group])),
            mean_low_motion_low_command_fraction=float(np.mean([r["low_motion_low_command_fraction"] for r in group])),
            mean_high_command_low_motion_fraction=float(np.mean([r["high_command_low_motion_fraction"] for r in group])),
            longest_low_motion_low_command_s=quantiles([r["longest_low_motion_low_command_s"] for r in group]),
            episodes_with_low_both_over_1s=sum(r["longest_low_motion_low_command_s"] >= 1 for r in group),
            episodes_with_high_command_low_motion_over_1s=sum(r["longest_high_command_low_motion_s"] >= 1 for r in group),
            mean_target_gt_30mm_fraction=float(np.mean([r["first_target_gt_30mm_fraction"] for r in group])),
            nonmonotonic_episode_count=sum(r["nonmonotonic_time"] for r in group),
            time_gap_over_120ms_count=sum(r["time_gap_over_120ms"] for r in group))
    result = dict(schema="aic_sfp_transport_data_quality/v1", labels=str(a.labels),
                  thresholds=dict(low_motion_mm=.1, low_command_mm=.5,
                                  high_command_mm=5, long_hold_s=1),
                  summary=summary, episodes=details, labels_modified=False,
                  final_scenes_opened=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
