#!/usr/bin/env python3
"""Read-only route and motion diagnosis for a matched SFP live gate."""
import argparse
import json
from pathlib import Path

import numpy as np


def actor_rows(path):
    return [r for line in (path / "runtime_trace.jsonl").open()
            if (r := json.loads(line)).get("event") == "proposal"]


def teacher_route(path):
    stream = json.loads((path / "scored_stream.json").read_text())
    rows = stream["streams"]["/aic_controller/controller_state"]
    return np.asarray([t / 1e9 for t, _ in rows]), np.asarray([
        [v["tcp_pose"]["position"][k] for k in "xyz"] for _, v in rows])


def nearest_path(xyz, teacher_xyz):
    # The teacher may run at another speed. Compare path geometry, not equal time.
    delta = xyz[:, None, :] - teacher_xyz[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    nearest = distance.argmin(axis=1)
    return distance[np.arange(len(xyz)), nearest] * 1000, nearest


def one_run(item):
    run, teacher = Path(item["run"]), Path(item["teacher"])
    rows = actor_rows(run)
    times = np.asarray([r["sim_time"] for r in rows])
    xyz = np.asarray([r["state"][:3] for r in rows])
    cmd = np.asarray([r["targets"][0][:3] for r in rows]) * 1000
    tt, tx = teacher_route(teacher)
    # Avoid the common stationary reset prefix when finding nearest route points.
    moving = np.flatnonzero(np.linalg.norm(tx - tx[0], axis=1) > .003)
    tx = tx[moving[0]:] if len(moving) else tx
    tt = tt[moving[0]:] if len(moving) else tt
    distance, nearest = nearest_path(xyz, tx[::5])
    nearest_time = tt[::5][nearest]
    moved = np.r_[np.nan, np.linalg.norm(np.diff(xyz, axis=0), axis=1) * 1000]
    command = np.linalg.norm(cmd, axis=1)
    # Sustained low movement despite a sizable demanded first chunk; descriptive,
    # not an attribution of the cause to contact or controller behavior.
    stalled = (command >= 10) & (moved < .5)
    windows = []
    for lo, hi in ((4, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 60)):
        mask = (times >= lo) & (times < hi)
        if mask.any():
            windows.append(dict(seconds=[lo, hi], rows=int(mask.sum()),
                                route_distance_median_mm=float(np.median(distance[mask])),
                                route_distance_p95_mm=float(np.percentile(distance[mask], 95)),
                                command_median_mm=float(np.median(command[mask])),
                                measured_step_median_mm=float(np.nanmedian(moved[mask])),
                                stalled_fraction=float(np.mean(stalled[mask])),
                                teacher_progress_median_s=float(np.median(nearest_time[mask]))))
    return dict(run=str(run), teacher=str(teacher), variant=run.name.rsplit("_", 1)[-1],
                scene=run.name.rsplit("_", 1)[0], windows=windows,
                first_path_error_gt_10mm_s=float(times[np.flatnonzero(distance > 10)[0]])
                if np.any(distance > 10) else None,
                first_path_error_gt_25mm_s=float(times[np.flatnonzero(distance > 25)[0]])
                if np.any(distance > 25) else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--posthoc", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve previous diagnosis")
    posthoc = json.loads(a.posthoc.read_text())
    runs = [one_run(item) for item in posthoc["runs"]]
    a.output.write_text(json.dumps(dict(schema="aic_sfp_paired_actor_live_diagnosis/v1",
                                        posthoc=str(a.posthoc),
                                        measured_tcp_and_teacher_route_only=True,
                                        no_geometry_actor_input=True,
                                        final_scenes_opened=False, runs=runs), indent=2) + "\n")
    for r in runs:
        print(r["scene"], r["variant"],
              [(w["seconds"], round(w["route_distance_median_mm"], 1),
                round(w["stalled_fraction"], 2)) for w in r["windows"]])


if __name__ == "__main__":
    main()
