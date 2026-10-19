#!/usr/bin/env python3
"""Posthoc, same-reset teacher/actor transport diagnosis. Never actor input."""
import argparse
import json
from pathlib import Path

import numpy as np


def interp(t, xyz, query):
    return np.column_stack([np.interp(query, t, xyz[:, axis]) for axis in range(3)])


def teacher_rows(path):
    data = json.loads((path / "scored_stream.json").read_text())
    states = data["streams"]["/aic_controller/controller_state"]
    t = np.array([stamp / 1e9 for stamp, _ in states])
    xyz = np.array([[row["tcp_pose"]["position"][axis] for axis in "xyz"] for _, row in states])
    commands = [(row["sim_ns"] / 1e9,
                 np.array([row["message"]["pose"]["position"][axis] for axis in "xyz"]))
                for row in data["commands"] if row["kind"] == "pose"]
    return t, xyz, commands, data["task"]


def actor_rows(path):
    rows = [json.loads(line) for line in (path / "runtime_trace.jsonl").open()]
    proposals = [row for row in rows if row["event"] == "proposal"]
    commands = [row for row in rows if row["event"] == "command"]
    t = np.array([row["sim_time"] for row in proposals])
    xyz = np.array([row["state"][:3] for row in proposals])
    return t, xyz, proposals, commands, rows


def sample_command(commands, time):
    past = [v for stamp, v in commands if stamp <= time]
    return past[-1] if past else np.zeros(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.summary.read_text())
    result = []
    for entry in source["source"]["runs"]:
        if not str(entry["run"]).endswith("_candidate"):
            continue
        at, ax, proposals, commands, events = actor_rows(Path(entry["run"]))
        tt, tx, teacher_commands, task = teacher_rows(Path(entry["teacher"]))
        lo = max(at.min(), tt.min())
        hi = min(at.max(), tt.max(), entry["teacher_cutoff_time_s"])
        keep = (at >= lo) & (at <= hi)
        times = at[keep]
        observed = ax[keep]
        reference = interp(tt, tx, times)
        error = np.linalg.norm(observed - reference, axis=1) * 1000
        route_keep = tt <= entry["teacher_cutoff_time_s"]
        route_t = tt[route_keep]
        route_x = tx[route_keep]
        route_distances = np.linalg.norm(ax[:,None,:]-route_x[None,:,:],axis=-1)
        nearest_route_mm = route_distances.min(axis=1)*1000
        nearest_teacher_time = route_t[route_distances.argmin(axis=1)]
        first = {}
        for threshold in (5, 10, 20, 30):
            hit = np.flatnonzero(error > threshold)
            first[str(threshold)] = float(times[hit[0]]) if len(hit) else None
        first_route = {}
        for threshold in (5,10,20,30):
            hit = np.flatnonzero(nearest_route_mm > threshold)
            first_route[str(threshold)] = float(at[hit[0]]) if len(hit) else None
        samples = []
        for seconds in (3.5, 4, 5, 6, 8, 10, 12, 15):
            if seconds < lo or seconds > hi:
                continue
            actor_xyz = interp(at, ax, [seconds])[0]
            teacher_xyz = interp(tt, tx, [seconds])[0]
            nearest = min(proposals, key=lambda row: abs(row["sim_time"] - seconds))
            ac = np.array(nearest["targets"][0][:3])
            tc = sample_command(teacher_commands, seconds)
            samples.append(dict(time_s=seconds, actor_xyz_m=actor_xyz.tolist(),
                                teacher_xyz_m=teacher_xyz.tolist(),
                                measured_error_mm=float(np.linalg.norm(actor_xyz-teacher_xyz)*1000),
                                actor_first_tcp_command_mm=(ac*1000).tolist(),
                                teacher_last_tcp_command_mm=(tc*1000).tolist(),
                                actor_image_age_ms={camera:(nearest["sim_time"]*1e9-stamp)/1e6
                                                    for camera, stamp in nearest["image_sim_ns"].items()},
                                nearest_teacher_route_mm=float(nearest_route_mm[np.argmin(abs(at-seconds))]),
                                nearest_teacher_route_time_s=float(nearest_teacher_time[np.argmin(abs(at-seconds))])))
        tracking = []
        for row in commands:
            time = row["sim_time"]
            if time < at.min() or time + .2 > at.max():
                continue
            delta = np.array(row["absolute_tcp_target"])[:3, 3] - np.array(row["state"][:3])
            motion = interp(at, ax, [time+.2])[0] - np.array(row["state"][:3])
            if np.linalg.norm(delta) >= .002 and np.linalg.norm(motion)>=.001:
                tracking.append(float(np.dot(delta, motion)/(np.linalg.norm(delta)*max(np.linalg.norm(motion),1e-9))))
        result.append(dict(run=entry["run"], teacher=entry["teacher"], task=task,
                           actor_request=proposals[0]["task_vector"],
                           matched_time_range_s=[float(lo),float(hi)],
                           first_measured_divergence_s=first,
                           first_progress_independent_route_divergence_s=first_route,
                           samples=samples,
                           command_motion_direction_cosine_median=float(np.median(tracking)) if tracking else None,
                           stale_chunk_drops=sum(row["event"]=="stale_chunk_member_dropped" for row in events),
                           actor_proposals=len(proposals), teacher_commands=len(teacher_commands),
                           note="Teacher and actor share reset and simulation clock; command directions are TCP-frame, measured positions are base-frame."))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema":"aic_sfp_paired_first_divergence/v1","runs":result,
                                        "geometry_actor_input":False,"final_scenes_opened":False},indent=2)+"\n")
    print(json.dumps([{k:r[k] for k in ("run","first_measured_divergence_s","command_motion_direction_cosine_median","stale_chunk_drops")} for r in result],indent=2))


if __name__ == "__main__":
    main()
