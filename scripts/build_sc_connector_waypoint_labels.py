#!/usr/bin/env python3
"""Build strictly timed, port-frame SC connector target chunks from scored bags.

The selected port and TCP-to-tip transforms are training labels only. Actor
inputs are images, ordinary robot state, and task encoding. This script does
not construct an autonomous pose estimator.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_sc_port_targets import matrix, pose_list, pose_matrix, select


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--port-edges", type=Path, required=True)
    p.add_argument("--tcp-tip-calibration", type=Path, required=True)
    p.add_argument("--episode-tip-calibrations", type=Path,
                   help="Scored physical TCP-to-tip transforms by episode; training labels only. Overrides the fixed transform and requires every episode to match.")
    p.add_argument("--port-opening-calibration", type=Path, required=True)
    p.add_argument("--labels-output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--step-seconds", type=float, default=0.05)
    p.add_argument("--time-tolerance-seconds", type=float, default=0.01)
    p.add_argument("--resample-mode", choices=("strict", "causal_hold"), default="strict")
    p.add_argument("--max-hold-seconds", type=float, default=0.15)
    a = p.parse_args()
    episodes = {int(row["episode_index"]): row for row in json.loads(a.manifest.read_text())["episodes"]}
    port_rows = json.loads(a.port_edges.read_text())["episodes"]
    calibration = json.loads(a.tcp_tip_calibration.read_text())["tcp_to_sc_tip"]
    tcp_tip = matrix(calibration)
    physical_by_episode = {}
    if a.episode_tip_calibrations:
        for row in json.loads(a.episode_tip_calibrations.read_text())["episodes"]:
            trial = row["trial"]
            if "episode_index" in row:
                episode_index = int(row["episode_index"])
            elif trial.startswith("expert_"):
                episode_index = int(trial.removeprefix("expert_"))
            elif trial.startswith("trial_"):
                episode_index = int(trial.removeprefix("trial_"))
            else:
                raise ValueError(f"Unknown physical grasp identity: {trial}")
            if episode_index in physical_by_episode:
                raise ValueError(f"Duplicate physical grasp episode {episode_index}")
            physical_by_episode[episode_index] = matrix({
                "xyz_m": row["samples"][0]["tcp_to_physical_sc_tip"][:3],
                "quat_xyzw": row["samples"][0]["tcp_to_physical_sc_tip"][3:7]})
    port_opening = matrix(json.loads(a.port_opening_calibration.read_text())["base_to_opening"])
    opening_port = np.linalg.inv(port_opening)
    counts = Counter()
    reports = []
    max_roundtrip_m = 0.0
    max_roundtrip_deg = 0.0
    a.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with a.labels_output.open("w") as output:
        for port in port_rows:
            episode = episodes[int(port["episode_index"])]
            tip_for_episode = (physical_by_episode[int(port["episode_index"])]
                               if a.episode_tip_calibrations else tcp_tip)
            if episode["task"]["task_family"] != "sc_to_sc":
                raise ValueError("Non-SC episode in port edge file")
            selected = int(port["target_port_index"])
            parent = f"task_board/sc_port_{selected}"
            name = f"{parent}/sc_port_base_link"
            edges = port["edges"]
            world_base = matrix(select(edges, "world", "tabletop")) @ matrix(select(edges, "tabletop", "base_link"))
            world_port = matrix(select(edges, "aic_world", "task_board")) @ matrix(select(edges, "task_board", parent)) @ matrix(select(edges, parent, name))
            base_port = np.linalg.inv(world_base) @ world_port
            port_base = np.linalg.inv(base_port)
            if np.linalg.norm(world_base[:3, 3] - np.array([-0.2, 0.2, 1.14])) > 1e-5:
                raise ValueError("Unexpected world/base alias in scored Gazebo TF")
            folder = Path(episode["canonical_episode_dir"])
            frames = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
            all_times = [float(row["sim_time"]) for row in frames]
            local_count = 0
            invalid_timing = 0
            missing_image = 0
            for j in range(len(frames) - 3):
                if a.resample_mode == "strict":
                    chunk = frames[j:j+4]
                    times = [float(row["sim_time"]) for row in chunk]
                    indices = [int(row["command_index"]) for row in chunk]
                    deltas = [y-x for x, y in zip(times, times[1:])]
                    if any(y != x+1 for x, y in zip(indices, indices[1:])) or any(
                        abs(dt - a.step_seconds) > a.time_tolerance_seconds for dt in deltas
                    ):
                        invalid_timing += 1
                        continue
                    source_rows = chunk
                    source_times = times
                else:
                    anchor_time = all_times[j]
                    # Ensure observations cover the entire four-command duration.
                    end = bisect.bisect_left(all_times, anchor_time + 4*a.step_seconds - 1e-6, lo=j)
                    if end >= len(frames) or all_times[end] - anchor_time > 0.25:
                        invalid_timing += 1
                        continue
                    if any(int(frames[k+1]["command_index"]) != int(frames[k]["command_index"]) + 1
                           for k in range(j, end)):
                        invalid_timing += 1
                        continue
                    source_indices = [bisect.bisect_right(all_times, anchor_time + h*a.step_seconds + 1e-6, lo=j, hi=end+1)-1
                                      for h in range(4)]
                    source_rows = [frames[k] for k in source_indices]
                    source_times = [all_times[k] for k in source_indices]
                    if any(anchor_time + h*a.step_seconds - source_times[h] > a.max_hold_seconds + 1e-6
                           for h in range(4)):
                        invalid_timing += 1
                        continue
                    chunk = source_rows
                    times = [anchor_time + h*a.step_seconds for h in range(4)]
                    indices = [int(row["command_index"]) for row in source_rows]
                images = chunk[0]["images"]
                if set(images) != {"center", "left", "right"} or not all((folder / v).is_file() for v in images.values()):
                    missing_image += 1
                    continue
                current_tcp = pose_matrix(chunk[0]["state"][:7])
                anchor_tip = port_base @ current_tcp @ tip_for_episode
                targets = []
                tcp_targets = []
                for row in chunk:
                    tcp_target = pose_matrix(row["teacher_target_pose"])
                    tcp_targets.append(pose_list(port_base @ tcp_target))
                    tip_target = port_base @ tcp_target @ tip_for_episode
                    recovered_tcp = base_port @ tip_target @ np.linalg.inv(tip_for_episode)
                    max_roundtrip_m = max(max_roundtrip_m, float(np.linalg.norm(recovered_tcp[:3, 3] - tcp_target[:3, 3])))
                    max_roundtrip_deg = max(max_roundtrip_deg, float(np.degrees(Rotation.from_matrix(recovered_tcp[:3, :3] @ tcp_target[:3, :3].T).magnitude())))
                    targets.append(pose_list(tip_target))
                opening_targets = [pose_list(opening_port @ pose_matrix(target)) for target in targets]
                tcp_opening_targets = [pose_list(opening_port @ pose_matrix(target)) for target in tcp_targets]
                output.write(json.dumps({
                    "episode_index": int(episode["episode_index"]),
                    "scene_sha256": episode["scene_sha256"],
                    "split": episode["split"],
                    "frame": chunk[0]["frame"],
                    "sim_time": times[0],
                    "episode_start_sim_time": all_times[0],
                    "command_indices": indices,
                    "command_time_offsets_s": [t-times[0] for t in times],
                    "source_command_time_offsets_s": [t-times[0] for t in source_times],
                    "held_target_ticks": sum(indices[k] == indices[k-1] for k in range(1, 4)),
                    "images": {k: str(folder / v) for k, v in images.items()},
                    "state": chunk[0]["state"],
                    "task_vector": episode["task_vector"],
                    "corrective_episode": "corrective_supervision" in episode.get("status", ""),
                    "teacher_rescued_direct_actor_takeover": "teacher_rescued_direct_actor_takeover" in episode.get("status", ""),
                    "selected_port_frame": name,
                    "selected_port_opening_frame": name + "_entrance",
                    "tcp_tip_label_source": ("episode_scored_physical_tf" if a.episode_tip_calibrations else "fixed_tcp_tip_proxy"),
                    "observed_sc_tip_pose_port_frame": pose_list(anchor_tip),
                    "observed_sc_tip_pose_opening_frame": pose_list(opening_port @ anchor_tip),
                    "observed_tcp_pose_port_frame": pose_list(port_base @ current_tcp),
                    "observed_tcp_pose_opening_frame": pose_list(opening_port @ port_base @ current_tcp),
                    "teacher_tcp_target_poses_port_frame": tcp_targets,
                    "teacher_tcp_target_poses_opening_frame": tcp_opening_targets,
                    "teacher_sc_tip_target_poses_port_frame": targets,
                    "teacher_sc_tip_target_poses_opening_frame": opening_targets,
                    "corrective_student_active_anchor": bool(chunk[0].get("student_active", False)),
                    "corrective_student_active_source_ticks": sum(bool(row.get("student_active", False)) for row in chunk),
                    "corrective_executed_target_pose_base_frame_anchor": chunk[0].get("executed_target_pose"),
                }, separators=(",", ":")) + "\n")
                local_count += 1
            counts[episode["split"]] += local_count
            reports.append({"episode_index": episode["episode_index"], "scene_sha256": episode["scene_sha256"],
                            "split": episode["split"], "frames": len(frames), "valid_chunks": local_count,
                            "invalid_timing_chunks": invalid_timing, "missing_image_chunks": missing_image})
    train_groups = {r["scene_sha256"] for r in reports if r["split"] == "train"}
    val_groups = {r["scene_sha256"] for r in reports if r["split"] != "train"}
    if train_groups & val_groups:
        raise ValueError("Scene leakage across SC splits")
    summary = {"schema": "aic_sc_connector_waypoint_labels/v1", "labels": str(a.labels_output),
               "training_label_only": True, "port_edges": str(a.port_edges), "tcp_tip_calibration": str(a.tcp_tip_calibration),
               "episode_tip_calibrations": str(a.episode_tip_calibrations) if a.episode_tip_calibrations else None,
               "port_opening_calibration": str(a.port_opening_calibration), "pose_quaternion_order": "xyzw",
               "timing": {"resample_mode": a.resample_mode, "step_s": a.step_seconds,
                          "tolerance_s": a.time_tolerance_seconds, "max_hold_s": a.max_hold_seconds,
                          "first_target_semantics": "teacher command recorded with anchor image; target held until next recorded command when resampling"},
               "counts_by_split": dict(counts), "episodes": len(reports),
               "train_scene_groups": len(train_groups), "validation_scene_groups": len(val_groups),
               "max_tcp_target_roundtrip_m": max_roundtrip_m, "max_tcp_target_roundtrip_deg": max_roundtrip_deg,
               "episode_reports": reports}
    a.summary_output.parent.mkdir(parents=True, exist_ok=True)
    a.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts_by_split", "episodes", "max_tcp_target_roundtrip_m")}))


if __name__ == "__main__":
    main()
