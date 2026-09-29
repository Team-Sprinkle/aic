#!/usr/bin/env python3
"""Create scene-grouped SC native-image pose and pixel training labels.

Bag TF and camera calibration are used only to create supervision. A runtime
locator must use images and ordinary state, never these labels or TF.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_sc_port_targets import matrix, pose_list, pose_matrix, select


WIDTH, HEIGHT, HFOV = 1152, 1024, 0.8718
FOCAL = WIDTH / (2 * math.tan(HFOV / 2))


def chain(edges, nodes):
    out = np.eye(4)
    for left, right in zip(nodes, nodes[1:]):
        out = out @ matrix(select(edges, left, right))
    return out


def project(point_base, base_optical):
    pos = (np.linalg.inv(base_optical) @ point_base)[:3, 3]
    if pos[2] <= 0:
        return [None, None, float(pos[2]), False]
    u = WIDTH / 2 + FOCAL * pos[0] / pos[2]
    v = HEIGHT / 2 + FOCAL * pos[1] / pos[2]
    return [float(u), float(v), float(pos[2]), bool(0 <= u < WIDTH and 0 <= v < HEIGHT)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("edges", type=Path)
    parser.add_argument("--tcp-tip-calibration", type=Path, required=True)
    parser.add_argument("--port-opening-calibration", type=Path, required=True)
    parser.add_argument("--labels-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--verified-manifest", type=Path,
                        default=Path("outputs/trajectory_datasets/expert_verified/manifest.json"))
    parser.add_argument("--include-failed-for-pose-only", action="store_true",
                        help="Retain scored failed repeats as geometry-labeled perception data; never BC expert data")
    parser.add_argument("--use-lowres-images", action="store_true",
                        help="Use recorded 288x256 RGB for a coarse global-view ablation when native frames are absent")
    parser.add_argument("--allow-new-development-scenes", action="store_true",
                        help="Admit explicitly marked trial_9xxxxx development resets outside the verified expert manifest for pose only")
    parser.add_argument("--allow-new-development-validation", action="store_true",
                        help="Legacy alias to admit only new_development_validation resets")
    args = parser.parse_args()
    edge_rows = json.loads(args.edges.read_text())["rows"]
    known = {}
    for row in json.loads(args.verified_manifest.read_text())["episodes"]:
        if row["task"]["task_family"] != "sc_to_sc":
            continue
        known.setdefault(row["trial_id"], []).append(row)
    split_by_trial = {}
    for item in edge_rows:
        if item.get("source_episode_index") is not None:
            matches = [row for rows in known.values() for row in rows
                       if row["episode_index"] == item["source_episode_index"]]
        else:
            matches = known.get(item["trial"], [])
        if len(matches) != 1:
            new_split = item.get('source_split')
            if ((args.allow_new_development_scenes or args.allow_new_development_validation) and not matches
                    and item['trial'].startswith('trial_9')
                    and new_split in (('new_development_validation','new_development_train')
                                      if args.allow_new_development_scenes else ('new_development_validation',))
                    and item.get('source_episode_index') is None):
                split_by_trial[item['trial']] = 'train' if new_split == 'new_development_train' else 'validation'
                continue
            raise ValueError(f"Expected one verified manifest row for {item['trial']}: {len(matches)}")
        if matches[0]["task"]["target_port_index"] != int(item["target_module_name"].rsplit("_", 1)[-1]):
            raise ValueError(f"Port mismatch for {item['trial']}")
        split_by_trial[item["trial"]] = matches[0]["split"]
        if item.get("source_split") and item["source_split"] != matches[0]["split"]:
            raise ValueError(f"Split mismatch for {item['trial']}")
    tcp_tip = matrix(json.loads(args.tcp_tip_calibration.read_text())["tcp_to_sc_tip"])
    port_opening = matrix(json.loads(args.port_opening_calibration.read_text())["base_to_opening"])
    camera_mount_tcp = chain(edge_rows[0]["edges"], [
        "tool0", "cam_mount/cam_mount_link", "ati/base_link", "ati/tool_link",
        "gripper/hande_base_link", "gripper/tcp"])
    local_cameras = {}
    for cam in ("center", "left", "right"):
        tool_camera = chain(edge_rows[0]["edges"], [
            "tool0", "cam_mount/cam_mount_link", f"{cam}_camera/camera_link",
            f"{cam}_camera/sensor_link", f"{cam}_camera/optical"])
        local_cameras[cam] = np.linalg.inv(camera_mount_tcp) @ tool_camera
    report = []
    args.labels_output.parent.mkdir(parents=True, exist_ok=True)
    with args.labels_output.open("w") as output:
        for episode_index, item in enumerate(edge_rows, start=1):
            edges = item["edges"]
            trial = item["trial"]
            expert_bc_eligible = item["official_tier3"] == 75 and not trial.startswith("trial_9")
            if item["official_tier3"] != 75 and not args.include_failed_for_pose_only:
                report.append({"trial": trial, "admitted": False, "reason": "official Tier 3 != 75"})
                continue
            episode = Path(item["episode"])
            port = int(item["target_module_name"].rsplit("_", 1)[-1])
            port_parent = f"task_board/sc_port_{port}"
            port_name = port_parent + "/sc_port_base_link"
            world_base = chain(edges, ["world", "tabletop", "base_link"])
            world_port = chain(edges, ["aic_world", "task_board", port_parent, port_name])
            base_opening = np.linalg.inv(world_base) @ world_port @ port_opening
            opening_base = np.linalg.inv(base_opening)
            scene_hash = item["scene_sha256"]
            count = 0
            near_count = 0
            visible = {cam: {"tip": 0, "opening": 0} for cam in local_cameras}
            for line in (episode / "frames.jsonl").open():
                row = json.loads(line)
                names = row.get("native_images")
                if not names and args.use_lowres_images:
                    names = row.get("images")
                if not names:
                    continue
                paths = {cam: str(episode / names[cam]) for cam in local_cameras}
                if not all(Path(path).is_file() for path in paths.values()):
                    raise FileNotFoundError(paths)
                base_tcp = pose_matrix(row["state"][:7])
                base_tip = base_tcp @ tcp_tip
                opening_tip = opening_base @ base_tip
                pixels = {}
                for cam, tcp_optical in local_cameras.items():
                    base_optical = base_tcp @ tcp_optical
                    p_tip = project(base_tip, base_optical)
                    p_port = project(base_opening, base_optical)
                    pixels[cam] = {"tip": p_tip, "opening": p_port}
                    visible[cam]["tip"] += int(p_tip[3])
                    visible[cam]["opening"] += int(p_port[3])
                task_vector = [0., 1., *([1., 0.] if port == 0 else [0., 1.]), *([0.] * 6)]
                record = {
                    "schema": "sc_native_pose_labels/v1", "trial": trial,
                    "official_tier3": item["official_tier3"],
                    "expert_bc_eligible": expert_bc_eligible,
                    "pose_label_only_when_failed": item["official_tier3"] != 75,
                    "pose_label_only": not expert_bc_eligible,
                    "episode_id": episode.name, "episode_index": episode_index,
                    "scene_sha256": scene_hash,
                    "split": split_by_trial[trial],
                    "frame": row["frame"], "sim_time": row["sim_time"],
                    "images": paths,
                    "lowres_images": {cam: str(episode / row["images"][cam]) for cam in local_cameras},
                    "state": row["state"], "task_vector": task_vector,
                    "observed_sc_tip_pose_opening_frame": pose_list(opening_tip),
                    "projected_training_pixels": pixels,
                    "pixel_label_source": "scored Gazebo TF + configured camera FOV + recorded TCP; training only",
                }
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
                near_count += int(abs(opening_tip[2, 3]) < 0.03)
            report.append({"trial": trial, "admitted": True, "official_tier3": item["official_tier3"],
                           "expert_bc_eligible": expert_bc_eligible,
                           "pose_label_only": not expert_bc_eligible,
                           "split": split_by_trial[trial],
                           "frames": count, "near_port_frames": near_count,
                           "in_frame_counts": visible, "scene_sha256": scene_hash})
    train = {r["scene_sha256"] for r in report if r.get("split") == "train"}
    val = {r["scene_sha256"] for r in report if r.get("split") == "validation"}
    if train & val:
        raise ValueError("Scene leakage")
    summary = {"schema": "sc_native_pose_labels_summary/v1", "labels": str(args.labels_output),
               "training_label_only": True, "image_size": [WIDTH, HEIGHT], "horizontal_fov_rad": HFOV,
               "focal_px": FOCAL, "verified_manifest": str(args.verified_manifest),
               "tcp_to_optical": {cam: pose_list(transform) for cam, transform in local_cameras.items()},
               "include_failed_for_pose_only": args.include_failed_for_pose_only,
               "validation_trials": sorted(k for k, v in split_by_trial.items() if v == "validation"),
               "episodes": report, "train_scene_count": len(train), "validation_scene_count": len(val)}
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"total_frames": sum(r.get("frames", 0) for r in report),
                      "near_port_frames": sum(r.get("near_port_frames", 0) for r in report)}))


if __name__ == "__main__":
    main()
