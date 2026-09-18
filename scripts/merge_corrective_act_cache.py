#!/usr/bin/env python3
"""Add score-verified corrective demonstrations without copying the legacy image cache."""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import yaml
from scipy.spatial.transform import Rotation


def inspect_collection(root, min_score):
    summary_path = root / "eval_collection/collection_config/eval_summary.json"
    summary = json.loads(summary_path.read_text())
    if not summary.get("evaluation_complete") or summary.get("evaluation_kind") != "privileged_corrective_data_collection":
        raise ValueError(f"Collection lacks a complete, explicitly privileged summary: {root}")
    config = yaml.safe_load((root / "engine_config.yaml").read_text())
    scores = yaml.safe_load(Path(summary["scoring_yaml"]).read_text())
    episodes = sorted((root / "episodes").glob("*/episode.json"), key=lambda p: int(p.parent.name.rsplit("_", 1)[1]))
    trials = list(config["trials"])
    if len(episodes) != len(trials):
        raise ValueError(f"Cannot establish one-to-one episode/trial lineage: {root}")
    accepted, rejected = [], []
    verified_student_files = {}
    for ordinal, (path, trial_name) in enumerate(zip(episodes, trials), 1):
        score = scores[trial_name]
        total = sum(float(score[key]["score"]) for key in ("tier_1", "tier_2", "tier_3"))
        record = {"collection": str(root.resolve()), "episode_dir": str(path.parent.resolve()),
                  "trial_id": trial_name, "source_trial_ordinal": ordinal,
                  "official_score_path": summary["scoring_yaml"], "official_total": total,
                  "official_tier3": float(score["tier_3"]["score"])}
        if abs(record["official_tier3"] - 75.) > 1e-6 or total < min_score:
            rejected.append(record)
            continue
        metadata = json.loads(path.read_text())
        rows = [json.loads(line) for line in (path.parent / "frames.jsonl").read_text().splitlines()]
        if len(rows) != metadata["frames"] or len(rows) < 2 or metadata["image_channel_order"] != "rgb":
            raise ValueError(f"Invalid corrective episode metadata: {path}")
        states = np.asarray([row["state"] for row in rows], dtype=np.float32)
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        targets = np.asarray([row["teacher_target_pose"] for row in rows])
        times = np.asarray([row["elapsed_sim_time"] for row in rows])
        if (states.shape != (len(rows), 32) or actions.shape != (len(rows), 6) or targets.shape != (len(rows), 7)
                or not np.isfinite(states).all() or not np.isfinite(actions).all() or not np.isfinite(targets).all()
                or not np.isfinite(times).all() or not np.all(np.diff(times) > 0)):
            raise ValueError(f"Invalid state/action/timestamp arrays: {path}")
        rotation = Rotation.from_quat(states[:, 3:7])
        pos_error = float(np.max(np.abs(states[:, :3] + rotation.apply(actions[:, :3]) - targets[:, :3])))
        rot_error = float(np.max(((rotation * Rotation.from_rotvec(actions[:, 3:])).inv() * Rotation.from_quat(targets[:, 3:])).magnitude()))
        if pos_error > 1e-6 or rot_error > 1e-6:
            raise ValueError(f"Teacher labels do not reconstruct their targets: {path}")
        for row in rows:
            if set(row["images"]) != {"center", "left", "right"} or not all((path.parent / value).is_file() for value in row["images"].values()):
                raise ValueError(f"Missing camera frame: {path}")
        perturbations = sum(np.linalg.norm(row.get("perturbation_xyz_rotvec", row["perturbation_xy_rotvec"])) > 1e-8 for row in rows)
        student = metadata.get("student_correction")
        student_probability = summary.get("runtime_settings", {}).get("corrective_student_probability", 0)
        student_unused_nominal = False
        if bool(student) != bool(student_probability):
            raise ValueError(f"Student correction provenance disagrees with runtime: {path}")
        if student:
            if student.get("probability_per_cycle") != student_probability:
                raise ValueError(f"Student correction probability disagrees with runtime: {path}")
            student_path = Path(student["torchscript"])
            if student_path not in verified_student_files:
                verified_student_files[student_path] = hashlib.sha256(student_path.read_bytes()).hexdigest()
            if verified_student_files[student_path] != student["sha256"]:
                raise ValueError(f"Student checkpoint hash mismatch: {path}")
            active = [row for row in rows if row.get("student_active")]
            if len(active) != metadata.get("student_recorded_frames"):
                raise ValueError(f"Student intervention frame count mismatch: {path}")
            student_unused_nominal = (not active and metadata.get("perturbation_config", {}).get("scale") == 0
                                      and summary.get("runtime_settings", {}).get("corrective_perturbation_scale") == 0)
            if not active and not student_unused_nominal:
                raise ValueError(f"No verified student intervention frames: {path}")
            for row in active:
                applied = np.asarray(row["perturbation_xyz_rotvec"])
                proposed = np.asarray(row["student_absolute_action"])
                executed = np.asarray(row["executed_target_pose"])
                teacher = np.asarray(row["teacher_target_pose"])
                if (applied.shape != (6,) or proposed.shape != (6,) or not np.isfinite(applied).all()
                        or not np.isfinite(proposed).all() or np.linalg.norm(applied[:3]) > .030001
                        or np.linalg.norm(applied[3:]) > .080001 or executed.shape != (7,)
                        or not np.isfinite(executed).all()):
                    raise ValueError(f"Invalid bounded student intervention: {path}")
                actual_delta = np.concatenate([executed[:3] - teacher[:3],
                    (Rotation.from_quat(teacher[3:]).inv() * Rotation.from_quat(executed[3:])).as_rotvec()])
                if not np.allclose(applied, actual_delta, atol=1e-6, rtol=0):
                    raise ValueError(f"Student intervention does not match executed target: {path}")
        explicitly_nominal = student_unused_nominal or (metadata.get("kind") == "privileged_aligned_demonstration"
                              and metadata.get("perturbation_config", {}).get("scale") == 0
                              and summary.get("runtime_settings", {}).get("corrective_perturbation_scale") == 0
                              and not student)
        if not perturbations and not explicitly_nominal:
            raise ValueError(f"No recorded perturbations in corrective episode: {path}")
        if explicitly_nominal:
            executed = np.asarray([row["executed_target_pose"] for row in rows])
            if perturbations or executed.shape != targets.shape or not np.allclose(executed, targets, atol=1e-9, rtol=0):
                raise ValueError(f"Nominal collection changed the teacher target: {path}")
        # Causal zero-order hold on an actual simulation-time grid. No invented
        # images or interpolated action labels; each frame points to an observed row.
        grid = np.arange(0., times[-1] - times[0] + 1e-8, .05)
        selected = np.searchsorted(times - times[0], grid + 1e-8, side="right") - 1
        board = config["trials"][trial_name]["scene"]["task_board"]
        nic_count = sum(bool(v.get("entity_present")) for k, v in board.items() if k.startswith("nic_rail_"))
        record.update({"nic_count": nic_count, "split": "validation" if ordinal % 5 == 0 else "train",
                       "demonstration_kind": ("nominal_expert_no_student_intervention" if student_unused_nominal else
                                              "nominal_aligned_expert" if explicitly_nominal else
                                              "student_corrective_expert" if student else "perturbed_expert_correction"),
                       "student_correction": student,
                       "student_recorded_frames": metadata.get("student_recorded_frames", 0),
                       "execution_frame": metadata.get("execution_frame", "gripper/tcp"),
                       "original_recorded_frames": len(rows), "resampled_frames": len(grid),
                       "perturbed_recorded_frames": int(perturbations), "target_position_roundtrip_error_m": pos_error,
                       "target_rotation_roundtrip_error_rad": rot_error,
                       "timestamp_source": "actual simulation stamp, resampled to 20 Hz by causal hold",
                       "trial_yaml": str((root / "trials" / f"{trial_name}.yaml").resolve())})
        accepted.append((record, rows, selected, grid.astype(np.float32), states[selected], actions[selected]))
    return accepted, rejected


def inherited_image_shards(base, base_root, collections):
    """Reuse immutable image shards and reject repeated collection inputs."""
    requested = [str(Path(path).resolve()) for path in collections]
    existing = {str(Path(row["collection"]).resolve()) for row in base["episodes"] if row.get("collection")}
    if len(set(requested)) != len(requested) or existing.intersection(requested):
        raise ValueError("Collection already present in cache or repeated in merge request")
    if base.get("image_shards"):
        return copy.deepcopy(base["image_shards"])
    return [{"root": str(base_root.resolve()), "from_index": 0, "to_index": base["frames"],
             "image_channel_order": base["image_channel_order"]}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--collections", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-total-score", type=float, default=80.)
    args = parser.parse_args()
    base = json.loads((args.base_cache / "cache.json").read_text())
    previous_shards = inherited_image_shards(base, args.base_cache, args.collections)
    accepted, rejected = [], []
    for collection in args.collections:
        good, bad = inspect_collection(collection, args.minimum_total_score)
        accepted.extend(good); rejected.extend(bad)
    if not accepted:
        raise ValueError("No score-verified corrective episodes passed acceptance")
    root = args.output_dir.resolve(); root.mkdir(parents=True, exist_ok=False)
    image_root = root / "corrective_images"; image_root.mkdir()
    new_frames = sum(len(entry[2]) for entry in accepted)
    images = {key: np.lib.format.open_memmap(image_root / (key + ".npy"), mode="w+", dtype=np.uint8,
                                            shape=(new_frames, *base["image_shape_hwc"])) for key in base["camera_keys"]}
    arrays = {key: [np.load(args.base_cache / (key + ".npy"))] for key in ("states", "actions", "episodes", "timestamps")}
    metadata = copy.deepcopy(base)
    offset = 0
    next_episode = int(arrays["episodes"][0].max()) + 1
    for record, rows, selected, grid, states, actions in accepted:
        episode = next_episode; next_episode += 1
        count = len(grid)
        for key in images:
            camera = key.removeprefix("observation.images.").removesuffix("_camera")
            last_index, last_image = None, None
            for j, index in enumerate(selected):
                if index != last_index:
                    image_path = Path(record["episode_dir"]) / rows[index]["images"][camera]
                    with Image.open(image_path) as image:
                        last_image = np.asarray(image.convert("RGB"))
                    if tuple(last_image.shape) != tuple(base["image_shape_hwc"]):
                        raise ValueError(f"Invalid corrective image: {image_path}")
                    last_index = index
                images[key][offset + j] = last_image
        arrays["states"].append(states); arrays["actions"].append(actions)
        arrays["episodes"].append(np.full(count, episode, np.int64)); arrays["timestamps"].append(grid)
        record.update({"episode_index": episode, "cache_from_index": base["frames"] + offset,
                       "cache_to_index": base["frames"] + offset + count,
                       "source_kind": "score_verified_privileged_corrective_labels"})
        metadata["episodes"].append(record)
        metadata["splits_by_nic_count"][str(record["nic_count"])][record["split"]].append(episode)
        offset += count
        print(json.dumps({"episode": episode, "frames": count, "score": record["official_total"], "split": record["split"]}), flush=True)
    for array in images.values():
        array.flush()
    for key, pieces in arrays.items():
        np.save(root / (key + ".npy"), np.concatenate(pieces))
    metadata.update({"frames": base["frames"] + new_frames, "base_cache": str(args.base_cache.resolve()),
                     "corrective_acceptance": "Official Tier 3 = 75 and total >= minimum_total_score, with complete image/label lineage",
                     "minimum_total_score": args.minimum_total_score,
                     "new_episode_split": "Every fifth original trial in each independent collection is validation; stable across merges",
                     "image_shards": previous_shards + [{"root": str(image_root), "from_index": base["frames"], "to_index": base["frames"] + new_frames, "image_channel_order": "rgb"}]})
    (root / "cache.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (root / "corrective_verification.json").write_text(json.dumps({"accepted": [x[0] for x in accepted], "rejected": rejected}, indent=2) + "\n")
    print(json.dumps({"accepted_new_episodes": len(accepted), "rejected_new_episodes": len(rejected), "frames": metadata["frames"]}))


if __name__ == "__main__":
    main()
