#!/usr/bin/env python3
"""Bounded, episode-balanced held-out ACT errors and observation ablations.

Defaults to CPU. This performs no training or simulator actions. For CUDA, the
caller must reserve a device and set CUDA_VISIBLE_DEVICES explicitly. Task swaps
report prediction changes only: the alternative target has no ground-truth label.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


def task_fields(episode):
    task = episode.get("task", episode)
    return (task["task_family"], int(task["target_card_index"]),
            int(task["target_port_index"]), int(task["target_card_valid"]))


def absolute_targets(states, relative_actions):
    """Match the trainer's absolute xyz and X-positive quaternion rotvec labels."""
    rotations = Rotation.from_quat(states[:, 3:7])
    xyz = states[:, :3] + rotations.apply(relative_actions[:, :3])
    quat = (rotations * Rotation.from_rotvec(relative_actions[:, 3:6])).as_quat()
    quat[quat[:, 0] < 0] *= -1
    length = np.linalg.norm(quat[:, :3], axis=1)
    angle = 2 * np.arctan2(length, quat[:, 3])
    rotvec = quat[:, :3] * (angle / np.maximum(length, 1e-12))[:, None]
    return np.concatenate([xyz, rotvec], axis=1).astype(np.float32)


def pose_errors(predicted, target):
    """Euclidean millimeters and geodesic degrees; handles equivalent rotvecs."""
    translation = np.linalg.norm(predicted[..., :3] - target[..., :3], axis=-1) * 1000
    shape = translation.shape
    rotation = (Rotation.from_rotvec(predicted[..., 3:6].reshape(-1, 3)).inv()
                * Rotation.from_rotvec(target[..., 3:6].reshape(-1, 3))).magnitude()
    return translation, np.degrees(rotation).reshape(shape)


def balanced_samples(episodes, episode_ids, frames_per_episode, rng):
    """Sample without replacement within each episode, with equal episode caps."""
    selected = []
    for episode_id in sorted(episode_ids):
        candidates = np.flatnonzero(episodes == episode_id)
        if not len(candidates):
            raise ValueError(f"Held-out episode {episode_id} has no cache frames")
        selected.extend(sorted(rng.choice(candidates, min(frames_per_episode, len(candidates)),
                                          replace=False).tolist()))
    return np.asarray(selected, dtype=np.int64)


def image_donors(selected, episodes, timestamps, metadata, heldout_ids,
                 phase_bins, time_bin_sec, rng):
    """Other-episode donors with identical task/count, time bin and phase bin.

    No fallback broadens a failed match. A -1 donor marks an unavailable ablation.
    All three camera views use the same donor frame to retain camera consistency.
    """
    pools = defaultdict(list)
    keys = {}
    for episode_id in sorted(heldout_ids):
        indices = np.flatnonzero(episodes == episode_id)
        elapsed = timestamps[indices] - timestamps[indices[0]]
        phase = np.minimum((elapsed / max(float(elapsed[-1]), 1e-9) * phase_bins).astype(int),
                           phase_bins - 1)
        time_bin = (elapsed / time_bin_sec).astype(int)
        scene = metadata[episode_id]
        base = (*task_fields(scene), int(scene["nic_count"]), int(scene.get("sc_count", -1)))
        for index, pbin, tbin in zip(indices, phase, time_bin):
            key = (*base, int(pbin), int(tbin))
            pools[key].append(int(index)); keys[int(index)] = key
    donors = np.full(len(selected), -1, dtype=np.int64)
    for row, index in enumerate(selected):
        candidates = np.asarray(pools[keys[int(index)]], dtype=np.int64)
        candidates = candidates[episodes[candidates] != episodes[index]]
        if len(candidates):
            donors[row] = rng.choice(candidates)
    return donors


def alternative_task_vector(episode):
    """Toggle to another physically present port, or skip if none is established."""
    family, card, port, valid = task_fields(episode)
    if family == "sc_to_sc" and int(episode.get("sc_count", 0)) < 2:
        return None
    if family not in {"sfp_to_nic", "sc_to_sc"}:
        return None
    # Every NIC has two SFP ports. Two SC modules establish both SC targets.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
    from lerobot_robot_aic.task_encoding import encode_task_vector
    return encode_task_vector(task_family=family, target_card_index=card,
                              target_port_index=1 - port, target_card_valid=valid)


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95))} if values.size else None


def summarize(records, metadata):
    groups = defaultdict(list)
    for row in records:
        episode = metadata[row["episode_index"]]
        family, card, port, _ = task_fields(episode)
        for key in ["all", f"family/{family}",
                    f"count/{family}/nic_{episode['nic_count']}/sc_{episode.get('sc_count', 'unknown')}",
                    f"target/{family}/card_{card}/port_{port}", f"episode/{row['episode_index']}"]:
            groups[key].append(row)
    output = {}
    for group, rows in sorted(groups.items()):
        metrics = {key for row in rows for key, value in row.items()
                   if key.endswith(("_mm", "_deg")) and value is not None}
        output[group] = {"frames": len(rows), "episodes": sorted({row["episode_index"] for row in rows}),
                         "metrics": {key: describe([row[key] for row in rows if row.get(key) is not None])
                                     for key in sorted(metrics)}}
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--training-config", type=Path,
                   help="Optional training metadata; otherwise use checkpoint training_step.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--frames-per-episode", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=2)
    p.add_argument("--phase-bins", type=int, default=5)
    p.add_argument("--time-bin-sec", type=float, default=4.)
    p.add_argument("--seed", type=int, default=9218)
    p.add_argument("--selection-only", action="store_true",
                   help="Validate sampling/donor/task plans on CPU without loading a model or images")
    p.add_argument("--bf16", action="store_true", help="Explicit optional CUDA autocast; default FP32")
    return p


def main():
    args = parser().parse_args()
    if min(args.frames_per_episode, args.batch_size, args.cpu_threads, args.phase_bins) < 1 or args.time_bin_sec <= 0:
        raise ValueError("Sampling, batch, thread, phase, and time-bin sizes must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace analysis: {args.output}")
    if not args.selection_only and not args.checkpoint:
        raise ValueError("Inference requires --checkpoint")
    cache = json.loads((args.cache / "cache.json").read_text())
    episodes = np.load(args.cache / "episodes.npy", mmap_mode="r")
    timestamps = np.load(args.cache / "timestamps.npy", mmap_mode="r")
    metadata = {int(row["episode_index"]): row for row in cache["episodes"]}
    training = None
    if args.training_config:
        training = json.loads(args.training_config.read_text())
    elif args.checkpoint:
        training = json.loads((args.checkpoint / "training_step.json").read_text())["training_config"]
    if training:
        if Path(training["source_cache"]).resolve() != args.cache.resolve():
            raise ValueError("Checkpoint/training metadata belongs to a different cache")
        heldout = set(map(int, training["validation_episodes"]))
        if heldout & set(map(int, training["train_episodes"])):
            raise ValueError("Training and validation episode IDs overlap")
    else:
        heldout = {i for i, row in metadata.items() if row.get("split") == "validation"}
    if not heldout or not heldout <= metadata.keys():
        raise ValueError("Need a nonempty held-out split represented in this cache")
    if any(metadata[i].get("split") != "validation" for i in heldout):
        raise ValueError("Selected evaluation episode is not marked validation")
    rng = np.random.default_rng(args.seed)
    selected = balanced_samples(episodes, heldout, args.frames_per_episode, rng)
    donors = image_donors(selected, episodes, timestamps, metadata, heldout,
                          args.phase_bins, args.time_bin_sec, rng)
    swaps = {i: alternative_task_vector(metadata[i]) for i in heldout}
    training_targets = {task_fields(metadata[i]) for i in training["train_episodes"]} if training else set()
    records = [{"frame_index": int(index), "episode_index": int(episodes[index]),
                "timestamp_sec": float(timestamps[index]),
                "image_donor_frame": int(donor) if donor >= 0 else None,
                "image_donor_episode": int(episodes[donor]) if donor >= 0 else None,
                "task_swap_vector": swaps[int(episodes[index])].tolist()
                if swaps[int(episodes[index])] is not None else None,
                "swapped_target_observed_in_training":
                (*task_fields(metadata[int(episodes[index])])[:2],
                 1 - task_fields(metadata[int(episodes[index])])[2],
                 task_fields(metadata[int(episodes[index])])[3]) in training_targets
                if training and swaps[int(episodes[index])] is not None else None}
               for index, donor in zip(selected, donors)]
    report = {"kind": "heldout_offline_ablation_diagnostic", "cache": str(args.cache.resolve()),
              "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
              "seed": args.seed, "frames_per_episode": args.frames_per_episode,
              "heldout_episodes": sorted(heldout), "sampled_frames": len(selected),
              "image_shuffle_matching": {"same_task_target_and_nic_sc_count": True,
                  "different_episode": True, "normalized_episode_phase_bins": args.phase_bins,
                  "elapsed_time_bin_seconds": args.time_bin_sec, "matched_frames": int((donors >= 0).sum()),
                  "unmatched_frames_are_skipped": True},
              "blank_image": "Spatially constant ImageNet mean RGB; normalized image tensor is zero",
              "metric_weighting": "Equal per-episode sample caps; executed_mean metrics average valid commands in the first four predicted slots, without executing them",
              "task_swap": "Toggle port within the same family/card only when another port is present; prediction change only",
              "task_swap_caveat": "A valid alternative target can be absent from training; inspect swapped_target_observed_in_training before interpreting sensitivity",
              "interpretation": "Artificial input ablations diagnose reliance; they do not measure closed-loop insertion success. Task-swap accuracy is deliberately undefined.",
              "selection_only": args.selection_only, "records": records}
    if not args.selection_only:
        import torch
        from safetensors.torch import load_file
        from lerobot.configs.policies import PreTrainedConfig
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
        from lerobot_robot_aic.act_backbone import load_act_policy
        from act_cache import load_cached_images, prepare_camera_arrays

        if args.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise ValueError("Set CUDA_VISIBLE_DEVICES to a reserved GPU before CUDA analysis")
        if args.bf16 and not args.device.startswith("cuda"):
            raise ValueError("BF16 option is only enabled for explicitly selected CUDA analysis")
        torch.set_num_threads(args.cpu_threads)
        device = torch.device(args.device)
        contract = json.loads((args.checkpoint / "aic_action_config.json").read_text())
        if (contract.get("action_representation") != "absolute_pose"
                or not contract.get("task_conditioned") or contract.get("task_vector_indices") != [32, 42]):
            raise ValueError("This diagnostic requires the audited absolute-action, 10D-task ACT contract")
        configs = PreTrainedConfig.from_pretrained(args.checkpoint, local_files_only=True)
        configs.device = args.device
        policy = load_act_policy(args.checkpoint, config=configs, local_files_only=True).to(device).eval()
        norm_paths = list(args.checkpoint.glob("*normalizer_processor.safetensors"))
        norms = {}
        for path in norm_paths:
            for key, value in load_file(str(path)).items():
                if key in norms and not torch.equal(norms[key], value):
                    raise ValueError(f"Inconsistent saved normalization for {key}")
                norms[key] = value
        state_mean = norms["observation.state.mean"].to(device)
        state_std = norms["observation.state.std"].to(device)
        action_mean = norms["action.mean"].to(device)
        action_std = norms["action.std"].to(device)
        if (not torch.isfinite(state_std).all() or (state_std <= 0).any()
                or not torch.isfinite(action_std).all() or (action_std <= 0).any()):
            raise ValueError("Invalid saved normalization")
        states = np.load(args.cache / "states.npy", mmap_mode="r")
        actions = np.load(args.cache / "actions.npy", mmap_mode="r")
        vectors = np.load(args.cache / "task_vectors.npy", mmap_mode="r")
        if states.shape != (len(episodes), 32) or vectors.shape != (len(episodes), 10):
            raise ValueError("Cache does not have the audited 32D-state/10D-task layout")
        from lerobot_robot_aic.task_encoding import encode_task_vector
        for episode_id in heldout:
            family, card, port, task_valid = task_fields(metadata[episode_id])
            expected = encode_task_vector(task_family=family, target_card_index=card,
                                          target_port_index=port, target_card_valid=task_valid)
            indices = selected[episodes[selected] == episode_id]
            if not np.all(vectors[indices] == expected):
                raise ValueError(f"Sampled task vectors disagree with metadata in episode {episode_id}")
        images = load_cached_images(args.cache, cache)
        if set(images) != set(policy.config.image_features):
            raise ValueError("Checkpoint cameras differ from cache cameras")
        mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1)
        target_order = contract["image_channel_order"]
        if cache.get("image_channel_order") not in {"rgb", "bgr"} or target_order not in {"rgb", "bgr"}:
            raise ValueError("Unaudited image channel convention")
        ends = {i: int(np.flatnonzero(episodes == i)[-1]) for i in heldout}
        executed = min(4, int(policy.config.chunk_size))
        started = time.monotonic()

        def inputs(indices, camera_indices):
            raw = np.array(states[indices], dtype=np.float32)
            sign = 3 if contract.get("quaternion_sign") == "x" else 6
            raw[raw[:, sign] < 0, 3:7] *= -1
            value = np.concatenate([raw, np.array(vectors[indices])], axis=1)
            if contract.get("include_elapsed_sim_time"):
                elapsed = np.minimum(timestamps[indices], contract["time_clip_sec"])
                value = np.column_stack([value, elapsed]).astype(np.float32)
            value[:, training.get("excluded_state_indices", [])] = 0
            if value.shape[1] != state_mean.numel():
                raise ValueError("Assembled state width differs from checkpoint normalization")
            batch = {"observation.state": (torch.as_tensor(value, device=device) - state_mean) / state_std}
            for key, pixels in prepare_camera_arrays(images, camera_indices):
                image = torch.as_tensor(pixels, device=device).float() / 255
                if cache["image_channel_order"] != target_order:
                    image = image[:, [2, 1, 0]]
                batch[key] = (image - mean) / std
            return batch

        def predict(batch):
            amp = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()
            with torch.inference_mode(), amp:
                prediction = policy.predict_action_chunk(batch)[:, :executed]
                prediction = prediction.float() * action_std + action_mean
            value = prediction.cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError("Checkpoint produced nonfinite commands")
            return value

        def add_errors(offsets, name, prediction, target, reference, valid):
            change_t, change_r = pose_errors(prediction, reference)
            error_t, error_r = pose_errors(prediction, target) if name != "task_swap" else (None, None)
            for j, offset in enumerate(offsets):
                row = records[offset]
                row[f"{name}_first_change_mm"] = float(change_t[j, 0])
                row[f"{name}_first_change_deg"] = float(change_r[j, 0])
                if error_t is not None:
                    row[f"{name}_first_error_mm"] = float(error_t[j, 0])
                    row[f"{name}_first_error_deg"] = float(error_r[j, 0])
                    row[f"{name}_executed_mean_error_mm"] = float(error_t[j][valid[j]].mean())
                    row[f"{name}_executed_mean_error_deg"] = float(error_r[j][valid[j]].mean())
                    if name != "baseline":
                        # Pair against the same frame: some image-shuffle donors
                        # are unavailable, so pooled baseline means can differ.
                        for units in ("mm", "deg"):
                            row[f"{name}_paired_first_error_increase_{units}"] = (
                                row[f"{name}_first_error_{units}"] - row[f"baseline_first_error_{units}"])

        for start in range(0, len(selected), args.batch_size):
            offsets = np.arange(start, min(start + args.batch_size, len(selected)))
            idx = selected[offsets]
            targets_idx = idx[:, None] + np.arange(executed)
            last = np.asarray([ends[int(episodes[i])] for i in idx])[:, None]
            valid = targets_idx <= last
            targets_idx = np.minimum(targets_idx, last)
            flat = targets_idx.reshape(-1)
            target = absolute_targets(states[flat], actions[flat]).reshape(len(idx), executed, 6)
            batch = inputs(idx, idx)
            baseline = predict(batch)
            add_errors(offsets, "baseline", baseline, target, baseline, valid)
            blank = dict(batch)
            for key in images:
                blank[key] = torch.zeros_like(batch[key])
            add_errors(offsets, "blank_images", predict(blank), target, baseline, valid)
            matched = donors[offsets] >= 0
            if matched.any():
                donor_input = inputs(idx[matched], donors[offsets][matched])
                add_errors(offsets[matched], "shuffled_images", predict(donor_input),
                           target[matched], baseline[matched], valid[matched])
            can_swap = np.asarray([swaps[int(episodes[i])] is not None for i in idx])
            if can_swap.any():
                swap_input = {key: value[can_swap].clone() for key, value in batch.items()}
                alternative = torch.as_tensor(np.stack([swaps[int(episodes[i])] for i in idx[can_swap]]), device=device)
                swap_input["observation.state"][:, 32:42] = (alternative - state_mean[32:42]) / state_std[32:42]
                add_errors(offsets[can_swap], "task_swap", predict(swap_input),
                           target[can_swap], baseline[can_swap], valid[can_swap])
            print(f"Analyzed {start + len(idx)}/{len(selected)} held-out frames", flush=True)
        report.update(device=args.device, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
                      precision="bf16" if args.bf16 else "fp32", elapsed_seconds=time.monotonic() - started,
                      executed_actions_evaluated=executed,
                      parameter_count=sum(p.numel() for p in policy.parameters()),
                      summary=summarize(records, metadata))
        weights = args.checkpoint / "model.safetensors"
        if weights.is_file():
            with weights.open("rb") as handle:
                report["checkpoint_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), "frames": len(selected),
                      "episodes": len(heldout), "shuffle_matches": int((donors >= 0).sum())}), flush=True)


if __name__ == "__main__":
    main()
