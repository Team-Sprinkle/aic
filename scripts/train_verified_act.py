#!/usr/bin/env python3
"""Train the installed LeRobot ACT model from a verified, decoded camera cache."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from act_cache import load_cached_images, prepare_camera_arrays
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
from lerobot_robot_aic.act_backbone import (apply_backbone_geometry, backbone_geometry, load_act_policy,
                                           read_backbone_geometry, save_backbone_geometry)


def load_task_inputs(cache_root, cache, episodes):
    """Fail closed when per-frame conditioning disagrees with episode metadata."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils/lerobot_robot_aic"))
    from lerobot_robot_aic.task_encoding import encode_task_vector, task_encoding_schema
    schema = task_encoding_schema()
    recorded = cache.get("task_encoding", {})
    if recorded.get("dim") != schema["dim"] or recorded.get("names") != schema["names"]:
        raise ValueError("Task conditioning requires the canonical audited 10D cache schema")
    vectors = np.load(cache_root / "task_vectors.npy")
    if vectors.shape != (len(episodes), schema["dim"]) or not np.isfinite(vectors).all():
        raise ValueError("Task vector array must contain one finite canonical vector per frame")
    metadata = {int(episode["episode_index"]): episode for episode in cache["episodes"]}
    for episode_id in np.unique(episodes):
        episode = metadata[int(episode_id)]
        task = episode.get("task", episode)
        expected = encode_task_vector(**{field: task[field] for field in
            ("task_family", "target_port_index", "target_card_index", "target_card_valid")})
        if not np.all(vectors[episodes == episode_id] == expected):
            raise ValueError(f"Task conditioning disagrees with metadata for episode {episode_id}")
    return vectors.astype(np.float32, copy=False), schema


def distributed_sample_indices(rng, batch_size, rank, world_size, groups, probabilities,
                               terminal_groups, terminal_probability):
    """Shard one deterministic global draw; ranks never replay the same RNG batch.

    Sampling is with replacement, as in the original trainer, so independent
    draws may still legitimately select the same source frame.
    """
    selected = sample_training_indices(rng, batch_size * world_size, groups, probabilities,
                                       terminal_groups, terminal_probability)
    return selected[rank * batch_size:(rank + 1) * batch_size]


def distributed_stop_flags(stopping, deadline, started, max_minutes, device, distributed):
    flags = torch.tensor([stopping, time.time() >= deadline,
                          time.monotonic() - started >= max_minutes * 60],
                         dtype=torch.int32, device=device)
    if distributed:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)
    return [bool(value) for value in flags.tolist()]


def stable_gaussian_kl(mu, log_variance):
    """KL to a unit Gaussian without bfloat16 cancellation near unit variance."""
    mu = mu.float()
    log_variance = log_variance.float()
    return .5 * (mu.square() + torch.expm1(log_variance) - log_variance).sum(-1).mean()


def sample_training_indices(rng, size, groups, probabilities, terminal_groups, terminal_probability):
    """Sample only eligible training groups, optionally emphasizing contact ends."""
    selected_groups = rng.choice(len(groups), size=size, p=probabilities)
    terminal = rng.random(size) < terminal_probability
    selected = np.empty(size, dtype=np.int64)
    for group_index, group in enumerate(groups):
        for at_end, candidates in [(False, group), (True, terminal_groups[group_index])]:
            mask = (selected_groups == group_index) & (terminal == at_end)
            if mask.any():
                selected[mask] = rng.choice(candidates, int(mask.sum()))
    return selected


def rebase_act_normalization(weights, old_stats, new_stats):
    """Preserve physical input/output units when a warm start changes statistics."""
    result = {key: value.detach().clone() for key, value in weights.items()}
    reference = result["model.action_head.weight"]
    parameters = {}
    for feature in ("observation.state", "action"):
        values = [torch.as_tensor(stats[feature][key], device=reference.device, dtype=reference.dtype)
                  for stats in (old_stats, new_stats) for key in ("mean", "std")]
        if (any(value.ndim != 1 or not torch.isfinite(value).all() for value in values)
                or len({tuple(value.shape) for value in values}) != 1
                or any((values[index] <= 0).any() for index in (1, 3))):
            raise ValueError(f"Invalid normalization statistics for {feature}")
        parameters[feature] = values
    for prefix, feature in (("model.encoder_robot_state_input_proj", "observation.state"),
                            ("model.vae_encoder_robot_state_input_proj", "observation.state"),
                            ("model.vae_encoder_action_input_proj", "action")):
        if prefix + ".weight" not in result:
            continue
        old_mean, old_std, new_mean, new_std = parameters[feature]
        weight = result[prefix + ".weight"]
        result[prefix + ".bias"] += weight @ ((new_mean - old_mean) / old_std)
        result[prefix + ".weight"] = weight * (new_std / old_std)[None, :]
    old_mean, old_std, new_mean, new_std = parameters["action"]
    result["model.action_head.weight"] *= (old_std / new_std)[:, None]
    result["model.action_head.bias"] = (result["model.action_head.bias"] * old_std + old_mean - new_mean) / new_std
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nic-counts", type=int, nargs="+", default=[1])
    parser.add_argument("--train-episode-indices", type=int, nargs="+",
                        help="Optional subset of the existing training split, for an explicitly labeled overfit diagnostic.")
    parser.add_argument("--training-probe-frames", type=int, default=0,
                        help="Report fitting error on this many training frames separately from all held-out metrics.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Per-GPU batch size; torchrun global batch is this value times WORLD_SIZE.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--image-load-workers", type=int, default=0,
                        help="CPU camera collation threads per rank; 0/1 is serial, 3 loads three cameras concurrently.")
    parser.add_argument("--profile-timing", action="store_true",
                        help="Synchronize CUDA at update phase boundaries and report preparation/forward/backward/optimizer timing.")
    parser.add_argument("--sampling-balance", choices=["frame", "task", "task_count"], default="frame",
                        help="Optionally balance task families or task/card-count groups, then sample frames within each group.")
    parser.add_argument("--include-task", action="store_true",
                        help="Append the canonical audited 10D task vector before optional elapsed time.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--lr-final-ratio", type=float, default=1.,
                        help="Cosine decay multiplier at the requested final update; 1 keeps constant learning rates.")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=3)
    parser.add_argument("--vision-backbone", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--dilate-backbone", action="store_true", help="Use stride-16 ResNet50 features for finer image detail.")
    parser.add_argument("--backbone-output-stride", type=int, choices=[16, 32],
                        help="Optional ResNet18 layer4 stride override without dilation; omitted inherits warm-start geometry, or uses stride32 for a fresh ResNet18.")
    parser.add_argument("--kl-weight", type=float, default=10.)
    parser.add_argument("--use-vae", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable the optional ACT VAE for a deterministic supervised objective.")
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-frames", type=int, default=512)
    parser.add_argument("--max-minutes", type=float, default=45.)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--rebase-normalization", action="store_true",
                        help="Adjust ACT input/output projections to preserve physical units under new training statistics; requires unchanged feature semantics.")
    parser.add_argument("--expand-time-input", action="store_true",
                        help="When adding elapsed time to an existing ACT checkpoint, append zero columns to its state projections.")
    parser.add_argument("--resize-action-chunk", action="store_true",
                        help="Warm start a different chunk length by linearly resampling decoder queries and regenerating fixed VAE positions.")
    parser.add_argument("--state-std-floor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--color-jitter", type=float, default=0.)
    parser.add_argument("--image-channel-order", choices=["rgb", "bgr"], default=None,
                        help="Physical channel convention stored in cache images; defaults to audited cache metadata.")
    parser.add_argument("--canonical-rgb", action="store_true",
                        help="Convert legacy BGR cache pixels to RGB before ImageNet preprocessing; export then expects RGB.")
    parser.add_argument("--state-features", choices=["full", "no_controller_error", "pose_joints_wrench", "pose_only", "wrench_only", "vision_only"], default="full",
                        help="Ablate command-related state channels; excluded projection columns are saved as exact zeros.")
    parser.add_argument("--action-representation", choices=["delta_pose", "absolute_pose"], default="delta_pose",
                        help="Preserve TCP-relative commands or express their target pose in base_link.")
    parser.add_argument("--include-time", action="store_true", help="Append elapsed episode simulation time to ACT state.")
    parser.add_argument("--mask-time", action="store_true", help="Retain the checkpoint's time-input dimension but zero its projection columns during fine tuning.")
    parser.add_argument("--corrective-sampling-probability", type=float, default=0.,
                        help="Fraction of training samples from verified corrective episodes; zero uses the ordinary full-cache sampler.")
    parser.add_argument("--terminal-sampling-probability", type=float, default=0.,
                        help="Within each training source group, sample this fraction from episode ends.")
    parser.add_argument("--terminal-window-sec", type=float, default=5.,
                        help="Duration of the contact-end window, measured using each source's recorded timestamps.")
    parser.add_argument("--time-clip-sec", type=float, default=40.)
    parser.add_argument("--quaternion-sign", choices=["w", "x"], default="w",
                        help="Choose a quaternion hemisphere; x is continuous around this collection's initial half-turn about X.")
    parser.add_argument("--state-position-noise", type=float, default=0., help="Training-only TCP position noise, metres; absolute targets only.")
    parser.add_argument("--state-rotation-noise", type=float, default=0., help="Training-only TCP rotation noise, radians per axis; absolute targets only.")
    parser.add_argument("--goal-loss-weight", type=float, default=0.,
                        help="Optional training-only final achieved TCP pose supervision of the ACT encoder, using zero VAE latent.")
    parser.add_argument("--goal-representation", choices=["absolute_pose", "relative_pose"], default="absolute_pose")
    parser.add_argument("--goal-images-only", action="store_true",
                        help="Zero proprioception in the auxiliary pass so goal prediction must use camera features.")
    parser.add_argument("--deadline-utc", default=None,
                        help="Optional absolute ISO-8601 deadline; --max-minutes still bounds each run.")
    args = parser.parse_args()
    geometry_config = {"vision_backbone": args.vision_backbone,
                       "replace_final_stride_with_dilation": args.dilate_backbone}
    try:
        geometry = (read_backbone_geometry(args.init_checkpoint, geometry_config)
                    if args.init_checkpoint and args.backbone_output_stride is None
                    else backbone_geometry(geometry_config, args.backbone_output_stride))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    is_primary = rank == 0
    if distributed and args.goal_loss_weight:
        raise ValueError("DDP does not support the experimental two-pass auxiliary goal head; set --goal-loss-weight 0")
    if args.sampling_balance != "frame" and args.corrective_sampling_probability:
        raise ValueError("Choose task balance or the legacy corrective mixture, not both")
    if args.rebase_normalization and (not args.init_checkpoint or args.expand_time_input or args.goal_loss_weight):
        raise ValueError("Normalization rebasing requires a checkpoint, unchanged state width, and no auxiliary goal head")
    if args.expand_time_input and (not args.init_checkpoint or not args.include_time):
        raise ValueError("Time-input expansion requires a checkpoint and --include-time")
    if args.resize_action_chunk and not args.init_checkpoint:
        raise ValueError("Chunk resizing requires an initialization checkpoint")
    if args.mask_time and not args.include_time:
        raise ValueError("--mask-time requires --include-time to preserve the checkpoint dimension")
    if not 0 <= args.corrective_sampling_probability <= 1:
        raise ValueError("Corrective sampling probability must be in [0, 1]")
    if not 0 <= args.terminal_sampling_probability <= 1 or args.terminal_window_sec <= 0:
        raise ValueError("Terminal sampling needs probability in [0, 1] and a positive window")
    if args.dilate_backbone and args.vision_backbone != "resnet50":
        raise ValueError("Dilated BasicBlock ResNet18 is unsupported by torchvision; choose resnet50")
    if not 0 < args.lr_final_ratio <= 1:
        raise ValueError("Final learning-rate ratio must be in (0, 1]")
    if min(args.state_position_noise, args.state_rotation_noise, args.goal_loss_weight) < 0:
        raise ValueError("State noise standard deviations must be nonnegative")
    if (args.state_position_noise or args.state_rotation_noise) and args.action_representation != "absolute_pose":
        raise ValueError("State noise currently requires unchanged absolute action targets")
    from datetime import datetime
    deadline = datetime.fromisoformat(args.deadline_utc).timestamp() if args.deadline_utc else float("inf")
    if deadline <= time.time():
        raise ValueError("Training deadline has already passed; no model was trained")
    if args.steps < 1 or args.max_minutes <= 0:
        raise ValueError("Training needs positive requested updates and a positive time limit")
    if min(args.batch_size, args.cpu_threads, args.validate_every, args.save_every, args.validation_frames) < 1:
        raise ValueError("Batch, thread, validation, and checkpoint settings must be positive")
    if args.image_load_workers < 0:
        raise ValueError("Image loading worker count must be nonnegative")
    if not torch.cuda.is_available():
        raise RuntimeError("This bounded trainer expects an explicitly selected CUDA GPU")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root = args.output_dir.resolve()
    if is_primary:
        root.mkdir(parents=True, exist_ok=False)
        (root / "trainer_source.py").write_text(Path(__file__).read_text())
        (root / "cache_loader_source.py").write_text(Path(__file__).with_name("act_cache.py").read_text())
        (root / "backbone_source.py").write_text((Path(__file__).resolve().parents[1] /
            "aic_utils/lerobot_robot_aic/lerobot_robot_aic/act_backbone.py").read_text())
    if distributed:
        dist.barrier()
    cache = json.loads((args.cache / "cache.json").read_text())
    # A relative TCP command is tied to the observation at which it was
    # recorded. Causal holding into synthetic 20 Hz rows repeats a physical
    # displacement that the expert did not issue. The strict pilot must use
    # its audited native observation-command pairs for delta training.
    if (args.action_representation == "delta_pose" and cache.get("strict_pilot_split")
            and not cache.get("native_observation_only")):
        raise ValueError("Strict-pilot TCP delta training requires native observation-command rows")
    if cache.get("native_observation_only") and (
            args.action_representation != "delta_pose"
            or cache.get("action_semantics") != "Native observed TCP-relative pose command; unchanged"):
        raise ValueError("Native TCP-delta cache has an incompatible action contract")
    source_dataset = Path(cache["source_dataset"]) if cache.get("source_dataset") else None
    if source_dataset is not None:
        if (source_dataset / "BUILD_IN_PROGRESS.json").exists():
            raise ValueError("Canonical expert materialization is incomplete; do not train a partial collection")
        if source_dataset.name == "expert_verified" and not (source_dataset / "manifest.json").is_file():
            raise ValueError("Canonical expert collection has no completed manifest")
    args.image_channel_order = args.image_channel_order or cache.get("image_channel_order")
    if args.image_channel_order not in {"rgb", "bgr"}:
        raise ValueError("Cache channel order is unaudited; specify --image-channel-order after checking source images")
    states = np.load(args.cache / "states.npy")
    if states.ndim != 2 or states.shape[1] != 32:
        raise ValueError("Verified ACT caches must store exactly 32 robot-state fields; task/time are separate")
    episodes = np.load(args.cache / "episodes.npy")
    if args.quaternion_sign == "x":
        states[states[:, 3] < 0, 3:7] *= -1
    source_poses = states[:, :7].copy()
    actions = np.load(args.cache / "actions.npy")
    if args.action_representation == "absolute_pose":
        from scipy.spatial.transform import Rotation
        current_rot = Rotation.from_quat(states[:, 3:7])
        target_position = states[:, :3] + current_rot.apply(actions[:, :3])
        target_quat = (current_rot * Rotation.from_rotvec(actions[:, 3:6])).as_quat()
        # This collection stays near a half-turn about X. Fix X's sign rather
        # than W's, avoiding a discontinuity when the target crosses 180 deg.
        target_quat[target_quat[:, 0] < 0] *= -1
        norm = np.linalg.norm(target_quat[:, :3], axis=1)
        angle = 2 * np.arctan2(norm, target_quat[:, 3])
        target_rotvec = target_quat[:, :3] * (angle / np.maximum(norm, 1e-12))[:, None]
        actions = np.concatenate([target_position, target_rotvec], axis=1).astype("float32")
    masked_indices = (list(range(13, 19)) if args.state_features == "no_controller_error" else
                      list(range(7, 19)) if args.state_features == "pose_joints_wrench" else
                      list(range(7, 32)) if args.state_features == "pose_only" else
                      list(range(0, 26)) if args.state_features == "wrench_only" else
                      list(range(0, 32)) if args.state_features == "vision_only" else [])
    # Modify this in-memory copy only. Zero input columns plus zero projection
    # weights make the saved ACT independent of excluded runtime channels.
    states[:, masked_indices] = 0.
    task_schema = None
    task_indices = []
    if args.include_task:
        task_vectors, task_schema = load_task_inputs(args.cache, cache, episodes)
        task_indices = list(range(states.shape[1], states.shape[1] + task_vectors.shape[1]))
        states = np.concatenate([states, task_vectors], axis=1)
    base_state_dim = states.shape[1]
    if args.include_time:
        if args.time_clip_sec <= 0:
            raise ValueError("Time clipping bound must be positive")
        elapsed = np.minimum(np.load(args.cache / "timestamps.npy"), args.time_clip_sec)
        states = np.column_stack([states, elapsed]).astype("float32")
        if args.mask_time:
            masked_indices.append(base_state_dim)
            states[:, base_state_dim] = 0.
    images = load_cached_images(args.cache, cache)
    image_executor = (ThreadPoolExecutor(max_workers=min(args.image_load_workers, len(images)),
                                         thread_name_prefix="act-camera")
                      if args.image_load_workers > 1 else None)
    train_eps, val_eps = [], []
    for count in args.nic_counts:
        split = cache["splits_by_nic_count"][str(count)]
        train_eps.extend(split["train"]); val_eps.extend(split["validation"])
    train_eps, val_eps = sorted(set(train_eps)), sorted(set(val_eps))
    if args.train_episode_indices is not None:
        if not set(args.train_episode_indices) <= set(train_eps):
            raise ValueError("The requested episode subset must belong to the existing training split")
        train_eps = sorted(set(args.train_episode_indices))
    if set(train_eps) & set(val_eps):
        raise ValueError("Training and validation episode IDs overlap")
    train_indices = np.flatnonzero(np.isin(episodes, train_eps))
    val_indices = np.flatnonzero(np.isin(episodes, val_eps))
    corrective_eps = [e["episode_index"] for e in cache["episodes"]
                      if e.get("source_kind") == "score_verified_privileged_corrective_labels" and e["episode_index"] in train_eps]
    corrective_indices = train_indices[np.isin(episodes[train_indices], corrective_eps)]
    legacy_indices = train_indices[~np.isin(episodes[train_indices], corrective_eps)]
    if args.corrective_sampling_probability and (not len(corrective_indices) or not len(legacy_indices)):
        raise ValueError("Corrective mixture sampling requires nonempty verified corrective and legacy training groups")
    if not len(train_indices) or not len(val_indices):
        raise ValueError("Both source-episode splits must be nonempty")
    end_index = np.empty(len(episodes), dtype="int64")
    for ep in np.unique(episodes):
        indices = np.flatnonzero(episodes == ep); end_index[indices] = indices[-1]
    sampling_groups = [legacy_indices, corrective_indices] if args.corrective_sampling_probability else [train_indices]
    sampling_probabilities = ([1 - args.corrective_sampling_probability, args.corrective_sampling_probability]
                              if args.corrective_sampling_probability else [1.])
    sampling_group_labels = ["legacy", "corrective"] if args.corrective_sampling_probability else ["all"]
    episode_metadata = {int(episode["episode_index"]): episode for episode in cache["episodes"]}
    def episode_group(episode):
        task = episode.get("task", episode)
        family = task.get("task_family", episode.get("task_family", "unknown"))
        if family not in {"sfp_to_nic", "sc_to_sc"}:
            raise ValueError("Task-group reporting/sampling requires canonical task_family metadata")
        return family, int(episode["nic_count"])
    if args.sampling_balance != "frame":
        group_episode_ids = {}
        for episode_id in train_eps:
            family, count = episode_group(episode_metadata[episode_id])
            name = family if args.sampling_balance == "task" else f"{family}/nic_{count}"
            group_episode_ids.setdefault(name, []).append(episode_id)
        sampling_group_labels = sorted(group_episode_ids)
        sampling_groups = [train_indices[np.isin(episodes[train_indices], group_episode_ids[name])]
                           for name in sampling_group_labels]
        sampling_probabilities = [1 / len(sampling_groups)] * len(sampling_groups)
    source_times = np.load(args.cache / "timestamps.npy")
    terminal_groups = [group[source_times[end_index[group]] - source_times[group] <= args.terminal_window_sec]
                       for group in sampling_groups]
    goals = None
    if args.goal_loss_weight:
        from scipy.spatial.transform import Rotation
        final_poses = source_poses[end_index]
        quat = Rotation.from_quat(final_poses[:, 3:7]).as_quat()
        quat[quat[:, 0] < 0] *= -1
        norm = np.linalg.norm(quat[:, :3], axis=-1)
        rotvec = quat[:, :3] * (2 * np.arctan2(norm, quat[:, 3]) / np.maximum(norm, 1e-12))[:, None]
        goals = np.concatenate([final_poses[:, :3], rotvec], axis=-1).astype("float32")
        if args.goal_representation == "relative_pose":
            current = Rotation.from_quat(source_poses[:, 3:7])
            goal_translation = current.apply(final_poses[:, :3] - source_poses[:, :3], inverse=True)
            goal_rotation = (current.inv() * Rotation.from_quat(final_poses[:, 3:7])).as_rotvec()
            goals = np.concatenate([goal_translation, goal_rotation], axis=-1).astype("float32")
        goal_mean = goals[train_indices].mean(0)
        goal_std = np.maximum(goals[train_indices].std(0), [.001] * 3 + [.01] * 3).astype("float32")
        if is_primary:
            np.savez(root / "goal_normalization.npz", mean=goal_mean, std=goal_std)
    state_mean = states[train_indices].mean(0)
    state_std = states[train_indices].std(0)
    floors = np.full(states.shape[1], 1e-8, dtype="float32")
    if args.state_std_floor:
        floors[:] = .001
        floors[10:13] = .005; floors[19:26] = .01
        floors[26:29] = .1; floors[29:32] = .01
    state_std = np.maximum(state_std, floors)
    state_std[masked_indices] = 1.
    state_mean[task_indices] = 0.
    state_std[task_indices] = 1.
    action_mean = actions[train_indices].mean(0)
    action_std = np.maximum(actions[train_indices].std(0), 1e-4)
    stats = {"observation.state": {"mean": state_mean, "std": state_std},
             "action": {"mean": action_mean, "std": action_std}}
    for key in images:
        stats[key] = {"mean": np.asarray([.485, .456, .406], dtype="float32").reshape(3, 1, 1),
                      "std": np.asarray([.229, .224, .225], dtype="float32").reshape(3, 1, 1)}
    height, width, channels = cache["image_shape_hwc"]
    config = ACTConfig(device=str(device), input_features={
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(states.shape[1],)),
        **{key: PolicyFeature(type=FeatureType.VISUAL, shape=(channels, height, width)) for key in images}},
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(actions.shape[1],))},
        chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
        dim_model=args.dim_model, dim_feedforward=args.dim_model * 4,
        n_encoder_layers=args.encoder_layers, n_decoder_layers=1, n_vae_encoder_layers=2,
        use_vae=args.use_vae, kl_weight=args.kl_weight, dropout=args.dropout,
        vision_backbone=args.vision_backbone,
        pretrained_backbone_weights="ResNet50_Weights.IMAGENET1K_V2" if args.vision_backbone == "resnet50" else "ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation=args.dilate_backbone,
        optimizer_lr=args.lr, optimizer_lr_backbone=args.backbone_lr)
    policy = apply_backbone_geometry(ACTPolicy(config), geometry).cuda()
    if args.init_checkpoint:
        previous_action_config = args.init_checkpoint / "aic_action_config.json"
        previous_representation = (json.loads(previous_action_config.read_text())["action_representation"]
                                   if previous_action_config.is_file() else "delta_pose")
        if previous_representation != args.action_representation:
            raise ValueError("Cannot initialize from a checkpoint with different action semantics")
        previous = load_act_policy(args.init_checkpoint, local_files_only=True)
        previous_weights = previous.state_dict()
        if args.rebase_normalization:
            from safetensors.torch import load_file
            previous_contract = json.loads(previous_action_config.read_text())
            previous_training = json.loads((args.init_checkpoint.parents[2] / "training_config.json").read_text())
            expected_contract = {"include_elapsed_sim_time": args.include_time,
                                 "elapsed_time_projection_masked": args.mask_time,
                                 "quaternion_sign": args.quaternion_sign,
                                 "image_channel_order": "rgb" if args.canonical_rgb else args.image_channel_order}
            if args.include_time:
                expected_contract["time_clip_sec"] = args.time_clip_sec
            if args.include_task:
                expected_contract["task_conditioned"] = True
                expected_contract["task_encoding"] = task_schema
            if (any(previous_contract.get(key) != value for key, value in expected_contract.items())
                    or previous_training["excluded_state_indices"] != masked_indices):
                raise ValueError("Normalization rebasing requires unchanged feature semantics and masks")
            tensors = load_file(str(args.init_checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
            previous_stats = {feature: {key: tensors[feature + "." + key] for key in ("mean", "std")}
                              for feature in ("observation.state", "action")}
            previous_weights = rebase_act_normalization(previous_weights, previous_stats, stats)
        if args.resize_action_chunk:
            key = "model.decoder_pos_embed.weight"
            queries = previous_weights[key]
            previous_weights[key] = torch.nn.functional.interpolate(
                queries.T.unsqueeze(0), size=args.chunk_size, mode="linear", align_corners=True).squeeze(0).T
            key = "model.vae_encoder_pos_enc"
            if key in previous_weights:
                previous_weights[key] = policy.state_dict()[key]
        if args.expand_time_input:
            previous_features = previous.config.robot_state_feature.shape[0]
            if previous_features != base_state_dim or states.shape[1] != previous_features + 1:
                raise ValueError("Only appending one time coordinate to the same base state is supported")
            for key in ["model.encoder_robot_state_input_proj.weight", "model.vae_encoder_robot_state_input_proj.weight"]:
                if key in previous_weights:
                    previous_weights[key] = torch.nn.functional.pad(previous_weights[key], (0, 1))
        policy.load_state_dict(previous_weights, strict=True)
        del previous
    state_projections = [policy.model.encoder_robot_state_input_proj]
    if args.use_vae:
        state_projections.append(policy.model.vae_encoder_robot_state_input_proj)
    with torch.no_grad():
        for projection in state_projections:
            projection.weight[:, masked_indices] = 0.
    pre, post = make_pre_post_processors(policy_cfg=config, dataset_stats=stats)
    optim_groups = policy.get_optim_params()
    optimization_parameters = list(policy.parameters())
    goal_head = None
    captured_encoder = {}
    if goals is not None:
        goal_head = torch.nn.Sequential(torch.nn.Linear(args.dim_model, args.dim_model), torch.nn.ReLU(),
                                        torch.nn.Linear(args.dim_model, 6)).cuda()
        if args.init_checkpoint and (args.init_checkpoint / "auxiliary_goal_head.pt").is_file():
            previous_step = json.loads((args.init_checkpoint / "training_step.json").read_text())
            previous_args = previous_step["training_config"]["args"]
            if (previous_args.get("goal_representation", "absolute_pose") == args.goal_representation
                    and previous_args.get("goal_images_only", False) == args.goal_images_only):
                goal_head.load_state_dict(torch.load(args.init_checkpoint / "auxiliary_goal_head.pt", weights_only=True))
        optim_groups.append({"params": list(goal_head.parameters()), "lr": args.lr})
        optimization_parameters.extend(goal_head.parameters())
        def capture_encoder(module, inputs, output):
            captured_encoder["value"] = output
        policy.model.encoder.register_forward_hook(capture_encoder)
        goal_mean_t = torch.as_tensor(goal_mean, device="cuda")
        goal_std_t = torch.as_tensor(goal_std, device="cuda")
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        args.lr_final_ratio + (1 - args.lr_final_ratio) * .5 * (1 + math.cos(math.pi * min(step / args.steps, 1.))))
    train_model = (DistributedDataParallel(policy.model, device_ids=[local_rank], output_device=local_rank,
                                          broadcast_buffers=False, gradient_as_bucket_view=True)
                   if distributed else policy.model)
    # Model initialization is identical across ranks; stochastic VAE/dropout and
    # augmentation streams are independent after the DDP parameter broadcast.
    torch.manual_seed(args.seed + rank)
    state_mean_t = torch.as_tensor(state_mean, device="cuda")
    state_std_t = torch.as_tensor(state_std, device="cuda")
    action_mean_t = torch.as_tensor(action_mean, device="cuda")
    action_std_t = torch.as_tensor(action_std, device="cuda")
    image_mean = torch.tensor([.485, .456, .406], device="cuda").view(1, 3, 1, 1)
    image_std = torch.tensor([.229, .224, .225], device="cuda").view(1, 3, 1, 1)
    horizon = np.arange(args.chunk_size)
    rng = np.random.default_rng(args.seed)
    validation_indices = np.random.default_rng(913).choice(val_indices, min(args.validation_frames, len(val_indices)), replace=False)
    corrective_val_eps = [e["episode_index"] for e in cache["episodes"]
                          if e.get("source_kind") == "score_verified_privileged_corrective_labels" and e["episode_index"] in val_eps]
    corrective_val_indices = val_indices[np.isin(episodes[val_indices], corrective_val_eps)]
    corrective_validation_indices = np.random.default_rng(914).choice(
        corrective_val_indices, min(args.validation_frames, len(corrective_val_indices)), replace=False)
    sampled_training_eps = corrective_eps if args.corrective_sampling_probability == 1 else train_eps
    probe_candidates = corrective_indices if args.corrective_sampling_probability == 1 else train_indices
    training_probe_indices = np.random.default_rng(915).choice(
        probe_candidates, min(max(0, args.training_probe_frames), len(probe_candidates)), replace=False)
    task_validation_groups = []
    if args.include_task:
        grouped_validation = {}
        for episode_id in val_eps:
            family, count = episode_group(episode_metadata[episode_id])
            grouped_validation.setdefault(f"{family}/nic_{count}", []).append(episode_id)
        group_rng = np.random.default_rng(916)
        for name, group_episodes in sorted(grouped_validation.items()):
            candidates = val_indices[np.isin(episodes[val_indices], group_episodes)]
            selected = group_rng.choice(candidates, min(128, args.validation_frames, len(candidates)), replace=False)
            task_validation_groups.append((name, selected, group_episodes))
    metadata = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "train_episodes": train_eps, "validation_episodes": val_eps,
                "training_frames": len(train_indices), "validation_frames": len(val_indices),
                "training_kind": ("Supervised ACT with LeRobot's L1 plus VAE KL objective; no RL updates" if args.use_vae
                                  else "Supervised ACT with LeRobot's L1 objective and VAE disabled; no RL updates"),
                "backbone_initialization": (f"Warm start from {args.init_checkpoint}; trainable" if args.init_checkpoint
                                            else f"ImageNet {args.vision_backbone}; trainable"),
                "optimizer_initialization": "Fresh AdamW optimizer and schedule; checkpoint optimizer state is not loaded",
                "warm_start_normalization": ("No ACT warm start" if not args.init_checkpoint else
                                             "Input/output projections rebased to the new statistics" if args.rebase_normalization else
                                             "No rebasing; loaded weights use this run's newly fitted statistics"),
                "normalization": "Training episodes only for state/action; fixed ImageNet image normalization",
                "source_cache": str(args.cache.resolve()), "physical_gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES")}
    import importlib.metadata
    import platform
    metadata["software"] = {"python": platform.python_version(), "torch": torch.__version__,
                            "lerobot": importlib.metadata.version("lerobot"), "numpy": np.__version__,
                            "gpu": torch.cuda.get_device_name(0)}
    metadata["excluded_state_indices"] = masked_indices
    metadata["backbone_geometry"] = geometry
    metadata["task_conditioned"] = args.include_task
    metadata["task_encoding"] = task_schema
    metadata["task_vector_indices"] = [32, 42] if args.include_task else None
    metadata["task_normalization"] = "Identity: categorical coordinates have mean 0 and std 1" if args.include_task else None
    metadata["distributed"] = {"world_size": world_size, "per_gpu_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size, "backend": "nccl" if distributed else None,
        "sampling": "One shared seeded global draw sharded across ranks; source sampling is with replacement",
        "validation": "Rank 0 evaluates the original module while other ranks wait at synchronized barriers",
        "precision": "bfloat16 autocast; float32 master parameters and KL"}
    metadata["image_loading"] = {"cpu_camera_workers_per_rank": min(args.image_load_workers, len(images)),
        "source_shards": len(cache.get("image_shards", [])),
        "coalesced_shards_per_camera": {key: len(value.shards) if hasattr(value, "shards") else 1
                                         for key, value in images.items()},
        "cuda_transfer": "Main rank thread only; camera workers collate uint8 CPU arrays"}
    metadata["corrective_training_episodes"] = corrective_eps
    metadata["sampled_training_episodes"] = sampled_training_eps
    metadata["normalization_training_episodes"] = train_eps
    metadata["corrective_validation_episodes"] = corrective_val_eps
    metadata["sampling"] = ("Independent mixture of uniform corrective and uniform legacy training frames" if args.corrective_sampling_probability
                            else f"Uniform {args.sampling_balance} groups, then uniform frames within each group" if args.sampling_balance != "frame"
                            else "Uniform sampling of training frames")
    metadata["terminal_sampling"] = {"probability": args.terminal_sampling_probability,
                                     "window_seconds": args.terminal_window_sec,
                                     "group_frame_counts": [len(group) for group in terminal_groups]}
    metadata["sampling_groups"] = {"balance": args.sampling_balance, "labels": sampling_group_labels,
                                   "probabilities": sampling_probabilities,
                                   "frame_counts": [len(group) for group in sampling_groups]}
    metadata["state_ablation_deployment"] = "Excluded input-projection columns are exactly zero in the saved weights; no runtime flag required."
    metadata["auxiliary_goal_supervision"] = ("Final achieved TCP pose from the same training episode, in the configured coordinate frame; extra encoder pass with zero latent and no action labels; auxiliary head is discarded at deployment." if goals is not None else None)
    metadata["kl_numerics"] = "Float32 mu^2 + expm1(log_variance) - log_variance avoids bfloat16 cancellation near the prior"
    if is_primary:
        (root / "training_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        np.savez(root / "normalization.npz", state_mean=state_mean, state_std=state_std, action_mean=action_mean, action_std=action_std)

    def batch(indices, augment=False):
        target_indices = indices[:, None] + horizon
        pads = target_indices > end_index[indices, None]
        target_indices = np.minimum(target_indices, end_index[indices, None])
        state = torch.as_tensor(states[indices], device="cuda")
        if augment and args.state_position_noise:
            state[:, :3] += torch.randn_like(state[:, :3]) * args.state_position_noise
        if augment and args.state_rotation_noise:
            rotvec = torch.randn_like(state[:, :3]) * args.state_rotation_noise
            angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
            perturbation = torch.cat([rotvec * (torch.sin(angle / 2) / angle.clamp_min(1e-12)), torch.cos(angle / 2)], dim=-1)
            quat = state[:, 3:7].clone()
            xyz = quat[:, 3:] * perturbation[:, :3] + perturbation[:, 3:] * quat[:, :3] + torch.linalg.cross(quat[:, :3], perturbation[:, :3])
            w = quat[:, 3:] * perturbation[:, 3:] - (quat[:, :3] * perturbation[:, :3]).sum(-1, keepdim=True)
            noisy = torch.cat([xyz, w], dim=-1)
            sign_index = 0 if args.quaternion_sign == "x" else 3
            noisy[noisy[:, sign_index] < 0] *= -1
            state[:, 3:7] = noisy
        result = {"observation.state": (state - state_mean_t) / state_std_t,
                  "action": (torch.as_tensor(actions[target_indices], device="cuda") - action_mean_t) / action_std_t,
                  "action_is_pad": torch.as_tensor(pads, device="cuda")}
        if goals is not None:
            result["auxiliary_goal"] = (torch.as_tensor(goals[indices], device="cuda") - goal_mean_t) / goal_std_t
        brightness = None
        if augment and args.color_jitter:
            brightness = 1 + (torch.rand(len(indices), 3, 1, 1, device="cuda") * 2 - 1) * args.color_jitter
        for key, cpu_array in prepare_camera_arrays(images, indices, image_executor):
            tensor = torch.as_tensor(cpu_array, device="cuda").float().div_(255.)
            if args.canonical_rgb and args.image_channel_order == "bgr":
                tensor = tensor[:, [2, 1, 0]]
            if brightness is not None:
                tensor = (tensor * brightness).clamp_(0, 1)
            result[key] = (tensor - image_mean) / image_std
        return result

    def validate(step):
        policy.eval(); absolute = []; squared = []; late = []; zero = []; rotation_errors = []; goal_errors = []
        with torch.inference_mode():
            for start in range(0, len(validation_indices), args.batch_size):
                idx = validation_indices[start:start + args.batch_size]
                data = batch(idx)
                predicted = policy.predict_action_chunk(data)
                if goal_head is not None:
                    if args.goal_images_only:
                        from lerobot.utils.constants import OBS_IMAGES
                        policy.model({"observation.state": torch.zeros_like(data["observation.state"]),
                                      OBS_IMAGES: [data[key] for key in policy.config.image_features]})
                    goal_pred = goal_head(captured_encoder["value"][1]) * goal_std_t + goal_mean_t
                    goal_errors.extend(torch.linalg.vector_norm(goal_pred[:, :3] - torch.as_tensor(goals[idx, :3], device="cuda"), dim=-1).tolist())
                mask = (~data["action_is_pad"]).unsqueeze(-1)
                absolute.append((torch.abs(predicted - data["action"]) * mask).sum().item())
                physical = predicted[:, 0] * action_std_t + action_mean_t
                actual = torch.as_tensor(actions[idx], device="cuda")
                from scipy.spatial.transform import Rotation
                pred_rot = Rotation.from_rotvec(physical[:, 3:6].cpu().numpy())
                true_rot = Rotation.from_rotvec(actual[:, 3:6].cpu().numpy())
                rotation_errors.extend(np.degrees((pred_rot.inv() * true_rot).magnitude()).tolist())
                squared.extend(torch.linalg.vector_norm(physical[:, :3] - actual[:, :3], dim=-1).tolist())
                hold_position = torch.as_tensor(source_poses[idx, :3], device="cuda") if args.action_representation == "absolute_pose" else 0
                zero.extend(torch.linalg.vector_norm(actual[:, :3] - hold_position, dim=-1).tolist())
                for offset, index in enumerate(idx):
                    if end_index[index] - index < 60:
                        late.append(float(torch.linalg.vector_norm(physical[offset, :3] - actual[offset, :3])))
        valid_coordinates = sum(min(args.chunk_size, int(end_index[i] - i + 1)) * actions.shape[1] for i in validation_indices)
        result = {"step": step, "validation_frames_sampled": len(validation_indices),
                  "inference_normalized_l1": sum(absolute) / valid_coordinates,
                  "first_action_translation_error_m": float(np.mean(squared)),
                  "first_action_rotation_error_deg": float(np.mean(rotation_errors)),
                  "first_action_translation_error_p95_m": float(np.percentile(squared, 95)),
                  "auxiliary_goal_translation_error_m": float(np.mean(goal_errors)) if goal_errors else None,
                  "hold_pose_translation_error_m": float(np.mean(zero)),
                  "last_3s_translation_error_m": float(np.mean(late)) if late else None,
                  "elapsed_seconds": time.monotonic() - started}
        # The pooled metric is dominated by legacy demonstrations. Keep a fixed,
        # separate sample of the newly collected holdouts visible during tuning.
        for group_name, group_indices, group_episodes in [
            ("corrective_holdout", corrective_validation_indices, corrective_val_eps),
            ("training_probe", training_probe_indices, sampled_training_eps),
            *task_validation_groups,
        ]:
            if not len(group_indices):
                continue
            errors, angles, late_errors = [], [], []
            with torch.inference_mode():
                for start in range(0, len(group_indices), args.batch_size):
                    idx = group_indices[start:start + args.batch_size]
                    predicted = policy.predict_action_chunk(batch(idx))[:, 0] * action_std_t + action_mean_t
                    actual = torch.as_tensor(actions[idx], device="cuda")
                    error = torch.linalg.vector_norm(predicted[:, :3] - actual[:, :3], dim=-1).cpu().numpy()
                    errors.extend(error.tolist())
                    late_errors.extend(error[end_index[idx] - idx < 60].tolist())
                    angles.extend(np.degrees((Rotation.from_rotvec(predicted[:, 3:6].cpu().numpy()).inv()
                                              * Rotation.from_rotvec(actions[idx, 3:6])).magnitude()).tolist())
            result[group_name] = {"frames_sampled": len(errors), "episodes": group_episodes,
                "translation_error_m": float(np.mean(errors)), "translation_error_p95_m": float(np.percentile(errors, 95)),
                "rotation_error_deg": float(np.mean(angles)),
                "last_3s_translation_error_m": float(np.mean(late_errors)) if late_errors else None}
        with (root / "validation.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print("VALIDATION " + json.dumps(result), flush=True)
        return result

    def save(step):
        for projection in state_projections:
            if torch.count_nonzero(projection.weight[:, masked_indices]):
                raise RuntimeError("Excluded state projection columns must remain zero for deployment parity")
        directory = root / "checkpoints" / f"{step:06d}"
        temporary = directory / "pretrained_model.tmp"
        temporary.mkdir(parents=True, exist_ok=False)
        policy.save_pretrained(temporary)
        save_backbone_geometry(policy, temporary)
        pre.save_pretrained(temporary)
        post.save_pretrained(temporary)
        if goal_head is not None:
            torch.save(goal_head.state_dict(), temporary / "auxiliary_goal_head.pt")
        (temporary / "aic_action_config.json").write_text(json.dumps({
            "backbone_geometry": geometry,
            "action_representation": args.action_representation,
            "action_frame": "gripper/tcp" if args.action_representation == "delta_pose" else "base_link",
            "delta_pose_reference": ("observation" if cache.get("native_observation_only") else None)
                                    if args.action_representation == "delta_pose" else None,
            "absolute_rotation_encoding": "rotation_vector_with_target_quaternion_x_nonnegative" if args.action_representation == "absolute_pose" else None,
            "source_action_representation": "full_tcp_relative_pose_command",
            "state_shape": [states.shape[1]],
            "base_state_dim": base_state_dim,
            "task_conditioned": args.include_task,
            "task_vector_indices": [32, 42] if args.include_task else None,
            "task_encoding": task_schema,
            "include_elapsed_sim_time": args.include_time,
            "elapsed_time_projection_masked": args.mask_time,
            "training_time_source": ("Per-source timestamps: legacy frame_index/fps from wall-clock recorder; corrective episodes use actual simulation timestamps" if cache.get("image_shards")
                                     else "dataset frame_index/fps from wall-clock recorder") if args.include_time else None,
            "time_clip_sec": args.time_clip_sec,
            "quaternion_sign": args.quaternion_sign,
            "image_channel_order": "rgb" if args.canonical_rgb else args.image_channel_order,
        }, indent=2) + "\n")
        (temporary / "training_step.json").write_text(json.dumps({"step": step, "training_config": metadata}, indent=2) + "\n")
        temporary.rename(directory / "pretrained_model")
        # Keep optimizer state only for the newest checkpoint, to limit disk use.
        torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(), "numpy_rng": rng.bit_generator.state,
                    "world_size": world_size, "rng_rank": 0,
                    "resume_note": "Optimizer snapshot is diagnostic; this trainer warm-starts weights, not exact optimizer/RNG resume"}, root / "optimizer_latest.pt.tmp")
        (root / "optimizer_latest.pt.tmp").replace(root / "optimizer_latest.pt")
        (root / "latest_checkpoint.json").write_text(json.dumps({"step": step, "path": str(directory / "pretrained_model")}) + "\n")
        print(f"CHECKPOINT {step}", flush=True)

    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    started = time.monotonic(); last_saved = -1; step = 0
    stop_flags = [False, False, False]
    timing_names = ["batch_prepare", "forward_loss_and_sync", "backward_and_ddp", "optimizer", "update_total"]
    timing_sum = np.zeros(len(timing_names), dtype=np.float64)
    timing_updates = 0
    def primary_evaluation(step, *, checkpoint=False):
        # Every rank takes the same branches. Never run a DDP forward on rank 0
        # alone: prediction uses policy.model, which is the unwrapped module.
        if distributed:
            dist.barrier()
        if is_primary:
            validate(step)
            if checkpoint:
                save(step)
        if distributed:
            dist.barrier()
    if is_primary:
        print("TRAINING " + json.dumps(metadata), flush=True)
    primary_evaluation(0)
    with ((root / "metrics.jsonl").open("a") if is_primary else nullcontext()) as log:
        for step in range(1, args.steps + 1):
            stop_flags = distributed_stop_flags(stopping, deadline, started, args.max_minutes, device, distributed)
            if any(stop_flags):
                step -= 1
                break
            policy.train()
            if args.profile_timing:
                phase_start = update_start = time.monotonic()
            selected = distributed_sample_indices(rng, args.batch_size, rank, world_size, sampling_groups,
                                                  sampling_probabilities, terminal_groups,
                                                  args.terminal_sampling_probability)
            data = batch(selected, augment=True)
            if args.profile_timing:
                torch.cuda.synchronize()
                phase_end = time.monotonic()
                timing_sum[0] += phase_end - phase_start
                phase_start = phase_end
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                from lerobot.utils.constants import OBS_IMAGES
                model_batch = {**data, OBS_IMAGES: [data[key] for key in policy.config.image_features]}
                prediction, (mu, log_variance) = train_model(model_batch)
                l1_loss = (torch.nn.functional.l1_loss(prediction, data["action"], reduction="none")
                           * (~data["action_is_pad"]).unsqueeze(-1)).mean()
                loss = l1_loss
                details = {"l1_loss": l1_loss.item()}
                if args.use_vae:
                    kl_loss = stable_gaussian_kl(mu, log_variance)
                    loss = loss + args.kl_weight * kl_loss
                    details["kld_loss"] = kl_loss.item()
                if goal_head is not None:
                    # The goal must be inferred from observations, without the
                    # training VAE's access to the ground-truth action chunk.
                    from lerobot.utils.constants import OBS_IMAGES
                    policy.model.eval()
                    observation_only = {"observation.state": (torch.zeros_like(data["observation.state"]) if args.goal_images_only else data["observation.state"]),
                                        OBS_IMAGES: [data[key] for key in policy.config.image_features]}
                    policy.model(observation_only)
                    goal_pred = goal_head(captured_encoder["value"][1])
                    goal_loss = torch.nn.functional.l1_loss(goal_pred, data["auxiliary_goal"])
                    policy.model.train()
                    loss = loss + args.goal_loss_weight * min(1., step / 1000.) * goal_loss
                    details["auxiliary_goal_l1"] = goal_loss.item()
            finite = torch.isfinite(loss).to(torch.int32)
            if distributed:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise RuntimeError(f"Nonfinite loss at {step}")
            if args.profile_timing:
                torch.cuda.synchronize()
                phase_end = time.monotonic()
                timing_sum[1] += phase_end - phase_start
                phase_start = phase_end
            loss.backward()
            if args.profile_timing:
                torch.cuda.synchronize()
                phase_end = time.monotonic()
                timing_sum[2] += phase_end - phase_start
                phase_start = phase_end
            grad = torch.nn.utils.clip_grad_norm_(optimization_parameters, 10., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if args.profile_timing:
                torch.cuda.synchronize()
                phase_end = time.monotonic()
                timing_sum[3] += phase_end - phase_start
                timing_sum[4] += phase_end - update_start
                timing_updates += 1
            if step == 1 or step % 50 == 0:
                values = torch.tensor([loss.item(), *details.values()], device=device)
                peak_memory = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device=device)
                if distributed:
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
                    values /= world_size
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                if args.profile_timing:
                    phase_means = torch.tensor(timing_sum / timing_updates, device=device)
                    phase_maxima = phase_means.clone()
                    if distributed:
                        dist.all_reduce(phase_means, op=dist.ReduceOp.SUM)
                        phase_means /= world_size
                        dist.all_reduce(phase_maxima, op=dist.ReduceOp.MAX)
                if is_primary:
                    row = {"step": step, "loss": values[0].item(),
                           **dict(zip(details, values[1:].tolist(), strict=True)), "grad_norm": float(grad),
                           "elapsed_seconds": time.monotonic() - started,
                           "learning_rates": scheduler.get_last_lr(),
                           "gpu_memory_gb": peak_memory.item(),
                           "global_examples_seen": step * args.batch_size * world_size}
                    if args.profile_timing:
                        row["profile_timing_seconds_per_update"] = {
                            "updates_in_interval": timing_updates,
                            "rank_mean": dict(zip(timing_names, phase_means.tolist(), strict=True)),
                            "rank_maximum": dict(zip(timing_names, phase_maxima.tolist(), strict=True))}
                    log.write(json.dumps(row) + "\n"); log.flush(); print(json.dumps(row), flush=True)
                timing_sum[:] = 0
                timing_updates = 0
            checkpoint = step % args.save_every == 0
            if step % args.validate_every == 0 or checkpoint:
                primary_evaluation(step, checkpoint=checkpoint)
                if checkpoint:
                    last_saved = step
        if last_saved != step:
            primary_evaluation(step, checkpoint=True)
    if is_primary:
        (root / "completion.json").write_text(json.dumps({"actual_updates": step, "requested_updates": args.steps,
            "elapsed_seconds": time.monotonic() - started, "stopped_by_signal": stop_flags[0],
            "deadline_reached": stop_flags[1] or time.time() >= deadline,
            "max_minutes_reached": stop_flags[2], "world_size": world_size,
            "global_batch_size": args.batch_size * world_size,
            "global_examples_seen": step * args.batch_size * world_size}, indent=2) + "\n")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    if image_executor is not None:
        image_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
