#!/usr/bin/env python3
"""Vision offline SERL pretraining using a LeRobot ACT checkpoint as actor."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler

from lerobot_robot_aic.act_warmstart import inspect_act_checkpoint
from lerobot_robot_aic.vision_offline_serl import (
    VisionOfflineSERLConfig,
    VisionOfflineSERLDataset,
    VisionOfflineSERLTrainer,
    load_act_actor,
    write_json,
)


def _unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--act-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="vision_offline_serl_smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adapter-lr", type=float, default=1e-4)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=0.0)
    parser.add_argument("--adapter-penalty-weight", type=float, default=1e-3)
    parser.add_argument("--act-preservation-weight", type=float, default=1e-2)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--actor-mode", choices=["act_direct", "act_adapter"], default="act_adapter")
    parser.add_argument(
        "--actor-update-mode",
        choices=["q_bc", "bc_only", "critic_only"],
        default="q_bc",
        help="Actor update objective. critic_only trains only critics while keeping the ACT adapter fixed.",
    )
    parser.add_argument("--freeze-act", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adapter-hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-num-layers", type=int, default=2)
    parser.add_argument("--adapter-arch", choices=["mlp", "gated"], default="mlp")
    parser.add_argument("--adapter-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--adapter-delta-clip", type=float, default=None)
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument(
        "--critic-image-encoder",
        choices=["small_conv", "resnet18", "resnet18_imagenet", "convnext_tiny", "convnext_tiny_imagenet"],
        default="small_conv",
        help="Visual backbone used by the offline SERL critics.",
    )
    parser.add_argument(
        "--critic-arch",
        choices=["concat", "multiplicative", "value_advantage"],
        default="concat",
        help="Critic state/action fusion architecture.",
    )
    parser.add_argument("--critic-feature-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-num-layers", type=int, default=2)
    parser.add_argument("--critic-per-camera-dim", type=int, default=64)
    parser.add_argument("--critic-layer-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--critic-activation", choices=["relu", "gelu"], default="gelu")
    parser.add_argument(
        "--state-encoding",
        choices=["none", "fourier"],
        default="fourier",
        help="Model-side state coordinate encoding applied to adapter and critic state inputs.",
    )
    parser.add_argument(
        "--state-encoding-indices",
        type=int,
        nargs="*",
        default=None,
        help="State indices to Fourier encode. Defaults to tcp position and tcp error xyz: 0 1 2 13 14 15.",
    )
    parser.add_argument("--state-encoding-num-bands", type=int, default=4)
    parser.add_argument("--state-encoding-max-freq", type=float, default=8.0)
    parser.add_argument("--state-encoding-scale", type=float, default=10.0)
    parser.add_argument(
        "--reward-mode",
        choices=["dataset", "final_success", "zero"],
        default="dataset",
    )
    parser.add_argument(
        "--camera-keys",
        nargs="+",
        default=None,
        help="LeRobot visual observation keys. Defaults to ACT checkpoint camera keys.",
    )
    parser.add_argument("--dataset-video-backend", default="pyav")
    parser.add_argument(
        "--swap-rgb-channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Reverse image channel order after LeRobot video decode. By default this follows "
            "meta/info.json aic_rgb_patch.swap_rgb_channels when present."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for train/validation video decoding.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin DataLoader memory before host-to-GPU transfer.",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-max-batches", type=int, default=16)
    parser.add_argument(
        "--early-stopping-metric",
        choices=["bc_loss", "actor_loss", "critic_loss", "td_loss"],
        default="bc_loss",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-wall-time-minutes", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ddp_env() -> dict[str, int]:
    return {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
    }


def _is_rank0() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def _barrier_if_distributed() -> None:
    if dist.is_initialized():
        dist.barrier()


def _setup_distributed(requested: str) -> tuple[bool, torch.device, dict[str, Any]]:
    env = _ddp_env()
    distributed = env["world_size"] > 1
    requested_device = torch.device(requested)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Retry with --device cpu.")
    if requested_device.type == "cuda":
        torch.cuda.set_device(env["local_rank"])
        device = torch.device("cuda", env["local_rank"])
    else:
        device = requested_device
    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, device, {
        **env,
        "distributed": distributed,
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "effective_device": str(device),
    }


def _wrap_ddp(trainer: VisionOfflineSERLTrainer, device: torch.device) -> None:
    ddp_kwargs: dict[str, Any] = {"find_unused_parameters": True}
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    trainer.actor = DistributedDataParallel(trainer.actor, **ddp_kwargs)
    trainer.critic1 = DistributedDataParallel(trainer.critic1, **ddp_kwargs)
    trainer.critic2 = DistributedDataParallel(trainer.critic2, **ddp_kwargs)
    trainer.target_critic1.load_state_dict(_unwrap_module(trainer.critic1).state_dict())
    trainer.target_critic2.load_state_dict(_unwrap_module(trainer.critic2).state_dict())


def _camera_keys(args: argparse.Namespace) -> list[str]:
    if args.camera_keys:
        return list(args.camera_keys)
    metadata = inspect_act_checkpoint(args.act_checkpoint)
    keys = metadata.get("camera_keys") or []
    if not keys:
        raise ValueError("No camera keys were found in ACT checkpoint metadata; pass --camera-keys explicitly.")
    return list(keys)


def _train_config(args: argparse.Namespace, dataset_summary: dict[str, Any], warmstart: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_root": str(args.dataset_root),
        "act_checkpoint": str(args.act_checkpoint),
        "output_dir": str(args.output_dir),
        "job_name": args.job_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "lr": args.lr,
        "adapter_lr": args.adapter_lr,
        "act_lr": args.act_lr,
        "critic_lr": args.critic_lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "bc_weight": args.bc_weight,
        "cql_weight": args.cql_weight,
        "adapter_penalty_weight": args.adapter_penalty_weight,
        "act_preservation_weight": args.act_preservation_weight,
        "smoothness_weight": args.smoothness_weight,
        "action_horizon": args.action_horizon,
        "actor_mode": args.actor_mode,
        "actor_update_mode": args.actor_update_mode,
        "freeze_act": args.freeze_act,
        "adapter_hidden_dim": args.adapter_hidden_dim,
        "adapter_num_layers": args.adapter_num_layers,
        "adapter_arch": args.adapter_arch,
        "adapter_layer_norm": args.adapter_layer_norm,
        "adapter_activation": args.adapter_activation,
        "adapter_scale": args.adapter_scale,
        "adapter_delta_clip": args.adapter_delta_clip,
        "action_clip": args.action_clip,
        "critic_image_encoder": args.critic_image_encoder,
        "critic_arch": args.critic_arch,
        "critic_feature_dim": args.critic_feature_dim,
        "critic_hidden_dim": args.critic_hidden_dim,
        "critic_num_layers": args.critic_num_layers,
        "critic_per_camera_dim": args.critic_per_camera_dim,
        "critic_layer_norm": args.critic_layer_norm,
        "critic_activation": args.critic_activation,
        "state_encoding": args.state_encoding,
        "state_encoding_indices": list(_state_encoding_indices(args)),
        "state_encoding_num_bands": args.state_encoding_num_bands,
        "state_encoding_max_freq": args.state_encoding_max_freq,
        "state_encoding_scale": args.state_encoding_scale,
        "reward_mode": args.reward_mode,
        "camera_keys": dataset_summary["camera_keys"],
        "dataset_video_backend": args.dataset_video_backend,
        "swap_rgb_channels": args.swap_rgb_channels,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "save_every": args.save_every,
        "val_fraction": args.val_fraction,
        "val_every": args.val_every,
        "val_max_batches": args.val_max_batches,
        "early_stopping_metric": args.early_stopping_metric,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "max_wall_time_minutes": args.max_wall_time_minutes,
        "dataset_summary": dataset_summary,
        "act_warmstart": warmstart,
    }


def _state_encoding_indices(args: argparse.Namespace) -> tuple[int, ...]:
    if args.state_encoding == "none":
        return ()
    if args.state_encoding_indices is not None and len(args.state_encoding_indices) > 0:
        return tuple(int(i) for i in args.state_encoding_indices)
    return (0, 1, 2, 13, 14, 15)


def _model_summary(trainer: VisionOfflineSERLTrainer) -> dict[str, int]:
    actor = _unwrap_module(trainer.actor)
    adapter_params = int(sum(p.numel() for p in getattr(actor, "adapter", nn.Module()).parameters()))
    return {
        "actor_parameters": int(sum(p.numel() for p in actor.parameters())),
        "actor_trainable_parameters": int(sum(p.numel() for p in actor.parameters() if p.requires_grad)),
        "act_trainable_parameters": int(sum(p.numel() for p in actor.act_policy.parameters() if p.requires_grad)),
        "adapter_parameters": adapter_params,
        "adapter_trainable_parameters": int(
            sum(p.numel() for p in getattr(actor, "adapter", nn.Module()).parameters() if p.requires_grad)
        ),
        "critic1_parameters": int(sum(p.numel() for p in _unwrap_module(trainer.critic1).parameters())),
        "critic2_parameters": int(sum(p.numel() for p in _unwrap_module(trainer.critic2).parameters())),
    }


def _split_by_episode(dataset: VisionOfflineSERLDataset, val_fraction: float) -> tuple[list[int], list[int]]:
    if val_fraction <= 0.0:
        return list(range(len(dataset))), []
    if val_fraction >= 1.0:
        raise ValueError("--val-fraction must be in [0, 1)")
    episodes = sorted(set(int(v) for v in dataset.episodes))
    if len(episodes) < 2:
        raise ValueError("Validation split requires at least two episodes")
    val_count = max(1, int(round(len(episodes) * val_fraction)))
    val_count = min(val_count, len(episodes) - 1)
    val_episodes = set(episodes[-val_count:])
    train_indices: list[int] = []
    val_indices: list[int] = []
    for idx, episode in enumerate(dataset.episodes):
        if int(episode) in val_episodes:
            val_indices.append(idx)
        else:
            train_indices.append(idx)
    return train_indices, val_indices


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*(row.keys() for row in rows)))
    return {
        key: float(sum(row[key] for row in rows if key in row) / sum(1 for row in rows if key in row))
        for key in keys
    }


def _validate(trainer: VisionOfflineSERLTrainer, loader: DataLoader, *, max_batches: int) -> dict[str, float]:
    # Validation is rank-0 only. If the modules remain DDP-wrapped, their
    # forward pass expects all ranks to participate and validation deadlocks
    # while nonzero ranks wait at the surrounding barrier.
    original_actor = trainer.actor
    original_critic1 = trainer.critic1
    original_critic2 = trainer.critic2
    trainer.actor = _unwrap_module(trainer.actor)
    trainer.critic1 = _unwrap_module(trainer.critic1)
    trainer.critic2 = _unwrap_module(trainer.critic2)
    actor_was_training = trainer.actor.training
    critic1_was_training = trainer.critic1.training
    critic2_was_training = trainer.critic2.training
    try:
        trainer.actor.eval()
        trainer.critic1.eval()
        trainer.critic2.eval()
        rows: list[dict[str, float]] = []
        for batch_index, batch in enumerate(loader, start=1):
            rows.append(trainer.eval_step(batch))
            if max_batches > 0 and batch_index >= max_batches:
                break
        if actor_was_training:
            trainer.actor.train()
            actor = _unwrap_module(trainer.actor)
            if hasattr(actor, "set_act_frozen"):
                actor.set_act_frozen(getattr(actor, "freeze_act", True))
        if critic1_was_training:
            trainer.critic1.train()
        if critic2_was_training:
            trainer.critic2.train()
        return _mean_metrics(rows)
    finally:
        trainer.actor = original_actor
        trainer.critic1 = original_critic1
        trainer.critic2 = original_critic2


def _as_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    try:
        return tuple(int(v) for v in value)
    except TypeError:
        return None


def _action_contract(
    *,
    args: argparse.Namespace,
    dataset_summary: dict[str, Any],
    warmstart: dict[str, Any],
    camera_keys: list[str],
) -> dict[str, Any]:
    """Validate the ACT warm start can supply the SERL action chunk contract."""
    errors: list[str] = []
    warnings: list[str] = []
    act_chunk_size = warmstart.get("chunk_size")
    act_n_action_steps = warmstart.get("n_action_steps")
    act_state_shape = _as_int_tuple(warmstart.get("state_shape"))
    act_action_shape = _as_int_tuple(warmstart.get("action_shape"))
    act_camera_keys = list(warmstart.get("camera_keys") or [])

    if act_chunk_size is None:
        warnings.append("ACT checkpoint metadata did not expose chunk_size; cannot verify chunk >= action_horizon.")
    elif int(act_chunk_size) < int(args.action_horizon):
        errors.append(
            f"ACT chunk_size={act_chunk_size} is smaller than offline SERL action_horizon={args.action_horizon}."
        )

    if act_n_action_steps is not None and int(act_n_action_steps) != int(args.action_horizon):
        warnings.append(
            "ACT checkpoint n_action_steps is an execution setting, not the model output size; "
            f"offline SERL will train on the first {args.action_horizon} actions from ACT chunk_size={act_chunk_size} "
            f"even though ACT was trained/evaluated with n_action_steps={act_n_action_steps}."
        )

    dataset_state_dim = int(dataset_summary["state_dim"])
    if act_state_shape is not None and act_state_shape and int(act_state_shape[0]) != dataset_state_dim:
        errors.append(
            f"ACT observation.state dim={act_state_shape[0]} does not match dataset state_dim={dataset_state_dim}."
        )

    dataset_single_action_dim = int(dataset_summary["single_action_dim"])
    if act_action_shape is not None and act_action_shape and int(act_action_shape[0]) != dataset_single_action_dim:
        errors.append(
            f"ACT per-step action dim={act_action_shape[0]} does not match dataset single_action_dim={dataset_single_action_dim}."
        )

    missing_cameras = sorted(set(camera_keys) - set(act_camera_keys))
    extra_cameras = sorted(set(act_camera_keys) - set(camera_keys))
    if missing_cameras or extra_cameras:
        errors.append(
            "ACT camera keys do not match offline SERL camera keys: "
            f"missing_from_act={missing_cameras}, extra_in_act={extra_cameras}."
        )

    expected_action_dim = dataset_single_action_dim * int(args.action_horizon)
    if int(dataset_summary["action_dim"]) != expected_action_dim:
        errors.append(
            f"Dataset action_dim={dataset_summary['action_dim']} does not equal "
            f"single_action_dim({dataset_single_action_dim}) * action_horizon({args.action_horizon})={expected_action_dim}."
        )

    contract = {
        "act_chunk_size": act_chunk_size,
        "act_n_action_steps": act_n_action_steps,
        "offline_serl_action_horizon": int(args.action_horizon),
        "runtime_n_action_steps_required": int(args.action_horizon),
        "dataset_state_dim": dataset_state_dim,
        "dataset_single_action_dim": dataset_single_action_dim,
        "offline_serl_action_dim": int(dataset_summary["action_dim"]),
        "act_state_shape": list(act_state_shape) if act_state_shape is not None else None,
        "act_action_shape": list(act_action_shape) if act_action_shape is not None else None,
        "camera_keys": list(camera_keys),
        "act_camera_keys": act_camera_keys,
        "uses_first_n_actions_from_act_chunk": int(args.action_horizon),
        "warnings": warnings,
        "errors": errors,
    }
    if errors:
        raise ValueError("Offline SERL/ACT contract check failed:\n- " + "\n- ".join(errors))
    return contract


def main() -> int:
    args = parse_args()
    distributed, device, distributed_summary = _setup_distributed(args.device)
    camera_keys = _camera_keys(args)

    dataset = VisionOfflineSERLDataset(
        args.dataset_root,
        camera_keys=camera_keys,
        action_horizon=args.action_horizon,
        reward_mode=args.reward_mode,
        video_backend=args.dataset_video_backend,
        swap_rgb_channels=args.swap_rgb_channels,
    )
    dataset_summary = asdict(dataset.summary)
    actor, warmstart = load_act_actor(
        args.act_checkpoint,
        action_horizon=args.action_horizon,
        device=device,
        actor_mode=args.actor_mode,
        state_dim=dataset.state_dim,
        adapter_hidden_dim=args.adapter_hidden_dim,
        adapter_num_layers=args.adapter_num_layers,
        adapter_arch=args.adapter_arch,
        adapter_layer_norm=args.adapter_layer_norm,
        adapter_activation=args.adapter_activation,
        state_encoding=args.state_encoding,
        state_encoding_indices=_state_encoding_indices(args),
        state_encoding_num_bands=args.state_encoding_num_bands,
        state_encoding_max_freq=args.state_encoding_max_freq,
        state_encoding_scale=args.state_encoding_scale,
        adapter_scale=args.adapter_scale,
        freeze_act=args.freeze_act,
        adapter_delta_clip=args.adapter_delta_clip,
        action_clip=args.action_clip,
    )
    contract = _action_contract(
        args=args,
        dataset_summary=dataset_summary,
        warmstart=warmstart,
        camera_keys=camera_keys,
    )
    warmstart["action_contract"] = contract
    if _is_rank0():
        print(json.dumps({"offline_serl_action_contract": contract}, indent=2, sort_keys=True))
    config = VisionOfflineSERLConfig(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        action_horizon=args.action_horizon,
        camera_keys=camera_keys,
        gamma=args.gamma,
        tau=args.tau,
        bc_weight=args.bc_weight,
        cql_weight=args.cql_weight,
        adapter_penalty_weight=args.adapter_penalty_weight,
        act_preservation_weight=args.act_preservation_weight,
        smoothness_weight=args.smoothness_weight,
        adapter_delta_clip=args.adapter_delta_clip,
        action_clip=args.action_clip,
        lr=args.lr,
        adapter_lr=args.adapter_lr,
        act_lr=args.act_lr,
        critic_lr=args.critic_lr,
        actor_mode=args.actor_mode,
        actor_update_mode=args.actor_update_mode,
        freeze_act=args.freeze_act,
        critic_image_encoder=args.critic_image_encoder,
        critic_arch=args.critic_arch,
        critic_feature_dim=args.critic_feature_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        critic_num_layers=args.critic_num_layers,
        critic_per_camera_dim=args.critic_per_camera_dim,
        critic_layer_norm=args.critic_layer_norm,
        critic_activation=args.critic_activation,
        adapter_arch=args.adapter_arch,
        adapter_layer_norm=args.adapter_layer_norm,
        adapter_activation=args.adapter_activation,
        state_encoding=args.state_encoding,
        state_encoding_indices=_state_encoding_indices(args),
        state_encoding_num_bands=args.state_encoding_num_bands,
        state_encoding_max_freq=args.state_encoding_max_freq,
        state_encoding_scale=args.state_encoding_scale,
    )
    trainer = VisionOfflineSERLTrainer(config=config, actor=actor, device=device)
    if distributed:
        _wrap_ddp(trainer, device)
    first_batch = next(iter(DataLoader(dataset, batch_size=min(args.batch_size, 2), shuffle=False, num_workers=0)))
    first_obs = trainer._obs_to_device(first_batch["obs"])
    with torch.no_grad():
        initial_components = trainer.actor_components(first_obs)
        warmstart["initial_delta_norm"] = float(initial_components["delta_action"].norm(dim=-1).mean().detach().cpu())
        warmstart["initial_final_minus_act_norm"] = float(
            (initial_components["final_action"] - initial_components["base_action"]).norm(dim=-1).mean().detach().cpu()
        )
    train_config = _train_config(args, dataset_summary, warmstart)
    train_indices, val_indices = _split_by_episode(dataset, args.val_fraction)
    train_config["split_summary"] = {
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "val_fraction": args.val_fraction,
        "split_by": "episode_index_tail_holdout",
    }
    summary = {
        "training_config": train_config,
        "model_summary": _model_summary(trainer),
        "distributed": distributed_summary,
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        if _is_rank0():
            print(json.dumps(summary, indent=2, sort_keys=True))
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0

    run_dir = args.output_dir
    if _is_rank0():
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "train_config.json", train_config)
        write_json(run_dir / "warmstart_report.json", warmstart)
        write_json(run_dir / "dataset_summary.json", dataset_summary)
        write_json(run_dir / "distributed.json", distributed_summary)

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None
    sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        if val_dataset is not None
        else None
    )
    metrics_path = run_dir / "metrics.jsonl"
    validation_path = run_dir / "validation_metrics.jsonl"
    if _is_rank0() and metrics_path.exists():
        metrics_path.unlink()
    if _is_rank0() and validation_path.exists():
        validation_path.unlink()

    step = 0
    best_val_metric = float("inf")
    best_val_step = 0
    bad_val_checks = 0
    stop_reason = "max_steps"
    started = time.monotonic()
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            if args.max_wall_time_minutes > 0.0 and (time.monotonic() - started) >= args.max_wall_time_minutes * 60.0:
                stop_reason = "max_wall_time"
                step = args.steps
                break
            step += 1
            metrics = trainer.train_step(batch)
            metrics["step"] = step
            if _is_rank0():
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, sort_keys=True) + "\n")
            should_save = args.save_every > 0 and step % args.save_every == 0
            should_validate = val_loader is not None and args.val_every > 0 and step % args.val_every == 0
            if should_save or should_validate:
                _barrier_if_distributed()
            if should_save:
                if _is_rank0():
                    trainer.save_checkpoint(
                        run_dir / f"checkpoint_{step:06d}.pt",
                        train_config=train_config,
                        dataset_summary=dataset_summary,
                        warmstart_report=warmstart,
                        step=step,
                    )
                _barrier_if_distributed()
            if should_validate:
                stop_now = False
                if _is_rank0():
                    val_metrics = _validate(trainer, val_loader, max_batches=args.val_max_batches)
                    val_metrics["step"] = step
                    with validation_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(val_metrics, sort_keys=True) + "\n")
                    metric = float(val_metrics[args.early_stopping_metric])
                    if metric < best_val_metric - args.early_stopping_min_delta:
                        best_val_metric = metric
                        best_val_step = step
                        bad_val_checks = 0
                        trainer.save_checkpoint(
                            run_dir / "checkpoint_best_val.pt",
                            train_config=train_config,
                            dataset_summary=dataset_summary,
                            warmstart_report=warmstart,
                            step=step,
                        )
                    else:
                        bad_val_checks += 1
                    if args.early_stopping_patience > 0 and bad_val_checks >= args.early_stopping_patience:
                        stop_reason = "early_stopping"
                        stop_now = True
                if dist.is_initialized():
                    payload = [stop_now]
                    dist.broadcast_object_list(payload, src=0)
                    stop_now = bool(payload[0])
                if stop_now:
                    step = args.steps
                _barrier_if_distributed()
                if step >= args.steps:
                    break
            if step >= args.steps:
                break

    latest = run_dir / "checkpoint_latest.pt"
    if _is_rank0():
        trainer.save_checkpoint(
            latest,
            train_config=train_config,
            dataset_summary=dataset_summary,
            warmstart_report={**warmstart, "distributed": distributed_summary},
            step=step,
        )
        run_summary = {
            "checkpoint_latest": str(latest),
            "metrics": str(metrics_path),
            "validation_metrics": str(validation_path) if val_loader is not None else None,
            "steps": step,
            "stop_reason": stop_reason,
            "best_val_metric": best_val_metric if best_val_step else None,
            "best_val_step": best_val_step if best_val_step else None,
            "best_val_checkpoint": str(run_dir / "checkpoint_best_val.pt") if best_val_step else None,
            "actor_mode": args.actor_mode,
            "actor_update_mode": args.actor_update_mode,
            "freeze_act": args.freeze_act,
            "act_checkpoint": str(args.act_checkpoint),
            "critic_init": "scratch",
            "critic_initialization_note": "Critic/value is scratch; ACT is used only as an actor/action prior.",
            "model_summary": _model_summary(trainer),
            "dataset_summary": dataset_summary,
            "warmstart_report": warmstart,
            "distributed": distributed_summary,
        }
        write_json(run_dir / "run_summary.json", run_summary)
        print(f"Wrote vision offline SERL checkpoint: {latest}")
        print(f"Wrote warm-start report: {run_dir / 'warmstart_report.json'}")
        print(f"Wrote metrics: {metrics_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
