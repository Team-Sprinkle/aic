#!/usr/bin/env python3
"""Minimal offline SERL-style pretraining on AIC LeRobot expert data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from lerobot_robot_aic.offline_rl_dataset import (
    OfflineRLTransitionDataset,
    load_lerobot_transitions,
    task_conditioning_summary,
)
from lerobot_robot_aic.act_warmstart import inspect_act_checkpoint, load_act_action_head_bias
from lerobot_robot_aic.run_metadata import git_info, write_json
from lerobot_robot_aic.task_encoding import TASK_VECTOR_DIM, task_encoding_schema
from lerobot_robot_aic.offline_serl import OfflineSERLConfig, OfflineSERLTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="offline_serl_smoke")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--bc-weight", type=float, default=1.0)
    parser.add_argument("--cql-weight", type=float, default=0.0)
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=1,
        help="Number of future action steps to flatten into each actor target.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument(
        "--reward-mode",
        choices=["dataset", "final_success", "zero"],
        default="dataset",
    )
    parser.add_argument("--obs-mode", choices=["lowdim"], default="lowdim")
    parser.add_argument("--include-task-vector", action="store_true")
    parser.add_argument("--task-metadata", type=Path, default=None)
    parser.add_argument("--missing-task-vector", choices=["error", "zeros"], default="error")
    parser.add_argument(
        "--critic-init",
        choices=["scratch", "checkpoint", "act"],
        default="scratch",
        help="Critic initialization source. 'act' is rejected because ACT has no value semantics.",
    )
    parser.add_argument("--critic-checkpoint", type=Path, default=None)
    parser.add_argument("--act-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--act-warmstart-mode",
        choices=["metadata", "action_head_bias"],
        default="action_head_bias",
        help=(
            "How to consume --act-checkpoint. 'action_head_bias' transfers the "
            "single-step ACT output bias as a repeated chunk prior; 'metadata' "
            "only validates and records checkpoint metadata."
        ),
    )
    parser.add_argument("--save-every", type=int, default=500)
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


def _setup_distributed(requested_device: str) -> tuple[bool, torch.device, dict[str, Any]]:
    env = _ddp_env()
    distributed = env["world_size"] > 1
    requested = torch.device(requested_device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Retry with --device cpu.")
        torch.cuda.set_device(env["local_rank"])
        device = torch.device("cuda", env["local_rank"])
    else:
        device = requested
    if distributed and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
    return distributed, device, {
        **env,
        "distributed": distributed,
        "backend": dist.get_backend() if dist.is_initialized() else None,
        "effective_device": str(device),
    }


def _wrap_ddp(trainer: OfflineSERLTrainer, device: torch.device) -> None:
    ddp_kwargs: dict[str, Any] = {}
    ddp_kwargs["find_unused_parameters"] = True
    if device.type == "cuda":
        ddp_kwargs["device_ids"] = [device.index]
        ddp_kwargs["output_device"] = device.index
    trainer.actor = DistributedDataParallel(trainer.actor, **ddp_kwargs)
    trainer.critic1 = DistributedDataParallel(trainer.critic1, **ddp_kwargs)
    trainer.critic2 = DistributedDataParallel(trainer.critic2, **ddp_kwargs)
    trainer.target_critic1.load_state_dict(trainer.critic1.module.state_dict())
    trainer.target_critic2.load_state_dict(trainer.critic2.module.state_dict())


def config_dict(args: argparse.Namespace, dataset: OfflineRLTransitionDataset) -> dict[str, Any]:
    original_obs_dim = dataset.obs_dim - (TASK_VECTOR_DIM if args.include_task_vector else 0)
    return {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "job_name": args.job_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "lr": args.lr,
        "gamma": args.gamma,
        "tau": args.tau,
        "bc_weight": args.bc_weight,
        "cql_weight": args.cql_weight,
        "action_horizon": args.action_horizon,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "reward_mode": args.reward_mode,
        "obs_mode": args.obs_mode,
        "include_task_vector": args.include_task_vector,
        "task_metadata": str(args.task_metadata) if args.task_metadata else None,
        "missing_task_vector": args.missing_task_vector,
        "task_vector_dim": TASK_VECTOR_DIM if args.include_task_vector else 0,
        "task_encoding_schema": task_encoding_schema() if args.include_task_vector else None,
        "original_obs_dim": original_obs_dim,
        "effective_obs_dim": dataset.obs_dim,
        "critic_init": args.critic_init,
        "critic_checkpoint": str(args.critic_checkpoint) if args.critic_checkpoint else None,
        "act_checkpoint": str(args.act_checkpoint) if args.act_checkpoint else None,
        "act_warmstart_semantics": (
            "actor/action bias prior only; ACT is not a value-function prior"
            if args.act_checkpoint
            else None
        ),
        "save_every": args.save_every,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "num_transitions": len(dataset),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    args = parse_args()
    distributed, device, distributed_summary = _setup_distributed(args.device)
    if args.critic_init == "act":
        raise ValueError("critic_init=act is invalid: ACT has no critic/value semantics")
    if args.critic_init == "checkpoint" and args.critic_checkpoint is None:
        raise ValueError("--critic-init checkpoint requires --critic-checkpoint")
    act_warmstart: dict[str, Any] | None = None
    if args.act_checkpoint is not None:
        act_warmstart = inspect_act_checkpoint(args.act_checkpoint)
        act_warmstart["mode"] = args.act_warmstart_mode

    arrays, schema = load_lerobot_transitions(
        args.dataset_root,
        reward_mode=args.reward_mode,
        obs_mode=args.obs_mode,
        action_horizon=args.action_horizon,
        include_task_vector=args.include_task_vector,
        task_metadata=args.task_metadata,
        missing_task_vector=args.missing_task_vector,
    )
    dataset = OfflineRLTransitionDataset(arrays, normalize=True)
    cfg = config_dict(args, dataset)

    summary = {
        "dataset_schema": schema.as_dict(),
        "training_config": cfg,
        "normalization_stats": dataset.stats.as_dict(),
        "reward_mean": float(arrays.reward.mean()),
        "done_count": int(arrays.done.sum()),
        "task_conditioning": task_conditioning_summary(
            include_task_vector=args.include_task_vector,
            original_obs_dim=cfg["original_obs_dim"],
            effective_obs_dim=dataset.obs_dim,
        ),
        "critic_initialization": {
            "mode": args.critic_init,
            "checkpoint": str(args.critic_checkpoint) if args.critic_checkpoint else None,
            "note": "Critic/value initializes from scratch unless a SERL/RL critic checkpoint is explicitly provided.",
        },
        "distributed": distributed_summary,
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
        _write_json(run_dir / "train_config.json", summary)
        _write_json(run_dir / "task_encoding_schema.json", task_encoding_schema() if args.include_task_vector else {})
        _write_json(run_dir / "git_info.json", git_info(Path(__file__).resolve().parents[3]))

    trainer = OfflineSERLTrainer(
        OfflineSERLConfig(
            obs_dim=dataset.obs_dim,
            action_dim=dataset.action_dim,
            gamma=args.gamma,
            tau=args.tau,
            bc_weight=args.bc_weight,
            cql_weight=args.cql_weight,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            action_horizon=args.action_horizon,
        ),
        device=device,
    )
    if distributed:
        _wrap_ddp(trainer, device)
    if args.act_checkpoint is not None and args.act_warmstart_mode == "action_head_bias":
        bias = load_act_action_head_bias(args.act_checkpoint)
        transfer_report = trainer.warm_start_actor_bias_from_action_head(bias)
        act_warmstart = {
            **(act_warmstart or {}),
            **transfer_report,
            "semantics": "actor/action bias prior only; critic/value remains scratch unless critic_init=checkpoint",
        }
    if args.critic_init == "checkpoint":
        critic_ckpt = torch.load(args.critic_checkpoint, map_location=trainer.device)
        trainer.load_critic_checkpoint(critic_ckpt)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )
    metrics_path = run_dir / "metrics.jsonl"
    latest_path = run_dir / "checkpoint_latest.pt"
    if _is_rank0() and metrics_path.exists():
        metrics_path.unlink()

    step = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            step += 1
            metrics = trainer.train_step(batch)
            metrics["step"] = step
            metrics["rank"] = distributed_summary["rank"]
            metrics["world_size"] = distributed_summary["world_size"]
            if _is_rank0():
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(metrics, sort_keys=True) + "\n")
            if _is_rank0() and args.save_every > 0 and step % args.save_every == 0:
                trainer.save_checkpoint(
                    run_dir / f"checkpoint_{step:06d}.pt",
                    train_config=cfg,
                    schema_summary=schema.as_dict(),
                    normalization_stats=dataset.stats.as_dict(),
                    step=step,
                    warmstart_metadata=act_warmstart,
                )
            if step >= args.steps:
                break

    if _is_rank0():
        trainer.save_checkpoint(
            latest_path,
            train_config=cfg,
            schema_summary=schema.as_dict(),
            normalization_stats=dataset.stats.as_dict(),
            step=step,
            warmstart_metadata=act_warmstart,
        )
        run_summary = {
            "checkpoint_latest": str(latest_path),
            "metrics": str(metrics_path),
            "steps": step,
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "include_task_vector": args.include_task_vector,
            "task_vector_dim": cfg["task_vector_dim"],
            "task_encoding_schema": cfg["task_encoding_schema"],
            "original_obs_dim": cfg["original_obs_dim"],
            "effective_obs_dim": cfg["effective_obs_dim"],
            "critic_init": args.critic_init,
            "act_warmstart": act_warmstart,
            "distributed": distributed_summary,
        }
        _write_json(run_dir / "run_summary.json", run_summary)
        print(f"Wrote offline SERL checkpoint: {latest_path}")
        print(f"Wrote metrics: {metrics_path}")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
