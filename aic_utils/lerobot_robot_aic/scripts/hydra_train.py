#!/usr/bin/env python3
"""Hydra launcher for AIC hybrid training stages.

The launcher intentionally delegates to the existing argparse scripts so those
entrypoints stay stable for direct use and for older automation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot_robot_aic.hardware import apply_cuda_visible_devices, select_cuda_devices  # noqa: E402
from lerobot_robot_aic.run_metadata import git_info, write_json  # noqa: E402
from lerobot_robot_aic.task_encoding import task_encoding_schema  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]


def _as_path(value: str | None) -> str | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else REPO_ROOT / path)


def _append_value_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{name}")
        return
    cmd.extend([f"--{name}", str(value)])


def _act_cmd(cfg: DictConfig) -> list[str]:
    train = cfg.train
    cmd = [sys.executable, str(REPO_ROOT / train.script)]
    _append_value_arg(cmd, "dataset-root", _as_path(train.dataset_root))
    _append_value_arg(cmd, "task-metadata", _as_path(train.task_metadata))
    for key in (
        "output_dir",
        "job_name",
        "steps",
        "batch_size",
        "device",
        "num_workers",
        "save_freq",
        "log_freq",
        "eval_freq",
        "chunk_size",
        "n_action_steps",
        "n_obs_steps",
        "task_conditioning",
    ):
        arg = key.replace("_", "-")
        value = _as_path(train[key]) if key == "output_dir" else train[key]
        _append_value_arg(cmd, arg, value)
    _append_value_arg(cmd, "dry-run", bool(train.dry_run))
    return cmd


def _act_lerobot_args(cfg: DictConfig) -> list[str]:
    train = cfg.train
    if str(train.task_conditioning) != "off":
        raise ValueError(
            "ACT torchrun launches LeRobot directly and does not materialize task-conditioned datasets. "
            "Use train.task_conditioning=off with a prebuilt task-conditioned dataset, or run the "
            "single-process train_act_policy.py wrapper first to create one."
        )
    dataset_root = Path(str(_as_path(train.dataset_root)))
    return [
        f"--dataset.repo_id=local/{dataset_root.resolve().name}",
        f"--dataset.root={dataset_root.resolve()}",
        "--policy.type=act",
        f"--output_dir={_as_path(train.output_dir)}",
        f"--job_name={train.job_name}",
        f"--policy.device={train.device}",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
        f"--num_workers={train.num_workers}",
        f"--batch_size={train.batch_size}",
        f"--optimizer.lr={train.lr}",
        f"--policy.optimizer_lr={train.lr}",
        f"--policy.optimizer_lr_backbone={train.lr}",
        f"--policy.chunk_size={train.chunk_size}",
        f"--policy.n_action_steps={train.n_action_steps}",
        f"--policy.n_obs_steps={train.n_obs_steps}",
        f"--steps={train.steps}",
        f"--save_freq={train.save_freq}",
        f"--log_freq={train.log_freq}",
        f"--eval_freq={train.eval_freq}",
        f"--dataset.video_backend={train.dataset_video_backend}",
    ]


def _offline_serl_cmd(cfg: DictConfig) -> list[str]:
    train = cfg.train
    cmd = [sys.executable, str(REPO_ROOT / train.script)]
    for key in (
        "dataset_root",
        "task_metadata",
        "output_dir",
        "job_name",
        "steps",
        "batch_size",
        "device",
        "lr",
        "gamma",
        "tau",
        "bc_weight",
        "cql_weight",
        "action_horizon",
        "hidden_dim",
        "num_layers",
        "reward_mode",
        "obs_mode",
        "missing_task_vector",
        "critic_init",
        "critic_checkpoint",
        "act_checkpoint",
        "act_warmstart_mode",
        "save_every",
    ):
        value = train[key]
        if key in {"dataset_root", "task_metadata", "output_dir", "critic_checkpoint", "act_checkpoint"}:
            value = _as_path(value)
        _append_value_arg(cmd, key.replace("_", "-"), value)
    _append_value_arg(cmd, "include-task-vector", bool(train.include_task_vector))
    _append_value_arg(cmd, "dry-run", bool(train.dry_run))
    return cmd


def _vision_offline_serl_cmd(cfg: DictConfig) -> list[str]:
    train = cfg.train
    cmd = [sys.executable, str(REPO_ROOT / train.script)]
    for key in (
        "dataset_root",
        "act_checkpoint",
        "output_dir",
        "job_name",
        "steps",
        "batch_size",
        "device",
        "lr",
        "adapter_lr",
        "act_lr",
        "critic_lr",
        "gamma",
        "tau",
        "bc_weight",
        "cql_weight",
        "adapter_penalty_weight",
        "act_preservation_weight",
        "smoothness_weight",
        "action_horizon",
        "actor_mode",
        "adapter_hidden_dim",
        "adapter_num_layers",
        "adapter_scale",
        "adapter_delta_clip",
        "action_clip",
        "reward_mode",
        "dataset_video_backend",
        "save_every",
    ):
        value = train[key]
        if key in {"dataset_root", "act_checkpoint", "output_dir"}:
            value = _as_path(value)
        _append_value_arg(cmd, key.replace("_", "-"), value)
    if bool(train.freeze_act):
        cmd.append("--freeze-act")
    else:
        cmd.append("--no-freeze-act")
    if train.get("camera_keys") is not None:
        cmd.append("--camera-keys")
        cmd.extend(str(v) for v in train.camera_keys)
    _append_value_arg(cmd, "dry-run", bool(train.dry_run))
    return cmd


def _online_gazebo_cmd(cfg: DictConfig) -> list[str]:
    train = cfg.train
    cmd = [sys.executable, str(REPO_ROOT / train.script)]
    for key in (
        "checkpoint",
        "act_torchscript",
        "output_dir",
        "workspace_dir",
        "engine_config",
        "sim_distrobox",
        "ground_truth",
        "gazebo_gui",
        "launch_rviz",
        "max_episodes",
        "max_steps",
        "updates",
        "batch_size",
        "replay_capacity",
        "max_minutes",
        "per_trial_timeout_sec",
        "device",
        "gamma",
        "tau",
        "adapter_lr",
        "critic_lr",
        "adapter_penalty_weight",
        "act_preservation_weight",
        "adapter_delta_clip",
        "action_clip",
        "critic_init",
        "critic_checkpoint",
        "critic_only_steps",
        "actor_update_delay",
        "task_family",
        "target_port_index",
        "target_card_index",
        "target_card_valid",
        "task_context_json",
    ):
        value = train[key]
        if key in {"checkpoint", "act_torchscript", "output_dir", "workspace_dir", "engine_config", "critic_checkpoint"}:
            value = _as_path(value)
        if key in {"ground_truth", "gazebo_gui", "launch_rviz"} and isinstance(value, bool):
            value = str(value).lower()
        _append_value_arg(cmd, key.replace("_", "-"), value)
    if not bool(train.include_images):
        cmd.append("--no-include-images")
    _append_value_arg(cmd, "allow-zero-images", bool(train.allow_zero_images))
    _append_value_arg(cmd, "dry-run", bool(train.dry_run))
    return cmd


def build_command(cfg: DictConfig) -> list[str]:
    stage = str(cfg.train.stage)
    if stage == "bc":
        return _act_cmd(cfg)
    if stage == "offline_adapter":
        if "train_vision_offline_serl.py" in str(cfg.train.script):
            return _vision_offline_serl_cmd(cfg)
        return _offline_serl_cmd(cfg)
    if stage == "online_adapter":
        return _online_gazebo_cmd(cfg)
    raise ValueError(f"Unsupported train.stage: {stage!r}")


def maybe_wrap_torchrun(cmd: list[str], cfg: DictConfig, selected_devices: list[int]) -> list[str]:
    distributed = cfg.hardware.get("distributed", {})
    if not bool(distributed.get("enabled", False)):
        return cmd
    if str(distributed.get("launcher", "torchrun")) != "torchrun":
        raise ValueError(f"Unsupported distributed launcher: {distributed.get('launcher')!r}")
    stage = str(cfg.train.stage)
    nproc = distributed.get("nproc_per_node")
    nproc_per_node = int(nproc) if nproc is not None else len(selected_devices)
    if nproc_per_node < 1:
        raise ValueError("hardware.distributed.nproc_per_node must be >= 1")
    wrapped = [sys.executable, "-m", "torch.distributed.run"]
    if bool(distributed.get("standalone", True)):
        wrapped.append("--standalone")
    wrapped.extend(["--nnodes", str(int(distributed.get("nnodes", 1)))])
    wrapped.extend(["--nproc-per-node", str(nproc_per_node)])
    if stage == "bc":
        wrapped.extend(["--module", "lerobot.scripts.lerobot_train"])
        wrapped.extend(_act_lerobot_args(cfg))
    elif stage == "offline_adapter":
        script_cmd = cmd[1:] if cmd and Path(str(cmd[0])).name.startswith("python") else cmd
        wrapped.extend(script_cmd)
    else:
        raise ValueError(f"Hydra torchrun/DDP is not implemented for train.stage={stage!r}")
    return wrapped


def _uses_direct_act_torchrun(cfg: DictConfig) -> bool:
    return bool(cfg.hardware.get("distributed", {}).get("enabled", False)) and str(cfg.train.stage) == "bc"


def _uses_lerobot_act_output_dir(cfg: DictConfig) -> bool:
    return str(cfg.train.stage) == "bc"


def _prepare_run_metadata(cfg: DictConfig, cuda_summary: dict[str, Any]) -> None:
    run_dir = Path(str(cfg.run.output_dir))
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(cfg, run_dir / "resolved_config.yaml", resolve=True)
    write_json(run_dir / "resolved_config.json", resolved)
    write_json(run_dir / "git_info.json", git_info(REPO_ROOT))
    if bool(cfg.train.get("include_task_vector", False)) or cfg.model.get("task_vector_dim", 0):
        write_json(run_dir / "task_encoding_schema.json", task_encoding_schema())
    write_json(run_dir / "hardware_selection.json", cuda_summary)


@hydra.main(version_base=None, config_path="../../../configs", config_name="experiment/hybrid_nominal_sfp2nic")
def main(cfg: DictConfig) -> int:
    hardware = cfg.hardware
    selected = select_cuda_devices(
        cuda_devices=list(hardware.cuda_devices) if hardware.cuda_devices is not None else None,
        auto_select_free_devices=bool(hardware.auto_select_free_devices),
        num_devices=int(hardware.num_devices),
        min_free_memory_gb=hardware.min_free_memory_gb,
    )
    cuda_summary = apply_cuda_visible_devices(selected) if bool(hardware.set_cuda_visible_devices) else {
        "selected_physical_gpus": selected,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "effective_training_device": str(hardware.effective_device),
    }
    cmd = maybe_wrap_torchrun(build_command(cfg), cfg, selected)
    if bool(hardware.dry_run_selection) or bool(cfg.train.get("dry_run", False)):
        print(json.dumps({"cuda": cuda_summary, "command": cmd}, indent=2, sort_keys=True))
    defer_metadata = _uses_lerobot_act_output_dir(cfg) and not bool(hardware.dry_run_selection)
    if not defer_metadata:
        _prepare_run_metadata(cfg, cuda_summary)
    if bool(hardware.dry_run_selection):
        return 0
    result = subprocess.run(cmd, check=False, cwd=REPO_ROOT).returncode
    if defer_metadata:
        _prepare_run_metadata(cfg, cuda_summary)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
