#!/usr/bin/env python3
"""Run a small LeRobot ACT training job for an AIC dataset."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_robot_aic.dataset_schema import summarize_dataset_schema
from lerobot_robot_aic.task_metadata import (
    append_task_vectors_to_observation_state_dataset,
    infer_task_metadata_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--dataset-root",
        type=Path,
        help="Local LeRobot dataset root, for example <output_dir>/accepted_dataset.",
    )
    source.add_argument("--dataset-repo-id", help="Hugging Face or local LeRobot dataset repo id.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--job-name", default="aic_act_smoke")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", default="1e-4", help="Optimizer learning rate.")
    parser.add_argument("--dataset-video-backend", default="pyav")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="ACT action prediction chunk size in environment steps.",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=100,
        help="Number of ACT chunk actions executed per policy invocation.",
    )
    parser.add_argument(
        "--n-obs-steps",
        type=int,
        default=1,
        help="Observation history steps. Installed LeRobot ACT currently supports 1.",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument(
        "--policy-repo-id",
        default=None,
        help="Optional policy repo id for push/checkpoint metadata.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the lerobot-train command without executing it.",
    )
    parser.add_argument(
        "--task-metadata",
        type=Path,
        default=None,
        help="Optional manifests/accepted.csv or episode_task_metadata.jsonl with 10-dim task vectors.",
    )
    parser.add_argument(
        "--task-conditioning",
        choices=["off", "append-state", "zeros"],
        default="off",
        help=(
            "Task conditioning mode. append-state creates a derived local dataset with "
            "the 10-dim task vector appended to observation.state. zeros appends zeros "
            "when metadata is unavailable."
        ),
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument passed to lerobot-train. Repeat as needed.",
    )
    return parser.parse_args()


def _local_repo_id(dataset_root: Path) -> str:
    return f"local/{dataset_root.resolve().name}"


def build_lerobot_train_cmd(args: argparse.Namespace) -> list[str]:
    if args.n_action_steps > args.chunk_size:
        raise ValueError("--n-action-steps must be <= --chunk-size for ACT")
    if args.dataset_root:
        dataset_root = args.dataset_root.resolve()
        summary = summarize_dataset_schema(dataset_root)
        if summary.action_mode == "unknown":
            raise ValueError(
                "Could not infer dataset action mode from meta/info.json. "
                "Run inspect_dataset_schema.py and verify the action schema."
            )
        repo_id = _local_repo_id(dataset_root)
    else:
        dataset_root = None
        repo_id = args.dataset_repo_id

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        "--policy.type=act",
        f"--output_dir={args.output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        "--policy.push_to_hub=false",
        f"--wandb.enable={str(bool(args.wandb)).lower()}",
        f"--num_workers={args.num_workers}",
        f"--batch_size={args.batch_size}",
        f"--optimizer.lr={args.lr}",
        f"--policy.optimizer_lr={args.lr}",
        f"--policy.optimizer_lr_backbone={args.lr}",
        f"--policy.chunk_size={args.chunk_size}",
        f"--policy.n_action_steps={args.n_action_steps}",
        f"--policy.n_obs_steps={args.n_obs_steps}",
        f"--steps={args.steps}",
        f"--dataset.video_backend={args.dataset_video_backend}",
    ]
    if dataset_root is not None:
        cmd.append(f"--dataset.root={dataset_root}")
    if args.policy_repo_id:
        cmd.append(f"--policy.repo_id={args.policy_repo_id}")
    cmd.extend(args.extra_arg)
    return cmd


def prepare_task_conditioned_dataset(args: argparse.Namespace) -> None:
    if args.task_conditioning == "off":
        return
    if not args.dataset_root:
        raise ValueError("--task-conditioning requires --dataset-root so a local derived dataset can be created")
    dataset_root = args.dataset_root.resolve()
    manifest_path = args.task_metadata.resolve() if args.task_metadata else infer_task_metadata_path(dataset_root)
    if args.task_conditioning == "append-state" and manifest_path is None:
        raise FileNotFoundError(
            "Task conditioning requested but no manifest was provided or found at "
            "<dataset_parent>/manifests/accepted.csv"
        )
    output_dir = args.output_dir.resolve()
    derived_root = output_dir.parent / f"{output_dir.name}_task_conditioned_dataset"
    if args.dry_run:
        source = str(manifest_path) if manifest_path else "zero-vector fallback"
        print(
            f"[dry-run] would create task-conditioned dataset at {derived_root} "
            f"from {dataset_root} using {source}"
        )
        return
    append_task_vectors_to_observation_state_dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_root=derived_root,
        missing="zeros" if args.task_conditioning == "zeros" else "error",
    )
    args.dataset_root = derived_root


def main() -> int:
    args = parse_args()
    prepare_task_conditioned_dataset(args)
    cmd = build_lerobot_train_cmd(args)
    rendered = " ".join(str(part) for part in cmd)
    if args.dry_run:
        print(rendered)
        return 0
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
