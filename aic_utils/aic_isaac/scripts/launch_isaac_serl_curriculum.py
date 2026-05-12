#!/usr/bin/env python3
"""Render staged Isaac SERL commands for curriculum episode shards.

Stage 1 is the production path today: one Isaac trainer process consumes one
episode directory. Stage 2 renders one independent command per GPU shard so we
can smoke multi-GPU scheduling before wiring a shared replay/central learner.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import yaml

from isaac_episode_configs import _split_filenames, materialize_many_episode_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal-yaml-dir", required=True, type=Path)
    parser.add_argument("--filenames", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--act-torchscript", required=True, type=Path)
    parser.add_argument("--stage", choices=["single", "sharded"], default="single")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--extra-train-arg", action="append", default=[])
    return parser.parse_args()


def _train_cmd(
    *,
    episode_dir: Path,
    output_dir: Path,
    checkpoint: Path,
    act_torchscript: Path,
    num_envs: int,
    steps: int,
    updates: int,
    batch_size: int,
    warmup_steps: int,
    run_name: str,
    extra: list[str],
) -> list[str]:
    return [
        "pixi",
        "run",
        "python",
        "aic_utils/aic_isaac/scripts/train_isaac_online_serl.py",
        "--checkpoint",
        str(checkpoint),
        "--act-torchscript",
        str(act_torchscript),
        "--episode-config-dir",
        str(episode_dir),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
        "--num-envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--updates",
        str(updates),
        "--batch-size",
        str(batch_size),
        "--warmup-steps",
        str(warmup_steps),
        *extra,
    ]


def main() -> int:
    args = parse_args()
    curriculum_dir = args.output_dir / "curriculum"
    summary = materialize_many_episode_configs(
        input_dir=args.minimal_yaml_dir,
        output_dir=curriculum_dir,
        filenames=_split_filenames(args.filenames),
        max_gpus=args.max_gpus,
    )
    commands = []
    if args.stage == "single":
        commands.append(
            {
                "name": "single",
                "gpu_id": None,
                "episode_dir": summary["episodes_dir"],
                "command": _train_cmd(
                    episode_dir=Path(summary["episodes_dir"]),
                    output_dir=args.output_dir / "single",
                    checkpoint=args.checkpoint,
                    act_torchscript=args.act_torchscript,
                    num_envs=args.num_envs,
                    steps=args.steps,
                    updates=args.updates,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    run_name="isaac_online_serl_single",
                    extra=args.extra_train_arg,
                ),
            }
        )
    else:
        for shard in summary["shards"]:
            gpu_id = int(shard["gpu_id"])
            commands.append(
                {
                    "name": f"gpu_{gpu_id}",
                    "gpu_id": gpu_id,
                    "episode_dir": shard["episodes_dir"],
                    "command": [
                        "env",
                        f"CUDA_VISIBLE_DEVICES={gpu_id}",
                        *_train_cmd(
                            episode_dir=Path(shard["episodes_dir"]),
                            output_dir=args.output_dir / f"gpu_{gpu_id}",
                            checkpoint=args.checkpoint,
                            act_torchscript=args.act_torchscript,
                            num_envs=args.num_envs,
                            steps=args.steps,
                            updates=args.updates,
                            batch_size=args.batch_size,
                            warmup_steps=args.warmup_steps,
                            run_name=f"isaac_online_serl_gpu_{gpu_id}",
                            extra=args.extra_train_arg,
                        ),
                    ],
                }
            )
    plan = {"curriculum": summary, "stage": args.stage, "commands": commands}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "launch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("Rendered commands:")
    for item in commands:
        print(shlex.join(item["command"]))
    if args.run:
        for item in commands:
            status = subprocess.run(item["command"], check=False).returncode
            if status != 0:
                return status
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
