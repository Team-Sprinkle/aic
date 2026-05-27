#!/usr/bin/env python3
"""Stage C-only launcher for Isaac online SERL curriculum runs.

The launcher materializes user-facing minimal YAMLs into fully specified child
episode YAMLs, shards them by GPU in curriculum order, and renders or runs one
independent trainer per GPU shard.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path

from isaac_episode_configs import _split_filenames, materialize_many_episode_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimal-yaml-dir", required=True, type=Path)
    parser.add_argument("--filenames", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--act-torchscript", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=["C"],
        default="C",
        help="Only Stage C is supported: one independent Isaac trainer per GPU shard.",
    )
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--isaaclab", default=os.environ.get("ISAACLAB_LAUNCHER", "isaaclab"))
    parser.add_argument(
        "--python-cmd",
        default=os.environ.get("AIC_PYTHON_CMD", "pixi run python"),
        help=(
            "Command used to launch the Python wrapper. Defaults to 'pixi run python'. "
            "Inside the Isaac container use '/workspace/isaaclab/aic/.pixi/envs/default/bin/python'."
        ),
    )
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
    save_every_steps: int,
    run_name: str,
    isaaclab: str,
    python_cmd: str,
    extra: list[str],
) -> list[str]:
    return [
        *shlex.split(python_cmd),
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
        "--save-every-steps",
        str(save_every_steps),
        "--isaaclab",
        isaaclab,
        *extra,
    ]


def _run_commands(commands: list[dict[str, object]], log_dir: Path) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[dict[str, object], Path, subprocess.Popen[bytes]]] = []
    for item in commands:
        log_path = log_dir / f"{item['name']}.log"
        log_file = log_path.open("wb")
        try:
            process = subprocess.Popen(item["command"], stdout=log_file, stderr=subprocess.STDOUT)
        finally:
            log_file.close()
        processes.append((item, log_path, process))

    status = 0
    for item, log_path, process in processes:
        code = process.wait()
        if code != 0 and status == 0:
            status = code
        if code != 0:
            print(f"Command {item['name']} failed with status {code}; log: {log_path}")
    return status


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
                        save_every_steps=args.save_every_steps,
                        run_name=f"isaac_online_serl_gpu_{gpu_id}",
                        isaaclab=args.isaaclab,
                        python_cmd=args.python_cmd,
                        extra=args.extra_train_arg,
                    ),
                ],
            }
        )

    plan = {
        "curriculum": summary,
        "stage": "C",
        "stage_description": "One independent Isaac trainer per GPU shard; processes run in parallel when --run is set.",
        "commands": commands,
        "multi_gpu_contract": (
            "Stage C only. Each GPU owns its shard, replay buffer, actor, critics, optimizer, metrics, "
            "and checkpoint. No central learner, worker transport, transition chunking, or policy sync is used."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "launch_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    print("Rendered commands:")
    for item in commands:
        print(shlex.join(item["command"]))
    if args.run:
        return _run_commands(commands, args.output_dir / "logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
