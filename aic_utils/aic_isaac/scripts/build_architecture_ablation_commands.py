#!/usr/bin/env python3
"""Generate reproducible Isaac SERL architecture ablation commands.

The script reads an existing ``train_config.json`` with ``argv``/``command``,
then writes short, bounded command files for history/vision ablations.  It does
not launch training; each generated command keeps the original run's reward,
success checker, reset config, and policy path unless explicitly overridden.
"""

from __future__ import annotations

import argparse
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    sets: dict[str, list[str]]
    enables: tuple[str, ...] = ()
    disables: tuple[str, ...] = ()
    removes: tuple[str, ...] = ()


def _load_argv(path: Path) -> list[str]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    argv = cfg.get("argv")
    if isinstance(argv, list) and argv:
        return [str(x) for x in argv]
    command = cfg.get("command")
    if isinstance(command, str) and command.strip():
        return shlex.split(command)
    raise ValueError(f"{path} has neither argv nor command")


def _strip_flag(argv: list[str], flag: str) -> list[str]:
    negated = f"--no-{flag[2:]}" if flag.startswith("--") else None
    out: list[str] = []
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token == flag or (negated is not None and token == negated):
            idx += 1
            while idx < len(argv) and not argv[idx].startswith("--"):
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def _set_flag(argv: list[str], flag: str, values: list[str]) -> list[str]:
    return [*_strip_flag(argv, flag), flag, *values]


def _enable_flag(argv: list[str], flag: str) -> list[str]:
    return [*_strip_flag(argv, flag), flag]


def _disable_flag(argv: list[str], flag: str) -> list[str]:
    if not flag.startswith("--"):
        raise ValueError(f"boolean flag must start with --: {flag}")
    return [*_strip_flag(argv, flag), f"--no-{flag[2:]}"]


def _common_smoke_sets(output_dir: Path, run_name: str, steps: int, num_envs: int, seed: int) -> dict[str, list[str]]:
    return {
        "--output_dir": [str(output_dir)],
        "--run_name": [run_name],
        "--steps": [str(int(steps))],
        "--updates": [str(int(steps))],
        "--num_envs": [str(int(num_envs))],
        "--seed": [str(int(seed))],
        "--save_every_steps": [str(max(1, int(steps) // 2))],
        "--save_latest_every_steps": [str(max(1, int(steps) // 2))],
        "--diagnostics_every": ["1"],
        "--log_every": ["1"],
        "--max_logged_image_steps": ["8"],
        "--image_log_every": ["2"],
    }


def _container_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return str(path)
    return str(Path("aic") / rel)


def _apply(argv: list[str], variant: Variant) -> list[str]:
    out = list(argv)
    for flag in variant.removes:
        out = _strip_flag(out, flag)
    for flag in variant.disables:
        out = _disable_flag(out, flag)
    for flag in variant.enables:
        out = _enable_flag(out, flag)
    for flag, values in variant.sets.items():
        out = _set_flag(out, flag, values)
    return out


def _variants(output_dir: Path, steps: int, num_envs: int, seed: int) -> list[Variant]:
    return [
        Variant(
            name="critic_convnext_hist4",
            description="Critic-only ConvNeXt Tiny ImageNet with critic state history 4; actor path unchanged.",
            sets={
                **_common_smoke_sets(output_dir, "arch_critic_convnext_hist4_smoke", steps, num_envs, seed),
                "--critic_image_encoder_override": ["convnext_tiny_imagenet"],
                "--critic_state_history_steps": ["4"],
                "--actor_state_history_steps": ["1"],
            },
        ),
        Variant(
            name="actorcritic_hist4_convnext",
            description="Actor and critic state history 4 plus ConvNeXt Tiny critic. This is a new-architecture branch.",
            sets={
                **_common_smoke_sets(output_dir, "arch_actorcritic_hist4_convnext_smoke", steps, num_envs, seed + 1),
                "--critic_image_encoder_override": ["convnext_tiny_imagenet"],
                "--critic_state_history_steps": ["4"],
                "--actor_state_history_steps": ["4"],
            },
        ),
        Variant(
            name="critic_resnet18_imagenet_hist4",
            description="Intermediate vision ablation: ImageNet ResNet18 critic with critic state history 4.",
            sets={
                **_common_smoke_sets(output_dir, "arch_critic_resnet18_imagenet_hist4_smoke", steps, num_envs, seed + 2),
                "--critic_image_encoder_override": ["resnet18_imagenet"],
                "--critic_state_history_steps": ["4"],
                "--actor_state_history_steps": ["1"],
            },
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-train-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--command-output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67430)
    parser.add_argument("--docker-container", default="")
    args = parser.parse_args()

    base_argv = _load_argv(args.base_train_config.resolve())
    command_dir = (args.command_output_dir or (args.output_dir / "commands")).resolve()
    command_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir.resolve()
    argv_output_dir = Path(_container_path(output_dir)) if args.docker_container else output_dir

    manifest: dict[str, Any] = {
        "base_train_config": str(args.base_train_config.resolve()),
        "output_dir": str(output_dir),
        "argv_output_dir": str(argv_output_dir),
        "steps": int(args.steps),
        "num_envs": int(args.num_envs),
        "variants": [],
    }
    for variant in _variants(argv_output_dir, int(args.steps), int(args.num_envs), int(args.seed)):
        argv = _apply(base_argv, variant)
        command_text = shlex.join(argv)
        if args.docker_container:
            command_text = (
                "docker exec -w /workspace/isaaclab "
                + shlex.quote(str(args.docker_container))
                + " bash -lc "
                + shlex.quote("./isaaclab.sh -p " + command_text)
            )
        command_path = command_dir / f"{variant.name}.sh"
        command_path.write_text(command_text + "\n", encoding="utf-8")
        manifest["variants"].append(
            {
                "name": variant.name,
                "description": variant.description,
                "command_path": str(command_path),
                "run_name": variant.sets["--run_name"][0],
                "sets": variant.sets,
            }
        )
    manifest_path = command_dir / "architecture_ablation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
