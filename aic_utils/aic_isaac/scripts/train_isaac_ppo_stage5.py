#!/usr/bin/env python3
"""Launch Stage 5 Isaac Lab PPO/RSL-RL training for AIC."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ISAAC_TRAIN = (
    REPO_ROOT / "aic_utils" / "aic_isaac" / "aic_isaaclab" / "scripts" / "rsl_rl" / "train.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="AIC-Task-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument(
        "--randomization-profile",
        choices=["none", "light", "heavy"],
        default="light",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("logs/rsl_rl"))
    parser.add_argument(
        "--isaaclab",
        default=os.environ.get("ISAACLAB_LAUNCHER", "isaaclab"),
        help="Isaac Lab launcher command or path. Defaults to ISAACLAB_LAUNCHER or 'isaaclab'.",
    )
    parser.add_argument("--run-name", default="stage5_ppo_smoke")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--load-run", default=None)
    parser.add_argument("--init-policy-checkpoint", type=Path, default=None)
    parser.add_argument("--insertion-distance-weight", type=float, default=0.0)
    parser.add_argument("--insertion-lateral-weight", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Additional raw argument passed to Isaac Lab train.py. Repeat as needed.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    if args.init_policy_checkpoint is not None:
        raise NotImplementedError(
            "--init-policy-checkpoint is reserved for a future checkpoint bridge. "
            "Current Stage 5 PPO starts from scratch or resumes an RSL-RL-native checkpoint."
        )

    cmd = [
        args.isaaclab,
        "-p",
        str(ISAAC_TRAIN),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        str(args.seed),
        "--run_name",
        args.run_name,
    ]
    if args.headless:
        cmd.append("--headless")
    cmd.append("--enable_cameras")
    if args.device:
        cmd.extend(["--device", args.device])
    if args.resume:
        cmd.append("--resume")
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.load_run:
        cmd.extend(["--load_run", args.load_run])
    cmd.extend(args.extra_arg)

    env = os.environ.copy()
    env["AIC_ISAAC_RANDOMIZATION_PROFILE"] = args.randomization_profile
    env["AIC_ISAAC_INSERTION_DISTANCE_WEIGHT"] = str(args.insertion_distance_weight)
    env["AIC_ISAAC_INSERTION_LATERAL_WEIGHT"] = str(args.insertion_lateral_weight)
    env["AIC_ISAAC_DISABLE_CAMERAS"] = "0"
    env["AIC_ISAAC_OUTPUT_DIR"] = str(args.output_dir)
    return cmd, env


def main() -> int:
    args = parse_args()
    cmd, env = build_command(args)
    rendered_env = (
        f"AIC_ISAAC_RANDOMIZATION_PROFILE={env['AIC_ISAAC_RANDOMIZATION_PROFILE']} "
        f"AIC_ISAAC_INSERTION_DISTANCE_WEIGHT={env['AIC_ISAAC_INSERTION_DISTANCE_WEIGHT']} "
        f"AIC_ISAAC_INSERTION_LATERAL_WEIGHT={env['AIC_ISAAC_INSERTION_LATERAL_WEIGHT']}"
    )
    rendered = " ".join(cmd)
    if args.dry_run:
        print(f"{rendered_env} {rendered}")
        return 0
    launcher = shutil.which(cmd[0]) if os.path.basename(cmd[0]) == cmd[0] else cmd[0]
    if launcher is None or not Path(launcher).exists():
        raise FileNotFoundError(
            f"Isaac Lab launcher '{cmd[0]}' was not found. Run this inside the Isaac Lab "
            "container, add 'isaaclab' to PATH, or pass --isaaclab /path/to/isaaclab.sh "
            "or set ISAACLAB_LAUNCHER."
        )
    return subprocess.run(cmd, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
