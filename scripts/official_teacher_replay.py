#!/usr/bin/env python3
"""Run or print the official teacher replay command."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))


def build_command(trajectory: str, action_mode: str = "relative_delta_gripper_tcp") -> list[str]:
    env_assignment = "AIC_OFFICIAL_TEACHER_TRAJECTORY=" + shlex.quote(trajectory)
    mode_assignment = " AIC_OFFICIAL_TEACHER_ACTION_MODE=" + shlex.quote(action_mode)
    return [
        "bash",
        "-lc",
        env_assignment
        + mode_assignment
        + " pixi run ros2 run aic_model aic_model --ros-args "
        + "-p use_sim_time:=true "
        + "-p policy:=aic_teacher_official.OfficialTeacherReplay",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="Smooth trajectory JSON.")
    parser.add_argument(
        "--action-mode",
        choices=["relative_delta_gripper_tcp", "absolute_cartesian_pose_base_link"],
        default="relative_delta_gripper_tcp",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the policy command. Without this, print a safe command stub.",
    )
    args = parser.parse_args()

    command = build_command(args.trajectory, args.action_mode)
    shell_command = command[-1]
    if not args.run:
        print("Start the official simulation/container and LeRobot recorder first.")
        print("Then run this replay policy command from the repo root:")
        print(shell_command)
        print()
        print(
            "For recording, pass this policy class to "
            "aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh:"
        )
        print("  --policy-class aic_teacher_official.OfficialTeacherReplay")
        print("  --teacher-trajectory", args.trajectory)
        print("  --teacher-action-mode", args.action_mode)
        return

    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
