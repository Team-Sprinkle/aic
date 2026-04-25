#!/usr/bin/env python3
"""Generate, smooth, and print/run the official teacher replay path."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.generate_piecewise import (
    PiecewiseGeneratorConfig,
    generate_piecewise_file,
)
from aic_teacher_official.postprocess import postprocess_file
from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.vlm_planner import call_gpt5_mini_delta_planner

from official_teacher_generate_piecewise import _vector
from official_teacher_replay import build_command


def default_run_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    timestamp = args.timestamp or datetime.now().strftime("%Y_%m%d_%H%M%S")
    run_name = f"{args.trial_label}_{timestamp}"
    planner = (
        Path(args.root_dir)
        / args.task_family
        / "vlm_planner"
        / args.scene_count_label
        / args.attempt_label
        / run_name
    )
    postprocessed = (
        Path(args.root_dir)
        / args.task_family
        / "vlm_planner_postprocessed"
        / args.scene_count_label
        / args.attempt_label
        / run_name
    )
    return planner, postprocessed


def recording_command(args: argparse.Namespace) -> str:
    cmd = [
        "bash",
        "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
        "--engine-config",
        args.engine_config,
        "--policy-class",
        "aic_teacher_official.OfficialTeacherReplay",
        "--teacher-trajectory",
        args.smooth_output,
        "--teacher-action-mode",
        args.action_mode,
        "--dataset-repo-id",
        args.dataset_repo_id,
        "--dataset-root",
        args.dataset_root,
        "--gazebo-gui",
        str(args.gazebo_gui).lower(),
        "--launch-rviz",
        str(args.launch_rviz).lower(),
        "--startup-delay-sec",
        str(args.startup_delay_sec),
        "--per-trial-timeout-sec",
        str(args.per_trial_timeout_sec),
        "--recorder-drain-sec",
        str(args.recorder_drain_sec),
        "--require-recorder-save-log",
        str(args.require_recorder_save_log).lower(),
    ]
    return " ".join(shlex.quote(part) for part in cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--piecewise-output", default="artifacts/piecewise_trajectory.json")
    parser.add_argument("--smooth-output", default="artifacts/smooth_trajectory.json")
    parser.add_argument(
        "--use-dataset-layout",
        action="store_true",
        help="Write artifacts under outputs/trajectory_datasets/.../vlm_planner* layout.",
    )
    parser.add_argument("--root-dir", default="outputs/trajectory_datasets")
    parser.add_argument("--task-family", default="sfp_to_nic")
    parser.add_argument("--scene-count-label", default="nic_cards_2")
    parser.add_argument("--attempt-label", default="n1")
    parser.add_argument("--trial-label", default="trial9")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--task-name", default="Insert cable into target port")
    parser.add_argument("--start-position", default="-0.35,0.35,0.32")
    parser.add_argument("--port-position", default="-0.10,0.45,0.12")
    parser.add_argument("--orientation-xyzw", default="1,0,0,0")
    parser.add_argument("--approach-offset", default="-0.08,-0.08,0.22")
    parser.add_argument("--engine-config", default="./outputs/configs/random_trials_10.yaml")
    parser.add_argument("--dataset-repo-id", default="${HF_USER}/official_teacher_dataset")
    parser.add_argument("--dataset-root", default="./outputs/lerobot_datasets")
    parser.add_argument(
        "--action-mode",
        choices=["relative_delta_gripper_tcp", "absolute_cartesian_pose_base_link"],
        default="relative_delta_gripper_tcp",
    )
    parser.add_argument("--context-json")
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--vlm-model", default="gpt-5-mini")
    parser.add_argument("--max-vlm-calls", type=int, default=20)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--gazebo-gui", action="store_true")
    parser.add_argument("--launch-rviz", action="store_true")
    parser.add_argument("--startup-delay-sec", type=int, default=8)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=0)
    parser.add_argument("--recorder-drain-sec", type=int, default=120)
    parser.add_argument("--require-recorder-save-log", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build artifacts and print commands.")
    parser.add_argument("--run", action="store_true", help="Run the replay policy command only.")
    parser.add_argument("--record", action="store_true", help="Run the official per-trial recording launcher.")
    args = parser.parse_args()

    planner_root = None
    postprocessed_root = None
    if args.use_dataset_layout:
        planner_root, postprocessed_root = default_run_roots(args)
        args.piecewise_output = str(planner_root / "piecewise_trajectory.json")
        args.smooth_output = str(postprocessed_root / "smooth_trajectory.json")
        args.dataset_root = str(postprocessed_root / "raw_dataset")

    context = OfficialTeacherContext.load_json(args.context_json) if args.context_json else None
    vlm_delta_plan = None
    if args.use_vlm:
        vlm_context = context or OfficialTeacherContext(
            start_position=_vector(args.start_position, length=3, name="--start-position"),
            port_position=_vector(args.port_position, length=3, name="--port-position"),
            orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
            diagnostics={"source": "explicit_cli"},
        )
        vlm_delta_plan = call_gpt5_mini_delta_planner(
            vlm_context,
            image_paths=[Path(path) for path in args.image],
            max_calls=args.max_vlm_calls,
            model=args.vlm_model,
        )

    config = PiecewiseGeneratorConfig(
        start_position=_vector(args.start_position, length=3, name="--start-position"),
        port_position=_vector(args.port_position, length=3, name="--port-position"),
        orientation_xyzw=_vector(args.orientation_xyzw, length=4, name="--orientation-xyzw"),
        approach_offset=_vector(args.approach_offset, length=3, name="--approach-offset"),
        task_name=args.task_name,
        context=context,
        vlm_delta_plan=vlm_delta_plan,
    )
    piecewise = generate_piecewise_file(config, args.piecewise_output)
    smooth = postprocess_file(args.piecewise_output, args.smooth_output, args.sample_dt)
    if planner_root is not None and postprocessed_root is not None:
        (planner_root / "metadata").mkdir(parents=True, exist_ok=True)
        (postprocessed_root / "metadata").mkdir(parents=True, exist_ok=True)
        (planner_root / "metadata" / "run_roots.json").write_text(
            __import__("json").dumps(
                {
                    "planner_root": str(planner_root),
                    "postprocessed_root": str(postprocessed_root),
                    "piecewise_output": args.piecewise_output,
                    "smooth_output": args.smooth_output,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {len(piecewise.waypoints)} piecewise waypoints to {args.piecewise_output}")
    print(f"Wrote {len(smooth.waypoints)} smooth waypoints to {args.smooth_output}")

    replay_shell = build_command(args.smooth_output, args.action_mode)[-1]
    record_shell = recording_command(args)
    print("\nReplay command:")
    print(replay_shell)
    print("\nLeRobot recording command:")
    print(record_shell)

    if args.record:
        raise SystemExit(subprocess.call(record_shell, shell=True))
    if args.run:
        raise SystemExit(subprocess.call(build_command(args.smooth_output, args.action_mode)))
    if not args.dry_run:
        print("\nDry-run default: commands printed but not executed. Pass --run or --record to execute.")


if __name__ == "__main__":
    main()
