#!/usr/bin/env python3
"""Run and bundle live agentic teleop failure-debug attempts."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "aic_teacher_official"))

from aic_teacher_official.debug_recorder import (  # noqa: E402
    DebugRecorder,
    build_failure_analysis_payload,
    build_failure_analysis_prompt,
    environment_metadata,
    read_json_if_exists,
    write_bundle,
    write_image_manifest,
    write_json,
)
from aic_teacher_official.postprocess import postprocess_file  # noqa: E402
from aic_teacher_official.trajectory import PiecewiseTrajectory, SmoothTrajectory  # noqa: E402

from analyze_agent_teleop_failure import run_analysis  # noqa: E402


def _run_id() -> str:
    return "agent_teleop_failure_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _recording_command(
    args: argparse.Namespace,
    *,
    attempt_dir: Path,
    policy_class: str,
    dataset_root: Path,
    results_root: Path,
    tmp_dir: Path,
    smooth_path: Path | None = None,
) -> list[str]:
    cmd = [
        "bash",
        "./aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh",
        "--engine-config",
        args.config,
        "--policy-class",
        policy_class,
        "--dataset-repo-id",
        f"local/agent_teleop_failure_{attempt_dir.name}_{policy_class.split('.')[-1]}",
        "--dataset-root",
        str(dataset_root),
        "--results-root",
        str(results_root),
        "--tmp-dir",
        str(tmp_dir),
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
        "--remove-bag-data",
        str(args.remove_bag_data).lower(),
    ]
    if args.sim_distrobox:
        cmd.extend(["--sim-distrobox", args.sim_distrobox])
    if smooth_path is not None:
        cmd.extend(
            [
                "--teacher-trajectory",
                str(smooth_path),
                "--teacher-action-mode",
                args.action_mode,
            ]
        )
    return cmd


def _planner_env(args: argparse.Namespace, attempt_dir: Path) -> dict[str, str]:
    env = {
        "AIC_OFFICIAL_TEACHER_PIECEWISE_OUTPUT": str(attempt_dir / "piecewise_trajectory.json"),
        "AIC_OFFICIAL_TEACHER_IMAGE_DIR": str(attempt_dir / "live_observation_images"),
        "AIC_OFFICIAL_TEACHER_DEBUG_OUTPUT_DIR": str(attempt_dir),
        "AIC_OFFICIAL_TEACHER_MAX_VLM_CALLS": str(args.max_vlm_calls),
        "AIC_OFFICIAL_TEACHER_USE_VLM": "true",
        "AIC_OFFICIAL_TEACHER_IMAGE_SAMPLE_PERIOD_SEC": str(args.image_sample_period),
        "AIC_OFFICIAL_TEACHER_IMAGE_CAPTURE_DURATION_SEC": str(args.image_capture_duration),
        "AIC_OFFICIAL_TEACHER_MAX_PLANNER_IMAGES": str(args.max_images),
    }
    return env


def _run_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdout_path: Path,
    stderr_path: Path,
    env_updates: dict[str, str] | None = None,
) -> int:
    env = None
    if env_updates is not None:
        import os

        env = {**os.environ, **env_updates}
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w",
        encoding="utf-8",
    ) as stderr:
        process = subprocess.run(cmd, cwd=cwd, text=True, stdout=stdout, stderr=stderr, env=env)
    return int(process.returncode)


def _write_attempt_trace(args: argparse.Namespace, attempt_dir: Path, metadata: dict[str, Any]) -> None:
    piecewise_path = attempt_dir / "piecewise_trajectory.json"
    smooth_path = attempt_dir / "smooth_trajectory.json"
    recorder = DebugRecorder(attempt_dir, sample_period=args.sample_period)
    if piecewise_path.exists() and smooth_path.exists():
        piecewise = PiecewiseTrajectory.load_json(piecewise_path)
        smooth = SmoothTrajectory.load_json(smooth_path)
        recorder.sample_smooth_trajectory(
            smooth,
            piecewise=piecewise,
            planner_prompt=read_json_if_exists(attempt_dir / "planner_prompt.json"),
            planner_response=read_json_if_exists(attempt_dir / "planner_response.json"),
            parsed_decision=piecewise.metadata.planning.get("vlm_delta_plan"),
            final_score_reward={
                "value": None,
                "available": False,
                "reason": "see_replay_results_score_summary_csv",
            },
        )
    else:
        recorder.record_event(
            "live_planner_or_postprocess_failed_before_trace",
            {
                "piecewise_exists": piecewise_path.exists(),
                "smooth_exists": smooth_path.exists(),
            },
        )
    recorder.write_trace(metadata=metadata)


def run_attempt(args: argparse.Namespace, run_dir: Path, attempt_index: int) -> int:
    attempt_dir = run_dir / f"attempt_{attempt_index}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed + attempt_index - 1

    planner_cmd = _recording_command(
        args,
        attempt_dir=attempt_dir,
        policy_class="aic_teacher_official.OfficialTeacherOraclePlanner",
        dataset_root=attempt_dir / "planner_dataset",
        results_root=attempt_dir / "planner_results",
        tmp_dir=attempt_dir / "planner_tmp",
    )
    planner_exit = _run_command(
        planner_cmd,
        cwd=REPO_ROOT,
        stdout_path=attempt_dir / "planner_record_stdout.txt",
        stderr_path=attempt_dir / "planner_record_stderr.txt",
        env_updates=_planner_env(args, attempt_dir),
    )

    postprocess_error = None
    if (attempt_dir / "piecewise_trajectory.json").exists():
        try:
            postprocess_file(
                attempt_dir / "piecewise_trajectory.json",
                attempt_dir / "smooth_trajectory.json",
                args.planner_sample_dt,
            )
        except Exception as ex:
            postprocess_error = {"type": type(ex).__name__, "message": str(ex)}
    else:
        postprocess_error = {"type": "MissingPiecewiseTrajectory", "message": "planner pass did not write piecewise_trajectory.json"}

    replay_exit = None
    replay_cmd: list[str] = []
    if postprocess_error is None:
        replay_cmd = _recording_command(
            args,
            attempt_dir=attempt_dir,
            policy_class="aic_teacher_official.OfficialTeacherReplay",
            dataset_root=attempt_dir / "replay_dataset",
            results_root=attempt_dir / "replay_results",
            tmp_dir=attempt_dir / "replay_tmp",
            smooth_path=attempt_dir / "smooth_trajectory.json",
        )
        replay_exit = _run_command(
            replay_cmd,
            cwd=REPO_ROOT,
            stdout_path=attempt_dir / "replay_record_stdout.txt",
            stderr_path=attempt_dir / "replay_record_stderr.txt",
        )

    write_image_manifest(
        attempt_dir,
        validate=args.validate_images,
        describe=args.describe_images,
        dry_run_descriptions=args.dry_run_image_descriptions,
        max_images=args.max_images,
        model=args.image_description_model,
    )
    _write_attempt_trace(
        args,
        attempt_dir,
        metadata={
            "attempt_index": attempt_index,
            "seed": seed,
            "run_mode": "live_planner_record_then_replay_record",
            "planner_command": " ".join(shlex.quote(part) for part in planner_cmd),
            "replay_command": " ".join(shlex.quote(part) for part in replay_cmd) if replay_cmd else None,
            "environment": environment_metadata(REPO_ROOT),
        },
    )
    command_result = {
        "run_mode": "live_planner_record_then_replay_record",
        "seed": seed,
        "attempt_index": attempt_index,
        "planner_command": " ".join(shlex.quote(part) for part in planner_cmd),
        "planner_argv": planner_cmd,
        "planner_exit_code": planner_exit,
        "postprocess_error": postprocess_error,
        "replay_command": " ".join(shlex.quote(part) for part in replay_cmd) if replay_cmd else None,
        "replay_argv": replay_cmd,
        "replay_exit_code": replay_exit,
        "exit_code": replay_exit if replay_exit is not None else 1,
    }
    write_json(attempt_dir / "command_result.json", command_result)
    return int(command_result["exit_code"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="./outputs/configs/random_trials_10.yaml")
    parser.add_argument("--output-dir", default="outputs/failure_analysis")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--sample-period", type=float, default=0.5)
    parser.add_argument("--planner-sample-dt", type=float, default=0.05)
    parser.add_argument("--image-sample-period", type=float, default=0.5)
    parser.add_argument("--image-capture-duration", type=float, default=2.0)
    parser.add_argument("--validate-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--describe-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run-image-descriptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--analyze-with-gpt5", action="store_true")
    parser.add_argument("--dry-run-analysis", action="store_true")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--max-vlm-calls", type=int, default=20)
    parser.add_argument("--image-description-model", default="gpt-5-mini")
    parser.add_argument("--analysis-model", default="gpt-5")
    parser.add_argument("--sim-distrobox", default="")
    parser.add_argument(
        "--action-mode",
        choices=["relative_delta_gripper_tcp", "absolute_cartesian_pose_base_link"],
        default="relative_delta_gripper_tcp",
    )
    parser.add_argument("--gazebo-gui", action="store_true")
    parser.add_argument("--launch-rviz", action="store_true")
    parser.add_argument("--startup-delay-sec", type=int, default=8)
    parser.add_argument("--per-trial-timeout-sec", type=int, default=0)
    parser.add_argument("--recorder-drain-sec", type=int, default=120)
    parser.add_argument("--require-recorder-save-log", action="store_true")
    parser.add_argument("--remove-bag-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--zip", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_attempts < 1:
        raise SystemExit("--num-attempts must be positive")
    run_dir = Path(args.output_dir) / (args.run_id or _run_id())
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "run_config.json",
        {
            "args": vars(args),
            "environment": environment_metadata(REPO_ROOT),
            "debugging_opt_in": True,
            "live_images_required": True,
        },
    )
    exit_codes = []
    for attempt_index in range(1, args.num_attempts + 1):
        print(f"Running live attempt {attempt_index}/{args.num_attempts} -> {run_dir / f'attempt_{attempt_index}'}")
        exit_codes.append(run_attempt(args, run_dir, attempt_index))
    payload = build_failure_analysis_payload(run_dir)
    (run_dir / "prompt.md").write_text(build_failure_analysis_prompt(payload), encoding="utf-8")
    write_bundle(run_dir, include_zip=args.zip)
    if args.dry_run_analysis or args.analyze_with_gpt5:
        run_analysis(
            run_dir=run_dir,
            bundle=None,
            model=args.analysis_model,
            dry_run=not args.analyze_with_gpt5,
        )
    if any(code != 0 for code in exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
