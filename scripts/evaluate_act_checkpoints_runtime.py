#!/usr/bin/env python3
"""Evaluate saved ACT checkpoints with the rootless AIC runtime container.

This is intended to run as a sidecar while LeRobot training writes checkpoints.
It polls ``<run-dir>/checkpoints/*/pretrained_model`` and evaluates each new
checkpoint once through the official AIC runtime stack in the ``aic_eval``
container.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import shlex
import sys
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_utils" / "gazebo_rl"))
from gazebo_rl.score_parser import parse_scoring_yaml


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--container", default="aic_eval")
    parser.add_argument("--workspace-host", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-container", default=None, help="Repository mount path; defaults to --workspace-host.")
    parser.add_argument("--artifact-host-root", type=Path,
                        help="Optional additional existing bind mount containing model/data/evaluation artifacts.")
    parser.add_argument("--artifact-container-root",
                        help="Container path paired with --artifact-host-root; does not change the code workspace.")
    parser.add_argument(
        "--docker-host",
        default=f"unix:///run/user/{os.getuid()}/docker.sock",
        help="Rootless Docker socket.",
    )
    parser.add_argument("--checkpoint-glob", default="checkpoints/[0-9]*/pretrained_model")
    parser.add_argument(
        "--eval-subdir",
        default="runtime_eval",
        help="Subdirectory under --run-dir where per-checkpoint eval outputs are written.",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once-existing", action="store_true")
    parser.add_argument("--max-runtime-sec", type=float, default=12.0)
    parser.add_argument("--max-simulation-sec", type=float, default=None,
                        help="Optional ACT simulation duration from center-camera timestamps; max-runtime-sec remains a wall watchdog.")
    parser.add_argument("--start-delay-sec", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--control-clock", choices=["wall", "simulation"], default="wall",
                        help="TorchScript ACT command cadence; choose according to the collection clock.")
    parser.add_argument("--image-channel-order", choices=["rgb", "bgr"], default=None,
                        help="Camera channel order expected by this ACT checkpoint's recorded training images.")
    parser.add_argument("--n-action-steps", type=int, default=4)
    parser.add_argument("--temporal-ensemble-coeff", type=float, default=None,
                        help="Optional absolute-ACT chunk ensembling; larger coefficients favor newer observations.")
    parser.add_argument("--translation-limit-mode", choices=["component", "norm"], default="component",
                        help="Norm limiting preserves the direction of an absolute target's translation.")
    parser.add_argument("--delta-pose-reference", choices=["controller", "observation"], default="controller",
                        help="For observation-relative labels, compose the target from observed TCP and send it in base_link.")
    parser.add_argument("--policy-device", default="cuda", help="Device passed to RunACT, e.g. cuda or cpu.")
    parser.add_argument("--policy-module", default="aic_example_policies.ros.RunACT")
    parser.add_argument("--act-torchscript", type=Path, default=None)
    parser.add_argument("--record-rollout", action="store_true", help="Save camera snapshots about once per simulation second.")
    parser.add_argument("--evaluation-purpose", choices=["development", "training_scene_diagnostic", "final_reliability"],
                        default="development", help="Declare whether scenes are for development, fitting diagnostics or the untouched final test.")
    parser.add_argument("--diagnostic-ground-truth", action="store_true",
                        help="Enable privileged geometry only for a CheatCode expert diagnostic.")
    parser.add_argument("--corrective-data-dir", type=Path,
                        help="Output for the privileged CollectCorrectiveCheatCode training-data pilot.")
    parser.add_argument("--corrective-perturbation-scale", type=float, default=1.,
                        help="Scale expert execution perturbations; zero explicitly records clean nominal demonstrations.")
    parser.add_argument("--corrective-execution-frame", choices=["gripper/tcp", "base_link"], default="gripper/tcp")
    parser.add_argument("--corrective-student-probability", type=float, default=0.,
                        help="Optional probability of bounded ACT segments during privileged correction collection.")
    parser.add_argument("--connect-scoring-world-frames", action=argparse.BooleanOptionalAction, default=True,
                        help="Publish the fixed world-to-aic_world identity needed by trajectory scoring; does not relay object poses.")
    parser.add_argument(
        "--serl-adapter-delta-clip",
        type=float,
        default=None,
        help="Optional runtime override for ACT-adapter SERL delta clip. Defaults unset so checkpoint metadata is used.",
    )
    parser.add_argument(
        "--serl-action-clip",
        type=float,
        default=None,
        help="Optional runtime override for ACT-adapter SERL final action clip. Defaults unset so checkpoint/default is used.",
    )
    parser.add_argument("--command-mode", required=True, choices=["none", "velocity", "delta_pose", "absolute_pose"],
                        help="Explicit execution mode. none is an intentional interface-only smoke check.")
    parser.add_argument("--command-frame", default=None,
                        help="Defaults to gripper/tcp for delta_pose and base_link otherwise.")
    parser.add_argument("--max-translation-delta", type=float, default=0.02)
    parser.add_argument("--max-rotation-delta", type=float, default=0.2)
    parser.add_argument("--translation-deadband", type=float, default=5e-4,
                        help="ACT command deadband in meters; use 0 for unmodified small insertion commands.")
    parser.add_argument("--rotation-deadband", type=float, default=1e-3,
                        help="ACT command deadband in radians.")
    parser.add_argument("--sim-wait-sec", type=float, default=25.0)
    parser.add_argument(
        "--eval-attempts",
        type=int,
        default=1,
        help="Number of clean container attempts per checkpoint before giving up.",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=10.0,
        help="Wall-clock delay between retry attempts after runtime failure.",
    )
    parser.add_argument("--readiness-timeout-sec", type=int, default=120)
    parser.add_argument("--engine-timeout-sec", type=float, default=300.0)
    parser.add_argument(
        "--engine-config",
        default=None,
        help="Engine configuration at a host or container path under one of the declared mounts.",
    )
    args = parser.parse_args(argv)
    if (args.artifact_host_root is None) != (args.artifact_container_root is None):
        parser.error("--artifact-host-root and --artifact-container-root must be supplied together")
    if args.eval_attempts < 1:
        parser.error("--eval-attempts must be at least 1")
    if args.translation_deadband < 0 or args.rotation_deadband < 0:
        parser.error("Command deadbands must be nonnegative")
    if args.temporal_ensemble_coeff is not None:
        import math
        if (not math.isfinite(args.temporal_ensemble_coeff) or args.temporal_ensemble_coeff < 0
                or args.n_action_steps != 1 or args.command_mode != "absolute_pose"
                or not args.policy_module.endswith(".RunACTTorchScript")):
            parser.error("Temporal ensembling requires absolute TorchScript ACT, n-action-steps=1 and a finite nonnegative coefficient")
    return args


def prepare_args(args: argparse.Namespace) -> None:
    args.run_dir = args.run_dir.resolve()
    args.workspace_host = args.workspace_host.resolve()
    args.workspace_container = args.workspace_container or str(args.workspace_host)
    if (args.artifact_host_root is None) != (args.artifact_container_root is None):
        raise ValueError("Artifact host/container roots must be supplied together")
    if args.artifact_host_root is not None:
        args.artifact_host_root = args.artifact_host_root.resolve()
        if not args.artifact_host_root.is_dir():
            raise ValueError(f"Artifact host root does not exist: {args.artifact_host_root}")
        args.artifact_container_root = str(Path(os.path.normpath(args.artifact_container_root)))
        if not Path(args.artifact_container_root).is_absolute():
            raise ValueError("Artifact container root must be an absolute path")
        if args.artifact_host_root == Path("/") or Path(args.artifact_container_root) == Path("/"):
            raise ValueError("Artifact mapping must name a specific directory, not the filesystem root")
    host_to_container(args.run_dir, args)
    host_to_container(args.run_dir / args.eval_subdir, args)
    if not math.isfinite(args.max_runtime_sec) or args.max_runtime_sec <= 0:
        raise ValueError("Wall watchdog duration must be finite and positive")
    if args.max_simulation_sec is not None:
        if not math.isfinite(args.max_simulation_sec) or args.max_simulation_sec <= 0:
            raise ValueError("Simulation duration must be finite and positive")
        if not args.policy_module.endswith(".RunACTTorchScript"):
            raise ValueError("Fixed simulation duration is supported only by RunACTTorchScript")
    expected_pose_frame = {"delta_pose": "gripper/tcp", "absolute_pose": "base_link"}.get(args.command_mode)
    args.command_frame = args.command_frame or expected_pose_frame or "base_link"
    if args.delta_pose_reference == "observation" and (
            args.command_mode != "delta_pose" or not args.policy_module.endswith(".RunACTTorchScript")):
        raise ValueError("Observation-referenced deltas require TorchScript ACT with delta_pose actions")
    if expected_pose_frame and args.command_frame != expected_pose_frame:
        raise ValueError(f"{args.command_mode} commands require frame {expected_pose_frame}")
    args.engine_config = args.engine_config or str(Path(args.workspace_container) / "aic_engine/config/sample_config.yaml")
    corrective = args.policy_module.endswith(".CollectCorrectiveCheatCode")
    if args.diagnostic_ground_truth and not args.policy_module.endswith((".CheatCode", ".CollectCorrectiveCheatCode")):
        raise ValueError("Privileged geometry is restricted to a separately labeled CheatCode diagnostic")
    if corrective and (not args.diagnostic_ground_truth or args.corrective_data_dir is None):
        raise ValueError("Corrective collection requires explicit privileged geometry and a data directory")
    if not 0 <= args.corrective_perturbation_scale <= 1:
        raise ValueError("Expert perturbation scale must be in [0, 1]")
    if not 0 <= args.corrective_student_probability <= 1:
        raise ValueError("Student correction probability must be in [0, 1]")
    if args.corrective_student_probability and (not corrective or args.act_torchscript is None
                                               or args.corrective_execution_frame != "base_link"):
        raise ValueError("Student corrections require the privileged collector, an ACT export and base_link execution")
    if corrective:
        expected_mode = "absolute_pose" if args.corrective_execution_frame == "base_link" else "delta_pose"
        if args.command_mode != expected_mode or args.command_frame != args.corrective_execution_frame:
            raise ValueError("Expert execution frame must match the declared command mode and frame")
    if args.corrective_data_dir is not None:
        if not corrective:
            raise ValueError("A corrective-data directory is only valid for the privileged training-data collector")
        args.corrective_data_dir = args.corrective_data_dir.resolve()
        host_to_container(args.corrective_data_dir, args)
    if not args.run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {args.run_dir}")
    args.engine_config_host = mapped_input_to_host(Path(args.engine_config), args)
    args.engine_config = host_to_container(args.engine_config_host, args)
    config = yaml.safe_load(args.engine_config_host.read_text())
    trials = config.get("trials") if isinstance(config, dict) else None
    if not isinstance(trials, dict) or not trials:
        raise ValueError("Engine config must contain a nonempty trials mapping")
    args.expected_trial_names = sorted(trials)
    args.expected_trial_order = list(trials)
    args.expected_runtime_tasks = [{"trial": trial_name, "task_id": task_id,
                                   **{key: task[key] for key in ("target_module_name", "port_name", "plug_name", "plug_type")}}
                                  for trial_name, trial in trials.items()
                                  for task_id, task in trial["tasks"].items()] if args.max_simulation_sec is not None else []
    if args.policy_module.endswith(("RunACTTorchScript", "RunACTAdapterSERL")) and args.act_torchscript is None:
        raise ValueError("This policy requires --act-torchscript")
    if args.act_torchscript is not None:
        args.act_torchscript = args.act_torchscript.resolve()
        host_to_container(args.act_torchscript, args)
        if not args.act_torchscript.is_file():
            raise ValueError(f"Missing ACT export: {args.act_torchscript}")
        metadata = json.loads(args.act_torchscript.with_suffix(".json").read_text())
        args.image_channel_order = args.image_channel_order or metadata.get("image_channel_order", "rgb")
        representation = metadata.get("action_representation", "delta_pose")
        if args.command_mode != "none" and (
            (representation == "absolute_pose") != (args.command_mode == "absolute_pose")
        ):
            raise ValueError("ACT export action representation and command mode disagree")
        normalizer_root = Path(metadata["checkpoint_dir"])
        if not normalizer_root.is_absolute():
            normalizer_root = args.workspace_host / normalizer_root
        normalizer_root = mapped_input_to_host(normalizer_root, args)
        normalizer = normalizer_root / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        host_to_container(normalizer, args)
        if not normalizer.is_file():
            raise ValueError(f"Missing ACT normalizer: {normalizer}")
        args.normalizer_path = normalizer
        if not 1 <= args.n_action_steps <= int(metadata["chunk_size"]):
            raise ValueError("n-action-steps must fit the exported action chunk")
    args.image_channel_order = args.image_channel_order or "rgb"


def evaluation_signature(checkpoint: Path, args: argparse.Namespace) -> str:
    ignored = {"once_existing", "poll_seconds", "eval_attempts", "retry_delay_sec"}
    settings = {key: value for key, value in vars(args).items() if key not in ignored}
    files = sorted(checkpoint.iterdir()) if checkpoint.is_dir() else [checkpoint]
    if args.act_torchscript is not None:
        files += [args.act_torchscript, args.act_torchscript.with_suffix(".json"), args.normalizer_path]
    identities = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file()]
    payload = {"settings": settings, "files": identities,
               "runtime_sources": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                   for name, path in runtime_sources(args).items()},
               "engine_config": args.engine_config_host.read_text(), "schema_version": 2}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def runtime_sources(args):
    paths = ["scripts/evaluate_act_checkpoints_runtime.py", "aic_model/aic_model/policy.py",
             "aic_model/aic_model/aic_model.py", "aic_model/aic_model/rollout_recording.py",
             "aic_utils/lerobot_robot_aic/lerobot_robot_aic/runtime_features.py",
             "aic_utils/lerobot_robot_aic/lerobot_robot_aic/act_state_contract.py",
             "aic_utils/lerobot_robot_aic/lerobot_robot_aic/task_encoding.py"]
    # Known repository policy layouts; missing optional files are skipped.
    module_path = args.policy_module.replace(".", "/") + ".py"
    paths += [module_path.split("/", 1)[0] + "/" + module_path,
              "aic_example_policies/aic_example_policies/ros/RunACTTorchScript.py"]
    if args.policy_module.endswith(".CollectCorrectiveCheatCode"):
        paths += ["aic_example_policies/aic_example_policies/ros/CheatCode.py",
                  "aic_example_policies/aic_example_policies/ros/cheatcode_start.py"]
    return {name: args.workspace_host / name for name in sorted(set(paths))
            if (args.workspace_host / name).is_file()}


def audit_simulation_stops(policy_log: Path, engine_log: Path, expected_trials: list[str],
                           expected_tasks: list[dict], requested_seconds: float) -> dict:
    """Join unique policy start/stop events to actual engine goals and task identities."""
    malformed, mapping_errors = [], []
    policy_lines = policy_log.read_text().splitlines() if policy_log.is_file() else []

    def timestamp(line):
        match = re.search(r"\[(\d+\.\d+)\]", line)
        if match is None:
            raise ValueError("Missing ROS log timestamp")
        return float(match.group(1))

    def events(marker):
        result = []
        for line in policy_lines:
            if marker not in line:
                continue
            try:
                record = json.loads(line.split(marker, 1)[1])
                episode = record.get("runtime_episode_id")
                if not isinstance(episode, str) or not episode:
                    raise ValueError("Missing unique runtime episode ID")
                result.append({**record, "event_wall_timestamp": timestamp(line)})
            except (ValueError, TypeError, AttributeError):
                malformed.append(line)
        return result

    starts, records = events("ACT_RUNTIME_START "), events("ACT_RUNTIME_STOP ")
    start_ids = [record["runtime_episode_id"] for record in starts]
    stop_ids = [record["runtime_episode_id"] for record in records]
    if (len(set(start_ids)) != len(start_ids) or len(set(stop_ids)) != len(stop_ids)
            or set(start_ids) != set(stop_ids)):
        mapping_errors.append("Missing, duplicated, or unmatched runtime episode IDs")
    engine_trials, goals = [], []
    current_trial = None
    for line in engine_log.read_text().splitlines() if engine_log.is_file() else []:
        trial = re.search(r"Starting trial '([^']+)'", line)
        if trial:
            current_trial = trial.group(1)
            engine_trials.append(current_trial)
        goal = re.search(r"Sending InsertCable goal for task \[([^\]]+)\]", line)
        if goal:
            try:
                goals.append({"trial": current_trial, "task_id": goal.group(1), "wall_timestamp": timestamp(line)})
            except ValueError:
                malformed.append(line)
    if engine_trials != expected_trials:
        mapping_errors.append("Actual engine trial order differs from configuration")
    if [(goal["trial"], goal["task_id"]) for goal in goals] != [(task["trial"], task["task_id"]) for task in expected_tasks]:
        mapping_errors.append("Actual engine goal sequence differs from configuration")
    if len(starts) != len(records) or len(records) != len(expected_tasks):
        mapping_errors.append("Runtime start/stop count differs from expected tasks")
    identity_keys = ("task_id", "target_module_name", "port_name", "plug_name", "plug_type")
    for index, (expected, goal) in enumerate(zip(expected_tasks, goals)):
        next_goal = goals[index + 1]["wall_timestamp"] if index + 1 < len(goals) else float("inf")
        candidates = [record for record in starts if goal["wall_timestamp"] <= record["event_wall_timestamp"] < next_goal]
        if len(candidates) != 1:
            mapping_errors.append(f"Expected one policy start in engine task interval: {expected['trial']}")
            continue
        started = candidates[0]
        ended = [record for record in records if record["runtime_episode_id"] == started["runtime_episode_id"]]
        if len(ended) != 1:
            continue
        record = ended[0]
        record["trial"] = expected["trial"]
        record["policy_start_wall_timestamp"] = started["event_wall_timestamp"]
        identity_matches = all(started.get(key) == record.get(key) == expected[key] for key in identity_keys)
        record["engine_task_mapping_valid"] = (identity_matches
            and started["event_wall_timestamp"] <= record["event_wall_timestamp"] < next_goal
            and goal["trial"] == expected["trial"] and goal["task_id"] == expected["task_id"])
        if not record["engine_task_mapping_valid"]:
            mapping_errors.append(f"Policy target identity or timing differs from engine task: {expected['trial']}")
    for record in records:
        duration = record.get("simulation_elapsed_sec")
        record["matched_simulation_budget"] = (record.get("engine_task_mapping_valid") is True
            and record.get("reason") == "simulation_limit" and record.get("simulation_budget_reached") is True
            and record.get("wall_watchdog_shortened") is False and record.get("simulation_limit_sec") == requested_seconds
            and isinstance(duration, (int, float)) and math.isfinite(duration) and duration >= requested_seconds - 1e-6)
    complete = (len(records) == len(expected_tasks) and not malformed and not mapping_errors
                and all(record["matched_simulation_budget"] for record in records))
    return {"requested_simulation_seconds": requested_seconds, "complete": complete,
            "expected_trials": expected_trials, "expected_tasks": expected_tasks, "records": records,
            "engine_trial_order": engine_trials, "engine_task_order": goals, "mapping_errors": mapping_errors,
            "wall_watchdog_shortened_trials": [record.get("trial") for record in records
                                                if record.get("wall_watchdog_shortened") is True],
            "malformed_stop_records": malformed,
            "classification": "matched_simulation_duration" if complete else "incomplete_simulation_duration"}


def evaluation_complete(summary: dict, expected_trials: list[str]) -> bool:
    if summary.get("simulation_duration_complete") is False:
        return False
    if summary.get("policy_ready") is not True or summary.get("engine_returncode") != 0 or summary.get("failure_reason"):
        return False
    path = summary.get("scoring_yaml")
    if not path or not Path(path).is_file():
        return False
    try:
        score = parse_scoring_yaml(path)
    except (OSError, ValueError, yaml.YAMLError):
        return False
    trials = score["trials"]
    return (score["total_score"] is not None and set(trials) == set(expected_trials)
            and all(all(value is not None for value in trial["tier_scores"].values())
                    for trial in trials.values()))


def docker_cmd(args: argparse.Namespace, *parts: str) -> list[str]:
    return ["docker", "--host", args.docker_host, *parts]


def host_to_container(path: Path, args: argparse.Namespace) -> str:
    resolved = Path(path).resolve()
    for host, container in sorted(path_mappings(args), key=lambda pair: len(pair[0].parts), reverse=True):
        if resolved.is_relative_to(host):
            return str(container / resolved.relative_to(host))
    raise ValueError(f"{resolved} is outside the declared workspace/artifact host roots")


def path_mappings(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    mappings = [(args.workspace_host.resolve(), Path(args.workspace_container))]
    if getattr(args, "artifact_host_root", None) is not None:
        mappings.append((args.artifact_host_root.resolve(), Path(args.artifact_container_root)))
    if len(mappings) == 2 and (mappings[0][0] == mappings[1][0] or mappings[0][1] == mappings[1][1]):
        if mappings[0] != mappings[1]:
            raise ValueError("Workspace and artifact mappings have conflicting roots")
    return mappings


def mapped_input_to_host(path: Path, args: argparse.Namespace) -> Path:
    """Accept declared host paths or the legacy container-path config spelling."""
    resolved = path.resolve()
    mappings = path_mappings(args)
    if any(resolved.is_relative_to(host) for host, _ in mappings):
        return resolved
    container_path = Path(os.path.normpath(str(path)))
    for host, container in sorted(mappings, key=lambda pair: len(pair[1].parts), reverse=True):
        if container_path.is_relative_to(container):
            candidate = (host / container_path.relative_to(container)).resolve()
            if not candidate.is_relative_to(host):
                raise ValueError(f"Mapped path escapes its declared host root: {path}")
            return candidate
    raise ValueError(f"{path} is outside the declared workspace/artifact roots")


def run_capture(cmd: list[str], log_path: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    return result


def container_bash(args: argparse.Namespace, script: str) -> list[str]:
    return docker_cmd(args, "exec", "-i", args.container, "bash", "-lc", script)


def start_long_container_bash(args: argparse.Namespace, script: str, log_path: Path) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        container_bash(args, script),
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def restart_container(args: argparse.Namespace, log_path: Path) -> int:
    result = run_capture(docker_cmd(args, "restart", "--time", "2", args.container), log_path=log_path, timeout=120)
    return result.returncode


def stop_process(proc: subprocess.Popen[str], timeout: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def run_engine_monitoring_sim(
    args: argparse.Namespace,
    engine_script: str,
    engine_log_path: Path,
    sim_proc: subprocess.Popen[str],
    sim_log_path: Path,
) -> tuple[int | None, str | None]:
    """Run aic_engine while failing fast if the simulator process exits."""
    engine_log_path.parent.mkdir(parents=True, exist_ok=True)
    with engine_log_path.open("w", encoding="utf-8") as log_file:
        engine_proc = subprocess.Popen(
            container_bash(args, engine_script),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        start = time.monotonic()
        sim_failed_after_score_at: float | None = None
        while True:
            engine_returncode = engine_proc.poll()
            if engine_returncode is not None:
                return engine_returncode, None

            sim_returncode = sim_proc.poll()
            engine_log = ""
            if engine_log_path.exists():
                try:
                    engine_log = engine_log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    engine_log = ""
            score_reported = "Finished scoring trial, total score is:" in engine_log
            if sim_returncode is not None and score_reported:
                if sim_failed_after_score_at is None:
                    sim_failed_after_score_at = time.monotonic()
                if time.monotonic() - sim_failed_after_score_at <= 45.0:
                    time.sleep(1.0)
                    continue
                stop_process(engine_proc)
                return None, (
                    "simulator exited after a trial score was reported, but aic_engine "
                    "did not finish within the post-score grace period "
                    f"(sim returncode={sim_returncode})"
                )
            if sim_returncode is not None:
                stop_process(engine_proc)
                return None, f"simulator exited while engine was running (returncode={sim_returncode})"

            if sim_log_path.exists():
                try:
                    sim_log = sim_log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    sim_log = ""
                if "ros_gz_container" in sim_log and "process has died" in sim_log and score_reported:
                    if sim_failed_after_score_at is None:
                        sim_failed_after_score_at = time.monotonic()
                    if time.monotonic() - sim_failed_after_score_at <= 45.0:
                        time.sleep(1.0)
                        continue
                    stop_process(engine_proc)
                    return None, (
                        "ros_gz_container died after a trial score was reported, but aic_engine "
                        "did not finish within the post-score grace period"
                    )
                if "ros_gz_container" in sim_log and "process has died" in sim_log:
                    stop_process(engine_proc)
                    return None, "ros_gz_container died while engine was running"

            elapsed = time.monotonic() - start
            if elapsed > args.engine_timeout_sec:
                stop_process(engine_proc)
                return None, f"aic_engine timed out after {args.engine_timeout_sec} seconds"

            time.sleep(1.0)


def wait_for_policy_ready(args: argparse.Namespace, node_name: str, action_name: str, log_path: Path) -> bool:
    script = f"""
source /ws_aic/install/setup.bash
export RMW_IMPLEMENTATION=rmw_zenoh_cpp
for i in $(seq 1 {args.readiness_timeout_sec}); do
  state=$(ros2 lifecycle get /{node_name} 2>/dev/null || true)
  actions=$(ros2 action list 2>/dev/null || true)
  if printf "%s" "$state" | grep -q "unconfigured" && printf "%s" "$actions" | grep -q "^/{action_name}$"; then
    echo ready
    echo "$state"
    ros2 action list
    exit 0
  fi
  sleep 1
done
echo not_ready
ros2 lifecycle get /{node_name} || true
ros2 action list || true
exit 1
"""
    result = run_capture(container_bash(args, script), log_path=log_path)
    return result.returncode == 0


def evaluate_checkpoint_once(
    checkpoint_path: Path,
    args: argparse.Namespace,
    eval_dir: Path,
    logs_dir: Path,
    attempt: int,
) -> dict[str, object]:
    step = checkpoint_path.parent.name if checkpoint_path.is_dir() else checkpoint_path.stem
    eval_dir.mkdir(parents=True, exist_ok=False)
    started_signature = evaluation_signature(checkpoint_path, args)
    source_hashes = {}
    for name, source in runtime_sources(args).items():
        content = source.read_bytes()
        target = eval_dir / "runtime_sources" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_hashes[name] = hashlib.sha256(content).hexdigest()

    checkpoint_container = host_to_container(checkpoint_path, args)
    eval_container = host_to_container(eval_dir, args)
    node_name = f"aic_model_act_{step}"
    action_name = f"insert_cable_act_{step}"

    summary: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_container": checkpoint_container,
        "eval_dir": str(eval_dir),
        "eval_dir_container": eval_container,
        "node_name": node_name,
        "action_name": action_name,
        "attempt": attempt,
        "runtime_source_sha256": source_hashes,
        "runtime_settings": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy_module": args.policy_module,
        "n_action_steps": args.n_action_steps,
        "control_clock": args.control_clock,
        "image_channel_order": args.image_channel_order,
        "command_mode": args.command_mode,
        "command_frame": args.command_frame,
        "translation_deadband": args.translation_deadband,
        "rotation_deadband": args.rotation_deadband,
        "serl_adapter_delta_clip_override": args.serl_adapter_delta_clip,
        "serl_action_clip_override": args.serl_action_clip,
        "serl_clip_override_note": (
            "None means the Gazebo SERL runtime uses values stored in the checkpoint config."
        ),
    }

    sim_proc = policy_proc = None
    try:
        if restart_container(args, logs_dir / "container_restart_before.log") != 0:
            raise RuntimeError("Container restart failed")
        sim_script = f"""
    set -e
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    cd {shlex.quote(str(args.workspace_container))}
    {f'ros2 run tf2_ros static_transform_publisher --frame-id world --child-frame-id aic_world --ros-args -p use_sim_time:=true > {shlex.quote(eval_container + "/world_frame.log")} 2>&1 &' if args.connect_scoring_world_frames and not args.diagnostic_ground_truth else ''}
    /entrypoint.sh ground_truth:={'true' if args.diagnostic_ground_truth else 'false'} start_aic_engine:=false gazebo_gui:=false launch_rviz:=false
    """
        sim_proc = start_long_container_bash(args, sim_script, logs_dir / "sim.log")
        time.sleep(args.sim_wait_sec)

        policy_script = f"""
    set -e
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    cd {shlex.quote(str(args.workspace_container))}
    export LD_LIBRARY_PATH={shlex.quote(str(args.workspace_container))}/.pixi/envs/default/lib:$LD_LIBRARY_PATH
    export PYTHONPATH={shlex.quote(str(args.workspace_container))}/.pixi/envs/default/lib/python3.12/site-packages:{shlex.quote(str(args.workspace_container))}/aic_model:{shlex.quote(str(args.workspace_container))}/scripts/pythonpath_bootstrap:$PYTHONPATH
    export AIC_CHECKOUT_PYTHONPATH={shlex.quote(str(args.workspace_container) + '/aic_model:' + str(args.workspace_container) + '/aic_example_policies')}
    export AIC_POLICY_RECORD_DIR={shlex.quote(str(eval_container) + '/rollout' if getattr(args, 'record_rollout', False) else '')}
    export AIC_ACT_POLICY_PATH={shlex.quote(str(checkpoint_container))}
    export AIC_ACT_TORCHSCRIPT={shlex.quote(host_to_container(args.act_torchscript, args) if args.act_torchscript else "")}
    export AIC_ACT_NORMALIZER_PATH={shlex.quote(host_to_container(args.normalizer_path, args) if args.act_torchscript else "")}
    export AIC_ACT_DEVICE={shlex.quote(str(args.policy_device))}
    export AIC_ACT_MAX_RUNTIME_SEC={args.max_runtime_sec}
    export AIC_ACT_MAX_SIMULATION_SEC={shlex.quote('' if args.max_simulation_sec is None else str(args.max_simulation_sec))}
    export AIC_ACT_START_DELAY_SEC={args.start_delay_sec}
    export AIC_ACT_CONTROL_HZ={args.control_hz}
    export AIC_ACT_CONTROL_CLOCK={args.control_clock}
    export AIC_ACT_IMAGE_CHANNEL_ORDER={args.image_channel_order}
    export AIC_CORRECTIVE_DATA_DIR={shlex.quote(host_to_container(args.corrective_data_dir, args)) if args.corrective_data_dir else "''"}
    export AIC_ACT_N_ACTION_STEPS={args.n_action_steps}
    export AIC_CORRECTIVE_PERTURBATION_SCALE={args.corrective_perturbation_scale}
    export AIC_CORRECTIVE_EXECUTION_FRAME={shlex.quote(args.corrective_execution_frame)}
    export AIC_CORRECTIVE_STUDENT_PROBABILITY={args.corrective_student_probability}
    export AIC_ACT_TRANSLATION_LIMIT_MODE={args.translation_limit_mode}
    export AIC_ACT_DELTA_POSE_REFERENCE={args.delta_pose_reference}
    export AIC_ACT_TEMPORAL_ENSEMBLE_COEFF={shlex.quote('' if args.temporal_ensemble_coeff is None else str(args.temporal_ensemble_coeff))}
    export AIC_ACT_COMMAND_MODE=none
    export AIC_ACT_RUNTIME_COMMAND_MODE={args.command_mode}
    export AIC_ACT_COMMAND_FRAME={shlex.quote(str(args.command_frame))}
    export AIC_ACT_MAX_TRANSLATION_DELTA={args.max_translation_delta}
    export AIC_ACT_MAX_ROTATION_DELTA={args.max_rotation_delta}
    export AIC_ACT_TRANSLATION_DEADBAND={args.translation_deadband}
    export AIC_ACT_ROTATION_DEADBAND={args.rotation_deadband}
    export AIC_SERL_CHECKPOINT={shlex.quote(str(checkpoint_container))}
    export AIC_SERL_ACT_TORCHSCRIPT={shlex.quote(host_to_container(args.act_torchscript, args) if args.act_torchscript else "")}
    export AIC_SERL_DEVICE={shlex.quote(str(args.policy_device))}
    export AIC_SERL_MAX_RUNTIME_SEC={args.max_runtime_sec}
    export AIC_SERL_START_DELAY_SEC={args.start_delay_sec}
    export AIC_SERL_CONTROL_HZ={args.control_hz}
    export AIC_SERL_N_ACTION_STEPS={args.n_action_steps}
    export AIC_SERL_COMMAND_MODE={args.command_mode}
    export AIC_SERL_COMMAND_FRAME={shlex.quote(str(args.command_frame))}
    export AIC_SERL_MAX_TRANSLATION_DELTA={args.max_translation_delta}
    export AIC_SERL_MAX_ROTATION_DELTA={args.max_rotation_delta}
    unset AIC_SERL_ADAPTER_DELTA_CLIP
    unset AIC_SERL_ACTION_CLIP
    {f"export AIC_SERL_ADAPTER_DELTA_CLIP={args.serl_adapter_delta_clip}" if args.serl_adapter_delta_clip is not None else ""}
    {f"export AIC_SERL_ACTION_CLIP={args.serl_action_clip}" if args.serl_action_clip is not None else ""}
    if [ -x .pixi/envs/default/bin/ros2 ]; then
      AIC_ROS2=.pixi/envs/default/bin/ros2
    else
      AIC_ROS2="pixi run ros2"
    fi
    $AIC_ROS2 run aic_model aic_model --ros-args \\
      -p use_sim_time:=true \\
      -p policy:={shlex.quote(str(args.policy_module))} \\
      -r __node:={node_name} \\
      -r /insert_cable:=/{action_name}
    """
        policy_proc = start_long_container_bash(args, policy_script, logs_dir / "policy.log")
        ready = wait_for_policy_ready(args, node_name, action_name, logs_dir / "readiness.log")
        summary["policy_ready"] = ready
        if ready:
            engine_script = f"""
    set -e
    source /ws_aic/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_zenoh_cpp
    cd {shlex.quote(str(args.workspace_container))}
    export AIC_RESULTS_DIR={shlex.quote(str(eval_container))}
    mkdir -p "$AIC_RESULTS_DIR"
    ros2 run aic_engine aic_engine --ros-args \\
      -r /insert_cable:=/{action_name} \\
      -p use_sim_time:=true \\
      -p config_file_path:={shlex.quote(str(args.engine_config))} \\
      -p model_node_name:={node_name} \\
      -p model_discovery_timeout_seconds:=60 \\
      -p model_configure_timeout_seconds:=120
    """
            try:
                engine_returncode, failure_reason = run_engine_monitoring_sim(
                    args, engine_script, logs_dir / "engine.log", sim_proc, logs_dir / "sim.log"
                )
                summary["engine_returncode"] = engine_returncode
                if failure_reason is not None:
                    summary["failure_reason"] = failure_reason
            except Exception as exc:
                summary["engine_returncode"] = None
                summary["failure_reason"] = f"engine launch failed: {exc}"
        else:
            summary["engine_returncode"] = None

    except Exception as exc:
        summary["failure_reason"] = f"Runtime failed: {exc}"
    finally:
        for proc in (policy_proc, sim_proc):
            if proc is not None:
                try:
                    stop_process(proc)
                except Exception as exc:
                    summary["failure_reason"] = f"Process cleanup failed: {exc}"
        try:
            if restart_container(args, logs_dir / "container_restart_after.log") != 0:
                summary["failure_reason"] = "Container cleanup restart failed"
        except Exception as exc:
            summary["failure_reason"] = f"Container cleanup failed: {exc}"

    scoring = eval_dir / "scoring.yaml"
    summary["scoring_yaml"] = str(scoring) if scoring.exists() else None
    if args.max_simulation_sec is not None:
        summary["simulation_duration_audit"] = audit_simulation_stops(
            logs_dir / "policy.log", logs_dir / "engine.log", args.expected_trial_order,
            args.expected_runtime_tasks, args.max_simulation_sec)
        summary["simulation_duration_complete"] = summary["simulation_duration_audit"]["complete"]
    summary["evaluation_complete"] = evaluation_complete(summary, args.expected_trial_names)
    summary["evaluation_kind"] = ("privileged_corrective_data_collection" if args.corrective_data_dir else
                                  "privileged_expert_diagnostic" if args.diagnostic_ground_truth else
                                  "recorded_trajectory_diagnostic" if args.policy_module.endswith("RunRecordedACTCommands") else
                                  "interface_smoke" if args.command_mode == "none" else
                                  "training_scene_diagnostic" if args.evaluation_purpose == "training_scene_diagnostic" else "policy_rollout")
    summary["evaluation_purpose"] = args.evaluation_purpose
    summary["evaluation_signature"] = started_signature
    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (eval_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def checkpoint_eval_step(checkpoint_path: Path) -> str:
    return checkpoint_path.parent.name if checkpoint_path.is_dir() else checkpoint_path.stem


def evaluate_checkpoint(checkpoint_path: Path, args: argparse.Namespace) -> dict[str, object]:
    lock_dir = args.workspace_host / "outputs" / ".runtime_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(f"{args.docker_host}/{args.container}".encode()).hexdigest()[:20]
    with (lock_dir / (identity + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another evaluation is using container {args.container}") from exc
        return _evaluate_checkpoint_locked(checkpoint_path, args)


def _evaluate_checkpoint_locked(checkpoint_path: Path, args: argparse.Namespace) -> dict[str, object]:
    host_to_container(checkpoint_path, args)
    step = checkpoint_eval_step(checkpoint_path)
    final_eval_dir = args.run_dir.resolve() / args.eval_subdir / step
    final_eval_dir.mkdir(parents=True, exist_ok=True)

    attempts: list[dict[str, object]] = []
    max_attempts = max(1, args.eval_attempts)
    for attempt in range(1, max_attempts + 1):
        # Keep previous failures and their score files separate from this attempt.
        attempt_number = 1
        while (final_eval_dir / f"attempt_{attempt_number:04d}").exists():
            attempt_number += 1
        eval_dir = final_eval_dir / f"attempt_{attempt_number:04d}"
        logs_dir = eval_dir / "logs"
        summary = evaluate_checkpoint_once(checkpoint_path, args, eval_dir, logs_dir, attempt)
        attempts.append(summary)
        if evaluation_complete(summary, args.expected_trial_names):
            if eval_dir != final_eval_dir:
                (final_eval_dir / "eval_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
                )
            (final_eval_dir / "attempts.json").write_text(
                json.dumps(attempts, indent=2, sort_keys=True), encoding="utf-8"
            )
            return summary
        if attempt < max_attempts:
            time.sleep(args.retry_delay_sec)

    summary = dict(attempts[-1])
    summary["attempts"] = attempts
    (final_eval_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (final_eval_dir / "attempts.json").write_text(
        json.dumps(attempts, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def iter_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = []
    for path in args.run_dir.glob(args.checkpoint_glob):
        host_to_container(path, args)
        if path.is_dir() and (path / "model.safetensors").exists():
            checkpoints.append(path)
        elif path.is_file() and path.suffix in {".pt", ".pth"}:
            checkpoints.append(path)
        elif args.corrective_data_dir is not None and path.is_file() and path.suffix == ".json":
            # Expert collection has a configuration artifact, not model weights.
            checkpoints.append(path)
    return sorted(checkpoints)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        prepare_args(args)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"Evaluation preflight failed: {exc}", file=sys.stderr)
        return 2
    evaluated: set[Path] = set()
    failed = False
    while True:
        try:
            checkpoints = iter_checkpoints(args)
        except ValueError as exc:
            print(f"Checkpoint path validation failed: {exc}", file=sys.stderr)
            return 2
        if args.once_existing and not checkpoints:
            print("No checkpoints matched; no evaluation was performed", file=sys.stderr)
            return 2
        for checkpoint in checkpoints:
            if checkpoint in evaluated:
                continue
            marker = args.run_dir / args.eval_subdir / checkpoint_eval_step(checkpoint) / "eval_summary.json"
            if marker.exists():
                try:
                    previous = json.loads(marker.read_text())
                except (OSError, ValueError):
                    previous = {}
                if (previous.get("evaluation_signature") == evaluation_signature(checkpoint, args)
                        and evaluation_complete(previous, args.expected_trial_names)):
                    evaluated.add(checkpoint)
                    continue
            try:
                summary = evaluate_checkpoint(checkpoint, args)
            except RuntimeError as exc:
                print(f"Evaluation could not start: {exc}", file=sys.stderr)
                return 1
            print(json.dumps(summary, sort_keys=True), flush=True)
            failed |= not evaluation_complete(summary, args.expected_trial_names)
            evaluated.add(checkpoint)
        if args.once_existing:
            return 1 if failed else 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
