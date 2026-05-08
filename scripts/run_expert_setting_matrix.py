#!/usr/bin/env python3
"""Run expert-generation modes over a fixed setting matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


MODE_ENV: dict[str, dict[str, str]] = {
    "nominal": {
        "AIC_EXPERT_VALIDATION_MAX_GUARDED_SPEED_MPS": "0.09",
        "AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0040",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_TIMEOUT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC": "2.0",
        "AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC": "1.5",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_GAIN": "0.7",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_STEP_LIMIT_M": "0.0010",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_MAX_BIAS_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_NOMINAL_PRECONTACT_REALIGN_ON_GATE_FAIL": "true",
        "AIC_OFFICIAL_TEACHER_NOMINAL_ALLOW_LOW_FORCE_GATE_MISS_M": "0.0040",
        "AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC": "0.25",
        "AIC_OFFICIAL_TEACHER_MICRO_ALIGN_SETTLE_SEC": "0.18",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC": "0.60",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0005",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_GAIN": "0.35",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_RESIDUAL_M": "0.0006",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_SEC": "1.00",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0100",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_GAIN": "0.90",
        "AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO": "true",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_GAIN": "0.35",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_STEP_LIMIT_M": "0.0004",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_MAX_BIAS_M": "0.0040",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_LIMIT_N": "5.0",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_ENABLED": "true",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_RADIUS_M": "0.0008",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_EXTRA_DEPTH_M": "0.0030",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_FORCE_LIMIT_N": "8.0",
        "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M": "0.080",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M": "0.0100",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N": "6.0",
        "AIC_OFFICIAL_TEACHER_SC_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY": "true",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_FORCE_THRESHOLD_N": "12.0",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_Z_OFFSET_M": "-0.010",
        "AIC_OFFICIAL_TEACHER_SC_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N": "18.0",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SPEED_MPS": "0.06",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS": "0.010",
        "AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_SETTLE_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET": "false",
    },
    "nominalrecovery": {
        "AIC_EXPERT_VALIDATION_MAX_GUARDED_SPEED_MPS": "0.03",
        "AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0040",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_TIMEOUT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC": "2.0",
        "AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC": "1.2",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_GAIN": "0.7",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_STEP_LIMIT_M": "0.0010",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_MAX_BIAS_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_NOMINAL_ALLOW_LOW_FORCE_GATE_MISS_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_MODE": "base_z_absolute",
        "AIC_OFFICIAL_TEACHER_REQUIRE_MEASURED_BACKOFF": "true",
        "AIC_OFFICIAL_TEACHER_RECOVERY_REQUIRED_MEASURED_BACKOFF_M": "0.004",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_TO_PREINSERT_SEC": "1.5",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_TIMEOUT_SEC": "1.2",
        "AIC_OFFICIAL_TEACHER_RECOVERY_POST_ALIGN_RETURN_SEC": "0.7",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS": "true",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_GAIN": "0.75",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_STEP_LIMIT_M": "0.0020",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS_MAX_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC": "0.60",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0005",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_GAIN": "0.35",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_RESIDUAL_M": "0.0006",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_SEC": "1.00",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0100",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_GAIN": "0.90",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_PORT_ALIGN_SEC": "0.80",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_ALIGN_SETTLE_SEC": "0.20",
        "AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO": "true",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_GAIN": "0.35",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_STEP_LIMIT_M": "0.0004",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_MAX_BIAS_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_LIMIT_N": "5.0",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_ENABLED": "true",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_RADIUS_M": "0.0008",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_EXTRA_DEPTH_M": "0.0030",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_FORCE_LIMIT_N": "8.0",
        "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M": "0.080",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M": "0.0100",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N": "6.0",
        "AIC_OFFICIAL_TEACHER_SC_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY": "true",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_FORCE_THRESHOLD_N": "12.0",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_Z_OFFSET_M": "-0.010",
        "AIC_OFFICIAL_TEACHER_SC_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N": "18.0",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_MODE": "base_z_absolute",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_DISTANCE_M": "0.004",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_MAX_BACKOFF_DISTANCE_M": "0.020",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_MIN_BACKOFF_DISTANCE_M": "0.002",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_REQUIRED_MEASURED_BACKOFF_M": "0.002",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_RELEASE_FORCE_THRESHOLD_N": "3.0",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_STRICT_RELEASE_FORCE_THRESHOLD_N": "3.0",
        "AIC_OFFICIAL_TEACHER_RECOVERY_MAX_RETRIES": "1",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS": "0.010",
        "AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET": "false",
    },
    "recovery": {
        "AIC_EXPERT_VALIDATION_MAX_GUARDED_SPEED_MPS": "0.03",
        "AIC_OFFICIAL_TEACHER_ENABLE_LIVE_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_RECOVERY_INITIAL_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_TIMEOUT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_TRACKING_GATE_REALIGN_SEC": "2.0",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_GAIN": "0.7",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_STEP_LIMIT_M": "0.0010",
        "AIC_OFFICIAL_TEACHER_PREINSERT_SERVO_MAX_BIAS_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_RECOVERY_INDUCE_FAILURE": "false",
        "AIC_OFFICIAL_TEACHER_RECOVERY_INDUCE_LATERAL_OFFSET_M": "0.0",
        "AIC_OFFICIAL_TEACHER_RECOVERY_ENABLE_PLANNED_INTERVENTION": "false",
        "AIC_OFFICIAL_TEACHER_RECOVERY_PLANNED_INTERVENTION_Z_OFFSET": "0.030",
        "AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_INDUCED_GATE_MISS": "true",
        "AIC_OFFICIAL_TEACHER_RECOVERY_ALLOW_INITIAL_LOW_FORCE_GATE_MISS_M": "0.0085",
        "AIC_OFFICIAL_TEACHER_RECOVERY_BACKOFF_MODE": "tcp_away_from_port",
        "AIC_OFFICIAL_TEACHER_REQUIRE_MEASURED_BACKOFF": "true",
        "AIC_OFFICIAL_TEACHER_RECOVERY_REQUIRED_MEASURED_BACKOFF_M": "0.004",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RELEASE_FORCE_THRESHOLD_N": "1.5",
        "AIC_OFFICIAL_TEACHER_RECOVERY_STRICT_RELEASE_FORCE_THRESHOLD_N": "1.5",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_TO_PREINSERT_SEC": "1.5",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETURN_Z_TIMEOUT_SEC": "1.2",
        "AIC_OFFICIAL_TEACHER_RECOVERY_POST_ALIGN_RETURN_SEC": "0.7",
        "AIC_OFFICIAL_TEACHER_LOCAL_PREINSERT_ALIGN_SEC": "0.6",
        "AIC_OFFICIAL_TEACHER_PRE_INSERT_SETTLE_SEC": "0.25",
        "AIC_OFFICIAL_TEACHER_MICRO_ALIGN_SETTLE_SEC": "0.18",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0005",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_GAIN": "0.25",
        "AIC_OFFICIAL_TEACHER_PRECONTACT_PORT_ALIGN_RESIDUAL_M": "0.0010",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_SEC": "0.70",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_MAX_OFFSET_M": "0.0050",
        "AIC_OFFICIAL_TEACHER_SC_PRECONTACT_PORT_ALIGN_GAIN": "0.75",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_PORT_ALIGN_SEC": "0.80",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_ALIGN_SETTLE_SEC": "0.20",
        "AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M": "0.0055",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO": "true",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_GAIN": "0.35",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_STEP_LIMIT_M": "0.0004",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_MAX_BIAS_M": "0.0060",
        "AIC_OFFICIAL_TEACHER_SC_GUARDED_INSERT_LATERAL_SERVO_FORCE_LIMIT_N": "5.0",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_ENABLED": "true",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_SEC": "2.5",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_RADIUS_M": "0.0008",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_EXTRA_DEPTH_M": "0.0030",
        "AIC_OFFICIAL_TEACHER_SC_FINAL_SEAT_FORCE_LIMIT_N": "8.0",
        "AIC_OFFICIAL_TEACHER_SC_ENABLE_LIVE_Z_REPAIR": "true",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_START_Z_OFFSET_M": "0.080",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_LATERAL_ERROR_M": "0.0100",
        "AIC_OFFICIAL_TEACHER_SC_LIVE_Z_REPAIR_MAX_FORCE_DELTA_N": "6.0",
        "AIC_OFFICIAL_TEACHER_SC_PRESERVE_LIVE_LATERAL_ON_Z_REPAIR": "false",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY": "true",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_FORCE_THRESHOLD_N": "12.0",
        "AIC_OFFICIAL_TEACHER_SC_NO_EVENT_RECOVERY_Z_OFFSET_M": "-0.010",
        "AIC_OFFICIAL_TEACHER_SC_LOW_FORCE_INSERTION_IGNORE_THRESHOLD_N": "18.0",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_MODE": "base_z_absolute",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_BACKOFF_DISTANCE_M": "0.004",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_MAX_BACKOFF_DISTANCE_M": "0.020",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_MIN_BACKOFF_DISTANCE_M": "0.002",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_REQUIRED_MEASURED_BACKOFF_M": "0.002",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_RELEASE_FORCE_THRESHOLD_N": "3.0",
        "AIC_OFFICIAL_TEACHER_SC_RECOVERY_STRICT_RELEASE_FORCE_THRESHOLD_N": "3.0",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_HANDOFF_SPEED_MPS": "0.06",
        "AIC_OFFICIAL_TEACHER_CHEATCODE_INSERTION_SPEED_MPS": "0.010",
        "AIC_OFFICIAL_TEACHER_INSERTION_EXACT_POSITION_SETTLE_SEC": "0.35",
        "AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET": "false",
        "AIC_OFFICIAL_TEACHER_ABOVE_CONTACT_SOFT_REALIGN_FORCE_THRESHOLD_N": "4.0",
        "AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_HOLD_THRESHOLD_N": "3.0",
        "AIC_OFFICIAL_TEACHER_LOW_FORCE_INSERTION_HOLD_SEC": "0.05",
        "AIC_OFFICIAL_TEACHER_RECOVERY_RETRY_XY_BIAS": "false",
    },
}

MODE_ARGS: dict[str, list[str]] = {
    "nominal": ["--ft-threshold", "15.0"],
    # First try the clean nominal insertion. Recovery should trigger only on
    # clear contact, not small preload during an otherwise aligned insert.
    "nominalrecovery": ["--ft-threshold", "15.0"],
    "recovery": ["--ft-threshold", "15.0"],
}


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    settings = data.get("settings") if isinstance(data, dict) else None
    if not isinstance(settings, list):
        raise ValueError(f"Manifest missing settings list: {path}")
    return settings


def _scores(summary_path: Path) -> list[float | None]:
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scores: list[float | None] = []
    for record in summary.get("records", []):
        validation = record.get("validation") or {}
        scores.append(validation.get("score"))
    return scores


def _accepted_count(summary_path: Path) -> int:
    if not summary_path.exists():
        return 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("accepted", 0))


def _run_one(
    *,
    mode: str,
    engine_config: Path,
    output_dir: Path,
    score_threshold: float,
    seed: int,
    debug: bool,
    timeout_sec: int,
    per_trial_timeout_sec: int,
    python_prefix: list[str],
    attempts_per_setting: int,
    max_total_attempts: int,
    candidates_per_scene: int,
) -> dict[str, Any]:
    cmd = [
        *python_prefix,
        str(REPO_ROOT / "scripts" / "generate_expert_trajectories.py"),
        f"--{mode}",
        "--target-accepted-trajectories",
        str(attempts_per_setting),
        "--max-total-attempts",
        str(max_total_attempts),
        "--candidates-per-scene",
        str(candidates_per_scene),
        "--score-threshold",
        str(score_threshold),
        "--config",
        str(engine_config),
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--use-gpt5-analysis",
        "true",
        "--per-trial-timeout-sec",
        str(per_trial_timeout_sec),
        *MODE_ARGS[mode],
        "--debug" if debug else "--dry-run-config",
    ]
    if not debug:
        cmd.pop()
    env = os.environ.copy()
    env.update(MODE_ENV[mode])
    try:
        result = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec if timeout_sec > 0 else None,
            check=False,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else (exc.stdout or b"").decode(errors="replace")
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else (exc.stderr or b"").decode(errors="replace")
        stderr = f"{stderr}\nTIMEOUT: command exceeded {timeout_sec}s".strip()
        timed_out = True
    summary_path = output_dir / "generation_summary.json"
    return {
        "cmd": cmd,
        "mode_args": MODE_ARGS[mode],
        "mode_env": MODE_ENV[mode],
        "returncode": returncode,
        "timed_out": timed_out,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "summary": str(summary_path),
        "accepted": _accepted_count(summary_path),
        "scores": _scores(summary_path),
    }


def _run_config_snapshot(args: argparse.Namespace, python_prefix: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "aic_expert_matrix_run_config/v1",
        "manifest": str(args.manifest),
        "output_root": str(args.output_root),
        "modes": list(args.modes),
        "score_threshold": args.score_threshold,
        "max_repeats": args.max_repeats,
        "attempts_per_setting": args.attempts_per_setting,
        "max_total_attempts_per_repeat": args.max_total_attempts_per_repeat,
        "candidates_per_scene": args.candidates_per_scene,
        "limit_settings": args.limit_settings,
        "start_index": args.start_index,
        "seed": args.seed,
        "debug": bool(args.debug),
        "timeout_sec": args.timeout_sec,
        "per_trial_timeout_sec": args.per_trial_timeout_sec,
        "python_prefix": python_prefix,
        "mode_args": {mode: MODE_ARGS[mode] for mode in args.modes},
        "mode_env": {mode: MODE_ENV[mode] for mode in args.modes},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("outputs/expert_matrix_configs/matrix_manifest.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/expert_matrix_runs"))
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_ENV), default=sorted(MODE_ENV))
    parser.add_argument("--score-threshold", type=float, default=92.0)
    parser.add_argument("--max-repeats", type=int, default=10)
    parser.add_argument("--attempts-per-setting", type=int, default=1)
    parser.add_argument(
        "--max-total-attempts-per-repeat",
        type=int,
        default=0,
        help="Maximum planner/replay attempts inside each repeat. Defaults to --attempts-per-setting.",
    )
    parser.add_argument(
        "--candidates-per-scene",
        type=int,
        default=0,
        help="Number of VLM/MoveIt candidate indices to cycle inside each repeat. Defaults to max-total attempts.",
    )
    parser.add_argument("--limit-settings", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument(
        "--per-trial-timeout-sec",
        type=int,
        default=360,
        help="Timeout passed to each planner/replay trial. Use 0 to disable.",
    )
    parser.add_argument(
        "--python-prefix",
        default="pixi run python",
        help="Command prefix used to invoke Python inside the repo environment.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.attempts_per_setting <= 0:
        raise ValueError("--attempts-per-setting must be > 0")
    if args.max_total_attempts_per_repeat <= 0:
        args.max_total_attempts_per_repeat = args.attempts_per_setting
    if args.candidates_per_scene <= 0:
        args.candidates_per_scene = args.max_total_attempts_per_repeat
    if args.max_total_attempts_per_repeat < args.attempts_per_setting:
        raise ValueError("--max-total-attempts-per-repeat must be >= --attempts-per-setting")
    if args.candidates_per_scene <= 0:
        raise ValueError("--candidates-per-scene must be > 0")
    settings = _load_manifest(args.manifest)
    settings = [setting for setting in settings if int(setting["index"]) >= args.start_index]
    if args.limit_settings > 0:
        settings = settings[: args.limit_settings]
    args.output_root.mkdir(parents=True, exist_ok=True)
    matrix_log = args.output_root / "matrix_results.jsonl"
    failures = 0
    with matrix_log.open("a", encoding="utf-8") as log:
        python_prefix = args.python_prefix.split()
        run_config = _run_config_snapshot(args, python_prefix)
        run_config_path = args.output_root / "matrix_run_config.json"
        if not run_config_path.exists():
            run_config_path.write_text(
                json.dumps(run_config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        for setting in settings:
            engine_config = Path(setting["engine_config"])
            for mode in args.modes:
                reached = False
                for repeat in range(1, args.max_repeats + 1):
                    seed = args.seed + int(setting["index"]) * 100 + repeat
                    output_dir = (
                        args.output_root
                        / mode
                        / f"setting_{int(setting['index']):04d}_{setting['suffix']}"
                        / f"repeat_{repeat:02d}"
                    )
                    result = _run_one(
                        mode=mode,
                        engine_config=engine_config,
                        output_dir=output_dir,
                        score_threshold=args.score_threshold,
                        seed=seed,
                        debug=args.debug,
                        timeout_sec=args.timeout_sec,
                        per_trial_timeout_sec=args.per_trial_timeout_sec,
                        python_prefix=python_prefix,
                        attempts_per_setting=args.attempts_per_setting,
                        max_total_attempts=args.max_total_attempts_per_repeat,
                        candidates_per_scene=args.candidates_per_scene,
                    )
                    passed = int(result["accepted"]) >= args.attempts_per_setting
                    row = {
                        "setting": setting,
                        "mode": mode,
                        "repeat": repeat,
                        "seed": seed,
                        "run_config": run_config,
                        "score_threshold": args.score_threshold,
                        "passed": passed,
                        "result": result,
                    }
                    log.write(json.dumps(row, sort_keys=True) + "\n")
                    log.flush()
                    print(
                        f"{mode} setting {setting['index']} repeat {repeat}: "
                        f"accepted={result['accepted']} scores={result['scores']}"
                    )
                    if row["passed"]:
                        reached = True
                        break
                if not reached:
                    failures += 1
    print(f"Matrix log: {matrix_log}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
