"""Expert trajectory debug artifacts and failure-analysis payloads."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
import mimetypes
import os
import shutil
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from aic_teacher_official.debug_recorder import load_openai_api_key
from aic_teacher_official.replay import SmoothTrajectoryReplayPolicy
from aic_teacher_official.trajectory import SmoothTrajectory, TrajectoryWaypoint


DEBUG_SAMPLE_PERIOD_SEC = 0.25
DEBUG_DOWNSAMPLED_PERIOD_SEC = 1.0
MAX_PROMPT_BYTES = 300_000
MAX_GPT5_IMAGES = 8
MAX_GPT5_SERIES_ROWS = 80
MAX_GPT5_RUNTIME_EVENTS = 80
VIDEO_SUFFIXES = {".mp4", ".avi", ".webm", ".mkv"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
OBSERVATION_STATE_KEYS = [
    "tcp_pose.position.x",
    "tcp_pose.position.y",
    "tcp_pose.position.z",
    "tcp_pose.orientation.x",
    "tcp_pose.orientation.y",
    "tcp_pose.orientation.z",
    "tcp_pose.orientation.w",
    "tcp_velocity.linear.x",
    "tcp_velocity.linear.y",
    "tcp_velocity.linear.z",
    "tcp_velocity.angular.x",
    "tcp_velocity.angular.y",
    "tcp_velocity.angular.z",
    "tcp_error.x",
    "tcp_error.y",
    "tcp_error.z",
    "tcp_error.rx",
    "tcp_error.ry",
    "tcp_error.rz",
    "joint_positions.0",
    "joint_positions.1",
    "joint_positions.2",
    "joint_positions.3",
    "joint_positions.4",
    "joint_positions.5",
    "joint_positions.6",
    "wrist_wrench.force.x",
    "wrist_wrench.force.y",
    "wrist_wrench.force.z",
    "wrist_wrench.torque.x",
    "wrist_wrench.torque.y",
    "wrist_wrench.torque.z",
]


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        return max(minimum, default)


@dataclass(frozen=True)
class DebugArtifactPaths:
    debug_dir: Path
    observations: Path
    actions: Path
    ft_windows: Path
    tracking_error: Path
    image_manifest: Path
    trajectory_segments: Path
    moveit_plan_summary: Path
    replay_command_trace: Path
    transition_metrics: Path
    gpt5_payload: Path
    gpt5_prompt: Path
    gpt5_analysis: Path


def debug_paths(dataset_root: str | Path) -> DebugArtifactPaths:
    debug_dir = Path(dataset_root) / "debug"
    return DebugArtifactPaths(
        debug_dir=debug_dir,
        observations=debug_dir / "observations_sampled.jsonl",
        actions=debug_dir / "actions_sampled.jsonl",
        ft_windows=debug_dir / "ft_windows.jsonl",
        tracking_error=debug_dir / "tracking_error_sampled.jsonl",
        image_manifest=debug_dir / "image_manifest.jsonl",
        trajectory_segments=debug_dir / "trajectory_segments.json",
        moveit_plan_summary=debug_dir / "moveit_plan_summary.json",
        replay_command_trace=debug_dir / "replay_command_trace.jsonl",
        transition_metrics=debug_dir / "transition_metrics.json",
        gpt5_payload=debug_dir / "gpt5_failure_payload.json",
        gpt5_prompt=debug_dir / "gpt5_failure_prompt.md",
        gpt5_analysis=debug_dir / "gpt5_failure_analysis.md",
    )


def write_debug_artifacts(
    dataset_root: str | Path,
    *,
    trajectory: SmoothTrajectory,
    replay_metrics: dict[str, Any] | None = None,
    validation: dict[str, Any] | None = None,
    sample_period_sec: float = DEBUG_SAMPLE_PERIOD_SEC,
    camera_images: list[str | Path] | None = None,
    lerobot_dataset_root: str | Path | None = None,
) -> DebugArtifactPaths:
    paths = debug_paths(dataset_root)
    paths.debug_dir.mkdir(parents=True, exist_ok=True)
    for camera in ("center", "left", "right"):
        (paths.debug_dir / "sampled_images" / camera).mkdir(parents=True, exist_ok=True)

    dataset_rows = load_lerobot_debug_rows(lerobot_dataset_root) if lerobot_dataset_root else []
    samples = sample_lerobot_rows(dataset_rows, trajectory=trajectory, sample_period_sec=sample_period_sec)
    if not samples:
        samples = sample_trajectory(trajectory, sample_period_sec=sample_period_sec)
    _write_jsonl(paths.observations, [sample["observation"] for sample in samples])
    _write_jsonl(paths.actions, [sample["action"] for sample in samples])
    _write_jsonl(paths.tracking_error, [sample["tracking_error"] for sample in samples])
    _write_jsonl(paths.replay_command_trace, [sample["replay_command"] for sample in samples])
    ft_windows = aggregate_ft_windows(dataset_rows, window_sec=sample_period_sec) if dataset_rows else []
    if replay_metrics and isinstance(replay_metrics.get("ft_windows"), list):
        ft_windows = list(replay_metrics["ft_windows"])
    elif replay_metrics and isinstance(replay_metrics.get("ft_samples"), list):
        ft_windows = aggregate_ft_windows(replay_metrics["ft_samples"], window_sec=sample_period_sec)
    _write_jsonl(paths.ft_windows, ft_windows)
    _write_json(paths.trajectory_segments, trajectory_segments(trajectory))
    _write_json(paths.moveit_plan_summary, moveit_plan_summary(trajectory))
    _write_json(paths.transition_metrics, compute_transition_metrics(trajectory))
    _write_json(paths.debug_dir / "phase_speed_metrics.json", compute_phase_speed_metrics(samples))
    if replay_metrics and replay_metrics.get("runtime_trace_path"):
        _write_jsonl(paths.debug_dir / "runtime_trace.jsonl", _load_jsonl(Path(replay_metrics["runtime_trace_path"])))
    _write_json(paths.debug_dir / "validation_metrics.json", validation or {})
    _write_json(paths.debug_dir / "score_failure_info.json", replay_metrics or {})
    _write_image_manifest(paths, list(camera_images or []) + dataset_video_paths(lerobot_dataset_root))
    payload = build_gpt5_failure_payload(paths.debug_dir, sample_period_sec=sample_period_sec, allow_missing=True)
    prompt = build_gpt5_failure_prompt(payload)
    _write_json(paths.gpt5_payload, payload)
    paths.gpt5_prompt.write_text(prompt, encoding="utf-8")
    paths.gpt5_analysis.write_text(
        "# GPT-5 Failure Analysis\n\nNot run. Use `scripts/analyze_expert_trajectory_failure.py --debug-dir "
        f"{paths.debug_dir}` after collecting replay F/T, action, and observation artifacts.\n",
        encoding="utf-8",
    )
    _write_json(
        paths.debug_dir / "debug_manifest.json",
        {
            "schema_version": "aic_expert_debug_manifest/v1",
            "sample_period_sec": sample_period_sec,
            "artifacts": {key: str(value) for key, value in paths.__dict__.items() if isinstance(value, Path)},
            "replay_metrics": replay_metrics or {},
            "validation": validation or {},
            "lerobot_dataset_root": str(lerobot_dataset_root) if lerobot_dataset_root else None,
        },
    )
    return paths


def load_lerobot_debug_rows(dataset_root: str | Path | None) -> list[dict[str, Any]]:
    if dataset_root is None:
        return []
    root = Path(dataset_root)
    data_files = sorted((root / "data").rglob("*.parquet"))
    if not data_files:
        return []
    try:
        import pandas as pd
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for path in data_files:
        frame = pd.read_parquet(path)
        for row in frame.to_dict("records"):
            timestamp = float(row.get("timestamp", 0.0))
            observation_state = _observation_state_dict(row.get("observation.state"))
            rows.append(
                {
                    "timestamp": timestamp,
                    "action": _numeric_list(row.get("action")),
                    "observation_state": observation_state,
                    "force": [
                        observation_state.get("wrist_wrench.force.x", 0.0),
                        observation_state.get("wrist_wrench.force.y", 0.0),
                        observation_state.get("wrist_wrench.force.z", 0.0),
                    ],
                    "torque": [
                        observation_state.get("wrist_wrench.torque.x", 0.0),
                        observation_state.get("wrist_wrench.torque.y", 0.0),
                        observation_state.get("wrist_wrench.torque.z", 0.0),
                    ],
                }
            )
    return sorted(rows, key=lambda item: item["timestamp"])


def sample_lerobot_rows(
    rows: list[dict[str, Any]],
    *,
    trajectory: SmoothTrajectory,
    sample_period_sec: float,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    replay = SmoothTrajectoryReplayPolicy(trajectory)
    start = rows[0]["timestamp"]
    end = rows[-1]["timestamp"]
    query_times = np.arange(start, end + sample_period_sec * 0.5, sample_period_sec)
    samples = []
    for query in query_times:
        row = min(rows, key=lambda item: abs(item["timestamp"] - float(query)))
        elapsed = float(row["timestamp"] - start)
        target = replay.sample(elapsed)
        waypoint = target.waypoint
        obs = row["observation_state"]
        force = row.get("force", [0.0, 0.0, 0.0])
        torque = row.get("torque", [0.0, 0.0, 0.0])
        command_source = command_source_for_waypoint(waypoint)
        action_representation = action_representation_for_waypoint(waypoint)
        frame = frame_for_action(action_representation, waypoint)
        actual_pose = {
            "position": [
                obs.get("tcp_pose.position.x"),
                obs.get("tcp_pose.position.y"),
                obs.get("tcp_pose.position.z"),
            ],
            "orientation_xyzw": [
                obs.get("tcp_pose.orientation.x"),
                obs.get("tcp_pose.orientation.y"),
                obs.get("tcp_pose.orientation.z"),
                obs.get("tcp_pose.orientation.w"),
            ],
        }
        tracking_translation = [
            obs.get("tcp_error.x", 0.0),
            obs.get("tcp_error.y", 0.0),
            obs.get("tcp_error.z", 0.0),
        ]
        tracking_rotation = [
            obs.get("tcp_error.rx", 0.0),
            obs.get("tcp_error.ry", 0.0),
            obs.get("tcp_error.rz", 0.0),
        ]
        observation = {
            "schema_version": "aic_expert_debug_observation/v1",
            "timestamp": float(row["timestamp"]),
            "elapsed": elapsed,
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "actual_tcp_pose": actual_pose,
            "target_tcp_pose": target.tcp_pose.to_dict(),
            "actual_tcp_velocity": {
                "linear": [
                    obs.get("tcp_velocity.linear.x"),
                    obs.get("tcp_velocity.linear.y"),
                    obs.get("tcp_velocity.linear.z"),
                ],
                "angular": [
                    obs.get("tcp_velocity.angular.x"),
                    obs.get("tcp_velocity.angular.y"),
                    obs.get("tcp_velocity.angular.z"),
                ],
            },
            "joint_positions": [obs.get(f"joint_positions.{idx}") for idx in range(7)],
            "wrench_force_torque": {
                "force": {"x": force[0], "y": force[1], "z": force[2]},
                "torque": {"x": torque[0], "y": torque[1], "z": torque[2]},
                "force_norm": float(np.linalg.norm(force)),
                "torque_norm": float(np.linalg.norm(torque)),
            },
            "frame": "base_link",
        }
        action = {
            "schema_version": "aic_expert_debug_action/v1",
            "timestamp": float(row["timestamp"]),
            "elapsed": elapsed,
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "action_representation": action_representation,
            "frame": frame,
            "recorded_action": row.get("action"),
            "joint_positions_target": target.joint_positions,
            "joint_velocities_target": target.joint_velocities,
            "absolute_cartesian_pose_target": target.tcp_pose.to_dict(),
            "relative_delta": row.get("action") if action_representation == "relative delta" else None,
            "velocity_command": target.tcp_velocity,
        }
        tracking_error = {
            "schema_version": "aic_expert_tracking_error/v1",
            "timestamp": float(row["timestamp"]),
            "elapsed": elapsed,
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "position_error_m": float(np.linalg.norm(tracking_translation)),
            "orientation_error_rad": float(np.linalg.norm(tracking_rotation)),
            "translation_error": tracking_translation,
            "rotation_error": tracking_rotation,
        }
        samples.append(
            {
                "observation": observation,
                "action": action,
                "tracking_error": tracking_error,
                "replay_command": {
                    "schema_version": "aic_expert_replay_command/v1",
                    "timestamp": float(row["timestamp"]),
                    "elapsed": elapsed,
                    "phase": waypoint.phase.value,
                    "command_source": command_source,
                    "action_representation": action_representation,
                    "frame": frame,
                    "target_tcp_pose": target.tcp_pose.to_dict(),
                    "target_joint_positions": target.joint_positions,
                },
            }
        )
    return samples


def sample_trajectory(trajectory: SmoothTrajectory, *, sample_period_sec: float) -> list[dict[str, Any]]:
    replay = SmoothTrajectoryReplayPolicy(trajectory)
    duration = max(0.0, replay.end_time - replay.start_time)
    query_times = np.arange(0.0, duration + sample_period_sec * 0.5, sample_period_sec)
    samples = []
    for elapsed in query_times:
        target = replay.sample(float(elapsed))
        waypoint = target.waypoint
        command_source = command_source_for_waypoint(waypoint)
        action_representation = action_representation_for_waypoint(waypoint)
        frame = frame_for_action(action_representation, waypoint)
        observation = {
            "schema_version": "aic_expert_debug_observation/v1",
            "timestamp": float(elapsed),
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "actual_tcp_pose": None,
            "target_tcp_pose": target.tcp_pose.to_dict(),
            "joint_positions": target.joint_positions,
            "wrench_force_torque": None,
            "frame": "base_link",
            "unavailable": {
                "actual_tcp_pose": "offline_trajectory_sample_without_controller_feedback",
                "wrench_force_torque": "offline_trajectory_sample_without_ft_stream",
            },
        }
        action = {
            "schema_version": "aic_expert_debug_action/v1",
            "timestamp": float(elapsed),
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "action_representation": action_representation,
            "frame": frame,
            "joint_positions": target.joint_positions,
            "joint_velocities": target.joint_velocities,
            "absolute_cartesian_pose": target.tcp_pose.to_dict(),
            "relative_delta": None,
            "velocity_command": target.tcp_velocity,
        }
        tracking_error = {
            "schema_version": "aic_expert_tracking_error/v1",
            "timestamp": float(elapsed),
            "phase": waypoint.phase.value,
            "command_source": command_source,
            "position_error_m": None,
            "orientation_error_rad": None,
            "reason": "controller_feedback_unavailable",
        }
        samples.append(
            {
                "observation": observation,
                "action": action,
                "tracking_error": tracking_error,
                "replay_command": {
                    "schema_version": "aic_expert_replay_command/v1",
                    "timestamp": float(elapsed),
                    "phase": waypoint.phase.value,
                    "command_source": command_source,
                    "action_representation": action_representation,
                    "frame": frame,
                },
            }
        )
    return samples


def compute_phase_speed_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual TCP speed by executed phase/source when live data exists."""
    buckets: dict[str, list[tuple[float, float]]] = {}
    for sample in samples:
        obs = sample.get("observation", {})
        phase = str(obs.get("phase") or "unknown")
        command_source = str(obs.get("command_source") or "")
        key = command_source if command_source in {
            "moveit_approach",
            "local_preinsert_align",
            "pre_insert_settle",
            "guarded_insert",
            "recovery_backoff",
            "recovery_realign",
            "retry_insert",
        } else phase
        velocity = ((obs.get("actual_tcp_velocity") or {}).get("linear") or [])
        if len(velocity) != 3 or any(value is None for value in velocity):
            continue
        elapsed = float(obs.get("elapsed", obs.get("timestamp", 0.0)))
        speed = float(np.linalg.norm(np.asarray(velocity, dtype=np.float64)))
        buckets.setdefault(key, []).append((elapsed, speed))
    phases = {}
    for phase, values in sorted(buckets.items()):
        speeds = [speed for _, speed in values]
        times = [elapsed for elapsed, _ in values]
        phases[phase] = {
            "sample_count": len(values),
            "duration_sec": max(times) - min(times) if len(times) > 1 else 0.0,
            "max_speed_mps": max(speeds),
            "p95_speed_mps": float(np.percentile(speeds, 95)),
            "median_speed_mps": statistics.median(speeds),
        }
    guarded = phases.get("guarded_insert") or phases.get("final_insertion") or {}
    return {
        "schema_version": "aic_expert_phase_speed_metrics/v1",
        "phases": phases,
        "max_guarded_insert_speed_mps": guarded.get("max_speed_mps"),
    }


def command_source_for_waypoint(waypoint: TrajectoryWaypoint) -> str:
    explicit = waypoint.diagnostics.get("command_source")
    if explicit:
        return str(explicit)
    phase = waypoint.phase.value
    source = waypoint.source.value
    if phase == "final_insertion" or source == "cheatcode":
        return "cheatcode_insert"
    if phase == "pre_insertion":
        return "moveit_approach"
    return "moveit_approach"


def action_representation_for_waypoint(waypoint: TrajectoryWaypoint) -> str:
    if waypoint.joint_positions is not None and waypoint.phase.value != "final_insertion":
        return "joint position"
    if waypoint.phase.value == "final_insertion":
        if waypoint.diagnostics.get("insertion_command_mode") == "exact_position":
            return "absolute Cartesian pose"
        return "relative delta"
    return "absolute Cartesian pose"


def frame_for_action(action_representation: str, waypoint: TrajectoryWaypoint) -> str:
    if action_representation == "relative delta":
        return "gripper/tcp"
    return "base_link"


def trajectory_segments(trajectory: SmoothTrajectory) -> dict[str, Any]:
    segments = []
    start = trajectory.waypoints[0]
    last = start
    for waypoint in trajectory.waypoints[1:]:
        if waypoint.phase != last.phase or command_source_for_waypoint(waypoint) != command_source_for_waypoint(last):
            segments.append(_segment_record(start, last))
            start = waypoint
        last = waypoint
    segments.append(_segment_record(start, last))
    return {"schema_version": "aic_expert_trajectory_segments/v1", "segments": segments}


def _segment_record(start: TrajectoryWaypoint, end: TrajectoryWaypoint) -> dict[str, Any]:
    return {
        "phase": start.phase.value,
        "command_source": command_source_for_waypoint(start),
        "start_timestamp": start.timestamp,
        "end_timestamp": end.timestamp,
        "duration_sec": max(0.0, end.timestamp - start.timestamp),
        "start_tcp_pose": start.tcp_pose.to_dict(),
        "end_tcp_pose": end.tcp_pose.to_dict(),
        "action_representation": action_representation_for_waypoint(start),
        "frame": frame_for_action(action_representation_for_waypoint(start), start),
    }


def moveit_plan_summary(trajectory: SmoothTrajectory) -> dict[str, Any]:
    planning = trajectory.metadata.planning
    moveit = planning.get("moveit") or {}
    return {
        "schema_version": "aic_expert_moveit_plan_summary/v1",
        "moveit": moveit,
        "replay_space": moveit.get("replay_space", "joint_position"),
        "segments": moveit.get("segments", []),
        "global_smoothing": trajectory.metadata.postprocessing,
        "warning": (
            "MoveIt trajectory is replayed in joint space; TCP smoothing metadata does not globally retime "
            "controller joint commands unless joint retiming is also present."
        ),
    }


def compute_transition_metrics(trajectory: SmoothTrajectory) -> dict[str, Any]:
    waypoints = trajectory.waypoints
    boundaries = []
    for index in range(1, len(waypoints)):
        before = waypoints[index - 1]
        after = waypoints[index]
        if before.phase == after.phase and command_source_for_waypoint(before) == command_source_for_waypoint(after):
            continue
        boundaries.append(_boundary_metrics(waypoints, index))
    return {
        "schema_version": "aic_expert_transition_metrics/v1",
        "boundary_count": len(boundaries),
        "boundaries": boundaries,
    }


def _boundary_metrics(waypoints: list[TrajectoryWaypoint], index: int) -> dict[str, Any]:
    before = waypoints[index - 1]
    after = waypoints[index]
    pos_jump = _norm(np.asarray(after.tcp_pose.position) - np.asarray(before.tcp_pose.position))
    ori_jump = quaternion_angle_rad(before.tcp_pose.orientation_xyzw, after.tcp_pose.orientation_xyzw)
    joint_jump = None
    if before.joint_positions is not None and after.joint_positions is not None and len(before.joint_positions) == len(after.joint_positions):
        joint_jump = _norm(np.asarray(after.joint_positions) - np.asarray(before.joint_positions))
    target_before = _velocity_at(waypoints, index - 1)
    target_after = _velocity_at(waypoints, index)
    accel_jump = None if target_before is None or target_after is None else _norm(np.asarray(target_after) - np.asarray(target_before))
    jerk_jump = _jerk_jump(waypoints, index)
    suspicious = bool(
        pos_jump > 0.01
        or ori_jump > 0.12
        or (joint_jump is not None and joint_jump > 0.08)
        or (accel_jump is not None and accel_jump > 0.1)
    )
    return {
        "phase_before": before.phase.value,
        "phase_after": after.phase.value,
        "command_source_before": command_source_for_waypoint(before),
        "command_source_after": command_source_for_waypoint(after),
        "timestamp": after.timestamp,
        "tcp_position_jump_m": pos_jump,
        "tcp_orientation_jump_rad": ori_jump,
        "joint_position_jump_norm": joint_jump,
        "target_velocity_before": target_before,
        "target_velocity_after": target_after,
        "actual_velocity_before": None,
        "actual_velocity_after": None,
        "estimated_acceleration_jump": accel_jump,
        "estimated_jerk_jump": jerk_jump,
        "ft_before_after": None,
        "tracking_error_before_after": None,
        "suspicious": suspicious,
        "suspicion_reasons": _suspicion_reasons(pos_jump, ori_jump, joint_jump, accel_jump),
    }


def quaternion_angle_rad(left: list[float], right: list[float]) -> float:
    q0 = np.asarray(left, dtype=np.float64)
    q1 = np.asarray(right, dtype=np.float64)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = abs(float(np.dot(q0, q1)))
    return float(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def aggregate_ft_windows(samples: list[dict[str, Any]], *, window_sec: float = DEBUG_SAMPLE_PERIOD_SEC) -> list[dict[str, Any]]:
    if window_sec <= 0.0:
        raise ValueError("window_sec must be positive")
    buckets: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        timestamp = float(sample.get("timestamp", sample.get("time", 0.0)))
        buckets.setdefault(int(timestamp // window_sec), []).append(sample)
    return [_ft_window(bucket * window_sec, (bucket + 1) * window_sec, values) for bucket, values in sorted(buckets.items())]


def _ft_window(start: float, end: float, samples: list[dict[str, Any]]) -> dict[str, Any]:
    axes = {axis: [] for axis in ("fx", "fy", "fz", "tx", "ty", "tz", "force_norm", "torque_norm")}
    for sample in samples:
        values = _ft_values(sample)
        for axis, value in values.items():
            axes[axis].append(value)
    stats = {axis: _stats(values) for axis, values in axes.items()}
    return {"window_start": start, "window_end": end, "sample_count": len(samples), **stats}


def _ft_values(sample: dict[str, Any]) -> dict[str, float]:
    force = sample.get("force") or sample.get("f") or {}
    torque = sample.get("torque") or sample.get("t") or {}
    fx = float(sample["fx"] if "fx" in sample else _axis_value(force, "x", 0))
    fy = float(sample["fy"] if "fy" in sample else _axis_value(force, "y", 1))
    fz = float(sample["fz"] if "fz" in sample else _axis_value(force, "z", 2))
    tx = float(sample["tx"] if "tx" in sample else _axis_value(torque, "x", 0))
    ty = float(sample["ty"] if "ty" in sample else _axis_value(torque, "y", 1))
    tz = float(sample["tz"] if "tz" in sample else _axis_value(torque, "z", 2))
    force_norm = float(sample.get("force_norm", math.sqrt(fx * fx + fy * fy + fz * fz)))
    torque_norm = float(sample.get("torque_norm", math.sqrt(tx * tx + ty * ty + tz * tz)))
    return {"fx": fx, "fy": fy, "fz": fz, "tx": tx, "ty": ty, "tz": tz, "force_norm": force_norm, "torque_norm": torque_norm}


def _axis_value(values: Any, key: str, index: int) -> float:
    if isinstance(values, dict):
        return float(values.get(key, 0.0))
    if isinstance(values, (list, tuple)) and len(values) > index:
        return float(values[index])
    return 0.0


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None, "median": None}
    return {"min": min(values), "max": max(values), "median": statistics.median(values)}


def build_gpt5_failure_payload(
    debug_dir: str | Path,
    *,
    sample_period_sec: float = DEBUG_SAMPLE_PERIOD_SEC,
    allow_missing: bool = False,
) -> dict[str, Any]:
    root = Path(debug_dir)
    required = {
        "observations": root / "observations_sampled.jsonl",
        "actions": root / "actions_sampled.jsonl",
        "ft_windows": root / "ft_windows.jsonl",
        "transition_metrics": root / "transition_metrics.json",
    }
    missing = [name for name, path in required.items() if not path.exists() or path.stat().st_size == 0]
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing required expert debug artifacts: {', '.join(missing)}")
    observations = _load_jsonl(required["observations"])
    actions = _load_jsonl(required["actions"])
    ft_windows = _load_jsonl(required["ft_windows"])
    if not allow_missing:
        if not observations:
            raise ValueError("no debug observations are available")
        if not actions:
            raise ValueError("no action trace is available")
        if not ft_windows:
            raise ValueError("no F/T data is available")
        if not (root / "transition_metrics.json").exists():
            raise ValueError("transition metrics cannot be computed")
    image_manifest = ensure_sampled_image_frames(root, sample_period_sec=sample_period_sec)
    if not allow_missing and image_manifest and not any(row.get("exists") for row in image_manifest):
        raise ValueError("images were expected but none exist")
    max_series_rows = _env_int("AIC_GPT5_FAILURE_ANALYSIS_MAX_SERIES_ROWS", MAX_GPT5_SERIES_ROWS, minimum=1)
    max_runtime_events = _env_int(
        "AIC_GPT5_FAILURE_ANALYSIS_MAX_RUNTIME_EVENTS",
        MAX_GPT5_RUNTIME_EVENTS,
        minimum=1,
    )
    max_images = _env_int("AIC_GPT5_FAILURE_ANALYSIS_MAX_IMAGES", MAX_GPT5_IMAGES, minimum=0)
    payload = {
        "schema_version": "aic_expert_gpt5_failure_payload/v2",
        "debug_dir": str(root),
        "coordinate_frame_contract": {
            "tcp_pose_frame": "base_link",
            "absolute_cartesian_actions_frame": "base_link",
            "relative_delta_actions_frame": "gripper/tcp",
            "force_frame_note": (
                "Runtime recovery_context rows include force in raw TCP-assumed form and a "
                "base_link estimate from current TCP orientation. LeRobot state force rows are raw wrist wrench."
            ),
            "sampling_period_sec": sample_period_sec,
        },
        "sample_period_sec": sample_period_sec,
        "observations_sampled": _compact_observation_rows(
            _limit_rows(_filter_by_period(observations, sample_period_sec), max_series_rows)
        ),
        "actions_sampled": _compact_action_rows(
            _limit_rows(_filter_by_period(actions, sample_period_sec), max_series_rows)
        ),
        "ft_windows": _limit_rows(_filter_by_period(ft_windows, sample_period_sec), max_series_rows),
        "tracking_error_sampled": _compact_tracking_rows(
            _limit_rows(
                _filter_by_period(_load_jsonl(root / "tracking_error_sampled.jsonl"), sample_period_sec),
                max_series_rows,
            )
        ),
        "image_manifest": _compact_image_manifest(image_manifest),
        "image_inputs": [
            {
                "timestamp": row.get("timestamp"),
                "camera": row.get("camera"),
                "debug_path": row.get("debug_path"),
                "media_type": row.get("media_type"),
            }
            for row in image_manifest
            if row.get("exists") and row.get("media_type") == "image"
        ][:max_images],
        "trajectory_segments": _load_json(root / "trajectory_segments.json"),
        "moveit_plan_summary": _compact_moveit_summary(_load_json(root / "moveit_plan_summary.json")),
        "replay_command_trace": _compact_replay_rows(
            _limit_rows(
                _filter_by_period(_load_jsonl(root / "replay_command_trace.jsonl"), sample_period_sec),
                max_series_rows,
            )
        ),
        "transition_metrics": _load_json(root / "transition_metrics.json"),
        "phase_speed_metrics": _load_json(root / "phase_speed_metrics.json"),
        "runtime_trace": _limit_rows(
            _filter_by_period(_load_jsonl(root / "runtime_trace.jsonl"), sample_period_sec),
            max_runtime_events,
        ),
        "validation_metrics": _compact_metrics_for_gpt(_load_json(root / "validation_metrics.json")),
        "score_failure_info": _load_json(root / "score_failure_info.json"),
        "code_context": code_context_snippets(),
    }
    return payload


def _limit_rows(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    if max_rows <= 2:
        return rows[:max_rows]
    # Preserve the beginning, end, and evenly spaced interior samples so long
    # episodes remain diagnosable without exceeding GPT payload limits.
    selected = {0, len(rows) - 1}
    interior = max_rows - len(selected)
    if interior > 0:
        step = (len(rows) - 1) / float(interior + 1)
        for idx in range(1, interior + 1):
            selected.add(int(round(idx * step)))
    return [rows[idx] for idx in sorted(selected)]


def _compact_metrics_for_gpt(metrics: dict[str, Any]) -> dict[str, Any]:
    compact = dict(metrics or {})
    phase_labels = compact.pop("phase_labels", None)
    if isinstance(phase_labels, list):
        counts: dict[str, int] = {}
        for label in phase_labels:
            counts[str(label)] = counts.get(str(label), 0) + 1
        compact["phase_label_count"] = len(phase_labels)
        compact["phase_label_counts"] = counts
        compact["phase_label_first"] = phase_labels[:10]
        compact["phase_label_last"] = phase_labels[-10:]
    return compact


def _compact_observation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        wrench = row.get("wrench_force_torque") or {}
        compact.append(
            {
                "timestamp": row.get("timestamp"),
                "elapsed": row.get("elapsed"),
                "phase": row.get("phase"),
                "command_source": row.get("command_source"),
                "frame": row.get("frame"),
                "actual_tcp_pose": row.get("actual_tcp_pose"),
                "target_tcp_pose": row.get("target_tcp_pose"),
                "actual_tcp_velocity": row.get("actual_tcp_velocity"),
                "wrist_force_raw": (wrench.get("force") if isinstance(wrench, dict) else None),
                "force_norm": wrench.get("force_norm"),
                "torque_norm": wrench.get("torque_norm"),
            }
        )
    return compact


def _compact_action_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for row in rows:
        compact.append(
            {
                "timestamp": row.get("timestamp"),
                "elapsed": row.get("elapsed"),
                "phase": row.get("phase"),
                "command_source": row.get("command_source"),
                "action_representation": row.get("action_representation"),
                "frame": row.get("frame"),
                "joint_positions_target": row.get("joint_positions_target"),
                "absolute_cartesian_pose_target": row.get("absolute_cartesian_pose_target"),
                "relative_delta": row.get("relative_delta"),
            }
        )
    return compact


def _compact_tracking_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": row.get("timestamp"),
            "elapsed": row.get("elapsed"),
            "phase": row.get("phase"),
            "command_source": row.get("command_source"),
            "position_error_m": row.get("position_error_m"),
            "orientation_error_rad": row.get("orientation_error_rad"),
        }
        for row in rows
    ]


def _compact_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": row.get("timestamp"),
            "elapsed": row.get("elapsed"),
            "phase": row.get("phase"),
            "command_source": row.get("command_source"),
            "action_representation": row.get("action_representation"),
            "frame": row.get("frame"),
        }
        for row in rows
    ]


def _compact_image_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "timestamp": row.get("timestamp"),
            "camera": row.get("camera"),
            "debug_path": row.get("debug_path"),
            "exists": row.get("exists"),
            "media_type": row.get("media_type"),
            "sampled_from_video": row.get("sampled_from_video", False),
        }
        for row in rows[:MAX_GPT5_IMAGES]
    ]


def _compact_moveit_summary(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(summary, dict):
        return summary
    moveit = summary.get("moveit") or {}
    segments = moveit.get("segments") or summary.get("segments") or []
    return {
        "schema_version": summary.get("schema_version"),
        "replay_space": summary.get("replay_space") or moveit.get("replay_space"),
        "segment_count": len(segments) if isinstance(segments, list) else None,
        "global_joint_retiming": summary.get("global_joint_retiming") or moveit.get("global_joint_retiming"),
        "warning": summary.get("warning"),
        "segments_brief": [
            {
                "index": idx,
                "phase": seg.get("phase") if isinstance(seg, dict) else None,
                "success": seg.get("success") if isinstance(seg, dict) else None,
                "waypoint_count": seg.get("waypoint_count") if isinstance(seg, dict) else None,
                "duration_sec": seg.get("duration_sec") if isinstance(seg, dict) else None,
            }
            for idx, seg in enumerate(segments[:12] if isinstance(segments, list) else [])
        ],
    }


def compact_payload_with_retry(
    debug_dir: str | Path,
    *,
    allow_missing: bool = False,
) -> tuple[dict[str, Any], float]:
    max_prompt_bytes = _env_int("AIC_GPT5_FAILURE_ANALYSIS_MAX_PROMPT_BYTES", MAX_PROMPT_BYTES, minimum=10_000)
    first = build_gpt5_failure_payload(
        debug_dir,
        sample_period_sec=DEBUG_SAMPLE_PERIOD_SEC,
        allow_missing=allow_missing,
    )
    if _payload_size(first) <= max_prompt_bytes:
        return first, DEBUG_SAMPLE_PERIOD_SEC
    second = build_gpt5_failure_payload(
        debug_dir,
        sample_period_sec=DEBUG_DOWNSAMPLED_PERIOD_SEC,
        allow_missing=allow_missing,
    )
    if _payload_size(second) <= max_prompt_bytes:
        return second, DEBUG_DOWNSAMPLED_PERIOD_SEC
    raise ValueError("GPT-5 payload remains too large after 1.0 second sampling")


def build_gpt5_failure_prompt(payload: dict[str, Any]) -> str:
    questions = [
        "Compare the initial candidate and repaired candidate if both are present.",
        "Did pre-contact repair smooth executable actions, or only metadata?",
        "Was the exact pre-insertion pose/orientation preserved before guarded insertion?",
        "Did local_preinsert_align improve near-port behavior without making it too slow?",
        "Did the tracking gate prevent rough insertion starts?",
        "Was guarded_insert actual TCP speed bounded by validation?",
        "For nominalrecovery/recovery, did F/T-gated recovery stop descent, physically back off by the measured minimum distance, wait for force release, realign, and retry before any lateral correction?",
        "Why is motion disconnected between segments?",
        "Are MoveIt segment plans globally smoothed/retimed properly?",
        "Is final insertion too fast?",
        "Is collision/contact/F/T spike risk caused by speed, frame mismatch, handoff discontinuity, or lack of settle/hold?",
        "Were insertions smooth or rough/colliding?",
        "Which trajectory-generation stage is problematic: VLM strategy, candidate generation, MoveIt planning, segment concatenation, global smoother, replay conversion, CheatCode handoff, or insertion speed?",
        "Does GPT-5-mini need TF lookup, MoveIt feasibility checking, candidate scoring, F/T summaries, visual validation, or collision-check summaries?",
        "Use the coordinate_frame_contract and runtime recovery_context rows to check body/TCP frame versus base_link frame usage. Which phases should use body frame versus base frame?",
        "Which phases should use absolute pose versus delta pose?",
        "Is replanning/global smoothing implemented correctly for smoothness, obstacle preservation, VLM cable-risk hints, jerk avoidance, and abrupt insertion avoidance?",
    ]
    return (
        "# AIC Expert Trajectory Failure Analysis\n\n"
        "Analyze the expert trajectory debug payload and the attached sampled camera images. Be concrete and tie conclusions to metrics, images, frames, action representations, and code context.\n\n"
        + "\n".join(f"{idx + 1}. {question}" for idx, question in enumerate(questions))
        + "\n\n## Payload\n```json\n"
        + json.dumps(payload, indent=2)
        + "\n```\n"
    )


def call_gpt5_failure_analysis(prompt: str, *, model: str = "gpt-5") -> str:
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for GPT-5 analysis")
    from openai import OpenAI

    timeout_sec = float(os.environ.get("AIC_GPT5_FAILURE_ANALYSIS_TIMEOUT_SEC", "180"))
    content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
    for image in _image_content_from_prompt(prompt):
        content.append(image)
    request: dict[str, Any] = {
        "model": model,
        "instructions": "You are a senior robotics trajectory and contact-debugging engineer.",
        "input": [{"role": "user", "content": content}],
    }
    reasoning_effort = os.environ.get("AIC_GPT5_FAILURE_ANALYSIS_REASONING_EFFORT", "minimal").strip()
    if reasoning_effort:
        request["reasoning"] = {"effort": reasoning_effort}
    max_output_tokens = _env_int("AIC_GPT5_FAILURE_ANALYSIS_MAX_OUTPUT_TOKENS", 4000, minimum=0)
    if max_output_tokens > 0:
        request["max_output_tokens"] = max_output_tokens
    response = OpenAI(api_key=api_key, timeout=timeout_sec).responses.create(
        **request,
    )
    text = getattr(response, "output_text", None)
    if not text:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", "") == "output_text":
                    chunks.append(str(getattr(content, "text", "")))
        text = "\n".join(chunks)
    if not str(text).strip():
        raise RuntimeError("GPT-5 returned empty/unparseable output")
    return str(text)


def _image_content_from_prompt(prompt: str) -> list[dict[str, Any]]:
    payload = _payload_from_prompt(prompt)
    images = (payload or {}).get("image_inputs") or []
    max_images = _env_int("AIC_GPT5_FAILURE_ANALYSIS_MAX_IMAGES", MAX_GPT5_IMAGES, minimum=0)
    content: list[dict[str, Any]] = []
    for image in images[:max_images]:
        path = Path(str(image.get("debug_path", "")))
        if not path.exists() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
                "detail": "low",
            }
        )
    return content


def _payload_from_prompt(prompt: str) -> dict[str, Any] | None:
    marker = "## Payload\n```json\n"
    start = prompt.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = prompt.find("\n```", start)
    if end < 0:
        return None
    try:
        return json.loads(prompt[start:end])
    except json.JSONDecodeError:
        return None


def code_context_snippets() -> dict[str, str]:
    return {
        "vlm_strategy_schema": "VLMStrategy accepts cable_risk/reason/mitigation/preferred approach only; executable low-level waypoints are forbidden.",
        "moveit_usage": "MoveItPyPlanningBackend plans rigid free-space approach segments and extracts joint trajectory for replay.",
        "trajectory_concat_smoothing": "postprocess_piecewise_trajectory smooths TCP metadata with C1 Hermite except final insertion; nominal repair may retime and resample executable joint targets through the pinned pre-insertion point.",
        "final_insertion": "Final insertion remains CheatCode-style geometric guarded_insert, not MoveIt. Default final insertion streams exact_position absolute base_link TCP targets from CheatCode geometry on a minimum-jerk profile. A pinned XY/orientation insertion experiment is available behind AIC_OFFICIAL_TEACHER_PIN_INSERTION_TARGET=true, but live testing showed it can miss insertion.",
        "replay_semantics": "joint_position_then_cheatcode sends MoveIt joint targets until online insertion. MoveIt/free-space replay is joint_position. Local_preinsert_align, guarded_insert exact_position, and recovery backoff send absolute base_link TCP pose targets. Debug action rows must state action_representation and frame explicitly.",
    }


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _load_json(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    skipped = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            skipped += 1
            rows.append(
                {
                    "event": "debug_artifact_jsonl_decode_skipped",
                    "source_path": str(path),
                    "line_number": line_number,
                    "error": str(exc),
                    "line_prefix": line[:240],
                }
            )
    if skipped:
        rows.append(
            {
                "event": "debug_artifact_jsonl_decode_summary",
                "source_path": str(path),
                "skipped_lines": skipped,
            }
        )
    return rows


def _write_image_manifest(paths: DebugArtifactPaths, camera_images: list[str | Path]) -> None:
    rows = []
    for index, image in enumerate(camera_images):
        src = Path(image)
        camera = _camera_name(src)
        suffix = src.suffix.lower()
        dst = paths.debug_dir / "sampled_images" / camera / f"{index:06d}{suffix or '.png'}"
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        rows.append(
            {
                "timestamp": index * DEBUG_SAMPLE_PERIOD_SEC,
                "camera": camera,
                "source_path": str(src),
                "debug_path": str(dst),
                "exists": src.exists(),
                "media_type": "video" if suffix in VIDEO_SUFFIXES else "image",
            }
        )
    _write_jsonl(paths.image_manifest, rows)


def _camera_name(path: Path) -> str:
    lower = path.name.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return "center"


def dataset_video_paths(dataset_root: str | Path | None) -> list[Path]:
    if dataset_root is None:
        return []
    root = Path(dataset_root)
    if not root.exists():
        return []
    return sorted(
        path
        for path in (root / "videos").rglob("*")
        if path.suffix.lower() in VIDEO_SUFFIXES
    )


def ensure_sampled_image_frames(debug_dir: str | Path, *, sample_period_sec: float) -> list[dict[str, Any]]:
    """Return image manifest rows, extracting video frames when replay videos are present."""
    root = Path(debug_dir)
    manifest_path = root / "image_manifest.jsonl"
    rows = _load_jsonl(manifest_path)
    if not rows:
        return []
    output_rows: list[dict[str, Any]] = []
    changed = False
    for row in rows:
        media_type = row.get("media_type")
        debug_path = Path(str(row.get("debug_path", "")))
        if media_type != "video":
            output_rows.append(row)
            continue
        source = debug_path if debug_path.exists() else Path(str(row.get("source_path", "")))
        extracted = _extract_video_frames_for_gpt(
            source,
            root=root,
            camera=str(row.get("camera") or _camera_name(source)),
            sample_period_sec=sample_period_sec,
        )
        if extracted:
            output_rows.extend(extracted)
            changed = True
        else:
            output_rows.append(row)
    if changed:
        output_rows.sort(key=lambda item: (float(item.get("timestamp", 0.0)), str(item.get("camera", ""))))
        _write_jsonl(manifest_path, output_rows)
    return output_rows


def _extract_video_frames_for_gpt(
    video_path: Path,
    *,
    root: Path,
    camera: str,
    sample_period_sec: float,
) -> list[dict[str, Any]]:
    if not video_path.exists():
        return []
    try:
        import cv2
    except Exception:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 1e-6 or frame_count <= 0:
        cap.release()
        return []
    duration = frame_count / fps
    frame_dir = root / "sampled_images" / camera
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    frame_index = 0
    timestamp = 0.0
    while timestamp <= duration + 1e-9 and len(rows) < MAX_GPT5_IMAGES:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        dst = frame_dir / f"video_{video_path.stem}_{frame_index:06d}.jpg"
        cv2.imwrite(str(dst), frame)
        rows.append(
            {
                "timestamp": round(timestamp, 3),
                "camera": camera,
                "source_path": str(video_path),
                "debug_path": str(dst),
                "exists": dst.exists(),
                "media_type": "image",
                "sampled_from_video": True,
            }
        )
        frame_index += 1
        timestamp += sample_period_sec
    cap.release()
    return rows


def _observation_state_dict(values: Any) -> dict[str, float]:
    numeric = _numeric_list(values)
    return {
        key: float(numeric[index]) if index < len(numeric) else 0.0
        for index, key in enumerate(OBSERVATION_STATE_KEYS)
    }


def _numeric_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        return [float(v) for v in values.tolist()]
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    try:
        return [float(v) for v in list(values)]
    except Exception:
        return []


def _filter_by_period(rows: list[dict[str, Any]], period: float) -> list[dict[str, Any]]:
    if period <= DEBUG_SAMPLE_PERIOD_SEC + 1e-9:
        return rows
    filtered = []
    last_bucket = None
    for row in rows:
        timestamp = float(row.get("timestamp", row.get("window_start", row.get("time_sec", 0.0))))
        bucket = round(timestamp / period)
        if bucket != last_bucket and abs(timestamp - bucket * period) <= DEBUG_SAMPLE_PERIOD_SEC * 0.75:
            filtered.append(row)
            last_bucket = bucket
    return filtered


def _payload_size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload).encode("utf-8"))


def _velocity_at(waypoints: list[TrajectoryWaypoint], index: int) -> list[float] | None:
    waypoint = waypoints[index]
    if waypoint.tcp_velocity is not None:
        return waypoint.tcp_velocity
    if 0 < index < len(waypoints) - 1:
        prev_w = waypoints[index - 1]
        next_w = waypoints[index + 1]
        dt = next_w.timestamp - prev_w.timestamp
        if dt > 0.0:
            return ((np.asarray(next_w.tcp_pose.position) - np.asarray(prev_w.tcp_pose.position)) / dt).tolist()
    return None


def _jerk_jump(waypoints: list[TrajectoryWaypoint], index: int) -> float | None:
    if index < 2 or index + 1 >= len(waypoints):
        return None
    v0 = _velocity_at(waypoints, index - 2)
    v1 = _velocity_at(waypoints, index - 1)
    v2 = _velocity_at(waypoints, index)
    v3 = _velocity_at(waypoints, index + 1)
    if None in (v0, v1, v2, v3):
        return None
    a_before = np.asarray(v1) - np.asarray(v0)
    a_after = np.asarray(v3) - np.asarray(v2)
    return _norm(a_after - a_before)


def _suspicion_reasons(pos_jump: float, ori_jump: float, joint_jump: float | None, accel_jump: float | None) -> list[str]:
    reasons = []
    if pos_jump > 0.01:
        reasons.append("tcp_position_jump_gt_1cm")
    if ori_jump > 0.12:
        reasons.append("tcp_orientation_jump_gt_0.12rad")
    if joint_jump is not None and joint_jump > 0.08:
        reasons.append("joint_jump_gt_0.08rad_norm")
    if accel_jump is not None and accel_jump > 0.1:
        reasons.append("velocity_discontinuity_gt_0.1mps")
    return reasons


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(values))
