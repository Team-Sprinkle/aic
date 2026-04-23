"""Transparent local trajectory score summaries for teacher-side analysis."""

from __future__ import annotations

from typing import Any

import numpy as np


def artifact_progress_metrics(artifact: dict[str, Any]) -> dict[str, float | None]:
    metadata = dict(artifact.get("metadata", {}))
    step_logs = list(artifact.get("step_logs", []))
    final_info = dict(artifact.get("final_info", {}))
    initial_summary = dict(metadata.get("initial_observation_summary", {}))
    initial_distance = _float_or_none(initial_summary.get("distance_to_target"))
    if initial_distance is None and step_logs:
        initial_distance = _distance_from_step(step_logs[0])
    final_distance = _float_or_none(final_info.get("distance_to_target"))
    path_length = _path_length(step_logs)
    progress = None
    if initial_distance is not None and final_distance is not None:
        progress = initial_distance - final_distance
    return {
        "initial_distance_to_target_m": initial_distance,
        "final_distance_to_target_m": final_distance,
        "progress_to_target_m": progress,
        "path_length_m": path_length,
    }


def replay_progress_metrics(
    *,
    original: dict[str, Any],
    replayed: dict[str, Any],
) -> dict[str, float | None]:
    original_initial = _float_or_none(
        dict(original.get("metadata", {})).get("initial_observation_summary", {}).get("distance_to_target")
    )
    replay_records = list(replayed.get("records", []))
    replay_initial = _float_or_none(replay_records[0].get("distance_to_target")) if replay_records else None
    replay_final = _float_or_none(replayed.get("final_info", {}).get("distance_to_target"))
    return {
        "original_initial_distance_to_target_m": original_initial,
        "replay_initial_distance_to_target_m": replay_initial,
        "replay_final_distance_to_target_m": replay_final,
        "replay_progress_to_target_m": None
        if replay_initial is None or replay_final is None
        else replay_initial - replay_final,
        "replay_path_length_m": _path_length_from_records(replay_records),
    }


def build_local_trajectory_score_summary(
    *,
    teacher_official_style_score: float,
    gym_final_score: float | None,
    rl_step_reward_total: float,
    progress_to_target_m: float | None,
    final_distance_to_target_m: float | None,
    path_length_m: float | None,
    quality_adjustment: float = 0.0,
    auxiliary_adjustment: float = 0.0,
    duplicate_penalty: float = 0.0,
) -> dict[str, Any]:
    clipped_reward = max(min(float(rl_step_reward_total), 100.0), -100.0)
    score = float(teacher_official_style_score) + 0.02 * clipped_reward
    if gym_final_score is not None:
        score += 0.15 * float(gym_final_score)
    if progress_to_target_m is not None:
        score += 10.0 * float(progress_to_target_m)
    if final_distance_to_target_m is not None:
        score -= 2.0 * float(final_distance_to_target_m)
    if path_length_m is not None:
        score += 0.5 * float(path_length_m)
    score += float(quality_adjustment) + float(auxiliary_adjustment) + float(duplicate_penalty)
    return {
        "score_label": "teacher_local_trajectory_score",
        "is_official": False,
        "notes": [
            "This is a transparent teacher-side local trajectory score summary.",
            "It combines gazebo-gym-side trajectory signals with teacher-local adjustments.",
            "It is not official_eval_score.",
        ],
        "scalar_score": score,
        "components": {
            "teacher_official_style_score": float(teacher_official_style_score),
            "gym_final_score": None if gym_final_score is None else float(gym_final_score),
            "rl_step_reward_total": float(rl_step_reward_total),
            "progress_to_target_m": None if progress_to_target_m is None else float(progress_to_target_m),
            "final_distance_to_target_m": None if final_distance_to_target_m is None else float(final_distance_to_target_m),
            "path_length_m": None if path_length_m is None else float(path_length_m),
            "quality_adjustment": float(quality_adjustment),
            "auxiliary_adjustment": float(auxiliary_adjustment),
            "duplicate_penalty": float(duplicate_penalty),
        },
    }


def _distance_from_step(step: dict[str, Any]) -> float | None:
    relative = step.get("observation_summary", {}).get("plug_to_port_relative")
    if not isinstance(relative, list) or len(relative) < 4:
        return None
    return _float_or_none(relative[3])


def _path_length(step_logs: list[dict[str, Any]]) -> float:
    positions: list[np.ndarray] = []
    for step in step_logs:
        pose = step.get("observation_summary", {}).get("tcp_pose")
        if isinstance(pose, list) and len(pose) >= 3:
            positions.append(np.asarray(pose[:3], dtype=np.float64))
    if len(positions) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:])))


def _path_length_from_records(records: list[dict[str, Any]]) -> float:
    positions: list[np.ndarray] = []
    for record in records:
        pose = record.get("tcp_pose")
        if isinstance(pose, list) and len(pose) >= 3:
            positions.append(np.asarray(pose[:3], dtype=np.float64))
    if len(positions) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:])))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
