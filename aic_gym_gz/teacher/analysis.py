"""Evaluation and analysis helpers for teacher rollouts, search, and replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .replay import TeacherReplayArtifact
from .scoring import OfficialStyleScoreEvaluator


@dataclass(frozen=True)
class TeacherEvaluationResult:
    summary: dict[str, Any]
    markdown: str


def load_json_file(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_rollout_artifact(
    artifact_payload: dict[str, Any],
    *,
    include_markdown: bool = True,
) -> TeacherEvaluationResult:
    metadata = artifact_payload.get("metadata", {})
    step_logs = list(artifact_payload.get("step_logs", []))
    planner_candidates = list(artifact_payload.get("planner_candidates", []))
    trajectory_segments = list(artifact_payload.get("trajectory_segments", []))
    final_info = dict(artifact_payload.get("final_info", {}))
    missing_fields = _missing_fields(
        artifact_payload,
        ["metadata", "step_logs", "trajectory_segments", "final_info"],
    )
    teacher_score = (
        metadata.get("teacher_official_style_score")
        or metadata.get("official_style_score")
        or OfficialStyleScoreEvaluator().evaluate_rollout(artifact_payload).to_dict()
    )
    phases = [segment.get("phase") for segment in trajectory_segments if segment.get("phase")]
    unique_phase_sequence = _collapse_adjacent_duplicates(phases)
    action_stats = _action_statistics(step_logs)
    tcp_positions = _tcp_positions(step_logs)
    path_length = _path_length(tcp_positions)
    duration = _duration(step_logs)
    candidate_count = len(planner_candidates)
    planner_calls = candidate_count
    data_quality = metadata.get("data_quality", {})
    signal_quality_summary = _signal_quality_summary(data_quality)
    tier3 = teacher_score.get("tier3", {})
    outcome = {
        "success": bool(final_info.get("success", False)),
        "wrong_port": bool(final_info.get("wrong_port", False)),
        "tier3_status": tier3.get("status"),
        "tier3_message": tier3.get("message"),
        "partial_insertion": "partial" in str(tier3.get("status", "")),
        "failure_mode": _failure_mode(final_info=final_info, teacher_score=teacher_score),
    }
    warnings = detect_rollout_warnings(
        {
            "phase_sequence": unique_phase_sequence,
            "segment_count": len(trajectory_segments),
            "action_stats": action_stats,
            "data_quality": data_quality,
            "outcome": outcome,
            "teacher_score": teacher_score,
            "path_length_m": path_length,
            "duration_s": duration,
        }
    )
    summary = {
        "artifact_type": "teacher_rollout",
        "metadata": {
            "trial_id": metadata.get("trial_id"),
            "task_id": metadata.get("task_id"),
            "planner_backend": metadata.get("planner_backend"),
            "teacher_version": metadata.get("teacher_version"),
            "missing_fields": missing_fields,
        },
        "counts": {
            "planner_calls": planner_calls,
            "candidate_count": candidate_count,
            "segment_count": len(trajectory_segments),
            "step_count": len(step_logs),
        },
        "phase_sequence": unique_phase_sequence,
        "scores": {
            "rl_step_reward_total": metadata.get("final_metrics", {}).get("rl_step_reward_total"),
            "gym_final_score": metadata.get("final_metrics", {}).get("gym_final_score"),
            "teacher_official_style_score": teacher_score.get("total_score"),
            "official_eval_score": metadata.get("final_metrics", {}).get("official_eval_score"),
        },
        "signal_quality": signal_quality_summary,
        "outcome": outcome,
        "trajectory": {
            "path_length_m": path_length,
            "duration_s": duration,
            "smoothness_jerk_mps3": teacher_score.get("tier2", {}).get("jerk_mps3"),
            "action_delta_mean": action_stats["action_delta_mean"],
            "action_repeat_fraction": action_stats["action_repeat_fraction"],
            "planner_adaptation": _planner_adaptation_label(action_stats, unique_phase_sequence),
        },
        "warnings": warnings,
    }
    markdown = format_rollout_markdown(summary) if include_markdown else ""
    return TeacherEvaluationResult(summary=summary, markdown=markdown)


def analyze_search_payload(
    search_payload: dict[str, Any],
    *,
    include_markdown: bool = True,
) -> TeacherEvaluationResult:
    ranked = list(search_payload.get("ranked_candidates", []))
    metadata = dict(search_payload.get("metadata", {}))
    missing_fields = _missing_fields(search_payload, ["metadata", "ranked_candidates"])
    candidate_summaries = [_candidate_signature(candidate) for candidate in ranked]
    pairwise = _pairwise_candidate_similarity(candidate_summaries)
    order_without_quality = sorted(
        ranked,
        key=lambda item: float(item.get("ranking_metrics", {}).get("teacher_official_style_score", -math.inf)),
        reverse=True,
    )
    top_without_quality = [item["candidate_spec"]["name"] for item in order_without_quality[: metadata.get("top_k", 3)]]
    top_with_quality = [item["candidate_spec"]["name"] for item in ranked[: metadata.get("top_k", 3)]]
    rank_changes = sum(
        1
        for index, item in enumerate(ranked)
        if index < len(order_without_quality)
        and item["candidate_spec"]["name"] != order_without_quality[index]["candidate_spec"]["name"]
    )
    best_planner_waypoint = next(
        (item for item in ranked if item["candidate_spec"]["mode"] == "planner_waypoint"),
        None,
    )
    top_candidate = ranked[0] if ranked else None
    value_over_single = None
    if top_candidate is not None and best_planner_waypoint is not None:
        value_over_single = float(top_candidate["ranking_metrics"]["composite_score"]) - float(
            best_planner_waypoint["ranking_metrics"]["composite_score"]
        )
    quality_penalty_magnitudes = [
        abs(float(item.get("ranking_metrics", {}).get("quality_adjustment", 0.0)))
        for item in ranked
    ]
    warnings = detect_search_warnings(
        {
            "candidate_count": len(ranked),
            "candidate_summaries": candidate_summaries,
            "pairwise_similarity": pairwise,
            "rank_changes_from_quality": rank_changes,
            "value_over_single_plan": value_over_single,
            "quality_penalty_magnitudes": quality_penalty_magnitudes,
        }
    )
    summary = {
        "artifact_type": "teacher_search",
        "metadata": {
            **metadata,
            "missing_fields": missing_fields,
        },
        "top_candidates": [
            {
                "rank": candidate.get("rank"),
                "name": candidate["candidate_spec"]["name"],
                "mode": candidate["candidate_spec"]["mode"],
                "composite_score": candidate["ranking_metrics"]["composite_score"],
                "teacher_official_style_score": candidate["ranking_metrics"]["teacher_official_style_score"],
                "gym_final_score": candidate["ranking_metrics"].get("gym_final_score"),
                "rl_step_reward_total": candidate["ranking_metrics"].get("rl_step_reward_total"),
                "quality_adjustment": candidate["ranking_metrics"].get("quality_adjustment"),
            }
            for candidate in ranked[: metadata.get("top_k", 3)]
        ],
        "ranking_analysis": {
            "rank_changes_from_quality": rank_changes,
            "top_k_without_quality": top_without_quality,
            "top_k_with_quality": top_with_quality,
            "value_over_single_plan": value_over_single,
            "metric_dominance": _metric_dominance(ranked),
        },
        "diversity_analysis": {
            "candidate_count": len(ranked),
            "nearly_duplicate_pairs": [item for item in pairwise if item["similarity"] >= 0.95],
            "pairwise_similarity": pairwise,
        },
        "warnings": warnings,
    }
    markdown = format_search_markdown(summary) if include_markdown else ""
    return TeacherEvaluationResult(summary=summary, markdown=markdown)


def analyze_replay_comparison(
    *,
    original: TeacherReplayArtifact,
    replayed: dict[str, Any],
    include_markdown: bool = True,
) -> TeacherEvaluationResult:
    original_steps = list(original.step_logs)
    replay_records = list(replayed.get("records", []))
    original_reward_total = sum(float(step.get("reward", 0.0)) for step in original_steps)
    replay_reward_total = sum(float(record.get("reward", 0.0)) for record in replay_records)
    original_final = original_steps[-1] if original_steps else {}
    replay_final = replay_records[-1] if replay_records else {}
    original_final_obs = original_final.get("observation_summary", {})
    replay_final_info = dict(replayed.get("final_info", {}))
    replay_final_eval = dict(replay_final_info.get("final_evaluation") or {})
    final_tcp_pose_delta = _norm_delta(
        original_final_obs.get("tcp_pose"),
        replay_final.get("tcp_pose"),
    )
    final_plug_target_delta = _norm_delta(
        original_final_obs.get("plug_to_port_relative"),
        replay_final.get("plug_to_port_relative"),
    )
    gym_final_score_delta = _float_or_none(replay_final_eval.get("gym_final_score"))
    if gym_final_score_delta is not None:
        gym_final_score_delta -= _float_or_none(original.metadata.get("final_metrics", {}).get("gym_final_score")) or 0.0
    fidelity_label = classify_replay_fidelity(
        step_delta=(len(replay_records) - max(len(original_steps), 1)),
        final_tcp_pose_delta=final_tcp_pose_delta,
        final_plug_target_delta=final_plug_target_delta,
        reward_total_delta=abs(replay_reward_total - original_reward_total),
        gym_final_score_delta=abs(gym_final_score_delta or 0.0),
    )
    warnings = detect_replay_warnings(
        {
            "fidelity_label": fidelity_label,
            "final_tcp_pose_delta": final_tcp_pose_delta,
            "final_plug_target_delta": final_plug_target_delta,
            "reward_total_delta": abs(replay_reward_total - original_reward_total),
            "gym_final_score_delta": abs(gym_final_score_delta or 0.0),
        }
    )
    summary = {
        "artifact_type": "teacher_replay_comparison",
        "fidelity": {
            "label": fidelity_label,
            "step_delta": len(replay_records) - max(len(original_steps), 1),
            "final_tcp_pose_delta": final_tcp_pose_delta,
            "final_plug_target_relation_delta": final_plug_target_delta,
            "reward_total_delta": abs(replay_reward_total - original_reward_total),
            "gym_final_score_delta": gym_final_score_delta,
        },
        "metadata_check": {
            "candidate_rank_present": "candidate_rank" in original.metadata,
            "ranking_metrics_present": "ranking_metrics" in original.metadata,
            "scenario_metadata_present": "scenario_metadata" in original.metadata,
            "task_metadata_present": "task_metadata" in original.metadata,
        },
        "warnings": warnings,
    }
    markdown = format_replay_markdown(summary) if include_markdown else ""
    return TeacherEvaluationResult(summary=summary, markdown=markdown)


def classify_replay_fidelity(
    *,
    step_delta: int,
    final_tcp_pose_delta: float | None,
    final_plug_target_delta: float | None,
    reward_total_delta: float,
    gym_final_score_delta: float,
) -> str:
    tcp_delta = final_tcp_pose_delta or 0.0
    relation_delta = final_plug_target_delta or 0.0
    if (
        abs(step_delta) <= 2
        and tcp_delta <= 0.02
        and relation_delta <= 0.02
        and reward_total_delta <= 10.0
        and gym_final_score_delta <= 5.0
    ):
        return "faithful"
    if (
        abs(step_delta) <= 16
        and tcp_delta <= 0.10
        and relation_delta <= 0.10
        and reward_total_delta <= 50.0
        and gym_final_score_delta <= 20.0
    ):
        return "approximately faithful"
    return "poor replay match"


def detect_rollout_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    phase_sequence = list(summary.get("phase_sequence", []))
    data_quality = dict(summary.get("data_quality", {}))
    action_stats = dict(summary.get("action_stats", {}))
    outcome = dict(summary.get("outcome", {}))
    teacher_score = dict(summary.get("teacher_score", {}))
    if len(set(phase_sequence)) <= 1 and summary.get("segment_count", 0) <= 1:
        warnings.append("Planner output collapse: rollout stayed in one phase with one segment.")
    if action_stats.get("action_repeat_fraction", 0.0) >= 0.95:
        warnings.append("Planner repeatedly chose nearly identical actions.")
    if not data_quality.get("wrench", {}).get("is_real", False):
        warnings.append("Rollout depended on approximate or missing wrench data.")
    if not data_quality.get("controller_state", {}).get("is_real", False):
        warnings.append("Controller-derived parity fields were unavailable during planning.")
    if outcome.get("failure_mode") not in {None, "success"} and (teacher_score.get("total_score") or 0.0) > 40.0:
        warnings.append("Suspiciously strong local score despite failed rollout outcome.")
    return warnings


def detect_search_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    candidate_count = int(summary.get("candidate_count", 0))
    value_over_single = summary.get("value_over_single_plan")
    pairwise = list(summary.get("pairwise_similarity", []))
    duplicates = [item for item in pairwise if item["similarity"] >= 0.95]
    if candidate_count and len(duplicates) >= max(1, candidate_count // 3):
        warnings.append("Search contains many near-duplicate candidates; planner diversity is low.")
    if value_over_single is not None and value_over_single <= 0.1:
        warnings.append("Search adds limited value over the best single planner candidate.")
    if int(summary.get("rank_changes_from_quality", 0)) == 0:
        warnings.append("Quality penalties are not materially changing rank order.")
    penalty_magnitudes = list(summary.get("quality_penalty_magnitudes", []))
    if penalty_magnitudes and max(penalty_magnitudes) >= 8.0:
        warnings.append("Ranking is heavily influenced by approximate-signal penalties.")
    return warnings


def detect_replay_warnings(summary: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if summary.get("fidelity_label") == "poor replay match":
        warnings.append("Replay fidelity is poor for this artifact and reset path.")
    if (summary.get("final_tcp_pose_delta") or 0.0) > 0.05:
        warnings.append("Replay final TCP pose drift is significant.")
    if (summary.get("gym_final_score_delta") or 0.0) > 10.0:
        warnings.append("Replay gym_final_score changed materially.")
    return warnings


def format_rollout_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Teacher Rollout Evaluation",
            f"- Trial: `{summary['metadata'].get('trial_id')}`",
            f"- Task: `{summary['metadata'].get('task_id')}`",
            f"- Planner backend: `{summary['metadata'].get('planner_backend')}`",
            f"- Planner calls: `{summary['counts']['planner_calls']}`",
            f"- Segments: `{summary['counts']['segment_count']}`",
            f"- Steps: `{summary['counts']['step_count']}`",
            f"- Phase sequence: `{summary['phase_sequence']}`",
            f"- `rl_step_reward_total`: `{summary['scores'].get('rl_step_reward_total')}`",
            f"- `gym_final_score`: `{summary['scores'].get('gym_final_score')}`",
            f"- `teacher_official_style_score`: `{summary['scores'].get('teacher_official_style_score')}`",
            f"- Outcome: `{summary['outcome']['failure_mode']}`",
            f"- Planner adaptation: `{summary['trajectory']['planner_adaptation']}`",
            "## Warnings",
            *([f"- {warning}" for warning in summary["warnings"]] or ["- None"]),
        ]
    )


def format_search_markdown(summary: dict[str, Any]) -> str:
    top_lines = [
        f"- Rank {candidate['rank']}: `{candidate['name']}` (`{candidate['mode']}`) composite=`{candidate['composite_score']:.3f}`"
        for candidate in summary["top_candidates"]
    ]
    return "\n".join(
        [
            "# Teacher Search Evaluation",
            f"- Planner backend: `{summary['metadata'].get('planner_backend')}`",
            f"- Top-K: `{summary['metadata'].get('top_k')}`",
            f"- Rank changes from quality penalties: `{summary['ranking_analysis']['rank_changes_from_quality']}`",
            f"- Value over best single planner candidate: `{summary['ranking_analysis']['value_over_single_plan']}`",
            "## Top Candidates",
            *(top_lines or ["- None"]),
            "## Warnings",
            *([f"- {warning}" for warning in summary["warnings"]] or ["- None"]),
        ]
    )


def format_replay_markdown(summary: dict[str, Any]) -> str:
    fidelity = summary["fidelity"]
    return "\n".join(
        [
            "# Teacher Replay Evaluation",
            f"- Fidelity: `{fidelity['label']}`",
            f"- Step delta: `{fidelity['step_delta']}`",
            f"- Final TCP pose delta: `{fidelity['final_tcp_pose_delta']}`",
            f"- Final plug-target relation delta: `{fidelity['final_plug_target_relation_delta']}`",
            f"- Reward total delta: `{fidelity['reward_total_delta']}`",
            f"- `gym_final_score` delta: `{fidelity['gym_final_score_delta']}`",
            "## Warnings",
            *([f"- {warning}" for warning in summary["warnings"]] or ["- None"]),
        ]
    )


def _candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    artifact = candidate.get("artifact", {})
    segments = list(artifact.get("trajectory_segments", []))
    phases = [segment.get("phase") for segment in segments if segment.get("phase")]
    waypoint_signature: list[list[float]] = []
    for segment in segments[:3]:
        points = list(segment.get("points", []))
        if points:
            pose = points[min(len(points) - 1, 0)].get("target_tcp_pose", [])[:3]
            waypoint_signature.append([float(value) for value in pose])
    return {
        "name": candidate["candidate_spec"]["name"],
        "mode": candidate["candidate_spec"]["mode"],
        "phase_sequence": _collapse_adjacent_duplicates(phases),
        "waypoint_signature": waypoint_signature,
        "composite_score": float(candidate.get("ranking_metrics", {}).get("composite_score", 0.0)),
    }


def _pairwise_candidate_similarity(candidate_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(candidate_summaries):
        for right in candidate_summaries[index + 1 :]:
            phase_similarity = 1.0 if left["phase_sequence"] == right["phase_sequence"] else 0.0
            waypoint_distance = _signature_distance(
                left.get("waypoint_signature", []),
                right.get("waypoint_signature", []),
            )
            waypoint_similarity = 1.0 / (1.0 + waypoint_distance)
            similarity = 0.6 * phase_similarity + 0.4 * waypoint_similarity
            pairs.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "similarity": similarity,
                    "phase_similarity": phase_similarity,
                    "waypoint_distance": waypoint_distance,
                }
            )
    return sorted(pairs, key=lambda item: item["similarity"], reverse=True)


def _signature_distance(left: list[list[float]], right: list[list[float]]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    count = min(len(left), len(right))
    total = 0.0
    for index in range(count):
        total += float(np.linalg.norm(np.asarray(left[index]) - np.asarray(right[index])))
    return total / count


def _metric_dominance(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranked:
        return {}
    teacher_scores = np.asarray(
        [item["ranking_metrics"]["teacher_official_style_score"] for item in ranked],
        dtype=np.float64,
    )
    quality_adjustments = np.asarray(
        [item["ranking_metrics"]["quality_adjustment"] for item in ranked],
        dtype=np.float64,
    )
    composite_scores = np.asarray(
        [item["ranking_metrics"]["composite_score"] for item in ranked],
        dtype=np.float64,
    )
    ranges = {
        "teacher_official_style_score": float(teacher_scores.max() - teacher_scores.min()),
        "quality_adjustment": float(quality_adjustments.max() - quality_adjustments.min()),
        "composite_score": float(composite_scores.max() - composite_scores.min()),
    }
    return {
        "teacher_score_range": ranges["teacher_official_style_score"],
        "quality_adjustment_range": ranges["quality_adjustment"],
        "composite_score_range": ranges["composite_score"],
        "dominant_metric": max(
            ("teacher_official_style_score", "quality_adjustment"),
            key=lambda name: ranges[name],
        ),
    }


def _missing_fields(payload: dict[str, Any], required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if field not in payload]


def _collapse_adjacent_duplicates(items: list[str]) -> list[str]:
    collapsed: list[str] = []
    for item in items:
        if not collapsed or collapsed[-1] != item:
            collapsed.append(item)
    return collapsed


def _action_statistics(step_logs: list[dict[str, Any]]) -> dict[str, float]:
    if len(step_logs) < 2:
        return {"action_delta_mean": 0.0, "action_repeat_fraction": 1.0}
    actions = np.asarray([step.get("trajectory_point", {}).get("action", [0.0] * 6) for step in step_logs], dtype=np.float64)
    deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    return {
        "action_delta_mean": float(deltas.mean()) if deltas.size else 0.0,
        "action_repeat_fraction": float(np.mean(deltas <= 1e-4)) if deltas.size else 1.0,
    }


def _planner_adaptation_label(action_stats: dict[str, float], phase_sequence: list[str]) -> str:
    if len(phase_sequence) > 1:
        return "phase-adaptive"
    if action_stats.get("action_delta_mean", 0.0) > 0.005:
        return "action-adaptive"
    return "weakly adaptive"


def _tcp_positions(step_logs: list[dict[str, Any]]) -> list[np.ndarray]:
    positions: list[np.ndarray] = []
    for step in step_logs:
        obs = step.get("observation_summary", {})
        pose = obs.get("tcp_pose")
        if isinstance(pose, list) and len(pose) >= 3:
            positions.append(np.asarray(pose[:3], dtype=np.float64))
    return positions


def _path_length(positions: list[np.ndarray]) -> float:
    if len(positions) < 2:
        return 0.0
    return float(sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:])))


def _duration(step_logs: list[dict[str, Any]]) -> float:
    if len(step_logs) < 2:
        return 0.0
    return float(step_logs[-1].get("sim_time", 0.0)) - float(step_logs[0].get("sim_time", 0.0))


def _signal_quality_summary(data_quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "signals": data_quality,
        "real_signals": sorted(
            signal for signal, quality in data_quality.items() if quality.get("is_real", False)
        ),
        "approximate_or_missing_signals": sorted(
            signal for signal, quality in data_quality.items() if not quality.get("is_real", False)
        ),
    }


def _failure_mode(*, final_info: dict[str, Any], teacher_score: dict[str, Any]) -> str:
    if bool(final_info.get("success", False)):
        return "success"
    if bool(final_info.get("wrong_port", False)):
        return "wrong_port"
    tier3_status = teacher_score.get("tier3", {}).get("status")
    if tier3_status:
        return str(tier3_status)
    return "failure"


def _norm_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.size == 0 or right_array.size == 0:
        return None
    count = min(left_array.size, right_array.size)
    return float(np.linalg.norm(left_array.reshape(-1)[:count] - right_array.reshape(-1)[:count]))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
