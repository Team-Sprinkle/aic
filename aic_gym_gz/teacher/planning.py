"""Shared teacher planning heuristics for candidate diversity and phase guidance."""

from __future__ import annotations

from typing import Any


def candidate_family_for_index(candidate_index: int) -> dict[str, Any]:
    families = (
        {
            "name": "baseline_safe",
            "role": "baseline safe approach",
            "preferred_phase": "free_space_approach",
            "preferred_motion_mode": "coarse_cartesian",
            "planner_instruction": "Prefer a safe baseline that makes measurable progress without overcommitting.",
        },
        {
            "name": "obstacle_clearance",
            "role": "clearance-biased obstacle-aware approach",
            "preferred_phase": "obstacle_avoidance",
            "preferred_motion_mode": "coarse_cartesian",
            "planner_instruction": "Prefer a wider, clearance-aware path that meaningfully differs from the baseline.",
        },
        {
            "name": "alignment_first",
            "role": "alignment-first entrance setup",
            "preferred_phase": "pre_insert_align",
            "preferred_motion_mode": "fine_cartesian",
            "planner_instruction": "Trade speed for earlier entrance alignment and corridor setup.",
        },
        {
            "name": "guarded_insert",
            "role": "insertion-first guarded micro-adjustment",
            "preferred_phase": "guarded_insert",
            "preferred_motion_mode": "guarded_insert",
            "planner_instruction": "If geometry permits, prefer guarded insertion micro-motions near the entrance.",
        },
        {
            "name": "recovery_backoff",
            "role": "contact-aware recovery or backoff",
            "preferred_phase": "backoff_and_retry",
            "preferred_motion_mode": "fine_cartesian",
            "planner_instruction": "If contact risk rises or progress stalls, back off and reset alignment rather than forcing insertion.",
        },
    )
    family = dict(families[candidate_index % len(families)])
    family["candidate_index"] = int(candidate_index)
    return family


def phase_guidance_from_state(
    *,
    current_phase: str,
    policy_context: dict[str, Any],
    temporal_context: dict[str, Any],
    obstacle_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    score_geometry = dict(policy_context.get("score_geometry", {}))
    auxiliary = dict(temporal_context.get("auxiliary_history_summary", {}))
    progress = dict(temporal_context.get("geometry_progress_summary", {}))
    distance_to_target = float(policy_context.get("distance_to_target", 0.0) or 0.0)
    distance_to_entrance = float(score_geometry.get("distance_to_entrance", distance_to_target) or distance_to_target)
    insertion_progress = float(score_geometry.get("insertion_progress", 0.0) or 0.0)
    lateral_misalignment = float(score_geometry.get("lateral_misalignment", 0.0) or 0.0)
    orientation_error = float(score_geometry.get("orientation_error", 0.0) or 0.0)
    hidden_contact_recent = bool(auxiliary.get("hidden_contact_recent", False))
    recent_contact = bool(auxiliary.get("had_contact_recent", False))
    off_limit_contact = bool(policy_context.get("off_limit_contact", False))
    obstacle_count = sum(int(item.get("present", False)) for item in obstacle_summary)
    is_far_from_target = bool(distance_to_target >= 0.15 and distance_to_entrance >= 0.12)
    is_near_entrance = bool(distance_to_entrance <= 0.08)
    in_insertion_zone = bool(distance_to_entrance <= 0.03 or insertion_progress >= 0.15)
    alignment_needed = bool(lateral_misalignment >= 0.012 or orientation_error >= 0.10)
    insertion_ready = bool(
        in_insertion_zone
        and lateral_misalignment <= 0.006
        and orientation_error <= 0.08
        and not hidden_contact_recent
        and not off_limit_contact
    )
    progress_delta = float(progress.get("net_distance_to_entrance_progress", 0.0) or 0.0)
    stuck_without_progress = bool(
        int(progress.get("history_items", 0)) >= 3
        and abs(progress_delta) <= 0.002
    )
    if off_limit_contact or hidden_contact_recent:
        recommended_phase = "backoff_and_retry"
    elif insertion_ready:
        recommended_phase = "guarded_insert"
    elif is_near_entrance or (distance_to_target <= 0.12 and alignment_needed):
        recommended_phase = "pre_insert_align"
    elif obstacle_count > 0 and distance_to_target <= 0.20:
        recommended_phase = "obstacle_avoidance"
    else:
        recommended_phase = "free_space_approach"
    allowed_phases = [recommended_phase]
    if recommended_phase == "free_space_approach":
        allowed_phases.append("obstacle_avoidance")
    elif recommended_phase == "obstacle_avoidance":
        allowed_phases.extend(["free_space_approach", "pre_insert_align"])
    elif recommended_phase == "pre_insert_align":
        allowed_phases.extend(["obstacle_avoidance", "guarded_insert"])
    elif recommended_phase == "guarded_insert":
        allowed_phases.extend(["pre_insert_align", "backoff_and_retry"])
    elif recommended_phase == "backoff_and_retry":
        allowed_phases.extend(["obstacle_avoidance", "pre_insert_align"])
    return {
        "recommended_phase": recommended_phase,
        "allowed_phases": allowed_phases,
        "distance_to_target": distance_to_target,
        "distance_to_entrance": distance_to_entrance,
        "is_far_from_target": is_far_from_target,
        "lateral_misalignment": lateral_misalignment,
        "orientation_error": orientation_error,
        "insertion_progress": insertion_progress,
        "is_near_entrance": is_near_entrance,
        "in_insertion_zone": in_insertion_zone,
        "alignment_needed": alignment_needed,
        "insertion_ready": insertion_ready,
        "hidden_contact_recent": hidden_contact_recent,
        "had_contact_recent": recent_contact,
        "off_limit_contact": off_limit_contact,
        "stuck_without_progress": stuck_without_progress,
        "current_phase": current_phase,
        "should_avoid_repeating_current_phase": bool(stuck_without_progress and recommended_phase != current_phase),
    }
