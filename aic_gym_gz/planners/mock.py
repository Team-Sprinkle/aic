"""Deterministic mock backend for tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .base import PlannerBackend
from ..teacher.planning import candidate_family_for_index
from ..teacher.types import TeacherPlan, TeacherPlanningState, TeacherWaypoint


@dataclass(frozen=True)
class DeterministicMockPlannerBackend(PlannerBackend):
    coarse_step_m: float = 0.06
    fine_step_m: float = 0.015

    @property
    def backend_name(self) -> str:
        return "mock-deterministic"

    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        family = candidate_family_for_index(candidate_index)
        plug = state.policy_context["plug_pose"]
        target = state.policy_context["target_port_pose"]
        entrance = state.policy_context.get("target_port_entrance_pose") or target
        phase_guidance = state.temporal_context.get("phase_guidance", {})
        distance_to_entrance = float(state.policy_context.get("distance_to_entrance", 0.0) or 0.0)
        insertion_progress = float(state.policy_context.get("insertion_progress", 0.0) or 0.0)
        lateral_misalignment = float(state.policy_context.get("lateral_misalignment", 0.0) or 0.0)
        orientation_error = float(state.policy_context.get("orientation_error", 0.0) or 0.0)
        hidden_contact_recent = bool(state.temporal_context.get("auxiliary_history_summary", {}).get("hidden_contact_recent", False))
        dx = float(target[0] - plug[0])
        dy = float(target[1] - plug[1])
        dz = float(target[2] - plug[2])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if hidden_contact_recent or bool(state.policy_context["off_limit_contact"]):
            phase = "backoff_and_retry"
            mode = "fine_cartesian"
            granularity = "fine"
            step = self.fine_step_m
            waypoint_count = 2
        elif bool(phase_guidance.get("insertion_ready", False)) or family["name"] == "guarded_insert" and distance_to_entrance <= 0.03:
            phase = "guarded_insert"
            mode = "guarded_insert"
            granularity = "guarded"
            step = min(self.fine_step_m, max(distance_to_entrance, 0.003))
            waypoint_count = 2
        elif bool(phase_guidance.get("alignment_needed", False)) or family["name"] == "alignment_first":
            phase = "pre_insert_align"
            mode = "fine_cartesian"
            granularity = "fine"
            step = self.fine_step_m
            waypoint_count = 2
        elif family["name"] == "obstacle_clearance" or distance > 0.10 and distance_to_entrance <= 0.18:
            phase = "obstacle_avoidance"
            mode = "coarse_cartesian"
            granularity = "coarse"
            step = self.coarse_step_m
            waypoint_count = 2
        else:
            phase = "free_space_approach"
            mode = "coarse_cartesian"
            granularity = "coarse"
            step = self.coarse_step_m
            waypoint_count = 2
        if distance_to_entrance <= 0.02 and insertion_progress < 0.05 and phase == "free_space_approach":
            phase = "pre_insert_align"
            mode = "fine_cartesian"
            granularity = "fine"
            step = self.fine_step_m
        lateral_bias = {
            "baseline_safe": 0.0,
            "obstacle_clearance": 0.02,
            "alignment_first": 0.01 if lateral_misalignment >= 0.0 else -0.01,
            "guarded_insert": 0.0,
            "recovery_backoff": -0.015,
        }.get(family["name"], 0.0)
        z_bias = -0.01 if family["name"] == "recovery_backoff" else 0.0
        waypoints = []
        for index in range(waypoint_count):
            alpha = min(1.0, ((index + 1) * step) / max(distance, step))
            target_ref = entrance if phase in {"pre_insert_align", "guarded_insert"} else target
            ref_dx = float(target_ref[0] - plug[0])
            ref_dy = float(target_ref[1] - plug[1])
            ref_dz = float(target_ref[2] - plug[2])
            waypoint_x = float(plug[0] + ref_dx * alpha)
            waypoint_y = float(plug[1] + ref_dy * alpha + lateral_bias * (1.0 - 0.35 * index))
            waypoint_z = float(plug[2] + ref_dz * alpha + z_bias)
            if phase == "guarded_insert":
                waypoint_x = float(plug[0] + ref_dx * min(alpha, 0.5))
                waypoint_y = float(plug[1] + ref_dy * min(alpha, 0.5))
                waypoint_z = float(plug[2] + ref_dz * min(alpha + 0.2, 1.0))
            waypoints.append(
                TeacherWaypoint(
                    position_xyz=(
                        waypoint_x,
                        waypoint_y,
                        waypoint_z,
                    ),
                    yaw=float(target[5] + (0.08 if family["name"] == "obstacle_clearance" else 0.0)),
                    speed_scale=(
                        0.35 if phase == "guarded_insert" else
                        0.45 if granularity != "coarse" or family["name"] == "alignment_first" else
                        0.9
                    ),
                    clearance_hint=(
                        0.08 if family["name"] == "obstacle_clearance" else
                        0.015 if granularity != "coarse" else
                        0.05
                    ),
                )
            )
        should_probe = (
            state.current_phase == "cable_probe"
            or (
                not state.dynamics_summary["quasi_static"]
                and state.dynamics_summary["cable_settling_score"] < 0.55
                and distance < 0.08
            )
        )
        rationale = (
            f"family={family['name']} distance={distance:.4f}m entrance={distance_to_entrance:.4f}m "
            f"phase={phase} candidate={candidate_index} progress={insertion_progress:.3f} "
            f"lateral={lateral_misalignment:.4f} orient={orientation_error:.4f} "
            f"settling={state.dynamics_summary['cable_settling_score']:.3f}"
        )
        return TeacherPlan(
            next_phase=phase,  # type: ignore[arg-type]
            waypoints=tuple(waypoints),
            motion_mode=mode,  # type: ignore[arg-type]
            caution_flag=distance < 0.03 or bool(state.policy_context["off_limit_contact"]),
            should_probe=should_probe,
            segment_horizon_steps=8 if granularity == "coarse" else (5 if granularity == "fine" else 4),
            segment_granularity=granularity,  # type: ignore[arg-type]
            rationale_summary=rationale,
        )
