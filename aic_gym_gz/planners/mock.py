"""Deterministic mock backend for tests and local demos."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .base import PlannerBackend
from ..teacher.types import TeacherPlan, TeacherPlanningState, TeacherWaypoint


@dataclass(frozen=True)
class DeterministicMockPlannerBackend(PlannerBackend):
    coarse_step_m: float = 0.06
    fine_step_m: float = 0.015

    @property
    def backend_name(self) -> str:
        return "mock-deterministic"

    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        plug = state.policy_context["plug_pose"]
        target = state.policy_context["target_port_pose"]
        dx = float(target[0] - plug[0])
        dy = float(target[1] - plug[1])
        dz = float(target[2] - plug[2])
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > 0.12:
            phase = "free_space_approach"
            mode = "coarse_cartesian"
            granularity = "coarse"
            step = self.coarse_step_m
            waypoint_count = 2
        elif distance > 0.025:
            phase = "pre_insert_align"
            mode = "fine_cartesian"
            granularity = "fine"
            step = self.fine_step_m
            waypoint_count = 2
        else:
            phase = "guarded_insert"
            mode = "guarded_insert"
            granularity = "guarded"
            step = min(self.fine_step_m, max(distance, 0.004))
            waypoint_count = 1
        lateral_bias = 0.006 * candidate_index
        waypoints = []
        for index in range(waypoint_count):
            alpha = min(1.0, ((index + 1) * step) / max(distance, step))
            waypoints.append(
                TeacherWaypoint(
                    position_xyz=(
                        float(plug[0] + dx * alpha),
                        float(plug[1] + dy * alpha + (lateral_bias if phase != "guarded_insert" else 0.0)),
                        float(plug[2] + dz * alpha),
                    ),
                    yaw=float(target[5]),
                    speed_scale=0.5 if granularity != "coarse" else 1.0,
                    clearance_hint=0.01 if granularity != "coarse" else 0.06,
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
            f"distance={distance:.4f}m phase={phase} candidate={candidate_index} "
            f"settling={state.dynamics_summary['cable_settling_score']:.3f}"
        )
        return TeacherPlan(
            next_phase=phase,  # type: ignore[arg-type]
            waypoints=tuple(waypoints),
            motion_mode=mode,  # type: ignore[arg-type]
            caution_flag=distance < 0.03 or bool(state.policy_context["off_limit_contact"]),
            should_probe=should_probe,
            segment_horizon_steps=12 if granularity == "coarse" else 6,
            segment_granularity=granularity,  # type: ignore[arg-type]
            rationale_summary=rationale,
        )
