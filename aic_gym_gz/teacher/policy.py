"""Hierarchical teacher controller."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..planners.base import PlannerBackend
from ..probes.library import ProbeLibrary
from ..trajectory.smoothing import MinimumJerkSmoother
from .close_range import CloseRangeInsertionPolicy
from .context import TeacherContextExtractor
from .history import TemporalObservationBuffer
from .types import CandidateEvaluation, TeacherPlan, TrajectorySegment


@dataclass
class TeacherConfig:
    candidate_plan_count: int = 2
    enable_probes: bool = True
    max_probe_actions: int = 3
    segment_limit: int = 8
    max_env_steps: int = 512
    run_until_env_done: bool = False
    max_planner_calls_per_episode: int = 10
    hold_ticks_per_action: int = 8
    planner_backend_name: str = "mock-deterministic"
    planner_output_mode: str = "absolute_cartesian_waypoint"
    prefer_live_scene_overview: bool = False
    enable_global_guidance: bool = False
    global_plan_interval_segments: int = 4
    enable_close_range_handoff: bool = True


@dataclass
class AgentTeacherController:
    planner: PlannerBackend
    context_extractor: TeacherContextExtractor = field(default_factory=TeacherContextExtractor)
    smoother: MinimumJerkSmoother = field(default_factory=MinimumJerkSmoother)
    probe_library: ProbeLibrary = field(default_factory=ProbeLibrary)
    config: TeacherConfig = field(default_factory=TeacherConfig)
    current_phase: str = "free_space_approach"
    last_rationale: str | None = None
    segment_index: int = 0
    planner_call_count: int = 0
    cached_global_guidance: dict[str, Any] | None = None
    close_range_policy: CloseRangeInsertionPolicy = field(default_factory=CloseRangeInsertionPolicy)

    def __post_init__(self) -> None:
        desired_base_dt = max(int(self.config.hold_ticks_per_action), 1) * 0.002
        if (
            self.smoother.planner_output_mode != self.config.planner_output_mode
            or abs(float(self.smoother.base_dt) - float(desired_base_dt)) > 1e-9
        ):
            self.smoother = replace(
                self.smoother,
                planner_output_mode=self.config.planner_output_mode,
                base_dt=desired_base_dt,
            )
        if self.context_extractor.prefer_live_scene_overview != self.config.prefer_live_scene_overview:
            self.context_extractor = replace(
                self.context_extractor,
                prefer_live_scene_overview=self.config.prefer_live_scene_overview,
            )

    def select_plan(
        self,
        *,
        scenario,
        task_id: str,
        state,
        temporal_buffer: TemporalObservationBuffer,
        recent_probe_results: list[dict[str, Any]],
        include_images: bool,
    ) -> tuple[TeacherPlan, TrajectorySegment, list[dict[str, Any]]]:
        planning_state = self.context_extractor.build_planning_state(
            scenario=scenario,
            task_id=task_id,
            state=state,
            temporal_buffer=temporal_buffer,
            current_phase=self.current_phase,
            recent_probe_results=recent_probe_results,
            include_images=include_images,
            last_teacher_rationale=self.last_rationale,
        )
        global_guidance = self._maybe_refresh_global_guidance(planning_state)
        planning_state.planning_metadata["planner_output_mode"] = self.config.planner_output_mode
        if global_guidance is not None:
            planning_state.temporal_context["global_guidance"] = global_guidance
            planning_state.planning_metadata["global_guidance"] = global_guidance
        candidates: list[CandidateEvaluation] = []
        best_plan: TeacherPlan | None = None
        best_segment: TrajectorySegment | None = None
        best_score: float | None = None
        for candidate_index in range(self.config.candidate_plan_count):
            plan = self.planner.plan(planning_state, candidate_index=candidate_index)
            self.planner_call_count += 1
            segment = self.smoother.smooth(state=state, plan=plan)
            score = self._score_candidate(
                plan=plan,
                segment=segment,
                dynamics=planning_state.dynamics_summary,
                temporal_context=planning_state.temporal_context,
                current_phase=planning_state.current_phase,
                policy_context=planning_state.policy_context,
                data_quality=planning_state.data_quality,
            )
            candidates.append(
                CandidateEvaluation(
                    name=f"candidate_{candidate_index}",
                    score=score,
                    plan=plan.to_dict(),
                    segment=segment.to_dict(),
                    notes="selected" if best_score is None or score > best_score else "",
                    metrics={
                        "candidate_index": candidate_index,
                        "history_window_size": planning_state.temporal_context.get("window_size", 0),
                        "cable_settling_score": planning_state.dynamics_summary.get("cable_settling_score", 0.0),
                    },
                    data_quality=planning_state.data_quality,
                )
            )
            if best_score is None or score > best_score:
                best_plan = plan
                best_segment = segment
                best_score = score
        assert best_plan is not None
        assert best_segment is not None
        self.current_phase = best_plan.next_phase
        self.last_rationale = best_plan.rationale_summary
        self.segment_index += 1
        return best_plan, best_segment, [candidate.to_dict() for candidate in candidates]

    def should_probe(self, plan: TeacherPlan, temporal_buffer: TemporalObservationBuffer) -> bool:
        dynamics = temporal_buffer.dynamics_summary()
        return bool(
            self.config.enable_probes
            and plan.should_probe
            and (not dynamics.quasi_static or dynamics.cable_settling_score < 0.7)
        )

    def initial_action(self) -> np.ndarray:
        return np.zeros(6, dtype=np.float32)

    def planner_budget_exhausted(self) -> bool:
        return self.planner_call_count >= max(int(self.config.max_planner_calls_per_episode), 1)

    def should_handoff_to_close_range(self, observation: dict[str, Any]) -> bool:
        if not self.config.enable_close_range_handoff:
            return False
        return self.close_range_policy.should_handoff(observation)

    def force_close_range_handoff(self) -> None:
        self.close_range_policy.force_handoff()

    def close_range_action(self, observation: dict[str, Any]) -> np.ndarray:
        return self.close_range_policy.action(observation)

    def _score_candidate(
        self,
        *,
        plan: TeacherPlan,
        segment: TrajectorySegment,
        dynamics: dict[str, Any],
        temporal_context: dict[str, Any],
        current_phase: str,
        policy_context: dict[str, Any],
        data_quality: dict[str, Any],
    ) -> float:
        duration_penalty = segment.expected_duration_s
        caution_penalty = 0.15 if plan.caution_flag else 0.0
        settling_bonus = float(dynamics["cable_settling_score"])
        granularity_bonus = 0.1 if plan.segment_granularity == "guarded" else 0.0
        controller_bonus = 0.05 if data_quality.get("controller_state", {}).get("is_real", False) else 0.0
        wrench_penalty = 0.1 if not data_quality.get("wrench", {}).get("is_real", False) else 0.0
        auxiliary = temporal_context.get("auxiliary_history_summary", {})
        phase_guidance = temporal_context.get("phase_guidance", {})
        hidden_contact_penalty = 0.05 if auxiliary.get("hidden_contact_recent", False) else 0.0
        repeated_contact_penalty = 0.02 * min(int(auxiliary.get("repeated_contact_rich_steps", 0)), 2)
        recommended_phase_bonus = 0.12 if plan.next_phase == phase_guidance.get("recommended_phase") else 0.0
        stuck_phase_penalty = (
            0.18
            if plan.next_phase == current_phase and phase_guidance.get("should_avoid_repeating_current_phase", False)
            else 0.0
        )
        insertion_zone_bonus = (
            0.10
            if phase_guidance.get("in_insertion_zone", False)
            and plan.segment_granularity in {"fine", "guarded"}
            else 0.0
        )
        guarded_insert_bonus = (
            0.12
            if phase_guidance.get("insertion_ready", False) and plan.next_phase == "guarded_insert"
            else 0.0
        )
        alignment_bonus = (
            0.08
            if policy_context.get("distance_to_entrance", 1.0) <= 0.08 and plan.next_phase == "pre_insert_align"
            else 0.0
        )
        progress_bonus, stagnation_penalty = self._progress_terms(plan=plan, policy_context=policy_context)
        global_phase_bonus, milestone_bonus = self._global_guidance_terms(
            plan=plan,
            policy_context=policy_context,
            temporal_context=temporal_context,
        )
        return (
            settling_bonus
            + granularity_bonus
            + controller_bonus
            + recommended_phase_bonus
            + insertion_zone_bonus
            + guarded_insert_bonus
            + alignment_bonus
            + progress_bonus
            + global_phase_bonus
            + milestone_bonus
            - duration_penalty
            - caution_penalty
            - wrench_penalty
            - hidden_contact_penalty
            - repeated_contact_penalty
            - stuck_phase_penalty
            - stagnation_penalty
        )

    def _progress_terms(
        self,
        *,
        plan: TeacherPlan,
        policy_context: dict[str, Any],
    ) -> tuple[float, float]:
        if not plan.waypoints:
            return 0.0, 0.2
        plug = np.asarray(policy_context.get("plug_pose", [0.0, 0.0, 0.0]), dtype=np.float64)[:3]
        target = np.asarray(policy_context.get("target_port_pose", [0.0, 0.0, 0.0]), dtype=np.float64)[:3]
        entrance = np.asarray(
            policy_context.get("target_port_entrance_pose") or policy_context.get("target_port_pose", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )[:3]
        target_ref = entrance if plan.next_phase in {"pre_insert_align", "guarded_insert"} else target
        current_distance = float(np.linalg.norm(plug - target_ref))
        planned_distance = float(np.linalg.norm(np.asarray(plan.waypoints[-1].position_xyz, dtype=np.float64) - target_ref))
        progress = max(current_distance - planned_distance, 0.0)
        progress_bonus = 8.0 * progress
        stagnation_penalty = 0.25 if progress <= 0.002 else 0.0
        return progress_bonus, stagnation_penalty

    def _global_guidance_terms(
        self,
        *,
        plan: TeacherPlan,
        policy_context: dict[str, Any],
        temporal_context: dict[str, Any],
    ) -> tuple[float, float]:
        guidance = temporal_context.get("global_guidance", {})
        if not isinstance(guidance, dict) or not guidance:
            return 0.0, 0.0
        phase_sequence = guidance.get("phase_sequence") or []
        current_phase = str(temporal_context.get("phase_guidance", {}).get("current_phase", ""))
        if not isinstance(phase_sequence, list):
            phase_sequence = []
        next_expected_phase = None
        for phase in phase_sequence:
            if phase != current_phase:
                next_expected_phase = phase
                break
        global_phase_bonus = 0.0
        if next_expected_phase is not None and plan.next_phase == next_expected_phase:
            global_phase_bonus = 0.12
        elif plan.next_phase in phase_sequence[:2]:
            global_phase_bonus = 0.06

        milestones = guidance.get("milestones") or []
        if not isinstance(milestones, list) or not milestones or not plan.waypoints:
            return global_phase_bonus, 0.0
        current_target = np.asarray(plan.waypoints[-1].position_xyz, dtype=np.float64)
        milestone_bonus = 0.0
        for milestone in milestones:
            if not isinstance(milestone, dict):
                continue
            position_xyz = milestone.get("position_xyz")
            phase = milestone.get("phase")
            if not isinstance(position_xyz, (list, tuple)) or len(position_xyz) < 3:
                continue
            milestone_target = np.asarray(position_xyz[:3], dtype=np.float64)
            distance = float(np.linalg.norm(current_target - milestone_target))
            if phase == plan.next_phase:
                if distance <= 0.03:
                    milestone_bonus = max(milestone_bonus, 0.18)
                elif distance <= 0.08:
                    milestone_bonus = max(milestone_bonus, 0.1)
            elif distance <= 0.03:
                milestone_bonus = max(milestone_bonus, 0.04)
        return global_phase_bonus, milestone_bonus

    def _maybe_refresh_global_guidance(self, planning_state) -> dict[str, Any] | None:
        if not self.config.enable_global_guidance:
            return self.cached_global_guidance
        if (
            self.cached_global_guidance is not None
            and self.segment_index % max(self.config.global_plan_interval_segments, 1) != 0
        ):
            return self.cached_global_guidance
        planner_guidance = self.planner.plan_global_guidance(planning_state, candidate_index=0)
        if planner_guidance is not None:
            self.cached_global_guidance = planner_guidance
            return planner_guidance
        phase_guidance = planning_state.temporal_context.get("phase_guidance", {})
        self.cached_global_guidance = {
            "source": "teacher_heuristic_global_guidance",
            "segment_index": self.segment_index,
            "current_phase": planning_state.current_phase,
            "recommended_phase": phase_guidance.get("recommended_phase"),
            "allowed_phases": phase_guidance.get("allowed_phases", []),
            "milestones": [
                {
                    "name": "entrance_setup",
                    "pose": planning_state.policy_context.get("target_port_entrance_pose"),
                    "phase": "pre_insert_align",
                },
                {
                    "name": "target_insert",
                    "pose": planning_state.policy_context.get("target_port_pose"),
                    "phase": "guarded_insert",
                },
            ],
        }
        return self.cached_global_guidance
