"""Hierarchical teacher controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..planners.base import PlannerBackend
from ..probes.library import ProbeLibrary
from ..trajectory.smoothing import MinimumJerkSmoother
from .context import TeacherContextExtractor
from .history import TemporalObservationBuffer
from .types import CandidateEvaluation, TeacherPlan, TrajectorySegment


@dataclass
class TeacherConfig:
    candidate_plan_count: int = 2
    enable_probes: bool = True
    max_probe_actions: int = 3
    segment_limit: int = 8
    hold_ticks_per_action: int = 8
    planner_backend_name: str = "mock-deterministic"


@dataclass
class AgentTeacherController:
    planner: PlannerBackend
    context_extractor: TeacherContextExtractor = field(default_factory=TeacherContextExtractor)
    smoother: MinimumJerkSmoother = field(default_factory=MinimumJerkSmoother)
    probe_library: ProbeLibrary = field(default_factory=ProbeLibrary)
    config: TeacherConfig = field(default_factory=TeacherConfig)
    current_phase: str = "free_space_approach"
    last_rationale: str | None = None

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
        candidates: list[CandidateEvaluation] = []
        best_plan: TeacherPlan | None = None
        best_segment: TrajectorySegment | None = None
        best_score: float | None = None
        for candidate_index in range(self.config.candidate_plan_count):
            plan = self.planner.plan(planning_state, candidate_index=candidate_index)
            segment = self.smoother.smooth(state=state, plan=plan)
            score = self._score_candidate(plan=plan, segment=segment, dynamics=planning_state.dynamics_summary)
            candidates.append(
                CandidateEvaluation(
                    name=f"candidate_{candidate_index}",
                    score=score,
                    plan=plan.to_dict(),
                    segment=segment.to_dict(),
                    notes="selected" if best_score is None or score > best_score else "",
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

    def _score_candidate(
        self,
        *,
        plan: TeacherPlan,
        segment: TrajectorySegment,
        dynamics: dict[str, Any],
    ) -> float:
        duration_penalty = segment.expected_duration_s
        caution_penalty = 0.15 if plan.caution_flag else 0.0
        settling_bonus = float(dynamics["cable_settling_score"])
        granularity_bonus = 0.1 if plan.segment_granularity == "guarded" else 0.0
        return settling_bonus + granularity_bonus - duration_penalty - caution_penalty
