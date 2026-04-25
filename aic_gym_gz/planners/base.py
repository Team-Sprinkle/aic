"""Planner backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..teacher.types import TeacherPlan, TeacherPlanningState


class PlannerBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable backend identifier."""

    @abstractmethod
    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        """Return a structured segment plan."""

    def plan_global_guidance(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any] | None:
        """Return optional low-frequency global guidance metadata."""
        del state, candidate_index
        return None

    def remaining_episode_plan_calls(self) -> int | None:
        """Return remaining low-level plan calls, or None when unlimited/unknown."""
        return None
