"""Planner backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..teacher.types import TeacherPlan, TeacherPlanningState


class PlannerBackend(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Stable backend identifier."""

    @abstractmethod
    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        """Return a structured segment plan."""
