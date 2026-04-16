"""OpenAI planner backend scaffold.

This module keeps all secret handling outside the repo. The API key is expected
to be supplied at runtime through the `OPENAI_API_KEY` environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any

from .base import PlannerBackend
from ..teacher.types import TeacherPlan, TeacherPlanningState


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    model: str = "gpt-5.4-mini"
    temperature: float = 0.1
    max_output_tokens: int = 1200
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str | None = None
    enabled: bool = False


@dataclass
class OpenAIPlannerBackend(PlannerBackend):
    config: OpenAIPlannerConfig

    @property
    def backend_name(self) -> str:
        return "openai"

    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        if not self.config.enabled:
            raise RuntimeError(
                "OpenAI planner backend is disabled. Enable it in config before use."
            )
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key in environment variable {self.config.api_key_env_var}."
            )
        raise NotImplementedError(
            "OpenAI planner execution is intentionally left as a scaffold. "
            "Use `build_request_payload()` to serialize state and map the JSON "
            "response back into `TeacherPlan` in a future integration."
        )

    def build_request_payload(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any]:
        return {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "candidate_index": candidate_index,
            "planning_state_json": json.dumps(state.__dict__, sort_keys=True),
            "instructions": (
                "Return a compact JSON object with fields: next_phase, waypoints, motion_mode, "
                "caution_flag, should_probe, segment_horizon_steps, segment_granularity, rationale_summary."
            ),
        }
