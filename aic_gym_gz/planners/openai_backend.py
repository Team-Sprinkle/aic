"""OpenAI planner backend for teacher segment planning.

This backend uses the OpenAI Responses API with strict JSON Schema output. The
API key is read only from the configured environment variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, ClassVar, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .base import PlannerBackend
from ..teacher.planning import candidate_family_for_index
from ..teacher.types import TeacherPlan, TeacherPlanningState, TeacherWaypoint


class _WaypointPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    position_xyz: tuple[float, float, float]
    yaw: float = 0.0
    speed_scale: float = Field(default=1.0, ge=0.0, le=1.5)
    clearance_hint: float = Field(default=0.0, ge=0.0, le=0.25)


class _PlanPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    next_phase: Literal[
        "free_space_approach",
        "obstacle_avoidance",
        "cable_probe",
        "pre_insert_align",
        "guarded_insert",
        "backoff_and_retry",
    ]
    waypoints: list[_WaypointPayload] = Field(min_length=1, max_length=4)
    motion_mode: Literal["coarse_cartesian", "fine_cartesian", "guarded_insert", "hold"]
    caution_flag: bool
    should_probe: bool
    segment_horizon_steps: int = Field(ge=1, le=64)
    segment_granularity: Literal["coarse", "fine", "guarded"]
    rationale_summary: str = Field(min_length=1, max_length=400)


@dataclass(frozen=True)
class OpenAIPlannerConfig:
    model: str = "gpt-5.4-mini"
    temperature: float | None = 0.1
    max_output_tokens: int = 1200
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = "low"
    text_verbosity: Literal["low", "medium", "high"] = "low"
    timeout_s: float = 20.0
    max_retries: int = 2
    max_calls_per_episode: int = 8
    max_calls_per_search: int = 64
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/responses"
    enabled: bool = False
    use_cache: bool = True
    cache_dir: str | None = None
    include_visual_context: bool = True
    max_recent_visual_images: int = 6
    max_scene_overview_images: int = 3


class OpenAIPlannerAPIError(RuntimeError):
    """Responses API request error with sanitized diagnostics."""


@dataclass
class OpenAIPlannerBackend(PlannerBackend):
    config: OpenAIPlannerConfig
    _episode_call_count: int = field(default=0, init=False)
    _search_budget_key: str | None = field(default=None, init=False)
    _plan_cache: dict[str, TeacherPlan] = field(default_factory=dict, init=False)

    _search_call_counts: ClassVar[dict[str, int]] = {}

    @property
    def backend_name(self) -> str:
        return "openai"

    def reset_episode_budget(self) -> None:
        self._episode_call_count = 0

    def set_search_budget_key(self, key: str) -> None:
        self._search_budget_key = key

    @classmethod
    def reset_search_budget(cls, key: str) -> None:
        cls._search_call_counts[key] = 0

    def plan(self, state: TeacherPlanningState, *, candidate_index: int = 0) -> TeacherPlan:
        if not self.config.enabled:
            raise RuntimeError("OpenAI planner backend is disabled. Enable it in config before use.")
        self._require_api_key()
        cache_key = self._cache_key(state=state, candidate_index=candidate_index)
        cached = self._load_cached_plan(cache_key)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                self._consume_budget()
                payload = self.build_request_payload(state, candidate_index=candidate_index)
                response_payload = self._post_responses_request(payload)
                plan = self._parse_response_payload(response_payload)
                self._store_cached_plan(cache_key, plan)
                return plan
            except (
                OpenAIPlannerAPIError,
                RuntimeError,
                ValidationError,
                json.JSONDecodeError,
                URLError,
                TimeoutError,
                socket.timeout,
            ) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
        raise RuntimeError(f"OpenAI planner request failed after retries: {last_error}") from last_error

    def build_request_payload(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any]:
        user_content = [
            {
                "type": "input_text",
                "text": json.dumps(
                    self._planner_state_payload(
                        state=state,
                        candidate_index=candidate_index,
                    ),
                    sort_keys=True,
                ),
            }
        ]
        user_content.extend(self._visual_content(state))
        payload = {
            "model": self.config.model,
            "max_output_tokens": self.config.max_output_tokens,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are planning one short, conservative teacher segment for a cable insertion "
                                "task. Use only the provided planning state. Respect data_quality metadata: if "
                                "signals are missing or approximate, remain conservative and do not pretend they "
                                "are official. Prefer compact, useful segments that reflect current phase, score "
                                "geometry, temporal summary, and insertion progress. Candidate_index is intentional: "
                                "each candidate must represent a distinct family with different phase preference, "
                                "clearance behavior, or insertion aggressiveness. Do not emit near-duplicate plans. "
                                "If the segment would be long, emit only the next short segment that makes progress "
                                "toward the phase guidance instead of a whole-episode path. Do not stay in one phase "
                                "indefinitely when the phase guidance recommends advancement."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "teacher_segment_plan",
                    "strict": True,
                    "schema": self.response_format_schema(),
                },
                "verbosity": self.config.text_verbosity,
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        return payload

    def build_debug_payload(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any]:
        return sanitize_payload(self.build_request_payload(state, candidate_index=candidate_index))

    def _compact_planning_brief(
        self,
        *,
        state: TeacherPlanningState,
        candidate_index: int,
    ) -> dict[str, Any]:
        policy = state.policy_context
        score_geometry = policy.get("score_geometry", {})
        candidate_family = candidate_family_for_index(candidate_index)
        return {
            "task": state.goal_summary,
            "current_phase": state.current_phase,
            "candidate_index": candidate_index,
            "candidate_guidance": candidate_family["planner_instruction"],
            "candidate_family": candidate_family,
            "distance_to_target": policy.get("distance_to_target"),
            "distance_to_entrance": score_geometry.get("distance_to_entrance"),
            "insertion_progress": score_geometry.get("insertion_progress"),
            "partial_insertion": score_geometry.get("partial_insertion"),
            "lateral_misalignment": score_geometry.get("lateral_misalignment"),
            "orientation_error": score_geometry.get("orientation_error"),
            "temporal_summary": state.temporal_context.get("dynamics_summary", {}),
            "geometry_progress_summary": state.temporal_context.get("geometry_progress_summary", {}),
            "phase_guidance": state.temporal_context.get("phase_guidance", {}),
            "auxiliary_contact_summary": state.temporal_context.get("auxiliary_history_summary", {}),
            "signal_quality_summary": {
                signal: {
                    "is_real": quality.get("is_real"),
                    "is_missing": quality.get("is_missing"),
                    "source": quality.get("source"),
                }
                for signal, quality in state.data_quality.items()
            },
        }

    def _planner_state_payload(
        self,
        *,
        state: TeacherPlanningState,
        candidate_index: int,
    ) -> dict[str, Any]:
        policy = state.policy_context
        return {
            "candidate_index": candidate_index,
            "compact_planning_brief": self._compact_planning_brief(
                state=state,
                candidate_index=candidate_index,
            ),
            "current_observation": {
                "tcp_pose": policy.get("tcp_pose"),
                "plug_pose": policy.get("plug_pose"),
                "target_port_pose": policy.get("target_port_pose"),
                "target_port_entrance_pose": policy.get("target_port_entrance_pose"),
                "distance_to_target": policy.get("distance_to_target"),
                "distance_to_entrance": policy.get("distance_to_entrance"),
                "lateral_misalignment": policy.get("lateral_misalignment"),
                "orientation_error": policy.get("orientation_error"),
                "insertion_progress": policy.get("insertion_progress"),
                "off_limit_contact": policy.get("off_limit_contact"),
            },
            "obstacle_summary": state.obstacle_summary[:6],
            "scene_geometry_context": {
                "board_pose_xyz_rpy": state.oracle_context.get("task_board_pose_xyz_rpy"),
                "target_port_pose": state.oracle_context.get("target_port_pose"),
                "target_port_entrance_pose": state.oracle_context.get("target_port_entrance_pose"),
                "plug_pose": state.oracle_context.get("plug_pose"),
                "clearance_summary": state.oracle_context.get("clearance_summary"),
                "scene_layout_summary": state.oracle_context.get("scene_layout_summary"),
                "cable_context": state.oracle_context.get("cable"),
            },
            "temporal_context_summary": {
                "dynamics_summary": state.dynamics_summary,
                "geometry_progress_summary": state.temporal_context.get("geometry_progress_summary", {}),
                "auxiliary_history_summary": state.temporal_context.get("auxiliary_history_summary", {}),
                "phase_guidance": state.temporal_context.get("phase_guidance", {}),
            },
            "signal_quality_context": state.data_quality,
            "recent_probe_results": state.recent_probe_results[-2:],
            "controller_context": {
                "controller_state_available": bool(state.controller_context.get("controller_state")),
                "tcp_error": state.controller_context.get("tcp_error"),
                "controller_target_mode": state.controller_context.get("controller_target_mode"),
            },
            "visual_context_summary": {
                "recent_visual_observations": [
                    {
                        "label": item.get("label"),
                        "camera_name": item.get("camera_name"),
                        "sim_tick": item.get("sim_tick"),
                        "sim_time": item.get("sim_time"),
                        "timestamp": item.get("timestamp"),
                        "source": item.get("source"),
                    }
                    for item in state.recent_visual_observations[: self.config.max_recent_visual_images]
                ],
                "scene_overview_images": [
                    {
                        "label": item.get("label"),
                        "view_name": item.get("view_name"),
                        "source": item.get("source"),
                    }
                    for item in state.scene_overview_images[: self.config.max_scene_overview_images]
                ],
            },
            "planning_metadata": {
                "include_images": state.planning_metadata.get("include_images"),
                "history_window_size": state.planning_metadata.get("history_window_size"),
                "teacher_history_is_additive": state.planning_metadata.get("teacher_history_is_additive"),
                "official_observation_contract_unchanged": state.planning_metadata.get("official_observation_contract_unchanged"),
                "auxiliary_force_contact_summary_is_teacher_side": state.planning_metadata.get("auxiliary_force_contact_summary_is_teacher_side"),
            },
            "last_teacher_rationale": state.last_teacher_rationale,
        }

    def _visual_content(self, state: TeacherPlanningState) -> list[dict[str, Any]]:
        if not self.config.include_visual_context:
            return []
        content: list[dict[str, Any]] = []
        recent_items = state.recent_visual_observations[: self.config.max_recent_visual_images]
        if recent_items:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "Recent official wrist-camera observations follow. These are official image observations "
                        "when image mode is enabled; use them to reason about local contact geometry, cable pose, "
                        "and corridor alignment over the last few steps."
                    ),
                }
            )
            for item in recent_items:
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"Image label={item.get('label')} camera={item.get('camera_name')} "
                            f"sim_tick={item.get('sim_tick')} sim_time={item.get('sim_time')} "
                            f"timestamp={item.get('timestamp')} source={item.get('source')}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": item.get("image_data_url"),
                        "detail": "low",
                    }
                )
        scene_items = state.scene_overview_images[: self.config.max_scene_overview_images]
        if scene_items:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "Teacher-side scene overview renders follow. These are multi-angle schematic views "
                        "derived from Gazebo/scenario geometry, not official participant observations. Use them "
                        "for global scene layout and obstacle-awareness only."
                    ),
                }
            )
            for item in scene_items:
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"Scene overview label={item.get('label')} view={item.get('view_name')} "
                            f"source={item.get('source')}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": item.get("image_data_url"),
                        "detail": "low",
                    }
                )
        return content

    def build_smoke_test_payload(self, *, prompt: str = "Return a short valid planner response.") -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "max_output_tokens": min(max(self.config.max_output_tokens, 300), 1200),
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Return valid JSON matching the provided schema."}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "teacher_segment_plan",
                    "strict": True,
                    "schema": self.response_format_schema(),
                },
                "verbosity": self.config.text_verbosity,
            },
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        return payload

    def run_smoke_test(self, *, prompt: str = "Return a cautious pre-insert plan.") -> dict[str, Any]:
        payload = self.build_smoke_test_payload(prompt=prompt)
        response_payload = self._post_responses_request(payload)
        plan = self._parse_response_payload(response_payload)
        return {
            "payload": sanitize_payload(payload),
            "response": sanitize_payload(response_payload),
            "plan": plan.to_dict(),
        }

    def _parse_response_payload(self, response_payload: dict[str, Any]) -> TeacherPlan:
        refusal = response_payload.get("refusal")
        if refusal:
            raise RuntimeError(f"OpenAI planner refused the request: {refusal}")
        text = self._extract_output_text(response_payload)
        payload = _PlanPayload.model_validate_json(text)
        return TeacherPlan(
            next_phase=payload.next_phase,
            waypoints=tuple(
                TeacherWaypoint(
                    position_xyz=tuple(float(axis) for axis in waypoint.position_xyz),
                    yaw=float(waypoint.yaw),
                    speed_scale=float(waypoint.speed_scale),
                    clearance_hint=float(waypoint.clearance_hint),
                )
                for waypoint in payload.waypoints
            ),
            motion_mode=payload.motion_mode,
            caution_flag=bool(payload.caution_flag),
            should_probe=bool(payload.should_probe),
            segment_horizon_steps=int(payload.segment_horizon_steps),
            segment_granularity=payload.segment_granularity,
            rationale_summary=payload.rationale_summary.strip(),
        )

    def _post_responses_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(payload).encode("utf-8")
        request = Request(
            self.config.base_url,
            data=encoded,
            headers={
                "Authorization": f"Bearer {self._require_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body_text = _http_error_body(exc)
            raise OpenAIPlannerAPIError(
                self._format_http_error_message(
                    status=exc.code,
                    body_text=body_text,
                    payload=payload,
                )
            ) from exc

    def _format_http_error_message(
        self,
        *,
        status: int,
        body_text: str,
        payload: dict[str, Any],
    ) -> str:
        return (
            "OpenAI Responses API request failed "
            f"(status={status}, model={self.config.model}, "
            f"structured_output_requested={bool(payload.get('text', {}).get('format'))}). "
            f"Response body: {body_text}"
        )

    def _extract_output_text(self, response_payload: dict[str, Any]) -> str:
        if isinstance(response_payload.get("output_text"), str) and response_payload["output_text"].strip():
            return str(response_payload["output_text"])
        texts: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                node_type = node.get("type")
                if node_type in {"output_text", "text"} and isinstance(node.get("text"), str):
                    texts.append(node["text"])
                    return
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(response_payload.get("output", []))
        if not texts:
            raise RuntimeError("OpenAI planner response did not contain any text output.")
        return "\n".join(texts).strip()

    def _cache_key(self, *, state: TeacherPlanningState, candidate_index: int) -> str:
        normalized = json.dumps(
            {
                "model": self.config.model,
                "temperature": self.config.temperature,
                "candidate_index": candidate_index,
                "planning_state": state.to_dict(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_cached_plan(self, cache_key: str) -> TeacherPlan | None:
        if cache_key in self._plan_cache:
            return self._plan_cache[cache_key]
        if not self.config.use_cache or not self.config.cache_dir:
            return None
        path = Path(self.config.cache_dir) / f"{cache_key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = self._parse_response_payload({"output_text": json.dumps(payload)})
        self._plan_cache[cache_key] = plan
        return plan

    def _store_cached_plan(self, cache_key: str, plan: TeacherPlan) -> None:
        if not self.config.use_cache:
            return
        self._plan_cache[cache_key] = plan
        if not self.config.cache_dir:
            return
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"{cache_key}.json").write_text(
            json.dumps(plan.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _consume_budget(self) -> None:
        if self._episode_call_count >= self.config.max_calls_per_episode:
            raise RuntimeError(
                f"OpenAI planner exceeded max_calls_per_episode={self.config.max_calls_per_episode}."
            )
        if self._search_budget_key is not None:
            used = self._search_call_counts.get(self._search_budget_key, 0)
            if used >= self.config.max_calls_per_search:
                raise RuntimeError(
                    f"OpenAI planner exceeded max_calls_per_search={self.config.max_calls_per_search}."
                )
            self._search_call_counts[self._search_budget_key] = used + 1
        self._episode_call_count += 1

    def _require_api_key(self) -> str:
        api_key = os.environ.get(self.config.api_key_env_var)
        if not api_key:
            raise RuntimeError(
                f"Missing OpenAI API key in environment variable {self.config.api_key_env_var}."
            )
        return api_key

    @staticmethod
    def response_format_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "next_phase": {
                    "type": "string",
                    "enum": [
                        "free_space_approach",
                        "obstacle_avoidance",
                        "cable_probe",
                        "pre_insert_align",
                        "guarded_insert",
                        "backoff_and_retry",
                    ],
                },
                "waypoints": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "position_xyz": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "yaw": {"type": "number"},
                            "speed_scale": {"type": "number"},
                            "clearance_hint": {"type": "number"},
                        },
                        "required": [
                            "position_xyz",
                            "yaw",
                            "speed_scale",
                            "clearance_hint",
                        ],
                    },
                },
                "motion_mode": {
                    "type": "string",
                    "enum": ["coarse_cartesian", "fine_cartesian", "guarded_insert", "hold"],
                },
                "caution_flag": {"type": "boolean"},
                "should_probe": {"type": "boolean"},
                "segment_horizon_steps": {"type": "integer", "minimum": 1, "maximum": 64},
                "segment_granularity": {
                    "type": "string",
                    "enum": ["coarse", "fine", "guarded"],
                },
                "rationale_summary": {"type": "string", "minLength": 1, "maxLength": 400},
            },
            "required": [
                "next_phase",
                "waypoints",
                "motion_mode",
                "caution_flag",
                "should_probe",
                "segment_horizon_steps",
                "segment_granularity",
                "rationale_summary",
            ],
        }


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            lower_key = str(key).lower()
            if lower_key in {"authorization", "api_key", "apikey"}:
                sanitized[key] = "<redacted>"
            elif lower_key == "image_url" and isinstance(value, str):
                sanitized[key] = "<image_data_url>"
            else:
                sanitized[key] = sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    return payload


def _http_error_body(error: HTTPError) -> str:
    try:
        raw = error.read()
    except Exception:
        return "<unable to read error body>"
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return repr(raw)
