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
from ..teacher.visual_context import select_visual_context_items


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
    global_max_calls_per_episode: int = 5
    api_key_env_var: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/responses"
    enabled: bool = False
    use_cache: bool = True
    cache_dir: str | None = None
    include_visual_context: bool = True
    max_recent_visual_images: int = 8
    max_scene_overview_images: int = 4
    max_visual_images_per_episode: int | None = 30
    visual_image_detail: Literal["low", "high", "auto"] = "high"
    enable_global_guidance: bool = False
    global_model: str | None = None
    global_temperature: float | None = 0.1
    global_max_output_tokens: int = 900
    trace_dir: str | None = None


class OpenAIPlannerAPIError(RuntimeError):
    """Responses API request error with sanitized diagnostics."""


@dataclass
class OpenAIPlannerBackend(PlannerBackend):
    config: OpenAIPlannerConfig
    _episode_call_count: int = field(default=0, init=False)
    _global_episode_call_count: int = field(default=0, init=False)
    _episode_visual_image_count: int = field(default=0, init=False)
    _search_budget_key: str | None = field(default=None, init=False)
    _plan_cache: dict[str, TeacherPlan] = field(default_factory=dict, init=False)
    _trace_call_index: int = field(default=0, init=False)

    _search_call_counts: ClassVar[dict[str, int]] = {}

    @property
    def backend_name(self) -> str:
        return "openai"

    def reset_episode_budget(self) -> None:
        self._episode_call_count = 0
        self._global_episode_call_count = 0
        self._episode_visual_image_count = 0
        self._trace_call_index = 0

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
                self._write_trace_record(
                    call_kind="segment_plan",
                    state=state,
                    candidate_index=candidate_index,
                    request_payload=payload,
                    response_payload=response_payload,
                    parsed_output=plan.to_dict(),
                )
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

    def plan_global_guidance(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any] | None:
        if not self.config.enable_global_guidance:
            return None
        if not self.config.enabled:
            return {
                "source": "openai_global_guidance_disabled_backend",
                "model": self.config.global_model or self.config.model,
                "budget_exhausted": False,
            }
        if self._global_episode_call_count >= self.config.global_max_calls_per_episode:
            return {
                "source": "openai_global_guidance_budget_exhausted",
                "model": self.config.global_model or self.config.model,
                "budget_exhausted": True,
            }
        self._require_api_key()
        self._global_episode_call_count += 1
        payload = self.build_global_guidance_request_payload(state, candidate_index=candidate_index)
        response_payload = self._post_responses_request(payload)
        guidance = self._parse_global_guidance_response_payload(response_payload)
        self._write_trace_record(
            call_kind="global_guidance",
            state=state,
            candidate_index=candidate_index,
            request_payload=payload,
            response_payload=response_payload,
            parsed_output=guidance,
        )
        return guidance

    def build_global_guidance_request_payload(
        self,
        state: TeacherPlanningState,
        *,
        candidate_index: int = 0,
    ) -> dict[str, Any]:
        brief = self._planner_state_payload(state=state, candidate_index=candidate_index)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": json.dumps(
                    {
                        "task": state.goal_summary,
                        "candidate_index": candidate_index,
                        "current_phase": state.current_phase,
                        "phase_guidance": state.temporal_context.get("phase_guidance", {}),
                        "global_inputs": {
                            "current_observation": brief.get("current_observation", {}),
                            "scene_geometry_context": brief.get("scene_geometry_context", {}),
                            "obstacle_summary": brief.get("obstacle_summary", []),
                            "temporal_context_summary": brief.get("temporal_context_summary", {}),
                            "signal_quality_context": brief.get("signal_quality_context", {}),
                            "planning_metadata": brief.get("planning_metadata", {}),
                        },
                    },
                    sort_keys=True,
                ),
            }
        ]
        recent_items, scene_items, selection_summary = select_visual_context_items(
            recent_visual_observations=state.recent_visual_observations,
            scene_overview_images=state.scene_overview_images,
            max_recent_visual_images=min(self.config.max_recent_visual_images, 3),
            max_scene_overview_images=min(self.config.max_scene_overview_images, 3),
            episode_remaining_budget=self._remaining_episode_visual_budget(),
        )
        self._episode_visual_image_count += len(recent_items) + len(scene_items)
        for item in recent_items:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Global-guidance recent image label={item.get('label')} "
                        f"camera={item.get('camera_name')} sim_tick={item.get('sim_tick')} "
                        f"sim_time={item.get('sim_time')} timestamp={item.get('timestamp')} "
                        f"age_from_latest_s={item.get('age_from_latest_s')} "
                        f"timepoint_label={item.get('timepoint_label')}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": item.get("image_data_url"),
                    "detail": self.config.visual_image_detail,
                }
            )
        for item in scene_items:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        f"Global-guidance scene overview label={item.get('label')} "
                        f"view={item.get('view_name')} source={item.get('source')}"
                    ),
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": item.get("image_data_url"),
                    "detail": self.config.visual_image_detail,
                }
            )
        content.append(
            {
                "type": "input_text",
                "text": (
                    "Visual selection summary for global guidance: "
                    f"recent_labels={selection_summary.recent_labels} "
                    f"scene_labels={selection_summary.scene_labels}"
                ),
            }
        )
        payload = {
            "model": self.config.global_model or self.config.model,
            "max_output_tokens": self.config.global_max_output_tokens,
            "store": False,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are producing low-frequency global guidance for a cable insertion teacher. "
                                "Do not emit executable low-level actions. Produce only phase-level strategy and a "
                                "small set of milestone waypoints that later local planning can refine."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "teacher_global_guidance",
                    "strict": True,
                    "schema": self.global_guidance_format_schema(),
                },
                "verbosity": self.config.text_verbosity,
            },
        }
        if self.config.global_temperature is not None:
            payload["temperature"] = self.config.global_temperature
        if self.config.reasoning_effort is not None:
            payload["reasoning"] = {"effort": self.config.reasoning_effort}
        return payload

    def _parse_global_guidance_response_payload(self, response_payload: dict[str, Any]) -> dict[str, Any]:
        text = self._extract_output_text(response_payload)
        payload = json.loads(text)
        return {
            "source": "openai_global_guidance",
            "model": self.config.global_model or self.config.model,
            **payload,
        }

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
            "wrench_contact_trend_summary": state.temporal_context.get("wrench_contact_trend_summary", {}),
            "compact_signal_samples": state.temporal_context.get("compact_signal_samples", {}),
            "phase_guidance": state.temporal_context.get("phase_guidance", {}),
            "global_guidance": state.temporal_context.get("global_guidance", {}),
            "auxiliary_contact_summary": state.temporal_context.get("auxiliary_history_summary", {}),
            "frame_context": state.policy_context.get("frame_context", {}),
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
                "tcp_velocity": policy.get("tcp_velocity"),
                "plug_pose": policy.get("plug_pose"),
                "target_port_pose": policy.get("target_port_pose"),
                "target_port_entrance_pose": policy.get("target_port_entrance_pose"),
                "wrench": policy.get("wrench"),
                "wrench_timestamp": policy.get("wrench_timestamp"),
                "distance_to_target": policy.get("distance_to_target"),
                "distance_to_entrance": policy.get("distance_to_entrance"),
                "lateral_misalignment": policy.get("lateral_misalignment"),
                "orientation_error": policy.get("orientation_error"),
                "insertion_progress": policy.get("insertion_progress"),
                "off_limit_contact": policy.get("off_limit_contact"),
                "relative_geometry": policy.get("relative_geometry"),
                "frame_context": policy.get("frame_context"),
                "world_entities_summary": policy.get("world_entities_summary"),
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
                "wrench_contact_trend_summary": state.temporal_context.get("wrench_contact_trend_summary", {}),
                "compact_signal_samples": state.temporal_context.get("compact_signal_samples", {}),
                "auxiliary_history_summary": state.temporal_context.get("auxiliary_history_summary", {}),
                "phase_guidance": state.temporal_context.get("phase_guidance", {}),
                "global_guidance": state.temporal_context.get("global_guidance", {}),
            },
            "signal_quality_context": state.data_quality,
            "recent_probe_results": state.recent_probe_results[-2:],
            "controller_context": {
                "controller_state_available": bool(state.controller_context.get("controller_state")),
                "reference_tcp_pose": state.controller_context.get("reference_tcp_pose"),
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
                        "age_from_latest_s": item.get("age_from_latest_s"),
                        "age_from_latest_steps": item.get("age_from_latest_steps"),
                        "timepoint_label": item.get("timepoint_label"),
                        "source": item.get("source"),
                    }
                    for item in state.recent_visual_observations[: self.config.max_recent_visual_images]
                ],
                "scene_overview_images": [
                    {
                        "label": item.get("label"),
                        "view_name": item.get("view_name"),
                        "timestamp": item.get("timestamp"),
                        "source": item.get("source"),
                    }
                    for item in state.scene_overview_images[: self.config.max_scene_overview_images]
                ],
                "scene_overview_sources": state.planning_metadata.get("scene_overview_sources", {}),
            },
            "planning_metadata": {
                "include_images": state.planning_metadata.get("include_images"),
                "history_window_size": state.planning_metadata.get("history_window_size"),
                "teacher_history_is_additive": state.planning_metadata.get("teacher_history_is_additive"),
                "official_observation_contract_unchanged": state.planning_metadata.get("official_observation_contract_unchanged"),
                "auxiliary_force_contact_summary_is_teacher_side": state.planning_metadata.get("auxiliary_force_contact_summary_is_teacher_side"),
                "frame_context": state.planning_metadata.get("frame_context"),
                "planner_output_mode": state.planning_metadata.get("planner_output_mode"),
                "scene_overview_sources": state.planning_metadata.get("scene_overview_sources"),
                "available_scene_overview_views": state.planning_metadata.get("available_scene_overview_views"),
                "scene_overview_live_source_used": state.planning_metadata.get("scene_overview_live_source_used"),
                "prefer_live_scene_overview": state.planning_metadata.get("prefer_live_scene_overview"),
                "global_guidance": state.planning_metadata.get("global_guidance"),
            },
            "last_teacher_rationale": state.last_teacher_rationale,
        }

    def _visual_content(self, state: TeacherPlanningState) -> list[dict[str, Any]]:
        if not self.config.include_visual_context:
            return []
        content: list[dict[str, Any]] = []
        remaining_budget = self._remaining_episode_visual_budget()
        recent_items, scene_items, selection_summary = select_visual_context_items(
            recent_visual_observations=state.recent_visual_observations,
            scene_overview_images=state.scene_overview_images,
            max_recent_visual_images=self.config.max_recent_visual_images,
            max_scene_overview_images=self.config.max_scene_overview_images,
            episode_remaining_budget=remaining_budget,
        )
        selected_count = len(recent_items) + len(scene_items)
        self._episode_visual_image_count += selected_count
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
                            f"timestamp={item.get('timestamp')} "
                            f"age_from_latest_s={item.get('age_from_latest_s')} "
                            f"age_from_latest_steps={item.get('age_from_latest_steps')} "
                            f"timepoint_label={item.get('timepoint_label')} "
                            f"source={item.get('source')}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": item.get("image_data_url"),
                        "detail": self.config.visual_image_detail,
                    }
                )
        if scene_items:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "Teacher-side scene overview images follow. Some views may come from fixed live Gazebo "
                        "overview cameras and others may fall back to teacher-side schematic renders. Treat them "
                        "as auxiliary global-scene context rather than official wrist observations."
                    ),
                }
            )
            for item in scene_items:
                content.append(
                    {
                        "type": "input_text",
                        "text": (
                            f"Scene overview label={item.get('label')} view={item.get('view_name')} "
                            f"timestamp={item.get('timestamp')} source={item.get('source')}"
                        ),
                    }
                )
                content.append(
                    {
                        "type": "input_image",
                        "image_url": item.get("image_data_url"),
                        "detail": self.config.visual_image_detail,
                    }
                )
        if selected_count:
            content.append(
                {
                    "type": "input_text",
                    "text": (
                        "Visual selection summary: "
                        f"recent_labels={selection_summary.recent_labels} "
                        f"scene_labels={selection_summary.scene_labels} "
                        f"remaining_episode_budget={selection_summary.remaining_episode_budget}"
                    ),
                }
            )
        return content

    def _remaining_episode_visual_budget(self) -> int | None:
        if self.config.max_visual_images_per_episode is None:
            return None
        return max(self.config.max_visual_images_per_episode - self._episode_visual_image_count, 0)

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

    def _write_trace_record(
        self,
        *,
        call_kind: str,
        state: TeacherPlanningState,
        candidate_index: int,
        request_payload: dict[str, Any],
        response_payload: dict[str, Any],
        parsed_output: dict[str, Any],
    ) -> None:
        if not self.config.trace_dir:
            return
        trace_dir = Path(self.config.trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        self._trace_call_index += 1
        text_output = None
        try:
            text_output = self._extract_output_text(response_payload)
        except Exception:
            text_output = None
        record = {
            "call_index": self._trace_call_index,
            "call_kind": call_kind,
            "candidate_index": candidate_index,
            "trial_id": state.trial_id,
            "task_id": state.task_id,
            "current_phase": state.current_phase,
            "request_payload": sanitize_payload(request_payload),
            "response_payload": sanitize_payload(response_payload),
            "response_text": text_output,
            "parsed_output": parsed_output,
            "planning_state_summary": {
                "goal_summary": state.goal_summary,
                "current_phase": state.current_phase,
                "policy_context_keys": sorted(state.policy_context.keys()),
                "temporal_context_keys": sorted(state.temporal_context.keys()),
                "scene_overview_sources": state.planning_metadata.get("scene_overview_sources", {}),
            },
        }
        path = trace_dir / f"{self._trace_call_index:03d}_{call_kind}_candidate{candidate_index}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

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

    @staticmethod
    def global_guidance_format_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "strategy_summary": {"type": "string", "minLength": 1, "maxLength": 400},
                "phase_sequence": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
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
                },
                "milestones": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "phase": {
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
                            "position_xyz": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "yaw": {"type": "number"},
                            "notes": {"type": "string", "minLength": 1, "maxLength": 240},
                        },
                        "required": ["name", "phase", "position_xyz", "yaw", "notes"],
                    },
                },
                "risks": {
                    "type": "array",
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
            },
            "required": ["strategy_summary", "phase_sequence", "milestones", "risks"],
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
