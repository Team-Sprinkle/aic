"""Audit the current planner payload and optionally ask the VLM what is missing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig, sanitize_payload
from aic_gym_gz.teacher.context import TeacherContextExtractor
from aic_gym_gz.teacher.history import TemporalObservationBuffer
from aic_gym_gz.teacher.quality import build_signal_quality_snapshot
from aic_gym_gz.utils import to_jsonable


def _image_timestamp_map(observation: dict[str, Any]) -> dict[str, float]:
    if "images" not in observation or "image_timestamps" not in observation:
        return {}
    timestamps = np.asarray(observation["image_timestamps"], dtype=np.float64)
    names = sorted(observation["images"].keys())
    return {name: float(timestamps[index]) for index, name in enumerate(names)}


def _payload_summary(planning_state: Any, request_payload: dict[str, Any]) -> dict[str, Any]:
    policy = dict(planning_state.policy_context)
    temporal = dict(planning_state.temporal_context)
    planner_state_payload = json.loads(request_payload["input"][1]["content"][0]["text"])
    return {
        "current_observation_fields_sent": sorted(
            planner_state_payload.get("current_observation", {}).keys()
        ),
        "scene_geometry_fields_sent": sorted(
            planner_state_payload.get("scene_geometry_context", {}).keys()
        ),
        "temporal_context_fields_sent": sorted(
            planner_state_payload.get("temporal_context_summary", {}).keys()
        ),
        "controller_fields_sent": sorted(
            planner_state_payload.get("controller_context", {}).keys()
        ),
        "has_recent_visual_observations": bool(planning_state.recent_visual_observations),
        "recent_visual_observation_count": len(planning_state.recent_visual_observations),
        "recent_visual_labels": [item.get("label") for item in planning_state.recent_visual_observations],
        "scene_overview_count": len(planning_state.scene_overview_images),
        "scene_overview_labels": [item.get("label") for item in planning_state.scene_overview_images],
        "history_window_size": int(temporal.get("window_size", 0)),
        "history_fields": sorted(temporal.keys()),
        "policy_context_fields": sorted(policy.keys()),
        "controller_context_fields": sorted(planning_state.controller_context.keys()),
        "camera_context_fields": sorted(planning_state.camera_context.keys()),
        "data_quality_fields": sorted(planning_state.data_quality.keys()),
        "obstacle_count": len(planning_state.obstacle_summary),
        "recent_probe_result_count": len(planning_state.recent_probe_results),
        "visual_input_item_count": sum(1 for item in request_payload["input"][1]["content"] if item.get("type") == "input_image"),
    }


def _audit_request_payload(
    *,
    backend: OpenAIPlannerBackend,
    model: str,
    planning_state: Any,
    planner_payload: dict[str, Any],
    context_summary: dict[str, Any],
) -> dict[str, Any]:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "overall_assessment": {"type": "string"},
            "missing_context": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "importance": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                        "reason": {"type": "string"},
                        "suggested_source": {"type": "string"},
                    },
                    "required": ["name", "importance", "reason", "suggested_source"],
                },
            },
            "additional_images": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "importance": {"type": "string", "enum": ["critical", "high", "medium", "low"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "importance", "reason"],
                },
            },
            "temporal_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "reason"],
                },
            },
            "geometry_and_frames": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "reason"],
                },
            },
            "ambiguities": {"type": "array", "items": {"type": "string"}},
            "downsample_or_remove": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "action": {"type": "string", "enum": ["keep", "downsample", "remove"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["name", "action", "reason"],
                },
            },
        },
        "required": [
            "overall_assessment",
            "missing_context",
            "additional_images",
            "temporal_summaries",
            "geometry_and_frames",
            "ambiguities",
            "downsample_or_remove",
        ],
    }
    prompt = {
        "planning_state": planning_state.to_dict(),
        "planner_payload_summary": context_summary,
        "planner_request_payload": sanitize_payload(planner_payload),
        "task": (
            "Do not plan. Audit the context only. Identify what crucial context is missing for good planning, "
            "which additional images would help, what temporal summaries would help, what geometry/frame/"
            "obstacle/controller information would help, what ambiguity remains, and what should be downsampled "
            "or removed to stay in a practical budget."
        ),
    }
    payload = {
        "model": model,
        "max_output_tokens": 1800,
        "store": False,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are auditing missing context in a robotics VLM planner payload. "
                            "Do not suggest policy changes unless they are directly about missing planner context. "
                            "Be specific about geometry, frames, controller semantics, observability gaps, and budget tradeoffs."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(prompt, sort_keys=True),
                    }
                ],
            },
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "missing_planner_context_audit",
                "strict": True,
                "schema": schema,
            },
            "verbosity": "low",
        },
    }
    if backend.config.reasoning_effort is not None:
        payload["reasoning"] = {"effort": backend.config.reasoning_effort}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="aic_gym_gz/artifacts/context_audit/missing_context_audit.json")
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument("--dump-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prefer-live-scene-overview", action="store_true")
    args = parser.parse_args()

    if args.include_images and not args.live:
        raise RuntimeError(
            "Real image context requires --live. "
            "The default mock env does not provide real wrist-camera frames."
        )
    env = (
        make_live_env(enable_randomization=False, include_images=args.include_images)
        if args.live
        else make_default_env(enable_randomization=True, include_images=False)
    )
    try:
        observation, _ = env.reset(seed=args.seed)
        assert env._scenario is not None
        assert env._state is not None
        task_id = next(iter(env._scenario.tasks.keys()))
        history = TemporalObservationBuffer()
        history.append(
            state=env._state,
            action=np.zeros(6, dtype=np.float32),
            images=observation.get("images"),
            image_timestamps=_image_timestamp_map(observation),
            camera_info=observation.get("camera_info"),
            signal_quality=build_signal_quality_snapshot(
                env._state,
                include_images=args.include_images,
                camera_info=observation.get("camera_info"),
            ),
        )
        planning_state = TeacherContextExtractor(
            prefer_live_scene_overview=args.prefer_live_scene_overview
        ).build_planning_state(
            scenario=env._scenario,
            task_id=task_id,
            state=env._state,
            temporal_buffer=history,
            current_phase="free_space_approach",
            recent_probe_results=[],
            include_images=args.include_images,
        )
    finally:
        env.close()

    backend = OpenAIPlannerBackend(
        OpenAIPlannerConfig(
            enabled=not args.dump_only,
            model=args.model,
        )
    )
    planner_payload = backend.build_request_payload(planning_state, candidate_index=0)
    context_summary = _payload_summary(planning_state, planner_payload)
    result = {
        "planning_state": planning_state.to_dict(),
        "current_payload_snapshot": sanitize_payload(planner_payload),
        "context_summary": context_summary,
        "vlm_response": None,
    }
    if not args.dump_only:
        try:
            audit_payload = _audit_request_payload(
                backend=backend,
                model=args.model,
                planning_state=planning_state,
                planner_payload=planner_payload,
                context_summary=context_summary,
            )
            response_payload = backend._post_responses_request(audit_payload)
            result["audit_request_payload"] = sanitize_payload(audit_payload)
            result["audit_response_payload"] = sanitize_payload(response_payload)
            result["vlm_response"] = json.loads(backend._extract_output_text(response_payload))
        except Exception as exc:
            result["vlm_error"] = str(exc)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(to_jsonable(result), indent=2 if args.pretty else None, sort_keys=True)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
