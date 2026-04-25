"""Post-rollout GPT-5 visual feedback for live teacher evaluation."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .utils import to_jsonable


VIEWPOINT_DESCRIPTIONS: dict[str, str] = {
    "camera_left": "left wrist camera, close-range gripper/plug view",
    "camera_center": "center wrist camera, close-range forward view along tool",
    "camera_right": "right wrist camera, close-range gripper/plug view",
    "overview_top_down_xy": "Gazebo fixed overview, top-down XY world-plane view",
    "overview_front_xz": "Gazebo fixed overview, front XZ world-plane view",
    "overview_side_yz": "Gazebo fixed overview, side YZ world-plane view",
    "overview_oblique_xy": "Gazebo fixed overview, oblique scene view",
}


@dataclass(frozen=True)
class FrameSample:
    stream_name: str
    video_path: str
    sample_index: int
    sample_count_for_stream: int
    frame_index: int
    frame_count: int
    timestamp_s: float
    viewpoint_description: str
    image_data_url: str
    diagnostic_image_path: str


class VlmFeedbackError(RuntimeError):
    pass


def run_final_gpt5_feedback(
    *,
    run_dir: Path,
    model: str = "gpt-5",
    api_key_env_var: str = "OPENAI_API_KEY",
    base_url: str = "https://api.openai.com/v1/responses",
    timeout_s: float = 90.0,
    max_output_tokens: int = 4000,
    frames_per_angle: int = 10,
) -> dict[str, Any]:
    artifact_path = run_dir / "teacher_rollout_artifact.json"
    analysis_path = run_dir / "rollout_analysis.json"
    video_dir = run_dir / "videos"
    if not artifact_path.exists():
        raise VlmFeedbackError(f"Missing rollout artifact: {artifact_path}")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    step_logs = list(artifact.get("step_logs", []))
    final_step = step_logs[-1] if step_logs else {}
    final_observation = dict(final_step.get("observation_summary", {}))
    final_info = dict(artifact.get("final_info", {}))
    frame_samples = collect_final_frame_samples(
        video_dir=video_dir,
        output_dir=run_dir / "vlm_feedback_frames",
        final_observation=final_observation,
        final_info=final_info,
        frames_per_angle=frames_per_angle,
    )
    payload = build_feedback_request_payload(
        model=model,
        frame_samples=frame_samples,
        artifact=artifact,
        analysis=analysis,
        final_observation=final_observation,
        max_output_tokens=max_output_tokens,
    )
    response_payload = _post_responses_request(
        payload,
        api_key=os.environ.get(api_key_env_var),
        base_url=base_url,
        timeout_s=timeout_s,
    )
    parse_error = None
    try:
        parsed = _parse_feedback_response(response_payload)
    except Exception as exc:
        parse_error = str(exc)
        parsed = {
            "failure_reason": "",
            "trajectory_quality_assessment": "",
            "coordinate_frame_concerns": [],
            "missing_context_or_tools": [],
            "recommended_visual_aids": [],
            "recommended_controller_changes": [],
            "is_context_sufficient_for_high_quality_trajectory": False,
            "satisfaction_level": "not_satisfied",
            "next_iteration_priority": "GPT-5 feedback response was not parseable; inspect response_payload.",
        }
    result = {
        "model": model,
        "request_frame_count": len(frame_samples),
        "frames": [sample.__dict__ for sample in frame_samples],
        "feedback": parsed,
        "parse_error": parse_error,
        "response_payload": _sanitize_payload(response_payload),
        "request_payload": _sanitize_payload(payload),
    }
    feedback_path = run_dir / "final_gpt5_vlm_feedback.json"
    feedback_path.write_text(json.dumps(to_jsonable(result), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "path": str(feedback_path),
        "model": model,
        "request_frame_count": len(frame_samples),
        "feedback": parsed,
        "parse_error": parse_error,
        "frames": [sample.__dict__ for sample in frame_samples],
    }


def collect_final_frame_samples(
    *,
    video_dir: Path,
    output_dir: Path,
    final_observation: dict[str, Any],
    final_info: dict[str, Any],
    frames_per_angle: int = 10,
) -> list[FrameSample]:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples: list[FrameSample] = []
    for stream_name, description in VIEWPOINT_DESCRIPTIONS.items():
        video_path = video_dir / f"{stream_name}.mp4"
        if not video_path.exists():
            continue
        frames = _read_evenly_spaced_video_frames(
            video_path,
            sample_count=max(int(frames_per_angle), 1),
        )
        if not frames:
            continue
        sample_count_for_stream = len(frames)
        for sample_index, (frame, frame_index, frame_count, fps) in enumerate(frames):
            timestamp_s = float(frame_index / fps) if fps > 0.0 else float(final_observation.get("sim_time") or 0.0)
            annotated = _annotate_feedback_frame(
                frame=frame,
                stream_name=stream_name,
                viewpoint_description=description,
                timestamp_s=timestamp_s,
                sample_index=sample_index,
                sample_count_for_stream=sample_count_for_stream,
                final_observation=final_observation,
                final_info=final_info,
            )
            image_path = output_dir / f"{stream_name}_sample_{sample_index:02d}_frame_{frame_index:06d}.png"
            Image.fromarray(annotated).save(image_path)
            samples.append(
                FrameSample(
                    stream_name=stream_name,
                    video_path=str(video_path),
                    sample_index=int(sample_index),
                    sample_count_for_stream=int(sample_count_for_stream),
                    frame_index=int(frame_index),
                    frame_count=int(frame_count),
                    timestamp_s=timestamp_s,
                    viewpoint_description=description,
                    image_data_url=_encode_image_data_url(annotated),
                    diagnostic_image_path=str(image_path),
                )
            )
    return samples


def build_feedback_request_payload(
    *,
    model: str,
    frame_samples: list[FrameSample],
    artifact: dict[str, Any],
    analysis: dict[str, Any],
    final_observation: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, Any]:
    final_info = dict(artifact.get("final_info", {}))
    run_summary = {
        "final_info": final_info,
        "analysis_scores": analysis.get("scores", {}),
        "analysis_outcome": analysis.get("outcome", {}),
        "counts": analysis.get("counts", {}),
        "final_observation": {
            "sim_time": final_observation.get("sim_time"),
            "sim_tick": final_observation.get("sim_tick"),
            "tcp_pose": final_observation.get("tcp_pose"),
            "plug_pose": final_observation.get("plug_pose"),
            "target_port_pose": final_observation.get("target_port_pose"),
            "target_port_entrance_pose": final_observation.get("target_port_entrance_pose"),
            "score_geometry": final_observation.get("score_geometry"),
            "controller_tcp_error": final_observation.get("controller_tcp_error"),
            "controller_reference_tcp_pose": final_observation.get("controller_reference_tcp_pose"),
        },
        "available_tools_and_visuals": [
            "numeric world-frame plug/target/entrance poses",
            "distance_to_target, distance_to_entrance, lateral_misalignment, insertion_progress",
            "3 wrist camera videos at 256x256",
            "4 fixed Gazebo overview videos with stable scale",
            "roughly 10 evenly spaced diagnostic frames per available camera angle",
            "each diagnostic frame includes timestamp, angle/viewpoint description, sample index, and metric overlay",
            "world-to-base_link conversion for controller velocity commands",
        ],
        "frame_sampling": _frame_sampling_summary(frame_samples),
    }
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": json.dumps(run_summary, sort_keys=True),
        }
    ]
    for sample in frame_samples:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Frame stream={sample.stream_name}; timestamp_s={sample.timestamp_s:.3f}; "
                    f"sample={sample.sample_index + 1}/{sample.sample_count_for_stream}; "
                    f"frame_index={sample.frame_index}/{sample.frame_count}; "
                    f"angle_description={sample.viewpoint_description}."
                ),
            }
        )
        content.append({"type": "input_image", "image_url": sample.image_data_url, "detail": "high"})
    return {
        "model": model,
        "max_output_tokens": max_output_tokens,
        "store": False,
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are diagnosing a live Gazebo robot cable-insertion rollout. "
                            "Use the timestamped wrist and fixed-scene camera frames plus numeric geometry. "
                            "Answer why the trajectory is failing or too aggressive, what missing context/tools "
                            "would help make a simpler smoother movement, and whether the current visual/tooling "
                            "context is sufficient to improve trajectory quality. Be concrete about coordinate "
                            "frames, target/insertion-axis visualization, arrows, labels, overlays, crops, and "
                            "controller/tooling deficiencies. Keep each string concise; the response must fit "
                            "comfortably inside the output budget as valid JSON."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "aic_rollout_vlm_feedback",
                "strict": True,
                "schema": feedback_schema(),
            },
            "verbosity": "low",
        },
        "reasoning": {"effort": "low"},
    }


def feedback_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "failure_reason": {"type": "string"},
            "trajectory_quality_assessment": {"type": "string"},
            "coordinate_frame_concerns": {"type": "array", "items": {"type": "string"}},
            "missing_context_or_tools": {"type": "array", "items": {"type": "string"}},
            "recommended_visual_aids": {"type": "array", "items": {"type": "string"}},
            "recommended_controller_changes": {"type": "array", "items": {"type": "string"}},
            "is_context_sufficient_for_high_quality_trajectory": {"type": "boolean"},
            "satisfaction_level": {"type": "string", "enum": ["satisfied", "partially_satisfied", "not_satisfied"]},
            "next_iteration_priority": {"type": "string"},
        },
        "required": [
            "failure_reason",
            "trajectory_quality_assessment",
            "coordinate_frame_concerns",
            "missing_context_or_tools",
            "recommended_visual_aids",
            "recommended_controller_changes",
            "is_context_sufficient_for_high_quality_trajectory",
            "satisfaction_level",
            "next_iteration_priority",
        ],
    }


def _read_evenly_spaced_video_frames(
    path: Path,
    *,
    sample_count: int,
) -> list[tuple[np.ndarray, int, int, float]]:
    cap = cv2.VideoCapture(str(path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        cap.release()
        return []
    target_count = min(max(int(sample_count), 1), frame_count)
    indices = np.linspace(0, frame_count - 1, num=target_count)
    frame_indices = sorted({int(round(value)) for value in indices})
    samples: list[tuple[np.ndarray, int, int, float]] = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        samples.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), frame_index, frame_count, fps))
    cap.release()
    return samples


def _annotate_feedback_frame(
    *,
    frame: np.ndarray,
    stream_name: str,
    viewpoint_description: str,
    timestamp_s: float,
    sample_index: int,
    sample_count_for_stream: int,
    final_observation: dict[str, Any],
    final_info: dict[str, Any],
) -> np.ndarray:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle([(0, 0), (width, 52)], fill=(15, 18, 22, 218))
    draw.text(
        (8, 8),
        f"{stream_name} | sample {sample_index + 1}/{sample_count_for_stream} | t={timestamp_s:.3f}s",
        fill=(255, 255, 255, 255),
    )
    draw.text((8, 28), viewpoint_description[:80], fill=(220, 232, 255, 255))
    score = dict(final_observation.get("score_geometry") or {})
    details = (
        f"target={_metric(final_info, score, 'distance_to_target'):.3f}m "
        f"entrance={_metric(final_info, score, 'distance_to_entrance'):.3f}m "
        f"lat={_metric(final_info, score, 'lateral_misalignment'):.3f}m "
        f"progress={_metric(final_info, score, 'insertion_progress'):.2f}"
    )
    draw.rectangle([(0, height - 34), (width, height)], fill=(15, 18, 22, 218))
    draw.text((8, height - 24), details, fill=(255, 255, 255, 255))
    # Non-calibrated visual prompt overlay: reminds the VLM which annotations are absent from raw frames.
    if stream_name.startswith("overview_"):
        draw.line([(width - 84, height - 58), (width - 24, height - 58)], fill=(240, 70, 70, 255), width=3)
        draw.text((width - 88, height - 78), "+world axis cue", fill=(240, 240, 240, 255))
    return np.asarray(image, dtype=np.uint8)


def _frame_sampling_summary(frame_samples: list[FrameSample]) -> dict[str, Any]:
    by_stream: dict[str, dict[str, Any]] = {}
    for sample in frame_samples:
        stream = by_stream.setdefault(
            sample.stream_name,
            {
                "angle_description": sample.viewpoint_description,
                "sample_count": 0,
                "timestamps_s": [],
                "frame_indices": [],
            },
        )
        stream["sample_count"] = int(stream["sample_count"]) + 1
        stream["timestamps_s"].append(float(sample.timestamp_s))
        stream["frame_indices"].append(int(sample.frame_index))
    return by_stream


def _metric(final_info: dict[str, Any], score: dict[str, Any], key: str) -> float:
    value = final_info.get(key)
    if value is None:
        value = score.get(key)
    if isinstance(value, list) and value:
        value = value[0]
    try:
        return float(value)
    except Exception:
        return 0.0


def _encode_image_data_url(image: np.ndarray, *, format: str = "PNG") -> str:
    with BytesIO() as buffer:
        Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buffer, format=format)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{format.lower()};base64,{encoded}"


def _post_responses_request(
    payload: dict[str, Any],
    *,
    api_key: str | None,
    base_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    if not api_key:
        raise VlmFeedbackError("Missing OPENAI_API_KEY for final GPT-5 feedback call.")
    request = Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise VlmFeedbackError(f"GPT-5 feedback request failed status={exc.code}: {body}") from exc
    except (URLError, TimeoutError, socket.timeout) as exc:
        raise VlmFeedbackError(f"GPT-5 feedback request failed: {exc}") from exc


def _parse_feedback_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    text = response_payload.get("output_text")
    if not isinstance(text, str) or not text.strip():
        texts: list[str] = []

        def visit(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") in {"output_text", "text"} and isinstance(node.get("text"), str):
                    texts.append(node["text"])
                    return
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(response_payload.get("output", []))
        text = "\n".join(texts).strip()
    if not text:
        raise VlmFeedbackError("GPT-5 feedback response did not contain output text.")
    return json.loads(text)


def _sanitize_payload(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: ("<image_data_url>" if key in {"image_url", "image_data_url"} and isinstance(value, str) else _sanitize_payload(value))
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_sanitize_payload(item) for item in node]
    return node
