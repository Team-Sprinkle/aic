"""Prepare review bundles for future GPT-5 VLM trajectory critique."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aic_teacher_official.trajectory import SmoothTrajectory
from aic_teacher_official.vlm_planner import _image_items, _extract_output_text, load_openai_api_key


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _find_images(image_dir: Path | None) -> list[Path]:
    if image_dir is None or not image_dir.exists():
        return []
    return sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def build_review_bundle(
    trajectory_path: str | Path,
    output_path: str | Path,
    *,
    wrist_image_dir: str | Path | None = None,
    gazebo_image_dir: str | Path | None = None,
    samples: int = 8,
) -> dict[str, Any]:
    trajectory_path = Path(trajectory_path)
    output_path = Path(output_path)
    trajectory = SmoothTrajectory.load_json(trajectory_path)
    if samples <= 0:
        raise ValueError("samples must be positive")

    stride = max(1, (len(trajectory.waypoints) - 1) // max(1, samples - 1))
    selected = trajectory.waypoints[::stride][:samples]
    if selected[-1] is not trajectory.waypoints[-1]:
        selected.append(trajectory.waypoints[-1])

    wrist_images = _find_images(Path(wrist_image_dir) if wrist_image_dir else None)
    gazebo_images = _find_images(Path(gazebo_image_dir) if gazebo_image_dir else None)

    manifest: dict[str, Any] = {
        "schema_version": "official_teacher_review_bundle/v0",
        "trajectory": str(trajectory_path),
        "trajectory_metadata": trajectory.metadata.to_dict(),
        "samples": [
            {
                "timestamp": waypoint.timestamp,
                "phase": waypoint.phase.value,
                "source": waypoint.source.value,
                "tcp_pose": waypoint.tcp_pose.to_dict(),
                "tcp_velocity": waypoint.tcp_velocity,
                "diagnostics": waypoint.diagnostics,
            }
            for waypoint in selected
        ],
        "images": {
            "wrist": [str(path) for path in wrist_images],
            "gazebo": [str(path) for path in gazebo_images],
            "missing_wrist_images": len(wrist_images) == 0,
            "missing_gazebo_images": len(gazebo_images) == 0,
        },
        "force_summary": {
            "available": False,
            "todo": "Attach wrist force/torque summary from LeRobot/ROS bag when available.",
        },
        "actions": {
            "available": False,
            "todo": "Attach executed action rows from LeRobot dataset after recording.",
        },
        "critique": {
            "model": "gpt-5-vlm",
            "api_called": False,
            "todo": "Send this manifest plus equidistant images to GPT-5 VLM in a later job.",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def call_gpt5_failure_review(
    manifest: dict[str, Any],
    *,
    model: str = "gpt-5",
) -> dict[str, Any]:
    """Run one GPT-5 VLM failure-analysis call for a prepared review manifest."""
    api_key = load_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set and could not be read from ~/.zshrc."
        )
    try:
        from openai import OpenAI
    except Exception as ex:
        raise RuntimeError("The openai Python package is required for GPT-5 review.") from ex

    image_paths = [
        Path(path)
        for path in [
            *manifest.get("images", {}).get("wrist", []),
            *manifest.get("images", {}).get("gazebo", []),
        ]
    ]
    prompt = {
        "task": "Analyze trajectory failure or quality risks for AIC cable insertion.",
        "success_preferences": {
            "score_target": ">80",
            "final_distance_to_port_m": "<0.05",
        },
        "required_analysis": [
            "Check whether VLM delta commands were well-formed for optimizer/replay.",
            "Compare phase labels, timestamps, TCP poses, actions if available, and images.",
            "Identify likely bugs in relative pose or coordinate calculations.",
            "Suggest concrete next trajectory/planner changes.",
        ],
        "manifest": manifest,
    }
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        instructions="You are a robotics failure-analysis reviewer. Output JSON only.",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": json.dumps(prompt, indent=2)},
                    *_image_items(image_paths[:16]),
                ],
            }
        ],
    )
    text = _extract_output_text(response)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"raw_text": text}
    parsed.setdefault("diagnostics", {})
    parsed["diagnostics"].update(
        {
            "model": model,
            "api_calls_used": 1,
            "image_count": len(image_paths[:16]),
        }
    )
    return parsed
