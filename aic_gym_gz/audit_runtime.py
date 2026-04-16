"""Observation/runtime/scoring parity audit for the gazebo-gym path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


OBSERVATION_AUDIT: list[dict[str, str]] = [
    {
        "official_item": "left_image / center_image / right_image",
        "current_gym_availability": "Available when include_images=True via RosCameraSidecarIO",
        "source_file_function": "aic_gym_gz/io.py:RosCameraSidecarIO.observation_from_state",
        "status": "matched",
        "notes": "Still sourced through ROS bridge sidecar rather than pure Gazebo transport.",
    },
    {
        "official_item": "left_camera_info / center_camera_info / right_camera_info",
        "current_gym_availability": "Available when include_images=True via ROS camera_info sidecar",
        "source_file_function": "aic_gym_gz/io.py:RosCameraSubscriber._camera_info_callback",
        "status": "matched",
        "notes": "Numeric CameraInfo fields are propagated into observation['camera_info'].",
    },
    {
        "official_item": "wrist_wrench",
        "current_gym_availability": "Available in RuntimeState and observation, populated from /fts_broadcaster/wrench with controller tare subtraction",
        "source_file_function": "aic_gym_gz/runtime.py:_RuntimeRosObserver, _wrench_from_ros_sample",
        "status": "matched",
        "notes": "Exactness depends on ROS topic availability in the live world.",
    },
    {
        "official_item": "joint_states",
        "current_gym_availability": "Available from Gazebo transport observation",
        "source_file_function": "aic_gym_gz/runtime.py:ScenarioGymGzBackend._runtime_state_from_observation",
        "status": "matched",
        "notes": "Joint positions are exact to transport payload; velocities remain finite-difference or zero-filled.",
    },
    {
        "official_item": "controller_state",
        "current_gym_availability": "Available as flattened observation fields and RuntimeState.controller_state",
        "source_file_function": "aic_gym_gz/runtime.py:_controller_state_from_ros_sample, aic_gym_gz/io.py:_base_observation",
        "status": "matched",
        "notes": "Requires ROS controller_state topic; without it fields zero-fill.",
    },
    {
        "official_item": "image timestamps",
        "current_gym_availability": "Available in observation['image_timestamps']",
        "source_file_function": "aic_gym_gz/io.py:RosCameraSubscriber._image_callback",
        "status": "matched",
        "notes": "Timestamps are ROS image header timestamps.",
    },
]


SCORING_AUDIT: list[dict[str, str]] = [
    {
        "official_metric": "tier2.duration",
        "current_gym_implementation": "Same inverse score range used locally",
        "source_file_function": "aic_gym_gz/reward.py:AicScoreCalculator.evaluate",
        "status": "matched",
        "notes": "Local path labeled gym_reward.",
    },
    {
        "official_metric": "tier2.trajectory_smoothness / jerk",
        "current_gym_implementation": "Central-window average jerk approximation aligned to official scorer structure",
        "source_file_function": "aic_gym_gz/reward.py:_official_average_linear_jerk",
        "status": "approximate",
        "notes": "Uses env velocity history rather than the official TF buffer implementation.",
    },
    {
        "official_metric": "tier2.insertion_force",
        "current_gym_implementation": "Penalty after >1s above 20N using live wrench samples",
        "source_file_function": "aic_gym_gz/reward.py:_time_above_force",
        "status": "matched",
        "notes": "Depends on live wrench topic availability.",
    },
    {
        "official_metric": "tier2.contacts",
        "current_gym_implementation": "Binary -24 penalty from off-limit contact topic",
        "source_file_function": "aic_gym_gz/runtime.py:_RuntimeRosObserver",
        "status": "matched",
        "notes": "Depends on contact topic availability.",
    },
    {
        "official_metric": "tier2.trajectory_efficiency",
        "current_gym_implementation": "Same inverse path-length score range used locally",
        "source_file_function": "aic_gym_gz/reward.py:AicScoreCalculator.evaluate",
        "status": "matched",
        "notes": "Uses env tcp path instead of official TF buffer poses.",
    },
    {
        "official_metric": "tier3.partial_insertion",
        "current_gym_implementation": "Uses explicit port and port-entrance poses when available",
        "source_file_function": "aic_gym_gz/reward.py:_tier3_score, aic_gym_gz/runtime.py:_build_score_geometry",
        "status": "matched",
        "notes": "Becomes exact relative to local geometry once port-entrance entity is found.",
    },
    {
        "official_metric": "official_eval_score",
        "current_gym_implementation": "Not executed by this audit",
        "source_file_function": "external official toolkit",
        "status": "missing",
        "notes": "Run official aic_scoring or toolkit evaluation for a true official_eval_score.",
    },
]


REPLAY_AUDIT: dict[str, Any] = {
    "checkpoint_restore": {
        "mock_backend": {
            "status": "matched",
            "mode": "mock_exact",
            "notes": "Exact checkpoint/restore implemented for MockStepperBackend.",
        },
        "live_backend": {
            "status": "approximate",
            "mode": "live_reset_replay",
            "notes": [
                "Exact mid-rollout restore remains unavailable because the live Gazebo transport path does not expose a world snapshot/restore service.",
                "The exported checkpoint supports deterministic reset-and-rerun from scenario start with the last observed state attached for diagnostics.",
            ],
        },
    }
}


def generate_runtime_audit() -> dict[str, Any]:
    return {
        "observation_parity": OBSERVATION_AUDIT,
        "scoring_parity": SCORING_AUDIT,
        "replay_support": REPLAY_AUDIT,
        "score_labels": {
            "gym_reward": "Local gazebo-gym reward/scoring path in aic_gym_gz.",
            "teacher_official_style_score": "Teacher-side approximation that may exist on feat/agent-teacher, not computed here.",
            "official_eval_score": "Actual official toolkit evaluation score, not computed by this audit unless run separately.",
        },
    }


def write_runtime_audit(*, output_json: str | None = None, output_markdown: str | None = None) -> dict[str, Any]:
    report = generate_runtime_audit()
    if output_json:
        Path(output_json).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if output_markdown:
        Path(output_markdown).write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Gazebo-Gym Runtime Parity Audit",
        "",
        "## Observation Parity",
        "",
        "| Official observation item | Current gym/live availability | Source file/function | Status | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["observation_parity"]:
        lines.append(
            f"| {row['official_item']} | {row['current_gym_availability']} | {row['source_file_function']} | {row['status']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Scoring Parity",
            "",
            "| Official metric/term | Current gym implementation | Source file/function | Status | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["scoring_parity"]:
        lines.append(
            f"| {row['official_metric']} | {row['current_gym_implementation']} | {row['source_file_function']} | {row['status']} | {row['notes']} |"
        )
    lines.extend(
        [
            "",
            "## Replay Support",
            "",
            f"- Mock backend checkpoint/restore: {report['replay_support']['checkpoint_restore']['mock_backend']['status']}",
            f"- Live backend checkpoint/restore: {report['replay_support']['checkpoint_restore']['live_backend']['status']}",
            "",
            "## Score Labels",
            "",
        ]
    )
    for label, description in report["score_labels"].items():
        lines.append(f"- `{label}`: {description}")
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            write_runtime_audit(
                output_json=args.output_json,
                output_markdown=args.output_markdown,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
