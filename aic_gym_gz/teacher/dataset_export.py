"""Dataset export for selected teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherDatasetExportResult:
    dataset_path: Path
    metadata_path: Path
    format: str


def export_teacher_jsonl_dataset(
    candidate_entry: dict[str, Any],
    *,
    output_dir: Path | str,
) -> TeacherDatasetExportResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "teacher_rollouts.jsonl"
    metadata_path = output_dir / "teacher_rollouts.metadata.json"
    artifact = candidate_entry["artifact"]
    score = candidate_entry["official_style_score"]
    records = []
    for step_index, step in enumerate(artifact["step_logs"]):
        records.append(
            {
                "episode_id": f"{artifact['metadata']['trial_id']}_{candidate_entry['rank']}",
                "step_index": step_index,
                "timestamp": step["sim_time"],
                "observation": step["observation_summary"],
                "action": step["trajectory_point"]["action"],
                "planner_rationale": step["planner_rationale"],
                "dynamics_summary": step["dynamics_summary"],
                "probe_result": step.get("probe_result"),
                "task_metadata": {
                    "task_id": artifact["metadata"]["task_id"],
                    "trial_id": artifact["metadata"]["trial_id"],
                },
                "planner_metadata": {
                    "candidate_spec": candidate_entry["candidate_spec"],
                    "candidate_rank": candidate_entry["rank"],
                    "selected_top_k": candidate_entry["selected_top_k"],
                    "near_perfect": candidate_entry["near_perfect"],
                },
                "score_breakdown": score,
            }
        )
    jsonl_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_spec": candidate_entry["candidate_spec"],
                "rank": candidate_entry["rank"],
                "official_style_score": score,
                "artifact_metadata": artifact["metadata"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TeacherDatasetExportResult(
        dataset_path=jsonl_path,
        metadata_path=metadata_path,
        format="jsonl",
    )


def export_teacher_lerobot_dataset(
    candidate_entry: dict[str, Any],
    *,
    repo_id: str,
    output_root: Path | str,
) -> TeacherDatasetExportResult:
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR

    output_root = Path(output_root)
    artifact = candidate_entry["artifact"]
    step_logs = artifact["step_logs"]
    if not step_logs:
        raise ValueError("Candidate artifact has no step logs to export.")
    sample_obs = step_logs[0]["observation_summary"]
    image_shape = _image_shape_from_summary(sample_obs.get("image_summaries", {}))
    observation_features = _lerobot_observation_features(image_shape=image_shape)
    action_features = {
        "linear.x": float,
        "linear.y": float,
        "linear.z": float,
        "angular.x": float,
        "angular.y": float,
        "angular.z": float,
    }
    action_processor, _, observation_processor = make_default_processors()
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=action_processor,
            initial_features=create_initial_features(action=action_features),
            use_videos=False,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=False,
        ),
    )
    dataset = LeRobotDataset.create(
        repo_id,
        fps=20,
        root=output_root,
        robot_type="ur5e_aic",
        features=dataset_features,
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
        vcodec="h264",
    )
    for step in step_logs:
        obs_raw = _lerobot_observation_from_step(step)
        obs_processed = observation_processor(obs_raw)
        action_processed = action_processor((_lerobot_action_from_step(step), obs_raw))
        frame = {
            **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
            **build_dataset_frame(dataset.features, action_processed, prefix=ACTION),
            "task": artifact["metadata"]["task_id"],
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    metadata_path = output_root / repo_id / "teacher_export_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "candidate_spec": candidate_entry["candidate_spec"],
                "rank": candidate_entry["rank"],
                "official_style_score": candidate_entry["official_style_score"],
                "selected_top_k": candidate_entry["selected_top_k"],
                "near_perfect": candidate_entry["near_perfect"],
                "artifact_metadata": artifact["metadata"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TeacherDatasetExportResult(
        dataset_path=output_root / repo_id,
        metadata_path=metadata_path,
        format="lerobot",
    )


def _lerobot_observation_features(*, image_shape: tuple[int, int, int]) -> dict[str, Any]:
    return {
        "tcp_pose.position.x": float,
        "tcp_pose.position.y": float,
        "tcp_pose.position.z": float,
        "tcp_pose.orientation.x": float,
        "tcp_pose.orientation.y": float,
        "tcp_pose.orientation.z": float,
        "tcp_pose.orientation.w": float,
        "tcp_velocity.linear.x": float,
        "tcp_velocity.linear.y": float,
        "tcp_velocity.linear.z": float,
        "tcp_velocity.angular.x": float,
        "tcp_velocity.angular.y": float,
        "tcp_velocity.angular.z": float,
        "tcp_error.x": float,
        "tcp_error.y": float,
        "tcp_error.z": float,
        "tcp_error.rx": float,
        "tcp_error.ry": float,
        "tcp_error.rz": float,
        "joint_positions.0": float,
        "joint_positions.1": float,
        "joint_positions.2": float,
        "joint_positions.3": float,
        "joint_positions.4": float,
        "joint_positions.5": float,
        "joint_positions.6": float,
        "wrist_wrench.force.x": float,
        "wrist_wrench.force.y": float,
        "wrist_wrench.force.z": float,
        "wrist_wrench.torque.x": float,
        "wrist_wrench.torque.y": float,
        "wrist_wrench.torque.z": float,
        "left_camera": image_shape,
        "center_camera": image_shape,
        "right_camera": image_shape,
    }


def _lerobot_observation_from_step(step: dict[str, Any]) -> dict[str, Any]:
    obs = step["observation_summary"]
    tcp_pose = obs["tcp_pose"]
    tcp_velocity = obs["tcp_velocity"]
    joint_positions = list(obs["joint_positions"])
    joint_positions = joint_positions[:7] + [0.0] * max(0, 7 - len(joint_positions))
    wrench = obs["wrench"]
    target_tcp_pose = step["trajectory_point"]["target_tcp_pose"]
    tcp_error = [float(target_tcp_pose[idx] - tcp_pose[idx]) for idx in range(6)]
    image_summaries = obs.get("image_summaries", {})
    return {
        "tcp_pose.position.x": float(tcp_pose[0]),
        "tcp_pose.position.y": float(tcp_pose[1]),
        "tcp_pose.position.z": float(tcp_pose[2]),
        "tcp_pose.orientation.x": 0.0,
        "tcp_pose.orientation.y": 0.0,
        "tcp_pose.orientation.z": float(tcp_pose[5]),
        "tcp_pose.orientation.w": float(tcp_pose[6]),
        "tcp_velocity.linear.x": float(tcp_velocity[0]),
        "tcp_velocity.linear.y": float(tcp_velocity[1]),
        "tcp_velocity.linear.z": float(tcp_velocity[2]),
        "tcp_velocity.angular.x": float(tcp_velocity[3]),
        "tcp_velocity.angular.y": float(tcp_velocity[4]),
        "tcp_velocity.angular.z": float(tcp_velocity[5]),
        "tcp_error.x": float(tcp_error[0]),
        "tcp_error.y": float(tcp_error[1]),
        "tcp_error.z": float(tcp_error[2]),
        "tcp_error.rx": float(tcp_error[3]),
        "tcp_error.ry": float(tcp_error[4]),
        "tcp_error.rz": float(tcp_error[5]),
        **{f"joint_positions.{idx}": float(joint_positions[idx]) for idx in range(7)},
        "wrist_wrench.force.x": float(wrench[0]),
        "wrist_wrench.force.y": float(wrench[1]),
        "wrist_wrench.force.z": float(wrench[2]),
        "wrist_wrench.torque.x": float(wrench[3]),
        "wrist_wrench.torque.y": float(wrench[4]),
        "wrist_wrench.torque.z": float(wrench[5]),
        "left_camera": _blank_or_image(image_summaries.get("left")),
        "center_camera": _blank_or_image(image_summaries.get("center")),
        "right_camera": _blank_or_image(image_summaries.get("right")),
    }


def _lerobot_action_from_step(step: dict[str, Any]) -> dict[str, float]:
    action = list(step["trajectory_point"]["action"])
    return {
        "linear.x": float(action[0]),
        "linear.y": float(action[1]),
        "linear.z": float(action[2]),
        "angular.x": float(action[3]),
        "angular.y": float(action[4]),
        "angular.z": float(action[5]),
    }


def _blank_or_image(summary: dict[str, Any] | None) -> np.ndarray:
    if summary is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    shape = tuple(int(x) for x in summary.get("shape", [64, 64, 3]))
    return np.zeros(shape, dtype=np.uint8)


def _image_shape_from_summary(image_summaries: dict[str, Any]) -> tuple[int, int, int]:
    left = image_summaries.get("left")
    if left is None:
        return (64, 64, 3)
    shape = left.get("shape", [64, 64, 3])
    return (int(shape[0]), int(shape[1]), int(shape[2]))
