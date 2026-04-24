"""Dataset export for selected teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils import to_jsonable

DEFAULT_IMAGE_SHAPE = (256, 256, 3)


@dataclass(frozen=True)
class TeacherDatasetExportResult:
    dataset_path: Path
    metadata_path: Path
    format: str


@dataclass(frozen=True)
class RolloutDatasetFrame:
    kind: str
    observation: dict[str, Any]
    action: list[float]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]
    planner_rationale: str | None
    phase: str | None


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
                "history_summary": step.get("history_summary", {}),
                "data_quality": step.get("data_quality", artifact["metadata"].get("data_quality", {})),
                "auxiliary_summary_available": step.get("auxiliary_summary_available", False),
                "auxiliary_force_contact_summary": step.get("auxiliary_force_contact_summary", {}),
                "auxiliary_contact_metrics": step.get("auxiliary_contact_metrics", {}),
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
                    "ranking_metrics": candidate_entry.get("ranking_metrics", {}),
                },
                "score_breakdown": score,
            }
        )
    jsonl_path.write_text(
        "".join(json.dumps(to_jsonable(record), sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    "candidate_spec": candidate_entry["candidate_spec"],
                    "rank": candidate_entry["rank"],
                    "official_style_score": score,
                    "ranking_metrics": candidate_entry.get("ranking_metrics", {}),
                    "artifact_metadata": artifact["metadata"],
                    "auxiliary_summary_metadata": artifact["metadata"].get("auxiliary_summary_metadata", {}),
                }
            ),
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
            to_jsonable(
                {
                    "candidate_spec": candidate_entry["candidate_spec"],
                    "rank": candidate_entry["rank"],
                    "official_style_score": candidate_entry["official_style_score"],
                    "ranking_metrics": candidate_entry.get("ranking_metrics", {}),
                    "selected_top_k": candidate_entry["selected_top_k"],
                    "near_perfect": candidate_entry["near_perfect"],
                    "artifact_metadata": artifact["metadata"],
                    "auxiliary_summary_metadata": artifact["metadata"].get("auxiliary_summary_metadata", {}),
                }
            ),
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


def export_rollout_lerobot_dataset(
    frames: list[RolloutDatasetFrame],
    *,
    repo_id: str,
    output_root: Path | str,
    single_task: str,
    fps: int,
    use_videos: bool = False,
    metadata: dict[str, Any] | None = None,
) -> TeacherDatasetExportResult:
    from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_STR

    if not frames:
        raise ValueError("No rollout frames were provided for dataset export.")
    sample_observation = frames[0].observation
    image_shape = _image_shape_from_observation(sample_observation)
    observation_features = _lerobot_rollout_observation_features(image_shape=image_shape)
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
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(observation=observation_features),
            use_videos=use_videos,
        ),
    )
    output_root = Path(output_root)
    dataset = LeRobotDataset.create(
        repo_id,
        fps=fps,
        root=output_root,
        robot_type="ur5e_aic",
        features=dataset_features,
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=0,
        batch_encoding_size=1,
        vcodec="h264",
    )
    for frame_entry in frames:
        obs_raw = _lerobot_rollout_observation_from_frame(frame_entry.observation)
        obs_processed = observation_processor(obs_raw)
        action_processed = action_processor((_lerobot_action_from_array(frame_entry.action), obs_raw))
        frame = {
            **build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR),
            **build_dataset_frame(dataset.features, action_processed, prefix=ACTION),
            "task": single_task,
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    metadata_path = output_root / repo_id / "rollout_export_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            to_jsonable(
                {
                    "repo_id": repo_id,
                    "single_task": single_task,
                    "fps": fps,
                    "use_videos": use_videos,
                    "frame_count": len(frames),
                    "metadata": metadata or {},
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    extras_path = output_root / repo_id / "rollout_extras.jsonl"
    extras_path.write_text(
        "".join(
            json.dumps(
                to_jsonable(
                    {
                        "index": index,
                        "kind": frame_entry.kind,
                        "reward": frame_entry.reward,
                        "terminated": frame_entry.terminated,
                        "truncated": frame_entry.truncated,
                        "planner_rationale": frame_entry.planner_rationale,
                        "phase": frame_entry.phase,
                        "info": frame_entry.info,
                    }
                ),
                sort_keys=True,
            )
            + "\n"
            for index, frame_entry in enumerate(frames)
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


def _lerobot_rollout_observation_features(*, image_shape: tuple[int, int, int]) -> dict[str, Any]:
    return {
        **_lerobot_observation_features(image_shape=image_shape),
        "sim_time": float,
        "sim_tick": float,
        "wrench_timestamp": float,
        "off_limit_contact": float,
        "plug_pose.position.x": float,
        "plug_pose.position.y": float,
        "plug_pose.position.z": float,
        "plug_pose.orientation.x": float,
        "plug_pose.orientation.y": float,
        "plug_pose.orientation.z": float,
        "plug_pose.orientation.w": float,
        "target_port_pose.position.x": float,
        "target_port_pose.position.y": float,
        "target_port_pose.position.z": float,
        "target_port_pose.orientation.x": float,
        "target_port_pose.orientation.y": float,
        "target_port_pose.orientation.z": float,
        "target_port_pose.orientation.w": float,
        "target_port_entrance_pose.position.x": float,
        "target_port_entrance_pose.position.y": float,
        "target_port_entrance_pose.position.z": float,
        "target_port_entrance_pose.orientation.x": float,
        "target_port_entrance_pose.orientation.y": float,
        "target_port_entrance_pose.orientation.z": float,
        "target_port_entrance_pose.orientation.w": float,
    }


def _lerobot_observation_from_step(step: dict[str, Any]) -> dict[str, Any]:
    obs = step["observation_summary"]
    tcp_pose = obs["tcp_pose"]
    tcp_velocity = obs["tcp_velocity"]
    joint_positions = list(obs["joint_positions"])
    joint_positions = joint_positions[:7] + [0.0] * max(0, 7 - len(joint_positions))
    wrench = obs["wrench"]
    target_tcp_pose = step["trajectory_point"]["target_tcp_pose"]
    tcp_error = obs.get("controller_tcp_error")
    if tcp_error is None:
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
        return np.zeros(DEFAULT_IMAGE_SHAPE, dtype=np.uint8)
    shape = tuple(int(x) for x in summary.get("shape", list(DEFAULT_IMAGE_SHAPE)))
    return np.zeros(shape, dtype=np.uint8)


def _image_shape_from_summary(image_summaries: dict[str, Any]) -> tuple[int, int, int]:
    left = image_summaries.get("left")
    if left is None:
        return DEFAULT_IMAGE_SHAPE
    shape = left.get("shape", list(DEFAULT_IMAGE_SHAPE))
    return (int(shape[0]), int(shape[1]), int(shape[2]))


def _image_shape_from_observation(observation: dict[str, Any]) -> tuple[int, int, int]:
    images = observation.get("images") or {}
    left = images.get("left")
    if isinstance(left, np.ndarray) and left.ndim == 3:
        return (int(left.shape[0]), int(left.shape[1]), int(left.shape[2]))
    return DEFAULT_IMAGE_SHAPE


def _lerobot_action_from_array(action: list[float] | tuple[float, ...]) -> dict[str, float]:
    values = list(action)[:6] + [0.0] * max(0, 6 - len(action))
    return {
        "linear.x": float(values[0]),
        "linear.y": float(values[1]),
        "linear.z": float(values[2]),
        "angular.x": float(values[3]),
        "angular.y": float(values[4]),
        "angular.z": float(values[5]),
    }


def _lerobot_rollout_observation_from_frame(observation: dict[str, Any]) -> dict[str, Any]:
    tcp_pose = list(np.asarray(observation.get("tcp_pose", np.zeros(7)), dtype=np.float64))
    tcp_velocity = list(np.asarray(observation.get("tcp_velocity", np.zeros(6)), dtype=np.float64))
    controller_tcp_error = list(np.asarray(observation.get("controller_tcp_error", np.zeros(6)), dtype=np.float64))
    joint_positions = list(np.asarray(observation.get("joint_positions", np.zeros(7)), dtype=np.float64))
    joint_positions = joint_positions[:7] + [0.0] * max(0, 7 - len(joint_positions))
    wrench = list(np.asarray(observation.get("wrench", np.zeros(6)), dtype=np.float64))
    plug_pose = list(np.asarray(observation.get("plug_pose", np.zeros(7)), dtype=np.float64))
    target_port_pose = list(np.asarray(observation.get("target_port_pose", np.zeros(7)), dtype=np.float64))
    target_port_entrance_pose = list(
        np.asarray(observation.get("target_port_entrance_pose", np.zeros(7)), dtype=np.float64)
    )
    images = observation.get("images") or {}
    return {
        "tcp_pose.position.x": float(tcp_pose[0]),
        "tcp_pose.position.y": float(tcp_pose[1]),
        "tcp_pose.position.z": float(tcp_pose[2]),
        "tcp_pose.orientation.x": float(tcp_pose[3]),
        "tcp_pose.orientation.y": float(tcp_pose[4]),
        "tcp_pose.orientation.z": float(tcp_pose[5]),
        "tcp_pose.orientation.w": float(tcp_pose[6]),
        "tcp_velocity.linear.x": float(tcp_velocity[0]),
        "tcp_velocity.linear.y": float(tcp_velocity[1]),
        "tcp_velocity.linear.z": float(tcp_velocity[2]),
        "tcp_velocity.angular.x": float(tcp_velocity[3]),
        "tcp_velocity.angular.y": float(tcp_velocity[4]),
        "tcp_velocity.angular.z": float(tcp_velocity[5]),
        "tcp_error.x": float(controller_tcp_error[0]),
        "tcp_error.y": float(controller_tcp_error[1]),
        "tcp_error.z": float(controller_tcp_error[2]),
        "tcp_error.rx": float(controller_tcp_error[3]),
        "tcp_error.ry": float(controller_tcp_error[4]),
        "tcp_error.rz": float(controller_tcp_error[5]),
        **{f"joint_positions.{index}": float(joint_positions[index]) for index in range(7)},
        "wrist_wrench.force.x": float(wrench[0]),
        "wrist_wrench.force.y": float(wrench[1]),
        "wrist_wrench.force.z": float(wrench[2]),
        "wrist_wrench.torque.x": float(wrench[3]),
        "wrist_wrench.torque.y": float(wrench[4]),
        "wrist_wrench.torque.z": float(wrench[5]),
        "left_camera": np.asarray(images.get("left", np.zeros(DEFAULT_IMAGE_SHAPE, dtype=np.uint8)), dtype=np.uint8),
        "center_camera": np.asarray(images.get("center", np.zeros(DEFAULT_IMAGE_SHAPE, dtype=np.uint8)), dtype=np.uint8),
        "right_camera": np.asarray(images.get("right", np.zeros(DEFAULT_IMAGE_SHAPE, dtype=np.uint8)), dtype=np.uint8),
        "sim_time": float(np.asarray(observation.get("sim_time", 0.0)).reshape(-1)[0]),
        "sim_tick": float(np.asarray(observation.get("sim_tick", 0.0)).reshape(-1)[0]),
        "wrench_timestamp": float(np.asarray(observation.get("wrench_timestamp", 0.0)).reshape(-1)[0]),
        "off_limit_contact": float(np.asarray(observation.get("off_limit_contact", 0.0)).reshape(-1)[0]),
        "plug_pose.position.x": float(plug_pose[0]),
        "plug_pose.position.y": float(plug_pose[1]),
        "plug_pose.position.z": float(plug_pose[2]),
        "plug_pose.orientation.x": float(plug_pose[3]),
        "plug_pose.orientation.y": float(plug_pose[4]),
        "plug_pose.orientation.z": float(plug_pose[5]),
        "plug_pose.orientation.w": float(plug_pose[6]),
        "target_port_pose.position.x": float(target_port_pose[0]),
        "target_port_pose.position.y": float(target_port_pose[1]),
        "target_port_pose.position.z": float(target_port_pose[2]),
        "target_port_pose.orientation.x": float(target_port_pose[3]),
        "target_port_pose.orientation.y": float(target_port_pose[4]),
        "target_port_pose.orientation.z": float(target_port_pose[5]),
        "target_port_pose.orientation.w": float(target_port_pose[6]),
        "target_port_entrance_pose.position.x": float(target_port_entrance_pose[0]),
        "target_port_entrance_pose.position.y": float(target_port_entrance_pose[1]),
        "target_port_entrance_pose.position.z": float(target_port_entrance_pose[2]),
        "target_port_entrance_pose.orientation.x": float(target_port_entrance_pose[3]),
        "target_port_entrance_pose.orientation.y": float(target_port_entrance_pose[4]),
        "target_port_entrance_pose.orientation.z": float(target_port_entrance_pose[5]),
        "target_port_entrance_pose.orientation.w": float(target_port_entrance_pose[6]),
    }
