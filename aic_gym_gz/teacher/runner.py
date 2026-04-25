"""End-to-end teacher rollout orchestration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..env import AicInsertionEnv
from ..io import summarize_image_batch
from .history import TemporalObservationBuffer
from .policy import AgentTeacherController
from .quality import (
    build_signal_quality_snapshot,
    controller_state_summary,
    normalize_auxiliary_force_contact_summary,
    serialize_nested,
    summarize_auxiliary_force_contact_summary,
    synthetic_auxiliary_force_contact_summary,
    summarize_camera_info,
)
from .replay import TeacherReplayArtifact, save_teacher_replay
from .types import TeacherRolloutLog, TeacherStepLog


def _feedback_track_dense_point(
    *,
    state,
    point,
    segment_metadata: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Track an optimized Cartesian target from the latest observed TCP pose.

    The smoother still owns the dense target path. This tracker only replaces
    stale open-loop deltas with a bounded feedback velocity toward the current
    dense target, which is important for live Gazebo runs where the cable/tool
    link does not exactly follow the previous command.
    """
    gate = segment_metadata.get("port_frame_alignment_gate")
    if not isinstance(gate, dict) or not bool(gate.get("active", False)):
        return np.asarray(point.action, dtype=np.float32), None
    if gate.get("gate_action") != "align_lateral_and_yaw_at_pre_insert_standoff":
        return np.asarray(point.action, dtype=np.float32), None
    target_pose = np.asarray(point.target_tcp_pose, dtype=np.float64)
    current_pose = np.asarray(state.tcp_pose, dtype=np.float64)
    dt = max(float(point.dt), 1e-6)
    pose_error = target_pose[:6] - current_pose[:6]
    pose_error[3:6] = [_wrap_to_pi(float(value)) for value in pose_error[3:6]]
    linear_limit = max(
        float(segment_metadata.get("segment_linear_speed_limit_mps") or 0.03),
        0.06,
    )
    angular_limit = 0.5
    linear = _clip_vector_norm(pose_error[:3] / dt, max_norm=linear_limit)
    angular = _clip_vector_norm(pose_error[3:6] / dt, max_norm=angular_limit)
    action = np.concatenate([linear, angular]).astype(np.float32)
    return action, {
        "control_source": "feedback_cartesian_tracker",
        "target_space": "tcp_pose",
        "linear_speed_limit_mps": float(linear_limit),
        "angular_speed_limit_radps": float(angular_limit),
        "tcp_position_error_m": [float(value) for value in pose_error[:3].tolist()],
        "tcp_orientation_error_rad": [float(value) for value in pose_error[3:6].tolist()],
    }


def _clip_vector_norm(vector: np.ndarray, *, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm) or norm <= 1e-12:
        return np.asarray(vector, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) * (float(max_norm) / norm)


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class TeacherRolloutResult:
    artifact: TeacherReplayArtifact
    output_path: Path | None = None


def run_teacher_rollout(
    *,
    env: AicInsertionEnv,
    controller: AgentTeacherController,
    seed: int = 123,
    trial_id: str | None = None,
    output_path: Path | None = None,
    probe_name: str = "hold_settle",
    trajectory_recorder: Any | None = None,
    step_callback: Any | None = None,
) -> TeacherRolloutResult:
    if hasattr(controller.planner, "reset_episode_budget"):
        controller.planner.reset_episode_budget()
    observation, info = env.reset(seed=seed, options={"trial_id": trial_id} if trial_id else {})
    initial_observation = dict(observation)
    scenario = env._scenario
    state = env._state
    if scenario is None or state is None:
        raise RuntimeError("Environment did not expose scenario/state after reset.")
    task_id = next(iter(scenario.tasks.keys()))
    include_images = env.task.include_images
    if trajectory_recorder is not None:
        trajectory_recorder.capture(
            observation=observation,
            scenario=scenario,
            state=state,
        )
    if step_callback is not None:
        step_callback(
            kind="reset",
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
            state=state,
            action=controller.initial_action(),
            planner_rationale=None,
            phase=controller.current_phase,
        )
    history = TemporalObservationBuffer()
    history.append(
        state=state,
        action=controller.initial_action(),
        images=observation.get("images"),
        image_timestamps=_image_timestamp_map(observation),
        image_summaries=_image_summary_map(observation),
        camera_info=observation.get("camera_info"),
        signal_quality=build_signal_quality_snapshot(
            state,
            include_images=include_images,
            camera_info=observation.get("camera_info"),
        ),
        auxiliary_force_contact_summary=normalize_auxiliary_force_contact_summary(info.get("auxiliary_force_contact_summary")),
        auxiliary_summary_available="auxiliary_force_contact_summary" in info,
    )
    start_wall = time.perf_counter()
    probe_results: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []
    trajectory_segments: list[dict[str, Any]] = []
    planner_candidates: list[dict[str, Any]] = []
    terminated = truncated = False
    last_info: dict[str, Any] = dict(info)
    planner_segment_count = 0
    no_progress_window: list[tuple[np.ndarray, float, float]] = []
    post_budget_step_count = 0
    limitations = [
        "Exact mid-rollout simulator state cloning is not available in the current stack.",
        "Branch-and-evaluate currently reuses deterministic planner variants instead of full simulator forks.",
        "Replay is exact for deterministic mock resets and best-effort for live Gazebo runs.",
    ]

    def _finalize_teacher_truncation(reason: str) -> None:
        nonlocal truncated, last_info
        truncated = True
        last_info = dict(last_info)
        last_info.setdefault("termination_reason", reason)
        if "final_evaluation" not in last_info:
            final_evaluation = env.task.final_evaluation()
            last_info["final_evaluation"] = final_evaluation
            last_info["evaluation"] = final_evaluation

    def _check_teacher_max_steps() -> bool:
        if not controller.config.run_until_env_done:
            return False
        if env._step_count < int(controller.config.max_env_steps):
            return False
        _finalize_teacher_truncation("teacher_max_env_steps_guard")
        return True

    def _check_post_budget_stagnation(current_state: Any) -> bool:
        nonlocal post_budget_step_count
        if not controller.planner_budget_exhausted():
            no_progress_window.clear()
            post_budget_step_count = 0
            return False
        post_budget_step_count += 1
        plug_xyz = np.asarray(current_state.plug_pose[:3], dtype=np.float64)
        target_xyz = np.asarray(current_state.target_port_pose[:3], dtype=np.float64)
        distance = float(np.linalg.norm(plug_xyz - target_xyz))
        orientation_error = float(current_state.score_geometry.get("orientation_error", 0.0) or 0.0)
        no_progress_window.append((plug_xyz.copy(), distance, orientation_error))
        if len(no_progress_window) > 80:
            no_progress_window.pop(0)
        if post_budget_step_count < 80 or len(no_progress_window) < 80:
            return False
        start_xyz, start_distance, start_orientation_error = no_progress_window[0]
        displacement = float(np.linalg.norm(plug_xyz - start_xyz))
        improvement = float(start_distance - distance)
        orientation_improvement = float(start_orientation_error - orientation_error)
        if displacement < 0.004 and improvement < 0.003 and orientation_improvement < 0.05:
            _finalize_teacher_truncation("post_budget_no_progress")
            last_info["post_budget_no_progress"] = {
                "window_steps": len(no_progress_window),
                "plug_displacement_m": displacement,
                "target_distance_improvement_m": improvement,
                "orientation_error_improvement_rad": orientation_improvement,
            }
            return True
        return False

    while True:
        if terminated or truncated:
            break
        if _check_teacher_max_steps():
            break
        use_close_range = False
        close_range_reason: str | None = None
        if controller.should_handoff_to_close_range(observation):
            use_close_range = True
            close_range_reason = "near_target_handoff"
        elif controller.planner_budget_exhausted():
            controller.force_close_range_handoff()
            use_close_range = True
            close_range_reason = "planner_budget_exhausted"
        elif (not controller.config.run_until_env_done) and planner_segment_count >= controller.config.segment_limit:
            break

        if use_close_range:
            action = controller.close_range_action(observation)
            observation, reward, terminated, truncated, last_info = env.step(action.astype(np.float32))
            state = env._state
            assert state is not None
            auxiliary_summary_available = "auxiliary_force_contact_summary" in last_info
            history.append(
                state=state,
                action=action,
                images=observation.get("images"),
                image_timestamps=_image_timestamp_map(observation),
                image_summaries=_image_summary_map(observation),
                camera_info=observation.get("camera_info"),
                signal_quality=build_signal_quality_snapshot(
                    state,
                    include_images=include_images,
                    camera_info=observation.get("camera_info"),
                ),
                auxiliary_force_contact_summary=_step_auxiliary_summary(
                    state=state,
                    step_info=last_info,
                ),
                auxiliary_summary_available=auxiliary_summary_available,
            )
            if trajectory_recorder is not None:
                trajectory_recorder.capture(
                    observation=observation,
                    scenario=scenario,
                    state=state,
                )
            if step_callback is not None:
                step_callback(
                    kind="close_range_step",
                    observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=last_info,
                    state=state,
                    action=action,
                    planner_rationale=close_range_reason,
                    phase=controller.close_range_policy.phase,
                )
            auxiliary_force_contact_summary = _step_auxiliary_summary(
                state=state,
                step_info=last_info,
            )
            auxiliary_contact_metrics = summarize_auxiliary_force_contact_summary(
                auxiliary_force_contact_summary=auxiliary_force_contact_summary,
                auxiliary_summary_available=auxiliary_summary_available,
                current_wrench=state.wrench,
            )
            step_logs.append(
                TeacherStepLog(
                    phase=controller.current_phase,  # type: ignore[arg-type]
                    sim_time=float(state.sim_time),
                    sim_tick=int(state.sim_tick),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    planner_rationale=close_range_reason or "",
                    trajectory_point={
                        "dt": float(env.task.hold_action_ticks) * 0.002,
                        "action": tuple(float(value) for value in np.asarray(action, dtype=np.float64).tolist()),
                        "target_tcp_pose": tuple(float(value) for value in np.asarray(state.tcp_pose, dtype=np.float64).tolist()),
                        "control_source": "close_range_policy",
                    },
                    dynamics_summary=history.dynamics_summary().to_dict(),
                    observation_summary=_observation_summary(
                        observation,
                        state=state,
                        include_images=include_images,
                        step_info=last_info,
                        target_tcp_pose=tuple(float(value) for value in np.asarray(state.tcp_pose, dtype=np.float64).tolist()),
                    ),
                    history_summary=history.teacher_memory_summary(),
                    data_quality=dict(history.latest().signal_quality),
                    auxiliary_force_contact_summary=auxiliary_force_contact_summary,
                    auxiliary_summary_available=auxiliary_summary_available,
                    auxiliary_contact_metrics=auxiliary_contact_metrics,
                    probe_result=probe_results[-1] if probe_results else None,
                ).to_dict()
            )
            if _check_teacher_max_steps() or _check_post_budget_stagnation(state):
                break
            continue

        plan, segment, candidates = controller.select_plan(
            scenario=scenario,
            task_id=task_id,
            state=state,
            temporal_buffer=history,
            recent_probe_results=probe_results,
            include_images=include_images,
        )
        planner_segment_count += 1
        planner_candidates.extend(candidates)
        if controller.should_probe(plan, history):
            before_state = deepcopy(state)
            before_history = deepcopy(history)
            for probe_action in controller.probe_library.actions_for(probe_name)[: controller.config.max_probe_actions]:
                observation, _, terminated, truncated, last_info = env.step(probe_action)
                state = env._state
                assert state is not None
                history.append(
                    state=state,
                    action=probe_action,
                    images=observation.get("images"),
                    image_timestamps=_image_timestamp_map(observation),
                    image_summaries=_image_summary_map(observation),
                    camera_info=observation.get("camera_info"),
                    signal_quality=build_signal_quality_snapshot(
                        state,
                        include_images=include_images,
                        camera_info=observation.get("camera_info"),
                    ),
                    auxiliary_force_contact_summary=_step_auxiliary_summary(
                        state=state,
                        step_info=last_info,
                    ),
                    auxiliary_summary_available="auxiliary_force_contact_summary" in last_info,
                )
                if trajectory_recorder is not None:
                    trajectory_recorder.capture(
                        observation=observation,
                        scenario=scenario,
                        state=state,
                    )
                if step_callback is not None:
                    step_callback(
                        kind="probe_step",
                        observation=observation,
                        reward=0.0,
                        terminated=terminated,
                        truncated=truncated,
                        info=last_info,
                        state=state,
                        action=probe_action,
                        planner_rationale=controller.last_rationale,
                        phase=controller.current_phase,
                    )
                if terminated or truncated:
                    break
                if _check_teacher_max_steps() or _check_post_budget_stagnation(state):
                    break
            probe_results.append(
                controller.probe_library.summarize_result(
                    probe_name=probe_name,
                    before_state=before_state,
                    after_state=state,
                    before_summary=before_history,
                    after_summary=history,
                    action_count=min(len(controller.probe_library.actions_for(probe_name)), controller.config.max_probe_actions),
                ).to_dict()
            )
            if terminated or truncated:
                break
        trajectory_segments.append(segment.to_dict())
        segment_metadata = dict(segment.conversion_metadata)
        for point in segment.points:
            if _check_teacher_max_steps():
                break
            state_before_action = env._state
            assert state_before_action is not None
            action, feedback_tracker_metadata = _feedback_track_dense_point(
                state=state_before_action,
                point=point,
                segment_metadata=segment_metadata,
            )
            observation, reward, terminated, truncated, last_info = env.step(action)
            state = env._state
            assert state is not None
            auxiliary_summary_available = "auxiliary_force_contact_summary" in last_info
            history.append(
                state=state,
                action=action,
                images=observation.get("images"),
                image_timestamps=_image_timestamp_map(observation),
                image_summaries=_image_summary_map(observation),
                camera_info=observation.get("camera_info"),
                signal_quality=build_signal_quality_snapshot(
                    state,
                    include_images=include_images,
                    camera_info=observation.get("camera_info"),
                ),
                auxiliary_force_contact_summary=_step_auxiliary_summary(
                    state=state,
                    step_info=last_info,
                ),
                auxiliary_summary_available=auxiliary_summary_available,
            )
            if trajectory_recorder is not None:
                trajectory_recorder.capture(
                    observation=observation,
                    scenario=scenario,
                    state=state,
                )
            if step_callback is not None:
                step_callback(
                    kind="segment_step",
                    observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=last_info,
                    state=state,
                    action=action,
                    planner_rationale=controller.last_rationale,
                    phase=controller.current_phase,
                )
            auxiliary_force_contact_summary = _step_auxiliary_summary(
                state=state,
                step_info=last_info,
            )
            auxiliary_contact_metrics = summarize_auxiliary_force_contact_summary(
                auxiliary_force_contact_summary=auxiliary_force_contact_summary,
                auxiliary_summary_available=auxiliary_summary_available,
                current_wrench=state.wrench,
            )
            step_logs.append(
                TeacherStepLog(
                    phase=controller.current_phase,  # type: ignore[arg-type]
                    sim_time=float(state.sim_time),
                    sim_tick=int(state.sim_tick),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    planner_rationale=controller.last_rationale or "",
                    trajectory_point={
                        **point.to_dict(),
                        "planned_action": [float(value) for value in point.action],
                        "action": [float(value) for value in action.tolist()],
                        **(
                            {"feedback_tracker": feedback_tracker_metadata}
                            if feedback_tracker_metadata is not None
                            else {}
                        ),
                        "executed_action": [float(value) for value in action.tolist()],
                    },
                    dynamics_summary=history.dynamics_summary().to_dict(),
                    observation_summary=_observation_summary(
                        observation,
                        state=state,
                        include_images=include_images,
                        step_info=last_info,
                        target_tcp_pose=point.target_tcp_pose,
                    ),
                    history_summary=history.teacher_memory_summary(),
                    data_quality=dict(history.latest().signal_quality),
                    auxiliary_force_contact_summary=auxiliary_force_contact_summary,
                    auxiliary_summary_available=auxiliary_summary_available,
                    auxiliary_contact_metrics=auxiliary_contact_metrics,
                    probe_result=probe_results[-1] if probe_results else None,
                ).to_dict()
            )
            if terminated or truncated:
                break
            if _check_teacher_max_steps() or _check_post_budget_stagnation(state):
                break
        if terminated or truncated:
            break
    total_wall_s = time.perf_counter() - start_wall
    final_history_summary = history.teacher_memory_summary()
    final_auxiliary_summary = dict(final_history_summary.get("auxiliary_history_summary", {}))
    final_data_quality = dict(history.latest().signal_quality)
    final_metrics = _final_metrics(last_info=last_info, step_logs=step_logs)
    rollout_log = TeacherRolloutLog(
        trial_id=scenario.trial_id,
        task_id=task_id,
        teacher_version="0.1.0",
        planner_backend=controller.planner.backend_name,
        seed=seed,
        scenario_metadata=dict(scenario.metadata),
        timing={
            "total_wall_s": total_wall_s,
            "step_count": len(step_logs),
            "segment_count": len(trajectory_segments),
            "planner_segment_count": planner_segment_count,
            "final_sim_time": float(state.sim_time),
        },
        initial_observation_summary={
            "sim_tick": int(initial_observation["sim_tick"]) if "sim_tick" in initial_observation else 0,
            "sim_time": float(initial_observation["sim_time"]) if "sim_time" in initial_observation else 0.0,
            "distance_to_target": float(initial_observation["plug_to_port_relative"][3]),
        },
        data_quality=final_data_quality,
        history_metadata=final_history_summary,
        auxiliary_summary_metadata=final_auxiliary_summary,
        planner_candidates=planner_candidates,
        probe_results=probe_results,
        trajectory_segments=trajectory_segments,
        step_logs=step_logs,
        final_info=last_info,
        limitations=limitations,
    )
    artifact = TeacherReplayArtifact(
        metadata={
            "trial_id": scenario.trial_id,
            "task_id": task_id,
            "seed": seed,
            "planner_backend": controller.planner.backend_name,
            "teacher_version": "0.1.0",
            "include_images": include_images,
            "hold_action_ticks": env.task.hold_action_ticks,
            "scenario_metadata": dict(scenario.metadata),
            "task_metadata": _task_metadata(scenario, task_id),
            "data_quality": final_data_quality,
            "initial_observation_summary": {
                "sim_tick": int(initial_observation["sim_tick"]) if "sim_tick" in initial_observation else 0,
                "sim_time": float(initial_observation["sim_time"]) if "sim_time" in initial_observation else 0.0,
                "distance_to_target": float(initial_observation["plug_to_port_relative"][3]),
            },
            "history_metadata": final_history_summary,
            "auxiliary_summary_metadata": final_auxiliary_summary,
            "auxiliary_summary_available": bool(final_auxiliary_summary.get("auxiliary_summary_available", False)),
            "planner_metadata": _planner_metadata(controller),
            "final_metrics": final_metrics,
            "run_until_env_done": bool(controller.config.run_until_env_done),
            "max_env_steps": int(controller.config.max_env_steps),
            "planner_segment_count": int(planner_segment_count),
        },
        trajectory_segments=trajectory_segments,
        probe_results=probe_results,
        planner_candidates=planner_candidates,
        step_logs=step_logs,
        final_info=last_info,
        limitations=limitations,
    )
    if output_path is not None:
        save_teacher_replay(artifact, output_path)
    return TeacherRolloutResult(artifact=artifact, output_path=output_path)


def _image_timestamp_map(observation: dict[str, Any]) -> dict[str, float]:
    if "images" not in observation or "image_timestamps" not in observation:
        return {}
    timestamps = np.asarray(observation["image_timestamps"], dtype=np.float64)
    names = sorted(observation["images"].keys())
    return {name: float(timestamps[index]) for index, name in enumerate(names)}


def _image_summary_map(observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if "images" not in observation or "image_timestamps" not in observation:
        return {}
    return summarize_image_batch(observation["images"], _image_timestamp_map(observation))


def _observation_summary(
    observation: dict[str, Any],
    *,
    state,
    include_images: bool,
    step_info: dict[str, Any] | None = None,
    target_tcp_pose: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    data_quality = build_signal_quality_snapshot(
        state,
        include_images=include_images,
        camera_info=observation.get("camera_info"),
    )
    summary = {
        "sim_tick": int(observation["sim_tick"]),
        "sim_time": float(observation["sim_time"]),
        "joint_positions": np.asarray(observation["joint_positions"], dtype=np.float64).tolist(),
        "joint_velocities": np.asarray(observation["joint_velocities"], dtype=np.float64).tolist(),
        "tcp_pose": np.asarray(observation["tcp_pose"], dtype=np.float64).tolist(),
        "tcp_velocity": np.asarray(observation["tcp_velocity"], dtype=np.float64).tolist(),
        "plug_pose": np.asarray(observation["plug_pose"], dtype=np.float64).tolist(),
        "target_port_pose": np.asarray(observation["target_port_pose"], dtype=np.float64).tolist(),
        "target_port_entrance_pose": np.asarray(
            observation["target_port_entrance_pose"], dtype=np.float64
        ).tolist(),
        "wrench": np.asarray(observation["wrench"], dtype=np.float64).tolist(),
        "wrench_timestamp": float(np.asarray(observation["wrench_timestamp"], dtype=np.float64).reshape(-1)[0]),
        "off_limit_contact": bool(np.asarray(observation["off_limit_contact"]).reshape(-1)[0] > 0.5),
        "plug_to_port_relative": np.asarray(observation["plug_to_port_relative"], dtype=np.float64).tolist(),
        "controller_tcp_pose": np.asarray(observation["controller_tcp_pose"], dtype=np.float64).tolist(),
        "controller_reference_tcp_pose": np.asarray(
            observation["controller_reference_tcp_pose"], dtype=np.float64
        ).tolist(),
        "controller_tcp_velocity": np.asarray(
            observation["controller_tcp_velocity"], dtype=np.float64
        ).tolist(),
        "controller_tcp_error": np.asarray(observation["controller_tcp_error"], dtype=np.float64).tolist(),
        "controller_reference_joint_state": np.asarray(
            observation["controller_reference_joint_state"], dtype=np.float64
        ).tolist(),
        "controller_target_mode": float(
            np.asarray(observation["controller_target_mode"], dtype=np.float64).reshape(-1)[0]
        ),
        "fts_tare_wrench": np.asarray(observation["fts_tare_wrench"], dtype=np.float64).tolist(),
        "score_geometry": serialize_nested(observation.get("score_geometry", {})),
        "world_entities_summary": serialize_nested(state.world_entities_summary),
        "controller_state_summary": controller_state_summary(state.controller_state),
        "data_quality": data_quality,
        "auxiliary_summary_available": bool(step_info and "auxiliary_force_contact_summary" in step_info),
    }
    if "images" in observation:
        summary["image_summaries"] = _image_summary_map(observation)
        summary["image_timestamps"] = _image_timestamp_map(observation)
        summary["camera_info"] = summarize_camera_info(observation.get("camera_info"))
    if target_tcp_pose is not None:
        target_array = np.asarray(target_tcp_pose, dtype=np.float64)
        tcp_pose = np.asarray(observation["tcp_pose"], dtype=np.float64)
        summary["step_target_tcp_pose"] = target_array.tolist()
        summary["step_target_tcp_position_error_m"] = float(np.linalg.norm(tcp_pose[:3] - target_array[:3]))
        if tcp_pose.shape[0] >= 7 and target_array.shape[0] >= 7:
            summary["step_target_tcp_orientation_error_l2"] = float(np.linalg.norm(tcp_pose[3:7] - target_array[3:7]))
    return summary


def _step_auxiliary_summary(*, state, step_info: dict[str, Any]) -> dict[str, Any]:
    if "auxiliary_force_contact_summary" not in step_info:
        return synthetic_auxiliary_force_contact_summary(state)
    return normalize_auxiliary_force_contact_summary(step_info.get("auxiliary_force_contact_summary"))


def _planner_metadata(controller: AgentTeacherController) -> dict[str, Any]:
    metadata = {
        "backend_name": controller.planner.backend_name,
        "candidate_plan_count": controller.config.candidate_plan_count,
        "segment_limit": controller.config.segment_limit,
        "max_planner_calls_per_episode": controller.config.max_planner_calls_per_episode,
        "run_until_env_done": controller.config.run_until_env_done,
        "max_env_steps": controller.config.max_env_steps,
        "hold_ticks_per_action": controller.config.hold_ticks_per_action,
        "enable_probes": controller.config.enable_probes,
        "planner_call_count": controller.planner_call_count,
    }
    planner_config = getattr(controller.planner, "config", None)
    if planner_config is not None:
        metadata["backend_config"] = {
            key: value
            for key, value in vars(planner_config).items()
            if "api_key" not in key.lower()
        }
    return metadata


def _task_metadata(scenario, task_id: str) -> dict[str, Any]:
    task = scenario.tasks[task_id]
    return {
        "plug_name": task.plug_name,
        "cable_name": task.cable_name,
        "cable_type": task.cable_type,
        "target_module_name": task.target_module_name,
        "port_name": task.port_name,
    }


def _final_metrics(*, last_info: dict[str, Any], step_logs: list[dict[str, Any]]) -> dict[str, Any]:
    final_evaluation = dict(last_info.get("final_evaluation") or {})
    reward_total = float(
        final_evaluation.get("training_reward_total", sum(float(step["reward"]) for step in step_logs))
    )
    gym_final_score = final_evaluation.get("gym_final_score")
    return {
        "rl_step_reward_total": reward_total,
        "gym_final_score": None if gym_final_score is None else float(gym_final_score),
        "official_eval_score": final_evaluation.get("official_eval_score"),
    }
