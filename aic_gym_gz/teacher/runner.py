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
from .replay import TeacherReplayArtifact, save_teacher_replay
from .types import TeacherRolloutLog, TeacherStepLog


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
) -> TeacherRolloutResult:
    observation, info = env.reset(seed=seed, options={"trial_id": trial_id} if trial_id else {})
    initial_observation = dict(observation)
    scenario = env._scenario
    state = env._state
    if scenario is None or state is None:
        raise RuntimeError("Environment did not expose scenario/state after reset.")
    task_id = next(iter(scenario.tasks.keys()))
    include_images = env.task.include_images
    history = TemporalObservationBuffer()
    history.append(
        state=state,
        action=controller.initial_action(),
        images=observation.get("images"),
        image_timestamps=_image_timestamp_map(observation),
        image_summaries=_image_summary_map(observation),
    )
    start_wall = time.perf_counter()
    probe_results: list[dict[str, Any]] = []
    step_logs: list[dict[str, Any]] = []
    trajectory_segments: list[dict[str, Any]] = []
    planner_candidates: list[dict[str, Any]] = []
    terminated = truncated = False
    last_info: dict[str, Any] = dict(info)
    limitations = [
        "Exact mid-rollout simulator state cloning is not available in the current stack.",
        "Branch-and-evaluate currently reuses deterministic planner variants instead of full simulator forks.",
        "Replay is exact for deterministic mock resets and best-effort for live Gazebo runs.",
    ]
    for _ in range(controller.config.segment_limit):
        plan, segment, candidates = controller.select_plan(
            scenario=scenario,
            task_id=task_id,
            state=state,
            temporal_buffer=history,
            recent_probe_results=probe_results,
            include_images=include_images,
        )
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
                )
                if terminated or truncated:
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
        for point in segment.points:
            action = np.asarray(point.action, dtype=np.float32)
            observation, reward, terminated, truncated, last_info = env.step(action)
            state = env._state
            assert state is not None
            history.append(
                state=state,
                action=action,
                images=observation.get("images"),
                image_timestamps=_image_timestamp_map(observation),
                image_summaries=_image_summary_map(observation),
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
                    trajectory_point=point.to_dict(),
                    dynamics_summary=history.dynamics_summary().to_dict(),
                    observation_summary=_observation_summary(observation),
                    probe_result=probe_results[-1] if probe_results else None,
                ).to_dict()
            )
            if terminated or truncated:
                break
        if terminated or truncated:
            break
    total_wall_s = time.perf_counter() - start_wall
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
            "final_sim_time": float(state.sim_time),
        },
        initial_observation_summary={
            "sim_tick": int(initial_observation["sim_tick"]) if "sim_tick" in initial_observation else 0,
            "sim_time": float(initial_observation["sim_time"]) if "sim_time" in initial_observation else 0.0,
            "distance_to_target": float(initial_observation["plug_to_port_relative"][3]),
        },
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


def _observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "sim_tick": int(observation["sim_tick"]),
        "sim_time": float(observation["sim_time"]),
        "joint_positions": np.asarray(observation["joint_positions"], dtype=np.float64).tolist(),
        "joint_velocities": np.asarray(observation["joint_velocities"], dtype=np.float64).tolist(),
        "tcp_pose": np.asarray(observation["tcp_pose"], dtype=np.float64).tolist(),
        "tcp_velocity": np.asarray(observation["tcp_velocity"], dtype=np.float64).tolist(),
        "plug_pose": np.asarray(observation["plug_pose"], dtype=np.float64).tolist(),
        "target_port_pose": np.asarray(observation["target_port_pose"], dtype=np.float64).tolist(),
        "wrench": np.asarray(observation["wrench"], dtype=np.float64).tolist(),
        "off_limit_contact": bool(np.asarray(observation["off_limit_contact"]).reshape(-1)[0] > 0.5),
        "plug_to_port_relative": np.asarray(observation["plug_to_port_relative"], dtype=np.float64).tolist(),
    }
    if "images" in observation:
        summary["image_summaries"] = _image_summary_map(observation)
        summary["image_timestamps"] = _image_timestamp_map(observation)
    return summary
