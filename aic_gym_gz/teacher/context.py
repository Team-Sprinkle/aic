"""Teacher-mode context extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..runtime import RuntimeState
from ..scenario import AicScenario
from .history import TemporalObservationBuffer
from .planning import phase_guidance_from_state
from .quality import controller_state_summary, serialize_nested
from .types import ObstacleSummary, TeacherPlanningState
from .visual_context import build_recent_visual_observations, build_scene_overview_images


@dataclass(frozen=True)
class TeacherContextExtractor:
    """Builds compact planner inputs from policy and oracle context."""

    def build_planning_state(
        self,
        *,
        scenario: AicScenario,
        task_id: str,
        state: RuntimeState,
        temporal_buffer: TemporalObservationBuffer,
        current_phase: str,
        recent_probe_results: list[dict[str, Any]],
        include_images: bool,
        last_teacher_rationale: str | None = None,
    ) -> TeacherPlanningState:
        task = scenario.tasks[task_id]
        policy_context = {
            "sim_tick": int(state.sim_tick),
            "sim_time": float(state.sim_time),
            "tcp_pose": state.tcp_pose.astype(float).tolist(),
            "tcp_velocity": state.tcp_velocity.astype(float).tolist(),
            "plug_pose": state.plug_pose.astype(float).tolist(),
            "target_port_pose": state.target_port_pose.astype(float).tolist(),
            "target_port_entrance_pose": (
                None
                if state.target_port_entrance_pose is None
                else state.target_port_entrance_pose.astype(float).tolist()
            ),
            "wrench": state.wrench.astype(float).tolist(),
            "official_current_wrench": state.wrench.astype(float).tolist(),
            "wrench_timestamp": float(state.wrench_timestamp),
            "off_limit_contact": bool(state.off_limit_contact),
            "distance_to_target": float(np.linalg.norm(state.plug_pose[:3] - state.target_port_pose[:3])),
            "distance_to_entrance": float(state.score_geometry.get("distance_to_entrance", 0.0) or 0.0),
            "lateral_misalignment": float(state.score_geometry.get("lateral_misalignment", 0.0) or 0.0),
            "orientation_error": float(state.score_geometry.get("orientation_error", 0.0) or 0.0),
            "insertion_progress": float(state.score_geometry.get("insertion_progress", 0.0) or 0.0),
            "partial_insertion": bool(state.score_geometry.get("partial_insertion", False)),
            "score_geometry": serialize_nested(state.score_geometry),
            "auxiliary_force_contact_summary": temporal_buffer.auxiliary_history_summary(max_items=1),
        }
        oracle_context = {
            "task_board_pose_xyz_rpy": list(scenario.task_board.pose_xyz_rpy),
            "target_port_pose": state.target_port_pose.astype(float).tolist(),
            "target_port_entrance_pose": (
                None
                if state.target_port_entrance_pose is None
                else state.target_port_entrance_pose.astype(float).tolist()
            ),
            "plug_pose": state.plug_pose.astype(float).tolist(),
            "board_pose": list(scenario.task_board.pose_xyz_rpy),
            "cable": {
                "name": task.cable_name,
                "type": task.cable_type,
                "scenario": scenario.cables[task.cable_name].__dict__,
            },
            "clearance_summary": self._clearance_summary(scenario),
            "scene_layout_summary": self._scene_layout_summary(scenario, state),
        }
        latest_frame = temporal_buffer.latest()
        image_refs = list(latest_frame.image_refs) if include_images else []
        recent_visual_observations = build_recent_visual_observations(
            frames=temporal_buffer.recent_visual_frames(max_frames=2),
            max_frames=2,
        ) if include_images else []
        scene_overview_images = build_scene_overview_images(
            scenario=scenario,
            state=state,
        )
        obstacle_summary = [obstacle.to_dict() for obstacle in self._obstacle_summary(scenario)]
        temporal_context = temporal_buffer.teacher_memory_summary()
        phase_guidance = phase_guidance_from_state(
            current_phase=current_phase,
            policy_context=policy_context,
            temporal_context=temporal_context,
            obstacle_summary=obstacle_summary,
        )
        return TeacherPlanningState(
            trial_id=scenario.trial_id,
            task_id=task_id,
            goal_summary=(
                f"Insert {task.plug_name} into {task.target_module_name}/{task.port_name} "
                f"for cable type {task.cable_type}."
            ),
            current_phase=current_phase,  # type: ignore[arg-type]
            policy_context=policy_context,
            oracle_context=oracle_context,
            obstacle_summary=obstacle_summary,
            dynamics_summary=temporal_buffer.dynamics_summary().to_dict(),
            image_refs=image_refs,
            image_timestamps=dict(latest_frame.image_timestamps) if include_images else {},
            image_summaries=dict(latest_frame.image_summaries) if include_images else {},
            recent_probe_results=recent_probe_results[-4:],
            recent_visual_observations=recent_visual_observations,
            scene_overview_images=scene_overview_images,
            controller_context={
                "controller_state": controller_state_summary(state.controller_state),
                "reference_tcp_pose": serialize_nested(state.controller_state.get("reference_tcp_pose")),
                "tcp_error": serialize_nested(state.controller_state.get("tcp_error")),
                "controller_target_mode": serialize_nested(state.controller_state.get("target_mode")),
            },
            camera_context={
                "camera_info": serialize_nested(latest_frame.camera_info) if include_images else {},
                "image_refs": image_refs,
                "image_timestamps": dict(latest_frame.image_timestamps) if include_images else {},
            },
            temporal_context={
                **temporal_context,
                "phase_guidance": phase_guidance,
            },
            data_quality=dict(latest_frame.signal_quality),
            planning_metadata={
                "include_images": bool(include_images),
                "recent_visual_frame_count": len(recent_visual_observations),
                "scene_overview_image_count": len(scene_overview_images),
                "history_window_size": len(temporal_buffer),
                "teacher_history_is_additive": True,
                "official_current_observation_only": temporal_buffer.current_observation_view(),
                "official_observation_contract_unchanged": True,
                "auxiliary_force_contact_summary_is_teacher_side": True,
                "phase_guidance": phase_guidance,
            },
            last_teacher_rationale=last_teacher_rationale,
        )

    def _obstacle_summary(self, scenario: AicScenario) -> list[ObstacleSummary]:
        obstacles: list[ObstacleSummary] = []
        for mapping in (
            scenario.task_board.nic_rails,
            scenario.task_board.sc_rails,
            scenario.task_board.mount_rails,
        ):
            for name, entity in mapping.items():
                obstacles.append(
                    ObstacleSummary(
                        object_name=name,
                        clearance_hint=0.04 if entity.present else 0.12,
                        present=bool(entity.present),
                        pose_hint=(entity.translation, 0.0, 0.0, entity.roll, entity.pitch, entity.yaw),
                    )
                )
        return obstacles

    def _clearance_summary(self, scenario: AicScenario) -> dict[str, float]:
        present_count = sum(
            int(entity.present)
            for mapping in (
                scenario.task_board.nic_rails,
                scenario.task_board.sc_rails,
                scenario.task_board.mount_rails,
            )
            for entity in mapping.values()
        )
        return {
            "present_obstacle_count": float(present_count),
            "nominal_open_space_clearance_m": 0.12,
            "nominal_insert_clearance_m": 0.01,
        }

    def _scene_layout_summary(self, scenario: AicScenario, state: RuntimeState) -> dict[str, Any]:
        board_pose = np.asarray(scenario.task_board.pose_xyz_rpy[:3], dtype=np.float64)
        present_obstacles: list[dict[str, Any]] = []
        for category, lane_offset, mapping in (
            ("nic", -0.06, scenario.task_board.nic_rails),
            ("sc", 0.0, scenario.task_board.sc_rails),
            ("mount", 0.06, scenario.task_board.mount_rails),
        ):
            for name, entity in mapping.items():
                if not entity.present:
                    continue
                approximate_xyz = [
                    float(board_pose[0] + entity.translation),
                    float(board_pose[1] + lane_offset),
                    float(board_pose[2]),
                ]
                present_obstacles.append(
                    {
                        "name": name,
                        "category": category,
                        "approximate_world_xyz": approximate_xyz,
                        "relative_translation_along_board_m": float(entity.translation),
                        "approximate_yaw": float(entity.yaw),
                    }
                )
        return {
            "board_pose_xyz_rpy": list(scenario.task_board.pose_xyz_rpy),
            "plug_pose_xyz": state.plug_pose[:3].astype(float).tolist(),
            "target_port_pose_xyz": state.target_port_pose[:3].astype(float).tolist(),
            "target_port_entrance_pose_xyz": (
                None
                if state.target_port_entrance_pose is None
                else state.target_port_entrance_pose[:3].astype(float).tolist()
            ),
            "present_obstacles": present_obstacles,
        }
