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
from .visual_context import (
    build_recent_visual_observations,
    build_scene_overview_images,
    latest_live_overview_images,
)


@dataclass(frozen=True)
class TeacherContextExtractor:
    """Builds compact planner inputs from policy and oracle context."""

    max_recent_visual_frames: int = 12
    prefer_live_scene_overview: bool = False

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
            "relative_geometry": self._relative_geometry_summary(state),
            "frame_context": self._frame_context(),
            "world_entities_summary": serialize_nested(state.world_entities_summary),
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
            "world_entities_summary": serialize_nested(state.world_entities_summary),
        }
        latest_frame = temporal_buffer.latest()
        image_refs = list(latest_frame.image_refs) if include_images else []
        recent_visual_observations = build_recent_visual_observations(
            frames=temporal_buffer.recent_visual_frames(max_frames=self.max_recent_visual_frames),
            max_frames=self.max_recent_visual_frames,
        ) if include_images else []
        live_overview_images = (
            latest_live_overview_images(image_size=(256, 256))
            if self.prefer_live_scene_overview
            else {}
        )
        scene_overview_images = build_scene_overview_images(
            scenario=scenario,
            state=state,
            live_images_by_view=live_overview_images,
        )
        scene_overview_sources = {
            str(item.get("view_name")): str(item.get("source"))
            for item in scene_overview_images
        }
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
                "recent_visual_timepoints": [
                    {
                        "label": item.get("label"),
                        "camera_name": item.get("camera_name"),
                        "sim_tick": item.get("sim_tick"),
                        "sim_time": item.get("sim_time"),
                        "timestamp": item.get("timestamp"),
                        "age_from_latest_s": item.get("age_from_latest_s"),
                        "age_from_latest_steps": item.get("age_from_latest_steps"),
                        "timepoint_label": item.get("timepoint_label"),
                    }
                    for item in recent_visual_observations
                ],
            },
            temporal_context={
                **temporal_context,
                "phase_guidance": phase_guidance,
            },
            data_quality=dict(latest_frame.signal_quality),
            planning_metadata={
                "frame_context": self._frame_context(),
                "include_images": bool(include_images),
                "recent_visual_frame_count": len(recent_visual_observations),
                "scene_overview_image_count": len(scene_overview_images),
                "scene_overview_sources": scene_overview_sources,
                "available_scene_overview_views": [
                    str(item.get("view_name")) for item in scene_overview_images
                ],
                "scene_overview_live_source_used": bool(
                    any(item.get("source") == "live_overview_topic" for item in scene_overview_images)
                ),
                "prefer_live_scene_overview": self.prefer_live_scene_overview,
                "recent_visual_frame_budget": self.max_recent_visual_frames,
                "history_window_size": len(temporal_buffer),
                "teacher_history_is_additive": True,
                "official_current_observation_only": temporal_buffer.current_observation_view(),
                "official_observation_contract_unchanged": True,
                "auxiliary_force_contact_summary_is_teacher_side": True,
                "phase_guidance": phase_guidance,
            },
            last_teacher_rationale=last_teacher_rationale,
        )

    def _relative_geometry_summary(self, state: RuntimeState) -> dict[str, Any]:
        tcp = np.asarray(state.tcp_pose[:3], dtype=np.float64)
        plug = np.asarray(state.plug_pose[:3], dtype=np.float64)
        target = np.asarray(state.target_port_pose[:3], dtype=np.float64)
        entrance = (
            target
            if state.target_port_entrance_pose is None
            else np.asarray(state.target_port_entrance_pose[:3], dtype=np.float64)
        )
        insertion_axis = target - entrance
        insertion_axis_norm = float(np.linalg.norm(insertion_axis))
        insertion_axis_unit = (
            insertion_axis / insertion_axis_norm
            if insertion_axis_norm > 1e-8
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        plug_to_entrance = entrance - plug
        tcp_to_entrance = entrance - tcp
        plug_to_target = target - plug
        tcp_to_target = target - tcp
        plug_to_tcp = tcp - plug
        axial_error_to_entrance = float(np.dot(plug_to_entrance, insertion_axis_unit))
        lateral_error_to_entrance = float(
            np.linalg.norm(plug_to_entrance - axial_error_to_entrance * insertion_axis_unit)
        )
        return {
            "tcp_to_target_port_xyz": tcp_to_target.astype(float).tolist(),
            "tcp_to_target_port_distance": float(np.linalg.norm(tcp_to_target)),
            "tcp_to_entrance_xyz": tcp_to_entrance.astype(float).tolist(),
            "tcp_to_entrance_distance": float(np.linalg.norm(tcp_to_entrance)),
            "plug_to_target_port_xyz": plug_to_target.astype(float).tolist(),
            "plug_to_target_port_distance": float(np.linalg.norm(plug_to_target)),
            "plug_to_entrance_xyz": plug_to_entrance.astype(float).tolist(),
            "plug_to_entrance_distance": float(np.linalg.norm(plug_to_entrance)),
            "plug_to_tcp_xyz": plug_to_tcp.astype(float).tolist(),
            "plug_to_tcp_distance": float(np.linalg.norm(plug_to_tcp)),
            "insertion_axis_world_xyz": insertion_axis_unit.astype(float).tolist(),
            "axial_error_to_entrance_m": axial_error_to_entrance,
            "lateral_error_to_entrance_m": lateral_error_to_entrance,
        }

    def _frame_context(self) -> dict[str, Any]:
        return {
            "runtime_pose_frame": "world",
            "runtime_action_command_frame": "world",
            "official_policy_reference_frame": "base_link",
            "planner_waypoint_space": "plug_position_in_runtime_pose_frame",
            "smoother_target_space": "tcp_pose_in_runtime_pose_frame",
            "controller_state_frame": "runtime_controller_topic_unspecified",
            "notes": [
                "Teacher payload poses come from runtime/Gazebo state and behave like world-frame poses.",
                "Official CheatCode and policy docs commonly reason in base_link.",
                "Controller topic fields may use controller-local conventions unless explicitly documented elsewhere.",
            ],
        }

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
