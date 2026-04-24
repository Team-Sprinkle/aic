from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher.context import TeacherContextExtractor
from aic_gym_gz.teacher.history import TemporalObservationBuffer
from aic_gym_gz.teacher.policy import AgentTeacherController
from aic_gym_gz.teacher.quality import build_signal_quality_snapshot
from aic_gym_gz.teacher.types import TeacherPlan, TeacherPlanningState, TeacherWaypoint
from aic_gym_gz.teacher.visual_context import build_scene_overview_images


class TeacherPlannerTest(unittest.TestCase):
    def test_mock_planner_returns_structured_segment_plan(self) -> None:
        backend = DeterministicMockPlannerBackend()
        state = TeacherPlanningState(
            trial_id="trial_0",
            task_id="task_0",
            goal_summary="goal",
            current_phase="free_space_approach",
            policy_context={
                "plug_pose": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "target_port_pose": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
                "target_port_entrance_pose": [0.0, 0.0, 0.92, 0.0, 0.0, 0.0, 1.0],
                "off_limit_contact": False,
                "distance_to_target": 0.1,
                "distance_to_entrance": 0.12,
                "lateral_misalignment": 0.015,
                "orientation_error": 0.1,
                "insertion_progress": 0.0,
            },
            oracle_context={},
            obstacle_summary=[],
            dynamics_summary={
                "quasi_static": True,
                "cable_settling_score": 0.9,
            },
            image_refs=[],
            image_timestamps={},
            image_summaries={},
            recent_probe_results=[],
            temporal_context={
                "phase_guidance": {
                    "recommended_phase": "pre_insert_align",
                    "alignment_needed": True,
                    "in_insertion_zone": False,
                    "insertion_ready": False,
                },
                "auxiliary_history_summary": {"hidden_contact_recent": False},
            },
        )
        plan = backend.plan(state)
        self.assertGreaterEqual(len(plan.waypoints), 1)
        self.assertIn(plan.motion_mode, {"coarse_cartesian", "fine_cartesian", "guarded_insert"})

    def test_mock_planner_can_progress_to_guarded_insert_near_entrance(self) -> None:
        backend = DeterministicMockPlannerBackend()
        state = TeacherPlanningState(
            trial_id="trial_0",
            task_id="task_0",
            goal_summary="goal",
            current_phase="pre_insert_align",
            policy_context={
                "plug_pose": [0.0, 0.0, 0.905, 0.0, 0.0, 0.0, 1.0],
                "target_port_pose": [0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0],
                "target_port_entrance_pose": [0.0, 0.0, 0.91, 0.0, 0.0, 0.0, 1.0],
                "off_limit_contact": False,
                "distance_to_target": 0.005,
                "distance_to_entrance": 0.01,
                "lateral_misalignment": 0.002,
                "orientation_error": 0.01,
                "insertion_progress": 0.3,
            },
            oracle_context={},
            obstacle_summary=[],
            dynamics_summary={"quasi_static": True, "cable_settling_score": 0.95},
            image_refs=[],
            image_timestamps={},
            image_summaries={},
            recent_probe_results=[],
            temporal_context={
                "phase_guidance": {
                    "recommended_phase": "guarded_insert",
                    "insertion_ready": True,
                    "in_insertion_zone": True,
                },
                "auxiliary_history_summary": {"hidden_contact_recent": False},
            },
        )
        plan = backend.plan(state, candidate_index=3)
        self.assertEqual(plan.next_phase, "guarded_insert")

    def test_context_extractor_surfaces_controller_and_quality_metadata(self) -> None:
        env = make_default_env(enable_randomization=True, include_images=False)
        try:
            observation, _ = env.reset(seed=123)
            assert env._scenario is not None
            assert env._state is not None
            task_id = next(iter(env._scenario.tasks.keys()))
            history = TemporalObservationBuffer()
            history.append(
                state=env._state,
                action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                signal_quality=build_signal_quality_snapshot(
                    env._state,
                    include_images=False,
                    camera_info=observation.get("camera_info"),
                ),
            )
            planning_state = TeacherContextExtractor().build_planning_state(
                scenario=env._scenario,
                task_id=task_id,
                state=env._state,
                temporal_buffer=history,
                current_phase="free_space_approach",
                recent_probe_results=[],
                include_images=False,
            )
        finally:
            env.close()
        self.assertIn("controller_state", planning_state.controller_context)
        self.assertIn("wrench", planning_state.data_quality)
        self.assertEqual(planning_state.temporal_context["window_size"], 1)
        self.assertIn("auxiliary_history_summary", planning_state.temporal_context)
        self.assertIn("wrench_contact_trend_summary", planning_state.temporal_context)
        self.assertIn("auxiliary_force_contact_summary", planning_state.policy_context)
        self.assertIn("relative_geometry", planning_state.policy_context)
        self.assertIn("frame_context", planning_state.policy_context)
        self.assertIn("world_entities_summary", planning_state.policy_context)
        self.assertIn("scene_overview_sources", planning_state.planning_metadata)
        self.assertIn("recent_visual_timepoints", planning_state.camera_context)

    def test_global_guidance_bonus_prefers_matching_phase_and_milestone(self) -> None:
        controller = AgentTeacherController(planner=DeterministicMockPlannerBackend())
        plan = TeacherPlan(
            next_phase="pre_insert_align",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.92)),),
            motion_mode="fine_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=6,
            segment_granularity="fine",
            rationale_summary="test",
        )
        bonus_phase, bonus_milestone = controller._global_guidance_terms(
            plan=plan,
            policy_context={"plug_pose": [0.0, 0.0, 1.0], "target_port_pose": [0.0, 0.0, 0.9]},
            temporal_context={
                "phase_guidance": {"current_phase": "free_space_approach"},
                "global_guidance": {
                    "phase_sequence": ["free_space_approach", "pre_insert_align", "guarded_insert"],
                    "milestones": [
                        {"name": "entrance_setup", "phase": "pre_insert_align", "position_xyz": [0.0, 0.0, 0.92]},
                    ],
                },
            },
        )
        self.assertGreater(bonus_phase, 0.0)
        self.assertGreater(bonus_milestone, 0.0)

    def test_live_overview_images_replace_matching_scene_entries(self) -> None:
        env = make_default_env(enable_randomization=True, include_images=False)
        try:
            env.reset(seed=123)
            assert env._scenario is not None
            assert env._state is not None
            live_images = {
                "top_down_xy": np.full((256, 256, 3), 127, dtype=np.uint8),
                "front_xz": np.full((256, 256, 3), 64, dtype=np.uint8),
            }
            scene_images = build_scene_overview_images(
                scenario=env._scenario,
                state=env._state,
                live_images_by_view=live_images,
            )
        finally:
            env.close()
        self.assertEqual(scene_images[0]["view_name"], "top_down_xy")
        self.assertEqual(scene_images[0]["source"], "live_overview_topic")
        self.assertEqual(scene_images[1]["source"], "live_overview_topic")
        self.assertEqual(scene_images[2]["source"], "teacher_schematic_scene_overview")
        self.assertEqual(scene_images[3]["source"], "teacher_schematic_scene_overview")


if __name__ == "__main__":
    unittest.main()
