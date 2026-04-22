from __future__ import annotations

import unittest

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
from aic_gym_gz.teacher.context import TeacherContextExtractor
from aic_gym_gz.teacher.history import TemporalObservationBuffer
from aic_gym_gz.teacher.quality import build_signal_quality_snapshot
from aic_gym_gz.teacher.types import TeacherPlanningState


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
                "off_limit_contact": False,
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
        )
        plan = backend.plan(state)
        self.assertGreaterEqual(len(plan.waypoints), 1)
        self.assertIn(plan.motion_mode, {"coarse_cartesian", "fine_cartesian", "guarded_insert"})

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


if __name__ == "__main__":
    unittest.main()
