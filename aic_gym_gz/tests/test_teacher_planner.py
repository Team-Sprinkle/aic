from __future__ import annotations

import unittest

from aic_gym_gz.planners.mock import DeterministicMockPlannerBackend
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


if __name__ == "__main__":
    unittest.main()
