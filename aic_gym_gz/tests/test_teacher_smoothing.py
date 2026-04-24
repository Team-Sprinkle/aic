from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.runtime import RuntimeState
from aic_gym_gz.teacher.types import TeacherPlan, TeacherWaypoint
from aic_gym_gz.trajectory.smoothing import MinimumJerkSmoother


class TeacherSmoothingTest(unittest.TestCase):
    def _state(self) -> RuntimeState:
        return RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_pose=np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_entrance_pose=np.array([0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            wrench=np.zeros(6, dtype=np.float64),
            off_limit_contact=False,
        )

    def test_minimum_jerk_segment_is_dense_and_continuous(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        plan = TeacherPlan(
            next_phase="pre_insert_align",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.95), yaw=0.0, speed_scale=0.5),),
            motion_mode="fine_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=64,
            segment_granularity="fine",
            rationale_summary="test",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertGreater(len(segment.points), 1)
        first = np.asarray(segment.points[0].target_tcp_pose)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertGreater(first[2], 0.0)
        self.assertAlmostEqual(last[2], 0.95, places=4)

    def test_segment_horizon_steps_limits_dense_segment_length(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        state.target_port_pose[2] = 0.8
        state.target_port_entrance_pose[2] = 0.85
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(
                TeacherWaypoint(position_xyz=(0.0, 0.0, 0.95), yaw=0.0, speed_scale=0.4),
                TeacherWaypoint(position_xyz=(0.0, 0.0, 0.85), yaw=0.0, speed_scale=0.4),
            ),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=4,
            segment_granularity="coarse",
            rationale_summary="test",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertLessEqual(len(segment.points), 4)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertLess(last[2], 0.9)

    def test_segment_horizon_steps_preserves_meaningful_progress(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        state.target_port_pose[2] = 0.75
        state.target_port_entrance_pose[2] = 0.8
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(
                TeacherWaypoint(position_xyz=(0.0, 0.0, 0.92), yaw=0.0, speed_scale=0.25),
                TeacherWaypoint(position_xyz=(0.0, 0.0, 0.82), yaw=0.0, speed_scale=0.25),
                TeacherWaypoint(position_xyz=(0.0, 0.0, 0.76), yaw=0.0, speed_scale=0.25),
            ),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=5,
            segment_granularity="coarse",
            rationale_summary="test",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertEqual(len(segment.points), 5)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertLess(last[2], 0.8)

    def test_delta_cartesian_waypoint_mode_logs_conversion_metadata(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02, planner_output_mode="delta_cartesian_waypoint")
        state = self._state()
        plan = TeacherPlan(
            next_phase="pre_insert_align",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, -0.05), yaw=0.1, speed_scale=0.5),),
            motion_mode="fine_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=8,
            segment_granularity="fine",
            rationale_summary="delta",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertEqual(segment.conversion_metadata["planner_output_mode"], "delta_cartesian_waypoint")
        self.assertEqual(segment.conversion_metadata["conversion_steps"][0]["mode"], "delta_cartesian_waypoint")
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertAlmostEqual(last[2], 0.95, places=4)

    def test_native_6d_action_mode_preserves_action_semantics(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02, planner_output_mode="native_6d_action")
        state = self._state()
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(TeacherWaypoint(position_xyz=(0.1, -0.2, 0.3), yaw=0.4, speed_scale=1.0),),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=4,
            segment_granularity="coarse",
            rationale_summary="native",
        )
        segment = smoother.smooth(state=state, plan=plan)
        first_action = np.asarray(segment.points[0].action)
        np.testing.assert_allclose(first_action[:3], [0.1, -0.2, 0.3])
        self.assertAlmostEqual(first_action[5], 0.4, places=6)
        self.assertTrue(segment.conversion_metadata["native_action_override_used"])


if __name__ == "__main__":
    unittest.main()
