from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.runtime import RuntimeState
from aic_gym_gz.teacher.types import TeacherPlan, TeacherWaypoint
from aic_gym_gz.trajectory.smoothing import MinimumJerkSmoother


class TeacherSmoothingTest(unittest.TestCase):
    def test_minimum_jerk_segment_is_dense_and_continuous(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = RuntimeState(
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
        state = RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_pose=np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_entrance_pose=np.array([0.0, 0.0, 0.85, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            wrench=np.zeros(6, dtype=np.float64),
            off_limit_contact=False,
        )
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
        state = RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_pose=np.array([0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            target_port_entrance_pose=np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            wrench=np.zeros(6, dtype=np.float64),
            off_limit_contact=False,
        )
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


if __name__ == "__main__":
    unittest.main()
