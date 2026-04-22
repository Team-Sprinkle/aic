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
            segment_horizon_steps=6,
            segment_granularity="fine",
            rationale_summary="test",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertGreater(len(segment.points), 1)
        first = np.asarray(segment.points[0].target_tcp_pose)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertGreater(first[2], 0.0)
        self.assertAlmostEqual(last[2], 0.95, places=4)


if __name__ == "__main__":
    unittest.main()
