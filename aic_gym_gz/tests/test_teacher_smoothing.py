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
        metrics = segment.conversion_metadata["optimizer_metrics"]
        self.assertEqual(metrics["sparse_waypoint_count"], 1)
        self.assertEqual(metrics["dense_point_count"], len(segment.points))
        self.assertGreater(metrics["dense_path_length_m"], 0.0)
        self.assertLessEqual(metrics["max_command_linear_speed_mps"], smoother.max_linear_speed + 1e-9)

    def test_optimizer_uses_physical_horizon_when_vlm_horizon_is_short(self) -> None:
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
        self.assertGreater(len(segment.points), 4)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertAlmostEqual(last[2], 0.85, places=4)
        self.assertGreater(segment.conversion_metadata["optimizer_horizon_steps"], 4)
        self.assertEqual(segment.conversion_metadata["optimizer_metrics"]["sparse_waypoint_count"], 2)

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
        self.assertGreater(len(segment.points), 5)
        last = np.asarray(segment.points[-1].target_tcp_pose)
        self.assertAlmostEqual(last[2], 0.76, places=4)

    def test_optimizer_limits_cartesian_speed_by_vector_norm(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02, max_linear_speed=0.05)
        state = self._state()
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(TeacherWaypoint(position_xyz=(0.25, 0.25, 0.75), yaw=0.0, speed_scale=1.0),),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=4,
            segment_granularity="coarse",
            rationale_summary="diagonal speed clamp",
        )
        segment = smoother.smooth(state=state, plan=plan)
        max_speed = max(float(np.linalg.norm(point.action[:3])) for point in segment.points)
        self.assertLessEqual(max_speed, smoother.max_linear_speed + 1e-9)
        self.assertLessEqual(
            segment.conversion_metadata["optimizer_metrics"]["max_command_linear_speed_mps"],
            smoother.max_linear_speed + 1e-9,
        )
        self.assertEqual(
            segment.conversion_metadata["optimizer_metrics"]["guarded_insert_speed_limit_mps"],
            smoother.guarded_insert_speed_limit,
        )

    def test_optimizer_slows_near_final_target(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02, max_linear_speed=0.12)
        state = self._state()
        state.tcp_pose[:3] = [0.0, 0.0, 0.92]
        state.plug_pose[:3] = [0.0, 0.0, 0.92]
        state.target_port_entrance_pose[:3] = [0.0, 0.0, 0.95]
        state.target_port_pose[:3] = [0.0, 0.0, 0.9]
        plan = TeacherPlan(
            next_phase="guarded_insert",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.902), yaw=0.0, speed_scale=1.0),),
            motion_mode="guarded_insert",
            caution_flag=True,
            should_probe=False,
            segment_horizon_steps=4,
            segment_granularity="guarded",
            rationale_summary="final approach clamp",
        )
        segment = smoother.smooth(state=state, plan=plan)
        near_target_speeds = [
            float(np.linalg.norm(point.action[:3]))
            for point in segment.points
            if np.linalg.norm(np.asarray(point.target_tcp_pose[:3]) - state.target_port_pose[:3]) <= 0.015
        ]
        self.assertTrue(near_target_speeds)
        self.assertLessEqual(max(near_target_speeds), 0.035 + 1e-9)

    def test_tiny_planner_horizon_still_produces_replayable_segment(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.9), yaw=0.0, speed_scale=0.5),),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=1,
            segment_granularity="coarse",
            rationale_summary="tiny horizon",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertGreaterEqual(len(segment.points), 4)
        self.assertEqual(segment.conversion_metadata["requested_horizon_steps"], 1)
        self.assertGreater(segment.conversion_metadata["effective_horizon_steps"], 4)

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

    def test_zero_absolute_yaw_defaults_to_target_yaw(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        state.target_port_pose[5] = 3.0
        plan = TeacherPlan(
            next_phase="free_space_approach",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.95), yaw=0.0, speed_scale=0.5),),
            motion_mode="coarse_cartesian",
            caution_flag=False,
            should_probe=False,
            segment_horizon_steps=4,
            segment_granularity="coarse",
            rationale_summary="yaw fallback",
        )
        segment = smoother.smooth(state=state, plan=plan)
        self.assertAlmostEqual(segment.points[-1].target_tcp_pose[5], 3.0, places=6)
        self.assertEqual(segment.conversion_metadata["conversion_steps"][0]["resolved_target_yaw"], 3.0)

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

    def test_port_frame_gate_blocks_guarded_insert_until_aligned(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.02)
        state = self._state()
        state.plug_pose[:3] = [0.10, 0.02, 1.0]
        state.tcp_pose[:3] = [0.10, 0.02, 1.0]
        state.plug_pose[5] = 0.8
        state.tcp_pose[5] = 0.8
        state.target_port_entrance_pose[:3] = [0.0, 0.0, 0.95]
        state.target_port_pose[:3] = [0.0, 0.0, 0.90]
        state.target_port_pose[5] = 0.0
        plan = TeacherPlan(
            next_phase="guarded_insert",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.90), yaw=0.0, speed_scale=1.0),),
            motion_mode="guarded_insert",
            caution_flag=True,
            should_probe=False,
            segment_horizon_steps=8,
            segment_granularity="guarded",
            rationale_summary="bad premature insertion",
        )

        segment = smoother.smooth(state=state, plan=plan)

        gate = segment.conversion_metadata["port_frame_alignment_gate"]
        self.assertTrue(gate["active"])
        self.assertEqual(gate["gate_action"], "align_lateral_and_yaw_at_pre_insert_standoff")
        final_pose = np.asarray(segment.points[-1].target_tcp_pose)
        # The insertion axis is negative world Z, so pre-insert is 25 mm before
        # the entrance plane in the opposite direction.
        self.assertAlmostEqual(final_pose[0], 0.0, places=4)
        self.assertAlmostEqual(final_pose[1], 0.0, places=4)
        self.assertAlmostEqual(final_pose[2], 0.975, places=4)
        self.assertAlmostEqual(final_pose[5], 0.0, places=4)
        self.assertAlmostEqual(gate["final_target_lateral_error_m"], 0.0, places=6)
        self.assertAlmostEqual(gate["final_target_axial_depth_m"], -0.025, places=6)
        self.assertAlmostEqual(gate["final_target_yaw_error_rad"], 0.0, places=6)
        self.assertEqual(gate["guarded_insert_speed_limit_mps"], smoother.guarded_insert_speed_limit)

    def test_port_frame_alignment_gate_uses_stable_near_port_speed_limit(self) -> None:
        smoother = MinimumJerkSmoother(base_dt=0.05)
        state = self._state()
        state.plug_pose[:3] = [0.16, 0.0, 1.0]
        state.tcp_pose[:3] = [0.16, 0.0, 1.0]
        state.target_port_entrance_pose[:3] = [0.0, 0.0, 0.95]
        state.target_port_pose[:3] = [0.0, 0.0, 0.90]
        plan = TeacherPlan(
            next_phase="pre_insert_align",
            waypoints=(TeacherWaypoint(position_xyz=(0.0, 0.0, 0.975), yaw=0.0, speed_scale=0.1),),
            motion_mode="fine_cartesian",
            caution_flag=True,
            should_probe=False,
            segment_horizon_steps=8,
            segment_granularity="fine",
            rationale_summary="slow VLM prealign should be stable near the cable",
        )

        segment = smoother.smooth(state=state, plan=plan)

        self.assertEqual(segment.conversion_metadata["gated_alignment_min_speed_scale"], 0.8)
        self.assertEqual(
            segment.conversion_metadata["segment_linear_speed_limit_mps"],
            smoother.port_alignment_unaligned_speed_limit,
        )
        self.assertLess(len(segment.points), 180)
        self.assertGreater(len(segment.points), 80)
        max_speed = max(float(np.linalg.norm(point.action[:3])) for point in segment.points)
        self.assertLessEqual(max_speed, smoother.port_alignment_unaligned_speed_limit + 1e-9)


if __name__ == "__main__":
    unittest.main()
