from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.runtime import AuxiliaryForceContactSummary, RuntimeState
from aic_gym_gz.teacher.history import TemporalObservationBuffer


def _state(*, sim_tick: int, sim_time: float, plug_xyz: tuple[float, float, float], tcp_xyz: tuple[float, float, float], force_z: float) -> RuntimeState:
    tcp_pose = np.array([*tcp_xyz, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    plug_pose = np.array([*plug_xyz, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return RuntimeState(
        sim_tick=sim_tick,
        sim_time=sim_time,
        joint_positions=np.zeros(6, dtype=np.float64),
        joint_velocities=np.zeros(6, dtype=np.float64),
        gripper_position=0.0,
        tcp_pose=tcp_pose,
        tcp_velocity=np.zeros(6, dtype=np.float64),
        plug_pose=plug_pose,
        target_port_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        target_port_entrance_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        wrench=np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0], dtype=np.float64),
        off_limit_contact=False,
        controller_state={"tcp_error": np.zeros(6, dtype=np.float64)},
        auxiliary_force_contact_summary=AuxiliaryForceContactSummary(
            source="mock_substep_history",
            sample_count=3,
            substep_tick_count=8,
            had_contact_recent=force_z > 0.0,
            wrench_current=np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0], dtype=np.float64),
            wrench_max_abs_recent=np.array([0.0, 0.0, force_z + 1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            wrench_mean_recent=np.array([0.0, 0.0, force_z / 2.0, 0.0, 0.0, 0.0], dtype=np.float64),
            wrench_max_force_abs_recent=force_z + 1.0,
        ),
    )


class TeacherHistoryTest(unittest.TestCase):
    def test_dynamics_summary_detects_recent_motion(self) -> None:
        history = TemporalObservationBuffer(max_frames=8, significant_motion_threshold=0.001)
        history.append(state=_state(sim_tick=0, sim_time=0.0, plug_xyz=(0.0, 0.0, 0.0), tcp_xyz=(0.0, 0.0, 0.0), force_z=0.0), action=np.zeros(6))
        history.append(state=_state(sim_tick=1, sim_time=0.1, plug_xyz=(0.0, 0.0, 0.004), tcp_xyz=(0.0, 0.0, 0.001), force_z=2.0), action=np.zeros(6))
        summary = history.dynamics_summary()
        self.assertGreater(summary.plug_oscillation_magnitude, 0.0)
        self.assertGreater(summary.recent_motion_energy, 0.0)
        self.assertLess(summary.cable_settling_score, 1.0)

    def test_quasi_static_after_small_deltas(self) -> None:
        history = TemporalObservationBuffer()
        for index in range(4):
            history.append(
                state=_state(
                    sim_tick=index,
                    sim_time=index * 0.1,
                    plug_xyz=(0.0, 0.0, 0.0001 * index),
                    tcp_xyz=(0.0, 0.0, 0.0001 * index),
                    force_z=0.1,
                ),
                action=np.zeros(6),
            )
        self.assertTrue(history.dynamics_summary().quasi_static)

    def test_teacher_memory_summary_keeps_current_observation_separate_from_history(self) -> None:
        history = TemporalObservationBuffer(max_frames=4)
        state = _state(
            sim_tick=3,
            sim_time=0.3,
            plug_xyz=(0.0, 0.0, 0.003),
            tcp_xyz=(0.0, 0.0, 0.001),
            force_z=0.5,
        )
        history.append(
            state=state,
            action=np.ones(6, dtype=np.float64),
            image_timestamps={"left": 1.0},
            image_summaries={"left": {"present": True}},
            camera_info={"left": {"size": np.array([64.0, 64.0], dtype=np.float32)}},
            signal_quality={"wrench": {"is_real": False}, "controller_state": {"is_real": True}},
        )
        current = history.current_observation_view()
        memory = history.teacher_memory_summary()
        self.assertEqual(current["sim_tick"], 3)
        self.assertEqual(len(memory["action_history"]), 1)
        self.assertEqual(memory["latest_signal_quality"]["controller_state"]["is_real"], True)
        self.assertNotIn("auxiliary_force_contact_summary", current)

    def test_auxiliary_history_is_stored_separately_and_summarized(self) -> None:
        history = TemporalObservationBuffer(max_frames=4)
        hidden_contact_state = _state(
            sim_tick=1,
            sim_time=0.1,
            plug_xyz=(0.0, 0.0, 0.001),
            tcp_xyz=(0.0, 0.0, 0.001),
            force_z=0.0,
        )
        history.append(
            state=hidden_contact_state,
            action=np.zeros(6),
            auxiliary_force_contact_summary={
                "source": "mock_substep_history",
                "sample_count": 4,
                "substep_tick_count": 8,
                "had_contact_recent": True,
                "wrench_max_force_abs_recent": 8.0,
            },
            auxiliary_summary_available=True,
        )
        summary = history.teacher_memory_summary()
        auxiliary = history.auxiliary_history_summary()
        self.assertIn("official_history_summary", summary)
        self.assertIn("auxiliary_history_summary", summary)
        self.assertEqual(auxiliary["hidden_contact_event_count_recent"], 1)
        self.assertEqual(auxiliary["latest_auxiliary_summary_status"], "real")

    def test_recent_visual_frames_keep_actual_image_arrays(self) -> None:
        history = TemporalObservationBuffer(max_frames=4)
        history.append(
            state=_state(
                sim_tick=2,
                sim_time=0.2,
                plug_xyz=(0.0, 0.0, 0.002),
                tcp_xyz=(0.0, 0.0, 0.002),
                force_z=0.0,
            ),
            action=np.zeros(6),
            images={"left": np.full((8, 8, 3), 127, dtype=np.uint8)},
            image_timestamps={"left": 0.2},
            image_summaries={"left": {"present": True}},
        )
        frames = history.recent_visual_frames(max_frames=2)
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]["images"]["left"].shape, (8, 8, 3))
        self.assertEqual(frames[0]["age_from_latest_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
