from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.runtime import RuntimeState
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
        wrench=np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0], dtype=np.float64),
        off_limit_contact=False,
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


if __name__ == "__main__":
    unittest.main()
