from __future__ import annotations

import unittest

import numpy as np

from aic_gym_gz.probes.library import ProbeLibrary
from aic_gym_gz.runtime import RuntimeState
from aic_gym_gz.teacher.history import TemporalObservationBuffer


def _state(sim_time: float, plug_z: float, force_z: float) -> RuntimeState:
    return RuntimeState(
        sim_tick=int(sim_time * 100),
        sim_time=sim_time,
        joint_positions=np.zeros(6, dtype=np.float64),
        joint_velocities=np.zeros(6, dtype=np.float64),
        gripper_position=0.0,
        tcp_pose=np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        tcp_velocity=np.zeros(6, dtype=np.float64),
        plug_pose=np.array([0.0, 0.0, plug_z, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        target_port_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        target_port_entrance_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        wrench=np.array([0.0, 0.0, force_z, 0.0, 0.0, 0.0], dtype=np.float64),
        off_limit_contact=False,
    )


class TeacherProbeTest(unittest.TestCase):
    def test_probe_catalog_contains_required_safe_probes(self) -> None:
        library = ProbeLibrary()
        self.assertEqual(
            set(library.list_probe_names()),
            {"hold_settle", "lift_and_hold", "micro_sweep_xy", "yaw_wiggle"},
        )

    def test_probe_result_logs_deltas(self) -> None:
        library = ProbeLibrary()
        before_history = TemporalObservationBuffer()
        after_history = TemporalObservationBuffer()
        before_state = _state(0.0, 1.0, 0.5)
        after_state = _state(0.2, 0.99, 1.5)
        before_history.append(state=before_state, action=np.zeros(6))
        after_history.append(state=before_state, action=np.zeros(6))
        after_history.append(state=after_state, action=np.zeros(6))
        result = library.summarize_result(
            probe_name="hold_settle",
            before_state=before_state,
            after_state=after_state,
            before_summary=before_history,
            after_summary=after_history,
            action_count=3,
        )
        self.assertEqual(result.probe_name, "hold_settle")
        self.assertGreater(result.peak_force, 0.0)
        self.assertGreater(result.plug_relative_motion, 0.0)


if __name__ == "__main__":
    unittest.main()
