"""Focused checks for the observation-only transport recovery gate."""
import importlib.util
from pathlib import Path
import sys

import numpy as np


SOURCE=Path(__file__).resolve().parents[2]/'scripts/transport_local_recovery.py'
spec=importlib.util.spec_from_file_location('transport_local_recovery_test',SOURCE)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)


def test_force_without_lost_progress_does_not_trigger():
    gate=module.TransportLoadStallMonitor()
    assert not any(gate.update(sim_time=i*.1,stage='transfer',goal_error_m=.1-i*.005,
                               load_change_n=12.) for i in range(6))


def test_stall_without_force_and_visual_timeout_do_not_trigger():
    gate=module.TransportLoadStallMonitor()
    assert not any(gate.update(sim_time=i*.1,stage='transfer',goal_error_m=.1,
                               load_change_n=2.) for i in range(6))
    gate.reset()
    assert not any(gate.update(sim_time=i*.1,stage='visual_acquisition',goal_error_m=.1,
                               load_change_n=2.) for i in range(6))


def test_sustained_load_and_stalled_waypoint_trigger():
    gate=module.TransportLoadStallMonitor()
    outputs=[gate.update(sim_time=i*.1,stage='transfer',goal_error_m=.1-i*.0001,
                         load_change_n=12.) for i in range(6)]
    assert outputs[-1] and not outputs[0]


def test_sideways_is_horizontal_and_orthogonal_to_measured_motion():
    positions=np.array([[0.,0.,0.],[.006,.002,0.],[.012,.004,0.]])
    axial=module.measured_approach_axis(positions)
    lateral=module.horizontal_lateral_axis(axial)
    assert abs(np.dot(axial,lateral))<1e-8
    assert lateral[2]==0.
    assert np.allclose(module.horizontal_lateral_axis(axial,-1),-lateral)
