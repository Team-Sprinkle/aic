import importlib.util
from pathlib import Path
import sys

import torch


PATH = Path(__file__).parents[1] / "aic_isaaclab/scripts/serl/measured_path_recovery.py"
spec = importlib.util.spec_from_file_location("measured_path_recovery", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def _step(controller, x, force, command=(0.001, 0.0, 0.0)):
    position = torch.tensor([[x, 0.0, 0.0]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    action = torch.tensor([list(command) + [0.0, 0.0, 0.0]])
    return controller.apply(
        position_world=position,
        quaternion_world_wxyz=quat,
        force_n=torch.tensor([[force]]),
        proposed_action_body=action,
    )


def test_force_trigger_replays_measured_path_and_masks_actor():
    cfg = m.MeasuredPathRecoveryConfig(
        trigger_consecutive_steps=2,
        min_clearance_m=0.001,
        backtrack_step_m=0.001,
        history_spacing_m=0.0001,
    )
    controller = m.MeasuredPathRecovery(1, cfg)
    _step(controller, 0.000, 0.0)
    _step(controller, 0.001, 0.0)
    _step(controller, 0.002, 9.0)
    _step(controller, 0.002, 9.0)
    action, owner, rows = _step(controller, 0.002, 9.0)
    assert rows[0]["mode_name"] == "backtrack"
    assert not bool(owner[0, 0])
    assert float(action[0, 0]) < 0.0


def test_clearance_hysteresis_returns_control_in_lateral_mode():
    cfg = m.MeasuredPathRecoveryConfig(
        trigger_consecutive_steps=1,
        min_clearance_m=0.001,
        backtrack_step_m=0.001,
        history_spacing_m=0.0001,
        lateral_policy_steps=3,
    )
    controller = m.MeasuredPathRecovery(1, cfg)
    _step(controller, 0.000, 0.0)
    _step(controller, 0.001, 9.0)
    _step(controller, 0.001, 9.0)
    action, owner, rows = _step(controller, 0.000, 3.0, command=(0.001, 0.001, 0.0))
    assert rows[0]["mode_name"] == "lateral"
    assert bool(owner[0, 0])
    # The component repeating the blocked +x direction is strongly suppressed;
    # the lateral +y proposal is preserved.
    assert abs(float(action[0, 0])) <= 0.00011
    assert abs(float(action[0, 1]) - 0.001) < 1.0e-7


def test_force_increase_during_retreat_aborts_motion():
    cfg = m.MeasuredPathRecoveryConfig(
        trigger_consecutive_steps=1,
        force_increase_abort_n=2.0,
        min_clearance_m=0.002,
    )
    controller = m.MeasuredPathRecovery(1, cfg)
    _step(controller, 0.000, 0.0)
    _step(controller, 0.001, 9.0)
    _step(controller, 0.001, 9.0)
    action, owner, rows = _step(controller, 0.0008, 12.0)
    assert rows[0]["mode_name"] == "abort"
    assert not bool(owner[0, 0])
    assert torch.equal(action, torch.zeros_like(action))


def test_near_lateral_projection_preserves_sideways_and_limits_toward_motion():
    result = m.project_near_lateral(
        torch.tensor([2.0, 3.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        toward_fraction=0.1,
        away_fraction=0.5,
    )
    assert torch.allclose(result, torch.tensor([0.2, 3.0, 0.0]))
