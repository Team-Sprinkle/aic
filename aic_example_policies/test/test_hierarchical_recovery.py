"""Behavioral checks for multi-level retreat and route-change gating."""

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/hierarchical_recovery.py"
spec = importlib.util.spec_from_file_location("hierarchical_recovery_test", SOURCE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)
MeasuredPose = module.MeasuredPose
RecoveryConfig = module.RecoveryConfig
HierarchicalRecovery = module.HierarchicalRecovery


def pose(x):
    return MeasuredPose((x, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))


def test_force_without_stall_does_not_trigger():
    controller = HierarchicalRecovery()
    for i in range(10):
        result = controller.step(pose=pose(i * 0.002), force_n=12.0,
                                 prior_command_m=0.002, phase="insertion", route="center")
    assert result.mode == "policy"
    assert result.total_backoffs == 0


def test_repeated_same_corridor_escalates_and_requires_new_route():
    controller = HierarchicalRecovery(RecoveryConfig(
        trigger_steps=2, local_retreat_m=0.008, route_retreat_m=0.050,
        history_spacing_m=0.001, max_command_step_m=0.003,
    ))
    for i in range(41):
        controller.step(pose=pose(i * 0.003), force_n=1.0,
                        prior_command_m=0.003, phase="transport", route="across")

    x = 0.120
    for attempt in range(3):
        for _ in range(2):
            result = controller.step(pose=pose(x), force_n=9.0,
                                     prior_command_m=0.003, phase="transport", route="across")
        assert result.mode == "backtrack"
        assert result.level == ("route" if attempt == 2 else "local")
        for _ in range(200):
            result = controller.step(pose=pose(x), force_n=1.0,
                                     prior_command_m=0.003, phase="transport", route="across")
            if result.mode == "retry":
                break
            assert result.mode == "backtrack"
            assert result.target is not None
            assert abs(result.target.xyz[0] - x) <= 0.003001
            x = result.target.xyz[0]
        assert result.mode == "retry"
        if attempt == 2:
            assert result.route_change_required
            assert result.forbidden_route == "across"
            assert not result.actor_owned
            assert result.target == pose(x)
            changed = controller.step(pose=pose(x), force_n=1.0,
                                      prior_command_m=0.0, phase="transport", route="outside")
            assert changed.actor_owned
        else:
            for i in range(1, 41):
                x = min(0.120, x + 0.003)
                controller.step(pose=pose(x), force_n=1.0,
                                prior_command_m=0.003, phase="transport", route="across")
            x = 0.120
    assert controller.total_backoffs == 3
    assert controller.route_resets == 1


def test_unverified_route_checkpoint_aborts_instead_of_pushing():
    controller = HierarchicalRecovery(RecoveryConfig(trigger_steps=1))
    controller.step(pose=pose(0.0), force_n=9.0, prior_command_m=0.0,
                    phase="transport", route="across")
    result = controller.step(pose=pose(0.0), force_n=9.0, prior_command_m=0.003,
                             phase="transport", route="across")
    assert result.mode == "abort"
    assert result.reason == "no_verified_safe_checkpoint"
    assert not result.actor_owned


def test_held_absolute_target_can_trigger_with_tracking_error():
    controller = HierarchicalRecovery(RecoveryConfig(trigger_steps=2, local_retreat_m=0.003))
    for i in range(20):
        controller.step(pose=pose(i * 0.001), force_n=1.0,
                        prior_command_m=0.001, phase="approach", route="center")
    for _ in range(2):
        result = controller.step(pose=pose(0.019), force_n=10.0,
                                 prior_command_m=0.0, tracking_error_m=0.010,
                                 phase="approach", route="center")
    assert result.mode == "backtrack"
    assert result.total_backoffs == 1
