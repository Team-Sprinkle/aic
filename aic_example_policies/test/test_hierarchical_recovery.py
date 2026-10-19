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


def test_stationary_tilt_is_retained_and_not_skipped_by_translation_tolerance():
    from math import cos, sin, radians
    def tilted(deg):
        return MeasuredPose((0., 0., 0.), (cos(radians(deg)/2), 0., sin(radians(deg)/2), 0.))
    controller = HierarchicalRecovery()
    for deg in (0, 5, 10, 15):
        controller.step(pose=tilted(deg), force_n=1., prior_command_m=0.,
                        phase='alignment', route='outside')
    assert len(controller.history) == 4
    # Reaching the previous position without its orientation cannot release
    # recovery and resume an actor with an incorrectly oriented tool.
    assert not controller._waypoint_reached(tilted(15), tilted(0))
    controller.mode='backtrack';controller.path=[tilted(0)];controller.anchor=tilted(0)
    controller.anchor_history_index=0
    controller.previous_force_n=1.
    decision=controller.step(pose=tilted(15), force_n=1., prior_command_m=0.,
                             phase='alignment', route='outside')
    assert decision.mode=='backtrack'
    assert not decision.actor_owned
    assert decision.target is not None
    assert module._angle_deg(tilted(15),decision.target)<=4.00001


def test_retry_history_prunes_failed_branch_and_deeper_anchor_precedes_B():
    controller=HierarchicalRecovery(RecoveryConfig(local_retreat_m=.008,route_retreat_m=.03))
    for i in range(81):
        controller.step(pose=pose(i*.001),force_n=1.,prior_command_m=.001,phase='alignment',route='same')
    controller._trigger(pose(.08),9.,'insertion','same')
    first_anchor=controller.anchor
    assert first_anchor is not None
    x=.08
    for _ in range(100):
        decision=controller.step(pose=pose(x),force_n=1.,prior_command_m=.003,phase='insertion',route='same')
        if decision.mode=='retry':break
        x=decision.target.xyz[0]
    assert controller.history[-1][0]==first_anchor
    assert all(item[0].xyz[0]<=first_anchor.xyz[0] for item in controller.history)
    # A new blockage >15mm from the previous contact still counts as another
    # failure of the same route, rather than resetting the hierarchy forever.
    controller.last_contact_pose=pose(.01);controller.last_contact_route='same';controller.local_attempts=2
    for i in range(1,40):
        controller.step(pose=pose(x+i*.001),force_n=1.,prior_command_m=.001,phase='insertion',route='same')
    contact=pose(x+.039);controller._trigger(contact,9.,'insertion','same')
    assert controller.level=='route'
    assert controller.anchor_history_index<min(controller.retry_checkpoint_indices)


def test_gradual_force_rise_during_retreat_aborts():
    controller=HierarchicalRecovery(RecoveryConfig(trigger_steps=1))
    for i in range(50):
        controller.step(pose=pose(i*.001),force_n=1.,prior_command_m=.001,
                        phase='approach',route='same')
    controller.step(pose=pose(.049),force_n=9.,prior_command_m=.003,
                    phase='approach',route='same')
    assert controller.mode=='backtrack'
    # Each rise is below the threshold; the accumulated new load is not.
    for force in [8.,9.,10.,11.,12.,13.,14.,15.]:
        decision=controller.step(pose=pose(.049),force_n=force,prior_command_m=.003,
                                phase='approach',route='same')
    assert decision.mode=='abort'
    assert decision.reason=='force_rising_during_retreat'


def test_route_reacquisition_returns_to_measured_initial_hold_not_60mm_anchor():
    controller=HierarchicalRecovery(RecoveryConfig(local_attempts_before_route=1))
    controller.step(pose=pose(0.),force_n=1.,prior_command_m=0.,
                    phase='acquisition_checkpoint',route='initial')
    for i in range(1,151):
        controller.step(pose=pose(i*.001),force_n=1.,prior_command_m=.001,
                        phase='approach',route='initial')
    controller.local_attempts=1;controller.last_contact_pose=pose(.15)
    controller.last_contact_route='initial'
    controller._trigger(pose(.15),9.,'contact','initial')
    assert controller.level=='route'
    assert controller.anchor.xyz[0]>.05
    assert controller.extend_route_to_acquisition()
    assert controller.anchor==pose(0.)
    assert controller.path==[item[0] for item in reversed(controller.history)]


def test_route_reacquisition_without_free_acquisition_hold_aborts():
    controller=HierarchicalRecovery()
    controller.history=[(pose(0.),9.,'acquisition_checkpoint','initial')]
    controller.mode='backtrack';controller.level='route'
    assert not controller.extend_route_to_acquisition()
    assert controller.mode=='abort'
    assert controller.reason=='no_free_acquisition_checkpoint'


def test_new_corridor_requires_real_motion_and_preserves_episode_budgets():
    import pytest
    controller=HierarchicalRecovery()
    controller.mode='retry';controller.level='route';controller.failed_route='old'
    controller.retry_checkpoint_indices=[0]
    controller.total_backoffs=3;controller.route_resets=1;controller.local_attempts=3
    with pytest.raises(ValueError):controller.confirm_new_corridor('new',.001)
    with pytest.raises(ValueError):controller.confirm_new_corridor('old',.010)
    assert controller.confirm_new_corridor('new',.010)=='new'
    assert controller.total_backoffs==3 and controller.route_resets==1
    assert controller.local_attempts==0 and controller.retry_checkpoint_indices==[]
    assert controller.mode=='policy'
