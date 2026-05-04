from __future__ import annotations

import pytest

from gazebo_rl.bridge_policy.GazeboRLBridgePolicy import GazeboRLBridgePolicy


class _Stamp:
    def to_msg(self):
        return self


class _Clock:
    def now(self):
        return _Stamp()


class _Parent:
    def get_clock(self):
        return _Clock()


def _policy() -> GazeboRLBridgePolicy:
    policy = GazeboRLBridgePolicy.__new__(GazeboRLBridgePolicy)
    policy._parent_node = _Parent()
    return policy


def test_send_delta_action_raises_when_move_robot_publisher_is_unavailable():
    def move_robot(*, motion_update=None, joint_motion_update=None):
        raise AttributeError("'NoneType' object has no attribute 'publish'")

    with pytest.raises(AttributeError, match="publish"):
        _policy()._send_delta_action(move_robot, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_send_delta_action_raises_when_move_robot_rejects_command():
    def move_robot(*, motion_update=None, joint_motion_update=None):
        return False

    with pytest.raises(RuntimeError, match="rejected"):
        _policy()._send_delta_action(move_robot, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_send_delta_action_builds_gripper_tcp_motion_update():
    sent = {}

    def move_robot(*, motion_update=None, joint_motion_update=None):
        sent["motion_update"] = motion_update
        sent["joint_motion_update"] = joint_motion_update
        return True

    _policy()._send_delta_action(move_robot, [1.0, -1.0, 0.0, 1.0, 0.0, 0.0])

    assert sent["joint_motion_update"] is None
    assert sent["motion_update"].header.frame_id == "gripper/tcp"
    assert sent["motion_update"].pose.position.x == pytest.approx(0.003)
    assert sent["motion_update"].pose.position.y == pytest.approx(-0.003)
