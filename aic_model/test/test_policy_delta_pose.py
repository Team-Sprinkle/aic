import importlib
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _install_policy_import_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    aic_control_mod = types.ModuleType("aic_control_interfaces.msg")
    aic_control_mod.JointMotionUpdate = type("JointMotionUpdate", (), {})
    aic_control_mod.MotionUpdate = type("MotionUpdate", (), {})
    aic_control_mod.TrajectoryGenerationMode = type(
        "TrajectoryGenerationMode",
        (),
        {"MODE_POSITION": 2},
    )
    monkeypatch.setitem(sys.modules, "aic_control_interfaces.msg", aic_control_mod)

    aic_model_mod = types.ModuleType("aic_model_interfaces.msg")
    aic_model_mod.Observation = type("Observation", (), {})
    monkeypatch.setitem(sys.modules, "aic_model_interfaces.msg", aic_model_mod)

    aic_task_mod = types.ModuleType("aic_task_interfaces.msg")
    aic_task_mod.Task = type("Task", (), {})
    monkeypatch.setitem(sys.modules, "aic_task_interfaces.msg", aic_task_mod)

    geom_mod = types.ModuleType("geometry_msgs.msg")

    class Point:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class Quaternion:
        def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
            self.x = x
            self.y = y
            self.z = z
            self.w = w

    class Pose:
        def __init__(self, position=None, orientation=None):
            self.position = position or Point()
            self.orientation = orientation or Quaternion()

    class Vector3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x = x
            self.y = y
            self.z = z

    class Wrench:
        def __init__(self, force=None, torque=None):
            self.force = force
            self.torque = torque

    geom_mod.Point = Point
    geom_mod.Pose = Pose
    geom_mod.Quaternion = Quaternion
    geom_mod.Vector3 = Vector3
    geom_mod.Wrench = Wrench
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geom_mod)

    duration_mod = types.ModuleType("rclpy.duration")
    duration_mod.Duration = type("Duration", (), {})
    monkeypatch.setitem(sys.modules, "rclpy.duration", duration_mod)

    std_msgs_mod = types.ModuleType("std_msgs.msg")
    std_msgs_mod.Header = type(
        "Header",
        (),
        {"__init__": lambda self, frame_id="", stamp=None: setattr(self, "__dict__", {"frame_id": frame_id, "stamp": stamp})},
    )
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_mod)

    tf_buf_mod = types.ModuleType("tf2_ros.buffer")
    tf_buf_mod.Buffer = type("Buffer", (), {})
    monkeypatch.setitem(sys.modules, "tf2_ros.buffer", tf_buf_mod)

    tf_listener_mod = types.ModuleType("tf2_ros.transform_listener")
    tf_listener_mod.TransformListener = type("TransformListener", (), {})
    monkeypatch.setitem(sys.modules, "tf2_ros.transform_listener", tf_listener_mod)


@pytest.fixture
def policy_module(monkeypatch: pytest.MonkeyPatch):
    package_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(package_root))
    _install_policy_import_stubs(monkeypatch)
    sys.modules.pop("aic_model.policy", None)
    return importlib.import_module("aic_model.policy")


def _pose(policy_mod, position_xyz, quat_xyzw):
    return policy_mod.Pose(
        position=policy_mod.Point(*position_xyz),
        orientation=policy_mod.Quaternion(*quat_xyzw),
    )


def test_compute_delta_pose_identity_frame(policy_module):
    mod = policy_module
    current_pose = _pose(mod, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    target_pose = _pose(
        mod,
        (0.1, -0.2, 0.3),
        (0.0, 0.0, math.sin(math.pi / 8.0), math.cos(math.pi / 8.0)),
    )

    delta_pose = mod.compute_delta_pose(current_pose, target_pose)
    delta_rotvec = mod.quaternion_xyzw_to_rotation_vector(
        np.array(
            [
                delta_pose.orientation.x,
                delta_pose.orientation.y,
                delta_pose.orientation.z,
                delta_pose.orientation.w,
            ],
            dtype=np.float64,
        )
    )

    assert delta_pose.position.x == pytest.approx(0.1)
    assert delta_pose.position.y == pytest.approx(-0.2)
    assert delta_pose.position.z == pytest.approx(0.3)
    np.testing.assert_allclose(delta_rotvec, np.array([0.0, 0.0, math.pi / 4.0]))


def test_compute_delta_pose_respects_current_tcp_frame(policy_module):
    mod = policy_module
    current_pose = _pose(
        mod,
        (1.0, 2.0, 0.0),
        (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
    )
    target_pose = _pose(
        mod,
        (1.0, 3.0, 0.0),
        (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
    )

    delta_pose = mod.compute_delta_pose(current_pose, target_pose)

    assert delta_pose.position.x == pytest.approx(1.0, abs=1e-6)
    assert delta_pose.position.y == pytest.approx(0.0, abs=1e-6)
    assert delta_pose.position.z == pytest.approx(0.0, abs=1e-6)
