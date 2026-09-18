"""A lateral connector offset must be rotated when forming a TCP target."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def test_sc_tcp_target_places_offset_tip_at_requested_position_after_rotation():
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/CollectCorrectiveCheatCode.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_tcp_position_for_tip_target")
    method.decorator_list = []
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    current_rotation = Rotation.from_euler("xyz", [2.8, -.2, .3])
    target_rotation = Rotation.from_euler("xyz", [3.1, .1, -.2])
    tcp_position = np.array([-.4, .3, .2])
    local_tip_offset = np.array([.015, .03, .04])
    tip_position = tcp_position + current_rotation.apply(local_tip_offset)
    target_tip_position = np.array([-.42, .4, .01])
    target_tcp = scope["_tcp_position_for_tip_target"](
        np.r_[tcp_position, current_rotation.as_quat()], tip_position,
        target_rotation.as_quat(), target_tip_position)
    np.testing.assert_allclose(target_tcp + target_rotation.apply(local_tip_offset), target_tip_position, atol=1e-12)
    assert np.linalg.norm(target_tcp[:2] - target_tip_position[:2]) > .02


def test_sfp_collection_delegates_pose_generation_without_sc_control_changes():
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/CollectCorrectiveCheatCode.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))

    class OriginalTeacher:
        def calc_gripper_pose(self, *args):
            return args

    scope = {"CheatCode": OriginalTeacher, "np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    collector_class = scope["CollectCorrectiveCheatCode"]
    collector = collector_class.__new__(collector_class)
    collector._task = SimpleNamespace(target_module_name="nic_card_mount_0")
    port = object()
    assert collector.calc_gripper_pose(port, .25, .5, .123, True) == (port, .25, .5, .123, True)
