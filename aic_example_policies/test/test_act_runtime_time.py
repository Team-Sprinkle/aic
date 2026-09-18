"""Check elapsed-time input construction without ROS or a model checkpoint."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest


def test_time_feature_uses_camera_stamp_and_clips_after_demo_horizon():
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_state_vector")
    scope = {"np": np, "Observation": object}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    runner = NS(feature_assembler=NS(uses_task_vector=False, assemble_ros=lambda obs: np.zeros(32, np.float32)),
                include_elapsed_sim_time=True, _episode_first_image_time=None, time_clip_sec=40., quaternion_sign="w")
    # The actual Observation message has no top-level header.
    stamp = NS(sec=100, nanosec=500_000_000)
    obs = NS(center_image=NS(header=NS(stamp=stamp)))
    state = scope["_state_vector"](runner, obs)
    assert state.shape == (33,) and state[-1] == 0
    stamp.sec, stamp.nanosec = 105, 250_000_000
    assert scope["_state_vector"](runner, obs)[-1] == 4.75
    stamp.sec = 200
    assert scope["_state_vector"](runner, obs)[-1] == 40
    runner._episode_first_image_time = None
    assert scope["_state_vector"](runner, obs)[-1] == 0
    runner.include_elapsed_sim_time = False
    assert scope["_state_vector"](runner, obs).shape == (32,)
    # Crossing W=0 near a half-turn must not flip the X hemisphere for a model
    # trained with quaternion_sign=x.
    runner.quaternion_sign = "x"
    for w in [-1e-5, 1e-5]:
        raw = np.zeros(32, np.float32)
        raw[3:7] = [1, 0, 0, w]
        if w < 0:
            raw[3:7] *= -1  # Base assembler's legacy W convention.
        runner.feature_assembler.assemble_ros = lambda obs: raw.copy()
        result = scope["_state_vector"](runner, obs)
        assert result[3] == 1 and np.isclose(result[6], w)


@pytest.mark.parametrize("encoding,values", [("rgb8", [10, 20, 30]), ("bgr8", [30, 20, 10]),
                                           ("rgba8", [10, 20, 30, 255]), ("bgra8", [30, 20, 10, 255])])
def test_camera_decoder_respects_encoding_and_row_padding(encoding, values):
    import cv2
    import torch
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef)
               and node.name in {"_image_msg_to_chw_float", "_normalized_image"}]
    for method in methods:
        method.decorator_list = []
    scope = {"np": np, "cv2": cv2, "torch": torch, "Any": object}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), scope)
    pixels = bytes(values + [99, 99] + values + [88, 88])
    msg = NS(data=pixels, height=2, width=1, step=len(values) + 2, encoding=encoding)
    decode = scope["_image_msg_to_chw_float"]
    expected = torch.tensor([10, 20, 30]).float() / 255
    assert torch.allclose(decode(msg, (3, 2, 1))[0, :, 0, 0], expected)
    runner = NS(_image_msg_to_chw_float=decode, camera_shapes={"camera": (3, 2, 1)}, device="cpu",
                image_channel_order="bgr", img_stats={"camera": {"mean": 0, "std": 1}})
    assert torch.allclose(scope["_normalized_image"](runner, "camera", msg)[0, :, 1, 0], expected.flip(0))


def test_temporal_ensemble_aligns_future_steps_and_expires_old_chunks():
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_ensemble_action")
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    runner = NS(temporal_ensemble_coefficient=0., _ensemble_predictions=[])
    combine = lambda values: scope["_ensemble_action"](runner, np.repeat(np.asarray(values)[:, None], 6, axis=1))
    np.testing.assert_allclose(combine([1., 2., 3.]), 1.)
    np.testing.assert_allclose(combine([10., 20., 30.]), 6.)
    np.testing.assert_allclose(combine([100., 200., 300.]), 41.)
    np.testing.assert_allclose(combine([1000., 2000., 3000.]), 410.)
    runner._ensemble_predictions.clear()
    runner.temporal_ensemble_coefficient = np.log(2)
    combine([1., 2.])
    np.testing.assert_allclose(combine([10., 20.]), (2. * .5 + 10.) / 1.5)
    assert np.isnan(combine([np.nan])).all()
    assert not runner._ensemble_predictions


def test_translation_norm_limit_preserves_direction_under_gripper_rotation():
    from scipy.spatial.transform import Rotation
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_limit_translation_norm")
    method.decorator_list = []
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    limit = scope["_limit_translation_norm"]
    rotation = Rotation.from_euler("xyz", [2.76, -.2, .05])
    world_delta = np.array([.003, -.002, -.065])
    local = rotation.apply(world_delta, inverse=True)
    actual = rotation.apply(limit(local, .02))
    np.testing.assert_allclose(actual, world_delta * .02 / np.linalg.norm(world_delta), atol=1e-12)
    np.testing.assert_allclose(limit(np.array([.001, -.002, .003]), .02), [.001, -.002, .003])
    np.testing.assert_array_equal(limit(np.zeros(3), .02), np.zeros(3))


def test_observed_delta_reconstructs_absolute_teacher_target():
    from scipy.spatial.transform import Rotation
    source = Path(__file__).resolve().parents[1] / "aic_example_policies/ros/RunACTTorchScript.py"
    cls = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_observed_delta_target_values")
    method.decorator_list = []
    scope = {"np": np}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
    current_rotation = Rotation.from_euler("xyz", [2.9, -.2, .4])
    current_position = np.array([.3, -.1, .2])
    target_position = np.array([.31, -.08, .17])
    target_rotation = Rotation.from_euler("xyz", [3.0, -.21, .42])
    q = current_rotation.as_quat()
    pose = NS(position=NS(x=current_position[0], y=current_position[1], z=current_position[2]),
              orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))
    delta_position = current_rotation.apply(target_position - current_position, inverse=True)
    delta_rotation = (current_rotation.inv() * target_rotation).as_rotvec()
    p, q = scope["_observed_delta_target_values"](pose, delta_position, delta_rotation)
    np.testing.assert_allclose(p, target_position, atol=1e-12)
    assert (Rotation.from_quat(q).inv() * target_rotation).magnitude() < 1e-12
    # Reapplying this label from a later controller pose would shift the target.
    assert np.linalg.norm((current_position + [.002, 0, 0] + current_rotation.apply(delta_position)) - p) > .0019
