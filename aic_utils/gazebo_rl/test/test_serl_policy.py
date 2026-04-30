from __future__ import annotations

import base64

import numpy as np
import pytest
import torch
from torch import nn

from gazebo_rl.serl_policy import (
    ACTAdapterSERLGazeboPolicy,
    TorchScriptACTAdapterActor,
    lowdim_state_from_gazebo_observation,
)


def test_lowdim_state_from_gazebo_observation_matches_dataset_order():
    obs = {
        "controller": {
            "current_tcp_pose": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            "tcp_velocity": {"linear": [4, 5, 6], "angular": [7, 8, 9]},
            "tcp_error": [10, 11, 12, 13, 14, 15],
        },
        "joints": {"position": [16, 17, 18, 19, 20, 21, 22]},
        "wrist_wrench": {"force": [23, 24, 25], "torque": [26, 27, 28]},
    }
    state = lowdim_state_from_gazebo_observation(obs)
    assert state.shape == (32,)
    assert np.allclose(state[:7], [1, 2, 3, 0, 0, 0, 1])
    assert np.allclose(state[-6:], [23, 24, 25, 26, 27, 28])


class _FakeACT(nn.Module):
    def forward(self, state, center, left, right):
        del center, left, right
        base = state[:, :6] * 0.1
        return base[:, None, :].repeat(1, 8, 1)


def _obs():
    return {
        "controller": {
            "current_tcp_pose": {
                "position": [1, 2, 3],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            "tcp_velocity": {"linear": [4, 5, 6], "angular": [7, 8, 9]},
            "tcp_error": [10, 11, 12, 13, 14, 15],
        },
        "joints": {"position": [16, 17, 18, 19, 20, 21, 22]},
        "wrist_wrench": {"force": [23, 24, 25], "torque": [26, 27, 28]},
    }


def _write_fake_act_adapter_checkpoint(tmp_path):
    state = torch.zeros(1, 32)
    image = torch.zeros(1, 3, 256, 288)
    act_path = tmp_path / "fake_act.pt"
    traced = torch.jit.trace(_FakeACT(), (state, image, image, image))
    traced.save(str(act_path))
    actor = TorchScriptACTAdapterActor(
        act_base=traced,
        state_dim=32,
        action_dim=48,
        action_horizon=8,
        hidden_dim=16,
        num_layers=2,
        adapter_scale=1.0,
        adapter_delta_clip=0.05,
        action_clip=None,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "vision_offline_serl_config": {
                "state_dim": 32,
                "action_dim": 48,
                "action_horizon": 8,
                "camera_keys": [
                    "observation.images.center_camera",
                    "observation.images.left_camera",
                    "observation.images.right_camera",
                ],
            },
            "warmstart_report": {"adapter_scale": 1.0},
        },
        checkpoint_path,
    )
    return checkpoint_path, act_path


def test_act_adapter_gazebo_policy_requires_images_by_default(tmp_path):
    checkpoint_path, act_path = _write_fake_act_adapter_checkpoint(tmp_path)
    policy = ACTAdapterSERLGazeboPolicy(checkpoint_path, act_torchscript=act_path)

    with pytest.raises(RuntimeError, match="lowdim-only"):
        policy.act(_obs())


def test_act_adapter_gazebo_policy_allows_explicit_zero_image_interface_mode(tmp_path):
    checkpoint_path, act_path = _write_fake_act_adapter_checkpoint(tmp_path)
    policy = ACTAdapterSERLGazeboPolicy(
        checkpoint_path,
        act_torchscript=act_path,
        allow_zero_images=True,
        action_clip=None,
    )

    action = policy.act(_obs())

    assert len(action) == 6
    assert np.allclose(action, [0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    assert policy.last_action_components["delta_action_norm"] == 0.0


def test_act_adapter_gazebo_policy_accepts_encoded_camera_images(tmp_path):
    checkpoint_path, act_path = _write_fake_act_adapter_checkpoint(tmp_path)
    policy = ACTAdapterSERLGazeboPolicy(checkpoint_path, act_torchscript=act_path, action_clip=None)
    image = {
        "height": 1,
        "width": 1,
        "encoding": "rgb8",
        "data_b64": base64.b64encode(bytes([10, 20, 30])).decode("ascii"),
    }
    obs = _obs()
    obs["images"] = {
        "observation.images.center_camera": image,
        "observation.images.left_camera": image,
        "observation.images.right_camera": image,
    }

    action = policy.act(obs)

    assert len(action) == 6
    assert np.allclose(action[:3], [0.1, 0.2, 0.3])


def test_act_adapter_gazebo_policy_accepts_jpeg_camera_images(tmp_path):
    import cv2

    checkpoint_path, act_path = _write_fake_act_adapter_checkpoint(tmp_path)
    policy = ACTAdapterSERLGazeboPolicy(checkpoint_path, act_torchscript=act_path, action_clip=None)
    rgb = np.zeros((256, 288, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    image = {
        "height": 256,
        "width": 288,
        "encoding": "jpeg_rgb8",
        "data_b64": base64.b64encode(encoded.tobytes()).decode("ascii"),
    }
    obs = _obs()
    obs["images"] = {
        "observation.images.center_camera": image,
        "observation.images.left_camera": image,
        "observation.images.right_camera": image,
    }

    action = policy.act(obs)

    assert len(action) == 6
    assert np.allclose(action[:3], [0.1, 0.2, 0.3])


def test_act_adapter_gazebo_policy_clips_loaded_adapter_and_action(tmp_path):
    checkpoint_path, act_path = _write_fake_act_adapter_checkpoint(tmp_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint["actor"]["adapter.4.bias"].fill_(10.0)
    torch.save(checkpoint, checkpoint_path)
    policy = ACTAdapterSERLGazeboPolicy(
        checkpoint_path,
        act_torchscript=act_path,
        allow_zero_images=True,
        adapter_delta_clip=0.1,
        action_clip=0.2,
    )

    action = policy.act(_obs())

    assert max(abs(value) for value in action) <= 0.20001
    assert policy.last_action_components["raw_delta_action_norm"] > policy.last_action_components["delta_action_norm"]
