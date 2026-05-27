from types import SimpleNamespace

import torch
from torch import nn

from lerobot_robot_aic.vision_offline_serl import (
    ACTAdapterSERLActor,
    ACTChunkActor,
    VisionCritic,
    VisionOfflineSERLConfig,
    VisionOfflineSERLTrainer,
    _set_processor_pipeline_device,
)


class _FakeACTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 6)

    def forward(self, batch):
        actions = self.linear(batch["observation.state"]).reshape(-1, 3, 2)
        return actions, (None, None)


class _FakeACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_features=["observation.images.cam"],
            output_features={"action": SimpleNamespace(shape=(2,))},
        )
        self.model = _FakeACTModel()


def test_set_processor_pipeline_device_updates_loaded_device_steps():
    class DeviceStep:
        def __init__(self):
            self.device = "cuda"
            self.tensor_device = torch.device("cuda")

        def __post_init__(self):
            self.tensor_device = torch.device(self.device)

    pipeline = SimpleNamespace(steps=[DeviceStep(), SimpleNamespace(other="unchanged")])

    _set_processor_pipeline_device(pipeline, "cpu")

    assert pipeline.steps[0].device == "cpu"
    assert pipeline.steps[0].tensor_device == torch.device("cpu")
    assert pipeline.steps[1].other == "unchanged"


def test_act_chunk_actor_flattens_chunks_and_preserves_gradients():
    actor = ACTChunkActor(_FakeACTPolicy(), action_horizon=2)
    obs = {
        "state": torch.randn(5, 4),
        "images": {"observation.images.cam": torch.randn(5, 3, 32, 32)},
    }

    action = actor.mean_action(obs)
    assert action.shape == (5, 4)

    loss = action.square().mean()
    loss.backward()
    assert actor.act_policy.model.linear.weight.grad is not None


def test_vision_critic_forward_shape():
    critic = VisionCritic(
        state_dim=4,
        camera_keys=["observation.images.cam"],
        action_dim=6,
    )
    obs = {
        "state": torch.randn(2, 4),
        "images": {"observation.images.cam": torch.randn(2, 3, 64, 64)},
    }
    action = torch.randn(2, 6)

    q = critic(obs, action)
    assert q.shape == (2, 1)


def test_adapter_actor_zero_init_matches_act_and_freezes_act():
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=True,
    )
    obs = {
        "state": torch.randn(5, 4),
        "images": {"observation.images.cam": torch.randn(5, 3, 32, 32)},
    }

    components = actor.action_components(obs)
    assert torch.allclose(components["final_action"], components["base_action"])
    assert torch.count_nonzero(components["delta_action"]) == 0
    assert all(not p.requires_grad for p in actor.act_policy.parameters())
    assert all(p.requires_grad for p in actor.adapter.parameters())


def test_adapter_actor_unfrozen_act_allows_act_gradients():
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=False,
    )
    obs = {
        "state": torch.randn(5, 4),
        "images": {"observation.images.cam": torch.randn(5, 3, 32, 32)},
    }

    loss = actor.mean_action(obs).square().mean()
    loss.backward()
    assert actor.act_policy.model.linear.weight.grad is not None


def test_adapter_actor_clips_delta_and_final_action():
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=True,
        adapter_delta_clip=0.1,
        action_clip=0.2,
    )
    with torch.no_grad():
        for param in actor.adapter.parameters():
            param.zero_()
        actor.adapter[-1].bias.fill_(10.0)
    obs = {
        "state": torch.randn(5, 4),
        "images": {"observation.images.cam": torch.randn(5, 3, 32, 32)},
    }

    components = actor.action_components(obs)

    assert components["raw_delta_action"].max() > 1.0
    assert components["delta_action"].abs().max() <= 0.10001
    assert components["final_action"].abs().max() <= 0.20001


def test_adapter_training_step_with_frozen_act():
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=True,
    )
    trainer = VisionOfflineSERLTrainer(
        config=VisionOfflineSERLConfig(
            state_dim=4,
            action_dim=4,
            action_horizon=2,
            camera_keys=["observation.images.cam"],
            adapter_lr=1e-3,
            critic_lr=1e-3,
            state_encoding="none",
            state_encoding_indices=(),
        ),
        actor=actor,
        device="cpu",
    )
    batch = {
        "obs": {
            "state": torch.randn(3, 4),
            "images": {"observation.images.cam": torch.randn(3, 3, 32, 32)},
        },
        "next_obs": {
            "state": torch.randn(3, 4),
            "images": {"observation.images.cam": torch.randn(3, 3, 32, 32)},
        },
        "action": torch.randn(3, 4),
        "reward": torch.zeros(3, 1),
        "done": torch.zeros(3, 1),
    }

    metrics = trainer.train_step(batch)
    assert "adapter_delta_norm" in metrics
    assert actor.act_policy.model.linear.weight.grad is None


def test_adapter_checkpoint_save_load(tmp_path):
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=True,
    )
    trainer = VisionOfflineSERLTrainer(
        config=VisionOfflineSERLConfig(
            state_dim=4,
            action_dim=4,
            action_horizon=2,
            camera_keys=["observation.images.cam"],
            state_encoding="none",
            state_encoding_indices=(),
        ),
        actor=actor,
        device="cpu",
    )
    path = tmp_path / "checkpoint_latest.pt"
    trainer.save_checkpoint(
        path,
        train_config={"actor_mode": "act_adapter"},
        dataset_summary={"state_dim": 4},
        warmstart_report={"initial_delta_norm": 0.0},
        step=3,
    )

    checkpoint = torch.load(path, map_location="cpu")
    assert checkpoint["step"] == 3
    assert checkpoint["vision_offline_serl_config"]["actor_mode"] == "act_adapter"
    assert "actor" in checkpoint


def test_fourier_gelu_adapter_and_critic_shapes():
    actor = ACTAdapterSERLActor(
        _FakeACTPolicy(),
        action_horizon=2,
        state_dim=4,
        adapter_hidden_dim=8,
        adapter_num_layers=1,
        freeze_act=True,
        adapter_activation="gelu",
        state_encoding="fourier",
        state_encoding_indices=(0, 1, 2),
        state_encoding_num_bands=4,
        state_encoding_max_freq=8.0,
        state_encoding_scale=10.0,
    )
    critic = VisionCritic(
        state_dim=4,
        camera_keys=["observation.images.cam"],
        action_dim=4,
        activation="gelu",
        state_encoding="fourier",
        state_encoding_indices=(0, 1, 2),
        state_encoding_num_bands=4,
        state_encoding_max_freq=8.0,
        state_encoding_scale=10.0,
    )
    obs = {
        "state": torch.randn(2, 4),
        "images": {"observation.images.cam": torch.randn(2, 3, 64, 64)},
    }

    assert actor.mean_action(obs).shape == (2, 4)
    assert critic(obs, torch.randn(2, 4)).shape == (2, 1)
