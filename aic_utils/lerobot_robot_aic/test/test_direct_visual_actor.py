"""Behavioral checks for the shared actor; no downloads, ROS, or simulator."""

from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lerobot_robot_aic.direct_visual_actor import DirectVisualActor, DirectVisualActorConfig, load_direct_visual_actor

torch.set_num_threads(1)


def make_actor(**kwargs):
    config = dict(state_dim=82, camera_keys=["observation.images.center_camera", "observation.images.left_camera"],
                  backbone="small_conv", image_size=28, hidden_dim=32, per_camera_dim=16)
    config.update(kwargs)
    return DirectVisualActor(DirectVisualActorConfig(**config))


def observation(actor):
    return {"state": torch.randn(2, actor.state_dim),
            "images": {key: torch.rand(2, 3, 32, 32) for key in actor.config.camera_keys}}


def payload(actor):
    return {"actor": actor.state_dict(), "vision_offline_serl_config": {
        "actor_mode": "direct_visual", "direct_visual_actor": asdict(actor.config),
        "state_dim": actor.state_dim, "action_dim": actor.action_dim,
        "action_horizon": actor.action_horizon, "camera_keys": actor.config.camera_keys}}


def test_images_affect_action_and_backbone_receives_gradients():
    torch.manual_seed(2)
    actor = make_actor()
    obs = observation(actor)
    for image in obs["images"].values():
        image.requires_grad_()
    action = actor.mean_action(obs)
    action.sum().backward()
    assert all(image.grad.abs().sum() > 0 for image in obs["images"].values())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.backbone.parameters())
    changed = {"state": obs["state"], "images": {k: torch.zeros_like(v) for k, v in obs["images"].items()}}
    assert not torch.allclose(action, actor.mean_action(changed), atol=1e-9)
    assert not hasattr(actor, "adapter") and not hasattr(actor, "act_base")


def test_actions_are_full_commands_with_independent_coordinate_limits():
    actor = make_actor(action_limits=(0.001, 0.002, 0.003, 0.01, 0.02, 0.03), action_horizon=2)
    with torch.no_grad():
        actor.head[-1].weight.zero_()
        actor.head[-1].bias.copy_(torch.tensor([100., -100.] * 6))
    parts = actor(observation(actor))
    assert torch.allclose(parts["final_action"].abs(), actor.action_limits.expand(2, -1))
    assert torch.count_nonzero(parts["delta_action"]) == 0
    assert torch.equal(parts["final_action"], parts["unclipped_final_action"])


def test_frozen_backbone_is_explicit_and_head_still_learns():
    actor = make_actor(freeze_backbone=True)
    actor.train()
    assert actor.training and not actor.backbone.training
    assert not any(p.requires_grad for p in actor.backbone.parameters())
    actor.mean_action(observation(actor)).sum().backward()
    assert actor.head[-1].weight.grad.abs().sum() > 0


def test_checkpoint_roundtrip_includes_normalization_and_needs_no_act():
    actor = make_actor()
    actor.fit_normalization(torch.randn(9, 82) * 3, torch.randn(9, 6) * 0.001)
    obs = observation(actor)
    restored = load_direct_visual_actor(payload(actor))
    assert torch.equal(actor.mean_action(obs), restored.mean_action(obs))
    assert restored.act_torchscript_path is None
    damaged = payload(actor)
    del damaged["actor"]["normalizer.state_std"]
    with pytest.raises(RuntimeError):
        load_direct_visual_actor(damaged)


@pytest.mark.parametrize("online_key", ["online_serl_config", "online_gazebo_serl_config"])
def test_online_checkpoints_share_actor_format(online_key):
    actor = make_actor()
    checkpoint = payload(actor)
    checkpoint[online_key] = {"checkpoint": {"vision_offline_serl_config": checkpoint.pop("vision_offline_serl_config")}}
    assert isinstance(load_direct_visual_actor(checkpoint), DirectVisualActor)


@pytest.mark.parametrize("bad", [torch.randn(2, 81), torch.randn(2, 83), torch.randn(82)])
def test_schema_mismatch_fails(bad):
    actor = make_actor()
    obs = observation(actor)
    obs["state"] = bad
    with pytest.raises(ValueError, match="State shape"):
        actor.mean_action(obs)


def test_missing_image_and_history_are_not_silently_ignored():
    actor = make_actor()
    obs = observation(actor)
    del obs["images"][actor.config.camera_keys[0]]
    with pytest.raises(KeyError):
        actor.mean_action(obs)
    obs = observation(actor)
    obs["actor_state"] = torch.randn(2, 164)
    with pytest.raises(ValueError, match="history"):
        actor.mean_action(obs)


@pytest.mark.parametrize("limits", [(0.,)*6, (float('nan'),)*6, (1.,)*5])
def test_invalid_physical_limits_rejected(limits):
    with pytest.raises(ValueError):
        make_actor(action_limits=limits)


def test_legacy_checkpoint_cannot_be_misread_as_direct_visual():
    with pytest.raises(ValueError, match="not a direct_visual"):
        load_direct_visual_actor({"vision_offline_serl_config": {"actor_mode": "act_direct"}})


def test_gazebo_loader_matches_offline_actor_without_act_export(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gazebo_rl"))
    from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy
    actor = make_actor()
    checkpoint = tmp_path / "direct.pt"
    torch.save(payload(actor), checkpoint)
    policy = ACTAdapterSERLGazeboPolicy(checkpoint)
    obs = observation(actor)
    obs = {"state": obs["state"][:1], "images": {k: v[:1] for k, v in obs["images"].items()}}
    monkeypatch.setattr(policy, "_obs_to_actor", lambda unused: obs)
    expected = actor.mean_action(obs).detach().numpy().reshape(1, 6)
    assert (policy.act_chunk({}, n_action_steps=1) == expected).all()


def test_bc_step_changes_visual_backbone():
    from lerobot_robot_aic.vision_offline_serl import VisionOfflineSERLConfig, VisionOfflineSERLTrainer
    actor = make_actor()
    cfg = VisionOfflineSERLConfig(state_dim=82, action_dim=6, action_horizon=1,
        camera_keys=actor.config.camera_keys, actor_mode="direct_visual", actor_update_mode="bc_only",
        adapter_penalty_weight=0, act_preservation_weight=0, state_encoding="none",
        critic_feature_dim=16, critic_hidden_dim=16, direct_visual_actor=actor.architecture_config())
    trainer = VisionOfflineSERLTrainer(config=cfg, actor=actor, device="cpu")
    before = next(actor.backbone.parameters()).detach().clone()
    metrics = trainer.train_step({"obs": observation(actor), "next_obs": observation(actor),
        "action": torch.randn(2, 6) * 0.01, "reward": torch.zeros(2, 1), "done": torch.ones(2, 1)})
    assert metrics["bc_loss"] > 0
    assert not torch.equal(before, next(actor.backbone.parameters()))
    assert metrics["adapter_penalty"] == metrics["act_preservation_loss"] == 0
