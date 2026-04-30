from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from gazebo_rl.serl_policy import ACTAdapterSERLGazeboPolicy, TorchScriptACTAdapterActor
from gazebo_rl.serl_train import ReplayBuffer, VisionCritic, load_trainer, run_training


class _FakeACT(nn.Module):
    def forward(self, state, center, left, right):
        del center, left, right
        return torch.zeros((state.shape[0], 8, 6), dtype=state.dtype, device=state.device)


def _write_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
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
        action_clip=0.05,
    )
    critic1 = VisionCritic(
        state_dim=32,
        camera_keys=[
            "observation.images.center_camera",
            "observation.images.left_camera",
            "observation.images.right_camera",
        ],
        action_dim=48,
    )
    critic2 = VisionCritic(
        state_dim=32,
        camera_keys=[
            "observation.images.center_camera",
            "observation.images.left_camera",
            "observation.images.right_camera",
        ],
        action_dim=48,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
            "target_critic1": critic1.state_dict(),
            "target_critic2": critic2.state_dict(),
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
            "dataset_summary": {"task_family": "test"},
            "warmstart_report": {"adapter_scale": 1.0},
        },
        checkpoint_path,
    )
    return checkpoint_path, act_path


def _args(tmp_path: Path) -> argparse.Namespace:
    checkpoint, act_path = _write_checkpoint(tmp_path)
    return argparse.Namespace(
        checkpoint=checkpoint,
        act_torchscript=act_path,
        output_dir=tmp_path / "out",
        workspace_dir=tmp_path,
        engine_config=None,
        sim_distrobox=None,
        ground_truth=True,
        gazebo_gui=False,
        launch_rviz=False,
        max_episodes=1,
        max_steps=1,
        updates=1,
        batch_size=1,
        replay_capacity=8,
        max_minutes=1.0,
        per_trial_timeout_sec=1.0,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        adapter_lr=1e-4,
        critic_lr=1e-4,
        adapter_penalty_weight=1e-2,
        act_preservation_weight=1e-1,
        adapter_delta_clip=0.05,
        action_clip=0.05,
        include_images=True,
        allow_zero_images=True,
        dry_run=True,
        record_lerobot=False,
        record_root=None,
        record_repo_id="local/test",
        record_single_task="test",
        record_video=True,
        record_fps=30,
        record_resume=False,
        record_drain_sec=0.0,
        record_image_writer_processes=0,
        record_image_writer_threads_per_camera=1,
        record_video_encoding_batch_size=1,
    )


def test_gazebo_serl_dry_run_loads_actor_and_critics(tmp_path: Path) -> None:
    args = _args(tmp_path)

    summary = run_training(args)

    assert summary["status"] == "dry_run"
    assert summary["state_dim"] == 32
    assert summary["action_dim"] == 48
    assert summary["action_horizon"] == 8
    assert (tmp_path / "out" / "dry_run_summary.json").is_file()


def test_gazebo_serl_trainer_saves_reloadable_checkpoint(tmp_path: Path) -> None:
    args = _args(tmp_path)
    trainer, train_config = load_trainer(args)
    path = tmp_path / "gazebo_online.pt"

    trainer.save_checkpoint(path, train_config=train_config, step=1)
    reloaded = ACTAdapterSERLGazeboPolicy(
        path,
        act_torchscript=args.act_torchscript,
        allow_zero_images=True,
    )

    assert reloaded.state_dim == 32
    assert reloaded.action_dim == 48
    assert reloaded.action_horizon == 8


def test_replay_buffer_shapes_for_single_transition() -> None:
    replay = ReplayBuffer(capacity=4)
    obs = {
        "state": torch.zeros(1, 32),
        "images": {
            "observation.images.center_camera": torch.zeros(1, 3, 256, 288),
            "observation.images.left_camera": torch.zeros(1, 3, 256, 288),
            "observation.images.right_camera": torch.zeros(1, 3, 256, 288),
        },
    }
    replay.append(
        {
            "obs": obs,
            "next_obs": obs,
            "action": torch.zeros(48),
            "reward": torch.zeros(1),
            "done": torch.zeros(1),
        }
    )

    batch = replay.sample(1, torch.device("cpu"))

    assert batch["obs"]["state"].shape == (1, 32)
    assert batch["action"].shape == (1, 48)
    assert batch["reward"].shape == (1, 1)
