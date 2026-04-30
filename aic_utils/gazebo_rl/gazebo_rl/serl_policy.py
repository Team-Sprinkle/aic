from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


def lowdim_state_from_gazebo_observation(obs: dict[str, Any]) -> np.ndarray:
    controller = obs.get("controller") or {}
    tcp_pose = controller.get("current_tcp_pose") or {}
    tcp_velocity = controller.get("tcp_velocity") or {}
    joints = obs.get("joints") or {}
    wrench = obs.get("wrist_wrench") or {}

    values: list[float] = []
    values.extend((tcp_pose.get("position") or [0.0, 0.0, 0.0])[:3])
    values.extend((tcp_pose.get("orientation_xyzw") or [0.0, 0.0, 0.0, 1.0])[:4])
    values.extend(((tcp_velocity or {}).get("linear") or [0.0, 0.0, 0.0])[:3])
    values.extend(((tcp_velocity or {}).get("angular") or [0.0, 0.0, 0.0])[:3])
    tcp_error = list(controller.get("tcp_error") or [])
    values.extend((tcp_error + [0.0] * 6)[:6])
    joint_positions = list(joints.get("position") or [])
    values.extend((joint_positions + [0.0] * 7)[:7])
    values.extend(((wrench or {}).get("force") or [0.0, 0.0, 0.0])[:3])
    values.extend(((wrench or {}).get("torque") or [0.0, 0.0, 0.0])[:3])
    return np.asarray(values[:32], dtype=np.float32)


class OfflineSERLGazeboPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str = "cpu"):
        from lerobot_robot_aic.offline_serl import GaussianActor

        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg = ckpt.get("offline_serl_config") or {}
        if int(cfg.get("obs_dim", 0)) != 32:
            raise ValueError(f"Gazebo SERL policy expects obs_dim=32, got {cfg.get('obs_dim')}")
        self.action_horizon = int(cfg.get("action_horizon", 1))
        self.single_action_dim = int((ckpt.get("dataset_schema") or {}).get("action_shape", [6])[0])
        hidden = tuple(int(cfg.get("hidden_dim", 256)) for _ in range(int(cfg.get("num_layers", 2))))
        self.actor = GaussianActor(
            obs_dim=int(cfg["obs_dim"]),
            action_dim=int(cfg["action_dim"]),
            hidden_dims=hidden,
        ).to(self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        stats = ckpt.get("normalization_stats") or {}
        self.obs_mean = torch.as_tensor(stats["obs_mean"], dtype=torch.float32, device=self.device)
        self.obs_std = torch.as_tensor(stats["obs_std"], dtype=torch.float32, device=self.device)
        self.action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=self.device)
        self.action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=self.device)

    def act(self, obs: dict[str, Any], *, explore: bool = False) -> list[float]:
        del explore
        lowdim = torch.as_tensor(lowdim_state_from_gazebo_observation(obs), device=self.device).unsqueeze(0)
        lowdim = (lowdim - self.obs_mean) / self.obs_std
        with torch.no_grad():
            normalized_action = self.actor.mean_action(lowdim).squeeze(0)
        action = normalized_action * self.action_std + self.action_mean
        return action[: self.single_action_dim].detach().cpu().numpy().astype(float).tolist()
