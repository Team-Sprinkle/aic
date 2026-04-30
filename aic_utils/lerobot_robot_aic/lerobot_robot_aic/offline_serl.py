#!/usr/bin/env python3

#
#  Copyright (C) 2026 Intrinsic Innovation LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Minimal offline SERL-style actor-critic pretraining components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def mlp(sizes: list[int], activation: type[nn.Module] = nn.ReLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[idx], sizes[idx + 1]))
        if idx < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        log_std_bounds: tuple[float, float] = (-5.0, 2.0),
    ):
        super().__init__()
        self.backbone = mlp([obs_dim, *hidden_dims], nn.ReLU)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_bounds = log_std_bounds

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(*self.log_std_bounds)
        return mean, log_std

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)[0]

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...] = (256, 256)):
        super().__init__()
        self.net = mlp([obs_dim + action_dim, *hidden_dims, 1], nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


@dataclass
class OfflineSERLConfig:
    obs_dim: int
    action_dim: int
    gamma: float = 0.99
    tau: float = 0.005
    bc_weight: float = 1.0
    cql_weight: float = 0.0
    lr: float = 3e-4
    hidden_dim: int = 256
    num_layers: int = 2
    action_horizon: int = 1


class OfflineSERLTrainer:
    def __init__(self, config: OfflineSERLConfig, device: torch.device | str):
        self.config = config
        self.device = torch.device(device)
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden = tuple(config.hidden_dim for _ in range(config.num_layers))
        self.actor = GaussianActor(config.obs_dim, config.action_dim, hidden).to(self.device)
        self.critic1 = Critic(config.obs_dim, config.action_dim, hidden).to(self.device)
        self.critic2 = Critic(config.obs_dim, config.action_dim, hidden).to(self.device)
        self.target_critic1 = Critic(config.obs_dim, config.action_dim, hidden).to(self.device)
        self.target_critic2 = Critic(config.obs_dim, config.action_dim, hidden).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=config.lr)

    def warm_start_actor_bias_from_action_head(self, action_head_bias: torch.Tensor) -> dict[str, Any]:
        """Seed the actor output bias from a compatible ACT action head bias.

        ACT and this offline SERL actor have different architectures, so only
        the output prior can be transferred without fabricating hidden-layer
        semantics. For chunked SERL actions, the single-step ACT bias is repeated
        across the configured action horizon.
        """
        if action_head_bias.ndim != 1:
            raise ValueError(f"Expected 1D ACT action_head.bias, got {tuple(action_head_bias.shape)}")
        single_action_dim = int(action_head_bias.numel())
        expected_dim = single_action_dim * int(self.config.action_horizon)
        if expected_dim != self.config.action_dim:
            raise ValueError(
                "ACT action head and SERL actor dimensions are incompatible: "
                f"{single_action_dim} * horizon {self.config.action_horizon} != {self.config.action_dim}"
            )
        repeated = action_head_bias.detach().to(self.device, dtype=self.actor.mean_head.bias.dtype)
        repeated = repeated.repeat(int(self.config.action_horizon))
        with torch.no_grad():
            self.actor.mean_head.bias.copy_(repeated)
        return {
            "mode": "action_head_bias",
            "single_action_dim": single_action_dim,
            "action_horizon": int(self.config.action_horizon),
            "transferred_tensors": ["model.action_head.bias -> actor.mean_head.bias"],
        }

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.device) for k, v in batch.items()}

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        batch = self._to_device(batch)
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action = self.actor.mean_action(next_obs)
            target_q = torch.minimum(
                self.target_critic1(next_obs, next_action),
                self.target_critic2(next_obs, next_action),
            )
            td_target = reward + self.config.gamma * (1.0 - done) * target_q

        q1 = self.critic1(obs, action)
        q2 = self.critic2(obs, action)
        td_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        conservative_loss = torch.zeros((), device=self.device)
        if self.config.cql_weight > 0.0:
            random_action = torch.randn_like(action)
            random_q1 = self.critic1(obs, random_action)
            random_q2 = self.critic2(obs, random_action)
            conservative_loss = (
                (random_q1 - q1.detach()).mean() + (random_q2 - q2.detach()).mean()
            )
        critic_loss = td_loss + self.config.cql_weight * conservative_loss

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_action = self.actor.mean_action(obs)
        actor_q = torch.minimum(
            self.critic1(obs, actor_action),
            self.critic2(obs, actor_action),
        )
        bc_loss = F.mse_loss(actor_action, action)
        actor_loss = -actor_q.mean() + self.config.bc_weight * bc_loss

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.soft_update_targets()

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "td_loss": float(td_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "q_mean": float(torch.minimum(q1, q2).mean().detach().cpu()),
            "reward_mean": float(reward.mean().detach().cpu()),
            "conservative_loss": float(conservative_loss.detach().cpu()),
        }

    def soft_update_targets(self) -> None:
        for target, source in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
                target_param.data.mul_(1.0 - self.config.tau)
                target_param.data.add_(self.config.tau * source_param.data)

    def save_checkpoint(
        self,
        path: Path,
        *,
        train_config: dict[str, Any],
        schema_summary: dict[str, Any],
        normalization_stats: dict[str, Any],
        step: int,
        warmstart_metadata: dict[str, Any] | None = None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "actor_optimizer": self.actor_opt.state_dict(),
                "critic_optimizer": self.critic_opt.state_dict(),
                "offline_serl_config": self.config.__dict__,
                "train_config": train_config,
                "dataset_schema": schema_summary,
                "normalization_stats": normalization_stats,
                "warmstart_metadata": warmstart_metadata or {},
                "step": step,
            },
            path,
        )
