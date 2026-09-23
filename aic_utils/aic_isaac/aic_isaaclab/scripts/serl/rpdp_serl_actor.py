"""Isaac runtime for the SAC-compatible RPDP trajectory mixture actor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from rpdp_policy_actor import IsaacRPDPPolicyActor


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy_module = load("rpdp_serl_policy_runtime", "rpdp_serl_policy.py")
geo = load("rpdp_serl_geometry_runtime", "rpdp_geometry.py")


class IsaacRPDPSERLActor(IsaacRPDPPolicyActor):
    policy_family = "rpdp_serl_mixture"

    def __init__(
        self,
        *,
        checkpoint: Path,
        source_bc_checkpoint: Path,
        perception_policy_checkpoint: Path,
        device: torch.device,
        stochastic_rollout: bool = False,
        component_hold_decisions: int = 3,
        seed: int = 20260922,
    ):
        super().__init__(
            checkpoint=source_bc_checkpoint,
            perception_policy_checkpoint=perception_policy_checkpoint,
            device=device,
        )
        bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.serl_checkpoint = Path(checkpoint)
        self.source_bc_checkpoint = Path(source_bc_checkpoint)
        self.stochastic_rollout = bool(stochastic_rollout)
        self.component_hold_decisions = max(1, int(component_hold_decisions))
        self._held_component: torch.Tensor | None = None
        self._held_component_remaining = 0
        self._sample_seed = int(seed)
        config = policy_module.MixturePolicyConfig(**bundle["config"])
        self.policy = policy_module.PortTrajectoryMixturePolicy(config).to(self.device)
        self.policy.load_state_dict(bundle["model_state_dict"])
        self.policy.eval()
        normalization = bundle["normalization"]
        self.register_buffer("serl_condition_mean", normalization["condition_mean"].float())
        self.register_buffer("serl_condition_std", normalization["condition_std"].float())
        self.register_buffer("serl_target_mean", normalization["target_mean"].float())
        self.register_buffer("serl_target_std", normalization["target_std"].float())
        self.condition_indices = list(bundle["condition_indices"])
        self.adapter = geo.ConnectorTCPAdapter(
            bundle["tcp_to_connector_p"].to(self.device),
            bundle["tcp_to_connector_q"].to(self.device),
        )
        del self.diffusion
        self._active_policy_sample: dict[str, torch.Tensor] | None = None

    def reset(self):
        super().reset()
        self._active_policy_sample = None
        self._held_component = None
        self._held_component_remaining = 0

    def _build_normalized_condition(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visual, pose, variance, visibility = self._perception(obs)
        state = obs["state"].to(self.device).float()
        full = torch.cat((visual, visibility, pose / 10.0, variance.clamp_min(0).sqrt() / 10.0, state), dim=1)
        if max(self.condition_indices, default=-1) >= 234:
            previous_state = state if self._previous_state is None else self._previous_state
            state_delta = state - previous_state
            previous_action = (
                torch.zeros((state.shape[0], 24), device=self.device)
                if self._previous_action is None
                else self._previous_action
            )
            force = state[:, 26:29]
            phase = torch.stack([
                torch.tensor(
                    [1.0, 0.0, 0.0, 0.0]
                    if float(torch.linalg.norm(pose[i])) > 3.0
                    else [0.0, 1.0, 0.0, 0.0]
                    if float(torch.linalg.norm(pose[i])) > 0.75
                    else [0.0, 0.0, 0.0, 1.0]
                    if float(torch.linalg.norm(force[i])) >= 5.0
                    else [0.0, 0.0, 1.0, 0.0],
                    device=self.device,
                )
                for i in range(state.shape[0])
            ])
            phase[torch.linalg.norm(force, dim=1) >= 5.0] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], device=self.device
            )
            full = torch.cat((full, state_delta, previous_action, phase), dim=1)
        selected = full[:, self.condition_indices]
        normalized = (selected - self.serl_condition_mean) / self.serl_condition_std
        return normalized, pose, state

    def _condition_from_replay(self, obs: dict[str, Any]) -> torch.Tensor:
        feature = obs.get("world_feature")
        if feature is None or feature.shape[-1] < self.policy.config.condition_dim:
            raise ValueError("RPDP SERL replay lacks the normalized causal condition feature")
        return feature[..., : self.policy.config.condition_dim].to(self.device).float()

    def _decode_action(
        self,
        normalized_pose6: torch.Tensor,
        *,
        obs: dict[str, Any],
        pose_world_mm: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state is None:
            state = obs["state"].to(self.device).float()
        if pose_world_mm is None:
            condition = self._condition_from_replay(obs)
            selected = condition * self.serl_condition_std + self.serl_condition_mean
            # Current condition contract is full and ordered: visual180,
            # visibility6, pose3/10, sigma3/10, state42, temporal context.
            pose_world_mm = selected[:, 186:189] * 10.0
        trajectory = normalized_pose6.reshape(-1, 4, 6) * self.serl_target_std + self.serl_target_mean
        next_p = trajectory[..., :3]
        next_q = geo.rotvec_to_quat(trajectory[..., 3:6])
        current_p, current_q = self._current_connector_pose(state, pose_world_mm)
        return self.adapter.waypoint_chunk(current_p, current_q, next_p, next_q).reshape(-1, 24)

    def distribution_from_obs(self, obs: dict[str, Any]):
        return self.policy.component_distribution(self._condition_from_replay(obs))

    def sac_samples(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        condition = self._condition_from_replay(obs)
        values = self.policy.actor_objective_samples(condition)
        batch, components, _ = values["samples"].shape
        repeated_obs = {
            key: (value[:, None].expand(-1, components, *value.shape[1:]).reshape(batch * components, *value.shape[1:])
                  if torch.is_tensor(value) else value)
            for key, value in obs.items()
            if key != "images"
        }
        repeated_obs["images"] = {
            key: value[:, None].expand(-1, components, *value.shape[1:]).reshape(batch * components, *value.shape[1:])
            for key, value in obs["images"].items()
        }
        decoded = self._decode_action(values["samples"].reshape(batch * components, -1), obs=repeated_obs)
        values["executed_actions"] = decoded.reshape(batch, components, -1)
        return values

    def mean_action(self, obs: dict[str, Any]) -> torch.Tensor:
        condition = self._condition_from_replay(obs)
        normalized, _ = self.policy.mode(condition)
        return self._decode_action(normalized, obs=obs)

    def sample_replay_action(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self._condition_from_replay(obs)
        sample = self.policy.sample(condition)
        return self._decode_action(sample["action"], obs=obs), sample["log_prob"]

    def _infer(self, obs: dict[str, Any]):
        with torch.no_grad():
            condition, pose, state = self._build_normalized_condition(obs)
            if self.stochastic_rollout:
                generator = torch.Generator(device=self.device).manual_seed(self._sample_seed)
                self._sample_seed += 1
                logits, components = self.policy.component_distribution(condition)
                probabilities = torch.softmax(logits, dim=-1)
                if self._held_component is None or self._held_component_remaining <= 0:
                    self._held_component = torch.multinomial(
                        probabilities, 1, generator=generator
                    ).squeeze(-1)
                    self._held_component_remaining = self.component_hold_decisions
                all_samples = components.rsample()
                index = self._held_component[:, None, None].expand(-1, 1, self.policy.action_dim)
                normalized = all_samples.gather(1, index).squeeze(1)
                sample = {
                    "action": normalized,
                    "component": self._held_component,
                    "log_prob": self.policy.mixture_log_prob(condition, normalized),
                    "probabilities": probabilities,
                }
                self._held_component_remaining -= 1
                self._active_policy_sample = {key: value.detach().clone() for key, value in sample.items()}
            else:
                normalized, component = self.policy.mode(condition)
                self._active_policy_sample = {
                    "action": normalized.detach().clone(),
                    "component": component.detach().clone(),
                    "log_prob": self.policy.mixture_log_prob(condition, normalized).detach().clone(),
                    "probabilities": torch.softmax(self.policy.parameters_for_distribution(condition)[0], dim=-1).detach().clone(),
                }
            action = self._decode_action(normalized, obs=obs, pose_world_mm=pose, state=state)
            feature = F.pad(condition, (0, max(0, 384 - condition.shape[1])))[:, :384].detach()
            obs["world_feature"] = feature
            self._previous_state = state.detach().clone()
            return action, feature

    def export_active_policy_sample(self) -> dict[str, torch.Tensor] | None:
        return self._active_policy_sample
