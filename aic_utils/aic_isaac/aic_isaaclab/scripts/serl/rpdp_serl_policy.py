"""BC-warm-started probabilistic port-frame trajectory policy for SERL/RLPD."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MixturePolicyConfig:
    condition_dim: int
    horizon: int = 4
    width: int = 192
    layers: int = 4
    components: int = 1
    covariance_rank: int = 4
    fusion: bool = True
    visual_dim: int = 186
    min_std: float = 0.02
    max_std: float = 1.0


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    x = t.float()[:, None] * freq[None]
    return torch.cat((x.sin(), x.cos()), dim=-1)


class PortTrajectoryMixturePolicy(nn.Module):
    """Mixture of low-rank Gaussians over complete 4x6 port-frame poses.

    Every component predicts a complete trajectory.  The low-rank covariance
    correlates exploration across waypoints while a positive diagonal floor
    keeps the density and log probability well defined.
    """

    def __init__(self, config: MixturePolicyConfig):
        super().__init__()
        if config.components < 1:
            raise ValueError("components must be positive")
        if config.covariance_rank < 1:
            raise ValueError("covariance_rank must be positive")
        self.config = config
        self.action_dim = config.horizon * 6
        self.query = nn.Parameter(torch.randn(1, config.horizon, config.width) * 0.01)
        if config.fusion:
            if not 0 < config.visual_dim < config.condition_dim:
                raise ValueError("visual_dim must split condition_dim")
            self.visual_condition = nn.Sequential(
                nn.LayerNorm(config.visual_dim),
                nn.Linear(config.visual_dim, config.width),
                nn.SiLU(),
                nn.Linear(config.width, config.width),
            )
            pose_dim = config.condition_dim - config.visual_dim
            self.pose_condition = nn.Sequential(
                nn.LayerNorm(pose_dim),
                nn.Linear(pose_dim, config.width),
                nn.SiLU(),
                nn.Linear(config.width, config.width),
            )
            self.visual_gate = nn.Linear(config.width, config.width)
        else:
            self.condition = nn.Sequential(
                nn.LayerNorm(config.condition_dim),
                nn.Linear(config.condition_dim, config.width),
                nn.SiLU(),
                nn.Linear(config.width, config.width),
            )
        layer = nn.TransformerEncoderLayer(
            config.width,
            6,
            config.width * 4,
            dropout=0.05,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, config.layers)
        self.output_norm = nn.LayerNorm(config.width)
        self.mean_head = nn.Linear(config.width, config.components * 6)
        self.pooled_norm = nn.LayerNorm(config.width)
        self.logit_head = nn.Linear(config.width, config.components)
        self.log_std_head = nn.Linear(config.width, config.components * self.action_dim)
        self.factor_head = nn.Linear(
            config.width,
            config.components * self.action_dim * config.covariance_rank,
        )
        self.reset_distribution_heads()

    def reset_distribution_heads(self) -> None:
        nn.init.zeros_(self.logit_head.weight)
        nn.init.zeros_(self.logit_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        initial_std = max(self.config.min_std * 2.0, 1.0e-4)
        nn.init.constant_(self.log_std_head.bias, math.log(initial_std))
        nn.init.zeros_(self.factor_head.weight)
        nn.init.zeros_(self.factor_head.bias)

    def _context(self, condition: torch.Tensor) -> torch.Tensor:
        if self.config.fusion:
            visual = self.visual_condition(condition[:, : self.config.visual_dim])
            pose = self.pose_condition(condition[:, self.config.visual_dim :])
            return pose + torch.sigmoid(self.visual_gate(pose)) * visual
        return self.condition(condition)

    def features(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self._context(condition)
        hidden = self.blocks(self.query.expand(condition.shape[0], -1, -1) + context[:, None])
        hidden = self.output_norm(hidden)
        return hidden, self.pooled_norm(hidden.mean(dim=1))

    def parameters_for_distribution(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden, pooled = self.features(condition)
        batch = condition.shape[0]
        means = self.mean_head(hidden).reshape(
            batch, self.config.horizon, self.config.components, 6
        ).permute(0, 2, 1, 3).reshape(batch, self.config.components, self.action_dim)
        logits = self.logit_head(pooled)
        raw_log_std = self.log_std_head(pooled).reshape(batch, self.config.components, self.action_dim)
        min_log = math.log(self.config.min_std)
        max_log = math.log(self.config.max_std)
        log_std = raw_log_std.clamp(min=min_log, max=max_log)
        cov_diag = torch.exp(2.0 * log_std)
        factor = self.factor_head(pooled).reshape(
            batch,
            self.config.components,
            self.action_dim,
            self.config.covariance_rank,
        )
        # Bounded factors prevent an untrained head from creating huge actions.
        factor = 0.25 * torch.tanh(factor)
        return logits, means, cov_diag, factor

    def component_distribution(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.distributions.LowRankMultivariateNormal]:
        logits, means, cov_diag, factor = self.parameters_for_distribution(condition)
        dist = torch.distributions.LowRankMultivariateNormal(
            loc=means,
            cov_factor=factor,
            cov_diag=cov_diag,
        )
        return logits, dist

    def mixture_log_prob(self, condition: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits, components = self.component_distribution(condition)
        value = action[:, None, :].expand(-1, self.config.components, -1)
        return torch.logsumexp(F.log_softmax(logits, dim=-1) + components.log_prob(value), dim=-1)

    def mode(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, means, _, _ = self.parameters_for_distribution(condition)
        selected = logits.argmax(dim=-1)
        index = selected[:, None, None].expand(-1, 1, self.action_dim)
        return means.gather(1, index).squeeze(1), selected

    def sample(self, condition: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        logits, components = self.component_distribution(condition)
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        all_samples = components.rsample()
        index = selected[:, None, None].expand(-1, 1, self.action_dim)
        action = all_samples.gather(1, index).squeeze(1)
        return {
            "action": action,
            "component": selected,
            "log_prob": self.mixture_log_prob(condition, action),
            "probabilities": probabilities,
        }

    def actor_objective_samples(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        """One reparameterized sample per component for low-variance SAC loss."""
        logits, means, cov_diag, factor = self.parameters_for_distribution(condition)
        components = torch.distributions.LowRankMultivariateNormal(
            loc=means,
            cov_factor=factor,
            cov_diag=cov_diag,
        )
        samples = components.rsample()
        log_mix = F.log_softmax(logits, dim=-1)
        pairwise_components = torch.distributions.LowRankMultivariateNormal(
            loc=means[:, None, :, :].expand(-1, self.config.components, -1, -1),
            cov_factor=factor[:, None, :, :, :].expand(-1, self.config.components, -1, -1, -1),
            cov_diag=cov_diag[:, None, :, :].expand(-1, self.config.components, -1, -1),
        )
        expanded = samples[:, :, None, :].expand(
            -1, -1, self.config.components, -1
        )
        component_log_probs = pairwise_components.log_prob(expanded)
        mixture_log_probs = torch.logsumexp(
            log_mix[:, None, :] + component_log_probs,
            dim=-1,
        )
        return {
            "samples": samples,
            "mixture_log_probs": mixture_log_probs,
            "probabilities": torch.softmax(logits, dim=-1),
        }

    def initialize_component_biases(self, lateral_std_units: float = 0.25) -> None:
        """Give extra components safe, distinct initial lateral means.

        Coordinates are normalized pose coordinates.  The first component is
        unchanged.  Additional components receive coherent x/y offsets across
        all four waypoints.  Learning may later repurpose the components.
        """
        if self.config.components <= 1:
            return
        with torch.no_grad():
            bias = self.mean_head.bias.reshape(self.config.components, 6)
            # Spread the nonprimary modes around the local port-opening plane.
            # For K=4 this gives three evenly spaced lateral alternatives plus
            # the untouched BC component, without favoring one image axis.
            for component in range(1, self.config.components):
                angle = 2.0 * math.pi * float(component - 1) / float(self.config.components - 1)
                x = lateral_std_units * math.cos(angle)
                y = lateral_std_units * math.sin(angle)
                bias[component, 0] += x
                bias[component, 1] += y
            # Start with 70% on exact BC and share 30% among alternatives.
            # Equal logits for the alternatives make the ratio primary:other
            # equal to 0.7 : 0.3/(K-1).
            self.logit_head.bias.zero_()
            primary_ratio = 0.7 * float(self.config.components - 1) / 0.3
            self.logit_head.bias[0] = math.log(primary_ratio)


def expand_single_to_mixture(
    source: PortTrajectoryMixturePolicy,
    *,
    components: int,
    lateral_std_units: float = 0.25,
) -> PortTrajectoryMixturePolicy:
    if source.config.components != 1:
        raise ValueError("source policy must have one component")
    config = MixturePolicyConfig(**{**source.config.__dict__, "components": int(components)})
    target = PortTrajectoryMixturePolicy(config).to(next(source.parameters()).device)
    source_state = source.state_dict()
    shared = {
        key: value
        for key, value in source_state.items()
        if key in target.state_dict() and target.state_dict()[key].shape == value.shape
    }
    target.load_state_dict(shared, strict=False)
    with torch.no_grad():
        src_w = source.mean_head.weight.reshape(1, 6, source.config.width)
        src_b = source.mean_head.bias.reshape(1, 6)
        target.mean_head.weight.copy_(src_w.expand(components, -1, -1).reshape_as(target.mean_head.weight))
        target.mean_head.bias.copy_(src_b.expand(components, -1).reshape_as(target.mean_head.bias))
        src_ls_w = source.log_std_head.weight.reshape(1, source.action_dim, source.config.width)
        src_ls_b = source.log_std_head.bias.reshape(1, source.action_dim)
        target.log_std_head.weight.copy_(src_ls_w.expand(components, -1, -1).reshape_as(target.log_std_head.weight))
        target.log_std_head.bias.copy_(src_ls_b.expand(components, -1).reshape_as(target.log_std_head.bias))
    target.initialize_component_biases(lateral_std_units=lateral_std_units)
    target.train(source.training)
    return target


def expected_sac_actor_loss(
    *,
    probabilities: torch.Tensor,
    mixture_log_probs: torch.Tensor,
    q_values: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if probabilities.shape != mixture_log_probs.shape or probabilities.shape != q_values.shape:
        raise ValueError("probabilities, log probabilities, and Q values must have matching [B,K] shapes")
    return (probabilities * (float(alpha) * mixture_log_probs - q_values)).sum(dim=-1).mean()


def component_separation(means: torch.Tensor) -> torch.Tensor:
    """Mean pairwise trajectory distance, useful for collapse diagnostics."""
    if means.shape[1] <= 1:
        return torch.zeros((), device=means.device, dtype=means.dtype)
    distances = torch.cdist(means, means)
    mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    return distances[mask].mean()


def policy_summary(policy: PortTrajectoryMixturePolicy) -> dict[str, Any]:
    return {
        "config": dict(policy.config.__dict__),
        "parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
    }
