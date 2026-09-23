import importlib.util
from pathlib import Path
import sys

import torch


PATH = Path(__file__).parents[1] / "aic_isaaclab/scripts/serl/rpdp_serl_policy.py"
spec = importlib.util.spec_from_file_location("rpdp_serl_policy", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def policy(components=4):
    return m.PortTrajectoryMixturePolicy(
        m.MixturePolicyConfig(condition_dim=12, width=24, layers=1, components=components, covariance_rank=2,
                              fusion=True, visual_dim=6)
    )


def test_distribution_shapes_and_finite_log_probability():
    model = policy(4)
    condition = torch.randn(5, 12)
    sample = model.sample(condition)
    assert sample["action"].shape == (5, 24)
    assert sample["probabilities"].shape == (5, 4)
    assert torch.isfinite(sample["log_prob"]).all()
    assert torch.allclose(sample["probabilities"].sum(-1), torch.ones(5))


def test_single_policy_expands_to_distinct_lateral_modes_without_changing_primary_mode():
    source = policy(1)
    source.eval()
    condition = torch.randn(3, 12)
    source_mode, _ = source.mode(condition)
    mixture = m.expand_single_to_mixture(source, components=4, lateral_std_units=0.2)
    mixture.eval()
    mixture_mode, selected = mixture.mode(condition)
    assert torch.equal(selected, torch.zeros_like(selected))
    assert torch.allclose(mixture_mode, source_mode, atol=1e-6)
    _, means, _, _ = mixture.parameters_for_distribution(condition)
    assert float(m.component_separation(means)) > 0.0


def test_actor_objective_backpropagates_through_all_components():
    model = policy(3)
    condition = torch.randn(4, 12)
    values = model.actor_objective_samples(condition)
    q = -values["samples"].square().mean(-1)
    loss = m.expected_sac_actor_loss(
        probabilities=values["probabilities"],
        mixture_log_probs=values["mixture_log_probs"],
        q_values=q,
        alpha=0.01,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.mean_head.weight.grad is not None
    assert torch.isfinite(model.mean_head.weight.grad).all()


def test_single_gaussian_is_supported_as_matched_baseline():
    model = policy(1)
    condition = torch.randn(2, 12)
    mode, selected = model.mode(condition)
    assert mode.shape == (2, 24)
    assert torch.equal(selected, torch.zeros_like(selected))
