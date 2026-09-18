"""Protect small KL values from cancellation under mixed precision."""
import ast
from pathlib import Path

import torch
import pytest


def load_kl():
    path = Path(__file__).resolve().parents[3] / "scripts/train_verified_act.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "stable_gaussian_kl")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    return scope["stable_gaussian_kl"]


def test_small_bfloat16_kl_matches_double_precision_gaussian_reference():
    mu = torch.tensor([[0., .001, -.001], [.003, 0., -.002]], dtype=torch.bfloat16)
    log_var = torch.tensor([[.001, -.002, .003], [-.004, .005, -.006]], dtype=torch.bfloat16)
    reference = .5 * (mu.double().square() + log_var.double().exp() - 1 - log_var.double()).sum(-1).mean()
    result = load_kl()(mu, log_var)
    assert result.dtype == torch.float32 and result >= 0
    torch.testing.assert_close(result.double(), reference, atol=1e-9, rtol=1e-5)


def test_stable_kl_gradient_and_exact_prior():
    kl = load_kl()
    assert kl(torch.zeros(2, 3), torch.zeros(2, 3)) == 0
    mu = torch.tensor([[.01, -.02]], requires_grad=True)
    log_var = torch.tensor([[.03, -.04]], requires_grad=True)
    kl(mu, log_var).backward()
    torch.testing.assert_close(mu.grad, mu.detach())
    torch.testing.assert_close(log_var.grad, .5 * (log_var.detach().exp() - 1), atol=1e-8, rtol=1e-5)


def test_terminal_sampler_preserves_source_and_holdout_boundaries():
    import numpy as np
    path = Path(__file__).resolve().parents[3] / "scripts/train_verified_act.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "sample_training_indices")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    sample = scope["sample_training_indices"]
    groups = [np.array([1, 2, 3, 4]), np.array([10, 11, 12])]
    ends = [np.array([4]), np.array([12])]
    # The IDs between the groups stand in for withheld episode frames.
    selected = sample(np.random.default_rng(1), 1000, groups, [.5, .5], ends, .7)
    assert set(selected) <= {1, 2, 3, 4, 10, 11, 12}
    assert set(sample(np.random.default_rng(1), 1000, groups, [0., 1.], ends, .5)) <= {10, 11, 12}
    assert set(sample(np.random.default_rng(1), 1000, groups, [.5, .5], ends, 1.)) == {4, 12}


def test_normalization_rebase_preserves_encoder_inputs_and_physical_actions():
    path = Path(__file__).resolve().parents[3] / "scripts/train_verified_act.py"
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "rebase_act_normalization")
    scope = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)
    rebase = scope["rebase_act_normalization"]
    generator = torch.Generator().manual_seed(28)
    def random(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)
    old = {key: {"mean": random(width), "std": random(width).exp()} for key, width in [("observation.state", 7), ("action", 3)]}
    new = {key: {"mean": random(width), "std": random(width).exp()} for key, width in [("observation.state", 7), ("action", 3)]}
    projections = [("model.encoder_robot_state_input_proj", "observation.state", 7),
                   ("model.vae_encoder_robot_state_input_proj", "observation.state", 7),
                   ("model.vae_encoder_action_input_proj", "action", 3)]
    weights = {"model.action_head.weight": random(3, 5), "model.action_head.bias": random(3)}
    for name, _, width in projections:
        weights[name + ".weight"] = random(5, width)
        weights[name + ".bias"] = random(5)
    saved = {key: value.clone() for key, value in weights.items()}
    updated = rebase(weights, old, new)
    for name, feature, width in projections:
        physical = random(11, width)
        before = torch.nn.functional.linear((physical - old[feature]["mean"]) / old[feature]["std"], weights[name + ".weight"], weights[name + ".bias"])
        after = torch.nn.functional.linear((physical - new[feature]["mean"]) / new[feature]["std"], updated[name + ".weight"], updated[name + ".bias"])
        torch.testing.assert_close(after, before)
    hidden = random(11, 5)
    before = torch.nn.functional.linear(hidden, weights["model.action_head.weight"], weights["model.action_head.bias"]) * old["action"]["std"] + old["action"]["mean"]
    after = torch.nn.functional.linear(hidden, updated["model.action_head.weight"], updated["model.action_head.bias"]) * new["action"]["std"] + new["action"]["mean"]
    torch.testing.assert_close(after, before)
    for key in weights:
        torch.testing.assert_close(weights[key], saved[key])
    new["action"]["std"][0] = 0
    with pytest.raises(ValueError, match="Invalid normalization"):
        rebase(weights, old, new)
