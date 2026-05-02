from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "aic_isaaclab"
    / "scripts"
    / "rsl_rl"
    / "serl_warmstart.py"
)
spec = importlib.util.spec_from_file_location("serl_warmstart", SCRIPT)
serl_warmstart = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(serl_warmstart)


def test_build_warmstarted_rsl_state_dict_uses_first_action_prior() -> None:
    rsl_state = {
        "std": torch.ones(6),
        "actor.0.weight": torch.zeros(512, 3154),
        "actor.0.bias": torch.zeros(512),
        "actor.6.bias": torch.zeros(6),
    }
    serl_ckpt = {
        "actor": {},
        "offline_serl_config": {"obs_dim": 32, "action_dim": 48, "action_horizon": 8},
        "normalization_stats": {
            "action_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6] + [0.0] * 42,
            "action_std": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] + [1.0] * 42,
        },
    }
    updated, report = serl_warmstart.build_warmstarted_rsl_state_dict(rsl_state, serl_ckpt)
    assert report["copied_action_prior"] is True
    assert updated["actor.6.bias"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert updated["std"].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
