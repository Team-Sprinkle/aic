from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

import pytest
import torch


FORCE_PENALTY = (
    Path(__file__).resolve().parents[1]
    / "aic_isaaclab"
    / "source"
    / "aic_task"
    / "aic_task"
    / "tasks"
    / "manager_based"
    / "aic_task"
    / "mdp"
    / "force_penalty.py"
)
spec = importlib.util.spec_from_file_location("aic_force_penalty", FORCE_PENALTY)
force_penalty = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules[spec.name] = force_penalty
spec.loader.exec_module(force_penalty)

force_delta_penalty_curve = force_penalty.force_delta_penalty_curve


def test_force_delta_penalty_curve_is_gentle_then_steep() -> None:
    delta = torch.tensor([5.0, 10.0, 15.0, 20.0, 22.0, 25.0, 28.0, 30.0, 40.0])

    penalty = force_delta_penalty_curve(
        delta,
        threshold=10.0,
        reference=20.0,
        knee_penalty_fraction=0.1,
        saturation=30.0,
        max_penalty=0.30,
    )

    expected = torch.tensor([0.0, 0.0, 0.0075, 0.03, 0.05808, 0.165, 0.27192, 0.30, 0.30])
    assert penalty.tolist() == pytest.approx(expected.tolist(), abs=1.0e-5)
