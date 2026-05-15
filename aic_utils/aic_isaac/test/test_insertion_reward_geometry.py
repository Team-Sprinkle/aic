from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch

MODULE = (
    Path(__file__).resolve().parents[1]
    / "aic_isaaclab"
    / "source"
    / "aic_task"
    / "aic_task"
    / "tasks"
    / "manager_based"
    / "aic_task"
    / "mdp"
    / "insertion_geometry.py"
)
AUDIT_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_insertion_reward_geometry.py"
spec = importlib.util.spec_from_file_location("insertion_geometry", MODULE)
insertion_geometry = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(insertion_geometry)


def _geom(depth: float, lateral: float, *, sigma: float = 0.0025):
    body = torch.tensor([[depth, lateral, 0.0]], dtype=torch.float32)
    entrance = torch.zeros_like(body)
    target = torch.tensor([[0.008, 0.0, 0.0]], dtype=torch.float32)
    axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    return insertion_geometry.compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=sigma,
    )


def test_on_center_forward_motion_is_positive() -> None:
    geom = _geom(0.001, 0.0)
    reward = insertion_geometry.signed_axial_progress_reward(
        previous_depth=torch.tensor([0.0005]),
        current_depth=geom.axial_depth,
        lateral_gate=geom.lateral_gate,
        scale=0.001,
    )
    assert reward.item() > 0.0


def test_off_center_forward_motion_is_negative() -> None:
    geom = _geom(0.001, 0.006)
    reward = insertion_geometry.signed_axial_progress_reward(
        previous_depth=torch.tensor([0.0005]),
        current_depth=geom.axial_depth,
        lateral_gate=geom.lateral_gate,
        scale=0.001,
    )
    assert reward.item() < 0.0


def test_bad_orientation_gate_makes_forward_motion_negative() -> None:
    geom = _geom(0.001, 0.0)
    reward = insertion_geometry.signed_axial_progress_reward(
        previous_depth=torch.tensor([0.0005]),
        current_depth=geom.axial_depth,
        lateral_gate=geom.lateral_gate,
        semantic_gate=torch.tensor([0.0]),
        scale=0.001,
    )
    assert reward.item() < 0.0


def test_reducing_lateral_error_is_positive() -> None:
    reward = insertion_geometry.lateral_progress_reward(
        previous_lateral_error=torch.tensor([0.004]),
        current_lateral_error=torch.tensor([0.002]),
        scale=0.001,
    )
    assert reward.item() > 0.0


def test_increasing_lateral_error_is_negative() -> None:
    reward = insertion_geometry.lateral_progress_reward(
        previous_lateral_error=torch.tensor([0.002]),
        current_lateral_error=torch.tensor([0.004]),
        scale=0.001,
    )
    assert reward.item() < 0.0


def test_seated_on_center_is_highest_probe_reward() -> None:
    seated = insertion_geometry.insertion_corridor_reward(_geom(0.008, 0.0), bypass_penalty_scale=2.0)
    half = insertion_geometry.insertion_corridor_reward(_geom(0.004, 0.0), bypass_penalty_scale=2.0)
    outside = insertion_geometry.insertion_corridor_reward(_geom(-0.001, 0.0), bypass_penalty_scale=2.0)
    assert seated.item() == pytest.approx(1.0)
    assert seated.item() > half.item() > outside.item()


def test_beside_port_deep_is_penalized() -> None:
    centered = insertion_geometry.insertion_corridor_reward(_geom(0.004, 0.0), bypass_penalty_scale=2.0)
    beside = insertion_geometry.insertion_corridor_reward(_geom(0.004, 0.006), bypass_penalty_scale=2.0)
    assert beside.item() < 0.0
    assert centered.item() > beside.item()


def test_deep_with_bad_orientation_gate_is_penalized() -> None:
    aligned = insertion_geometry.insertion_corridor_reward(
        _geom(0.004, 0.0),
        bypass_penalty_scale=2.0,
        semantic_gate=torch.tensor([1.0]),
    )
    misaligned = insertion_geometry.insertion_corridor_reward(
        _geom(0.004, 0.0),
        bypass_penalty_scale=2.0,
        semantic_gate=torch.tensor([0.0]),
    )
    assert aligned.item() > 0.0
    assert misaligned.item() < 0.0


def test_invalid_tiny_target_depth_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="target_depth_m"):
        insertion_geometry.compute_insertion_geometry(
            body_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            entrance_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            target_pos_w=torch.tensor([[0.001, 0.0, 0.0]], dtype=torch.float32),
            axis_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            lateral_gate_sigma=0.0025,
        )


def test_off_axis_target_fails_loudly() -> None:
    with pytest.raises(RuntimeError, match="target_lateral_residual_m"):
        insertion_geometry.compute_insertion_geometry(
            body_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            entrance_pos_w=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            target_pos_w=torch.tensor([[0.008, 0.003, 0.0]], dtype=torch.float32),
            axis_w=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            lateral_gate_sigma=0.0025,
        )


def test_reward_geometry_audit_default_depth_is_valid(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT_SCRIPT),
            "--output-root",
            str(tmp_path),
            "--run-name",
            "audit_default",
            "--grid",
            "17",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "off_axis_forward_motion_penalized" in result.stdout
    assert (tmp_path / "audit_default" / "summary.json").exists()
