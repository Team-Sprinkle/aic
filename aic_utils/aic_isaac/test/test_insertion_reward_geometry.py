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


def _geom10(depth: float, lateral: float, *, sigma: float = 0.0015):
    body = torch.tensor([[depth, lateral, 0.0]], dtype=torch.float32)
    entrance = torch.zeros_like(body)
    target = torch.tensor([[0.010, 0.0, 0.0]], dtype=torch.float32)
    axis = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    return insertion_geometry.compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=sigma,
    )


def _phase(depth: float, lateral: float, theta: float, *, prev_depth: float, prev_lateral: float, prev_theta: float):
    return insertion_geometry.cheatcode_insertion_phase_reward(
        geometry=_geom10(depth, lateral),
        previous_depth=torch.tensor([prev_depth], dtype=torch.float32),
        previous_lateral_error=torch.tensor([prev_lateral], dtype=torch.float32),
        orientation_error=torch.tensor([theta], dtype=torch.float32),
        previous_orientation_error=torch.tensor([prev_theta], dtype=torch.float32),
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


def test_cheatcode_start_like_inward_motion_is_negative() -> None:
    reward = _phase(-0.005, 0.006, 0.10, prev_depth=-0.006, prev_lateral=0.006, prev_theta=0.10)
    assert reward.axial_progress.item() < 0.0
    assert reward.total.item() < 0.0


def test_cheatcode_start_like_lateral_improvement_is_positive() -> None:
    reward = _phase(-0.006, 0.005, 0.10, prev_depth=-0.006, prev_lateral=0.006, prev_theta=0.10)
    assert reward.lateral_progress.item() > 0.0


def test_cheatcode_start_like_orientation_improvement_is_positive() -> None:
    reward = _phase(-0.006, 0.006, 0.08, prev_depth=-0.006, prev_lateral=0.006, prev_theta=0.10)
    assert reward.orientation_progress.item() > 0.0


def test_cheatcode_aligned_preentry_inward_motion_is_positive() -> None:
    reward = _phase(-0.003, 0.0005, 0.03, prev_depth=-0.004, prev_lateral=0.0005, prev_theta=0.03)
    assert reward.axial_progress.item() > 0.0


def test_cheatcode_laterally_misaligned_inward_motion_is_negative() -> None:
    reward = _phase(-0.003, 0.003, 0.03, prev_depth=-0.004, prev_lateral=0.003, prev_theta=0.03)
    assert reward.axial_progress.item() < 0.0


def test_cheatcode_orientation_misaligned_inward_motion_is_negative() -> None:
    reward = _phase(-0.003, 0.0005, 0.15, prev_depth=-0.004, prev_lateral=0.0005, prev_theta=0.15)
    assert reward.axial_progress.item() < 0.0


def test_cheatcode_inside_misaligned_penalized_and_not_success() -> None:
    reward = _phase(0.004, 0.003, 0.03, prev_depth=0.004, prev_lateral=0.003, prev_theta=0.03)
    assert reward.inside_alignment.item() < 0.0
    assert reward.success_candidate.item() == 0.0


def test_cheatcode_seated_aligned_is_success_candidate_and_high() -> None:
    reward = _phase(0.010, 0.0002, 0.02, prev_depth=0.009, prev_lateral=0.0003, prev_theta=0.03)
    assert reward.success_candidate.item() == 1.0
    assert reward.total.item() > 1.0


def test_cheatcode_seated_laterally_bad_is_not_success_and_penalized() -> None:
    reward = _phase(0.010, 0.003, 0.02, prev_depth=0.009, prev_lateral=0.003, prev_theta=0.02)
    assert reward.success_candidate.item() == 0.0
    assert reward.corridor.item() < 0.0


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
