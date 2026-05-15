"""Pure insertion-geometry helpers shared by rewards and diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import torch


class InsertionGeometry(NamedTuple):
    axial_depth: torch.Tensor
    target_depth: torch.Tensor
    target_lateral_residual: torch.Tensor
    lateral_error: torch.Tensor
    lateral_gate: torch.Tensor
    depth_fraction: torch.Tensor
    axis: torch.Tensor


def normalize_axis(axis_w: torch.Tensor) -> torch.Tensor:
    """Return a unit insertion axis, preserving batched shape."""
    return axis_w / torch.linalg.norm(axis_w, dim=-1, keepdim=True).clamp(min=1.0e-9)


def compute_insertion_geometry(
    *,
    body_pos_w: torch.Tensor,
    entrance_pos_w: torch.Tensor,
    target_pos_w: torch.Tensor,
    axis_w: torch.Tensor,
    lateral_gate_sigma: float,
) -> InsertionGeometry:
    """Compute semantic port-frame insertion coordinates.

    ``axis_w`` must point from the port entrance into the port.  ``axial_depth``
    is positive inside the entrance plane and negative outside/above it.
    """
    axis = normalize_axis(axis_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype))
    entrance = entrance_pos_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
    target = target_pos_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
    target_delta = target - entrance
    raw_target_depth = torch.sum(target_delta * axis, dim=1)
    target_lateral_residual = torch.linalg.norm(
        target_delta - raw_target_depth.unsqueeze(1) * axis,
        dim=1,
    )
    min_target_depth_m = 0.003
    max_target_depth_m = 0.030
    max_target_lateral_residual_m = 0.002
    invalid_depth = (raw_target_depth < min_target_depth_m) | (raw_target_depth > max_target_depth_m)
    invalid_residual = target_lateral_residual > max_target_lateral_residual_m
    if bool((invalid_depth | invalid_residual).any().detach().cpu()):
        bad_ids = torch.nonzero(invalid_depth | invalid_residual, as_tuple=False).flatten()
        first = int(bad_ids[0].detach().cpu())
        raise RuntimeError(
            "Invalid insertion geometry: target/entrance/axis are inconsistent. "
            f"env_index={first} target_depth_m={float(raw_target_depth[first].detach().cpu()):.6f} "
            f"target_lateral_residual_m={float(target_lateral_residual[first].detach().cpu()):.6f}. "
            f"Expected {min_target_depth_m:.3f} <= target_depth_m <= {max_target_depth_m:.3f} "
            f"and lateral residual <= {max_target_lateral_residual_m:.3f}. "
            "Do not compute insertion rewards from collapsed or off-axis targets."
        )
    delta_from_entrance = body_pos_w - entrance
    axial_depth = torch.sum(delta_from_entrance * axis, dim=1)
    target_depth = raw_target_depth
    axial_component = axial_depth.unsqueeze(1) * axis
    lateral_error = torch.linalg.norm(delta_from_entrance - axial_component, dim=1)
    sigma = max(float(lateral_gate_sigma), 1.0e-9)
    lateral_gate = torch.exp(-torch.square(lateral_error / sigma))
    depth_fraction = (axial_depth / target_depth).clamp(min=0.0, max=1.0)
    return InsertionGeometry(
        axial_depth=axial_depth,
        target_depth=target_depth,
        target_lateral_residual=target_lateral_residual,
        lateral_error=lateral_error,
        lateral_gate=lateral_gate,
        depth_fraction=depth_fraction,
        axis=axis,
    )


def insertion_corridor_reward(
    geometry: InsertionGeometry,
    *,
    bypass_penalty_scale: float,
    semantic_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reward centered seated depth and penalize deep off-center bypass."""
    depth_fraction = geometry.depth_fraction
    gate = geometry.lateral_gate if semantic_gate is None else geometry.lateral_gate * semantic_gate.to(geometry.lateral_gate.device)
    centered_depth_reward = depth_fraction * gate
    bypass_penalty = depth_fraction * (1.0 - gate) * max(float(bypass_penalty_scale), 0.0)
    return centered_depth_reward - bypass_penalty


def signed_axial_progress_reward(
    *,
    previous_depth: torch.Tensor,
    current_depth: torch.Tensor,
    lateral_gate: torch.Tensor,
    scale: float,
    semantic_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reward signed progress from entrance toward seated depth.

    Forward progress is positive only when centered.  Forward progress while
    off-center is negative.  Backward progress remains negative; the term should
    not pay the policy for retreating beside the port.
    """
    progress = ((current_depth - previous_depth) / max(float(scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    gate = lateral_gate if semantic_gate is None else lateral_gate * semantic_gate.to(lateral_gate.device)
    signed_gate = 2.0 * gate - 1.0
    return torch.where(progress > 0.0, progress * signed_gate, progress)


def lateral_progress_reward(
    *,
    previous_lateral_error: torch.Tensor,
    current_lateral_error: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reward one-step reduction in lateral error."""
    return ((previous_lateral_error - current_lateral_error) / max(float(scale), 1.0e-9)).clamp(
        min=-1.0,
        max=1.0,
    )
