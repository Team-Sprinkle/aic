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


class CheatcodeInsertionRewardComponents(NamedTuple):
    total: torch.Tensor
    lateral_progress: torch.Tensor
    orientation_progress: torch.Tensor
    near_misaligned: torch.Tensor
    preinsert_hover: torch.Tensor
    axial_progress: torch.Tensor
    corridor: torch.Tensor
    inside_alignment: torch.Tensor
    retreat: torch.Tensor
    success_candidate: torch.Tensor
    g_lat_pre: torch.Tensor
    g_ori_pre: torch.Tensor
    g_align_pre: torch.Tensor
    g_lat_insert: torch.Tensor
    g_ori_insert: torch.Tensor
    g_align_insert: torch.Tensor
    near_gate: torch.Tensor
    inside_gate: torch.Tensor


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


def _pow4_gate(error: torch.Tensor, sigma: float) -> torch.Tensor:
    scaled = error / max(float(sigma), 1.0e-9)
    return torch.exp(-(scaled * scaled * scaled * scaled))


def _sigmoid_gate(value: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.sigmoid(value / max(float(scale), 1.0e-9))


def cheatcode_insertion_phase_reward(
    *,
    geometry: InsertionGeometry,
    previous_depth: torch.Tensor,
    previous_lateral_error: torch.Tensor,
    orientation_error: torch.Tensor,
    previous_orientation_error: torch.Tensor,
    sigma_lat_pre: float = 0.0025,
    sigma_lat_insert: float = 0.0015,
    sigma_theta_pre: float = 0.10,
    sigma_theta_insert: float = 0.06,
    lateral_progress_scale: float = 0.001,
    orientation_progress_scale: float = 0.02,
    axial_progress_scale: float = 0.001,
    hover_depth: float = -0.004,
    hover_scale: float = 0.002,
    near_gate_start: float = -0.008,
    near_gate_scale: float = 0.001,
    hover_gate_start: float = -0.100,
    hover_gate_scale: float = 0.010,
    inside_gate_scale: float = 0.001,
    near_misaligned_lateral_threshold: float = 0.0015,
    near_misaligned_orientation_threshold: float = 0.06,
    inside_lateral_scale: float = 0.001,
    inside_orientation_scale: float = 0.04,
    bypass_penalty_scale: float = 6.0,
    success_depth_fraction: float = 0.9,
    success_lateral_threshold: float = 0.0005,
    success_orientation_threshold: float = 0.03,
    lateral_progress_weight: float = 0.40,
    orientation_progress_weight: float = 0.30,
    near_misaligned_weight: float = 0.25,
    hover_weight: float = 0.15,
    axial_progress_weight: float = 0.30,
    corridor_weight: float = 1.50,
    inside_alignment_weight: float = 0.20,
    retreat_weight: float = 0.20,
) -> CheatcodeInsertionRewardComponents:
    """Phase-aware insertion reward modeled after the Gazebo CheatCode state machine.

    The reward first pays for lateral/orientation alignment outside the port,
    then permits forward insertion only inside a sharp alignment tube. Inward
    motion while lateral or orientation error is large is explicitly negative.
    """
    s = geometry.axial_depth
    r = geometry.lateral_error
    theta = orientation_error.to(device=s.device, dtype=s.dtype)
    previous_depth = previous_depth.to(device=s.device, dtype=s.dtype)
    previous_lateral_error = previous_lateral_error.to(device=s.device, dtype=s.dtype)
    previous_orientation_error = previous_orientation_error.to(device=s.device, dtype=s.dtype)

    g_lat_pre = _pow4_gate(r, sigma_lat_pre)
    g_ori_pre = _pow4_gate(theta, sigma_theta_pre)
    g_align_pre = g_lat_pre * g_ori_pre
    g_lat_insert = _pow4_gate(r, sigma_lat_insert)
    g_ori_insert = _pow4_gate(theta, sigma_theta_insert)
    g_align_insert = g_lat_insert * g_ori_insert

    near_gate = _sigmoid_gate(s - float(near_gate_start), near_gate_scale)
    hover_gate = _sigmoid_gate(s - float(hover_gate_start), hover_gate_scale)
    inside_gate = _sigmoid_gate(s, inside_gate_scale)

    r_lateral_progress = lateral_progress_reward(
        previous_lateral_error=previous_lateral_error,
        current_lateral_error=r,
        scale=lateral_progress_scale,
    )
    r_orientation_progress = ((previous_orientation_error - theta) / max(float(orientation_progress_scale), 1.0e-9)).clamp(
        min=-1.0,
        max=1.0,
    )
    r_near_misaligned = -near_gate * (
        torch.relu(r - float(near_misaligned_lateral_threshold))
        / max(float(near_misaligned_lateral_threshold), 1.0e-9)
        + torch.relu(theta - float(near_misaligned_orientation_threshold))
        / max(float(near_misaligned_orientation_threshold), 1.0e-9)
    )
    r_hover = -torch.abs(s - float(hover_depth)) / max(float(hover_scale), 1.0e-9)
    # Keep the pre-insertion attractor active across the near-gate curriculum
    # region. Otherwise backing far away can look better than staying near the
    # entrance while misaligned because near_gate intentionally decays outside.
    r_preinsert_hover = (1.0 - g_align_pre) * torch.maximum(near_gate, hover_gate) * r_hover

    delta_s = s - previous_depth
    forward = (delta_s / max(float(axial_progress_scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    forward_reward = forward * (2.0 * g_align_insert - 1.0)
    retreat_penalty = -inside_gate * torch.relu(-delta_s / max(float(axial_progress_scale), 1.0e-9))
    r_axial = torch.where(delta_s > 0.0, forward_reward, retreat_penalty)

    centered_depth = geometry.depth_fraction * g_align_insert
    bypass_penalty = geometry.depth_fraction * (1.0 - g_align_insert) * max(float(bypass_penalty_scale), 0.0)
    r_corridor = centered_depth - bypass_penalty
    r_inside_alignment = -inside_gate * (
        torch.square(r / max(float(inside_lateral_scale), 1.0e-9))
        + torch.square(theta / max(float(inside_orientation_scale), 1.0e-9))
    )
    r_retreat = -inside_gate * torch.relu(-delta_s / max(float(axial_progress_scale), 1.0e-9))
    success_candidate = (
        (geometry.depth_fraction >= float(success_depth_fraction))
        & (r <= float(success_lateral_threshold))
        & (theta <= float(success_orientation_threshold))
    ).to(dtype=s.dtype)

    total = (
        float(lateral_progress_weight) * r_lateral_progress
        + float(orientation_progress_weight) * r_orientation_progress
        + float(near_misaligned_weight) * r_near_misaligned
        + float(hover_weight) * r_preinsert_hover
        + float(axial_progress_weight) * r_axial
        + float(corridor_weight) * r_corridor
        + float(inside_alignment_weight) * r_inside_alignment
        + float(retreat_weight) * r_retreat
        + success_candidate
    )
    return CheatcodeInsertionRewardComponents(
        total=total,
        lateral_progress=r_lateral_progress,
        orientation_progress=r_orientation_progress,
        near_misaligned=r_near_misaligned,
        preinsert_hover=r_preinsert_hover,
        axial_progress=r_axial,
        corridor=r_corridor,
        inside_alignment=r_inside_alignment,
        retreat=r_retreat,
        success_candidate=success_candidate,
        g_lat_pre=g_lat_pre,
        g_ori_pre=g_ori_pre,
        g_align_pre=g_align_pre,
        g_lat_insert=g_lat_insert,
        g_ori_insert=g_ori_insert,
        g_align_insert=g_align_insert,
        near_gate=near_gate,
        inside_gate=inside_gate,
    )
