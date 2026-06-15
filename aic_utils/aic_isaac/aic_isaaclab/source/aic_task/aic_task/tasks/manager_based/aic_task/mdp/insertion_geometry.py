"""Pure insertion-geometry helpers shared by rewards and diagnostics."""

from __future__ import annotations

from typing import NamedTuple

import torch


class InsertionGeometry(NamedTuple):
    axial_depth: torch.Tensor
    target_depth: torch.Tensor
    target_lateral_residual: torch.Tensor
    lateral_residual: torch.Tensor
    lateral_error: torch.Tensor
    lateral_gate: torch.Tensor
    depth_fraction: torch.Tensor
    axis: torch.Tensor


class CheatcodeInsertionRewardComponents(NamedTuple):
    total: torch.Tensor
    lateral_progress: torch.Tensor
    orientation_progress: torch.Tensor
    lat_gate_phase: torch.Tensor
    lateral_alignment_action: torch.Tensor
    lateral_alignment_axial_quiet_gate: torch.Tensor
    off_axis_axial_action_penalty: torch.Tensor
    lateral_error_state_penalty: torch.Tensor
    action_toward_axis: torch.Tensor
    lateral_funnel: torch.Tensor
    near_misaligned: torch.Tensor
    preinsert_hover: torch.Tensor
    axial_progress: torch.Tensor
    preinsert_aligned_axial: torch.Tensor
    corridor: torch.Tensor
    inside_alignment: torch.Tensor
    retreat: torch.Tensor
    semantic_progress: torch.Tensor
    module_depth_progress: torch.Tensor
    success_candidate: torch.Tensor
    g_lat_pre: torch.Tensor
    g_ori_pre: torch.Tensor
    g_align_pre: torch.Tensor
    g_lat_insert: torch.Tensor
    g_ori_insert: torch.Tensor
    g_align_insert: torch.Tensor
    g_semantic: torch.Tensor
    g_action_axis: torch.Tensor
    g_insert_combined: torch.Tensor
    action_axial: torch.Tensor
    action_lateral: torch.Tensor
    action_rotation: torch.Tensor
    axial_forward_action_penalty: torch.Tensor
    action_lateral_sigma: torch.Tensor
    action_forward_gate: torch.Tensor
    near_gate: torch.Tensor
    inside_gate: torch.Tensor
    sigma_lat_pre: torch.Tensor
    sigma_lat_insert: torch.Tensor
    sigma_theta_pre: torch.Tensor
    sigma_theta_insert: torch.Tensor


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
    max_target_depth_m = 0.060
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
    lateral_residual = delta_from_entrance - axial_component
    lateral_error = torch.linalg.norm(lateral_residual, dim=1)
    sigma = max(float(lateral_gate_sigma), 1.0e-9)
    lateral_gate = torch.exp(-torch.square(lateral_error / sigma))
    depth_fraction = (axial_depth / target_depth).clamp(min=0.0, max=1.0)
    return InsertionGeometry(
        axial_depth=axial_depth,
        target_depth=target_depth,
        target_lateral_residual=target_lateral_residual,
        lateral_residual=lateral_residual,
        lateral_error=lateral_error,
        lateral_gate=lateral_gate,
        depth_fraction=depth_fraction,
        axis=axis,
    )


def insertion_corridor_reward(
    geometry: InsertionGeometry,
    *,
    bypass_penalty_scale: float,
    bypass_gate_tolerance: float = 0.0,
    semantic_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reward centered seated depth and penalize deep off-center bypass."""
    depth_fraction = geometry.depth_fraction
    gate = geometry.lateral_gate if semantic_gate is None else geometry.lateral_gate * semantic_gate.to(geometry.lateral_gate.device)
    centered_depth_reward = depth_fraction * gate
    tolerance = min(max(float(bypass_gate_tolerance), 0.0), 0.99)
    effective_tolerance = tolerance * depth_fraction
    gate_failure = torch.relu((1.0 - gate) - effective_tolerance) / (1.0 - effective_tolerance).clamp(min=1.0e-9)
    bypass_penalty = depth_fraction * gate_failure * max(float(bypass_penalty_scale), 0.0)
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


def _pow4_gate(error: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    if isinstance(sigma, torch.Tensor):
        sigma_t = sigma.to(device=error.device, dtype=error.dtype).clamp(min=1.0e-9)
    else:
        sigma_t = torch.as_tensor(max(float(sigma), 1.0e-9), device=error.device, dtype=error.dtype)
    scaled = error / sigma_t
    return torch.exp(-(scaled * scaled * scaled * scaled))


def _scheduled_lateral_sigma(
    axial_depth: torch.Tensor,
    *,
    near_sigma: float,
    far_sigma: float,
    far_depth: float,
    near_depth: float,
) -> torch.Tensor:
    """Linearly shrink lateral reward radius as the tip approaches the entrance."""
    denom = max(float(near_depth) - float(far_depth), 1.0e-9)
    alpha = ((axial_depth - float(far_depth)) / denom).clamp(min=0.0, max=1.0)
    return float(far_sigma) + alpha * (float(near_sigma) - float(far_sigma))


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
    schedule_lateral_radius: bool = False,
    sigma_lat_pre_far: float = 0.004,
    sigma_lat_insert_far: float = 0.004,
    lateral_radius_schedule_far_depth: float = -0.020,
    lateral_radius_schedule_near_depth: float = 0.0,
    sigma_theta_pre: float = 0.10,
    sigma_theta_insert: float = 0.06,
    schedule_orientation_tolerance: bool = False,
    sigma_theta_pre_far: float = 0.12,
    sigma_theta_insert_far: float = 0.10,
    orientation_tolerance_schedule_far_depth: float = -0.020,
    orientation_tolerance_schedule_near_depth: float = 0.0,
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
    near_misaligned_max: float = 25.0,
    lateral_funnel_scale: float = 0.010,
    lateral_funnel_max: float = 4.0,
    inside_lateral_scale: float = 0.001,
    inside_orientation_scale: float = 0.04,
    inside_alignment_max: float = 25.0,
    bypass_penalty_scale: float = 6.0,
    bypass_gate_tolerance: float = 0.0,
    success_depth_fraction: float = 0.9,
    success_lateral_threshold: float = 0.0005,
    success_orientation_threshold: float = 0.03,
    success_candidate_weight: float = 1.0,
    lateral_progress_weight: float = 0.40,
    orientation_progress_weight: float = 0.30,
    lateral_funnel_weight: float = 0.0,
    near_misaligned_weight: float = 0.25,
    hover_weight: float = 0.15,
    axial_progress_weight: float = 0.30,
    lateral_alignment_action_weight: float = 0.0,
    lateral_alignment_action_scale: float = 0.001,
    lateral_alignment_off_axis_threshold: float = 0.001,
    lateral_alignment_require_axial_quiet: bool = False,
    lateral_alignment_axial_quiet_scale: float = 0.0005,
    off_axis_axial_action_penalty_weight: float = 0.0,
    off_axis_axial_action_scale: float = 0.001,
    off_axis_axial_action_penalty_max: float = 0.0,
    lateral_error_state_penalty_weight: float = 0.0,
    lateral_error_state_penalty_scale: float = 0.010,
    lateral_error_state_penalty_max: float = 4.0,
    preinsert_aligned_axial_weight: float = 0.0,
    preinsert_aligned_axial_start: float = -0.035,
    preinsert_aligned_axial_scale: float = 0.005,
    phase_gate_insertion_credit: bool = False,
    stateful_phase_mode: bool = False,
    stateful_lateral_enter_threshold: float = 0.0010,
    stateful_orientation_enter_threshold: float = 0.035,
    stateful_axial_action_quiet_scale: float = 0.0005,
    stateful_axial_lateral_action_penalty_weight: float = 0.0,
    stateful_axial_lateral_action_scale: float = 0.00015,
    stateful_axial_lateral_action_penalty_max: float = 4.0,
    stateful_axial_alignment_loss_penalty_weight: float = 0.0,
    stateful_axial_alignment_loss_lateral_scale: float = 0.0005,
    stateful_axial_alignment_loss_orientation_scale: float = 0.02,
    stateful_axial_alignment_loss_penalty_max: float = 8.0,
    stateful_axial_pure_action_weight: float = 0.0,
    stateful_axial_impure_action_penalty_weight: float = 0.0,
    stateful_axial_impure_action_penalty_max: float = 4.0,
    stateful_axial_rotation_action_penalty_weight: float = 0.0,
    stateful_axial_rotation_action_scale: float = 0.00015,
    stateful_axial_rotation_action_penalty_max: float = 4.0,
    stateful_axial_forward_action_penalty_weight: float = 0.0,
    stateful_axial_forward_action_scale: float = 0.00005,
    stateful_axial_forward_action_penalty_max: float = 4.0,
    corridor_weight: float = 1.50,
    inside_alignment_weight: float = 0.20,
    retreat_weight: float = 0.20,
    action_delta_w: torch.Tensor | None = None,
    action_rotation_norm: torch.Tensor | None = None,
    action_axis_gate: bool = False,
    action_lateral_sigma: float = 0.00005,
    action_lateral_sigma_far: float = 0.00030,
    action_radius_schedule_far_depth: float = -0.020,
    action_radius_schedule_near_depth: float = 0.0,
    action_forward_scale: float = 0.00005,
    action_min_forward: float = 0.0,
    semantic_gate: torch.Tensor | None = None,
    previous_semantic_gate: torch.Tensor | None = None,
    semantic_progress_scale: float = 0.10,
    semantic_progress_weight: float = 0.0,
    semantic_loss_weight: float = 0.0,
    module_depth: torch.Tensor | None = None,
    previous_module_depth: torch.Tensor | None = None,
    module_depth_progress_scale: float = 0.001,
    module_depth_progress_weight: float = 0.0,
    module_depth_loss_weight: float = 0.0,
) -> CheatcodeInsertionRewardComponents:
    """Additive phase reward based on privileged insertion geometry."""
    s = geometry.axial_depth
    r = geometry.lateral_error
    theta = orientation_error.to(device=s.device, dtype=s.dtype)
    previous_depth = previous_depth.to(device=s.device, dtype=s.dtype)
    previous_lateral_error = previous_lateral_error.to(device=s.device, dtype=s.dtype)
    previous_orientation_error = previous_orientation_error.to(device=s.device, dtype=s.dtype)

    effective_sigma_lat_pre: torch.Tensor
    effective_sigma_lat_insert: torch.Tensor
    if bool(schedule_lateral_radius):
        effective_sigma_lat_pre = _scheduled_lateral_sigma(
            s,
            near_sigma=sigma_lat_pre,
            far_sigma=sigma_lat_pre_far,
            far_depth=lateral_radius_schedule_far_depth,
            near_depth=lateral_radius_schedule_near_depth,
        )
        effective_sigma_lat_insert = _scheduled_lateral_sigma(
            s,
            near_sigma=sigma_lat_insert,
            far_sigma=sigma_lat_insert_far,
            far_depth=lateral_radius_schedule_far_depth,
            near_depth=lateral_radius_schedule_near_depth,
        )
    else:
        effective_sigma_lat_pre = torch.full_like(s, float(sigma_lat_pre))
        effective_sigma_lat_insert = torch.full_like(s, float(sigma_lat_insert))

    effective_sigma_theta_pre: torch.Tensor
    effective_sigma_theta_insert: torch.Tensor
    if bool(schedule_orientation_tolerance):
        effective_sigma_theta_pre = _scheduled_lateral_sigma(
            s,
            near_sigma=sigma_theta_pre,
            far_sigma=sigma_theta_pre_far,
            far_depth=orientation_tolerance_schedule_far_depth,
            near_depth=orientation_tolerance_schedule_near_depth,
        )
        effective_sigma_theta_insert = _scheduled_lateral_sigma(
            s,
            near_sigma=sigma_theta_insert,
            far_sigma=sigma_theta_insert_far,
            far_depth=orientation_tolerance_schedule_far_depth,
            near_depth=orientation_tolerance_schedule_near_depth,
        )
    else:
        effective_sigma_theta_pre = torch.full_like(s, float(sigma_theta_pre))
        effective_sigma_theta_insert = torch.full_like(s, float(sigma_theta_insert))

    g_lat_pre = _pow4_gate(r, effective_sigma_lat_pre)
    g_ori_pre = _pow4_gate(theta, effective_sigma_theta_pre)
    g_align_pre = g_lat_pre * g_ori_pre
    g_lat_insert = _pow4_gate(r, effective_sigma_lat_insert)
    g_ori_insert = _pow4_gate(theta, effective_sigma_theta_insert)
    g_align_insert = g_lat_insert * g_ori_insert
    if action_rotation_norm is None:
        action_rotation = torch.zeros_like(s)
    else:
        action_rotation = action_rotation_norm.to(device=s.device, dtype=s.dtype)
        if action_rotation.shape != s.shape:
            action_rotation = action_rotation.reshape(s.shape)
    action_rotation_quiet_gate = torch.exp(
        -torch.square(action_rotation / max(float(stateful_axial_rotation_action_scale), 1.0e-9))
    )
    if bool(action_axis_gate) and action_delta_w is not None:
        action = action_delta_w.to(device=s.device, dtype=s.dtype)
        if action.ndim != 2 or action.shape[-1] < 3:
            raise ValueError("action_delta_w must have shape (num_envs, >=3) when action_axis_gate is enabled")
        action_xyz = action[:, :3]
        action_axial = torch.sum(action_xyz * geometry.axis, dim=1)
        action_lateral = torch.linalg.norm(action_xyz - action_axial.unsqueeze(1) * geometry.axis, dim=1)
        action_sigma = _scheduled_lateral_sigma(
            s,
            near_sigma=action_lateral_sigma,
            far_sigma=action_lateral_sigma_far,
            far_depth=action_radius_schedule_far_depth,
            near_depth=action_radius_schedule_near_depth,
        )
        action_direction_gate = _pow4_gate(action_lateral, action_sigma)
        action_forward_gate = _sigmoid_gate(action_axial - float(action_min_forward), action_forward_scale)
        g_action_axis = action_direction_gate * action_forward_gate * action_rotation_quiet_gate
    else:
        action_xyz = torch.zeros((s.shape[0], 3), device=s.device, dtype=s.dtype)
        action_axial = torch.zeros_like(s)
        action_lateral = torch.zeros_like(s)
        action_sigma = torch.full_like(s, float(action_lateral_sigma))
        action_direction_gate = torch.ones_like(s)
        action_forward_gate = torch.ones_like(s)
        g_action_axis = torch.ones_like(s)
    lateral_dir = geometry.lateral_residual / r.unsqueeze(1).clamp(min=1.0e-9)
    action_lateral_vec = action_xyz - action_axial.unsqueeze(1) * geometry.axis
    action_toward_axis = -torch.sum(action_lateral_vec * lateral_dir, dim=1)
    if semantic_gate is None:
        g_semantic = torch.ones_like(s)
    else:
        g_semantic = semantic_gate.to(device=s.device, dtype=s.dtype).clamp(min=0.0, max=1.0)
    if previous_semantic_gate is None:
        previous_semantic = g_semantic.detach()
    else:
        previous_semantic = previous_semantic_gate.to(device=s.device, dtype=s.dtype).clamp(min=0.0, max=1.0)
    # The trailing-body consistency gate is a seated-depth check. Applying it
    # from the start prevents the policy from receiving credit for beginning a
    # valid aligned insertion because the trailing body cannot be near its
    # seated-depth reference until the plug has already advanced. Ramp it in
    # with depth: early insertion is governed by tip alignment/action, while
    # near full depth the consistency body becomes mandatory.
    semantic_depth_weight = geometry.depth_fraction.clamp(min=0.0, max=1.0)
    g_semantic_depth = (1.0 - semantic_depth_weight) + semantic_depth_weight * g_semantic
    g_insert_combined = g_align_insert * g_action_axis * g_semantic_depth

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
    r_lateral_funnel = -torch.minimum(
        torch.square(r / max(float(lateral_funnel_scale), 1.0e-9)),
        torch.full_like(r, max(float(lateral_funnel_max), 0.0)),
    )
    lat_gate_phase = torch.exp(-torch.square(r / effective_sigma_lat_pre.clamp(min=1.0e-9)))
    off_axis_alignment_phase = (r > float(lateral_alignment_off_axis_threshold)).to(dtype=s.dtype)
    axis_to_center = -lateral_dir
    action_side_vec = action_lateral_vec - action_toward_axis.unsqueeze(1) * axis_to_center
    action_side_slip = torch.linalg.norm(action_side_vec, dim=1)
    toward_axis_score = torch.relu(action_toward_axis / max(float(lateral_alignment_action_scale), 1.0e-9)).clamp(
        max=1.0,
    )
    away_axis_penalty = torch.relu(-action_toward_axis / max(float(lateral_alignment_action_scale), 1.0e-9)).clamp(
        max=1.0,
    )
    axial_forward_for_lateral = torch.relu(action_axial)
    r_lateral_alignment_axial_quiet_gate = (
        torch.exp(-torch.square(axial_forward_for_lateral / max(float(lateral_alignment_axial_quiet_scale), 1.0e-9)))
        if bool(lateral_alignment_require_axial_quiet)
        else torch.ones_like(toward_axis_score)
    )
    lateral_alignment_side_quiet_gate = torch.exp(
        -torch.square(action_side_slip / max(float(lateral_alignment_action_scale), 1.0e-9))
    )
    lateral_did_not_worsen = (r_lateral_progress >= -1.0e-6).to(dtype=s.dtype)
    r_lateral_alignment_action = (
        toward_axis_score
        * r_lateral_alignment_axial_quiet_gate
        * lateral_alignment_side_quiet_gate
        * lateral_did_not_worsen
        - away_axis_penalty
    ).clamp(min=-1.0, max=1.0)
    r_off_axis_axial_action_penalty = (
        (1.0 - lat_gate_phase)
        * torch.square(torch.relu(action_axial / max(float(off_axis_axial_action_scale), 1.0e-9)))
    )
    r_any_axis_action_penalty = torch.square(
        torch.abs(action_axial) / max(float(stateful_axial_action_quiet_scale), 1.0e-9)
    )
    r_axial_phase_lateral_action_penalty = torch.square(
        action_lateral / max(float(stateful_axial_lateral_action_scale), 1.0e-9)
    )
    r_axial_phase_rotation_action_penalty = torch.square(
        action_rotation / max(float(stateful_axial_rotation_action_scale), 1.0e-9)
    )
    r_axial_phase_forward_action_penalty = torch.square(
        torch.relu(float(action_min_forward) - action_axial)
        / max(float(stateful_axial_forward_action_scale), 1.0e-9)
    )
    r_axial_phase_alignment_loss_penalty = (
        torch.relu(r - float(stateful_lateral_enter_threshold))
        / max(float(stateful_axial_alignment_loss_lateral_scale), 1.0e-9)
        + torch.relu(theta - float(stateful_orientation_enter_threshold))
        / max(float(stateful_axial_alignment_loss_orientation_scale), 1.0e-9)
    )
    axial_phase_lateral_worsening = torch.relu(r - previous_lateral_error) / max(
        float(stateful_axial_alignment_loss_lateral_scale),
        1.0e-9,
    )
    axial_phase_orientation_worsening = torch.relu(theta - previous_orientation_error) / max(
        float(stateful_axial_alignment_loss_orientation_scale),
        1.0e-9,
    )
    axial_phase_alignment_preservation_gate = torch.exp(
        -torch.square(axial_phase_lateral_worsening) - torch.square(axial_phase_orientation_worsening)
    )
    axial_phase_alignment_nonworsening_gate = (
        (r <= previous_lateral_error + 1.0e-7)
        & (theta <= previous_orientation_error + 1.0e-5)
    ).to(dtype=s.dtype)
    r_axial_phase_alignment_loss_penalty = (
        r_axial_phase_alignment_loss_penalty
        + axial_phase_lateral_worsening
        + axial_phase_orientation_worsening
    )
    action_norm = torch.linalg.norm(action_xyz, dim=1)
    action_forward_score = torch.relu(action_axial / max(float(action_forward_scale), 1.0e-9)).clamp(max=1.0)
    r_axial_phase_pure_action = (
        g_align_insert
        * axial_phase_alignment_preservation_gate
        * axial_phase_alignment_nonworsening_gate
        * action_direction_gate
        * action_rotation_quiet_gate
        * action_forward_gate
        * action_forward_score
    )
    action_activity = torch.relu(
        action_norm / max(float(action_lateral_sigma), float(action_forward_scale), 1.0e-9)
    ).clamp(max=1.0)
    r_axial_phase_impure_action_penalty = action_activity * (1.0 - r_axial_phase_pure_action)
    if float(off_axis_axial_action_penalty_max) > 0.0:
        r_off_axis_axial_action_penalty = torch.minimum(
            r_off_axis_axial_action_penalty,
            torch.full_like(r_off_axis_axial_action_penalty, float(off_axis_axial_action_penalty_max)),
        )
        r_any_axis_action_penalty = torch.minimum(
            r_any_axis_action_penalty,
            torch.full_like(r_any_axis_action_penalty, float(off_axis_axial_action_penalty_max)),
        )
    if float(stateful_axial_lateral_action_penalty_max) > 0.0:
        r_axial_phase_lateral_action_penalty = torch.minimum(
            r_axial_phase_lateral_action_penalty,
            torch.full_like(r_axial_phase_lateral_action_penalty, float(stateful_axial_lateral_action_penalty_max)),
        )
    if float(stateful_axial_alignment_loss_penalty_max) > 0.0:
        r_axial_phase_alignment_loss_penalty = torch.minimum(
            r_axial_phase_alignment_loss_penalty,
            torch.full_like(
                r_axial_phase_alignment_loss_penalty,
                float(stateful_axial_alignment_loss_penalty_max),
            ),
        )
    if float(stateful_axial_impure_action_penalty_max) > 0.0:
        r_axial_phase_impure_action_penalty = torch.minimum(
            r_axial_phase_impure_action_penalty,
            torch.full_like(r_axial_phase_impure_action_penalty, float(stateful_axial_impure_action_penalty_max)),
        )
    if float(stateful_axial_rotation_action_penalty_max) > 0.0:
        r_axial_phase_rotation_action_penalty = torch.minimum(
            r_axial_phase_rotation_action_penalty,
            torch.full_like(
                r_axial_phase_rotation_action_penalty,
                float(stateful_axial_rotation_action_penalty_max),
            ),
        )
    if float(stateful_axial_forward_action_penalty_max) > 0.0:
        r_axial_phase_forward_action_penalty = torch.minimum(
            r_axial_phase_forward_action_penalty,
            torch.full_like(
                r_axial_phase_forward_action_penalty,
                float(stateful_axial_forward_action_penalty_max),
            ),
        )
    r_lateral_error_state_penalty = torch.minimum(
        torch.square(r / max(float(lateral_error_state_penalty_scale), 1.0e-9)),
        torch.full_like(r, max(float(lateral_error_state_penalty_max), 0.0)),
    )
    near_misaligned_error = torch.minimum(
        torch.relu(r - float(near_misaligned_lateral_threshold))
        / max(float(near_misaligned_lateral_threshold), 1.0e-9)
        + torch.relu(theta - float(near_misaligned_orientation_threshold))
        / max(float(near_misaligned_orientation_threshold), 1.0e-9),
        torch.full_like(r, max(float(near_misaligned_max), 0.0)),
    )
    r_near_misaligned = -near_gate * near_misaligned_error
    r_hover = -torch.abs(s - float(hover_depth)) / max(float(hover_scale), 1.0e-9)
    # Keep the pre-insertion attractor active across the near-gate curriculum
    # region. Otherwise backing far away can look better than staying near the
    # entrance while misaligned because near_gate intentionally decays outside.
    r_preinsert_hover = (1.0 - g_align_pre) * torch.maximum(near_gate, hover_gate) * r_hover

    delta_s = s - previous_depth
    forward = (delta_s / max(float(axial_progress_scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    forward_reward = forward * (2.0 * g_insert_combined - 1.0)
    retreat_penalty = -inside_gate * torch.relu(-delta_s / max(float(axial_progress_scale), 1.0e-9))
    r_axial = torch.where(delta_s > 0.0, forward_reward, retreat_penalty)
    preinsert_aligned_window = (1.0 - inside_gate) * _sigmoid_gate(
        s - float(preinsert_aligned_axial_start),
        preinsert_aligned_axial_scale,
    )
    r_preinsert_aligned_axial = (
        torch.relu(forward)
        * g_align_pre
        * g_action_axis
        * preinsert_aligned_window
    )

    centered_depth = geometry.depth_fraction * g_insert_combined
    tolerance = min(max(float(bypass_gate_tolerance), 0.0), 0.99)
    effective_bypass_tolerance = tolerance * geometry.depth_fraction
    gate_failure = torch.relu((1.0 - g_insert_combined) - effective_bypass_tolerance) / (
        1.0 - effective_bypass_tolerance
    ).clamp(min=1.0e-9)
    bypass_penalty = geometry.depth_fraction * gate_failure * max(float(bypass_penalty_scale), 0.0)
    r_corridor = centered_depth - bypass_penalty
    inside_alignment_error = torch.minimum(
        torch.square(r / max(float(inside_lateral_scale), 1.0e-9))
        + torch.square(theta / max(float(inside_orientation_scale), 1.0e-9)),
        torch.full_like(r, max(float(inside_alignment_max), 0.0)),
    )
    r_inside_alignment = -inside_gate * inside_alignment_error
    r_retreat = -inside_gate * torch.relu(-delta_s / max(float(axial_progress_scale), 1.0e-9))
    semantic_progress_raw = ((g_semantic - previous_semantic) / max(float(semantic_progress_scale), 1.0e-9)).clamp(
        min=-1.0,
        max=1.0,
    )
    semantic_active = torch.maximum(inside_gate, geometry.depth_fraction.clamp(min=0.0, max=1.0))
    r_semantic_progress = semantic_active * semantic_progress_raw
    if module_depth is None or previous_module_depth is None:
        r_module_depth_progress = torch.zeros_like(s)
    else:
        module_s = module_depth.to(device=s.device, dtype=s.dtype)
        previous_module_s = previous_module_depth.to(device=s.device, dtype=s.dtype)
        module_delta = module_s - previous_module_s
        module_forward = (module_delta / max(float(module_depth_progress_scale), 1.0e-9)).clamp(
            min=-1.0,
            max=1.0,
        )
        r_module_depth_progress = torch.where(
            module_delta > 0.0,
            torch.relu(module_forward) * g_align_insert * g_action_axis,
            module_forward,
        )
    success_candidate = (
        (geometry.depth_fraction >= float(success_depth_fraction))
        & (r <= float(success_lateral_threshold))
        & (theta <= float(success_orientation_threshold))
    ).to(dtype=s.dtype)

    positive_lateral_progress = torch.relu(r_lateral_progress)
    negative_lateral_progress = torch.minimum(r_lateral_progress, torch.zeros_like(r_lateral_progress))
    phase_lateral_progress = off_axis_alignment_phase * (
        positive_lateral_progress + negative_lateral_progress
    )
    positive_orientation_progress = torch.relu(r_orientation_progress)
    negative_orientation_progress = torch.minimum(r_orientation_progress, torch.zeros_like(r_orientation_progress))
    positive_axial_progress = torch.relu(forward) * g_insert_combined
    bad_forward_progress = torch.relu(forward) * (1.0 - g_insert_combined)
    off_axis_inserted_depth_penalty = torch.minimum(
        torch.relu(s) / max(float(axial_progress_scale), 1.0e-9),
        torch.full_like(s, max(float(off_axis_axial_action_penalty_max), 0.0)),
    )
    positive_corridor = geometry.depth_fraction * g_insert_combined
    negative_corridor = -geometry.depth_fraction * gate_failure * max(
        float(bypass_penalty_scale),
        0.0,
    )

    insertion_credit_gate = lat_gate_phase if bool(phase_gate_insertion_credit) else torch.ones_like(lat_gate_phase)
    gated_positive_axial = lat_gate_phase * positive_axial_progress
    gated_preinsert_aligned_axial = insertion_credit_gate * r_preinsert_aligned_axial
    gated_positive_corridor = insertion_credit_gate * positive_corridor
    gated_success_candidate = insertion_credit_gate * success_candidate * g_insert_combined
    additive_total = (
        float(lateral_alignment_action_weight) * off_axis_alignment_phase * r_lateral_alignment_action
        + float(axial_progress_weight) * gated_positive_axial
        - float(off_axis_axial_action_penalty_weight) * r_off_axis_axial_action_penalty
        - float(lateral_error_state_penalty_weight) * r_lateral_error_state_penalty
        + float(lateral_progress_weight) * phase_lateral_progress
        + float(orientation_progress_weight) * (positive_orientation_progress * g_lat_pre + negative_orientation_progress)
        + float(lateral_funnel_weight) * r_lateral_funnel
        + float(near_misaligned_weight) * r_near_misaligned
        + float(hover_weight) * r_preinsert_hover
        - float(axial_progress_weight) * (1.0 - lat_gate_phase) * bad_forward_progress
        + float(axial_progress_weight) * retreat_penalty
        + float(preinsert_aligned_axial_weight) * gated_preinsert_aligned_axial
        + float(corridor_weight) * (gated_positive_corridor + negative_corridor)
        + float(inside_alignment_weight) * r_inside_alignment
        + float(retreat_weight) * r_retreat
        + float(semantic_progress_weight) * torch.relu(r_semantic_progress)
        + float(semantic_loss_weight) * torch.minimum(r_semantic_progress, torch.zeros_like(r_semantic_progress))
        + float(module_depth_progress_weight) * torch.relu(r_module_depth_progress)
        + float(module_depth_loss_weight) * torch.minimum(r_module_depth_progress, torch.zeros_like(r_module_depth_progress))
        + float(success_candidate_weight) * gated_success_candidate
    )
    if bool(stateful_phase_mode):
        lateral_state = (r > float(stateful_lateral_enter_threshold)).to(dtype=s.dtype)
        orientation_excess = torch.relu(
            (theta - float(stateful_orientation_enter_threshold))
            / max(float(orientation_progress_scale), 1.0e-9)
        ).clamp(max=1.0)
        orientation_state = (
            (1.0 - lateral_state)
            * (theta > float(stateful_orientation_enter_threshold)).to(dtype=s.dtype)
        )
        axial_state = 1.0 - torch.maximum(lateral_state, orientation_state)
        previous_lateral_state = (previous_lateral_error > float(stateful_lateral_enter_threshold)).to(dtype=s.dtype)
        previous_orientation_state = (
            (1.0 - previous_lateral_state)
            * (previous_orientation_error > float(stateful_orientation_enter_threshold)).to(dtype=s.dtype)
        )
        previous_axial_state = 1.0 - torch.maximum(previous_lateral_state, previous_orientation_state)
        lateral_total = (
            float(lateral_progress_weight) * r_lateral_progress
            + float(lateral_alignment_action_weight) * r_lateral_alignment_action
            - float(axial_progress_weight) * torch.relu(forward)
            - float(stateful_axial_forward_action_penalty_weight) * torch.relu(forward)
            - float(off_axis_axial_action_penalty_weight) * torch.relu(forward)
            - float(off_axis_axial_action_penalty_weight) * off_axis_inserted_depth_penalty
            - float(off_axis_axial_action_penalty_weight) * r_any_axis_action_penalty
            - float(stateful_axial_rotation_action_penalty_weight) * r_axial_phase_rotation_action_penalty
            - float(lateral_error_state_penalty_weight)
            * (r_lateral_error_state_penalty + torch.ones_like(r_lateral_error_state_penalty))
        )
        orientation_total = (
            float(orientation_progress_weight) * r_orientation_progress
            - float(orientation_progress_weight)
            * orientation_excess
            * (1.0 - torch.relu(r_orientation_progress).clamp(max=1.0))
            + float(lateral_progress_weight) * torch.minimum(r_lateral_progress, torch.zeros_like(r_lateral_progress))
            - float(axial_progress_weight) * torch.relu(forward)
            - float(stateful_axial_forward_action_penalty_weight) * torch.relu(forward)
            - float(off_axis_axial_action_penalty_weight) * torch.relu(forward)
            - float(off_axis_axial_action_penalty_weight) * off_axis_inserted_depth_penalty
            - float(off_axis_axial_action_penalty_weight) * r_any_axis_action_penalty
            - float(stateful_axial_lateral_action_penalty_weight) * r_axial_phase_lateral_action_penalty
            - float(stateful_axial_alignment_loss_penalty_weight) * r_axial_phase_alignment_loss_penalty
            - float(lateral_error_state_penalty_weight) * r_lateral_error_state_penalty
        )
        axial_pure_progress = torch.relu(forward) * r_axial_phase_pure_action
        axial_pure_corridor = positive_corridor * r_axial_phase_pure_action
        axial_pure_preinsert = r_preinsert_aligned_axial * r_axial_phase_pure_action
        axial_total = (
            float(axial_progress_weight) * axial_pure_progress
            + float(stateful_axial_pure_action_weight) * r_axial_phase_pure_action
            + float(preinsert_aligned_axial_weight) * axial_pure_preinsert
            + float(corridor_weight) * axial_pure_corridor
            + float(success_candidate_weight) * success_candidate * g_insert_combined * axial_phase_alignment_nonworsening_gate
            + float(retreat_weight) * retreat_penalty
            + float(lateral_progress_weight) * torch.minimum(r_lateral_progress, torch.zeros_like(r_lateral_progress))
            + float(orientation_progress_weight)
            * torch.minimum(r_orientation_progress, torch.zeros_like(r_orientation_progress))
            + float(inside_alignment_weight) * r_inside_alignment
            - float(off_axis_axial_action_penalty_weight) * bad_forward_progress
            - float(stateful_axial_lateral_action_penalty_weight) * r_axial_phase_lateral_action_penalty
            - float(stateful_axial_rotation_action_penalty_weight) * r_axial_phase_rotation_action_penalty
            - float(stateful_axial_forward_action_penalty_weight) * r_axial_phase_forward_action_penalty
            - float(stateful_axial_alignment_loss_penalty_weight) * r_axial_phase_alignment_loss_penalty
            - float(stateful_axial_impure_action_penalty_weight) * r_axial_phase_impure_action_penalty
        )
        total = (
            lateral_state * lateral_total
            + orientation_state * orientation_total
            + axial_state * axial_total
            - previous_axial_state
            * float(stateful_axial_alignment_loss_penalty_weight)
            * r_axial_phase_alignment_loss_penalty
        )
    else:
        total = additive_total
    return CheatcodeInsertionRewardComponents(
        total=total,
        lateral_progress=r_lateral_progress,
        orientation_progress=r_orientation_progress,
        lat_gate_phase=lat_gate_phase,
        lateral_alignment_action=r_lateral_alignment_action,
        lateral_alignment_axial_quiet_gate=r_lateral_alignment_axial_quiet_gate,
        off_axis_axial_action_penalty=r_off_axis_axial_action_penalty,
        lateral_error_state_penalty=r_lateral_error_state_penalty,
        action_toward_axis=action_toward_axis,
        lateral_funnel=r_lateral_funnel,
        near_misaligned=r_near_misaligned,
        preinsert_hover=r_preinsert_hover,
        axial_progress=r_axial,
        preinsert_aligned_axial=r_preinsert_aligned_axial,
        corridor=r_corridor,
        inside_alignment=r_inside_alignment,
        retreat=r_retreat,
        semantic_progress=r_semantic_progress,
        module_depth_progress=r_module_depth_progress,
        success_candidate=success_candidate,
        g_lat_pre=g_lat_pre,
        g_ori_pre=g_ori_pre,
        g_align_pre=g_align_pre,
        g_lat_insert=g_lat_insert,
        g_ori_insert=g_ori_insert,
        g_align_insert=g_align_insert,
        g_semantic=g_semantic,
        g_action_axis=g_action_axis,
        g_insert_combined=g_insert_combined,
        action_axial=action_axial,
        action_lateral=action_lateral,
        action_rotation=action_rotation,
        axial_forward_action_penalty=r_axial_phase_forward_action_penalty,
        action_lateral_sigma=action_sigma,
        action_forward_gate=action_forward_gate,
        near_gate=near_gate,
        inside_gate=inside_gate,
        sigma_lat_pre=effective_sigma_lat_pre,
        sigma_lat_insert=effective_sigma_lat_insert,
        sigma_theta_pre=effective_sigma_theta_pre,
        sigma_theta_insert=effective_sigma_theta_insert,
    )
