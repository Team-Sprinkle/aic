# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the aic task (UR5e assembly with task board).

Includes:
- Command-tracking rewards with exponential / tanh kernels (inspired by the
  gear-assembly deploy environment).
- A sparse reaching bonus.
- Smoothness and safety penalties (torques, joint acceleration, action rate).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Command-pose tracking (position)
# ---------------------------------------------------------------------------


def position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def position_command_error_exp(
    env: ManagerBasedRLEnv, sigma: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward position tracking using a Gaussian (exponential) kernel.

    Unlike tanh, this kernel drops off very steeply beyond *sigma*, providing
    almost no gradient far from the target while giving a strong signal
    close-in — ideal for fine insertion tasks.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    dist_sq = torch.sum(torch.square(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


# ---------------------------------------------------------------------------
# Command-pose tracking (orientation)
# ---------------------------------------------------------------------------


def orientation_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize orientation error (shortest-path angular distance in rad)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward orientation tracking using the tanh kernel.

    Maps the angular error through ``1 - tanh(error / std)`` so that perfectly
    aligned orientations yield 1.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    ang_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


# ---------------------------------------------------------------------------
# Sparse reaching bonus
# ---------------------------------------------------------------------------


def ee_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Sparse +1 bonus when the EE is within *threshold* (m) of the command position."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b
    )
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return (distance < threshold).float()


# ---------------------------------------------------------------------------
# Optional insertion-aware shaping
# ---------------------------------------------------------------------------


def _target_position_w(
    env: ManagerBasedRLEnv,
    target_asset: RigidObject,
    target_position_offset: tuple[float, float, float] | list[float] | None,
) -> torch.Tensor:
    episode_positions = _episode_target_position_w(env)
    if episode_positions is not None:
        return episode_positions.to(device=target_asset.data.root_pos_w.device, dtype=target_asset.data.root_pos_w.dtype)
    if target_position_offset is None:
        return target_asset.data.root_pos_w
    offset = torch.tensor(
        target_position_offset,
        dtype=target_asset.data.root_pos_w.dtype,
        device=target_asset.data.root_pos_w.device,
    ).reshape(1, 3)
    return target_asset.data.root_pos_w + quat_apply(target_asset.data.root_quat_w, offset.expand_as(target_asset.data.root_pos_w))


def _episode_target_position_w(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    episode_by_env = getattr(env, "_aic_current_episode_by_env", None)
    if not episode_by_env:
        return None
    origins = env.scene.env_origins
    rows: list[torch.Tensor] = []
    for env_id in range(env.num_envs):
        episode = episode_by_env.get(env_id)
        if not episode:
            return None
        target = ((episode.get("scene") or {}).get("target") or {}).get("target_pose_world") or {}
        position = target.get("position")
        if position is None:
            return None
        rows.append(torch.tensor(position, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _body_position_w(
    body_asset: RigidObject,
    body_id: int,
    body_position_offset: tuple[float, float, float] | list[float] | None,
) -> torch.Tensor:
    body_pos_w = body_asset.data.body_pos_w[:, body_id]  # type: ignore
    if body_position_offset is None:
        return body_pos_w
    offset = torch.tensor(
        body_position_offset,
        dtype=body_pos_w.dtype,
        device=body_pos_w.device,
    ).reshape(1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_id]  # type: ignore
    return body_pos_w + quat_apply(body_quat_w, offset.expand_as(body_pos_w))


def _offset_quat_w(
    base_quat_w: torch.Tensor,
    orientation_offset: tuple[float, float, float, float] | list[float] | None,
) -> torch.Tensor:
    if orientation_offset is None:
        return base_quat_w
    offset = torch.tensor(
        orientation_offset,
        dtype=base_quat_w.dtype,
        device=base_quat_w.device,
    ).reshape(1, 4)
    return quat_mul(base_quat_w, offset.expand_as(base_quat_w))


def body_to_object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward a body approaching a target object's root pose.

    This is intentionally generic: it can be pointed at an SC port, NIC card,
    fixture, or future insertion target without changing the runtime policy
    interface. It is disabled by default in the env config because current
    assets do not expose a canonical cable-tip or port-insertion frame.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    distance = torch.norm(body_pos_w - target_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def body_to_object_distance_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward precise body-to-target proximity using an exponential kernel."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    dist_sq = torch.sum(torch.square(body_pos_w - target_pos_w), dim=1)
    return torch.exp(-dist_sq / (sigma**2))


def body_to_object_orientation_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    body_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward semantic body-frame alignment to a semantic target frame."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_quat_w = _offset_quat_w(
        body_asset.data.body_quat_w[:, body_cfg.body_ids[0]],  # type: ignore
        body_orientation_offset,
    )
    target_quat_w = _offset_quat_w(target_asset.data.root_quat_w, target_orientation_offset)
    ang_error = quat_error_magnitude(body_quat_w, target_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


def body_to_object_reaching_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Sparse +1 when the selected body is within *threshold* of the target object."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    distance = torch.norm(body_pos_w - target_pos_w, dim=1)
    return (distance < threshold).float()


def body_to_object_lateral_error(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    axis: int = 0,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Penalize lateral error to a target object's root pose.

    ``axis`` is the assumed insertion axis in world coordinates. The current
    Isaac scene does not yet expose a semantic port insertion frame, so this is
    kept optional and low-level until those frames are added.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    delta = body_pos_w - _target_position_w(env, target_asset, target_position_offset)
    mask = torch.ones(3, dtype=torch.bool, device=delta.device)
    mask[axis] = False
    return torch.norm(delta[:, mask], dim=1)


def force_delta_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 3.0,
    reference: float = 20.0,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sudden changes in incoming body wrench.

    The returned value is non-positive. Use a positive reward weight to apply
    this as a penalty, matching the offline dense reward convention.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wrench = getattr(asset.data, "body_incoming_wrench_w", None)
    if wrench is None:
        wrench = getattr(asset.data, "body_incoming_wrench_b", None)
    if wrench is None:
        current = torch.zeros(env.num_envs, 3, device=env.device)
    else:
        body_ids = asset_cfg.body_ids
        selected = wrench if body_ids is None or body_ids == slice(None) else wrench[:, body_ids, :]
        current = selected[..., :3].sum(dim=1)
    previous = getattr(env, "_aic_previous_force_w", None)
    if previous is None or previous.shape != current.shape:
        previous = current.detach().clone()
    delta_norm = torch.norm(current - previous.to(current.device), dim=1)
    setattr(env, "_aic_previous_force_w", current.detach().clone())
    normalized = ((delta_norm - float(threshold)) / max(float(reference), 1e-6)).clamp(min=0.0, max=float(max_penalty))
    return -normalized


# ---------------------------------------------------------------------------
# Smoothness / safety penalties
# ---------------------------------------------------------------------------


def joint_torques_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize applied joint torques (L2 squared)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1
    )


def joint_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint accelerations (L2 squared) for smoother motion."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_pos_limits(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joints that exceed their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def body_lin_acc_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize linear acceleration of selected bodies (encourages gentle motion)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1
    )
