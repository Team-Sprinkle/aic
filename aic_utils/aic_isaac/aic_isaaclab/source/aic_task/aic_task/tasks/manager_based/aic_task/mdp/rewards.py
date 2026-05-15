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

from .force_penalty import force_delta_penalty_curve

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _contact_sensor_force(
    env: ManagerBasedRLEnv,
    preferred_body_names: list[str],
) -> torch.Tensor | None:
    sensors = getattr(env.scene, "sensors", {})
    rows = []
    for sensor_name in ("contact_forces",):
        sensor = sensors.get(sensor_name)
        if sensor is None:
            continue
        net = getattr(sensor.data, "net_forces_w", None)
        if net is None:
            continue
        sensor_body_names = list(getattr(sensor, "body_names", []) or [])
        body_ids = [idx for idx, name in enumerate(sensor_body_names) if name in preferred_body_names]
        if not body_ids:
            continue
        rows.append(net[:, body_ids, :3].sum(dim=1))
    if rows:
        return torch.stack(rows, dim=0).sum(dim=0)
    return None


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
        target = ((episode.get("scene") or {}).get("target") or {})
        pose = target.get("target_pose_world") or {}
        position = pose.get("position")
        if position is None:
            return None
        position_tensor = torch.tensor(position, dtype=origins.dtype, device=origins.device)
        entrance = (target.get("entrance_pose_world") or {}).get("position")
        axis = target.get("insertion_axis_world")
        if entrance is not None and axis is not None:
            entrance_tensor = torch.tensor(entrance, dtype=origins.dtype, device=origins.device)
            axis_tensor = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
            axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor).clamp(min=1.0e-9)
            seated_depth = torch.sum((position_tensor - entrance_tensor) * axis_tensor).clamp(min=0.0)
            position_tensor = entrance_tensor + seated_depth * axis_tensor
        rows.append(position_tensor + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_insertion_axis_w(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    episode_by_env = getattr(env, "_aic_current_episode_by_env", None)
    if not episode_by_env:
        return None
    origins = env.scene.env_origins
    rows: list[torch.Tensor] = []
    for env_id in range(env.num_envs):
        episode = episode_by_env.get(env_id)
        if not episode:
            return None
        target = ((episode.get("scene") or {}).get("target") or {})
        axis = target.get("insertion_axis_world")
        if axis is None:
            return None
        axis_tensor = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
        rows.append(axis_tensor / torch.linalg.norm(axis_tensor).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _episode_target_orientation_w(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    episode_by_env = getattr(env, "_aic_current_episode_by_env", None)
    if not episode_by_env:
        return None
    rows: list[torch.Tensor] = []
    device = env.scene.env_origins.device
    dtype = env.scene.env_origins.dtype
    for env_id in range(env.num_envs):
        episode = episode_by_env.get(env_id)
        if not episode:
            return None
        target = ((episode.get("scene") or {}).get("target") or {}).get("target_pose_world") or {}
        orientation = target.get("orientation_wxyz")
        if orientation is None:
            return None
        rows.append(torch.tensor(orientation, dtype=dtype, device=device))
    out = torch.stack(rows, dim=0)
    return out / torch.linalg.norm(out, dim=1, keepdim=True).clamp(min=1.0e-9)


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


def _target_orientation_w(
    env: ManagerBasedRLEnv,
    target_asset: RigidObject,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None,
) -> torch.Tensor:
    episode_orientations = _episode_target_orientation_w(env)
    if episode_orientations is not None:
        return episode_orientations.to(device=target_asset.data.root_quat_w.device, dtype=target_asset.data.root_quat_w.dtype)
    return _offset_quat_w(target_asset.data.root_quat_w, target_orientation_offset)


def _quat_conjugate_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([quat[:, 0:1], -quat[:, 1:4]], dim=1)


def _target_frame_delta(
    env: ManagerBasedRLEnv,
    body_asset: RigidObject,
    body_id: int,
    target_asset: RigidObject,
    *,
    target_position_offset: tuple[float, float, float] | list[float] | None,
    body_position_offset: tuple[float, float, float] | list[float] | None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None,
) -> torch.Tensor:
    body_pos_w = _body_position_w(body_asset, body_id, body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    target_quat_w = _target_orientation_w(env, target_asset, target_orientation_offset)
    return quat_apply(_quat_conjugate_wxyz(target_quat_w), body_pos_w - target_pos_w)


def _lateral_error_from_delta(delta_target: torch.Tensor, insertion_axis: int) -> torch.Tensor:
    axis = int(insertion_axis)
    if axis < 0 or axis > 2:
        raise ValueError(f"insertion_axis must be 0, 1, or 2, got {insertion_axis}")
    mask = torch.ones(3, dtype=torch.bool, device=delta_target.device)
    mask[axis] = False
    return torch.norm(delta_target[:, mask], dim=1)


def _axis_lateral_error_w(body_pos_w: torch.Tensor, target_pos_w: torch.Tensor, axis_w: torch.Tensor) -> torch.Tensor:
    delta = body_pos_w - target_pos_w
    axial = torch.sum(delta * axis_w, dim=1, keepdim=True) * axis_w
    lateral = delta - axial
    return torch.norm(lateral, dim=1)


def _semantic_axial_lateral_errors(
    env: ManagerBasedRLEnv,
    body_asset: RigidObject,
    body_id: int,
    target_asset: RigidObject,
    *,
    insertion_axis: int,
    target_position_offset: tuple[float, float, float] | list[float] | None,
    body_position_offset: tuple[float, float, float] | list[float] | None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return absolute axial and lateral error using episode insertion-axis metadata when present."""
    axis_w = _episode_insertion_axis_w(env)
    if axis_w is not None:
        body_pos_w = _body_position_w(body_asset, body_id, body_position_offset)
        target_pos_w = _target_position_w(env, target_asset, target_position_offset)
        axis_w = axis_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        delta_w = body_pos_w - target_pos_w
        axial_error = torch.abs(torch.sum(delta_w * axis_w, dim=1))
        lateral_error = _axis_lateral_error_w(body_pos_w, target_pos_w, axis_w)
        return axial_error, lateral_error

    delta_target = _target_frame_delta(
        env,
        body_asset,
        body_id,
        target_asset,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    axis = int(insertion_axis)
    if axis < 0 or axis > 2:
        raise ValueError(f"insertion_axis must be 0, 1, or 2, got {insertion_axis}")
    return torch.abs(delta_target[:, axis]), _lateral_error_from_delta(delta_target, axis)


def _semantic_lateral_error(
    env: ManagerBasedRLEnv,
    body_asset: RigidObject,
    body_id: int,
    target_asset: RigidObject,
    *,
    insertion_axis: int,
    target_position_offset: tuple[float, float, float] | list[float] | None,
    body_position_offset: tuple[float, float, float] | list[float] | None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None,
) -> torch.Tensor:
    axis_w = _episode_insertion_axis_w(env)
    if axis_w is not None:
        body_pos_w = _body_position_w(body_asset, body_id, body_position_offset)
        target_pos_w = _target_position_w(env, target_asset, target_position_offset)
        return _axis_lateral_error_w(body_pos_w, target_pos_w, axis_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype))
    delta_target = _target_frame_delta(
        env,
        body_asset,
        body_id,
        target_asset,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return _lateral_error_from_delta(delta_target, insertion_axis)


def _episode_entrance_position_axis_w(
    env: ManagerBasedRLEnv,
    *,
    target_pos_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    episode_by_env = getattr(env, "_aic_current_episode_by_env", None)
    if not episode_by_env:
        return None
    origins = env.scene.env_origins
    entrance_rows: list[torch.Tensor] = []
    axis_rows: list[torch.Tensor] = []
    for env_id in range(env.num_envs):
        target = (((episode_by_env.get(env_id) or {}).get("scene") or {}).get("target") or {})
        entrance = (target.get("entrance_pose_world") or {}).get("position")
        axis = target.get("insertion_axis_world")
        if entrance is None or axis is None:
            return None
        entrance_tensor = torch.tensor(entrance, dtype=target_pos_w.dtype, device=target_pos_w.device)
        axis_tensor = torch.tensor(axis, dtype=target_pos_w.dtype, device=target_pos_w.device)
        axis_tensor = axis_tensor / torch.linalg.norm(axis_tensor).clamp(min=1.0e-9)
        entrance_rows.append(entrance_tensor + origins[env_id].to(device=target_pos_w.device, dtype=target_pos_w.dtype))
        axis_rows.append(axis_tensor)
    return torch.stack(entrance_rows, dim=0), torch.stack(axis_rows, dim=0)


def _reset_previous_on_episode_start(
    previous: torch.Tensor,
    current: torch.Tensor,
    reset_mask: torch.Tensor | None,
) -> torch.Tensor:
    previous = previous.to(current.device)
    if reset_mask is None:
        return previous
    mask = reset_mask.to(current.device)
    while mask.ndim < current.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask <= 1, current.detach(), previous)


def _positive_reward_lateral_gate(
    reward: torch.Tensor,
    lateral_error: torch.Tensor,
    lateral_gate_sigma: float | None,
) -> torch.Tensor:
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    gate = torch.exp(-torch.square(lateral_error / max(float(lateral_gate_sigma), 1.0e-9)))
    return torch.where(reward > 0.0, reward * gate, reward)


def _insertion_progress_lateral_gate(
    reward: torch.Tensor,
    lateral_error: torch.Tensor,
    lateral_gate_sigma: float | None,
) -> torch.Tensor:
    """Gate insertion-direction progress and punish positive progress while off-center.

    A pure positive gate still lets policies collect small-but-positive reward by
    moving along the insertion axis beside the port. For insertion-specific
    progress terms, positive progress should only stay positive when the lateral
    error is already small. When the tip is laterally misaligned, the same
    insertion-direction motion is treated as a bad move because it bypasses the
    entrance rather than entering it.
    """
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    gate = torch.exp(-torch.square(lateral_error / max(float(lateral_gate_sigma), 1.0e-9)))
    signed_gate = 2.0 * gate - 1.0
    return torch.where(reward > 0.0, reward * signed_gate, reward)


def body_to_object_distance_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
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
    reward = 1.0 - torch.tanh(distance / std)
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return _positive_reward_lateral_gate(reward, lateral_error, lateral_gate_sigma)


def body_to_object_distance_exp(
    env: ManagerBasedRLEnv,
    sigma: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward precise body-to-target proximity using an exponential kernel."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    dist_sq = torch.sum(torch.square(body_pos_w - target_pos_w), dim=1)
    reward = torch.exp(-dist_sq / (sigma**2))
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return _positive_reward_lateral_gate(reward, lateral_error, lateral_gate_sigma)


def body_to_object_distance_progress(
    env: ManagerBasedRLEnv,
    scale: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward one-step progress toward the semantic target point.

    This mirrors the offline dense reward term:
    ``clip((previous_distance - current_distance) / scale, -1, 1)``.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    distance = torch.norm(body_pos_w - target_pos_w, dim=1)
    previous = getattr(env, "_aic_previous_target_distance", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if previous is None or previous.shape != distance.shape:
        previous = distance.detach().clone()
    else:
        previous = previous.to(distance.device)
        if reset_mask is not None:
            previous = torch.where(reset_mask.to(distance.device) <= 1, distance.detach(), previous)
    progress = ((previous - distance) / max(float(scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    setattr(env, "_aic_previous_target_distance", distance.detach().clone())
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return progress
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return _insertion_progress_lateral_gate(progress, lateral_error, lateral_gate_sigma)


def body_to_object_motion_projection(
    env: ManagerBasedRLEnv,
    scale: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward actual tip motion whose realized displacement points toward the target.

    This uses the measured body pose after physics, not the commanded action. It is
    therefore useful for catching frame/sign mistakes and controller realization
    issues: if the policy command moves the tip away from the gate, the term is
    negative even if the action vector itself looked reasonable.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    current = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target = _target_position_w(env, target_asset, target_position_offset)
    previous = getattr(env, "_aic_previous_motion_projection_body_pos_w", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if previous is None or previous.shape != current.shape:
        previous = current.detach().clone()
    else:
        previous = _reset_previous_on_episode_start(previous, current, reset_mask)
    motion = current - previous
    direction = target - previous
    direction = direction / torch.linalg.norm(direction, dim=1, keepdim=True).clamp(min=1.0e-9)
    projection = torch.sum(motion * direction, dim=1)
    reward = (projection / max(float(scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    setattr(env, "_aic_previous_motion_projection_body_pos_w", current.detach().clone())
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return _insertion_progress_lateral_gate(reward, lateral_error, lateral_gate_sigma)


def body_to_object_lateral_progress(
    env: ManagerBasedRLEnv,
    scale: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward reduction of lateral error in the target/port frame."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    previous = getattr(env, "_aic_previous_target_lateral_error", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if previous is None or previous.shape != lateral_error.shape:
        previous = lateral_error.detach().clone()
    else:
        previous = _reset_previous_on_episode_start(previous, lateral_error, reset_mask)
    reward = ((previous - lateral_error) / max(float(scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    setattr(env, "_aic_previous_target_lateral_error", lateral_error.detach().clone())
    return reward


def body_to_object_axial_progress(
    env: ManagerBasedRLEnv,
    scale: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward reducing signed gap along the insertion axis in the target frame.

    Euclidean distance can improve when the tip moves diagonally toward the side
    of the card. This term is stricter: it pays only when the absolute axial
    gap to the target insertion point shrinks, and positive rewards are laterally
    gated so off-center downward motion is not treated as good insertion.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    axis_w = _episode_insertion_axis_w(env)
    if axis_w is not None:
        body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
        target_pos_w = _target_position_w(env, target_asset, target_position_offset)
        delta_w = body_pos_w - target_pos_w
        axis_w = axis_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        axial_error = torch.abs(torch.sum(delta_w * axis_w, dim=1))
        lateral_error = _axis_lateral_error_w(body_pos_w, target_pos_w, axis_w)
    else:
        delta_target = _target_frame_delta(
            env,
            body_asset,
            body_cfg.body_ids[0],
            target_asset,
            target_position_offset=target_position_offset,
            body_position_offset=body_position_offset,
            target_orientation_offset=target_orientation_offset,
        )
        axis = int(insertion_axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"insertion_axis must be 0, 1, or 2, got {insertion_axis}")
        axial_error = torch.abs(delta_target[:, axis])
        lateral_error = _lateral_error_from_delta(delta_target, axis)
    previous = getattr(env, "_aic_previous_target_axial_error", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if previous is None or previous.shape != axial_error.shape:
        previous = axial_error.detach().clone()
    else:
        previous = _reset_previous_on_episode_start(previous, axial_error, reset_mask)
    reward = ((previous - axial_error) / max(float(scale), 1.0e-9)).clamp(min=-1.0, max=1.0)
    setattr(env, "_aic_previous_target_axial_error", axial_error.detach().clone())
    if lateral_gate_sigma is None or float(lateral_gate_sigma) <= 0.0:
        return reward
    return _insertion_progress_lateral_gate(reward, lateral_error, lateral_gate_sigma)


def body_to_object_insertion_corridor(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    lateral_gate_sigma: float = 0.0025,
    bypass_penalty_scale: float = 1.0,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward being seated along the entrance corridor and punish off-axis bypass.

    Depth past the entrance only counts as good insertion when the tip is close to
    the port centerline. Moving down beside the card can reduce Euclidean
    distance, but here it produces a negative bypass term because the lateral
    gate is near zero while signed depth is positive.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    entrance_axis = _episode_entrance_position_axis_w(env, target_pos_w=target_pos_w)
    if entrance_axis is not None:
        entrance_w, axis_w = entrance_axis
        axis_w = axis_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        entrance_w = entrance_w.to(device=body_pos_w.device, dtype=body_pos_w.dtype)
        depth = torch.sum((body_pos_w - entrance_w) * axis_w, dim=1)
        target_depth = torch.sum((target_pos_w - entrance_w) * axis_w, dim=1).clamp(min=1.0e-6)
        lateral_error = _axis_lateral_error_w(body_pos_w, entrance_w, axis_w)
    else:
        delta_target = _target_frame_delta(
            env,
            body_asset,
            body_cfg.body_ids[0],
            target_asset,
            target_position_offset=target_position_offset,
            body_position_offset=body_position_offset,
            target_orientation_offset=target_orientation_offset,
        )
        axis = int(insertion_axis)
        if axis < 0 or axis > 2:
            raise ValueError(f"insertion_axis must be 0, 1, or 2, got {insertion_axis}")
        depth = -delta_target[:, axis]
        target_depth = torch.full_like(depth, 0.01)
        lateral_error = _lateral_error_from_delta(delta_target, axis)

    depth_fraction = (depth / target_depth).clamp(min=0.0, max=1.0)
    gate = torch.exp(-torch.square(lateral_error / max(float(lateral_gate_sigma), 1.0e-9)))
    centered_depth_reward = depth_fraction * gate
    bypass_penalty = depth_fraction * (1.0 - gate) * max(float(bypass_penalty_scale), 0.0)
    return centered_depth_reward - bypass_penalty


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
    target_quat_w = _target_orientation_w(env, target_asset, target_orientation_offset)
    ang_error = quat_error_magnitude(body_quat_w, target_quat_w)
    return 1.0 - torch.tanh(ang_error / std)


def body_to_object_orientation_gated_exp(
    env: ManagerBasedRLEnv,
    std: float,
    gate_sigma: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    body_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Reward orientation alignment only near the target, matching offline rewards."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_quat_w = _offset_quat_w(
        body_asset.data.body_quat_w[:, body_cfg.body_ids[0]],  # type: ignore
        body_orientation_offset,
    )
    target_quat_w = _target_orientation_w(env, target_asset, target_orientation_offset)
    ang_error = quat_error_magnitude(body_quat_w, target_quat_w)
    orientation_alignment = torch.exp(-torch.square(ang_error / max(float(std), 1.0e-9)))
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    distance = torch.norm(body_pos_w - target_pos_w, dim=1)
    orientation_gate = torch.exp(-torch.square(distance / max(float(gate_sigma), 1.0e-9)))
    return orientation_alignment * orientation_gate


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


def body_to_object_success(
    env: ManagerBasedRLEnv,
    threshold: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    axial_threshold: float | None = None,
    lateral_threshold: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Terminate when the selected body is centered and seated at the semantic target."""
    if float(threshold) <= 0.0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    distance_success = torch.norm(body_pos_w - target_pos_w, dim=1) <= float(threshold)
    if axial_threshold is None and lateral_threshold is None:
        return distance_success

    axial_error, lateral_error = _semantic_axial_lateral_errors(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    axial_limit = float(threshold if axial_threshold is None else axial_threshold)
    lateral_limit = float(threshold if lateral_threshold is None else lateral_threshold)
    return torch.logical_and(axial_error <= axial_limit, lateral_error <= lateral_limit)


def body_to_object_success_once_bonus(
    env: ManagerBasedRLEnv,
    threshold: float,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    axial_threshold: float | None = None,
    lateral_threshold: float | None = None,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Emit a sparse +1 once when an episode first satisfies strict insertion geometry."""
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    body_pos_w = _body_position_w(body_asset, body_cfg.body_ids[0], body_position_offset)
    target_pos_w = _target_position_w(env, target_asset, target_position_offset)
    success = torch.norm(body_pos_w - target_pos_w, dim=1) <= float(threshold)
    if axial_threshold is not None or lateral_threshold is not None:
        axial_error, lateral_error = _semantic_axial_lateral_errors(
            env,
            body_asset,
            body_cfg.body_ids[0],
            target_asset,
            insertion_axis=insertion_axis,
            target_position_offset=target_position_offset,
            body_position_offset=body_position_offset,
            target_orientation_offset=target_orientation_offset,
        )
        axial_limit = float(threshold if axial_threshold is None else axial_threshold)
        lateral_limit = float(threshold if lateral_threshold is None else lateral_threshold)
        success = torch.logical_and(axial_error <= axial_limit, lateral_error <= lateral_limit)
    achieved = getattr(env, "_aic_success_bonus_emitted", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if achieved is None or achieved.shape != success.shape:
        achieved = torch.zeros_like(success)
    else:
        achieved = achieved.to(success.device)
        if reset_mask is not None:
            achieved = torch.where(reset_mask.to(success.device) <= 1, torch.zeros_like(achieved), achieved)
    bonus = torch.logical_and(success, torch.logical_not(achieved)).float()
    setattr(env, "_aic_success_bonus_emitted", torch.logical_or(achieved, success).detach().clone())
    return bonus


def body_to_object_lateral_error(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    insertion_axis: int = 0,
    scale: float = 0.006,
    target_position_offset: tuple[float, float, float] | list[float] | None = None,
    body_position_offset: tuple[float, float, float] | list[float] | None = None,
    target_orientation_offset: tuple[float, float, float, float] | list[float] | None = None,
) -> torch.Tensor:
    """Normalized lateral error in the target/port frame.

    This is intentionally target-frame based: moving diagonally toward the side
    of a card should not look like good insertion progress simply because the
    Euclidean distance to the entrance point got smaller.
    """
    body_asset: RigidObject = env.scene[body_cfg.name]
    target_asset: RigidObject = env.scene[target_cfg.name]
    lateral_error = _semantic_lateral_error(
        env,
        body_asset,
        body_cfg.body_ids[0],
        target_asset,
        insertion_axis=insertion_axis,
        target_position_offset=target_position_offset,
        body_position_offset=body_position_offset,
        target_orientation_offset=target_orientation_offset,
    )
    return torch.tanh(lateral_error / max(float(scale), 1.0e-9))


def force_delta_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 10.0,
    reference: float = 20.0,
    knee_penalty_fraction: float = 0.1,
    saturation: float | None = None,
    max_penalty: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize sudden changes in incoming body wrench.

    The returned value is non-positive. Use a positive reward weight to apply
    this as a penalty, matching the offline dense reward convention.

    The curve is intentionally gentle below ``reference`` and steep above it:
    no penalty below ``threshold``, a quadratic ramp up to
    ``knee_penalty_fraction * max_penalty`` at ``reference``, then a smoothstep
    ramp to ``max_penalty`` at ``saturation``. This allows normal insertion
    contact while strongly discouraging hard impacts.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    wrench = getattr(asset.data, "body_incoming_wrench_w", None)
    if wrench is None:
        wrench = getattr(asset.data, "body_incoming_wrench_b", None)
    if wrench is None:
        wrench = getattr(asset.data, "body_incoming_joint_wrench_b", None)
    if wrench is None:
        requested_names = getattr(asset_cfg, "body_names", None)
        preferred = [requested_names] if isinstance(requested_names, str) else list(requested_names or [])
        # gripper_tcp/sfp_tip_link can be fixed frames without contact reports; these are the closest
        # physical contact-reporting proxies available in the current Isaac asset.
        preferred.extend(["sfp_tip_link", "sfp_module_link", "gripper_tcp", "wrist_3_link"])
        contact_force = _contact_sensor_force(env, preferred)
        if contact_force is None:
            current = torch.zeros(env.num_envs, 3, device=env.device)
        else:
            current = contact_force
    else:
        body_ids = asset_cfg.body_ids
        selected = wrench if body_ids is None or body_ids == slice(None) else wrench[:, body_ids, :]
        current = selected[..., :3].sum(dim=1)
    previous = getattr(env, "_aic_previous_force_w", None)
    reset_mask = getattr(env, "episode_length_buf", None)
    if previous is None or previous.shape != current.shape:
        previous = current.detach().clone()
    elif reset_mask is not None:
        previous = previous.to(current.device)
        previous = torch.where(reset_mask.to(current.device).reshape(-1, 1) <= 1, current.detach(), previous)
    delta_norm = torch.norm(current - previous.to(current.device), dim=1)
    setattr(env, "_aic_previous_force_w", current.detach().clone())
    return -force_delta_penalty_curve(
        delta_norm,
        threshold=float(threshold),
        reference=float(reference),
        knee_penalty_fraction=float(knee_penalty_fraction),
        saturation=saturation,
        max_penalty=float(max_penalty),
    )


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
