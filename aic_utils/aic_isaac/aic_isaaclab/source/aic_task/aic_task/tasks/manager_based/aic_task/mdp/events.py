from __future__ import annotations

import math
import os
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING

import omni.usd
import torch
import yaml
from pxr import Gf, Sdf, UsdGeom, UsdLux
from isaaclab.utils.math import compute_pose_error, quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Matches the regex form Isaac Lab uses to instantiate per-env prim paths.
_ENV_REGEX_RE = re.compile(r"env_(?:\.\*|\[\^/\]\*)")

# Orientations captured from PhysX on the first reset and reused on every
# subsequent reset. Holding the quaternion fixed keeps the composed child
# transforms from referenced USDs (e.g. port frames) correctly aligned.
_cached_orientations: dict[str, torch.Tensor] = {}
_episode_config_cache: dict[str, object] = {"root": None, "episodes": [], "cursor": 0}


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (1500.0, 3500.0),
    color_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.5, 0.5, 0.5),
        (1.0, 1.0, 1.0),
    ),
) -> None:
    """Randomize the dome light's intensity and color on reset.

    The light is a single shared prim, so the randomization is global across
    all environments regardless of ``env_ids``.
    """
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/light")
    if not light_prim.IsValid():
        return
    light = UsdLux.DomeLight(light_prim)

    intensity = torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item()
    light.GetIntensityAttr().Set(intensity)

    color_min, color_max = color_range
    r = torch.empty(1).uniform_(color_min[0], color_max[0]).item()
    g = torch.empty(1).uniform_(color_min[1], color_max[1]).item()
    b = torch.empty(1).uniform_(color_min[2], color_max[2]).item()
    light.GetColorAttr().Set(Gf.Vec3f(r, g, b))


def _sample_axis(pose_range: dict, snap_step: dict, axis: str) -> float:
    """Sample an axis offset, snapping to a grid step when configured."""
    lo, hi = pose_range.get(axis, (0.0, 0.0))
    step = snap_step.get(axis, 0.0)
    if step > 0 and (hi - lo) > 0:
        n_lo = math.ceil(lo / step)
        n_hi = math.floor(hi / step)
        return random.randint(n_lo, n_hi) * step
    return torch.empty(1).uniform_(lo, hi).item()


def _quat_mul_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Multiply quaternions stored as wxyz tensors."""
    w1, x1, y1, z1 = lhs.unbind(dim=-1)
    w2, x2, y2, z2 = rhs.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _yaw_offset_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Create a world-z yaw quaternion in wxyz order."""
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack(
        (torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)),
        dim=-1,
    )


def _apply_yaw_noise(base_rot: torch.Tensor, yaw_range: tuple[float, float]) -> torch.Tensor:
    lo, hi = yaw_range
    if lo == 0.0 and hi == 0.0:
        return base_rot
    yaw = torch.empty(base_rot.shape[0], device=base_rot.device).uniform_(lo, hi)
    return _quat_mul_wxyz(_yaw_offset_quat(yaw), base_rot)


def _range_tuple(value, default=(0.0, 0.0)) -> tuple[float, float]:
    if value is None:
        return (float(default[0]), float(default[1]))
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Expected fixed value or two-value range, got {value!r}")


def _load_episode_configs_from_env() -> list[dict]:
    root_raw = os.environ.get("AIC_ISAAC_EPISODE_CONFIG_DIR")
    if not root_raw:
        return []
    if _episode_config_cache["root"] == root_raw:
        return list(_episode_config_cache["episodes"])
    root = Path(root_raw)
    episodes_dir = root if root.name == "episodes" else root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"AIC_ISAAC_EPISODE_CONFIG_DIR has no episodes directory: {root}")
    episodes = []
    for path in sorted(episodes_dir.glob("episode_*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Isaac episode config must be a mapping: {path}")
        episodes.append(data)
    if not episodes:
        raise ValueError(f"No episode_*.yaml files found in {episodes_dir}")
    _episode_config_cache.update({"root": root_raw, "episodes": episodes, "cursor": 0})
    return episodes


def _episode_for_envs(env_ids: torch.Tensor) -> dict[int, dict]:
    episodes = _load_episode_configs_from_env()
    if not episodes:
        return {}
    cursor = int(_episode_config_cache.get("cursor", 0))
    by_env: dict[int, dict] = {}
    for env_id in env_ids.tolist():
        episode = episodes[cursor % len(episodes)]
        cursor += 1
        by_env[int(env_id)] = episode
    _episode_config_cache["cursor"] = cursor
    return by_env


def _episode_randomization_for_envs(env: ManagerBasedEnv, env_ids: torch.Tensor) -> dict[int, dict]:
    episode_by_env = _episode_for_envs(env_ids)
    if not episode_by_env:
        return {}
    current = dict(getattr(env, "_aic_current_episode_by_env", {}))
    randomization_by_env: dict[int, dict] = {}
    for env_id, episode in episode_by_env.items():
        randomization = episode.get("isaac_randomization") or {}
        if not isinstance(randomization, dict):
            raise ValueError(f"episode {episode.get('episode_id')} has invalid isaac_randomization")
        current[int(env_id)] = episode
        randomization_by_env[int(env_id)] = randomization
    setattr(env, "_aic_current_episode_by_env", current)
    return randomization_by_env


def _part_cfg_by_name(parts: list[dict]) -> dict[str, dict]:
    return {str(part["scene_name"]): part for part in parts}


def _write_usd_xform_pose(
    stage,
    prim_path_template: str,
    env_ids: torch.Tensor,
    env_origins: torch.Tensor,
    world_pos: torch.Tensor,
    world_rot: torch.Tensor,
) -> None:
    """Mirror a per-env rigid body pose onto its USD Xform.

    The prim translate is authored relative to its env root, so the world
    position is converted to env-local coordinates before writing.
    """
    ids = env_ids.tolist()
    local_pos = (world_pos - env_origins).tolist()
    rot = world_rot.tolist()

    for i, env_id in enumerate(ids):
        prim_path = _ENV_REGEX_RE.sub(f"env_{env_id}", prim_path_template)
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue

        xf = UsdGeom.Xformable(prim)
        tx, ty, tz = local_pos[i]
        qw, qx, qy, qz = rot[i]

        for op in xf.GetOrderedXformOps():
            name = op.GetOpName()
            if "translate" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Float3:
                    op.Set(Gf.Vec3f(tx, ty, tz))
                else:
                    op.Set(Gf.Vec3d(tx, ty, tz))
            elif "orient" in name:
                if op.GetTypeName() == Sdf.ValueTypeNames.Quatf:
                    op.Set(Gf.Quatf(qw, qx, qy, qz))
                else:
                    op.Set(Gf.Quatd(qw, qx, qy, qz))


def randomize_board_and_parts(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    board_scene_name: str = "task_board",
    board_default_pos: tuple = (0.0, 0.0, 0.0),
    board_range: dict = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
    parts: list[dict] = (),
    sync_usd_xforms: bool = True,
) -> None:
    """Randomize the task board and its attached parts on reset.

    The board position is drawn from ``board_range`` around ``board_default_pos``.
    ``board_range`` and per-part ``pose_range`` support x/y/z offsets and a
    world-z ``yaw`` offset in radians.
    Each part is offset from the board by a fixed ``offset`` plus a random
    delta from ``pose_range`` (optionally snapped to ``snap_step``).

    When ``sync_usd_xforms`` is True (default) the pose is mirrored onto the
    USD Xform so the viewport tracks physics state. Training workloads should
    set this False to skip the per-env USD writes.
    """
    order = list(getattr(env, "_aic_reset_event_order", []) or [])
    order.append(
        {
            "event": "randomize_board_and_parts",
            "env_ids": [int(v) for v in env_ids.detach().cpu().tolist()],
            "index": len(order),
        }
    )
    setattr(env, "_aic_reset_event_order", order[-64:])

    device = env.device
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    stage = omni.usd.get_context().get_stage() if sync_usd_xforms else None
    episode_by_env = _episode_randomization_for_envs(env, env_ids)

    all_names = [board_scene_name] + [p["scene_name"] for p in parts]
    if not _cached_orientations:
        for name in all_names:
            _cached_orientations[name] = (
                env.scene[name].data.root_state_w[:, 3:7].clone()
            )

    # Board pose.
    board_asset = env.scene[board_scene_name]
    board_rot = _cached_orientations[board_scene_name][env_ids].clone()
    board_pos = torch.empty(n, 3, device=device)
    for local_idx, env_id in enumerate(env_ids.tolist()):
        randomization = episode_by_env.get(int(env_id), {})
        base = tuple(float(v) for v in randomization.get("board_default_pos", board_default_pos))
        ranges = randomization.get("board_range", board_range)
        board_pos[local_idx] = torch.tensor(base, device=device)
        for axis_idx, axis in enumerate(("x", "y", "z")):
            lo, hi = _range_tuple(ranges.get(axis), (0.0, 0.0))
            board_pos[local_idx, axis_idx] += torch.empty(1, device=device).uniform_(lo, hi).item()
        yaw_range = _range_tuple(ranges.get("yaw"), (0.0, 0.0))
        board_rot[local_idx : local_idx + 1] = _apply_yaw_noise(
            board_rot[local_idx : local_idx + 1],
            yaw_range,
        )
    board_world_pos = board_pos + env_origins

    board_asset.write_root_pose_to_sim(
        torch.cat([board_world_pos, board_rot], dim=-1), env_ids=env_ids
    )
    board_asset.write_root_velocity_to_sim(
        torch.zeros(n, 6, device=device), env_ids=env_ids
    )
    if sync_usd_xforms:
        _write_usd_xform_pose(
            stage,
            board_asset.cfg.prim_path,
            env_ids,
            env_origins,
            board_world_pos,
            board_rot,
        )

    # Part poses, anchored to the board.
    default_parts_by_name = _part_cfg_by_name(list(parts))
    for part_cfg in parts:
        pname = part_cfg["scene_name"]
        part_asset = env.scene[pname]
        part_rot = _cached_orientations[pname][env_ids].clone()

        part_pos = board_world_pos.clone()
        for idx, env_id in enumerate(env_ids.tolist()):
            randomization = episode_by_env.get(int(env_id), {})
            parts_by_name = _part_cfg_by_name(randomization.get("parts", []))
            effective_part = parts_by_name.get(pname, default_parts_by_name[pname])
            pr = effective_part.get("pose_range", {})
            snap = effective_part.get("snap_step", {})
            ox, oy, oz = tuple(float(v) for v in effective_part.get("offset", part_cfg["offset"]))
            part_pos[idx, 0] += ox + _sample_axis(pr, snap, "x")
            part_pos[idx, 1] += oy + _sample_axis(pr, snap, "y")
            part_pos[idx, 2] = board_world_pos[idx, 2] + oz + _sample_axis(pr, snap, "z")
            part_rot[idx : idx + 1] = _apply_yaw_noise(
                part_rot[idx : idx + 1],
                _range_tuple(pr.get("yaw"), (0.0, 0.0)),
            )

        part_asset.write_root_pose_to_sim(
            torch.cat([part_pos, part_rot], dim=-1), env_ids=env_ids
        )
        part_asset.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=device), env_ids=env_ids
        )
        if sync_usd_xforms:
            _write_usd_xform_pose(
                stage,
                part_asset.cfg.prim_path,
                env_ids,
                env_origins,
                part_pos,
                part_rot,
            )


def reset_robot_tcp_to_episode_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    body_name: str = "gripper_tcp",
    joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ),
    max_iterations: int = 8,
    position_tolerance: float = 0.002,
    orientation_tolerance: float = 0.05,
    damping: float = 0.05,
    max_joint_delta: float = 0.25,
    sync_action_term_after_reset: bool = True,
) -> None:
    """Move robot reset body to child-YAML near-gate reset poses.

    New episode YAMLs store start_near_gate.body_start_position_world and
    reset_body_name so near-gate curricula can place either the reward body or
    a physically controlled proxy near the port entrance. Older YAMLs with
    tcp_start_position_world continue to work and use the function-level
    body_name default.
    """
    order = list(getattr(env, "_aic_reset_event_order", []) or [])
    order.append(
        {
            "event": "reset_robot_tcp_to_episode_start",
            "env_ids": [int(v) for v in env_ids.detach().cpu().tolist()],
            "index": len(order),
        }
    )
    setattr(env, "_aic_reset_event_order", order[-64:])

    episode_by_env = getattr(env, "_aic_current_episode_by_env", {}) or {}
    robot = env.scene["robot"]
    report = dict(getattr(env, "_aic_tcp_reset_report_by_env", {}) or {})
    joint_state_targets: list[tuple[int, dict]] = []
    for local_idx, env_id in enumerate(env_ids.tolist()):
        start = ((episode_by_env.get(int(env_id)) or {}).get("scene") or {}).get("start_near_gate") or {}
        if start.get("reset_mode") == "robot_joint_state":
            joint_state_targets.append((local_idx, start))
    if joint_state_targets:
        active_env_ids = torch.tensor(
            [int(env_ids[local_idx].item()) for local_idx, _ in joint_state_targets],
            dtype=torch.long,
            device=env.device,
        )
        joint_names_in = joint_state_targets[0][1].get("robot_joint_names")
        joint_positions_in = joint_state_targets[0][1].get("robot_joint_positions")
        if not isinstance(joint_names_in, (list, tuple)) or not isinstance(joint_positions_in, (list, tuple)):
            raise ValueError("robot_joint_state reset requires robot_joint_names and robot_joint_positions lists")
        reset_joint_names = [str(name) for name in joint_names_in]
        joint_ids, resolved_joint_names = robot.find_joints(reset_joint_names, preserve_order=True)
        if len(joint_ids) != len(reset_joint_names):
            raise RuntimeError(
                f"Could not resolve all robot_joint_state joints {reset_joint_names}; resolved {resolved_joint_names}"
            )
        q_rows = []
        qd_rows = []
        for _, start in joint_state_targets:
            raw_q = start.get("robot_joint_positions")
            raw_qd = start.get("robot_joint_velocities")
            if not isinstance(raw_q, (list, tuple)) or len(raw_q) != len(reset_joint_names):
                raise ValueError("robot_joint_positions length must match robot_joint_names")
            if raw_qd is None:
                raw_qd = [0.0] * len(reset_joint_names)
            if not isinstance(raw_qd, (list, tuple)) or len(raw_qd) != len(reset_joint_names):
                raise ValueError("robot_joint_velocities length must match robot_joint_names")
            q_rows.append(torch.tensor([float(v) for v in raw_q], dtype=torch.float32, device=env.device))
            qd_rows.append(torch.tensor([float(v) for v in raw_qd], dtype=torch.float32, device=env.device))
        q = torch.stack(q_rows, dim=0)
        qd = torch.stack(qd_rows, dim=0)
        robot.write_joint_state_to_sim(q, qd, joint_ids=joint_ids, env_ids=active_env_ids)
        robot.set_joint_position_target(q, joint_ids=joint_ids, env_ids=active_env_ids)
        if hasattr(env, "sim"):
            env.sim.forward()
        robot.update(0.0)
        if sync_action_term_after_reset:
            action_manager = getattr(env, "action_manager", None)
            if action_manager is not None:
                try:
                    action_term = action_manager.get_term("arm_action")
                except Exception:
                    action_term = None
                if action_term is not None and hasattr(action_term, "process_actions"):
                    zeros_action = torch.zeros(
                        (getattr(env, "num_envs", robot.data.joint_pos.shape[0]), action_term.action_dim),
                        dtype=robot.data.joint_pos.dtype,
                        device=env.device,
                    )
                    action_term.process_actions(zeros_action)
                    try:
                        action_manager._action[active_env_ids] = 0.0
                        action_manager._prev_action[active_env_ids] = 0.0
                    except Exception:
                        pass
        for row, env_id in enumerate(active_env_ids.tolist()):
            episode = episode_by_env.get(int(env_id)) or {}
            start = joint_state_targets[row][1]
            report[int(env_id)] = {
                "body_name": None,
                "episode_id": episode.get("episode_id"),
                "reset_mode": "robot_joint_state",
                "note": "direct per-episode robot joint state reset",
                "robot_joint_names": reset_joint_names,
                "robot_joint_positions": [float(v) for v in q[row].detach().cpu().tolist()],
                "robot_joint_velocities": [float(v) for v in qd[row].detach().cpu().tolist()],
                "full_joint_velocities_zeroed": bool(start.get("robot_joint_velocities") is None),
                "action_term_synced_after_reset": bool(sync_action_term_after_reset),
            }
        setattr(env, "_aic_tcp_reset_report_by_env", report)

    targets: list[tuple[int, torch.Tensor, torch.Tensor | None, str]] = []
    for local_idx, env_id in enumerate(env_ids.tolist()):
        start = ((episode_by_env.get(int(env_id)) or {}).get("scene") or {}).get("start_near_gate") or {}
        reset_mode = start.get("reset_mode")
        if reset_mode == "robot_joint_state":
            continue
        if reset_mode == "body_start_position_world":
            raw_target = start.get("body_start_position_world")
            reset_body_name = str(start.get("reset_body_name") or body_name)
            raw_orientation = start.get("body_start_orientation_wxyz") or start.get("tcp_start_orientation_world")
        elif reset_mode == "tcp_start_position_world":
            raw_target = start.get("tcp_start_position_world")
            reset_body_name = body_name
            raw_orientation = start.get("tcp_start_orientation_world")
        else:
            continue
        if not isinstance(raw_target, (list, tuple)) or len(raw_target) != 3:
            continue
        target_local = torch.tensor(raw_target, dtype=torch.float32, device=env.device)
        target_orientation = None
        if isinstance(raw_orientation, (list, tuple)) and len(raw_orientation) == 4:
            target_orientation = torch.tensor(raw_orientation, dtype=torch.float32, device=env.device)
            target_orientation = target_orientation / torch.linalg.norm(target_orientation).clamp(min=1.0e-9)
        targets.append((local_idx, target_local + env.scene.env_origins[env_ids[local_idx]], target_orientation, reset_body_name))
    if not targets:
        return
    reset_body_names = {name for _, _, _, name in targets}
    if len(reset_body_names) != 1:
        raise RuntimeError(f"Mixed near-gate reset bodies in one reset batch are not supported: {sorted(reset_body_names)}")
    active_body_name = next(iter(reset_body_names))

    body_ids, body_names = robot.find_bodies(active_body_name, preserve_order=True)
    if not body_ids:
        raise RuntimeError(f"Robot body {active_body_name!r} not found; available bodies: {robot.body_names}")
    joint_ids, resolved_joint_names = robot.find_joints(list(joint_names), preserve_order=True)
    if len(joint_ids) != len(joint_names):
        raise RuntimeError(
            f"Could not resolve all near-gate reset joints {joint_names}; resolved {resolved_joint_names}"
        )

    local_indices = torch.tensor([idx for idx, _, _, _ in targets], dtype=torch.long, device=env.device)
    active_env_ids = env_ids[local_indices]
    target_pos = torch.stack([target for _, target, _, _ in targets], dim=0)
    body_id = int(body_ids[0])
    jacobian_body_id = max(body_id - 1, 0)
    q = robot.data.joint_pos[active_env_ids][:, joint_ids].clone()
    zeros = torch.zeros_like(q)
    has_orientation = any(target_quat is not None for _, _, target_quat, _ in targets)
    if has_orientation:
        target_quat_rows = []
        for row, (_, _, target_quat, _) in enumerate(targets):
            if target_quat is None:
                target_quat_rows.append(robot.data.body_quat_w[active_env_ids[row], body_id].to(dtype=torch.float32))
            else:
                target_quat_rows.append(target_quat)
        target_quat = torch.stack(target_quat_rows, dim=0)
    else:
        target_quat = None
    eye_dim = 6 if has_orientation else 3
    eye = torch.eye(eye_dim, dtype=torch.float32, device=env.device).unsqueeze(0)
    initial_error = torch.linalg.norm(
        target_pos - robot.data.body_pos_w[active_env_ids, body_id].to(dtype=torch.float32),
        dim=1,
    )
    initial_orientation_error = (
        None
        if target_quat is None
        else quat_error_magnitude(
            robot.data.body_quat_w[active_env_ids, body_id].to(dtype=torch.float32),
            target_quat,
        )
    )

    for _ in range(max(1, int(max_iterations))):
        current_pos = robot.data.body_pos_w[active_env_ids, body_id].to(dtype=torch.float32)
        current_quat = robot.data.body_quat_w[active_env_ids, body_id].to(dtype=torch.float32)
        if target_quat is None:
            error = target_pos - current_pos
            converged = torch.linalg.norm(error, dim=1) <= float(position_tolerance)
        else:
            pos_error, rot_error = compute_pose_error(
                current_pos,
                current_quat,
                target_pos,
                target_quat,
                rot_error_type="axis_angle",
            )
            error = torch.cat([pos_error, rot_error], dim=1)
            converged = (torch.linalg.norm(pos_error, dim=1) <= float(position_tolerance)) & (
                torch.linalg.norm(rot_error, dim=1) <= float(orientation_tolerance)
            )
        if bool(converged.all()):
            break
        jacobians = robot.root_physx_view.get_jacobians()
        jac_rows = 6 if has_orientation else 3
        jac = jacobians[active_env_ids, jacobian_body_id, :jac_rows, :][:, :, joint_ids].to(dtype=torch.float32)
        jac_t = jac.transpose(1, 2)
        lhs = jac @ jac_t + (float(damping) ** 2) * eye
        dq = jac_t @ torch.linalg.solve(lhs, error.unsqueeze(-1))
        dq = dq.squeeze(-1).clamp(min=-float(max_joint_delta), max=float(max_joint_delta))
        q = q + dq
        limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if limits is not None:
            lo = limits[active_env_ids][:, joint_ids, 0]
            hi = limits[active_env_ids][:, joint_ids, 1]
            q = torch.max(torch.min(q, hi), lo)
        robot.write_joint_state_to_sim(q, zeros, joint_ids=joint_ids, env_ids=active_env_ids)
        robot.set_joint_position_target(q, joint_ids=joint_ids, env_ids=active_env_ids)
        if hasattr(env, "sim"):
            env.sim.forward()
        robot.update(0.0)

    full_q = robot.data.joint_pos[active_env_ids].clone()
    full_q[:, joint_ids] = q
    full_qd = torch.zeros_like(full_q)
    robot.write_joint_state_to_sim(full_q, full_qd, env_ids=active_env_ids)
    robot.set_joint_position_target(q, joint_ids=joint_ids, env_ids=active_env_ids)
    if hasattr(env, "sim"):
        env.sim.forward()
    robot.update(0.0)
    if sync_action_term_after_reset:
        action_manager = getattr(env, "action_manager", None)
        if action_manager is not None:
            try:
                action_term = action_manager.get_term("arm_action")
            except Exception:
                action_term = None
            if action_term is not None and hasattr(action_term, "process_actions"):
                zeros_action = torch.zeros(
                    (getattr(env, "num_envs", robot.data.joint_pos.shape[0]), action_term.action_dim),
                    dtype=robot.data.joint_pos.dtype,
                    device=env.device,
                )
                action_term.process_actions(zeros_action)
                try:
                    action_manager._action[active_env_ids] = 0.0
                    action_manager._prev_action[active_env_ids] = 0.0
                except Exception:
                    pass

    final_error = torch.linalg.norm(
        target_pos - robot.data.body_pos_w[active_env_ids, body_id].to(dtype=torch.float32),
        dim=1,
    )
    final_orientation_error = (
        None
        if target_quat is None
        else quat_error_magnitude(
            robot.data.body_quat_w[active_env_ids, body_id].to(dtype=torch.float32),
            target_quat,
        )
    )
    for row, env_id in enumerate(active_env_ids.tolist()):
        episode = episode_by_env.get(int(env_id)) or {}
        start = ((episode.get("scene") or {}).get("start_near_gate") or {})
        report[int(env_id)] = {
            "body_name": active_body_name,
            "episode_id": episode.get("episode_id"),
            "start_orientation_world": start.get("body_start_orientation_wxyz") or start.get("tcp_start_orientation_world"),
            "start_orientation_used": target_quat is not None,
            "note": "6D damped IK reset" if target_quat is not None else "position-only damped IK reset",
            "target_position_world": [float(v) for v in target_pos[row].detach().cpu().tolist()],
            "initial_error_m": float(initial_error[row].detach().cpu()),
            "final_error_m": float(final_error[row].detach().cpu()),
            "initial_orientation_error_rad": None
            if initial_orientation_error is None
            else float(initial_orientation_error[row].detach().cpu()),
            "final_orientation_error_rad": None
            if final_orientation_error is None
            else float(final_orientation_error[row].detach().cpu()),
            "max_iterations": int(max_iterations),
            "position_tolerance_m": float(position_tolerance),
            "orientation_tolerance_rad": float(orientation_tolerance),
            "full_joint_velocities_zeroed": True,
            "action_term_synced_after_reset": bool(sync_action_term_after_reset),
        }
    setattr(env, "_aic_tcp_reset_report_by_env", report)
