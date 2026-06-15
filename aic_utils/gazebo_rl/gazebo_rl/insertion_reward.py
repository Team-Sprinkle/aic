from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class GazeboRewardConfig:
    distance_weight: float = 0.25
    close_weight: float = 0.35
    progress_weight: float = 0.25
    progress_scale: float = 0.003
    orientation_weight: float = 0.10
    orientation_std: float = 0.03
    orientation_gate_sigma: float = 0.012
    reaching_weight: float = 0.0
    terminal_score_weight: float = 1.0
    lateral_weight: float = -0.05
    lateral_gate_sigma: float = 0.012
    lateral_error_scale: float = 0.006
    lateral_progress_weight: float = 0.0
    lateral_progress_scale: float = 0.001
    axial_progress_weight: float = 0.0
    axial_progress_scale: float = 0.001
    corridor_weight: float = 0.0
    corridor_sigma: float = 0.0025
    bypass_penalty_scale: float = 1.0
    corridor_orientation_gate_std: float = 0.0
    cheatcode_phase_weight: float = 0.0
    cheatcode_lateral_progress_weight: float = 0.40
    cheatcode_orientation_progress_weight: float = 0.30
    cheatcode_near_misaligned_weight: float = 0.25
    cheatcode_hover_weight: float = 0.15
    cheatcode_axial_progress_weight: float = 0.30
    cheatcode_corridor_weight: float = 1.50
    cheatcode_inside_alignment_weight: float = 0.20
    cheatcode_retreat_weight: float = 0.20
    target_reward_distance_std: float = 0.02
    target_reward_close_sigma: float = 0.006
    target_reward_reaching_threshold: float = 0.01
    target_reward_orientation_error_mode: str = "quat"
    target_reward_orientation_axis_local: tuple[float, float, float] | None = None
    force_delta_penalty_weight: float = 0.3
    force_delta_threshold: float = 10.0
    force_delta_reference: float = 20.0
    force_delta_saturation: float = 30.0
    force_delta_knee_penalty_fraction: float = 0.1
    insertion_target_depth_m: float = 0.0458

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def xyzw_to_wxyz(quat_xyzw: Any) -> list[float]:
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(-1)[:4]
    if q.shape[0] != 4:
        raise ValueError("xyzw quaternion must have 4 values")
    return _normalize_quat_wxyz([q[3], q[0], q[1], q[2]]).tolist()


def wxyz_to_xyzw(quat_wxyz: Any) -> list[float]:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)[:4]
    if q.shape[0] != 4:
        raise ValueError("wxyz quaternion must have 4 values")
    q = _normalize_quat_wxyz(q)
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def normalize_axis(axis: Any) -> np.ndarray:
    arr = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1.0e-9:
        raise ValueError("Cannot normalize near-zero insertion axis")
    return arr / norm


@dataclass
class InsertionGeometry:
    axial_depth: float
    target_depth: float
    target_lateral_residual: float
    lateral_error: float
    lateral_gate: float
    depth_fraction: float
    axis: np.ndarray


def compute_insertion_geometry(
    *,
    body_pos_w: Any,
    entrance_pos_w: Any,
    target_pos_w: Any,
    axis_w: Any,
    lateral_gate_sigma: float,
) -> InsertionGeometry:
    body = np.asarray(body_pos_w, dtype=np.float64).reshape(3)
    entrance = np.asarray(entrance_pos_w, dtype=np.float64).reshape(3)
    target = np.asarray(target_pos_w, dtype=np.float64).reshape(3)
    axis = normalize_axis(axis_w)
    target_delta = target - entrance
    target_depth = float(np.dot(target_delta, axis))
    target_lateral = float(np.linalg.norm(target_delta - target_depth * axis))
    if target_depth < 0.003 or target_depth > 0.060 or target_lateral > 0.002:
        raise RuntimeError(
            "Invalid insertion geometry: target/entrance/axis are inconsistent "
            f"(target_depth_m={target_depth:.6f}, target_lateral_residual_m={target_lateral:.6f})"
        )
    delta = body - entrance
    axial_depth = float(np.dot(delta, axis))
    lateral_error = float(np.linalg.norm(delta - axial_depth * axis))
    sigma = max(float(lateral_gate_sigma), 1.0e-9)
    lateral_gate = float(math.exp(-((lateral_error / sigma) ** 2)))
    depth_fraction = float(np.clip(axial_depth / target_depth, 0.0, 1.0))
    return InsertionGeometry(
        axial_depth=axial_depth,
        target_depth=target_depth,
        target_lateral_residual=target_lateral,
        lateral_error=lateral_error,
        lateral_gate=lateral_gate,
        depth_fraction=depth_fraction,
        axis=axis,
    )


def insertion_corridor_reward(geometry: InsertionGeometry, *, bypass_penalty_scale: float, semantic_gate: float = 1.0) -> float:
    gate = geometry.lateral_gate * float(semantic_gate)
    return float(geometry.depth_fraction * gate - geometry.depth_fraction * (1.0 - gate) * max(float(bypass_penalty_scale), 0.0))


def signed_axial_progress_reward(
    *,
    previous_depth: float,
    current_depth: float,
    lateral_gate: float,
    scale: float,
    semantic_gate: float = 1.0,
) -> float:
    progress = float(np.clip((current_depth - previous_depth) / max(float(scale), 1.0e-9), -1.0, 1.0))
    signed_gate = 2.0 * float(lateral_gate) * float(semantic_gate) - 1.0
    return float(progress * signed_gate if progress > 0.0 else progress)


def lateral_progress_reward(*, previous_lateral_error: float, current_lateral_error: float, scale: float) -> float:
    return float(np.clip((previous_lateral_error - current_lateral_error) / max(float(scale), 1.0e-9), -1.0, 1.0))


def cheatcode_insertion_phase_reward(
    *,
    geometry: InsertionGeometry,
    previous_depth: float,
    previous_lateral_error: float,
    orientation_error: float,
    previous_orientation_error: float,
    lateral_progress_weight: float = 0.40,
    orientation_progress_weight: float = 0.30,
    near_misaligned_weight: float = 0.25,
    hover_weight: float = 0.15,
    axial_progress_weight: float = 0.30,
    corridor_weight: float = 1.50,
    inside_alignment_weight: float = 0.20,
    retreat_weight: float = 0.20,
) -> dict[str, float]:
    s = float(geometry.axial_depth)
    r = float(geometry.lateral_error)
    theta = float(orientation_error)
    g_lat_pre = _pow4_gate(r, 0.0025)
    g_ori_pre = _pow4_gate(theta, 0.10)
    g_align_pre = g_lat_pre * g_ori_pre
    g_lat_insert = _pow4_gate(r, 0.0015)
    g_ori_insert = _pow4_gate(theta, 0.06)
    g_align_insert = g_lat_insert * g_ori_insert
    near_gate = _sigmoid_gate(s - (-0.008), 0.001)
    hover_gate = _sigmoid_gate(s - (-0.100), 0.010)
    inside_gate = _sigmoid_gate(s, 0.001)
    lat_progress = lateral_progress_reward(
        previous_lateral_error=previous_lateral_error,
        current_lateral_error=r,
        scale=0.001,
    )
    ori_progress = float(np.clip((previous_orientation_error - theta) / 0.02, -1.0, 1.0))
    near_misaligned = -near_gate * (
        max(0.0, r - 0.0015) / 0.0015 + max(0.0, theta - 0.06) / 0.06
    )
    hover = (1.0 - g_align_pre) * max(near_gate, hover_gate) * (-abs(s - (-0.004)) / 0.002)
    delta_s = s - float(previous_depth)
    forward = float(np.clip(delta_s / 0.001, -1.0, 1.0))
    axial = forward * (2.0 * g_align_insert - 1.0) if delta_s > 0.0 else -inside_gate * max(0.0, -delta_s / 0.001)
    corridor = geometry.depth_fraction * g_align_insert - geometry.depth_fraction * (1.0 - g_align_insert) * 6.0
    inside_alignment = -inside_gate * ((r / 0.001) ** 2 + (theta / 0.04) ** 2)
    retreat = -inside_gate * max(0.0, -delta_s / 0.001)
    total = (
        lateral_progress_weight * lat_progress
        + orientation_progress_weight * ori_progress
        + near_misaligned_weight * near_misaligned
        + hover_weight * hover
        + axial_progress_weight * axial
        + corridor_weight * corridor
        + inside_alignment_weight * inside_alignment
        + retreat_weight * retreat
    )
    return {
        "cheatcode_phase_total": float(total),
        "cheatcode/lateral_progress": float(lat_progress),
        "cheatcode/orientation_progress": float(ori_progress),
        "cheatcode/near_misaligned": float(near_misaligned),
        "cheatcode/preinsert_hover": float(hover),
        "cheatcode/axial_progress": float(axial),
        "cheatcode/corridor": float(corridor),
        "cheatcode/inside_alignment": float(inside_alignment),
        "cheatcode/retreat": float(retreat),
    }


def calculate_gazebo_insertion_reward(
    *,
    prev_obs: dict[str, Any] | None,
    obs: dict[str, Any] | None,
    terminal_score: float = 0.0,
    episode_config: dict[str, Any] | None = None,
    config: GazeboRewardConfig | None = None,
) -> tuple[float, dict[str, float]]:
    cfg = config or GazeboRewardConfig()
    target = _target_with_live_gazebo_geometry(
        ((episode_config or {}).get("scene") or {}).get("target") or {},
        obs,
        cfg,
    )
    body_pose = _reward_body_pose(obs)
    prev_body_pose = _reward_body_pose(prev_obs)
    if body_pose is None:
        return 0.0, {"task_geometry_reward_available": 0.0, "insertion_reward_available": 0.0}
    body_pos = _position(body_pose)
    target_pos = _target_position(target, obs, cfg)
    if body_pos is None or target_pos is None:
        return 0.0, {"task_geometry_reward_available": 0.0, "insertion_reward_available": 0.0}

    distance = float(np.linalg.norm(target_pos - body_pos))
    prev_distance = None
    prev_body_pos = _position(prev_body_pose)
    if prev_body_pos is not None:
        prev_distance = float(np.linalg.norm(target_pos - prev_body_pos))
    progress = 0.0 if prev_distance is None else float(np.clip((prev_distance - distance) / max(cfg.progress_scale, 1.0e-9), -1.0, 1.0))
    distance_reward = 1.0 - math.tanh(distance / max(cfg.target_reward_distance_std, 1.0e-9))
    close_reward = math.exp(-((distance * distance) / max(cfg.target_reward_close_sigma * cfg.target_reward_close_sigma, 1.0e-12)))
    orientation_error = _orientation_error(body_pose, target, obs, cfg)
    prev_orientation_error = _orientation_error(prev_body_pose, target, prev_obs, cfg) if prev_body_pose else orientation_error
    orientation_reward = 0.0 if orientation_error is None else 1.0 - math.tanh(orientation_error / max(cfg.orientation_std, 1.0e-9))

    reward = (
        cfg.distance_weight * distance_reward
        + cfg.close_weight * close_reward
        + cfg.progress_weight * progress
        + cfg.orientation_weight * orientation_reward
        + cfg.reaching_weight * (1.0 if distance <= cfg.target_reward_reaching_threshold else 0.0)
        + cfg.terminal_score_weight * float(terminal_score)
    )
    info: dict[str, float] = {
        "task_geometry_reward_available": 1.0,
        "insertion_reward_available": 1.0,
        "distance_m": distance,
        "distance_reward": float(distance_reward),
        "close_reward": float(close_reward),
        "progress": float(progress),
        "orientation_error_rad": -1.0 if orientation_error is None else float(orientation_error),
        "orientation_reward": float(orientation_reward),
        "terminal_score_reward": float(terminal_score),
    }

    entrance_pos = _entrance_position(target)
    axis = target.get("insertion_axis_world")
    if entrance_pos is not None and axis is not None:
        geom = compute_insertion_geometry(
            body_pos_w=body_pos,
            entrance_pos_w=entrance_pos,
            target_pos_w=target_pos,
            axis_w=axis,
            lateral_gate_sigma=cfg.lateral_gate_sigma,
        )
        prev_geom = geom
        if prev_body_pos is not None:
            prev_geom = compute_insertion_geometry(
                body_pos_w=prev_body_pos,
                entrance_pos_w=entrance_pos,
                target_pos_w=target_pos,
                axis_w=axis,
                lateral_gate_sigma=cfg.lateral_gate_sigma,
            )
        semantic_gate = 1.0
        if cfg.corridor_orientation_gate_std > 0.0 and orientation_error is not None:
            semantic_gate = math.exp(-((orientation_error / cfg.corridor_orientation_gate_std) ** 2))
        lateral_progress = lateral_progress_reward(
            previous_lateral_error=prev_geom.lateral_error,
            current_lateral_error=geom.lateral_error,
            scale=cfg.lateral_progress_scale,
        )
        axial_progress = signed_axial_progress_reward(
            previous_depth=prev_geom.axial_depth,
            current_depth=geom.axial_depth,
            lateral_gate=geom.lateral_gate,
            scale=cfg.axial_progress_scale,
            semantic_gate=semantic_gate,
        )
        corridor = insertion_corridor_reward(
            geom,
            bypass_penalty_scale=cfg.bypass_penalty_scale,
            semantic_gate=semantic_gate,
        )
        lateral_penalty = cfg.lateral_weight * (geom.lateral_error / max(cfg.lateral_error_scale, 1.0e-9))
        reward += (
            lateral_penalty
            + cfg.lateral_progress_weight * lateral_progress
            + cfg.axial_progress_weight * axial_progress
            + cfg.corridor_weight * corridor
        )
        info.update(
            {
                "lateral_error_m": float(geom.lateral_error),
                "axial_depth_m": float(geom.axial_depth),
                "target_depth_m": float(geom.target_depth),
                "lateral_gate": float(geom.lateral_gate),
                "lateral_progress": float(lateral_progress),
                "axial_progress": float(axial_progress),
                "corridor": float(corridor),
                "lateral_error_reward": float(lateral_penalty),
            }
        )
        if cfg.cheatcode_phase_weight:
            cheat = cheatcode_insertion_phase_reward(
                geometry=geom,
                previous_depth=prev_geom.axial_depth,
                previous_lateral_error=prev_geom.lateral_error,
                orientation_error=0.0 if orientation_error is None else orientation_error,
                previous_orientation_error=0.0 if prev_orientation_error is None else prev_orientation_error,
                lateral_progress_weight=cfg.cheatcode_lateral_progress_weight,
                orientation_progress_weight=cfg.cheatcode_orientation_progress_weight,
                near_misaligned_weight=cfg.cheatcode_near_misaligned_weight,
                hover_weight=cfg.cheatcode_hover_weight,
                axial_progress_weight=cfg.cheatcode_axial_progress_weight,
                corridor_weight=cfg.cheatcode_corridor_weight,
                inside_alignment_weight=cfg.cheatcode_inside_alignment_weight,
                retreat_weight=cfg.cheatcode_retreat_weight,
            )
            reward += cfg.cheatcode_phase_weight * cheat["cheatcode_phase_total"]
            info.update(cheat)

    force_penalty = _force_delta_penalty(prev_obs, obs, cfg)
    reward += force_penalty
    info["force_delta_penalty"] = float(force_penalty)
    info["reward/total"] = float(reward)
    return float(reward), info


def _pow4_gate(error: float, sigma: float) -> float:
    scaled = float(error) / max(float(sigma), 1.0e-9)
    return float(math.exp(-(scaled ** 4)))


def _sigmoid_gate(value: float, scale: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(value) / max(float(scale), 1.0e-9))))


def _normalize_quat_wxyz(quat: Any) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    q = q / norm
    return -q if q[0] < 0.0 else q


def _quat_error_wxyz(a: Any, b: Any) -> float:
    qa = _normalize_quat_wxyz(a)
    qb = _normalize_quat_wxyz(b)
    dot = float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0))
    return float(2.0 * math.acos(dot))


def _position(pose: dict[str, Any] | None) -> np.ndarray | None:
    if not isinstance(pose, dict) or pose.get("position") is None:
        return None
    arr = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
    return arr[:3] if arr.shape[0] >= 3 else None


def _reward_body_pose(obs: dict[str, Any] | None) -> dict[str, Any] | None:
    oracle = (obs or {}).get("oracle") or {}
    return oracle.get("plug_pose_base_link") or ((obs or {}).get("controller") or {}).get("current_tcp_pose")


def _target_with_live_gazebo_geometry(
    target: dict[str, Any],
    obs: dict[str, Any] | None,
    cfg: GazeboRewardConfig,
) -> dict[str, Any]:
    oracle = (obs or {}).get("oracle") or {}
    entrance_pose = oracle.get("target_port_entrance_pose_base_link")
    port_pose = oracle.get("target_port_pose_base_link")
    entrance = _position(entrance_pose)
    port = _position(port_pose)
    if entrance is None or port is None:
        return dict(target)
    axis = port - entrance
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-9:
        return dict(target)
    axis = axis / norm
    depth = float(target.get("insertion_target_depth_m") or target.get("seated_depth_m") or cfg.insertion_target_depth_m)
    live_target = dict(target)
    orientation_xyzw = (port_pose or {}).get("orientation_xyzw") or (entrance_pose or {}).get("orientation_xyzw")
    pose: dict[str, Any] = {"position": (entrance + axis * depth).astype(float).tolist()}
    if orientation_xyzw is not None:
        pose["orientation_wxyz"] = xyzw_to_wxyz(orientation_xyzw)
    live_target["target_pose_world"] = pose
    live_target["entrance_pose_world"] = {
        "position": entrance.astype(float).tolist(),
        **({"orientation_wxyz": xyzw_to_wxyz((entrance_pose or {}).get("orientation_xyzw"))} if (entrance_pose or {}).get("orientation_xyzw") is not None else {}),
    }
    live_target["insertion_axis_world"] = axis.astype(float).tolist()
    live_target["gazebo_live_geometry"] = True
    return live_target


def _target_position(target: dict[str, Any], obs: dict[str, Any] | None, cfg: GazeboRewardConfig) -> np.ndarray | None:
    pose = target.get("target_pose_world") or {}
    pos = _position(pose)
    entrance = _entrance_position(target)
    axis_raw = target.get("insertion_axis_world")
    if pos is not None and entrance is not None and axis_raw is not None:
        axis = normalize_axis(axis_raw)
        delta = pos - entrance
        depth = float(np.dot(delta, axis))
        lateral = float(np.linalg.norm(delta - depth * axis))
        if 0.003 <= depth <= 0.060 and lateral <= 0.002:
            return pos
        depth = float(target.get("insertion_target_depth_m") or target.get("seated_depth_m") or cfg.insertion_target_depth_m)
        return entrance + axis * depth
    if pos is not None:
        return pos
    return _position(((obs or {}).get("oracle") or {}).get("target_port_pose_base_link"))


def _entrance_position(target: dict[str, Any]) -> np.ndarray | None:
    return _position(target.get("entrance_pose_world") or {})


def _orientation_error(
    body_pose: dict[str, Any] | None,
    target: dict[str, Any],
    obs: dict[str, Any] | None,
    cfg: GazeboRewardConfig,
) -> float | None:
    if body_pose is None:
        return None
    body_q_xyzw = body_pose.get("orientation_xyzw")
    if body_q_xyzw is None:
        return None
    target_q = ((target.get("target_pose_world") or {}).get("orientation_wxyz"))
    if target_q is None:
        oracle_target_q = (((obs or {}).get("oracle") or {}).get("target_port_pose_base_link") or {}).get("orientation_xyzw")
        if oracle_target_q is None:
            return None
        target_q = xyzw_to_wxyz(oracle_target_q)
    return _quat_error_wxyz(xyzw_to_wxyz(body_q_xyzw), target_q)


def _force_delta_penalty(prev_obs: dict[str, Any] | None, obs: dict[str, Any] | None, cfg: GazeboRewardConfig) -> float:
    if cfg.force_delta_penalty_weight <= 0.0:
        return 0.0
    prev_force = (((prev_obs or {}).get("wrist_wrench") or {}).get("force"))
    force = (((obs or {}).get("wrist_wrench") or {}).get("force"))
    if prev_force is None or force is None:
        return 0.0
    delta = float(np.linalg.norm(np.asarray(force, dtype=np.float64)[:3] - np.asarray(prev_force, dtype=np.float64)[:3]))
    if delta <= cfg.force_delta_threshold:
        return 0.0
    span = max(cfg.force_delta_saturation - cfg.force_delta_threshold, 1.0e-9)
    x = min((delta - cfg.force_delta_threshold) / span, 1.0)
    knee = cfg.force_delta_knee_penalty_fraction * min(delta / max(cfg.force_delta_reference, 1.0e-9), 1.0)
    return -float(cfg.force_delta_penalty_weight) * max(float(knee), x * x)
