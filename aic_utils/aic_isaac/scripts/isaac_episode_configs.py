#!/usr/bin/env python3
"""Materialize Isaac Lab episode configs from the user-facing minimal YAML."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

TASK_FAMILIES = {"sfp_to_nic", "sc_to_sc"}
DEFAULT_BOARD_POS = (0.2837, 0.229, 0.0)
DEFAULT_BOARD_RANGE = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)}
DEFAULT_PARTS = {
    "sc_port": {"scene_name": "sc_port", "offset": (0.0067, -0.0362, 0.005), "pose_range": {}},
    "sc_port_2": {"scene_name": "sc_port_2", "offset": (0.0076, -0.0783, 0.005), "pose_range": {}},
    "nic_card": {"scene_name": "nic_card", "offset": (-0.03235, 0.02329, 0.0743), "pose_range": {}, "snap_step": {"y": 0.04}},
}
NIC_CARD_ROT_WXYZ = (0.0, 0.0, -0.7068252, 0.7073883)
SFP_PORT_SEATED_TARGET_ROOT_LOCAL = {
    0: (0.01059, -0.07594, 0.01540),
    1: (-0.01261, -0.07594, 0.01540),
}
SFP_PORT_LOCAL = {
    0: (0.01295, -0.031572, 0.00501),
    1: (-0.01025, -0.031572, 0.00501),
}
SFP_PORT_RPY = (4.69895, 0.0, 0.0)
SFP_PORT_ENTRANCE_LOCAL = (0.0, 0.0, -0.0458)
SC_PORT_TARGET_LOCAL = (0.093, 0.140, 0.020)
SC_INSERTION_AXIS_WORLD = (0.0, 1.0, 0.0)


class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _safe_dump(data: Any) -> str:
    return yaml.dump(data, Dumper=_NoAliasDumper, sort_keys=False)


def load_request(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Isaac user config must be a mapping: {path}")
    return data


def _require(data: dict[str, Any], dotted: str) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise ValueError(f"Missing required field: {dotted}")
        cur = cur[part]
    return cur


def validate_request(request: dict[str, Any]) -> None:
    family = _require(request, "task_family")
    if family not in TASK_FAMILIES:
        raise ValueError(f"task_family must be one of {sorted(TASK_FAMILIES)}")
    generation = request.get("generation") or {}
    if generation and "target_accepted_trajectories" in generation:
        if int(generation["target_accepted_trajectories"]) <= 0:
            raise ValueError("generation.target_accepted_trajectories must be positive")
    acceptance = request.get("acceptance") or {}
    if acceptance.get("stop_near_gate") is not None and acceptance.get("near_gate") is not None:
        raise ValueError("Use acceptance.stop_near_gate or deprecated acceptance.near_gate, not both")
    scene = request.get("scene") or {}
    if family == "sfp_to_nic":
        _require(request, "scene.nic_cards.count")
    else:
        _require(request, "scene.sc_ports.count")
    start = scene.get("start_near_gate")
    if start is not None:
        if not isinstance(start, dict):
            raise ValueError("scene.start_near_gate must be a mapping")
        if "distance" not in start and (
            "axial_distance_m" not in start or "lateral_distance_m" not in start
        ):
            raise ValueError("scene.start_near_gate requires distance or axial_distance_m/lateral_distance_m")
        offset = start.get("reset_body_offset_from_reference_world")
        if offset is not None and (not isinstance(offset, (list, tuple)) or len(offset) != 3):
            raise ValueError("scene.start_near_gate.reset_body_offset_from_reference_world must be a 3-value list")
        orientation = start.get("reset_body_orientation_wxyz")
        if orientation is not None and (not isinstance(orientation, (list, tuple)) or len(orientation) != 4):
            raise ValueError("scene.start_near_gate.reset_body_orientation_wxyz must be a 4-value list")


def _choices(value: Any, default: list[Any]) -> list[Any]:
    if value is None or value == "auto":
        return list(default)
    if isinstance(value, list):
        if not value:
            raise ValueError("choice list must not be empty")
        return value
    return [value]


def _sample_int(value: Any, default: list[int], rng: random.Random) -> int:
    return int(rng.choice([int(v) for v in _choices(value, default)]))


def _port_index(value: Any) -> int:
    if isinstance(value, str):
        if value.startswith("sfp_port_"):
            return int(value.removeprefix("sfp_port_"))
        if value.startswith("sc_port_"):
            return int(value.removeprefix("sc_port_"))
    return int(value)


def _vadd(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return (a[0] * scale, a[1] * scale, a[2] * scale)


def _vdot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vnorm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_vdot(a, a))


def _normalize(a: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _vnorm(a)
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return _vscale(a, 1.0 / norm)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> tuple[tuple[float, float, float], ...]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matvec(
    mat: tuple[tuple[float, float, float], ...],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(sum(mat[i][j] * vec[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _quat_conj_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def _quat_normalize_wxyz(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-12:
        raise ValueError("Cannot normalize near-zero quaternion")
    return tuple(float(v) / norm for v in q)  # type: ignore[return-value]


def _quat_mul_wxyz(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = lhs
    rw, rx, ry, rz = rhs
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return _quat_normalize_wxyz(
        (
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        )
    )


def _round4(value: tuple[float, float, float, float]) -> list[float]:
    return [round(float(v), 6) for v in _quat_normalize_wxyz(value)]


def _quat_apply_wxyz(
    quat: tuple[float, float, float, float],
    vec: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(quat, (0.0, vec[0], vec[1], vec[2])), _quat_conj_wxyz(quat))
    return (rotated[1], rotated[2], rotated[3])


def _perpendicular(axis: tuple[float, float, float], rng: random.Random) -> tuple[float, float, float]:
    axis = _normalize(axis)
    seed = (0.0, 0.0, 1.0) if abs(axis[2]) < 0.9 else (1.0, 0.0, 0.0)
    a = _normalize(_cross(axis, seed))
    b = _cross(axis, a)
    theta = rng.uniform(0.0, 2.0 * math.pi)
    return _vadd(_vscale(a, math.cos(theta)), _vscale(b, math.sin(theta)))


def _round3(value: tuple[float, float, float]) -> list[float]:
    return [round(float(v), 6) for v in value]


def _sample_context(request: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    scene = request.get("scene") or {}
    family = str(rng.choice(_choices(request.get("task_family"), [request["task_family"]])))
    if family == "sfp_to_nic":
        nic = scene.get("nic_cards") or {}
        count = max(1, min(5, _sample_int(nic.get("count"), [1, 2, 3, 4, 5], rng)))
        target_card_raw = nic.get("target_card", "auto")
        target_card = rng.randrange(count) if target_card_raw == "auto" else _sample_int(target_card_raw, list(range(count)), rng)
        target_port = _port_index(rng.choice(_choices(nic.get("target_port"), [0, 1])))
        if target_port not in {0, 1}:
            raise ValueError(f"sfp_to_nic target_port must resolve to 0 or 1, got {target_port}")
        return {
            "task_family": family,
            "target_port_index": target_port,
            "target_card_index": target_card,
            "target_card_valid": 1,
            "nic_card_count": count,
            "sc_port_count": 0,
        }
    if family == "sc_to_sc":
        sc = scene.get("sc_ports") or {}
        count = max(1, min(2, _sample_int(sc.get("count"), [1, 2], rng)))
        target_raw = sc.get("target_port", "auto")
        target_port = rng.randrange(count) if target_raw == "auto" else _port_index(rng.choice(_choices(target_raw, list(range(count)))))
        if target_port not in {0, 1}:
            raise ValueError(f"sc_to_sc target_port must resolve to 0 or 1, got {target_port}")
        return {
            "task_family": family,
            "target_port_index": target_port,
            "target_card_index": -1,
            "target_card_valid": 0,
            "nic_card_count": 0,
            "sc_port_count": count,
        }
    raise ValueError(f"Unsupported task_family: {family}")


def _target_spec(
    context: dict[str, Any],
    board_pos: tuple[float, float, float],
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if context["task_family"] == "sfp_to_nic":
        scene = (request or {}).get("scene") or {}
        target_cfg = scene.get("target") or {}
        entrance_axis_offset_m = float(target_cfg.get("entrance_axis_offset_m", 0.0))
        part = DEFAULT_PARTS["nic_card"]
        target_port_index = int(context["target_port_index"])
        target_offset = SFP_PORT_SEATED_TARGET_ROOT_LOCAL[target_port_index]
        root_position = _vadd(board_pos, part["offset"])
        position = _vadd(root_position, _quat_apply_wxyz(NIC_CARD_ROT_WXYZ, target_offset))
        port_rotation = _rpy_matrix(*SFP_PORT_RPY)
        target_orientation = _quat_mul_wxyz(NIC_CARD_ROT_WXYZ, _quat_from_rpy(*SFP_PORT_RPY))
        body_orientation = _quat_mul_wxyz(target_orientation, _quat_conj_wxyz(_quat_from_rpy(0.0, math.pi, 0.0)))
        entrance_offset_local = _vadd(
            SFP_PORT_LOCAL[target_port_index],
            _matvec(port_rotation, SFP_PORT_ENTRANCE_LOCAL),
        )
        entrance_position = _vadd(root_position, _quat_apply_wxyz(NIC_CARD_ROT_WXYZ, entrance_offset_local))
        entrance_axis = _normalize(_quat_apply_wxyz(NIC_CARD_ROT_WXYZ, _matvec(port_rotation, (0.0, 0.0, 1.0))))
        if entrance_axis_offset_m:
            entrance_position = _vadd(entrance_position, _vscale(entrance_axis, entrance_axis_offset_m))
        seated_depth_m = target_cfg.get("seated_depth_m")
        if seated_depth_m is not None:
            seated_depth = float(seated_depth_m)
            if seated_depth <= 0.0:
                raise ValueError("scene.target.seated_depth_m must be positive")
            position = _vadd(entrance_position, _vscale(entrance_axis, seated_depth))
        return {
            "scene_name": "nic_card",
            "target_reward_body": "sfp_tip_link",
            "target_position_offset": _round3(target_offset),
            "body_position_offset": [0.0, 0.0, 0.0],
            "target_pose_world": {"position": _round3(position), "orientation_wxyz": _round4(target_orientation)},
            "entrance_pose_world": {"position": _round3(entrance_position), "orientation_wxyz": _round4(target_orientation)},
            "body_start_orientation_wxyz": _round4(body_orientation),
            "entrance_position_offset": _round3(entrance_offset_local),
            "entrance_axis_offset_m": round(entrance_axis_offset_m, 6),
            "seated_depth_m": None if seated_depth_m is None else round(float(seated_depth_m), 6),
            "insertion_axis_world": _round3(entrance_axis),
        }
    scene_name = "sc_port" if int(context["target_port_index"]) == 0 else "sc_port_2"
    part = DEFAULT_PARTS[scene_name]
    position = _vadd(_vadd(board_pos, part["offset"]), SC_PORT_TARGET_LOCAL)
    return {
        "scene_name": scene_name,
        "target_reward_body": "sfp_tip_link",
        "target_position_offset": _round3(SC_PORT_TARGET_LOCAL),
        "body_position_offset": [0.0, 0.0, 0.0],
        "target_pose_world": {"position": _round3(position), "orientation_wxyz": None},
        "insertion_axis_world": _round3(SC_INSERTION_AXIS_WORLD),
    }


def _apply_start_near_gate(
    *,
    request: dict[str, Any],
    context: dict[str, Any],
    board_pos: tuple[float, float, float],
    rng: random.Random,
) -> tuple[tuple[float, float, float], dict[str, Any] | None]:
    """Compute near-gate body-start metadata without moving the target scene.

    Near-gate curricula should place the plug tip just outside the port
    entrance.  The child YAML therefore records the reset body and desired body
    start position while leaving the board/target scene unchanged.
    """
    start = (request.get("scene") or {}).get("start_near_gate")
    if not isinstance(start, dict):
        return board_pos, None
    target = _target_spec(context, board_pos, request)
    gate_position = tuple(
        float(v)
        for v in (target.get("entrance_pose_world") or target["target_pose_world"])["position"]
    )
    axis = tuple(float(v) for v in target["insertion_axis_world"])
    reset_body_name = str(start.get("reset_body_name") or target.get("target_reward_body") or "sfp_tip_link")
    min_clearance = float(start.get("min_clearance_m", 0.02))
    if "distance" in start:
        distance = float(start["distance"])
        if distance < 0.0:
            raise ValueError("scene.start_near_gate.distance must be non-negative")
        if distance < min_clearance:
            raise ValueError("scene.start_near_gate.distance must be at least min_clearance_m")
        reference_raw = start.get("reference_body_position") or start.get("reference_tcp_position")
        if reference_raw is None:
            reference = _vadd(gate_position, _vscale(axis, distance))
        else:
            if not isinstance(reference_raw, (list, tuple)) or len(reference_raw) != 3:
                raise ValueError("scene.start_near_gate.reference_body_position must be a 3-value list")
            reference = (float(reference_raw[0]), float(reference_raw[1]), float(reference_raw[2]))
        direction = _vsub(gate_position, reference)
        if _vnorm(direction) < 1e-6:
            direction = _perpendicular(axis, rng)
        requested = {"distance": distance}
    else:
        axial = float(start["axial_distance_m"])
        lateral = float(start["lateral_distance_m"])
        if axial < 0.0:
            raise ValueError(
                "scene.start_near_gate.axial_distance_m must be non-negative so the tip starts outside "
                "the entrance plane, not already past it."
            )
        if lateral < 0.0:
            raise ValueError("scene.start_near_gate.lateral_distance_m must be non-negative")
        if math.sqrt(axial * axial + lateral * lateral) < min_clearance:
            raise ValueError("scene.start_near_gate combined distance must be at least min_clearance_m")
        lateral_dir = _perpendicular(axis, rng)
        reference_raw = start.get("reference_body_position") or start.get("reference_tcp_position")
        if reference_raw is None:
            # ``axis`` points from the entrance into the port.  Near-gate
            # resets should start outside the entrance plane and then insert
            # along +axis, so the axial offset is opposite the insertion axis.
            reference = _vadd(gate_position, _vadd(_vscale(axis, -axial), _vscale(lateral_dir, lateral)))
        else:
            if not isinstance(reference_raw, (list, tuple)) or len(reference_raw) != 3:
                raise ValueError("scene.start_near_gate.reference_body_position must be a 3-value list")
            reference = (float(reference_raw[0]), float(reference_raw[1]), float(reference_raw[2]))
        requested = {
            "axial_distance_m": axial,
            "lateral_distance_m": lateral,
            "lateral_direction_world": _round3(lateral_dir),
        }
    axes = str(start.get("axes", "xyz")).lower()
    if axes not in {"xyz", "xy"}:
        raise ValueError("scene.start_near_gate.axes must be 'xyz' or 'xy'")
    offset_raw = start.get("reset_body_offset_from_reference_world")
    reset_body_offset = (0.0, 0.0, 0.0)
    if offset_raw is not None:
        if not isinstance(offset_raw, (list, tuple)) or len(offset_raw) != 3:
            raise ValueError("scene.start_near_gate.reset_body_offset_from_reference_world must be a 3-value list")
        reset_body_offset = (float(offset_raw[0]), float(offset_raw[1]), float(offset_raw[2]))
    reset_body_position = _vadd(reference, reset_body_offset)
    orientation_raw = start.get("reset_body_orientation_wxyz")
    reset_body_orientation = target.get("body_start_orientation_wxyz")
    if orientation_raw is not None:
        if not isinstance(orientation_raw, (list, tuple)) or len(orientation_raw) != 4:
            raise ValueError("scene.start_near_gate.reset_body_orientation_wxyz must be a 4-value list")
        reset_body_orientation = [round(float(v), 6) for v in orientation_raw]
    ref_delta = _vsub(reference, gate_position)
    axial_component = abs(_vdot(ref_delta, axis))
    lateral_component = math.sqrt(max(0.0, _vnorm(ref_delta) ** 2 - axial_component * axial_component))
    return board_pos, {
        **requested,
        "reset_mode": "body_start_position_world",
        "reset_body_name": reset_body_name,
        "body_start_position_world": _round3(reset_body_position),
        "body_start_orientation_wxyz": reset_body_orientation,
        "reference_reward_body_name": target.get("target_reward_body") or "sfp_tip_link",
        "reference_reward_body_start_position_world": _round3(reference),
        "reference_reward_body_start_orientation_wxyz": target.get("body_start_orientation_wxyz"),
        "reset_body_offset_from_reference_world": _round3(reset_body_offset),
        "reset_body_orientation_wxyz": reset_body_orientation,
        "reference_body_position": _round3(reference),
        # Backward-compatible aliases for older diagnostics.
        "tcp_start_position_world": _round3(reset_body_position),
        "tcp_start_orientation_world": reset_body_orientation,
        "reference_tcp_position": _round3(reference),
        "target_gate_position": _round3(gate_position),
        "target_gate_axis_world": _round3(axis),
        "target_gate_source": "entrance_pose_world" if target.get("entrance_pose_world") else "target_pose_world",
        "achieved_distance": round(_vnorm(ref_delta), 6),
        "achieved_axial_distance_m": round(axial_component, 6),
        "achieved_lateral_distance_m": round(lateral_component, 6),
        "axes": axes,
        "min_clearance_m": min_clearance,
    }


def _episode_config(request: dict[str, Any], index: int, rng: random.Random) -> dict[str, Any]:
    context = _sample_context(request, rng)
    board_pos, start_metadata = _apply_start_near_gate(
        request=request,
        context=context,
        board_pos=DEFAULT_BOARD_POS,
        rng=rng,
    )
    target = _target_spec(context, board_pos, request)
    parts = [dict(DEFAULT_PARTS[name]) for name in ("sc_port", "sc_port_2", "nic_card")]
    task_description = (
        f"Insert SFP cable into NIC card {context['target_card_index']} port {context['target_port_index']}"
        if context["task_family"] == "sfp_to_nic"
        else f"Insert SC cable into SC port {context['target_port_index']}"
    )
    stop_near_gate = (request.get("acceptance") or {}).get("stop_near_gate")
    if stop_near_gate is None:
        stop_near_gate = (request.get("acceptance") or {}).get("near_gate")
    return {
        "episode_id": f"episode_{index:06d}",
        "episode_index": index,
        "task_description": task_description,
        "task_context": context,
        "scene": {
            "task_board": {
                "position_world": _round3(board_pos),
                "yaw": 0.0,
            },
            "parts": {
                part["scene_name"]: {
                    "offset_from_board": _round3(tuple(float(v) for v in part["offset"])),
                    "position_world": _round3(_vadd(board_pos, tuple(float(v) for v in part["offset"]))),
                    "pose_range": part.get("pose_range", {}),
                    "snap_step": part.get("snap_step", {}),
                }
                for part in parts
            },
            "target": target,
            "start_near_gate": start_metadata,
            "stop_near_gate": stop_near_gate or None,
        },
        "isaac_randomization": {
            "board_scene_name": "task_board",
            "board_default_pos": _round3(board_pos),
            "board_range": {k: [float(v[0]), float(v[1])] for k, v in DEFAULT_BOARD_RANGE.items()},
            "parts": [
                {
                    **part,
                    "offset": _round3(tuple(float(v) for v in part["offset"])),
                    "pose_range": part.get("pose_range", {}),
                    "snap_step": part.get("snap_step", {}),
                }
                for part in parts
            ],
        },
    }


def materialize_episode_configs(
    request_path: Path,
    output_dir: Path,
    *,
    episode_count: int | None = None,
    episode_start_index: int = 1,
    source_name: str | None = None,
) -> dict[str, Any]:
    request = load_request(request_path)
    validate_request(request)
    generation = request.get("generation") or {}
    count = int(episode_count or generation.get("target_accepted_trajectories") or 1)
    if count <= 0:
        raise ValueError("episode_count must be positive")
    seed = int(generation.get("seed", 1))
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(request_path, output_dir / "request.yaml")
    episodes: list[dict[str, Any]] = []
    for local_index in range(1, count + 1):
        episode = _episode_config(request, episode_start_index + local_index - 1, rng)
        episode["source_request"] = {
            "path": str(request_path),
            "name": source_name or request_path.stem,
            "local_episode_index": local_index,
        }
        episodes.append(episode)
        (episodes_dir / f"{episode['episode_id']}.yaml").write_text(
            _safe_dump(episode),
            encoding="utf-8",
        )
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "episode_id",
            "task_description",
            "task_family",
            "target_port_index",
            "target_card_index",
            "target_card_valid",
            "target_scene_name",
            "target_position_x",
            "target_position_y",
            "target_position_z",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            context = episode["task_context"]
            target = episode["scene"]["target"]
            pos = target["target_pose_world"]["position"]
            writer.writerow(
                {
                    "episode_id": episode["episode_id"],
                    "task_description": episode["task_description"],
                    "task_family": context["task_family"],
                    "target_port_index": context["target_port_index"],
                    "target_card_index": context["target_card_index"],
                    "target_card_valid": context["target_card_valid"],
                    "target_scene_name": target["scene_name"],
                    "target_position_x": pos[0],
                    "target_position_y": pos[1],
                    "target_position_z": pos[2],
                }
            )
    task_distribution_path = output_dir / "task_distribution.yaml"
    task_distribution_path.write_text(
        _safe_dump(
            {
                "task_family": sorted({ep["task_context"]["task_family"] for ep in episodes}),
                "episodes": [
                    {
                        "episode_id": ep["episode_id"],
                        **ep["task_context"],
                    }
                    for ep in episodes
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "request_yaml": str(request_path),
        "output_dir": str(output_dir),
        "episodes_dir": str(episodes_dir),
        "manifest_csv": str(manifest_path),
        "task_distribution_yaml": str(task_distribution_path),
        "episode_count": count,
        "seed": seed,
        "first_episode": episodes[0] if episodes else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _split_filenames(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _available_gpu_ids(max_gpus: int) -> list[int]:
    if max_gpus <= 0:
        return [0]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        ids = [int(item.strip()) for item in visible.split(",") if item.strip().isdigit()]
        return ids[:max_gpus] or list(range(max_gpus))
    return list(range(max_gpus))


def materialize_many_episode_configs(
    *,
    input_dir: Path,
    output_dir: Path,
    filenames: list[str] | None = None,
    input_yamls: list[Path] | None = None,
    max_gpus: int = 1,
    episode_count_override: int | None = None,
) -> dict[str, Any]:
    """Materialize multiple minimal request YAMLs into one curriculum-ordered child YAML set.

    Episodes are generated request-by-request. Shards are then assigned in the
    same curriculum order with round-robin GPU ids:
    a_001 -> gpu0, a_002 -> gpu1, ..., b_001 -> gpu0, ...
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = output_dir / "episodes"
    shards_dir = output_dir / "shards"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    shards_dir.mkdir(parents=True, exist_ok=True)

    request_paths: list[Path] = []
    for path in input_yamls or []:
        request_paths.append(path)
    for name in filenames or []:
        request_paths.append(input_dir / name)
    if not request_paths and input_dir.exists():
        request_paths = sorted(input_dir.glob("*.yaml")) + sorted(input_dir.glob("*.yml"))
    if not request_paths:
        raise ValueError("No minimal YAML inputs were provided")

    gpu_ids = _available_gpu_ids(max_gpus)
    episode_rows: list[dict[str, Any]] = []
    global_index = 1
    for request_path in request_paths:
        if not request_path.exists():
            raise FileNotFoundError(f"Minimal YAML does not exist: {request_path}")
        tmp_dir = output_dir / "_per_request" / request_path.stem
        summary = materialize_episode_configs(
            request_path,
            tmp_dir,
            episode_count=episode_count_override,
            episode_start_index=global_index,
            source_name=request_path.stem,
        )
        generated = sorted((Path(summary["episodes_dir"])).glob("episode_*.yaml"))
        for yaml_path in generated:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            episode_id = f"episode_{global_index:06d}"
            data["episode_id"] = episode_id
            data["episode_index"] = global_index
            gpu_id = gpu_ids[(global_index - 1) % len(gpu_ids)]
            data["curriculum"] = {
                "global_episode_index": global_index,
                "gpu_id": gpu_id,
                "shard_index": gpu_ids.index(gpu_id),
                "num_shards": len(gpu_ids),
            }
            out = episodes_dir / f"{episode_id}.yaml"
            out.write_text(_safe_dump(data), encoding="utf-8")
            episode_rows.append(
                {
                    "episode_id": episode_id,
                    "episode_yaml": str(out),
                    "global_episode_index": global_index,
                    "gpu_id": gpu_id,
                    "source_request": request_path.name,
                    **data["task_context"],
                    "start_near_gate": bool((data.get("scene") or {}).get("start_near_gate")),
                }
            )
            global_index += 1

    fieldnames = [
        "episode_id",
        "episode_yaml",
        "global_episode_index",
        "gpu_id",
        "source_request",
        "task_family",
        "target_port_index",
        "target_card_index",
        "target_card_valid",
        "nic_card_count",
        "sc_port_count",
        "start_near_gate",
    ]
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(episode_rows)

    shard_summaries = []
    for gpu_id in gpu_ids:
        shard_episode_dir = shards_dir / f"gpu_{gpu_id}" / "episodes"
        shard_episode_dir.mkdir(parents=True, exist_ok=True)
        rows = [row for row in episode_rows if int(row["gpu_id"]) == gpu_id]
        for local_index, row in enumerate(rows, start=1):
            data = yaml.safe_load(Path(row["episode_yaml"]).read_text(encoding="utf-8"))
            data["curriculum"]["local_shard_episode_index"] = local_index
            (shard_episode_dir / f"episode_{local_index:06d}.yaml").write_text(
                _safe_dump(data),
                encoding="utf-8",
            )
        shard_manifest = shards_dir / f"gpu_{gpu_id}" / "manifest.csv"
        with shard_manifest.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        shard_summaries.append(
            {
                "gpu_id": gpu_id,
                "episodes_dir": str(shard_episode_dir),
                "manifest_csv": str(shard_manifest),
                "episode_count": len(rows),
            }
        )

    task_distribution_path = output_dir / "task_distribution.yaml"
    task_distribution_path.write_text(
        _safe_dump(
            {
                "task_family": sorted({row["task_family"] for row in episode_rows}),
                "episodes": [
                    {
                        key: row[key]
                        for key in (
                            "episode_id",
                            "task_family",
                            "target_port_index",
                            "target_card_index",
                            "target_card_valid",
                        )
                    }
                    for row in episode_rows
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = {
        "input_dir": str(input_dir),
        "input_yamls": [str(path) for path in request_paths],
        "output_dir": str(output_dir),
        "episodes_dir": str(episodes_dir),
        "manifest_csv": str(manifest_path),
        "task_distribution_yaml": str(task_distribution_path),
        "max_gpus": max_gpus,
        "gpu_ids": gpu_ids,
        "episode_count": len(episode_rows),
        "shards": shard_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-yaml", type=Path)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--filenames", default="")
    parser.add_argument("--input-yaml", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode-count", type=int, default=None)
    parser.add_argument("--max-gpus", type=int, default=1)
    parser.add_argument("--multi", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.multi or args.input_yaml or args.filenames or args.request_yaml is None:
        input_yamls = list(args.input_yaml)
        if args.request_yaml is not None:
            input_yamls.insert(0, args.request_yaml)
        summary = materialize_many_episode_configs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            filenames=_split_filenames(args.filenames),
            input_yamls=input_yamls,
            max_gpus=args.max_gpus,
            episode_count_override=args.episode_count,
        )
    else:
        summary = materialize_episode_configs(args.request_yaml, args.output_dir, episode_count=args.episode_count)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
