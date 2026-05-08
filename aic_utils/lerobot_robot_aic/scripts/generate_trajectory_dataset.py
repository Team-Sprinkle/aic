#!/usr/bin/env python3
"""Generate user-facing trajectory datasets with the LeRobot recorder pipeline."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_SCRIPT_DIR = REPO_ROOT / "aic_engine" / "scripts"
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(ENGINE_SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_random_trials_config import (  # noqa: E402
    PROFILE_QUALIFICATION_EVAL_LIKE,
    _build_trial,
    _profile_defaults,
)
from run_expert_setting_matrix import MODE_ENV as EXPERT_MODE_ENV  # noqa: E402

TASK_FAMILIES = {"sfp_to_nic", "sc_to_sc"}
POLICY_CLASS = {
    "cheatcode": "aic_example_policies.ros.CheatCode",
    "agent": "aic_teacher_official expert generator",
}
AGENT_EXPERT_MODES = {"nominal", "nominalrecovery", "recovery"}
NIC_RAILS = [f"nic_rail_{i}" for i in range(5)]
SC_RAILS = [f"sc_rail_{i}" for i in range(2)]
MOUNT_RAILS = [
    "lc_mount_rail_0",
    "sfp_mount_rail_0",
    "sc_mount_rail_0",
    "lc_mount_rail_1",
    "sfp_mount_rail_1",
    "sc_mount_rail_1",
]
MOUNT_ENTITY_NAMES = {
    "lc_mount_rail_0": "lc_mount_0",
    "sfp_mount_rail_0": "sfp_mount_0",
    "sc_mount_rail_0": "sc_mount_0",
    "lc_mount_rail_1": "lc_mount_1",
    "sfp_mount_rail_1": "sfp_mount_1",
    "sc_mount_rail_1": "sc_mount_1",
}
DEFAULT_START_NEAR_GATE_REFERENCE_TCP = (-0.4, 0.2, 0.3)
DEFAULT_ROBOT_WORLD_POSITION = (-0.2, 0.2, 1.14)
DEFAULT_ROBOT_WORLD_YAW = -3.141
NIC_RAIL_Y = {
    0: -0.1745,
    1: -0.1345,
    2: -0.0945,
    3: -0.0545,
    4: -0.0145,
}
SC_RAIL_Y = {0: 0.0295, 1: 0.0705}
SFP_PORT_LOCAL = {
    "sfp_port_0": (0.01295, -0.031572, 0.00501),
    "sfp_port_1": (-0.01025, -0.031572, 0.00501),
}
LIMITS = {
    "nic_translation": (-0.0215, 0.0234),
    "nic_yaw_deg": (-10.0, 10.0),
    "sc_translation": (-0.06, 0.055),
    "fixture_translation": (-0.09425, 0.09425),
    "fixture_yaw_deg": (-60.0, 60.0),
}
DEFAULT_TEMPLATE = REPO_ROOT / "aic_engine" / "config" / "sample_config.yaml"
DEFAULT_EXPERT_SETTING_REGISTRY = REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "config" / "expert_setting_registry.json"
DEFAULT_EXPERT_REGISTRY_OVERLAY_DIR = (
    REPO_ROOT / "aic_utils" / "lerobot_robot_aic" / "config" / "expert_setting_registry_overlays"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-yaml", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--num-trials-override", type=int, default=None)
    parser.add_argument("--target-accepted-override", type=int, default=None)
    parser.add_argument("--max-attempts-override", type=int, default=None)
    parser.add_argument("--skip-recording", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument(
        "--inspect-reference-dataset",
        default=None,
        help="Optional Hugging Face LeRobot reference dataset repo id.",
    )
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        request = yaml.safe_load(f)
    if not isinstance(request, dict):
        raise ValueError(f"Request YAML must be a map: {path}")
    return request


def require_path(data: dict[str, Any], dotted: str) -> Any:
    current: Any = data
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Missing required field: {dotted}")
        current = current[part]
    return current


def validate_request(request: dict[str, Any]) -> None:
    require_path(request, "root_dir")
    task_family = require_path(request, "task_family")
    if task_family not in TASK_FAMILIES:
        raise ValueError(f"task_family must be one of {sorted(TASK_FAMILIES)}")
    require_path(request, "generation.target_accepted_trajectories")
    require_path(request, "generation.max_attempts")
    policy = require_path(request, "generation.policy")
    if policy not in POLICY_CLASS:
        raise ValueError(f"Unsupported generation.policy '{policy}'. Supported: {sorted(POLICY_CLASS)}")
    expert_mode = effective_expert_mode(request)
    if policy == "agent" and expert_mode not in AGENT_EXPERT_MODES:
        raise ValueError(
            f"generation.expert_mode must be one of {sorted(AGENT_EXPERT_MODES)} for policy='agent'"
        )
    require_path(request, "acceptance.min_score")
    stop_near_gate = (request.get("acceptance") or {}).get("stop_near_gate")
    legacy_near_gate = (request.get("acceptance") or {}).get("near_gate")
    if stop_near_gate is not None and legacy_near_gate is not None:
        raise ValueError("Use acceptance.stop_near_gate or deprecated acceptance.near_gate, not both")
    start_near_gate = (request.get("scene") or {}).get("start_near_gate")
    if start_near_gate is not None:
        if not isinstance(start_near_gate, dict):
            raise ValueError("scene.start_near_gate must be a map")
        if "distance" in start_near_gate:
            distance = float(start_near_gate["distance"])
            if distance <= 0.0:
                raise ValueError("scene.start_near_gate.distance must be > 0")
        else:
            if "axial_distance_m" not in start_near_gate or "lateral_distance_m" not in start_near_gate:
                raise ValueError(
                    "scene.start_near_gate requires axial_distance_m and lateral_distance_m "
                    "when legacy distance is omitted"
                )
            axial_distance = float(start_near_gate["axial_distance_m"])
            lateral_distance = float(start_near_gate["lateral_distance_m"])
            if axial_distance < 0.0:
                raise ValueError("scene.start_near_gate.axial_distance_m must be >= 0")
            if lateral_distance < 0.0:
                raise ValueError("scene.start_near_gate.lateral_distance_m must be >= 0")
    if task_family == "sfp_to_nic":
        require_path(request, "scene.nic_cards.count")
    else:
        require_path(request, "scene.sc_ports.count")


def sample_value(spec: Any, default: Any, rng: random.Random) -> Any:
    if spec is None:
        if isinstance(default, (tuple, list)) and len(default) == 2:
            return rng.uniform(float(default[0]), float(default[1]))
        return default
    if isinstance(spec, list):
        if not spec:
            raise ValueError("List override must not be empty")
        return rng.choice(spec)
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        lo = float(spec["min"])
        hi = float(spec["max"])
        if lo > hi:
            raise ValueError(f"Invalid range override: min {lo} > max {hi}")
        return rng.uniform(lo, hi)
    return spec


def _as_count(value: Any, field: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{field} must resolve to an integer count, got {value!r}")
    if value < 0:
        raise ValueError(f"{field} must be >= 0")
    return value


def _count_label(task_family: str, request: dict[str, Any]) -> str:
    if task_family == "sfp_to_nic":
        spec = request["scene"]["nic_cards"]["count"]
        label = spec if isinstance(spec, int) else "mixed"
        return f"nic_cards_{label}"
    spec = request["scene"]["sc_ports"]["count"]
    label = spec if isinstance(spec, int) else "mixed"
    return f"sc_ports_{label}"


def derive_output_dir(request: dict[str, Any]) -> Path:
    policy = request["generation"]["policy"]
    target = int(request["generation"]["target_accepted_trajectories"])
    suffix = str(request.get("suffix", "dataset"))
    return (
        Path(request["root_dir"])
        / request["task_family"]
        / policy
        / _count_label(request["task_family"], request)
        / f"n{target}__{suffix}"
    )


def derived_dataset_name(output_dir: Path) -> str:
    parts = output_dir.parts[-5:]
    return "__".join(p.replace("/", "_") for p in parts)


def _validate_spec_range(spec: Any, lo: float, hi: float, field: str) -> None:
    if spec is None:
        return
    vals: list[float]
    if isinstance(spec, dict) and "min" in spec and "max" in spec:
        vals = [float(spec["min"]), float(spec["max"])]
    elif isinstance(spec, list):
        vals = [float(v) for v in spec]
    else:
        vals = [float(spec)]
    bad = [v for v in vals if v < lo or v > hi]
    if bad:
        raise ValueError(f"{field} override {bad[0]} is outside official limits [{lo}, {hi}]")


def validate_override_limits(request: dict[str, Any]) -> None:
    scene = request.get("scene", {})
    nic = scene.get("nic_cards", {}) if isinstance(scene.get("nic_cards", {}), dict) else {}
    sc = scene.get("sc_ports", {}) if isinstance(scene.get("sc_ports", {}), dict) else {}
    mounts = scene.get("fixture_mounts", {}) if isinstance(scene.get("fixture_mounts", {}), dict) else {}
    _validate_spec_range(nic.get("translation"), *LIMITS["nic_translation"], "scene.nic_cards.translation")
    _validate_spec_range(nic.get("yaw_deg"), *LIMITS["nic_yaw_deg"], "scene.nic_cards.yaw_deg")
    _validate_spec_range(sc.get("translation"), *LIMITS["sc_translation"], "scene.sc_ports.translation")
    _validate_spec_range(mounts.get("translation"), *LIMITS["fixture_translation"], "scene.fixture_mounts.translation")
    _validate_spec_range(mounts.get("yaw_deg"), *LIMITS["fixture_yaw_deg"], "scene.fixture_mounts.yaw_deg")


def _deg_field(section: dict[str, Any], name: str, default_rad: Any, rng: random.Random) -> float:
    spec = section.get(f"{name}_deg")
    if spec is None:
        return round(float(sample_value(None, default_rad, rng)), 5)
    return round(math.radians(float(sample_value(spec, None, rng))), 5)


def _range_or_fixed(value: Any) -> tuple[float, float] | float:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        if float(value[0]) == float(value[1]):
            return float(value[0])
        return (float(value[0]), float(value[1]))
    return float(value)


def _rot_x(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _rot_y(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rot_z(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    return _matmul(_matmul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


def _matvec(m: list[list[float]], v: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    return tuple(sum(m[i][j] * float(v[j]) for j in range(3)) for i in range(3))


def _vadd(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vsub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vscale(a: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (a[0] * s, a[1] * s, a[2] * s)


def _vnorm(a: tuple[float, float, float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _transform_point(
    translation: tuple[float, float, float],
    rpy: tuple[float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return _vadd(translation, _matvec(_rpy_matrix(*rpy), point))


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _vnorm(v)
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero vector")
    return _vscale(v, 1.0 / norm)


def _world_to_base_point(point_world: tuple[float, float, float]) -> tuple[float, float, float]:
    return _matvec(
        _rot_z(-DEFAULT_ROBOT_WORLD_YAW),
        _vsub(point_world, DEFAULT_ROBOT_WORLD_POSITION),
    )


def _world_to_base_vector(vector_world: tuple[float, float, float]) -> tuple[float, float, float]:
    return _matvec(_rot_z(-DEFAULT_ROBOT_WORLD_YAW), vector_world)


def _base_to_world_vector(vector_base: tuple[float, float, float]) -> tuple[float, float, float]:
    return _matvec(_rot_z(DEFAULT_ROBOT_WORLD_YAW), vector_base)


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _random_perpendicular_unit(
    axis: tuple[float, float, float],
    rng: random.Random,
) -> tuple[float, float, float]:
    axis = _normalize(axis)
    seed = (0.0, 0.0, 1.0)
    if abs(axis[2]) > 0.9:
        seed = (1.0, 0.0, 0.0)
    basis_a = _normalize(_cross(axis, seed))
    basis_b = _cross(axis, basis_a)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return _vadd(_vscale(basis_a, math.cos(angle)), _vscale(basis_b, math.sin(angle)))


def _sample_pose_on_rail(
    section: dict[str, Any],
    defaults: dict[str, Any],
    translation_default: tuple[float, float],
    rng: random.Random,
) -> dict[str, float]:
    return {
        "translation": round(float(sample_value(section.get("translation"), translation_default, rng)), 5),
        "roll": _deg_field(section, "roll", defaults["roll"], rng),
        "pitch": _deg_field(section, "pitch", defaults["pitch"], rng),
        "yaw": _deg_field(section, "yaw", defaults["yaw"], rng),
    }


def _selected_rails(section: dict[str, Any], all_rails: list[str], count: int, rng: random.Random) -> list[str]:
    rails_spec = section.get("rails", all_rails)
    rails = list(rails_spec)
    unknown = sorted(set(rails) - set(all_rails))
    if unknown:
        raise ValueError(f"Unknown rails in request: {unknown}")
    if count > len(rails):
        raise ValueError(f"Requested count {count} exceeds available rails {rails}")
    return sorted(rng.sample(rails, count))


def _target_index(target_spec: Any, present_rails: list[str], prefix: str, rng: random.Random) -> int:
    if target_spec in (None, "auto"):
        rail = rng.choice(present_rails)
    elif isinstance(target_spec, int):
        rail = f"{prefix}_{target_spec}"
    else:
        text = str(target_spec)
        rail = text if text.startswith(prefix) else f"{prefix}_{text.split('_')[-1]}"
    if rail not in present_rails:
        raise ValueError(f"Target {target_spec!r} must be among present rails {present_rails}")
    return int(rail.rsplit("_", 1)[1])


def _apply_board_overrides(trial: dict[str, Any], section: dict[str, Any], rng: random.Random) -> None:
    if not section:
        return
    pose = trial["scene"]["task_board"]["pose"]
    for key in ("x", "y", "z"):
        if key in section:
            pose[key] = round(float(sample_value(section[key], pose[key], rng)), 5)
    for key in ("roll", "pitch", "yaw"):
        deg_key = f"{key}_deg"
        if deg_key in section:
            pose[key] = round(math.radians(float(sample_value(section[deg_key], None, rng))), 5)


def _apply_nic_overrides(
    trial: dict[str, Any],
    section: dict[str, Any],
    required_exact: bool,
    profile_cfg: dict[str, Any],
    rng: random.Random,
) -> int | None:
    if "count" not in section and not required_exact:
        return None
    count = _as_count(sample_value(section.get("count"), None, rng), "scene.nic_cards.count")
    present = _selected_rails(section, NIC_RAILS, count, rng)
    nic_defaults = profile_cfg["nic_pose"]
    for rail in NIC_RAILS:
        if rail not in present:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        idx = int(rail.rsplit("_", 1)[1])
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"nic_card_{idx}",
            "entity_pose": _sample_pose_on_rail(section, nic_defaults, LIMITS["nic_translation"], rng),
        }
    if not present:
        return None
    return _target_index(section.get("target_card"), present, "nic_rail", rng)


def _apply_sc_overrides(
    trial: dict[str, Any],
    section: dict[str, Any],
    required_exact: bool,
    profile_cfg: dict[str, Any],
    rng: random.Random,
) -> int | None:
    if "count" not in section and not required_exact:
        return None
    count = _as_count(sample_value(section.get("count"), None, rng), "scene.sc_ports.count")
    present = _selected_rails(section, SC_RAILS, count, rng)
    sc_defaults = profile_cfg["sc_pose"]
    for rail in SC_RAILS:
        if rail not in present:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        idx = int(rail.rsplit("_", 1)[1])
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": f"sc_mount_{idx}",
            "entity_pose": _sample_pose_on_rail(section, sc_defaults, LIMITS["sc_translation"], rng),
        }
    if not present:
        return None
    return _target_index(section.get("target_port"), present, "sc_rail", rng)


def _apply_fixture_mount_overrides(
    trial: dict[str, Any], section: dict[str, Any], profile_cfg: dict[str, Any], rng: random.Random
) -> None:
    if not section:
        return
    rails = list(section.get("rails", MOUNT_RAILS))
    unknown = sorted(set(rails) - set(MOUNT_RAILS))
    if unknown:
        raise ValueError(f"Unknown fixture_mounts rails in request: {unknown}")
    present_prob = float(sample_value(section.get("present_probability"), profile_cfg["mount_pose"]["present_prob"], rng))
    for rail in MOUNT_RAILS:
        if rail not in rails:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        if rng.random() > present_prob:
            trial["scene"]["task_board"][rail] = {"entity_present": False}
            continue
        trial["scene"]["task_board"][rail] = {
            "entity_present": True,
            "entity_name": MOUNT_ENTITY_NAMES[rail],
            "entity_pose": _sample_pose_on_rail(section, profile_cfg["mount_pose"], LIMITS["fixture_translation"], rng),
        }


def _apply_family_task_and_cable(
    trial: dict[str, Any],
    task_family: str,
    target_nic: int | None,
    target_sc: int | None,
    scene: dict[str, Any],
    rng: random.Random,
) -> None:
    cable_section = scene.get("cable", {}) if isinstance(scene.get("cable", {}), dict) else {}
    if task_family == "sfp_to_nic":
        if target_nic is None:
            raise ValueError("sfp_to_nic requires at least one present NIC card")
        cable_name = "cable_0"
        cable_type = str(sample_value(cable_section.get("cable_type"), "sfp_sc_cable", rng))
        port_name = str(sample_value(scene.get("nic_cards", {}).get("target_port"), "auto", rng))
        if port_name == "auto":
            port_name = rng.choice(["sfp_port_0", "sfp_port_1"])
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sfp",
            "plug_name": "sfp_tip",
            "port_type": "sfp",
            "port_name": port_name,
            "target_module_name": f"nic_card_mount_{target_nic}",
            "time_limit": 180,
        }
        default_offset = {"x": 0.0, "y": 0.015385, "z": 0.04245}
    else:
        if target_sc is None:
            raise ValueError("sc_to_sc requires at least one present SC port")
        cable_name = "cable_1"
        cable_type = str(sample_value(cable_section.get("cable_type"), "sfp_sc_cable_reversed", rng))
        task = {
            "cable_type": "sfp_sc",
            "cable_name": cable_name,
            "plug_type": "sc",
            "plug_name": "sc_tip",
            "port_type": "sc",
            "port_name": "sc_port_base",
            "target_module_name": f"sc_port_{target_sc}",
            "time_limit": 180,
        }
        default_offset = {"x": 0.0, "y": 0.015385, "z": 0.04045}

    existing_pose = next(iter(trial["scene"]["cables"].values()))["pose"]
    offset_section = cable_section.get("gripper_offset", {})
    pose = {
        "gripper_offset": {
            axis: round(float(sample_value(offset_section.get(axis), default_offset[axis], rng)), 5)
            for axis in ("x", "y", "z")
        },
        "roll": round(math.radians(float(sample_value(cable_section.get("roll_deg"), math.degrees(existing_pose["roll"]), rng))), 5),
        "pitch": round(math.radians(float(sample_value(cable_section.get("pitch_deg"), math.degrees(existing_pose["pitch"]), rng))), 5),
        "yaw": round(math.radians(float(sample_value(cable_section.get("yaw_deg"), math.degrees(existing_pose["yaw"]), rng))), 5),
    }
    trial["scene"]["cables"] = {
        cable_name: {
            "pose": pose,
            "attach_cable_to_gripper": True,
            "cable_type": cable_type,
        }
    }
    trial["tasks"] = {"task_1": task}


def _target_gate_position_in_board_frame(trial: dict[str, Any]) -> tuple[float, float, float]:
    return _target_gate_frame_in_board_frame(trial)[0]


def _target_gate_frame_in_board_frame(
    trial: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    task = trial["tasks"]["task_1"]
    board = trial["scene"]["task_board"]
    if task["port_type"] == "sfp":
        target_idx = int(str(task["target_module_name"]).rsplit("_", 1)[1])
        rail = board[f"nic_rail_{target_idx}"]
        pose = rail["entity_pose"]
        mount_translation = (
            -0.081418 + float(pose["translation"]),
            NIC_RAIL_Y[target_idx],
            0.012,
        )
        port_translation = SFP_PORT_LOCAL[str(task["port_name"])]
        entrance_translation = (0.0, 0.0, -0.0458)
        mount_rpy = (float(pose["roll"]), float(pose["pitch"]), float(pose["yaw"]))
        port_rpy = (4.69895, 0.0, 0.0)
        point = _transform_point(port_translation, port_rpy, entrance_translation)
        position = _transform_point(mount_translation, mount_rpy, point)
        rotation = _matmul(_rpy_matrix(*mount_rpy), _rpy_matrix(*port_rpy))
        axis = _matvec(rotation, (0.0, 0.0, 1.0))
        return position, _normalize(axis)

    target_idx = int(str(task["target_module_name"]).rsplit("_", 1)[1])
    rail = board[f"sc_rail_{target_idx}"]
    pose = rail["entity_pose"]
    sc_translation = (
        -0.075 + float(pose["translation"]),
        SC_RAIL_Y[target_idx],
        0.0165,
    )
    sc_rpy = (
        1.57 + float(pose["roll"]),
        float(pose["pitch"]),
        1.57 + float(pose["yaw"]),
    )
    base_translation = (0.0, -0.002, 0.0)
    base_rpy = (1.5708, 3.14159, 0.0)
    entrance_translation = (0.0, 0.0, -0.01564)
    point = _transform_point(base_translation, base_rpy, entrance_translation)
    position = _transform_point(sc_translation, sc_rpy, point)
    rotation = _matmul(_rpy_matrix(*sc_rpy), _rpy_matrix(*base_rpy))
    axis = _matvec(rotation, (0.0, 0.0, 1.0))
    return position, _normalize(axis)


def _target_gate_position_world(trial: dict[str, Any]) -> tuple[float, float, float]:
    return _target_gate_frame_world(trial)[0]


def _target_gate_frame_world(
    trial: dict[str, Any],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    board_pose = trial["scene"]["task_board"]["pose"]
    board_translation = (
        float(board_pose["x"]),
        float(board_pose["y"]),
        float(board_pose["z"]),
    )
    board_rpy = (
        float(board_pose["roll"]),
        float(board_pose["pitch"]),
        float(board_pose["yaw"]),
    )
    position_board, axis_board = _target_gate_frame_in_board_frame(trial)
    board_rotation = _rpy_matrix(*board_rpy)
    position_world = _vadd(board_translation, _matvec(board_rotation, position_board))
    axis_world = _normalize(_matvec(board_rotation, axis_board))
    return _world_to_base_point(position_world), _normalize(_world_to_base_vector(axis_world))


def _apply_start_near_gate(
    trial: dict[str, Any],
    section: dict[str, Any] | None,
    rng: random.Random,
) -> None:
    if not section:
        return
    min_clearance = float(section.get("min_clearance_m", 0.02))
    reference = section.get("reference_tcp_position", DEFAULT_START_NEAR_GATE_REFERENCE_TCP)
    if not isinstance(reference, (list, tuple)) or len(reference) != 3:
        raise ValueError("scene.start_near_gate.reference_tcp_position must be a 3-value list when provided")
    reference_tcp = (float(reference[0]), float(reference[1]), float(reference[2]))
    current_target, gate_axis_world = _target_gate_frame_world(trial)
    metadata: dict[str, Any]
    if "distance" in section:
        distance = float(section["distance"])
        if distance < min_clearance:
            raise ValueError(
                "scene.start_near_gate.distance must be at least min_clearance_m "
                f"({distance} < {min_clearance})"
            )
        direction = _vsub(current_target, reference_tcp)
        norm = _vnorm(direction)
        if norm < 1e-6:
            direction = _random_perpendicular_unit(gate_axis_world, rng)
            norm = 1.0
        desired_target = _vadd(reference_tcp, _vscale(direction, distance / norm))
        metadata = {
            "distance": distance,
            "achieved_distance": None,
        }
    else:
        axial_distance = float(section["axial_distance_m"])
        lateral_distance = float(section["lateral_distance_m"])
        total_distance = math.sqrt(axial_distance * axial_distance + lateral_distance * lateral_distance)
        if total_distance < min_clearance:
            raise ValueError(
                "scene.start_near_gate combined axial/lateral distance must be at least min_clearance_m "
                f"({total_distance} < {min_clearance})"
            )
        lateral_direction = _random_perpendicular_unit(gate_axis_world, rng)
        reference_offset = _vadd(
            _vscale(gate_axis_world, axial_distance),
            _vscale(lateral_direction, lateral_distance),
        )
        desired_target = _vsub(reference_tcp, reference_offset)
        metadata = {
            "axial_distance_m": axial_distance,
            "lateral_distance_m": lateral_distance,
            "lateral_direction_world": [round(v, 6) for v in lateral_direction],
        }
    delta = _base_to_world_vector(_vsub(desired_target, current_target))
    board_pose = trial["scene"]["task_board"]["pose"]
    axes = str(section.get("axes", "xyz")).lower()
    if axes not in {"xyz", "xy"}:
        raise ValueError("scene.start_near_gate.axes must be 'xyz' or 'xy'")
    board_pose["x"] = round(float(board_pose["x"]) + delta[0], 5)
    board_pose["y"] = round(float(board_pose["y"]) + delta[1], 5)
    if axes == "xyz":
        board_pose["z"] = round(float(board_pose["z"]) + delta[2], 5)
    achieved, achieved_axis = _target_gate_frame_world(trial)
    reference_delta = _vsub(reference_tcp, achieved)
    axial_component = abs(sum(reference_delta[i] * achieved_axis[i] for i in range(3)))
    lateral_component = math.sqrt(max(0.0, _vnorm(reference_delta) ** 2 - axial_component * axial_component))
    metadata.update(
        {
            "reference_tcp_position": [round(v, 6) for v in reference_tcp],
            "target_gate_position": [round(v, 6) for v in achieved],
            "target_gate_axis_world": [round(v, 6) for v in achieved_axis],
            "achieved_distance": round(_vnorm(reference_delta), 6),
            "achieved_axial_distance_m": round(axial_component, 6),
            "achieved_lateral_distance_m": round(lateral_component, 6),
            "axes": axes,
            "min_clearance_m": min_clearance,
            "overlap_check": (
                "rigid task-board transform preserves board component clearances; "
                "distance must exceed min_clearance_m"
            ),
        }
    )
    trial.setdefault("generated_metadata", {})["start_near_gate"] = metadata


def generate_trials(request: dict[str, Any], num_trials: int) -> dict[str, Any]:
    base = yaml.safe_load(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    limits = copy.deepcopy(base.get("task_board_limits", {}))
    profile_cfg = _profile_defaults(PROFILE_QUALIFICATION_EVAL_LIKE)
    rng = random.Random(request.get("generation", {}).get("seed"))
    scene = request.get("scene", {})
    generated: dict[str, Any] = {}
    for idx in range(1, num_trials + 1):
        nic_section = scene.get("nic_cards", {})
        sc_section = scene.get("sc_ports", {})
        if request["task_family"] == "sfp_to_nic" and "sc_ports" not in scene:
            sc_section = {"count": 0}
        if request["task_family"] == "sc_to_sc" and "nic_cards" not in scene:
            nic_section = {"count": 0}
        raw = _build_trial(
            rng,
            idx,
            limits,
            profile_cfg=profile_cfg,
            sfp_to_nic_weight=1.0 if request["task_family"] == "sfp_to_nic" else 0.0,
            sc_to_sc_weight=1.0 if request["task_family"] == "sc_to_sc" else 0.0,
        )
        _apply_board_overrides(raw, scene.get("board", {}), rng)
        target_nic = _apply_nic_overrides(
            raw,
            nic_section,
            request["task_family"] == "sfp_to_nic",
            profile_cfg,
            rng,
        )
        target_sc = _apply_sc_overrides(
            raw,
            sc_section,
            request["task_family"] == "sc_to_sc",
            profile_cfg,
            rng,
        )
        _apply_fixture_mount_overrides(raw, scene.get("fixture_mounts", {}), profile_cfg, rng)
        _apply_family_task_and_cable(raw, request["task_family"], target_nic, target_sc, scene, rng)
        start_near_gate = scene.get("start_near_gate") if isinstance(scene, dict) else None
        _apply_start_near_gate(raw, start_near_gate if isinstance(start_near_gate, dict) else None, rng)
        generated[f"trial_{idx:06d}"] = raw
    return generated


def write_engine_configs(request: dict[str, Any], output_dir: Path, num_trials: int) -> Path:
    base = yaml.safe_load(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
    trials = generate_trials(request, num_trials)
    engine_config = copy.deepcopy(base)
    engine_config["trials"] = trials
    engine_config["generated"] = {
        "script": "aic_utils/lerobot_robot_aic/scripts/generate_trajectory_dataset.py",
        "seed": request.get("generation", {}).get("seed"),
        "task_family": request["task_family"],
        "num_trials": num_trials,
        "request_yaml_semantics": {
            "missing": "team default randomization from generate_random_trials_config.py",
            "list": "uniform categorical choice",
            "min_max": "continuous uniform range; min == max is fixed",
            "degrees": "request fields ending in _deg are converted to radians in engine_config.yaml",
        },
    }
    out_path = output_dir / "engine_config.yaml"
    out_path.write_text(yaml.safe_dump(engine_config, sort_keys=False), encoding="utf-8")
    trials_dir = output_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)
    for trial_id, trial in trials.items():
        single = copy.deepcopy(engine_config)
        single["trials"] = {trial_id: trial}
        (trials_dir / f"{trial_id}.yaml").write_text(
            yaml.safe_dump(single, sort_keys=False), encoding="utf-8"
        )
    return out_path


def run_command(cmd: list[str], dry_run: bool, env: dict[str, str] | None = None) -> dict[str, Any]:
    rendered = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {rendered}")
        return {"cmd": cmd, "skipped": True, "returncode": None}
    print(f"[run] {rendered}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {rendered}")
    return {"cmd": cmd, "skipped": False, "returncode": result.returncode}


def run_command_allowing_codes(
    cmd: list[str],
    dry_run: bool,
    allowed: set[int],
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    rendered = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {rendered}")
        return {"cmd": cmd, "skipped": True, "returncode": None}
    print(f"[run] {rendered}")
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, env=env)
    if result.returncode not in allowed:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {rendered}")
    return {"cmd": cmd, "skipped": False, "returncode": result.returncode}


def count_selected(selection_report: Path) -> int | None:
    if not selection_report.exists():
        return None
    with selection_report.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for row in csv.DictReader(f) if str(row.get("selected", "")).lower() == "true")


def compare_reference(local_dataset: Path, reference_repo_id: str) -> dict[str, Any]:
    if not (local_dataset / "meta" / "info.json").exists():
        return {"status": "skipped", "reason": f"missing {local_dataset / 'meta' / 'info.json'}"}
    cmd = [
        "pixi",
        "run",
        "aic-validate-dataset-compat",
        f"--base.repo_id={reference_repo_id}",
        f"--candidate.repo_id={local_dataset.name}",
        f"--candidate.root={local_dataset}",
        "--json",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    return {
        "status": "succeeded" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "cmd": cmd,
    }


def _bool_string(value: Any) -> str:
    return str(bool(value)).lower()


def _agent_score_csv_for_dataset(
    *,
    output_dir: Path,
    dataset_root: Path,
    record: dict[str, Any],
    index: int,
) -> Path:
    score_dir = output_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    path = score_dir / f"agent_attempt_{index:06d}_score_summary.csv"
    validation = record.get("validation") or {}
    replay_metrics = record.get("replay_metrics") or {}
    score = validation.get("score", replay_metrics.get("score"))
    if score is None:
        score = 0.0
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id", "run_index", "status", "total_score"])
        writer.writeheader()
        writer.writerow(
            {
                "trial_id": dataset_root.parent.name,
                "run_index": 1,
                "status": "OK",
                "total_score": float(score),
            }
        )
    return path


def _accepted_agent_datasets(agent_output_dir: Path, output_dir: Path) -> tuple[list[Path], list[Path], int]:
    summary_path = agent_output_dir / "generation_summary.json"
    if not summary_path.exists():
        return [], [], 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    datasets: list[Path] = []
    score_csvs: list[Path] = []
    for record in summary.get("records", []):
        if not record.get("accepted"):
            continue
        replay_metrics = record.get("replay_metrics") or {}
        trajectory_path = replay_metrics.get("trajectory_path")
        if not trajectory_path:
            continue
        dataset_root = Path(trajectory_path).parent / "dataset"
        if not (dataset_root / "meta" / "info.json").exists():
            continue
        datasets.append(dataset_root)
        score_csvs.append(
            _agent_score_csv_for_dataset(
                output_dir=output_dir,
                dataset_root=dataset_root,
                record=record,
                index=len(datasets),
            )
        )
    return datasets, score_csvs, int(summary.get("accepted", len(datasets)))


def agent_filter_min_score(request: dict[str, Any]) -> float:
    return float(request["acceptance"]["min_score"])


def stop_near_gate_config(request: dict[str, Any]) -> dict[str, Any]:
    acceptance = request.get("acceptance") or {}
    config = acceptance.get("stop_near_gate")
    if config is None:
        config = acceptance.get("near_gate", {})
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError("acceptance.stop_near_gate must be a map")
    return config


def has_stop_near_gate_config(request: dict[str, Any]) -> bool:
    acceptance = request.get("acceptance") or {}
    return acceptance.get("stop_near_gate") is not None or acceptance.get("near_gate") is not None


def _rail_indices(task_board: dict[str, Any], rails: list[str]) -> list[int]:
    indices: list[int] = []
    for rail in rails:
        if task_board.get(rail, {}).get("entity_present"):
            indices.append(int(rail.rsplit("_", 1)[1]))
    return indices


def _parse_index_from_suffix(text: str, prefix: str) -> int:
    if not text.startswith(prefix):
        raise ValueError(f"Expected {text!r} to start with {prefix!r}")
    return int(text[len(prefix) :])


def infer_registry_suffix_from_trial(task_family: str, trial: dict[str, Any]) -> str | None:
    task = trial.get("tasks", {}).get("task_1", {})
    task_board = trial.get("scene", {}).get("task_board", {})
    if task_family == "sfp_to_nic":
        present = _rail_indices(task_board, NIC_RAILS)
        if not present:
            return None
        target = _parse_index_from_suffix(str(task.get("target_module_name", "")), "nic_card_mount_")
        port = _parse_index_from_suffix(str(task.get("port_name", "")), "sfp_port_")
        present_label = "".join(str(idx) for idx in present)
        return f"matrix_sfp2nic_cards{len(present)}_present{present_label}_target{target}_port{port}"
    if task_family == "sc_to_sc":
        present = _rail_indices(task_board, SC_RAILS)
        if not present:
            return None
        target = _parse_index_from_suffix(str(task.get("target_module_name", "")), "sc_port_")
        nic_count = len(_rail_indices(task_board, NIC_RAILS))
        present_label = "".join(str(idx) for idx in present)
        return f"matrix_sc2sc_sc{len(present)}_present{present_label}_target{target}_nic{nic_count}"
    return None


def infer_registry_suffixes_from_engine_config(engine_config_path: Path, task_family: str) -> list[str]:
    config = yaml.safe_load(engine_config_path.read_text(encoding="utf-8"))
    trials = config.get("trials") if isinstance(config, dict) else None
    if not isinstance(trials, dict):
        return []
    suffixes: list[str] = []
    for trial in trials.values():
        if not isinstance(trial, dict):
            continue
        suffix = infer_registry_suffix_from_trial(task_family, trial)
        if suffix:
            suffixes.append(suffix)
    return suffixes


def effective_registry_suffix(request: dict[str, Any]) -> str:
    generation = request.get("generation", {})
    inferred = generation.get("_inferred_expert_registry_suffix")
    if inferred:
        return str(inferred)
    explicit = generation.get("expert_registry_suffix")
    if explicit:
        return str(explicit)
    return str(request.get("suffix", ""))


def effective_expert_mode(request: dict[str, Any]) -> str:
    return str(request.get("generation", {}).get("expert_mode", "nominal"))


def effective_max_agent_attempts(request: dict[str, Any], max_attempts: int) -> int:
    """Return the planner/replay attempt budget for agent generation.

    generation.max_attempts remains the historical overall attempt budget. The
    optional generation.max_planner_attempts field is a stricter retry budget
    for full planner+replay attempts. generation.max_replay_attempts is kept as
    a backwards-compatible alias for older request files.
    """
    generation = request.get("generation", {})
    configured = generation.get("max_planner_attempts", generation.get("max_replay_attempts"))
    if configured is None:
        return max_attempts
    agent_attempts = int(configured)
    if agent_attempts <= 0:
        raise ValueError("generation.max_planner_attempts must be > 0")
    return min(max_attempts, agent_attempts)


def effective_max_replay_attempts(request: dict[str, Any], max_attempts: int) -> int:
    return effective_max_agent_attempts(request, max_attempts)


def build_agent_generation_cmd(
    *,
    request: dict[str, Any],
    engine_config_path: Path,
    output_dir: Path,
    target: int,
    max_attempts: int,
) -> list[str]:
    generation = request.get("generation", {})
    expert_mode = effective_expert_mode(request)
    cmd = [
        "pixi",
        "run",
        "python",
        str(REPO_ROOT / "scripts/generate_expert_trajectories.py"),
        f"--{expert_mode}",
        "--target-accepted-trajectories",
        str(target),
        "--max-total-attempts",
        str(max_attempts),
        "--candidates-per-scene",
        str(int(generation.get("candidates_per_scene", min(max_attempts, 5)))),
        "--score-threshold",
        str(float(request["acceptance"]["min_score"])),
        "--require-insertion-event",
        _bool_string(request.get("acceptance", {}).get("success_only", True)),
        "--config",
        str(engine_config_path),
        "--output-dir",
        str(output_dir / "agent_generation"),
        "--seed",
        str(int(generation.get("seed", 0) or 0)),
        "--use-gpt5-analysis",
        _bool_string(generation.get("use_gpt5_analysis", generation.get("auto_improve_on_failure", False))),
        "--per-trial-timeout-sec",
        str(int(generation.get("per_trial_timeout_sec", 360))),
        "--ft-threshold",
        str(float(generation.get("ft_threshold", 15.0))),
    ]
    if bool(generation.get("debug", False)) or bool(generation.get("auto_improve_on_failure", False)):
        cmd.append("--debug")
    if "strategy_model" in generation:
        cmd.extend(["--strategy-model", str(generation["strategy_model"])])
    if "analysis_model" in generation:
        cmd.extend(["--analysis-model", str(generation["analysis_model"])])
    if "max_retries" in generation:
        cmd.extend(["--max-retries", str(int(generation["max_retries"]))])
    if "backup_distance_m" in generation:
        cmd.extend(["--backup-distance-m", str(float(generation["backup_distance_m"]))])
    near_gate = stop_near_gate_config(request)
    if has_stop_near_gate_config(request):
        cmd.extend(["--allow-near-gate-acceptance", "true"])
        generation.setdefault("env", {})
        if not isinstance(generation["env"], dict):
            raise ValueError("generation.env must be a map of environment variable names to values")
        generation["env"].setdefault("AIC_OFFICIAL_TEACHER_STOP_AT_NEAR_GATE", "true")
        if "max_lateral_error_m" in near_gate:
            cmd.extend(["--near-gate-max-lateral-error-m", str(float(near_gate["max_lateral_error_m"]))])
            max_lateral_error = str(float(near_gate["max_lateral_error_m"]))
            generation["env"]["AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_LATERAL_ERROR_M"] = max_lateral_error
            generation["env"]["AIC_OFFICIAL_TEACHER_TRACKING_GATE_MAX_LATERAL_ERROR_M"] = max_lateral_error
            if request.get("task_family") == "sc_to_sc":
                generation["env"]["AIC_OFFICIAL_TEACHER_SC_TRACKING_GATE_MAX_LATERAL_ERROR_M"] = max_lateral_error
        if "max_axial_error_m" in near_gate:
            max_axial_error = str(float(near_gate["max_axial_error_m"]))
            cmd.extend(["--near-gate-max-axial-error-m", max_axial_error])
            generation["env"]["AIC_OFFICIAL_TEACHER_NEAR_GATE_MAX_AXIAL_ERROR_M"] = max_axial_error
        if "max_tcp_speed_mps" in near_gate:
            cmd.extend(["--near-gate-max-tcp-speed-mps", str(float(near_gate["max_tcp_speed_mps"]))])
        if "max_force_delta_n" in near_gate:
            cmd.extend(["--near-gate-max-force-delta-n", str(float(near_gate["max_force_delta_n"]))])
        if "max_force_n" in near_gate:
            cmd.extend(["--near-gate-max-force-n", str(float(near_gate["max_force_n"]))])
    recorder_drain_sec = int(generation.get("recorder_drain_sec", 120))
    planner_recorder_drain_sec = int(generation.get("planner_recorder_drain_sec", 45))
    if not bool(generation.get("allow_short_agent_drain", False)):
        recorder_drain_sec = max(recorder_drain_sec, 45)
        planner_recorder_drain_sec = max(planner_recorder_drain_sec, 45)
    cmd.extend(["--recorder-drain-sec", str(recorder_drain_sec)])
    cmd.extend(["--planner-recorder-drain-sec", str(planner_recorder_drain_sec)])
    if "startup_delay_sec" in generation:
        cmd.extend(["--startup-delay-sec", str(int(generation["startup_delay_sec"]))])
    return cmd


def _overlay_paths(generation: dict[str, Any]) -> list[Path]:
    overlay_dir = Path(generation.get("expert_registry_overlay_dir", DEFAULT_EXPERT_REGISTRY_OVERLAY_DIR))
    if not overlay_dir.exists():
        return []
    return sorted(overlay_dir.glob("*.jsonl"))


def _merge_registry_overlay_entry(registry: dict[str, Any], entry: dict[str, Any]) -> None:
    suffix = str(entry.get("suffix") or "")
    mode = str(entry.get("mode") or "")
    if not suffix or mode not in AGENT_EXPERT_MODES:
        return
    setting = (registry.get("settings") or {}).get(suffix)
    if not isinstance(setting, dict):
        return
    modes = setting.setdefault("modes", {})
    mode_entry = modes.setdefault(mode, {})
    score = entry.get("score")
    status = str(entry.get("status") or mode_entry.get("status") or "attempted_not_passing")
    previous_status = str(mode_entry.get("status") or "")
    if status == "passed":
        mode_entry["status"] = "passed"
    elif status == "near_gate_passed":
        if previous_status != "passed":
            mode_entry["status"] = "near_gate_passed"
    elif previous_status not in {"passed", "near_gate_passed"}:
        mode_entry["status"] = status
    mode_entry["last_summary"] = entry.get("summary") or mode_entry.get("last_summary")
    mode_entry["last_reason"] = entry.get("reason") or mode_entry.get("last_reason")
    if isinstance(entry.get("mode_env"), dict):
        mode_entry["last_mode_env"] = {str(key): str(value) for key, value in entry["mode_env"].items()}
    if score is not None and status != "near_gate_passed":
        score_f = float(score)
        if mode_entry.get("best_score") is None or score_f > float(mode_entry.get("best_score")):
            mode_entry["best_score"] = score_f
            mode_entry["best_summary"] = entry.get("summary")
            if isinstance(entry.get("mode_env"), dict):
                mode_entry["best_mode_env"] = {str(key): str(value) for key, value in entry["mode_env"].items()}
    if status == "passed" and isinstance(entry.get("mode_env"), dict):
        mode_entry["best_mode_env"] = {str(key): str(value) for key, value in entry["mode_env"].items()}
    if status == "near_gate_passed" and previous_status != "passed" and isinstance(entry.get("mode_env"), dict):
        mode_entry["best_mode_env"] = {str(key): str(value) for key, value in entry["mode_env"].items()}
        mode_entry["best_summary"] = entry.get("summary") or mode_entry.get("best_summary")
    mode_entry.setdefault("history", []).append(entry)


def load_expert_registry(generation: dict[str, Any]) -> dict[str, Any] | None:
    registry_path = Path(generation.get("expert_setting_registry", DEFAULT_EXPERT_SETTING_REGISTRY))
    if not registry_path.exists():
        return None
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(registry, dict):
        return None
    for overlay_path in _overlay_paths(generation):
        for line in overlay_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                _merge_registry_overlay_entry(registry, entry)
    return registry


def expert_registry_mode_entry(request: dict[str, Any]) -> dict[str, Any] | None:
    generation = request.get("generation", {})
    if request.get("generation", {}).get("policy") != "agent":
        return None
    registry = load_expert_registry(generation)
    if registry is None:
        return None
    suffix = effective_registry_suffix(request)
    mode = effective_expert_mode(request)
    setting = (registry.get("settings") or {}).get(suffix)
    if not isinstance(setting, dict):
        return None
    mode_entry = (setting.get("modes") or {}).get(mode)
    return mode_entry if isinstance(mode_entry, dict) else None


def _expert_registry_mode_env_from_entry(mode_entry: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(mode_entry, dict) or mode_entry.get("status") not in {"passed", "near_gate_passed"}:
        return {}
    env = mode_entry.get("best_mode_env")
    if not isinstance(env, dict):
        return {}
    return {str(key): str(value) for key, value in env.items()}


def expert_registry_mode_env_for_suffix(request: dict[str, Any], suffix: str) -> dict[str, str]:
    generation = request.get("generation", {})
    if bool(generation.get("ignore_expert_setting_registry", False)):
        return {}
    if not bool(generation.get("use_expert_registry_env", True)):
        return {}
    if request.get("generation", {}).get("policy") != "agent":
        return {}
    registry = load_expert_registry(generation)
    if registry is None:
        return {}
    setting = (registry.get("settings") or {}).get(suffix)
    if not isinstance(setting, dict):
        return {}
    mode_entry = (setting.get("modes") or {}).get(effective_expert_mode(request))
    return _expert_registry_mode_env_from_entry(mode_entry if isinstance(mode_entry, dict) else None)


def expert_registry_mode_env(request: dict[str, Any]) -> dict[str, str]:
    generation = request.get("generation", {})
    if bool(generation.get("ignore_expert_setting_registry", False)):
        return {}
    if not bool(generation.get("use_expert_registry_env", True)):
        return {}
    return _expert_registry_mode_env_from_entry(expert_registry_mode_entry(request))


def expert_registry_mode_env_by_suffix(request: dict[str, Any], suffixes: list[str]) -> dict[str, dict[str, str]]:
    yaml_env = request.get("generation", {}).get("env", {})
    if yaml_env is not None and not isinstance(yaml_env, dict):
        raise ValueError("generation.env must be a map of environment variable names to values")
    yaml_env_str = {str(key): str(value) for key, value in yaml_env.items()} if isinstance(yaml_env, dict) else {}
    env_by_suffix: dict[str, dict[str, str]] = {}
    for suffix in sorted(set(suffixes)):
        suffix_env = expert_registry_mode_env_for_suffix(request, suffix)
        if suffix_env:
            suffix_env.update(yaml_env_str)
            env_by_suffix[suffix] = suffix_env
    return env_by_suffix


def agent_mode_env_for_registry_suffix(request: dict[str, Any], suffix: str | None = None) -> dict[str, str]:
    generation = request.get("generation", {})
    expert_mode = effective_expert_mode(request)
    if expert_mode not in EXPERT_MODE_ENV:
        raise ValueError(f"No expert generation environment preset exists for mode '{expert_mode}'")
    mode_env = dict(EXPERT_MODE_ENV[expert_mode])
    if suffix:
        mode_env.update(expert_registry_mode_env_for_suffix(request, suffix))
    else:
        mode_env.update(expert_registry_mode_env(request))
    yaml_env = generation.get("env", {})
    if yaml_env is not None and not isinstance(yaml_env, dict):
        raise ValueError("generation.env must be a map of environment variable names to values")
    if isinstance(yaml_env, dict):
        mode_env.update({str(key): str(value) for key, value in yaml_env.items()})
    return mode_env


def build_agent_generation_env(
    request: dict[str, Any],
    *,
    registry_suffixes: list[str] | None = None,
) -> tuple[dict[str, str], dict[str, str]]:
    generation = request.get("generation", {})
    expert_mode = effective_expert_mode(request)
    if expert_mode not in EXPERT_MODE_ENV:
        raise ValueError(f"No expert generation environment preset exists for mode '{expert_mode}'")
    mode_env = agent_mode_env_for_registry_suffix(request)
    env = os.environ.copy()
    env.update(mode_env)
    env["AIC_EXPERT_TASK_FAMILY"] = str(request.get("task_family", ""))
    if registry_suffixes:
        env_by_suffix = expert_registry_mode_env_by_suffix(request, registry_suffixes)
        if env_by_suffix:
            env["AIC_EXPERT_REGISTRY_MODE_ENV_BY_SUFFIX"] = json.dumps(env_by_suffix, sort_keys=True)
    return env, mode_env


def expert_registry_skip_reason(request: dict[str, Any]) -> str | None:
    generation = request.get("generation", {})
    if bool(generation.get("ignore_expert_setting_registry", False)):
        return None
    mode_entry = expert_registry_mode_entry(request)
    if not isinstance(mode_entry, dict):
        return None
    suffix = effective_registry_suffix(request)
    mode = effective_expert_mode(request)
    status = str(mode_entry.get("status", "unresolved"))
    if status.startswith("skipped"):
        reason = mode_entry.get("last_reason") or "marked skipped in expert setting registry"
        return f"{suffix}/{mode} is {status}: {reason}"
    return None


def _overlay_id() -> str:
    raw = os.environ.get("AIC_EXPERT_REGISTRY_OVERLAY_ID") or socket.gethostname() or "local"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def write_expert_registry_overlay(
    *,
    request: dict[str, Any],
    output_dir: Path,
    agent_mode_env: dict[str, str] | None,
) -> Path | None:
    generation = request.get("generation", {})
    if request.get("generation", {}).get("policy") != "agent":
        return None
    if not bool(generation.get("write_expert_registry_overlay", True)):
        return None
    summary_path = output_dir / "agent_generation" / "generation_summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    mode = effective_expert_mode(request)
    overlay_dir = Path(generation.get("expert_registry_overlay_dir", DEFAULT_EXPERT_REGISTRY_OVERLAY_DIR))
    overlay_dir.mkdir(parents=True, exist_ok=True)
    path = overlay_dir / f"{_overlay_id()}.jsonl"
    entries: list[dict[str, Any]] = []
    for record in summary.get("records", []) or []:
        validation = record.get("validation") or {}
        replay_metrics = record.get("replay_metrics") or {}
        score = validation.get("score", replay_metrics.get("score"))
        engine_config = replay_metrics.get("engine_config")
        suffixes = (
            infer_registry_suffixes_from_engine_config(Path(engine_config), str(request.get("task_family")))
            if engine_config
            else []
        )
        suffix = suffixes[0] if suffixes else effective_registry_suffix(request)
        if not suffix:
            continue
        reasons = [str(reason) for reason in validation.get("reasons", []) or []]
        acceptance_type = str(validation.get("acceptance_type") or "")
        status = "passed" if bool(record.get("accepted")) else "attempted_not_passing"
        if acceptance_type == "near_gate":
            status = "near_gate_passed"
        gpt5_paths: list[str] = []
        gpt5 = record.get("gpt5_failure_analysis") or {}
        for item in gpt5.values() if isinstance(gpt5, dict) else []:
            if isinstance(item, dict) and item.get("analysis"):
                gpt5_paths.append(str(item["analysis"]))
        entries.append(
            {
                "schema_version": "aic_expert_registry_overlay/v1",
                "timestamp_unix": time.time(),
                "overlay_id": _overlay_id(),
                "suffix": suffix,
                "request_suffix": request.get("suffix"),
                "mode": mode,
                "task_family": request.get("task_family"),
                "status": status,
                "acceptance_type": acceptance_type or "full_insertion",
                "score": float(score) if score is not None else None,
                "summary": str(summary_path),
                "output_dir": str(output_dir),
                "mode_env": agent_mode_env_for_registry_suffix(request, suffix),
                "reason": "; ".join(sorted(set(reasons)))[:500] if reasons else summary.get("stopped_reason"),
                "gpt5_analysis_paths": gpt5_paths,
            }
        )
    if not entries:
        suffix = effective_registry_suffix(request)
        if not suffix:
            return None
        accepted = int(summary.get("accepted", 0) or 0)
        entries.append(
            {
                "schema_version": "aic_expert_registry_overlay/v1",
                "timestamp_unix": time.time(),
                "overlay_id": _overlay_id(),
                "suffix": suffix,
                "request_suffix": request.get("suffix"),
                "mode": mode,
                "task_family": request.get("task_family"),
                "status": (
                    "passed"
                    if accepted >= int(request["generation"]["target_accepted_trajectories"])
                    else "attempted_not_passing"
                ),
                "score": None,
                "summary": str(summary_path),
                "output_dir": str(output_dir),
                "mode_env": agent_mode_env or {},
                "reason": summary.get("stopped_reason"),
                "gpt5_analysis_paths": [],
            }
        )
    with path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    return path


def main() -> int:
    args = parse_args()
    request = load_request(args.request_yaml)
    if args.target_accepted_override is not None:
        request.setdefault("generation", {})["target_accepted_trajectories"] = args.target_accepted_override
    if args.max_attempts_override is not None:
        request.setdefault("generation", {})["max_attempts"] = args.max_attempts_override
    validate_request(request)
    validate_override_limits(request)

    target = int(request["generation"]["target_accepted_trajectories"])
    max_attempts = int(request["generation"]["max_attempts"])
    if target <= 0 or max_attempts <= 0:
        raise ValueError("target_accepted_trajectories and max_attempts must be > 0")
    policy = request["generation"]["policy"]
    agent_attempts = effective_max_agent_attempts(request, max_attempts) if policy == "agent" else max_attempts
    num_trials = args.num_trials_override or agent_attempts
    if num_trials <= 0:
        raise ValueError("--num-trials-override must be > 0")
    if num_trials > agent_attempts:
        raise ValueError("--num-trials-override cannot exceed the effective agent attempt budget")

    output_dir = derive_output_dir(request)
    append_if_exists = bool(request.get("generation", {}).get("append_if_exists", False))
    if output_dir.exists() and not append_if_exists and not args.dry_run:
        raise FileExistsError(f"Output directory exists and append_if_exists is false: {output_dir}")
    for child in ("scores", "trials", "logs"):
        (output_dir / child).mkdir(parents=True, exist_ok=True)
    if args.dry_run or args.skip_recording:
        for child in ("raw_dataset", "accepted_dataset"):
            (output_dir / child).mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.request_yaml, output_dir / "request.yaml")
    engine_config_path = write_engine_configs(request, output_dir, num_trials)
    inferred_suffixes = infer_registry_suffixes_from_engine_config(engine_config_path, request["task_family"])
    if len(set(inferred_suffixes)) == 1:
        request.setdefault("generation", {})["_inferred_expert_registry_suffix"] = inferred_suffixes[0]
    skip_reason = expert_registry_skip_reason(request)
    if skip_reason is not None:
        summary = {
            "request_yaml": str(args.request_yaml),
            "output_dir": str(output_dir),
            "task_family": request["task_family"],
            "policy": request["generation"]["policy"],
            "expert_mode": effective_expert_mode(request),
            "target_accepted_trajectories": target,
            "max_attempts": max_attempts,
            "min_score": float(request["acceptance"]["min_score"]),
            "number_attempted": 0,
            "number_accepted": 0,
            "status": "skipped",
            "skip_reason": skip_reason,
            "expert_registry_suffix": effective_registry_suffix(request),
        }
        (output_dir / "generation_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"Skipped expert dataset generation: {skip_reason}")
        print(f"Wrote trajectory dataset request artifacts under: {output_dir}")
        return 0

    commands: list[dict[str, Any]] = []
    dataset_repo_id = f"local/{derived_dataset_name(output_dir)}"
    recording_cmd = [
        "bash",
        str(REPO_ROOT / "aic_utils/lerobot_robot_aic/scripts/launch_policy_recording_per_trial.sh"),
        "--engine-config",
        str(engine_config_path),
        "--dataset-repo-id",
        dataset_repo_id,
        "--dataset-root",
        str(output_dir / "raw_dataset"),
        "--results-root",
        str(output_dir / "scores"),
        "--policy-class",
        POLICY_CLASS[request["generation"]["policy"]],
        "--gazebo-gui",
        "false",
        "--launch-rviz",
        "false",
        "--require-recorder-save-log",
        "true",
        "--remove-bag-data",
        "true",
        "--tmp-dir",
        str(output_dir / "logs" / "per_trial_tmp"),
    ]
    generation = request.get("generation", {})
    if "per_trial_timeout_sec" in generation:
        recording_cmd.extend(["--per-trial-timeout-sec", str(int(generation["per_trial_timeout_sec"]))])
    if "recorder_drain_sec" in generation:
        recording_cmd.extend(["--recorder-drain-sec", str(int(generation["recorder_drain_sec"]))])
    if "startup_delay_sec" in generation:
        recording_cmd.extend(["--startup-delay-sec", str(int(generation["startup_delay_sec"]))])
    agent_mode_env: dict[str, str] | None = None
    if policy == "agent":
        agent_cmd = build_agent_generation_cmd(
            request=request,
            engine_config_path=engine_config_path,
            output_dir=output_dir,
            target=target,
            max_attempts=agent_attempts,
        )
        agent_env, agent_mode_env = build_agent_generation_env(
            request,
            registry_suffixes=inferred_suffixes,
        )
        if not args.skip_recording:
            commands.append(run_command_allowing_codes(agent_cmd, args.dry_run, {0, 2}, env=agent_env))
            write_expert_registry_overlay(
                request=request,
                output_dir=output_dir,
                agent_mode_env=agent_mode_env,
            )
        agent_datasets, agent_score_csvs, agent_accepted = (
            ([], [], None)
            if args.dry_run or args.skip_recording
            else (*_accepted_agent_datasets(output_dir / "agent_generation", output_dir),)
        )
    else:
        agent_datasets = []
        agent_score_csvs = []
        agent_accepted = None
        if not args.skip_recording:
            commands.append(run_command(recording_cmd, args.dry_run))

    filter_cmd = [
        "pixi",
        "run",
        "python",
        str(REPO_ROOT / "aic_utils/lerobot_robot_aic/scripts/filter_merge_lerobot_by_score.py"),
        "--datasets",
        str(output_dir / "raw_dataset"),
        "--score-csvs",
        str(output_dir / "scores" / "score_summary.csv"),
        "--min-score",
        str(float(request["acceptance"]["min_score"])),
        "--output",
        str(output_dir / "accepted_dataset"),
        "--include-videos",
        "--overwrite",
    ]
    filter_min_score = agent_filter_min_score(request) if policy == "agent" else float(request["acceptance"]["min_score"])
    if policy == "agent" and agent_datasets:
        filter_cmd = [
            "pixi",
            "run",
            "python",
            str(REPO_ROOT / "aic_utils/lerobot_robot_aic/scripts/filter_merge_lerobot_by_score.py"),
            "--datasets",
            *[str(path) for path in agent_datasets],
            "--score-csvs",
            *[str(path) for path in agent_score_csvs],
            "--min-score",
            str(filter_min_score),
            "--output",
            str(output_dir / "accepted_dataset"),
            "--include-videos",
            "--overwrite",
        ]
    can_filter_existing = (
        (output_dir / "raw_dataset" / "meta" / "info.json").exists()
        and (output_dir / "scores" / "score_summary.csv").exists()
    )
    can_filter_agent = policy == "agent" and bool(agent_datasets)
    if policy == "agent":
        should_filter = can_filter_agent
    else:
        should_filter = (not args.skip_recording or can_filter_existing) and can_filter_existing
    if not args.skip_filter and should_filter:
        commands.append(run_command(filter_cmd, args.dry_run))

    report_src = output_dir / "accepted_dataset" / "selection_report.csv"
    if report_src.exists():
        shutil.copy2(report_src, output_dir / "selection_report.csv")
    accepted = count_selected(report_src)
    if accepted is None and policy == "agent":
        accepted = agent_accepted
    actual_attempted = num_trials
    if policy == "agent":
        agent_summary_path = output_dir / "agent_generation" / "generation_summary.json"
        if agent_summary_path.exists():
            try:
                agent_summary = json.loads(agent_summary_path.read_text(encoding="utf-8"))
                if agent_summary.get("attempts") is not None:
                    actual_attempted = int(agent_summary["attempts"])
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                actual_attempted = num_trials
    schema_comparison = None
    if args.inspect_reference_dataset:
        schema_comparison = compare_reference(output_dir / "accepted_dataset", args.inspect_reference_dataset)

    summary = {
        "request_yaml": str(args.request_yaml),
        "output_dir": str(output_dir),
        "task_family": request["task_family"],
        "policy": request["generation"]["policy"],
        "expert_mode": effective_expert_mode(request) if policy == "agent" else None,
        "count_label": _count_label(request["task_family"], request),
        "target_accepted_trajectories": target,
        "max_attempts": max_attempts,
        "max_planner_attempts": agent_attempts if policy == "agent" else None,
        "min_score": float(request["acceptance"]["min_score"]),
        "seed": request.get("generation", {}).get("seed"),
        "raw_dataset": str(output_dir / "raw_dataset"),
        "accepted_dataset": str(output_dir / "accepted_dataset"),
        "agent_generation": str(output_dir / "agent_generation") if policy == "agent" else None,
        "agent_mode_env": agent_mode_env,
        "expert_registry_suffix": effective_registry_suffix(request) if policy == "agent" else None,
        "expert_registry_overlay_dir": (
            str(Path(request.get("generation", {}).get("expert_registry_overlay_dir", DEFAULT_EXPERT_REGISTRY_OVERLAY_DIR)))
            if policy == "agent"
            else None
        ),
        "agent_replay_datasets": [str(path) for path in agent_datasets],
        "scores": str(output_dir / "scores"),
        "number_attempted": actual_attempted,
        "number_accepted": accepted,
        "generated_engine_config": str(engine_config_path),
        "command_lines_run": commands,
        "schema_comparison": schema_comparison,
        "notes": {
            "dataset_format": "raw_dataset and accepted_dataset are native LeRobot dataset roots.",
            "cable_jitter": (
                "Missing cable fields inherit the existing generate_random_trials_config.py "
                "internal cable jitter. Explicit cable fields in request YAML override it."
            ),
            "attempt_strategy": (
                "Agent generation creates at most generation.max_planner_attempts full planner+replay attempts "
                "when set, otherwise generation.max_attempts, unless --num-trials-override is lower."
            ),
            "opposite_family_defaults": (
                "Minimal sfp_to_nic requests default sc_ports.count to 0, and minimal sc_to_sc "
                "requests default nic_cards.count to 0. Explicit opposite-family scene sections "
                "are preserved for obstacle-heavy data generation."
            ),
            "agent_recorder_drain": (
                "Agent generation clamps recorder_drain_sec and planner_recorder_drain_sec to at "
                "least 45s unless generation.allow_short_agent_drain is true; shorter drains can "
                "drop otherwise valid planner/replay outputs during process teardown."
            ),
        },
    }
    (output_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"Wrote trajectory dataset request artifacts under: {output_dir}")
    if accepted is not None and accepted < target:
        print(f"Accepted {accepted}/{target}; generate additional attempts or adjust acceptance criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
