#!/usr/bin/env python3
"""Copy generated near-gate episodes and pin the semantic lateral direction.

This is intended for controlled 40/10 ablations.  It keeps the existing target
scene and calibrated reset-body offset, then recomputes the semantic tip
reference and reset body position from:

    reference = gate - axis * axial_distance + lateral_dir * lateral_distance

Random lateral starts remain the default in ``isaac_episode_configs.py``; this
script is only for fixed-direction experiment folders.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


def _vec(raw: Any, *, name: str) -> list[float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(v) for v in raw]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: list[float], *, name: str) -> list[float]:
    n = _norm(a)
    if n < 1.0e-10:
        raise ValueError(f"{name} must be non-zero")
    return [a[i] / n for i in range(3)]


def _round_vec(v: list[float]) -> list[float]:
    return [round(float(x), 9) for x in v]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * s for i in range(3)]


def _episode_files(config_dir: Path) -> list[Path]:
    episodes_dir = config_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"{episodes_dir} does not exist")
    episodes = sorted(episodes_dir.glob("*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode YAMLs found under {episodes_dir}")
    return episodes


def _start_mapping(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("start_near_gate"), dict):
        return scene["start_near_gate"]
    raise ValueError("episode has no scene.start_near_gate mapping")


def _project_lateral_direction(raw_direction: list[float], axis: list[float]) -> list[float]:
    axis = _normalize(axis, name="target_gate_axis_world")
    raw = _normalize(raw_direction, name="lateral_direction_world")
    projected = _sub(raw, _scale(axis, _dot(raw, axis)))
    if _norm(projected) < 1.0e-8:
        raise ValueError("lateral_direction_world must not be parallel to target_gate_axis_world")
    return _normalize(projected, name="projected lateral_direction_world")


def set_episode_lateral_direction(path: Path, *, lateral_direction_world: list[float], note: str) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    start = _start_mapping(data)
    gate = _vec(start.get("target_gate_position"), name="target_gate_position")
    axis = _normalize(_vec(start.get("target_gate_axis_world"), name="target_gate_axis_world"), name="target_gate_axis_world")
    axial = float(start.get("axial_distance_m"))
    lateral = float(start.get("lateral_distance_m"))
    direction = _project_lateral_direction(lateral_direction_world, axis)
    reset_offset = _vec(
        start.get("reset_body_offset_from_reference_world", [0.0, 0.0, 0.0]),
        name="reset_body_offset_from_reference_world",
    )

    old_reference = _vec(start.get("reference_tip_center_position_world"), name="reference_tip_center_position_world")
    old_body = _vec(start.get("body_start_position_world"), name="body_start_position_world")
    reference = _add(_add(gate, _scale(axis, -axial)), _scale(direction, lateral))
    body = _add(reference, reset_offset)
    ref_delta = _sub(reference, gate)
    axial_component = abs(_dot(ref_delta, axis))
    lateral_component = math.sqrt(max(0.0, _norm(ref_delta) ** 2 - axial_component * axial_component))

    for key in (
        "reference_tip_center_position_world",
        "reference_reward_body_start_position_world",
        "reference_body_position",
        "reference_tcp_position",
    ):
        if key in start:
            start[key] = _round_vec(reference)
    for key in ("body_start_position_world", "tcp_start_position_world"):
        if key in start:
            start[key] = _round_vec(body)
    start["lateral_direction_world"] = _round_vec(direction)
    start["achieved_distance"] = round(_norm(ref_delta), 9)
    start["achieved_axial_distance_m"] = round(axial_component, 9)
    start["achieved_lateral_distance_m"] = round(lateral_component, 9)
    start["fixed_lateral_direction_calibration"] = {
        "method": "reference = gate - axis * axial_distance + lateral_dir * lateral_distance; body_start = reference + existing reset_body_offset_from_reference_world",
        "requested_lateral_direction_world": _round_vec(lateral_direction_world),
        "applied_lateral_direction_world": _round_vec(direction),
        "old_reference_tip_center_position_world": _round_vec(old_reference),
        "new_reference_tip_center_position_world": _round_vec(reference),
        "old_body_start_position_world": _round_vec(old_body),
        "new_body_start_position_world": _round_vec(body),
        "note": note,
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {
        "episode": path.name,
        "old_reference_tip_center_position_world": _round_vec(old_reference),
        "new_reference_tip_center_position_world": _round_vec(reference),
        "old_body_start_position_world": _round_vec(old_body),
        "new_body_start_position_world": _round_vec(body),
        "applied_lateral_direction_world": _round_vec(direction),
        "achieved_axial_distance_m": round(axial_component, 9),
        "achieved_lateral_distance_m": round(lateral_component, 9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--lateral-direction-world", nargs=3, required=True, type=float)
    parser.add_argument("--note", default="")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.input_dir.resolve()
    dst = args.output_dir.resolve()
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    direction = [float(v) for v in args.lateral_direction_world]
    changes = [
        set_episode_lateral_direction(path, lateral_direction_world=direction, note=str(args.note))
        for path in _episode_files(dst)
    ]
    summary = {
        "input_dir": str(src),
        "output_dir": str(dst),
        "requested_lateral_direction_world": _round_vec(direction),
        "note": str(args.note),
        "episodes": changes,
    }
    (dst / "fixed_lateral_direction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

