#!/usr/bin/env python3
"""Shift generated near-gate reset-body poses without moving the target scene.

This is a small experiment utility for cases where an episode YAML has the
right semantic tip reference, but Isaac reset/settle places the realized tip a
few millimeters too far in or out.  It shifts only reset-body fields; the port,
target, entrance, and semantic reference remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _vec(raw: Any, *, name: str) -> list[float]:
    if not isinstance(raw, list | tuple) or len(raw) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(v) for v in raw]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _round_vec(v: list[float]) -> list[float]:
    return [round(float(x), 9) for x in v]


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
    target = scene.get("target") if isinstance(scene, dict) else None
    if isinstance(target, dict) and isinstance(target.get("start_near_gate"), dict):
        return target["start_near_gate"]
    if isinstance(data.get("start_near_gate"), dict):
        return data["start_near_gate"]
    raise ValueError("episode has no start_near_gate mapping")


def shift_episode(path: Path, *, axis_shift_m: float, lateral_shift_m: float, note: str | None) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a YAML mapping")
    start = _start_mapping(data)
    axis = _vec(start.get("target_gate_axis_world"), name="target_gate_axis_world")
    lateral = [0.0, 0.0, 0.0]
    if abs(float(lateral_shift_m)) > 0.0:
        lateral = _vec(start.get("lateral_direction_world"), name="lateral_direction_world")
    shift = [axis_shift_m * axis[i] + lateral_shift_m * lateral[i] for i in range(3)]
    changed: dict[str, Any] = {
        "episode": path.name,
        "axis_shift_m": axis_shift_m,
        "lateral_shift_m": lateral_shift_m,
        "world_shift_m": _round_vec(shift),
        "fields": {},
    }
    for key in ("body_start_position_world", "tcp_start_position_world"):
        if key in start:
            old = _vec(start[key], name=key)
            new = _add(old, shift)
            start[key] = _round_vec(new)
            changed["fields"][key] = {"old": _round_vec(old), "new": _round_vec(new)}
    if "reset_body_offset_from_reference_world" in start:
        old = _vec(start["reset_body_offset_from_reference_world"], name="reset_body_offset_from_reference_world")
        new = _add(old, shift)
        start["reset_body_offset_from_reference_world"] = _round_vec(new)
        changed["fields"]["reset_body_offset_from_reference_world"] = {
            "old": _round_vec(old),
            "new": _round_vec(new),
        }
    start["axis_shift_calibration"] = {
        "method": "reset body position += axis_shift_m * target_gate_axis_world + lateral_shift_m * lateral_direction_world",
        "axis_shift_m": axis_shift_m,
        "lateral_shift_m": lateral_shift_m,
        "world_shift_m": _round_vec(shift),
        "note": note or "",
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--axis-shift-m", required=True, type=float)
    parser.add_argument(
        "--lateral-shift-m",
        type=float,
        default=0.0,
        help="Optional shift along start_near_gate.lateral_direction_world.",
    )
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

    changes = [
        shift_episode(
            path,
            axis_shift_m=float(args.axis_shift_m),
            lateral_shift_m=float(args.lateral_shift_m),
            note=str(args.note),
        )
        for path in _episode_files(dst)
    ]
    summary = {
        "input_dir": str(src),
        "output_dir": str(dst),
        "axis_shift_m": float(args.axis_shift_m),
        "lateral_shift_m": float(args.lateral_shift_m),
        "note": str(args.note),
        "episodes": changes,
    }
    (dst / "axis_shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
