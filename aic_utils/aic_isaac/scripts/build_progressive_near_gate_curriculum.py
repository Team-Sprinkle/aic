#!/usr/bin/env python3
"""Build a progressive near-gate curriculum with randomized lateral directions.

The generated episodes interpolate semantic tip start geometry from an outside
near-gate start to a farther, wider start while preserving each base episode's
calibrated reset-body offset, orientation, and other randomization metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import yaml


def _vec(raw: Any, *, name: str, length: int = 3) -> list[float]:
    if not isinstance(raw, list | tuple) or len(raw) != length:
        raise ValueError(f"{name} must be a {length}-value list")
    return [float(v) for v in raw]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * float(s) for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _norm(v: list[float]) -> float:
    return math.sqrt(_dot(v, v))


def _normalize(v: list[float], *, name: str) -> list[float]:
    n = _norm(v)
    if n < 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector: {name}")
    return [x / n for x in v]


def _round(v: list[float], digits: int = 9) -> list[float]:
    return [round(float(x), digits) for x in v]


def _qnormalize(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1.0e-12:
        raise ValueError("cannot normalize near-zero quaternion")
    return [float(v) / n for v in q]


def _orthonormal_basis(axis: list[float], seed_dir: list[float]) -> tuple[list[float], list[float]]:
    axis_n = _normalize(axis, name="target_gate_axis_world")
    seed_perp = _sub(seed_dir, _scale(axis_n, _dot(seed_dir, axis_n)))
    if _norm(seed_perp) < 1.0e-9:
        fallback = [1.0, 0.0, 0.0] if abs(axis_n[0]) < 0.9 else [0.0, 1.0, 0.0]
        seed_perp = _sub(fallback, _scale(axis_n, _dot(fallback, axis_n)))
    e1 = _normalize(seed_perp, name="projected lateral_direction_world")
    e2 = [
        axis_n[1] * e1[2] - axis_n[2] * e1[1],
        axis_n[2] * e1[0] - axis_n[0] * e1[2],
        axis_n[0] * e1[1] - axis_n[1] * e1[0],
    ]
    return e1, _normalize(e2, name="axis cross lateral basis")


def _episode_files(config_dir: Path) -> list[Path]:
    episodes = sorted((config_dir / "episodes").glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml files under {config_dir / 'episodes'}")
    return episodes


def _start(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("start_near_gate"), dict):
        return scene["start_near_gate"]
    raise ValueError("episode has no scene.start_near_gate mapping")


def _write_episode(
    *,
    base_episode: Path,
    out_path: Path,
    episode_index: int,
    episode_count: int,
    axial_start_m: float,
    axial_end_m: float,
    lateral_start_m: float,
    lateral_end_m: float,
    rng: random.Random,
) -> dict[str, Any]:
    data = yaml.safe_load(base_episode.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{base_episode} did not parse to a mapping")
    start = _start(data)
    gate = _vec(start.get("target_gate_position"), name="target_gate_position")
    axis = _normalize(_vec(start.get("target_gate_axis_world"), name="target_gate_axis_world"), name="target_gate_axis_world")
    seed_lateral = _vec(start.get("lateral_direction_world"), name="lateral_direction_world")
    e1, e2 = _orthonormal_basis(axis, seed_lateral)
    t = 0.0 if episode_count <= 1 else float(episode_index - 1) / float(episode_count - 1)
    signed_axial = float(axial_start_m) + t * (float(axial_end_m) - float(axial_start_m))
    lateral = float(lateral_start_m) + t * (float(lateral_end_m) - float(lateral_start_m))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    lateral_dir = _normalize(
        _add(_scale(e1, math.cos(angle)), _scale(e2, math.sin(angle))),
        name="random lateral direction",
    )
    reference = _add(_add(gate, _scale(axis, signed_axial)), _scale(lateral_dir, lateral))
    lowtheta = start.get("lowtheta_metric_reset") or {}
    lowtheta_offset = lowtheta.get("metric_reset_body_minus_tip_world_m")
    reset_offset = (
        _vec(lowtheta_offset, name="lowtheta_metric_reset.metric_reset_body_minus_tip_world_m")
        if isinstance(lowtheta_offset, (list, tuple)) and len(lowtheta_offset) == 3
        else _vec(start.get("reset_body_offset_from_reference_world"), name="reset_body_offset_from_reference_world")
    )
    lowtheta_quat = lowtheta.get("reset_body_quat_wxyz")
    body = _add(reference, reset_offset)

    reference_keys = (
        "reference_tip_center_position_world",
        "reference_reward_body_start_position_world",
        "reference_tcp_position",
        "reference_body_position",
    )
    for key in reference_keys:
        if key in start:
            start[key] = _round(reference)
    for key in ("body_start_position_world", "tcp_start_position_world"):
        if key in start:
            start[key] = _round(body)
    if isinstance(lowtheta_quat, (list, tuple)) and len(lowtheta_quat) == 4:
        quat = _round(_qnormalize([float(v) for v in lowtheta_quat]), digits=9)
        for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
            if key in start:
                start[key] = quat
        start["reset_body_offset_from_reference_world"] = _round(reset_offset)
    start["lateral_direction_world"] = _round(lateral_dir)
    start["axial_distance_m"] = round(signed_axial, 9)
    start["lateral_distance_m"] = round(lateral, 9)
    start["achieved_axial_distance_m"] = round(signed_axial, 9)
    start["achieved_lateral_distance_m"] = round(lateral, 9)
    start["achieved_distance"] = round(math.sqrt(signed_axial * signed_axial + lateral * lateral), 9)
    start["progressive_curriculum_variant"] = {
        "source_episode": str(base_episode),
        "episode_index": episode_index,
        "episode_count": episode_count,
        "interpolation_t": round(t, 9),
        "target_signed_depth_m": round(signed_axial, 9),
        "requested_lateral_m": round(lateral, 9),
        "lateral_direction_world": _round(lateral_dir),
        "method": "linear signed-depth/lateral curriculum with per-episode random lateral direction; calibrated reset-body offset and orientation preserved",
        "used_lowtheta_metric_reset": isinstance(lowtheta_quat, (list, tuple)) and len(lowtheta_quat) == 4,
    }
    data["episode_id"] = f"{data.get('episode_id', base_episode.stem)}_progressive_{episode_index:06d}"
    data["episode_index"] = episode_index
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return start["progressive_curriculum_variant"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--axial-start-m", type=float, default=-0.020)
    parser.add_argument("--axial-end-m", type=float, default=-0.060)
    parser.add_argument("--lateral-start-m", type=float, default=0.0)
    parser.add_argument("--lateral-end-m", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    (output_root / "episodes").mkdir(parents=True)

    base_episodes = _episode_files(args.base_config_dir.resolve())
    rng = random.Random(int(args.seed))
    records = []
    for idx in range(1, int(args.episodes) + 1):
        base_episode = base_episodes[(idx - 1) % len(base_episodes)]
        out_path = output_root / "episodes" / f"episode_{idx:06d}.yaml"
        record = _write_episode(
            base_episode=base_episode,
            out_path=out_path,
            episode_index=idx,
            episode_count=int(args.episodes),
            axial_start_m=float(args.axial_start_m),
            axial_end_m=float(args.axial_end_m),
            lateral_start_m=float(args.lateral_start_m),
            lateral_end_m=float(args.lateral_end_m),
            rng=rng,
        )
        records.append({"episode": out_path.name, **record})

    manifest = {
        "base_config_dir": str(args.base_config_dir.resolve()),
        "base_episode_count": len(base_episodes),
        "episodes": int(args.episodes),
        "axial_start_m": float(args.axial_start_m),
        "axial_end_m": float(args.axial_end_m),
        "lateral_start_m": float(args.lateral_start_m),
        "lateral_end_m": float(args.lateral_end_m),
        "seed": int(args.seed),
        "records": records,
    }
    (output_root / "progressive_curriculum_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
