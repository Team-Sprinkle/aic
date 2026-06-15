#!/usr/bin/env python3
"""Build constant-distance actual-tip 40x10 reset episodes.

Each output episode places the actual reward body (normally ``sfp_tip_link``)
at a fixed signed axial distance and fixed lateral radius from the entrance
while varying the lateral direction.  The IK reset target is the reset body
(normally ``gripper_tcp``), so the builder derives that body pose from the
measured metric ``reset_body_minus_tip`` offset stored in the template episode.
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


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * s for i in range(3)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: list[float], *, name: str) -> list[float]:
    n = _norm(a)
    if n < 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector: {name}")
    return [x / n for x in a]


def _round(a: list[float], digits: int = 9) -> list[float]:
    return [round(float(x), digits) for x in a]


def _start(data: dict[str, Any]) -> dict[str, Any]:
    start = ((data.get("scene") or {}).get("start_near_gate") or {})
    if not isinstance(start, dict):
        raise ValueError("episode has no scene.start_near_gate mapping")
    return start


def _episode_files(config_dir: Path) -> list[Path]:
    episodes = sorted((config_dir / "episodes").glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml files under {config_dir / 'episodes'}")
    return episodes


def _project_lateral_basis(axis: list[float], preferred: list[float]) -> tuple[list[float], list[float]]:
    axis_n = _unit(axis, name="target_gate_axis_world")
    projected = _sub(preferred, _scale(axis_n, _dot(preferred, axis_n)))
    e1 = _unit(projected, name="projected lateral_direction_world")
    e2 = _unit(_cross(axis_n, e1), name="axis x lateral_direction_world")
    return e1, e2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-config-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episode-count", type=int, default=48)
    parser.add_argument("--axial-mm", type=float, default=-40.0)
    parser.add_argument("--lateral-mm", type=float, default=10.0)
    parser.add_argument(
        "--axial-mm-min",
        type=float,
        default=None,
        help="Optional minimum signed axial start distance in mm. If set with --axial-mm-max, per-episode axial distance is sampled uniformly.",
    )
    parser.add_argument(
        "--axial-mm-max",
        type=float,
        default=None,
        help="Optional maximum signed axial start distance in mm. If set with --axial-mm-min, per-episode axial distance is sampled uniformly.",
    )
    parser.add_argument(
        "--lateral-mm-min",
        type=float,
        default=None,
        help="Optional minimum lateral start radius in mm. If set with --lateral-mm-max, per-episode lateral distance is sampled uniformly.",
    )
    parser.add_argument(
        "--lateral-mm-max",
        type=float,
        default=None,
        help="Optional maximum lateral start radius in mm. If set with --lateral-mm-min, per-episode lateral distance is sampled uniformly.",
    )
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument(
        "--placement-axial-mm",
        type=float,
        default=None,
        help=(
            "Optional pre-settle axial placement in mm. Metadata still records --axial-mm as the "
            "requested/achieved target; use this only when compensating a measured post-reset settle drift."
        ),
    )
    parser.add_argument(
        "--placement-lateral-mm",
        type=float,
        default=None,
        help=(
            "Optional pre-settle lateral placement in mm. Metadata still records --lateral-mm as the "
            "requested/achieved target; use this only when compensating a measured post-reset settle drift."
        ),
    )
    parser.add_argument("--start-angle-deg", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    (output_dir / "episodes").mkdir(parents=True)

    templates = _episode_files(args.template_config_dir.resolve())
    first = yaml.safe_load(templates[0].read_text(encoding="utf-8"))
    first_start = _start(first)
    axis = _unit(_vec(first_start.get("target_gate_axis_world"), name="target_gate_axis_world"), name="target_gate_axis_world")
    preferred_lateral = _vec(first_start.get("lateral_direction_world"), name="lateral_direction_world")
    e1, e2 = _project_lateral_basis(axis, preferred_lateral)
    axial_m = float(args.axial_mm) / 1000.0
    lateral_m = abs(float(args.lateral_mm)) / 1000.0
    use_axial_range = args.axial_mm_min is not None or args.axial_mm_max is not None
    use_lateral_range = args.lateral_mm_min is not None or args.lateral_mm_max is not None
    if use_axial_range and (args.axial_mm_min is None or args.axial_mm_max is None):
        raise ValueError("--axial-mm-min and --axial-mm-max must be set together")
    if use_lateral_range and (args.lateral_mm_min is None or args.lateral_mm_max is None):
        raise ValueError("--lateral-mm-min and --lateral-mm-max must be set together")
    axial_min_m = float(args.axial_mm_min) / 1000.0 if use_axial_range else axial_m
    axial_max_m = float(args.axial_mm_max) / 1000.0 if use_axial_range else axial_m
    lateral_min_m = abs(float(args.lateral_mm_min)) / 1000.0 if use_lateral_range else lateral_m
    lateral_max_m = abs(float(args.lateral_mm_max)) / 1000.0 if use_lateral_range else lateral_m
    if axial_min_m > axial_max_m:
        axial_min_m, axial_max_m = axial_max_m, axial_min_m
    if lateral_min_m > lateral_max_m:
        lateral_min_m, lateral_max_m = lateral_max_m, lateral_min_m
    rng = random.Random(int(args.seed))
    placement_axial_m = (
        float(args.placement_axial_mm) / 1000.0 if args.placement_axial_mm is not None else axial_m
    )
    placement_lateral_m = (
        abs(float(args.placement_lateral_mm)) / 1000.0 if args.placement_lateral_mm is not None else lateral_m
    )
    fixed_distance_m = math.sqrt(axial_m * axial_m + lateral_m * lateral_m)
    placement_distance_m = math.sqrt(placement_axial_m * placement_axial_m + placement_lateral_m * placement_lateral_m)

    records: list[dict[str, Any]] = []
    count = int(args.episode_count)
    for idx in range(count):
        sample_axial_m = rng.uniform(axial_min_m, axial_max_m) if use_axial_range else axial_m
        sample_lateral_m = rng.uniform(lateral_min_m, lateral_max_m) if use_lateral_range else lateral_m
        sample_placement_axial_m = (
            float(args.placement_axial_mm) / 1000.0 if args.placement_axial_mm is not None else sample_axial_m
        )
        sample_placement_lateral_m = (
            abs(float(args.placement_lateral_mm)) / 1000.0
            if args.placement_lateral_mm is not None
            else sample_lateral_m
        )
        sample_fixed_distance_m = math.sqrt(sample_axial_m * sample_axial_m + sample_lateral_m * sample_lateral_m)
        sample_placement_distance_m = math.sqrt(
            sample_placement_axial_m * sample_placement_axial_m
            + sample_placement_lateral_m * sample_placement_lateral_m
        )
        template_path = templates[idx % len(templates)]
        data = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{template_path} did not parse to a mapping")
        start = _start(data)
        gate = _vec(start.get("target_gate_position"), name="target_gate_position")
        body = _vec(start.get("body_start_position_world") or start.get("tcp_start_position_world"), name="body_start_position_world")
        old_reference = _vec(
            start.get("reference_tip_center_position_world") or start.get("reference_reward_body_start_position_world"),
            name="reference_tip_center_position_world",
        )
        metric_reset = start.get("lowtheta_metric_reset") or {}
        if isinstance(metric_reset, dict) and metric_reset.get("metric_reset_body_minus_tip_world_m") is not None:
            reset_body_offset = _vec(
                metric_reset.get("metric_reset_body_minus_tip_world_m"),
                name="lowtheta_metric_reset.metric_reset_body_minus_tip_world_m",
            )
            offset_source = "lowtheta_metric_reset.metric_reset_body_minus_tip_world_m"
        else:
            # Legacy fallback for templates without a live measured body-tip offset.
            reset_body_offset = _sub(body, old_reference)
            offset_source = "body_start_position_world - reference_tip_center_position_world"

        angle = math.radians(float(args.start_angle_deg)) + (2.0 * math.pi * idx / max(count, 1))
        lateral_dir = _unit(
            _add(_scale(e1, math.cos(angle)), _scale(e2, math.sin(angle))),
            name="sampled lateral direction",
        )
        actual_tip = _add(_add(gate, _scale(axis, sample_placement_axial_m)), _scale(lateral_dir, sample_placement_lateral_m))
        body_start = _add(actual_tip, reset_body_offset)

        for key in ("reference_tip_center_position_world", "reference_reward_body_start_position_world", "reference_body_position", "reference_tcp_position"):
            if key in start:
                start[key] = _round(actual_tip)
        for key in ("body_start_position_world", "tcp_start_position_world"):
            if key in start:
                start[key] = _round(body_start)
        start["reset_body_offset_from_reference_world"] = _round(reset_body_offset)
        start["lateral_direction_world"] = _round(lateral_dir)
        start["axial_distance_m"] = round(sample_axial_m, 9)
        start["lateral_distance_m"] = round(sample_lateral_m, 9)
        start["achieved_distance"] = round(sample_fixed_distance_m, 9)
        start["achieved_axial_distance_m"] = round(sample_axial_m, 9)
        start["achieved_lateral_distance_m"] = round(sample_lateral_m, 9)
        start["constant_distance_40x10_variant"] = {
            "source_episode": str(template_path),
            "sample_index": idx,
            "angle_rad": round(angle, 9),
            "target_signed_depth_m": round(sample_axial_m, 9),
            "requested_lateral_m": round(sample_lateral_m, 9),
            "fixed_distance_to_entrance_m": round(sample_fixed_distance_m, 9),
            "placement_signed_depth_m": round(sample_placement_axial_m, 9),
            "placement_lateral_m": round(sample_placement_lateral_m, 9),
            "placement_distance_to_entrance_m": round(sample_placement_distance_m, 9),
            "axial_range_m": [round(axial_min_m, 9), round(axial_max_m, 9)],
            "lateral_range_m": [round(lateral_min_m, 9), round(lateral_max_m, 9)],
            "reset_body_offset_source": offset_source,
            "method": "actual_tip = entrance + axis * placement_axial_m + sampled_lateral_dir * placement_lateral_m; reset_body = actual_tip + measured reset_body_minus_tip offset",
        }
        # Remove older labels that can otherwise imply a different generation method.
        start.pop("randomized_curriculum_variant", None)
        start.pop("axis_shift_calibration", None)

        data["episode_id"] = f"{data.get('episode_id', template_path.stem)}_constant40x10_{idx + 1:06d}"
        out_path = output_dir / "episodes" / f"episode_{idx + 1:06d}.yaml"
        out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "episode": out_path.name,
                "source_episode": str(template_path),
                "angle_rad": round(angle, 9),
                "target_signed_depth_m": round(sample_axial_m, 9),
                "requested_lateral_m": round(sample_lateral_m, 9),
                "fixed_distance_to_entrance_m": round(sample_fixed_distance_m, 9),
                "placement_signed_depth_m": round(sample_placement_axial_m, 9),
                "placement_lateral_m": round(sample_placement_lateral_m, 9),
                "placement_distance_to_entrance_m": round(sample_placement_distance_m, 9),
                "actual_tip_position_world": _round(actual_tip),
                "reset_body_position_world": _round(body_start),
                "reset_body_offset_source": offset_source,
            }
        )

    manifest = {
        "template_config_dir": str(args.template_config_dir.resolve()),
        "episode_count": count,
        "axial_m": round(axial_m, 9),
        "lateral_m": round(lateral_m, 9),
        "axial_range_m": [round(axial_min_m, 9), round(axial_max_m, 9)],
        "lateral_range_m": [round(lateral_min_m, 9), round(lateral_max_m, 9)],
        "fixed_distance_to_entrance_m": round(fixed_distance_m, 9),
        "placement_axial_m": round(placement_axial_m, 9),
        "placement_lateral_m": round(placement_lateral_m, 9),
        "placement_distance_to_entrance_m": round(placement_distance_m, 9),
        "seed": int(args.seed),
        "records": records,
    }
    (output_dir / "constant_distance_40x10_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: manifest[k] for k in ("template_config_dir", "episode_count", "axial_range_m", "lateral_range_m")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
