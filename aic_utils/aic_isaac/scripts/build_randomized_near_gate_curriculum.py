#!/usr/bin/env python3
"""Build randomized near-gate curriculum episode folders from calibrated resets.

The base Isaac episode generator only accepts non-negative near-gate axial
distances, so shallow positive-depth final-window starts are represented as
calibrated reset-body shifts from an already validated episode.  The seated
target, strict success checker, and asset geometry are left unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Variant:
    bucket: str
    target_s_m: float
    lateral_m: float
    rotvec_rad: tuple[float, float, float]


def _vec(raw: Any, *, name: str, length: int = 3) -> list[float]:
    if not isinstance(raw, list | tuple) or len(raw) != length:
        raise ValueError(f"{name} must be a {length}-value list")
    return [float(v) for v in raw]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * s for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: list[float], *, name: str) -> list[float]:
    n = _norm(a)
    if n < 1.0e-12:
        raise ValueError(f"cannot normalize near-zero vector: {name}")
    return [x / n for x in a]


def _round(v: list[float], digits: int = 9) -> list[float]:
    return [round(float(x), digits) for x in v]


def _qnorm(q: list[float]) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1.0e-12:
        raise ValueError("cannot normalize near-zero quaternion")
    return [float(v) / n for v in q]


def _qmul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = _qnorm(a)
    bw, bx, by, bz = _qnorm(b)
    return _qnorm(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _qconj(q: list[float]) -> list[float]:
    qw, qx, qy, qz = _qnorm(q)
    return [qw, -qx, -qy, -qz]


def _qrot(q: list[float], v: list[float]) -> list[float]:
    qw, qx, qy, qz = _qnorm(q)
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    # Quaternion-vector rotation without normalizing the pure-vector quaternion.
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _rotvec_quat(rx: float, ry: float, rz: float) -> list[float]:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return [math.cos(half), rx * scale, ry * scale, rz * scale]


def _measured_tip_pos_in_reset_body(start: dict[str, Any]) -> list[float] | None:
    for key in ("semantic_tip_transform_repair", "semantic_tip_transform", "calibrated_from_reset_diagnostic"):
        raw = (start.get(key) or {}).get("measured_tip_pos_in_reset_body")
        if isinstance(raw, list | tuple) and len(raw) == 3:
            return _vec(raw, name=f"{key}.measured_tip_pos_in_reset_body")
    metric_reset = start.get("lowtheta_metric_reset") or {}
    body_minus_tip = metric_reset.get("metric_reset_body_minus_tip_world_m")
    reset_quat = metric_reset.get("reset_body_quat_wxyz") or start.get("reset_body_orientation_wxyz")
    if (
        isinstance(body_minus_tip, list | tuple)
        and len(body_minus_tip) == 3
        and isinstance(reset_quat, list | tuple)
        and len(reset_quat) == 4
    ):
        tip_minus_body_world = [-float(v) for v in body_minus_tip]
        return _qrot(_qconj(_vec(reset_quat, name="reset_body_quat_wxyz", length=4)), tip_minus_body_world)
    return None


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


def _write_variant(
    *,
    base_episode: Path,
    out_path: Path,
    variant: Variant,
    base_settled_s_m: float,
    lateral_sign: float,
    rng: random.Random,
) -> dict[str, Any]:
    data = yaml.safe_load(base_episode.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{base_episode} did not parse to a mapping")
    start = _start(data)
    axis = _vec(start.get("target_gate_axis_world"), name="target_gate_axis_world")
    lateral_dir = _vec(start.get("lateral_direction_world"), name="lateral_direction_world")
    axis_shift_m = float(variant.target_s_m) - float(base_settled_s_m)
    lateral_shift_m = lateral_sign * float(variant.lateral_m)
    # Small deterministic jitter prevents identical starts while keeping the
    # requested bucket center auditable.
    axis_shift_m += rng.uniform(-0.00015, 0.00015)
    lateral_shift_m += rng.uniform(-0.00005, 0.00005)
    shift = _add(_scale(axis, axis_shift_m), _scale(lateral_dir, lateral_shift_m))

    reference_keys = (
        "reference_tip_center_position_world",
        "reference_reward_body_start_position_world",
        "reference_tcp_position",
        "reference_body_position",
    )
    shifted_references: dict[str, list[float]] = {}
    for key in reference_keys:
        if key in start:
            shifted_references[key] = _add(_vec(start[key], name=key), shift)

    base_q = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
    base_q = _vec(base_q, name="body_start_orientation_wxyz", length=4)
    dq = _rotvec_quat(*variant.rotvec_rad)
    q_new = _qmul(dq, base_q)
    tip_local = _measured_tip_pos_in_reset_body(start)
    reference_tip = shifted_references.get("reference_tip_center_position_world") or shifted_references.get(
        "reference_reward_body_start_position_world"
    )
    for key, value in shifted_references.items():
        start[key] = _round(value)
    if tip_local is not None and reference_tip is not None:
        body_position = _add(reference_tip, _scale(_qrot(q_new, tip_local), -1.0))
        for key in ("body_start_position_world", "tcp_start_position_world"):
            if key in start:
                start[key] = _round(body_position)
        if "reset_body_offset_from_reference_world" in start:
            start["reset_body_offset_from_reference_world"] = _round(
                _add(body_position, _scale(reference_tip, -1.0))
            )
    else:
        for key in ("body_start_position_world", "tcp_start_position_world"):
            if key in start:
                start[key] = _round(_add(_vec(start[key], name=key), shift))
        # Pure translation preserves the calibrated reset-body-to-reference offset.
        # Without a measured semantic tip transform, a rotation perturbation cannot
        # compensate the reset body position to keep sfp_tip_link fixed; record this
        # explicitly instead of corrupting the offset metadata.
    for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
        if key in start:
            start[key] = _round(q_new)

    reference_for_metadata = shifted_references.get("reference_tip_center_position_world") or shifted_references.get(
        "reference_reward_body_start_position_world"
    )
    if reference_for_metadata is not None:
        gate = _vec(start.get("target_gate_position"), name="target_gate_position")
        axis_n = _normalize(axis, name="target_gate_axis_world")
        lateral_dir_n = _normalize(
            _sub(lateral_dir, _scale(axis_n, _dot(lateral_dir, axis_n))),
            name="projected lateral_direction_world",
        )
        ref_delta = _sub(reference_for_metadata, gate)
        signed_depth = _dot(ref_delta, axis_n)
        signed_lateral = _dot(ref_delta, lateral_dir_n)
        lateral_residual = _sub(ref_delta, _scale(axis_n, signed_depth))
        lateral_norm = _norm(lateral_residual)
        start["axial_distance_m"] = round(signed_depth, 9)
        start["lateral_distance_m"] = round(signed_lateral, 9)
        start["achieved_distance"] = round(_norm(ref_delta), 9)
        start["achieved_axial_distance_m"] = round(signed_depth, 9)
        start["achieved_lateral_distance_m"] = round(lateral_norm, 9)

    data["episode_id"] = f"{data.get('episode_id', base_episode.stem)}_{variant.bucket}_{out_path.stem}"
    start["randomized_curriculum_variant"] = {
        "source_episode": str(base_episode),
        "bucket": variant.bucket,
        "base_settled_s_m": base_settled_s_m,
        "target_signed_depth_m": variant.target_s_m,
        "requested_lateral_m": variant.lateral_m,
        "lateral_sign": lateral_sign,
        "axis_shift_m": round(axis_shift_m, 9),
        "lateral_shift_m": round(lateral_shift_m, 9),
        "world_shift_m": _round(shift),
        "rotvec_rad": [round(float(v), 9) for v in variant.rotvec_rad],
        "method": "semantic-tip-preserving reset body pose shift/rotation from calibrated base; target/full-depth strict success unchanged",
        "tip_preserving_rotation": tip_local is not None and reference_tip is not None,
    }
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return start["randomized_curriculum_variant"]


def _phase_variants(phase: str) -> list[Variant]:
    if phase == "shallow_final":
        depths = [-0.003, -0.002, -0.001, 0.0, 0.001, 0.002]
        laterals = [0.0, 0.00025, 0.0005, 0.0010]
        rots = [(0.0, 0.0, 0.0), (0.004, 0.0, 0.0), (0.0, 0.008, 0.0), (0.0, 0.0, 0.014)]
    elif phase == "near_gate":
        depths = [-0.002, -0.004, -0.006, -0.010]
        laterals = [0.0, 0.00025, 0.0005, 0.0010, 0.0020]
        rots = [(0.0, 0.0, 0.0), (0.008, 0.0, 0.0), (0.0, 0.016, 0.0), (0.0, 0.0, 0.030)]
    elif phase == "bridge":
        depths = [-0.010, -0.020]
        laterals = [0.0005, 0.0010, 0.0020, 0.0040]
        rots = [(0.0, 0.0, 0.0), (0.014, 0.0, 0.0), (0.0, 0.028, 0.0), (0.0, 0.0, 0.050)]
    elif phase == "heldout_40x10":
        depths = [-0.040]
        laterals = [0.0100]
        rots = [(0.0, 0.0, 0.0), (0.030, 0.0, 0.0), (0.0, 0.060, 0.0)]
    else:
        raise ValueError(f"unknown phase {phase}")
    return [Variant(phase, d, l, r) for d in depths for l in laterals for r in rots]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--base-settled-s-m", type=float, default=-0.00205)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--episodes-per-phase", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    base_episode = _episode_files(args.base_config_dir.resolve())[0]
    rng = random.Random(int(args.seed))
    manifest: dict[str, Any] = {
        "base_config_dir": str(args.base_config_dir.resolve()),
        "base_episode": str(base_episode),
        "base_settled_s_m": float(args.base_settled_s_m),
        "seed": int(args.seed),
        "phases": {},
    }
    mixed_records: list[tuple[Path, dict[str, Any]]] = []
    for phase in ("shallow_final", "near_gate", "bridge", "heldout_40x10"):
        phase_dir = output_root / phase
        (phase_dir / "episodes").mkdir(parents=True)
        variants = _phase_variants(phase)
        rng.shuffle(variants)
        count = min(int(args.episodes_per_phase), len(variants))
        if phase == "heldout_40x10":
            count = min(8, len(variants) * 2)
        records = []
        for idx in range(count):
            variant = variants[idx % len(variants)]
            lateral_sign = -1.0 if idx % 2 else 1.0
            out_path = phase_dir / "episodes" / f"episode_{idx + 1:06d}.yaml"
            record = _write_variant(
                base_episode=base_episode,
                out_path=out_path,
                variant=variant,
                base_settled_s_m=float(args.base_settled_s_m),
                lateral_sign=lateral_sign,
                rng=rng,
            )
            records.append({"episode": out_path.name, **record})
            if phase != "heldout_40x10":
                mixed_records.append((out_path, record))
        (phase_dir / "curriculum_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        manifest["phases"][phase] = {"dir": str(phase_dir), "count": len(records)}

    mixed_dir = output_root / "train_mixed_50_30_20"
    (mixed_dir / "episodes").mkdir(parents=True)
    buckets = {
        "shallow_final": [p for p, r in mixed_records if r["bucket"] == "shallow_final"],
        "near_gate": [p for p, r in mixed_records if r["bucket"] == "near_gate"],
        "bridge": [p for p, r in mixed_records if r["bucket"] == "bridge"],
    }
    ordered = []
    for cycle in range(4):
        ordered.extend(buckets["shallow_final"][cycle * 3 : cycle * 3 + 3])
        ordered.extend(buckets["near_gate"][cycle * 2 : cycle * 2 + 2])
        ordered.extend(buckets["bridge"][cycle : cycle + 1])
    ordered = ordered[:24]
    for idx, source in enumerate(ordered, start=1):
        shutil.copy2(source, mixed_dir / "episodes" / f"episode_{idx:06d}.yaml")
    manifest["phases"]["train_mixed_50_30_20"] = {
        "dir": str(mixed_dir),
        "count": len(ordered),
        "probability_note": "episode list approximates 50% shallow_final, 30% near_gate, 20% bridge for uniform episode assignment",
    }
    (output_root / "randomized_curriculum_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
