#!/usr/bin/env python3
"""Build stateful insertion curriculum episodes from a high-level YAML config."""

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
    if not isinstance(raw, (list, tuple)) or len(raw) != length:
        raise ValueError(f"{name} must be a {length}-value list")
    return [float(v) for v in raw]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * float(s) for i in range(3)]


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


def _qmul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _qconj(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _qapply(q: list[float], v: list[float]) -> list[float]:
    qv = [0.0, float(v[0]), float(v[1]), float(v[2])]
    out = _qmul(_qmul(q, qv), _qconj(q))
    return out[1:4]


def _qpow_shortest(q: list[float], exponent: float) -> list[float]:
    qn = _qnormalize(q)
    if qn[0] < 0.0:
        qn = [-v for v in qn]
    w = max(-1.0, min(1.0, qn[0]))
    vnorm = math.sqrt(qn[1] * qn[1] + qn[2] * qn[2] + qn[3] * qn[3])
    if vnorm < 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vnorm, w)
    scaled = 0.5 * float(exponent) * angle
    s = math.sin(scaled) / vnorm
    return _qnormalize([math.cos(scaled), qn[1] * s, qn[2] * s, qn[3] * s])


def _axis_angle_quat(axis: list[float], angle: float) -> list[float]:
    axis_n = _normalize(axis, name="orientation perturbation axis")
    half = 0.5 * float(angle)
    return [math.cos(half), *(math.sin(half) * v for v in axis_n)]


def _quat_between_vectors(src: list[float], dst: list[float]) -> list[float]:
    a = _normalize(src, name="source orientation axis")
    b = _normalize(dst, name="target orientation axis")
    dot = max(-1.0, min(1.0, _dot(a, b)))
    if dot > 1.0 - 1.0e-10:
        return [1.0, 0.0, 0.0, 0.0]
    if dot < -1.0 + 1.0e-10:
        e1, _ = _orthonormal_basis(a, [1.0, 0.0, 0.0])
        return _axis_angle_quat(e1, math.pi)
    cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
    return _qnormalize([1.0 + dot, cross[0], cross[1], cross[2]])


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
    episodes_dir = config_dir / "episodes" if (config_dir / "episodes").is_dir() else config_dir
    episodes = sorted(episodes_dir.glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml files under {episodes_dir}")
    return episodes


def _start(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("start_near_gate"), dict):
        return scene["start_near_gate"]
    raise ValueError("episode has no scene.start_near_gate mapping")


def _range_value(cfg: dict[str, Any], name: str, t: float) -> float:
    raw = cfg.get(name)
    if isinstance(raw, dict):
        start = raw.get("initial", raw.get("start"))
        end = raw.get("terminal", raw.get("end"))
        if start is None or end is None:
            raise ValueError(f"{name} must define initial/start and terminal/end")
        return float(start) + float(t) * (float(end) - float(start))
    if raw is None:
        raise ValueError(f"missing range field: {name}")
    return float(raw)


def _write_episode(
    *,
    base_episode: Path,
    out_path: Path,
    episode_index: int,
    episode_count: int,
    cfg: dict[str, Any],
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
    start_cfg = cfg["start_near_gate"]
    axial_distance = _range_value(start_cfg, "axial_distance_m", t)
    signed_axial = float(axial_distance) if bool(start_cfg.get("signed_axial_distance", False)) else -abs(axial_distance)
    lateral = _range_value(start_cfg, "lateral_distance_m", t)
    theta = _range_value(start_cfg, "orientation_error_rad", t)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    lateral_dir = _normalize(_add(_scale(e1, math.cos(angle)), _scale(e2, math.sin(angle))), name="random lateral direction")
    orientation_axis = _normalize(_add(_scale(e1, math.cos(angle + math.pi / 2.0)), _scale(e2, math.sin(angle + math.pi / 2.0))), name="orientation perturbation axis")
    reference = _add(_add(gate, _scale(axis, signed_axial)), _scale(lateral_dir, lateral))
    lowtheta = start.get("lowtheta_metric_reset") or {}
    reset_offset_raw = lowtheta.get("metric_reset_body_minus_tip_world_m") or start.get("reset_body_offset_from_reference_world")
    reset_offset = _vec(reset_offset_raw, name="reset body offset")
    body = _add(reference, reset_offset)
    base_quat_raw = lowtheta.get("reset_body_quat_wxyz") or start.get("body_start_orientation_wxyz")
    base_quat = _qnormalize(_vec(base_quat_raw, name="reset orientation", length=4))
    base_alignment_error = None
    alignment_correction = [1.0, 0.0, 0.0, 0.0]
    if bool(start_cfg.get("align_orientation_axis_at_zero", False)):
        source_tip_quat_raw = lowtheta.get("source_tip_quat_wxyz")
        if source_tip_quat_raw is None:
            raise ValueError("align_orientation_axis_at_zero requires lowtheta_metric_reset.source_tip_quat_wxyz")
        source_tip_quat = _qnormalize(_vec(source_tip_quat_raw, name="source tip orientation", length=4))
        local_axis = _normalize(
            _vec(start_cfg.get("orientation_axis_local", [0.0, 0.0, 1.0]), name="orientation_axis_local"),
            name="orientation_axis_local",
        )
        source_axis = _normalize(_qapply(source_tip_quat, local_axis), name="source tip insertion axis")
        base_alignment_error = math.acos(max(-1.0, min(1.0, _dot(source_axis, axis))))
        alignment_correction = _quat_between_vectors(source_axis, axis)
        blend = max(0.0, min(1.0, float(start_cfg.get("orientation_alignment_blend", 1.0))))
        alignment_correction = _qpow_shortest(alignment_correction, blend)
        base_quat = _qnormalize(_qmul(alignment_correction, base_quat))
        reset_offset = _qapply(alignment_correction, reset_offset)
        body = _add(reference, reset_offset)
    quat = _qnormalize(_qmul(_axis_angle_quat(orientation_axis, theta), base_quat))
    settle_compensation = start_cfg.get("settle_compensation") or {}
    settle_shift = [0.0, 0.0, 0.0]
    settle_orientation_correction = [1.0, 0.0, 0.0, 0.0]
    if isinstance(settle_compensation, dict):
        if settle_compensation.get("position_shift_world_m") is not None:
            settle_shift = _vec(settle_compensation.get("position_shift_world_m"), name="settle_compensation.position_shift_world_m")
            body = _add(body, settle_shift)
        if settle_compensation.get("orientation_correction_wxyz") is not None:
            settle_orientation_correction = _qnormalize(
                _vec(
                    settle_compensation.get("orientation_correction_wxyz"),
                    name="settle_compensation.orientation_correction_wxyz",
                    length=4,
                )
            )
            quat = _qnormalize(_qmul(settle_orientation_correction, quat))

    for key in ("reference_tip_center_position_world", "reference_reward_body_start_position_world", "reference_tcp_position", "reference_body_position"):
        if key in start:
            start[key] = _round(reference)
    for key in ("body_start_position_world", "tcp_start_position_world"):
        if key in start:
            start[key] = _round(body)
    for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
        if key in start:
            start[key] = _round(quat)
    start["reset_body_offset_from_reference_world"] = _round(reset_offset)
    start["lateral_direction_world"] = _round(lateral_dir)
    start["axial_distance_m"] = round(signed_axial, 9)
    start["lateral_distance_m"] = round(lateral, 9)
    start["achieved_axial_distance_m"] = round(signed_axial, 9)
    start["achieved_lateral_distance_m"] = round(lateral, 9)
    start["stateful_curriculum_variant"] = {
        "source_episode": str(base_episode),
        "episode_index": episode_index,
        "episode_count": episode_count,
        "interpolation_t": round(t, 9),
        "requested_axial_distance_m": round(axial_distance, 9),
        "target_signed_depth_m": round(signed_axial, 9),
        "signed_axial_distance": bool(start_cfg.get("signed_axial_distance", False)),
        "requested_lateral_m": round(lateral, 9),
        "requested_orientation_error_rad": round(theta, 9),
        "align_orientation_axis_at_zero": bool(start_cfg.get("align_orientation_axis_at_zero", False)),
        "orientation_alignment_blend": round(float(start_cfg.get("orientation_alignment_blend", 1.0)), 9),
        "source_axis_alignment_error_rad": (
            None if base_alignment_error is None else round(base_alignment_error, 12)
        ),
        "alignment_correction_quat_wxyz": _round(alignment_correction, digits=12),
        "lateral_direction_world": _round(lateral_dir),
        "orientation_perturbation_axis_world": _round(orientation_axis),
        "progression_policy": cfg.get("progression", {}).get("policy", "scheduled"),
        "settle_compensation_position_shift_world_m": _round(settle_shift),
        "settle_compensation_orientation_correction_wxyz": _round(settle_orientation_correction, digits=12),
    }
    data["episode_id"] = f"{data.get('episode_id', base_episode.stem)}_stateful_{episode_index:06d}"
    data["episode_index"] = episode_index
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return start["stateful_curriculum_variant"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to a mapping")
    output_root = (args.output_root or Path(cfg["output_root"])).resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists; pass --overwrite")
        shutil.rmtree(output_root)
    (output_root / "episodes").mkdir(parents=True)
    episodes = int(cfg.get("episodes", 1000))
    rng = random.Random(int(cfg.get("seed", 20260613)))
    base_episodes = _episode_files(Path(cfg["base_config_dir"]).resolve())
    records = []
    for idx in range(1, episodes + 1):
        base_episode = base_episodes[(idx - 1) % len(base_episodes)]
        record = _write_episode(
            base_episode=base_episode,
            out_path=output_root / "episodes" / f"episode_{idx:06d}.yaml",
            episode_index=idx,
            episode_count=episodes,
            cfg=cfg,
            rng=rng,
        )
        records.append({"episode": f"episode_{idx:06d}.yaml", **record})
    manifest = {
        "config": cfg,
        "output_root": str(output_root),
        "episodes": episodes,
        "base_episode_count": len(base_episodes),
        "records": records,
    }
    (output_root / "stateful_curriculum_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
