#!/usr/bin/env python3
"""Build candidate low-theta reset episodes from a measured successful partial.

The script uses a measured semantic-tip pose from a metrics row and the
calibrated tip-in-reset-body transform stored in an existing episode YAML.  It
then solves the wrist/reset-body pose that should place the semantic tip at the
requested signed depth while preserving the measured low-theta orientation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


def _vec(raw: Any, length: int, name: str) -> list[float]:
    if not isinstance(raw, list | tuple) or len(raw) != length:
        raise ValueError(f"{name} must be a {length}-value list")
    return [float(v) for v in raw]


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _unit(v: list[float]) -> list[float]:
    n = _norm(v)
    if n < 1.0e-12:
        raise ValueError("cannot normalize near-zero vector")
    return [x / n for x in v]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _scale(v: list[float], s: float) -> list[float]:
    return [x * s for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _round(values: list[float], digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in values]


def _qnorm(q: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in q))
    if n < 1.0e-12:
        raise ValueError("cannot normalize near-zero quaternion")
    return [x / n for x in q]


def _qconj(q: list[float]) -> list[float]:
    qw, qx, qy, qz = _qnorm(q)
    return [qw, -qx, -qy, -qz]


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


def _qrot(q: list[float], v: list[float]) -> list[float]:
    qw, qx, qy, qz = _qnorm(q)
    vx, vy, vz = v
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _episode_file(config_dir: Path) -> Path:
    episodes = sorted((config_dir / "episodes").glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml under {config_dir / 'episodes'}")
    return episodes[0]


def _start(data: dict[str, Any]) -> dict[str, Any]:
    start = ((data.get("scene") or {}).get("start_near_gate") or {})
    if not isinstance(start, dict):
        raise ValueError("episode has no scene.start_near_gate mapping")
    return start


def _metric_row(metrics_jsonl: Path, step: int | None) -> dict[str, Any]:
    rows = [json.loads(line) for line in metrics_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{metrics_jsonl} has no metric rows")
    if step is not None:
        for row in rows:
            if int(row.get("step", -1)) == int(step):
                return row
        raise ValueError(f"no row with step={step}")
    return max(rows, key=lambda row: float((row.get("post_step_insertion_geometry") or {}).get("signed_depth_m_env0", -999.0)))


def _by_env(raw: Any, env_index: int, name: str, length: int) -> list[float]:
    if isinstance(raw, list) and env_index < len(raw) and isinstance(raw[env_index], list):
        return _vec(raw[env_index], length, name)
    if env_index == 0 and isinstance(raw, list) and raw and not isinstance(raw[0], list):
        return _vec(raw, length, name)
    raise ValueError(f"{name} missing env index {env_index}")


def _metric_tip_world_quat(row: dict[str, Any], geom: dict[str, Any], env_index: int) -> list[float]:
    actual = row.get("actual_body_orientation_wxyz_by_env")
    if isinstance(actual, dict) and isinstance(actual.get("sfp_tip_link"), list):
        return _by_env(actual["sfp_tip_link"], env_index, "actual sfp_tip_link world quaternion", 4)
    frame_offsets = row.get("post_step_body_frame_offsets")
    if isinstance(frame_offsets, dict):
        world_quats = frame_offsets.get("world_quat_wxyz_by_env")
        if isinstance(world_quats, dict):
            tip_quats = world_quats.get("sfp_tip_link")
            if isinstance(tip_quats, list):
                return _by_env(tip_quats, env_index, "post_step sfp_tip_link world quaternion", 4)
    # Older metrics may only carry the selected geometry orientation.  Keep this
    # fallback for compatibility, but prefer the explicit body-frame world pose
    # above because body_orientation_wxyz_by_env can be a checker-frame value.
    if env_index != 0:
        raise ValueError("metric row has no per-env tip quaternion for env_index != 0")
    return _vec((geom.get("body_orientation_wxyz_by_env") or [None])[0], 4, "metric tip quat")


def _metric_body_pose(row: dict[str, Any], body_name: str, env_index: int) -> tuple[list[float], list[float]] | None:
    actual_pos = row.get("actual_body_position_world_by_env")
    actual_quat = row.get("actual_body_orientation_wxyz_by_env")
    if isinstance(actual_pos, dict) and isinstance(actual_quat, dict):
        body_positions = actual_pos.get(body_name)
        body_quats = actual_quat.get(body_name)
        if isinstance(body_positions, list) and isinstance(body_quats, list):
            return (
                _by_env(body_positions, env_index, f"actual {body_name} world position", 3),
                _by_env(body_quats, env_index, f"actual {body_name} world quaternion", 4),
            )
    frame_offsets = row.get("post_step_body_frame_offsets")
    if not isinstance(frame_offsets, dict):
        return None
    world_pos = frame_offsets.get("world_position_by_env")
    world_quat = frame_offsets.get("world_quat_wxyz_by_env")
    if not isinstance(world_pos, dict) or not isinstance(world_quat, dict):
        return None
    body_positions = world_pos.get(body_name)
    body_quats = world_quat.get(body_name)
    if isinstance(body_positions, list) and isinstance(body_quats, list):
        return (
            _by_env(body_positions, env_index, f"post_step {body_name} world position", 3),
            _by_env(body_quats, env_index, f"post_step {body_name} world quaternion", 4),
        )
    return None


def _tip_calibration(start: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        start.get("calibrated_from_reset_diagnostic"),
        start.get("semantic_tip_transform_repair"),
        start.get("semantic_tip_transform"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("measured_tip_pos_in_reset_body") is not None and candidate.get("measured_tip_quat_in_reset_body") is not None:
            return candidate
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("measured_tip_pos_in_reset_body") is not None:
            raise ValueError("found measured_tip_pos_in_reset_body but no measured_tip_quat_in_reset_body")
    raise ValueError("no tip-in-reset-body calibration found in start_near_gate metadata")


def _lateral_basis(axis: list[float], preferred: list[float], tip: list[float], entrance: list[float]) -> list[float]:
    delta = _sub(tip, entrance)
    residual = _sub(delta, _scale(axis, _dot(delta, axis)))
    if _norm(residual) > 1.0e-6:
        return _unit(residual)
    preferred_residual = _sub(preferred, _scale(axis, _dot(preferred, axis)))
    return _unit(preferred_residual)


def _env_origin_delta(row: dict[str, Any], env_index: int, spacing: float) -> list[float]:
    if env_index == 0:
        return [0.0, 0.0, 0.0]
    actual = row.get("actual_body_position_world_by_env")
    if not isinstance(actual, dict) or not isinstance(actual.get("sfp_tip_link"), list):
        raise ValueError("--env-index requires actual_body_position_world_by_env.sfp_tip_link")
    tip0 = _by_env(actual["sfp_tip_link"], 0, "actual sfp_tip_link env0 position", 3)
    tipn = _by_env(actual["sfp_tip_link"], env_index, "actual sfp_tip_link selected env position", 3)
    raw = _sub(tipn, tip0)
    if spacing <= 0.0:
        return raw
    # Isaac cloned-env origins are laid out on a regular grid.  Round x/y to
    # the nearest env-spacing multiple so local pose differences in the sweep do
    # not get baked into the episode-local reset coordinates.
    return [round(raw[0] / spacing) * spacing, round(raw[1] / spacing) * spacing, 0.0]


def _selected_env_origin(
    row: dict[str, Any],
    start: dict[str, Any],
    geom: dict[str, Any],
    env_index: int,
    spacing: float,
) -> tuple[list[float], list[float], list[float]]:
    entrance_env0 = _vec(geom.get("entrance_world_env0") or start.get("target_gate_position"), 3, "entrance")
    base_entrance = _vec(start.get("target_gate_position"), 3, "target_gate_position")
    env0_origin = _sub(entrance_env0, base_entrance)
    env_delta = _env_origin_delta(row, env_index, spacing)
    return _add(env0_origin, env_delta), env0_origin, env_delta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--metrics-jsonl", required=True, type=Path)
    parser.add_argument("--metric-step", type=int)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument(
        "--env-spacing-m",
        type=float,
        default=4.0,
        help="Cloned Isaac env spacing used to convert selected-env world poses back to episode-local coordinates.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-s-mm", type=float, nargs="+", default=[-2.0, 0.0, 4.0, 8.0])
    parser.add_argument("--lateral-mm", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument(
        "--pose-source",
        choices=["metric_reset_body", "calibrated_tip_transform"],
        default="metric_reset_body",
        help=(
            "How to reconstruct the reset body pose. metric_reset_body preserves the measured "
            "post-step reset-body-to-tip relation from metrics when available; calibrated_tip_transform "
            "uses the episode's stored semantic tip calibration."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_dir} exists; pass --overwrite")
        shutil.rmtree(output_dir)
    (output_dir / "episodes").mkdir(parents=True)

    base_episode = _episode_file(args.base_config_dir.resolve())
    base = yaml.safe_load(base_episode.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base episode must parse to a mapping")
    start = _start(base)

    row = _metric_row(args.metrics_jsonl.resolve(), args.metric_step)
    geom = row.get("post_step_insertion_geometry") or {}
    env_index = int(args.env_index)
    actual_pos = row.get("actual_body_position_world_by_env")
    if isinstance(actual_pos, dict) and isinstance(actual_pos.get("sfp_tip_link"), list):
        tip_world = _by_env(actual_pos["sfp_tip_link"], env_index, "actual sfp_tip_link world position", 3)
    else:
        if env_index != 0:
            raise ValueError("metric row has no per-env tip world position for env_index != 0")
        tip_world = _vec(geom.get("body_world_env0") or (geom.get("body_world_by_env") or [None])[0], 3, "metric tip world")
    tip_quat_world = _metric_tip_world_quat(row, geom, env_index)
    env_origin, env0_origin, env_delta = _selected_env_origin(row, start, geom, env_index, float(args.env_spacing_m))
    entrance_env0 = _vec(geom.get("entrance_world_env0") or start.get("target_gate_position"), 3, "entrance")
    entrance = _add(entrance_env0, env_delta)
    axis = _unit(_vec(geom.get("axis_world_env0") or start.get("target_gate_axis_world"), 3, "axis"))
    preferred_lat = _vec(start.get("lateral_direction_world"), 3, "lateral_direction_world")
    lateral_dir = _lateral_basis(axis, preferred_lat, tip_world, entrance)

    reset_body_name = str(start.get("reset_body_name") or "wrist_3_link")
    measured_body_pose = _metric_body_pose(row, reset_body_name, env_index)
    use_metric_body_pose = str(args.pose_source) == "metric_reset_body" and measured_body_pose is not None
    if use_metric_body_pose:
        metric_body_world, reset_body_quat = measured_body_pose
        metric_body_minus_tip = _sub(metric_body_world, tip_world)
    else:
        repair = _tip_calibration(start)
        tip_in_body = _vec(repair.get("measured_tip_pos_in_reset_body"), 3, "measured_tip_pos_in_reset_body")
        tip_quat_in_body = _vec(repair.get("measured_tip_quat_in_reset_body"), 4, "measured_tip_quat_in_reset_body")
        reset_body_quat = _qmul(tip_quat_world, _qconj(tip_quat_in_body))
        rotated_tip_local = _qrot(reset_body_quat, tip_in_body)
    records: list[dict[str, Any]] = []
    idx = 1
    for s_mm in args.target_s_mm:
        for lat_mm in args.lateral_mm:
            for sign in (1.0, -1.0) if lat_mm > 0 else (1.0,):
                data = yaml.safe_load(base_episode.read_text(encoding="utf-8"))
                child_start = _start(data)
                target_tip = _add(
                    _add(entrance, _scale(axis, float(s_mm) / 1000.0)),
                    _scale(lateral_dir, sign * float(lat_mm) / 1000.0),
                )
                body_pos = _add(target_tip, metric_body_minus_tip) if use_metric_body_pose else _sub(target_tip, rotated_tip_local)
                body_pos_local = _sub(body_pos, env_origin)
                target_tip_local = _sub(target_tip, env_origin)
                child_start["reset_mode"] = "body_start_position_world"
                child_start["reset_body_name"] = reset_body_name
                child_start["body_start_position_world"] = _round(body_pos_local)
                child_start["tcp_start_position_world"] = _round(body_pos_local)
                child_start["body_start_orientation_wxyz"] = _round(reset_body_quat)
                child_start["reset_body_orientation_wxyz"] = _round(reset_body_quat)
                child_start["tcp_start_orientation_world"] = _round(reset_body_quat)
                child_start["reference_tip_center_position_world"] = _round(target_tip_local)
                child_start["reference_body_position"] = _round(target_tip_local)
                child_start["reference_tcp_position"] = _round(target_tip_local)
                child_start["reference_reward_body_start_position_world"] = _round(target_tip_local)
                child_start["reset_body_offset_from_reference_world"] = _round(_sub(body_pos_local, target_tip_local))
                target_delta = _sub(target_tip, entrance)
                signed_depth = _dot(target_delta, axis)
                signed_lateral = _dot(target_delta, lateral_dir)
                lateral_residual = _sub(target_delta, _scale(axis, signed_depth))
                child_start["axial_distance_m"] = round(signed_depth, 9)
                child_start["lateral_distance_m"] = round(signed_lateral, 9)
                child_start["achieved_distance"] = round(_norm(target_delta), 9)
                child_start["achieved_axial_distance_m"] = round(signed_depth, 9)
                child_start["achieved_lateral_distance_m"] = round(_norm(lateral_residual), 9)
                child_start.pop("randomized_curriculum_variant", None)
                child_start.pop("axis_shift_calibration", None)
                child_start["lowtheta_metric_reset"] = {
                    "source_metrics": str(args.metrics_jsonl.resolve()),
                    "source_step": int(row.get("step", -1)),
                    "source_env_index": env_index,
                    "env0_origin_world_m": _round(env0_origin),
                    "selected_env_delta_world_m": _round(env_delta),
                    "selected_env_origin_world_m": _round(env_origin),
                    "source_tip_signed_depth_m": (
                        (geom.get("signed_depth_m_by_env") or [geom.get("signed_depth_m_env0")])[env_index]
                        if isinstance(geom.get("signed_depth_m_by_env"), list)
                        and len(geom.get("signed_depth_m_by_env")) > env_index
                        else geom.get("signed_depth_m_env0")
                    ),
                    "source_tip_lateral_error_m": (
                        (geom.get("lateral_error_m_by_env") or [geom.get("lateral_error_m_env0")])[env_index]
                        if isinstance(geom.get("lateral_error_m_by_env"), list)
                        and len(geom.get("lateral_error_m_by_env")) > env_index
                        else geom.get("lateral_error_m_env0")
                    ),
                    "source_tip_orientation_error_rad": (
                        (geom.get("orientation_error_rad_by_env") or [geom.get("orientation_error_rad_env0")])[env_index]
                        if isinstance(geom.get("orientation_error_rad_by_env"), list)
                        and len(geom.get("orientation_error_rad_by_env")) > env_index
                        else geom.get("orientation_error_rad_env0")
                    ),
                    "source_tip_world": _round(tip_world),
                    "source_tip_quat_wxyz": _round(tip_quat_world),
                    "source_tip_quat_source": (
                        "post_step_body_frame_offsets.world_quat_wxyz_by_env.sfp_tip_link"
                        if isinstance(row.get("post_step_body_frame_offsets"), dict)
                        and isinstance((row["post_step_body_frame_offsets"].get("world_quat_wxyz_by_env") or {}).get("sfp_tip_link"), list)
                        else "actual_body_orientation_wxyz_by_env.sfp_tip_link"
                    ),
                    "target_signed_depth_m": round(float(s_mm) / 1000.0, 9),
                    "target_lateral_m": round(sign * float(lat_mm) / 1000.0, 9),
                    "target_tip_world": _round(target_tip),
                    "target_tip_episode_local": _round(target_tip_local),
                    "reset_body_quat_wxyz": _round(reset_body_quat),
                    "reset_body_episode_local": _round(body_pos_local),
                    "pose_source": str(args.pose_source),
                    "metric_reset_body_pose_available": bool(measured_body_pose is not None),
                    "metric_reset_body_minus_tip_world_m": _round(metric_body_minus_tip) if use_metric_body_pose else None,
                    "method": (
                        "body_start = target_tip + measured(metric_reset_body_world - tip_world); q_reset = measured metric reset-body world quat"
                        if use_metric_body_pose
                        else "body_start = target_tip - q_reset * measured_tip_pos_in_reset_body; q_reset = measured_lowtheta_tip_quat * inv(tip_quat_in_reset_body)"
                    ),
                }
                data["episode_id"] = f"{data.get('episode_id', base_episode.stem)}_lowtheta_metric_s{s_mm:g}_lat{sign * lat_mm:g}_{idx:06d}"
                out = output_dir / "episodes" / f"episode_{idx:06d}.yaml"
                out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
                records.append({"episode": out.name, **child_start["lowtheta_metric_reset"]})
                idx += 1
    manifest = {
        "base_episode": str(base_episode),
        "metrics_jsonl": str(args.metrics_jsonl.resolve()),
        "metric_step": row.get("step"),
        "pose_source": str(args.pose_source),
        "episode_count": len(records),
        "records": records,
    }
    (output_dir / "lowtheta_metric_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("base_episode", "metrics_jsonl", "metric_step", "episode_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
