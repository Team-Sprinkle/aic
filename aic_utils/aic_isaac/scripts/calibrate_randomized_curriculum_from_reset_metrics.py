#!/usr/bin/env python3
"""Calibrate randomized start episodes from zero-action settle metrics.

This script fixes episode reset placement only.  It leaves the strict target
depth, success checker, reward, and policy path unchanged.  For each env in a
reset-validation row, it shifts the matching episode so the semantic tip's
post-step position would land at the randomized curriculum target
``(s, lateral)`` encoded in the episode metadata.
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


def _add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _scale(a: list[float], s: float) -> list[float]:
    return [a[i] * s for i in range(3)]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: list[float], *, name: str) -> list[float]:
    n = _norm(a)
    if n < 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return [x / n for x in a]


def _round(a: list[float], digits: int = 9) -> list[float]:
    return [round(float(x), digits) for x in a]


def _qnorm(q: list[float], *, name: str) -> list[float]:
    n = math.sqrt(sum(float(v) * float(v) for v in q))
    if n < 1.0e-12:
        raise ValueError(f"{name} must be non-zero")
    return [float(v) / n for v in q]


def _qconj(q: list[float]) -> list[float]:
    q = _qnorm(q, name="quaternion")
    return [q[0], -q[1], -q[2], -q[3]]


def _qmul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = _qnorm(a, name="lhs quaternion")
    bw, bx, by, bz = _qnorm(b, name="rhs quaternion")
    return _qnorm(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        name="product quaternion",
    )


def _q_to_axis_angle(q: list[float]) -> tuple[list[float], float]:
    q = _qnorm(q, name="quaternion")
    if q[0] < 0.0:
        q = [-v for v in q]
    w = max(-1.0, min(1.0, q[0]))
    angle = 2.0 * math.acos(w)
    sin_half = math.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1.0e-9 or abs(angle) < 1.0e-9:
        return [1.0, 0.0, 0.0], 0.0
    return [q[1] / sin_half, q[2] / sin_half, q[3] / sin_half], angle


def _axis_angle_to_q(axis: list[float], angle: float) -> list[float]:
    axis = _normalize(axis, name="axis-angle axis")
    half = 0.5 * float(angle)
    s = math.sin(half)
    return _qnorm([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], name="axis-angle quaternion")


def _qscale_rotation(q: list[float], gain: float) -> list[float]:
    axis, angle = _q_to_axis_angle(q)
    return _axis_angle_to_q(axis, angle * float(gain))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _row_for_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("step", -1)) == int(step):
            return row
    raise ValueError(f"metrics has no row for step {step}")


def _start(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene")
    if isinstance(scene, dict) and isinstance(scene.get("start_near_gate"), dict):
        return scene["start_near_gate"]
    raise ValueError("episode has no scene.start_near_gate mapping")


def _episode_files(config_dir: Path) -> list[Path]:
    episodes_dir = config_dir / "episodes" if (config_dir / "episodes").is_dir() else config_dir
    episodes = sorted(episodes_dir.glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml under {episodes_dir}")
    return episodes


def _body_world(row: dict[str, Any], env_id: int) -> list[float]:
    geom = row.get("post_step_insertion_geometry") or {}
    bodies = geom.get("body_world_by_env")
    if isinstance(bodies, list) and env_id < len(bodies):
        return _vec(bodies[env_id], name=f"body_world_by_env[{env_id}]")
    bodies = (row.get("post_step_selected_body_poses") or {}).get("positions_w_by_env") or {}
    tip_bodies = bodies.get("sfp_tip_link")
    if isinstance(tip_bodies, list) and env_id < len(tip_bodies):
        return _vec(tip_bodies[env_id], name=f"sfp_tip_link[{env_id}]")
    raise ValueError(f"metrics row has no semantic tip world position for env {env_id}")


def _body_quat(row: dict[str, Any], env_id: int) -> list[float] | None:
    geom = row.get("post_step_insertion_geometry") or {}
    quats = geom.get("body_orientation_wxyz_by_env")
    if isinstance(quats, list) and env_id < len(quats):
        raw = quats[env_id]
        if isinstance(raw, list) and len(raw) == 4:
            return _qnorm([float(v) for v in raw], name=f"body_orientation_wxyz_by_env[{env_id}]")
    quats = (row.get("post_step_body_frame_offsets") or {}).get("world_quat_wxyz_by_env") or {}
    tip_quats = quats.get("sfp_tip_link")
    if isinstance(tip_quats, list) and env_id < len(tip_quats):
        raw = tip_quats[env_id]
        if isinstance(raw, list) and len(raw) == 4:
            return _qnorm([float(v) for v in raw], name=f"sfp_tip_link quat[{env_id}]")
    return None


def _target_quat(row: dict[str, Any], env_id: int) -> list[float] | None:
    geom = row.get("post_step_insertion_geometry") or {}
    quats = geom.get("target_orientation_wxyz_by_env")
    if isinstance(quats, list) and env_id < len(quats):
        raw = quats[env_id]
        if isinstance(raw, list) and len(raw) == 4:
            return _qnorm([float(v) for v in raw], name=f"target_orientation_wxyz_by_env[{env_id}]")
    raw = geom.get("target_orientation_wxyz_env0")
    if env_id == 0 and isinstance(raw, list) and len(raw) == 4:
        return _qnorm([float(v) for v in raw], name="target_orientation_wxyz_env0")
    return None


def _target_tip_world(start: dict[str, Any], row: dict[str, Any], env_id: int) -> list[float]:
    variant = start.get("randomized_curriculum_variant") or {}
    target_s = float(variant.get("target_signed_depth_m", 0.0))
    requested_lateral = float(variant.get("requested_lateral_m", 0.0))
    lateral_sign = float(variant.get("lateral_sign", 1.0))
    axis = _normalize(_vec(start.get("target_gate_axis_world"), name="target_gate_axis_world"), name="target_gate_axis_world")
    lateral_dir = _normalize(_vec(start.get("lateral_direction_world"), name="lateral_direction_world"), name="lateral_direction_world")
    lateral_dir = _normalize(_sub(lateral_dir, _scale(axis, _dot(lateral_dir, axis))), name="projected lateral_direction_world")
    geom = row.get("post_step_insertion_geometry") or {}
    targets = geom.get("target_world_by_env")
    depths = geom.get("target_depth_m_by_env")
    if isinstance(targets, list) and env_id < len(targets) and isinstance(depths, list) and env_id < len(depths):
        full_target = _vec(targets[env_id], name=f"target_world_by_env[{env_id}]")
        gate = _sub(full_target, _scale(axis, float(depths[env_id])))
    else:
        gate = _vec(start.get("target_gate_position"), name="target_gate_position")
    return _add(_add(gate, _scale(axis, target_s)), _scale(lateral_dir, lateral_sign * requested_lateral))


def _shift_start(start: dict[str, Any], correction: list[float]) -> None:
    for key in (
        "body_start_position_world",
        "tcp_start_position_world",
        "reference_tip_center_position_world",
        "reference_reward_body_start_position_world",
        "reference_body_position",
        "reference_tcp_position",
    ):
        if isinstance(start.get(key), list) and len(start[key]) == 3:
            start[key] = _round(_add([float(x) for x in start[key]], correction))
    if isinstance(start.get("reset_body_offset_from_reference_world"), list) and len(start["reset_body_offset_from_reference_world"]) == 3:
        start["reset_body_offset_from_reference_world"] = _round(
            _add([float(x) for x in start["reset_body_offset_from_reference_world"]], correction)
        )


def _calibrate_orientation(
    start: dict[str, Any],
    row: dict[str, Any],
    env_id: int,
    *,
    orientation_gain: float,
) -> dict[str, Any] | None:
    desired = _target_quat(row, env_id)
    current_tip = _body_quat(row, env_id)
    reset_q = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
    if (
        desired is None
        or current_tip is None
        or not isinstance(reset_q, list)
        or len(reset_q) != 4
    ):
        return None
    correction_full = _qmul(desired, _qconj(current_tip))
    correction = _qscale_rotation(correction_full, float(orientation_gain))
    calibrated = _qmul(correction, [float(v) for v in reset_q])
    for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
        if isinstance(start.get(key), list) and len(start[key]) == 4:
            start[key] = _round(calibrated)
    return {
        "desired_tip_quat_wxyz": _round(desired),
        "actual_tip_quat_wxyz": _round(current_tip),
        "orientation_gain": float(orientation_gain),
        "orientation_correction_full_wxyz": _round(correction_full),
        "orientation_correction_wxyz": _round(correction),
        "calibrated_reset_quat_wxyz": _round(calibrated),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config-dir", required=True, type=Path)
    parser.add_argument("--validation-run", required=True, type=Path)
    parser.add_argument("--output-config-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--max-envs", type=int, default=0)
    parser.add_argument(
        "--position-gain",
        type=float,
        default=1.0,
        help="Scale the post-step semantic tip position correction before applying it to reset/reference positions.",
    )
    parser.add_argument(
        "--calibrate-orientation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also left-multiply each reset-body quaternion by desired_tip_quat * inv(actual_post_step_tip_quat).",
    )
    parser.add_argument(
        "--orientation-gain",
        type=float,
        default=1.0,
        help="Scale the orientation correction in axis-angle space when --calibrate-orientation is enabled.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = args.input_config_dir.resolve()
    output = args.output_config_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)

    row = _row_for_step(_jsonl(args.validation_run.resolve() / "metrics.jsonl"), int(args.step))
    episodes = _episode_files(source)
    max_envs = int(args.max_envs) if int(args.max_envs) > 0 else len(episodes)
    records: list[dict[str, Any]] = []
    for env_id, episode in enumerate(episodes[:max_envs]):
        data = yaml.safe_load(episode.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{episode} is not a YAML mapping")
        start = _start(data)
        actual = _body_world(row, env_id)
        target = _target_tip_world(start, row, env_id)
        correction_full = _sub(target, actual)
        correction = _scale(correction_full, float(args.position_gain))
        _shift_start(start, correction)
        orientation_record = (
            _calibrate_orientation(start, row, env_id, orientation_gain=float(args.orientation_gain))
            if bool(args.calibrate_orientation)
            else None
        )
        start["post_step_randomized_curriculum_calibration"] = {
            "method": "shift reset/reference positions to post-step semantic tip target; optionally left-multiply reset orientation by post-step tip error",
            "source_validation_run": str(args.validation_run.resolve()),
            "source_step": int(args.step),
            "env_id": int(env_id),
            "actual_tip_world": _round(actual),
            "target_tip_world": _round(target),
            "body_start_world_shift_full": _round(correction_full),
            "position_gain": float(args.position_gain),
            "body_start_world_shift": _round(correction),
            "orientation_calibration": orientation_record,
        }
        out_path = output / "episodes" / f"episode_{env_id + 1:06d}.yaml"
        data["episode_id"] = f"{data.get('episode_id', episode.stem)}_poststepcal"
        out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "source_episode": str(episode),
                "output_episode": str(out_path),
                "env_id": int(env_id),
                "actual_tip_world": actual,
                "target_tip_world": target,
                "body_start_world_shift_full": correction_full,
                "position_gain": float(args.position_gain),
                "body_start_world_shift": correction,
                "orientation_calibration": orientation_record,
            }
        )

    for path in source.iterdir():
        if path.name == "episodes":
            continue
        dst = output / path.name
        if path.is_dir():
            shutil.copytree(path, dst)
        elif path.is_file():
            shutil.copy2(path, dst)
    (output / "post_step_calibration_summary.json").write_text(
        json.dumps({"input_config_dir": str(source), "validation_run": str(args.validation_run.resolve()), "step": int(args.step), "episodes": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_config_dir": str(output), "episode_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
