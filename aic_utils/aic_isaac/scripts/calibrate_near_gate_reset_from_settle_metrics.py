#!/usr/bin/env python3
"""Calibrate near-gate reset body poses from post-step settle metrics.

The episode generator can place the semantic tip correctly immediately after
IK reset, while the first zero/action settle step still moves the cable.  This
script reads a reset diagnostic and a metrics row, estimates where
``sfp_tip_link`` landed after that settle step, and shifts the reset-body pose
so the post-step tip lands on the episode's semantic reference point.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def _vec(value: Any, *, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-value list")
    return [float(v) for v in value]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _round(values: list[float], digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in values]


def _qnorm(q: list[float]) -> list[float]:
    norm = sum(float(v) * float(v) for v in q) ** 0.5
    if norm < 1.0e-12:
        raise ValueError("cannot normalize near-zero quaternion")
    return [float(v) / norm for v in q]


def _qconj(q: list[float]) -> list[float]:
    q = _qnorm(q)
    return [q[0], -q[1], -q[2], -q[3]]


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


def _qapply(q: list[float], v: list[float]) -> list[float]:
    def raw_mul(lhs: list[float], rhs: list[float]) -> list[float]:
        lw, lx, ly, lz = lhs
        rw, rx, ry, rz = rhs
        return [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]

    q = _qnorm(q)
    rotated = raw_mul(raw_mul(q, [0.0, *[float(x) for x in v]]), _qconj(q))
    return [float(x) for x in rotated[1:]]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _row_for_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    for row in rows:
        try:
            if int(row.get("step")) == int(step):
                return row
        except (TypeError, ValueError):
            pass
    raise ValueError(f"metrics has no row for step {step}")


def _start_mapping(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    target = scene.get("target") if isinstance(scene.get("target"), dict) else {}
    for value in (target.get("start_near_gate"), scene.get("start_near_gate"), data.get("start_near_gate")):
        if isinstance(value, dict):
            return value
    raise ValueError("episode has no start_near_gate mapping")


def _reference_tip(start: dict[str, Any]) -> list[float]:
    for key in (
        "reference_reward_body_start_position_world",
        "reference_tip_center_position_world",
        "reference_body_position",
        "reference_tcp_position",
    ):
        value = start.get(key)
        if isinstance(value, list) and len(value) == 3:
            return [float(v) for v in value]
    raise ValueError("start_near_gate has no semantic tip reference position")


def _geometry_settle_correction(start: dict[str, Any], metrics_row: dict[str, Any], *, env_id: int) -> dict[str, Any]:
    geom = metrics_row.get("post_step_insertion_geometry")
    if not isinstance(geom, dict):
        raise ValueError("metrics row has no post_step_insertion_geometry")
    axis = _vec(start.get("target_gate_axis_world"), name="target_gate_axis_world")
    gate = _vec(start.get("target_gate_position"), name="target_gate_position")
    reference = _reference_tip(start)

    metric_axis = _vec(geom.get("axis_world_env0"), name="post_step_insertion_geometry.axis_world_env0")
    metric_entrance = _vec(geom.get("entrance_world_env0"), name="post_step_insertion_geometry.entrance_world_env0")
    body_by_env = geom.get("body_world_by_env")
    s_by_env = geom.get("signed_depth_m_by_env")
    if not isinstance(body_by_env, list) or not body_by_env:
        raise ValueError("post_step_insertion_geometry has no body_world_by_env")
    if not isinstance(s_by_env, list) or not s_by_env:
        raise ValueError("post_step_insertion_geometry has no signed_depth_m_by_env")
    idx = min(max(int(env_id), 0), len(body_by_env) - 1, len(s_by_env) - 1)
    metric_body = _vec(body_by_env[idx], name=f"post_step_insertion_geometry.body_world_by_env[{idx}]")
    settled_s = float(s_by_env[idx])

    # The metric positions include the Isaac env origin, while episode YAML
    # reference poses do not.  Compute only the insertion-frame residual from
    # the metric world values; the env origin cancels in body - entrance.
    settled_residual = _sub(_sub(metric_body, metric_entrance), [settled_s * v for v in metric_axis])
    settled_canonical = _add(_add(gate, [settled_s * v for v in axis]), settled_residual)
    correction = _sub(reference, settled_canonical)
    return {
        "correction": correction,
        "reference_tip_world": reference,
        "settled_tip_world": settled_canonical,
        "settled_signed_depth_m": settled_s,
        "settled_lateral_residual_world": settled_residual,
        "metric_env_id": idx,
    }


def _settled_tip_by_env(reset_diagnostic: dict[str, Any], metrics_row: dict[str, Any], *, tip_body: str) -> list[list[float]]:
    reset_positions = (reset_diagnostic.get("actual_body_position_world_by_env") or {}).get(tip_body)
    if not isinstance(reset_positions, list):
        positions = (metrics_row.get("actual_body_position_world_by_env") or {}).get(tip_body)
        if not isinstance(positions, list):
            positions = ((metrics_row.get("post_step_selected_body_poses") or {}).get("positions_w_by_env") or {}).get(
                tip_body
            )
        if isinstance(positions, list):
            return [_vec(pos, name=f"settled {tip_body}[{idx}]") for idx, pos in enumerate(positions)]
        raise ValueError(f"reset diagnostic and metrics row have no positions for {tip_body}")
    motion = metrics_row.get("realized_body_motion")
    if not isinstance(motion, dict) or tip_body not in motion:
        raise ValueError(f"metrics row has no realized_body_motion for {tip_body}")
    deltas = motion[tip_body].get("delta_world_by_env")
    if not isinstance(deltas, list):
        raise ValueError(f"metrics row has no delta_world_by_env for {tip_body}")
    settled: list[list[float]] = []
    for idx, reset_pos_raw in enumerate(reset_positions):
        reset_pos = _vec(reset_pos_raw, name=f"reset {tip_body}[{idx}]")
        delta = _vec(deltas[min(idx, len(deltas) - 1)], name=f"delta {tip_body}[{idx}]")
        settled.append(_add(reset_pos, delta))
    return settled


def _settled_tip_quat(metrics_row: dict[str, Any], *, tip_body: str, env_id: int) -> list[float] | None:
    orientations = (metrics_row.get("actual_body_orientation_wxyz_by_env") or {}).get(tip_body)
    if not isinstance(orientations, list) or not orientations:
        orientations = (
            (metrics_row.get("post_step_selected_body_poses") or {}).get("orientations_wxyz_by_env") or {}
        ).get(tip_body)
    if not isinstance(orientations, list) or not orientations:
        orientations = (metrics_row.get("post_step_insertion_geometry") or {}).get("body_orientation_wxyz_by_env")
    if not isinstance(orientations, list) or not orientations:
        return None
    quat = orientations[min(int(env_id), len(orientations) - 1)]
    if not isinstance(quat, list) or len(quat) != 4:
        return None
    return _qnorm([float(x) for x in quat])


def calibrate_episode(
    episode_path: Path,
    out_path: Path,
    *,
    reset_diagnostic: dict[str, Any],
    metrics_row: dict[str, Any],
    env_id: int,
    tip_body: str,
    calibrate_orientation: bool,
    geometry_frame: bool,
) -> dict[str, Any]:
    data = yaml.safe_load(episode_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{episode_path} is not a YAML mapping")
    start = _start_mapping(data)
    if geometry_frame:
        calibration = _geometry_settle_correction(start, metrics_row, env_id=env_id)
        reference = calibration["reference_tip_world"]
        settled_tip = calibration["settled_tip_world"]
        correction = calibration["correction"]
    else:
        reference = _reference_tip(start)
        settled = _settled_tip_by_env(reset_diagnostic, metrics_row, tip_body=tip_body)
        settled_tip = settled[min(int(env_id), len(settled) - 1)]
        correction = _sub(reference, settled_tip)
    orientation_correction = None

    if calibrate_orientation:
        desired_tip_quat = start.get("reference_reward_body_start_orientation_wxyz")
        actual_tip_quat = _settled_tip_quat(metrics_row, tip_body=tip_body, env_id=env_id)
        reset_quat = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
        if (
            isinstance(desired_tip_quat, list)
            and len(desired_tip_quat) == 4
            and actual_tip_quat is not None
            and isinstance(reset_quat, list)
            and len(reset_quat) == 4
        ):
            orientation_correction = _qmul([float(x) for x in desired_tip_quat], _qconj(actual_tip_quat))
            new_reset_quat = _qmul(orientation_correction, [float(x) for x in reset_quat])
            for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
                if key in start:
                    start[key] = _round(new_reset_quat)
            repair = start.get("semantic_tip_transform_repair")
            rel_tip = repair.get("measured_tip_pos_in_reset_body") if isinstance(repair, dict) else None
            settle_shift = repair.get("applied_settle_compensation_shift") if isinstance(repair, dict) else None
            post_step_shift = correction
            if isinstance(rel_tip, list) and len(rel_tip) == 3:
                base_body_start = [reference[i] - _qapply(new_reset_quat, [float(x) for x in rel_tip])[i] for i in range(3)]
                if isinstance(settle_shift, list) and len(settle_shift) == 3:
                    base_body_start = _add(base_body_start, [float(x) for x in settle_shift])
                base_body_start = _add(base_body_start, post_step_shift)
                current_body = start.get("body_start_position_world")
                if isinstance(current_body, list) and len(current_body) == 3:
                    orientation_position_delta = _sub(base_body_start, [float(x) for x in current_body])
                    correction = _add(correction, orientation_position_delta)

    for key in ("body_start_position_world", "tcp_start_position_world"):
        if isinstance(start.get(key), list) and len(start[key]) == 3:
            start[key] = _round(_add([float(v) for v in start[key]], correction))
    if isinstance(start.get("reset_body_offset_from_reference_world"), list) and len(start["reset_body_offset_from_reference_world"]) == 3:
        start["reset_body_offset_from_reference_world"] = _round(
            _add([float(v) for v in start["reset_body_offset_from_reference_world"]], correction)
        )
    start["post_step_settle_calibration"] = {
        "method": (
            "body_start += reference_tip - settled_tip_from_insertion_geometry"
            if geometry_frame
            else "body_start += reference_tip - (reset_tip + realized_tip_delta_at_step)"
        ),
        "source_episode": str(episode_path),
        "source_step": int(metrics_row.get("step")),
        "env_id": int(env_id),
        "tip_body": tip_body,
        "reference_tip_world": _round(reference),
        "settled_tip_world": _round(settled_tip),
        "body_start_world_shift": _round(correction),
        "orientation_correction_wxyz": _round(orientation_correction) if orientation_correction is not None else None,
    }
    if geometry_frame:
        start["post_step_settle_calibration"].update(
            {
                "settled_signed_depth_m": round(float(calibration["settled_signed_depth_m"]), 9),
                "settled_lateral_residual_world": _round(calibration["settled_lateral_residual_world"]),
                "metric_env_id": int(calibration["metric_env_id"]),
                "frame_note": "metric env-origin cancels in body - entrance before applying canonical episode gate",
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {
        "episode": episode_path.name,
        "env_id": int(env_id),
        "reference_tip_world": reference,
        "settled_tip_world": settled_tip,
        "body_start_world_shift": correction,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--tip-body", default="sfp_tip_link")
    parser.add_argument("--calibrate-orientation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--geometry-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use post_step_insertion_geometry signed depth and lateral residual to avoid mixing Isaac env-origin "
            "world coordinates with canonical episode YAML coordinates."
        ),
    )
    parser.add_argument("--copy-non-episode-files", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source_episodes = args.episodes_dir.resolve()
    run_dir = args.run_dir.resolve()
    out_root = args.output_dir.resolve()
    out_episodes = out_root / "episodes"
    out_episodes.mkdir(parents=True, exist_ok=True)

    metrics_row = _row_for_step(_jsonl(run_dir / "metrics.jsonl"), int(args.step))
    reset_diagnostic_path = run_dir / "reset_diagnostic.json"
    reset_diagnostic = (
        json.loads(reset_diagnostic_path.read_text(encoding="utf-8"))
        if reset_diagnostic_path.is_file()
        else {}
    )
    summaries = []
    for episode in sorted(source_episodes.glob("episode_*.yaml")):
        summaries.append(
            calibrate_episode(
                episode,
                out_episodes / episode.name,
                reset_diagnostic=reset_diagnostic,
                metrics_row=metrics_row,
                env_id=int(args.env_id),
                tip_body=str(args.tip_body),
                calibrate_orientation=bool(args.calibrate_orientation),
                geometry_frame=bool(args.geometry_frame),
            )
        )
    if args.copy_non_episode_files:
        for path in source_episodes.parent.iterdir():
            if path.name == "episodes":
                continue
            dst = out_root / path.name
            if path.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(path, dst)
            elif path.is_file():
                shutil.copy2(path, dst)
    (out_root / "settle_calibration_summary.json").write_text(
        json.dumps({"source_run": str(run_dir), "step": int(args.step), "episodes": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
