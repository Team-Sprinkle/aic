#!/usr/bin/env python3
"""Repair near-gate reset-body poses from calibrated semantic tip transforms.

Some generated insertion episodes carry a manually edited wrist/reset-body pose
and a semantic tip reference pose.  If those two fields drift apart, the reset
IK can satisfy the wrist pose while ``sfp_tip_link`` starts centimeters away
from the intended near-gate geometry.  This utility treats the semantic tip
reference and the calibrated reset-body-to-tip transform as authoritative, then
rewrites the reset-body pose accordingly.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


def _qnorm(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
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


def _qslerp_identity(q: list[float], scale: float) -> list[float]:
    q = _qnorm(q)
    scale = float(scale)
    if scale <= 0.0:
        return [1.0, 0.0, 0.0, 0.0]
    if scale >= 1.0:
        return q
    if q[0] < 0.0:
        q = [-x for x in q]
    w = max(min(q[0], 1.0), -1.0)
    angle = math.acos(w)
    sin_angle = math.sin(angle)
    if abs(sin_angle) < 1.0e-9:
        return [1.0, 0.0, 0.0, 0.0]
    coeff_identity = math.sin((1.0 - scale) * angle) / sin_angle
    coeff_q = math.sin(scale * angle) / sin_angle
    return _qnorm(
        [
            coeff_identity + coeff_q * q[0],
            coeff_q * q[1],
            coeff_q * q[2],
            coeff_q * q[3],
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


def _round_vec(values: list[float], digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in values]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _start_mapping(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    target = scene.get("target") if isinstance(scene.get("target"), dict) else {}
    start = target.get("start_near_gate")
    if isinstance(start, dict):
        return start
    start = scene.get("start_near_gate")
    if isinstance(start, dict):
        return start
    start = data.get("start_near_gate")
    if isinstance(start, dict):
        return start
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
            return [float(x) for x in value]
    raise ValueError("start_near_gate has no semantic tip reference position")


def _orientation(start: dict[str, Any], calibrated: dict[str, Any], *, apply_strict_theta_correction: bool) -> tuple[list[float], list[float] | None]:
    return _orientation_with_mode(
        start,
        calibrated,
        apply_strict_theta_correction=apply_strict_theta_correction,
        strict_theta_correction_mode="forward",
    )


def _orientation_with_mode(
    start: dict[str, Any],
    calibrated: dict[str, Any],
    *,
    apply_strict_theta_correction: bool,
    strict_theta_correction_mode: str,
    strict_theta_correction_scale: float = 1.0,
) -> tuple[list[float], list[float] | None]:
    desired_tip_quat = start.get("reference_reward_body_start_orientation_wxyz")
    applied_correction = None
    strict_correction = start.get("measured_strict_theta_correction")
    if (
        apply_strict_theta_correction
        and isinstance(strict_correction, dict)
        and isinstance(strict_correction.get("q_corr_world_wxyz"), list)
        and len(strict_correction["q_corr_world_wxyz"]) == 4
        and isinstance(desired_tip_quat, list)
        and len(desired_tip_quat) == 4
    ):
        applied_correction = [float(x) for x in strict_correction["q_corr_world_wxyz"]]
        if strict_theta_correction_mode == "inverse":
            applied_correction = _qconj(applied_correction)
        elif strict_theta_correction_mode != "forward":
            raise ValueError(f"unsupported strict theta correction mode: {strict_theta_correction_mode!r}")
        applied_correction = _qslerp_identity(applied_correction, strict_theta_correction_scale)
        desired_tip_quat = _qmul(applied_correction, [float(x) for x in desired_tip_quat])
    rel_tip_quat = calibrated.get("measured_tip_quat_in_reset_body")
    if isinstance(desired_tip_quat, list) and len(desired_tip_quat) == 4 and isinstance(rel_tip_quat, list) and len(rel_tip_quat) == 4:
        return _qmul([float(x) for x in desired_tip_quat], _qconj([float(x) for x in rel_tip_quat])), applied_correction
    for key in ("reset_body_orientation_wxyz", "body_start_orientation_wxyz", "tcp_start_orientation_world"):
        value = start.get(key)
        if isinstance(value, list) and len(value) == 4:
            return _qnorm([float(x) for x in value]), applied_correction
    raise ValueError("start_near_gate has no usable reset orientation")


def repair_episode(
    path: Path,
    out_path: Path,
    *,
    apply_settle_compensation: bool,
    apply_strict_theta_correction: bool,
    strict_theta_correction_mode: str = "forward",
    strict_theta_correction_scale: float = 1.0,
) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    start = _start_mapping(data)
    calibrated = start.get("calibrated_from_reset_diagnostic")
    if not isinstance(calibrated, dict):
        raise ValueError(f"{path} has no calibrated_from_reset_diagnostic metadata")
    rel_tip = calibrated.get("measured_tip_pos_in_reset_body")
    if not isinstance(rel_tip, list) or len(rel_tip) != 3:
        raise ValueError(f"{path} calibration has no measured_tip_pos_in_reset_body")

    reference = _reference_tip(start)
    orientation, applied_strict_theta_correction = _orientation_with_mode(
        start,
        calibrated,
        apply_strict_theta_correction=apply_strict_theta_correction,
        strict_theta_correction_mode=str(strict_theta_correction_mode),
        strict_theta_correction_scale=float(strict_theta_correction_scale),
    )
    reset_to_tip_world = _qapply(orientation, [float(x) for x in rel_tip])
    body_start = [reference[i] - reset_to_tip_world[i] for i in range(3)]
    settle_shift = None
    settle = start.get("settle_compensation")
    if apply_settle_compensation and isinstance(settle, dict):
        raw_shift = settle.get("body_start_world_shift")
        if isinstance(raw_shift, list) and len(raw_shift) == 3:
            settle_shift = [float(x) for x in raw_shift]
            body_start = [body_start[i] + settle_shift[i] for i in range(3)]
    old_body = start.get("body_start_position_world") or start.get("tcp_start_position_world")
    old_offset = _sub([float(x) for x in old_body], reference) if isinstance(old_body, list) and len(old_body) == 3 else None
    new_offset = _sub(body_start, reference)

    start["reset_body_name"] = str(start.get("reset_body_name") or calibrated.get("reset_body") or "wrist_3_link")
    start["body_start_position_world"] = _round_vec(body_start)
    start["tcp_start_position_world"] = _round_vec(body_start)
    start["body_start_orientation_wxyz"] = _round_vec(orientation)
    start["reset_body_orientation_wxyz"] = _round_vec(orientation)
    start["tcp_start_orientation_world"] = _round_vec(orientation)
    start["reset_body_offset_from_reference_world"] = _round_vec(new_offset)
    start["semantic_tip_transform_repair"] = {
        "method": "body_start = semantic_tip_reference - q_reset * measured_tip_pos_in_reset_body",
        "source_episode": str(path),
        "old_reset_body_offset_from_reference_world": _round_vec(old_offset) if old_offset is not None else None,
        "new_reset_body_offset_from_reference_world": _round_vec(new_offset),
        "measured_tip_pos_in_reset_body": _round_vec([float(x) for x in rel_tip]),
        "applied_settle_compensation_shift": _round_vec(settle_shift) if settle_shift is not None else None,
        "applied_strict_theta_correction_wxyz": (
            _round_vec(applied_strict_theta_correction) if applied_strict_theta_correction is not None else None
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return {
        "episode": path.name,
        "old_offset_m": old_offset,
        "new_offset_m": new_offset,
        "offset_delta_m": None if old_offset is None else _sub(new_offset, old_offset),
        "body_start_position_world": body_start,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--copy-non-episode-files", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--apply-settle-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply start.settle_compensation.body_start_world_shift after repairing the semantic tip transform.",
    )
    parser.add_argument(
        "--apply-strict-theta-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compose start.measured_strict_theta_correction.q_corr_world_wxyz onto the desired semantic tip quaternion.",
    )
    parser.add_argument(
        "--strict-theta-correction-mode",
        choices=("forward", "inverse"),
        default="forward",
        help="Whether to compose the stored strict theta correction directly or with its inverse.",
    )
    parser.add_argument(
        "--strict-theta-correction-scale",
        type=float,
        default=1.0,
        help="Slerp fraction from identity to the stored strict theta correction.",
    )
    args = parser.parse_args()

    source_episodes = args.episodes_dir.resolve()
    out_root = args.output_dir.resolve()
    out_episodes = out_root / "episodes"
    out_episodes.mkdir(parents=True, exist_ok=True)
    summaries = []
    for episode in sorted(source_episodes.glob("episode_*.yaml")):
        summaries.append(
            repair_episode(
                episode,
                out_episodes / episode.name,
                apply_settle_compensation=bool(args.apply_settle_compensation),
                apply_strict_theta_correction=bool(args.apply_strict_theta_correction),
                strict_theta_correction_mode=str(args.strict_theta_correction_mode),
                strict_theta_correction_scale=float(args.strict_theta_correction_scale),
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
    (out_root / "repair_summary.json").write_text(json.dumps({"episodes": summaries}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
