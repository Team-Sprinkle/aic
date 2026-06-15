#!/usr/bin/env python3
"""Calibrate near-gate reset YAMLs from an Isaac reset diagnostic.

The episode generator stores a requested reset-body pose, typically
``gripper_tcp``. Isaac's damped-IK reset can satisfy that pose while the
semantic insertion frame, ``sfp_tip_link``, lands with a residual position or
orientation error. This utility uses the measured post-reset transform from
reset body to semantic tip and rewrites the reset-body pose so the semantic tip
lands on the requested reference pose instead.
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
        raise ValueError("Cannot normalize near-zero quaternion")
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
    out = _qmul_raw(_qmul_raw(_qnorm(q), [0.0, *[float(x) for x in v]]), _qconj(q))
    return [float(x) for x in out[1:]]


def _qmul_raw(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _round_vec(v: list[float]) -> list[float]:
    return [round(float(x), 6) for x in v]


def _start_dict(episode: dict[str, Any]) -> dict[str, Any]:
    scene = episode.get("scene") or {}
    start = scene.get("start_near_gate")
    if isinstance(start, dict):
        return start
    target = scene.get("target") or {}
    start = target.get("start_near_gate")
    if isinstance(start, dict):
        return start
    raise ValueError("episode has no scene.start_near_gate mapping")


def _copy_sidecar_files(source_dir: Path, output_dir: Path) -> None:
    for name in ("request.yaml", "summary.json", "manifest.csv", "task_distribution.yaml", "source.txt"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def calibrate(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_episode_config_dir)
    source_episodes = source_dir / "episodes"
    if not source_episodes.is_dir():
        raise FileNotFoundError(f"missing source episodes directory: {source_episodes}")
    output_dir = Path(args.output_episode_config_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_episodes = output_dir / "episodes"
    output_episodes.mkdir(parents=True)

    diagnostic = json.loads(Path(args.reset_diagnostic).read_text(encoding="utf-8"))
    env_id = str(int(args.env_id))
    reset_body = str(args.reset_body)
    tip_body = str(args.tip_body)
    actual_pos = diagnostic["actual_body_position_world_by_env"]
    actual_quat = diagnostic["actual_body_orientation_wxyz_by_env"]
    body_pos = [float(x) for x in actual_pos[reset_body][int(args.env_id)]]
    tip_pos = [float(x) for x in actual_pos[tip_body][int(args.env_id)]]
    body_quat = _qnorm([float(x) for x in actual_quat[reset_body][int(args.env_id)]])
    tip_quat = _qnorm([float(x) for x in actual_quat[tip_body][int(args.env_id)]])
    rel_pos_body = _qapply(_qconj(body_quat), _sub(tip_pos, body_pos))
    rel_quat_body_tip = _qmul(_qconj(body_quat), tip_quat)

    episode_start = diagnostic.get("episode_start_by_env", {}).get(env_id, {})
    desired_tip_quat = args.desired_tip_orientation_wxyz
    if desired_tip_quat is None:
        desired_tip_quat = episode_start.get("reference_reward_body_start_orientation_wxyz")
    if not isinstance(desired_tip_quat, list) or len(desired_tip_quat) != 4:
        raise ValueError("desired tip orientation is missing; pass --desired-tip-orientation-wxyz")
    desired_tip_quat = _qnorm([float(x) for x in desired_tip_quat])

    calibrated = []
    for episode_path in sorted(source_episodes.glob("episode_*.yaml")):
        episode = yaml.safe_load(episode_path.read_text(encoding="utf-8"))
        start = _start_dict(episode)
        desired_tip_pos = start.get("reference_reward_body_start_position_world") or start.get(
            "reference_tip_center_position_world"
        )
        if not isinstance(desired_tip_pos, list) or len(desired_tip_pos) != 3:
            raise ValueError(f"{episode_path} has no reference tip position")
        desired_tip_pos = [float(x) for x in desired_tip_pos]
        new_body_quat = _qmul(desired_tip_quat, _qconj(rel_quat_body_tip))
        new_body_pos = _sub(desired_tip_pos, _qapply(new_body_quat, rel_pos_body))
        start["reset_body_name"] = reset_body
        start["body_start_position_world"] = _round_vec(new_body_pos)
        start["tcp_start_position_world"] = _round_vec(new_body_pos)
        start["body_start_orientation_wxyz"] = _round_vec(new_body_quat)
        start["reset_body_orientation_wxyz"] = _round_vec(new_body_quat)
        start["tcp_start_orientation_world"] = _round_vec(new_body_quat)
        start["reset_body_offset_from_reference_world"] = _round_vec(_sub(new_body_pos, desired_tip_pos))
        start["calibrated_from_reset_diagnostic"] = {
            "reset_diagnostic": str(args.reset_diagnostic),
            "env_id": int(args.env_id),
            "reset_body": reset_body,
            "tip_body": tip_body,
            "measured_reset_body_pos_w": _round_vec(body_pos),
            "measured_tip_pos_w": _round_vec(tip_pos),
            "measured_reset_body_quat_wxyz": _round_vec(body_quat),
            "measured_tip_quat_wxyz": _round_vec(tip_quat),
            "measured_tip_pos_in_reset_body": _round_vec(rel_pos_body),
            "measured_tip_quat_in_reset_body": _round_vec(rel_quat_body_tip),
            "desired_tip_quat_wxyz": _round_vec(desired_tip_quat),
        }
        (output_episodes / episode_path.name).write_text(yaml.safe_dump(episode, sort_keys=False), encoding="utf-8")
        calibrated.append({"episode": episode_path.name, "body_start_position_world": new_body_pos})

    _copy_sidecar_files(source_dir, output_dir)
    summary = {
        "source_episode_config_dir": str(source_dir),
        "reset_diagnostic": str(args.reset_diagnostic),
        "env_id": int(args.env_id),
        "reset_body": reset_body,
        "tip_body": tip_body,
        "measured_tip_pos_in_reset_body": rel_pos_body,
        "measured_tip_quat_in_reset_body": rel_quat_body_tip,
        "desired_tip_orientation_wxyz": desired_tip_quat,
        "calibrated_episode_count": len(calibrated),
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-episode-config-dir", required=True)
    parser.add_argument("--reset-diagnostic", required=True)
    parser.add_argument("--output-episode-config-dir", required=True)
    parser.add_argument("--env-id", type=int, default=0)
    parser.add_argument("--reset-body", default="gripper_tcp")
    parser.add_argument("--tip-body", default="sfp_tip_link")
    parser.add_argument("--desired-tip-orientation-wxyz", type=float, nargs=4, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    out = calibrate(parse_args())
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
