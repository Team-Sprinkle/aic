#!/usr/bin/env python3
"""Generate near-gate reset orientation variants from episode YAMLs.

This is for empirical reset calibration: YAML quaternions and analytical
tip-frame corrections have not reliably predicted post-step semantic-tip
orientation, so generate a small candidate set and validate it through the
runtime settle path.
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
    vx, vy, vz = [float(x) for x in v]
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _vadd(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _vsub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _rotvec_quat(rx: float, ry: float, rz: float) -> list[float]:
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1.0e-12:
        return [1.0, 0.0, 0.0, 0.0]
    half = 0.5 * angle
    scale = math.sin(half) / angle
    return [math.cos(half), rx * scale, ry * scale, rz * scale]


def _round(values: list[float], digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in values]


def _start_mapping(data: dict[str, Any]) -> dict[str, Any]:
    scene = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    target = scene.get("target") if isinstance(scene.get("target"), dict) else {}
    for value in (target.get("start_near_gate"), scene.get("start_near_gate"), data.get("start_near_gate")):
        if isinstance(value, dict):
            return value
    raise ValueError("episode has no start_near_gate mapping")


def _episode_files(config_dir: Path) -> list[Path]:
    episodes_dir = config_dir / "episodes" if (config_dir / "episodes").is_dir() else config_dir
    episodes = sorted(episodes_dir.glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml files under {episodes_dir}")
    return episodes


def _parse_values(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("empty sweep values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--rx-values",
        default="0.0",
        help="Comma-separated radians for reset-orientation x-axis perturbations.",
    )
    parser.add_argument(
        "--ry-values",
        default="0.0",
        help="Comma-separated radians for reset-orientation y-axis perturbations.",
    )
    parser.add_argument(
        "--rz-values",
        default="0.0",
        help="Comma-separated radians for reset-orientation z-axis perturbations.",
    )
    parser.add_argument(
        "--composition",
        choices=("world", "body"),
        default="world",
        help="world: q_new=dq*q_base. body: q_new=q_base*dq.",
    )
    parser.add_argument(
        "--preserve-reference-tip-position",
        action="store_true",
        help=(
            "Rotate the reset body about reference_tip_center_position_world/reference_reward_body_start_position_world. "
            "This keeps the semantic tip at the original requested near-gate position while sweeping orientation."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of generated variants.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src = args.input_dir.resolve()
    dst = args.output_dir.resolve()
    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite")
        shutil.rmtree(dst)
    (dst / "episodes").mkdir(parents=True)

    source_episodes = _episode_files(src)
    rx_values = _parse_values(args.rx_values)
    ry_values = _parse_values(args.ry_values)
    rz_values = _parse_values(args.rz_values)

    variants: list[dict[str, Any]] = []
    out_idx = 1
    for source in source_episodes:
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{source} is not a YAML mapping")
        start = _start_mapping(data)
        base_q = start.get("body_start_orientation_wxyz") or start.get("reset_body_orientation_wxyz")
        if not isinstance(base_q, list) or len(base_q) != 4:
            raise ValueError(f"{source} missing body_start_orientation_wxyz/reset_body_orientation_wxyz")
        base_body_position = start.get("body_start_position_world") or start.get("tcp_start_position_world")
        base_tip_position = start.get("reference_tip_center_position_world") or start.get(
            "reference_reward_body_start_position_world"
        )
        tip_in_base_body: list[float] | None = None
        if args.preserve_reference_tip_position:
            if not isinstance(base_body_position, list) or len(base_body_position) != 3:
                raise ValueError(f"{source} missing body_start_position_world/tcp_start_position_world")
            if not isinstance(base_tip_position, list) or len(base_tip_position) != 3:
                raise ValueError(
                    f"{source} missing reference_tip_center_position_world/reference_reward_body_start_position_world"
                )
            tip_world_delta = _vsub([float(v) for v in base_tip_position], [float(v) for v in base_body_position])
            tip_in_base_body = _qrot(_qconj([float(v) for v in base_q]), tip_world_delta)
        for rx in rx_values:
            for ry in ry_values:
                for rz in rz_values:
                    if args.limit > 0 and out_idx > int(args.limit):
                        break
                    trial = yaml.safe_load(source.read_text(encoding="utf-8"))
                    trial_start = _start_mapping(trial)
                    dq = _rotvec_quat(float(rx), float(ry), float(rz))
                    q_new = _qmul(dq, [float(v) for v in base_q]) if args.composition == "world" else _qmul(
                        [float(v) for v in base_q], dq
                    )
                    for key in ("body_start_orientation_wxyz", "reset_body_orientation_wxyz", "tcp_start_orientation_world"):
                        if key in trial_start:
                            trial_start[key] = _round(q_new)
                    if args.preserve_reference_tip_position and tip_in_base_body is not None:
                        tip_world = [float(v) for v in base_tip_position]
                        new_body_position = _vsub(tip_world, _qrot(q_new, tip_in_base_body))
                        for key in ("body_start_position_world", "tcp_start_position_world"):
                            if key in trial_start:
                                trial_start[key] = _round(new_body_position)
                        trial_start["reset_body_offset_from_reference_world"] = _round(_vsub(new_body_position, tip_world))
                    trial_start["orientation_sweep_variant"] = {
                        "source_episode": source.name,
                        "base_orientation_wxyz": _round([float(v) for v in base_q]),
                        "composition": str(args.composition),
                        "preserve_reference_tip_position": bool(args.preserve_reference_tip_position),
                        "rotvec_rad": [float(rx), float(ry), float(rz)],
                        "orientation_wxyz": _round(q_new),
                    }
                    trial["episode_id"] = f"{trial.get('episode_id', source.stem)}_orisweep_{out_idx:06d}"
                    out_path = dst / "episodes" / f"episode_{out_idx:06d}.yaml"
                    out_path.write_text(yaml.safe_dump(trial, sort_keys=False), encoding="utf-8")
                    variants.append(
                        {
                            "episode": out_path.name,
                            "source_episode": source.name,
                            "composition": str(args.composition),
                            "preserve_reference_tip_position": bool(args.preserve_reference_tip_position),
                            "rotvec_rad": [float(rx), float(ry), float(rz)],
                            "orientation_wxyz": _round(q_new),
                        }
                    )
                    out_idx += 1
                if args.limit > 0 and out_idx > int(args.limit):
                    break
            if args.limit > 0 and out_idx > int(args.limit):
                break
        if args.limit > 0 and out_idx > int(args.limit):
            break

    for path in src.iterdir() if (src / "episodes").is_dir() else src.parent.iterdir():
        if path.name == "episodes":
            continue
        dst_path = dst / path.name
        if path.is_file():
            shutil.copy2(path, dst_path)
        elif path.is_dir() and not dst_path.exists():
            shutil.copytree(path, dst_path)
    summary = {
        "input_dir": str(src),
        "output_dir": str(dst),
        "composition": str(args.composition),
        "preserve_reference_tip_position": bool(args.preserve_reference_tip_position),
        "rx_values": rx_values,
        "ry_values": ry_values,
        "rz_values": rz_values,
        "variant_count": len(variants),
        "variants": variants,
    }
    (dst / "orientation_sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("output_dir", "composition", "variant_count")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
