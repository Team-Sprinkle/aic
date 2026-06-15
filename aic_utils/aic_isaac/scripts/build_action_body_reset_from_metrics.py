#!/usr/bin/env python3
"""Build episode resets that target a measured controller/action body pose.

This is a reset/controller diagnostic helper.  It keeps the scene, target,
reward, and success definitions unchanged, and only replaces
``scene.start_near_gate`` so the reset body matches the controller body used by
Differential IK, usually ``wrist_3_link``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


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


def _episode_files(config_dir: Path) -> list[Path]:
    episodes_dir = config_dir / "episodes" if (config_dir / "episodes").is_dir() else config_dir
    episodes = sorted(episodes_dir.glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml under {episodes_dir}")
    return episodes


def _vec(value: Any, *, name: str, length: int = 3) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-value list")
    return [float(v) for v in value]


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _round(values: list[float], digits: int = 9) -> list[float]:
    return [round(float(v), digits) for v in values]


def _body_pose(row: dict[str, Any], body_name: str, env_id: int, *, pre_step: bool) -> tuple[list[float], list[float]]:
    key = "pre_step_selected_body_poses" if pre_step else "post_step_selected_body_poses"
    poses = row.get(key)
    if isinstance(poses, dict):
        positions = (poses.get("positions_w_by_env") or {}).get(body_name)
        orientations = (poses.get("orientations_wxyz_by_env") or {}).get(body_name)
    else:
        positions = None
        orientations = None
    if positions is None or orientations is None:
        offsets_key = "pre_step_body_frame_offsets" if pre_step else "post_step_body_frame_offsets"
        offsets = row.get(offsets_key)
        if isinstance(offsets, dict):
            positions = (offsets.get("world_position_by_env") or {}).get(body_name)
            orientations = (offsets.get("world_quat_wxyz_by_env") or {}).get(body_name)
    if positions is None or orientations is None:
        positions = (row.get("actual_body_position_world_by_env") or {}).get(body_name)
        orientations = (row.get("actual_body_orientation_wxyz_by_env") or {}).get(body_name)
    if not isinstance(positions, list) or env_id >= len(positions):
        raise ValueError(f"metrics row has no position for {body_name} env {env_id}")
    if not isinstance(orientations, list) or env_id >= len(orientations):
        raise ValueError(f"metrics row has no orientation for {body_name} env {env_id}")
    return (
        _vec(positions[env_id], name=f"{body_name} position env {env_id}", length=3),
        _vec(orientations[env_id], name=f"{body_name} orientation env {env_id}", length=4),
    )


def _env_origin(row: dict[str, Any], data: dict[str, Any], env_id: int) -> list[float]:
    target = ((data.get("scene") or {}).get("target") or {})
    local_target = _vec(target.get("target_pose_world", {}).get("position"), name="target_pose_world.position")
    geom = row.get("post_step_insertion_geometry") or {}
    targets = geom.get("target_world_by_env")
    if isinstance(targets, list) and env_id < len(targets):
        return _sub(_vec(targets[env_id], name=f"target_world_by_env[{env_id}]"), local_target)
    target_env0 = geom.get("target_world_env0")
    if env_id == 0 and isinstance(target_env0, list):
        return _sub(_vec(target_env0, name="target_world_env0"), local_target)
    return [0.0, 0.0, 0.0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config-dir", required=True, type=Path)
    parser.add_argument("--metrics-jsonl", required=True, type=Path)
    parser.add_argument("--output-config-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--body-name", default="wrist_3_link")
    parser.add_argument("--pose-row", choices=["pre", "post"], default="pre")
    parser.add_argument("--max-envs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output_config_dir.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite")
        shutil.rmtree(output)
    (output / "episodes").mkdir(parents=True)

    row = _row_for_step(_jsonl(args.metrics_jsonl.resolve()), int(args.step))
    episodes = _episode_files(args.input_config_dir.resolve())
    max_envs = int(args.max_envs) if int(args.max_envs) > 0 else len(episodes)
    records: list[dict[str, Any]] = []
    for env_id, episode in enumerate(episodes[:max_envs]):
        data = yaml.safe_load(episode.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{episode} is not a YAML mapping")
        start = ((data.get("scene") or {}).get("start_near_gate") or {})
        pos_world, quat = _body_pose(row, str(args.body_name), env_id, pre_step=args.pose_row == "pre")
        origin = _env_origin(row, data, env_id)
        pos_local = _sub(pos_world, origin)
        start["reset_mode"] = "body_start_position_world"
        start["reset_body_name"] = str(args.body_name)
        start["body_start_position_world"] = _round(pos_local)
        start["body_start_orientation_wxyz"] = _round(quat)
        start["tcp_start_position_world"] = _round(pos_local)
        start["tcp_start_orientation_world"] = _round(quat)
        start["action_body_reset_source"] = {
            "metrics_jsonl": str(args.metrics_jsonl.resolve()),
            "step": int(args.step),
            "env_id": int(env_id),
            "pose_row": str(args.pose_row),
            "body_name": str(args.body_name),
            "body_position_world": _round(pos_world),
            "body_orientation_wxyz": _round(quat),
            "env_origin": _round(origin),
        }
        data["episode_id"] = f"{data.get('episode_id', episode.stem)}_{args.body_name}_reset_step{args.step}_env{env_id}"
        out_path = output / "episodes" / f"episode_{env_id + 1:06d}.yaml"
        out_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        records.append(
            {
                "source_episode": str(episode),
                "output_episode": str(out_path),
                "env_id": int(env_id),
                "body_position_world": pos_world,
                "body_orientation_wxyz": quat,
                "env_origin": origin,
            }
        )

    manifest = {
        "input_config_dir": str(args.input_config_dir.resolve()),
        "metrics_jsonl": str(args.metrics_jsonl.resolve()),
        "source_step": int(args.step),
        "pose_row": str(args.pose_row),
        "body_name": str(args.body_name),
        "episode_count": len(records),
        "records": records,
    }
    (output / "action_body_reset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_config_dir": str(output), "episode_count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
