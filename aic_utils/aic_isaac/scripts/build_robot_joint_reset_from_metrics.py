#!/usr/bin/env python3
"""Build replayable robot-joint-state reset episodes from SERL metrics.

This is used when a rollout finds a promising articulated state whose semantic
tip geometry cannot be reconstructed reliably from a wrist pose alone.  The
output episode YAMLs keep the source scene/target metadata and replace only the
near-gate reset with the measured robot joint positions/velocities.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import yaml


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _episode_file(config_dir: Path) -> Path:
    episodes = sorted((config_dir / "episodes").glob("episode_*.yaml"))
    if not episodes:
        raise FileNotFoundError(f"no episode_*.yaml under {config_dir / 'episodes'}")
    return episodes[0]


def _by_env(mapping: dict[str, Any], key: str, env_index: int) -> float:
    values = mapping.get(f"{key}_by_env")
    if isinstance(values, list) and env_index < len(values):
        return float(values[env_index])
    if env_index == 0 and mapping.get(f"{key}_env0") is not None:
        return float(mapping[f"{key}_env0"])
    return float("nan")


def _vec_by_env(mapping: dict[str, Any], key: str, env_index: int) -> list[float] | None:
    values = mapping.get(f"{key}_by_env")
    if isinstance(values, list) and env_index < len(values) and isinstance(values[env_index], list):
        return [float(v) for v in values[env_index]]
    if env_index == 0 and isinstance(mapping.get(f"{key}_env0"), list):
        return [float(v) for v in mapping[f"{key}_env0"]]
    return None


def _robot_state_for_env(row: dict[str, Any], env_index: int, *, post_step: bool) -> dict[str, Any] | None:
    key = "post_step_robot_state" if post_step else "pre_step_robot_state"
    state = row.get(key)
    if not isinstance(state, dict):
        return None
    joint_names = state.get("joint_names")
    q_by_env = state.get("joint_positions_by_env")
    qd_by_env = state.get("joint_velocities_by_env")
    if not isinstance(joint_names, list) or not isinstance(q_by_env, list) or env_index >= len(q_by_env):
        return None
    q = q_by_env[env_index]
    if not isinstance(q, list):
        return None
    qd = None
    if isinstance(qd_by_env, list) and env_index < len(qd_by_env) and isinstance(qd_by_env[env_index], list):
        qd = qd_by_env[env_index]
    return {
        "joint_names": [str(name) for name in joint_names],
        "joint_positions": [float(v) for v in q],
        "joint_velocities": [float(v) for v in qd] if qd is not None else [0.0] * len(q),
    }


def _candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        geom = row.get("post_step_insertion_geometry") or {}
        module = (row.get("post_step_all_body_insertion_geometry") or {}).get("sfp_module_link") or {}
        if not isinstance(geom, dict):
            continue
        sample_count = max(
            len(geom.get("signed_depth_m_by_env") or []),
            len(geom.get("lateral_error_m_by_env") or []),
            len(geom.get("orientation_error_rad_by_env") or []),
            1,
        )
        for env_index in range(sample_count):
            state = _robot_state_for_env(row, env_index, post_step=True)
            if state is None:
                continue
            s = _by_env(geom, "signed_depth_m", env_index)
            r = _by_env(geom, "lateral_error_m", env_index)
            theta = _by_env(geom, "orientation_error_rad", env_index)
            module_s = _by_env(module, "signed_depth_m", env_index)
            module_r = _by_env(module, "lateral_error_m", env_index)
            if not all(math.isfinite(v) for v in (s, r, theta)):
                continue
            if s < args.min_s_m or s > args.max_s_m:
                continue
            if r > args.max_r_m or theta > args.max_theta_rad:
                continue
            if math.isfinite(module_r) and module_r > args.max_module_r_m:
                continue
            consistency_error = abs((s - module_s) - args.expected_tip_module_gap_m) if math.isfinite(module_s) else 1.0
            score = (
                (s * args.score_s_weight)
                - (r * args.score_r_weight)
                - (theta * args.score_theta_weight)
                - (consistency_error * args.score_consistency_weight)
            )
            out.append(
                {
                    "step": int(row.get("step", -1)),
                    "env_index": env_index,
                    "s": s,
                    "r": r,
                    "theta": theta,
                    "module_s": module_s,
                    "module_r": module_r,
                    "consistency_error_m": consistency_error,
                    "score": score,
                    "tip_world": _vec_by_env(geom, "body_world", env_index),
                    "tip_orientation_wxyz": _vec_by_env(geom, "body_orientation_wxyz", env_index),
                    "robot_state": state,
                }
            )
    return sorted(out, key=lambda item: item["score"], reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config-dir", required=True, type=Path)
    parser.add_argument("--metrics-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-s-m", type=float, default=-0.010)
    parser.add_argument("--max-s-m", type=float, default=0.020)
    parser.add_argument("--max-r-m", type=float, default=0.0010)
    parser.add_argument("--max-theta-rad", type=float, default=0.035)
    parser.add_argument("--max-module-r-m", type=float, default=0.0020)
    parser.add_argument("--expected-tip-module-gap-m", type=float, default=0.02365)
    parser.add_argument("--score-s-weight", type=float, default=1.0)
    parser.add_argument("--score-r-weight", type=float, default=20.0)
    parser.add_argument("--score-theta-weight", type=float, default=0.10)
    parser.add_argument("--score-consistency-weight", type=float, default=15.0)
    parser.add_argument(
        "--zero-joint-velocities",
        action="store_true",
        help="Write robot_joint_velocities as explicit zeros instead of replaying recorded velocities.",
    )
    parser.add_argument(
        "--omit-joint-velocities",
        action="store_true",
        help="Omit robot_joint_velocities so the runtime reset path uses its zero-velocity default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.zero_joint_velocities and args.omit_joint_velocities:
        parser.error("--zero-joint-velocities and --omit-joint-velocities are mutually exclusive")

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
    rows = _jsonl(args.metrics_jsonl.resolve())
    candidates = _candidates(rows, args)
    if not candidates:
        raise RuntimeError("no rows matched the requested robot-state reset filters")

    selected = candidates[: max(1, int(args.top_k))]
    for idx, candidate in enumerate(selected, start=1):
        data = yaml.safe_load(base_episode.read_text(encoding="utf-8"))
        start = ((data.get("scene") or {}).get("start_near_gate") or {})
        robot_state = candidate["robot_state"]
        start["reset_mode"] = "robot_joint_state"
        start["robot_joint_names"] = robot_state["joint_names"]
        start["robot_joint_positions"] = robot_state["joint_positions"]
        if args.omit_joint_velocities:
            start.pop("robot_joint_velocities", None)
            velocity_policy = "omitted_runtime_zero_default"
        elif args.zero_joint_velocities:
            start["robot_joint_velocities"] = [0.0] * len(robot_state["joint_positions"])
            velocity_policy = "explicit_zero"
        else:
            start["robot_joint_velocities"] = robot_state["joint_velocities"]
            velocity_policy = "recorded_post_step"
        start["robot_joint_state_reset_source"] = {
            "metrics_jsonl": str(args.metrics_jsonl.resolve()),
            "step": candidate["step"],
            "env_index": candidate["env_index"],
            "post_step_s_m": candidate["s"],
            "post_step_r_m": candidate["r"],
            "post_step_theta_rad": candidate["theta"],
            "post_step_module_s_m": candidate["module_s"],
            "post_step_module_r_m": candidate["module_r"],
            "post_step_tip_world": candidate["tip_world"],
            "post_step_tip_orientation_wxyz": candidate["tip_orientation_wxyz"],
            "consistency_error_m": candidate["consistency_error_m"],
            "score": candidate["score"],
            "velocity_policy": velocity_policy,
        }
        data["episode_id"] = f"{data.get('episode_id', base_episode.stem)}_robot_state_step{candidate['step']}_env{candidate['env_index']}_{idx:06d}"
        out = output_dir / "episodes" / f"episode_{idx:06d}.yaml"
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    manifest = {
        "base_episode": str(base_episode),
        "metrics_jsonl": str(args.metrics_jsonl.resolve()),
        "episode_count": len(selected),
        "velocity_policy": (
            "omitted_runtime_zero_default"
            if args.omit_joint_velocities
            else "explicit_zero"
            if args.zero_joint_velocities
            else "recorded_post_step"
        ),
        "filter": {
            "min_s_m": args.min_s_m,
            "max_s_m": args.max_s_m,
            "max_r_m": args.max_r_m,
            "max_theta_rad": args.max_theta_rad,
            "max_module_r_m": args.max_module_r_m,
            "expected_tip_module_gap_m": args.expected_tip_module_gap_m,
        },
        "selected": [
            {k: v for k, v in candidate.items() if k != "robot_state"}
            for candidate in selected
        ],
    }
    (output_dir / "robot_joint_reset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "episode_count": len(selected), "best": manifest["selected"][0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
