#!/usr/bin/env python3
"""Append causal contact/recovery features to a LeRobot dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_DIR))

from lerobot_robot_aic.contact_recovery_features import (  # noqa: E402
    CONTACT_RECOVERY_FEATURE_DIM,
    CONTACT_RECOVERY_FEATURE_NAMES,
    ContactRecoveryFeatureComputer,
    ContactRecoveryFeatureConfig,
)
from lerobot_robot_aic.task_encoding import TASK_VECTOR_NAMES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validation-report", type=Path, default=None)
    return parser.parse_args()


def _state_indices(names: list[str]) -> dict[str, list[int]]:
    required = {
        "position": [f"tcp_pose.position.{axis}" for axis in "xyz"],
        "orientation": [f"tcp_pose.orientation.{axis}" for axis in ("x", "y", "z", "w")],
        "force": [f"wrist_wrench.force.{axis}" for axis in "xyz"],
        "torque": [f"wrist_wrench.torque.{axis}" for axis in "xyz"],
    }
    return {key: [names.index(name) for name in needed] for key, needed in required.items()}


def _frame_time(row: Any, fps: float) -> float:
    if "timestamp" in row and row["timestamp"] is not None:
        return float(row["timestamp"])
    return float(row["frame_index"]) / fps


def _compute_stats(states: list[Any]) -> dict[str, Any]:
    matrix = np.stack([np.asarray(state, dtype=np.float32).reshape(-1) for state in states], axis=0)
    std = matrix.std(axis=0)
    safe_std = np.where(np.abs(std) < 1e-8, 1.0, std)
    return {
        "min": matrix.min(axis=0).tolist(),
        "max": matrix.max(axis=0).tolist(),
        "mean": matrix.mean(axis=0).tolist(),
        "std": safe_std.tolist(),
        "count": [int(matrix.shape[0])],
        "q01": np.quantile(matrix, 0.01, axis=0).tolist(),
        "q10": np.quantile(matrix, 0.10, axis=0).tolist(),
        "q50": np.quantile(matrix, 0.50, axis=0).tolist(),
        "q90": np.quantile(matrix, 0.90, axis=0).tolist(),
        "q99": np.quantile(matrix, 0.99, axis=0).tolist(),
    }


def _append_features_to_frame_group(
    group: pd.DataFrame,
    *,
    indices: dict[str, list[int]],
    fps: float,
    config: ContactRecoveryFeatureConfig,
    task_vector_dim: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    computer = ContactRecoveryFeatureComputer(config)
    rows = group.sort_values(["frame_index", "timestamp"] if "timestamp" in group.columns else ["frame_index"]).copy()
    new_states: list[np.ndarray] = []
    events: list[dict[str, Any]] = []
    threshold1_latest_time_index = CONTACT_RECOVERY_FEATURE_NAMES.index("force_thresh_1.time_since_latest_sec")
    threshold1_latest_norm_index = CONTACT_RECOVERY_FEATURE_NAMES.index("force_thresh_1.latest_delta_norm")
    for _, row in rows.iterrows():
        state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
        time_sec = _frame_time(row, fps)
        features = computer.update(
            time_sec=time_sec,
            tcp_position_base=state[indices["position"]],
            tcp_orientation_xyzw=state[indices["orientation"]],
            force=state[indices["force"]],
            torque=state[indices["torque"]],
        )
        if float(features[threshold1_latest_time_index]) == 0.0 and float(features[threshold1_latest_norm_index]) > 0.0:
            events.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "time_sec": time_sec,
                    "threshold_n": 1.0,
                    "force_delta_norm": float(features[threshold1_latest_norm_index]),
                    "tcp_position_base": state[indices["position"]].astype(float).tolist(),
                }
            )
        if task_vector_dim:
            new_states.append(
                np.concatenate([state[:-task_vector_dim], features, state[-task_vector_dim:]]).astype(np.float32)
            )
        else:
            new_states.append(np.concatenate([state, features]).astype(np.float32))
    rows["observation.state"] = new_states
    return rows.sort_index(), events


def _validate_events(
    events: list[dict[str, Any]],
    data_frames: list[pd.DataFrame],
    short_window_sec: float,
    *,
    task_vector_dim: int,
) -> dict[str, Any]:
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        by_episode.setdefault(int(event["episode_index"]), []).append(event)
    duplicate_short_window_count = 0
    min_event_spacing_sec: float | None = None
    for ep_events in by_episode.values():
        times = [float(event["time_sec"]) for event in ep_events]
        for prev, curr in zip(times, times[1:]):
            spacing = curr - prev
            min_event_spacing_sec = spacing if min_event_spacing_sec is None else min(min_event_spacing_sec, spacing)
            if spacing < short_window_sec - 1e-9:
                duplicate_short_window_count += 1

    motion_checks = []
    all_frames = pd.concat(data_frames, ignore_index=True)
    for event in events:
        ep = int(event["episode_index"])
        t0 = float(event["time_sec"])
        p0 = np.asarray(event["tcp_position_base"], dtype=np.float64)
        future = all_frames[
            (all_frames["episode_index"].astype(int) == ep)
            & (all_frames["timestamp"].astype(float) >= t0)
            & (all_frames["timestamp"].astype(float) <= t0 + 0.5)
        ]
        max_motion = 0.0
        for _, row in future.iterrows():
            state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
            pos = state[:3].astype(np.float64)
            max_motion = max(max_motion, float(np.linalg.norm(pos - p0)))
        motion_checks.append(
            {
                "episode_index": ep,
                "frame_index": int(event["frame_index"]),
                "max_tcp_motion_500ms_m": max_motion,
                "has_position_response_gt_1mm": bool(max_motion > 0.001),
            }
        )

    return {
        "event_count": len(events),
        "episode_count_with_events": len(by_episode),
        "duplicate_short_window_count": duplicate_short_window_count,
        "min_event_spacing_sec": min_event_spacing_sec,
        "position_response_gt_1mm_count": sum(int(item["has_position_response_gt_1mm"]) for item in motion_checks),
        "motion_checks": motion_checks[:200],
    }


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(args.output_root)
        shutil.rmtree(args.output_root)
    shutil.copytree(args.dataset_root, args.output_root)

    info_path = args.output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info.get("fps", 20.0))
    state_spec = info["features"]["observation.state"]
    names = state_spec.get("names")
    if not isinstance(names, list):
        raise ValueError("observation.state must have names to append contact features")
    if any(name in names for name in CONTACT_RECOVERY_FEATURE_NAMES):
        raise ValueError("dataset already appears to contain contact/recovery features")
    indices = _state_indices(names)
    task_vector_dim = len(TASK_VECTOR_NAMES) if names[-len(TASK_VECTOR_NAMES):] == TASK_VECTOR_NAMES else 0
    config = ContactRecoveryFeatureConfig()

    all_states: list[Any] = []
    all_events: list[dict[str, Any]] = []
    processed_frames: list[pd.DataFrame] = []
    data_files = sorted((args.output_root / "data").rglob("*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {args.output_root / 'data'}")
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        groups = []
        for _, group in df.groupby("episode_index", sort=False):
            processed, events = _append_features_to_frame_group(
                group,
                indices=indices,
                fps=fps,
                config=config,
                task_vector_dim=task_vector_dim,
            )
            groups.append(processed)
            all_events.extend(events)
        out_df = pd.concat(groups).sort_index()
        all_states.extend(out_df["observation.state"].tolist())
        out_df.to_parquet(data_file, index=False)
        processed_frames.append(out_df)

    old_shape = int(state_spec["shape"][0])
    state_spec["shape"] = [old_shape + CONTACT_RECOVERY_FEATURE_DIM]
    if task_vector_dim:
        state_spec["names"] = [*names[:-task_vector_dim], *CONTACT_RECOVERY_FEATURE_NAMES, *names[-task_vector_dim:]]
    else:
        state_spec["names"] = [*names, *CONTACT_RECOVERY_FEATURE_NAMES]
    info["aic_contact_recovery_features"] = {
        "schema_version": "aic_contact_recovery_features/v3",
        "method": "causal_one_step_force_delta_threshold_memory",
        "feature_names": CONTACT_RECOVERY_FEATURE_NAMES,
        "config": asdict(config),
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True), encoding="utf-8")

    stats_path = args.output_root / "meta" / "stats.json"
    if stats_path.exists() and all_states:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["observation.state"] = _compute_stats(all_states)
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "fps": fps,
        "config": asdict(config),
        "feature_dim": CONTACT_RECOVERY_FEATURE_DIM,
        "validation": _validate_events(
            all_events,
            processed_frames,
            0.2,
            task_vector_dim=task_vector_dim,
        ),
    }
    report_path = args.validation_report or (args.output_root / "meta" / "contact_recovery_feature_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("output_root", "feature_dim", "validation")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
