#!/usr/bin/env python3
"""Audit stored demonstrations and score lineage without labeling acceptance as success."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def finite_quantile(values, quantile):
    """Keep an audit report writable when a subset is empty or nonfinite."""
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, quantile)) if values.size else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.dataset_root / "meta/info.json").read_text())
    frames = pd.concat([pd.read_parquet(p) for p in sorted((args.dataset_root / "data").rglob("*.parquet"))], ignore_index=True)
    frames = frames.sort_values(["episode_index", "frame_index"])
    manifest = pd.read_csv(args.manifest)
    if manifest.accepted_episode_index.duplicated().any():
        raise ValueError("Ambiguous combined episode lineage")
    if set(manifest.accepted_episode_index) != set(frames.episode_index):
        raise ValueError("Manifest does not cover exactly the dataset episode IDs")
    actions = np.stack(frames.action).astype(np.float64)
    states = np.stack(frames["observation.state"]).astype(np.float64)
    names = info["features"]["observation.state"]["names"]
    velocity_indices = [names.index("tcp_velocity.linear." + axis) for axis in "xyz"]
    tcp_speed = np.linalg.norm(states[:, velocity_indices], axis=1)
    zero_action = np.linalg.norm(actions, axis=1) < 1e-8
    limits = np.array([0.02, 0.02, 0.02, 0.2, 0.2, 0.2])
    sources, labels = [], []
    for source, group in manifest.groupby("dataset_root", sort=True):
        root = Path(source)
        generation_path = root.parent / "generation_summary.json"
        generation = json.loads(generation_path.read_text()) if generation_path.exists() else {}
        stop_near_gate = (generation.get("agent_mode_env") or {}).get("AIC_OFFICIAL_TEACHER_STOP_AT_NEAR_GATE")
        report_path = root / "selection_report.csv"
        if not report_path.exists():
            report_path = root.parent / "selection_report.csv"
        selection = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
        sources.append({"source": source, "episodes": len(group), "source_exists": root.exists(),
                        "generation_summary_exists": generation_path.exists(),
                        "stop_at_near_gate": stop_near_gate,
                        "acceptance_min_total_score": generation.get("min_score"),
                        "selection_report": str(report_path) if report_path.exists() else None})
        for row in group.itertuples():
            matching = selection[selection.trial_id.astype(str) == str(row.trial_id)] if "trial_id" in selection else pd.DataFrame()
            scores = sorted(set(float(v) for v in matching.get("total_score", []) if np.isfinite(v)))
            # A scalar total does not supply a Tier 3 insertion outcome.
            labels.append({"episode_index": row.accepted_episode_index, "task_family": row.task_family,
                           "target_card_index": row.target_card_index, "target_port_index": row.target_port_index,
                           "source": source, "trial_id": row.trial_id, "stop_at_near_gate": stop_near_gate,
                           "historical_total_score": scores[0] if len(scores) == 1 else None,
                           "official_insertion_label": "unknown"})
    episodes = pd.DataFrame(labels).sort_values("episode_index")
    episode_ids = sorted(frames.episode_index.unique())
    num_val = max(1, int(len(episode_ids) * args.val_fraction))
    heldout_ids = set(episode_ids[-num_val:])
    episodes["split"] = episodes.episode_index.map(lambda n: "validation" if n in heldout_ids else "train")
    lengths = frames.groupby("episode_index").size()
    episodes["frame_count"] = episodes.episode_index.map(lengths)
    episodes.to_csv(args.output_dir / "episode_audit.csv", index=False)
    groups = ["split", "task_family", "target_card_index", "target_port_index"]
    balance = episodes.groupby(groups, dropna=False).agg(episodes=("episode_index", "size"), frames=("frame_count", "sum")).reset_index()
    balance.to_csv(args.output_dir / "task_balance.csv", index=False)
    differences = frames.groupby("episode_index").timestamp.diff().dropna().to_numpy()
    score_counts = episodes.historical_total_score.value_counts(dropna=False)
    result = {
        "dataset_root": str(args.dataset_root.resolve()), "manifest": str(args.manifest.resolve()),
        "frames": len(frames), "episodes": len(episode_ids), "state_dim": states.shape[1], "fps": info["fps"],
        "columns": frames.columns.tolist(), "reward_column_present": "reward" in frames,
        "nonfinite_action_values": int((~np.isfinite(actions)).sum()), "nonfinite_state_values": int((~np.isfinite(states)).sum()),
        "duplicate_episode_frame_pairs": int(frames.duplicated(["episode_index", "frame_index"]).sum()),
        "nonincreasing_timestamps": int((differences <= 0).sum()),
        "action_abs_quantiles_per_coordinate": {str(q): [finite_quantile(abs(actions[:, i]), q) for i in range(actions.shape[1])] for q in (0.5, 0.95, 0.99, 1.0)},
        "direct_actor_limits": limits.tolist(),
        "frames_outside_limits": int((abs(actions) > limits).any(axis=1).sum()),
        "outside_limits_fraction_by_coordinate": (abs(actions) > limits).mean(axis=0).tolist(),
        "zero_action_frame_fraction": float((np.linalg.norm(actions, axis=1) < 1e-8).mean()),
        "zero_action_frames_with_tcp_speed_above_1mm_s": int((zero_action & (tcp_speed > 0.001)).sum()),
        "zero_action_tcp_speed_quantiles_m_s": {str(q): finite_quantile(tcp_speed[zero_action], q) for q in (0.5, 0.9, 0.99)},
        "historical_total_score_counts": {str(k): int(v) for k, v in score_counts.items()},
        "official_insertion_labels": "not established by these scalar-score selection reports",
        "episodes_with_explicit_stop_near_gate": int(episodes.stop_at_near_gate.astype(str).str.lower().eq("true").sum()),
        "validation_episode_ids": sorted(int(n) for n in heldout_ids), "balance": balance.to_dict("records"),
        "sources": sources,
        "interpretation": "Acceptance, a total score, or an episode ending is not a verified insertion label. Missing provenance stays unknown.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"balance", "sources", "columns", "validation_episode_ids"}}, indent=2))


if __name__ == "__main__":
    main()
