#!/usr/bin/env python3
"""Audit verified SC LLB episodes and prepare a non-destructive BC cleanup manifest.

The output is deliberately marked metadata-only: action chunks still must be
extracted from the scored ROS bags and filtered before it can train an actor.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trial_score(audit):
    score = yaml.safe_load((audit / "results/scoring.yaml").read_text())
    trials = [value for key, value in score.items() if key.startswith("trial_")]
    if len(trials) != 1:
        raise ValueError(f"Expected one scored trial in {audit}")
    return float(trials[0]["tier_3"]["score"])


def hold_markers(plan):
    rows = json.loads((plan / "moveit_joint_trajectory.json").read_text())["waypoints"]
    result = []
    for index, row in enumerate(rows):
        marker = row.get("diagnostics", {})
        if marker.get("replay_modifier") != "endpoint_joint_hold":
            continue
        previous = rows[index - 1] if index else None
        if previous is None or marker.get("segment") != previous.get("diagnostics", {}).get("segment"):
            raise ValueError(f"Malformed held segment in {plan}")
        duration = float(marker["hold_duration_s"])
        if abs(float(row["timestamp"]) - float(previous["timestamp"]) - duration) > 1e-5:
            raise ValueError(f"Hold duration disagrees with timestamps in {plan}")
        result.append({
            "segment": marker["segment"],
            "plan_start_s": float(previous["timestamp"]),
            "plan_end_s": float(row["timestamp"]),
            "duration_s": duration,
            "tcp_target_base_frame": previous["tcp_pose"],
            "joint_target": previous["joint_positions"],
        })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; this script never overwrites a dataset")
    source = json.loads(args.source.read_text())
    if not source.get("complete") or source.get("official_full_insertions") != 104:
        raise ValueError("Expected the completed 104-success source manifest")
    episodes = []
    for episode in source["episodes"]:
        # The curated episode path is itself a symlink; its parent is the
        # dataset directory, not the collection containing the plan.
        audit = Path(episode["action_trace"]).parent
        if not audit.is_dir() or not Path(episode["bag"]).is_dir():
            raise FileNotFoundError(f"Missing scored episode or bag: {episode['case']}")
        if trial_score(audit) != 75 or episode["score_tier3"] != 75:
            raise ValueError(f"Non-success entered verified source: {episode['case']}")
        plan = audit.parents[1] / "plans" / (episode["case"] + "_insert")
        markers = hold_markers(plan)
        if {item["segment"] for item in markers} != {"sweep_after_row", "tilt_clear_of_cards"}:
            raise ValueError(f"Missing expected explicit holds: {episode['case']}")
        trace = Path(episode["action_trace"])
        events = [json.loads(line) for line in trace.open()]
        local_starts = [event for event in events if event.get("event") == "local_preinsert_align_started"]
        local_success = [event for event in events if event.get("event") == "guarded_insert_success"]
        if len(local_starts) != 1 or len(local_success) != 1:
            raise ValueError(f"Missing explicit local boundary/success: {episode['case']}")
        if local_starts[0]["time_sec"] >= local_success[0]["time_sec"]:
            raise ValueError(f"Invalid local event ordering: {episode['case']}")
        config = yaml.safe_load((audit / "eval_config.yaml").read_text())
        trial = next(iter(config["trials"].values()))
        start_state = json.loads((audit.parents[1] / "scenes" / episode["case"] /
                                  "measured_start_joint_state.json").read_text())
        reset_identity = {"scene": trial["scene"], "robot": config.get("robot"),
                          "start_joints": dict(zip(start_state["name"], start_state["position"]))}
        group = hashlib.sha256(json.dumps(reset_identity, sort_keys=True).encode()).hexdigest()
        episodes.append({
            "case": episode["case"], "selected_port": episode["selected_port"],
            "card_count": episode["card_count"], "scene_reset_group": group,
            "split": None,
            "source_episode": str(audit), "scored_bag": episode["bag"],
            "action_trace": str(trace), "insert_plan": str(plan),
            "hold_markers": markers,
            "local_start_sim_s": float(local_starts[0]["time_sec"]),
            "local_success_sim_s": float(local_success[0]["time_sec"]),
            "guarded_z_holds": sum(event.get("event") == "sc_guarded_insert_z_gate_hold" for event in events),
            "guarded_speed_advances": sum(event.get("event") == "guarded_insert_speed_gate_advance" for event in events),
        })
    if len(episodes) != 104 or len({row["case"] for row in episodes}) != 104:
        raise ValueError("Unexpected source episode count or duplicate case")
    # All present groups are unique, but retain a group-level assignment so
    # repeated resets cannot leak if this dataset is extended later.
    group_rows = {}
    for row in episodes:
        group_rows.setdefault(row["scene_reset_group"], []).append(row)
    strata = {}
    for group, rows in group_rows.items():
        keys = {(row["card_count"], row["selected_port"]) for row in rows}
        if len(keys) != 1:
            raise ValueError(f"Reset group spans card/port strata: {group}")
        strata.setdefault(next(iter(keys)), []).append(group)
    validation_groups = set()
    for groups in strata.values():
        groups.sort(key=lambda group: hashlib.sha256(("sc_llb_clean_v1:" + group).encode()).hexdigest())
        validation_groups.update(groups[:max(1, round(len(groups) * 0.2))])
    for row in episodes:
        row["split"] = "validation" if row["scene_reset_group"] in validation_groups else "train"
    train = {row["scene_reset_group"] for row in episodes if row["split"] == "train"}
    validation = {row["scene_reset_group"] for row in episodes if row["split"] == "validation"}
    if train & validation or not train or not validation:
        raise ValueError("Invalid episode-grouped split")
    summary = {split: {str(port): sum(row["split"] == split and row["selected_port"] == port for row in episodes)
                       for port in (0, 1)} for split in ("train", "validation")}
    if any(not summary[split][str(port)] for split in summary for port in (0, 1)):
        raise ValueError("Both selected ports must occur in each split")
    result = {
        "schema": "aic_sc_llb_cleaned_bc_manifest/v1",
        "status": "metadata_only_action_chunks_pending",
        "training_ready": False,
        "source_manifest": str(args.source.resolve()),
        "source_manifest_sha256": sha256(args.source),
        "source_episode_count": len(episodes),
        "split_counts_by_port": summary,
        "split_policy": "SHA256 rank of whole scene/reset group within each card-count/selected-port stratum; about 4:1 train/validation; no transition-level random split",
        "cleanup_policy": {
            "transport": "Drop the interior of exact endpoint_joint_hold plateaus after aligning them to recorded target commands; retain both transition edges and contiguous 200 ms chunks.",
            "local": "Preserve guarded contact decisions and progress. Downsample only repeated no-motion/no-target-change windows using measured TCP, force, and command traces; never interpolate actions across omitted time.",
            "geometry_inputs_at_deployment": False,
            "source_episodes_mutated": False,
        },
        "reserved_final_scenes_opened": False,
        "bc_training_started": False,
        "episodes": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "episodes": len(episodes),
                      "split_counts_by_port": summary, "training_ready": False}))


if __name__ == "__main__":
    main()
