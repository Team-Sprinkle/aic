#!/usr/bin/env python3
"""Create cleaned BC chunks from command-derived SC LLB chunks, preserving sources.

Only training anchors are filtered. Validation anchors remain untouched. The
recorded four-command target chunk in each retained row is never modified.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import median


def pose7(value):
    if isinstance(value, dict):
        return value["position"] + value["orientation_xyzw"]
    return value


def pose_distance(a, b):
    a, b = pose7(a), pose7(b)
    translation = math.dist(a[:3], b[:3])
    dot = abs(sum(x * y for x, y in zip(a[3:7], b[3:7])))
    rotation_deg = math.degrees(2 * math.acos(max(-1.0, min(1.0, dot))))
    return translation, rotation_deg


def near_hold_target(row, marker):
    translation, rotation = pose_distance(row["teacher_tcp_target_poses_base_frame"][0],
                                          marker["tcp_target_base_frame"])
    return translation < 0.003 and rotation < 3.0


def block_holds(rows, markers):
    blocked = set()
    baseline_force = {}
    matches = []
    for marker in markers:
        runs = []
        current = []
        for index, row in enumerate(rows):
            if near_hold_target(row, marker):
                current.append(index)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        for run in runs:
            duration = rows[run[-1]]["sim_time"] - rows[run[0]]["sim_time"]
            if duration < marker["duration_s"] * 0.6:
                continue
            # The raw wrist wrench includes approximately 20 N of tool weight.
            # Compare to this stationary hold's measured baseline, not zero.
            baseline = [median(rows[index]["state"][axis] for index in run)
                        for axis in range(26, 29)]
            first = rows[run[0]]["sim_time"]
            last = rows[run[-1]]["sim_time"]
            for index in run:
                t = rows[index]["sim_time"]
                if t > first + 0.4 and t < last - 0.4:
                    blocked.add(index)
                    baseline_force[index] = baseline
            matches.append({"segment": marker["segment"], "observed_plateau_s": round(duration, 3),
                            "eligible_interior_anchors": sum(index in blocked for index in run)})
    return blocked, baseline_force, matches


def unchanging_local(row, previous):
    state, old = row["state"], previous["state"]
    tcp_mm, tcp_deg = pose_distance(state[:7], old[:7])
    targets = row["teacher_tcp_target_poses_base_frame"]
    old_targets = previous["teacher_tcp_target_poses_base_frame"]
    if len(targets) != 4 or len(old_targets) != 4:
        raise ValueError("Expected complete recorded 200 ms target chunks")
    target_changes = [pose_distance(target, old_target)
                      for target, old_target in zip(targets, old_targets)]
    force_change = math.dist(state[26:29], old[26:29])
    force_magnitude = math.sqrt(sum(value * value for value in state[26:29]))
    return (tcp_mm < 0.0001 and tcp_deg < 0.2 and
            all(distance < 0.0001 and angle < 0.2 for distance, angle in target_changes) and
            force_change < 0.2 and force_magnitude < 2.0)


def read_grouped(path):
    current_case, rows = None, []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            case = row["episode_id"]
            if current_case is not None and case != current_case:
                yield current_case, rows
                rows = []
            current_case = case
            rows.append(row)
    if current_case is not None:
        yield current_case, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--raw-transport", type=Path, required=True)
    parser.add_argument("--raw-local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; source and existing output are immutable")
    spec = json.loads(args.spec.read_text())
    if spec.get("status") != "metadata_only_action_chunks_pending" or spec.get("source_episode_count") != 104:
        parser.error("Expected the audited 104-episode cleaning specification")
    by_case = {episode["case"]: episode for episode in spec["episodes"]}
    args.output.mkdir(parents=True)
    totals = Counter()
    reports = []
    for phase, raw in (("transport", args.raw_transport), ("local", args.raw_local)):
        seen = set()
        with (args.output / f"{phase}_chunks.jsonl").open("w") as output:
            for case, rows in read_grouped(raw):
                if case not in by_case or case in seen:
                    raise ValueError(f"Unknown or repeated episode group: {case}")
                seen.add(case)
                episode = by_case[case]
                if any(row["split"] != episode["split"] or row["phase"] != phase or
                       row["official_tier3"] != 75 for row in rows):
                    raise ValueError(f"Chunk provenance/phase mismatch: {case}")
                blocked, baseline_force, matches = (block_holds(rows, episode["hold_markers"])
                                    if phase == "transport" and episode["split"] == "train"
                                    else (set(), {}, []))
                previous = None
                last_hold_kept_time = None
                kept = 0
                dropped_hold = 0
                dropped_still = 0
                for index, row in enumerate(rows):
                    if index in blocked:
                        # The target plateau may include robot settling or
                        # cable motion. Keep measured motion, excess force,
                        # and at least one anchor per second of the wait.
                        tcp_motion_m, tcp_rotation_deg = pose_distance(
                            row["state"][:7], rows[index - 1]["state"][:7])
                        quiet = math.dist(row["state"][26:29], baseline_force[index]) < 2.0
                        still = tcp_motion_m < 0.0001 and tcp_rotation_deg < 0.15
                        recent = (last_hold_kept_time is not None and
                                  row["sim_time"] - last_hold_kept_time < 1.0)
                        if quiet and still and recent:
                            dropped_hold += 1
                            continue
                        last_hold_kept_time = row["sim_time"]
                    if (phase == "local" and episode["split"] == "train" and previous is not None and
                        row["sim_time"] - previous["sim_time"] < 0.5 and unchanging_local(row, previous)):
                        dropped_still += 1
                        continue
                    output.write(json.dumps(row, separators=(",", ":")) + "\n")
                    previous = row
                    kept += 1
                totals[f"{phase}_raw"] += len(rows)
                totals[f"{phase}_kept"] += kept
                totals[f"{phase}_hold_dropped"] += dropped_hold
                totals[f"{phase}_still_dropped"] += dropped_still
                reports.append({"case": case, "phase": phase, "split": episode["split"],
                                "raw": len(rows), "kept": kept, "hold_dropped": dropped_hold,
                                "still_dropped": dropped_still, "matched_holds": matches})
        if seen != set(by_case):
            raise ValueError(f"Missing {phase} episode groups: {sorted(set(by_case)-seen)}")
    totals["transport"] = totals["transport_kept"]
    totals["local"] = totals["local_kept"]
    result = {"schema": "aic_sc_llb_cleaned_bc_chunks/v1", "source_spec": str(args.spec.resolve()),
              "raw_transport": str(args.raw_transport.resolve()), "raw_local": str(args.raw_local.resolve()),
              "episode_count": len(by_case), "counts": dict(totals), "episodes": reports,
              "source_episodes_mutated": False, "target_commands_modified": False,
              "validation_rows_filtered": False, "training_ready": True,
              "final_scenes_opened": False, "actor_geometry_input": False}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "counts": dict(totals)}))


if __name__ == "__main__":
    main()
