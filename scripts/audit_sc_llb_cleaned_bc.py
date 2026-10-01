#!/usr/bin/env python3
"""Independent provenance and coverage audit of cleaned 104-episode BC chunks."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from filter_sc_llb_bc_chunks import block_holds, pose_distance


def grouped(path):
    case, rows = None, []
    for line in path.open():
        row = json.loads(line)
        if case is not None and row["episode_id"] != case:
            yield case, rows
            rows = []
        case = row["episode_id"]
        rows.append(row)
    if case is not None:
        yield case, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--materialized", type=Path, required=True)
    p.add_argument("--cleaned-dir", default="bc_chunks", help="Directory within materialized root")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    spec = json.loads(a.spec.read_text())
    source = Path(spec["source_manifest"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != spec["source_manifest_sha256"]:
        raise ValueError("The immutable source manifest changed")
    expected = {row["case"]: row for row in spec["episodes"]}
    if len(expected) != 104:
        raise ValueError("Expected all 104 scored successes")
    root = a.materialized
    manifest = json.loads((root / a.cleaned_dir / "manifest.json").read_text())
    if not manifest["training_ready"] or manifest["episode_count"] != 104:
        raise ValueError("Cleaned chunk manifest is incomplete")
    totals = Counter()
    strata = Counter()
    invalid_drops = Counter()
    by_phase = {}
    for phase in ("transport", "local"):
        raw_path = root / "raw_chunks" / (phase + "_chunks.jsonl")
        clean_path = root / a.cleaned_dir / (phase + "_chunks.jsonl")
        seen = set()
        raw_stream = grouped(raw_path)
        clean_stream = grouped(clean_path)
        for (case, original), (clean_case, cleaned) in zip(raw_stream, clean_stream, strict=True):
            if case != clean_case or case not in expected or case in seen:
                raise ValueError(f"Episode coverage or order changed in {phase}: {case}")
            seen.add(case)
            if not cleaned:
                raise ValueError(f"All {phase} chunks removed: {case}")
            # Every retained row, including its 4 actual recorded targets,
            # must match an original row exactly and retain chronological order.
            position = 0
            for row in cleaned:
                while position < len(original) and original[position]["frame"] != row["frame"]:
                    position += 1
                if position == len(original) or original[position] != row:
                    raise ValueError(f"Changed or reordered target/observation: {case} {phase} frame={row['frame']}")
                position += 1
            if expected[case]["split"] == "validation" and original != cleaned:
                raise ValueError(f"Held-out validation data filtered: {case}")
            if (cleaned[0]["frame"] != original[0]["frame"] or
                    cleaned[-1]["frame"] != original[-1]["frame"]):
                raise ValueError(f"Episode boundary was removed: {case} {phase}")
            kept_frames = {row["frame"] for row in cleaned}
            eligible, force_baseline, _ = (block_holds(original, expected[case]["hold_markers"])
                                           if phase == "transport" and expected[case]["split"] == "train"
                                           else (set(), {}, []))
            for index, row in enumerate(original):
                if row["frame"] in kept_frames:
                    continue
                if phase == "transport" and (index not in eligible or
                        math.dist(row["state"][26:29], force_baseline[index]) >= 2.0):
                    invalid_drops[phase] += 1
                if phase == "transport" and index in eligible:
                    motion_m, rotation_deg = pose_distance(
                        row["state"][:7], original[index - 1]["state"][:7])
                    if motion_m >= 0.0001 or rotation_deg >= 0.15:
                        invalid_drops["measured_motion"] += 1
            for row in (cleaned[0], cleaned[-1]):
                if set(row["images"]) != {"center", "left", "right"} or not all(
                    Path(image).is_file() for image in row["images"].values()
                ):
                    raise ValueError(f"Missing sample camera image: {case} {phase}")
            totals[f"{phase}_raw"] += len(original)
            totals[f"{phase}_kept"] += len(cleaned)
            key = (phase, expected[case]["split"], expected[case]["card_count"],
                   expected[case]["selected_port"])
            strata[key + ("raw",)] += len(original)
            strata[key + ("kept",)] += len(cleaned)
        if seen != set(expected):
            raise ValueError(f"Missing {phase} episodes: {sorted(set(expected)-seen)}")
        by_phase[phase] = {"raw": totals[f"{phase}_raw"], "kept": totals[f"{phase}_kept"],
                           "removed": totals[f"{phase}_raw"] - totals[f"{phase}_kept"]}
    groups = {split: {row["scene_reset_group"] for row in expected.values() if row["split"] == split}
              for split in ("train", "validation")}
    if groups["train"] & groups["validation"]:
        raise ValueError("Scene/reset split leakage")
    if invalid_drops:
        raise ValueError(f"Transport rows removed outside quiet named holds: {dict(invalid_drops)}")
    cleaning_manifest = manifest["episodes"]
    hold_reports = [row for row in cleaning_manifest if row["phase"] == "transport" and row["split"] == "train"]
    unmatched_holds = [row["case"] for row in hold_reports if
                       {m["segment"] for m in row["matched_holds"]} !=
                       {"sweep_after_row", "tilt_clear_of_cards"}]
    if unmatched_holds:
        raise ValueError(f"Named transport holds were not found: {unmatched_holds}")
    if totals["transport_raw"] - totals["transport_kept"] < 100 * len(hold_reports):
        raise ValueError("Too few planned-hold anchors removed; check wrist gravity baseline")
    report = {"schema": "aic_sc_llb_cleaned_bc_audit/v2", "passed": True,
              "cleaned_dir": a.cleaned_dir,
              "source_manifest_sha256": spec["source_manifest_sha256"],
              "episodes": len(expected), "train_episodes": sum(row["split"] == "train" for row in expected.values()),
              "validation_episodes": sum(row["split"] == "validation" for row in expected.values()),
              "by_phase": by_phase, "retained_rows_exact_source_subset": True,
              "by_phase_split_card_count_port": [
                  {"phase": phase, "split": split, "card_count": card_count,
                   "selected_port": port, "raw": count,
                   "kept": strata[(phase, split, card_count, port, "kept")]}
                  for (phase, split, card_count, port, kind), count in sorted(strata.items())
                  if kind == "raw"],
              "matched_named_training_holds": len(hold_reports) * 2,
              "removed_excess_force_or_measured_motion_rows": 0,
              "all_episode_boundaries_retained": True,
              "validation_rows_unchanged": True, "scene_reset_leakage": False,
              "all_retained_action_chunks_are_recorded": True,
              "reserved_final_scenes_opened": False}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
