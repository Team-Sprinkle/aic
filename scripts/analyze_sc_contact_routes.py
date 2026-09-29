#!/usr/bin/env python3
"""Summarize named Isaac SC contacts and measured motion from route probes."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def magnitude(values):
    return math.sqrt(sum(float(v) ** 2 for v in values))


def peak_contact(row, key, first=3, last=8):
    matrix = row.get(key, [])
    return max((float(v) for group in matrix for v in group[first:last]), default=0.0)


def summarize(path):
    data = json.loads(path.read_text())
    rows = data["rows"]
    named = {}
    for label, key, first, last in (
        ("gripper_base_card", "debug_gripper_base_contact_by_scene_n", 3, 8),
        ("finger_l_card", "debug_gripper_finger_l_contact_by_scene_n", 3, 8),
        ("finger_r_card", "debug_gripper_finger_r_contact_by_scene_n", 3, 8),
        ("plug_card", "debug_sc_plug_contact_by_scene_n", 3, 8),
    ):
        values = [peak_contact(row, key, first, last) for row in rows]
        named[label] = {"peak_n": max(values), "steps_over_1n": sum(v > 1 for v in values),
                        "steps_over_100n": sum(v > 100 for v in values)}
    for link in (1, 2, 5, 10, 15, 19, 20):
        key = f"debug_rope_link{link}_scene_contact_by_scene_n"
        if any(key in row for row in rows):
            values = [peak_contact(row, key) for row in rows]
            named[f"rope_link{link}_card"] = {"peak_n": max(values),
                                               "steps_over_1n": sum(v > 1 for v in values)}
    windows = {}
    for start, stop in ((25, 99), (100, 110)):
        selected = [row for row in rows if start <= row["step"] <= stop]
        if len(selected) < 2:
            continue
        a, b = selected[0], selected[-1]
        windows[f"{start}_{stop}"] = {
            "gripper_motion_mm": 1000 * magnitude([x-y for x, y in zip(
                b["gripper_hande_base_link_position_world"], a["gripper_hande_base_link_position_world"])]),
            "plug_motion_mm": 1000 * magnitude([x-y for x, y in zip(
                b["sc_plug_link_position_world"], a["sc_plug_link_position_world"])]),
            "peak_command_m": max(float(row["cmd_root_pos_norm_m"]) for row in selected),
            "peak_wrist_force_n": max(float(row["force_norm"]) for row in selected),
        }
    return {"trace": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(rows),
            "contact_filter_paths": data.get("debug_contact_filter_paths", []),
            "named_contacts": named, "windows": windows,
            "final_tip_target_distance_mm": 1000 * float(rows[-1]["distance_m"]),
            "peak_wrist_force_n": max(float(row["force_norm"]) for row in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = {path.stem: summarize(path) for path in args.traces}
    result = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
