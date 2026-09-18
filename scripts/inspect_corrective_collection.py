#!/usr/bin/env python3
"""Audit official corrective-data scores and render sampled terminal camera frames."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from merge_corrective_act_cache import inspect_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--sample-episodes", type=int, default=6)
    args = parser.parse_args()
    if args.sample_episodes < 1:
        raise ValueError("Need at least one sampled episode")
    accepted, rejected = inspect_collection(args.collection, 80.)
    root = args.collection / "verification"
    root.mkdir(exist_ok=False)
    records = [(record, rows) for record, rows, *_ in accepted]
    chosen = np.linspace(0, len(records) - 1, min(args.sample_episodes, len(records)), dtype=int) if records else []
    sampled = [(records[i][0], records[i][1], "ACCEPT") for i in chosen]
    for record in rejected[:2]:
        path = Path(record["episode_dir"]) / "frames.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows:
            sampled.append((record, rows, "REJECT"))
    pages = []
    for start in range(0, len(sampled), 2):
        entries = sampled[start:start + 2]
        canvas = Image.new("RGB", (864, len(entries) * 3 * 282), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (record, rows, decision) in enumerate(entries):
            for j, offset in enumerate([2, 1, 0]):
                row = min(rows, key=lambda row: abs(row["sim_time"] - (rows[-1]["sim_time"] - offset)))
                y = (i * 3 + j) * 282
                title = (f"{decision} {record['trial_id']} | total {record['official_total']:.2f} | "
                         f"Tier3 {record['official_tier3']:.2f} | end - {offset}s | left / center / right")
                draw.text((4, y + 5), title, fill="black")
                for col, camera in enumerate(["left", "center", "right"]):
                    with Image.open(Path(record["episode_dir"]) / row["images"][camera]) as image:
                        canvas.paste(image.convert("RGB").resize((288, 256)), (col * 288, y + 26))
        path = root / f"terminal_frames_{len(pages) + 1:02d}.jpg"
        canvas.save(path, quality=92)
        pages.append(str(path.resolve()))
    report = {"accepted": [entry[0] for entry in accepted], "rejected": rejected,
              "sampled_terminal_trials": [entry[0]["trial_id"] for entry in sampled], "contact_sheets": pages,
              "visual_review": "Contact sheets generated; human/assistant visual inspection must be recorded separately"}
    (root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": len(accepted), "rejected": len(rejected), "contact_sheets": pages}, indent=2))


if __name__ == "__main__":
    main()
