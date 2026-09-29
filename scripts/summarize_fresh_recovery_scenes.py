#!/usr/bin/env python3
"""Freeze scored fresh-scene route diagnostics with hashes and trace metrics."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_cohorts", ROOT / "scripts/summarize_recovery_trace_cohorts.py")
cohorts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohorts)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bulk-root", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.trace_dir = args.trace_dir.resolve()
    rows = []
    for number in (1, 2, 3):
        name = f"fresh_scene_{number:02}"
        root = args.bulk_root / name
        scoring = yaml.safe_load((root / "results/scoring.yaml").read_text())
        trial = scoring["trial_000001"]
        video = ROOT / "artifacts/hierarchical_recovery" / name / "all_views_20fps.mp4"
        config = ROOT / "configs/hierarchical_recovery" / f"{name}.eval_config.yaml"
        rows.append({
            "scene": name,
            "route": ("stock_cheatcode" if number == 3 else
                      "privileged_across_cards_then_stock_cheatcode"),
            "official_total": scoring["total"],
            "official_tier3": trial["tier_3"]["score"],
            "official_tier3_message": trial["tier_3"]["message"],
            "official_force_message": trial["tier_2"]["categories"]["insertion force"]["message"],
            "config": str(config.relative_to(ROOT)), "config_sha256": sha(config),
            "bag": str(next((root / "results").glob("bag_trial_000001_*"))),
            "video": str(video.relative_to(ROOT)), "video_sha256": sha(video),
            "trace": str((args.trace_dir / f"{name}_trace.jsonl").relative_to(ROOT)),
            "trace_metrics": cohorts.summary(args.trace_dir / f"{name}_trace.jsonl"),
        })
    result = {
        "schema": "aic_fresh_recovery_scenes/v1",
        "official_eval_image_digest": "sha256:9aa2ffdbb946d38edde1bac7b5f02a44cfbea26e3b04a9c74e09f14c97472923",
        "normal_collisions": True,
        "autonomous_policy": False,
        "scenes": rows,
        "interpretation": "Three development variations provide one full and two partial controls for the post-hoc 80 mm tracking line. The third perturbs a previously non-inserting rail-0 layout by 7 mm in board X and 4 mm in board Y yet becomes partial; no new no-insertion sensitivity test, named contact, or learned recovery result.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scenes": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
