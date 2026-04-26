#!/usr/bin/env python3
"""Build a review bundle manifest for later GPT-5 VLM critique."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.review import (
    build_comparison_review_bundle,
    build_review_bundle,
    call_gpt5_failure_review,
)


def _comparison_run(value: str) -> dict[str, str]:
    parts = value.split("|")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--comparison-run must be label|trajectory_json|dataset_root|scoring_yaml"
        )
    return {
        "label": parts[0],
        "trajectory_path": parts[1],
        "dataset_root": parts[2],
        "scoring_path": parts[3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", help="Smooth trajectory JSON.")
    parser.add_argument("--output", required=True, help="Review bundle manifest JSON.")
    parser.add_argument("--wrist-image-dir", help="Optional directory of wrist images.")
    parser.add_argument("--gazebo-image-dir", help="Optional directory of Gazebo images.")
    parser.add_argument("--dataset-root", help="Optional LeRobot dataset root for actions/observations/videos.")
    parser.add_argument("--scoring-path", help="Optional official scoring.yaml path.")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--comparison-run",
        action="append",
        type=_comparison_run,
        default=[],
        help=(
            "Add a run to a multi-loop comparison bundle as "
            "label|trajectory_json|dataset_root|scoring_yaml. Repeat for loop_1, loop_2, etc."
        ),
    )
    parser.add_argument("--use-gpt5-review", action="store_true")
    parser.add_argument("--review-output", help="Optional GPT-5 review JSON output.")
    args = parser.parse_args()

    if args.comparison_run:
        manifest = build_comparison_review_bundle(
            args.comparison_run,
            args.output,
            samples=args.samples,
        )
    else:
        if not args.trajectory:
            raise SystemExit("--trajectory is required unless --comparison-run is provided.")
        manifest = build_review_bundle(
            args.trajectory,
            args.output,
            wrist_image_dir=args.wrist_image_dir,
            gazebo_image_dir=args.gazebo_image_dir,
            dataset_root=args.dataset_root,
            scoring_path=args.scoring_path,
            samples=args.samples,
        )
    print(
        f"Wrote review bundle to {args.output}; "
        f"samples={manifest.get('samples_per_run', len(manifest.get('samples', [])))}; "
        f"runs={len(manifest.get('runs', []))}; "
        f"wrist_images={sum(len(run.get('images', {}).get('wrist', [])) for run in manifest.get('runs', [])) if manifest.get('runs') else len(manifest['images']['wrist'])}; "
        f"gazebo_images={sum(len(run.get('images', {}).get('gazebo', [])) for run in manifest.get('runs', [])) if manifest.get('runs') else len(manifest['images']['gazebo'])}"
    )
    if args.use_gpt5_review:
        review = call_gpt5_failure_review(manifest)
        review_output = args.review_output or args.output.replace(".json", "_gpt5_review.json")
        Path(review_output).write_text(
            __import__("json").dumps(review, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote GPT-5 review to {review_output}")


if __name__ == "__main__":
    main()
