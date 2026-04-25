#!/usr/bin/env python3
"""Build a review bundle manifest for later GPT-5 VLM critique."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.review import build_review_bundle, call_gpt5_failure_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, help="Smooth trajectory JSON.")
    parser.add_argument("--output", required=True, help="Review bundle manifest JSON.")
    parser.add_argument("--wrist-image-dir", help="Optional directory of wrist images.")
    parser.add_argument("--gazebo-image-dir", help="Optional directory of Gazebo images.")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--use-gpt5-review", action="store_true")
    parser.add_argument("--review-output", help="Optional GPT-5 review JSON output.")
    args = parser.parse_args()

    manifest = build_review_bundle(
        args.trajectory,
        args.output,
        wrist_image_dir=args.wrist_image_dir,
        gazebo_image_dir=args.gazebo_image_dir,
        samples=args.samples,
    )
    print(
        f"Wrote review bundle to {args.output}; "
        f"samples={len(manifest['samples'])}; "
        f"wrist_images={len(manifest['images']['wrist'])}; "
        f"gazebo_images={len(manifest['images']['gazebo'])}"
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
