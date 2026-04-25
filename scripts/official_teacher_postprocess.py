#!/usr/bin/env python3
"""Convert a piecewise official teacher plan into a smooth replay trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "aic_teacher_official"))

from aic_teacher_official.postprocess import postprocess_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Piecewise trajectory JSON path.")
    parser.add_argument("--output", required=True, help="Smooth trajectory JSON path.")
    parser.add_argument(
        "--sample-dt",
        type=float,
        default=0.05,
        help="Output sample period in seconds. Default: 0.05.",
    )
    args = parser.parse_args()

    smooth = postprocess_file(args.input, args.output, args.sample_dt)
    print(
        f"Wrote {len(smooth.waypoints)} smooth waypoints to {args.output} "
        f"from {args.input}"
    )


if __name__ == "__main__":
    main()
