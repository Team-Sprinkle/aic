"""Emit teacher parity and compatibility audit tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.teacher.audit import (
    dataset_compatibility_rows,
    observation_parity_rows,
    scoring_parity_rows,
)
from aic_gym_gz.utils import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = {
        "observation_parity": [row.to_dict() for row in observation_parity_rows()],
        "scoring_parity": [row.to_dict() for row in scoring_parity_rows()],
        "dataset_compatibility": [row.to_dict() for row in dataset_compatibility_rows()],
    }
    text = json.dumps(to_jsonable(payload), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
