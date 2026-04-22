"""Evaluate a single teacher rollout artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.teacher.analysis import analyze_rollout_artifact, load_json_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    artifact = load_json_file(args.artifact)
    result = analyze_rollout_artifact(artifact)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(result.summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.output_markdown:
        Path(args.output_markdown).write_text(result.markdown, encoding="utf-8")
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
