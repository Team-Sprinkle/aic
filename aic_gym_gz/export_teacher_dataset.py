"""Export selected teacher candidates to dataset formats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.teacher.dataset_export import (
    export_teacher_jsonl_dataset,
    export_teacher_lerobot_dataset,
)
from aic_gym_gz.utils import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-artifact", required=True)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--format", choices=("jsonl", "lerobot"), default="jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-id", default="local/teacher_dataset")
    args = parser.parse_args()

    payload = json.loads(Path(args.search_artifact).read_text(encoding="utf-8"))
    candidate = next(
        item for item in payload["ranked_candidates"] if int(item["rank"]) == args.candidate_rank
    )
    if args.format == "jsonl":
        result = export_teacher_jsonl_dataset(candidate, output_dir=args.output_dir)
    else:
        result = export_teacher_lerobot_dataset(
            candidate,
            repo_id=args.repo_id,
            output_root=args.output_dir,
        )
    print(
        json.dumps(
            to_jsonable(
                {
                    "dataset_path": str(result.dataset_path),
                    "metadata_path": str(result.metadata_path),
                    "format": result.format,
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
