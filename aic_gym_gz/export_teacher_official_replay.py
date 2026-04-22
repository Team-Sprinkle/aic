"""Export a selected teacher candidate as an official replay artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.teacher.search import export_selected_candidate_to_replay
from aic_gym_gz.utils import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-artifact", required=True)
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.search_artifact).read_text(encoding="utf-8"))
    artifact = export_selected_candidate_to_replay(
        payload,
        candidate_rank=args.candidate_rank,
        output_path=args.output,
    )
    print(
        json.dumps(
            to_jsonable(
                {
                    "output": args.output,
                    "trial_id": artifact.metadata["trial_id"],
                    "task_id": artifact.metadata["task_id"],
                    "candidate_rank": artifact.metadata["candidate_rank"],
                }
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
