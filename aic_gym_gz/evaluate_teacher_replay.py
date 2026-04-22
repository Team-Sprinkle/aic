"""Replay an artifact and evaluate replay faithfulness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher.analysis import analyze_replay_comparison
from aic_gym_gz.teacher.replay import TeacherReplayRunner, load_teacher_replay
from aic_gym_gz.utils import to_jsonable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-markdown", default=None)
    args = parser.parse_args()

    artifact = load_teacher_replay(args.artifact)
    env = make_default_env(enable_randomization=True)
    try:
        replayed = TeacherReplayRunner(env=env).replay(artifact)
    finally:
        env.close()
    result = analyze_replay_comparison(original=artifact, replayed=replayed)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(to_jsonable(result.summary), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.output_markdown:
        Path(args.output_markdown).write_text(result.markdown, encoding="utf-8")
    print(json.dumps(to_jsonable(result.summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
