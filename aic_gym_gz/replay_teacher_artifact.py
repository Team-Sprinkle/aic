"""Replay a saved teacher artifact against the current env."""

from __future__ import annotations

import argparse
import json

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher import TeacherReplayRunner, load_teacher_replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()

    env = make_default_env(enable_randomization=True)
    try:
        artifact = load_teacher_replay(args.artifact)
        replay = TeacherReplayRunner(env=env).replay(artifact)
        print(json.dumps(replay, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
