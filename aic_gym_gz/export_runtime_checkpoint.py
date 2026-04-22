"""Export a runtime checkpoint for replay / branch-and-evaluate tooling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .env import make_default_env


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    env = make_default_env()
    try:
        env.reset(seed=args.seed)
        env.step(np.array([0.02, -0.01, 0.0, 0.0, 0.0, 0.05], dtype=np.float32))
        checkpoint = env.runtime.export_checkpoint()
        Path(args.output).write_text(
            json.dumps(
                {
                    "mode": checkpoint.mode,
                    "exact": checkpoint.exact,
                    "limitations": checkpoint.limitations,
                    "summary": {
                        "checkpoint_label": "runtime_checkpoint",
                        "mock_restore_semantics": "exact" if checkpoint.exact else "approximate",
                        "step_reward_label": "rl_step_reward",
                        "final_score_label": "gym_final_score",
                    },
                    "payload": checkpoint.payload,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
