"""Collect temporal diagnostics over a short rollout."""

from __future__ import annotations

import argparse
import json

import numpy as np

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher.history import TemporalObservationBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    env = make_default_env(enable_randomization=True)
    history = TemporalObservationBuffer()
    try:
        env.reset(seed=args.seed)
        assert env._state is not None
        zero = np.zeros(6, dtype=np.float32)
        history.append(state=env._state, action=zero)
        snapshots = [history.compact_state()]
        for _ in range(args.steps):
            env.step(zero)
            assert env._state is not None
            history.append(state=env._state, action=zero)
            snapshots.append(history.compact_state())
        print(json.dumps({"snapshots": snapshots}, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
