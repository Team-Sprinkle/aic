"""Execute one probe primitive on the mock env and report dynamics deltas."""

from __future__ import annotations

import argparse
import json

import numpy as np

from aic_gym_gz.env import make_default_env
from aic_gym_gz.probes.library import ProbeLibrary
from aic_gym_gz.teacher.history import TemporalObservationBuffer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="micro_sweep_xy")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = make_default_env(enable_randomization=True)
    probe_library = ProbeLibrary()
    try:
        observation, _ = env.reset(seed=args.seed)
        assert env._state is not None
        buffer_before = TemporalObservationBuffer()
        buffer_before.append(state=env._state, action=np.zeros(6, dtype=np.float32))
        before_state = env._state
        history = TemporalObservationBuffer()
        history.append(state=env._state, action=np.zeros(6, dtype=np.float32))
        for action in probe_library.actions_for(args.probe):
            observation, _, terminated, truncated, _ = env.step(action)
            assert env._state is not None
            history.append(state=env._state, action=action)
            if terminated or truncated:
                break
        after_state = env._state
        assert after_state is not None
        result = probe_library.summarize_result(
            probe_name=args.probe,
            before_state=before_state,
            after_state=after_state,
            before_summary=buffer_before,
            after_summary=history,
            action_count=len(probe_library.actions_for(args.probe)),
        )
        payload = {
            "probe": args.probe,
            "result": result.to_dict(),
            "final_distance_to_target": float(observation["plug_to_port_relative"][3]),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
