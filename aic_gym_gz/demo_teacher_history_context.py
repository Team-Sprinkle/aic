"""Show how teacher-side history augments the base current observation."""

from __future__ import annotations

import json

import numpy as np

from aic_gym_gz.env import make_default_env
from aic_gym_gz.teacher.history import TemporalObservationBuffer
from aic_gym_gz.teacher.quality import build_signal_quality_snapshot


def main() -> None:
    env = make_default_env(enable_randomization=True, include_images=False)
    try:
        observation, _ = env.reset(seed=123)
        assert env._state is not None
        history = TemporalObservationBuffer()
        history.append(
            state=env._state,
            action=np.zeros(6, dtype=np.float32),
            signal_quality=build_signal_quality_snapshot(
                env._state,
                include_images=False,
                camera_info=observation.get("camera_info"),
            ),
        )
        for _ in range(2):
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert env._state is not None
            history.append(
                state=env._state,
                action=np.zeros(6, dtype=np.float32),
                signal_quality=build_signal_quality_snapshot(
                    env._state,
                    include_images=False,
                    camera_info=observation.get("camera_info"),
                ),
            )
            if terminated or truncated:
                break
        print(
            json.dumps(
                {
                    "official_compatible_current_observation": history.current_observation_view(),
                    "teacher_side_history": history.teacher_memory_summary(),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
