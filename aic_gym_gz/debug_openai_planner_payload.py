"""Build and inspect the exact payload sent to the Responses API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic_gym_gz.env import make_default_env
from aic_gym_gz.planners.openai_backend import OpenAIPlannerBackend, OpenAIPlannerConfig
from aic_gym_gz.teacher.context import TeacherContextExtractor
from aic_gym_gz.teacher.history import TemporalObservationBuffer
from aic_gym_gz.teacher.quality import build_signal_quality_snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--prefer-live-scene-overview", action="store_true")
    args = parser.parse_args()

    env = make_default_env(enable_randomization=True, include_images=False)
    try:
        observation, _ = env.reset(seed=123)
        assert env._scenario is not None
        assert env._state is not None
        task_id = next(iter(env._scenario.tasks.keys()))
        history = TemporalObservationBuffer()
        history.append(
            state=env._state,
            action=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            signal_quality=build_signal_quality_snapshot(
                env._state,
                include_images=False,
                camera_info=observation.get("camera_info"),
            ),
        )
        planning_state = TeacherContextExtractor(
            prefer_live_scene_overview=args.prefer_live_scene_overview
        ).build_planning_state(
            scenario=env._scenario,
            task_id=task_id,
            state=env._state,
            temporal_buffer=history,
            current_phase="free_space_approach",
            recent_probe_results=[],
            include_images=False,
        )
    finally:
        env.close()

    backend = OpenAIPlannerBackend(
        OpenAIPlannerConfig(
            enabled=True,
            model=args.model,
        )
    )
    payload = backend.build_debug_payload(planning_state, candidate_index=args.candidate_index)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
