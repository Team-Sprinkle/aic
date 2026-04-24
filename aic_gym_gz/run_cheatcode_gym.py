"""Run a thin CheatCode-inspired staged policy inside the gazebo gym env."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aic_gym_gz.env import make_default_env, make_live_env
from aic_gym_gz.teacher.close_range import CloseRangeInsertionPolicy
from aic_gym_gz.teacher.dataset_export import RolloutDatasetFrame, export_rollout_lerobot_dataset
from aic_gym_gz.utils import to_jsonable
from aic_gym_gz.video import HeadlessTrajectoryVideoRecorder, build_run_name, default_video_output_dir


class CheatCodeGymAdapter(CloseRangeInsertionPolicy):
    """Thin gym-native adapter inspired by the official TF-based CheatCode policy.

    The official policy issues absolute pose targets using ground-truth TF in
    `base_link`. This adapter preserves the staged behavior but maps it onto the
    gym's native 6D Cartesian velocity action interface.
    """


def _rollout_fps_from_ticks(ticks_per_step: int) -> int:
    return max(1, int(round(1.0 / (max(int(ticks_per_step), 1) * 0.002))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--ticks-per-step", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--attach-to-existing", action="store_true")
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="transport")
    parser.add_argument("--live-timeout", type=float, default=20.0)
    parser.add_argument("--attach-ready-timeout", type=float, default=90.0)
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--dataset-repo-id", default="local/aic_gym_gz_cheatcode")
    parser.add_argument("--dataset-root", default="aic_gym_gz/artifacts/lerobot_datasets")
    parser.add_argument("--dataset-no-videos", action="store_true")
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--output", default="aic_gym_gz/artifacts/cheatcode_gym_summary.json")
    args = parser.parse_args()

    include_images = bool(args.include_images or not args.disable_video)
    image_shape = (int(args.image_height), int(args.image_width), 3)
    if include_images and not args.live:
        raise RuntimeError(
            "Real images/video require --live. "
            "The default mock env does not provide real wrist-camera frames."
        )
    env = (
        make_live_env(
            include_images=include_images,
            enable_randomization=False,
            ticks_per_step=args.ticks_per_step,
            attach_to_existing=args.attach_to_existing,
            transport_backend=args.transport_backend,
            timeout=args.live_timeout,
            attach_ready_timeout=args.attach_ready_timeout,
            image_shape=image_shape,
        )
        if args.live
        else make_default_env(
            include_images=include_images,
            enable_randomization=True,
            ticks_per_step=args.ticks_per_step,
            image_shape=image_shape,
        )
    )
    try:
        episode_summaries: list[dict[str, Any]] = []
        dataset_outputs: list[dict[str, Any]] = []
        for episode in range(args.episodes):
            observation, info = env.reset(seed=args.seed + episode)
            scenario = env._scenario
            state = env._state
            if scenario is None or state is None:
                raise RuntimeError("Environment did not expose scenario/state after reset.")
            run_name = build_run_name(
                prefix=f"cheatcode_gym_ep{episode}",
                seed=args.seed + episode,
                trial_id=info.get("trial_id"),
            )
            recorder = HeadlessTrajectoryVideoRecorder(
                output_dir=Path(args.video_dir) / f"episode_{episode}"
                if args.video_dir
                else default_video_output_dir(run_name=run_name),
                enabled=not args.disable_video,
                require_real_wrist_images=not args.disable_video,
                require_live_overview=not args.disable_video,
            )
            recorder.capture(observation=observation, scenario=scenario, state=state)
            policy = CheatCodeGymAdapter()
            total_reward = 0.0
            terminated = truncated = False
            last_info = dict(info)
            step_count = 0
            dataset_frames = [
                RolloutDatasetFrame(
                    kind="reset",
                    observation=dict(observation),
                    action=[0.0] * 6,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info=dict(info),
                    planner_rationale=None,
                    phase=policy.phase,
                )
            ]
            while not (terminated or truncated):
                action = policy.action(observation)
                observation, reward, terminated, truncated, last_info = env.step(action)
                total_reward += float(reward)
                step_count += 1
                state = env._state
                assert state is not None
                recorder.capture(observation=observation, scenario=scenario, state=state)
                dataset_frames.append(
                    RolloutDatasetFrame(
                        kind="policy_step",
                        observation=dict(observation),
                        action=[float(value) for value in action.tolist()],
                        reward=float(reward),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        info=dict(last_info),
                        planner_rationale=None,
                        phase=policy.phase,
                    )
                )
                if step_count >= args.max_steps and not (terminated or truncated):
                    truncated = True
                    last_info = dict(last_info)
                    last_info["termination_reason"] = "max_steps_guard"
            final_eval = dict(last_info.get("final_evaluation") or {})
            video_summary = recorder.close()
            dataset_repo_id = f"{args.dataset_repo_id}_ep{episode}"
            dataset_result = export_rollout_lerobot_dataset(
                dataset_frames,
                repo_id=dataset_repo_id,
                output_root=Path(args.dataset_root),
                single_task="Insert cable into target port",
                fps=_rollout_fps_from_ticks(args.ticks_per_step),
                use_videos=not args.dataset_no_videos,
                metadata={
                    "trial_id": info.get("trial_id"),
                    "seed": args.seed + episode,
                    "policy": "CheatCodeGymAdapter",
                    "ticks_per_step": int(args.ticks_per_step),
                },
            )
            dataset_outputs.append(
                {
                    "episode": episode,
                    "dataset_path": str(dataset_result.dataset_path),
                    "metadata_path": str(dataset_result.metadata_path),
                    "format": dataset_result.format,
                }
            )
            episode_summaries.append(
                {
                    "episode": episode,
                    "seed": args.seed + episode,
                    "trial_id": info.get("trial_id"),
                    "return": total_reward,
                    "length": int(observation["step_count"]),
                    "termination_reason": (
                        "terminated"
                        if terminated
                        else ("truncated" if truncated else "completed")
                    ),
                    "gym_final_score": final_eval.get("gym_final_score"),
                    "phase_at_end": policy.phase,
                    "video_output": video_summary,
                    "dataset_output": dataset_outputs[-1],
                }
            )
        payload = {
            "adapter_notes": [
                "The official CheatCode policy uses TF ground truth and absolute pose targets in base_link.",
                "This gym adapter preserves the staged strategy but outputs native 6D Cartesian velocity actions.",
                "The gym observation does not expose the full official TF tree or the exact motion-update interface.",
            ],
            "episodes": episode_summaries,
            "datasets": dataset_outputs,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
