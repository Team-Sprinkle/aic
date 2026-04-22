"""Minimal live training smoke loop for the real aic_gym_gz environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic_gym_gz.env import live_env_health_check, make_live_env
    from aic_gym_gz.policies import deterministic_policy_actions
    from aic_gym_gz.utils import to_jsonable
else:
    from .env import live_env_health_check, make_live_env
    from .policies import deterministic_policy_actions
    from .utils import to_jsonable


def _select_action(*, policy: str, step_idx: int, rng: np.random.Generator) -> np.ndarray:
    if policy == "zero":
        return np.zeros(6, dtype=np.float32)
    if policy == "random":
        return rng.uniform(
            low=np.array([-0.01, -0.01, -0.01, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([0.01, 0.01, 0.01, 0.0, 0.0, 0.0], dtype=np.float32),
            size=(6,),
        ).astype(np.float32)
    scripted = deterministic_policy_actions()
    return scripted[min(step_idx, len(scripted) - 1)].as_env_action()


def run_training_smoke(
    *,
    include_images: bool = False,
    policy: str = "scripted",
    episodes: int = 2,
    max_steps: int = 16,
    seed: int = 123,
    attach_to_existing: bool = True,
    transport_backend: str = "cli",
) -> dict[str, Any]:
    health = live_env_health_check(
        include_images=include_images,
        seed=seed,
        attach_to_existing=attach_to_existing,
        transport_backend=transport_backend,
    )
    env = make_live_env(
        include_images=include_images,
        enable_randomization=False,
        attach_to_existing=attach_to_existing,
        transport_backend=transport_backend,
    )
    rng = np.random.default_rng(seed)
    episode_summaries: list[dict[str, Any]] = []
    try:
        for episode_index in range(episodes):
            observation, info = env.reset(seed=seed + episode_index)
            total_reward = 0.0
            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}
            for step_idx in range(max_steps):
                action = _select_action(policy=policy, step_idx=step_idx, rng=rng)
                observation, reward, terminated, truncated, final_info = env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            episode_summaries.append(
                {
                    "episode_index": episode_index,
                    "trial_id": info["trial_id"],
                    "steps": step_idx + 1,
                    "terminated": terminated,
                    "truncated": truncated,
                    "total_reward": total_reward,
                    "final_distance_to_target": float(final_info.get("distance_to_target", np.nan)),
                    "evaluation": final_info.get("evaluation"),
                    "final_sim_tick": int(observation["sim_tick"]),
                    "images_present": (
                        bool(
                            all(int(observation["images"][name].sum()) > 0 for name in ("left", "center", "right"))
                        )
                        if include_images
                        else False
                    ),
                }
            )
        return {
            "policy": policy,
            "include_images": include_images,
            "episodes": episode_summaries,
            "health": health,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--policy", choices=("scripted", "zero", "random"), default="scripted")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--attach-to-existing", action="store_true", default=True)
    parser.add_argument("--transport-backend", choices=("transport", "cli"), default="cli")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    payload = run_training_smoke(
        include_images=args.include_images,
        policy=args.policy,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        attach_to_existing=args.attach_to_existing,
        transport_backend=args.transport_backend,
    )
    if args.output:
        Path(args.output).write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
