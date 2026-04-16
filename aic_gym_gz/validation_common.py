"""Shared helpers for validation-only rollout scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .env import make_default_env


PolicyFn = Callable[[dict[str, Any], int], np.ndarray]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    description: str
    factory: Callable[[], PolicyFn]


def heuristic_action(observation: dict[str, Any]) -> np.ndarray:
    plug_position = observation["plug_pose"][:3]
    target_position = observation["target_port_pose"][:3]
    entrance_position = observation["target_port_entrance_pose"][:3]
    insertion_progress = float(observation["score_geometry"]["insertion_progress"][0])
    target_point = entrance_position if insertion_progress < 0.2 else target_position
    direction = target_point - plug_position
    linear = np.clip(2.5 * direction, -0.05, 0.05)
    action = np.zeros(6, dtype=np.float32)
    action[:3] = linear.astype(np.float32)
    return action


def _toward_target_factory() -> PolicyFn:
    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del step_idx
        return heuristic_action(observation)

    return _policy


def _aggressive_toward_target_factory() -> PolicyFn:
    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del step_idx
        plug_position = observation["plug_pose"][:3]
        target_position = observation["target_port_pose"][:3]
        entrance_position = observation["target_port_entrance_pose"][:3]
        insertion_progress = float(observation["score_geometry"]["insertion_progress"][0])
        target_point = entrance_position if insertion_progress < 0.2 else target_position
        direction = target_point - plug_position
        action = np.zeros(6, dtype=np.float32)
        action[:3] = np.clip(12.5 * direction, -0.25, 0.25).astype(np.float32)
        return action

    return _policy


def _away_from_target_factory() -> PolicyFn:
    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del step_idx
        action = -heuristic_action(observation)
        action[:3] = np.clip(action[:3], -0.05, 0.05)
        return action.astype(np.float32)

    return _policy


def _oscillate_in_place_factory() -> PolicyFn:
    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del observation
        action = np.zeros(6, dtype=np.float32)
        action[0] = 0.04 if step_idx % 2 == 0 else -0.04
        return action

    return _policy


def _collide_intentionally_factory() -> PolicyFn:
    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del observation, step_idx
        action = np.zeros(6, dtype=np.float32)
        action[2] = -0.25
        return action

    return _policy


def _random_factory(seed: int) -> PolicyFn:
    rng = np.random.default_rng(seed)

    def _policy(observation: dict[str, Any], step_idx: int) -> np.ndarray:
        del observation, step_idx
        return rng.uniform(
            low=np.array([-0.05, -0.05, -0.05, -0.1, -0.1, -0.1], dtype=np.float32),
            high=np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1], dtype=np.float32),
            size=(6,),
        ).astype(np.float32)

    return _policy


def policy_specs(*, seed: int) -> dict[str, PolicySpec]:
    return {
        "random": PolicySpec(
            name="random",
            description="Random bounded Cartesian velocity actions.",
            factory=lambda: _random_factory(seed),
        ),
        "heuristic": PolicySpec(
            name="heuristic",
            description="Simple plug-to-entrance then plug-to-target heuristic.",
            factory=_toward_target_factory,
        ),
        "toward_target": PolicySpec(
            name="toward_target",
            description="Fixed shaped policy that moves toward entrance/target.",
            factory=_toward_target_factory,
        ),
        "aggressive_toward_target": PolicySpec(
            name="aggressive_toward_target",
            description="Same target-seeking direction with much larger action magnitude.",
            factory=_aggressive_toward_target_factory,
        ),
        "away_from_target": PolicySpec(
            name="away_from_target",
            description="Fixed anti-progress policy that inverts the heuristic direction.",
            factory=_away_from_target_factory,
        ),
        "oscillate_in_place": PolicySpec(
            name="oscillate_in_place",
            description="Alternating x-axis oscillation to trigger action-delta and oscillation penalties.",
            factory=_oscillate_in_place_factory,
        ),
        "collide_intentionally": PolicySpec(
            name="collide_intentionally",
            description="Aggressive downward motion until off-limit contact/truncation.",
            factory=_collide_intentionally_factory,
        ),
    }


def rollout_policy(
    *,
    policy_name: str,
    seed: int,
    max_steps: int,
    ticks_per_step: int = 8,
) -> dict[str, Any]:
    specs = policy_specs(seed=seed)
    if policy_name not in specs:
        raise KeyError(f"Unknown policy {policy_name!r}. Available: {sorted(specs)}")
    env = make_default_env(ticks_per_step=ticks_per_step, enable_randomization=True)
    env.task.max_episode_steps = max(env.task.max_episode_steps, max_steps + 1)
    policy_fn = specs[policy_name].factory()
    try:
        observation, info = env.reset(seed=seed)
        records: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        final_info: dict[str, Any] | None = None
        cumulative_reward = 0.0
        for step_idx in range(max_steps):
            action = policy_fn(observation, step_idx)
            observation, reward, terminated, truncated, step_info = env.step(action)
            cumulative_reward += float(reward)
            reward_terms = dict(step_info["reward_terms"])
            reward_metrics = dict(step_info["reward_metrics"])
            records.append(
                {
                    "step_idx": step_idx,
                    "rl_step_reward": float(reward),
                    "cumulative_rl_reward": cumulative_reward,
                    "distance_to_target": float(step_info["distance_to_target"]),
                    "distance_to_entrance": float(step_info["distance_to_entrance"]),
                    "action_l2_norm": float(np.linalg.norm(action)),
                    "tcp_velocity_l2_norm": float(np.linalg.norm(observation["tcp_velocity"])),
                    "wrench_force_l2_norm": float(np.linalg.norm(observation["wrench"][:3])),
                    "wrench_timestamp": float(observation["wrench_timestamp"][0]),
                    "off_limit_contact": bool(observation["off_limit_contact"][0] > 0.5),
                    "sim_tick": int(observation["sim_tick"]),
                    "sim_time": float(observation["sim_time"]),
                    "reward_terms": reward_terms,
                    "reward_metrics": reward_metrics,
                }
            )
            if terminated or truncated:
                final_info = dict(step_info)
                break
        if final_info is None:
            final_info = dict(step_info)
        final_evaluation = final_info.get("final_evaluation")
        if final_evaluation is None:
            final_evaluation = env.task.final_evaluation()
            final_info["final_evaluation"] = final_evaluation
        return {
            "policy": policy_name,
            "description": specs[policy_name].description,
            "seed": seed,
            "trial_id": info["trial_id"],
            "ticks_per_step": ticks_per_step,
            "terminated": terminated,
            "truncated": truncated,
            "num_steps": len(records),
            "records": records,
            "final_evaluation": final_info.get("final_evaluation"),
        }
    finally:
        env.close()


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])
