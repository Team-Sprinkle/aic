"""Benchmark helpers for the standalone env."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from .env import AicInsertionEnv, make_default_env


@dataclass(frozen=True)
class BenchmarkResult:
    reset_latency_s: float
    mean_step_latency_s: float
    simulated_seconds_per_wall_second: float
    samples_per_second: float


def benchmark_env(env: AicInsertionEnv, *, num_steps: int = 128, seed: int = 123) -> BenchmarkResult:
    t0 = time.perf_counter()
    observation, _ = env.reset(seed=seed)
    reset_latency = time.perf_counter() - t0
    sim_time_start = float(observation["sim_time"])

    wall_start = time.perf_counter()
    for _ in range(num_steps):
        action = env.action_space.sample().astype(np.float32)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset(seed=seed)
    wall_elapsed = time.perf_counter() - wall_start
    sim_elapsed = float(observation["sim_time"]) - sim_time_start
    return BenchmarkResult(
        reset_latency_s=reset_latency,
        mean_step_latency_s=wall_elapsed / max(num_steps, 1),
        simulated_seconds_per_wall_second=sim_elapsed / max(wall_elapsed, 1e-9),
        samples_per_second=num_steps / max(wall_elapsed, 1e-9),
    )


def main() -> None:
    env = make_default_env()
    result = benchmark_env(env)
    print("aic_gym_gz benchmark")
    print(f"  reset_latency_s: {result.reset_latency_s:.6f}")
    print(f"  mean_step_latency_s: {result.mean_step_latency_s:.6f}")
    print(
        "  simulated_seconds_per_wall_second: "
        f"{result.simulated_seconds_per_wall_second:.3f}"
    )
    print(f"  samples_per_second: {result.samples_per_second:.2f}")


if __name__ == "__main__":
    main()
