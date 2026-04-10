#!/usr/bin/env python3
"""Benchmark live Gazebo observation and step latency across transport modes."""

from __future__ import annotations

import json
import statistics
import time

from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig, GazeboTransportClient
from aic_gazebo_env.protocol import GetObservationRequest, ResetRequest, StepRequest


WORLD_NAME = "aic_world"
WORLD_PATH = "/tmp/aic.sdf"
SOURCE_ENTITY = "ati/tool_link"
TARGET_ENTITY = "tabletop"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
INITIAL_JOINTS = (-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110)
JOINT_DELTA = (0.02, -0.01, 0.015, 0.0, 0.0, 0.0)
SAMPLES = 5


def make_config(*, transport_backend: str, observation_transport: str) -> GazeboCliClientConfig:
    return GazeboCliClientConfig(
        executable="gz",
        world_path=WORLD_PATH,
        timeout=10.0,
        world_name=WORLD_NAME,
        source_entity_name=SOURCE_ENTITY,
        target_entity_name=TARGET_ENTITY,
        joint_command_model_name="ur5e",
        joint_names=JOINT_NAMES,
        initial_joint_positions=INITIAL_JOINTS,
        transport_backend=transport_backend,
        observation_transport=observation_transport,
    )


def make_client(mode: str):
    if mode == "cli_one_shot":
        return GazeboCliClient(make_config(transport_backend="cli", observation_transport="one_shot"))
    if mode == "cli_persistent":
        return GazeboCliClient(make_config(transport_backend="cli", observation_transport="persistent"))
    if mode == "transport_cpp":
        return GazeboTransportClient(
            make_config(transport_backend="transport", observation_transport="persistent")
        )
    raise ValueError(f"unsupported mode {mode}")


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "median_ms": round(statistics.median(values) * 1000.0, 3),
        "worst_ms": round(max(values) * 1000.0, 3),
        "best_ms": round(min(values) * 1000.0, 3),
    }


def observation_benchmark(client) -> dict[str, float]:
    timings: list[float] = []
    for _ in range(SAMPLES):
        start = time.perf_counter()
        client.get_observation(GetObservationRequest())
        timings.append(time.perf_counter() - start)
    return summarize(timings)


def step_benchmark(client) -> dict[str, object]:
    client.reset(ResetRequest(seed=0, options={"mode": "benchmark"}))
    timings: list[float] = []
    settled_flags: list[object] = []
    distance_values: list[float] = []
    for sample in range(SAMPLES):
        sign = 1.0 if sample % 2 == 0 else -1.0
        positions = [
            initial + (sign * delta)
            for initial, delta in zip(INITIAL_JOINTS, JOINT_DELTA)
        ]
        start = time.perf_counter()
        response = client.step(
            StepRequest(
                action={
                    "set_joint_positions": {
                        "model_name": "ur5e",
                        "joint_names": list(JOINT_NAMES),
                        "positions": positions,
                    },
                    "multi_step": 1,
                }
            )
        )
        timings.append(time.perf_counter() - start)
        settled_flags.append(response.info.get("joint_target_settled"))
        distance_values.append(
            float(response.observation["task_geometry"]["tracked_entity_pair"]["distance"])
        )
    summary = summarize(timings)
    summary["final_distance"] = round(distance_values[-1], 6)
    return {
        **summary,
        "joint_target_settled": settled_flags,
    }


def main() -> None:
    results: dict[str, object] = {}
    for mode in ("cli_one_shot", "cli_persistent", "transport_cpp"):
        client = make_client(mode)
        try:
            results[mode] = {}
            try:
                results[mode]["observation"] = observation_benchmark(client)
            except Exception as exc:
                results[mode]["observation_error"] = str(exc)
            try:
                results[mode]["step"] = step_benchmark(client)
            except Exception as exc:
                results[mode]["step_error"] = str(exc)
        finally:
            if hasattr(client, "close"):
                client.close()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
