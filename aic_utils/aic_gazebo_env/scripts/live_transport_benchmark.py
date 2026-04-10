#!/usr/bin/env python3
"""Benchmark live Gazebo observation and step latency across transport modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from aic_gazebo_env.live_runtime import (
    DEFAULT_WORLD_NAME,
    LiveRuntimeManager,
)
from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig, GazeboTransportClient
from aic_gazebo_env.protocol import GetObservationRequest, ResetRequest, StepRequest


WORLD_NAME = DEFAULT_WORLD_NAME
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
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-build", action="store_true")
    parser.add_argument("--auto-launch", action="store_true")
    parser.add_argument("--worker-benchmark", action="store_true")
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Print only the final JSON payload.",
    )
    return parser.parse_args(argv)


def _base_health(mode: str) -> dict[str, object]:
    return {
        "mode": mode,
        "transport_backend": mode,
        "helper_startup_ok": False,
        "helper_ready_ok": False,
        "world_control_ok": False,
        "reset_ok": False,
        "first_observation_ok": False,
        "fallback_used": False,
    }


def _merge_health(health: dict[str, object], info: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(info, dict):
        return health
    for key in (
        "helper_startup_ok",
        "helper_ready_ok",
        "world_control_ok",
        "reset_ok",
        "first_observation_ok",
        "fallback_used",
        "transport_backend",
    ):
        if key in info:
            health[key] = info[key]
    return health


def _classify_error(exc: Exception) -> str:
    message = str(exc).lower()
    if "status=gz_not_found" in message:
        return "gz_not_found"
    if "status=helper_not_found" in message:
        return "helper_not_found"
    if "no such file or directory: 'gz'" in message:
        return "gz_not_found"
    if "setup_status=workspace_setup_script_found_but_not_sourced" in message:
        return "workspace_setup_script_found_but_not_sourced"
    if "setup_status=no_workspace_setup_script_found" in message:
        return "no_workspace_setup_script_found"
    if any(
        token in message
        for token in (
            "broken pipe",
            "helper exited",
            "json decode",
            "timed out waiting for transport bridge response",
            "could not resolve `aic_gz_transport_bridge`",
            "no such file or directory: 'gz'",
        )
    ):
        return "helper_process_or_ipc"
    if any(token in message for token in ("failed readiness handshake", "timed out waiting for initial transport samples", "state_callback_count")):
        return "state_topic_readiness"
    if any(token in message for token in ("reset service failure", "reset world advance failure", "joint target request failed", "/control")):
        return "world_control_service_timeout"
    if any(token in message for token in ("state publication not fresh yet", "timed out waiting for state sample", "timed out reading gazebo topic")):
        return "observation_freshness_timeout"
    if "reset observation failure" in message or "reset instability" in message:
        return "reset_instability"
    return "unknown"


def _error_payload(exc: Exception) -> dict[str, object]:
    return {
        "category": _classify_error(exc),
        "message": str(exc),
    }


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


def observation_benchmark(client) -> dict[str, object]:
    timings: list[float] = []
    last_info: dict[str, object] | None = None
    for _ in range(SAMPLES):
        start = time.perf_counter()
        response = client.get_observation(GetObservationRequest())
        timings.append(time.perf_counter() - start)
        last_info = response.info
    summary: dict[str, object] = summarize(timings)
    if isinstance(last_info, dict):
        summary.update(
            {
                "transport_backend": last_info.get("transport_backend"),
                "fallback_used": last_info.get("fallback_used", False),
                "helper_startup_ok": last_info.get("helper_startup_ok", False),
                "helper_ready_ok": last_info.get("helper_ready_ok", False),
                "first_observation_ok": last_info.get("first_observation_ok", True),
                "sim_step_count_raw": last_info.get("sim_step_count_raw"),
            }
        )
    return summary


def step_benchmark(client) -> dict[str, object]:
    reset_response = client.reset(ResetRequest(seed=0, options={"mode": "benchmark"}))
    timings: list[float] = []
    settled_flags: list[object] = []
    distance_values: list[float] = []
    last_info: dict[str, object] | None = reset_response.info
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
        last_info = response.info
    summary = summarize(timings)
    summary["final_distance"] = round(distance_values[-1], 6)
    if isinstance(last_info, dict):
        summary.update(
            {
                "transport_backend": last_info.get("transport_backend"),
                "fallback_used": last_info.get("fallback_used", False),
                "helper_startup_ok": last_info.get("helper_startup_ok", False),
                "helper_ready_ok": last_info.get("helper_ready_ok", False),
                "world_control_ok": last_info.get("world_control_ok", False),
                "reset_ok": reset_response.info.get("reset_ok", False),
                "first_observation_ok": last_info.get("first_observation_ok", False),
                "sim_step_count_raw": last_info.get("sim_step_count_raw"),
            }
        )
    return {
        **summary,
        "joint_target_settled": settled_flags,
    }


def probe_mode(client, mode: str) -> dict[str, object]:
    health = _base_health(mode)
    result: dict[str, object] = {"health": health}
    try:
        observation = client.get_observation(GetObservationRequest())
        _merge_health(health, observation.info)
        health["first_observation_ok"] = True
        result["initial_observation"] = {
            "entity_count": observation.observation.get("entity_count"),
            "joint_count": observation.observation.get("joint_count"),
            "sim_step_count_raw": observation.info.get("sim_step_count_raw"),
        }
    except Exception as exc:
        result["initial_observation_error"] = _error_payload(exc)
        return result

    try:
        reset_response = client.reset(ResetRequest(seed=0, options={"mode": "benchmark-probe"}))
        _merge_health(health, reset_response.info)
        result["reset_probe"] = {
            "transport_backend": reset_response.info.get("transport_backend"),
            "fallback_used": reset_response.info.get("fallback_used", False),
            "sim_step_count_raw": reset_response.info.get("sim_step_count_raw"),
        }
    except Exception as exc:
        result["reset_error"] = _error_payload(exc)
    return result


def run_benchmark() -> dict[str, object]:
    results: dict[str, object] = {}
    for mode in ("cli_one_shot", "cli_persistent", "transport_cpp"):
        client = make_client(mode)
        try:
            mode_result = probe_mode(client, mode)
            try:
                mode_result["observation"] = observation_benchmark(client)
            except Exception as exc:
                mode_result["observation_error"] = _error_payload(exc)
            try:
                mode_result["step"] = step_benchmark(client)
            except Exception as exc:
                mode_result["step_error"] = _error_payload(exc)
            if "initial_observation_error" in mode_result or "reset_error" in mode_result:
                mode_result["note"] = (
                    "Runtime is still unstable in this mode. See categorized errors to "
                    "separate helper readiness, world-control, observation freshness, "
                    "and reset instability."
                )
            results[mode] = mode_result
        finally:
            if hasattr(client, "close"):
                client.close()
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker_benchmark:
        print(json.dumps(run_benchmark(), indent=None if args.json_only else 2, sort_keys=True))
        return

    manager = LiveRuntimeManager(world_name=WORLD_NAME, world_path=WORLD_PATH)
    preflight = manager.preflight()
    context = manager.prepare(auto_build=args.auto_build, auto_launch=args.auto_launch)
    health = manager.wait_for_health(context, timeout_s=120.0).to_dict()
    if health.get("no_op_step_ok"):
        script_path = Path(__file__).resolve()
        command = (
            f"PYTHONPATH={manager.repo_root / 'aic_utils' / 'aic_gazebo_env'} "
            f"{sys.executable} {script_path} --worker-benchmark --json-only"
        )
        result = manager.run_context_command(context, command, timeout_s=300.0)
        benchmark_results = json.loads(result.stdout) if result.returncode == 0 else {
            "benchmark_error": {
                "category": "benchmark_worker_failed",
                "message": result.stderr or result.stdout,
            }
        }
    else:
        benchmark_results = {
            "benchmark_error": {
                "category": "live_health_unavailable",
                "message": health.get("diagnostics", {}).get("last_error", "live health checks did not pass"),
            }
        }
    payload = {
        "preflight": preflight,
        "context": context.to_dict(),
        "health": health,
        "results": benchmark_results,
    }
    recommendation = preflight.get("recommendation")
    if not args.json_only and recommendation:
        print(f"benchmark_preflight_recommendation: {recommendation}", file=sys.stderr)
    print(json.dumps(payload, indent=None if args.json_only else 2, sort_keys=True))


if __name__ == "__main__":
    main()
