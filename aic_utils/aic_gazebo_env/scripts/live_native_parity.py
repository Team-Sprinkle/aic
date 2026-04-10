#!/usr/bin/env python3
"""Compare native Gazebo bridge access against GazeboCliClient on a live world."""

from __future__ import annotations

import json
import math
import ast
import re
import struct
import subprocess
import sys
import time

from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig
from aic_gazebo_env.protocol import GetObservationRequest, ResetRequest, StepRequest

WORLD = "aic_world"
SOURCE = "ati/tool_link"
TARGET = "tabletop"
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_ENTITY_IDS = [102, 103, 104, 105, 106, 107]
POLICY = [
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
    [0.02, -0.01, 0.015, 0.0, 0.0, 0.0],
]
JOINT_TOL = 1e-4
POS_TOL = 1e-4
ORI_TOL = 1e-3


def run(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def wait_for_live_topics() -> None:
    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            gz_topic(f"/world/{WORLD}/state")
            gz_topic(f"/world/{WORLD}/pose/info")
            return
        except Exception as exc:  # pragma: no cover - live-only diagnostic path
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for live Gazebo topics: {last_error}")


def gz_topic(topic: str) -> str:
    return run(["gz", "topic", "-e", "-n", "1", "-t", topic])


def decode_component_payload(payload: str) -> bytes:
    try:
        return ast.literal_eval(f'"{payload}"').encode("latin1")
    except (SyntaxError, ValueError):
        return payload.encode("utf-8").decode("unicode_escape").encode("latin1")


def gz_service(service: str, reqtype: str, reptype: str, req: str) -> str:
    return run(
        [
            "gz",
            "service",
            "-s",
            service,
            "--reqtype",
            reqtype,
            "--reptype",
            reptype,
            "--timeout",
            "10000",
            "--req",
            req,
        ]
    )


def decode_joint_positions() -> dict[str, float]:
    payload = gz_topic(f"/world/{WORLD}/state")
    blocks = re.findall(
        r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
        payload,
        re.S,
    )
    decoded: dict[int, float] = {}
    for entity_id_text, body in blocks:
        entity_id = int(entity_id_text)
        if entity_id not in JOINT_ENTITY_IDS:
            continue
        for match in re.finditer(
            r'type: (\d+)\n          component: "((?:\\.|[^"])*)"',
            body,
            re.S,
        ):
            component_type, component_payload = match.groups()
            if component_type != "8319580315957903596":
                continue
            raw = decode_component_payload(component_payload)
            if len(raw) >= 10 and raw[0] == 10 and raw[1] == 8:
                decoded[entity_id] = struct.unpack("<d", raw[2:10])[0]
                break
    return {
        joint_name: decoded[entity_id]
        for joint_name, entity_id in zip(JOINT_NAMES, JOINT_ENTITY_IDS)
    }


def decode_pose_entities() -> dict[str, dict[str, list[float]]]:
    payload = gz_topic(f"/world/{WORLD}/pose/info")
    entities: dict[str, dict[str, list[float]]] = {}
    for match in re.finditer(
        r'pose \{\n  name: "([^"]+)"\n(.*?)\n\}',
        payload,
        re.S,
    ):
        name, block = match.groups()
        position_match = re.search(
            r"position \{\n    x: ([^\n]+)\n    y: ([^\n]+)\n    z: ([^\n]+)\n  \}",
            block,
            re.S,
        )
        orientation_match = re.search(
            r"orientation \{\n(?:    x: ([^\n]+)\n)?(?:    y: ([^\n]+)\n)?(?:    z: ([^\n]+)\n)?(?:    w: ([^\n]+)\n)?  \}",
            block,
            re.S,
        )
        if position_match is None or orientation_match is None:
            continue
        entities[name] = {
            "position": [
                float(position_match.group(1)),
                float(position_match.group(2)),
                float(position_match.group(3)),
            ],
            "orientation": [
                float(orientation_match.group(1) or 0.0),
                float(orientation_match.group(2) or 0.0),
                float(orientation_match.group(3) or 0.0),
                float(orientation_match.group(4) or 0.0),
            ],
        }
    return entities


def normalize_quaternion(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    return [component / norm for component in quaternion]


def conjugate_quaternion(quaternion: list[float]) -> list[float]:
    x, y, z, w = quaternion
    return [-x, -y, -z, w]


def multiply_quaternions(left: list[float], right: list[float]) -> list[float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def orientation_error(source: list[float], target: list[float]) -> float:
    normalized_source = normalize_quaternion(source)
    normalized_target = normalize_quaternion(target)
    dot = sum(a * b for a, b in zip(normalized_source, normalized_target))
    clamped = min(1.0, max(0.0, abs(dot)))
    return 2.0 * math.acos(clamped)


def build_trace(step_index: int, joints: dict[str, float], poses: dict[str, dict[str, list[float]]]) -> dict[str, object]:
    source = poses[SOURCE]
    target = poses[TARGET]
    relative_position = [
        target_axis - source_axis
        for source_axis, target_axis in zip(source["position"], target["position"])
    ]
    distance = math.sqrt(sum(axis * axis for axis in relative_position))
    return {
        "step_index": step_index,
        "joint_positions": [joints[name] for name in JOINT_NAMES],
        "tcp_position": list(source["position"]),
        "relative_position": relative_position,
        "distance": distance,
        "orientation_error": orientation_error(
            source["orientation"],
            target["orientation"],
        ),
        "reward": -distance + (10.0 if distance <= 1.0 else 0.0),
        "terminated": distance <= 1.0,
        "truncated": False,
    }


def native_step(delta: list[float], step_index: int) -> dict[str, object]:
    joints = decode_joint_positions()
    targets = [joints[name] + axis for name, axis in zip(JOINT_NAMES, delta)]
    request = (
        'data: "model_name=ur;'
        f"joint_names={','.join(JOINT_NAMES)};"
        f"positions={','.join(str(value) for value in targets)}\""
    )
    gz_service(
        f"/world/{WORLD}/joint_target",
        "gz.msgs.StringMsg",
        "gz.msgs.Boolean",
        request,
    )
    time.sleep(0.2)
    return build_trace(step_index, decode_joint_positions(), decode_pose_entities())


def run_native_rollout() -> list[dict[str, object]]:
    wait_for_live_topics()
    traces: list[dict[str, object]] = []
    for step_index, delta in enumerate(POLICY):
        trace = native_step(delta, step_index)
        traces.append(trace)
        if trace["terminated"] or trace["truncated"]:
            break
    return traces


def run_env_rollout() -> list[dict[str, object]]:
    wait_for_live_topics()
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/tmp/aic.sdf",
            timeout=30.0,
            world_name=WORLD,
            source_entity_name=SOURCE,
            target_entity_name=TARGET,
            joint_command_model_name="ur",
            joint_names=JOINT_NAMES,
        )
    )
    traces: list[dict[str, object]] = []
    for step_index, delta in enumerate(POLICY):
        response = client.step(
            StepRequest(
                action={
                    "joint_position_delta": list(delta),
                    "multi_step": 1,
                }
            )
        )
        tracked = response.observation["task_geometry"]["tracked_entity_pair"]
        traces.append(
            {
                "step_index": step_index,
                "joint_positions": list(response.observation["joint_positions"]),
                "tcp_position": list(
                    response.observation["entities_by_name"][SOURCE]["position"]
                ),
                "relative_position": list(tracked["relative_position"]),
                "distance": float(tracked["distance"]),
                "orientation_error": float(tracked["orientation_error"]),
                "reward": float(response.reward),
                "terminated": bool(response.terminated),
                "truncated": bool(response.truncated),
            }
        )
        if response.terminated or response.truncated:
            break
    return traces


def compare(native: list[dict[str, object]], env: list[dict[str, object]]) -> dict[str, object]:
    for native_step_data, env_step_data in zip(native, env):
        joint_diff = max(
            abs(a - b)
            for a, b in zip(
                native_step_data["joint_positions"],
                env_step_data["joint_positions"],
            )
        )
        tcp_diff = max(
            abs(a - b)
            for a, b in zip(
                native_step_data["tcp_position"],
                env_step_data["tcp_position"],
            )
        )
        relative_diff = max(
            abs(a - b)
            for a, b in zip(
                native_step_data["relative_position"],
                env_step_data["relative_position"],
            )
        )
        distance_diff = abs(native_step_data["distance"] - env_step_data["distance"])
        orientation_diff = abs(
            native_step_data["orientation_error"] - env_step_data["orientation_error"]
        )
        reward_diff = abs(native_step_data["reward"] - env_step_data["reward"])
        flags_match = (
            native_step_data["terminated"] == env_step_data["terminated"]
            and native_step_data["truncated"] == env_step_data["truncated"]
        )
        if not (
            joint_diff <= JOINT_TOL
            and tcp_diff <= POS_TOL
            and relative_diff <= POS_TOL
            and distance_diff <= POS_TOL
            and orientation_diff <= ORI_TOL
            and reward_diff <= POS_TOL
            and flags_match
        ):
            return {
                "match": False,
                "first_mismatch": {
                    "step_index": native_step_data["step_index"],
                    "joint_diff": joint_diff,
                    "tcp_diff": tcp_diff,
                    "relative_diff": relative_diff,
                    "distance_diff": distance_diff,
                    "orientation_diff": orientation_diff,
                    "reward_diff": reward_diff,
                    "native": native_step_data,
                    "env": env_step_data,
                },
            }
    return {"match": len(native) == len(env), "first_mismatch": None}


def benchmark(label: str, traces: list[dict[str, object]], elapsed: float) -> dict[str, object]:
    return {
        "label": label,
        "steps": len(traces),
        "elapsed_s": elapsed,
        "steps_per_s": len(traces) / elapsed if elapsed > 0 else math.inf,
    }


def run_mode(mode: str) -> dict[str, object]:
    if mode == "native":
        start = time.perf_counter()
        traces = run_native_rollout()
        elapsed = time.perf_counter() - start
        return {
            "mode": mode,
            "policy": POLICY,
            "trace": traces,
            "benchmark": benchmark("native_toolkit_path", traces, elapsed),
        }
    if mode == "env":
        start = time.perf_counter()
        traces = run_env_rollout()
        elapsed = time.perf_counter() - start
        return {
            "mode": mode,
            "policy": POLICY,
            "trace": traces,
            "benchmark": benchmark("gazebo_cli_env_path", traces, elapsed),
        }
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: live_native_parity.py [native|env]")
    print(json.dumps(run_mode(sys.argv[1]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
