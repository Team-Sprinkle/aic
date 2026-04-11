#!/usr/bin/env python3
"""Run one native Gazebo joint-delta step and print comparable trace fields."""

from __future__ import annotations

import ast
import json
import math
import re
import struct
import subprocess
import time

WORLD = "aic_world"
SOURCE = "ati/tool_link"
TARGET = "tabletop"
SOURCE_ID = 79
JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
JOINT_ENTITY_IDS = [102, 103, 104, 105, 106, 107]
JOINT_DELTA = [0.02, -0.01, 0.015, 0.0, 0.0, 0.0]
TABLETOP_POSITION = [-0.2, 0.2, 1.14]
TABLETOP_YAW = -3.141
SETTLE_TOL = 5e-3
SETTLE_TIMEOUT_S = 3.0


def run(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout


def gz_topic(topic: str) -> str:
    try:
        return run(["gz", "topic", "-e", "-n", "1", "-t", topic])
    except subprocess.TimeoutExpired:
        if topic.endswith("/state"):
            return read_state_after_world_step(topic)
        raise


def read_state_after_world_step(topic: str) -> str:
    process = subprocess.Popen(
        ["gz", "topic", "-e", "-n", "1", "-t", topic],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.5)
        subprocess.run(
            [
                "gz",
                "service",
                "-s",
                f"/world/{WORLD}/control",
                "--reqtype",
                "gz.msgs.WorldControl",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "10000",
                "--req",
                "multi_step: 1",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        stdout, stderr = process.communicate(timeout=15)
    except Exception:
        process.kill()
        process.communicate()
        raise
    if stderr.strip():
        raise RuntimeError(stderr.strip())
    return stdout


def decode_component_payload(payload: str) -> bytes:
    return ast.literal_eval(f'"{payload}"').encode("latin1")


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


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> list[float]:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def decode_source_pose() -> dict[str, list[float]]:
    payload = gz_topic(f"/world/{WORLD}/state")
    blocks = re.findall(
        r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
        payload,
        re.S,
    )
    for entity_id_text, body in blocks:
        if int(entity_id_text) != SOURCE_ID:
            continue
        match = re.search(
            r'type: 10918813941671183356\n\s+component: "([^"]+)"',
            body,
            re.S,
        )
        if match is None:
            break
        values = [float(value) for value in match.group(1).split()]
        if len(values) != 6:
            break
        x, y, z, roll, pitch, yaw = values
        return {
            "position": [x, y, z],
            "orientation": quaternion_from_rpy(roll, pitch, yaw),
        }
    raise RuntimeError("ati/tool_link pose not found in world state")


def normalize_quaternion(quaternion: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in quaternion))
    return [component / norm for component in quaternion]


def orientation_error(source: list[float], target: list[float]) -> float:
    normalized_source = normalize_quaternion(source)
    normalized_target = normalize_quaternion(target)
    dot = sum(a * b for a, b in zip(normalized_source, normalized_target))
    clamped = min(1.0, max(0.0, abs(dot)))
    return 2.0 * math.acos(clamped)


def send_joint_target(targets: list[float]) -> str:
    request = (
        'data: "model_name=ur;'
        f"joint_names={','.join(JOINT_NAMES)};"
        f"positions={','.join(str(value) for value in targets)}\""
    )
    return run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{WORLD}/joint_target",
            "--reqtype",
            "gz.msgs.StringMsg",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "10000",
            "--req",
            request,
        ]
    )


def wait_for_joint_targets(targets: list[float]) -> dict[str, float]:
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while time.monotonic() < deadline:
        joints = decode_joint_positions()
        max_error = max(
            abs(joints[name] - target)
            for name, target in zip(JOINT_NAMES, targets)
        )
        if max_error <= SETTLE_TOL:
            return joints
        time.sleep(0.1)
    return decode_joint_positions()


def main() -> None:
    pre_joints = decode_joint_positions()
    targets = [pre_joints[name] + delta for name, delta in zip(JOINT_NAMES, JOINT_DELTA)]
    start = time.perf_counter()
    service_reply = send_joint_target(targets)
    post_joints = wait_for_joint_targets(targets)
    elapsed = time.perf_counter() - start
    source = decode_source_pose()
    target = {
        "position": list(TABLETOP_POSITION),
        "orientation": quaternion_from_rpy(0.0, 0.0, TABLETOP_YAW),
    }
    relative_position = [
        target_axis - source_axis
        for source_axis, target_axis in zip(source["position"], target["position"])
    ]
    distance = math.sqrt(sum(axis * axis for axis in relative_position))
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "joint_positions": [post_joints[name] for name in JOINT_NAMES],
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
                "service_reply": service_reply.strip(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
