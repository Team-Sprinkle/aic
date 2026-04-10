#!/usr/bin/env python3
"""Probe live Gazebo joint-target effects using native topics/services."""

from __future__ import annotations

import ast
import json
import re
import struct
import subprocess
import time

STATE_TOPIC = "/world/aic_world/state"
POSE_TOPIC = "/world/aic_world/pose/info"
SERVICE = "/world/aic_world/joint_target"
JOINT_ENTITY_IDS = [102, 103, 104, 105, 106, 107]
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def run(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def read_state_text(topic: str) -> str:
    return run(["gz", "topic", "-e", "-n", "1", "-t", topic])


def decode_joint_positions() -> dict[str, float | None]:
    out = read_state_text(STATE_TOPIC)
    blocks = re.findall(
        r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
        out,
        re.S,
    )
    decoded: dict[int, float] = {}
    for entity_id_text, body in blocks:
        entity_id = int(entity_id_text)
        if entity_id not in JOINT_ENTITY_IDS:
            continue
        for match in re.finditer(
            r'type: (\d+)\n          component: "(.*?)"',
            body,
            re.S,
        ):
            component_type, payload = match.groups()
            if component_type != "8319580315957903596":
                continue
            raw = ast.literal_eval(f'"{payload}"').encode("latin1")
            if len(raw) >= 10 and raw[0] == 10 and raw[1] == 8:
                decoded[entity_id] = struct.unpack("<d", raw[2:10])[0]
    return {
        joint_name: decoded.get(entity_id)
        for joint_name, entity_id in zip(JOINT_NAMES, JOINT_ENTITY_IDS)
    }


def read_named_pose(name: str) -> dict[str, list[float]]:
    out = read_state_text(POSE_TOPIC)
    block_match = re.search(
        rf'pose \{{\n  name: "{re.escape(name)}"\n(.*?)\n\}}',
        out,
        re.S,
    )
    if block_match is None:
        raise RuntimeError(f"Pose block not found for {name}")
    block = block_match.group(1)
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
        raise RuntimeError(f"Incomplete pose block for {name}")
    return {
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


def send_joint_target() -> dict[str, object]:
    request = (
        'data: "model_name=ur5e;'
        "joint_names=shoulder_pan_joint,shoulder_lift_joint,elbow_joint,"
        'wrist_1_joint,wrist_2_joint,wrist_3_joint;'
        'positions=0.20,-1.67,-1.47,-1.57,1.57,1.41"'
    )
    completed = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            SERVICE,
            "--reqtype",
            "gz.msgs.StringMsg",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "10000",
            "--req",
            request,
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    pre_joints = decode_joint_positions()
    pre_tool = read_named_pose("ati/tool_link")
    pre_wrist = read_named_pose("wrist_3_link")
    start = time.perf_counter()
    reply = send_joint_target()
    time.sleep(0.2)
    elapsed = time.perf_counter() - start
    post_joints = decode_joint_positions()
    post_tool = read_named_pose("ati/tool_link")
    post_wrist = read_named_pose("wrist_3_link")
    print(
        json.dumps(
            {
                "pre_joints": pre_joints,
                "post_joints": post_joints,
                "pre_tool": pre_tool,
                "post_tool": post_tool,
                "pre_wrist": pre_wrist,
                "post_wrist": post_wrist,
                "service_reply": reply,
                "elapsed_s": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
