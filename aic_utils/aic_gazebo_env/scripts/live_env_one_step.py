#!/usr/bin/env python3
"""Run one live GazeboCliClient step from reset and print a comparable trace."""

from __future__ import annotations

import faulthandler
import json
import sys
import time

from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig
from aic_gazebo_env.protocol import StepRequest

HOME_JOINTS = [-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]
JOINT_DELTA = [0.02, -0.01, 0.015, 0.0, 0.0, 0.0]
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


def main() -> None:
    faulthandler.enable(file=sys.stderr)
    faulthandler.dump_traceback_later(30, repeat=False, file=sys.stderr)
    client = GazeboCliClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path="/tmp/aic.sdf",
            timeout=30.0,
            world_name="aic_world",
            source_entity_name="ati/tool_link",
            target_entity_name="tabletop",
            joint_command_model_name="ur",
        )
    )
    start = time.perf_counter()
    response = client.step(
        StepRequest(
            action={
                "set_joint_positions": {
                    "model_name": "ur5e",
                    "joint_names": JOINT_NAMES,
                    "positions": [
                        home + delta for home, delta in zip(HOME_JOINTS, JOINT_DELTA)
                    ],
                },
                "multi_step": 1,
            }
        )
    )
    elapsed = time.perf_counter() - start
    faulthandler.cancel_dump_traceback_later()
    tracked = response.observation["task_geometry"]["tracked_entity_pair"]
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "joint_positions": response.observation["joint_positions"],
                "tcp_position": response.observation["entities_by_name"]["ati/tool_link"][
                    "position"
                ],
                "relative_position": tracked["relative_position"],
                "distance": tracked["distance"],
                "orientation_error": tracked["orientation_error"],
                "reward": response.reward,
                "terminated": response.terminated,
                "truncated": response.truncated,
                "joint_target_service": response.info["joint_target_service"],
                "pose_service": response.info["pose_service"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
