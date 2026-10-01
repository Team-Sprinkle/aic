#!/usr/bin/env python3
"""CPU-only, measured-start MoveIt smoke test; never executes in Gazebo."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_gazebo_assets
from aic_teacher_official.expert_generator.moveit_py_backend import (
    MoveItPyBackendConfig, MoveItPyPlanningBackend, _apply_verified_scene,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--joint-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = MoveItPyBackendConfig(workspace_dir=str(root), use_sim_time=False)
    backend = MoveItPyPlanningBackend(cfg)
    objects = object_geometries_from_gazebo_assets(args.config, repo_root=root)
    joints = json.loads(args.joint_state.read_text())
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    by_name = dict(zip(joints["name"], joints["position"], strict=True))
    positions = np.asarray([by_name[name] for name in names], dtype=np.float64)
    moveit = MoveItPy(node_name="aic_fixed_scene_probe", config_dict=backend._moveit_config_dict())
    count = _apply_verified_scene(moveit, objects, required=True)
    group = moveit.get_planning_component(cfg.planning_group)
    start = RobotState(moveit.get_robot_model())
    start.set_joint_group_active_positions(cfg.planning_group, positions)
    start.update()
    tcp = start.get_pose(cfg.end_effector_link)
    goal = RobotState(moveit.get_robot_model())
    target = positions.copy()
    target[3] += 0.01
    goal.set_joint_group_active_positions(cfg.planning_group, target)
    goal.update()
    group.set_start_state(robot_state=start)
    group.set_goal_state(robot_state=goal)
    result = group.plan()
    trajectory = getattr(result, "trajectory", None)
    message = trajectory.get_robot_trajectory_msg() if trajectory is not None else None
    points = list(message.joint_trajectory.points) if message is not None else []
    summary = {
        "schema": "aic_moveit_fixed_scene_probe/v1",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "joint_state_sha256": hashlib.sha256(args.joint_state.read_bytes()).hexdigest(),
        "collision_object_count": count,
        "start_joint_names": names,
        "start_joint_positions": positions.tolist(),
        "goal_joint_positions": target.tolist(),
        "moveit_tcp_pose": {
            "position": [tcp.position.x, tcp.position.y, tcp.position.z],
            "orientation_xyzw": [tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w],
        },
        "moveit_success": bool(getattr(result, "success", bool(points))),
        "trajectory_points": len(points),
        "gazebo_executed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
