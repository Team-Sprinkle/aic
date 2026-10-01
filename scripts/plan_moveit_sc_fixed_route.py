#!/usr/bin/env python3
"""Plan a rigid collision-checked SC corridor from the archived measured start."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy

from aic_teacher_official.expert_generator.candidate_generation import ApproachCandidate, RouteSubgoal
from aic_teacher_official.expert_generator.collision_scene import object_geometries_from_gazebo_assets
from aic_teacher_official.expert_generator.moveit_py_backend import MoveItPyBackendConfig, MoveItPyPlanningBackend
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot, SerializablePose
from aic_teacher_official.expert_generator.vlm_strategy import CableRisk, ExpertMode, VLMStrategy
from aic_teacher_official.trajectory import SmoothTrajectory, TrajectoryMetadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--joint-state", type=Path, required=True)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lane-extra-m", type=float, default=0.0)
    parser.add_argument("--behind-extra-m", type=float, default=0.0)
    parser.add_argument("--forward-extra-m", type=float, default=0.0)
    parser.add_argument("--strict-return", action="store_true")
    parser.add_argument("--low-cross-z-m", type=float,
                        help="TCP height in base frame at outside-lane crossing; final standoff unchanged")
    parser.add_argument("--final-standoff-z-m", type=float,
                        help="Override the last TCP height independently of the low crossing")
    parser.add_argument("--diagonal-egress", action="store_true",
                        help="move back, left, and down together on the first corridor segment")
    parser.add_argument("--aligned-orientation-xyzw", type=str,
                        help="optional comma-separated port-aligned TCP quaternion for beyond-row and return goals")
    parser.add_argument("--prealign-from-goal-index", type=int, default=3,
                        help="route-plan waypoint index where port-aligned wrist orientation begins")
    parser.add_argument("--tilt-before-return", action="store_true",
                        help="insert a separate, position-fixed wrist-tilt goal after the beyond-row sweep and before return")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_config = yaml.safe_load(args.config.read_text())
    if len(source_config["trials"]) != 1:
        parser.error("Expected exactly one scene trial")
    trial_id, trial_config = next(iter(source_config["trials"].items()))
    if len(trial_config["tasks"]) != 1:
        parser.error("Expected exactly one insertion task")
    task = next(iter(trial_config["tasks"].values()))
    if task["plug_type"] != "sc" or task["port_type"] != "sc":
        parser.error("This planner currently supports SC-to-SC only")
    cfg = MoveItPyBackendConfig(workspace_dir=str(root), use_sim_time=False, planning_time=10.0)
    backend = MoveItPyPlanningBackend(cfg)
    objects = object_geometries_from_gazebo_assets(args.config, repo_root=root)
    joints = json.loads(args.joint_state.read_text())
    route = json.loads(args.route_plan.read_text())
    by_name = dict(zip(joints["name"], joints["position"], strict=True))
    names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
             "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    positions = np.asarray([by_name[name] for name in names], dtype=np.float64)
    moveit = MoveItPy(node_name="aic_sc_corridor_planner", config_dict=backend._moveit_config_dict())
    start = RobotState(moveit.get_robot_model())
    start.set_joint_group_active_positions(cfg.planning_group, positions)
    start.update()
    tcp = start.get_pose(cfg.end_effector_link)
    orientation = [tcp.orientation.x, tcp.orientation.y, tcp.orientation.z, tcp.orientation.w]
    aligned_orientation = None
    if args.aligned_orientation_xyzw:
        aligned_orientation = [float(item) for item in args.aligned_orientation_xyzw.split(",")]
        if len(aligned_orientation) != 4 or not np.isclose(np.linalg.norm(aligned_orientation), 1.0, atol=1e-3):
            parser.error("--aligned-orientation-xyzw needs a normalized x,y,z,w quaternion")
    if args.tilt_before_return and (aligned_orientation is None or not args.strict_return):
        parser.error("--tilt-before-return requires --aligned-orientation-xyzw and --strict-return")
    chosen = [0, 2, 3, 4, 5] if args.strict_return else [0, 2, 3, 5]
    low_z = args.low_cross_z_m
    if low_z is not None and not 0.12 < low_z < 0.23:
        parser.error("--low-cross-z-m must be between 0.12 and 0.23 m")
    outside_x = route["waypoints"][2]["tcp_base_m"][0] - args.lane_extra_m
    goals = []
    for idx in chosen:
        item = route["waypoints"][idx]
        position = list(item["tcp_base_m"])
        if args.diagonal_egress and idx == 0:
            # Start-to-first-goal motion leaves the row while moving both left
            # and down; the next goal completes the low crossing behind it.
            position[0] = float(tcp.position.x + 0.5 * (outside_x - tcp.position.x))
            position[2] = float(tcp.position.z + 0.5 * ((low_z or route["waypoints"][2]["tcp_base_m"][2])
                                                         - tcp.position.z))
        if idx in (0, 2):
            position[1] -= args.behind_extra_m
        if idx in (2, 3):
            position[0] -= args.lane_extra_m
        if idx in (3, 4):
            position[1] += args.forward_extra_m
        if low_z is not None and idx in (2, 3, 4):
            position[2] = low_z
        if args.final_standoff_z_m is not None and idx == 5:
            position[2] = args.final_standoff_z_m
        goal_orientation = (aligned_orientation if aligned_orientation is not None
                            and idx >= args.prealign_from_goal_index
                            and not (args.tilt_before_return and idx == 4) else orientation)
        goals.append(RouteSubgoal(item["name"], SerializablePose(position, goal_orientation, "base_link"), "obstacle_avoidance"))
        if args.tilt_before_return and idx == 4:
            # MoveIt plans the rotation as its own collision-checked segment.
            # The preceding sweep finishes with the original orientation; only
            # after it reaches the clear beyond-row pose does the wrist tilt.
            goals.append(RouteSubgoal("tilt_clear_of_cards", SerializablePose(
                position.copy(), aligned_orientation, "base_link"), "obstacle_avoidance"))
    candidate = ApproachCandidate(
        index=0, name="outside_left_moveit_corridor",
        safe_lift_pose=goals[0].pose,
        approach_standoff_pose=goals[-1].pose,
        pre_insert_pose=goals[-1].pose,
        metadata={"lane_extra_m": args.lane_extra_m, "behind_extra_m": args.behind_extra_m,
                  "forward_extra_m": args.forward_extra_m, "strict_return": args.strict_return,
                  "low_cross_z_m": low_z, "diagonal_egress": args.diagonal_egress,
                  "final_standoff_z_m": args.final_standoff_z_m,
                  "aligned_orientation_xyzw": aligned_orientation,
                  "prealign_from_goal_index": args.prealign_from_goal_index,
                  "tilt_before_return": args.tilt_before_return},
        route_subgoals=tuple(goals),
    )
    snapshot = SceneSnapshot(
        run_id=str(trial_id), seed=51500,
        scene_id=args.config.parent.name, mode="nominal",
        joint_state=joints, collision_objects=objects,
        task_config={"plug_type": "sc", "port_type": "sc",
                     "target_module_name": task["target_module_name"]},
    )
    strategy = VLMStrategy(
        mode=ExpertMode.NOMINAL, approach_side="above_left", cable_risk=CableRisk.HIGH,
        reason="Fixed measured corridor pilot", mitigation="Gazebo cable validation required",
        preferred_clearance_m=0.02, avoid_regions=[], insertion_strategy="straight_slow_descent",
    )
    backend._moveit_py = moveit
    result = backend.plan_free_space_approach(snapshot, strategy, candidate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "aic_sc_moveit_corridor_plan/v1",
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "joint_state_sha256": hashlib.sha256(args.joint_state.read_bytes()).hexdigest(),
        "route_plan_sha256": hashlib.sha256(args.route_plan.read_bytes()).hexdigest(),
        "candidate": candidate.to_dict(),
        "result": result.to_dict(),
        "gazebo_executed": False,
    }
    if hasattr(result, "waypoints"):
        trajectory = SmoothTrajectory(result.waypoints, TrajectoryMetadata(
            planning={"backend": "moveit_py", "rigid_collision_scene": "gazebo_assets", "cable_validated": False},
        ))
        trajectory.save_json(args.output_dir / "moveit_joint_trajectory.json")
        summary["trajectory_waypoints"] = len(result.waypoints)
    (args.output_dir / "plan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"success": hasattr(result, "waypoints"), "summary": str(args.output_dir / "plan_summary.json"),
                      "waypoints": summary.get("trajectory_waypoints", 0), "reason": getattr(result, "reason", None)}))


if __name__ == "__main__":
    main()
