"""MoveItPy backend for free-space approach validation/planning."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import yaml

from aic_teacher_official.expert_generator.candidate_generation import ApproachCandidate
from aic_teacher_official.expert_generator.moveit_planner import (
    PlannedApproach,
    PlanningFailure,
)
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot
from aic_teacher_official.expert_generator.vlm_strategy import VLMStrategy
from aic_teacher_official.trajectory import PhaseLabel, SourceLabel, TCPPose, TrajectoryWaypoint


@dataclass(frozen=True)
class MoveItPyBackendConfig:
    node_name: str = "aic_expert_moveit_py"
    planning_group: str = "ur_manipulator"
    end_effector_link: str = "gripper/tcp"
    workspace_dir: str | None = None
    ur_type: str = "ur5e"
    description_file: str | None = None
    controllers_file: str | None = None
    planning_time: float = 5.0
    planning_attempts: int = 5
    max_velocity_scaling_factor: float = 0.1
    max_acceleration_scaling_factor: float = 0.1
    approach_segment_mode: str = "direct_pre_insert"


class MoveItPyPlanningBackend:
    """Adapter around `moveit.planning`.

    The backend requires a live MoveItPy configuration/parameter environment. If
    MoveItPy cannot initialize or cannot produce a plan, the candidate fails.
    """

    def __init__(self, config: MoveItPyBackendConfig | None = None, *, moveit_py: Any | None = None):
        self.config = config or MoveItPyBackendConfig()
        self._moveit_py = moveit_py

    def plan_free_space_approach(
        self,
        scene_snapshot: SceneSnapshot,
        strategy: VLMStrategy,
        candidate_pose: ApproachCandidate,
    ) -> PlannedApproach | PlanningFailure:
        try:
            moveit_py = self._moveit_py or self._create_moveit_py()
            planning_component = moveit_py.get_planning_component(self.config.planning_group)
            planning_component.set_start_state_to_current_state()
            all_waypoints: list[TrajectoryWaypoint] = []
            segment_summaries: list[dict[str, Any]] = []
            cumulative_time = 0.0
            last_robot_state = None
            segment_specs = self._segment_specs(candidate_pose)
            for segment_index, (segment_name, pose, phase) in enumerate(segment_specs):
                if segment_index > 0 and last_robot_state is not None:
                    _set_planning_start_state(planning_component, last_robot_state)
                pose_stamped = _pose_stamped_from_serializable(pose)
                planning_component.set_goal_state(
                    pose_stamped_msg=pose_stamped,
                    pose_link=self.config.end_effector_link,
                )
                plan_params = self._plan_request_parameters(moveit_py)
                plan_result = (
                    planning_component.plan(plan_params)
                    if plan_params is not None
                    else planning_component.plan()
                )
                if not _plan_result_success(plan_result):
                    return PlanningFailure(
                        reason="moveit_plan_failed",
                        recoverable=True,
                        details={
                            "candidate": candidate_pose.name,
                            "segment": segment_name,
                            "target_pose": pose.to_dict(),
                            "plan_result": _plan_result_summary(plan_result),
                        },
                    )
                segment_waypoints, last_robot_state, segment_summary = _waypoints_from_plan_result(
                    plan_result,
                    segment_name=segment_name,
                    segment_index=segment_index,
                    phase=phase,
                    candidate_name=candidate_pose.name,
                    end_effector_link=self.config.end_effector_link,
                    planning_group=self.config.planning_group,
                    cumulative_time=cumulative_time,
                    previous_timestamp=all_waypoints[-1].timestamp if all_waypoints else None,
                )
                if not segment_waypoints:
                    return PlanningFailure(
                        reason="moveit_plan_had_no_replayable_waypoints",
                        recoverable=True,
                        details={
                            "candidate": candidate_pose.name,
                            "segment": segment_name,
                            "plan_result": _plan_result_summary(plan_result),
                        },
                    )
                all_waypoints.extend(segment_waypoints)
                cumulative_time = all_waypoints[-1].timestamp
                segment_summaries.append(segment_summary)
            return PlannedApproach(
                candidate_name=candidate_pose.name,
                waypoints=all_waypoints,
                planning_scene_objects=[],
                metadata={
                    "backend": "moveit_py",
                    "planning_group": self.config.planning_group,
                    "end_effector_link": self.config.end_effector_link,
                    "candidate": candidate_pose.name,
                    "replay_space": "joint_position",
                    "segments": segment_summaries,
                },
            )
        except Exception as ex:
            return PlanningFailure(
                reason="moveit_py_exception",
                recoverable=False,
                details={"type": type(ex).__name__, "message": str(ex)},
            )

    def _segment_specs(
        self,
        candidate_pose: ApproachCandidate,
    ) -> tuple[tuple[str, Any, PhaseLabel], ...]:
        if self.config.approach_segment_mode == "three_stage":
            return (
                ("safe_lift", candidate_pose.safe_lift_pose, PhaseLabel.APPROACH),
                ("approach_standoff", candidate_pose.approach_standoff_pose, PhaseLabel.ALIGNMENT),
                ("pre_insert", candidate_pose.pre_insert_pose, PhaseLabel.PRE_INSERTION),
            )
        if self.config.approach_segment_mode != "direct_pre_insert":
            raise ValueError(
                "approach_segment_mode must be 'direct_pre_insert' or 'three_stage' "
                f"(got {self.config.approach_segment_mode!r})"
            )
        return (("pre_insert", candidate_pose.pre_insert_pose, PhaseLabel.PRE_INSERTION),)

    def _create_moveit_py(self) -> Any:
        from moveit.planning import MoveItPy

        return MoveItPy(node_name=self.config.node_name, config_dict=self._moveit_config_dict())

    def _moveit_config_dict(self) -> dict[str, Any]:
        workspace_dir = _workspace_dir(self.config.workspace_dir)
        description_file = Path(
            self.config.description_file
            or os.environ.get("AIC_EXPERT_MOVEIT_DESCRIPTION_FILE", "")
            or workspace_dir / "aic_description" / "urdf" / "ur_gz.urdf.xacro"
        )
        controllers_file = Path(
            self.config.controllers_file
            or os.environ.get("AIC_EXPERT_MOVEIT_CONTROLLERS_FILE", "")
            or workspace_dir / "aic_bringup" / "config" / "aic_ros2_controllers.yaml"
        )
        config_dir = workspace_dir / "aic_moveit_config" / "config"
        robot_description = _xacro_robot_description(
            description_file=description_file,
            controllers_file=controllers_file,
            ur_type=self.config.ur_type,
        )
        ompl_config = _load_yaml(config_dir / "ompl_planning.yaml")
        moveit_cpp_config = _load_yaml(config_dir / "moveit_cpp.yaml")
        plan_request_params = dict(moveit_cpp_config.get("plan_request_params", {}))
        ompl_config["plan_request_params"] = plan_request_params
        params: dict[str, Any] = {
            "robot_description": robot_description,
            "robot_description_semantic": (config_dir / "aic.srdf").read_text(encoding="utf-8"),
            "robot_description_kinematics": _load_yaml(config_dir / "kinematics.yaml"),
            "robot_description_planning": _load_yaml(config_dir / "joint_limits.yaml"),
            "ompl": ompl_config,
            "use_sim_time": True,
        }
        params.update(_load_yaml(config_dir / "moveit_controllers.yaml"))
        params.update(moveit_cpp_config)
        return params

    def _plan_request_parameters(self, moveit_py: Any) -> Any:
        from moveit.planning import PlanRequestParameters

        try:
            params = PlanRequestParameters(moveit_py, "ompl")
        except TypeError:
            return None
        params.planning_time = self.config.planning_time
        params.planning_attempts = self.config.planning_attempts
        params.max_velocity_scaling_factor = self.config.max_velocity_scaling_factor
        params.max_acceleration_scaling_factor = self.config.max_acceleration_scaling_factor
        return params


def _pose_stamped_from_serializable(pose: Any) -> Any:
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.frame_id = pose.frame_id
    msg.pose.position.x = float(pose.position[0])
    msg.pose.position.y = float(pose.position[1])
    msg.pose.position.z = float(pose.position[2])
    msg.pose.orientation.x = float(pose.orientation_xyzw[0])
    msg.pose.orientation.y = float(pose.orientation_xyzw[1])
    msg.pose.orientation.z = float(pose.orientation_xyzw[2])
    msg.pose.orientation.w = float(pose.orientation_xyzw[3])
    return msg


def _plan_result_success(plan_result: Any) -> bool:
    if plan_result is None:
        return False
    success = getattr(plan_result, "success", None)
    if success is not None:
        return bool(success)
    trajectory = getattr(plan_result, "trajectory", None) or getattr(plan_result, "planned_trajectory", None)
    return trajectory is not None


def _plan_result_summary(plan_result: Any) -> dict[str, Any]:
    if plan_result is None:
        return {"available": False}
    summary = {"type": type(plan_result).__name__}
    for attr in ("success", "planning_time", "error_code"):
        if hasattr(plan_result, attr):
            summary[attr] = str(getattr(plan_result, attr))
    trajectory = getattr(plan_result, "trajectory", None) or getattr(plan_result, "planned_trajectory", None)
    if trajectory is not None:
        trajectory_msg = _robot_trajectory_msg(trajectory)
        joint_traj = getattr(trajectory_msg, "joint_trajectory", None)
        points = getattr(joint_traj, "points", []) if joint_traj is not None else []
        summary["joint_trajectory_points"] = len(points)
    return summary


def _waypoints_from_plan_result(
    plan_result: Any,
    *,
    segment_name: str,
    segment_index: int,
    phase: PhaseLabel,
    candidate_name: str,
    end_effector_link: str,
    planning_group: str,
    cumulative_time: float,
    previous_timestamp: float | None,
) -> tuple[list[TrajectoryWaypoint], Any | None, dict[str, Any]]:
    trajectory = getattr(plan_result, "trajectory", None) or getattr(plan_result, "planned_trajectory", None)
    trajectory_msg = _robot_trajectory_msg(trajectory)
    joint_trajectory = getattr(trajectory_msg, "joint_trajectory", None)
    points = list(getattr(joint_trajectory, "points", []) or [])
    joint_names = [str(v) for v in getattr(joint_trajectory, "joint_names", [])]
    if not points or not joint_names:
        return [], None, {
            "segment": segment_name,
            "joint_names": joint_names,
            "joint_trajectory_points": len(points),
        }

    robot_model = _trajectory_robot_model(trajectory)
    waypoints: list[TrajectoryWaypoint] = []
    last_robot_state = None
    for point_index, point in enumerate(points):
        timestamp = cumulative_time + _duration_seconds(getattr(point, "time_from_start", None))
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            continue
        joint_positions = [float(v) for v in getattr(point, "positions", [])]
        joint_velocities = [float(v) for v in getattr(point, "velocities", [])]
        if len(joint_positions) != len(joint_names):
            continue
        tcp_pose, last_robot_state = _tcp_pose_from_joint_point(
            robot_model,
            planning_group=planning_group,
            end_effector_link=end_effector_link,
            joint_positions=joint_positions,
        )
        waypoints.append(
            TrajectoryWaypoint(
                timestamp=float(timestamp),
                tcp_pose=tcp_pose,
                phase=phase,
                source=SourceLabel.OPTIMIZER,
                joint_names=joint_names,
                joint_positions=joint_positions,
                joint_velocities=(
                    joint_velocities
                    if len(joint_velocities) == len(joint_positions)
                    else [0.0] * len(joint_positions)
                ),
                diagnostics={
                    "planner": "moveit_py",
                    "candidate": candidate_name,
                    "segment": segment_name,
                    "segment_index": segment_index,
                    "point_index": point_index,
                    "replay_space": "joint_position",
                },
            )
        )
    return waypoints, last_robot_state, {
        "segment": segment_name,
        "joint_names": joint_names,
        "joint_trajectory_points": len(points),
        "replay_waypoints": len(waypoints),
    }


def _robot_trajectory_msg(trajectory: Any) -> Any:
    if trajectory is None:
        return None
    if hasattr(trajectory, "get_robot_trajectory_msg"):
        return trajectory.get_robot_trajectory_msg()
    return trajectory


def _trajectory_robot_model(trajectory: Any) -> Any:
    robot_model = getattr(trajectory, "robot_model", None)
    return robot_model() if callable(robot_model) else robot_model


def _tcp_pose_from_joint_point(
    robot_model: Any,
    *,
    planning_group: str,
    end_effector_link: str,
    joint_positions: list[float],
) -> tuple[TCPPose, Any]:
    if robot_model is None:
        raise RuntimeError("MoveIt trajectory does not expose robot_model; cannot compute TCP replay pose.")
    from moveit.core.robot_state import RobotState

    state = RobotState(robot_model)
    positions = np.asarray(joint_positions, dtype=np.float64)
    try:
        state.set_joint_group_active_positions(planning_group, positions)
    except Exception:
        state.set_joint_group_positions(planning_group, positions)
    try:
        state.update(True, "all")
    except TypeError:
        state.update()
    pose = state.get_pose(end_effector_link)
    return (
        TCPPose(
            position=[
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            orientation_xyzw=[
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        ),
        state,
    )


def _duration_seconds(duration: Any) -> float:
    if duration is None:
        return 0.0
    return float(getattr(duration, "sec", 0.0)) + 1e-9 * float(getattr(duration, "nanosec", 0.0))


def _set_planning_start_state(planning_component: Any, robot_state: Any) -> None:
    try:
        planning_component.set_start_state(robot_state=robot_state)
    except Exception:
        planning_component.set_start_state(robot_state)


def _workspace_dir(configured: str | None) -> Path:
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("AIC_WORKSPACE_DIR"):
        return Path(os.environ["AIC_WORKSPACE_DIR"]).expanduser().resolve()
    return Path.cwd().resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _xacro_robot_description(*, description_file: Path, controllers_file: Path, ur_type: str) -> str:
    if not description_file.exists():
        raise FileNotFoundError(f"MoveIt description xacro not found: {description_file}")
    if not controllers_file.exists():
        raise FileNotFoundError(f"MoveIt controller config not found: {controllers_file}")
    result = subprocess.run(
        [
            "xacro",
            str(description_file),
            "name:=ur",
            f"ur_type:={ur_type}",
            f"simulation_controllers:={controllers_file}",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout
