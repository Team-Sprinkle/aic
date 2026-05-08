"""Candidate staging pose generation for MoveIt free-space planning."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable

from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot, SerializablePose
from aic_teacher_official.expert_generator.vlm_strategy import VLMStrategy
from transforms3d._gohlketransforms import quaternion_multiply


@dataclass(frozen=True)
class RouteSubgoal:
    name: str
    pose: SerializablePose
    phase: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "pose": self.pose.to_dict(),
            "phase": self.phase,
        }


@dataclass(frozen=True)
class ApproachCandidate:
    index: int
    name: str
    safe_lift_pose: SerializablePose
    approach_standoff_pose: SerializablePose
    pre_insert_pose: SerializablePose
    metadata: dict[str, object]
    route_subgoals: tuple[RouteSubgoal, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "name": self.name,
            "safe_lift_pose": self.safe_lift_pose.to_dict(),
            "approach_standoff_pose": self.approach_standoff_pose.to_dict(),
            "pre_insert_pose": self.pre_insert_pose.to_dict(),
            "metadata": dict(self.metadata),
            "route_subgoals": [subgoal.to_dict() for subgoal in self.route_subgoals],
        }


DEFAULT_CANDIDATE_ORDER = [
    "above",
    "back",
    "above_left",
    "above_right",
    "high_clearance_vertical",
    "front",
]
SIDE_OFFSETS = {
    "above": (0.0, 0.0),
    "above_left": (-0.04, 0.04),
    "above_right": (0.04, 0.04),
    "high_clearance_vertical": (0.0, 0.0),
    "front": (0.0, -0.05),
    "back": (0.0, 0.05),
}
DIAGONAL_BYPASS_PROGRESS_FRACTION = 0.20
SC_NIC_BYPASS_LEFT_OFFSET_M = 0.08
SC_NIC_APPROACH_LEFT_OFFSET_M = 0.05
SC_NIC_BYPASS_Y_MARGIN_M = 0.14
SC_NIC_TARGET_Y_OVERSHOOT_M = 0.04


def _unique_candidate_names(preferred: str, count: int) -> list[str]:
    ordered = [preferred] + [name for name in DEFAULT_CANDIDATE_ORDER if name != preferred]
    return ordered[: max(0, count)]


def _ordered_candidate_names(snapshot: SceneSnapshot, strategy: VLMStrategy, count: int) -> list[str]:
    if _sc_to_sc_with_nic_obstacles(snapshot):
        preferred = strategy.approach_side
        if preferred in {"front", "above"}:
            preferred = "above_left"
        obstacle_safe_order = [
            preferred,
            "above_left",
            "back",
            "high_clearance_vertical",
            "above_right",
            "above",
        ]
        ordered: list[str] = []
        for name in obstacle_safe_order:
            if name in SIDE_OFFSETS and name not in ordered:
                ordered.append(name)
        return ordered[: max(0, count)]
    return _unique_candidate_names(strategy.approach_side, count)


def _pose_at(base: SerializablePose, *, x: float, y: float, z: float, orientation_xyzw: Iterable[float]) -> SerializablePose:
    return SerializablePose(
        position=[float(x), float(y), float(z)],
        orientation_xyzw=[float(v) for v in orientation_xyzw],
        frame_id=base.frame_id,
    )


def generate_approach_candidates(
    snapshot: SceneSnapshot,
    strategy: VLMStrategy,
    *,
    count: int = 5,
    pre_insert_z_offset_m: float = 0.20,
) -> list[ApproachCandidate]:
    """Generate symbolic candidate poses.

    Position candidates are deterministic and conservative. Orientation comes
    from the target-port pose, which mirrors the CheatCode-style "align plug to
    port geometry" contract without asking the VLM for orientation.
    """

    if snapshot.target_port_pose is None:
        raise ValueError("target_port_pose is required to generate approach candidates")
    if snapshot.tcp_pose is None:
        raise ValueError("tcp_pose is required to generate approach candidates")
    target = snapshot.target_port_pose
    current = snapshot.tcp_pose
    near_start = _distance(current.position, target.position) <= 0.08
    effective_z_offset = _effective_pre_insert_z_offset(
        snapshot,
        default_z_offset=pre_insert_z_offset_m,
        near_start=near_start,
    )
    cheatcode_pre_insert = _cheatcode_gripper_target(snapshot, z_offset=effective_z_offset)
    orientation = list(cheatcode_pre_insert.orientation_xyzw)
    clearance = float(strategy.preferred_clearance_m)
    nic_obstacles = _nic_obstacles(snapshot)
    sc_nic_obstacle_route = _sc_to_sc_with_nic_obstacles(snapshot)
    sc_bypass_left_offset = _env_float("AIC_EXPERT_SC_NIC_BYPASS_LEFT_OFFSET_M", SC_NIC_BYPASS_LEFT_OFFSET_M)
    sc_approach_left_offset = _env_float(
        "AIC_EXPERT_SC_NIC_APPROACH_LEFT_OFFSET_M",
        min(SC_NIC_APPROACH_LEFT_OFFSET_M, sc_bypass_left_offset),
    )
    names = _ordered_candidate_names(snapshot, strategy, count)
    candidates: list[ApproachCandidate] = []
    for idx, name in enumerate(names):
        dx, dy = SIDE_OFFSETS.get(name, (0.0, 0.0))
        extra_clearance = 0.10 if sc_nic_obstacle_route else 0.06 if name == "high_clearance_vertical" else 0.0
        route_around_nics = sc_nic_obstacle_route and not near_start
        if near_start:
            dx = 0.0
            dy = 0.0
            extra_clearance = 0.0
        staging_z = cheatcode_pre_insert.position[2]
        lift_z = max(current.position[2], staging_z + extra_clearance)
        diagonal_bypass_progress_fraction = 0.0
        if route_around_nics:
            lift_z = max(lift_z, staging_z + clearance + extra_clearance)
            bypass_y = _nic_bypass_y(nic_obstacles, target.position[1])
            wide_left_x = float(current.position[0]) - sc_bypass_left_offset
            if name == "high_clearance_vertical":
                dx = max(-0.5 * sc_bypass_left_offset, wide_left_x - cheatcode_pre_insert.position[0])
                dy = bypass_y - cheatcode_pre_insert.position[1]
            elif name == "back":
                dx = max(-0.75 * sc_approach_left_offset, wide_left_x - cheatcode_pre_insert.position[0])
                dy = bypass_y - cheatcode_pre_insert.position[1]
            elif name == "above_left":
                dx = max(-sc_approach_left_offset, wide_left_x - cheatcode_pre_insert.position[0])
                dy = bypass_y - cheatcode_pre_insert.position[1]
            elif name == "above_right":
                dx = 0.05
                dy = bypass_y - cheatcode_pre_insert.position[1]
        safe_lift_x = wide_left_x if route_around_nics else current.position[0]
        approach_z = staging_z + (min(clearance, 0.02) if near_start else clearance) + extra_clearance
        safe_lift_z = lift_z
        if route_around_nics:
            safe_lift_z = max(safe_lift_z, approach_z + _env_float("AIC_EXPERT_SC_NIC_LEFT_CLEARANCE_EXTRA_Z_M", 0.04))
        safe_lift = _pose_at(
            cheatcode_pre_insert,
            x=safe_lift_x,
            y=current.position[1],
            z=safe_lift_z,
            orientation_xyzw=orientation,
        )
        approach = _pose_at(
            cheatcode_pre_insert,
            x=cheatcode_pre_insert.position[0] + dx,
            y=cheatcode_pre_insert.position[1] if route_around_nics else cheatcode_pre_insert.position[1] + dy,
            z=approach_z,
            orientation_xyzw=orientation,
        )
        pre_insert = _pose_at(
            cheatcode_pre_insert,
            x=cheatcode_pre_insert.position[0],
            y=cheatcode_pre_insert.position[1],
            z=staging_z,
            orientation_xyzw=orientation,
        )
        route_subgoals: tuple[RouteSubgoal, ...] = ()
        if route_around_nics:
            left_descent = _pose_at(
                cheatcode_pre_insert,
                x=wide_left_x,
                y=current.position[1],
                z=approach_z,
                orientation_xyzw=orientation,
            )
            outside_lane_forward = _pose_at(
                cheatcode_pre_insert,
                x=wide_left_x,
                y=bypass_y,
                z=approach_z,
                orientation_xyzw=orientation,
            )
            right_sweep = _pose_at(
                cheatcode_pre_insert,
                x=cheatcode_pre_insert.position[0] + dx,
                y=bypass_y,
                z=approach_z,
                orientation_xyzw=orientation,
            )
            route_subgoals = (
                RouteSubgoal("camera_left_clearance", safe_lift, "approach"),
                RouteSubgoal("left_lane_descent", left_descent, "approach"),
                RouteSubgoal("outside_lane_forward_past_cards", outside_lane_forward, "obstacle_avoidance"),
                RouteSubgoal("right_sweep_toward_port", right_sweep, "alignment"),
                RouteSubgoal("port_standoff", approach, "alignment"),
                RouteSubgoal("pre_insert", pre_insert, "pre_insertion"),
            )
        candidates.append(
            ApproachCandidate(
                index=idx,
                name=name,
                safe_lift_pose=safe_lift,
                approach_standoff_pose=approach,
                pre_insert_pose=pre_insert,
                metadata={
                    "orientation_source": "cheatcode_style_target_port_geometry",
                    "vlm_preferred": name == strategy.approach_side,
                    "pre_insert_z_offset_m": effective_z_offset,
                    "requested_pre_insert_z_offset_m": pre_insert_z_offset_m,
                    "pre_insert_pose_source": "cheatcode_calc_gripper_pose",
                    "sc_nic_obstacle_route": sc_nic_obstacle_route,
                    "near_start_short_approach": near_start,
                    "nic_obstacle_names": [obj.name for obj in nic_obstacles],
                    "diagonal_bypass_progress_fraction": diagonal_bypass_progress_fraction,
                    "route_subgoal_names": [subgoal.name for subgoal in route_subgoals],
                    "sc_nic_bypass_left_offset_m": sc_bypass_left_offset if route_around_nics else 0.0,
                    "sc_nic_approach_left_offset_m": sc_approach_left_offset if route_around_nics else 0.0,
                    "sc_nic_bypass_y": bypass_y if route_around_nics else None,
                    "route_policy": (
                        "near_start_short_approach"
                        if near_start
                        else "camera_left_then_outside_lane_moveit_around_present_nic_cards"
                        if sc_nic_obstacle_route
                        else "standard"
                    ),
                },
                route_subgoals=route_subgoals,
            )
        )
    return candidates


def _nic_obstacles(snapshot: SceneSnapshot) -> list[object]:
    return [obj for obj in snapshot.collision_objects if obj.role == "nic_card"]


def _sc_to_sc_with_nic_obstacles(snapshot: SceneSnapshot) -> bool:
    task = snapshot.task_config or {}
    return (
        str(task.get("plug_type", "")).lower() == "sc"
        and str(task.get("port_type", "")).lower() == "sc"
        and bool(_nic_obstacles(snapshot))
    )


def _nic_bypass_y(nic_obstacles: list[object], target_y: float) -> float:
    if not nic_obstacles:
        return target_y + _env_float("AIC_EXPERT_SC_NIC_TARGET_Y_OVERSHOOT_M", SC_NIC_TARGET_Y_OVERSHOOT_M)
    max_y = max(float(obj.pose.position[1]) + 0.5 * float(obj.dimensions[1]) for obj in nic_obstacles)
    margin = _env_float("AIC_EXPERT_SC_NIC_BYPASS_Y_MARGIN_M", SC_NIC_BYPASS_Y_MARGIN_M)
    target_overshoot = _env_float("AIC_EXPERT_SC_NIC_TARGET_Y_OVERSHOOT_M", SC_NIC_TARGET_Y_OVERSHOOT_M)
    return max(float(target_y) + target_overshoot, max_y + margin)


def _distance(a: Iterable[float], b: Iterable[float]) -> float:
    aa = [float(v) for v in a]
    bb = [float(v) for v in b]
    return sum((aa[i] - bb[i]) ** 2 for i in range(3)) ** 0.5


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _effective_pre_insert_z_offset(
    snapshot: SceneSnapshot,
    *,
    default_z_offset: float,
    near_start: bool,
) -> float:
    if not near_start or snapshot.tcp_pose is None or snapshot.target_port_pose is None:
        return default_z_offset
    plug_tip_gripper_offset_z = 0.0
    if snapshot.plug_pose is not None:
        plug_tip_gripper_offset_z = snapshot.tcp_pose.position[2] - snapshot.plug_pose.position[2]
    max_preinsert_gripper_z = snapshot.tcp_pose.position[2] + 0.03
    capped_offset = max_preinsert_gripper_z - snapshot.target_port_pose.position[2] + plug_tip_gripper_offset_z
    return max(0.02, min(default_z_offset, capped_offset))


def _cheatcode_gripper_target(snapshot: SceneSnapshot, *, z_offset: float) -> SerializablePose:
    target = snapshot.target_port_pose
    current = snapshot.tcp_pose
    if target is None or current is None:
        raise ValueError("target_port_pose and tcp_pose are required")
    if snapshot.plug_pose is None:
        return _pose_at(
            target,
            x=target.position[0],
            y=target.position[1],
            z=target.position[2] + z_offset,
            orientation_xyzw=target.orientation_xyzw,
        )

    q_port = _wxyz(target.orientation_xyzw)
    q_plug = _wxyz(snapshot.plug_pose.orientation_xyzw)
    q_gripper = _wxyz(current.orientation_xyzw)
    q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
    q_diff = quaternion_multiply(q_port, q_plug_inv)
    q_gripper_target = quaternion_multiply(q_diff, q_gripper)
    plug_tip_gripper_offset_z = current.position[2] - snapshot.plug_pose.position[2]
    return _pose_at(
        target,
        x=target.position[0],
        y=target.position[1],
        z=target.position[2] + z_offset - plug_tip_gripper_offset_z,
        orientation_xyzw=_xyzw(q_gripper_target),
    )


def _wxyz(q_xyzw: Iterable[float]) -> tuple[float, float, float, float]:
    q = [float(v) for v in q_xyzw]
    return (q[3], q[0], q[1], q[2])


def _xyzw(q_wxyz: Iterable[float]) -> list[float]:
    q = [float(v) for v in q_wxyz]
    return [q[1], q[2], q[3], q[0]]
