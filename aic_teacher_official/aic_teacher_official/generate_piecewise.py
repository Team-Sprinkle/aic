"""Generate first-pass official teacher piecewise trajectory artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from aic_teacher_official.context import OfficialTeacherContext
from aic_teacher_official.trajectory import (
    PhaseLabel,
    PiecewiseTrajectory,
    SourceLabel,
    TCPPose,
    TrajectoryMetadata,
    TrajectoryWaypoint,
)


@dataclass(frozen=True)
class PiecewiseGeneratorConfig:
    start_position: list[float]
    port_position: list[float]
    orientation_xyzw: list[float]
    approach_offset: list[float]
    alignment_height: float = 0.16
    pre_insertion_height: float = 0.03
    insertion_depth: float = -0.015
    approach_duration: float = 2.0
    alignment_duration: float = 2.0
    pre_insertion_duration: float = 1.0
    insertion_duration: float = 12.0
    task_name: str = "Insert cable into target port"
    context: OfficialTeacherContext | None = None
    vlm_delta_plan: dict[str, Any] | None = None
    planner_feedback: dict[str, Any] | None = None


def _pose(position: list[float], orientation_xyzw: list[float]) -> TCPPose:
    return TCPPose(
        position=[float(v) for v in position],
        orientation_xyzw=[float(v) for v in orientation_xyzw],
    )


def _add(position: list[float], offset: list[float]) -> list[float]:
    return (np.asarray(position, dtype=np.float64) + np.asarray(offset, dtype=np.float64)).tolist()


def _xyzw_to_wxyz(quaternion_xyzw: list[float]) -> tuple[float, float, float, float]:
    return (
        float(quaternion_xyzw[3]),
        float(quaternion_xyzw[0]),
        float(quaternion_xyzw[1]),
        float(quaternion_xyzw[2]),
    )


def _wxyz_to_xyzw(quaternion_wxyz: tuple[float, float, float, float]) -> list[float]:
    return [
        float(quaternion_wxyz[1]),
        float(quaternion_wxyz[2]),
        float(quaternion_wxyz[3]),
        float(quaternion_wxyz[0]),
    ]


def _cheatcode_gripper_orientation(
    *,
    gripper_orientation_xyzw: list[float],
    port_orientation_xyzw: list[float] | None,
    plug_orientation_xyzw: list[float] | None,
) -> list[float]:
    if port_orientation_xyzw is None or plug_orientation_xyzw is None:
        return [float(v) for v in gripper_orientation_xyzw]

    from transforms3d._gohlketransforms import quaternion_multiply

    q_port = _xyzw_to_wxyz(port_orientation_xyzw)
    q_plug = _xyzw_to_wxyz(plug_orientation_xyzw)
    # Mirrors aic_example_policies/.../CheatCode.py. That policy has been the
    # most reliable geometric insertion reference in the official environment.
    q_plug_inv = (-q_plug[0], q_plug[1], q_plug[2], q_plug[3])
    q_gripper = _xyzw_to_wxyz(gripper_orientation_xyzw)
    return _wxyz_to_xyzw(quaternion_multiply(quaternion_multiply(q_port, q_plug_inv), q_gripper))


def _cheatcode_gripper_position(
    *,
    port: list[float],
    gripper_start: list[float],
    plug_position: list[float] | None,
    z_offset: float,
) -> list[float]:
    if plug_position is None:
        return [port[0], port[1], port[2] + z_offset]
    plug_tip_gripper_offset = (
        np.asarray(gripper_start, dtype=np.float64) - np.asarray(plug_position, dtype=np.float64)
    )
    return [
        float(port[0]),
        float(port[1]),
        float(port[2] + z_offset - plug_tip_gripper_offset[2]),
    ]


def _phase_from_vlm(value: str) -> PhaseLabel:
    try:
        phase = PhaseLabel(value)
    except ValueError:
        return PhaseLabel.APPROACH
    if phase == PhaseLabel.FINAL_INSERTION:
        return PhaseLabel.PRE_INSERTION
    return phase


def generate_piecewise_trajectory(config: PiecewiseGeneratorConfig) -> PiecewiseTrajectory:
    """Build a deterministic placeholder oracle plan.

    Approach/alignment waypoints can come from a VLM delta plan or a deterministic
    placeholder. Final insertion is deterministic geometry: hold the aligned TCP
    x/y and descend along the port z-axis, mirroring the final CheatCode descent
    in `aic_example_policies/aic_example_policies/ros/CheatCode.py`.
    """
    start_position = (
        config.context.start_position if config.context is not None else config.start_position
    )
    port = [
        float(v)
        for v in (config.context.port_position if config.context is not None else config.port_position)
    ]
    start_orientation = [
        float(v)
        for v in (
            config.context.orientation_xyzw
            if config.context is not None
            else config.orientation_xyzw
        )
    ]
    orientation = _cheatcode_gripper_orientation(
        gripper_orientation_xyzw=start_orientation,
        port_orientation_xyzw=(
            config.context.port_orientation_xyzw if config.context is not None else None
        ),
        plug_orientation_xyzw=(
            config.context.plug_orientation_xyzw if config.context is not None else None
        ),
    )

    approach_stage = _add(port, config.approach_offset)
    alignment = _cheatcode_gripper_position(
        port=port,
        gripper_start=start_position,
        plug_position=config.context.plug_position if config.context is not None else None,
        z_offset=config.alignment_height,
    )
    pre_insertion = _cheatcode_gripper_position(
        port=port,
        gripper_start=start_position,
        plug_position=config.context.plug_position if config.context is not None else None,
        z_offset=config.pre_insertion_height,
    )
    inserted = _cheatcode_gripper_position(
        port=port,
        gripper_start=start_position,
        plug_position=config.context.plug_position if config.context is not None else None,
        z_offset=config.insertion_depth,
    )

    t0 = 0.0
    t1 = t0 + config.approach_duration
    t2 = t1 + config.alignment_duration
    t3 = t2 + config.pre_insertion_duration
    t4 = t3 + config.insertion_duration

    metadata = TrajectoryMetadata(
        task={"name": config.task_name},
        planning={
            "method": (
                "gpt5_mini_delta_plan_plus_cheatcode_geometry_v0"
                if config.vlm_delta_plan
                else "placeholder_vlm_optimizer_plus_cheatcode_geometry_v0"
            ),
            "vlm_pause_allowed": True,
            "context": config.context.to_dict() if config.context is not None else None,
            "vlm_delta_plan": config.vlm_delta_plan,
            "planner_feedback": config.planner_feedback,
            "automatic_context_extraction": (
                "official_ros_tf" if config.context is not None else "explicit_cli_or_todo"
            ),
        },
        diagnostics={
            "cheatcode_reference": "aic_example_policies/aic_example_policies/ros/CheatCode.py",
            "cheatcode_adapter": (
                "uses plug-tip-to-gripper z offset plus port/plug/gripper quaternion correction "
                "when official TF context includes plug and port orientation"
            ),
            "action_mode_assumption": "VLM plans deltas; replay defaults to relative TCP deltas",
        },
    )
    waypoints: list[TrajectoryWaypoint] = [
        TrajectoryWaypoint(
            timestamp=t0,
            tcp_pose=_pose(start_position, start_orientation),
            phase=PhaseLabel.APPROACH,
            source=SourceLabel.VLM if config.vlm_delta_plan else SourceLabel.PLACEHOLDER_VLM,
            diagnostics={
                "oracle_role": "start",
                "context_source": "official_ros_tf" if config.context is not None else "explicit_cli",
            },
        )
    ]
    if config.vlm_delta_plan:
        from aic_teacher_official.vlm_planner import sanitize_delta_plan

        current = np.asarray(start_position, dtype=np.float64)
        timestamp = t0
        for index, vlm_waypoint in enumerate(sanitize_delta_plan(config.vlm_delta_plan)):
            current = current + np.asarray(vlm_waypoint["delta_xyz"], dtype=np.float64)
            timestamp += float(vlm_waypoint["duration"])
            waypoints.append(
                TrajectoryWaypoint(
                    timestamp=timestamp,
                    tcp_pose=_pose(current.tolist(), orientation),
                    phase=_phase_from_vlm(vlm_waypoint["phase"]),
                    source=SourceLabel.VLM,
                    diagnostics={
                        "oracle_role": "vlm_delta_waypoint",
                        "delta_xyz": vlm_waypoint["delta_xyz"],
                        "rationale": vlm_waypoint["rationale"],
                        "vlm_used": True,
                    },
                )
            )
        # Optimizer placeholder snaps the VLM plan onto a conservative staging
        # line above the target port before geometric insertion.
        t3 = timestamp + config.pre_insertion_duration
    else:
        waypoints.extend(
            [
                TrajectoryWaypoint(
                    timestamp=t1,
                    tcp_pose=_pose(approach_stage, orientation),
                    phase=PhaseLabel.APPROACH,
                    source=SourceLabel.PLACEHOLDER_VLM,
                    diagnostics={
                        "oracle_role": "coarse approach / obstacle avoidance",
                        "vlm_pause_allowed": True,
                    },
                ),
                TrajectoryWaypoint(
                    timestamp=t2,
                    tcp_pose=_pose(alignment, orientation),
                    phase=PhaseLabel.ALIGNMENT,
                    source=SourceLabel.PLACEHOLDER_OPTIMIZER,
                    diagnostics={
                        "oracle_role": "optimizer alignment staging",
                        "todo": "Replace with constrained trajectory optimizer output.",
                    },
                ),
            ]
        )
    waypoints.extend(
        [
            TrajectoryWaypoint(
                timestamp=t3,
                tcp_pose=_pose(pre_insertion, orientation),
                phase=PhaseLabel.PRE_INSERTION,
                source=SourceLabel.OPTIMIZER if config.vlm_delta_plan else SourceLabel.PLACEHOLDER_OPTIMIZER,
                diagnostics={
                    "oracle_role": "pre-insertion staging",
                    "geometry": "CheatCode-style target TCP pose from port pose and plug-tip offset",
                    "optimizer_role": "snap approach plan to deterministic insertion staging pose",
                },
            ),
            TrajectoryWaypoint(
                timestamp=t3 + config.insertion_duration,
                tcp_pose=_pose(inserted, orientation),
                phase=PhaseLabel.FINAL_INSERTION,
                source=SourceLabel.CHEATCODE,
                diagnostics={
                    "oracle_role": "final insertion",
                    "cheatcode_derived": True,
                    "generation": (
                        "deterministic CheatCode-style descent using port frame, "
                        "plug-tip/gripper offset, slow final timing, and no VLM calls"
                    ),
                    "vlm_used": False,
                },
            ),
        ]
    )
    return PiecewiseTrajectory(waypoints=waypoints, metadata=metadata)


def generate_piecewise_file(config: PiecewiseGeneratorConfig, output_path: str | Path) -> PiecewiseTrajectory:
    trajectory = generate_piecewise_trajectory(config)
    trajectory.save_json(output_path)
    return trajectory
