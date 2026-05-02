"""Nominal expert trajectory assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic_teacher_official.expert_generator.candidate_generation import (
    ApproachCandidate,
    generate_approach_candidates,
)
from aic_teacher_official.expert_generator.moveit_planner import (
    PlannedApproach,
    PlanningFailure,
)
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode, VLMStrategy
from aic_teacher_official.trajectory import (
    PhaseLabel,
    PiecewiseTrajectory,
    SourceLabel,
    TCPPose,
    TrajectoryMetadata,
    TrajectoryWaypoint,
)


@dataclass(frozen=True)
class ExpertTrajectoryResult:
    accepted_for_replay: bool
    trajectory: PiecewiseTrajectory | None
    candidate: ApproachCandidate | None
    planning_result: PlannedApproach | PlanningFailure | None
    metadata: dict[str, Any]


def append_cheatcode_insertion_segment(
    approach: PlannedApproach,
    *,
    insertion_depth_m: float = 0.070,
    insertion_speed_mps: float = 0.0013,
    local_preinsert_align_sec: float = 2.25,
    pre_insert_settle_sec: float = 1.0,
    handoff_blend_sec: float = 2.0,
) -> list[TrajectoryWaypoint]:
    if not approach.waypoints:
        raise ValueError("approach plan must contain at least one waypoint")
    if insertion_speed_mps <= 0.0:
        raise ValueError("insertion_speed_mps must be positive")
    if local_preinsert_align_sec < 0.0 or pre_insert_settle_sec < 0.0 or handoff_blend_sec < 0.0:
        raise ValueError("local alignment, settle, and handoff durations must be non-negative")
    waypoints = list(approach.waypoints)
    pre = waypoints[-1]
    timestamp = pre.timestamp
    if local_preinsert_align_sec > 0.0:
        timestamp += local_preinsert_align_sec
        waypoints.append(
            TrajectoryWaypoint(
                timestamp=timestamp,
                tcp_pose=pre.tcp_pose,
                tcp_velocity=[0.0, 0.0, 0.0],
                phase=PhaseLabel.LOCAL_PREINSERT_ALIGN,
                source=SourceLabel.CHEATCODE,
                joint_names=pre.joint_names,
                joint_positions=pre.joint_positions,
                joint_velocities=[0.0] * len(pre.joint_positions) if pre.joint_positions else pre.joint_velocities,
                diagnostics={
                    "insertion_source": "cheatcode_style_ground_truth_geometry",
                    "command_source": "local_preinsert_align",
                    "local_preinsert_align_sec": local_preinsert_align_sec,
                    "interpolation": "minimum_jerk_translation_plus_quaternion_slerp",
                    "preinsert_pose_pinned": True,
                    "moveit_used": False,
                    "ft_correction_used": False,
                },
            )
        )
        pre = waypoints[-1]
    if handoff_blend_sec > 0.0:
        timestamp += handoff_blend_sec
        waypoints.append(
            TrajectoryWaypoint(
                timestamp=timestamp,
                tcp_pose=pre.tcp_pose,
                tcp_velocity=[0.0, 0.0, 0.0],
                phase=PhaseLabel.PRE_INSERTION,
                source=SourceLabel.CHEATCODE,
                joint_names=pre.joint_names,
                joint_positions=pre.joint_positions,
                joint_velocities=[0.0] * len(pre.joint_positions) if pre.joint_positions else pre.joint_velocities,
                diagnostics={
                    "insertion_source": "cheatcode_style_ground_truth_geometry",
                    "command_source": "blended_handoff",
                    "handoff_blend_sec": handoff_blend_sec,
                    "preinsert_pose_pinned": True,
                    "moveit_used": False,
                    "ft_correction_used": False,
                },
            )
        )
        pre = waypoints[-1]
    if pre_insert_settle_sec > 0.0:
        timestamp += pre_insert_settle_sec
        waypoints.append(
            TrajectoryWaypoint(
                timestamp=timestamp,
                tcp_pose=pre.tcp_pose,
                tcp_velocity=[0.0, 0.0, 0.0],
                phase=PhaseLabel.HOLD,
                source=SourceLabel.CHEATCODE,
                joint_names=pre.joint_names,
                joint_positions=pre.joint_positions,
                joint_velocities=[0.0] * len(pre.joint_positions) if pre.joint_positions else pre.joint_velocities,
                diagnostics={
                    "insertion_source": "cheatcode_style_ground_truth_geometry",
                    "command_source": "pre_insert_settle",
                    "pre_insert_settle_sec": pre_insert_settle_sec,
                    "tracking_gate_checked": True,
                    "tracking_gate_threshold_m": 0.02,
                    "tracking_gate_timeout_sec": 2.0,
                    "preinsert_pose_pinned": True,
                    "moveit_used": False,
                    "ft_correction_used": False,
                },
            )
        )
        pre = waypoints[-1]
    inserted_position = list(pre.tcp_pose.position)
    inserted_position[2] -= insertion_depth_m
    insertion_duration_sec = insertion_depth_m / insertion_speed_mps
    waypoints.append(
        TrajectoryWaypoint(
            timestamp=pre.timestamp + insertion_duration_sec,
            tcp_pose=TCPPose(
                position=inserted_position,
                orientation_xyzw=list(pre.tcp_pose.orientation_xyzw),
            ),
            tcp_velocity=[0.0, 0.0, -insertion_depth_m / insertion_duration_sec],
            phase=PhaseLabel.FINAL_INSERTION,
            source=SourceLabel.CHEATCODE,
            diagnostics={
                "insertion_source": "cheatcode_style_ground_truth_geometry",
                "command_source": "guarded_insert",
                "insertion_speed_mps": insertion_speed_mps,
                "insertion_depth_m": insertion_depth_m,
                "descent_profile": "minimum_jerk",
                "insertion_command_mode": "exact_position",
                "action_frame": "base_link",
                "max_actual_guarded_insert_speed_mps": 0.02,
                "moveit_used": False,
                "ft_correction_used": False,
            },
        )
    )
    return waypoints


class NominalExpert:
    """Build nominal candidates without F/T correction or VLM waypoints."""

    mode = ExpertMode.NOMINAL

    def __init__(self, *, moveit_planner: Any, candidates_per_scene: int = 5):
        self.moveit_planner = moveit_planner
        self.candidates_per_scene = candidates_per_scene

    def generate_candidate(
        self,
        snapshot: SceneSnapshot,
        strategy: VLMStrategy,
        *,
        candidate: ApproachCandidate | None = None,
    ) -> ExpertTrajectoryResult:
        if strategy.mode != ExpertMode.NOMINAL:
            raise ValueError("NominalExpert requires nominal VLMStrategy")
        if strategy.recovery_allowed:
            raise ValueError("Nominal mode must not enable recovery")
        selected = candidate or generate_approach_candidates(
            snapshot,
            strategy,
            count=self.candidates_per_scene,
        )[0]
        planning_result = self.moveit_planner.plan_free_space_approach(snapshot, strategy, selected)
        if isinstance(planning_result, PlanningFailure):
            return ExpertTrajectoryResult(
                accepted_for_replay=False,
                trajectory=None,
                candidate=selected,
                planning_result=planning_result,
                metadata={
                    "mode": "nominal",
                    "ft_correction_used": False,
                    "vlm_waypoints_used": False,
                    "planning_failure": planning_result.to_dict(),
                },
            )

        waypoints = append_cheatcode_insertion_segment(planning_result)
        trajectory = PiecewiseTrajectory(
            waypoints=waypoints,
            metadata=TrajectoryMetadata(
                planning={
                    "expert_mode": "nominal",
                    "vlm_strategy": strategy.to_dict(),
                    "candidate": selected.to_dict(),
                    "moveit": planning_result.to_dict(),
                    "vlm_waypoints_used": False,
                    "ft_correction_used": False,
                },
                diagnostics={
                    "moveit_required": True,
                    "geometric_fallback_available": False,
                },
            ),
        )
        return ExpertTrajectoryResult(
            accepted_for_replay=True,
            trajectory=trajectory,
            candidate=selected,
            planning_result=planning_result,
            metadata={
                "mode": "nominal",
                "ft_correction_used": False,
                "vlm_waypoints_used": False,
            },
        )
