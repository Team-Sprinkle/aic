"""Recovery expert for guarded insertion demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aic_teacher_official.expert_generator.ft_guard import FTGuard, FTGuardConfig, RecoveryPhase
from aic_teacher_official.expert_generator.nominal_expert import (
    ExpertTrajectoryResult,
    append_cheatcode_insertion_segment,
)
from aic_teacher_official.expert_generator.scene_snapshot import SceneSnapshot
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode, VLMStrategy
from aic_teacher_official.trajectory import PiecewiseTrajectory, TrajectoryMetadata


@dataclass(frozen=True)
class RecoveryAction:
    phase: RecoveryPhase
    command: str
    metadata: dict[str, Any]


class RecoveryExpert:
    """Generate recovery demonstrations and expose online recovery guidance."""

    mode = ExpertMode.RECOVERY

    def __init__(self, *, moveit_planner: Any, ft_config: FTGuardConfig | None = None, candidates_per_scene: int = 5):
        self.moveit_planner = moveit_planner
        self.ft_config = ft_config or FTGuardConfig()
        self.candidates_per_scene = candidates_per_scene

    def recover_from_state(
        self,
        current_observation: dict[str, Any],
        current_scene_snapshot: SceneSnapshot,
    ) -> RecoveryAction:
        guard = FTGuard(self.ft_config)
        phase = guard.update(current_observation.get("wrench_force_torque"))
        command = {
            RecoveryPhase.GUARDED_DESCENT: "continue_guarded_descent",
            RecoveryPhase.SOFT_CONTACT: "stop_descent",
            RecoveryPhase.BACKOFF: "backoff",
            RecoveryPhase.REALIGN: "realign_to_cheatcode_geometry",
            RecoveryPhase.RETRY: "retry_insertion",
            RecoveryPhase.SUCCESS: "finish",
            RecoveryPhase.FAILURE: "abort",
        }[phase]
        return RecoveryAction(
            phase=phase,
            command=command,
            metadata={
                "scene_id": current_scene_snapshot.scene_id,
                "ft_guard": guard.metadata(),
            },
        )

    def generate_candidate(self, snapshot: SceneSnapshot, strategy: VLMStrategy, *, candidate: Any) -> ExpertTrajectoryResult:
        if strategy.mode not in {ExpertMode.RECOVERY, ExpertMode.NOMINAL_RECOVERY}:
            raise ValueError("RecoveryExpert requires recovery or nominalrecovery VLMStrategy")
        planning_result = self.moveit_planner.plan_free_space_approach(snapshot, strategy, candidate)
        from aic_teacher_official.expert_generator.moveit_planner import PlanningFailure

        if isinstance(planning_result, PlanningFailure):
            return ExpertTrajectoryResult(
                accepted_for_replay=False,
                trajectory=None,
                candidate=candidate,
                planning_result=planning_result,
                metadata={"mode": "recovery", "planning_failure": planning_result.to_dict()},
            )
        waypoints = append_cheatcode_insertion_segment(
            planning_result,
            insertion_depth_m=0.070,
            insertion_speed_mps=0.0009,
            local_preinsert_align_sec=0.8,
            pre_insert_settle_sec=0.35,
            handoff_blend_sec=0.5,
        )
        mode_value = strategy.mode.value
        trajectory = PiecewiseTrajectory(
            waypoints=waypoints,
            metadata=TrajectoryMetadata(
                planning={
                    "expert_mode": mode_value,
                    "vlm_strategy": strategy.to_dict(),
                    "candidate": candidate.to_dict(),
                    "moveit": planning_result.to_dict(),
                    "vlm_waypoints_used": False,
                    "ft_correction_used": True,
                    "ft_guard": FTGuard(self.ft_config).metadata(),
                },
                diagnostics={
                    "moveit_required": True,
                    "geometric_fallback_available": False,
                    "recover_from_state_api": "recovery_expert.recover_from_state(current_observation, current_scene_snapshot)",
                },
            ),
        )
        return ExpertTrajectoryResult(
            accepted_for_replay=True,
            trajectory=trajectory,
            candidate=candidate,
            planning_result=planning_result,
            metadata={"mode": mode_value, "ft_correction_used": True, "vlm_waypoints_used": False},
        )
