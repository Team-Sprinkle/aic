"""Static parity audit helpers for the teacher stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditRow:
    item: str
    current_availability: str
    source: str
    status: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "current_availability": self.current_availability,
            "source": self.source,
            "status": self.status,
            "notes": self.notes,
        }


def observation_parity_rows() -> list[AuditRow]:
    return [
        AuditRow(
            item="wrist camera images",
            current_availability="Matched in live teacher path; blank placeholders in mock path",
            source="aic_gym_gz/io.py:RosCameraSidecarIO.observation_from_state; aic_gym_gz/teacher/runner.py:_observation_summary",
            status="matched",
            notes="Planner path now carries image summaries end-to-end; mock env still uses zero images.",
        ),
        AuditRow(
            item="image timestamps",
            current_availability="Matched in live teacher path; zeros in mock path",
            source="aic_gym_gz/io.py:RosCameraSidecarIO.observation_from_state; aic_gym_gz/teacher/context.py:TeacherContextExtractor.build_planning_state",
            status="matched",
            notes="Teacher planning state now includes timestamp map and image summaries.",
        ),
        AuditRow(
            item="force/torque sensor data",
            current_availability="Approximate in live gym path, synthetic in mock path",
            source="aic_gym_gz/runtime.py:ScenarioGymGzBackend._runtime_state_from_observation; aic_gym_gz/runtime.py:MockStepperBackend.step_ticks",
            status="approximate",
            notes="Official path uses `Observation.wrist_wrench`; live gym runtime still fills zeros today unless downstream bridge exposes wrench.",
        ),
        AuditRow(
            item="TCP pose",
            current_availability="Matched",
            source="aic_gym_gz/runtime.py:_runtime_state_from_observation; aic_gym_gz/io.py:_base_observation",
            status="matched",
        ),
        AuditRow(
            item="TCP velocity",
            current_availability="Matched in gym observation; not yet sourced from official controller_state",
            source="aic_gym_gz/runtime.py:_runtime_state_from_observation; aic_gym_gz/io.py:_base_observation",
            status="matched",
            notes="Live gym derives velocity from successive samples; official path publishes controller velocity directly.",
        ),
        AuditRow(
            item="joint positions/velocities",
            current_availability="Matched for gym path",
            source="aic_gym_gz/runtime.py:_runtime_state_from_observation; aic_gym_gz/io.py:_base_observation",
            status="matched",
        ),
        AuditRow(
            item="controller state / reference tcp pose / tcp_error",
            current_availability="Missing in teacher planner path",
            source="aic_interfaces/aic_model_interfaces/msg/Observation.msg; aic_adapter/src/aic_adapter.cpp",
            status="missing",
            notes="Official ROS observations include `controller_state`; teacher path currently logs only the gym observation slice.",
        ),
        AuditRow(
            item="camera info",
            current_availability="Missing in teacher planner path",
            source="aic_interfaces/aic_model_interfaces/msg/Observation.msg; aic_adapter/src/aic_adapter.cpp",
            status="missing",
            notes="Official policy receives `CameraInfo`; teacher planner currently only uses image arrays and timestamps.",
        ),
        AuditRow(
            item="plug pose / target pose / privileged oracle board metadata",
            current_availability="Matched and extended in teacher mode",
            source="aic_gym_gz/io.py:_base_observation; aic_gym_gz/teacher/context.py",
            status="matched",
            notes="This is teacher-only privileged context, not part of official participant observations.",
        ),
    ]


def scoring_parity_rows() -> list[AuditRow]:
    return [
        AuditRow(
            item="trajectory smoothness / jerk",
            current_availability="Now official-style in teacher selection; older gym reward remains approximate",
            source="aic_scoring/src/ScoringTier2.cc:GetTrajectoryJerkScore; aic_gym_gz/teacher/scoring.py:OfficialStyleScoreEvaluator._official_like_average_jerk; aic_gym_gz/reward.py:_average_linear_jerk",
            status="approximate",
            notes="Teacher search uses quadratic-window jerk like official scorer; env reward still uses simpler finite differences.",
        ),
        AuditRow(
            item="task duration",
            current_availability="Matched",
            source="aic_scoring/src/ScoringTier2.cc:GetTaskDurationScore; aic_gym_gz/teacher/scoring.py:OfficialStyleScoreEvaluator.evaluate_rollout",
            status="matched",
        ),
        AuditRow(
            item="trajectory efficiency",
            current_availability="Matched",
            source="aic_scoring/src/ScoringTier2.cc:GetTrajectoryEfficiencyScore; aic_gym_gz/teacher/scoring.py:OfficialStyleScoreEvaluator.evaluate_rollout",
            status="matched",
        ),
        AuditRow(
            item="insertion force penalty",
            current_availability="Matched in formula, limited by live wrench availability",
            source="aic_scoring/src/ScoringTier2.cc:GetInsertionForceScore; aic_gym_gz/teacher/scoring.py:OfficialStyleScoreEvaluator.evaluate_rollout",
            status="approximate",
            notes="Thresholds match; live teacher accuracy depends on wrench propagation from runtime.",
        ),
        AuditRow(
            item="off-limit contact penalty",
            current_availability="Matched",
            source="aic_scoring/src/ScoringTier2.cc:GetContactsScore; aic_gym_gz/teacher/scoring.py:OfficialStyleScoreEvaluator.evaluate_rollout",
            status="matched",
        ),
        AuditRow(
            item="successful insertion / wrong port",
            current_availability="Matched",
            source="aic_scoring/src/ScoringTier2.cc:ComputeTier3Score; aic_gym_gz/task.py:evaluate_step; aic_gym_gz/teacher/scoring.py:_tier3_score",
            status="matched",
        ),
        AuditRow(
            item="partial insertion depth",
            current_availability="Approximate",
            source="aic_scoring/src/ScoringTier2.cc:GetDistanceScore; aic_gym_gz/teacher/scoring.py:_tier3_score",
            status="approximate",
            notes="Official scorer uses port entrance TF and bounding box; gym path currently estimates with target pose and configurable depth.",
        ),
        AuditRow(
            item="tier1 validity",
            current_availability="Approximate",
            source="docs/scoring.md; aic_model/aic_model/aic_model.py",
            status="approximate",
            notes="Teacher search assumes valid rollouts locally; official Tier 1 still requires running through `aic_model`.",
        ),
    ]


def dataset_compatibility_rows() -> list[AuditRow]:
    return [
        AuditRow(
            item="teleop / LeRobot dataset schema",
            current_availability="Compatible adapter implemented",
            source="origin/exp/data:aic_utils/lerobot_robot_aic/lerobot_robot_aic/policy_recorder.py; aic_gym_gz/teacher/dataset_export.py",
            status="matched",
            notes="Teacher exporter uses the same observation/action key family as exp/data policy recorder.",
        ),
        AuditRow(
            item="rich planner/scenario/score metadata",
            current_availability="Available via sidecar metadata JSON",
            source="aic_gym_gz/teacher/dataset_export.py",
            status="matched",
            notes="Native LeRobot frames do not natively carry all teacher metadata, so sidecar export is used.",
        ),
        AuditRow(
            item="controller-state-specific fields",
            current_availability="Approximate compatibility",
            source="origin/exp/data:aic_utils/lerobot_robot_aic/lerobot_robot_aic/policy_recorder.py; aic_gym_gz/teacher/dataset_export.py",
            status="approximate",
            notes="`tcp_error.*` is synthesized from commanded target vs observed TCP because gym teacher artifacts do not carry official controller error directly.",
        ),
    ]
