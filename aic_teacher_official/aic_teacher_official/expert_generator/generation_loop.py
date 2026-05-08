"""Accepted-trajectory-targeting generation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aic_teacher_official.expert_generator.candidate_generation import generate_approach_candidates
from aic_teacher_official.expert_generator.dataset_writer import DatasetMetadataWriter, ExpertEpisodeMetadata
from aic_teacher_official.expert_generator.trajectory_validator import TrajectoryValidator, ValidationResult
from aic_teacher_official.expert_generator.vlm_strategy import ExpertMode


class SceneProvider(Protocol):
    def next_scene(self, *, attempt_index: int, rerandomize_scene: bool, respawn_assets: bool) -> Any:
        ...


class StrategyProvider(Protocol):
    def strategy_for_scene(self, snapshot: Any, *, mode: ExpertMode, output_dir: Any | None = None) -> Any:
        ...


class ReplayRunner(Protocol):
    def replay_and_score(self, trajectory: Any, *, attempt_index: int, candidate_index: int) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class GenerationConfig:
    expert_mode: ExpertMode
    target_accepted_trajectories: int = 100
    max_total_attempts: int = 300
    candidates_per_scene: int = 5
    rerandomize_scene: bool = True
    respawn_assets: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_mode": self.expert_mode.value,
            "target_accepted_trajectories": self.target_accepted_trajectories,
            "max_total_attempts": self.max_total_attempts,
            "candidates_per_scene": self.candidates_per_scene,
            "rerandomize_scene": self.rerandomize_scene,
            "respawn_assets": self.respawn_assets,
        }


@dataclass(frozen=True)
class GenerationSummary:
    accepted: int
    attempts: int
    stopped_reason: str
    results: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aic_expert_generation_summary/v1",
            "accepted": self.accepted,
            "attempts": self.attempts,
            "stopped_reason": self.stopped_reason,
            "results": list(self.results),
        }


class ExpertGenerationLoop:
    def __init__(
        self,
        *,
        config: GenerationConfig,
        scene_provider: SceneProvider,
        strategy_provider: StrategyProvider,
        expert: Any,
        replay_runner: ReplayRunner,
        validator: TrajectoryValidator,
        metadata_writer: DatasetMetadataWriter,
    ):
        if config.target_accepted_trajectories <= 0:
            raise ValueError("target_accepted_trajectories must be > 0")
        if config.max_total_attempts <= 0:
            raise ValueError("max_total_attempts must be > 0")
        self.config = config
        self.scene_provider = scene_provider
        self.strategy_provider = strategy_provider
        self.expert = expert
        self.replay_runner = replay_runner
        self.validator = validator
        self.metadata_writer = metadata_writer

    def run(self) -> GenerationSummary:
        accepted = 0
        attempts = 0
        scene_index = 0
        results: list[dict[str, Any]] = []
        while accepted < self.config.target_accepted_trajectories and attempts < self.config.max_total_attempts:
            scene_index += 1
            snapshot = self.scene_provider.next_scene(
                attempt_index=scene_index,
                rerandomize_scene=self.config.rerandomize_scene,
                respawn_assets=self.config.respawn_assets,
            )
            strategy = self.strategy_provider.strategy_for_scene(snapshot, mode=self.config.expert_mode)
            candidates = generate_approach_candidates(
                snapshot,
                strategy,
                count=self.config.candidates_per_scene,
            )
            for candidate in candidates:
                if accepted >= self.config.target_accepted_trajectories or attempts >= self.config.max_total_attempts:
                    break
                attempts += 1
                generated = self.expert.generate_candidate(snapshot, strategy, candidate=candidate)
                if generated.trajectory is None:
                    results.append({"attempt": attempts, "accepted": False, "reason": "planning_failed", "metadata": generated.metadata})
                    continue
                replay_metrics = self.replay_runner.replay_and_score(
                    generated.trajectory,
                    attempt_index=attempts,
                    candidate_index=candidate.index,
                )
                validation = self.validator.evaluate(
                    {
                        **replay_metrics,
                        "mode": self.config.expert_mode.value,
                        "vlm_cable_risk": strategy.cable_risk.value,
                        "moveit_success": True,
                        "candidate_index": candidate.index,
                        "scene_seed": snapshot.seed,
                        "phase_labels": [wp.phase.value for wp in generated.trajectory.waypoints],
                    }
                )
                results.append({"attempt": attempts, "accepted": validation.accepted, "validation": validation.to_dict()})
                if validation.accepted:
                    self._write_metadata(accepted, snapshot, strategy, candidate, generated, validation)
                    accepted += 1
                    break
        reason = "target_reached" if accepted >= self.config.target_accepted_trajectories else "max_attempts_exhausted"
        return GenerationSummary(accepted=accepted, attempts=attempts, stopped_reason=reason, results=results)

    def _write_metadata(self, episode_index: int, snapshot: Any, strategy: Any, candidate: Any, generated: Any, validation: ValidationResult) -> None:
        planning_dict = generated.planning_result.to_dict() if generated.planning_result else {}
        phase_labels = [
            {
                "timestamp": wp.timestamp,
                "phase": wp.phase.value,
                "source": wp.source.value,
            }
            for wp in generated.trajectory.waypoints
        ]
        self.metadata_writer.append_episode(
            ExpertEpisodeMetadata(
                episode_index=episode_index,
                mode=self.config.expert_mode.value,
                scene_id=snapshot.scene_id,
                candidate_index=candidate.index,
                trajectory_path=None,
                validation=validation.to_dict(),
                vlm_strategy=strategy.to_dict(),
                moveit=planning_dict,
                phase_labels=phase_labels,
                extra={
                    "snapshot": snapshot.to_dict(),
                    "candidate": candidate.to_dict(),
                    "priority_order": [
                        "highest_scoring_nominal_experts",
                        "recovery_experts",
                        "diversity",
                        "speed",
                    ],
                },
            )
        )
