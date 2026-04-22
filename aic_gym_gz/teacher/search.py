"""Candidate generation, rollout search, and ranking for teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from ..env import AicInsertionEnv
from ..planners.base import PlannerBackend
from ..utils import to_jsonable
from .policy import AgentTeacherController, TeacherConfig
from .quality import ranking_quality_adjustment
from .replay import TeacherReplayArtifact, save_teacher_replay
from .runner import TeacherRolloutResult, run_teacher_rollout
from .scoring import OfficialStyleScoreEvaluator
from .types import TeacherPlan, TeacherWaypoint


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mode: str
    planner_candidate_index: int = 0
    perturbation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "planner_candidate_index": self.planner_candidate_index,
            "perturbation_xyz": list(self.perturbation_xyz),
            "yaw_offset": self.yaw_offset,
        }


@dataclass
class TeacherSearchConfig:
    planner_candidate_count: int = 3
    local_perturbation_count: int = 4
    top_k: int = 3
    near_perfect_threshold: float = 90.0
    candidate_segment_limit: int = 16
    enable_probes: bool = True


class CandidatePlannerBackend(PlannerBackend):
    def __init__(self, base_backend: PlannerBackend, spec: CandidateSpec) -> None:
        self._base_backend = base_backend
        self._spec = spec

    @property
    def backend_name(self) -> str:
        return f"{self._base_backend.backend_name}:{self._spec.mode}"

    def plan(self, state, *, candidate_index: int = 0) -> TeacherPlan:
        del candidate_index
        base_plan = self._base_backend.plan(
            state,
            candidate_index=self._spec.planner_candidate_index,
        )
        if self._spec.mode != "local_perturbation":
            return base_plan
        adjusted_waypoints = []
        scale = {
            "free_space_approach": 1.0,
            "obstacle_avoidance": 0.75,
            "pre_insert_align": 0.35,
            "guarded_insert": 0.15,
            "cable_probe": 0.25,
            "backoff_and_retry": 0.5,
        }.get(base_plan.next_phase, 0.5)
        for waypoint in base_plan.waypoints:
            adjusted_waypoints.append(
                TeacherWaypoint(
                    position_xyz=(
                        float(waypoint.position_xyz[0] + self._spec.perturbation_xyz[0] * scale),
                        float(waypoint.position_xyz[1] + self._spec.perturbation_xyz[1] * scale),
                        float(waypoint.position_xyz[2] + self._spec.perturbation_xyz[2] * scale),
                    ),
                    yaw=float(waypoint.yaw + self._spec.yaw_offset * scale),
                    speed_scale=waypoint.speed_scale,
                    clearance_hint=waypoint.clearance_hint,
                )
            )
        return TeacherPlan(
            next_phase=base_plan.next_phase,
            waypoints=tuple(adjusted_waypoints),
            motion_mode=base_plan.motion_mode,
            caution_flag=base_plan.caution_flag,
            should_probe=base_plan.should_probe,
            segment_horizon_steps=base_plan.segment_horizon_steps,
            segment_granularity=base_plan.segment_granularity,
            rationale_summary=(
                f"{base_plan.rationale_summary}; mode={self._spec.mode}; "
                f"delta={self._spec.perturbation_xyz}; yaw_offset={self._spec.yaw_offset:.4f}"
            ),
        )


class TeacherCandidateGenerator:
    def __init__(self, config: TeacherSearchConfig) -> None:
        self._config = config

    def generate(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for index in range(self._config.planner_candidate_count):
            specs.append(
                CandidateSpec(
                    name=f"planner_candidate_{index}",
                    mode="planner_waypoint",
                    planner_candidate_index=index,
                )
            )
        perturbations = (
            (0.010, 0.0, 0.0, 0.0),
            (-0.010, 0.0, 0.0, 0.0),
            (0.0, 0.010, 0.0, 0.06),
            (0.0, -0.010, 0.0, -0.06),
        )
        for index, (dx, dy, dz, yaw) in enumerate(perturbations[: self._config.local_perturbation_count]):
            specs.append(
                CandidateSpec(
                    name=f"local_perturbation_{index}",
                    mode="local_perturbation",
                    planner_candidate_index=0,
                    perturbation_xyz=(dx, dy, dz),
                    yaw_offset=yaw,
                )
            )
        return specs


@dataclass(frozen=True)
class TeacherSearchResult:
    payload: dict[str, Any]
    output_path: Path | None = None


class TeacherCandidateSearch:
    def __init__(
        self,
        *,
        env_factory: Callable[[], AicInsertionEnv],
        planner_factory: Callable[[], PlannerBackend],
        config: TeacherSearchConfig = TeacherSearchConfig(),
    ) -> None:
        self.env_factory = env_factory
        self.planner_factory = planner_factory
        self.config = config
        self.generator = TeacherCandidateGenerator(config)
        self.score_evaluator = OfficialStyleScoreEvaluator(
            near_perfect_threshold=config.near_perfect_threshold
        )

    def run(
        self,
        *,
        seed: int = 123,
        trial_id: str | None = None,
        output_path: Path | None = None,
    ) -> TeacherSearchResult:
        ranked: list[dict[str, Any]] = []
        limitations = [
            "Exact intermediate-state cloning is unavailable; candidate search replays from the same seed/trial reset.",
            "Search is exact on the deterministic mock path and best-effort on live Gazebo depending on reset determinism.",
        ]
        planner_backend = self.planner_factory()
        search_budget_key = f"teacher_search:{seed}:{trial_id or 'default'}:{planner_backend.backend_name}"
        if hasattr(planner_backend, "reset_search_budget"):
            planner_backend.reset_search_budget(search_budget_key)
        for spec in self.generator.generate():
            env = self.env_factory()
            try:
                base_backend = self.planner_factory()
                if hasattr(base_backend, "set_search_budget_key"):
                    base_backend.set_search_budget_key(search_budget_key)
                planner = CandidatePlannerBackend(base_backend, spec)
                controller = AgentTeacherController(
                    planner=planner,
                    config=TeacherConfig(
                        candidate_plan_count=1,
                        enable_probes=self.config.enable_probes,
                        segment_limit=self.config.candidate_segment_limit,
                    ),
                )
                rollout = run_teacher_rollout(
                    env=env,
                    controller=controller,
                    seed=seed,
                    trial_id=trial_id,
                )
                artifact_dict = rollout.artifact.to_dict()
                score = self.score_evaluator.evaluate_rollout(artifact_dict)
                ranking_metrics = self._ranking_metrics(
                    artifact=artifact_dict,
                    teacher_score=score.to_dict(),
                )
                ranked.append(
                    {
                        "candidate_spec": spec.to_dict(),
                        "artifact": artifact_dict,
                        "teacher_official_style_score": score.to_dict(),
                        "official_style_score": score.to_dict(),
                        "ranking_metrics": ranking_metrics,
                    }
                )
            finally:
                env.close()
        ranked.sort(
            key=lambda item: float(item["ranking_metrics"]["composite_score"]),
            reverse=True,
        )
        for index, entry in enumerate(ranked):
            entry["rank"] = index + 1
            entry["selected_top_k"] = index < self.config.top_k
            entry["near_perfect"] = bool(
                entry["teacher_official_style_score"]["selection"]["near_perfect"]
            )
        payload = {
            "metadata": {
                "seed": seed,
                "trial_id": trial_id,
                "near_perfect_threshold": self.config.near_perfect_threshold,
                "top_k": self.config.top_k,
                "planner_backend": planner_backend.backend_name,
            },
            "ranked_candidates": ranked,
            "top_candidates": ranked[: self.config.top_k],
            "near_perfect_candidates": [item for item in ranked if item["near_perfect"]],
            "limitations": limitations,
        }
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return TeacherSearchResult(payload=payload, output_path=output_path)

    def _ranking_metrics(
        self,
        *,
        artifact: dict[str, Any],
        teacher_score: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = artifact.get("metadata", {})
        final_metrics = metadata.get("final_metrics", {})
        data_quality = metadata.get("data_quality", {})
        quality_adjustment, quality_penalties = ranking_quality_adjustment(data_quality)
        gym_final_score = final_metrics.get("gym_final_score")
        rl_step_reward_total = float(final_metrics.get("rl_step_reward_total", 0.0))
        teacher_total = float(teacher_score["total_score"])
        composite_score = teacher_total
        if gym_final_score is not None:
            composite_score += 0.15 * float(gym_final_score)
        composite_score += 0.02 * max(min(rl_step_reward_total, 100.0), -100.0)
        composite_score += quality_adjustment
        return {
            "teacher_official_style_score": teacher_total,
            "gym_final_score": gym_final_score,
            "rl_step_reward_total": rl_step_reward_total,
            "data_quality": data_quality,
            "quality_adjustment": quality_adjustment,
            "quality_penalties": quality_penalties,
            "composite_score": composite_score,
            "signals_exact_vs_approximate": {
                signal: quality.get("source")
                for signal, quality in data_quality.items()
            },
        }


def export_selected_candidate_to_replay(
    search_payload: dict[str, Any],
    *,
    candidate_rank: int = 1,
    output_path: Path | str,
) -> TeacherReplayArtifact:
    ranked = search_payload["ranked_candidates"]
    selected = next(item for item in ranked if int(item["rank"]) == candidate_rank)
    artifact_payload = dict(selected["artifact"])
    artifact_payload["metadata"] = {
        **artifact_payload["metadata"],
        "candidate_rank": candidate_rank,
        "candidate_mode": selected["candidate_spec"]["mode"],
        "official_style_score": selected["official_style_score"],
        "teacher_official_style_score": selected["teacher_official_style_score"],
        "ranking_metrics": selected["ranking_metrics"],
    }
    artifact = TeacherReplayArtifact(
        metadata=artifact_payload["metadata"],
        trajectory_segments=artifact_payload["trajectory_segments"],
        probe_results=artifact_payload.get("probe_results", []),
        planner_candidates=artifact_payload.get("planner_candidates", []),
        step_logs=artifact_payload.get("step_logs", []),
        final_info=artifact_payload.get("final_info", {}),
        limitations=artifact_payload.get("limitations", []),
    )
    save_teacher_replay(artifact, output_path)
    return artifact
