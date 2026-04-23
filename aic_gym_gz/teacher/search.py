"""Candidate generation, rollout search, and ranking for teacher trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..env import AicInsertionEnv
from ..planners.base import PlannerBackend
from ..utils import to_jsonable
from .local_scores import artifact_progress_metrics, build_local_trajectory_score_summary
from .policy import AgentTeacherController, TeacherConfig
from .planning import candidate_family_for_index
from .quality import ranking_auxiliary_adjustment, ranking_quality_adjustment
from .replay import TeacherReplayArtifact, save_teacher_replay
from .runner import TeacherRolloutResult, run_teacher_rollout
from .scoring import OfficialStyleScoreEvaluator
from .types import TeacherPlan, TeacherWaypoint


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    mode: str
    planner_candidate_index: int = 0
    family: str = "baseline_safe"
    source_candidate_name: str | None = None
    refinement_style: str | None = None
    segment_limit_override: int | None = None
    perturbation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw_offset: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "planner_candidate_index": self.planner_candidate_index,
            "family": self.family,
            "source_candidate_name": self.source_candidate_name,
            "refinement_style": self.refinement_style,
            "segment_limit_override": self.segment_limit_override,
            "perturbation_xyz": list(self.perturbation_xyz),
            "yaw_offset": self.yaw_offset,
        }


@dataclass
class TeacherSearchConfig:
    planner_candidate_count: int = 4
    local_perturbation_count: int = 4
    top_k: int = 3
    near_perfect_threshold: float = 90.0
    candidate_segment_limit: int = 8
    refinement_segment_limit: int = 8
    enable_probes: bool = True
    auxiliary_hidden_contact_event_weight: float = 0.35
    auxiliary_quiet_hidden_force_weight: float = 0.15
    auxiliary_repeated_contact_rich_weight: float = 0.05
    duplicate_similarity_threshold: float = 0.985
    duplicate_penalty_weight: float = 1.25


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
        if self._spec.mode == "planner_waypoint":
            return _apply_candidate_family(base_plan=base_plan, state=state, spec=self._spec)
        transformed = _apply_candidate_family(base_plan=base_plan, state=state, spec=self._spec)
        transformed = _apply_refinement_style(base_plan=transformed, state=state, spec=self._spec)
        adjusted_waypoints = []
        scale = {
            "free_space_approach": 1.0,
            "obstacle_avoidance": 0.75,
            "pre_insert_align": 0.35,
            "guarded_insert": 0.15,
            "cable_probe": 0.25,
            "backoff_and_retry": 0.5,
        }.get(transformed.next_phase, 0.5)
        for waypoint in transformed.waypoints:
            adjusted_waypoints.append(
                TeacherWaypoint(
                    position_xyz=(
                        float(waypoint.position_xyz[0] + self._spec.perturbation_xyz[0] * scale),
                        float(waypoint.position_xyz[1] + self._spec.perturbation_xyz[1] * scale),
                        float(waypoint.position_xyz[2] + self._spec.perturbation_xyz[2] * scale),
                    ),
                    yaw=float(waypoint.yaw + self._spec.yaw_offset * scale),
                    speed_scale=max(0.2, float(waypoint.speed_scale)),
                    clearance_hint=waypoint.clearance_hint,
                )
            )
        return TeacherPlan(
            next_phase=transformed.next_phase,
            waypoints=tuple(adjusted_waypoints),
            motion_mode=transformed.motion_mode,
            caution_flag=transformed.caution_flag,
            should_probe=transformed.should_probe,
            segment_horizon_steps=transformed.segment_horizon_steps,
            segment_granularity=transformed.segment_granularity,
            rationale_summary=(
                f"{transformed.rationale_summary}; mode={self._spec.mode}; family={self._spec.family}; "
                f"refinement_style={self._spec.refinement_style}; "
                f"delta={self._spec.perturbation_xyz}; yaw_offset={self._spec.yaw_offset:.4f}"
            ),
        )


class TeacherCandidateGenerator:
    def __init__(self, config: TeacherSearchConfig) -> None:
        self._config = config

    def planner_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for index in range(self._config.planner_candidate_count):
            family = candidate_family_for_index(index)
            specs.append(
                CandidateSpec(
                    name=f"{family['name']}_{index}",
                    mode="planner_waypoint",
                    planner_candidate_index=index,
                    family=str(family["name"]),
                )
            )
        return specs

    def refinement_specs(self, planner_ranked: list[dict[str, Any]]) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        seeds = _select_refinement_seeds(
            planner_ranked=planner_ranked,
            limit=self._config.local_perturbation_count,
        )
        for slot, seed in enumerate(seeds):
            style = _refinement_style_for_seed(seed=seed, slot=slot)
            if style is None:
                continue
            dx, dy, dz, yaw = _refinement_offsets(style=style)
            candidate_spec = seed["candidate_spec"]
            specs.append(
                CandidateSpec(
                    name=f"{candidate_spec['family']}_{style}_{slot}",
                    mode="adaptive_refinement",
                    planner_candidate_index=int(candidate_spec["planner_candidate_index"]),
                    family=str(candidate_spec["family"]),
                    source_candidate_name=str(candidate_spec["name"]),
                    refinement_style=style,
                    segment_limit_override=max(
                        self._config.refinement_segment_limit,
                        self._config.candidate_segment_limit,
                    ),
                    perturbation_xyz=(dx, dy, dz),
                    yaw_offset=yaw,
                )
            )
        perturbations = (
            ("alignment_first", 0.006, 0.0, 0.0, 0.0),
            ("guarded_insert", -0.006, 0.0, 0.0, 0.0),
            ("obstacle_clearance", 0.0, 0.008, 0.0, 0.08),
            ("recovery_backoff", 0.0, -0.008, -0.004, -0.08),
        )
        for index, (family, dx, dy, dz, yaw) in enumerate(perturbations):
            if len(specs) >= self._config.local_perturbation_count:
                break
            specs.append(
                CandidateSpec(
                    name=f"{family}_perturbation_{index}",
                    mode="local_perturbation",
                    planner_candidate_index=index,
                    family=family,
                    segment_limit_override=max(
                        self._config.refinement_segment_limit,
                        self._config.candidate_segment_limit,
                    ),
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
        selected_top: list[dict[str, Any]] = []
        limitations = [
            "Exact intermediate-state cloning is unavailable; candidate search replays from the same seed/trial reset.",
            "Search is exact on the deterministic mock path and best-effort on live Gazebo depending on reset determinism.",
        ]
        planner_backend = self.planner_factory()
        search_budget_key = f"teacher_search:{seed}:{trial_id or 'default'}:{planner_backend.backend_name}"
        if hasattr(planner_backend, "reset_search_budget"):
            planner_backend.reset_search_budget(search_budget_key)
        planner_specs = self.generator.planner_specs()
        for spec in planner_specs:
            ranked.append(
                self._evaluate_candidate(
                    spec=spec,
                    seed=seed,
                    trial_id=trial_id,
                    search_budget_key=search_budget_key,
                )
            )
        planner_ranked = sorted(
            [item for item in ranked if item["candidate_spec"]["mode"] == "planner_waypoint"],
            key=lambda item: float(item["ranking_metrics"]["composite_score"]),
            reverse=True,
        )
        for spec in self.generator.refinement_specs(planner_ranked):
            ranked.append(
                self._evaluate_candidate(
                    spec=spec,
                    seed=seed,
                    trial_id=trial_id,
                    search_budget_key=search_budget_key,
                )
            )
        ranked.sort(
            key=lambda item: float(item["ranking_metrics"]["composite_score"]),
            reverse=True,
        )
        self._apply_duplicate_penalties(ranked)
        ranked.sort(
            key=lambda item: float(item["ranking_metrics"]["composite_score"]),
            reverse=True,
        )
        for index, entry in enumerate(ranked):
            entry["rank"] = index + 1
            entry["selected_top_k"] = False
            entry["near_perfect"] = bool(
                entry["teacher_official_style_score"]["selection"]["near_perfect"]
            )
        selected_top = self._select_diverse_top_k(ranked)
        selected_names = {entry["candidate_spec"]["name"] for entry in selected_top}
        for entry in ranked:
            entry["selected_top_k"] = entry["candidate_spec"]["name"] in selected_names
        payload = {
            "metadata": {
                "seed": seed,
                "trial_id": trial_id,
                "near_perfect_threshold": self.config.near_perfect_threshold,
                "top_k": self.config.top_k,
                "planner_backend": planner_backend.backend_name,
            },
            "ranked_candidates": ranked,
            "top_candidates": selected_top,
            "near_perfect_candidates": [item for item in ranked if item["near_perfect"]],
            "limitations": limitations,
        }
        if output_path is not None:
            Path(output_path).write_text(
                json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return TeacherSearchResult(payload=payload, output_path=output_path)

    def _evaluate_candidate(
        self,
        *,
        spec: CandidateSpec,
        seed: int,
        trial_id: str | None,
        search_budget_key: str,
    ) -> dict[str, Any]:
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
                    segment_limit=(
                        spec.segment_limit_override
                        if spec.segment_limit_override is not None
                        else self.config.candidate_segment_limit
                    ),
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
            return {
                "candidate_spec": spec.to_dict(),
                "artifact": artifact_dict,
                "teacher_official_style_score": score.to_dict(),
                "official_style_score": score.to_dict(),
                "ranking_metrics": ranking_metrics,
            }
        finally:
            env.close()

    def _ranking_metrics(
        self,
        *,
        artifact: dict[str, Any],
        teacher_score: dict[str, Any],
    ) -> dict[str, Any]:
        metadata = artifact.get("metadata", {})
        final_metrics = metadata.get("final_metrics", {})
        data_quality = metadata.get("data_quality", {})
        auxiliary_metrics = metadata.get("auxiliary_summary_metadata", {})
        quality_adjustment, quality_penalties = ranking_quality_adjustment(data_quality)
        auxiliary_adjustment, auxiliary_penalties = ranking_auxiliary_adjustment(
            auxiliary_metrics,
            hidden_contact_event_weight=self.config.auxiliary_hidden_contact_event_weight,
            quiet_hidden_force_weight=self.config.auxiliary_quiet_hidden_force_weight,
            repeated_contact_rich_weight=self.config.auxiliary_repeated_contact_rich_weight,
        )
        gym_final_score = final_metrics.get("gym_final_score")
        rl_step_reward_total = float(final_metrics.get("rl_step_reward_total", 0.0))
        teacher_total = float(teacher_score["total_score"])
        progress_metrics = artifact_progress_metrics(artifact)
        local_summary = build_local_trajectory_score_summary(
            teacher_official_style_score=teacher_total,
            gym_final_score=None if gym_final_score is None else float(gym_final_score),
            rl_step_reward_total=rl_step_reward_total,
            progress_to_target_m=progress_metrics.get("progress_to_target_m"),
            final_distance_to_target_m=progress_metrics.get("final_distance_to_target_m"),
            path_length_m=progress_metrics.get("path_length_m"),
            quality_adjustment=quality_adjustment,
            auxiliary_adjustment=auxiliary_adjustment,
        )
        return {
            "teacher_official_style_score": teacher_total,
            "gym_final_score": gym_final_score,
            "rl_step_reward_total": rl_step_reward_total,
            "progress_metrics": progress_metrics,
            "local_trajectory_score_summary": local_summary,
            "data_quality": data_quality,
            "auxiliary_summary_metadata": auxiliary_metrics,
            "quality_adjustment": quality_adjustment,
            "quality_penalties": quality_penalties,
            "auxiliary_adjustment": auxiliary_adjustment,
            "auxiliary_penalties": auxiliary_penalties,
            "duplicate_penalty": 0.0,
            "composite_score": float(local_summary["scalar_score"]),
            "signals_exact_vs_approximate": {
                signal: quality.get("source")
                for signal, quality in data_quality.items()
            },
        }

    def _apply_duplicate_penalties(self, ranked: list[dict[str, Any]]) -> None:
        signatures: list[dict[str, Any]] = []
        for entry in ranked:
            signature = _candidate_signature(entry)
            penalty = 0.0
            duplicate_of = None
            for earlier in signatures:
                similarity = _candidate_similarity(signature, earlier)
                if similarity >= self.config.duplicate_similarity_threshold:
                    penalty = -self.config.duplicate_penalty_weight * similarity
                    duplicate_of = earlier["name"]
                    break
            entry["ranking_metrics"]["duplicate_penalty"] = penalty
            if duplicate_of is not None:
                entry["ranking_metrics"]["duplicate_of"] = duplicate_of
            entry["ranking_metrics"]["composite_score"] = float(entry["ranking_metrics"]["composite_score"]) + penalty
            local_summary = dict(entry["ranking_metrics"].get("local_trajectory_score_summary", {}))
            components = dict(local_summary.get("components", {}))
            components["duplicate_penalty"] = penalty
            local_summary["components"] = components
            local_summary["scalar_score"] = float(local_summary.get("scalar_score", 0.0)) + penalty
            entry["ranking_metrics"]["local_trajectory_score_summary"] = local_summary
            signatures.append(signature)

    def _select_diverse_top_k(self, ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for entry in ranked:
            signature = _candidate_signature(entry)
            if any(
                _candidate_similarity(signature, _candidate_signature(existing))
                >= self.config.duplicate_similarity_threshold
                for existing in selected
            ):
                continue
            selected.append(entry)
            if len(selected) >= self.config.top_k:
                break
        if len(selected) < self.config.top_k:
            for entry in ranked:
                if entry not in selected:
                    selected.append(entry)
                if len(selected) >= self.config.top_k:
                    break
        return selected


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


def _apply_candidate_family(*, base_plan: TeacherPlan, state, spec: CandidateSpec) -> TeacherPlan:
    family = spec.family
    policy = state.policy_context
    phase_guidance = state.temporal_context.get("phase_guidance", {})
    entrance = policy.get("target_port_entrance_pose") or policy.get("target_port_pose")
    target = policy.get("target_port_pose")
    waypoints = list(base_plan.waypoints)
    next_phase = base_plan.next_phase
    motion_mode = base_plan.motion_mode
    segment_granularity = base_plan.segment_granularity
    horizon_steps = base_plan.segment_horizon_steps
    caution_flag = base_plan.caution_flag
    should_probe = base_plan.should_probe
    if family == "obstacle_clearance" and next_phase == "free_space_approach":
        next_phase = "obstacle_avoidance"
        if waypoints:
            shifted = []
            for waypoint in waypoints:
                shifted.append(
                    TeacherWaypoint(
                        position_xyz=(
                            float(waypoint.position_xyz[0]),
                            float(waypoint.position_xyz[1] + 0.02),
                            float(waypoint.position_xyz[2] + 0.01),
                        ),
                        yaw=waypoint.yaw + 0.08,
                        speed_scale=min(waypoint.speed_scale, 0.75),
                        clearance_hint=max(waypoint.clearance_hint, 0.08),
                    )
                )
            waypoints = shifted
    elif family == "alignment_first" and (
        phase_guidance.get("recommended_phase") == "pre_insert_align"
        or phase_guidance.get("in_insertion_zone", False)
    ):
        next_phase = "pre_insert_align"
        motion_mode = "fine_cartesian"
        segment_granularity = "fine"
        horizon_steps = min(horizon_steps, 6)
        waypoints = [
            TeacherWaypoint(
                position_xyz=(
                    float(entrance[0]),
                    float(entrance[1]),
                    float(entrance[2] + 0.01),
                ),
                yaw=float(target[5]),
                speed_scale=0.45,
                clearance_hint=0.02,
            )
        ]
    elif family == "guarded_insert" and (
        phase_guidance.get("insertion_ready", False)
        or (
            phase_guidance.get("in_insertion_zone", False)
            and not phase_guidance.get("is_far_from_target", False)
        )
    ):
        next_phase = "guarded_insert"
        motion_mode = "guarded_insert"
        segment_granularity = "guarded"
        horizon_steps = min(horizon_steps, 4)
        waypoints = [
            TeacherWaypoint(
                position_xyz=(
                    float(entrance[0]),
                    float(entrance[1]),
                    float(entrance[2]),
                ),
                yaw=float(target[5]),
                speed_scale=0.3,
                clearance_hint=0.01,
            ),
            TeacherWaypoint(
                position_xyz=(
                    float(target[0]),
                    float(target[1]),
                    float(target[2]),
                ),
                yaw=float(target[5]),
                speed_scale=0.25,
                clearance_hint=0.005,
            ),
        ]
    elif family == "recovery_backoff" and (
        phase_guidance.get("hidden_contact_recent", False)
        or phase_guidance.get("off_limit_contact", False)
        or phase_guidance.get("stuck_without_progress", False)
    ):
        next_phase = "backoff_and_retry"
        motion_mode = "fine_cartesian"
        segment_granularity = "fine"
        caution_flag = True
        should_probe = True
        waypoints = [
            TeacherWaypoint(
                position_xyz=(
                    float(policy["plug_pose"][0] - 0.02),
                    float(policy["plug_pose"][1]),
                    float(policy["plug_pose"][2] + 0.02),
                ),
                yaw=float(target[5]),
                speed_scale=0.4,
                clearance_hint=0.06,
            )
        ]
    return TeacherPlan(
        next_phase=next_phase,
        waypoints=tuple(waypoints),
        motion_mode=motion_mode,
        caution_flag=caution_flag,
        should_probe=should_probe,
        segment_horizon_steps=horizon_steps,
        segment_granularity=segment_granularity,
        rationale_summary=f"{base_plan.rationale_summary}; family={family}",
    )


def _apply_refinement_style(*, base_plan: TeacherPlan, state, spec: CandidateSpec) -> TeacherPlan:
    if spec.refinement_style is None:
        return base_plan
    policy = state.policy_context
    entrance = policy.get("target_port_entrance_pose") or policy.get("target_port_pose")
    target = policy.get("target_port_pose")
    waypoints = list(base_plan.waypoints)
    next_phase = base_plan.next_phase
    motion_mode = base_plan.motion_mode
    segment_granularity = base_plan.segment_granularity
    horizon_steps = base_plan.segment_horizon_steps
    caution_flag = base_plan.caution_flag
    should_probe = base_plan.should_probe
    style = spec.refinement_style
    if style == "progress_push" and waypoints:
        refined = []
        for waypoint in waypoints:
            position = np.asarray(waypoint.position_xyz, dtype=np.float64)
            target_vector = np.asarray(target[:3], dtype=np.float64) - position
            norm = float(np.linalg.norm(target_vector))
            direction = np.zeros(3, dtype=np.float64) if norm <= 1e-6 else target_vector / norm
            pushed = position + 0.018 * direction
            refined.append(
                TeacherWaypoint(
                    position_xyz=tuple(float(value) for value in pushed),
                    yaw=float(waypoint.yaw),
                    speed_scale=min(max(float(waypoint.speed_scale) + 0.15, 0.25), 0.9),
                    clearance_hint=max(float(waypoint.clearance_hint), 0.04),
                )
            )
        waypoints = refined
    elif style == "clearance_arc" and waypoints:
        refined = []
        for waypoint in waypoints:
            refined.append(
                TeacherWaypoint(
                    position_xyz=(
                        float(waypoint.position_xyz[0]),
                        float(waypoint.position_xyz[1] + 0.015),
                        float(waypoint.position_xyz[2] + 0.008),
                    ),
                    yaw=float(waypoint.yaw + 0.06),
                    speed_scale=min(float(waypoint.speed_scale), 0.7),
                    clearance_hint=max(float(waypoint.clearance_hint), 0.08),
                )
            )
        waypoints = refined
        next_phase = "obstacle_avoidance"
    elif style == "micro_align":
        next_phase = "pre_insert_align"
        motion_mode = "fine_cartesian"
        segment_granularity = "fine"
        horizon_steps = min(horizon_steps, 5)
        waypoints = [
            TeacherWaypoint(
                position_xyz=(
                    float(entrance[0]),
                    float(entrance[1]),
                    float(entrance[2] + 0.008),
                ),
                yaw=float(target[5]),
                speed_scale=0.35,
                clearance_hint=0.015,
            ),
            TeacherWaypoint(
                position_xyz=(
                    float(entrance[0]),
                    float(entrance[1]),
                    float(entrance[2] + 0.002),
                ),
                yaw=float(target[5]),
                speed_scale=0.25,
                clearance_hint=0.01,
            ),
        ]
    elif style == "guarded_probe":
        next_phase = "guarded_insert"
        motion_mode = "guarded_insert"
        segment_granularity = "guarded"
        horizon_steps = min(horizon_steps, 4)
        waypoints = [
            TeacherWaypoint(
                position_xyz=(float(entrance[0]), float(entrance[1]), float(entrance[2])),
                yaw=float(target[5]),
                speed_scale=0.22,
                clearance_hint=0.01,
            ),
            TeacherWaypoint(
                position_xyz=(float(target[0]), float(target[1]), float(target[2])),
                yaw=float(target[5]),
                speed_scale=0.18,
                clearance_hint=0.005,
            ),
        ]
    elif style == "backoff_probe":
        next_phase = "backoff_and_retry"
        motion_mode = "fine_cartesian"
        segment_granularity = "fine"
        horizon_steps = min(horizon_steps, 5)
        caution_flag = True
        should_probe = True
        plug = policy.get("plug_pose", [0.0, 0.0, 0.0])
        waypoints = [
            TeacherWaypoint(
                position_xyz=(
                    float(plug[0] - 0.02),
                    float(plug[1]),
                    float(plug[2] + 0.02),
                ),
                yaw=float(target[5]),
                speed_scale=0.35,
                clearance_hint=0.05,
            )
        ]
    return TeacherPlan(
        next_phase=next_phase,
        waypoints=tuple(waypoints),
        motion_mode=motion_mode,
        caution_flag=caution_flag,
        should_probe=should_probe,
        segment_horizon_steps=horizon_steps,
        segment_granularity=segment_granularity,
        rationale_summary=f"{base_plan.rationale_summary}; refinement_style={style}",
    )


def _candidate_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    phase_similarity = 1.0 if left["phase_sequence"] == right["phase_sequence"] else 0.0
    waypoint_distance = _signature_distance(
        left.get("waypoint_signature", []),
        right.get("waypoint_signature", []),
    )
    return 0.6 * phase_similarity + 0.4 * (1.0 / (1.0 + waypoint_distance))


def _candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    artifact = candidate.get("artifact", {})
    segments = list(artifact.get("trajectory_segments", []))
    phases = [segment.get("phase") for segment in segments if segment.get("phase")]
    waypoint_signature: list[list[float]] = []
    for segment in segments[:3]:
        points = list(segment.get("points", []))
        if points:
            pose = points[-1].get("target_tcp_pose", [])[:3]
            waypoint_signature.append([float(value) for value in pose])
    return {
        "name": candidate["candidate_spec"]["name"],
        "mode": candidate["candidate_spec"]["mode"],
        "phase_sequence": _collapse_adjacent_duplicates(phases),
        "waypoint_signature": waypoint_signature,
    }


def _collapse_adjacent_duplicates(items: list[str]) -> list[str]:
    collapsed: list[str] = []
    for item in items:
        if not collapsed or collapsed[-1] != item:
            collapsed.append(item)
    return collapsed


def _signature_distance(left: list[list[float]], right: list[list[float]]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    count = min(len(left), len(right))
    total = 0.0
    for index in range(count):
        total += sum((float(a) - float(b)) ** 2 for a, b in zip(left[index], right[index])) ** 0.5
    return total / count


def _select_refinement_seeds(*, planner_ranked: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    for entry in planner_ranked:
        if not seeds:
            seeds.append(entry)
            continue
        if len(seeds) >= limit:
            break
        signature = _candidate_signature(entry)
        if any(
            _candidate_similarity(signature, _candidate_signature(existing)) >= 0.9
            for existing in seeds
        ):
            continue
        seeds.append(entry)
    for entry in planner_ranked:
        if len(seeds) >= limit:
            break
        if entry not in seeds:
            seeds.append(entry)
    return seeds[:limit]


def _refinement_style_for_seed(*, seed: dict[str, Any], slot: int) -> str | None:
    artifact = seed.get("artifact", {})
    phases = [
        segment.get("phase")
        for segment in artifact.get("trajectory_segments", [])
        if segment.get("phase")
    ]
    last_phase = phases[-1] if phases else None
    final_distance = artifact.get("final_info", {}).get("distance_to_target")
    family = seed.get("candidate_spec", {}).get("family")
    if last_phase == "guarded_insert":
        return "guarded_probe"
    if last_phase == "pre_insert_align":
        return "micro_align"
    if last_phase == "backoff_and_retry":
        return "backoff_probe"
    if last_phase == "obstacle_avoidance":
        return "progress_push" if slot == 0 else "clearance_arc"
    if family == "alignment_first" and final_distance is not None and float(final_distance) <= 0.75:
        return "micro_align"
    if family == "recovery_backoff":
        return "backoff_probe"
    return "progress_push"


def _refinement_offsets(*, style: str) -> tuple[float, float, float, float]:
    offsets = {
        "progress_push": (0.012, 0.0, -0.008, 0.0),
        "clearance_arc": (0.0, 0.012, 0.006, 0.08),
        "micro_align": (0.004, 0.0, -0.004, 0.0),
        "guarded_probe": (-0.003, 0.0, 0.0, 0.0),
        "backoff_probe": (-0.012, 0.0, 0.012, -0.05),
    }
    return offsets.get(style, (0.0, 0.0, 0.0, 0.0))
