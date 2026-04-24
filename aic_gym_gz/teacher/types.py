"""Dataclasses shared by the agent-teacher stack."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TeacherPhase = Literal[
    "free_space_approach",
    "obstacle_avoidance",
    "cable_probe",
    "pre_insert_align",
    "guarded_insert",
    "backoff_and_retry",
]


MotionMode = Literal["coarse_cartesian", "fine_cartesian", "guarded_insert", "hold"]


@dataclass(frozen=True)
class DynamicsSummary:
    plug_oscillation_magnitude: float
    cable_settling_score: float
    recent_motion_energy: float
    quasi_static: bool
    time_since_last_significant_cable_motion: float
    wrench_energy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeObservationSummary:
    sim_time: float
    tcp_position_xyz: tuple[float, float, float]
    plug_position_xyz: tuple[float, float, float]
    wrench_force_xyz: tuple[float, float, float]
    contact: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    probe_name: str
    before: ProbeObservationSummary
    after: ProbeObservationSummary
    duration_s: float
    action_count: int
    plug_relative_motion: float
    peak_force: float
    settling_delta: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObstacleSummary:
    object_name: str
    clearance_hint: float
    present: bool
    pose_hint: tuple[float, float, float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherPlanningState:
    trial_id: str
    task_id: str
    goal_summary: str
    task_definition: dict[str, Any]
    current_phase: TeacherPhase
    policy_context: dict[str, Any]
    oracle_context: dict[str, Any]
    obstacle_summary: list[dict[str, Any]]
    dynamics_summary: dict[str, Any]
    image_refs: list[str]
    image_timestamps: dict[str, float]
    image_summaries: dict[str, dict[str, Any]]
    recent_probe_results: list[dict[str, Any]]
    recent_visual_observations: list[dict[str, Any]] = field(default_factory=list)
    scene_overview_images: list[dict[str, Any]] = field(default_factory=list)
    controller_context: dict[str, Any] = field(default_factory=dict)
    camera_context: dict[str, Any] = field(default_factory=dict)
    temporal_context: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, dict[str, Any]] = field(default_factory=dict)
    planning_metadata: dict[str, Any] = field(default_factory=dict)
    last_teacher_rationale: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["recent_visual_observations"] = [
            {
                "label": item.get("label"),
                "camera_name": item.get("camera_name"),
                "sim_tick": item.get("sim_tick"),
                "sim_time": item.get("sim_time"),
                "timestamp": item.get("timestamp"),
                "age_from_latest_s": item.get("age_from_latest_s"),
                "age_from_latest_steps": item.get("age_from_latest_steps"),
                "timepoint_label": item.get("timepoint_label"),
                "source": item.get("source"),
            }
            for item in self.recent_visual_observations
        ]
        payload["scene_overview_images"] = [
            {
                "label": item.get("label"),
                "view_name": item.get("view_name"),
                "timestamp": item.get("timestamp"),
                "source": item.get("source"),
            }
            for item in self.scene_overview_images
        ]
        return payload


@dataclass(frozen=True)
class TeacherWaypoint:
    position_xyz: tuple[float, float, float]
    yaw: float = 0.0
    speed_scale: float = 1.0
    clearance_hint: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherPlan:
    next_phase: TeacherPhase
    waypoints: tuple[TeacherWaypoint, ...]
    motion_mode: MotionMode
    caution_flag: bool
    should_probe: bool
    segment_horizon_steps: int
    segment_granularity: Literal["coarse", "fine", "guarded"]
    rationale_summary: str
    decision_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["waypoints"] = [waypoint.to_dict() for waypoint in self.waypoints]
        return payload


@dataclass(frozen=True)
class DenseTrajectoryPoint:
    dt: float
    action: tuple[float, float, float, float, float, float]
    target_tcp_pose: tuple[float, float, float, float, float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectorySegment:
    phase: TeacherPhase
    motion_mode: MotionMode
    rationale_summary: str
    points: tuple[DenseTrajectoryPoint, ...]
    expected_duration_s: float
    conversion_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["points"] = [point.to_dict() for point in self.points]
        return payload


@dataclass(frozen=True)
class CandidateEvaluation:
    name: str
    score: float
    plan: dict[str, Any]
    segment: dict[str, Any]
    mode: str = "planner_waypoint"
    notes: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeacherStepLog:
    phase: TeacherPhase
    sim_time: float
    sim_tick: int
    reward: float
    terminated: bool
    truncated: bool
    planner_rationale: str
    trajectory_point: dict[str, Any]
    dynamics_summary: dict[str, Any]
    observation_summary: dict[str, Any]
    history_summary: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    auxiliary_force_contact_summary: dict[str, Any] = field(default_factory=dict)
    auxiliary_summary_available: bool = False
    auxiliary_contact_metrics: dict[str, Any] = field(default_factory=dict)
    probe_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TeacherRolloutLog:
    trial_id: str
    task_id: str
    teacher_version: str
    planner_backend: str
    seed: int | None
    scenario_metadata: dict[str, Any]
    timing: dict[str, Any]
    initial_observation_summary: dict[str, Any]
    data_quality: dict[str, Any] = field(default_factory=dict)
    history_metadata: dict[str, Any] = field(default_factory=dict)
    auxiliary_summary_metadata: dict[str, Any] = field(default_factory=dict)
    planner_candidates: list[dict[str, Any]] = field(default_factory=list)
    probe_results: list[dict[str, Any]] = field(default_factory=list)
    trajectory_segments: list[dict[str, Any]] = field(default_factory=list)
    step_logs: list[dict[str, Any]] = field(default_factory=list)
    final_info: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
