"""MoveIt planner wrapper.

MoveIt is intentionally required for this generator. There is no geometric
fallback because dataset quality depends on verifying that rigid obstacle-aware
planning is actually being used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any

from aic_teacher_official.expert_generator.candidate_generation import ApproachCandidate
from aic_teacher_official.expert_generator.scene_snapshot import ObjectGeometry, SceneSnapshot
from aic_teacher_official.expert_generator.vlm_strategy import VLMStrategy
from aic_teacher_official.trajectory import PhaseLabel, SourceLabel, TCPPose, TrajectoryWaypoint


class MoveItUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanningFailure:
    reason: str
    recoverable: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": False,
            "reason": self.reason,
            "recoverable": self.recoverable,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class PlannedApproach:
    candidate_name: str
    waypoints: list[TrajectoryWaypoint]
    planning_scene_objects: list[ObjectGeometry]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "candidate_name": self.candidate_name,
            "waypoints": [wp.to_dict() for wp in self.waypoints],
            "planning_scene_objects": [obj.to_dict() for obj in self.planning_scene_objects],
            "metadata": dict(self.metadata),
        }


class MoveItPlanner:
    """Thin abstraction around the real MoveIt integration.

    Unit tests can inject ``backend``. Production must use an actual MoveIt
    backend; if none is importable this class fails loudly.
    """

    def __init__(
        self,
        *,
        planning_group: str = "arm",
        required: bool = True,
        backend: Any | None = None,
        import_names: tuple[str, ...] = ("moveit_commander", "moveit.planning", "pymoveit2"),
    ):
        self.planning_group = planning_group
        self.required = required
        self.backend = backend
        self.import_names = import_names
        self.available_backend_name = self._detect_backend_name() if backend is None else "injected"
        if required and backend is None and self.available_backend_name is None:
            raise MoveItUnavailableError(
                "MoveIt is required for expert generation, but no supported Python MoveIt "
                f"module was found. Checked: {', '.join(import_names)}. No geometric fallback is available."
            )

    def _detect_backend_name(self) -> str | None:
        for name in self.import_names:
            try:
                if find_spec(name) is not None:
                    return name
            except ModuleNotFoundError:
                continue
        return None

    def build_planning_scene(
        self,
        snapshot: SceneSnapshot,
        strategy: VLMStrategy,
    ) -> list[ObjectGeometry]:
        objects = list(snapshot.collision_objects)
        for region in strategy.avoid_regions:
            if snapshot.target_port_pose is None:
                continue
            objects.append(
                ObjectGeometry(
                    name=f"vlm_keepout_{region}",
                    pose=snapshot.target_port_pose,
                    shape="box",
                    dimensions=[0.12, 0.12, 0.12],
                    role="vlm_avoid_region",
                    metadata={
                        "source": "vlm_strategy",
                        "region": region,
                        "inflated": True,
                    },
                )
            )
        return objects

    def plan_free_space_approach(
        self,
        scene_snapshot: SceneSnapshot,
        strategy: VLMStrategy,
        candidate_pose: ApproachCandidate,
    ) -> PlannedApproach | PlanningFailure:
        if self.backend is not None:
            return self.backend.plan_free_space_approach(scene_snapshot, strategy, candidate_pose)
        if self.available_backend_name is None:
            return PlanningFailure(
                reason="moveit_unavailable",
                recoverable=False,
                details={"message": "MoveIt required; no fallback planner is implemented."},
            )
        return PlanningFailure(
            reason="moveit_backend_adapter_not_configured",
            recoverable=False,
            details={
                "backend": self.available_backend_name,
                "message": "A ROS MoveIt adapter must be provided for live planning in this environment.",
            },
        )


def waypoints_from_candidate(candidate: ApproachCandidate, *, start_time: float = 0.0) -> list[TrajectoryWaypoint]:
    """Helper for injected test backends that emulate a successful MoveIt plan."""

    return [
        TrajectoryWaypoint(
            timestamp=start_time,
            tcp_pose=TCPPose(
                position=list(candidate.safe_lift_pose.position),
                orientation_xyzw=list(candidate.safe_lift_pose.orientation_xyzw),
            ),
            phase=PhaseLabel.APPROACH,
            source=SourceLabel.OPTIMIZER,
            diagnostics={"planner": "moveit", "candidate": candidate.name},
        ),
        TrajectoryWaypoint(
            timestamp=start_time + 2.0,
            tcp_pose=TCPPose(
                position=list(candidate.approach_standoff_pose.position),
                orientation_xyzw=list(candidate.approach_standoff_pose.orientation_xyzw),
            ),
            phase=PhaseLabel.ALIGNMENT,
            source=SourceLabel.OPTIMIZER,
            diagnostics={"planner": "moveit", "candidate": candidate.name},
        ),
        TrajectoryWaypoint(
            timestamp=start_time + 4.0,
            tcp_pose=TCPPose(
                position=list(candidate.pre_insert_pose.position),
                orientation_xyzw=list(candidate.pre_insert_pose.orientation_xyzw),
            ),
            phase=PhaseLabel.PRE_INSERTION,
            source=SourceLabel.OPTIMIZER,
            diagnostics={"planner": "moveit", "candidate": candidate.name},
        ),
    ]
