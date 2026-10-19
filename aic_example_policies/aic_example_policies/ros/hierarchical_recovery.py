"""Observation-only, measured-pose recovery supervisor for cable insertion.

The supervisor decides when to retrace; the caller supplies an observation-
based route choice after a route-level retreat. It never reads simulator poses
of the cable, plug, port, or cards. This module is simulator-independent and
contains no ROS dependency so the same decisions can be used in Isaac and
Gazebo. Motion commands remain the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, sin, sqrt, isfinite


@dataclass(frozen=True)
class MeasuredPose:
    xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class RecoveryConfig:
    trigger_force_n: float = 8.0
    clear_force_n: float = 4.0
    trigger_steps: int = 2
    min_command_m: float = 0.00005
    max_realized_m: float = 0.00015
    history_spacing_m: float = 0.001
    history_spacing_deg: float = 2.0
    local_retreat_m: float = 0.012
    route_retreat_m: float = 0.060
    same_contact_radius_m: float = 0.015
    local_attempts_before_route: int = 2
    max_total_backoffs: int = 5
    max_route_resets: int = 2
    max_command_step_m: float = 0.003
    max_command_rotation_deg: float = 4.0
    force_rise_abort_n: float = 6.0
    waypoint_tolerance_m: float = 0.0015
    waypoint_tolerance_deg: float = 2.0
    max_history: int = 1000

    def validate(self) -> None:
        if not (0 < self.clear_force_n < self.trigger_force_n):
            raise ValueError("clear force must be positive and below trigger")
        if not (0 < self.local_retreat_m < self.route_retreat_m):
            raise ValueError("route retreat must exceed local retreat")
        if self.trigger_steps < 1 or self.local_attempts_before_route < 1:
            raise ValueError("trigger and retry counts must be positive")
        if self.max_total_backoffs < 1 or self.max_route_resets < 1:
            raise ValueError("recovery budgets must be positive")
        if self.max_history < 3:
            raise ValueError("history must retain at least three poses")
        if self.history_spacing_deg <= 0 or self.waypoint_tolerance_deg <= 0:
            raise ValueError("rotation history spacing and tolerance must be positive")


@dataclass(frozen=True)
class RecoveryDecision:
    mode: str
    actor_owned: bool
    target: MeasuredPose | None
    level: str | None
    route_change_required: bool
    forbidden_route: str | None
    local_attempts: int
    total_backoffs: int
    route_resets: int
    reason: str | None = None


def _distance(a: MeasuredPose, b: MeasuredPose) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a.xyz, b.xyz)))


def _unit_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    n = sqrt(sum(v * v for v in q))
    if n < 1e-8:
        raise ValueError("measured orientation has zero norm")
    return tuple(v / n for v in q)  # type: ignore[return-value]


def _angle_deg(a: MeasuredPose, b: MeasuredPose) -> float:
    dot = abs(sum(x * y for x, y in zip(_unit_quat(a.quat_wxyz), _unit_quat(b.quat_wxyz))))
    return 2 * acos(min(1.0, dot)) * 180.0 / 3.141592653589793


def _slerp(a: tuple[float, float, float, float], b: tuple[float, float, float, float], fraction: float) -> tuple[float, float, float, float]:
    qa, qb = _unit_quat(a), _unit_quat(b)
    dot = sum(x * y for x, y in zip(qa, qb))
    if dot < 0:
        qb = tuple(-x for x in qb)
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _unit_quat(tuple(x + fraction * (y - x) for x, y in zip(qa, qb)))
    angle = acos(dot)
    scale = sin(angle)
    wa, wb = sin((1 - fraction) * angle) / scale, sin(fraction * angle) / scale
    return _unit_quat(tuple(wa * x + wb * y for x, y in zip(qa, qb)))


class HierarchicalRecovery:
    """Per-episode recovery state with local and route-level checkpoints."""

    def __init__(self, config: RecoveryConfig = RecoveryConfig()):
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.mode = "policy"
        self.level: str | None = None
        self.history: list[tuple[MeasuredPose, float, str, str]] = []
        self.path: list[MeasuredPose] = []
        self.path_index = 0
        self.anchor: MeasuredPose | None = None
        self.previous_pose: MeasuredPose | None = None
        self.previous_command_m = 0.0
        self.previous_force_n = 0.0
        self.retreat_force_floor_n: float | None = None
        self.trigger_count = 0
        self.last_contact_pose: MeasuredPose | None = None
        self.last_contact_route: str | None = None
        self.local_attempts = 0
        self.total_backoffs = 0
        self.route_resets = 0
        self.failed_route: str | None = None
        self.reason: str | None = None
        self.anchor_history_index: int | None = None
        self.retry_checkpoint_indices: list[int] = []

    def _append_history(self, pose: MeasuredPose, force_n: float, phase: str, route: str) -> None:
        if (not self.history or _distance(self.history[-1][0], pose) >= self.config.history_spacing_m
                or _angle_deg(self.history[-1][0], pose) >= self.config.history_spacing_deg
                or phase != self.history[-1][2]):
            self.history.append((pose, force_n, phase, route))
            if len(self.history) > self.config.max_history:
                self.history.pop(0)
                self.retry_checkpoint_indices=[i-1 for i in self.retry_checkpoint_indices if i>0]

    def _select_path(self, pose: MeasuredPose, level: str, blocked_phase: str) -> tuple[list[MeasuredPose], MeasuredPose] | None:
        required = self.config.local_retreat_m if level == "local" else self.config.route_retreat_m
        candidates = list(reversed(list(enumerate(self.history))))
        if not candidates:
            return None
        traveled = 0.0
        previous = pose
        path = []
        for index,(old_pose, old_force, old_phase, _route) in candidates:
            traveled += _distance(previous, old_pose)
            previous = old_pose
            if self._waypoint_reached(pose, old_pose):
                continue
            path.append(old_pose)
            # A route-level anchor must be outside the blocked corridor in
            # traveled distance. A low-force older point is mandatory at both
            # levels so a stalled path cannot be mistaken for a safe retreat.
            route_anchor_is_distinct = (
                level != "route"
                or _distance(pose, old_pose) >= self.config.route_retreat_m
                or old_phase != blocked_phase
            )
            # A deeper retry must precede the already tried checkpoints.
            # Counting the path around repeated B->C->B loops is not a fresh
            # route restart, even if it accumulates enough travelled length.
            before_previous_retry=(level!='route' or not self.retry_checkpoint_indices
                                   or index<min(self.retry_checkpoint_indices))
            if traveled >= required and old_force <= self.config.clear_force_n and route_anchor_is_distinct and before_previous_retry:
                self.anchor_history_index=index
                return path, old_pose
        return None

    def _waypoint_reached(self, pose: MeasuredPose, target: MeasuredPose) -> bool:
        return (_distance(pose, target) <= self.config.waypoint_tolerance_m
                and _angle_deg(pose, target) <= self.config.waypoint_tolerance_deg)

    def extend_route_to_acquisition(self) -> bool:
        """Return through measured poses to the initial free acquisition hold.

        The deployment caller seeds that checkpoint before actor commands.
        Never replace a missing checkpoint with a synthetic straight retreat.
        """
        if self.mode!='backtrack' or self.level!='route':
            raise ValueError('Only an active route-level retreat can be extended')
        if (not self.history or self.history[0][2]!='acquisition_checkpoint'
                or self.history[0][1]>self.config.clear_force_n):
            self.mode='abort';self.reason='no_free_acquisition_checkpoint'
            return False
        self.path=[item[0] for item in reversed(self.history)]
        self.path_index=0;self.anchor=self.history[0][0]
        self.anchor_history_index=0
        return True

    def confirm_new_corridor(self,route: str,measured_lateral_m: float) -> str:
        """Require actual lateral motion while preserving episode budgets.

        Earlier checkpoints remain mandatory on the same failed corridor.
        A genuinely different corridor can reuse the root acquisition hold.
        """
        if (self.mode!='retry' or self.level!='route' or route==self.failed_route
                or not isfinite(measured_lateral_m) or measured_lateral_m<.003):
            raise ValueError('A new route requires completed retreat and measured lateral motion')
        self.retry_checkpoint_indices=[]
        self.local_attempts=0;self.trigger_count=0
        self.last_contact_pose=None;self.last_contact_route=None
        self.failed_route=None;self.mode='policy';self.level=None
        self.retreat_force_floor_n=None
        return route

    def _trigger(self, pose: MeasuredPose, force_n: float, phase: str, route: str) -> None:
        # A second blockage farther along the same failed corridor must not
        # erase its retry count simply because C is >15mm away from A.
        same = self.last_contact_pose is not None and route == self.last_contact_route
        self.local_attempts = self.local_attempts + 1 if same else 1
        self.last_contact_pose = pose
        self.last_contact_route = route
        self.total_backoffs += 1
        if self.total_backoffs > self.config.max_total_backoffs:
            self.mode, self.reason = "abort", "backoff_budget_exceeded"
            return
        level = "route" if self.local_attempts > self.config.local_attempts_before_route else "local"
        if level == "route":
            self.route_resets += 1
            if self.route_resets > self.config.max_route_resets:
                self.mode, self.reason = "abort", "route_budget_exceeded"
                return
        selected = self._select_path(pose, level, phase)
        if selected is None:
            self.mode, self.reason = "abort", "no_verified_safe_checkpoint"
            return
        self.path, self.anchor = selected
        self.path_index = 0
        self.level = level
        self.failed_route = route if level == "route" else None
        self.mode = "backtrack"
        self.previous_force_n = force_n
        self.retreat_force_floor_n = force_n
        self.trigger_count = 0

    def _bounded_target(self, pose: MeasuredPose, target: MeasuredPose) -> MeasuredPose:
        distance = _distance(pose, target)
        fraction = min(1.0, self.config.max_command_step_m / max(distance, 1e-12))
        dot = abs(sum(x * y for x, y in zip(_unit_quat(pose.quat_wxyz), _unit_quat(target.quat_wxyz))))
        angle_deg = 2 * acos(min(1.0, dot)) * 180.0 / 3.141592653589793
        fraction = min(fraction, self.config.max_command_rotation_deg / max(angle_deg, 1e-12))
        xyz = tuple(x + fraction * (y - x) for x, y in zip(pose.xyz, target.xyz))
        return MeasuredPose(xyz, _slerp(pose.quat_wxyz, target.quat_wxyz, fraction))

    def step(self, *, pose: MeasuredPose, force_n: float, prior_command_m: float,
             phase: str, route: str, tracking_error_m: float = 0.0) -> RecoveryDecision:
        """Process one measured observation and the previous executed command.

        `force_n` must be tared or baseline-corrected. `prior_command_m` is
        translation requested during the preceding interval, not the new
        actor proposal. `tracking_error_m` allows a held absolute target to
        count as continuing demand in asynchronous Gazebo control.
        """
        if self.mode in {"policy", "retry"}:
            realized = 0.0 if self.previous_pose is None else _distance(self.previous_pose, pose)
            stalled = (
                self.previous_pose is not None
                and max(prior_command_m, tracking_error_m) >= self.config.min_command_m
                and realized <= self.config.max_realized_m
                and force_n >= self.config.trigger_force_n
            )
            self.trigger_count = self.trigger_count + 1 if stalled else 0
            self._append_history(pose, force_n, phase, route)
            if self.trigger_count >= self.config.trigger_steps:
                self._trigger(pose, force_n, phase, route)

        target = None
        if self.mode == "backtrack":
            if self.retreat_force_floor_n is None:
                self.retreat_force_floor_n=self.previous_force_n
            self.retreat_force_floor_n=min(self.retreat_force_floor_n,force_n)
            # Compare with the lowest force on this retreat, not only the
            # previous tick: a gradual force rise is still a renewed load.
            if force_n > self.retreat_force_floor_n + self.config.force_rise_abort_n:
                self.mode, self.reason = "abort", "force_rising_during_retreat"
            else:
                while self.path_index < len(self.path) - 1 and self._waypoint_reached(pose, self.path[self.path_index]):
                    self.path_index += 1
                if self.anchor is not None and self._waypoint_reached(pose, self.anchor):
                    if force_n <= self.config.clear_force_n:
                        if self.anchor_history_index is None:
                            raise RuntimeError('Recovery anchor has no measured history index')
                        # Keep a history of the current route, not all failed
                        # branches. Full event/replay logs remain the caller's
                        # responsibility and must never be truncated here.
                        self.history=self.history[:self.anchor_history_index+1]
                        self.retry_checkpoint_indices=[i for i in self.retry_checkpoint_indices
                                                       if i<=self.anchor_history_index]
                        self.retry_checkpoint_indices.append(self.anchor_history_index)
                        self.mode = "retry"
                    else:
                        self.mode, self.reason = "abort", "force_not_released_at_anchor"
                elif self.path_index < len(self.path):
                    target = self._bounded_target(pose, self.path[self.path_index])
                else:
                    self.mode, self.reason = "abort", "retrace_path_exhausted"

        self.previous_pose = pose
        self.previous_command_m = prior_command_m
        self.previous_force_n = force_n
        route_change = self.mode == "retry" and self.level == "route"
        if route_change and route == self.failed_route:
            # Caller must select another observation-supported corridor before
            # actor motion resumes; waiting is safer than repeating the path.
            target = pose
        actor_owned = self.mode in {"policy", "retry"} and not (route_change and route == self.failed_route)
        return RecoveryDecision(
            mode=self.mode, actor_owned=actor_owned, target=target,
            level=self.level, route_change_required=route_change,
            forbidden_route=self.failed_route if route_change else None,
            local_attempts=self.local_attempts, total_backoffs=self.total_backoffs,
            route_resets=self.route_resets, reason=self.reason,
        )
