"""Backend abstraction and temporary stub bridge for runtime communication."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
import random

from .action import Action
from .observation import Observation
from .protocol import (
    GetObservationRequest,
    GetObservationResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from .reward import Reward
from .task import MinimalTask, MinimalTaskState
from .termination import Termination


class Backend(ABC):
    """Abstract transport-neutral backend interface used by runtimes."""

    @abstractmethod
    def reset(self, request: ResetRequest) -> ResetResponse:
        """Handle a reset request."""

    @abstractmethod
    def step(self, request: StepRequest) -> StepResponse:
        """Handle a step request."""

    @abstractmethod
    def get_observation(
        self,
        request: GetObservationRequest,
    ) -> GetObservationResponse:
        """Handle an observation request."""


@dataclass
class StubBackend(Backend):
    """Temporary in-process backend that mimics a real request/response bridge.

    The backend deliberately serializes requests to dicts and deserializes
    responses back into protocol objects so tests exercise a true roundtrip
    through the protocol layer even before any Gazebo transport exists.
    """

    step_count: int = 0
    last_seed: int | None = None
    last_options: dict[str, object] = field(default_factory=dict)
    last_action: dict[str, object] = field(default_factory=dict)
    joint_count: int = 6
    max_abs_joint_delta: float = 0.25
    substeps_per_step: int = 4
    substep_dt: float = 0.05
    reset_settling_steps: int = 2
    observation_mutator: Callable[[dict[str, object]], dict[str, object]] | None = None
    sim_time: float = 0.0
    sim_time_offset: float = 0.0
    joint_positions_state: list[float] = field(default_factory=list)
    joint_velocities_state: list[float] = field(default_factory=list)
    task_object_pose_state: dict[str, list[float]] = field(default_factory=dict)
    entity_ids: list[str] = field(default_factory=list)
    reward_model: Reward = field(default_factory=Reward)
    termination_model: Termination = field(default_factory=Termination)
    task: MinimalTask = field(default_factory=MinimalTask)
    task_state: MinimalTaskState | None = None

    def reset(self, request: ResetRequest) -> ResetResponse:
        """Reset stub backend state and return a protocol response."""
        decoded = ResetRequest.from_dict(request.to_dict())
        settling_steps = self._resolve_settling_steps(decoded.options)
        rng = random.Random(decoded.seed if decoded.seed is not None else 0)
        self.step_count = 0
        self.last_seed = decoded.seed
        self.last_options = dict(decoded.options)
        self.last_action = {}
        self.sim_time = 0.0
        self.sim_time_offset = 0.0
        if decoded.seed is None:
            self.joint_positions_state = [0.0] * self.joint_count
        else:
            self.joint_positions_state = [
                round(rng.uniform(-0.05, 0.05), 6) for _ in range(self.joint_count)
            ]
        self.joint_velocities_state = [0.0] * self.joint_count
        self.task_state = self.task.reset(seed=decoded.seed)
        self.task_object_pose_state = {
            "position": list(self.task_state.object_pose["position"]),
            "orientation": list(self.task_state.object_pose["orientation"]),
        }
        self.entity_ids = ["robot", "task_object"]
        self._apply_settling_steps(settling_steps)
        if decoded.seed is None:
            self.sim_time_offset = self.sim_time
        response = ResetResponse(
            observation=self._make_observation(),
            info={
                "backend": "stub",
                "runtime": "stub",
                "seed": decoded.seed,
                "options": dict(decoded.options),
                "settling_steps": settling_steps,
                "entity_count": len(self.entity_ids),
                "task_object_pose": self._task_object_pose_info(),
                "privileged_observation": self.task.privileged_observation(self.task_state),
            },
        )
        return ResetResponse.from_dict(response.to_dict())

    def step(self, request: StepRequest) -> StepResponse:
        """Advance stub backend state and return a protocol response."""
        decoded = StepRequest.from_dict(request.to_dict())
        action = Action.from_dict(
            decoded.action,
            expected_length=self.joint_count,
        ).clipped(max_abs_delta=self.max_abs_joint_delta)
        self.step_count += 1
        self.last_action = action.to_dict()
        per_substep = [
            delta / float(self.substeps_per_step)
            for delta in action.joint_position_delta
        ]
        self.joint_velocities_state = [
            delta / self.substep_dt for delta in per_substep
        ]
        for _ in range(self.substeps_per_step):
            self.joint_positions_state = [
                position + delta
                for position, delta in zip(
                    self.joint_positions_state,
                    per_substep,
                    strict=True,
                )
            ]
            self.sim_time += self.substep_dt
        ee_position = self._ee_position()
        if self.task_state is None:
            raise RuntimeError("Task state was not initialized before stepping.")
        self.task_state = self.task.advance(
            state=self.task_state,
            ee_position=ee_position,
        )
        self.task_object_pose_state = {
            "position": list(self.task_state.object_pose["position"]),
            "orientation": list(self.task_state.object_pose["orientation"]),
        }
        response = StepResponse(
            observation=self._make_observation(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={},
        )
        object_position = list(self.task_state.object_pose["position"])
        target_position = list(self.task_state.target_pose["position"])
        success = self.task.is_success(self.task_state)
        reward_position = ee_position if not self.task_state.grasped else object_position
        reward, distance_to_target = self.reward_model.compute(
            ee_position=reward_position,
            target_position=target_position,
            action_delta=action.joint_position_delta,
            success=success,
        )
        terminated, truncated, reason = self.termination_model.evaluate(
            step_count=self.step_count,
            distance_to_target=distance_to_target,
            ee_position=reward_position,
        )
        if reason == "success" and not success:
            reward, distance_to_target = self.reward_model.compute(
                ee_position=reward_position,
                target_position=target_position,
                action_delta=action.joint_position_delta,
                success=True,
            )
        response = StepResponse(
            observation=response.observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "backend": "stub",
                "runtime": "stub",
                "step_count": self.step_count,
                "applied_action": action.to_dict(),
                "substeps": self.substeps_per_step,
                "distance_to_target": distance_to_target,
                "termination_reason": reason,
                "target_position": target_position,
                "privileged_observation": self.task.privileged_observation(self.task_state),
            },
        )
        return StepResponse.from_dict(response.to_dict())

    def get_observation(
        self,
        request: GetObservationRequest,
    ) -> GetObservationResponse:
        """Return the latest stub observation through the protocol layer."""
        decoded = GetObservationRequest.from_dict(request.to_dict())
        del decoded
        if self.task_state is None:
            raise RuntimeError("Task state was not initialized before reading observation.")
        response = GetObservationResponse(
            observation=self._make_observation(),
            info={
                "backend": "stub",
                "runtime": "stub",
                "privileged_observation": self.task.privileged_observation(self.task_state),
            },
        )
        return GetObservationResponse.from_dict(response.to_dict())

    def _resolve_settling_steps(self, options: dict[str, object]) -> int:
        """Resolve the number of reset settling steps."""
        settling_steps = options.get("settling_steps", self.reset_settling_steps)
        if not isinstance(settling_steps, int) or settling_steps < 0:
            raise ValueError("Reset option 'settling_steps' must be a non-negative int.")
        return settling_steps

    def _apply_settling_steps(self, settling_steps: int) -> None:
        """Advance simulated time for post-reset settling without changing state."""
        for _ in range(settling_steps):
            self.sim_time += self.substep_dt

    def _make_observation(self) -> dict[str, object]:
        """Build the current stub observation payload."""
        raw_observation: dict[str, object] = {
            "step_count": self.step_count,
            "sim_time": round(self.sim_time - self.sim_time_offset, 10),
            "joint_positions": list(self.joint_positions_state),
            "joint_velocities": list(self.joint_velocities_state),
            "end_effector_pose": {
                "position": [
                    sum(self.joint_positions_state[:2]),
                    sum(self.joint_positions_state[2:4]),
                    0.5 + sum(self.joint_positions_state[4:6]),
                ],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
        }
        if self.observation_mutator is not None:
            raw_observation = self.observation_mutator(raw_observation)
        return Observation.from_dict(raw_observation).to_dict()

    def _ee_position(self) -> list[float]:
        """Return the current end-effector position implied by joint state."""
        return [
            sum(self.joint_positions_state[:2]),
            sum(self.joint_positions_state[2:4]),
            0.5 + sum(self.joint_positions_state[4:6]),
        ]

    def _task_object_pose_info(self) -> dict[str, list[float]]:
        """Return the current task object pose payload."""
        return {
            "position": list(self.task_object_pose_state["position"]),
            "orientation": list(self.task_object_pose_state["orientation"]),
        }
