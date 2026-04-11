"""Runtime and backend abstractions for synchronous AIC training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .scenario import AicScenario


@dataclass(frozen=True)
class RuntimeState:
    sim_tick: int
    sim_time: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    gripper_position: float
    tcp_pose: np.ndarray
    tcp_velocity: np.ndarray
    plug_pose: np.ndarray
    target_port_pose: np.ndarray
    wrench: np.ndarray
    off_limit_contact: bool
    insertion_event: str | None = None


class RuntimeBackend(ABC):
    """Backend contract for exact-tick stepping."""

    @abstractmethod
    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        """Reset the backend and return the initial simulator state."""

    @abstractmethod
    def apply_action(self, action: np.ndarray) -> None:
        """Store the latest command to be applied during stepping."""

    @abstractmethod
    def step_ticks(self, tick_count: int) -> RuntimeState:
        """Advance exactly ``tick_count`` simulator iterations."""

    @abstractmethod
    def close(self) -> None:
        """Release backend resources."""


@dataclass
class AicGazeboRuntime:
    """Synchronous exact-tick runtime wrapper."""

    backend: RuntimeBackend
    ticks_per_step: int = 8
    sim_dt: float = 0.002

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        return self.backend.reset(seed=seed, scenario=scenario)

    def step(self, action: np.ndarray, *, ticks: int | None = None) -> RuntimeState:
        if action.shape != (6,):
            raise ValueError(f"Expected action shape (6,), received {action.shape}.")
        self.backend.apply_action(action.astype(np.float64, copy=False))
        return self.backend.step_ticks(ticks if ticks is not None else self.ticks_per_step)

    def close(self) -> None:
        self.backend.close()


class ScenarioGymGzBackend(RuntimeBackend):
    """Placeholder for the intended ScenarIO + gym-gz backend."""

    def __init__(self, *, world_name: str = "aic_world") -> None:
        self._world_name = world_name
        try:
            __import__("scenario")
            __import__("gym_gz")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ScenarIO / gym-gz dependencies are not installed in this environment. "
                "Use MockStepperBackend for local tests, or install the simulator stack "
                "inside the eval container / training image."
            ) from exc

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        raise NotImplementedError(
            "Real ScenarIO + gym-gz integration is intentionally isolated here and "
            "requires runtime dependencies that are not available in this shell."
        )

    def apply_action(self, action: np.ndarray) -> None:
        raise NotImplementedError

    def step_ticks(self, tick_count: int) -> RuntimeState:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MockStepperBackend(RuntimeBackend):
    """Deterministic backend used for tests and the random-policy demo."""

    def __init__(self, *, sim_dt: float = 0.002) -> None:
        self._sim_dt = sim_dt
        self._rng = np.random.default_rng(0)
        self._state: RuntimeState | None = None
        self._action = np.zeros(6, dtype=np.float64)

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        task = next(iter(scenario.tasks.values()))
        board_x, board_y, board_z, _, _, board_yaw = scenario.task_board.pose_xyz_rpy
        target_pose = np.array(
            [
                board_x - 0.081418,
                board_y - 0.1745 + scenario.task_board.nic_rails["nic_rail_0"].translation,
                board_z + 0.012,
                0.0,
                0.0,
                board_yaw,
                1.0,
            ],
            dtype=np.float64,
        )
        tcp_pose = np.array([-0.45, 0.2, 1.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        cable = scenario.cables[task.cable_name]
        plug_pose = tcp_pose.copy()
        plug_pose[:3] += np.array(cable.gripper_offset_xyz, dtype=np.float64)
        self._action[:] = 0.0
        self._state = RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.array([-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110]),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0073 if cable.cable_type == "sfp_sc_cable" else 0.00655,
            tcp_pose=tcp_pose,
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=plug_pose,
            target_port_pose=target_pose,
            wrench=np.zeros(6, dtype=np.float64),
            off_limit_contact=False,
            insertion_event=None,
        )
        return self._state

    def apply_action(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float64, copy=True)

    def step_ticks(self, tick_count: int) -> RuntimeState:
        if self._state is None:
            raise RuntimeError("Backend must be reset before stepping.")
        tick_count = int(tick_count)
        if tick_count <= 0:
            raise ValueError("tick_count must be positive.")

        state = self._state
        for _ in range(tick_count):
            linear = np.clip(self._action[:3], -0.25, 0.25)
            angular = np.clip(self._action[3:], -2.0, 2.0)
            tcp_velocity = np.concatenate([linear, angular])
            next_tcp_pose = state.tcp_pose.copy()
            next_tcp_pose[:3] += linear * self._sim_dt
            # Store yaw-only orientation integration in pose[5].
            next_tcp_pose[5] += angular[2] * self._sim_dt
            next_tcp_pose[6] = 1.0

            direction = state.target_port_pose[:3] - state.plug_pose[:3]
            plug_correction = 0.15 * direction * self._sim_dt
            next_plug_pose = state.plug_pose.copy()
            next_plug_pose[:3] = next_tcp_pose[:3] + (state.plug_pose[:3] - state.tcp_pose[:3]) + plug_correction
            next_plug_pose[5] = next_tcp_pose[5]
            next_plug_pose[6] = 1.0

            joint_velocities = np.tanh(np.linspace(0.8, 1.2, 6) * linear[0]) * 0.2
            joint_positions = state.joint_positions + joint_velocities * self._sim_dt
            distance = np.linalg.norm(next_plug_pose[:3] - state.target_port_pose[:3])
            insertion_event = None
            if distance < 0.004:
                insertion_event = f"{next(iter(['nic_card_mount_0']))}/{next(iter(['sfp_port_0']))}"
            off_limit_contact = bool(next_tcp_pose[2] < 1.0)
            force_mag = max(0.0, (0.02 - distance) * 1800.0)
            wrench = np.array([0.0, 0.0, force_mag, 0.0, 0.0, 0.0], dtype=np.float64)

            state = RuntimeState(
                sim_tick=state.sim_tick + 1,
                sim_time=state.sim_time + self._sim_dt,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                gripper_position=state.gripper_position,
                tcp_pose=next_tcp_pose,
                tcp_velocity=tcp_velocity,
                plug_pose=next_plug_pose,
                target_port_pose=state.target_port_pose,
                wrench=wrench,
                off_limit_contact=off_limit_contact,
                insertion_event=insertion_event,
            )
        self._state = state
        return state

    def close(self) -> None:
        self._state = None
