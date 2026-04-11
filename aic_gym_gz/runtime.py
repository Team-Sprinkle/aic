"""Runtime and backend abstractions for synchronous AIC training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from pathlib import Path
import re
import subprocess
import sys
import time
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
    """Live Gazebo backend routed through the training-only transport runtime.

    This backend keeps the stable `aic_gym_gz` API while delegating simulator
    lifecycle and exact-tick stepping to `aic_utils/aic_gazebo_env`.

    It is the real backend path for local/containerized Gazebo runs. The mock
    backend remains available for deterministic unit tests.
    """

    def __init__(
        self,
        *,
        world_name: str = "aic_world",
        world_path: str | None = None,
        headless: bool = True,
        timeout: float = 10.0,
        source_entity_name: str = "ati/tool_link",
        target_entity_name: str = "tabletop",
        transport_backend: str = "transport",
        attach_to_existing: bool = False,
    ) -> None:
        self._world_name = world_name
        self._world_path = world_path or str(
            Path(__file__).resolve().parents[1] / "aic_description" / "world" / "aic.sdf"
        )
        self._headless = headless
        self._timeout = timeout
        self._source_entity_name = source_entity_name
        self._target_entity_name = target_entity_name
        self._transport_backend = transport_backend
        self._attach_to_existing = attach_to_existing
        self._runtime = None
        self._action = np.zeros(6, dtype=np.float64)
        self._last_state: RuntimeState | None = None
        self._last_observation: dict[str, Any] | None = None
        self._last_info: dict[str, Any] | None = None

        repo_gazebo_env = Path(__file__).resolve().parents[1] / "aic_utils" / "aic_gazebo_env"
        if repo_gazebo_env.exists():
            sys.path.insert(0, str(repo_gazebo_env))
        try:
            from aic_gazebo_env.runtime import (  # type: ignore
                GazeboAttachedRuntime,
                GazeboRuntime,
                GazeboRuntimeConfig,
            )
            from aic_gazebo_env.gazebo_client import GazeboCliClient, GazeboCliClientConfig  # type: ignore
            from aic_gazebo_env.protocol import GetObservationRequest  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The live Gazebo backend depends on `aic_utils/aic_gazebo_env`. "
                "Ensure the repo-local training runtime is present and on PYTHONPATH."
            ) from exc

        self._runtime_type = GazeboAttachedRuntime if attach_to_existing else GazeboRuntime
        self._runtime_config_type = GazeboRuntimeConfig
        self._cli_client_type = GazeboCliClient
        self._cli_config_type = GazeboCliClientConfig
        self._get_observation_request_type = GetObservationRequest

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        del scenario
        if self._runtime is None:
            self._start_runtime()
        self._action[:] = 0.0
        if self._attach_to_existing:
            return self.connect_existing_world()
        observation, info = self._runtime.reset(seed=seed, options={})
        return self._await_runtime_state(observation=observation, info=info, timeout_s=self._timeout)

    def connect_existing_world(self) -> RuntimeState:
        if self._runtime is None:
            self._start_runtime()
        deadline = time.monotonic() + self._timeout
        last_error: Exception | None = None
        bootstrap_state = self._bootstrap_existing_world_state()
        if bootstrap_state is not None:
            return bootstrap_state
        while time.monotonic() < deadline:
            try:
                observation, info = self._runtime.get_observation()
                self._action[:] = 0.0
                self._last_observation = dict(observation)
                self._last_info = dict(info)
                self._last_state = self._runtime_state_from_observation(observation, info)
                return self._last_state
            except RuntimeError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
            except TimeoutError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
        raise RuntimeError(f"Timed out waiting for attached world readiness: {last_error}")

    def apply_action(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float64, copy=True)

    def step_ticks(self, tick_count: int) -> RuntimeState:
        if self._runtime is None:
            raise RuntimeError("Live backend must be reset before stepping.")
        tick_count = int(tick_count)
        if tick_count <= 0:
            raise ValueError("tick_count must be positive.")
        position_delta = (np.clip(self._action[:3], -0.25, 0.25) * 0.002 * tick_count).tolist()
        rotation_delta = _angular_delta_to_quaternion(
            np.clip(self._action[3:], -2.0, 2.0) * 0.002 * tick_count
        )
        observation, _, _, _, info = self._runtime.step(
            {
                "ee_delta_action": {
                    "position_delta": position_delta,
                    "orientation_delta": rotation_delta,
                    "frame": "world",
                },
                "multi_step": tick_count,
            }
        )
        self._last_observation = dict(observation)
        self._last_info = dict(info)
        self._last_state = self._runtime_state_from_observation(observation, info)
        return self._last_state

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None
        self._last_state = None
        self._last_observation = None
        self._last_info = None

    def last_native_trace_fields(self) -> dict[str, Any]:
        if self._last_observation is None:
            raise RuntimeError("No live observation has been recorded yet.")
        observation = self._last_observation
        entities_by_name = observation.get("entities_by_name") or {}

        def pose_fields(name: str | None) -> tuple[str | None, list[float] | None, list[float] | None]:
            if name is None:
                return None, None, None
            entity = entities_by_name.get(name)
            if not isinstance(entity, dict):
                return name, None, None
            position = entity.get("position")
            orientation = entity.get("orientation")
            return (
                name,
                list(position) if isinstance(position, list) else None,
                list(orientation) if isinstance(orientation, list) else None,
            )

        plug_name = None
        for candidate in ("lc_plug_link", "sc_plug_link", "sfp_module_link", "cable_0"):
            if isinstance(entities_by_name.get(candidate), dict):
                plug_name = candidate
                break

        tracked = ((observation.get("task_geometry") or {}).get("tracked_entity_pair") or {})
        tcp_name, tcp_position, tcp_orientation = pose_fields(self._source_entity_name)
        plug_name, plug_position, plug_orientation = pose_fields(plug_name)
        target_name, target_position, target_orientation = pose_fields(self._target_entity_name)
        return {
            "joint_positions": list(observation.get("joint_positions") or []),
            "joint_velocities": list(observation.get("joint_velocities") or []),
            "sim_time": (
                None
                if self._last_state is None
                else float(self._last_state.sim_time)
            ),
            "tcp_entity_name": tcp_name,
            "tcp_position": tcp_position,
            "tcp_orientation": tcp_orientation,
            "plug_entity_name": plug_name,
            "plug_position": plug_position,
            "plug_orientation": plug_orientation,
            "target_entity_name": target_name,
            "target_position": target_position,
            "target_orientation": target_orientation,
            "relative_position": list(tracked.get("relative_position") or []),
            "distance_to_target": tracked.get("distance"),
            "orientation_error": tracked.get("orientation_error"),
            "success_like": bool(tracked.get("success", False)),
            "force_magnitude": 0.0,
            "off_limit_contact": False,
            "entity_count": observation.get("entity_count"),
            "joint_count": observation.get("joint_count"),
            "step_count_raw": observation.get("step_count"),
        }

    def _start_runtime(self) -> None:
        self._runtime = self._runtime_type(
            self._runtime_config_type(
                world_path=self._world_path,
                headless=self._headless,
                timeout=self._timeout,
                world_name=self._world_name,
                source_entity_name=self._source_entity_name,
                target_entity_name=self._target_entity_name,
                transport_backend=self._transport_backend,
            )
        )
        self._runtime.start()

    def _tick_existing_world_for_sample(self) -> None:
        try:
            subprocess.run(
                [
                    "gz",
                    "service",
                    "-s",
                    f"/world/{self._world_name}/control",
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(int(self._timeout * 1000.0)),
                    "--req",
                    "multi_step: 1",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=max(self._timeout, 1.0),
            )
        except Exception:
            if self._runtime is None:
                return
            try:
                self._runtime.step({"multi_step": 1})
            except Exception:
                return

    def _await_runtime_state(
        self,
        *,
        observation: dict[str, Any],
        info: dict[str, Any],
        timeout_s: float,
    ) -> RuntimeState:
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        current_observation = observation
        current_info = info
        while True:
            try:
                self._last_observation = dict(current_observation)
                self._last_info = dict(current_info)
                self._last_state = self._runtime_state_from_observation(
                    current_observation,
                    current_info,
                )
                return self._last_state
            except RuntimeError as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Timed out waiting for a sane live observation after reset: {last_error}"
                    ) from exc
                time.sleep(0.1)
                current_observation, current_info = self._runtime.get_observation()

    def _bootstrap_existing_world_state(self) -> RuntimeState | None:
        try:
            client = self._cli_client_type(
                self._cli_config_type(
                    executable="gz",
                    world_path=self._world_path,
                    timeout=self._timeout,
                    world_name=self._world_name,
                    source_entity_name=self._source_entity_name,
                    target_entity_name=self._target_entity_name,
                    transport_backend="cli",
                    observation_transport="one_shot",
                )
            )
        except Exception:
            return None
        try:
            deadline = time.monotonic() + self._timeout
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    response = client.get_observation(self._get_observation_request_type())
                    observation = dict(response.observation)
                    info = dict(response.info)
                    self._action[:] = 0.0
                    self._last_observation = observation
                    self._last_info = info
                    self._last_state = self._runtime_state_from_observation(observation, info)
                    return self._last_state
                except Exception as exc:
                    last_error = exc
                    self._tick_existing_world_for_sample()
                    time.sleep(0.1)
            if last_error is not None:
                raise last_error
        except Exception:
            return None
        finally:
            try:
                client.close()
            except Exception:
                pass
        return None

    def _runtime_state_from_observation(
        self,
        observation: dict[str, Any],
        info: dict[str, Any],
    ) -> RuntimeState:
        entities_by_name = observation.get("entities_by_name") or {}
        source = entities_by_name.get(self._source_entity_name)
        target = entities_by_name.get(self._target_entity_name)
        plug = None
        for plug_name in ("lc_plug_link", "sc_plug_link", "sfp_module_link", "cable_0"):
            candidate = entities_by_name.get(plug_name)
            if isinstance(candidate, dict):
                plug = candidate
                break
        if not isinstance(source, dict):
            raise RuntimeError(
                f"Live observation did not contain source entity '{self._source_entity_name}'."
            )
        if not isinstance(target, dict):
            raise RuntimeError(
                f"Live observation did not contain target entity '{self._target_entity_name}'."
            )

        joint_positions = np.asarray(observation.get("joint_positions") or [], dtype=np.float64)
        joint_velocities = np.zeros_like(joint_positions)
        step_count = int(observation.get("step_count") or 0)
        sim_time = _parse_sim_time_seconds(info.get("state_text_raw") or info.get("state_text") or "")
        tcp_pose = _pose_dict_to_array(source.get("pose") or source)
        target_pose = _pose_dict_to_array(target.get("pose") or target)
        plug_pose = _pose_dict_to_array((plug.get("pose") if isinstance(plug, dict) else None) or plug or source.get("pose") or source)
        previous_state = self._last_state
        if previous_state is None or previous_state.sim_time >= sim_time:
            tcp_velocity = np.zeros(6, dtype=np.float64)
        else:
            dt = sim_time - previous_state.sim_time
            tcp_velocity = np.concatenate(
                [
                    (tcp_pose[:3] - previous_state.tcp_pose[:3]) / dt,
                    np.zeros(3, dtype=np.float64),
                ]
            )

        return RuntimeState(
            sim_tick=step_count,
            sim_time=sim_time,
            joint_positions=joint_positions,
            joint_velocities=joint_velocities,
            gripper_position=0.0,
            tcp_pose=tcp_pose,
            tcp_velocity=tcp_velocity,
            plug_pose=plug_pose,
            target_port_pose=target_pose,
            wrench=np.zeros(6, dtype=np.float64),
            off_limit_contact=False,
            insertion_event=None,
        )


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


def _pose_dict_to_array(pose_like: dict[str, Any]) -> np.ndarray:
    position = pose_like.get("position") or [0.0, 0.0, 0.0]
    orientation = pose_like.get("orientation") or [0.0, 0.0, 0.0, 1.0]
    quaternion = np.asarray(orientation, dtype=np.float64)
    yaw = _quaternion_to_yaw(quaternion)
    return np.array(
        [
            float(position[0]),
            float(position[1]),
            float(position[2]),
            0.0,
            0.0,
            yaw,
            float(quaternion[3]),
        ],
        dtype=np.float64,
    )


def _quaternion_to_yaw(quaternion: np.ndarray) -> float:
    x, y, z, w = quaternion.tolist()
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _angular_delta_to_quaternion(delta_rpy: np.ndarray) -> list[float]:
    roll, pitch, yaw = [float(value) for value in delta_rpy.tolist()]
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _parse_sim_time_seconds(state_text: str) -> float:
    sec_match = re.search(r"sim_time\s*\{\s*sec:\s*(\d+)", state_text)
    nsec_match = re.search(r"sim_time\s*\{.*?nsec:\s*(\d+)", state_text, re.S)
    if sec_match is None:
        return 0.0
    seconds = float(sec_match.group(1))
    nanoseconds = float(nsec_match.group(1)) if nsec_match is not None else 0.0
    return seconds + nanoseconds * 1e-9
