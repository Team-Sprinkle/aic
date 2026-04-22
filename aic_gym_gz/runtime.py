"""Runtime and backend abstractions for synchronous AIC training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
    target_port_entrance_pose: np.ndarray | None
    wrench: np.ndarray
    off_limit_contact: bool
    wrench_timestamp: float = 0.0
    insertion_event: str | None = None
    controller_state: dict[str, Any] = field(default_factory=dict)
    score_geometry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeCheckpoint:
    mode: str
    payload: dict[str, Any]
    exact: bool
    limitations: list[str] = field(default_factory=list)


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

    def export_checkpoint(self) -> RuntimeCheckpoint:
        raise NotImplementedError("This backend does not support checkpoint export.")

    def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> RuntimeState:
        raise NotImplementedError("This backend does not support checkpoint restore.")


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

    def export_checkpoint(self) -> RuntimeCheckpoint:
        return self.backend.export_checkpoint()

    def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> RuntimeState:
        return self.backend.restore_checkpoint(checkpoint)


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
        self._last_seed: int | None = None
        self._last_trial_id: str | None = None
        self._task = None
        self._ros_observer: _RuntimeRosObserver | None = None

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
        self._last_seed = seed
        self._last_trial_id = scenario.trial_id
        self._task = next(iter(scenario.tasks.values()))
        if self._runtime is None:
            self._start_runtime()
        if self._ros_observer is None:
            self._ros_observer = _RuntimeRosObserver()
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
        if self._ros_observer is not None:
            self._ros_observer.close()
            self._ros_observer = None
        if self._runtime is not None:
            self._runtime.stop()
            self._runtime = None
        self._last_state = None
        self._last_observation = None
        self._last_info = None

    def export_checkpoint(self) -> RuntimeCheckpoint:
        if self._last_state is None:
            raise RuntimeError("No live runtime state is available to checkpoint.")
        return RuntimeCheckpoint(
            mode="live_reset_replay",
            exact=False,
            payload={
                "seed": self._last_seed,
                "trial_id": self._last_trial_id,
                "state": _serialize_runtime_state(self._last_state),
                "last_observation": dict(self._last_observation or {}),
                "last_info": dict(self._last_info or {}),
            },
            limitations=[
                "Mid-rollout exact restore is not available because the live Gazebo transport path does not expose a world snapshot/restore service.",
                "This checkpoint is suitable for deterministic reset-and-rerun from scenario start, not exact intermediate replay.",
            ],
        )

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
        source_name, source = _resolve_named_entity(
            entities_by_name,
            self._source_entity_name,
            "ati/tool_link",
            "wrist_3_link",
        )
        target_name, target = _resolve_named_entity(
            entities_by_name,
            *self._target_candidates(),
        )
        plug_name, plug = _resolve_named_entity(
            entities_by_name,
            *self._plug_candidates(),
        )
        entrance_name, entrance = _resolve_named_entity(
            entities_by_name,
            *self._target_port_entrance_candidates(),
        )
        for fallback_name in ("lc_plug_link", "sc_plug_link", "sfp_module_link", "cable_0"):
            candidate = entities_by_name.get(fallback_name)
            if isinstance(candidate, dict):
                plug = candidate
                plug_name = fallback_name
                break
        if not isinstance(source, dict):
            raise RuntimeError(
                f"Live observation did not contain source entity '{self._source_entity_name}'."
            )
        if not isinstance(target, dict):
            raise RuntimeError(
                "Live observation did not contain the configured target port entity."
            )

        joint_positions = np.asarray(observation.get("joint_positions") or [], dtype=np.float64)
        joint_velocities = np.zeros_like(joint_positions)
        step_count = int(observation.get("step_count") or 0)
        sim_time = _parse_sim_time_seconds(info.get("state_text_raw") or info.get("state_text") or "")
        tcp_pose = _pose_dict_to_array(source.get("pose") or source)
        target_pose = _pose_dict_to_array(target.get("pose") or target)
        plug_pose = _pose_dict_to_array((plug.get("pose") if isinstance(plug, dict) else None) or plug or source.get("pose") or source)
        entrance_pose = (
            _pose_dict_to_array(entrance.get("pose") or entrance)
            if isinstance(entrance, dict)
            else None
        )
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
        ros_sample = self._ros_observer.snapshot() if self._ros_observer is not None else {}
        wrench, wrench_timestamp = _wrench_from_ros_sample(ros_sample)
        controller_state = _controller_state_from_ros_sample(ros_sample)
        off_limit_contact = bool(ros_sample.get("off_limit_contact", False))
        score_geometry = _build_score_geometry(
            plug_name=plug_name,
            target_name=target_name,
            entrance_name=entrance_name,
            plug_pose=plug_pose,
            target_pose=target_pose,
            entrance_pose=entrance_pose,
            tracked_pair=observation.get("task_geometry", {}).get("tracked_entity_pair", {}),
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
            target_port_entrance_pose=entrance_pose,
            wrench=wrench,
            wrench_timestamp=wrench_timestamp,
            off_limit_contact=off_limit_contact,
            insertion_event=None,
            controller_state=controller_state,
            score_geometry=score_geometry,
        )

    def _plug_candidates(self) -> tuple[str, ...]:
        if self._task is None:
            return ("lc_plug_link", "sc_plug_link", "sfp_module_link", "cable_0")
        return (
            f"{self._task.plug_name}_link",
            self._task.plug_name,
            "lc_plug_link",
            "sc_plug_link",
            "sfp_module_link",
            "cable_0",
        )

    def _target_candidates(self) -> tuple[str, ...]:
        if self._task is None:
            return (self._target_entity_name, "tabletop", "task_board_base_link")
        return (
            f"{self._task.target_module_name}::{self._task.port_name}_link",
            f"{self._task.target_module_name}/{self._task.port_name}_link",
            f"{self._task.port_name}_link",
            self._target_entity_name,
            "tabletop",
            "task_board_base_link",
        )

    def _target_port_entrance_candidates(self) -> tuple[str, ...]:
        if self._task is None:
            return ()
        return (
            f"{self._task.target_module_name}::{self._task.port_name}_link_entrance",
            f"{self._task.target_module_name}/{self._task.port_name}_link_entrance",
            f"{self._task.port_name}_link_entrance",
        )


class MockStepperBackend(RuntimeBackend):
    """Deterministic backend used for tests and the random-policy demo."""

    def __init__(self, *, sim_dt: float = 0.002) -> None:
        self._sim_dt = sim_dt
        self._rng = np.random.default_rng(0)
        self._state: RuntimeState | None = None
        self._action = np.zeros(6, dtype=np.float64)
        self._scenario: AicScenario | None = None

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._scenario = scenario
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
        entrance_pose = target_pose.copy()
        entrance_pose[2] -= 0.0458
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
            target_port_entrance_pose=entrance_pose,
            wrench=np.zeros(6, dtype=np.float64),
            wrench_timestamp=0.0,
            off_limit_contact=False,
            insertion_event=None,
            controller_state={},
            score_geometry=_build_score_geometry(
                plug_name=f"{task.plug_name}_link",
                target_name=f"{task.target_module_name}::{task.port_name}_link",
                entrance_name=f"{task.target_module_name}::{task.port_name}_link_entrance",
                plug_pose=plug_pose,
                target_pose=target_pose,
                entrance_pose=entrance_pose,
                tracked_pair={},
            ),
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
                target_port_entrance_pose=state.target_port_entrance_pose,
                wrench=wrench,
                wrench_timestamp=state.sim_time + self._sim_dt,
                off_limit_contact=off_limit_contact,
                insertion_event=insertion_event,
                controller_state=state.controller_state,
                score_geometry=_build_score_geometry(
                    plug_name=str(state.score_geometry.get("plug_name") or "plug"),
                    target_name=str(state.score_geometry.get("target_port_name") or "target"),
                    entrance_name=str(state.score_geometry.get("port_entrance_name") or "entrance"),
                    plug_pose=next_plug_pose,
                    target_pose=state.target_port_pose,
                    entrance_pose=state.target_port_entrance_pose,
                    tracked_pair={},
                ),
            )
        self._state = state
        return state

    def close(self) -> None:
        self._state = None

    def export_checkpoint(self) -> RuntimeCheckpoint:
        if self._state is None:
            raise RuntimeError("No mock runtime state is available to checkpoint.")
        return RuntimeCheckpoint(
            mode="mock_exact",
            exact=True,
            payload={
                "state": _serialize_runtime_state(self._state),
                "action": self._action.tolist(),
                "rng_state": self._rng.bit_generator.state,
            },
        )

    def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> RuntimeState:
        if checkpoint.mode != "mock_exact":
            raise ValueError(f"Unsupported checkpoint mode for MockStepperBackend: {checkpoint.mode}")
        self._state = _deserialize_runtime_state(checkpoint.payload["state"])
        self._action = np.asarray(checkpoint.payload.get("action", np.zeros(6)), dtype=np.float64)
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = checkpoint.payload["rng_state"]
        return self._state


class _RuntimeRosObserver:
    """Best-effort ROS side observer for score-critical live fields.

    This is additive to the Gazebo transport observation path. It improves
    parity for wrench, controller_state, and off-limit-contact semantics but
    still does not turn the gym runtime into the official evaluation path.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._lock = None
        self._stop_event = None
        self._thread = None
        self._ready = False
        try:
            import threading

            self._lock = threading.Lock()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._spin, daemon=True, name="aic-runtime-ros")
            self._thread.start()
            self._ready = True
        except Exception:
            self._ready = False

    def snapshot(self) -> dict[str, Any]:
        if not self._ready or self._lock is None:
            return {}
        with self._lock:
            return dict(self._data)

    def close(self) -> None:
        if not self._ready or self._stop_event is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _spin(self) -> None:
        import threading

        import rclpy
        from geometry_msgs.msg import WrenchStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from ros_gz_interfaces.msg import Contacts

        from aic_control_interfaces.msg import ControllerState

        context = Context()
        rclpy.init(context=context)
        node = Node("aic_gym_gz_runtime_observer", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)

        controller_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        def set_value(key: str, value: Any) -> None:
            assert self._lock is not None
            with self._lock:
                self._data[key] = value

        def wrench_callback(message: WrenchStamped) -> None:
            set_value("wrench", message)

        def controller_callback(message: ControllerState) -> None:
            set_value("controller_state", message)

        def contact_callback(message: Contacts) -> None:
            active = bool(getattr(message, "contacts", []) or getattr(message, "states", []))
            set_value("off_limit_contact", active)

        node.create_subscription(
            WrenchStamped,
            "/fts_broadcaster/wrench",
            wrench_callback,
            10,
        )
        for topic in (
            "/aic_controller/controller_state",
            "/controller_manager/aic_controller/controller_state",
            "/controller_manager/controller_state",
        ):
            node.create_subscription(
                ControllerState,
                topic,
                controller_callback,
                controller_qos,
            )
        node.create_subscription(
            Contacts,
            "/aic/gazebo/contacts/off_limit",
            contact_callback,
            10,
        )
        while rclpy.ok(context=context) and self._stop_event is not None and not self._stop_event.is_set():
            executor.spin_once(timeout_sec=0.1)
        try:
            executor.remove_node(node)
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown(context=context)
        except Exception:
            pass


def _resolve_named_entity(
    entities_by_name: dict[str, dict[str, Any]],
    *candidates: str,
) -> tuple[str | None, dict[str, Any] | None]:
    for candidate in candidates:
        entity = entities_by_name.get(candidate)
        if isinstance(entity, dict):
            return candidate, entity
    return None, None


def _wrench_from_ros_sample(sample: dict[str, Any]) -> tuple[np.ndarray, float]:
    wrench_msg = sample.get("wrench")
    controller_msg = sample.get("controller_state")
    if wrench_msg is None:
        return np.zeros(6, dtype=np.float64), 0.0
    tare = np.zeros(6, dtype=np.float64)
    if controller_msg is not None:
        tare = np.array(
            [
                float(controller_msg.fts_tare_offset.wrench.force.x),
                float(controller_msg.fts_tare_offset.wrench.force.y),
                float(controller_msg.fts_tare_offset.wrench.force.z),
                float(controller_msg.fts_tare_offset.wrench.torque.x),
                float(controller_msg.fts_tare_offset.wrench.torque.y),
                float(controller_msg.fts_tare_offset.wrench.torque.z),
            ],
            dtype=np.float64,
        )
    wrench = np.array(
        [
            float(wrench_msg.wrench.force.x),
            float(wrench_msg.wrench.force.y),
            float(wrench_msg.wrench.force.z),
            float(wrench_msg.wrench.torque.x),
            float(wrench_msg.wrench.torque.y),
            float(wrench_msg.wrench.torque.z),
        ],
        dtype=np.float64,
    ) - tare
    timestamp = float(wrench_msg.header.stamp.sec) + float(wrench_msg.header.stamp.nanosec) * 1e-9
    return wrench, timestamp


def _controller_state_from_ros_sample(sample: dict[str, Any]) -> dict[str, Any]:
    message = sample.get("controller_state")
    if message is None:
        return {}
    return {
        "tcp_pose": np.array(
            [
                float(message.tcp_pose.position.x),
                float(message.tcp_pose.position.y),
                float(message.tcp_pose.position.z),
                float(message.tcp_pose.orientation.x),
                float(message.tcp_pose.orientation.y),
                float(message.tcp_pose.orientation.z),
                float(message.tcp_pose.orientation.w),
            ],
            dtype=np.float64,
        ),
        "tcp_velocity": np.array(
            [
                float(message.tcp_velocity.linear.x),
                float(message.tcp_velocity.linear.y),
                float(message.tcp_velocity.linear.z),
                float(message.tcp_velocity.angular.x),
                float(message.tcp_velocity.angular.y),
                float(message.tcp_velocity.angular.z),
            ],
            dtype=np.float64,
        ),
        "reference_tcp_pose": np.array(
            [
                float(message.reference_tcp_pose.position.x),
                float(message.reference_tcp_pose.position.y),
                float(message.reference_tcp_pose.position.z),
                float(message.reference_tcp_pose.orientation.x),
                float(message.reference_tcp_pose.orientation.y),
                float(message.reference_tcp_pose.orientation.z),
                float(message.reference_tcp_pose.orientation.w),
            ],
            dtype=np.float64,
        ),
        "tcp_error": np.asarray(message.tcp_error, dtype=np.float64),
        "reference_joint_state": np.asarray(message.reference_joint_state.positions, dtype=np.float64),
        "target_mode": int(message.target_mode.mode),
        "fts_tare_offset": np.array(
            [
                float(message.fts_tare_offset.wrench.force.x),
                float(message.fts_tare_offset.wrench.force.y),
                float(message.fts_tare_offset.wrench.force.z),
                float(message.fts_tare_offset.wrench.torque.x),
                float(message.fts_tare_offset.wrench.torque.y),
                float(message.fts_tare_offset.wrench.torque.z),
            ],
            dtype=np.float64,
        ),
    }


def _build_score_geometry(
    *,
    plug_name: str | None,
    target_name: str | None,
    entrance_name: str | None,
    plug_pose: np.ndarray,
    target_pose: np.ndarray,
    entrance_pose: np.ndarray | None,
    tracked_pair: dict[str, Any],
) -> dict[str, Any]:
    target_distance = float(np.linalg.norm(target_pose[:3] - plug_pose[:3]))
    geometry: dict[str, Any] = {
        "plug_name": plug_name,
        "target_port_name": target_name,
        "port_entrance_name": entrance_name,
        "distance_to_target": target_distance,
        "tracked_distance": tracked_pair.get("distance"),
        "orientation_error": tracked_pair.get("orientation_error"),
    }
    if entrance_pose is not None:
        insertion_axis = target_pose[:3] - entrance_pose[:3]
        insertion_length = float(np.linalg.norm(insertion_axis))
        axis_unit = (
            insertion_axis / insertion_length
            if insertion_length > 1e-8
            else np.array([0.0, 0.0, 1.0], dtype=np.float64)
        )
        plug_offset = plug_pose[:3] - entrance_pose[:3]
        axial_depth = float(np.dot(plug_offset, axis_unit))
        clipped_axial_depth = float(np.clip(axial_depth, 0.0, insertion_length))
        lateral_offset = plug_offset - (axial_depth * axis_unit)
        lateral_misalignment = float(np.linalg.norm(lateral_offset))
        distance_to_entrance = float(np.linalg.norm(entrance_pose[:3] - plug_pose[:3]))
        partial_insertion = bool(
            lateral_misalignment < 0.005 and clipped_axial_depth > 0.0 and clipped_axial_depth < insertion_length + 0.01
        )
        geometry["distance_threshold"] = insertion_length
        geometry["distance_to_entrance"] = distance_to_entrance
        geometry["partial_insertion"] = partial_insertion
        geometry["plug_to_port_depth"] = float(plug_pose[2] - target_pose[2])
        geometry["port_to_entrance_depth"] = float(entrance_pose[2] - target_pose[2])
        geometry["corridor_axis"] = axis_unit.tolist()
        geometry["lateral_misalignment"] = lateral_misalignment
        geometry["axial_depth"] = axial_depth
        geometry["insertion_progress"] = (
            0.0 if insertion_length <= 1e-8 else clipped_axial_depth / insertion_length
        )
    else:
        geometry["distance_to_entrance"] = target_distance
        geometry["lateral_misalignment"] = 0.0
        geometry["insertion_progress"] = 0.0
        geometry["partial_insertion"] = False
    return geometry


def _serialize_runtime_state(state: RuntimeState) -> dict[str, Any]:
    return {
        "sim_tick": state.sim_tick,
        "sim_time": state.sim_time,
        "joint_positions": state.joint_positions.tolist(),
        "joint_velocities": state.joint_velocities.tolist(),
        "gripper_position": state.gripper_position,
        "tcp_pose": state.tcp_pose.tolist(),
        "tcp_velocity": state.tcp_velocity.tolist(),
        "plug_pose": state.plug_pose.tolist(),
        "target_port_pose": state.target_port_pose.tolist(),
        "target_port_entrance_pose": None
        if state.target_port_entrance_pose is None
        else state.target_port_entrance_pose.tolist(),
        "wrench": state.wrench.tolist(),
        "wrench_timestamp": state.wrench_timestamp,
        "off_limit_contact": state.off_limit_contact,
        "insertion_event": state.insertion_event,
        "controller_state": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in state.controller_state.items()
        },
        "score_geometry": dict(state.score_geometry),
    }


def _deserialize_runtime_state(payload: dict[str, Any]) -> RuntimeState:
    controller_state = {
        key: np.asarray(value, dtype=np.float64)
        if isinstance(value, list)
        else value
        for key, value in dict(payload.get("controller_state", {})).items()
    }
    entrance_pose = payload.get("target_port_entrance_pose")
    return RuntimeState(
        sim_tick=int(payload["sim_tick"]),
        sim_time=float(payload["sim_time"]),
        joint_positions=np.asarray(payload["joint_positions"], dtype=np.float64),
        joint_velocities=np.asarray(payload["joint_velocities"], dtype=np.float64),
        gripper_position=float(payload["gripper_position"]),
        tcp_pose=np.asarray(payload["tcp_pose"], dtype=np.float64),
        tcp_velocity=np.asarray(payload["tcp_velocity"], dtype=np.float64),
        plug_pose=np.asarray(payload["plug_pose"], dtype=np.float64),
        target_port_pose=np.asarray(payload["target_port_pose"], dtype=np.float64),
        target_port_entrance_pose=(
            None if entrance_pose is None else np.asarray(entrance_pose, dtype=np.float64)
        ),
        wrench=np.asarray(payload["wrench"], dtype=np.float64),
        wrench_timestamp=float(payload.get("wrench_timestamp", 0.0)),
        off_limit_contact=bool(payload["off_limit_contact"]),
        insertion_event=payload.get("insertion_event"),
        controller_state=controller_state,
        score_geometry=dict(payload.get("score_geometry", {})),
    )


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
