"""Runtime and backend abstractions for synchronous AIC training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

from .scenario import AicScenario


def _configure_ros_eval_session_env() -> None:
    """Best-effort middleware setup so in-process ROS clients join eval Zenoh."""
    if os.environ.get("AIC_GYM_GZ_FORCE_CYCLONEDDS", "").strip() in {"1", "true", "TRUE"}:
        os.environ["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
        os.environ.pop("ZENOH_SESSION_CONFIG_URI", None)
        os.environ.pop("ZENOH_CONFIG_OVERRIDE", None)
        return
    repo_root = Path(__file__).resolve().parents[1]
    eval_dir = repo_root / "docker" / "aic_eval"
    session_config = eval_dir / "aic_zenoh_config.json5"
    credentials = eval_dir / "credentials.txt"
    if session_config.exists():
        os.environ.setdefault("ZENOH_SESSION_CONFIG_URI", str(session_config))
    if credentials.exists():
        override = (
            'transport/auth/usrpwd/user="eval";'
            'transport/auth/usrpwd/password="CHANGE_IN_PROD";'
            f'transport/auth/usrpwd/dictionary_file="{credentials}";'
            'transport/shared_memory/enabled=false'
        )
        os.environ.setdefault("ZENOH_CONFIG_OVERRIDE", override)
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_zenoh_cpp")


@dataclass(frozen=True)
class AuxiliaryForceContactSummary:
    """Auxiliary within-step force/contact summary.

    This payload is explicitly non-official and is not part of the
    official-compatible observation surface. It exists to summarize internal
    sub-samples collected during one `env.step()`.
    """

    is_official_observation: bool = False
    source: str = "final_sample_only"
    substep_tick_count: int = 0
    sample_count: int = 0
    wrench_current: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    wrench_max_abs_recent: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    wrench_mean_recent: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    wrench_max_force_abs_recent: float = 0.0
    wrench_max_torque_abs_recent: float = 0.0
    had_contact_recent: bool = False
    max_contact_indicator_recent: float = 0.0
    first_wrench_recent: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    last_wrench_recent: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    time_of_peak_within_step: float | None = None
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_official_observation": bool(self.is_official_observation),
            "source": self.source,
            "substep_tick_count": int(self.substep_tick_count),
            "sample_count": int(self.sample_count),
            "wrench_current": self.wrench_current.astype(np.float32).copy(),
            "wrench_max_abs_recent": self.wrench_max_abs_recent.astype(np.float32).copy(),
            "wrench_mean_recent": self.wrench_mean_recent.astype(np.float32).copy(),
            "wrench_max_force_abs_recent": float(self.wrench_max_force_abs_recent),
            "wrench_max_torque_abs_recent": float(self.wrench_max_torque_abs_recent),
            "had_contact_recent": bool(self.had_contact_recent),
            "max_contact_indicator_recent": float(self.max_contact_indicator_recent),
            "first_wrench_recent": self.first_wrench_recent.astype(np.float32).copy(),
            "last_wrench_recent": self.last_wrench_recent.astype(np.float32).copy(),
            "time_of_peak_within_step": (
                None
                if self.time_of_peak_within_step is None
                else float(self.time_of_peak_within_step)
            ),
            "limitations": list(self.limitations),
        }


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
    world_entities_summary: dict[str, Any] = field(default_factory=dict)
    auxiliary_force_contact_summary: AuxiliaryForceContactSummary = field(
        default_factory=AuxiliaryForceContactSummary
    )


@dataclass(frozen=True)
class RuntimeCheckpoint:
    mode: str
    payload: dict[str, Any]
    exact: bool
    limitations: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MockTransientContactConfig:
    """Optional deterministic narrow-band contact model for validation."""

    contact_band_z: tuple[float, float] | None = None
    peak_force_newtons: float = 30.0
    peak_torque_newton_meters: float = 3.0


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


def _single_sample_force_contact_summary(
    *,
    wrench: np.ndarray,
    contact: bool,
    substep_tick_count: int,
    source: str,
    limitations: list[str] | tuple[str, ...] = (),
) -> AuxiliaryForceContactSummary:
    return AuxiliaryForceContactSummary(
        source=source,
        substep_tick_count=int(substep_tick_count),
        sample_count=1,
        wrench_current=wrench.copy(),
        wrench_max_abs_recent=np.abs(wrench),
        wrench_mean_recent=wrench.copy(),
        wrench_max_force_abs_recent=float(np.linalg.norm(wrench[:3])),
        wrench_max_torque_abs_recent=float(np.linalg.norm(wrench[3:])),
        had_contact_recent=bool(contact),
        max_contact_indicator_recent=1.0 if contact else 0.0,
        first_wrench_recent=wrench.copy(),
        last_wrench_recent=wrench.copy(),
        time_of_peak_within_step=0.0,
        limitations=tuple(str(item) for item in limitations),
    )


def _force_contact_summary_from_samples(
    *,
    wrench_samples: list[np.ndarray],
    timestamps: list[float],
    contact_indicators: list[float],
    substep_tick_count: int,
    source: str,
    limitations: list[str] | tuple[str, ...] = (),
) -> AuxiliaryForceContactSummary:
    if not wrench_samples:
        return _single_sample_force_contact_summary(
            wrench=np.zeros(6, dtype=np.float64),
            contact=False,
            substep_tick_count=substep_tick_count,
            source=source,
            limitations=list(limitations) + ["No within-step samples were available."],
        )
    stack = np.stack([np.asarray(sample, dtype=np.float64) for sample in wrench_samples], axis=0)
    force_norms = np.linalg.norm(stack[:, :3], axis=1)
    torque_norms = np.linalg.norm(stack[:, 3:], axis=1)
    peak_index = int(np.argmax(force_norms)) if force_norms.size else 0
    if timestamps:
        start_time = float(timestamps[0])
        peak_time = float(timestamps[min(peak_index, len(timestamps) - 1)]) - start_time
    else:
        peak_time = None
    return AuxiliaryForceContactSummary(
        source=source,
        substep_tick_count=int(substep_tick_count),
        sample_count=int(stack.shape[0]),
        wrench_current=stack[-1].copy(),
        wrench_max_abs_recent=np.max(np.abs(stack), axis=0),
        wrench_mean_recent=np.mean(stack, axis=0),
        wrench_max_force_abs_recent=float(force_norms.max(initial=0.0)),
        wrench_max_torque_abs_recent=float(torque_norms.max(initial=0.0)),
        had_contact_recent=bool(any(indicator > 0.0 for indicator in contact_indicators)),
        max_contact_indicator_recent=float(max(contact_indicators, default=0.0)),
        first_wrench_recent=stack[0].copy(),
        last_wrench_recent=stack[-1].copy(),
        time_of_peak_within_step=peak_time,
        limitations=tuple(str(item) for item in limitations),
    )


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
        attach_ready_timeout: float | None = None,
        source_entity_name: str = "ati/tool_link",
        target_entity_name: str = "tabletop",
        transport_backend: str = "transport",
        attach_to_existing: bool = False,
        allow_synthetic_tcp_pose: bool = True,
        allow_synthetic_plug_pose: bool = True,
        use_controller_velocity_commands: bool = False,
        live_mode: str = "gazebo_training_fast",
        observation_transport_override: str | None = None,
        state_observation_mode: str = "honest_live",
    ) -> None:
        self._world_name = world_name
        self._world_path = world_path or str(
            Path(__file__).resolve().parents[1] / "aic_description" / "world" / "aic.sdf"
        )
        self._headless = headless
        self._timeout = timeout
        self._attach_ready_timeout = (
            float(attach_ready_timeout) if attach_ready_timeout is not None else max(float(timeout), 30.0)
        )
        self._source_entity_name = source_entity_name
        self._target_entity_name = target_entity_name
        self._transport_backend = transport_backend
        self._attach_to_existing = attach_to_existing
        self._allow_synthetic_tcp_pose = bool(allow_synthetic_tcp_pose)
        self._allow_synthetic_plug_pose = bool(allow_synthetic_plug_pose)
        self._use_controller_velocity_commands = bool(use_controller_velocity_commands)
        self._live_mode = str(live_mode).strip().lower()
        self._observation_transport_override = (
            None if observation_transport_override is None else str(observation_transport_override).strip().lower()
        )
        self._state_observation_mode = str(state_observation_mode).strip().lower()
        self._runtime = None
        self._action = np.zeros(6, dtype=np.float64)
        self._last_state: RuntimeState | None = None
        self._last_observation: dict[str, Any] | None = None
        self._last_info: dict[str, Any] | None = None
        self._last_seed: int | None = None
        self._last_trial_id: str | None = None
        self._scenario: AicScenario | None = None
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
        print('{"backend_stage":"reset_enter"}', flush=True)
        self._last_seed = seed
        self._last_trial_id = scenario.trial_id
        self._scenario = scenario
        self._task = next(iter(scenario.tasks.values()))
        if self._live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"} and not self._attach_to_existing:
            if self._runtime is not None:
                try:
                    self._runtime.stop()
                finally:
                    self._runtime = None
        if self._runtime is None:
            print('{"backend_stage":"start_runtime_begin"}', flush=True)
            self._start_runtime()
            print('{"backend_stage":"start_runtime_done"}', flush=True)
        if (
            self._ros_observer is None
            and (self._attach_to_existing or self._use_controller_velocity_commands)
        ):
            self._ros_observer = _RuntimeRosObserver()
        self._action[:] = 0.0
        if self._attach_to_existing:
            # Attaching to an already-launched official scene must be read-only.
            # Issuing a world reset here perturbs the scene/controller startup path
            # and can invalidate evaluation-style attach semantics.
            return self.connect_existing_world()
        if (
            self._live_mode == "gazebo_training_fast"
            and getattr(self, "_state_observation_mode", "honest_live") == "synthetic_training"
        ):
            synthetic_reset = self._synthetic_training_fast_reset_state()
            if synthetic_reset is not None:
                self._last_state = synthetic_reset
                self._seed_fast_source_pose_cache(synthetic_reset)
                print('{"backend_stage":"synthetic_training_reset_shortcut"}', flush=True)
                return synthetic_reset
        if self._live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast", "controller_velocity_wip"}:
            print('{"backend_stage":"connect_started_world_begin"}', flush=True)
            return self._connect_started_world(timeout_s=min(max(self._timeout, 5.0), 10.0))
        observation, info = self._runtime.reset(seed=seed, options={})
        print('{"backend_stage":"runtime_reset_done"}', flush=True)
        return self._await_runtime_state(observation=observation, info=info, timeout_s=self._timeout)

    def _connect_started_world(self, *, timeout_s: float) -> RuntimeState:
        if self._runtime is None:
            raise RuntimeError("Live runtime must be started before connecting to a fresh world.")
        bootstrap_timeout_s = min(max(timeout_s * 0.2, 5.0), 20.0)
        if os.environ.get("AIC_GYM_GZ_ENABLE_CLI_BOOTSTRAP") == "1":
            print('{"backend_stage":"bootstrap_cli_begin"}', flush=True)
            bootstrap_state = self._bootstrap_existing_world_state(timeout_s=bootstrap_timeout_s)
            if bootstrap_state is not None:
                bootstrap_state = self._settle_training_fast_home(bootstrap_state)
                self._seed_fast_source_pose_cache(bootstrap_state)
                print('{"backend_stage":"bootstrap_cli_done"}', flush=True)
                return bootstrap_state
            print('{"backend_stage":"bootstrap_cli_miss"}', flush=True)
        deadline = time.monotonic() + timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                observation, info = self._runtime.get_observation()
                self._action[:] = 0.0
                self._last_observation = dict(observation)
                self._last_info = dict(info)
                self._last_state = self._runtime_state_from_observation(observation, info)
                if not self._state_is_sane_for_live_reset(self._last_state):
                    raise RuntimeError("Fresh training-fast world is not yet exposing a sane reset pose.")
                self._last_state = self._settle_training_fast_home(self._last_state)
                self._seed_fast_source_pose_cache(self._last_state)
                print('{"backend_stage":"connect_started_world_done"}', flush=True)
                return self._last_state
            except RuntimeError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
            except TimeoutError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
        fallback_state = self._synthetic_training_fast_reset_state()
        if fallback_state is not None:
            print('{"backend_stage":"synthetic_reset_fallback"}', flush=True)
            self._last_state = fallback_state
            self._seed_fast_source_pose_cache(fallback_state)
            return fallback_state
        raise RuntimeError(
            "Timed out waiting for fresh training-fast world readiness: "
            f"{last_error}. timeout={timeout_s}s, "
            f"transport_backend={self._transport_backend}, source_entity={self._source_entity_name}, "
            f"target_entity={self._target_entity_name}"
        )

    def _settle_training_fast_home(self, state: RuntimeState) -> RuntimeState:
        if self._live_mode != "gazebo_training_fast" or self._runtime is None:
            return state
        target = np.array([-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110], dtype=np.float64)
        latest_state = state
        for chunk in (500, 500, 500, 500):
            try:
                observation, _, _, _, info = self._runtime.step(
                    {
                        "set_joint_positions": {
                            "model_name": "ur5e",
                            "joint_names": [
                                "shoulder_pan_joint",
                                "shoulder_lift_joint",
                                "elbow_joint",
                                "wrist_1_joint",
                                "wrist_2_joint",
                                "wrist_3_joint",
                            ],
                            "positions": target.tolist(),
                        },
                        "multi_step": int(chunk),
                    }
                )
                self._last_observation = dict(observation)
                self._last_info = dict(info)
                latest_state = self._runtime_state_from_observation(observation, info)
                joints = np.asarray(latest_state.joint_positions, dtype=np.float64)
                if joints.shape == target.shape and np.max(np.abs(joints - target)) < 0.05:
                    return latest_state
            except Exception:
                return latest_state
        return latest_state

    def _synthetic_training_fast_reset_state(self) -> RuntimeState | None:
        if getattr(self, "_state_observation_mode", "honest_live") != "synthetic_training":
            return None
        if self._scenario is None or self._task is None:
            return None
        synthetic_target = _synthetic_target_pose_from_scenario(
            scenario=self._scenario,
            task=self._task,
        )
        if synthetic_target is None:
            return None
        target_pose = _pose_dict_to_array(synthetic_target["target_pose_dict"])
        entrance_pose = (
            _pose_dict_to_array(synthetic_target["entrance_pose_dict"])
            if synthetic_target["entrance_pose_dict"] is not None
            else None
        )
        tcp_pose = np.array([-0.45, 0.2, 1.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        entities_by_name = self._supplement_entities_from_cli_once() or {}
        source_name, source = _resolve_named_entity(
            entities_by_name,
            self._source_entity_name,
            "ati/tool_link",
            "wrist_3_link",
        )
        if isinstance(source, dict):
            try:
                tcp_pose = _pose_dict_to_array(source.get("pose") or source)
            except Exception:
                pass
        cable = self._scenario.cables.get(self._task.cable_name)
        if cable is None:
            return None
        plug_pose = np.asarray(tcp_pose, dtype=np.float64).copy()
        plug_pose[:3] += np.array(cable.gripper_offset_xyz, dtype=np.float64)
        controller_state: dict[str, Any] = {}
        score_geometry = _build_score_geometry(
            plug_name="synthetic_plug_from_tcp",
            target_name=f"{self._task.target_module_name}::{self._task.port_name}_link",
            entrance_name=f"{self._task.target_module_name}::{self._task.port_name}_link_entrance",
            plug_pose=plug_pose,
            target_pose=target_pose,
            entrance_pose=entrance_pose,
            tracked_pair={},
        )
        world_entities_summary = _world_entities_summary(
            entities_by_name=entities_by_name,
            source_name=source_name or "synthetic_tcp_pose",
            target_name=f"{self._task.target_module_name}::{self._task.port_name}_link",
            plug_name="synthetic_plug_from_tcp",
            entrance_name=f"{self._task.target_module_name}::{self._task.port_name}_link_entrance",
        )
        world_entities_summary["runtime_diagnostics"] = {
            "attach_to_existing": False,
            "live_mode": self._live_mode,
            "synthetic_bootstrap_reset": True,
            "synthetic_tcp_pose_used": True,
            "synthetic_plug_pose_used": True,
            "scene_alignment_ok": True,
        }
        return RuntimeState(
            sim_tick=0,
            sim_time=0.0,
            joint_positions=np.zeros(6, dtype=np.float64),
            joint_velocities=np.zeros(6, dtype=np.float64),
            gripper_position=0.0,
            tcp_pose=tcp_pose,
            tcp_velocity=np.zeros(6, dtype=np.float64),
            plug_pose=plug_pose,
            target_port_pose=target_pose,
            target_port_entrance_pose=entrance_pose,
            wrench=np.zeros(6, dtype=np.float64),
            wrench_timestamp=0.0,
            off_limit_contact=False,
            insertion_event=None,
            controller_state=controller_state,
            score_geometry=score_geometry,
            world_entities_summary=world_entities_summary,
            auxiliary_force_contact_summary=_single_sample_force_contact_summary(
                wrench=np.zeros(6, dtype=np.float64),
                contact=False,
                substep_tick_count=0,
                source="synthetic_bootstrap_reset",
            ),
        )

    def _seed_fast_source_pose_cache(self, state: RuntimeState | None) -> None:
        if state is None or self._runtime is None:
            return
        try:
            client = self._runtime._client()
        except Exception:
            return
        setter = getattr(client, "set_source_pose_cache", None)
        if not callable(setter):
            return
        try:
            setter(
                entity_name=self._source_entity_name,
                position=np.asarray(state.tcp_pose[:3], dtype=np.float64).tolist(),
                orientation=np.asarray(state.tcp_pose[3:7], dtype=np.float64).tolist(),
            )
        except Exception:
            return

    def connect_existing_world(self) -> RuntimeState:
        if self._runtime is None:
            self._start_runtime()
        deadline = time.monotonic() + self._attach_ready_timeout
        last_error: Exception | None = None
        if self._transport_backend == "cli" and os.environ.get("AIC_GYM_GZ_ENABLE_CLI_BOOTSTRAP") == "1":
            bootstrap_timeout_s = min(max(self._attach_ready_timeout * 0.25, 5.0), 20.0)
            bootstrap_state = self._bootstrap_existing_world_state(timeout_s=bootstrap_timeout_s)
            if bootstrap_state is not None:
                return bootstrap_state
        while time.monotonic() < deadline:
            try:
                observation, info = self._runtime.get_observation()
                self._action[:] = 0.0
                self._last_observation = dict(observation)
                self._last_info = dict(info)
                self._last_state = self._runtime_state_from_observation(observation, info)
                if not self._state_is_sane_for_live_reset(self._last_state):
                    raise RuntimeError("Attached world is not yet exposing a sane live reset pose.")
                return self._last_state
            except RuntimeError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
            except TimeoutError as exc:
                last_error = exc
                self._tick_existing_world_for_sample()
                time.sleep(0.1)
        raise RuntimeError(
            "Timed out waiting for attached world readiness: "
            f"{last_error}. attach_ready_timeout={self._attach_ready_timeout}s, "
            f"transport_backend={self._transport_backend}, source_entity={self._source_entity_name}, "
            f"target_entity={self._target_entity_name}"
        )

    def apply_action(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float64, copy=True)

    def step_ticks(self, tick_count: int) -> RuntimeState:
        if self._runtime is None:
            raise RuntimeError("Live backend must be reset before stepping.")
        tick_count = int(tick_count)
        if tick_count <= 0:
            raise ValueError("tick_count must be positive.")
        if getattr(self, "_state_observation_mode", "honest_live") == "synthetic_training":
            return self._step_ticks_synthetic_training(tick_count)
        step_window_start = time.monotonic()
        used_controller_command = False
        used_pose_command = False
        position_delta = np.clip(self._action[:3], -0.25, 0.25) * 0.002 * tick_count
        rotation_delta = _angular_delta_to_quaternion(
            np.clip(self._action[3:], -2.0, 2.0) * 0.002 * tick_count
        )
        if self._use_controller_velocity_commands and self._ros_observer is not None:
            controller_linear, controller_angular = self._action_twist_for_controller_frame(
                linear_xyz=np.clip(self._action[:3], -0.12, 0.12),
                angular_xyz=np.clip(self._action[3:], -0.5, 0.5),
                frame_id="base_link",
            )
            used_controller_command = self._ros_observer.publish_velocity_command(
                linear_xyz=controller_linear,
                angular_xyz=controller_angular,
                frame_id="base_link",
                timeout_s=min(max(self._timeout, 1.0), 2.0),
            )
        elif self._attach_to_existing and self._ros_observer is not None:
            used_pose_command = self._ros_observer.publish_pose_command(
                position_xyz=position_delta,
                orientation_xyzw=np.asarray(rotation_delta, dtype=np.float64),
                frame_id="gripper/tcp",
                timeout_s=min(max(self._timeout, 1.0), 2.0),
            )
        if used_controller_command or used_pose_command:
            observation, _, _, _, info = self._runtime.step({"multi_step": tick_count})
            if used_controller_command and self._ros_observer is not None:
                stopped = self._ros_observer.publish_velocity_command(
                    linear_xyz=np.zeros(3, dtype=np.float64),
                    angular_xyz=np.zeros(3, dtype=np.float64),
                    frame_id="base_link",
                    timeout_s=min(max(self._timeout, 1.0), 2.0),
                )
                info = dict(info)
                info["sent_controller_velocity_stop"] = bool(stopped)
        else:
            if getattr(self, "_live_mode", "gazebo_training_fast") == "gazebo_training_fast":
                joint_action = self._cartesian_action_to_joint_action(
                    self._action,
                    tick_count=tick_count,
                )
                joint_delta = joint_action * 0.002 * tick_count
                observation, _, _, _, info = self._runtime.step(
                    {
                        "joint_position_delta": joint_delta.tolist(),
                        "multi_step": tick_count,
                    }
                )
            else:
                observation, _, _, _, info = self._runtime.step(
                    {
                        "delta_source_pose": {
                            "position_delta": position_delta.tolist(),
                            "orientation_delta": rotation_delta,
                        },
                        "multi_step": tick_count,
                    }
                )
        step_window_end = time.monotonic()
        self._last_observation = dict(observation)
        self._last_info = dict(info)
        self._last_info["used_controller_velocity_command"] = bool(used_controller_command)
        self._last_info["used_ros_pose_command"] = bool(used_pose_command)
        self._last_state = self._runtime_state_from_observation(
            observation,
            info,
            step_tick_count=tick_count,
            wall_time_window=(step_window_start, step_window_end),
        )
        return self._last_state

    def _cartesian_action_to_joint_action(self, action: np.ndarray, *, tick_count: int) -> np.ndarray:
        """Map world-frame Cartesian plug velocity action to local joint action.

        The VLM / teacher stack emits Cartesian actions in the Gazebo world
        frame. The no-ROS portable Gazebo path accepts only UR5e joint targets,
        so this uses a measured local response matrix around the official
        ground-truth home pose instead of treating xyz as joints 0/1/2.
        Columns are plug xyz displacement, in meters, observed over one
        128-tick command interval for a +0.25 action on each joint channel.
        """
        linear_velocity = np.clip(np.asarray(action[:3], dtype=np.float64), -0.25, 0.25)
        desired_displacement = linear_velocity * 0.002 * float(tick_count)
        if float(np.linalg.norm(desired_displacement)) <= 1e-9:
            joint_action = np.zeros(6, dtype=np.float64)
        else:
            # Positive elbow, wrist_1 and wrist_3 actions are the measured
            # directions that move the held SFP tip toward the board-side SFP
            # port from the official home neighborhood. The full unconstrained
            # pseudo-inverse over all six joints saturates into directions that
            # move away from the target because several joints have opposite
            # local signs under cable load.
            joint_action = np.zeros(6, dtype=np.float64)
            # Elbow-positive is the only measured command that reliably beats
            # the passive cable/gravity drift while moving the SFP tip toward
            # the board (+x, -y, -z). Keep wrist commands small; large wrist
            # combinations dominated the least-squares solution and moved away.
            approach_axis = np.array([0.25, -0.08, -0.86], dtype=np.float64)
            approach_axis /= float(np.linalg.norm(approach_axis))
            approach_amount = float(np.dot(desired_displacement, approach_axis))
            joint_action[2] = np.clip(approach_amount / 0.025, 0.0, 0.25)
            joint_action[5] = np.clip(0.15 * joint_action[2], 0.0, 0.04)
        yaw_action = float(np.clip(action[5], -0.25, 0.25))
        joint_action[5] += 0.25 * yaw_action
        return np.clip(joint_action, -0.25, 0.25)

    def _action_twist_for_controller_frame(
        self,
        *,
        linear_xyz: np.ndarray,
        angular_xyz: np.ndarray,
        frame_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        linear = np.asarray(linear_xyz, dtype=np.float64).copy()
        angular = np.asarray(angular_xyz, dtype=np.float64).copy()
        if frame_id != "base_link":
            return linear, angular
        yaw = self._robot_base_yaw_in_world()
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        world_from_base = np.array(
            [
                [cos_yaw, -sin_yaw, 0.0],
                [sin_yaw, cos_yaw, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_from_world = world_from_base.T
        return base_from_world @ linear, base_from_world @ angular

    def _robot_base_yaw_in_world(self) -> float:
        metadata = getattr(self._scenario, "metadata", {}) if self._scenario is not None else {}
        for key in ("robot_yaw", "robot_base_yaw", "robot_yaw_world"):
            if key in metadata:
                try:
                    return float(metadata[key])
                except (TypeError, ValueError):
                    pass
        # Official aic_gz_bringup default. The gym planner state is in Gazebo
        # world coordinates, while the official Cartesian controller accepts
        # base_link-frame twists.
        return -3.141

    def _step_ticks_synthetic_training(self, tick_count: int) -> RuntimeState:
        if self._runtime is None or self._last_state is None:
            raise RuntimeError("Synthetic training step requires a reset live state.")
        client = self._runtime._client()
        position_delta = np.clip(self._action[:3], -0.25, 0.25) * 0.002 * tick_count
        rotation_delta = _angular_delta_to_quaternion(
            np.clip(self._action[3:], -2.0, 2.0) * 0.002 * tick_count
        )
        try:
            cached_apply = getattr(client, "_apply_delta_source_pose_action_cached", None)
            if callable(cached_apply):
                cached_apply(
                    {
                        "position_delta": position_delta.tolist(),
                        "orientation_delta": rotation_delta,
                    }
                )
            else:
                client._maybe_apply_pose_action(  # type: ignore[attr-defined]
                    {
                        "delta_source_pose": {
                            "position_delta": position_delta.tolist(),
                            "orientation_delta": rotation_delta,
                        }
                    }
                )
            if self._attach_to_existing and self._advance_world_control_cli(tick_count):
                pass
            else:
                client._advance_world(tick_count)  # type: ignore[attr-defined]
            if hasattr(client, "_logical_step_count"):
                client._logical_step_count = int(getattr(client, "_logical_step_count", 0)) + tick_count
        except Exception:
            pass

        previous_state = self._last_state
        sim_dt = 0.002 * float(tick_count)
        tcp_pose = np.asarray(previous_state.tcp_pose, dtype=np.float64).copy()
        tcp_pose[:3] = tcp_pose[:3] + np.clip(self._action[:3], -0.25, 0.25) * sim_dt
        tcp_pose[5] = float(tcp_pose[5] + np.clip(self._action[5], -2.0, 2.0) * sim_dt)
        tcp_velocity = np.concatenate(
            [
                np.clip(self._action[:3], -0.25, 0.25),
                np.clip(self._action[3:], -2.0, 2.0),
            ]
        ).astype(np.float64)
        plug_pose = np.asarray(previous_state.plug_pose, dtype=np.float64).copy()
        synthesized_plug_pose = self._maybe_synthesize_plug_pose(
            tcp_pose=tcp_pose,
            plug_pose=plug_pose,
            controller_state=previous_state.controller_state,
        )
        if synthesized_plug_pose is not None:
            plug_pose = synthesized_plug_pose
        else:
            plug_pose[:3] = plug_pose[:3] + (tcp_pose[:3] - previous_state.tcp_pose[:3])
            plug_pose[5] = float(tcp_pose[5])
        ros_sample = self._ros_observer.snapshot() if self._ros_observer is not None else {}
        wrench, wrench_timestamp = _wrench_from_ros_sample(ros_sample)
        controller_state = _controller_state_from_ros_sample(ros_sample)
        off_limit_contact = bool(ros_sample.get("off_limit_contact", False))
        score_geometry = _build_score_geometry(
            plug_name=str(previous_state.score_geometry.get("plug_name") or "plug"),
            target_name=str(previous_state.score_geometry.get("target_port_name") or "target"),
            entrance_name=str(previous_state.score_geometry.get("port_entrance_name") or "entrance"),
            plug_pose=plug_pose,
            target_pose=previous_state.target_port_pose,
            entrance_pose=previous_state.target_port_entrance_pose,
            tracked_pair={},
        )
        insertion_event, insertion_event_source = self._resolve_live_insertion_event(
            ros_sample=ros_sample,
            geometry=score_geometry,
            allow_geometry_success=False,
        )
        world_entities_summary = dict(previous_state.world_entities_summary or {})
        world_entities_summary["runtime_diagnostics"] = {
            **dict(world_entities_summary.get("runtime_diagnostics") or {}),
            "state_observation_mode": "synthetic_training",
            "insertion_event_source": insertion_event_source,
        }
        next_state = RuntimeState(
            sim_tick=int(previous_state.sim_tick) + tick_count,
            sim_time=float(previous_state.sim_time) + sim_dt,
            joint_positions=np.asarray(previous_state.joint_positions, dtype=np.float64).copy(),
            joint_velocities=np.asarray(previous_state.joint_velocities, dtype=np.float64).copy(),
            gripper_position=float(previous_state.gripper_position),
            tcp_pose=tcp_pose,
            tcp_velocity=tcp_velocity,
            plug_pose=plug_pose,
            target_port_pose=np.asarray(previous_state.target_port_pose, dtype=np.float64).copy(),
            target_port_entrance_pose=(
                None
                if previous_state.target_port_entrance_pose is None
                else np.asarray(previous_state.target_port_entrance_pose, dtype=np.float64).copy()
            ),
            wrench=wrench,
            wrench_timestamp=wrench_timestamp,
            off_limit_contact=off_limit_contact,
            insertion_event=insertion_event,
            controller_state=controller_state,
            score_geometry=score_geometry,
            world_entities_summary=world_entities_summary,
            auxiliary_force_contact_summary=_single_sample_force_contact_summary(
                wrench=wrench,
                contact=off_limit_contact,
                substep_tick_count=tick_count,
                source="synthetic_training_final_sample",
                limitations=(
                    "State is kinematically advanced from the previous step while force/contact remains a real async side channel.",
                ),
            ),
        )
        self._last_state = next_state
        self._last_observation = None
        self._last_info = {
            "used_synthetic_training_state": True,
            "multi_step": tick_count,
        }
        return next_state

    def _advance_world_control_cli(self, tick_count: int) -> bool:
        try:
            completed = subprocess.run(
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
                    "1000",
                    "--req",
                    f"multi_step: {int(tick_count)}",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.5,
            )
        except Exception:
            return False
        return completed.returncode == 0 and "data: true" in completed.stdout

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
        for candidate in self._plug_candidates():
            if isinstance(entities_by_name.get(candidate), dict):
                plug_name = candidate
                break

        tracked = ((observation.get("task_geometry") or {}).get("tracked_entity_pair") or {})
        tcp_name, tcp_position, tcp_orientation = pose_fields(self._source_entity_name)
        plug_name, plug_position, plug_orientation = pose_fields(plug_name)
        target_name, target_position, target_orientation = pose_fields(self._target_entity_name)
        return {
            "live_mode": self._live_mode,
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
        observation_transport = "auto"
        if self._observation_transport_override in {"auto", "one_shot", "persistent"}:
            observation_transport = self._observation_transport_override
        elif self._attach_to_existing and self._transport_backend == "cli":
            observation_transport = "persistent"
        elif (
            not self._attach_to_existing
            and self._live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"}
            and self._transport_backend == "cli"
        ):
            observation_transport = "one_shot"
        helper_startup_timeout_s = 5.0
        helper_startup_settle_s = 3.0
        if not self._attach_to_existing and self._live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"}:
            helper_startup_timeout_s = min(max(float(self._timeout), 5.0), 10.0)
            helper_startup_settle_s = 1.0
        self._runtime = self._runtime_type(
            self._runtime_config_type(
                world_path=self._world_path,
                headless=self._headless,
                timeout=self._timeout,
                world_name=self._world_name,
                source_entity_name=self._source_entity_name,
                target_entity_name=self._target_entity_name,
                transport_backend=self._transport_backend,
                observation_transport=observation_transport,
                helper_startup_timeout_s=helper_startup_timeout_s,
                helper_startup_settle_s=helper_startup_settle_s,
                allow_world_step_on_observation_timeout=not self._attach_to_existing,
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
                if not self._state_is_sane_for_live_reset(self._last_state):
                    raise RuntimeError(
                        "Live observation is still waiting for a sane controller/world pose sample."
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

    def _state_is_sane_for_live_reset(self, state: RuntimeState) -> bool:
        if state.controller_state:
            controller_tcp_pose = state.controller_state.get("tcp_pose")
            if self._is_sane_live_tcp_pose(
                controller_tcp_pose,
                observed_tcp_pose=state.tcp_pose,
                previous_state=self._last_state,
                step_tick_count=0,
            ):
                return True
        return float(state.tcp_pose[2]) > 0.5 and float(state.plug_pose[2]) > 0.2

    def _bootstrap_existing_world_state(self, *, timeout_s: float) -> RuntimeState | None:
        bootstrap_cli_timeout = min(max(float(timeout_s) * 0.2, 3.0), 8.0)
        bootstrap_deadline_s = min(float(timeout_s), bootstrap_cli_timeout + 1.0)
        try:
            client = self._cli_client_type(
                self._cli_config_type(
                    executable="gz",
                    world_path=self._world_path,
                    timeout=bootstrap_cli_timeout,
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
            deadline = time.monotonic() + bootstrap_deadline_s
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
                    if not self._state_is_sane_for_live_reset(self._last_state):
                        raise RuntimeError("Bootstrap observation was not yet a sane live reset pose.")
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
        *,
        step_tick_count: int = 0,
        wall_time_window: tuple[float, float] | None = None,
    ) -> RuntimeState:
        entities_by_name = observation.get("entities_by_name") or {}
        if (
            self._attach_to_existing
            and self._transport_backend == "cli"
            and len(entities_by_name) <= 4
        ):
            supplemented_entities = self._supplement_entities_from_cli_once()
            if supplemented_entities:
                entities_by_name = supplemented_entities
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
        for fallback_name in self._plug_candidates():
            candidate = entities_by_name.get(fallback_name)
            if isinstance(candidate, dict):
                plug = candidate
                plug_name = fallback_name
                break
        source = self._maybe_promote_child_pose_to_world(
            entities_by_name=entities_by_name,
            entity_name=source_name,
            entity_value=source,
            parent_candidates=(
                getattr(self._task, "cable_name", None),
                "cable_0",
                "ur5e",
            ),
        )
        plug = self._maybe_promote_child_pose_to_world(
            entities_by_name=entities_by_name,
            entity_name=plug_name,
            entity_value=plug,
            parent_candidates=(
                getattr(self._task, "cable_name", None),
                "cable_0",
            ),
        )
        target = self._maybe_promote_child_pose_to_world(
            entities_by_name=entities_by_name,
            entity_name=target_name,
            entity_value=target,
            parent_candidates=(
                getattr(self._task, "target_module_name", None),
            ),
        )
        entrance = self._maybe_promote_child_pose_to_world(
            entities_by_name=entities_by_name,
            entity_name=entrance_name,
            entity_value=entrance,
            parent_candidates=(
                getattr(self._task, "target_module_name", None),
            ),
        )
        synthetic_target = self._synthetic_target_pose(
            entities_by_name=entities_by_name,
            observed_target_name=target_name,
            observed_target_value=target,
        )
        synthetic_target_used = False
        target_pose_frame_corrected = False
        if synthetic_target is not None and (
            not isinstance(target, dict)
            or target_name in {"tabletop", "task_board_base_link", self._target_entity_name}
            or bool(synthetic_target.get("replace_observed_local_pose", False))
        ):
            target_name = synthetic_target["target_name"]
            target = synthetic_target["target_pose_dict"]
            target_pose_frame_corrected = bool(
                synthetic_target.get("replace_observed_local_pose", False)
            )
            synthetic_target_used = not target_pose_frame_corrected
            if (
                not isinstance(entrance, dict)
                or bool(synthetic_target.get("replace_observed_local_pose", False))
            ):
                entrance_name = synthetic_target["entrance_name"]
                entrance = synthetic_target["entrance_pose_dict"]
        if not isinstance(target, dict):
            raise RuntimeError(
                "Live observation did not contain the configured target port entity."
            )

        joint_positions = np.asarray(observation.get("joint_positions") or [], dtype=np.float64)
        joint_velocities = np.zeros_like(joint_positions)
        step_count = int(observation.get("step_count") or 0)
        sim_time = _parse_sim_time_seconds(info.get("state_text_raw") or info.get("state_text") or "")
        if sim_time <= 0.0 and step_count > 0:
            sim_time = float(step_count) * 0.002
        tcp_pose = (
            _pose_dict_to_array(source.get("pose") or source)
            if isinstance(source, dict)
            else np.zeros(7, dtype=np.float64)
        )
        target_pose = _pose_dict_to_array(target.get("pose") or target)
        plug_pose = (
            _pose_dict_to_array((plug.get("pose") if isinstance(plug, dict) else None) or plug)
            if isinstance(plug, dict)
            else np.zeros(7, dtype=np.float64)
        )
        missing_real_plug_pose = not isinstance(plug, dict)
        entrance_pose = (
            _pose_dict_to_array(entrance.get("pose") or entrance)
            if isinstance(entrance, dict)
            else None
        )
        ros_sample = self._ros_observer.snapshot() if self._ros_observer is not None else {}
        wrench, wrench_timestamp = _wrench_from_ros_sample(ros_sample)
        controller_state = _controller_state_from_ros_sample(ros_sample)
        previous_state = self._last_state
        controller_tcp_pose = controller_state.get("tcp_pose")
        if self._is_sane_live_tcp_pose(
            controller_tcp_pose,
            observed_tcp_pose=tcp_pose,
            previous_state=previous_state,
            step_tick_count=step_tick_count,
        ):
            tcp_pose = _controller_pose_to_runtime_pose(controller_tcp_pose)
            if source_name is None:
                source_name = "controller_tcp_pose"
        synthesized_tcp_pose = (
            self._maybe_synthesize_tcp_pose(
                observed_tcp_pose=tcp_pose,
                previous_state=previous_state,
                step_tick_count=step_tick_count,
            )
            if self._allow_synthetic_tcp_pose
            else None
        )
        synthesized_tcp_pose_used = synthesized_tcp_pose is not None
        if synthesized_tcp_pose is not None:
            tcp_pose = synthesized_tcp_pose
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
        synthesized_plug_pose = None
        if self._allow_synthetic_plug_pose:
            if missing_real_plug_pose:
                synthesized_plug_pose = self._maybe_synthesize_plug_pose(
                    tcp_pose=tcp_pose,
                    plug_pose=tcp_pose,
                    controller_state=controller_state,
                )
            else:
                synthesized_plug_pose = self._maybe_synthesize_plug_pose(
                    tcp_pose=tcp_pose,
                    plug_pose=plug_pose,
                    controller_state=controller_state,
                )
        synthesized_plug_pose_used = synthesized_plug_pose is not None
        if synthesized_plug_pose is not None:
            plug_pose = synthesized_plug_pose
            if plug_name is None or missing_real_plug_pose:
                plug_name = "synthetic_plug_from_tcp"
        elif missing_real_plug_pose:
            if previous_state is not None:
                plug_pose = np.asarray(previous_state.plug_pose, dtype=np.float64).copy()
                plug_name = plug_name or "previous_plug_pose_fallback"
                synthesized_plug_pose_used = True
            else:
                plug_pose = np.asarray(tcp_pose, dtype=np.float64).copy()
                plug_name = plug_name or "tcp_pose_fallback"
                synthesized_plug_pose_used = True
        if source_name is None and float(tcp_pose[2]) > 0.1:
            source_name = "synthetic_tcp_pose"
        if source_name is None:
            raise RuntimeError(
                f"Live observation did not contain source entity '{self._source_entity_name}', "
                "and no sane controller tcp pose was available."
            )
        off_limit_contact = bool(ros_sample.get("off_limit_contact", False))
        auxiliary_summary = _single_sample_force_contact_summary(
            wrench=wrench,
            contact=off_limit_contact,
            substep_tick_count=step_tick_count,
            source="final_sample_only",
            limitations=(
                []
                if step_tick_count <= 1
                else [
                    "Exact simulator sub-sample history is not exposed on this live path; only the current sample is guaranteed."
                ]
            ),
        )
        if (
            self._ros_observer is not None
            and wall_time_window is not None
            and step_tick_count > 0
        ):
            window_samples = self._ros_observer.history_between(*wall_time_window)
            wrench_samples = [item["wrench"] for item in window_samples["wrench_samples"]]
            timestamps = [float(item["timestamp"]) for item in window_samples["wrench_samples"]]
            contact_indicators = [
                float(item["indicator"]) for item in window_samples["contact_samples"]
            ]
            if wrench_samples or contact_indicators:
                if not wrench_samples:
                    wrench_samples = [wrench.copy()]
                    timestamps = [float(wrench_timestamp)]
                if not contact_indicators:
                    contact_indicators = [1.0 if off_limit_contact else 0.0]
                limitations: list[str] = []
                if window_samples["timestamp_basis"] != "ros_header":
                    limitations.append(
                        "Within-step live aggregation is derived from ROS callback arrival windows rather than exact simulator substeps."
                    )
                auxiliary_summary = _force_contact_summary_from_samples(
                    wrench_samples=wrench_samples,
                    timestamps=timestamps,
                    contact_indicators=contact_indicators,
                    substep_tick_count=step_tick_count,
                    source=window_samples["source"],
                    limitations=limitations,
                )
        score_geometry = _build_score_geometry(
            plug_name=plug_name,
            target_name=target_name,
            entrance_name=entrance_name,
            plug_pose=plug_pose,
            target_pose=target_pose,
            entrance_pose=entrance_pose,
            tracked_pair=observation.get("task_geometry", {}).get("tracked_entity_pair", {}),
        )
        insertion_event, insertion_event_source = self._resolve_live_insertion_event(
            ros_sample=ros_sample,
            geometry=score_geometry,
            allow_geometry_success=(
                not synthesized_tcp_pose_used and not synthesized_plug_pose_used
            ),
        )
        world_entities_summary = _world_entities_summary(
            entities_by_name=entities_by_name,
            source_name=source_name,
            target_name=target_name,
            plug_name=plug_name,
            entrance_name=entrance_name,
        )
        runtime_diagnostics = self._scene_alignment_diagnostics(
            entities_by_name=entities_by_name,
            target_name=target_name,
            synthetic_target_used=synthetic_target_used,
            target_pose_frame_corrected=target_pose_frame_corrected,
            synthesized_tcp_pose_used=synthesized_tcp_pose_used,
            synthesized_plug_pose_used=synthesized_plug_pose_used,
            insertion_event_source=insertion_event_source,
            geometry_success_candidate=self._live_insertion_event_from_geometry(score_geometry) is not None,
        )
        world_entities_summary["runtime_diagnostics"] = runtime_diagnostics
        if self._attach_to_existing and not bool(runtime_diagnostics.get("scene_alignment_ok", False)):
            raise RuntimeError(
                "Attach-mode live scene is not aligned with the sampled task. "
                f"diagnostics={runtime_diagnostics}"
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
            insertion_event=insertion_event,
            controller_state=controller_state,
            score_geometry=score_geometry,
            world_entities_summary=world_entities_summary,
            auxiliary_force_contact_summary=auxiliary_summary,
        )

    def _maybe_promote_child_pose_to_world(
        self,
        *,
        entities_by_name: dict[str, Any],
        entity_name: str | None,
        entity_value: Any,
        parent_candidates: tuple[str | None, ...],
    ) -> Any:
        if not isinstance(entity_value, dict) or not entity_name:
            return entity_value
        try:
            child_pose_like = entity_value.get("pose") or entity_value
            child_pose = _pose_like_to_pose_dict(child_pose_like)
        except Exception:
            return entity_value
        child_position = np.asarray(child_pose["position"], dtype=np.float64)
        # Attached-world observations sometimes expose child-link poses in the
        # parent model frame; promote those into world space before reset sanity
        # checks if a plausible world-space parent pose is available.
        if float(child_position[2]) > 0.5:
            return entity_value
        for parent_name in parent_candidates:
            if not parent_name:
                continue
            parent_value = entities_by_name.get(parent_name)
            if not isinstance(parent_value, dict):
                continue
            try:
                parent_pose = _pose_like_to_pose_dict(parent_value.get("pose") or parent_value)
            except Exception:
                continue
            parent_position = np.asarray(parent_pose["position"], dtype=np.float64)
            if float(parent_position[2]) <= 0.5:
                continue
            world_pose = _compose_pose(parent_pose, child_pose)
            return {
                "position": world_pose["position"].tolist(),
                "orientation": world_pose["orientation"].tolist(),
                "promoted_from_local_frame": True,
                "source_entity_name": entity_name,
                "parent_entity_name": parent_name,
            }
        return entity_value

    def _supplement_entities_from_cli_once(self) -> dict[str, dict[str, Any]] | None:
        try:
            client = self._cli_client_type(
                self._cli_config_type(
                    executable="gz",
                    world_path=self._world_path,
                    timeout=min(max(self._timeout, 1.0), 3.0),
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
            response = client.get_observation(self._get_observation_request_type())
            supplemented = response.observation.get("entities_by_name")
            if isinstance(supplemented, dict) and len(supplemented) > 4:
                return supplemented
        except Exception:
            return None
        finally:
            try:
                client.close()
            except Exception:
                pass
        return None

    def _plug_candidates(self) -> tuple[str, ...]:
        if self._task is None:
            return ("sfp_tip_link", "sfp_module_link", "lc_plug_link", "sc_plug_link", "cable_0")
        task_candidates = [
            f"{self._task.plug_name}_link",
            self._task.plug_name,
        ]
        plug_type = str(getattr(self._task, "plug_type", ""))
        plug_name = str(getattr(self._task, "plug_name", ""))
        if plug_type == "sfp" or plug_name.startswith("sfp"):
            task_candidates.extend(["sfp_tip_link", "sfp_module_link"])
        elif plug_type == "lc" or plug_name.startswith("lc"):
            task_candidates.append("lc_plug_link")
        elif plug_type == "sc" or plug_name.startswith("sc"):
            task_candidates.append("sc_plug_link")
        task_candidates.extend(["lc_plug_link", "sc_plug_link", "sfp_module_link", "cable_0"])
        return tuple(dict.fromkeys(task_candidates))

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

    def _synthetic_target_pose(
        self,
        *,
        entities_by_name: dict[str, dict[str, Any]],
        observed_target_name: str | None,
        observed_target_value: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._task is None or self._scenario is None:
            return None
        replace_observed_local_pose = False
        if observed_target_name and observed_target_name not in {
            self._target_entity_name,
            "tabletop",
            "task_board_base_link",
        }:
            if not self._observed_target_pose_looks_local_frame(
                entities_by_name=entities_by_name,
                observed_target_name=observed_target_name,
                observed_target_value=observed_target_value,
            ):
                return None
            replace_observed_local_pose = True

        module_pose = None
        module_entity = entities_by_name.get(self._task.target_module_name)
        if isinstance(module_entity, dict):
            module_pose = _pose_dict_to_full_pose(module_entity.get("pose") or module_entity)
            if (
                module_pose is not None
                and self._attach_to_existing
                and float(np.asarray(module_pose["position"], dtype=np.float64)[2]) <= 0.5
            ):
                module_pose = None
        if module_pose is None and not self._attach_to_existing:
            module_pose = _target_module_pose_from_scenario(
                scenario=self._scenario,
                task=self._task,
            )
        if module_pose is None and replace_observed_local_pose:
            module_pose = _target_module_pose_from_scenario(
                scenario=self._scenario,
                task=self._task,
            )
        if module_pose is None:
            return None

        local_target, local_entrance = _local_target_and_entrance_for_task(self._task)
        if local_target is None:
            return None
        target_pose = _compose_pose(module_pose, local_target)
        entrance_pose = _compose_pose(target_pose, local_entrance) if local_entrance is not None else None
        return {
            "target_name": f"{self._task.target_module_name}::{self._task.port_name}_synthetic",
            "entrance_name": (
                None
                if entrance_pose is None
                else f"{self._task.target_module_name}::{self._task.port_name}_entrance_synthetic"
            ),
            "target_pose_dict": _full_pose_to_pose_dict(target_pose),
            "entrance_pose_dict": None if entrance_pose is None else _full_pose_to_pose_dict(entrance_pose),
            "replace_observed_local_pose": replace_observed_local_pose,
        }

    def _observed_target_pose_looks_local_frame(
        self,
        *,
        entities_by_name: dict[str, dict[str, Any]],
        observed_target_name: str | None,
        observed_target_value: dict[str, Any] | None,
    ) -> bool:
        if not observed_target_name or not isinstance(observed_target_value, dict):
            return False
        try:
            target_pose = _pose_like_to_pose_dict(
                observed_target_value.get("pose") or observed_target_value
            )
        except Exception:
            return False
        target_position = np.asarray(target_pose["position"], dtype=np.float64)
        if float(target_position[2]) > 0.5:
            return False
        board_entity = entities_by_name.get("task_board") or entities_by_name.get("task_board_base_link")
        if not isinstance(board_entity, dict):
            return False
        try:
            board_pose = _pose_like_to_pose_dict(board_entity.get("pose") or board_entity)
        except Exception:
            return False
        board_position = np.asarray(board_pose["position"], dtype=np.float64)
        if float(board_position[2]) <= 0.5:
            return False
        expected_port_prefix = f"{getattr(self._task, 'port_name', '')}_link"
        return observed_target_name.startswith(expected_port_prefix)

    def _maybe_synthesize_plug_pose(
        self,
        *,
        tcp_pose: np.ndarray,
        plug_pose: np.ndarray,
        controller_state: dict[str, Any],
    ) -> np.ndarray | None:
        if self._task is None or self._scenario is None:
            return None
        cable = self._scenario.cables.get(self._task.cable_name)
        if cable is None:
            return None
        synthesized = np.asarray(tcp_pose, dtype=np.float64).copy()
        synthesized[:3] = synthesized[:3] + np.asarray(cable.gripper_offset_xyz, dtype=np.float64)
        position_gap = float(
            np.linalg.norm(
                np.asarray(plug_pose[:3], dtype=np.float64) - np.asarray(synthesized[:3], dtype=np.float64)
            )
        )
        if position_gap <= 0.08 and float(plug_pose[2]) > 0.5:
            return None
        return synthesized

    def _maybe_synthesize_tcp_pose(
        self,
        *,
        observed_tcp_pose: np.ndarray,
        previous_state: RuntimeState | None,
        step_tick_count: int,
    ) -> np.ndarray | None:
        if previous_state is None or step_tick_count <= 0:
            return None
        if float(observed_tcp_pose[2]) > 0.5:
            expected = np.asarray(previous_state.tcp_pose, dtype=np.float64).copy()
            delta_t = 0.002 * float(step_tick_count)
            expected[:3] = expected[:3] + np.clip(self._action[:3], -0.25, 0.25) * delta_t
            expected[5] = float(expected[5] + np.clip(self._action[5], -2.0, 2.0) * delta_t)
            if float(np.linalg.norm(observed_tcp_pose[:3] - expected[:3])) <= 0.08:
                return None
        synthesized = np.asarray(previous_state.tcp_pose, dtype=np.float64).copy()
        delta_t = 0.002 * float(step_tick_count)
        synthesized[:3] = synthesized[:3] + np.clip(self._action[:3], -0.25, 0.25) * delta_t
        synthesized[5] = float(synthesized[5] + np.clip(self._action[5], -2.0, 2.0) * delta_t)
        return synthesized

    def _is_sane_live_tcp_pose(
        self,
        candidate_pose: Any,
        *,
        observed_tcp_pose: np.ndarray,
        previous_state: RuntimeState | None,
        step_tick_count: int,
    ) -> bool:
        if not isinstance(candidate_pose, np.ndarray) or candidate_pose.shape[0] < 7:
            return False
        candidate = np.asarray(candidate_pose, dtype=np.float64)
        if float(candidate[2]) <= 0.5:
            return False
        if previous_state is not None and step_tick_count > 0:
            expected = np.asarray(previous_state.tcp_pose, dtype=np.float64).copy()
            delta_t = 0.002 * float(step_tick_count)
            expected[:3] = expected[:3] + np.clip(self._action[:3], -0.25, 0.25) * delta_t
            expected[5] = float(expected[5] + np.clip(self._action[5], -2.0, 2.0) * delta_t)
            return float(np.linalg.norm(candidate[:3] - expected[:3])) <= 0.08
        if float(observed_tcp_pose[2]) > 0.5:
            return float(np.linalg.norm(candidate[:3] - observed_tcp_pose[:3])) <= 0.08
        return True

    def _live_insertion_event_from_geometry(
        self,
        geometry: dict[str, Any],
    ) -> str | None:
        if self._task is None:
            return None
        target_distance = float(geometry.get("distance_to_target", np.inf))
        lateral_misalignment = float(geometry.get("lateral_misalignment", np.inf))
        insertion_progress = float(geometry.get("insertion_progress", 0.0))
        if (
            target_distance <= 0.003
            and lateral_misalignment <= 0.005
            and insertion_progress >= 0.95
        ):
            return f"{self._task.target_module_name}/{self._task.port_name}"
        return None

    def _resolve_live_insertion_event(
        self,
        *,
        ros_sample: dict[str, Any],
        geometry: dict[str, Any],
        allow_geometry_success: bool = False,
    ) -> tuple[str | None, str]:
        official_event = ros_sample.get("official_insertion_event")
        if isinstance(official_event, str) and official_event.strip():
            return official_event.strip(), "official_topic"
        if allow_geometry_success:
            geometry_event = self._live_insertion_event_from_geometry(geometry)
            if geometry_event is not None:
                return geometry_event, "world_frame_geometry"
        return None, "none"

    def _scene_alignment_diagnostics(
        self,
        *,
        entities_by_name: dict[str, dict[str, Any]],
        target_name: str | None,
        synthetic_target_used: bool,
        target_pose_frame_corrected: bool,
        synthesized_tcp_pose_used: bool,
        synthesized_plug_pose_used: bool,
        insertion_event_source: str,
        geometry_success_candidate: bool,
    ) -> dict[str, Any]:
        expected_target_name = (
            None if self._task is None else f"{self._task.target_module_name}::{self._task.port_name}_link"
        )
        expected_module_name = None if self._task is None else self._task.target_module_name
        expected_cable_name = None
        if self._task is not None and self._scenario is not None:
            cable = self._scenario.cables.get(self._task.cable_name)
            if cable is not None:
                expected_cable_name = cable.cable_name
        expected_entities_present = {
            "task_board": "task_board" in entities_by_name or "task_board_base_link" in entities_by_name,
            "tabletop": "tabletop" in entities_by_name,
            "target_module": bool(expected_module_name and expected_module_name in entities_by_name),
            "cable": bool(expected_cable_name and expected_cable_name in entities_by_name),
        }
        scene_alignment_ok = (
            all(expected_entities_present.values())
            and not synthetic_target_used
            and (
                target_pose_frame_corrected
                or target_name not in {"tabletop", "task_board_base_link", self._target_entity_name}
            )
        )
        return {
            "attach_to_existing": bool(self._attach_to_existing),
            "live_mode": self._live_mode,
            "expected_target_name": expected_target_name,
            "resolved_target_name": target_name,
            "expected_module_name": expected_module_name,
            "expected_cable_name": expected_cable_name,
            "expected_entities_present": expected_entities_present,
            "synthetic_target_used": bool(synthetic_target_used),
            "target_pose_frame_corrected": bool(target_pose_frame_corrected),
            "synthetic_tcp_pose_used": bool(synthesized_tcp_pose_used),
            "synthetic_plug_pose_used": bool(synthesized_plug_pose_used),
            "insertion_event_source": insertion_event_source,
            "geometry_success_candidate": bool(geometry_success_candidate),
            "scene_alignment_ok": bool(scene_alignment_ok),
        }


class MockStepperBackend(RuntimeBackend):
    """Deterministic backend used for tests and the random-policy demo."""

    def __init__(
        self,
        *,
        sim_dt: float = 0.002,
        transient_contact_config: MockTransientContactConfig | None = None,
    ) -> None:
        self._sim_dt = sim_dt
        self._rng = np.random.default_rng(0)
        self._state: RuntimeState | None = None
        self._action = np.zeros(6, dtype=np.float64)
        self._scenario: AicScenario | None = None
        self._transient_contact_config = transient_contact_config or MockTransientContactConfig()

    def reset(self, *, seed: int | None, scenario: AicScenario) -> RuntimeState:
        self._rng = np.random.default_rng(seed if seed is not None else 0)
        self._scenario = scenario
        task = next(iter(scenario.tasks.values()))
        synthetic_target = _synthetic_target_pose_from_scenario(scenario=scenario, task=task)
        if synthetic_target is None:
            raise RuntimeError(
                f"Could not synthesize target pose for task {task.target_module_name}/{task.port_name}."
            )
        target_pose = _pose_dict_to_array(synthetic_target["target_pose_dict"])
        tcp_pose = np.array([-0.45, 0.2, 1.30, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        cable = scenario.cables[task.cable_name]
        plug_pose = tcp_pose.copy()
        plug_pose[:3] += np.array(cable.gripper_offset_xyz, dtype=np.float64)
        entrance_pose = (
            _pose_dict_to_array(synthetic_target["entrance_pose_dict"])
            if synthetic_target["entrance_pose_dict"] is not None
            else None
        )
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
            auxiliary_force_contact_summary=_single_sample_force_contact_summary(
                wrench=np.zeros(6, dtype=np.float64),
                contact=False,
                substep_tick_count=0,
                source="mock_reset",
            ),
        )
        return self._state

    def apply_action(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float64, copy=True)

    def step_ticks(self, tick_count: int) -> RuntimeState:
        if self._state is None:
            raise RuntimeError("Backend must be reset before stepping.")
        if self._scenario is None:
            raise RuntimeError("Backend must retain a scenario before stepping.")
        task = next(iter(self._scenario.tasks.values()))
        tick_count = int(tick_count)
        if tick_count <= 0:
            raise ValueError("tick_count must be positive.")

        state = self._state
        wrench_samples: list[np.ndarray] = []
        timestamps: list[float] = []
        contact_indicators: list[float] = []
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
                insertion_event = f"{task.target_module_name}/{task.port_name}"
            off_limit_contact = bool(next_tcp_pose[2] < 1.0)
            force_mag = max(0.0, (0.02 - distance) * 1800.0)
            torque_mag = 0.0
            contact_band = self._transient_contact_config.contact_band_z
            if contact_band is not None:
                band_lo, band_hi = sorted((float(contact_band[0]), float(contact_band[1])))
                band_center = 0.5 * (band_lo + band_hi)
                half_width = max(0.5 * (band_hi - band_lo), 1e-6)
                band_overlap = max(0.0, 1.0 - abs(float(next_tcp_pose[2]) - band_center) / half_width)
                if band_overlap > 0.0:
                    off_limit_contact = True
                    speed_scale = 1.0 + min(abs(float(linear[2])) / 0.25, 1.0)
                    force_mag = max(
                        force_mag,
                        self._transient_contact_config.peak_force_newtons * band_overlap * speed_scale,
                    )
                    torque_mag = (
                        self._transient_contact_config.peak_torque_newton_meters
                        * band_overlap
                        * speed_scale
                    )
            wrench = np.array([0.0, 0.0, force_mag, 0.0, torque_mag, 0.0], dtype=np.float64)
            sample_timestamp = state.sim_time + self._sim_dt
            wrench_samples.append(wrench.copy())
            timestamps.append(sample_timestamp)
            contact_indicators.append(1.0 if off_limit_contact else 0.0)

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
                wrench_timestamp=sample_timestamp,
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
                auxiliary_force_contact_summary=_force_contact_summary_from_samples(
                    wrench_samples=wrench_samples,
                    timestamps=timestamps,
                    contact_indicators=contact_indicators,
                    substep_tick_count=tick_count,
                    source="mock_substeps_exact",
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
                "transient_contact_config": {
                    "contact_band_z": None
                    if self._transient_contact_config.contact_band_z is None
                    else list(self._transient_contact_config.contact_band_z),
                    "peak_force_newtons": self._transient_contact_config.peak_force_newtons,
                    "peak_torque_newton_meters": (
                        self._transient_contact_config.peak_torque_newton_meters
                    ),
                },
            },
        )

    def restore_checkpoint(self, checkpoint: RuntimeCheckpoint) -> RuntimeState:
        if checkpoint.mode != "mock_exact":
            raise ValueError(f"Unsupported checkpoint mode for MockStepperBackend: {checkpoint.mode}")
        self._state = _deserialize_runtime_state(checkpoint.payload["state"])
        self._action = np.asarray(checkpoint.payload.get("action", np.zeros(6)), dtype=np.float64)
        self._rng = np.random.default_rng()
        self._rng.bit_generator.state = checkpoint.payload["rng_state"]
        transient_config = checkpoint.payload.get("transient_contact_config", {})
        self._transient_contact_config = MockTransientContactConfig(
            contact_band_z=(
                None
                if transient_config.get("contact_band_z") is None
                else tuple(transient_config["contact_band_z"])
            ),
            peak_force_newtons=float(transient_config.get("peak_force_newtons", 30.0)),
            peak_torque_newton_meters=float(
                transient_config.get("peak_torque_newton_meters", 3.0)
            ),
        )
        return self._state


class _RuntimeRosObserver:
    """Best-effort ROS side observer for score-critical live fields.

    This is additive to the Gazebo transport observation path. It improves
    parity for wrench, controller_state, and off-limit-contact semantics but
    still does not turn the gym runtime into the official evaluation path.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._wrench_history: deque[dict[str, Any]] = deque(maxlen=4096)
        self._contact_history: deque[dict[str, Any]] = deque(maxlen=4096)
        self._lock = None
        self._stop_event = None
        self._thread = None
        self._ready = False
        self._command_queue: deque[dict[str, Any]] = deque()
        self._router_process = None
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

    def history_between(self, start_time: float, end_time: float) -> dict[str, Any]:
        if not self._ready or self._lock is None:
            return {
                "wrench_samples": [],
                "contact_samples": [],
                "timestamp_basis": "unavailable",
                "source": "final_sample_only",
            }
        with self._lock:
            wrench_samples = [
                {
                    "wrench": item["wrench"].copy(),
                    "timestamp": float(item["timestamp"]),
                }
                for item in self._wrench_history
                if start_time <= float(item["wall_time"]) <= end_time
            ]
            contact_samples = [
                {
                    "indicator": float(item["indicator"]),
                    "timestamp": float(item["timestamp"]),
                }
                for item in self._contact_history
                if start_time <= float(item["wall_time"]) <= end_time
            ]
            timestamp_basis = "ros_header" if wrench_samples else "wall_time"
        return {
            "wrench_samples": wrench_samples,
            "contact_samples": contact_samples,
            "timestamp_basis": timestamp_basis,
            "source": "ros_callback_window",
        }

    def close(self) -> None:
        if not self._ready or self._stop_event is None:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        router_process = self._router_process
        self._router_process = None
        if router_process is not None and router_process.poll() is None:
            router_process.terminate()
            try:
                router_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                router_process.kill()

    def publish_velocity_command(
        self,
        *,
        linear_xyz: np.ndarray,
        angular_xyz: np.ndarray,
        frame_id: str = "base_link",
        timeout_s: float = 2.0,
    ) -> bool:
        if not self._ready or self._lock is None:
            return False
        import threading

        request = {
            "linear_xyz": np.asarray(linear_xyz, dtype=np.float64).copy(),
            "angular_xyz": np.asarray(angular_xyz, dtype=np.float64).copy(),
            "frame_id": str(frame_id),
            "done": threading.Event(),
            "ok": False,
            "error": None,
        }
        with self._lock:
            self._command_queue.append(request)
        if not request["done"].wait(timeout_s):
            return False
        return bool(request["ok"])

    def publish_pose_command(
        self,
        *,
        position_xyz: np.ndarray,
        orientation_xyzw: np.ndarray,
        frame_id: str = "gripper/tcp",
        timeout_s: float = 2.0,
    ) -> bool:
        if not self._ready or self._lock is None:
            return False
        import threading

        request = {
            "mode": "position",
            "position_xyz": np.asarray(position_xyz, dtype=np.float64).copy(),
            "orientation_xyzw": np.asarray(orientation_xyzw, dtype=np.float64).copy(),
            "frame_id": str(frame_id),
            "done": threading.Event(),
            "ok": False,
            "error": None,
        }
        with self._lock:
            self._command_queue.append(request)
        if not request["done"].wait(timeout_s):
            return False
        return bool(request["ok"])

    def _spin(self) -> None:
        import threading

        _configure_ros_eval_session_env()
        self._maybe_start_zenoh_router()
        import rclpy
        from geometry_msgs.msg import Twist, Vector3
        from geometry_msgs.msg import WrenchStamped
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from ros_gz_interfaces.msg import Contacts
        from std_msgs.msg import String

        from aic_control_interfaces.msg import ControllerState, MotionUpdate, TrajectoryGenerationMode

        context = Context()
        rclpy.init(context=context)
        node = Node("aic_gym_gz_runtime_observer", context=context)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        motion_pub = node.create_publisher(
            MotionUpdate,
            "/aic_controller/pose_commands",
            10,
        )

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
            wall_time = time.monotonic()
            wrench, timestamp = _wrench_from_ros_sample({"wrench": message})
            assert self._lock is not None
            with self._lock:
                self._data["wrench"] = message
                self._wrench_history.append(
                    {
                        "wrench": wrench,
                        "timestamp": timestamp if timestamp > 0.0 else wall_time,
                        "wall_time": wall_time,
                    }
                )

        def controller_callback(message: ControllerState) -> None:
            set_value("controller_state", message)

        def contact_callback(message: Contacts) -> None:
            active = bool(getattr(message, "contacts", []) or getattr(message, "states", []))
            wall_time = time.monotonic()
            timestamp = _message_stamp_seconds(message)
            assert self._lock is not None
            with self._lock:
                self._data["off_limit_contact"] = active
                self._contact_history.append(
                    {
                        "indicator": 1.0 if active else 0.0,
                        "timestamp": timestamp if timestamp is not None else wall_time,
                        "wall_time": wall_time,
                    }
                )

        def insertion_event_callback(message: String) -> None:
            payload = str(getattr(message, "data", "")).strip()
            if not payload:
                return
            set_value("official_insertion_event", payload)

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
        node.create_subscription(
            String,
            "/scoring/insertion_event",
            insertion_event_callback,
            10,
        )
        while rclpy.ok(context=context) and self._stop_event is not None and not self._stop_event.is_set():
            pending: dict[str, Any] | None = None
            assert self._lock is not None
            with self._lock:
                if self._command_queue:
                    pending = self._command_queue.popleft()
            if pending is not None:
                try:
                    message = MotionUpdate()
                    message.header.stamp = node.get_clock().now().to_msg()
                    message.header.frame_id = str(pending["frame_id"])
                    if str(pending.get("mode", "velocity")) == "position":
                        position_xyz = np.asarray(pending["position_xyz"], dtype=np.float64)
                        orientation_xyzw = np.asarray(pending["orientation_xyzw"], dtype=np.float64)
                        message.pose = Pose(
                            position=Point(
                                x=float(position_xyz[0]),
                                y=float(position_xyz[1]),
                                z=float(position_xyz[2]),
                            ),
                            orientation=Quaternion(
                                x=float(orientation_xyzw[0]),
                                y=float(orientation_xyzw[1]),
                                z=float(orientation_xyzw[2]),
                                w=float(orientation_xyzw[3]),
                            ),
                        )
                        message.trajectory_generation_mode.mode = (
                            TrajectoryGenerationMode.MODE_POSITION
                        )
                    else:
                        linear_xyz = np.asarray(pending["linear_xyz"], dtype=np.float64)
                        angular_xyz = np.asarray(pending["angular_xyz"], dtype=np.float64)
                        message.velocity = Twist(
                            linear=Vector3(
                                x=float(linear_xyz[0]),
                                y=float(linear_xyz[1]),
                                z=float(linear_xyz[2]),
                            ),
                            angular=Vector3(
                                x=float(angular_xyz[0]),
                                y=float(angular_xyz[1]),
                                z=float(angular_xyz[2]),
                            ),
                        )
                        message.trajectory_generation_mode.mode = (
                            TrajectoryGenerationMode.MODE_VELOCITY
                        )
                    message.target_stiffness = np.diag([75.0] * 6).flatten().tolist()
                    message.target_damping = np.diag([35.0] * 6).flatten().tolist()
                    message.wrench_feedback_gains_at_tip = [0.0] * 6
                    if motion_pub.get_subscription_count() <= 0:
                        raise RuntimeError("No /aic_controller/pose_commands subscriber is ready.")
                    motion_pub.publish(message)
                    pending["ok"] = True
                except Exception as exc:
                    pending["error"] = exc
                    pending["ok"] = False
                finally:
                    pending["done"].set()
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

    def _maybe_start_zenoh_router(self) -> None:
        if self._router_process is not None:
            return
        try:
            existing = subprocess.run(
                ["bash", "-lc", "pgrep -f 'rmw_zenohd' >/dev/null"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
            if existing.returncode == 0:
                return
        except Exception:
            return
        router_executable = shutil.which("rmw_zenohd")
        router_command: list[str] | None = None
        if router_executable is not None:
            router_command = [router_executable]
        else:
            ros2_executable = shutil.which("ros2")
            if ros2_executable is not None:
                router_command = [ros2_executable, "run", "rmw_zenoh_cpp", "rmw_zenohd"]
        if router_command is None:
            return
        try:
            self._router_process = subprocess.Popen(
                router_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
        except Exception:
            self._router_process = None


def _resolve_named_entity(
    entities_by_name: dict[str, dict[str, Any]],
    *candidates: str,
) -> tuple[str | None, dict[str, Any] | None]:
    for candidate in candidates:
        entity = entities_by_name.get(candidate)
        if isinstance(entity, dict):
            return candidate, entity
    return None, None


def _message_stamp_seconds(message: Any) -> float | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return float(sec) + float(nanosec) * 1e-9


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
    timestamp = _message_stamp_seconds(wrench_msg) or 0.0
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
    tracked_orientation_error = tracked_pair.get("orientation_error")
    yaw_orientation_error = abs(_wrap_to_pi(float(plug_pose[5] - target_pose[5])))
    geometry: dict[str, Any] = {
        "plug_name": plug_name,
        "target_port_name": target_name,
        "port_entrance_name": entrance_name,
        "distance_to_target": target_distance,
        "tracked_distance": tracked_pair.get("distance"),
        "tracked_orientation_error": tracked_orientation_error,
        "orientation_error": yaw_orientation_error,
        "orientation_error_source": "plug_target_yaw_in_runtime_world",
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


def _serialize_auxiliary_force_contact_summary(
    summary: AuxiliaryForceContactSummary,
) -> dict[str, Any]:
    return {
        "is_official_observation": bool(summary.is_official_observation),
        "source": summary.source,
        "substep_tick_count": int(summary.substep_tick_count),
        "sample_count": int(summary.sample_count),
        "wrench_current": summary.wrench_current.tolist(),
        "wrench_max_abs_recent": summary.wrench_max_abs_recent.tolist(),
        "wrench_mean_recent": summary.wrench_mean_recent.tolist(),
        "wrench_max_force_abs_recent": float(summary.wrench_max_force_abs_recent),
        "wrench_max_torque_abs_recent": float(summary.wrench_max_torque_abs_recent),
        "had_contact_recent": bool(summary.had_contact_recent),
        "max_contact_indicator_recent": float(summary.max_contact_indicator_recent),
        "first_wrench_recent": summary.first_wrench_recent.tolist(),
        "last_wrench_recent": summary.last_wrench_recent.tolist(),
        "time_of_peak_within_step": summary.time_of_peak_within_step,
        "limitations": list(summary.limitations),
    }


def _deserialize_auxiliary_force_contact_summary(
    payload: dict[str, Any],
) -> AuxiliaryForceContactSummary:
    if not payload:
        return AuxiliaryForceContactSummary()
    return AuxiliaryForceContactSummary(
        is_official_observation=bool(payload.get("is_official_observation", False)),
        source=str(payload.get("source", "final_sample_only")),
        substep_tick_count=int(payload.get("substep_tick_count", 0)),
        sample_count=int(payload.get("sample_count", 0)),
        wrench_current=np.asarray(payload.get("wrench_current", np.zeros(6)), dtype=np.float64),
        wrench_max_abs_recent=np.asarray(
            payload.get("wrench_max_abs_recent", np.zeros(6)),
            dtype=np.float64,
        ),
        wrench_mean_recent=np.asarray(payload.get("wrench_mean_recent", np.zeros(6)), dtype=np.float64),
        wrench_max_force_abs_recent=float(payload.get("wrench_max_force_abs_recent", 0.0)),
        wrench_max_torque_abs_recent=float(payload.get("wrench_max_torque_abs_recent", 0.0)),
        had_contact_recent=bool(payload.get("had_contact_recent", False)),
        max_contact_indicator_recent=float(payload.get("max_contact_indicator_recent", 0.0)),
        first_wrench_recent=np.asarray(payload.get("first_wrench_recent", np.zeros(6)), dtype=np.float64),
        last_wrench_recent=np.asarray(payload.get("last_wrench_recent", np.zeros(6)), dtype=np.float64),
        time_of_peak_within_step=(
            None
            if payload.get("time_of_peak_within_step") is None
            else float(payload["time_of_peak_within_step"])
        ),
        limitations=tuple(str(item) for item in payload.get("limitations", [])),
    )


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
        "auxiliary_force_contact_summary": _serialize_auxiliary_force_contact_summary(
            state.auxiliary_force_contact_summary
        ),
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
        auxiliary_force_contact_summary=_deserialize_auxiliary_force_contact_summary(
            payload.get("auxiliary_force_contact_summary", {})
        ),
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


def _controller_pose_to_runtime_pose(pose: Any) -> np.ndarray:
    """Convert controller [x,y,z,qx,qy,qz,qw] pose to runtime yaw pose."""
    array = np.asarray(pose, dtype=np.float64)
    if array.shape[0] < 7:
        raise ValueError("Controller pose must contain position plus quaternion.")
    quaternion = array[3:7]
    yaw = _quaternion_to_yaw(quaternion)
    return np.array(
        [
            float(array[0]),
            float(array[1]),
            float(array[2]),
            0.0,
            0.0,
            float(yaw),
            float(quaternion[3]),
        ],
        dtype=np.float64,
    )


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _pose_like_to_pose_dict(pose_like: dict[str, Any]) -> dict[str, np.ndarray]:
    position = np.asarray(pose_like.get("position") or [0.0, 0.0, 0.0], dtype=np.float64)
    orientation = np.asarray(pose_like.get("orientation") or [0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return {
        "position": position,
        "orientation": orientation,
    }


def _pose_dict_to_full_pose(pose_like: dict[str, Any]) -> dict[str, np.ndarray] | None:
    position = pose_like.get("position") or [0.0, 0.0, 0.0]
    orientation = pose_like.get("orientation") or [0.0, 0.0, 0.0, 1.0]
    if len(position) < 3 or len(orientation) < 4:
        return None
    return {
        "position": np.asarray(position[:3], dtype=np.float64),
        "orientation": np.asarray(orientation[:4], dtype=np.float64),
    }


def _full_pose_to_pose_dict(pose: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {
        "position": [float(value) for value in pose["position"].tolist()],
        "orientation": [float(value) for value in pose["orientation"].tolist()],
    }


def _rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1.tolist()
    x2, y2, z2, w2 = q2.tolist()
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quaternion(vector: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion.tolist()
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation @ vector


def _compose_pose(parent_pose: dict[str, np.ndarray], child_pose: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    parent_position = np.asarray(parent_pose["position"], dtype=np.float64)
    parent_orientation = np.asarray(parent_pose["orientation"], dtype=np.float64)
    child_position = np.asarray(child_pose["position"], dtype=np.float64)
    child_orientation = np.asarray(child_pose["orientation"], dtype=np.float64)
    return {
        "position": parent_position + _rotate_vector_by_quaternion(child_position, parent_orientation),
        "orientation": _quaternion_multiply(parent_orientation, child_orientation),
    }


def _target_module_pose_from_scenario(
    *,
    scenario: AicScenario,
    task: Any,
) -> dict[str, np.ndarray] | None:
    board_x, board_y, board_z, board_roll, board_pitch, board_yaw = scenario.task_board.pose_xyz_rpy
    board_pose = {
        "position": np.array([board_x, board_y, board_z], dtype=np.float64),
        "orientation": _rpy_to_quaternion(board_roll, board_pitch, board_yaw),
    }
    module_pose = _module_pose_in_board_frame(scenario=scenario, task=task)
    if module_pose is None:
        return None
    return _compose_pose(board_pose, module_pose)


def _synthetic_target_pose_from_scenario(
    *,
    scenario: AicScenario,
    task: Any,
) -> dict[str, Any] | None:
    module_pose = _target_module_pose_from_scenario(scenario=scenario, task=task)
    if module_pose is None:
        return None
    local_target, local_entrance = _local_target_and_entrance_for_task(task)
    if local_target is None:
        return None
    target_pose = _compose_pose(module_pose, local_target)
    entrance_pose = _compose_pose(target_pose, local_entrance) if local_entrance is not None else None
    return {
        "target_name": f"{task.target_module_name}::{task.port_name}_synthetic",
        "entrance_name": (
            None if entrance_pose is None else f"{task.target_module_name}::{task.port_name}_entrance_synthetic"
        ),
        "target_pose_dict": _full_pose_to_pose_dict(target_pose),
        "entrance_pose_dict": None if entrance_pose is None else _full_pose_to_pose_dict(entrance_pose),
    }


def _module_pose_in_board_frame(*, scenario: AicScenario, task: Any) -> dict[str, np.ndarray] | None:
    target_module_name = str(task.target_module_name)
    if target_module_name.startswith("nic_card_mount_"):
        try:
            index = int(target_module_name.rsplit("_", 1)[-1])
        except ValueError:
            return None
        rail = scenario.task_board.nic_rails.get(f"nic_rail_{index}")
        if rail is None:
            return None
        base_y_by_index = {
            0: -0.1745,
            1: -0.1345,
            2: -0.0945,
            3: -0.0545,
            4: -0.0145,
        }
        if index not in base_y_by_index:
            return None
        return {
            "position": np.array(
                [
                    -0.081418 + float(rail.translation),
                    base_y_by_index[index],
                    0.012,
                ],
                dtype=np.float64,
            ),
            "orientation": _rpy_to_quaternion(float(rail.roll), float(rail.pitch), float(rail.yaw)),
        }
    if target_module_name.startswith("sc_port_"):
        try:
            index = int(target_module_name.rsplit("_", 1)[-1])
        except ValueError:
            return None
        rail = scenario.task_board.sc_rails.get(f"sc_rail_{index}")
        if rail is None:
            return None
        base_y_by_index = {
            0: 0.0295,
            1: 0.0705,
        }
        if index not in base_y_by_index:
            return None
        return {
            "position": np.array(
                [
                    -0.075 + float(rail.translation),
                    base_y_by_index[index],
                    0.0165,
                ],
                dtype=np.float64,
            ),
            "orientation": _rpy_to_quaternion(
                1.57 + float(rail.roll),
                float(rail.pitch),
                1.57 + float(rail.yaw),
            ),
        }
    return None


def _local_target_and_entrance_for_task(task: Any) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None]:
    target_module_name = str(task.target_module_name)
    port_name = str(task.port_name)
    if target_module_name.startswith("nic_card_mount_"):
        nic_card_pose = {
            "position": np.array([-0.002, -0.01785, 0.0899], dtype=np.float64),
            "orientation": _rpy_to_quaternion(-1.57, 0.0, 0.0),
        }
        if port_name == "sfp_port_0":
            port_pose = {
                "position": np.array([0.01295, -0.031572, 0.00501], dtype=np.float64),
                "orientation": _rpy_to_quaternion(4.69895, 0.0, 0.0),
            }
        elif port_name == "sfp_port_1":
            port_pose = {
                "position": np.array([-0.01025, -0.031572, 0.00501], dtype=np.float64),
                "orientation": _rpy_to_quaternion(4.69895, 0.0, 0.0),
            }
        else:
            return None, None
        target_pose = _compose_pose(nic_card_pose, port_pose)
        entrance_pose = {
            "position": np.array([0.0, 0.0, -0.0458], dtype=np.float64),
            "orientation": _rpy_to_quaternion(0.0, 0.0, 0.0),
        }
        return target_pose, entrance_pose
    if target_module_name.startswith("sc_port_") and port_name == "sc_port_base":
        target_pose = {
            "position": np.array([0.0, -0.002, 0.0], dtype=np.float64),
            "orientation": _rpy_to_quaternion(1.5708, 3.14159, 0.0),
        }
        entrance_pose = {
            "position": np.array([0.0, 0.0, -0.01564], dtype=np.float64),
            "orientation": _rpy_to_quaternion(0.0, 0.0, 0.0),
        }
        return target_pose, entrance_pose
    return None, None


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


def _world_entities_summary(
    *,
    entities_by_name: dict[str, dict[str, Any]],
    source_name: str | None,
    target_name: str | None,
    plug_name: str | None,
    entrance_name: str | None,
) -> dict[str, Any]:
    interesting_prefixes = (
        "overview_",
        "task_board",
        "tabletop",
        "cable_",
        "wrist_",
        "shoulder_",
        "upper_arm_",
        "forearm_",
        "ee_link",
        "tool0",
        "ati/",
        "ati_",
        "lc_",
        "sc_",
    )
    named_entities: dict[str, dict[str, Any]] = {}
    for name in sorted(entities_by_name.keys()):
        if not any(name.startswith(prefix) for prefix in interesting_prefixes) and name not in {
            source_name,
            target_name,
            plug_name,
            entrance_name,
            "ur5e",
            "task_board",
            "tabletop",
        }:
            continue
        entity = entities_by_name.get(name)
        if not isinstance(entity, dict):
            continue
        pose = entity.get("pose") if isinstance(entity.get("pose"), dict) else entity
        position = pose.get("position") if isinstance(pose, dict) else None
        orientation = pose.get("orientation") if isinstance(pose, dict) else None
        named_entities[name] = {
            "id": entity.get("id"),
            "position": list(position) if isinstance(position, list) else None,
            "orientation": list(orientation) if isinstance(orientation, list) else None,
        }
    return {
        "entity_count": len(entities_by_name),
        "tracked_names": {
            "source_entity_name": source_name,
            "target_entity_name": target_name,
            "plug_entity_name": plug_name,
            "entrance_entity_name": entrance_name,
        },
        "named_entities": named_entities,
        "all_entity_names_sample": sorted(list(entities_by_name.keys()))[:128],
        "pose_rich_entity_count": sum(
            1
            for entity in entities_by_name.values()
            if isinstance(entity, dict)
            and isinstance(entity.get("position"), list)
            and len(entity.get("position", [])) == 3
        ),
    }
