"""Runtime abstraction and runtime implementations for the training-only env."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import time
from typing import Any

from .backend import Backend, StubBackend
from .discovery import resolve_gz_executable
from .gazebo_client import (
    GazeboCliClient,
    GazeboCliClientConfig,
    GazeboClient,
    GazeboTransportClient,
)
from .protocol import GetObservationRequest, ResetRequest, StepRequest


class Runtime(ABC):
    """Abstract runtime interface for future Gazebo-backed implementations."""

    @abstractmethod
    def start(self) -> None:
        """Start the runtime."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the runtime."""

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the runtime and return `(observation, info)`."""

    @abstractmethod
    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action and return Gymnasium-style step outputs."""

    @abstractmethod
    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the latest observation and metadata from the runtime."""


@dataclass(frozen=True)
class GazeboRuntimeConfig:
    """Configuration for a future Gazebo subprocess runtime."""

    world_path: str
    headless: bool = True
    timeout: float = 5.0
    executable: str = "gz"
    world_name: str | None = None
    source_entity_name: str = "robot"
    target_entity_name: str = "task_board"
    joint_command_model_name: str = "ur5e"
    joint_names: tuple[str, ...] = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    success_distance_threshold: float = 1.0
    orientation_success_threshold: float | None = None
    max_episode_steps: int | None = None
    success_bonus: float = 10.0
    reward_mode: str = "heuristic"
    transport_backend: str = "auto"
    transport_helper_executable: str | None = None
    helper_startup_timeout_s: float = 5.0
    helper_request_timeout_s: float = 5.0
    helper_startup_settle_s: float = 3.0
    require_pose_for_ready: bool = True
    reset_post_reset_ticks: int = 4
    action_post_step_ticks: int = 1
    settle_step_ticks: int = 1


@dataclass
class GazeboRuntime(Runtime):
    """Runtime that manages a `gz sim` subprocess lifecycle without ROS.

    Real Gazebo runtimes are wrappers around an external world, not isolated
    simulator instances. Multiple `GazeboRuntime` objects that point at the
    same world/executable observe and mutate the same underlying world state.
    """

    config: GazeboRuntimeConfig
    process: subprocess.Popen[str] | None = None
    client: GazeboClient | None = None

    def _resolved_executable(self) -> str:
        resolution = resolve_gz_executable(self.config.executable)
        if resolution.resolved_path is None:
            searched = "\n".join(f"  - {path}" for path in resolution.searched_locations)
            setup_hint = ""
            if resolution.discovered_setup_script:
                setup_hint = (
                    "\nSetup script found but not sourced. Try:\n"
                    f"  bash -lc \"source {resolution.discovered_setup_script} && <your command>\""
                )
            raise FileNotFoundError(
                f"Gazebo executable was not found: {self.config.executable}. "
                f"status={resolution.status}. setup_status={resolution.setup_status}. "
                f"Searched:\n{searched}{setup_hint}"
            )
        return resolution.resolved_path

    def start(self) -> None:
        """Start the Gazebo subprocess and verify it survives initial startup."""
        if self.process is not None and self.process.poll() is None:
            return

        world_path = Path(self.config.world_path)
        if not world_path.exists():
            raise FileNotFoundError(
                f"Gazebo world file does not exist: {world_path}"
            )

        command = self._build_command(world_path, executable=self._resolved_executable())
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + self.config.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                raise RuntimeError(
                    "Gazebo process crashed during startup "
                    f"(exit code {self.process.returncode}). "
                    f"stdout: {stdout.strip()} stderr: {stderr.strip()}"
                )
            time.sleep(0.05)

    def stop(self) -> None:
        """Stop the Gazebo subprocess cleanly, or kill it on timeout."""
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        if self.process is None:
            return

        if self.process.poll() is not None:
            self.process = None
            return

        self.process.terminate()
        try:
            self.process.wait(timeout=self.config.timeout)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait(timeout=self.config.timeout)
            self.process = None
            raise TimeoutError(
                "Timed out while waiting for Gazebo process to stop cleanly."
            ) from exc

        self.process = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the running Gazebo world through the real client bridge."""
        self._ensure_running("reset")
        response = self._client().reset(
            ResetRequest(seed=seed, options=dict(options or {}))
        )
        return dict(response.observation), dict(response.info)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance Gazebo through the real client bridge."""
        self._ensure_running("step")
        response = self._client().step(StepRequest(action=dict(action)))
        return (
            dict(response.observation),
            response.reward,
            response.terminated,
            response.truncated,
            dict(response.info),
        )

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the latest Gazebo observation through the real client bridge."""
        self._ensure_running("get observation")
        response = self._client().get_observation(GetObservationRequest())
        return dict(response.observation), dict(response.info)

    def _build_command(self, world_path: Path, *, executable: str | None = None) -> list[str]:
        """Build the subprocess command line for `gz sim`."""
        command = [executable or self._resolved_executable(), "sim"]
        if self.config.headless:
            command.append("-s")
        command.append(str(world_path))
        return command

    def _client(self) -> GazeboClient:
        """Return the configured real Gazebo client."""
        if self.client is None:
            client_config = GazeboCliClientConfig(
                executable=self.config.executable,
                world_path=self.config.world_path,
                timeout=self.config.timeout,
                world_name=self.config.world_name,
                source_entity_name=self.config.source_entity_name,
                target_entity_name=self.config.target_entity_name,
                joint_command_model_name=self.config.joint_command_model_name,
                joint_names=self.config.joint_names,
                success_distance_threshold=self.config.success_distance_threshold,
                orientation_success_threshold=self.config.orientation_success_threshold,
                max_episode_steps=self.config.max_episode_steps,
                success_bonus=self.config.success_bonus,
                reward_mode=self.config.reward_mode,
                transport_backend=self.config.transport_backend,
                transport_helper_executable=self.config.transport_helper_executable,
                helper_startup_timeout_s=self.config.helper_startup_timeout_s,
                helper_request_timeout_s=self.config.helper_request_timeout_s,
                helper_startup_settle_s=self.config.helper_startup_settle_s,
                require_pose_for_ready=self.config.require_pose_for_ready,
                reset_post_reset_ticks=self.config.reset_post_reset_ticks,
                action_post_step_ticks=self.config.action_post_step_ticks,
                settle_step_ticks=self.config.settle_step_ticks,
            )
            backend = self.config.transport_backend
            if backend == "transport" or (
                backend == "auto"
                and Path(self.config.executable).name == "gz"
                and GazeboCliClient.transport_helper_available(client_config)
            ):
                self.client = GazeboTransportClient(client_config)
            else:
                self.client = GazeboCliClient(client_config)
        return self.client

    def _ensure_running(self, operation: str) -> None:
        """Ensure the Gazebo subprocess is active before using the client."""
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError(
                f"Cannot {operation} before GazeboRuntime.start() or after stop()."
            )


@dataclass
class GazeboAttachedRuntime(Runtime):
    """Runtime that attaches to an already-running Gazebo world."""

    config: GazeboRuntimeConfig
    client: GazeboClient | None = None
    is_started: bool = False

    def start(self) -> None:
        if self.is_started:
            return
        self._client()
        self.is_started = True

    def stop(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        self.client = None
        self.is_started = False

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_started("reset")
        response = self._client().reset(
            ResetRequest(seed=seed, options=dict(options or {}))
        )
        return dict(response.observation), dict(response.info)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_started("step")
        response = self._client().step(StepRequest(action=dict(action)))
        return (
            dict(response.observation),
            response.reward,
            response.terminated,
            response.truncated,
            dict(response.info),
        )

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_started("get observation")
        response = self._client().get_observation(GetObservationRequest())
        return dict(response.observation), dict(response.info)

    def _client(self) -> GazeboClient:
        if self.client is None:
            client_config = GazeboCliClientConfig(
                executable=self.config.executable,
                world_path=self.config.world_path,
                timeout=self.config.timeout,
                world_name=self.config.world_name,
                source_entity_name=self.config.source_entity_name,
                target_entity_name=self.config.target_entity_name,
                joint_command_model_name=self.config.joint_command_model_name,
                joint_names=self.config.joint_names,
                success_distance_threshold=self.config.success_distance_threshold,
                orientation_success_threshold=self.config.orientation_success_threshold,
                max_episode_steps=self.config.max_episode_steps,
                success_bonus=self.config.success_bonus,
                reward_mode=self.config.reward_mode,
                transport_backend=self.config.transport_backend,
                transport_helper_executable=self.config.transport_helper_executable,
                helper_startup_timeout_s=self.config.helper_startup_timeout_s,
                helper_request_timeout_s=self.config.helper_request_timeout_s,
                helper_startup_settle_s=self.config.helper_startup_settle_s,
                require_pose_for_ready=self.config.require_pose_for_ready,
                reset_post_reset_ticks=self.config.reset_post_reset_ticks,
                action_post_step_ticks=self.config.action_post_step_ticks,
                settle_step_ticks=self.config.settle_step_ticks,
            )
            backend = self.config.transport_backend
            if backend == "transport" or (
                backend == "auto"
                and Path(self.config.executable).name == "gz"
                and GazeboCliClient.transport_helper_available(client_config)
            ):
                self.client = GazeboTransportClient(client_config)
            else:
                self.client = GazeboCliClient(client_config)
        return self.client

    def _ensure_started(self, operation: str) -> None:
        if not self.is_started:
            raise RuntimeError(
                f"Cannot {operation} before GazeboAttachedRuntime.start()."
            )


@dataclass
class FakeRuntime(Runtime):
    """Stub runtime used by the public env API during early milestones.

    This class intentionally does not launch a simulator. It only tracks basic
    lifecycle state so the public environment API can be exercised by tests.
    """

    is_started: bool = False
    is_stopped: bool = False
    last_seed: int | None = None
    last_options: dict[str, Any] = field(default_factory=dict)
    backend: Backend = field(default_factory=StubBackend)
    last_legacy_action: list[float] = field(default_factory=list)

    def start(self) -> None:
        """Start the fake runtime."""
        self.is_started = True
        self.is_stopped = False

    def stop(self) -> None:
        """Stop the fake runtime."""
        self.is_stopped = True

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset through the backend bridge and return `(observation, info)`."""
        if not self.is_started:
            self.start()
        self.last_seed = seed
        self.last_options = dict(options or {})
        self.last_legacy_action = []
        response = self.backend.reset(
            ResetRequest(seed=seed, options=dict(options or {}))
        )
        observation = dict(response.observation)
        if options:
            observation["runtime"] = (
                "fake" if self.last_options.get("mode") == "test" else "stub"
            )
        if self.last_options.get("mode") == "integration":
            observation["backend"] = "stub"
        return observation, self._compat_info(response.info)

    def step(
        self,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one step through the backend bridge."""
        if not self.is_started or self.is_stopped:
            raise RuntimeError("Cannot step runtime before start or after stop.")
        command = action.get("command")
        if isinstance(command, list) and all(isinstance(item, (int, float)) for item in command):
            self.last_legacy_action = [float(item) for item in command]
        response = self.backend.step(StepRequest(action=dict(action)))
        observation = dict(response.observation)
        if "command" in action:
            observation["last_action"] = list(self.last_legacy_action)
            reward = 0.0
        else:
            reward = response.reward
        return (
            observation,
            reward,
            response.terminated,
            response.truncated,
            self._compat_info(response.info),
        )

    def get_observation(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read the latest observation through the backend bridge."""
        if not self.is_started or self.is_stopped:
            raise RuntimeError(
                "Cannot get observation before start or after stop."
            )
        response = self.backend.get_observation(GetObservationRequest())
        observation = dict(response.observation)
        observation["backend"] = "stub"
        observation["last_action"] = {"command": list(self.last_legacy_action)}
        return observation, self._compat_info(response.info)

    def _compat_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """Return info with legacy public API compatibility fields."""
        compatible = dict(info)
        compatible["runtime"] = "fake"
        compatible.setdefault("seed", self.last_seed)
        compatible.setdefault("options", dict(self.last_options))
        return compatible
