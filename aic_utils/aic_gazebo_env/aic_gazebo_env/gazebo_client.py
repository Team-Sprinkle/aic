"""Minimal Gazebo CLI client for the real runtime path."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import ast
import math
import os
from pathlib import Path
import re
import shutil
import select
import struct
import subprocess
import threading
import time
import xml.etree.ElementTree as ET

from .official_scoring import OfficialTier3TrackedPairScorer
from .protocol import (
    GetObservationRequest,
    GetObservationResponse,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
)
from .transport_bridge import (
    GazeboTransportBridge,
    GazeboTransportBridgeConfig,
    GazeboTransportBridgeError,
)


class GazeboClient(ABC):
    """Transport abstraction used by the real Gazebo runtime."""

    @abstractmethod
    def reset(self, request: ResetRequest) -> ResetResponse:
        """Reset the Gazebo world and return a protocol response."""

    @abstractmethod
    def get_observation(
        self,
        request: GetObservationRequest,
    ) -> GetObservationResponse:
        """Read the latest Gazebo observation and return a protocol response."""

    @abstractmethod
    def step(self, request: StepRequest) -> StepResponse:
        """Advance the Gazebo world by one step."""


@dataclass
class PersistentGazeboTopicReader:
    """Keep a live `gz topic -e` process running and cache whole samples.

    Gazebo's one-shot CLI subscriptions are expensive enough to dominate RL
    stepping latency. This reader amortizes that cost by keeping one
    subscription process alive and slicing the byte stream into samples using a
    short quiet period between messages.
    """

    executable: str
    topic: str
    quiet_period_s: float

    def __post_init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._latest_sample: str | None = None
        self._latest_generation = 0

    def start(self) -> None:
        """Start the persistent reader if it is not already running."""
        process = self._process
        if process is not None and process.poll() is None:
            return

        self.stop()
        self._stop_event.clear()
        self._process = subprocess.Popen(
            [self.executable, "topic", "-e", "-t", self.topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"gz-topic-reader:{self.topic}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the persistent reader process and worker thread."""
        self._stop_event.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        self._process = None

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def current_generation(self) -> int:
        """Return the current sample generation counter."""
        with self._condition:
            return self._latest_generation

    def get_sample(
        self,
        *,
        after_generation: int | None = None,
        timeout: float,
    ) -> tuple[str, int]:
        """Return the latest sample, optionally waiting for a newer one."""
        self.start()
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                latest_sample = self._latest_sample
                latest_generation = self._latest_generation
                if latest_sample is not None and (
                    after_generation is None or latest_generation > after_generation
                ):
                    return latest_sample, latest_generation

                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Timed out waiting for Gazebo topic sample on {self.topic}."
                    )
                self._condition.wait(timeout=remaining)

    def _reader_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return

        fd = process.stdout.fileno()
        buffer = ""
        last_data_time: float | None = None

        while not self._stop_event.is_set():
            if process.poll() is not None:
                break

            try:
                ready, _, _ = select.select([fd], [], [], self.quiet_period_s)
            except (OSError, ValueError):
                break

            if ready:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", "replace")
                last_data_time = time.monotonic()
                continue

            if buffer and last_data_time is not None:
                if time.monotonic() - last_data_time >= self.quiet_period_s:
                    self._publish_sample(buffer)
                    buffer = ""
                    last_data_time = None

        if buffer:
            self._publish_sample(buffer)

    def _publish_sample(self, payload: str) -> None:
        sample = payload.strip()
        if not sample:
            return
        with self._condition:
            self._latest_sample = sample
            self._latest_generation += 1
            self._condition.notify_all()


@dataclass(frozen=True)
class GazeboCliClientConfig:
    """Configuration for the CLI-backed Gazebo client."""

    executable: str
    world_path: str
    timeout: float
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
    initial_joint_positions: tuple[float, ...] | None = (
        -0.1597,
        -1.3542,
        -1.6648,
        -1.6933,
        1.5710,
        1.4110,
    )
    step_after_action: bool = True
    joint_settle_tolerance: float = 5e-3
    joint_settle_timeout_s: float = 3.0
    joint_settle_poll_interval_s: float = 0.1
    live_state_entity_ids: tuple[tuple[str, int], ...] = (
        ("ati/tool_link", 79),
        ("wrist_3_link", 50),
    )
    tabletop_position: tuple[float, float, float] = (-0.2, 0.2, 1.14)
    tabletop_rpy: tuple[float, float, float] = (0.0, 0.0, -3.141)
    success_distance_threshold: float = 1.0
    orientation_success_threshold: float | None = None
    max_episode_steps: int | None = None
    success_bonus: float = 10.0
    reward_mode: str = "heuristic"
    transport_backend: str = "cli"
    transport_helper_executable: str | None = None
    helper_startup_timeout_s: float = 5.0
    helper_request_timeout_s: float = 5.0
    helper_startup_settle_s: float = 3.0
    require_pose_for_ready: bool = True
    reset_post_reset_ticks: int = 4
    action_post_step_ticks: int = 1
    settle_step_ticks: int = 1
    observation_transport: str = "auto"
    persistent_topic_quiet_period_s: float = 0.05
    allow_world_step_on_observation_timeout: bool = True


@dataclass
class GazeboCliClient(GazeboClient):
    """Client that talks to Gazebo through the `gz` CLI."""

    config: GazeboCliClientConfig

    def __post_init__(self) -> None:
        self._official_tier3_scorer = OfficialTier3TrackedPairScorer()
        self._state_reader: PersistentGazeboTopicReader | None = None
        self._pose_reader: PersistentGazeboTopicReader | None = None
        self._logical_step_count = 0
        self._last_source_entity_name: str | None = None
        self._last_source_position: list[float] | None = None
        self._last_source_orientation: list[float] | None = None

    def _transport_backend_name(self) -> str:
        """Return the concrete transport backend label used for info/debugging."""
        return "cli"

    def _transport_health_flags(self) -> dict[str, object]:
        return {
            "helper_startup_ok": False,
            "helper_ready_ok": False,
            "fallback_used": False,
        }

    def _logicalize_observation(
        self,
        observation: dict[str, object],
        *,
        step_count: int | None = None,
    ) -> dict[str, object]:
        logical_observation = dict(observation)
        logical_observation["step_count"] = (
            self._logical_step_count if step_count is None else step_count
        )
        return logical_observation

    def _logicalize_state_text(
        self,
        state_text: str,
        *,
        step_count: int | None = None,
    ) -> str:
        logical_step_count = self._logical_step_count if step_count is None else step_count
        return re.sub(
            r"step_count:\s*-?\d+",
            f"step_count: {logical_step_count}",
            state_text,
            count=1,
        )

    def _validate_observation_sane(self, observation: dict[str, object]) -> None:
        entity_count = observation.get("entity_count")
        joint_count = observation.get("joint_count")
        if not isinstance(entity_count, int) or entity_count <= 0:
            raise RuntimeError("observation sanity check failed: entity_count")
        if not isinstance(joint_count, int) or joint_count < 0:
            raise RuntimeError("observation sanity check failed: joint_count")

    def _advance_world(self, ticks: int) -> str:
        if ticks <= 0:
            raise ValueError("ticks must be positive")
        return self._run(
            [
                "service",
                "-s",
                f"/world/{self._world_name()}/control",
                "--reqtype",
                "gz.msgs.WorldControl",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                str(int(self.config.timeout * 1000)),
                "--req",
                f"multi_step: {ticks}",
            ]
        )

    def _reset_and_observe(
        self,
        *,
        pre_reset_generation: int | None,
    ) -> tuple[GetObservationResponse, dict[str, object]]:
        reset_info: dict[str, object] = {
            "reset_ok": False,
            "world_control_ok": False,
            "first_observation_ok": False,
            "fallback_used": False,
        }
        try:
            reset_reply = self._run(
                [
                    "service",
                    "-s",
                    f"/world/{self._world_name()}/control",
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(int(self.config.timeout * 1000)),
                    "--req",
                    "reset: {all: true}",
                ]
            )
        except Exception as exc:
            raise RuntimeError(f"reset service failure: {exc}") from exc
        reset_info["reset_ok"] = True
        reset_info["reset_service"] = reset_reply
        try:
            step_reply = self._advance_world(self.config.reset_post_reset_ticks)
        except Exception as exc:
            raise RuntimeError(f"reset world advance failure: {exc}") from exc
        reset_info["world_control_ok"] = True
        reset_info["reset_post_reset_step_service"] = step_reply
        try:
            observation_response = self._get_observation_after_generation(
                state_generation=(
                    pre_reset_generation if self._uses_persistent_observation_transport() else None
                ),
                pose_generation=(
                    self._current_pose_generation()
                    if self._uses_persistent_observation_transport()
                    else None
                ),
            )
            self._validate_observation_sane(observation_response.observation)
        except TimeoutError as exc:
            raise RuntimeError(
                f"reset freshness failure: state publication not fresh yet: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"reset observation failure: observation parse failure: {exc}"
            ) from exc
        reset_info["first_observation_ok"] = True
        reset_info["fallback_used"] = (
            observation_response.info.get("transport_backend") == "transport_cli_fallback"
        )
        return observation_response, reset_info

    @classmethod
    def transport_helper_available(
        cls,
        config: GazeboCliClientConfig,
    ) -> bool:
        """Return whether the persistent C++ transport helper is available."""
        return (
            GazeboTransportBridge.find_helper_explicit_or_on_path(
                config.transport_helper_executable
            )
            is not None
        )

    def reset(self, request: ResetRequest) -> ResetResponse:
        decoded = ResetRequest.from_dict(request.to_dict())
        state_generation = self._current_state_generation()
        observation_response, reset_details = self._reset_and_observe(
            pre_reset_generation=state_generation
        )
        self._logical_step_count = 0
        normalized_observation = self._logicalize_observation(
            observation_response.observation,
            step_count=0,
        )
        response = ResetResponse(
            observation=normalized_observation,
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "transport_backend": self._transport_backend_name(),
                "seed": decoded.seed,
                "options": dict(decoded.options),
                "world_name": self._world_name(),
                "observation_topic": observation_response.info.get("observation_topic"),
                "sim_step_count_raw": observation_response.observation.get("step_count"),
                **self._transport_health_flags(),
                **reset_details,
            },
        )
        self._remember_initial_tracked_distance(normalized_observation)
        self._remember_joint_positions(normalized_observation)
        return ResetResponse.from_dict(response.to_dict())

    def get_observation(
        self,
        request: GetObservationRequest,
    ) -> GetObservationResponse:
        decoded = GetObservationRequest.from_dict(request.to_dict())
        del decoded
        world_name = self._world_name()
        state_topic = f"/world/{world_name}/state"
        pose_topic = f"/world/{world_name}/pose/info"
        dynamic_pose_topic = f"/world/{world_name}/dynamic_pose/info"
        state_payload, _ = self._read_state_sample(
            topic=state_topic,
            after_generation=None,
        )
        pose_payload = self._read_pose_sample(topic=pose_topic)
        if pose_payload is None:
            pose_payload = self._read_pose_sample(topic=dynamic_pose_topic)
        observation = self._parse_live_observation(
            state_payload=state_payload,
            pose_payload=pose_payload,
            world_name=world_name,
        )
        if observation["entity_count"] == 0 and observation["joint_count"] == 0:
            observation = self._parse_state_text(state_payload)
        response = GetObservationResponse(
            observation=self._logicalize_observation(observation),
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "transport_backend": self._transport_backend_name(),
                "observation_topic": state_topic,
                "pose_topic": pose_topic,
                "dynamic_pose_topic": dynamic_pose_topic,
                "state_text": self._logicalize_state_text(state_payload),
                "state_text_raw": state_payload,
                "pose_text": pose_payload,
                "sim_step_count_raw": observation.get("step_count"),
            },
        )
        return GetObservationResponse.from_dict(response.to_dict())

    def close(self) -> None:
        """Release any persistent Gazebo topic readers."""
        if self._state_reader is not None:
            self._state_reader.stop()
            self._state_reader = None
        if self._pose_reader is not None:
            self._pose_reader.stop()
            self._pose_reader = None

    def _read_topic_sample(self, topic: str) -> str:
        deadline = time.monotonic() + (self.config.timeout * 3.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._run(["topic", "-e", "-n", "1", "-t", topic])
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if topic.endswith("/state") and self.config.allow_world_step_on_observation_timeout:
                    fallback_payload = self._read_state_sample_after_world_step(topic)
                    if fallback_payload is not None:
                        return fallback_payload
                time.sleep(0.2)
        raise RuntimeError(f"Timed out reading Gazebo topic {topic}: {last_error}")

    def _read_state_sample(
        self,
        *,
        topic: str,
        after_generation: int | None,
    ) -> tuple[str, int | None]:
        if not self._uses_persistent_observation_transport():
            return self._read_topic_sample(topic), None
        reader = self._state_topic_reader(topic)
        try:
            return reader.get_sample(
                after_generation=after_generation,
                timeout=min(self.config.timeout, 2.0),
            )
        except TimeoutError:
            if self.config.allow_world_step_on_observation_timeout:
                fallback_payload = self._read_state_sample_after_world_step(topic)
                if fallback_payload is not None:
                    return fallback_payload, None
            raise

    def _read_pose_sample(
        self,
        *,
        topic: str,
        after_generation: int | None = None,
    ) -> str | None:
        if not self._uses_persistent_observation_transport():
            attempts = 3
            for _ in range(attempts):
                payload = self._try_run_with_timeout(
                    ["topic", "-e", "-n", "1", "-t", topic],
                    timeout=min(self.config.timeout, 3.0),
                )
                if payload:
                    return payload
                time.sleep(0.2)
            return None
        reader = self._pose_topic_reader(topic)
        try:
            payload, _ = reader.get_sample(
                after_generation=after_generation,
                timeout=min(self.config.timeout, 1.5),
            )
            return payload
        except TimeoutError:
            return None

    def _try_run_with_timeout(self, args: list[str], *, timeout: float) -> str | None:
        try:
            completed = subprocess.run(
                [self.config.executable, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def _read_state_sample_after_world_step(self, topic: str) -> str | None:
        process = subprocess.Popen(
            [self.config.executable, "topic", "-e", "-n", "1", "-t", topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(0.5)
            step_reply = self._try_run_with_timeout(
                [
                    "service",
                    "-s",
                    f"/world/{self._world_name()}/control",
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(int(self.config.timeout * 1000)),
                    "--req",
                    "multi_step: 1",
                ],
                timeout=self.config.timeout,
            )
            if step_reply is None:
                process.kill()
                process.communicate()
                return None
            stdout, stderr = process.communicate(timeout=self.config.timeout)
            if process.returncode not in (0, None):
                return None
            if stderr.strip():
                return None
            return stdout.strip() or None
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return None

    def step(self, request: StepRequest) -> StepResponse:
        decoded = StepRequest.from_dict(request.to_dict())
        translated_action = self._translate_policy_action(decoded.action)
        translated_action = self._translate_joint_delta_action(translated_action)
        joint_reply = self._maybe_apply_joint_action(translated_action)
        pose_reply = self._maybe_apply_pose_action(translated_action)
        multi_step = self._resolve_multi_step(translated_action)
        reply: str | None = None
        settle_info: dict[str, object] = {}
        if joint_reply is None and pose_reply is None:
            state_generation = self._current_state_generation()
            pose_generation = self._current_pose_generation()
            reply = self._advance_world(multi_step)
            observation_response = self._get_observation_after_generation(
                state_generation=(
                    state_generation if self._uses_persistent_observation_transport() else None
                ),
                pose_generation=(
                    pose_generation if self._uses_persistent_observation_transport() else None
                ),
            )
            settle_info["world_control_ok"] = True
        else:
            if self.config.step_after_action:
                reply = self._advance_world(multi_step)
                settle_info["world_control_ok"] = True
            observation_response, settle_info = self._observe_after_action(
                translated_action=translated_action,
                joint_reply=joint_reply,
                pose_reply=pose_reply,
            )
        expected_step_count = self._logical_step_count + multi_step
        logical_observation = self._logicalize_observation(
            observation_response.observation,
            step_count=expected_step_count,
        )
        self._logical_step_count = expected_step_count
        reward, terminated, truncated, reward_details = self._compute_step_outcome(
            logical_observation
        )
        self._remember_joint_positions(logical_observation)
        response = StepResponse(
            observation=logical_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "transport_backend": self._transport_backend_name(),
                "world_name": self._world_name(),
                "applied_action": dict(translated_action),
                "joint_target_service": joint_reply,
                "pose_service": pose_reply,
                "multi_step": multi_step,
                "step_service": reply,
                "observation_topic": observation_response.info.get("observation_topic"),
                "source_entity_name": self.config.source_entity_name,
                "target_entity_name": self.config.target_entity_name,
                "success_distance_threshold": self.config.success_distance_threshold,
                "orientation_success_threshold": self.config.orientation_success_threshold,
                "max_episode_steps": self.config.max_episode_steps,
                "success_bonus": self.config.success_bonus,
                "reward_mode": self.config.reward_mode,
                "terminated": terminated,
                "truncated": truncated,
                "reward": reward,
                "reward_details": reward_details,
                "sim_step_count_raw": observation_response.observation.get("step_count"),
                "first_observation_ok": True,
                **self._transport_health_flags(),
                **settle_info,
            },
        )
        return StepResponse.from_dict(response.to_dict())

    def _observe_after_action(
        self,
        *,
        translated_action: dict[str, object],
        joint_reply: str | None,
        pose_reply: str | None,
    ) -> tuple[GetObservationResponse, dict[str, object]]:
        """Observe the world after an action with transport-aware settling."""
        settle_info: dict[str, object] = {}
        joint_action = translated_action.get("set_joint_positions")
        if joint_reply is not None and isinstance(joint_action, dict):
            positions = joint_action.get("positions")
            if isinstance(positions, list) and all(
                isinstance(value, (int, float)) for value in positions
            ):
                observation_response, settled = self._wait_for_joint_target_settle(
                    [float(value) for value in positions]
                )
                settle_info["joint_target_settled"] = settled
                settle_info["joint_settle_tolerance"] = self.config.joint_settle_tolerance
                settle_info["joint_settle_timeout_s"] = self.config.joint_settle_timeout_s
                return observation_response, settle_info

        if pose_reply is not None:
            if not self.config.step_after_action:
                self._advance_world(self.config.action_post_step_ticks)
                settle_info["world_control_ok"] = True
            settle_info["joint_target_settled"] = None

        if not self._uses_persistent_observation_transport():
            return self.get_observation(GetObservationRequest()), settle_info

        return self._get_observation_after_generation(
            state_generation=self._current_state_generation(),
            pose_generation=self._current_pose_generation(),
        ), settle_info

    def _translate_policy_action(self, action: dict[str, object]) -> dict[str, object]:
        """Translate the canonical policy-facing action into bridge-level actions."""
        translated_action = dict(action)
        policy_action = translated_action.get("policy_action")
        if policy_action is None:
            return translated_action
        if not isinstance(policy_action, dict):
            raise ValueError("Gazebo step action 'policy_action' must be a dict.")
        if any(
            translated_action.get(key) is not None
            for key in ("set_entity_position", "set_entity_pose", "delta_source_pose")
        ):
            raise ValueError(
                "Gazebo step action 'policy_action' must not be combined with bridge-level pose actions."
            )

        delta_source_pose: dict[str, object] = {}
        if "position_delta" in policy_action:
            delta_source_pose["position_delta"] = policy_action["position_delta"]
        if "orientation_delta" in policy_action:
            delta_source_pose["orientation_delta"] = policy_action["orientation_delta"]
        if not delta_source_pose:
            raise ValueError(
                "Gazebo step action 'policy_action' must provide 'position_delta', "
                "'orientation_delta', or both."
            )

        translated_action.pop("policy_action")
        translated_action["delta_source_pose"] = delta_source_pose
        return translated_action

    def _translate_joint_delta_action(
        self,
        action: dict[str, object],
    ) -> dict[str, object]:
        """Translate env-facing joint deltas into absolute joint targets."""
        translated_action = dict(action)
        joint_position_delta = translated_action.get("joint_position_delta")
        if joint_position_delta is None:
            return translated_action
        if translated_action.get("set_joint_positions") is not None:
            raise ValueError(
                "Gazebo step action must not combine 'joint_position_delta' with 'set_joint_positions'."
            )
        if any(
            translated_action.get(key) is not None
            for key in ("set_entity_position", "set_entity_pose", "delta_source_pose")
        ):
            raise ValueError(
                "Gazebo step action 'joint_position_delta' must not be combined with pose actions."
            )
        if not isinstance(joint_position_delta, list) or len(joint_position_delta) != len(
            self.config.joint_names
        ):
            raise ValueError(
                "Gazebo step action 'joint_position_delta' must be a list with one value per configured joint."
            )
        if not all(isinstance(value, (int, float)) for value in joint_position_delta):
            raise ValueError(
                "Gazebo step action 'joint_position_delta' must contain only numbers."
            )

        current_joint_positions = self._resolve_current_joint_positions()
        next_positions = [
            current_position + float(delta)
            for current_position, delta in zip(
                current_joint_positions,
                joint_position_delta,
            )
        ]
        self._last_known_joint_positions = tuple(next_positions)
        translated_action.pop("joint_position_delta")
        translated_action["set_joint_positions"] = {
            "model_name": self.config.joint_command_model_name,
            "joint_names": list(self.config.joint_names),
            "positions": next_positions,
        }
        return translated_action

    def _run(self, args: list[str]) -> str:
        completed = subprocess.run(
            [self.config.executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(self.config.timeout, 1.0),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Gazebo CLI command failed with exit code {completed.returncode}: "
                f"{[self.config.executable, *args]!r}. "
                f"stdout: {completed.stdout.strip()} stderr: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    def _try_run(self, args: list[str]) -> str | None:
        try:
            return self._run(args)
        except (RuntimeError, subprocess.TimeoutExpired):
            return None

    def _resolve_multi_step(self, action: dict[str, object]) -> int:
        """Resolve the minimal diagnostic step count from the action payload."""
        multi_step = action.get("multi_step", 1)
        if not isinstance(multi_step, int) or multi_step <= 0:
            raise ValueError("Gazebo step action requires positive integer 'multi_step'.")
        return multi_step

    def _maybe_apply_joint_action(self, action: dict[str, object]) -> str | None:
        """Apply a Gazebo-native joint target action if provided."""
        joint_action = action.get("set_joint_positions")
        if joint_action is None:
            return None
        if any(
            action.get(key) is not None
            for key in ("set_entity_position", "set_entity_pose", "delta_source_pose")
        ):
            raise ValueError(
                "Gazebo step action must not combine 'set_joint_positions' with pose actions."
            )
        if not isinstance(joint_action, dict):
            raise ValueError("Gazebo step action 'set_joint_positions' must be a dict.")
        model_name = joint_action.get("model_name", self.config.joint_command_model_name)
        joint_names = joint_action.get("joint_names")
        positions = joint_action.get("positions")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError(
                "Gazebo step action 'set_joint_positions.model_name' must be a non-empty str."
            )
        if not isinstance(joint_names, list) or not joint_names:
            raise ValueError(
                "Gazebo step action 'set_joint_positions.joint_names' must be a non-empty list."
            )
        if not all(isinstance(name, str) and name for name in joint_names):
            raise ValueError(
                "Gazebo step action 'set_joint_positions.joint_names' must contain only non-empty strings."
            )
        if not isinstance(positions, list) or len(positions) != len(joint_names):
            raise ValueError(
                "Gazebo step action 'set_joint_positions.positions' must match the joint_names list length."
            )
        if not all(isinstance(value, (int, float)) for value in positions):
            raise ValueError(
                "Gazebo step action 'set_joint_positions.positions' must contain only numbers."
            )
        return self._send_joint_target_request(
            model_name=model_name,
            joint_names=joint_names,
            positions=[float(value) for value in positions],
        )

    def _maybe_apply_pose_action(self, action: dict[str, object]) -> str | None:
        """Apply the minimal real action if the caller provided one."""
        position_action = action.get("set_entity_position")
        pose_action = action.get("set_entity_pose")
        delta_pose_action = action.get("delta_source_pose")
        provided_action_count = sum(
            value is not None
            for value in (position_action, pose_action, delta_pose_action)
        )
        if provided_action_count > 1:
            raise ValueError(
                "Gazebo step action must provide at most one of "
                "'set_entity_position', 'set_entity_pose', or 'delta_source_pose'."
            )

        if pose_action is None and position_action is None and delta_pose_action is None:
            return None

        if delta_pose_action is not None:
            return self._apply_delta_source_pose_action(delta_pose_action)

        action_name = "set_entity_pose" if pose_action is not None else "set_entity_position"
        selected_action = pose_action if pose_action is not None else position_action
        if not isinstance(selected_action, dict):
            raise ValueError(f"Gazebo step action '{action_name}' must be a dict.")

        entity_name = selected_action.get("name")
        if not isinstance(entity_name, str) or not entity_name:
            raise ValueError(
                f"Gazebo step action '{action_name}.name' must be a non-empty str."
            )

        position = selected_action.get("position")
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError(
                f"Gazebo step action '{action_name}.position' must be a 3-item list."
            )
        if not all(isinstance(value, (int, float)) for value in position):
            raise ValueError(
                f"Gazebo step action '{action_name}.position' must contain only numbers."
            )

        x, y, z = (float(value) for value in position)
        orientation = selected_action.get("orientation", [0.0, 0.0, 0.0, 1.0])
        if not isinstance(orientation, list) or len(orientation) != 4:
            raise ValueError(
                f"Gazebo step action '{action_name}.orientation' must be a 4-item list."
            )
        if not all(isinstance(value, (int, float)) for value in orientation):
            raise ValueError(
                f"Gazebo step action '{action_name}.orientation' must contain only numbers."
            )
        qx, qy, qz, qw = (float(value) for value in orientation)
        return self._send_set_pose_request(
            entity_name=entity_name,
            position=[x, y, z],
            orientation=[qx, qy, qz, qw],
        )

    def _apply_delta_source_pose_action(self, delta_pose_action: object) -> str:
        """Apply an incremental pose update to the configured source entity."""
        return self._apply_delta_source_pose_action_internal(
            delta_pose_action,
            prefer_cached_source_pose=False,
        )

    def _apply_delta_source_pose_action_cached(self, delta_pose_action: object) -> str:
        """Apply an incremental pose update using cached source pose when available."""
        return self._apply_delta_source_pose_action_internal(
            delta_pose_action,
            prefer_cached_source_pose=True,
        )

    def set_source_pose_cache(
        self,
        *,
        entity_name: str,
        position: list[float],
        orientation: list[float],
    ) -> None:
        """Seed the cached source pose for fast delta updates."""
        self._last_source_entity_name = str(entity_name)
        self._last_source_position = [float(value) for value in position]
        self._last_source_orientation = self._normalize_quaternion(
            [float(value) for value in orientation]
        )

    def _apply_delta_source_pose_action_internal(
        self,
        delta_pose_action: object,
        *,
        prefer_cached_source_pose: bool,
    ) -> str:
        """Apply an incremental pose update to the configured source entity."""
        if not isinstance(delta_pose_action, dict):
            raise ValueError("Gazebo step action 'delta_source_pose' must be a dict.")

        position_delta = delta_pose_action.get("position_delta", [0.0, 0.0, 0.0])
        if not isinstance(position_delta, list) or len(position_delta) != 3:
            raise ValueError(
                "Gazebo step action 'delta_source_pose.position_delta' must be a 3-item list."
            )
        if not all(isinstance(value, (int, float)) for value in position_delta):
            raise ValueError(
                "Gazebo step action 'delta_source_pose.position_delta' must contain only numbers."
            )

        orientation_delta = delta_pose_action.get("orientation_delta", [0.0, 0.0, 0.0, 1.0])
        if not isinstance(orientation_delta, list) or len(orientation_delta) != 4:
            raise ValueError(
                "Gazebo step action 'delta_source_pose.orientation_delta' must be a 4-item list."
            )
        if not all(isinstance(value, (int, float)) for value in orientation_delta):
            raise ValueError(
                "Gazebo step action 'delta_source_pose.orientation_delta' must contain only numbers."
            )

        if "position_delta" not in delta_pose_action and "orientation_delta" not in delta_pose_action:
            raise ValueError(
                "Gazebo step action 'delta_source_pose' must provide 'position_delta', "
                "'orientation_delta', or both."
            )

        source_entity_name = self.config.source_entity_name
        pose_entities: dict[str, dict[str, object]] = {}
        current_position = (
            None if self._last_source_position is None else list(self._last_source_position)
        )
        current_orientation = (
            None if self._last_source_orientation is None else list(self._last_source_orientation)
        )
        if self._last_source_entity_name:
            source_entity_name = self._last_source_entity_name
        source_entity = None
        if not prefer_cached_source_pose or current_position is None or current_orientation is None:
            observation_response = self.get_observation(GetObservationRequest())
            entities_by_name = observation_response.observation.get("entities_by_name")
            if not isinstance(entities_by_name, dict):
                raise RuntimeError("Real observation did not contain 'entities_by_name'.")
            source_entity = self._lookup_named_entity(
                entities_by_name,
                source_entity_name,
            )
            if source_entity is None:
                world_name = self._world_name()
                pose_payload = self._read_pose_sample(
                    topic=f"/world/{world_name}/pose/info",
                    after_generation=None,
                )
                if pose_payload is None:
                    pose_payload = self._read_pose_sample(
                        topic=f"/world/{world_name}/dynamic_pose/info",
                        after_generation=None,
                    )
                if isinstance(pose_payload, str) and pose_payload:
                    pose_entities = {
                        entity["name"]: entity
                        for entity in self._extract_poses(pose_payload)
                        if isinstance(entity.get("name"), str)
                    }
                    source_entity = self._lookup_named_entity(
                        pose_entities,
                        source_entity_name,
                    )
            if source_entity is None:
                for candidate_name, _candidate_id in self.config.live_state_entity_ids:
                    if candidate_name == source_entity_name:
                        continue
                    source_entity = self._lookup_named_entity(entities_by_name, candidate_name)
                    if source_entity is None and pose_entities:
                        source_entity = self._lookup_named_entity(pose_entities, candidate_name)
                    if source_entity is not None:
                        source_entity_name = candidate_name
                        break
            if source_entity is not None:
                current_position = self._lookup_entity_position(source_entity)
                current_orientation = self._lookup_entity_orientation(source_entity)
        if current_position is None or current_orientation is None:
            current_position = (
                None if self._last_source_position is None else list(self._last_source_position)
            )
            current_orientation = (
                None
                if self._last_source_orientation is None
                else list(self._last_source_orientation)
            )
            source_entity_name = self._last_source_entity_name or source_entity_name
        if current_position is None or current_orientation is None:
            raise RuntimeError(
                f"Configured source entity was not found in real observation: {self.config.source_entity_name}"
            )
        self._last_source_entity_name = source_entity_name
        self._last_source_position = list(current_position)
        self._last_source_orientation = list(current_orientation)

        next_position = [
            current_axis + float(delta_axis)
            for current_axis, delta_axis in zip(current_position, position_delta)
        ]
        normalized_orientation_delta = self._normalize_quaternion(
            [float(value) for value in orientation_delta]
        )
        next_orientation = self._normalize_quaternion(
            self._multiply_quaternions(
                normalized_orientation_delta,
                current_orientation,
            )
        )
        self._last_source_position = list(next_position)
        self._last_source_orientation = list(next_orientation)
        return self._send_set_pose_request(
            entity_name=source_entity_name,
            position=next_position,
            orientation=next_orientation,
        )

    def _send_set_pose_request(
        self,
        *,
        entity_name: str,
        position: list[float],
        orientation: list[float],
    ) -> str:
        """Send the absolute pose request through Gazebo's set_pose service."""
        x, y, z = position
        qx, qy, qz, qw = orientation
        return self._run(
            [
                "service",
                "-s",
                f"/world/{self._world_name()}/set_pose",
                "--reqtype",
                "gz.msgs.Pose",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                str(int(self.config.timeout * 1000)),
                "--req",
                (
                    f'name: "{entity_name}" '
                    f"position: {{x: {x}, y: {y}, z: {z}}} "
                    f"orientation: {{x: {qx}, y: {qy}, z: {qz}, w: {qw}}}"
                ),
            ]
        )

    def _send_joint_target_request(
        self,
        *,
        model_name: str,
        joint_names: list[str],
        positions: list[float],
    ) -> str:
        """Send absolute joint targets through the Gazebo-native joint bridge."""
        published = self._publish_joint_target_topics(
            joint_names=joint_names,
            positions=positions,
        )
        if published:
            return "data: true"
        joint_names_text = ",".join(joint_names)
        positions_text = ",".join(self._format_joint_position(position) for position in positions)
        request_text = (
            f"model_name={model_name};"
            f"joint_names={joint_names_text};"
            f"positions={positions_text}"
        )
        last_error: Exception | None = None
        for _ in range(3):
            try:
                return self._run(
                    [
                        "service",
                        "-s",
                        f"/world/{self._world_name()}/joint_target",
                        "--reqtype",
                        "gz.msgs.StringMsg",
                        "--reptype",
                        "gz.msgs.Boolean",
                        "--timeout",
                        str(int(self.config.timeout * 1000)),
                        "--req",
                        f'data: "{request_text}"',
                    ]
                )
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"Joint target request failed after retries: {last_error}")

    def _publish_joint_target_topics(
        self,
        *,
        joint_names: list[str],
        positions: list[float],
    ) -> bool:
        """Publish UR5e joint targets to Gazebo's native position controllers."""
        if len(joint_names) != len(positions):
            return False
        for _ in range(3):
            for joint_name, position in zip(joint_names, positions):
                try:
                    GazeboCliClient._run(
                        self,
                        [
                            "topic",
                            "-t",
                            f"/aic/joint_target/{joint_name}",
                            "-m",
                            "gz.msgs.Double",
                            "-p",
                            f"data: {self._format_joint_position(position)}",
                        ]
                    )
                except (RuntimeError, subprocess.TimeoutExpired):
                    return False
            time.sleep(0.05)
        return True

    def _resolve_current_joint_positions(self) -> list[float]:
        cached_positions = getattr(self, "_last_known_joint_positions", None)
        if isinstance(cached_positions, tuple) and len(cached_positions) == len(
            self.config.joint_names
        ):
            return [float(position) for position in cached_positions]

        configured_positions = self.config.initial_joint_positions
        try:
            observation_response = self.get_observation(GetObservationRequest())
            observed_positions = self._lookup_current_joint_positions(
                observation_response.observation
            )
            if (
                isinstance(configured_positions, tuple)
                and len(configured_positions) == len(self.config.joint_names)
                and max(abs(position) for position in observed_positions) < 1e-3
                and max(abs(position) for position in configured_positions) > 0.1
            ):
                # Exported no-ROS training worlds can initially report zero
                # joint positions even though the official URDF link poses and
                # controller initial positions are the nonzero ground-truth
                # home configuration.
                return [float(position) for position in configured_positions]
            return observed_positions
        except RuntimeError:
            if isinstance(configured_positions, tuple) and len(configured_positions) == len(
                self.config.joint_names
            ):
                return [float(position) for position in configured_positions]
            raise

    def _remember_joint_positions(self, observation: dict[str, object]) -> None:
        try:
            current_joint_positions = self._lookup_current_joint_positions(observation)
        except RuntimeError:
            return
        self._last_known_joint_positions = tuple(current_joint_positions)

    def _remember_initial_tracked_distance(self, observation: dict[str, object]) -> None:
        tracked_distance = self._extract_tracked_distance(observation)
        if tracked_distance is not None:
            self._initial_tracked_distance = tracked_distance

    def _resolve_initial_tracked_distance(self, observation: dict[str, object]) -> float | None:
        initial_distance = getattr(self, "_initial_tracked_distance", None)
        if isinstance(initial_distance, float):
            return initial_distance
        tracked_distance = self._extract_tracked_distance(observation)
        if tracked_distance is not None:
            self._initial_tracked_distance = tracked_distance
        return tracked_distance

    def _extract_tracked_distance(self, observation: dict[str, object]) -> float | None:
        task_geometry = observation.get("task_geometry")
        if not isinstance(task_geometry, dict):
            return None
        pair_geometry = task_geometry.get("robot_to_task_board")
        if not isinstance(pair_geometry, dict):
            pair_geometry = task_geometry.get("tracked_entity_pair")
        if not isinstance(pair_geometry, dict):
            return None
        distance = pair_geometry.get("distance")
        if not isinstance(distance, float):
            return None
        return distance

    def _wait_for_joint_target_settle(
        self,
        target_positions: list[float],
    ) -> tuple[GetObservationResponse, bool]:
        """Poll observations until the configured joints are close to target."""
        deadline = time.monotonic() + self.config.joint_settle_timeout_s
        latest_observation_response: GetObservationResponse | None = None
        while time.monotonic() < deadline:
            latest_observation_response = self.get_observation(GetObservationRequest())
            try:
                current_joint_positions = self._lookup_current_joint_positions(
                    latest_observation_response.observation
                )
            except RuntimeError:
                time.sleep(self.config.joint_settle_poll_interval_s)
                continue
            max_error = max(
                abs(target - current)
                for target, current in zip(target_positions, current_joint_positions)
            )
            if max_error <= self.config.joint_settle_tolerance:
                return latest_observation_response, True
            self._advance_world(self.config.settle_step_ticks)
            time.sleep(self.config.joint_settle_poll_interval_s)

        if latest_observation_response is None:
            latest_observation_response = self.get_observation(GetObservationRequest())
        return latest_observation_response, False

    def _get_observation_after_generation(
        self,
        *,
        state_generation: int | None,
        pose_generation: int | None = None,
    ) -> GetObservationResponse:
        world_name = self._world_name()
        state_topic = f"/world/{world_name}/state"
        pose_topic = f"/world/{world_name}/pose/info"
        dynamic_pose_topic = f"/world/{world_name}/dynamic_pose/info"
        try:
            state_payload, _ = self._read_state_sample(
                topic=state_topic,
                after_generation=state_generation
                if self._uses_persistent_observation_transport()
                else None,
            )
        except TimeoutError:
            fallback_payload = self._read_state_sample_after_world_step(state_topic)
            if fallback_payload is None:
                raise
            state_payload = fallback_payload
        pose_payload = self._read_pose_sample(
            topic=pose_topic,
            after_generation=pose_generation,
        )
        if pose_payload is None:
            pose_payload = self._read_pose_sample(
                topic=dynamic_pose_topic,
                after_generation=pose_generation,
            )
        observation = self._parse_live_observation(
            state_payload=state_payload,
            pose_payload=pose_payload,
            world_name=world_name,
        )
        if observation["entity_count"] == 0 and observation["joint_count"] == 0:
            observation = self._parse_state_text(state_payload)
        return GetObservationResponse(
            observation=observation,
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "observation_topic": state_topic,
                "pose_topic": pose_topic,
                "dynamic_pose_topic": dynamic_pose_topic,
                "state_text": state_payload,
                "pose_text": pose_payload,
            },
        )

    def _current_state_generation(self) -> int | None:
        if self._state_reader is None:
            return None
        return self._state_reader.current_generation()

    def _current_pose_generation(self) -> int | None:
        if self._pose_reader is None:
            return None
        return self._pose_reader.current_generation()

    def _state_topic_reader(self, topic: str) -> PersistentGazeboTopicReader:
        if self._state_reader is None:
            self._state_reader = PersistentGazeboTopicReader(
                executable=self.config.executable,
                topic=topic,
                quiet_period_s=self.config.persistent_topic_quiet_period_s,
            )
        return self._state_reader

    def _pose_topic_reader(self, topic: str) -> PersistentGazeboTopicReader:
        if self._pose_reader is None:
            self._pose_reader = PersistentGazeboTopicReader(
                executable=self.config.executable,
                topic=topic,
                quiet_period_s=self.config.persistent_topic_quiet_period_s,
            )
        return self._pose_reader

    def _uses_persistent_observation_transport(self) -> bool:
        mode = self.config.observation_transport
        if mode == "persistent":
            return True
        if mode == "one_shot":
            return False
        executable_name = Path(self.config.executable).name
        return executable_name == "gz"

    def _format_joint_position(self, position: float) -> str:
        return f"{float(position):.6f}"

    def _parse_state_text(self, payload: str) -> dict[str, object]:
        """Parse a structured observation from `gz topic -e` text."""
        world_name = self._extract_scalar_string(payload, "world") or self._world_name()
        step_count = self._extract_scalar_int(payload, "step_count")
        entities = self._extract_entities(payload)
        joints = self._extract_joints(payload)
        entities_by_name = {
            entity["name"]: dict(entity) for entity in entities if isinstance(entity["name"], str)
        }
        return {
            "world_name": world_name,
            "step_count": 0 if step_count is None else step_count,
            "entity_count": len(entities),
            "entity_names": [entity["name"] for entity in entities],
            "entities": entities,
            "entities_by_name": entities_by_name,
            "joint_count": len(joints),
            "joint_names": [joint["name"] for joint in joints],
            "joint_positions": [joint["position"] for joint in joints],
            "joints": joints,
            "joints_by_name": {
                joint["name"]: dict(joint) for joint in joints if isinstance(joint["name"], str)
            },
            "task_geometry": self._build_task_geometry(entities_by_name),
        }

    def _parse_live_observation(
        self,
        *,
        state_payload: str,
        pose_payload: str | None,
        world_name: str,
    ) -> dict[str, object]:
        """Parse a structured observation from live Gazebo pose/state topics."""
        entities = (
            self._extract_poses(pose_payload)
            if isinstance(pose_payload, str) and pose_payload
            else self._extract_live_state_entities(state_payload)
        )
        state_text_entities = self._extract_entities(state_payload)
        entities = self._merge_entity_views(
            primary_entities=entities,
            secondary_entities=state_text_entities,
        )
        entities_by_name = {
            entity["name"]: dict(entity) for entity in entities if isinstance(entity["name"], str)
        }
        entities_by_name = self._normalize_entity_pose_frames(entities_by_name)
        entities = [entities_by_name[name] for name in sorted(entities_by_name.keys())]
        joints = self._decode_joint_components(state_payload)
        state_text_joints = self._extract_joints(state_payload)
        if len(state_text_joints) > len(joints):
            joints = state_text_joints
        return {
            "world_name": world_name,
            "step_count": self._extract_state_step_count(state_payload),
            "entity_count": len(entities),
            "entity_names": [entity["name"] for entity in entities],
            "entities": entities,
            "entities_by_name": entities_by_name,
            "joint_count": len(joints),
            "joint_names": [joint["name"] for joint in joints],
            "joint_positions": [joint["position"] for joint in joints],
            "joints": joints,
            "joints_by_name": {
                joint["name"]: dict(joint) for joint in joints if isinstance(joint["name"], str)
            },
            "task_geometry": self._build_task_geometry(entities_by_name),
        }

    def _normalize_entity_pose_frames(
        self,
        entities_by_name: dict[str, dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        normalized = {name: dict(entity) for name, entity in entities_by_name.items()}
        cache: dict[str, dict[str, object]] = {}
        for name in list(normalized.keys()):
            resolved = self._resolve_entity_world_pose(name=name, entities_by_name=normalized, cache=cache, stack=set())
            if resolved is not None:
                normalized[name] = resolved
        return normalized

    def _resolve_entity_world_pose(
        self,
        *,
        name: str,
        entities_by_name: dict[str, dict[str, object]],
        cache: dict[str, dict[str, object]],
        stack: set[str],
    ) -> dict[str, object] | None:
        cached = cache.get(name)
        if cached is not None:
            return dict(cached)
        entity = entities_by_name.get(name)
        if not isinstance(entity, dict):
            return None
        if name in stack:
            return dict(entity)
        pose = self._entity_pose_dict(entity)
        if pose is None:
            cache[name] = dict(entity)
            return dict(entity)
        if self._pose_looks_world_space(name=name, entity=entity, entities_by_name=entities_by_name):
            cache[name] = dict(entity)
            return dict(entity)
        parent_name = self._candidate_parent_name(name=name, entities_by_name=entities_by_name)
        if parent_name is None:
            cache[name] = dict(entity)
            return dict(entity)
        stack = set(stack)
        stack.add(name)
        parent_entity = self._resolve_entity_world_pose(
            name=parent_name,
            entities_by_name=entities_by_name,
            cache=cache,
            stack=stack,
        )
        if not isinstance(parent_entity, dict):
            cache[name] = dict(entity)
            return dict(entity)
        parent_pose = self._entity_pose_dict(parent_entity)
        if parent_pose is None:
            cache[name] = dict(entity)
            return dict(entity)
        composed = self._compose_pose_dicts(parent_pose, pose)
        resolved = dict(entity)
        resolved["position"] = list(composed["position"])
        resolved["orientation"] = list(composed["orientation"])
        resolved["pose"] = {
            "position": list(composed["position"]),
            "orientation": list(composed["orientation"]),
        }
        resolved["pose_frame_status"] = "parent_composed_world"
        resolved["pose_parent_name"] = parent_name
        cache[name] = dict(resolved)
        return resolved

    def _pose_looks_world_space(
        self,
        *,
        name: str,
        entity: dict[str, object],
        entities_by_name: dict[str, dict[str, object]],
    ) -> bool:
        position = self._lookup_entity_position(entity)
        if position is None:
            return True
        if name in {"task_board", "tabletop", "cable_0", self.config.source_entity_name}:
            return True
        if position[2] > 0.5:
            return True
        board = entities_by_name.get("task_board")
        board_position = self._lookup_entity_position(board) if isinstance(board, dict) else None
        if board_position is None or board_position[2] <= 0.5:
            return True
        return False

    def _candidate_parent_name(
        self,
        *,
        name: str,
        entities_by_name: dict[str, dict[str, object]],
    ) -> str | None:
        if name.endswith("_entrance"):
            candidate = name[: -len("_entrance")]
            if candidate in entities_by_name:
                return candidate
        if name in {"lc_plug_link", "sc_plug_link", "sfp_module_link"} and "cable_0" in entities_by_name:
            return "cable_0"
        if name.startswith("nic_card_mount_") and "task_board" in entities_by_name:
            return "task_board"
        if name == "nic_card_link":
            for candidate in sorted(entities_by_name.keys()):
                if candidate.startswith("nic_card_mount_"):
                    return candidate
        if re.fullmatch(r"sfp_port_\d+_link", name):
            if "nic_card_link" in entities_by_name:
                return "nic_card_link"
        if re.fullmatch(r"sc_port_base_link", name):
            for candidate in sorted(entities_by_name.keys()):
                if candidate.startswith("sc_port_") and candidate in entities_by_name:
                    return candidate
        return None

    def _entity_pose_dict(
        self,
        entity: dict[str, object],
    ) -> dict[str, list[float]] | None:
        position = self._lookup_entity_position(entity)
        orientation = self._lookup_entity_orientation(entity)
        if position is None or orientation is None:
            return None
        return {"position": list(position), "orientation": list(orientation)}

    def _compose_pose_dicts(
        self,
        parent_pose: dict[str, list[float]],
        child_pose: dict[str, list[float]],
    ) -> dict[str, list[float]]:
        parent_position = parent_pose["position"]
        parent_orientation = self._normalize_quaternion(parent_pose["orientation"])
        child_position = child_pose["position"]
        child_orientation = self._normalize_quaternion(child_pose["orientation"])
        rotated_child = self._rotate_vector_by_quaternion(child_position, parent_orientation)
        return {
            "position": [
                float(parent_axis + child_axis)
                for parent_axis, child_axis in zip(parent_position, rotated_child)
            ],
            "orientation": self._normalize_quaternion(
                self._multiply_quaternions(parent_orientation, child_orientation)
            ),
        }

    def _rotate_vector_by_quaternion(
        self,
        vector: list[float],
        quaternion: list[float],
    ) -> list[float]:
        x, y, z, w = quaternion
        return [
            (1.0 - 2.0 * (y * y + z * z)) * vector[0]
            + (2.0 * (x * y - z * w)) * vector[1]
            + (2.0 * (x * z + y * w)) * vector[2],
            (2.0 * (x * y + z * w)) * vector[0]
            + (1.0 - 2.0 * (x * x + z * z)) * vector[1]
            + (2.0 * (y * z - x * w)) * vector[2],
            (2.0 * (x * z - y * w)) * vector[0]
            + (2.0 * (y * z + x * w)) * vector[1]
            + (1.0 - 2.0 * (x * x + y * y)) * vector[2],
        ]

    def _merge_entity_views(
        self,
        *,
        primary_entities: list[dict[str, object]],
        secondary_entities: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        merged_by_name: dict[str, dict[str, object]] = {}
        for entity in secondary_entities:
            name = entity.get("name")
            if isinstance(name, str):
                merged_by_name[name] = dict(entity)
        for entity in primary_entities:
            name = entity.get("name")
            if not isinstance(name, str):
                continue
            existing = merged_by_name.get(name, {})
            combined = dict(existing)
            combined.update(entity)
            existing_pose = existing.get("pose") if isinstance(existing.get("pose"), dict) else {}
            current_pose = entity.get("pose") if isinstance(entity.get("pose"), dict) else {}
            pose = dict(existing_pose)
            pose.update(current_pose)
            if pose:
                combined["pose"] = pose
                if "position" not in combined and isinstance(pose.get("position"), list):
                    combined["position"] = pose["position"]
                if "orientation" not in combined and isinstance(pose.get("orientation"), list):
                    combined["orientation"] = pose["orientation"]
            merged_by_name[name] = combined
        return [merged_by_name[name] for name in sorted(merged_by_name.keys())]

    def _extract_live_state_entities(self, payload: str) -> list[dict[str, object]]:
        entities_by_id = {
            entity_id: name for name, entity_id in self.config.live_state_entity_ids
        }
        entities: list[dict[str, object]] = []
        blocks = re.findall(
            r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
            payload,
            re.S,
        )
        for entity_id_text, body in blocks:
            entity_id = int(entity_id_text)
            entity_name = entities_by_id.get(entity_id)
            if entity_name is None:
                continue
            pose = self._extract_state_pose(body)
            if pose is None:
                continue
            entities.append(
                {
                    "name": entity_name,
                    "id": entity_id,
                    "position": pose["position"],
                    "orientation": pose["orientation"],
                    "pose": dict(pose),
                }
            )

        tabletop_orientation = self._quaternion_from_rpy(*self.config.tabletop_rpy)
        entities.append(
            {
                "name": "tabletop",
                "position": list(self.config.tabletop_position),
                "orientation": tabletop_orientation,
                "pose": {
                    "position": list(self.config.tabletop_position),
                    "orientation": tabletop_orientation,
                },
            }
        )
        return entities

    def _extract_state_pose(self, payload: str) -> dict[str, list[float]] | None:
        match = re.search(
            r'type: 10918813941671183356\n\s+component: "([^"]+)"',
            payload,
            re.S,
        )
        if match is None:
            return None
        values = [float(value) for value in match.group(1).split()]
        if len(values) != 6:
            return None
        x, y, z, roll, pitch, yaw = values
        return {
            "position": [x, y, z],
            "orientation": self._quaternion_from_rpy(roll, pitch, yaw),
        }

    def _quaternion_from_rpy(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> list[float]:
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)
        return [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]

    def _extract_state_step_count(self, payload: str) -> int:
        step_count = self._extract_scalar_int(payload, "step_count")
        if step_count is not None:
            return step_count
        iterations_match = re.search(r"iterations:\s*(\d+)", payload)
        if iterations_match is None:
            return 0
        return int(iterations_match.group(1))

    def _build_task_geometry(
        self,
        entities_by_name: dict[str, dict[str, object]],
    ) -> dict[str, object]:
        """Build a small task-relevant geometry slice from named entities."""
        geometry: dict[str, object] = {}
        tracked_entity_pair = self._build_relative_geometry(
            entities_by_name,
            source_name=self.config.source_entity_name,
            target_name=self.config.target_entity_name,
        )
        if tracked_entity_pair is not None:
            geometry["tracked_entity_pair"] = tracked_entity_pair
            if self._uses_default_pair():
                geometry["robot_to_task_board"] = dict(tracked_entity_pair)
        return geometry

    def _uses_default_pair(self) -> bool:
        """Return whether the configured tracked pair matches the legacy default."""
        return (
            self.config.source_entity_name == "robot"
            and self.config.target_entity_name == "task_board"
        )

    def _build_relative_geometry(
        self,
        entities_by_name: dict[str, dict[str, object]],
        *,
        source_name: str,
        target_name: str,
    ) -> dict[str, object] | None:
        """Build relative position geometry for a named entity pair."""
        source = self._lookup_named_entity(entities_by_name, source_name)
        target = self._lookup_named_entity(entities_by_name, target_name)
        if source is None or target is None:
            return None

        source_position = self._lookup_entity_position(source)
        target_position = self._lookup_entity_position(target)
        source_orientation = self._lookup_entity_orientation(source)
        target_orientation = self._lookup_entity_orientation(target)
        if (
            source_position is None
            or target_position is None
            or source_orientation is None
            or target_orientation is None
        ):
            return None

        relative_position = [
            target_axis - source_axis
            for source_axis, target_axis in zip(source_position, target_position)
        ]
        distance = math.sqrt(sum(axis * axis for axis in relative_position))
        relative_orientation = self._relative_orientation(
            source_orientation,
            target_orientation,
        )
        orientation_error = self._orientation_error(
            source_orientation,
            target_orientation,
        )
        distance_success = distance <= self.config.success_distance_threshold
        orientation_success = self._orientation_success(orientation_error)
        success = distance_success and orientation_success
        return {
            "source": source_name,
            "target": target_name,
            "relative_position": relative_position,
            "distance": distance,
            "source_orientation": source_orientation,
            "target_orientation": target_orientation,
            "relative_orientation": relative_orientation,
            "orientation_error": orientation_error,
            "success_threshold": self.config.success_distance_threshold,
            "orientation_success_threshold": self.config.orientation_success_threshold,
            "distance_success": distance_success,
            "orientation_success": orientation_success,
            "success": success,
        }

    def _compute_step_outcome(
        self,
        observation: dict[str, object],
    ) -> tuple[float, bool, bool, dict[str, object]]:
        """Compute the minimal geometry-based step outcome for the real path."""
        task_geometry = observation.get("task_geometry")
        if not isinstance(task_geometry, dict):
            return 0.0, False, self._compute_truncation(observation), {
                "mode": self.config.reward_mode,
                "reason": "missing_task_geometry",
            }

        pair_geometry = task_geometry.get("robot_to_task_board")
        if not isinstance(pair_geometry, dict):
            pair_geometry = task_geometry.get("tracked_entity_pair")
        if not isinstance(pair_geometry, dict):
            return 0.0, False, self._compute_truncation(observation), {
                "mode": self.config.reward_mode,
                "reason": "missing_tracked_pair",
            }

        distance = pair_geometry.get("distance")
        success = pair_geometry.get("success")
        if not isinstance(distance, float) or not isinstance(success, bool):
            return 0.0, False, self._compute_truncation(observation), {
                "mode": self.config.reward_mode,
                "reason": "missing_distance_or_success",
            }

        if self.config.reward_mode == "official_tier3":
            reward, reward_details = self._official_tier3_scorer.score(
                tracked_pair=pair_geometry,
                initial_distance=self._resolve_initial_tracked_distance(observation),
            )
        else:
            reward = -distance
            if success:
                reward += self.config.success_bonus
            reward_details = {
                "mode": "heuristic",
                "distance": distance,
                "success_bonus": self.config.success_bonus,
                "tracked_pair_success": success,
            }
        return reward, success, self._compute_truncation(observation), reward_details

    def _compute_truncation(self, observation: dict[str, object]) -> bool:
        """Return whether the configured real episode budget has been reached."""
        max_episode_steps = self.config.max_episode_steps
        if max_episode_steps is None:
            return False
        step_count = observation.get("step_count")
        if not isinstance(step_count, int):
            return False
        return step_count >= max_episode_steps

    def _lookup_named_entity(
        self,
        entities_by_name: dict[str, dict[str, object]],
        name: str,
    ) -> dict[str, object] | None:
        """Return a named entity dict if present and well formed."""
        entity = entities_by_name.get(name)
        if not isinstance(entity, dict):
            scoped_matches = [
                candidate
                for candidate, value in entities_by_name.items()
                if isinstance(value, dict)
                and (
                    candidate.endswith(f"::{name}")
                    or f"::{name}::" in candidate
                    or candidate.endswith(f"/{name}")
                    or f"/{name}/" in candidate
                )
            ]
            if len(scoped_matches) == 1:
                entity = entities_by_name.get(scoped_matches[0])
            else:
                return None
        if not isinstance(entity, dict):
            return None
        return entity

    def _lookup_current_joint_positions(
        self,
        observation: dict[str, object],
    ) -> list[float]:
        """Return current joint positions in configured joint order."""
        joints_by_name = observation.get("joints_by_name")
        if isinstance(joints_by_name, dict):
            positions: list[float] = []
            for joint_name in self.config.joint_names:
                joint_state = joints_by_name.get(joint_name)
                if not isinstance(joint_state, dict):
                    break
                position = joint_state.get("position")
                if not isinstance(position, float):
                    break
                positions.append(position)
            if len(positions) == len(self.config.joint_names):
                return positions

        joint_names = observation.get("joint_names")
        joint_positions = observation.get("joint_positions")
        if (
            isinstance(joint_names, list)
            and isinstance(joint_positions, list)
            and len(joint_names) == len(joint_positions)
        ):
            positions_by_name = {
                name: position
                for name, position in zip(joint_names, joint_positions)
                if isinstance(name, str) and isinstance(position, float)
            }
            positions = [positions_by_name.get(joint_name) for joint_name in self.config.joint_names]
            if all(isinstance(position, float) for position in positions):
                return [float(position) for position in positions]

        raise RuntimeError(
            "Real observation did not contain joint positions for all configured joints."
        )

    def _lookup_entity_position(self, entity: dict[str, object]) -> list[float] | None:
        """Return an entity position if the parsed payload contains one."""
        position = entity.get("position")
        if isinstance(position, list) and len(position) == 3 and all(
            isinstance(value, float) for value in position
        ):
            return position
        return None

    def _lookup_entity_orientation(self, entity: dict[str, object]) -> list[float] | None:
        """Return an entity orientation if the parsed payload contains one."""
        orientation = entity.get("orientation")
        if isinstance(orientation, list) and len(orientation) == 4 and all(
            isinstance(value, float) for value in orientation
        ):
            return orientation
        return None

    def _orientation_success(self, orientation_error: float) -> bool:
        """Return whether the optional orientation threshold is satisfied."""
        threshold = self.config.orientation_success_threshold
        if threshold is None:
            return True
        return orientation_error <= threshold

    def _normalize_quaternion(self, quaternion: list[float]) -> list[float]:
        """Return a normalized quaternion."""
        norm = math.sqrt(sum(component * component for component in quaternion))
        if norm == 0.0:
            raise ValueError("Quaternion norm must be non-zero.")
        return [component / norm for component in quaternion]

    def _conjugate_quaternion(self, quaternion: list[float]) -> list[float]:
        """Return the quaternion conjugate."""
        x, y, z, w = quaternion
        return [-x, -y, -z, w]

    def _multiply_quaternions(
        self,
        left: list[float],
        right: list[float],
    ) -> list[float]:
        """Multiply two quaternions in xyzw order."""
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]

    def _relative_orientation(
        self,
        source_orientation: list[float],
        target_orientation: list[float],
    ) -> list[float]:
        """Return the normalized source-to-target relative orientation."""
        normalized_source = self._normalize_quaternion(source_orientation)
        normalized_target = self._normalize_quaternion(target_orientation)
        return self._normalize_quaternion(
            self._multiply_quaternions(
                normalized_target,
                self._conjugate_quaternion(normalized_source),
            )
        )

    def _orientation_error(
        self,
        source_orientation: list[float],
        target_orientation: list[float],
    ) -> float:
        """Return the angular distance between two quaternions in radians."""
        normalized_source = self._normalize_quaternion(source_orientation)
        normalized_target = self._normalize_quaternion(target_orientation)
        dot = sum(
            source_component * target_component
            for source_component, target_component in zip(
                normalized_source,
                normalized_target,
            )
        )
        clamped = min(1.0, max(0.0, abs(dot)))
        return 2.0 * math.acos(clamped)

    def _extract_scalar_string(self, payload: str, key: str) -> str | None:
        """Extract a simple string scalar from a textproto-like payload."""
        quoted = re.search(rf"{re.escape(key)}:\s*\"([^\"]+)\"", payload)
        if quoted:
            return quoted.group(1)
        plain = re.search(rf"{re.escape(key)}:\s*([^\s{{}}]+)", payload)
        if plain:
            return plain.group(1)
        return None

    def _extract_scalar_int(self, payload: str, key: str) -> int | None:
        """Extract an integer scalar from a textproto-like payload."""
        match = re.search(rf"{re.escape(key)}:\s*(-?\d+)", payload)
        if match is None:
            return None
        return int(match.group(1))

    def _extract_entities(self, payload: str) -> list[dict[str, object]]:
        """Extract entity pose data from a textproto-like world-state payload."""
        entities: list[dict[str, object]] = []
        for block in self._extract_repeated_blocks(payload, "entity"):
            name = self._extract_scalar_string(block, "name")
            if name is None:
                continue
            entity: dict[str, object] = {"name": name}
            entity_id = self._extract_scalar_int(block, "id")
            if entity_id is not None:
                entity["id"] = entity_id
            pose_block = self._extract_first_block(block, "pose")
            if pose_block is not None:
                pose: dict[str, list[float]] = {}
                position = self._extract_xyz_vector(pose_block, "position")
                if position is not None:
                    pose["position"] = position
                    entity["position"] = position
                orientation = self._extract_xyzw_vector(pose_block, "orientation")
                if orientation is not None:
                    pose["orientation"] = orientation
                    entity["orientation"] = orientation
                if pose:
                    entity["pose"] = pose
            entities.append(entity)
        return entities

    def _extract_joints(self, payload: str) -> list[dict[str, object]]:
        """Extract joint position data from a textproto-like world-state payload."""
        joints: list[dict[str, object]] = []
        for block in self._extract_repeated_blocks(payload, "joint"):
            name = self._extract_scalar_string(block, "name")
            if name is None:
                continue
            joint: dict[str, object] = {"name": name}
            position = self._extract_scalar_float(block, "position")
            if position is None:
                axis1_block = self._extract_first_block(block, "axis1")
                if axis1_block is not None:
                    position = self._extract_scalar_float(axis1_block, "position")
            if position is None:
                continue
            joint["position"] = position
            joints.append(joint)
        return joints

    def _extract_poses(self, payload: str) -> list[dict[str, object]]:
        """Extract entity pose data from Gazebo pose-info payloads."""
        entities: list[dict[str, object]] = []
        for block in self._extract_repeated_blocks(payload, "pose"):
            name = self._extract_scalar_string(block, "name")
            if name is None:
                continue
            entity: dict[str, object] = {"name": name}
            entity_id = self._extract_scalar_int(block, "id")
            if entity_id is not None:
                entity["id"] = entity_id
            position = self._extract_xyz_vector(block, "position")
            orientation = self._extract_xyzw_vector_allow_partial(block, "orientation")
            pose: dict[str, list[float]] = {}
            if position is not None:
                pose["position"] = position
                entity["position"] = position
            if orientation is not None:
                pose["orientation"] = orientation
                entity["orientation"] = orientation
            if pose:
                entity["pose"] = pose
            entities.append(entity)
        return entities

    def _decode_joint_components(self, payload: str) -> list[dict[str, object]]:
        """Decode live joint positions from SerializedStepMap component payloads."""
        blocks = re.findall(
            r"entities \{\n    key: (\d+)\n    value \{(.*?)\n    \}\n  \}",
            payload,
            re.S,
        )
        decoded_positions: list[float] = []
        for _, body in blocks:
            for match in re.finditer(
                r'type: (\d+)\n          component: "((?:\\.|[^"])*)"',
                body,
                re.S,
            ):
                component_type, component_payload = match.groups()
                if component_type != "8319580315957903596":
                    continue
                raw = self._decode_textproto_bytes(component_payload)
                if len(raw) < 10 or raw[0] != 10 or raw[1] != 8:
                    continue
                decoded_positions.append(struct.unpack("<d", raw[2:10])[0])
                break
        if len(decoded_positions) < len(self.config.joint_names):
            return []
        return [
            {"name": joint_name, "position": float(position)}
            for joint_name, position in zip(
                self.config.joint_names,
                decoded_positions[: len(self.config.joint_names)],
            )
        ]

    def _decode_textproto_bytes(self, payload: str) -> bytes:
        """Decode an escaped textproto bytes literal into raw bytes."""
        return ast.literal_eval(f'"{payload}"').encode("latin1")

    def _extract_xyz_vector(self, payload: str, key: str) -> list[float] | None:
        """Extract a three-axis vector block if all values are present."""
        block = self._extract_first_block(payload, key)
        if block is None:
            return None
        values: list[float] = []
        for axis in ("x", "y", "z"):
            value = self._extract_scalar_float(block, axis)
            if value is None:
                return None
            values.append(value)
        return values

    def _extract_xyzw_vector(self, payload: str, key: str) -> list[float] | None:
        """Extract a quaternion-like vector block if all values are present."""
        block = self._extract_first_block(payload, key)
        if block is None:
            return None
        values: list[float] = []
        for axis in ("x", "y", "z", "w"):
            value = self._extract_scalar_float(block, axis)
            if value is None:
                return None
            values.append(value)
        return values

    def _extract_xyzw_vector_allow_partial(
        self,
        payload: str,
        key: str,
    ) -> list[float] | None:
        """Extract a quaternion block, defaulting missing axes to zero."""
        block = self._extract_first_block(payload, key)
        if block is None:
            return None
        values: list[float] = []
        for axis in ("x", "y", "z", "w"):
            value = self._extract_scalar_float(block, axis)
            values.append(0.0 if value is None else value)
        return values

    def _extract_scalar_float(self, payload: str, key: str) -> float | None:
        """Extract a float scalar from a textproto-like payload."""
        match = re.search(
            rf"{re.escape(key)}:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            payload,
        )
        if match is None:
            return None
        return float(match.group(1))

    def _extract_first_block(self, payload: str, key: str) -> str | None:
        """Extract the first nested block for a key from a textproto-like payload."""
        blocks = self._extract_repeated_blocks(payload, key)
        if not blocks:
            return None
        return blocks[0]

    def _extract_repeated_blocks(self, payload: str, key: str) -> list[str]:
        """Extract repeated nested blocks for a key from a textproto-like payload."""
        prefix = f"{key} {{"
        blocks: list[str] = []
        start = 0
        while True:
            index = payload.find(prefix, start)
            if index < 0:
                return blocks
            brace_start = payload.find("{", index)
            if brace_start < 0:
                return blocks
            depth = 0
            cursor = brace_start
            while cursor < len(payload):
                char = payload[cursor]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        blocks.append(payload[brace_start + 1 : cursor])
                        start = cursor + 1
                        break
                cursor += 1
            else:
                return blocks

    def _world_name(self) -> str:
        discovered_world_name = getattr(self, "_discovered_world_name", None)
        if isinstance(discovered_world_name, str) and discovered_world_name:
            return discovered_world_name

        if self.config.world_name:
            return self.config.world_name

        running_world_name = self._discover_running_world_name()
        if running_world_name:
            self._discovered_world_name = running_world_name
            return running_world_name

        return self._world_name_from_path()

    def _world_name_from_path(self) -> str:
        """Infer the configured world name from the world file."""
        if self.config.world_name:
            return self.config.world_name

        world_path = Path(self.config.world_path)
        try:
            root = ET.parse(world_path).getroot()
        except ET.ParseError:
            return world_path.stem

        world_element = root.find("world")
        if world_element is not None:
            world_name = world_element.get("name")
            if world_name:
                return world_name

        return world_path.stem

    def _discover_running_world_name(self) -> str | None:
        service_list = self._try_run(["service", "-l"])
        if not service_list:
            return None

        world_names: set[str] = set()
        preferred_world_names: set[str] = set()
        for service_name in service_list.splitlines():
            match = re.search(r"/world/([^/\s]+)/", service_name)
            if match is None:
                continue
            world_name = match.group(1)
            world_names.add(world_name)
            if service_name.strip().endswith(("/control", "/joint_target")):
                preferred_world_names.add(world_name)

        candidates = preferred_world_names or world_names
        if len(candidates) == 1:
            return next(iter(candidates))
        if self.config.world_name in candidates:
            return self.config.world_name
        return None


@dataclass
class GazeboTransportClient(GazeboCliClient):
    """Persistent Gazebo Transport client backed by the C++ helper process."""

    def __post_init__(self) -> None:
        super().__post_init__()
        world_name = self._world_name_from_path()
        self._last_state_generation: int | None = None
        self._last_pose_generation: int | None = None
        self._bridge = GazeboTransportBridge(
            GazeboTransportBridgeConfig(
                world_name=world_name,
                state_topic=f"/world/{world_name}/state",
                pose_topic=f"/world/{world_name}/pose/info",
                helper_executable=self.config.transport_helper_executable,
                startup_timeout_s=self.config.helper_startup_timeout_s,
                request_timeout_s=self.config.helper_request_timeout_s,
                startup_settle_s=self.config.helper_startup_settle_s,
                require_pose_for_ready=self.config.require_pose_for_ready,
            )
        )

    def _transport_backend_name(self) -> str:
        return "transport"

    def _publish_joint_target_topics(
        self,
        *,
        joint_names: list[str],
        positions: list[float],
    ) -> bool:
        del joint_names, positions
        return False

    def _transport_health_flags(self) -> dict[str, object]:
        return self._bridge.health_flags()

    def close(self) -> None:
        super().close()
        self._bridge.close()

    def get_observation(
        self,
        request: GetObservationRequest,
    ) -> GetObservationResponse:
        decoded = GetObservationRequest.from_dict(request.to_dict())
        del decoded
        world_name = self._world_name()
        state_topic = f"/world/{world_name}/state"
        pose_topic = f"/world/{world_name}/pose/info"
        try:
            response = self._bridge.request(
                {
                    "op": "get_observation",
                    "timeout_ms": int(self.config.timeout * 1000),
                    "pose_timeout_ms": int(min(self.config.timeout, 1.0) * 1000),
                }
            )
        except GazeboTransportBridgeError:
            return self._get_observation_via_cli_fallback(
                world_name=world_name,
                state_topic=state_topic,
                pose_topic=pose_topic,
            )
        state_payload = response["state_text"]
        pose_payload = response.get("pose_text")
        state_generation = response.get("state_generation")
        pose_generation = response.get("pose_generation")
        if isinstance(state_generation, int):
            self._last_state_generation = state_generation
        if isinstance(pose_generation, int):
            self._last_pose_generation = pose_generation
        observation = self._parse_live_observation(
            state_payload=state_payload,
            pose_payload=pose_payload if isinstance(pose_payload, str) else None,
            world_name=world_name,
        )
        if observation["entity_count"] == 0 and observation["joint_count"] == 0:
            observation = self._parse_state_text(state_payload)
        return GetObservationResponse(
            observation=self._logicalize_observation(observation),
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "transport_backend": "transport",
                "observation_topic": state_topic,
                "pose_topic": pose_topic,
                "state_text": self._logicalize_state_text(state_payload),
                "state_text_raw": state_payload,
                "pose_text": pose_payload,
                "sim_step_count_raw": observation.get("step_count"),
                "state_generation": state_generation,
                "pose_generation": pose_generation,
                "world_control_ok": True,
                "first_observation_ok": True,
                "fallback_used": False,
                **self._transport_health_flags(),
            },
        )

    def _get_observation_via_cli_fallback(
        self,
        *,
        world_name: str,
        state_topic: str,
        pose_topic: str,
    ) -> GetObservationResponse:
        state_payload = self._cli_read_topic_sample(state_topic)
        pose_payload = self._cli_read_pose_sample(pose_topic)
        observation = self._parse_live_observation(
            state_payload=state_payload,
            pose_payload=pose_payload,
            world_name=world_name,
        )
        if observation["entity_count"] == 0 and observation["joint_count"] == 0:
            observation = self._parse_state_text(state_payload)
        return GetObservationResponse(
            observation=self._logicalize_observation(observation),
            info={
                "backend": "gazebo",
                "runtime": "gazebo",
                "transport_backend": "transport_cli_fallback",
                "observation_topic": state_topic,
                "pose_topic": pose_topic,
                "state_text": self._logicalize_state_text(state_payload),
                "state_text_raw": state_payload,
                "pose_text": pose_payload,
                "sim_step_count_raw": observation.get("step_count"),
                "state_generation": None,
                "pose_generation": None,
                "world_control_ok": True,
                "first_observation_ok": True,
                "fallback_used": True,
                **self._transport_health_flags(),
            },
        )

    def _read_state_sample(
        self,
        *,
        topic: str,
        after_generation: int | None,
    ) -> tuple[str, int | None]:
        try:
            response = self._bridge.request(
                {
                    "op": "get_observation",
                    "after_generation": after_generation,
                    "timeout_ms": int(self.config.timeout * 1000),
                    "pose_timeout_ms": 0,
                }
            )
        except GazeboTransportBridgeError as exc:
            message = str(exc)
            if "timed out waiting for state sample" in message or "timed out waiting for transport bridge response" in message:
                raise TimeoutError(message) from exc
            raise
        payload = response["state_text"]
        generation = response.get("state_generation")
        if isinstance(generation, int):
            self._last_state_generation = generation
            return payload, generation
        return payload, None

    def _read_pose_sample(
        self,
        *,
        topic: str,
        after_generation: int | None = None,
    ) -> str | None:
        del topic
        try:
            response = self._bridge.request(
                {
                    "op": "get_observation",
                    "pose_after_generation": after_generation,
                    "timeout_ms": int(self.config.timeout * 1000),
                    "pose_timeout_ms": int(min(self.config.timeout, 1.0) * 1000),
                }
            )
        except GazeboTransportBridgeError:
            return None
        pose_payload = response.get("pose_text")
        pose_generation = response.get("pose_generation")
        if isinstance(pose_generation, int):
            self._last_pose_generation = pose_generation
        return pose_payload if isinstance(pose_payload, str) else None

    def _read_state_sample_after_world_step(self, topic: str) -> str | None:
        try:
            response = self._bridge.request(
                {
                    "op": "world_control",
                    "service": f"/world/{self._world_name()}/control",
                    "multi_step": 1,
                    "timeout_ms": int(self.config.timeout * 1000),
                }
            )
            if not response:
                return None
            state_response = self._bridge.request(
                {
                    "op": "get_observation",
                    "timeout_ms": int(min(self.config.timeout, 2.0) * 1000),
                    "pose_timeout_ms": 0,
                }
            )
        except Exception:
            return self._cli_read_state_sample_after_world_step(topic)
        state_text = state_response.get("state_text")
        return state_text if isinstance(state_text, str) else None

    def _current_state_generation(self) -> int | None:
        try:
            response = self._bridge.request({"op": "status"})
        except Exception:
            return self._last_state_generation
        generation = response.get("state_generation")
        if isinstance(generation, int):
            self._last_state_generation = generation
            return generation
        return self._last_state_generation

    def _current_pose_generation(self) -> int | None:
        try:
            response = self._bridge.request({"op": "status"})
        except Exception:
            return self._last_pose_generation
        generation = response.get("pose_generation")
        if isinstance(generation, int):
            self._last_pose_generation = generation
            return generation
        return self._last_pose_generation

    def _cli_run(self, args: list[str]) -> str:
        completed = subprocess.run(
            [self.config.executable, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(self.config.timeout, 1.0),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Gazebo CLI command failed with exit code {completed.returncode}: "
                f"{[self.config.executable, *args]!r}. "
                f"stdout: {completed.stdout.strip()} stderr: {completed.stderr.strip()}"
            )
        return completed.stdout.strip()

    def _cli_try_run_with_timeout(self, args: list[str], *, timeout: float) -> str | None:
        try:
            completed = subprocess.run(
                [self.config.executable, *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    def _cli_read_topic_sample(self, topic: str) -> str:
        deadline = time.monotonic() + (self.config.timeout * 3.0)
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return self._cli_run(["topic", "-e", "-n", "1", "-t", topic])
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                last_error = exc
                if topic.endswith("/state"):
                    fallback_payload = self._cli_read_state_sample_after_world_step(topic)
                    if fallback_payload is not None:
                        return fallback_payload
                time.sleep(0.2)
        raise RuntimeError(f"Timed out reading Gazebo topic {topic}: {last_error}")

    def _cli_read_pose_sample(self, topic: str) -> str | None:
        return self._cli_try_run_with_timeout(
            ["topic", "-e", "-n", "1", "-t", topic],
            timeout=min(self.config.timeout, 1.0),
        )

    def _cli_read_state_sample_after_world_step(self, topic: str) -> str | None:
        process = subprocess.Popen(
            [self.config.executable, "topic", "-e", "-n", "1", "-t", topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(0.5)
            step_reply = self._cli_try_run_with_timeout(
                [
                    "service",
                    "-s",
                    f"/world/{self._world_name()}/control",
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    str(int(self.config.timeout * 1000)),
                    "--req",
                    "multi_step: 1",
                ],
                timeout=self.config.timeout,
            )
            if step_reply is None:
                process.kill()
                process.communicate()
                return None
            stdout, stderr = process.communicate(timeout=self.config.timeout)
            if process.returncode not in (0, None):
                return None
            if stderr.strip():
                return None
            return stdout.strip() or None
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            return None

    def _run(self, args: list[str]) -> str:
        if args[:2] == ["service", "-l"]:
            # Service discovery remains a CLI-only fallback; use configured world
            # names for the transport path.
            fallback = super()._try_run(args)
            if fallback is None:
                raise RuntimeError("Gazebo service discovery is not available")
            return fallback

        if len(args) < 2 or args[0] != "service" or args[1] != "-s":
            raise RuntimeError(f"Unsupported transport helper command: {args!r}")

        service = args[2]
        timeout_ms = int(self.config.timeout * 1000)
        if "--timeout" in args:
            timeout_ms = int(args[args.index("--timeout") + 1])

        if service.endswith("/control"):
            request_text = args[args.index("--req") + 1]
            payload: dict[str, object] = {
                "op": "world_control",
                "service": service,
                "timeout_ms": timeout_ms,
            }
            multi_step_match = re.search(r"multi_step:\s*(\d+)", request_text)
            if multi_step_match is not None:
                payload["multi_step"] = int(multi_step_match.group(1))
            if "reset:" in request_text and "all: true" in request_text:
                payload["reset_all"] = True
            response = self._bridge.request(payload)
            return response["reply_text"]

        if service.endswith("/set_pose"):
            request_text = args[args.index("--req") + 1]
            name_match = re.search(r'name:\s*"([^"]+)"', request_text)
            position_match = re.search(
                r"position:\s*\{x:\s*([^,]+),\s*y:\s*([^,]+),\s*z:\s*([^}]+)\}",
                request_text,
            )
            orientation_match = re.search(
                r"orientation:\s*\{x:\s*([^,]+),\s*y:\s*([^,]+),\s*z:\s*([^,]+),\s*w:\s*([^}]+)\}",
                request_text,
            )
            if name_match is None or position_match is None or orientation_match is None:
                raise RuntimeError(f"Could not parse set_pose request: {request_text}")
            response = self._bridge.request(
                {
                    "op": "set_pose",
                    "service": service,
                    "name": name_match.group(1),
                    "position": [
                        float(position_match.group(1)),
                        float(position_match.group(2)),
                        float(position_match.group(3)),
                    ],
                    "orientation": [
                        float(orientation_match.group(1)),
                        float(orientation_match.group(2)),
                        float(orientation_match.group(3)),
                        float(orientation_match.group(4)),
                    ],
                    "timeout_ms": timeout_ms,
                }
            )
            return response["reply_text"]

        if service.endswith("/joint_target"):
            request_text = args[args.index("--req") + 1]
            data_match = re.search(r'data:\s*"([^"]+)"', request_text)
            if data_match is None:
                raise RuntimeError(f"Could not parse joint_target request: {request_text}")
            response = self._bridge.request(
                {
                    "op": "joint_target",
                    "service": service,
                    "request_text": data_match.group(1),
                    "timeout_ms": timeout_ms,
                }
            )
            return response["reply_text"]

        raise RuntimeError(f"Unsupported service for transport helper: {service}")

    def _try_run(self, args: list[str]) -> str | None:
        try:
            return self._run(args)
        except Exception:
            return None

    def _uses_persistent_observation_transport(self) -> bool:
        return True
