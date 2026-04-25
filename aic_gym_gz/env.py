"""Gymnasium environment entry point."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .io import AicGazeboIO, MockGazeboIO, RosCameraSidecarIO
from .randomizer import AicEnvRandomizer
from .runtime import AicGazeboRuntime, MockStepperBackend, ScenarioGymGzBackend
from .task import AicInsertionTask


@dataclass
class AicInsertionEnv(gym.Env[dict[str, Any], np.ndarray]):
    metadata = {"render_modes": []}

    runtime: AicGazeboRuntime
    task: AicInsertionTask
    io: AicGazeboIO
    randomizer: AicEnvRandomizer
    trial_id: str | None = None

    def __post_init__(self) -> None:
        self.action_space = self.task.action_space
        self.observation_space = self.task.observation_space
        self._scenario = None
        self._state = None
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        scenario = self.randomizer.sample(
            seed=seed,
            trial_id=str(options.get("trial_id", self.trial_id)) if options.get("trial_id", self.trial_id) else None,
        )
        state = self.runtime.reset(seed=seed, scenario=scenario)
        print('{"env_stage":"runtime_reset_returned"}', flush=True)
        self.task.reset(scenario=scenario, initial_state=state)
        self._scenario = scenario
        self._state = state
        self._step_count = 0
        try:
            print('{"env_stage":"final_observation_begin"}', flush=True)
            observation = self.io.observation_from_state(
                state,
                include_images=self.task.include_images,
                step_count=self._step_count,
            )
            if (
                self._requires_live_wrist_images()
                and not self._observation_has_live_wrist_images(observation)
            ):
                self._state = self._warm_images_on_reset(state)
                observation = self.io.observation_from_state(
                    self._state,
                    include_images=True,
                    step_count=self._step_count,
                )
                if not self._observation_has_live_wrist_images(observation):
                    raise TimeoutError(
                        "Timed out waiting for real wrist camera images during reset warm-up."
                    )
                print('{"env_stage":"final_observation_done_after_warmup"}', flush=True)
            else:
                print('{"env_stage":"final_observation_done"}', flush=True)
        except TimeoutError as exc:
            if str(exc).startswith("Timed out waiting for real wrist camera images"):
                raise
            if not self.task.include_images:
                raise
            self._state = self._warm_images_on_reset(state)
            observation = self.io.observation_from_state(
                self._state,
                include_images=True,
                step_count=self._step_count,
            )
            if (
                self._requires_live_wrist_images()
                and not self._observation_has_live_wrist_images(observation)
            ):
                raise TimeoutError(
                    "Timed out waiting for real wrist camera images during reset retry."
                )
            print('{"env_stage":"final_observation_done_after_retry"}', flush=True)
        info = {
            "trial_id": scenario.trial_id,
            "scenario_metadata": scenario.metadata,
            "task_definition": next(iter(scenario.tasks.values())).__dict__,
            "reward_label": "rl_step_reward",
            "score_label": "gym_final_score",
            "runtime_scene_validation": dict(state.world_entities_summary.get("runtime_diagnostics", {})),
        }
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Environment must be reset before step().")
        sanitized = self.io.sanitize_action(action)
        previous_state = self._state
        current_state = self.runtime.step(sanitized, ticks=self.task.hold_action_ticks)
        self._step_count += 1
        observation = self.io.observation_from_state(
            current_state,
            include_images=self.task.include_images,
            step_count=self._step_count,
        )
        reward, terminated, truncated, info = self.task.evaluate_step(
            previous_state=previous_state,
            current_state=current_state,
            action=sanitized,
            step_count=self._step_count,
        )
        info["official_compatible_observation_semantics"] = {
            "wrench_is_current_sample_only": True,
            "auxiliary_force_contact_summary_is_non_official": True,
        }
        info["runtime_scene_validation"] = dict(
            current_state.world_entities_summary.get("runtime_diagnostics", {})
        )
        info["auxiliary_force_contact_summary"] = (
            current_state.auxiliary_force_contact_summary.to_dict()
        )
        if terminated or truncated:
            final_evaluation = self.task.final_evaluation()
            info["final_evaluation"] = final_evaluation
            info["evaluation"] = final_evaluation
        self._state = current_state
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.io.close()
        self.runtime.close()

    def _warm_images_on_reset(self, state):
        try:
            observation = self.io.observation_from_state(
                state,
                include_images=True,
                step_count=0,
            )
            if self._observation_has_live_wrist_images(observation):
                return state
        except TimeoutError:
            pass
        zero_action = np.zeros(6, dtype=np.float32)
        warmed_state = state
        warm_ticks = max(2, int(self.task.hold_action_ticks))
        for _ in range(160):
            warmed_state = self.runtime.step(zero_action, ticks=warm_ticks)
            try:
                observation = self.io.observation_from_state(
                    warmed_state,
                    include_images=True,
                    step_count=0,
                )
                if self._observation_has_live_wrist_images(observation):
                    return warmed_state
            except TimeoutError:
                continue
        return warmed_state

    @staticmethod
    def _observation_has_live_wrist_images(observation: dict[str, Any]) -> bool:
        images = observation.get("images")
        timestamps = observation.get("image_timestamps")
        if not isinstance(images, dict):
            return False
        if timestamps is None:
            return False
        timestamp_array = np.asarray(timestamps, dtype=np.float32).reshape(-1)
        if timestamp_array.size < 3 or not np.all(timestamp_array[:3] > 0.0):
            return False
        return all(
            name in images
            and images[name].size > 0
            and int(np.asarray(images[name], dtype=np.uint8).sum()) > 0
            for name in ("left", "center", "right")
        )

    def _requires_live_wrist_images(self) -> bool:
        return bool(self.task.include_images and isinstance(self.io, RosCameraSidecarIO))


def live_env_health_check(
    *,
    include_images: bool = False,
    enable_randomization: bool = False,
    ticks_per_step: int = 8,
    world_path: str | None = None,
    attach_to_existing: bool = False,
    transport_backend: str = "transport",
    timeout: float = 10.0,
    attach_ready_timeout: float | None = None,
    seed: int = 123,
    image_shape: tuple[int, int, int] = (256, 256, 3),
) -> dict[str, Any]:
    env = make_live_env(
        include_images=include_images,
        enable_randomization=enable_randomization,
        ticks_per_step=ticks_per_step,
        world_path=world_path,
        attach_to_existing=attach_to_existing,
        transport_backend=transport_backend,
        timeout=timeout,
        attach_ready_timeout=attach_ready_timeout,
        image_shape=image_shape,
    )
    try:
        observation, info = env.reset(seed=seed)
        summary = {
            "trial_id": info["trial_id"],
            "sim_tick": int(observation["sim_tick"]),
            "sim_time": float(observation["sim_time"]),
            "joint_count": int(observation["joint_positions"].shape[0]),
            "images_ready": False,
            "observation_keys": sorted(observation.keys()),
        }
        if include_images:
            summary["images_ready"] = all(
                observation["images"][name].shape == image_shape
                and observation["images"][name].dtype == np.uint8
                and int(observation["images"][name].sum()) > 0
                for name in ("left", "center", "right")
            )
        return summary
    finally:
        env.close()


def make_default_env(
    *,
    include_images: bool = False,
    enable_randomization: bool = True,
    ticks_per_step: int = 8,
    allow_mock_images: bool = False,
    image_shape: tuple[int, int, int] = (256, 256, 3),
) -> AicInsertionEnv:
    if include_images and not allow_mock_images:
        raise RuntimeError(
            "make_default_env() cannot provide real camera images. "
            "Use make_live_env(include_images=True, ...) for real wrist-camera data. "
            "Pass allow_mock_images=True only for tests or explicitly mock-only workflows."
        )
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=MockStepperBackend(),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(
            hold_action_ticks=ticks_per_step,
            include_images=include_images,
            image_shape=image_shape,
        ),
        io=MockGazeboIO(image_shape=image_shape),
        randomizer=AicEnvRandomizer(enable_randomization=enable_randomization),
    )


def make_live_env(
    *,
    include_images: bool = False,
    enable_randomization: bool = True,
    ticks_per_step: int = 8,
    world_path: str | None = None,
    attach_to_existing: bool = False,
    transport_backend: str = "transport",
    timeout: float = 10.0,
    attach_ready_timeout: float | None = None,
    image_shape: tuple[int, int, int] = (256, 256, 3),
    allow_synthetic_tcp_pose: bool = False,
    allow_synthetic_plug_pose: bool = False,
    use_controller_velocity_commands: bool = False,
    source_entity_name: str = "ati/tool_link",
    live_mode: str = "gazebo_training_fast",
    image_observation_mode: str = "artifact_validation",
    observation_transport_override: str | None = None,
    state_observation_mode: str | None = None,
) -> AicInsertionEnv:
    normalized_live_mode = str(live_mode).strip().lower()
    if normalized_live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"}:
        resolved_use_controller_velocity = False
    elif normalized_live_mode == "controller_velocity_wip":
        resolved_use_controller_velocity = True
    else:
        raise ValueError(
            "Unsupported live_mode. Expected 'gazebo_training_fast', "
            f"'gazebo_pose_delta_fast', or 'controller_velocity_wip', received {live_mode!r}."
        )
    resolved_transport_backend = transport_backend
    if normalized_live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"} and not attach_to_existing:
        resolved_transport_backend = "auto"
    image_ready_timeout_s = (
        1.0
        if normalized_live_mode in {"gazebo_training_fast", "gazebo_pose_delta_fast"}
        else 10.0
    )
    normalized_image_mode = str(image_observation_mode).strip().lower()
    if normalized_image_mode not in {"artifact_validation", "async_training"}:
        raise ValueError(
            "Unsupported image_observation_mode. Expected 'artifact_validation' or "
            f"'async_training', received {image_observation_mode!r}."
        )
    resolved_state_observation_mode = (
        str(state_observation_mode).strip().lower()
        if state_observation_mode is not None
        else ("synthetic_training" if normalized_image_mode == "async_training" else "honest_live")
    )
    resolved_allow_synthetic_tcp_pose = (
        bool(allow_synthetic_tcp_pose)
        and resolved_state_observation_mode == "synthetic_training"
    )
    resolved_allow_synthetic_plug_pose = (
        bool(allow_synthetic_plug_pose)
        and resolved_state_observation_mode == "synthetic_training"
    )
    return AicInsertionEnv(
        runtime=AicGazeboRuntime(
            backend=ScenarioGymGzBackend(
                world_path=world_path,
                timeout=timeout,
                attach_ready_timeout=attach_ready_timeout,
                attach_to_existing=attach_to_existing,
                transport_backend=resolved_transport_backend,
                source_entity_name=source_entity_name,
                allow_synthetic_tcp_pose=resolved_allow_synthetic_tcp_pose,
                allow_synthetic_plug_pose=resolved_allow_synthetic_plug_pose,
                use_controller_velocity_commands=(
                    bool(use_controller_velocity_commands) or resolved_use_controller_velocity
                ),
                live_mode=normalized_live_mode,
                observation_transport_override=observation_transport_override,
                state_observation_mode=resolved_state_observation_mode,
            ),
            ticks_per_step=ticks_per_step,
        ),
        task=AicInsertionTask(
            hold_action_ticks=ticks_per_step,
            include_images=include_images,
            image_shape=image_shape,
        ),
        io=RosCameraSidecarIO(image_shape=image_shape, ready_timeout_s=image_ready_timeout_s)
        if include_images and normalized_image_mode == "artifact_validation"
        else RosCameraSidecarIO(
            image_shape=image_shape,
            ready_timeout_s=10.0,
            start_bridge=True,
            blocking_bootstrap=True,
            allow_direct_fetch_fallback=True,
            background_direct_fetch=False,
        )
        if include_images
        else MockGazeboIO(image_shape=image_shape),
        randomizer=AicEnvRandomizer(enable_randomization=enable_randomization),
    )
