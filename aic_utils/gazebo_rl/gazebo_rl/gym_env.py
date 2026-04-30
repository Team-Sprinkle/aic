from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from gazebo_rl.ipc import IPCMessage, IPCServer, JsonLineConnection
from gazebo_rl.runner import GazeboRLRunner, GazeboRLRunnerConfig
from gazebo_rl.score_parser import dense_training_reward, score_from_scoring_yaml


class GazeboRLEnv:
    def __init__(
        self,
        *,
        workspace_dir: str | Path,
        engine_config: str | None = None,
        sim_distrobox: str | None = None,
        ground_truth: bool = True,
        gazebo_gui: bool = False,
        launch_rviz: bool = False,
        max_steps: int = 25,
        per_trial_timeout_sec: float = 300.0,
        host: str = "127.0.0.1",
        port: int = 0,
        command_dt_sec: float = 0.05,
        results_dir: str | Path = "outputs/gazebo_rl/results",
        record_lerobot: bool = False,
        record_root: str | Path | None = None,
        record_repo_id: str = "local/gazebo_rl_rollout",
        record_single_task: str = "gazebo_rl rollout",
        record_video: bool = True,
        record_fps: int = 30,
        record_resume: bool = False,
        record_drain_sec: float = 20.0,
        record_image_writer_processes: int = 0,
        record_image_writer_threads_per_camera: int = 4,
        record_video_encoding_batch_size: int = 1,
        include_images: bool = False,
    ):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.max_steps = int(max_steps)
        self.per_trial_timeout_sec = float(per_trial_timeout_sec)
        self._server = IPCServer(host=host, port=port)
        self._conn: JsonLineConnection | None = None
        self._hello: dict[str, Any] | None = None
        self._closed = False
        self._step_count = 0
        self.runner = GazeboRLRunner(
            GazeboRLRunnerConfig(
                workspace_dir=self.workspace_dir,
                engine_config=engine_config,
                sim_distrobox=sim_distrobox,
                ground_truth=ground_truth,
                gazebo_gui=gazebo_gui,
                launch_rviz=launch_rviz,
                max_steps=max_steps,
                per_trial_timeout_sec=per_trial_timeout_sec,
                host=host,
                port=self._server.port,
                command_dt_sec=command_dt_sec,
                results_dir=Path(results_dir).resolve(),
                record_lerobot=record_lerobot,
                record_root=Path(record_root).resolve() if record_root is not None else None,
                record_repo_id=record_repo_id,
                record_single_task=record_single_task,
                record_video=record_video,
                record_fps=record_fps,
                record_resume=record_resume,
                record_drain_sec=record_drain_sec,
                record_image_writer_processes=record_image_writer_processes,
                record_image_writer_threads_per_camera=record_image_writer_threads_per_camera,
                record_video_encoding_batch_size=record_video_encoding_batch_size,
                include_images=include_images,
            )
        )

    @property
    def results_dir(self) -> Path:
        return self.runner.config.results_dir

    @property
    def first_observation_keys(self) -> list[str]:
        return sorted(self._last_obs.keys()) if hasattr(self, "_last_obs") else []

    def _recv_until_observation(self, deadline: float) -> IPCMessage:
        while time.monotonic() < deadline:
            if self._conn is None:
                raise RuntimeError("IPC connection is not open")
            timeout = max(0.1, min(30.0, deadline - time.monotonic()))
            try:
                msg = self._conn.recv(timeout_sec=timeout)
            except TimeoutError:
                continue
            if msg.type == "hello":
                self._hello = msg.payload
                continue
            if msg.type in {"observation", "done", "error"}:
                return msg
            raise RuntimeError(f"Unexpected IPC message type: {msg.type}")
        raise TimeoutError("Timed out waiting for observation from Gazebo RL bridge")

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("Cannot reset a closed GazeboRLEnv")
        self._step_count = 0
        self.runner.start()
        deadline = time.monotonic() + self.per_trial_timeout_sec
        while self._conn is None and time.monotonic() < deadline:
            self.runner.check_processes()
            try:
                self._conn = self._server.accept(timeout_sec=1.0)
            except TimeoutError:
                continue
        if self._conn is None:
            raise TimeoutError("Timed out waiting for Gazebo RL bridge connection")
        msg = self._recv_until_observation(deadline)
        if msg.type != "observation":
            raise RuntimeError(f"Expected first observation, got {msg.type}: {msg.payload}")
        obs = msg.payload["observation"]
        self._last_obs = obs
        return obs, {"hello": self._hello, "results_dir": str(self.results_dir)}

    def step(self, action: list[float] | tuple[float, ...]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("Call reset() before step()")
        self._conn.send("action", {"action": list(action), "step_count": self._step_count})
        self._step_count += 1
        deadline = time.monotonic() + self.per_trial_timeout_sec
        msg = self._recv_until_observation(deadline)
        if msg.type == "error":
            raise RuntimeError(f"Bridge error: {msg.payload}")
        if msg.type == "done":
            parsed = score_from_scoring_yaml(self.results_dir)
            return self._last_obs, dense_training_reward(terminal=True, results_dir=self.results_dir), True, False, {
                "done": msg.payload,
                "score": parsed,
            }
        obs = msg.payload["observation"]
        self._last_obs = obs
        terminated = bool(msg.payload.get("terminated", False))
        truncated = self._step_count >= self.max_steps
        reward = dense_training_reward(terminal=terminated, results_dir=self.results_dir)
        info = {"step_count": self._step_count, "results_dir": str(self.results_dir)}
        if truncated:
            self._conn.send("done", {"reason": "env_max_steps", "step_count": self._step_count})
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self._closed = True
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None
        self.runner.close()
        self._server.close()
