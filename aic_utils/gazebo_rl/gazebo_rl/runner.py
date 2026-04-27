from __future__ import annotations

import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _parse_distrobox_names(output: str) -> set[str]:
    names: set[str] = set()
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if "|" in stripped:
            columns = [column.strip() for column in stripped.split("|")]
            lowered = [column.lower() for column in columns]
            if "name" in lowered:
                continue
            if len(columns) >= 2 and columns[1]:
                names.add(columns[1])
            continue

        tokens = stripped.split()
        if not tokens or tokens[0].lower() in {"id", "name"}:
            continue
        if len(tokens) >= 2:
            names.add(tokens[1])
        else:
            names.add(tokens[0])
    return names


@dataclass
class GazeboRLRunnerConfig:
    workspace_dir: Path
    engine_config: str | None = None
    sim_distrobox: str | None = None
    ground_truth: bool = True
    gazebo_gui: bool = False
    launch_rviz: bool = False
    max_steps: int = 25
    per_trial_timeout_sec: float = 300.0
    host: str = "127.0.0.1"
    port: int = 8765
    command_dt_sec: float = 0.05
    results_dir: Path = field(default_factory=lambda: Path("outputs/gazebo_rl/results"))
    record_lerobot: bool = False
    record_root: Path | None = None
    record_repo_id: str = "local/gazebo_rl_rollout"
    record_single_task: str = "gazebo_rl rollout"
    record_video: bool = True
    record_fps: int = 30
    record_max_episodes: int = 1
    record_resume: bool = False
    record_drain_sec: float = 20.0
    record_image_writer_processes: int = 0
    record_image_writer_threads_per_camera: int = 4
    record_video_encoding_batch_size: int = 1


class ManagedProcess:
    def __init__(self, name: str, process: subprocess.Popen):
        self.name = name
        self.process = process

    def terminate(self, timeout_sec: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout_sec)


class GazeboRLRunner:
    def __init__(self, config: GazeboRLRunnerConfig):
        self.config = config
        self.processes: list[ManagedProcess] = []

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "DBX_CONTAINER_MANAGER": env.get("DBX_CONTAINER_MANAGER", "docker"),
                "AIC_GAZEBO_RL_HOST": self.config.host,
                "AIC_GAZEBO_RL_PORT": str(self.config.port),
                "AIC_GAZEBO_RL_COMMAND_DT_SEC": str(self.config.command_dt_sec),
                "AIC_GAZEBO_RL_MAX_STEPS": str(self.config.max_steps),
                "AIC_GAZEBO_RL_GROUND_TRUTH": _bool_arg(self.config.ground_truth),
                "AIC_RESULTS_DIR": str(self.config.results_dir),
            }
        )
        return env

    def simulation_command(self) -> list[str]:
        if self.config.sim_distrobox:
            cmd = [
                "distrobox",
                "enter",
                "-r",
                "--no-tty",
                self.config.sim_distrobox,
                "--",
                "/entrypoint.sh",
                f"ground_truth:={_bool_arg(self.config.ground_truth)}",
                "start_aic_engine:=true",
                f"gazebo_gui:={_bool_arg(self.config.gazebo_gui)}",
                f"launch_rviz:={_bool_arg(self.config.launch_rviz)}",
            ]
            if self.config.engine_config:
                cmd.append(f"aic_engine_config_file:={self.config.engine_config}")
            return cmd

        config = self.config.engine_config or str(
            self.config.workspace_dir / "aic_engine" / "config" / "sample_config.yaml"
        )
        return [
            "pixi",
            "run",
            "ros2",
            "launch",
            "aic_bringup",
            "aic_gz_bringup.launch.py",
            "start_aic_engine:=true",
            "shutdown_on_aic_engine_exit:=true",
            f"aic_engine_config_file:={config}",
            f"ground_truth:={_bool_arg(self.config.ground_truth)}",
            f"gazebo_gui:={_bool_arg(self.config.gazebo_gui)}",
            f"launch_rviz:={_bool_arg(self.config.launch_rviz)}",
        ]

    def policy_command(self) -> list[str]:
        return [
            "pixi",
            "run",
            "ros2",
            "run",
            "aic_model",
            "aic_model",
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "policy:=gazebo_rl.bridge_policy.GazeboRLBridgePolicy",
        ]

    def recorder_command(self) -> list[str]:
        record_root = self.config.record_root or (
            self.config.results_dir.parent / "lerobot_rollout_dataset"
        )
        cmd = [
            "pixi",
            "run",
            "aic-policy-recorder",
            f"--dataset.repo_id={self.config.record_repo_id}",
            f"--dataset.single_task={self.config.record_single_task}",
            f"--dataset.root={record_root}",
            f"--dataset.fps={self.config.record_fps}",
            f"--dataset.num_image_writer_processes={self.config.record_image_writer_processes}",
            (
                "--dataset.num_image_writer_threads_per_camera="
                f"{self.config.record_image_writer_threads_per_camera}"
            ),
            f"--dataset.video_encoding_batch_size={self.config.record_video_encoding_batch_size}",
            f"--max_episodes={self.config.record_max_episodes}",
            "--save_failed_episodes",
            "--action_mode=cartesian",
        ]
        if self.config.record_video:
            cmd.append("--dataset.video")
        else:
            cmd.append("--no-dataset.video")
        if self.config.record_resume:
            cmd.append("--dataset.resume")
        return cmd

    def start(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self._validate_distrobox()
        env = self._env()
        sim = subprocess.Popen(
            self.simulation_command(),
            cwd=self.config.workspace_dir,
            env=env,
            text=True,
        )
        self.processes.append(ManagedProcess("simulation", sim))
        time.sleep(2.0)
        if self.config.record_lerobot:
            recorder = subprocess.Popen(
                self.recorder_command(),
                cwd=self.config.workspace_dir,
                env=env,
                text=True,
            )
            self.processes.append(ManagedProcess("recorder", recorder))
            time.sleep(1.0)
        policy = subprocess.Popen(
            self.policy_command(),
            cwd=self.config.workspace_dir,
            env=env,
            text=True,
        )
        self.processes.append(ManagedProcess("policy", policy))

    def check_processes(self) -> None:
        for managed in self.processes:
            code = managed.process.poll()
            if code is not None:
                raise RuntimeError(f"{managed.name} process exited early with code {code}")

    def _validate_distrobox(self) -> None:
        if not self.config.sim_distrobox:
            return
        if shutil.which("distrobox") is None:
            raise RuntimeError(
                "distrobox is not installed or is not on PATH, but "
                f"--sim-distrobox {self.config.sim_distrobox!r} was provided."
            )
        try:
            result = subprocess.run(
                ["distrobox", "list"],
                check=True,
                capture_output=True,
                text=True,
                env=self._env(),
            )
        except OSError as ex:
            raise RuntimeError(f"Could not list distrobox containers: {ex}") from ex
        except subprocess.CalledProcessError as ex:
            detail = ex.stderr.strip() or ex.stdout.strip() or str(ex)
            raise RuntimeError(f"Could not list distrobox containers: {detail}") from ex

        names = _parse_distrobox_names(result.stdout)
        if self.config.sim_distrobox not in names:
            raise RuntimeError(
                f"Distrobox container '{self.config.sim_distrobox}' was not found. "
                "This is a user-created container name, not a toolkit resource. "
                "Pass --sim-distrobox <your-container-name>, or omit --sim-distrobox "
                "to use the local pixi launch path."
            )

    def close(self) -> None:
        recorder_processes = [managed for managed in self.processes if managed.name == "recorder"]
        for managed in recorder_processes:
            if managed.process.poll() is None and self.config.record_drain_sec > 0.0:
                try:
                    managed.process.wait(timeout=self.config.record_drain_sec)
                except subprocess.TimeoutExpired:
                    pass
        for managed in reversed([p for p in self.processes if p.name != "recorder"]):
            managed.terminate()
        self._cleanup_distrobox_processes()
        for managed in recorder_processes:
            managed.terminate()
        self.processes.clear()

    def _cleanup_distrobox_processes(self) -> None:
        if not self.config.sim_distrobox:
            return
        patterns = [
            "/entrypoint.sh",
            "aic_gz_bringup",
            "rmw_zenohd",
            "/aic_engine/aic_engine",
            "aic_adapter",
            "robot_state_publisher",
            "component_container",
            "topic_tools",
            "static_transform_publisher",
            "controller_manager/spawner",
        ]
        for signal_name in ("-INT", "-TERM"):
            for pattern in patterns:
                try:
                    subprocess.run(
                        [
                            "distrobox",
                            "enter",
                            "-r",
                            "--no-tty",
                            self.config.sim_distrobox,
                            "--",
                            "pkill",
                            signal_name,
                            "-f",
                            pattern,
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=self._env(),
                        timeout=5.0,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if signal_name == "-INT":
                time.sleep(2.0)
