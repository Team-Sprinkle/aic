from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


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
                "--name",
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
        try:
            result = subprocess.run(
                ["distrobox", "list"],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as ex:
            raise RuntimeError(f"Could not list distrobox containers: {ex}") from ex
        names = {
            line.split("|")[1].strip()
            for line in result.stdout.splitlines()
            if "|" in line and not line.startswith("ID")
        }
        if self.config.sim_distrobox not in names:
            raise RuntimeError(
                f"Distrobox container '{self.config.sim_distrobox}' was not found. "
                "Create it first with the aic_eval image from docs/getting_started.md."
            )

    def close(self) -> None:
        for managed in reversed(self.processes):
            managed.terminate()
        self.processes.clear()
