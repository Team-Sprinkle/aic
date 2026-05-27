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
    sim_docker_container: str | None = None
    docker_host: str | None = None
    workspace_container: Path = Path("/home/chmin/yj/ws_aic/src/aic")
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
    clean_stale_zenoh: bool = True
    include_images: bool = False
    episode_config: Path | None = None
    preposition_start_near_gate: bool = False
    start_near_gate_mode: str = "task_board"
    preposition_max_steps: int = 240
    preposition_tolerance_m: float = 0.003
    preposition_max_translation_m: float = 0.003


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
        source_python_paths = [
            str(self.config.workspace_dir / "aic_utils" / "gazebo_rl"),
        ]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            source_python_paths.append(existing_pythonpath)
        env.update(
            {
                "DBX_CONTAINER_MANAGER": env.get("DBX_CONTAINER_MANAGER", "docker"),
                "PYTHONPATH": os.pathsep.join(source_python_paths),
                "AIC_GAZEBO_RL_HOST": self.config.host,
                "AIC_GAZEBO_RL_PORT": str(self.config.port),
                "AIC_GAZEBO_RL_COMMAND_DT_SEC": str(self.config.command_dt_sec),
                "AIC_GAZEBO_RL_MAX_STEPS": str(self.config.max_steps),
                "AIC_GAZEBO_RL_GROUND_TRUTH": _bool_arg(self.config.ground_truth),
                "AIC_GAZEBO_RL_INCLUDE_IMAGES": _bool_arg(self.config.include_images),
                "AIC_RESULTS_DIR": str(self.config.results_dir),
                "AIC_GAZEBO_RL_PREPOSITION_START_NEAR_GATE": _bool_arg(self.config.preposition_start_near_gate),
                "AIC_GAZEBO_RL_START_NEAR_GATE_MODE": self.config.start_near_gate_mode,
                "AIC_GAZEBO_RL_SIM_DISTROBOX": self.config.sim_distrobox or "",
                "AIC_GAZEBO_RL_PREPOSITION_MAX_STEPS": str(self.config.preposition_max_steps),
                "AIC_GAZEBO_RL_PREPOSITION_TOLERANCE_M": str(self.config.preposition_tolerance_m),
                "AIC_GAZEBO_RL_PREPOSITION_MAX_TRANSLATION_M": str(self.config.preposition_max_translation_m),
            }
        )
        if self.config.episode_config is not None:
            env["AIC_GAZEBO_RL_EPISODE_CONFIG"] = str(self.config.episode_config)
        return env

    def _runtime_env_exports(self) -> str:
        env = self._env()
        pythonpath_prefix = os.pathsep.join(
            [
                str(self.config.workspace_container / "aic_utils" / "gazebo_rl"),
                str(self.config.workspace_container / "aic_example_policies"),
            ]
        )
        runtime_env = {
            "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
            "AIC_GAZEBO_RL_HOST": env["AIC_GAZEBO_RL_HOST"],
            "AIC_GAZEBO_RL_PORT": env["AIC_GAZEBO_RL_PORT"],
            "AIC_GAZEBO_RL_COMMAND_DT_SEC": env["AIC_GAZEBO_RL_COMMAND_DT_SEC"],
            "AIC_GAZEBO_RL_MAX_STEPS": env["AIC_GAZEBO_RL_MAX_STEPS"],
            "AIC_GAZEBO_RL_GROUND_TRUTH": env["AIC_GAZEBO_RL_GROUND_TRUTH"],
            "AIC_GAZEBO_RL_INCLUDE_IMAGES": env["AIC_GAZEBO_RL_INCLUDE_IMAGES"],
            "AIC_RESULTS_DIR": self._container_path(self.config.results_dir),
            "AIC_GAZEBO_RL_PREPOSITION_START_NEAR_GATE": env["AIC_GAZEBO_RL_PREPOSITION_START_NEAR_GATE"],
            "AIC_GAZEBO_RL_START_NEAR_GATE_MODE": env["AIC_GAZEBO_RL_START_NEAR_GATE_MODE"],
            "AIC_GAZEBO_RL_SIM_DISTROBOX": env["AIC_GAZEBO_RL_SIM_DISTROBOX"],
            "AIC_GAZEBO_RL_PREPOSITION_MAX_STEPS": env["AIC_GAZEBO_RL_PREPOSITION_MAX_STEPS"],
            "AIC_GAZEBO_RL_PREPOSITION_TOLERANCE_M": env["AIC_GAZEBO_RL_PREPOSITION_TOLERANCE_M"],
            "AIC_GAZEBO_RL_PREPOSITION_MAX_TRANSLATION_M": env["AIC_GAZEBO_RL_PREPOSITION_MAX_TRANSLATION_M"],
        }
        if self.config.episode_config is not None:
            runtime_env["AIC_GAZEBO_RL_EPISODE_CONFIG"] = self._container_path(self.config.episode_config)
        exports = [f"export {key}={shlex_quote(value)}" for key, value in runtime_env.items()]
        exports.append(f"export PYTHONPATH={shlex_quote(pythonpath_prefix)}:${{PYTHONPATH:-}}")
        return "\n".join(exports)

    def _container_path(self, path: str | Path | None) -> str | None:
        if path is None:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            return str(candidate)
        try:
            rel = candidate.resolve().relative_to(self.config.workspace_dir.resolve())
        except ValueError:
            return str(candidate)
        return str(self.config.workspace_container / rel)

    def _docker_base_cmd(self) -> list[str]:
        cmd = ["docker"]
        docker_host = self.config.docker_host or os.environ.get("DOCKER_HOST")
        if docker_host:
            cmd.extend(["--host", docker_host])
        return cmd

    def _docker_exec_cmd(self, script: str, *, interactive: bool = True) -> list[str]:
        if not self.config.sim_docker_container:
            raise RuntimeError("sim_docker_container is not configured")
        cmd = self._docker_base_cmd() + ["exec"]
        if interactive:
            cmd.append("-i")
        cmd.extend([self.config.sim_docker_container, "bash", "-lc", script])
        return cmd

    def _runtime_bash(self, body: str) -> str:
        return "\n".join(
            [
                "source /ws_aic/install/setup.bash",
                self._runtime_env_exports(),
                f"cd {shlex_quote(str(self.config.workspace_container))}",
                body,
            ]
        )

    def simulation_command(self) -> list[str]:
        if self.config.sim_docker_container:
            args = [
                f"ground_truth:={_bool_arg(self.config.ground_truth)}",
                "start_aic_engine:=false",
                f"gazebo_gui:={_bool_arg(self.config.gazebo_gui)}",
                f"launch_rviz:={_bool_arg(self.config.launch_rviz)}",
            ]
            return self._docker_exec_cmd(
                self._runtime_bash("/entrypoint.sh " + " ".join(shlex_quote(arg) for arg in args))
            )
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
        if self.config.sim_docker_container:
            return self._docker_exec_cmd(
                self._runtime_bash(
                    "pixi run ros2 run aic_model aic_model --ros-args "
                    "-p use_sim_time:=true "
                    "-p policy:=gazebo_rl.bridge_policy.GazeboRLBridgePolicy "
                    "-r __node:=aic_model_gazebo_rl"
                )
            )
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
        dataset_root = self._container_path(record_root) if self.config.sim_docker_container else str(record_root)
        cmd = [
            "pixi",
            "run",
            "aic-policy-recorder",
            f"--dataset.repo_id={self.config.record_repo_id}",
            f"--dataset.single_task={self.config.record_single_task}",
            f"--dataset.root={dataset_root}",
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
        if self.config.sim_docker_container:
            return self._docker_exec_cmd(self._runtime_bash(" ".join(shlex_quote(part) for part in cmd)))
        return cmd

    def engine_command(self) -> list[str] | None:
        if not self.config.sim_docker_container:
            return None
        config = self._container_path(
            self.config.engine_config
            or str(self.config.workspace_dir / "aic_engine" / "config" / "sample_config.yaml")
        )
        body = (
            "ros2 run aic_engine aic_engine --ros-args "
            f"-p use_sim_time:=true "
            f"-p config_file_path:={shlex_quote(str(config))} "
            "-p model_node_name:=aic_model_gazebo_rl "
            "-p model_discovery_timeout_seconds:=60 "
            "-p model_configure_timeout_seconds:=90"
        )
        return self._docker_exec_cmd(self._runtime_bash(body))

    def start(self) -> None:
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        self._validate_runtime_container()
        if self.config.clean_stale_zenoh:
            self._cleanup_stale_zenoh_router()
        env = self._env()
        sim = subprocess.Popen(
            self.simulation_command(),
            cwd=self.config.workspace_dir,
            env=env,
            text=True,
        )
        self.processes.append(ManagedProcess("simulation", sim))
        time.sleep(8.0 if self.config.sim_docker_container else 2.0)
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
        engine_cmd = self.engine_command()
        if engine_cmd is not None:
            time.sleep(4.0)
            engine = subprocess.Popen(
                engine_cmd,
                cwd=self.config.workspace_dir,
                env=env,
                text=True,
            )
            self.processes.append(ManagedProcess("engine", engine))

    def check_processes(self) -> None:
        for managed in self.processes:
            code = managed.process.poll()
            if code is not None:
                raise RuntimeError(f"{managed.name} process exited early with code {code}")

    def _validate_runtime_container(self) -> None:
        if self.config.sim_distrobox and self.config.sim_docker_container:
            raise RuntimeError("--sim-distrobox and --sim-docker-container are mutually exclusive")
        self._validate_distrobox()
        self._validate_docker_container()

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

    def _validate_docker_container(self) -> None:
        if not self.config.sim_docker_container:
            return
        if shutil.which("docker") is None:
            raise RuntimeError(
                "docker is not installed or is not on PATH, but "
                f"--sim-docker-container {self.config.sim_docker_container!r} was provided."
            )
        try:
            result = subprocess.run(
                self._docker_base_cmd() + ["ps", "--format", "{{.Names}}"],
                check=True,
                capture_output=True,
                text=True,
                env=self._env(),
            )
        except OSError as ex:
            raise RuntimeError(f"Could not list docker containers: {ex}") from ex
        except subprocess.CalledProcessError as ex:
            detail = ex.stderr.strip() or ex.stdout.strip() or str(ex)
            raise RuntimeError(f"Could not list docker containers: {detail}") from ex

        names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
        if self.config.sim_docker_container not in names:
            raise RuntimeError(
                f"Docker container '{self.config.sim_docker_container}' was not found. "
                "For rootless Docker, run from the initialized shell or pass --docker-host."
            )

    def _cleanup_stale_zenoh_router(self) -> None:
        patterns = ["rmw_zenohd", "rmw_zenoh_cpp rmw_zenohd"]
        for pattern in patterns:
            try:
                subprocess.run(
                    ["pkill", "-f", pattern],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=self._env(),
                    timeout=5.0,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass

        if self.config.sim_distrobox:
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
        if self.config.sim_docker_container:
            for pattern in patterns:
                try:
                    subprocess.run(
                        self._docker_exec_cmd(f"pkill -f {shlex_quote(pattern)}", interactive=True),
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=self._env(),
                        timeout=5.0,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
        time.sleep(2.0)

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
        patterns = [
            "/entrypoint.sh",
            "aic_gz_bringup",
            "rmw_zenohd",
            "/aic_engine/aic_engine",
            "aic_model",
            "aic_model_gazebo_rl",
            "aic_adapter",
            "robot_state_publisher",
            "component_container",
            "topic_tools",
            "static_transform_publisher",
            "controller_manager/spawner",
        ]
        if self.config.sim_distrobox:
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
        if self.config.sim_docker_container:
            for signal_name in ("-INT", "-TERM"):
                for pattern in patterns:
                    try:
                        subprocess.run(
                            self._docker_exec_cmd(
                                f"pkill {signal_name} -f {shlex_quote(pattern)}",
                                interactive=True,
                            ),
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


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(str(value))
