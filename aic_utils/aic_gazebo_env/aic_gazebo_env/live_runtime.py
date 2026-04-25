"""Live runtime orchestration and e2e helpers for the training-only Gazebo path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

from .discovery import (
    SetupScriptDiscovery,
    discover_setup_script,
    find_repo_root,
    resolve_gz_executable,
    resolve_transport_helper_executable,
)


DEFAULT_WORLD_NAME = "aic_world"
DEFAULT_SOURCE_ENTITY = "ati/tool_link"
DEFAULT_TARGET_ENTITY = "tabletop"
DEFAULT_WORLD_PATH = "/tmp/aic.sdf"
DEFAULT_CONTAINER_NAME = "aic_eval"
DEFAULT_CONTAINER_IMAGE = "ghcr.io/intrinsic-dev/aic/aic_eval:latest"
DEFAULT_ENTRYPOINT_LOG = "/tmp/aic_gazebo_env_entrypoint.log"


@dataclass(frozen=True)
class LiveCommandResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class LiveEnvironmentContext:
    repo_root: str
    mode: str
    setup_script: str | None = None
    workspace_root: str | None = None
    container_name: str | None = None
    container_image: str | None = None
    launch_command: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_root": self.repo_root,
            "mode": self.mode,
            "setup_script": self.setup_script,
            "workspace_root": self.workspace_root,
            "container_name": self.container_name,
            "container_image": self.container_image,
            "launch_command": self.launch_command,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class LiveHealthReport:
    gz_reachable: bool = False
    helper_reachable: bool = False
    world_control_reachable: bool = False
    state_topic_live: bool = False
    first_observation_ok: bool = False
    reset_ok: bool = False
    no_op_step_ok: bool = False
    action_step_ok: bool = False
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_world_file(repo_root: Path | None = None) -> str:
    repo = find_repo_root(repo_root or Path(__file__))
    candidate = repo / "aic_description" / "world" / "aic.sdf"
    if candidate.exists():
        return str(candidate)
    return DEFAULT_WORLD_PATH


def live_prerequisites(repo_root: Path | None = None) -> dict[str, Any]:
    repo = find_repo_root(repo_root or Path(__file__))
    setup = discover_setup_script(repo_root=repo)
    gz = resolve_gz_executable("gz", repo_root=repo)
    helper = resolve_transport_helper_executable(None, repo_root=repo)
    return {
        "repo_root": str(repo),
        "setup_script": setup.script_path,
        "setup_explanation": setup.explanation,
        "searched_setup_locations": list(setup.searched_locations),
        "gz_resolved_path": gz.resolved_path,
        "gz_status": gz.status,
        "gz_setup_status": gz.setup_status,
        "searched_gz_locations": list(gz.searched_locations),
        "helper_resolved_path": helper.resolved_path,
        "helper_status": helper.status,
        "helper_setup_status": helper.setup_status,
        "searched_helper_locations": list(helper.searched_locations),
        "distrobox_path": shutil.which("distrobox"),
        "docker_path": shutil.which("docker"),
        "colcon_path": shutil.which("colcon"),
        "ros_setup_exists": Path("/opt/ros/kilted/setup.bash").exists(),
    }


class LiveRuntimeManager:
    """Discover, prepare, and exercise a live runtime shell context."""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        world_name: str = DEFAULT_WORLD_NAME,
        world_path: str | None = None,
        container_name: str = DEFAULT_CONTAINER_NAME,
        container_image: str = DEFAULT_CONTAINER_IMAGE,
    ) -> None:
        self.repo_root = find_repo_root(repo_root or Path(__file__))
        self.world_name = world_name
        self.world_path = world_path or default_world_file(self.repo_root)
        self.container_name = container_name
        self.container_image = container_image

    def preflight(self) -> dict[str, Any]:
        report = live_prerequisites(self.repo_root)
        setup_script = report.get("setup_script")
        recommendation = None
        if isinstance(setup_script, str) and (
            report.get("gz_resolved_path") is None or report.get("helper_resolved_path") is None
        ):
            recommendation = self.source_recommendation(setup_script)
        report["world_name"] = self.world_name
        report["world_path"] = self.world_path
        report["repo_pythonpath"] = str(self.repo_root / "aic_utils" / "aic_gazebo_env")
        report["recommendation"] = recommendation
        report["canonical_e2e_command"] = (
            f"python3 aic_utils/aic_gazebo_env/scripts/run_live_e2e.py "
            f"--auto-build --auto-launch"
        )
        return report

    def source_recommendation(self, setup_script: str, *, script_relative: str = "aic_utils/aic_gazebo_env/scripts/run_live_e2e.py") -> str:
        command = (
            f"cd {shlex.quote(str(self.repo_root))} && "
            f"source {shlex.quote(setup_script)} && "
            f"python3 {shlex.quote(script_relative)}"
        )
        return f"bash -lc {shlex.quote(command)}"

    def discover_context(self) -> LiveEnvironmentContext:
        preflight = self.preflight()
        setup_script = preflight.get("setup_script")
        if preflight.get("gz_resolved_path") and preflight.get("helper_resolved_path"):
            return LiveEnvironmentContext(
                repo_root=str(self.repo_root),
                mode="local_direct",
                setup_script=setup_script,
                workspace_root=_workspace_root_from_setup(setup_script),
                diagnostics=preflight,
            )
        if self._container_exists():
            return LiveEnvironmentContext(
                repo_root=str(self.repo_root),
                mode="distrobox_attach",
                setup_script="/ws_aic/install/setup.bash",
                workspace_root="/ws_aic",
                container_name=self.container_name,
                container_image=self.container_image,
                diagnostics=preflight,
            )
        docker_container = self._docker_container_name()
        if docker_container is not None:
            return LiveEnvironmentContext(
                repo_root=str(self.repo_root),
                mode="docker_exec",
                setup_script="/ws_aic/install/setup.bash",
                workspace_root="/ws_aic",
                container_name=docker_container,
                container_image=self.container_image,
                diagnostics=preflight,
            )
        if isinstance(setup_script, str):
            return LiveEnvironmentContext(
                repo_root=str(self.repo_root),
                mode="local_sourced",
                setup_script=setup_script,
                workspace_root=_workspace_root_from_setup(setup_script),
                diagnostics=preflight,
            )
        return LiveEnvironmentContext(
            repo_root=str(self.repo_root),
            mode="unavailable",
            container_name=self.container_name,
            container_image=self.container_image,
            diagnostics=preflight,
        )

    def prepare(
        self,
        *,
        auto_build: bool = False,
        auto_launch: bool = False,
    ) -> LiveEnvironmentContext:
        context = self.discover_context()
        if context.mode == "local_sourced" and auto_build:
            self._maybe_build_helper_locally(context)
            context = self.discover_context()
        if context.mode == "unavailable" and auto_launch:
            self._maybe_create_container()
            context = self.discover_context()
        if context.mode == "distrobox_attach" and auto_launch:
            if not self._world_ready(context):
                launch_command = (
                    "/entrypoint.sh ground_truth:=false start_aic_engine:=true "
                    "gazebo_gui:=false launch_rviz:=false"
                )
                self._launch_background(context, launch_command, timeout_s=15.0)
                context.launch_command = launch_command
        elif context.mode == "docker_exec" and auto_launch:
            if not self._world_ready(context):
                launch_command = (
                    "/entrypoint.sh ground_truth:=false start_aic_engine:=true "
                    "gazebo_gui:=false launch_rviz:=false"
                )
                self._launch_background(context, launch_command, timeout_s=15.0)
                context.launch_command = launch_command
        elif context.mode == "local_sourced" and auto_launch:
            if not self._world_ready(context):
                launch_command = (
                    "ros2 launch aic_bringup aic_gz_bringup.launch.py "
                    "ground_truth:=false start_aic_engine:=true "
                    "gazebo_gui:=false launch_rviz:=false"
                )
                self._launch_background(context, launch_command, timeout_s=15.0)
                context.launch_command = launch_command
        return context

    def wait_for_health(
        self,
        context: LiveEnvironmentContext,
        *,
        timeout_s: float = 120.0,
    ) -> LiveHealthReport:
        report = LiveHealthReport()
        deadline = time.monotonic() + timeout_s
        report.diagnostics["context"] = context.to_dict()
        if context.mode == "unavailable":
            report.diagnostics["last_error"] = (
                "live environment is unavailable; no local sourced workspace or attachable distrobox container was found"
            )
            return report
        while time.monotonic() < deadline:
            try:
                result = self._run_worker(context, "health", timeout_s=45.0)
                payload = json.loads(result.stdout)
                return LiveHealthReport(
                    gz_reachable=bool(payload.get("gz_reachable")),
                    helper_reachable=bool(payload.get("helper_reachable")),
                    world_control_reachable=bool(payload.get("world_control_reachable")),
                    state_topic_live=bool(payload.get("state_topic_live")),
                    first_observation_ok=bool(payload.get("first_observation_ok")),
                    reset_ok=bool(payload.get("reset_ok")),
                    no_op_step_ok=bool(payload.get("no_op_step_ok")),
                    action_step_ok=bool(payload.get("action_step_ok")),
                    stage_timings_ms=dict(payload.get("stage_timings_ms") or {}),
                    diagnostics=payload,
                )
            except Exception as exc:
                report.diagnostics["last_error"] = str(exc)
                time.sleep(1.0)
        return report

    def run_context_command(
        self,
        context: LiveEnvironmentContext,
        command: str,
        *,
        timeout_s: float,
    ) -> LiveCommandResult:
        return self._run_context_shell(context, command, timeout_s=timeout_s)

    def run_e2e(
        self,
        context: LiveEnvironmentContext,
        *,
        timeout_s: float = 120.0,
    ) -> dict[str, Any]:
        health = self.wait_for_health(context, timeout_s=timeout_s)
        if not health.no_op_step_ok or not health.action_step_ok:
            return {
                "health": health.to_dict(),
                "smoke": None,
                "parity": None,
                "ok": False,
            }
        smoke = json.loads(self._run_worker(context, "smoke", timeout_s=60.0).stdout)
        parity = json.loads(self._run_worker(context, "parity", timeout_s=60.0).stdout)
        return {
            "health": health.to_dict(),
            "smoke": smoke,
            "parity": parity,
            "ok": bool(smoke.get("ok")) and bool(parity.get("ok")),
        }

    def _container_exists(self) -> bool:
        distrobox = shutil.which("distrobox")
        if distrobox is None:
            return False
        completed = subprocess.run(
            [distrobox, "list", "--no-color"],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return False
        return any(
            line.split("|", 1)[0].strip() == self.container_name
            or line.strip().startswith(f"{self.container_name} ")
            for line in completed.stdout.splitlines()
        )

    def _docker_container_name(self) -> str | None:
        docker = shutil.which("docker")
        if docker is None:
            return None
        completed = subprocess.run(
            [
                docker,
                "ps",
                "--format",
                "{{.Names}}|{{.Image}}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return None
        fallback: str | None = None
        for line in completed.stdout.splitlines():
            if "|" not in line:
                continue
            name, image = (part.strip() for part in line.split("|", 1))
            if name == self.container_name:
                return name
            if image == self.container_image or image.startswith(f"{self.container_image}@"):
                fallback = fallback or name
        return fallback

    def _maybe_create_container(self) -> None:
        distrobox = shutil.which("distrobox")
        if distrobox is None:
            raise RuntimeError("distrobox is not available for container launch")
        if self._container_exists():
            return
        env = dict(os.environ)
        env["DBX_CONTAINER_MANAGER"] = "docker"
        docker = shutil.which("docker")
        if docker is not None:
            pull = subprocess.run(
                [docker, "pull", self.container_image],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if pull.returncode != 0:
                raise RuntimeError(
                    "Failed to pull aic_eval image.\n"
                    f"stdout:\n{pull.stdout}\nstderr:\n{pull.stderr}"
                )
        command = [
            distrobox,
            "create",
            "-r",
            "-i",
            self.container_image,
            self.container_name,
        ]
        if shutil.which("nvidia-smi") is not None:
            command.insert(3, "--nvidia")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Failed to create distrobox container.\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )

    def _maybe_build_helper_locally(self, context: LiveEnvironmentContext) -> None:
        if context.setup_script is None:
            return
        if resolve_transport_helper_executable(None, repo_root=self.repo_root).resolved_path:
            return
        ros_setup = "/opt/ros/kilted/setup.bash"
        source_prefix = (
            f"source {shlex.quote(ros_setup)} && "
            if Path(ros_setup).exists()
            else f"source {shlex.quote(context.setup_script)} && "
        )
        workspace_root = context.workspace_root or str(self.repo_root)
        build_command = (
            f"{source_prefix}cd {shlex.quote(workspace_root)} && "
            "colcon build --packages-select aic_gazebo_transport_bridge --executor sequential"
        )
        result = self._run_context_shell(context, build_command, timeout_s=600.0)
        if result.returncode != 0:
            raise RuntimeError(
                "Targeted helper build failed.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    def _world_ready(self, context: LiveEnvironmentContext) -> bool:
        try:
            result = self._run_context_shell(
                context,
                "gz service -l",
                timeout_s=10.0,
            )
        except Exception:
            return False
        if result.returncode != 0:
            return False
        return f"/world/{self.world_name}/control" in result.stdout

    def _launch_background(
        self,
        context: LiveEnvironmentContext,
        command: str,
        *,
        timeout_s: float,
    ) -> LiveCommandResult:
        log_path = DEFAULT_ENTRYPOINT_LOG
        wrapped = f"nohup {command} > {shlex.quote(log_path)} 2>&1 & echo $!"
        return self._run_context_shell(context, wrapped, timeout_s=timeout_s)

    def _run_worker(
        self,
        context: LiveEnvironmentContext,
        worker_mode: str,
        *,
        timeout_s: float,
    ) -> LiveCommandResult:
        script_path = (
            self.repo_root
            / "aic_utils"
            / "aic_gazebo_env"
            / "scripts"
            / "run_live_e2e.py"
        )
        command = (
            f"PYTHONPATH={shlex.quote(str(self.repo_root / 'aic_utils' / 'aic_gazebo_env'))} "
            f"{shlex.quote(sys.executable)} {shlex.quote(str(script_path))} "
            f"--worker {shlex.quote(worker_mode)} "
            f"--world-name {shlex.quote(self.world_name)} "
            f"--world-path {shlex.quote(self.world_path)} "
            f"--json-only"
        )
        return self._run_context_shell(context, command, timeout_s=timeout_s)

    def _run_context_shell(
        self,
        context: LiveEnvironmentContext,
        command: str,
        *,
        timeout_s: float,
    ) -> LiveCommandResult:
        repo_root = shlex.quote(str(self.repo_root))
        if context.mode == "local_direct":
            shell_command = f"cd {repo_root} && {command}"
            process = subprocess.run(
                ["bash", "-lc", shell_command],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        elif context.mode == "local_sourced":
            if context.setup_script is None:
                raise RuntimeError("local_sourced context is missing setup_script")
            shell_command = (
                f"source {shlex.quote(context.setup_script)} && cd {repo_root} && {command}"
            )
            process = subprocess.run(
                ["bash", "-lc", shell_command],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        elif context.mode == "distrobox_attach":
            if context.container_name is None:
                raise RuntimeError("distrobox_attach context is missing container_name")
            inner = (
                f"source /ws_aic/install/setup.bash && cd {repo_root} && {command}"
            )
            process = subprocess.run(
                [
                    "bash",
                    "-lc",
                    (
                        f"DBX_CONTAINER_MANAGER=docker distrobox enter -r "
                        f"{shlex.quote(context.container_name)} -- bash -lc {shlex.quote(inner)}"
                    ),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        elif context.mode == "docker_exec":
            if context.container_name is None:
                raise RuntimeError("docker_exec context is missing container_name")
            inner = (
                f"source /ws_aic/install/setup.bash && cd {repo_root} && {command}"
            )
            process = subprocess.run(
                [
                    "docker",
                    "exec",
                    context.container_name,
                    "bash",
                    "-lc",
                    inner,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        else:
            raise RuntimeError(f"Unsupported live context mode: {context.mode}")
        return LiveCommandResult(
            command=command,
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
        )


def perform_live_health_check(*, world_name: str, world_path: str) -> dict[str, Any]:
    """Direct in-env health check once the shell is prepared."""
    report: dict[str, Any] = {
        "gz_reachable": False,
        "helper_reachable": False,
        "world_control_reachable": False,
        "state_topic_live": False,
        "first_observation_ok": False,
        "reset_ok": False,
        "no_op_step_ok": False,
        "action_step_ok": False,
        "stage_timings_ms": {},
    }
    stage_starts: dict[str, float] = {}

    def _start_stage(name: str) -> None:
        stage_starts[name] = time.perf_counter()

    def _finish_stage(name: str) -> None:
        start = stage_starts.get(name)
        if start is not None:
            report["stage_timings_ms"][name] = round((time.perf_counter() - start) * 1000.0, 3)

    _start_stage("gz_reachable")
    gz = resolve_gz_executable("gz")
    report["gz_reachable"] = gz.resolved_path is not None
    _finish_stage("gz_reachable")
    _start_stage("helper_reachable")
    helper = resolve_transport_helper_executable(None)
    report["helper_reachable"] = helper.resolved_path is not None
    _finish_stage("helper_reachable")
    if not report["gz_reachable"] or not report["helper_reachable"]:
        return report

    import subprocess as _subprocess

    _start_stage("world_control_reachable")
    services = _subprocess.run(
        ["gz", "service", "-l"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    if services.returncode != 0:
        report["services_stderr"] = services.stderr
        _finish_stage("world_control_reachable")
        return report
    report["world_control_reachable"] = f"/world/{world_name}/control" in services.stdout
    report["service_list"] = services.stdout
    _finish_stage("world_control_reachable")
    if not report["world_control_reachable"]:
        return report

    from .gazebo_client import GazeboCliClientConfig, GazeboTransportClient
    from .protocol import GetObservationRequest, ResetRequest, StepRequest

    client = GazeboTransportClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path=world_path,
            timeout=10.0,
            world_name=world_name,
            source_entity_name=DEFAULT_SOURCE_ENTITY,
            target_entity_name=DEFAULT_TARGET_ENTITY,
            transport_backend="transport",
        )
    )
    try:
        _start_stage("first_observation")
        observation = client.get_observation(GetObservationRequest())
        report["state_topic_live"] = True
        report["first_observation_ok"] = True
        report["first_observation_info"] = observation.info
        _finish_stage("first_observation")
        _start_stage("reset")
        reset_response = client.reset(ResetRequest(seed=0, options={"mode": "health"}))
        report["reset_ok"] = True
        report["reset_info"] = reset_response.info
        _finish_stage("reset")
        _start_stage("no_op_step")
        step_response = client.step(StepRequest(action={"multi_step": 1}))
        report["no_op_step_ok"] = True
        report["no_op_step_info"] = step_response.info
        _finish_stage("no_op_step")
        _start_stage("action_step")
        action_response = client.step(
            StepRequest(action={"position_delta": [0.001, 0.0, 0.0], "multi_step": 1})
        )
        report["action_step_ok"] = True
        report["action_step_info"] = action_response.info
        _finish_stage("action_step")
    finally:
        client.close()
    return report


def perform_live_smoke_sequence(*, world_name: str, world_path: str) -> dict[str, Any]:
    from .gazebo_client import GazeboCliClientConfig, GazeboTransportClient
    from .protocol import GetObservationRequest, ResetRequest, StepRequest

    client = GazeboTransportClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path=world_path,
            timeout=10.0,
            world_name=world_name,
            source_entity_name=DEFAULT_SOURCE_ENTITY,
            target_entity_name=DEFAULT_TARGET_ENTITY,
            transport_backend="transport",
        )
    )
    try:
        observation = client.get_observation(GetObservationRequest())
        reset_response = client.reset(ResetRequest(seed=7, options={"mode": "live-e2e"}))
        no_op = client.step(StepRequest(action={"multi_step": 1}))
        pose_step = client.step(
            StepRequest(
                action={
                    "position_delta": [0.002, 0.0, 0.0],
                    "multi_step": 1,
                }
            )
        )
        joint_step = client.step(
            StepRequest(
                action={
                    "joint_position_delta": [0.01, -0.005, 0.005, 0.0, 0.0, 0.0],
                    "multi_step": 1,
                }
            )
        )
    finally:
        client.close()

    return {
        "ok": True,
        "observation": _summarize_observation(observation.observation),
        "reset": {
            "info": reset_response.info,
            "observation": _summarize_observation(reset_response.observation),
        },
        "no_op_step": {
            "reward": no_op.reward,
            "terminated": no_op.terminated,
            "truncated": no_op.truncated,
            "info": no_op.info,
            "observation": _summarize_observation(no_op.observation),
        },
        "pose_step": {
            "reward": pose_step.reward,
            "terminated": pose_step.terminated,
            "truncated": pose_step.truncated,
            "info": pose_step.info,
            "observation": _summarize_observation(pose_step.observation),
        },
        "joint_step": {
            "reward": joint_step.reward,
            "terminated": joint_step.terminated,
            "truncated": joint_step.truncated,
            "info": joint_step.info,
            "observation": _summarize_observation(joint_step.observation),
        },
    }


def perform_live_parity_sequence(*, world_name: str, world_path: str) -> dict[str, Any]:
    from .gazebo_client import GazeboCliClientConfig, GazeboTransportClient
    from .official_scoring import OfficialTier3TrackedPairScorer
    from .protocol import ResetRequest, StepRequest

    client = GazeboTransportClient(
        GazeboCliClientConfig(
            executable="gz",
            world_path=world_path,
            timeout=10.0,
            world_name=world_name,
            source_entity_name=DEFAULT_SOURCE_ENTITY,
            target_entity_name=DEFAULT_TARGET_ENTITY,
            transport_backend="transport",
        )
    )
    scorer = OfficialTier3TrackedPairScorer()
    sequence = [
        {"multi_step": 1},
        {"multi_step": 1},
        {"joint_position_delta": [0.005, -0.0025, 0.0025, 0.0, 0.0, 0.0], "multi_step": 1},
    ]
    try:
        reset_response = client.reset(ResetRequest(seed=11, options={"mode": "parity"}))
        steps = [client.step(StepRequest(action=action)) for action in sequence]
    finally:
        client.close()

    observations = [reset_response.observation, *(response.observation for response in steps)]
    tracked = [obs.get("task_geometry", {}).get("tracked_entity_pair", {}) for obs in observations]
    step_counts = [obs.get("step_count") for obs in observations]
    raw_counts = [response.info.get("sim_step_count_raw") for response in [reset_response, *steps]]
    relative_positions = [pair.get("relative_position") for pair in tracked]
    distances = [pair.get("distance") for pair in tracked]
    success_flags = [pair.get("success") for pair in tracked]
    initial_distance = distances[0] if isinstance(distances[0], float) else None
    official_scores: list[dict[str, Any]] = []
    for pair in tracked:
        if isinstance(pair, dict):
            score_value, score_details = scorer.score(
                tracked_pair=pair,
                initial_distance=initial_distance,
            )
            official_scores.append(
                {
                    "score": score_value,
                    "details": score_details,
                }
            )
        else:
            official_scores.append({"score": 0.0, "details": {"reason": "missing_pair"}})

    parity_checks = {
        "world_name_matches": all(obs.get("world_name") == world_name for obs in observations),
        "entity_count_sane": all(isinstance(obs.get("entity_count"), int) and obs["entity_count"] > 0 for obs in observations),
        "tracked_pair_present": all(isinstance(pair, dict) and pair for pair in tracked),
        "relative_position_present": all(isinstance(value, list) and len(value) == 3 for value in relative_positions),
        "distance_present": all(isinstance(value, float) for value in distances),
        "success_flag_present": all(isinstance(value, bool) for value in success_flags),
        "step_count_monotonic": _is_monotonic(step_counts),
        "sim_step_count_raw_monotonic": _is_monotonic(raw_counts),
        "repeated_no_op_stable": _no_op_stable(relative_positions[:3], tolerance=1e-4),
        "distance_changes_sensibly": len({round(value, 6) for value in distances if isinstance(value, float)}) >= 2,
        "official_score_slice_present": all(
            isinstance(item.get("score"), float) and isinstance(item.get("details"), dict)
            for item in official_scores
        ),
    }
    return {
        "ok": all(parity_checks.values()),
        "checks": parity_checks,
        "step_counts": step_counts,
        "sim_step_count_raw": raw_counts,
        "distances": distances,
        "relative_positions": relative_positions,
        "official_scores": official_scores,
    }


def _summarize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    tracked = observation.get("task_geometry", {}).get("tracked_entity_pair", {})
    return {
        "world_name": observation.get("world_name"),
        "step_count": observation.get("step_count"),
        "entity_count": observation.get("entity_count"),
        "joint_count": observation.get("joint_count"),
        "tracked_pair_present": bool(tracked),
        "tracked_relative_position": tracked.get("relative_position"),
        "tracked_distance": tracked.get("distance"),
        "tracked_success": tracked.get("success"),
    }


def _is_monotonic(values: list[Any]) -> bool:
    filtered = [value for value in values if isinstance(value, int)]
    return all(left <= right for left, right in zip(filtered, filtered[1:]))


def _no_op_stable(relative_positions: list[Any], *, tolerance: float) -> bool:
    if len(relative_positions) < 2:
        return False
    base = relative_positions[0]
    if not isinstance(base, list) or len(base) != 3:
        return False
    for candidate in relative_positions[1:]:
        if not isinstance(candidate, list) or len(candidate) != 3:
            return False
        if any(abs(a - b) > tolerance for a, b in zip(base, candidate)):
            return False
    return True


def _workspace_root_from_setup(setup_script: str | None) -> str | None:
    if setup_script is None:
        return None
    return str(Path(setup_script).resolve().parents[1])
