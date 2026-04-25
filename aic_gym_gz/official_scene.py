"""Helpers for launching official Gazebo scenes from sampled scenarios."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import tempfile
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from .scenario import AicScenario, RailEntity


@dataclass(frozen=True)
class OfficialSceneLaunchSpec:
    scenario: AicScenario
    shell_command: str
    ros_launch_args: tuple[str, ...]
    expected_entities: tuple[str, ...]
    shell_environment: dict[str, str]
    ros_command_prefix: str
    launch_mode: str
    router_command_prefix: str | None


def _load_overview_rig_models() -> str:
    world_sdf = (Path(__file__).resolve().parent.parent / "aic_description" / "world" / "aic.sdf").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'(\s*<model name="overview_camera_rig">.*?<model name="overview_oblique_camera_rig">.*?</model>\s*)\n\s*<!-- Cable is now spawned via cable\.sdf\.xacro in launch file -->',
        world_sdf,
        re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not locate overview camera rig definitions in aic.sdf")
    return match.group(1)


_OVERVIEW_RIG_MODELS = _load_overview_rig_models()


def resolve_official_setup_script() -> Path:
    candidates = [
        Path.cwd() / "install" / "setup.bash",
        Path.cwd().parent / "install" / "setup.bash",
        Path.cwd().parent.parent / "install" / "setup.bash",
        Path("/ws_aic/install/setup.bash"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the official ROS setup script. "
        f"checked={[str(candidate) for candidate in candidates]}"
    )


def export_training_world_for_scenario(
    scenario: AicScenario,
    *,
    output_path: str | Path,
    setup_script: str | Path | None = None,
    timeout_s: float = 120.0,
) -> dict[str, object]:
    _cleanup_stale_training_world_processes()
    setup_path = Path(setup_script) if setup_script is not None else resolve_official_setup_script()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    launch_spec = build_official_launch_spec(
        scenario,
        setup_script=setup_path,
        ground_truth=False,
        start_aic_engine=False,
        gazebo_gui=False,
        launch_rviz=False,
    )
    launch_log_path = output.with_suffix(".launch.log")
    launch_handle = launch_log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-lc", launch_spec.shell_command],
        stdout=launch_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_world_entities(
            ros_command_prefix=launch_spec.ros_command_prefix,
            world_name="aic_world",
            required_entities=("task_board", "cable_0"),
            timeout_s=timeout_s,
            launch_log_path=launch_log_path,
        )
        world_sdf = _generate_world_sdf(
            ros_command_prefix=launch_spec.ros_command_prefix,
            world_name="aic_world",
            timeout_s=timeout_s,
        )
        sanitized = sanitize_training_world_sdf(
            world_sdf,
            overview_models=_generated_overview_rig_models(setup_script=setup_path),
            scene_probe_model=_generated_scene_probe_model(setup_script=setup_path),
        )
        output.write_text(sanitized, encoding="utf-8")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15.0)
        launch_handle.close()
    return {
        "output_path": str(output),
        "launch_log_path": str(launch_log_path),
        "launch_mode": launch_spec.launch_mode,
        "ros_launch_args": list(launch_spec.ros_launch_args),
    }


def _cleanup_stale_training_world_processes() -> None:
    cleanup_script = r"""
python3 - <<'PY'
import os
import signal
import subprocess

patterns = [
    "/entrypoint.sh",
    "ros2 launch aic_bringup aic_gz_bringup.launch.py",
    "rmw_zenohd",
    "gz sim",
    "gz_server",
    "aic_adapter",
    "robot_state_publisher",
    "component_container",
    "controller_manager/spawner",
]
current = os.getpid()
ppid = os.getppid()
output = subprocess.run(
    ["ps", "-eo", "pid,args"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
for line in output[1:]:
    try:
        pid_text, args = line.strip().split(None, 1)
    except ValueError:
        continue
    pid = int(pid_text)
    if pid in {current, ppid}:
        continue
    if any(pattern in args for pattern in patterns):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
PY
rm -f /dev/shm/sem.fastdds* /dev/shm/fastdds* /dev/shm/fastrtps* 2>/dev/null || true
sleep 2
"""
    subprocess.run(
        ["bash", "-lc", cleanup_script],
        check=False,
        capture_output=True,
        text=True,
        timeout=20.0,
    )


def sanitize_training_world_sdf(
    world_sdf: str,
    *,
    overview_models: str | None = None,
    scene_probe_model: str | None = None,
) -> str:
    plugin_pattern = re.compile(
        r"\s*<plugin[^>]*WorldSdfGeneratorPlugin[^>]*>.*?</plugin>",
        re.DOTALL,
    )
    sanitized = re.sub(plugin_pattern, "", world_sdf)
    overview_pattern = re.compile(
        r"\s*<model name=['\"]overview_camera_rig['\"].*?<model name=['\"]overview_oblique_camera_rig['\"].*?</model>\s*",
        re.DOTALL,
    )
    sanitized = re.sub(overview_pattern, "\n", sanitized)
    selected_scene_probe_model = scene_probe_model or _generated_scene_probe_model(setup_script=resolve_official_setup_script())
    if "scene_probe_camera" not in sanitized:
        closing = "</world>"
        idx = sanitized.rfind(closing)
        if idx != -1:
            sanitized = sanitized[:idx] + selected_scene_probe_model + "\n" + sanitized[idx:]
    return sanitized


def _generated_overview_rig_models(*, setup_script: str | Path) -> str:
    xacro_path = Path(__file__).resolve().parent.parent / "aic_description" / "urdf" / "overview_camera_array.urdf.xacro"
    try:
        import xacro  # type: ignore
    except Exception:
        return _OVERVIEW_RIG_MODELS
    try:
        document = xacro.process_file(str(xacro_path))
        urdf_text = document.toprettyxml()
    except Exception:
        return _OVERVIEW_RIG_MODELS
    with tempfile.TemporaryDirectory(prefix="aic_overview_xacro_") as temp_dir:
        urdf_path = Path(temp_dir) / "overview_camera_array.urdf"
        urdf_path.write_text(urdf_text, encoding="utf-8")
        completed = subprocess.run(
            [
                "bash",
                "-lc",
                (
                    f"source {shlex.quote(str(setup_script))} && "
                    f"gz sdf --print {shlex.quote(str(urdf_path))}"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
    if completed.returncode != 0:
        return _OVERVIEW_RIG_MODELS
    match = re.search(
        r"(<model name=['\"]overview_camera_array['\"].*?</model>)",
        completed.stdout,
        re.DOTALL,
    )
    if match is None:
        return _OVERVIEW_RIG_MODELS
    return "\n" + match.group(1) + "\n"


def _generated_scene_probe_model(*, setup_script: str | Path) -> str:
    del setup_script
    return """
<model name="scene_probe_camera">
  <pose>0.95 -0.10 1.30 0 0 3.14</pose>
  <static>false</static>
  <link name="camera_link">
    <gravity>false</gravity>
    <inertial>
      <pose>0 0 0 0 0 0</pose>
      <mass>0.01</mass>
      <inertia>
        <ixx>0.00001</ixx>
        <ixy>0</ixy>
        <ixz>0</ixz>
        <iyy>0.00001</iyy>
        <iyz>0</iyz>
        <izz>0.00001</izz>
      </inertia>
    </inertial>
    <visual name="body_visual">
      <geometry>
        <box>
          <size>0.03 0.03 0.03</size>
        </box>
      </geometry>
    </visual>
    <sensor name="scene_probe" type="camera">
      <always_on>true</always_on>
      <update_rate>20</update_rate>
      <topic>/scene_probe/image</topic>
      <camera>
        <camera_info_topic>/scene_probe/camera_info</camera_info_topic>
        <horizontal_fov>0.8718</horizontal_fov>
        <image>
          <width>1152</width>
          <height>1024</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.07</near>
          <far>20</far>
        </clip>
      </camera>
    </sensor>
  </link>
</model>
"""


def _wait_for_world_entities(
    *,
    ros_command_prefix: str,
    world_name: str,
    required_entities: tuple[str, ...],
    timeout_s: float,
    launch_log_path: str | Path | None = None,
) -> None:
    deadline = time.monotonic() + float(timeout_s)
    last_error = ""
    while time.monotonic() < deadline:
        if launch_log_path is not None:
            try:
                log_text = Path(launch_log_path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                log_text = ""
            if all(f"named [{name}]" in log_text for name in required_entities):
                return
        service_probe = subprocess.run(
            [
                "bash",
                "-lc",
                f"{ros_command_prefix} && gz service -l",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )
        if f"/world/{world_name}/generate_world_sdf" not in service_probe.stdout:
            last_error = service_probe.stderr or service_probe.stdout
            time.sleep(1.0)
            continue
        topic_probe = subprocess.run(
            [
                "bash",
                "-lc",
                f"{ros_command_prefix} && timeout 12s gz topic -e -n 1 -t /world/{world_name}/state",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )
        state_text = topic_probe.stdout
        if all(name in state_text for name in required_entities):
            return
        last_error = topic_probe.stderr or state_text[-500:]
        time.sleep(1.0)
    raise RuntimeError(
        "Timed out waiting for spawned training-world entities. "
        f"required_entities={required_entities}, last_error={last_error}"
    )


def _generate_world_sdf(
    *,
    ros_command_prefix: str,
    world_name: str,
    timeout_s: float,
) -> str:
    request = "global_entity_gen_config { expand_include_tags { data: true } }"
    completed = subprocess.run(
        [
            "bash",
            "-lc",
            (
                f"{ros_command_prefix} && "
                f"gz service -s /world/{world_name}/generate_world_sdf "
                "--reqtype gz.msgs.SdfGeneratorConfig "
                "--reptype gz.msgs.StringMsg "
                f"--timeout {int(max(timeout_s, 20.0) * 1000.0)} "
                f"--req {shlex.quote(request)}"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=max(timeout_s, 30.0),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "generate_world_sdf failed. "
            f"stdout={completed.stdout.strip()} stderr={completed.stderr.strip()}"
        )
    payload = completed.stdout.strip()
    if not payload.startswith("data: "):
        raise RuntimeError(f"Unexpected generate_world_sdf payload: {payload[:200]}")
    return ast.literal_eval(payload[len("data: "):])


def bringup_launch_args_for_scenario(
    scenario: AicScenario,
    *,
    ground_truth: bool,
    start_aic_engine: bool,
    gazebo_gui: bool = False,
    launch_rviz: bool = False,
) -> list[str]:
    board_x, board_y, board_z, board_roll, board_pitch, board_yaw = scenario.task_board.pose_xyz_rpy
    task = next(iter(scenario.tasks.values()))
    cable = scenario.cables[task.cable_name]
    cable_spawn_xyz = _official_cable_spawn_xyz(cable.cable_type, attach_to_gripper=cable.attach_to_gripper)
    args = [
        "ground_truth:=" + _bool_arg(ground_truth),
        "start_aic_engine:=" + _bool_arg(start_aic_engine),
        "gazebo_gui:=" + _bool_arg(gazebo_gui),
        "launch_rviz:=" + _bool_arg(launch_rviz),
        "spawn_task_board:=true",
        "spawn_cable:=true",
        f"task_board_x:={board_x}",
        f"task_board_y:={board_y}",
        f"task_board_z:={board_z}",
        f"task_board_roll:={board_roll}",
        f"task_board_pitch:={board_pitch}",
        f"task_board_yaw:={board_yaw}",
        f"cable_type:={cable.cable_type}",
        "attach_cable_to_gripper:=" + _bool_arg(cable.attach_to_gripper),
        f"cable_x:={cable_spawn_xyz[0]}",
        f"cable_y:={cable_spawn_xyz[1]}",
        f"cable_z:={cable_spawn_xyz[2]}",
        f"cable_roll:={cable.rpy[0]}",
        f"cable_pitch:={cable.rpy[1]}",
        f"cable_yaw:={cable.rpy[2]}",
    ]
    args.extend(_task_board_args(scenario))
    return args


def expected_scene_entities(scenario: AicScenario) -> tuple[str, ...]:
    task = next(iter(scenario.tasks.values()))
    cable = scenario.cables[task.cable_name]
    expected = {
        "task_board",
        "task_board_base_link",
        "tabletop",
        cable.cable_name,
        task.target_module_name,
    }
    for group in (
        scenario.task_board.nic_rails,
        scenario.task_board.sc_rails,
        scenario.task_board.mount_rails,
    ):
        for entity in group.values():
            if entity.present and entity.name:
                expected.add(entity.name)
    return tuple(sorted(expected))


def build_official_launch_spec(
    scenario: AicScenario,
    *,
    setup_script: str | Path,
    ground_truth: bool,
    start_aic_engine: bool,
    gazebo_gui: bool = False,
    launch_rviz: bool = False,
) -> OfficialSceneLaunchSpec:
    launch_args = bringup_launch_args_for_scenario(
        scenario,
        ground_truth=ground_truth,
        start_aic_engine=start_aic_engine,
        gazebo_gui=gazebo_gui,
        launch_rviz=launch_rviz,
    )
    ros_launch = " ".join(
        ["ros2 launch aic_bringup aic_gz_bringup.launch.py", *[shlex.quote(arg) for arg in launch_args]]
    )
    shell_environment = _official_scene_shell_environment(setup_script=setup_script)
    ros_command_prefix = _shell_prefix(
        setup_script=setup_script,
        shell_environment=shell_environment,
    )
    router_command_prefix = _router_shell_prefix(setup_script=setup_script)
    launch_mode = "entrypoint" if _should_use_container_entrypoint(setup_script=setup_script) else "ros2_launch"
    if launch_mode == "entrypoint":
        router_command_prefix = None
        shell_command = " && ".join(
            [
                _export_shell_environment(shell_environment),
                " ".join(["/entrypoint.sh", *[shlex.quote(arg) for arg in launch_args]]),
            ]
        )
    else:
        router_start = None
        if router_command_prefix is not None:
            router_start = f"{router_command_prefix} && ros2 run rmw_zenoh_cpp rmw_zenohd"
        if router_start is not None:
            shell_command = (
                f"({router_start}) & sleep 2 && "
                f"{ros_command_prefix} && {ros_launch}"
            )
        else:
            shell_command = f"{ros_command_prefix} && {ros_launch}"
    return OfficialSceneLaunchSpec(
        scenario=scenario,
        shell_command=shell_command,
        ros_launch_args=tuple(launch_args),
        expected_entities=expected_scene_entities(scenario),
        shell_environment=shell_environment,
        ros_command_prefix=ros_command_prefix,
        launch_mode=launch_mode,
        router_command_prefix=router_command_prefix,
    )


def _task_board_args(scenario: AicScenario) -> list[str]:
    args: list[str] = []
    for key, entity in scenario.task_board.mount_rails.items():
        args.extend(_rail_entity_args(key, entity))
    for key, entity in scenario.task_board.sc_rails.items():
        suffix = key.removeprefix("sc_rail_")
        args.extend(_rail_entity_args(f"sc_port_{suffix}", entity))
    for key, entity in scenario.task_board.nic_rails.items():
        suffix = key.removeprefix("nic_rail_")
        args.extend(_rail_entity_args(f"nic_card_mount_{suffix}", entity))
    return args


def _rail_entity_args(prefix: str, entity: RailEntity) -> list[str]:
    return [
        f"{prefix}_present:={_bool_arg(entity.present)}",
        f"{prefix}_translation:={entity.translation}",
        f"{prefix}_roll:={entity.roll}",
        f"{prefix}_pitch:={entity.pitch}",
        f"{prefix}_yaw:={entity.yaw}",
    ]


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _official_cable_spawn_xyz(cable_type: str, *, attach_to_gripper: bool) -> tuple[float, float, float]:
    if attach_to_gripper:
        if cable_type == "sfp_sc_cable_reversed":
            return (0.172, 0.024, 1.508)
        return (0.172, 0.024, 1.518)
    return (-0.35, 0.4, 1.15)


def _official_scene_shell_environment(*, setup_script: str | Path) -> dict[str, str]:
    if not _should_enable_official_eval_middleware(setup_script=setup_script):
        return {}
    if _should_use_container_entrypoint(setup_script=setup_script):
        return {
            "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
            "ZENOH_CONFIG_OVERRIDE": "transport/shared_memory/enabled=false",
        }
    session_script = _eval_session_script(setup_script=setup_script)
    if session_script is not None:
        return {}
    return {
        "RMW_IMPLEMENTATION": "rmw_zenoh_cpp",
        "ZENOH_CONFIG_OVERRIDE": "transport/shared_memory/enabled=false",
    }


def _should_enable_official_eval_middleware(*, setup_script: str | Path) -> bool:
    setup_path = str(setup_script)
    return (
        "/ws_aic/install/" in setup_path
        or Path("/entrypoint.sh").exists()
        or os.environ.get("AIC_USE_ZENOH_OFFICIAL_SCENE", "").strip() in {"1", "true", "TRUE"}
    )


def _should_use_container_entrypoint(*, setup_script: str | Path) -> bool:
    return Path("/entrypoint.sh").exists() and "/ws_aic/install/" in str(setup_script)


def _export_shell_environment(shell_environment: dict[str, str]) -> str:
    if not shell_environment:
        return "true"
    return " && ".join(
        f"export {name}={shlex.quote(value)}"
        for name, value in shell_environment.items()
    )


def _shell_prefix(*, setup_script: str | Path, shell_environment: dict[str, str]) -> str:
    parts: list[str] = []
    if shell_environment:
        parts.append(_export_shell_environment(shell_environment))
    parts.append(f"source {shlex.quote(str(setup_script))}")
    session_script = (
        _eval_session_script(setup_script=setup_script)
        if _should_enable_official_eval_middleware(setup_script=setup_script)
        else None
    )
    if session_script is not None:
        parts.append(f"source {shlex.quote(str(session_script))}")
    return " && ".join(parts)


def _router_shell_prefix(*, setup_script: str | Path) -> str | None:
    if not _should_enable_official_eval_middleware(setup_script=setup_script):
        return None
    router_script = _router_script(setup_script=setup_script)
    if router_script is None:
        return None
    return " && ".join(
        [
            f"source {shlex.quote(str(setup_script))}",
            f"source {shlex.quote(str(router_script))}",
        ]
    )


def _eval_session_script(*, setup_script: str | Path) -> Path | None:
    setup_root = Path(setup_script).resolve().parent.parent
    candidates = [
        setup_root / "docker" / "aic_eval" / "zenoh_config_eval_session.sh",
        setup_root / "src" / "aic" / "docker" / "aic_eval" / "zenoh_config_eval_session.sh",
        Path.cwd() / "docker" / "aic_eval" / "zenoh_config_eval_session.sh",
    ]
    for script in candidates:
        if script.exists():
            return script
    return None


def _router_script(*, setup_script: str | Path) -> Path | None:
    setup_root = Path(setup_script).resolve().parent.parent
    candidates = [
        setup_root / "docker" / "aic_eval" / "zenoh_config_router.sh",
        setup_root / "src" / "aic" / "docker" / "aic_eval" / "zenoh_config_router.sh",
        Path.cwd() / "docker" / "aic_eval" / "zenoh_config_router.sh",
    ]
    for script in candidates:
        if script.exists():
            return script
    return None
