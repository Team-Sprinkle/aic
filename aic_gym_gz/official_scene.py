"""Helpers for launching official Gazebo scenes from sampled scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
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
    launch_mode = "entrypoint" if _should_use_container_entrypoint(setup_script=setup_script) else "ros2_launch"
    if launch_mode == "entrypoint":
        shell_command = " && ".join(
            [
                _export_shell_environment(shell_environment),
                " ".join(["/entrypoint.sh", *[shlex.quote(arg) for arg in launch_args]]),
            ]
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
    return " && ".join(parts)
