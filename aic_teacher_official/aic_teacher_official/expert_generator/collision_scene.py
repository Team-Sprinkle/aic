"""Best-effort rigid planning-scene extraction from AIC engine YAML."""

from __future__ import annotations

import math
from pathlib import Path
import subprocess
from typing import Any
import xml.etree.ElementTree as ET

import yaml

from aic_teacher_official.expert_generator.scene_snapshot import ObjectGeometry, SerializablePose


BOARD_BOX_DIMENSIONS_M = [0.75, 0.38, 0.035]
NIC_BOX_DIMENSIONS_M = [0.12, 0.06, 0.03]
SC_MOUNT_BOX_DIMENSIONS_M = [0.08, 0.05, 0.04]
FIXTURE_BOX_DIMENSIONS_M = [0.06, 0.04, 0.035]


def load_engine_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Engine config must be a YAML map: {config_path}")
    return data


def first_trial_id(config: dict[str, Any]) -> str:
    trials = config.get("trials")
    if not isinstance(trials, dict) or not trials:
        raise ValueError("Engine config must contain a non-empty 'trials' map")
    return next(iter(trials.keys()))


def trial_config(config: dict[str, Any], trial_id: str | None = None) -> tuple[str, dict[str, Any]]:
    resolved = trial_id or first_trial_id(config)
    trials = config.get("trials")
    if not isinstance(trials, dict) or resolved not in trials:
        raise ValueError(f"Trial {resolved!r} not found in engine config")
    trial = trials[resolved]
    if not isinstance(trial, dict):
        raise ValueError(f"Trial {resolved!r} must be a YAML map")
    return resolved, trial


def object_geometries_from_engine_config(
    config_or_path: dict[str, Any] | str | Path,
    *,
    trial_id: str | None = None,
) -> list[ObjectGeometry]:
    config = (
        load_engine_config(config_or_path)
        if isinstance(config_or_path, (str, Path))
        else config_or_path
    )
    resolved_trial_id, trial = trial_config(config, trial_id)
    scene = trial.get("scene", {}) if isinstance(trial.get("scene", {}), dict) else {}
    board = scene.get("task_board", {}) if isinstance(scene.get("task_board", {}), dict) else {}
    board_pose = board.get("pose", {}) if isinstance(board.get("pose", {}), dict) else {}
    objects: list[ObjectGeometry] = []
    objects.append(
        ObjectGeometry(
            name=f"{resolved_trial_id}_task_board_base",
            pose=SerializablePose(
                position=[
                    float(board_pose.get("x", 0.0)),
                    float(board_pose.get("y", 0.0)),
                    float(board_pose.get("z", 0.0)),
                ],
                orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                frame_id="world",
            ),
            shape="box",
            dimensions=BOARD_BOX_DIMENSIONS_M,
            role="task_board",
            metadata={"source": "engine_config", "yaw_rad": float(board_pose.get("yaw", 0.0))},
        )
    )
    for rail_name, rail in board.items():
        if not isinstance(rail, dict) or not rail.get("entity_present"):
            continue
        if not rail_name.endswith("_rail_0") and "_rail_" not in rail_name:
            continue
        pose = rail.get("entity_pose", {}) if isinstance(rail.get("entity_pose", {}), dict) else {}
        entity_name = str(rail.get("entity_name", rail_name))
        role, dims = _role_and_dimensions(rail_name)
        objects.append(
            ObjectGeometry(
                name=f"{resolved_trial_id}_{entity_name}",
                pose=SerializablePose(
                    position=[
                        float(board_pose.get("x", 0.0)),
                        float(board_pose.get("y", 0.0)) + float(pose.get("translation", 0.0)),
                        float(board_pose.get("z", 0.0)),
                    ],
                    orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
                    frame_id="world",
                ),
                shape="box",
                dimensions=dims,
                role=role,
                metadata={
                    "source": "engine_config",
                    "rail": rail_name,
                    "translation_along_rail_m": float(pose.get("translation", 0.0)),
                    "yaw_rad": float(pose.get("yaw", 0.0)),
                },
            )
        )
    return objects


def _role_and_dimensions(rail_name: str) -> tuple[str, list[float]]:
    if rail_name.startswith("nic_rail"):
        return "nic_card", NIC_BOX_DIMENSIONS_M
    if rail_name.startswith("sc_rail"):
        return "sc_port", SC_MOUNT_BOX_DIMENSIONS_M
    return "fixture_mount", FIXTURE_BOX_DIMENSIONS_M


def object_geometries_from_gazebo_assets(
    config_or_path: dict[str, Any] | str | Path,
    *,
    repo_root: str | Path,
    trial_id: str | None = None,
    xacro_executable: str = "xacro",
) -> list[ObjectGeometry]:
    """Expand the trial's actual task-board xacro and nested SDF collision shapes.

    This deliberately fails on unsupported collision geometry or missing assets.
    The approximate boxes above are retained for legacy diagnostics, but must
    not be fed to MoveIt as if they matched Gazebo.
    """
    config = load_engine_config(config_or_path) if isinstance(config_or_path, (str, Path)) else config_or_path
    resolved_trial_id, trial = trial_config(config, trial_id)
    board = trial["scene"]["task_board"]
    board_pose = board["pose"]
    root = Path(repo_root).resolve()
    xacro_file = root / "aic_description/urdf/task_board.urdf.xacro"
    args = [xacro_executable, str(xacro_file)]
    for field in ("x", "y", "z", "roll", "pitch", "yaw"):
        args.append(f"{field}:={float(board_pose.get(field, 0.0))}")
    for rail_name, rail in board.items():
        if not isinstance(rail, dict) or not rail.get("entity_present"):
            continue
        if rail_name.startswith("nic_rail_"):
            xacro_name = rail_name.replace("nic_rail_", "nic_card_mount_", 1)
        elif rail_name.startswith("sc_rail_"):
            xacro_name = rail_name.replace("sc_rail_", "sc_port_", 1)
        elif rail_name.startswith(("lc_mount_rail_", "sfp_mount_rail_", "sc_mount_rail_")):
            xacro_name = rail_name
        else:
            raise ValueError(f"Unknown present board rail: {rail_name}")
        args.append(f"{xacro_name}_present:=true")
        pose = rail.get("entity_pose", {})
        for field in ("translation", "roll", "pitch", "yaw"):
            args.append(f"{xacro_name}_{field}:={float(pose.get(field, 0.0))}")
    expanded = subprocess.run(args, check=True, text=True, capture_output=True).stdout
    urdf = ET.fromstring(expanded)
    world_board = _rpy_transform([float(board_pose.get(k, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")])
    objects: list[ObjectGeometry] = []
    base_link = urdf.find("./link[@name='task_board_base_link']")
    if base_link is None:
        raise ValueError("Expanded task board lacks task_board_base_link")
    link_transforms = {"task_board_base_link": world_board}
    for joint in urdf.findall("./joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None or parent.get("link") != "task_board_base_link":
            continue
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"Missing fixed-joint origin: {joint.get('name')}")
        link_transforms[child.get("link")] = _compose(
            world_board, _rpy_transform(_numbers(origin.get("xyz"), 3) + _numbers(origin.get("rpy"), 3)),
        )
    for link in urdf.findall("./link"):
        link_name = link.get("name")
        if link_name not in link_transforms:
            raise ValueError(f"Unexpected task-board link without fixed transform: {link_name}")
        for collision in link.findall("collision"):
            role = "task_board" if link_name == "task_board_base_link" else "fixture"
            objects.append(_collision_object(collision, link_transforms[link_name], f"{resolved_trial_id}_{link_name}", role))

    includes = urdf.findall("./gazebo/include")
    expected = sum(
        bool(isinstance(rail, dict) and rail.get("entity_present"))
        for name, rail in board.items() if name.startswith(("nic_rail_", "sc_rail_"))
    )
    if len(includes) != expected:
        raise ValueError(f"Expanded task board has {len(includes)} includes; expected {expected}")
    for include in includes:
        include_name = include.findtext("name")
        uri = include.findtext("uri")
        if not include_name or not uri or not uri.startswith("package://aic_assets/models/"):
            raise ValueError(f"Unsupported Gazebo include: {include_name!r}, {uri!r}")
        model_dir = root / "aic_assets/models" / uri.removeprefix("package://aic_assets/models/")
        sdf_path = model_dir / "model.sdf"
        sdf = ET.parse(sdf_path).getroot()
        include_tf = _compose(world_board, _rpy_transform(_numbers(include.findtext("pose"), 6)))
        model_tf = _compose(include_tf, _rpy_transform(_numbers(sdf.findtext("./model/pose"), 6)))
        count = 0
        for link in sdf.findall("./model/link"):
            link_tf = _compose(model_tf, _rpy_transform(_numbers(link.findtext("pose"), 6)))
            for collision in link.findall("collision"):
                obj = _collision_object(
                    collision, link_tf, f"{resolved_trial_id}_{include_name}_{link.get('name')}",
                    "nic_card" if include_name.startswith("nic_card_mount") else "fixture",
                )
                obj.metadata["gazebo_model_name"] = include_name
                obj.metadata["model_pose_world"] = {
                    "position": model_tf[0], "orientation_xyzw": model_tf[1],
                }
                objects.append(obj)
                count += 1
        if not count:
            raise ValueError(f"Included Gazebo model has no supported collisions: {sdf_path}")
    if not objects:
        raise ValueError("Gazebo task board produced no collision objects")
    return objects


def _numbers(value: str | None, count: int) -> list[float]:
    result = [float(x) for x in value.split()] if value else [0.0] * count
    if len(result) != count:
        raise ValueError(f"Expected {count} pose values, got {result}")
    return result


def _multiply_quaternions(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def _rpy_transform(values: list[float]) -> tuple[list[float], list[float]]:
    x, y, z, roll, pitch, yaw = values
    qx = [math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)]
    qy = [0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]
    qz = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
    return [x, y, z], _multiply_quaternions(_multiply_quaternions(qz, qy), qx)


def _compose(
    parent: tuple[list[float], list[float]], child: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    p, q = parent
    c, r = child
    v = _multiply_quaternions(_multiply_quaternions(q, [*c, 0.0]), [-q[0], -q[1], -q[2], q[3]])
    return [p[i] + v[i] for i in range(3)], _multiply_quaternions(q, r)


def _collision_object(
    collision: ET.Element, parent: tuple[list[float], list[float]], prefix: str, role: str,
) -> ObjectGeometry:
    origin = collision.find("origin")
    if origin is not None:
        xyz = _numbers(origin.get("xyz"), 3)
        rpy = _numbers(origin.get("rpy"), 3)
        local = _rpy_transform(xyz + rpy)
    else:
        local = _rpy_transform(_numbers(collision.findtext("pose"), 6))
    position, quat = _compose(parent, local)
    geometry = collision.find("geometry")
    if geometry is None:
        raise ValueError(f"Collision lacks geometry: {prefix}/{collision.get('name')}")
    box = geometry.find("box")
    cylinder = geometry.find("cylinder")
    if box is not None:
        shape, dimensions = "box", _numbers(box.get("size") or box.findtext("size"), 3)
    elif cylinder is not None:
        shape = "cylinder"
        dimensions = [float(cylinder.findtext("length")), float(cylinder.findtext("radius"))]
    else:
        raise ValueError(f"Unsupported collision shape: {prefix}/{collision.get('name')}")
    if min(dimensions) <= 0:
        raise ValueError(f"Nonpositive collision dimension: {prefix}/{collision.get('name')}")
    return ObjectGeometry(
        name=f"{prefix}_{collision.get('name')}",
        pose=SerializablePose(position, quat, frame_id="world"),
        shape=shape,
        dimensions=dimensions,
        role=role,
        metadata={"source": "gazebo_assets", "link_or_model": prefix},
    )
