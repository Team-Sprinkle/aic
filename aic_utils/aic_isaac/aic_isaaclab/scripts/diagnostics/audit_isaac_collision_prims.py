#!/usr/bin/env python3
"""Dump Isaac-stage collision prim transforms/AABBs for SFP/NIC insertion assets."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AIC-Task-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--output_dir", default="aic/outputs/agentic_reward_curriculum_20260529/collision_prim_audits")
parser.add_argument("--run_name", default="isaac_collision_prim_audit")
parser.add_argument("--episode_config_dir", default=None)
parser.add_argument("--episode_length_s", type=float, default=2.0)
parser.add_argument("--near_gate_reset_max_iterations", type=int, default=0)
parser.add_argument("--near_gate_reset_position_tolerance", type=float, default=0.0)
parser.add_argument("--near_gate_reset_orientation_tolerance", type=float, default=0.0)
parser.add_argument("--isaac_action_scale", type=float, default=1.0)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--target_reward_body", default="sfp_tip_link")
parser.add_argument("--target_reward_consistency_body", default="sfp_module_link")
parser.add_argument("--target_reward_orientation_error_mode", choices=["axis", "quat"], default="quat")
parser.add_argument("--target_reward_orientation_axis_local", type=float, nargs=3, default=(0.0, 0.0, 1.0))
parser.add_argument("--target_reward_consistency_axial_std", type=float, default=0.0008)
parser.add_argument("--target_reward_consistency_lateral_sigma", type=float, default=0.0012)
parser.add_argument(
    "--prim_regex",
    action="append",
    default=[
        "sfp_module_link/collisions",
        "sfp_tip_link/collisions",
        "runtime_sdf",
        "10099100-011lfc001",
        "sc_port",
        "nic_card",
    ],
    help="Regex matched against USD prim paths to include in the audit. Repeatable.",
)
parser.add_argument(
    "--disable_collision_prim_regex",
    action="append",
    default=[],
    help="Regex matched against USD prim paths; matching collision prims are disabled before reset.",
)
parser.add_argument(
    "--replace_sfp_body_sdf_collision_with_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable converted SFP body_sdf_collision mesh and add Gazebo SDF body_collider_box* USD cube colliders.",
)
parser.add_argument(
    "--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Like --replace_sfp_body_sdf_collision_with_sdf_boxes, but shrink each body box by per-axis margins.",
)
parser.add_argument(
    "--replace_sfp_module_sdf_collision_with_active_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable converted SFP body_sdf_collision mesh and add Gazebo-active sfp_sc_cable SFP module box colliders.",
)
parser.add_argument("--sfp_shrunk_box_margin_m", type=float, nargs=3, default=(0.00015, 0.0, 0.00015))
parser.add_argument(
    "--replace_nic_cage_p0_with_sdf_boxes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable converted NIC cage_p0_* meshes and add first-port Gazebo SDF cage box colliders.",
)
parser.add_argument(
    "--replace_nic_cage_p0_with_aligned_cubes",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Disable converted NIC cage_p0_* meshes and add USD cubes using each original prim's local transform.",
)
parser.add_argument(
    "--sfp_module_sdf",
    default="aic/aic_assets/models/SFP Module/model.sdf",
    help="SDF source for runtime SFP body-box replacement.",
)
parser.add_argument(
    "--sfp_cable_sdf",
    default="aic/aic_assets/models/sfp_sc_cable/model.sdf",
    help="SDF wrapper source used to remove inactive SFP module colliders for active-collider replacement.",
)
parser.add_argument(
    "--nic_card_sdf",
    default="aic/aic_assets/models/NIC Card/model.sdf",
    help="SDF source for --replace_nic_cage_p0_with_sdf_boxes.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.episode_config_dir:
    os.environ["AIC_ISAAC_EPISODE_CONFIG_DIR"] = str(args_cli.episode_config_dir)
os.environ["AIC_ISAAC_ENABLE_CONTACT_SENSOR"] = "1"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
import omni.usd  # noqa: E402

import aic_task.tasks  # noqa: F401,E402
from aic_task.tasks.manager_based.aic_task.mdp.insertion_geometry import compute_insertion_geometry  # noqa: E402


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / ".git").exists():
            return parent
    cwd = Path.cwd()
    for parent in (cwd, *cwd.parents):
        if (parent / ".git").exists():
            return parent
    return None


def _run_git(args: list[str]) -> str:
    root = _repo_root()
    try:
        return subprocess.run(
            ["git", *args],
            cwd=None if root is None else root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        ).stdout
    except Exception as exc:
        return f"<git failed: {exc}>"


def _jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _parse_sdf_body_boxes(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for collision in root.findall(".//collision"):
        name = str(collision.attrib.get("name", ""))
        if not name.startswith("body_collider_box"):
            continue
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected SDF body box pose/size for {name}: {pose} {size}")
        out.append({"name": name.replace(".", "_"), "translation_m": pose[:3], "rotation_rpy_rad": pose[3:], "size_m": size})
    return out


def _parse_sdf_box_colliders(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for collision in root.findall(".//collision"):
        name = str(collision.attrib.get("name", ""))
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected SDF box pose/size for {name}: {pose} {size}")
        out.append({"name": name.replace(".", "_"), "translation_m": pose[:3], "rotation_rpy_rad": pose[3:], "size_m": size})
    return out


def _parse_removed_sfp_module_colliders(path: Path) -> set[str]:
    removed: set[str] = set()
    text = path.read_text(encoding="utf-8")
    for match in re.finditer(r'<collision\s+element_id="sfp_module_link::([^"]+)"\s+action="remove"\s*/>', text):
        removed.add(match.group(1).replace(".", "_"))
    return removed


def _parse_nic_cage_p0_sdf_boxes(path: Path) -> list[dict[str, Any]]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    cage_names = {
        "10099100-011lfc001_collider_box",
        "10099100-011lfc001_collider_box.001",
        "10099100-011lfc001_collider_box.002",
        "10099100-011lfc001_collider_box.003",
        "10099100-011lfc001_collider_box.004",
    }
    for link in root.findall(".//link"):
        if link.attrib.get("name") == "nic_card_link":
            collisions = link.findall("./collision")
            break
    else:
        collisions = []
    out: list[dict[str, Any]] = []
    for collision in collisions:
        name = str(collision.attrib.get("name", ""))
        if name not in cage_names:
            continue
        box = collision.find("./geometry/box")
        size_elem = None if box is None else box.find("./size")
        if size_elem is None:
            continue
        pose_elem = collision.find("./pose")
        pose = [float(v) for v in (pose_elem.text or "0 0 0 0 0 0").split()] if pose_elem is not None else [0.0] * 6
        size = [float(v) for v in (size_elem.text or "").split()]
        if len(pose) != 6 or len(size) != 3:
            raise ValueError(f"unexpected NIC cage box pose/size for {name}: {pose} {size}")
        out.append(
            {
                "name": re.sub(r"[^A-Za-z0-9_]", "_", name),
                "translation_m": pose[:3],
                "rotation_rpy_rad": pose[3:],
                "size_m": size,
            }
        )
    return out


def _collision_enabled(prim) -> bool | None:
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        return None
    attr = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr()
    if attr and attr.HasValue():
        return bool(attr.Get())
    return True


def _disable_matching_collision_prims(run_dir: Path) -> dict[str, Any]:
    patterns = [str(p) for p in args_cli.disable_collision_prim_regex if str(p).strip()]
    if not patterns:
        return {"enabled": False, "patterns": [], "matched": [], "matched_count": 0}
    compiled = [re.compile(p) for p in patterns]
    stage = omni.usd.get_context().get_stage()
    matched: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not any(regex.search(path) for regex in compiled):
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
    report = {"enabled": True, "patterns": patterns, "matched": matched, "matched_count": len(matched)}
    (run_dir / "collision_toggle_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _replace_sfp_body_collision(run_dir: Path) -> dict[str, Any]:
    body_replacement = bool(args_cli.replace_sfp_body_sdf_collision_with_sdf_boxes)
    shrunk_body_replacement = bool(args_cli.replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes)
    active_replacement = bool(args_cli.replace_sfp_module_sdf_collision_with_active_sdf_boxes)
    if not body_replacement and not shrunk_body_replacement and not active_replacement:
        return {"enabled": False, "matched": [], "matched_count": 0, "created": [], "created_count": 0}
    if sum(bool(v) for v in (body_replacement, shrunk_body_replacement, active_replacement)) > 1:
        raise ValueError(
            "SFP collision replacement modes are mutually exclusive: choose only one of "
            "--replace_sfp_body_sdf_collision_with_sdf_boxes, "
            "--replace_sfp_body_sdf_collision_with_shrunk_sdf_boxes, or "
            "--replace_sfp_module_sdf_collision_with_active_sdf_boxes"
        )
    stage = omni.usd.get_context().get_stage()
    if active_replacement:
        removed = _parse_removed_sfp_module_colliders(Path(args_cli.sfp_cable_sdf))
        boxes = [box for box in _parse_sdf_box_colliders(Path(args_cli.sfp_module_sdf)) if str(box["name"]) not in removed]
        mode = "gazebo_active_sfp_module_link_boxes"
    else:
        boxes = _parse_sdf_body_boxes(Path(args_cli.sfp_module_sdf))
        if shrunk_body_replacement:
            margins = [max(float(v), 0.0) for v in args_cli.sfp_shrunk_box_margin_m]
            shrunk_boxes: list[dict[str, Any]] = []
            for box in boxes:
                size = [float(v) for v in box["size_m"]]
                shrunk_size = [max(size[i] - 2.0 * margins[i], 1.0e-6) for i in range(3)]
                shrunk = dict(box)
                shrunk["name"] = f"{box['name']}_shrunk"
                shrunk["size_m"] = shrunk_size
                shrunk["original_size_m"] = size
                shrunk["shrink_margin_m"] = margins
                shrunk_boxes.append(shrunk)
            boxes = shrunk_boxes
            mode = "shrunk_body_boxes"
        else:
            mode = "body_boxes_only"
    matched: list[dict[str, Any]] = []
    created: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if not path.endswith("/body_sdf_collision") or "/sfp_module/sfp_module_link/collisions/" not in path:
            continue
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
        parent_path = str(prim.GetParent().GetPath())
        for box in boxes:
            box_path = f"{parent_path}/runtime_sdf_{box['name']}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in box["translation_m"]]))
            xform.AddRotateXYZOp().Set(Gf.Vec3f(*[float(math.degrees(v)) for v in box["rotation_rpy_rad"]]))
            xform.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in box["size_m"]]))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            created.append({"path": box_path, **box})
    report = {"enabled": True, "mode": mode, "matched": matched, "matched_count": len(matched), "created": created, "created_count": len(created)}
    (run_dir / "sfp_body_collision_replacement_report.json").write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n")
    return report


def _replace_nic_cage_p0_collision(run_dir: Path) -> dict[str, Any]:
    sdf_replacement = bool(args_cli.replace_nic_cage_p0_with_sdf_boxes)
    aligned_replacement = bool(args_cli.replace_nic_cage_p0_with_aligned_cubes)
    if not sdf_replacement and not aligned_replacement:
        return {"enabled": False, "matched": [], "matched_count": 0, "created": [], "created_count": 0}
    if sdf_replacement and aligned_replacement:
        raise ValueError("NIC cage replacement modes are mutually exclusive")
    boxes = _parse_nic_cage_p0_sdf_boxes(Path(args_cli.nic_card_sdf)) if sdf_replacement else []
    if sdf_replacement and not boxes:
        raise ValueError(f"no NIC cage p0 replacement boxes found in {args_cli.nic_card_sdf}")
    stage = omni.usd.get_context().get_stage()
    matched: list[dict[str, Any]] = []
    parent_paths: set[str] = set()
    aligned_sources: list[dict[str, Any]] = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if "/nic_card/collisions/cage_p0_" not in path:
            continue
        xform = UsdGeom.Xformable(prim)
        local_matrix = xform.GetLocalTransformation() if xform else Gf.Matrix4d(1.0)
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        attr = collision_api.GetCollisionEnabledAttr()
        previous = attr.Get() if attr and attr.HasValue() else None
        if attr:
            attr.Set(False)
        else:
            collision_api.CreateCollisionEnabledAttr(False)
        matched.append({"path": path, "type": prim.GetTypeName(), "previous_collision_enabled": previous})
        parent_paths.add(str(prim.GetParent().GetPath()))
        aligned_sources.append({"name": prim.GetName(), "parent_path": str(prim.GetParent().GetPath()), "local_matrix": local_matrix})
    created: list[dict[str, Any]] = []
    if sdf_replacement:
        for parent_path in sorted(parent_paths):
            for box in boxes:
                box_path = f"{parent_path}/runtime_sdf_nic_p0_{box['name']}"
                cube = UsdGeom.Cube.Define(stage, box_path)
                cube.CreateSizeAttr(1.0)
                xform = UsdGeom.Xformable(cube.GetPrim())
                xform.ClearXformOpOrder()
                tx, ty, tz = box["translation_m"]
                rx, ry, rz = box["rotation_rpy_rad"]
                sx, sy, sz = box["size_m"]
                xform.AddTranslateOp().Set(Gf.Vec3d(float(tx), float(ty), float(tz)))
                xform.AddRotateXYZOp().Set(
                    Gf.Vec3f(float(math.degrees(rx)), float(math.degrees(ry)), float(math.degrees(rz)))
                )
                xform.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), float(sz)))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
                created.append(
                    {
                        "path": box_path,
                        "source_sdf_collision": box["name"],
                        "translation_m": list(box["translation_m"]),
                        "rotation_rpy_rad": list(box["rotation_rpy_rad"]),
                        "size_m": list(box["size_m"]),
                    }
                )
        mode = "nic_cage_p0_sdf_boxes"
    else:
        for source in aligned_sources:
            box_path = f"{source['parent_path']}/runtime_aligned_cube_{source['name']}"
            cube = UsdGeom.Cube.Define(stage, box_path)
            cube.CreateSizeAttr(1.0)
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(source["local_matrix"])
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
            created.append({"path": box_path, "source_isaac_collision": source["name"]})
        mode = "nic_cage_p0_aligned_cubes"
    report = {
        "enabled": True,
        "mode": mode,
        "matched": matched,
        "matched_count": len(matched),
        "created": created,
        "created_count": len(created),
        "source": str(args_cli.nic_card_sdf),
    }
    (run_dir / "nic_cage_p0_replacement_report.json").write_text(
        json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _matrix_rows(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[i][j]) for j in range(4)] for i in range(4)]


def _basis_lengths(matrix: Gf.Matrix4d) -> dict[str, Any]:
    rows = [[float(matrix[i][j]) for j in range(3)] for i in range(3)]
    cols = [[float(matrix[i][j]) for i in range(3)] for j in range(3)]
    row_lengths = [math.sqrt(sum(v * v for v in row)) for row in rows]
    col_lengths = [math.sqrt(sum(v * v for v in col)) for col in cols]
    return {
        "row_lengths_m": row_lengths,
        "row_lengths_mm": [1000.0 * v for v in row_lengths],
        "col_lengths_m": col_lengths,
        "col_lengths_mm": [1000.0 * v for v in col_lengths],
    }


def _attr_value(prim, name: str) -> Any:
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasValue():
        return None
    value = attr.Get()
    if isinstance(value, Gf.Range3d):
        return _range_dict(value)
    if isinstance(value, (Gf.Vec2f, Gf.Vec2d, Gf.Vec3f, Gf.Vec3d)):
        return [float(value[i]) for i in range(len(value))]
    if isinstance(value, (list, tuple)):
        return _jsonable(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _primitive_metadata(prim, world: Gf.Matrix4d) -> dict[str, Any]:
    type_name = prim.GetTypeName()
    metadata: dict[str, Any] = {"type": type_name, "transform_basis": _basis_lengths(world)}
    if type_name == "Cube":
        size = _attr_value(prim, "size")
        metadata["cube_size"] = size
        if isinstance(size, (int, float)):
            row_lengths = metadata["transform_basis"]["row_lengths_m"]
            metadata["approx_world_obb_size_m"] = [float(size) * float(v) for v in row_lengths]
            metadata["approx_world_obb_size_mm"] = [1000.0 * v for v in metadata["approx_world_obb_size_m"]]
    elif type_name == "Mesh":
        extent = _attr_value(prim, "extent")
        metadata["mesh_extent"] = extent
        if isinstance(extent, list) and len(extent) == 2 and all(isinstance(row, list) and len(row) == 3 for row in extent):
            local_size = [float(extent[1][i]) - float(extent[0][i]) for i in range(3)]
            row_lengths = metadata["transform_basis"]["row_lengths_m"]
            metadata["mesh_local_size_m"] = local_size
            metadata["mesh_local_size_mm"] = [1000.0 * v for v in local_size]
            metadata["approx_world_obb_size_m"] = [abs(local_size[i] * row_lengths[i]) for i in range(3)]
            metadata["approx_world_obb_size_mm"] = [1000.0 * v for v in metadata["approx_world_obb_size_m"]]
    else:
        for name in ("radius", "height", "size", "extent"):
            value = _attr_value(prim, name)
            if value is not None:
                metadata[name] = value
    return metadata


def _range_dict(rng: Gf.Range3d) -> dict[str, Any]:
    if rng.IsEmpty():
        return {"empty": True}
    mn = rng.GetMin()
    mx = rng.GetMax()
    size = [float(mx[i] - mn[i]) for i in range(3)]
    return {
        "empty": False,
        "min_m": [float(mn[i]) for i in range(3)],
        "max_m": [float(mx[i]) for i in range(3)],
        "size_m": size,
        "size_mm": [1000.0 * v for v in size],
        "center_m": [float((mn[i] + mx[i]) * 0.5) for i in range(3)],
    }


def _body_position(env, body_name: str) -> torch.Tensor | None:
    robot = env.unwrapped.scene.articulations.get("robot")
    if robot is None:
        return None
    names = list(getattr(robot.data, "body_names", []))
    if body_name not in names:
        return None
    return robot.data.body_pos_w[:, names.index(body_name), :]


def _episode_tensor(env, key_path: tuple[str, ...]) -> torch.Tensor | None:
    episodes = dict(getattr(env.unwrapped, "_aic_current_episode_by_env", {}) or {})
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        value: Any = episodes.get(env_id) or {}
        for key in key_path:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return None
        rows.append(torch.tensor(value, dtype=origins.dtype, device=origins.device) + origins[env_id])
    return torch.stack(rows, dim=0)


def _episode_axis(env) -> torch.Tensor | None:
    episodes = dict(getattr(env.unwrapped, "_aic_current_episode_by_env", {}) or {})
    if not episodes:
        return None
    origins = env.unwrapped.scene.env_origins
    rows = []
    for env_id in range(env.unwrapped.num_envs):
        target = (((episodes.get(env_id) or {}).get("scene") or {}).get("target") or {})
        axis = target.get("insertion_axis_world")
        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            return None
        axis_t = torch.tensor(axis, dtype=origins.dtype, device=origins.device)
        rows.append(axis_t / torch.linalg.norm(axis_t).clamp(min=1.0e-9))
    return torch.stack(rows, dim=0)


def _geometry(env, body_name: str) -> dict[str, Any]:
    body = _body_position(env, body_name)
    entrance = _episode_tensor(env, ("scene", "target", "entrance_pose_world", "position"))
    target = _episode_tensor(env, ("scene", "target", "target_pose_world", "position"))
    axis = _episode_axis(env)
    if body is None or entrance is None or target is None or axis is None:
        return {"available": False, "body": body_name}
    geom = compute_insertion_geometry(
        body_pos_w=body,
        entrance_pos_w=entrance,
        target_pos_w=target,
        axis_w=axis,
        lateral_gate_sigma=0.0005,
    )
    return {
        "available": True,
        "body": body_name,
        "signed_depth_m": _jsonable(geom.axial_depth),
        "lateral_error_m": _jsonable(geom.lateral_error),
        "target_depth_m": _jsonable(geom.target_depth),
        "axial_error_m": _jsonable(torch.abs(geom.target_depth - geom.axial_depth)),
    }


def _prim_report(patterns: list[str]) -> list[dict[str, Any]]:
    compiled = [re.compile(p) for p in patterns if p.strip()]
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy])
    rows: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if compiled and not any(regex.search(path) for regex in compiled):
            continue
        imageable = UsdGeom.Imageable(prim)
        xformable = UsdGeom.Xformable(prim)
        world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if xformable else Gf.Matrix4d(1.0)
        try:
            aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            bbox = _range_dict(aligned)
        except Exception as exc:
            bbox = {"error": f"{type(exc).__name__}: {exc}"}
        rows.append(
            {
                "path": path,
                "name": prim.GetName(),
                "type": prim.GetTypeName(),
                "active": bool(prim.IsActive()),
                "imageable": bool(imageable),
                "visibility": str(imageable.ComputeVisibility()) if imageable else None,
                "has_collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                "collision_enabled": _collision_enabled(prim),
                "world_translation_m": [float(v) for v in world.ExtractTranslation()],
                "world_matrix": _matrix_rows(world),
                "primitive_metadata": _primitive_metadata(prim, world),
                "world_aabb": bbox,
            }
        )
    rows.sort(key=lambda item: item["path"])
    return rows


def _cage_registration_report(env, prims: list[dict[str, Any]]) -> dict[str, Any]:
    entrance = _episode_tensor(env, ("scene", "target", "entrance_pose_world", "position"))
    axis = _episode_axis(env)
    if entrance is None or axis is None:
        return {"available": False}
    entrance0 = entrance[0].detach().cpu()
    axis0 = axis[0].detach().cpu()
    axis0 = axis0 / torch.linalg.norm(axis0).clamp(min=1.0e-9)
    rows: list[dict[str, Any]] = []
    for prim in prims:
        path = str(prim.get("path", ""))
        if "cage_p0_" not in path and "runtime_sdf_nic_p0" not in path and "runtime_aligned_cube_cage_p0" not in path:
            continue
        bbox = prim.get("world_aabb") or {}
        center = bbox.get("center_m")
        center_source = "world_aabb"
        if not isinstance(center, list) or len(center) != 3:
            center = prim.get("world_translation_m")
            center_source = "world_translation"
        if not isinstance(center, list) or len(center) != 3:
            continue
        center_t = torch.tensor(center, dtype=axis0.dtype)
        delta = center_t - entrance0
        axial = float(torch.dot(delta, axis0))
        lateral_vec = delta - axial * axis0
        size_mm = bbox.get("size_mm")
        rows.append(
            {
                "path": path,
                "collision_enabled": prim.get("collision_enabled"),
                "center_m": center,
                "center_source": center_source,
                "center_axial_from_entrance_m": axial,
                "center_axial_from_entrance_mm": axial * 1000.0,
                "center_lateral_from_entrance_m": float(torch.linalg.norm(lateral_vec)),
                "center_lateral_from_entrance_mm": float(torch.linalg.norm(lateral_vec)) * 1000.0,
                "world_aabb_size_mm": size_mm,
                "approx_world_obb_size_mm": (prim.get("primitive_metadata") or {}).get("approx_world_obb_size_mm"),
            }
        )
    rows.sort(key=lambda item: (str(item["path"]).replace("runtime_sdf_nic_p0", "zz_runtime_sdf_nic_p0")))
    return {
        "available": True,
        "entrance_world_m": _jsonable(entrance0),
        "axis_world": _jsonable(axis0),
        "rows": rows,
    }


def _configure_env_cfg(env_cfg) -> None:
    env_cfg.seed = int(args_cli.seed)
    if float(args_cli.episode_length_s) > 0.0:
        env_cfg.episode_length_s = float(args_cli.episode_length_s)
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.center_rgb = None
        env_cfg.observations.policy.left_rgb = None
        env_cfg.observations.policy.right_rgb = None
    env_cfg.actions.arm_action.scale = float(args_cli.isaac_action_scale)
    _configure_semantic_reward_terms(env_cfg)
    reset_event = getattr(getattr(env_cfg, "events", None), "reset_robot_tcp_to_episode_start", None)
    params = getattr(reset_event, "params", None)
    if isinstance(params, dict):
        if int(args_cli.near_gate_reset_max_iterations) > 0:
            params["max_iterations"] = int(args_cli.near_gate_reset_max_iterations)
        if float(args_cli.near_gate_reset_position_tolerance) > 0.0:
            params["position_tolerance"] = float(args_cli.near_gate_reset_position_tolerance)
        if float(args_cli.near_gate_reset_orientation_tolerance) > 0.0:
            params["orientation_tolerance"] = float(args_cli.near_gate_reset_orientation_tolerance)


def _configure_semantic_reward_terms(env_cfg) -> None:
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None:
        return
    body_names = [str(args_cli.target_reward_body)]
    for name in (
        "target_distance_tanh",
        "target_distance_exp",
        "target_distance_progress",
        "target_orientation_tanh",
        "target_orientation_gated_exp",
        "target_reaching_bonus",
        "target_success_once_bonus",
        "target_lateral_error",
        "target_motion_projection",
        "target_lateral_progress",
        "target_axial_progress",
        "target_insertion_corridor",
        "target_cheatcode_phase_reward",
    ):
        term = getattr(rewards, name, None)
        params = None if term is None else getattr(term, "params", None)
        if not isinstance(params, dict):
            continue
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = str(args_cli.target_reward_orientation_error_mode)
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)
        if "consistency_axial_std" in params:
            params["consistency_axial_std"] = float(args_cli.target_reward_consistency_axial_std)
        if "consistency_lateral_sigma" in params:
            params["consistency_lateral_sigma"] = float(args_cli.target_reward_consistency_lateral_sigma)
    terminations = getattr(env_cfg, "terminations", None)
    target_success = None if terminations is None else getattr(terminations, "target_success", None)
    params = None if target_success is None else getattr(target_success, "params", None)
    if isinstance(params, dict):
        body_cfg = params.get("body_cfg")
        if body_cfg is not None:
            body_cfg.body_names = body_names
        target_cfg = params.get("target_cfg")
        if target_cfg is not None:
            target_cfg.name = "nic_card"
        if "orientation_error_mode" in params:
            params["orientation_error_mode"] = str(args_cli.target_reward_orientation_error_mode)
        if "orientation_axis_local" in params:
            params["orientation_axis_local"] = tuple(float(x) for x in args_cli.target_reward_orientation_axis_local)
        if "consistency_body_name" in params:
            params["consistency_body_name"] = str(args_cli.target_reward_consistency_body)


def _write_summary(run_dir: Path, data: dict[str, Any]) -> None:
    prims = data.get("prims", [])
    collision_prims = [p for p in prims if p.get("has_collision_api")]
    enabled = [p for p in collision_prims if p.get("collision_enabled") is not False]
    lines = [
        "# Isaac Collision Prim Audit",
        "",
        f"Strict success claimed: `false`.",
        f"Included prims: `{len(prims)}`.",
        f"CollisionAPI prims: `{len(collision_prims)}`.",
        f"Enabled collision prims: `{len(enabled)}`.",
        "",
        "## Semantic Geometry At Reset",
        "",
        "```json",
        json.dumps(data.get("semantic_geometry", {}), indent=2)[:4000],
        "```",
        "",
        "## Largest Included AABBs",
        "",
    ]
    sized = []
    for prim in prims:
        bbox = prim.get("world_aabb") or {}
        size = bbox.get("size_mm")
        if isinstance(size, list) and len(size) == 3:
            sized.append((max(float(v) for v in size), prim))
    for _, prim in sorted(sized, key=lambda item: item[0], reverse=True)[:20]:
        lines.append(f"- `{prim['path']}` size_mm={prim['world_aabb']['size_mm']} collision={prim.get('collision_enabled')}")
    obb_sized = []
    for prim in prims:
        meta = prim.get("primitive_metadata") or {}
        size = meta.get("approx_world_obb_size_mm")
        if isinstance(size, list) and len(size) == 3:
            obb_sized.append((max(float(v) for v in size), prim))
    if obb_sized:
        lines.extend(["", "## Largest Transform-Derived OBBs", ""])
        for _, prim in sorted(obb_sized, key=lambda item: item[0], reverse=True)[:20]:
            size = prim["primitive_metadata"]["approx_world_obb_size_mm"]
            lines.append(f"- `{prim['path']}` approx_obb_size_mm={size} collision={prim.get('collision_enabled')}")
    cage_registration = data.get("cage_registration") or {}
    rows = cage_registration.get("rows") if isinstance(cage_registration, dict) else None
    if isinstance(rows, list):
        lines.extend(
            [
                "",
                "## NIC Cage Registration",
                "",
                "| prim | collision | center axial mm | center lateral mm | size mm |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in rows[:40]:
            lines.append(
                "| "
                f"`{row.get('path')}` | "
                f"`{row.get('collision_enabled')}` | "
                f"{float(row.get('center_axial_from_entrance_mm', 0.0)):.3f} | "
                f"{float(row.get('center_lateral_from_entrance_mm', 0.0)):.3f} | "
                f"`{row.get('approx_world_obb_size_mm') or row.get('world_aabb_size_mm')}` |"
            )
        lines.append("")
        lines.append("Center is computed from `world_aabb` when available, otherwise from `world_translation`.")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    output_root = Path(args_cli.output_dir)
    run_dir = output_root / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M-%S')}_{args_cli.run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(str(x)) for x in sys.argv) + "\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(_run_git(["status", "--short", "--branch"]), encoding="utf-8")
    (run_dir / "git_diff.patch").write_text(_run_git(["diff", "--", "."]), encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(vars(args_cli), indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[collision-audit] run_dir={run_dir}", flush=True)
    env = None
    try:
        status_path.write_text(json.dumps({"stage": "parse_env_cfg"}, indent=2) + "\n")
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
        _configure_env_cfg(env_cfg)
        status_path.write_text(json.dumps({"stage": "gym_make"}, indent=2) + "\n")
        env = gym.make(args_cli.task, cfg=env_cfg)
        replacement = _replace_sfp_body_collision(run_dir)
        nic_replacement = _replace_nic_cage_p0_collision(run_dir)
        collision_toggle = _disable_matching_collision_prims(run_dir)
        status_path.write_text(json.dumps({"stage": "reset"}, indent=2) + "\n")
        env.reset(seed=int(args_cli.seed))
        prims = _prim_report([str(p) for p in args_cli.prim_regex])
        data = {
            "run_dir": str(run_dir),
            "replacement_report": replacement,
            "nic_cage_p0_replacement_report": nic_replacement,
            "collision_toggle_report": collision_toggle,
            "semantic_geometry": {
                "sfp_tip_link": _geometry(env, "sfp_tip_link"),
                "sfp_module_link": _geometry(env, "sfp_module_link"),
            },
            "prims": prims,
            "cage_registration": _cage_registration_report(env, prims),
        }
        (run_dir / "collision_prim_audit.json").write_text(json.dumps(_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary(run_dir, data)
        status_path.write_text(json.dumps({"stage": "complete"}, indent=2) + "\n")
        print(json.dumps({"run_dir": str(run_dir), "prim_count": len(data["prims"])}, indent=2), flush=True)
        return 0
    except Exception as exc:
        payload = {"stage": "error", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        (run_dir / "error.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        return 1
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
