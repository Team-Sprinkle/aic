#!/usr/bin/env python3
"""Replace the SC port's visual-mesh convex hulls with source SDF colliders.

The distributed SC port USD uses its GLB visual meshes as convex collisions.
One convex hull closes the receptacle opening.  The Gazebo source instead has
explicit boxes and cylinders in ``aic_assets/models/SC Port/model.sdf``.
Write a sibling USD so relative material/mesh references remain valid.
"""

from __future__ import annotations

import argparse
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--sdf", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--omit-collision-name", action="append", default=[],
    help="Diagnostic ablation: omit an exact SDF collision name from the generated asset.",
)
parser.add_argument(
    "--include-collision-name", action="append", default=[],
    help="Diagnostic ablation: include only these exact SDF collision names.",
)
parser.add_argument(
    "--omit-all-sdf-collisions", action="store_true",
    help="Diagnostic ablation: disable every SC port collider while preserving visuals.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics


def main() -> None:
    source = args.source.resolve()
    output = args.output.resolve()
    if source.parent != output.parent:
        raise ValueError("SC port output USD must be beside its source to preserve relative references")
    if source == output:
        raise ValueError("Do not overwrite the distributed SC port USD")
    root_xml = ET.parse(args.sdf).getroot()
    link = root_xml.find("./model/link[@name='sc_port_link']")
    if link is None:
        raise ValueError("SDF has no sc_port_link")
    collisions = link.findall("collision")
    if not collisions:
        raise ValueError("SDF has no SC port collisions")

    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    rigid_root = stage.GetPrimAtPath("/World/sc_port_visual")
    if not rigid_root.IsValid() or not rigid_root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("SC port USD has no expected rigid root")
    # The visual GLB is in centimeters and Y-up. Its authored root includes
    # +90 degrees about X and a 0.01 scale. The Isaac scene spawn was built
    # for that GLB, whereas Gazebo spawns the source SDF model. Comparing the
    # scored Gazebo port-link and base-link TFs with the live Isaac rigid root
    # shows one further 180-degree Y rotation between those model frames.
    # Without it, the SDF bottom plate appears on the insertion side and
    # blocks a scored Gazebo plug pose by about 7.7 mm.
    conversion = UsdGeom.Xformable(rigid_root).GetLocalTransformation()
    conversion_inverse = conversion.GetInverse()
    model_alignment = Gf.Matrix4d(1.0)
    model_alignment.SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 180.0))
    disabled_visual_colliders = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            disabled_visual_colliders.append(str(prim.GetPath()))

    authored = []
    omitted = set(args.omit_collision_name)
    included = set(args.include_collision_name)
    present = {element.get("name") for element in collisions}
    if (omitted | included) - present:
        raise ValueError(f"Unknown SDF collision names: {sorted((omitted | included) - present)}")
    for index, element in enumerate(collisions):
        name = element.get("name") or f"collision_{index}"
        if args.omit_all_sdf_collisions or name in omitted or (included and name not in included):
            print(f"omitted_source_collider={name}", flush=True)
            continue
        pose_values = [float(v) for v in (element.findtext("pose") or "0 0 0 0 0 0").split()]
        if len(pose_values) != 6:
            raise ValueError(f"Invalid pose for {name}: {pose_values}")
        x, y, z, roll, pitch, yaw = pose_values
        geometry = element.find("geometry")
        if geometry is None:
            raise ValueError(f"Missing geometry for {name}")
        path = f"{rigid_root.GetPath()}/sdf_collision_{index:02d}"
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        if box is not None:
            size = [float(v) for v in (box.findtext("size") or "").split()]
            if len(size) != 3 or any(v <= 0 for v in size):
                raise ValueError(f"Invalid box size for {name}: {size}")
            shape = UsdGeom.Cube.Define(stage, path)
            shape.CreateSizeAttr(1.0)
            shape_scale = Gf.Vec3d(*size)
            kind = "box"
        elif cylinder is not None:
            radius = float(cylinder.findtext("radius") or "0")
            length = float(cylinder.findtext("length") or "0")
            if radius <= 0 or length <= 0:
                raise ValueError(f"Invalid cylinder size for {name}")
            shape = UsdGeom.Cylinder.Define(stage, path)
            shape.CreateRadiusAttr(radius)
            shape.CreateHeightAttr(length)
            shape_scale = Gf.Vec3d(1.0)
            kind = "cylinder"
        else:
            raise ValueError(f"Unsupported SDF collision geometry for {name}")

        scale = Gf.Matrix4d(1.0)
        scale.SetScale(shape_scale)
        rotation = Gf.Matrix4d(1.0)
        # SDF uses fixed-axis roll, pitch, yaw: Rz(yaw) Ry(pitch) Rx(roll).
        rotation.SetRotate(
            Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(yaw))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(pitch))
            * Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(roll))
        )
        translation = Gf.Matrix4d(1.0)
        translation.SetTranslate(Gf.Vec3d(x, y, z))
        shape.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
            scale * rotation * translation * model_alignment * conversion_inverse
        )
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        physx = PhysxSchema.PhysxCollisionAPI.Apply(shape.GetPrim())
        physx.CreateContactOffsetAttr().Set(0.00001)
        physx.CreateRestOffsetAttr().Set(0.0)
        authored.append((name, kind, path, tuple(pose_values)))

    stage.GetRootLayer().Save()
    print(f"source={source}", flush=True)
    print(f"sdf={args.sdf.resolve()}", flush=True)
    print(f"disabled_visual_colliders={len(disabled_visual_colliders)}", flush=True)
    for name, kind, path, pose in authored:
        print(f"source_collider={name} kind={kind} prim={path} pose={pose}", flush=True)
    print(f"output={output}", flush=True)


try:
    main()
finally:
    simulation_app.close()
