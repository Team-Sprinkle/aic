#!/usr/bin/env python3
"""Build a stable SC-gripped visual variant of the distributed Isaac asset.

The distributed robot/cable USD rigidly attaches ``lc_plug_link`` to the
gripper. Rewiring that articulation to the free SC body proved unstable. This
builder preserves every joint in the tested articulation, hides the LC/SFP
geometry, and copies the SC render and primitive collision geometry onto the
already gripped body. Control and reward code must interpret
``lc_plug_link`` through the documented SC semantic-tip transform.

The output must be beside the source USD because the source contains relative
asset references. This also avoids copying a large asset tree.
"""

import argparse
import math
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument(
    "--mode",
    choices=("stable_proxy", "stable_proxy_aligned", "aligned_topology", "reversed_topology"),
    default="stable_proxy",
    help="Build the collision proxy or the SC-gripped topology diagnostic.",
)
parser.add_argument("--proxy-delta-wxyz", type=float, nargs=4)
parser.add_argument("--print-collision-paths", action="store_true")
parser.add_argument("--collision-path-output", type=Path)
parser.add_argument("--stage-path-output", type=Path)
parser.add_argument("--print-joint-drives", action="store_true")
parser.add_argument("--print-topology-transforms", action="store_true")
parser.add_argument(
    "--use-source-reversed-grasp",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Apply the reversed Gazebo cable spawn/endpoint transform relative to the prepared normal grasp.",
)
parser.add_argument(
    "--sc-grasp-extension-m",
    type=float,
    default=0.03,
    help="Move the SC body forward in its local tip direction to provide board clearance in the Isaac grasp.",
)
parser.add_argument(
    "--sc-grasp-roll-deg",
    type=float,
    default=0.0,
    help="Rotate the gripper grasp about the SC connector's local insertion axis.",
)
parser.add_argument(
    "--disable-collision-substring",
    action="append",
    default=[],
    help="Diagnostic only: disable collision API on paths containing this substring.",
)
parser.add_argument(
    "--deinstance-substring",
    action="append",
    default=[],
    help="Make matching instance roots editable before applying diagnostic collision changes.",
)
parser.add_argument(
    "--expand-arm-joint-limits",
    action="store_true",
    help="Use the UR arm's continuous +/-360 degree range for source-grasp reachability diagnostics.",
)
parser.add_argument(
    "--apply-source-reversed-collision-contract",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "For reversed_topology, reproduce the official Gazebo model's cable/gripper collision exceptions: "
        "remove endpoint-0/connection-0 collision and shorten/shift the first cable-link collider."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args)
simulation_app = app.app

from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics


source = args.source.resolve()
output = args.output.resolve()
if source.parent != output.parent:
    raise ValueError(
        "Output must be in the source USD directory so relative mesh and texture references remain valid: "
        f"source={source}, output={output}"
    )
output.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(source, output)
stage = Usd.Stage.Open(str(output))
layer = stage.GetRootLayer()
base = "/World/cable"


def _set_quaternion(attr, value: Gf.Quatd) -> None:
    if attr.GetTypeName() == Sdf.ValueTypeNames.Quatf:
        attr.Set(Gf.Quatf(float(value.GetReal()), Gf.Vec3f(*value.GetImaginary())))
    else:
        attr.Set(value)


def _replace_local_transform(prim: Usd.Prim, matrix: Gf.Matrix4d) -> None:
    """Author one matrix op while preserving the prim and all its children."""
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(matrix)


def _joint_frame(position: tuple[float, float, float], yaw: float) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), float(yaw) * 180.0 / 3.141592653589793))
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    return matrix


def _reversed_topology() -> None:
    """Rebuild the endpoint topology for the source reversed cable.

    The prepared USD is the normal cable: LC is fixed to rope link 0 and SC is
    fixed to link 20.  The reversed Gazebo model instead fixes SC to endpoint
    0 and LC/SFP to endpoint 1.  Keep the tested rope articulation, derive the
    collapsed endpoint transforms from the source SDF poses, and move the rope
    as one rigid initial assembly so the newly gripped SC body starts exactly
    where the prepared LC body was.  All authored body poses then satisfy both
    endpoint fixed joints at time zero.
    """
    time = Usd.TimeCode.Default()
    cable = stage.GetPrimAtPath(base)
    rope = stage.GetPrimAtPath(base + "/Rope")
    link0 = stage.GetPrimAtPath(base + "/Rope/Rope/link_0")
    link20 = stage.GetPrimAtPath(base + "/Rope/Rope/link_20")
    lc_parent = stage.GetPrimAtPath(base + "/lc_plug")
    sc_parent = stage.GetPrimAtPath(base + "/sc_plug")
    required = (cable, rope, link0, link20, lc_parent, sc_parent)
    if not all(prim.IsValid() for prim in required):
        raise RuntimeError("Prepared cable USD does not contain the expected rope/plug prims")

    cable_world = UsdGeom.Xformable(cable).ComputeLocalToWorldTransform(time)
    old_lc_world = UsdGeom.Xformable(lc_parent).ComputeLocalToWorldTransform(time)
    link0_local = UsdGeom.Xformable(link0).GetLocalTransformation()
    link20_local = UsdGeom.Xformable(link20).GetLocalTransformation()

    # These joint frames are the source reversed SDF poses expressed against
    # the endpoint bodies that the USD importer collapsed into rope links.
    # See the derivation in docs/SC_CABLE_SNAG_RECOVERY_PLAN.md.
    link0_to_sc = _joint_frame((0.015539, -0.000748, -0.001346), 0.0)
    link20_to_lc = _joint_frame((0.02225, 0.0, 0.0), -1.5707963267948966)

    desired_sc_world = old_lc_world
    if args.use_source_reversed_grasp:
        # Difference between the mean source Gazebo grasps:
        #   normal:   cable spawn * normal connection_0 * LC(-41 mm, +90 deg yaw)
        #   reversed: cable spawn * reversed connection_0 * SC(-52 mm, 180 deg yaw)
        # Expressed in the prepared gripped-LC frame. The cable plugin welds
        # the plug at this existing relative pose; it does not move the plug
        # origin onto the tool origin.
        grasp_delta = Gf.Matrix4d(1.0)
        grasp_delta.SetRotate(
            Gf.Rotation(
                Gf.Quatd(
                    0.4524538566732309,
                    Gf.Vec3d(0.5435549803762655, 0.49199366526501526, 0.5077161847221076),
                )
            )
        )
        grasp_delta.SetTranslateOnly(Gf.Vec3d(-0.0000214963851, 0.0125764861, 0.00485154245))
        desired_sc_world = grasp_delta * old_lc_world
    elif abs(float(args.sc_grasp_extension_m)) > 0.0 or abs(float(args.sc_grasp_roll_deg)) > 0.0:
        grasp_adjustment = Gf.Matrix4d(1.0)
        grasp_adjustment.SetRotate(
            Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), float(args.sc_grasp_roll_deg))
        )
        grasp_adjustment.SetTranslateOnly(Gf.Vec3d(float(args.sc_grasp_extension_m), 0.0, 0.0))
        desired_sc_world = grasp_adjustment * old_lc_world
    desired_link0_world = link0_to_sc.GetInverse() * desired_sc_world
    desired_rope_world = link0_local.GetInverse() * desired_link0_world
    desired_link20_world = link20_local * desired_rope_world
    desired_lc_world = link20_to_lc * desired_link20_world

    _replace_local_transform(rope, desired_rope_world * cable_world.GetInverse())
    _replace_local_transform(sc_parent, desired_sc_world * cable_world.GetInverse())
    _replace_local_transform(lc_parent, desired_lc_world * cable_world.GetInverse())

    fixed0 = UsdPhysics.Joint(stage.GetPrimAtPath(base + "/Rope/fixedJoint"))
    fixed1 = UsdPhysics.Joint(stage.GetPrimAtPath(base + "/Rope/fixedJoint2"))
    fixed0.GetBody1Rel().SetTargets([Sdf.Path(base + "/sc_plug/sc_plug_link")])
    fixed1.GetBody1Rel().SetTargets([Sdf.Path(base + "/lc_plug/lc_plug_link")])
    fixed0.GetLocalPos0Attr().Set(Gf.Vec3f(0.015539, -0.000748, -0.001346))
    _set_quaternion(fixed0.GetLocalRot0Attr(), Gf.Quatd(1.0, Gf.Vec3d(0.0)))
    fixed1.GetLocalPos0Attr().Set(Gf.Vec3f(0.02225, 0.0, 0.0))
    _set_quaternion(
        fixed1.GetLocalRot0Attr(),
        Gf.Quatd(0.7071067811865476, Gf.Vec3d(0.0, 0.0, -0.7071067811865475)),
    )
    for joint in (fixed0, fixed1):
        joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0))
        _set_quaternion(joint.GetLocalRot1Attr(), Gf.Quatd(1.0, Gf.Vec3d(0.0)))

    gripper_joint = UsdPhysics.Joint(stage.GetPrimAtPath(base + "/gripper_attach_joint"))
    gripper_joint.GetBody1Rel().SetTargets([Sdf.Path(base + "/sc_plug/sc_plug_link")])
    # Preserve the source plugin behavior: weld the plug at its authored
    # relative grasp instead of forcing the plug body origin onto the old LC
    # joint frame.
    gripper_body0_path = gripper_joint.GetBody0Rel().GetTargets()[0]
    gripper_body0_world = UsdGeom.Xformable(stage.GetPrimAtPath(gripper_body0_path)).ComputeLocalToWorldTransform(time)
    gripper_local0 = Gf.Matrix4d(1.0)
    gripper_local0.SetRotate(Gf.Rotation(gripper_joint.GetLocalRot0Attr().Get()))
    gripper_local0.SetTranslateOnly(Gf.Vec3d(*gripper_joint.GetLocalPos0Attr().Get()))
    sc_body_world = UsdGeom.Xformable(
        stage.GetPrimAtPath(base + "/sc_plug/sc_plug_link")
    ).ComputeLocalToWorldTransform(time)
    gripper_local1 = gripper_local0 * gripper_body0_world * sc_body_world.GetInverse()
    gripper_joint.GetLocalPos1Attr().Set(Gf.Vec3f(*gripper_local1.ExtractTranslation()))
    _set_quaternion(gripper_joint.GetLocalRot1Attr(), gripper_local1.ExtractRotationQuat())


def _aligned_topology() -> None:
    """Grip the SC body and align its geometry at the native robot pose.

    The downloaded articulation was authored for the LC end at the gripper.
    Swapping only joint targets leaves the SC semantic tip about 130 degrees
    from the fixed SC-port frame.  Preserve the stable LC-end cable joint and
    rotate the SC render/collision/tip frames within that gripped rigid body.
    This avoids teleporting the rope or changing its generalized coordinates.
    """
    lc_parent = stage.GetPrimAtPath(base + "/lc_plug")
    sc_parent = stage.GetPrimAtPath(base + "/sc_plug")
    for name in ("xformOp:translate", "xformOp:orient"):
        lc_attr = lc_parent.GetAttribute(name)
        sc_attr = sc_parent.GetAttribute(name)
        lc_value, sc_value = lc_attr.Get(), sc_attr.Get()
        lc_attr.Set(sc_value)
        sc_attr.Set(lc_value)

    swaps = (
        (base + "/Rope/fixedJoint", base + "/sc_plug/sc_plug_link"),
        (base + "/Rope/fixedJoint2", base + "/lc_plug/lc_plug_link"),
        (base + "/gripper_attach_joint", base + "/sc_plug/sc_plug_link"),
    )
    for joint_path, body_path in swaps:
        UsdPhysics.Joint(stage.GetPrimAtPath(joint_path)).GetBody1Rel().SetTargets([Sdf.Path(body_path)])

    # q_delta = inverse(native gripped SC body orientation) * desired SC body
    # orientation.  The desired orientation makes sc_tip coincide with the
    # fixed Isaac sc_port_base frame at board yaw zero.
    delta = Gf.Quatd(
        0.5687546808819202,
        Gf.Vec3d(-0.42054428140969796, 0.5679921325413939, -0.42076781925378465),
    )
    visual = UsdGeom.Xformable(stage.GetPrimAtPath(base + "/sc_plug/sc_plug_link/visual"))
    visual_orient = next(
        (op for op in visual.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
        None,
    )
    if visual_orient is None:
        visual.AddOrientOp().Set(delta)
    else:
        value = visual_orient.Get()
        old = Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*value.GetImaginary()))
        _set_quaternion(visual_orient.GetAttr(), delta * old)

    collisions = stage.GetPrimAtPath(base + "/sc_plug/sc_plug_link/collisions")
    for prim in Usd.PrimRange(collisions):
        if prim == collisions or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateContactOffsetAttr().Set(0.00001)
        physx_collision.CreateRestOffsetAttr().Set(0.0)
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                value = op.Get()
                rotated = Gf.Rotation(delta).TransformDir(Gf.Vec3d(*value))
                op.Set(type(value)(*rotated))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                value = op.Get()
                old = Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*value.GetImaginary()))
                _set_quaternion(op.GetAttr(), delta * old)

    tip_joint = UsdPhysics.Joint(stage.GetPrimAtPath(base + "/sc_plug/sc_tip_joint"))
    old_pos = tip_joint.GetLocalPos0Attr().Get()
    new_pos = Gf.Rotation(delta).TransformDir(Gf.Vec3d(*old_pos))
    tip_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*new_pos))
    old_rot_value = tip_joint.GetLocalRot0Attr().Get()
    old_rot = Gf.Quatd(float(old_rot_value.GetReal()), Gf.Vec3d(*old_rot_value.GetImaginary()))
    _set_quaternion(tip_joint.GetLocalRot0Attr(), delta * old_rot)


def _rotate_copied_sc_geometry(delta: Gf.Quatd) -> None:
    """Rotate copied SC geometry inside the stable gripped LC rigid body."""
    visual = UsdGeom.Xformable(stage.GetPrimAtPath(base + "/lc_plug/lc_plug_link/sc_visual"))
    visual_orient = next(
        (op for op in visual.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeOrient),
        None,
    )
    if visual_orient is None:
        visual.AddOrientOp().Set(delta)
    else:
        value = visual_orient.Get()
        old = Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*value.GetImaginary()))
        _set_quaternion(visual_orient.GetAttr(), delta * old)

    collisions = stage.GetPrimAtPath(base + "/lc_plug/lc_plug_link/sc_collisions")
    for prim in Usd.PrimRange(collisions):
        if prim == collisions or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        physx_collision.CreateContactOffsetAttr().Set(0.00001)
        physx_collision.CreateRestOffsetAttr().Set(0.0)
        xform = UsdGeom.Xformable(prim)
        for op in xform.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                value = op.Get()
                rotated = Gf.Rotation(delta).TransformDir(Gf.Vec3d(*value))
                op.Set(type(value)(*rotated))
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                value = op.Get()
                old = Gf.Quatd(float(value.GetReal()), Gf.Vec3d(*value.GetImaginary()))
                _set_quaternion(op.GetAttr(), delta * old)


def _print_joint_drives() -> None:
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        drive_rows = []
        for axis in ("angular", "rotX", "rotY", "rotZ"):
            drive = UsdPhysics.DriveAPI.Get(prim, axis)
            if drive:
                drive_rows.append(
                    (
                        axis,
                        drive.GetStiffnessAttr().Get(),
                        drive.GetDampingAttr().Get(),
                        drive.GetMaxForceAttr().Get(),
                    )
                )
        print(f"{prim.GetPath()} type={prim.GetTypeName()} drives={drive_rows}", flush=True)
    for prim in stage.Traverse():
        mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
        if mass_api and ("/Rope/Rope/link_" in str(prim.GetPath()) or "/plug/" in str(prim.GetPath())):
            print(f"{prim.GetPath()} mass={mass_api.GetMassAttr().Get()}", flush=True)


def _print_topology_transforms() -> None:
    time = Usd.TimeCode.Default()
    for path in (base + "/Rope/fixedJoint", base + "/Rope/fixedJoint2", base + "/gripper_attach_joint"):
        joint = UsdPhysics.Joint(stage.GetPrimAtPath(path))
        body0_path = joint.GetBody0Rel().GetTargets()[0]
        body1_path = joint.GetBody1Rel().GetTargets()[0]
        body0 = UsdGeom.Xformable(stage.GetPrimAtPath(body0_path)).ComputeLocalToWorldTransform(time)
        body1 = UsdGeom.Xformable(stage.GetPrimAtPath(body1_path)).ComputeLocalToWorldTransform(time)
        local0 = Gf.Matrix4d(1.0)
        local0.SetRotate(Gf.Rotation(joint.GetLocalRot0Attr().Get()))
        local0.SetTranslateOnly(Gf.Vec3d(*joint.GetLocalPos0Attr().Get()))
        local1 = Gf.Matrix4d(1.0)
        local1.SetRotate(Gf.Rotation(joint.GetLocalRot1Attr().Get()))
        local1.SetTranslateOnly(Gf.Vec3d(*joint.GetLocalPos1Attr().Get()))
        for convention, frame0, frame1 in (
            ("local_times_world", local0 * body0, local1 * body1),
            ("world_times_local", body0 * local0, body1 * local1),
        ):
            p0, p1 = frame0.ExtractTranslation(), frame1.ExtractTranslation()
            residual = (p0 - p1).GetLength()
            q0, q1 = frame0.ExtractRotationQuat(), frame1.ExtractRotationQuat()
            q0v = (q0.GetReal(), *q0.GetImaginary())
            q1v = (q1.GetReal(), *q1.GetImaginary())
            dot = abs(sum(float(a) * float(b) for a, b in zip(q0v, q1v)))
            q0n = math.sqrt(sum(float(v) ** 2 for v in q0v))
            q1n = math.sqrt(sum(float(v) ** 2 for v in q1v))
            angle = 2.0 * math.acos(max(-1.0, min(1.0, dot / max(q0n * q1n, 1.0e-12))))
            print(
                f"{path} {convention} body0={body0_path} body1={body1_path} "
                f"p0={tuple(p0)} p1={tuple(p1)} translation_residual_m={residual} "
                f"rotation_residual_rad={angle}",
                flush=True,
            )


def _apply_requested_collision_disables() -> list[str]:
    collision_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    disabled_paths = []
    for path in collision_paths:
        if any(fragment in path for fragment in args.disable_collision_substring):
            UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr().Set(False)
            disabled_paths.append(path)
    if args.disable_collision_substring:
        print(f"disabled_collisions={len(disabled_paths)}", flush=True)
    if args.collision_path_output is not None:
        args.collision_path_output.write_text("\n".join(collision_paths) + "\n", encoding="utf-8")
    return collision_paths


def _apply_requested_deinstancing() -> None:
    if not args.deinstance_substring:
        return
    changed = []
    for prim in list(stage.Traverse()):
        path = str(prim.GetPath())
        if prim.IsInstance() and any(fragment in path for fragment in args.deinstance_substring):
            prim.SetInstanceable(False)
            changed.append(path)
    print(f"deinstanced_prims={changed}", flush=True)


def _apply_arm_joint_limits() -> None:
    if not args.expand_arm_joint_limits:
        return
    arm_names = {
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    }
    changed = []
    for prim in stage.Traverse():
        if prim.GetName() not in arm_names or not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        joint.CreateLowerLimitAttr().Set(-360.0)
        joint.CreateUpperLimitAttr().Set(360.0)
        changed.append(str(prim.GetPath()))
    if len(changed) != len(arm_names):
        raise RuntimeError(f"Expected {len(arm_names)} arm joints, expanded {len(changed)}: {changed}")
    print(f"expanded_arm_joint_limits={changed}", flush=True)


def _apply_source_reversed_collision_contract() -> None:
    """Mirror the collision edits declared by sfp_sc_cable_reversed/model.sdf.

    Gazebo removes the endpoint-0 and connection-0 colliders because they
    overlap the gripper palm. It replaces link_1's 48 mm cylinder with a 36 mm
    cylinder shifted 6 mm away from that palm. The SDF importer collapsed each
    rope rigid body and its collider onto one USD prim, so create a child
    collider rather than moving the link_1 rigid body itself.
    """
    if args.mode != "reversed_topology" or not args.apply_source_reversed_collision_contract:
        return

    link0 = stage.GetPrimAtPath(base + "/Rope/Rope/link_0")
    link1 = stage.GetPrimAtPath(base + "/Rope/Rope/link_1")
    if not link0.IsValid() or not link1.IsValid():
        raise RuntimeError("Expected imported rope links 0 and 1 for reversed collision contract")
    for prim in (link0, link1):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"Expected collision API on imported rope body {prim.GetPath()}")
        collision = UsdPhysics.CollisionAPI(prim)
        collision.CreateCollisionEnabledAttr().Set(False)

    replacement_path = link1.GetPath().AppendChild("source_reversed_link1_collision")
    replacement = UsdGeom.Capsule.Define(stage, replacement_path)
    replacement.CreateAxisAttr().Set(UsdGeom.Tokens.z)
    replacement.CreateRadiusAttr().Set(0.002)
    replacement.CreateHeightAttr().Set(0.036)
    replacement.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.006))
    UsdPhysics.CollisionAPI.Apply(replacement.GetPrim()).CreateCollisionEnabledAttr().Set(True)
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(replacement.GetPrim())
    physx_collision.CreateContactOffsetAttr().Set(0.00001)
    physx_collision.CreateRestOffsetAttr().Set(0.0)
    print(
        "source_reversed_collision_contract="
        f"disabled:{link0.GetPath()},{link1.GetPath()} replacement:{replacement_path}",
        flush=True,
    )


if args.mode == "aligned_topology":
    _aligned_topology()
    layer.Save()
    print(output)
    simulation_app.close()
    raise SystemExit(0)

if args.mode == "reversed_topology":
    _reversed_topology()
    _apply_source_reversed_collision_contract()
    _apply_arm_joint_limits()
    _apply_requested_deinstancing()
    collision_paths = _apply_requested_collision_disables()
    layer.Save()
    if args.print_joint_drives:
        _print_joint_drives()
    if args.print_topology_transforms:
        _print_topology_transforms()
    print(output)
    simulation_app.close()
    raise SystemExit(0)

# Present the SC connector at the gripped LC rigid body without changing the
# articulation topology. The SC collision hierarchy consists of simple box
# and cylinder primitives and replaces the incompatible LC/SFP collision.
for source_name, target_name in (
    ("visual", "sc_visual"),
    ("collisions", "sc_collisions"),
):
    Sdf.CopySpec(
        layer,
        Sdf.Path(base + f"/sc_plug/sc_plug_link/{source_name}"),
        layer,
        Sdf.Path(base + f"/lc_plug/lc_plug_link/{target_name}"),
    )
for path in (
    base + "/lc_plug/lc_plug_link/visual",
    base + "/lc_plug/lc_plug_link/collisions",
    base + "/sfp_module/sfp_module_link/visual",
    base + "/sfp_module/sfp_module_link/collisions",
    base + "/sc_plug/sc_plug_link/visual",
):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive(False)

if args.mode == "stable_proxy_aligned":
    # Keep the distributed cable articulation unchanged. Rotate only the SC
    # render/collision proxy so its semantic tip has the official SC-port
    # keying orientation at the native robot pose. Control code uses the same
    # delta to derive the semantic tip position/orientation from lc_plug_link.
    delta_values = args.proxy_delta_wxyz or (
        0.5687546808819202,
        -0.42054428140969796,
        0.5679921325413939,
        -0.42076781925378465,
    )
    _rotate_copied_sc_geometry(
        Gf.Quatd(
            float(delta_values[0]),
            Gf.Vec3d(*[float(v) for v in delta_values[1:4]]),
        )
    )

layer.Save()
collision_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
if args.stage_path_output is not None:
    args.stage_path_output.write_text(
        "\n".join(str(prim.GetPath()) for prim in stage.Traverse()) + "\n", encoding="utf-8"
    )
collision_paths = _apply_requested_collision_disables()
layer.Save()
if args.print_collision_paths:
    for path in collision_paths:
        print(path, flush=True)
print(output)
simulation_app.close()
