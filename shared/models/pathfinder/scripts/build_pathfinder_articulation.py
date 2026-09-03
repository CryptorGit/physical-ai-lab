"""Pathfinder素材USDからデバッグ用Isaac Sim Articulationを生成する。"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher


# Kitがfd 1を差し替える前に複製し、生成レポートを確実に端末へ返す。
REPORT_FD = os.dup(1)
PATHFINDER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PATHFINDER_DIR / "source" / "pathfinder_visual_collision.usdc"
DEFAULT_OUTPUT = PATHFINDER_DIR / "usd" / "pathfinder_articulation.usd"

# 角度と角速度は、USDへdegree / degree-per-secondとして書き込む。
JOINT_SETTINGS = {
    "waist": {"type": "fixed", "axis": "Z"},
    "neck": {"type": "fixed", "axis": "Z"},
    "wrist_L": {"type": "fixed", "axis": "Z"},
    "wrist_R": {"type": "fixed", "axis": "Z"},
    "shoulder_L": {"type": "revolute", "axis": "Y", "limits": (-90.0, 90.0)},
    "shoulder_R": {"type": "revolute", "axis": "Y", "limits": (-90.0, 90.0)},
    "elbow_L": {"type": "revolute", "axis": "Z", "limits": (-10.0, 135.0)},
    "elbow_R": {"type": "revolute", "axis": "Z", "limits": (-10.0, 135.0)},
    "hip_L": {"type": "revolute", "axis": "Y", "limits": (-60.0, 50.0)},
    "hip_R": {"type": "revolute", "axis": "Y", "limits": (-60.0, 50.0)},
    "knee_L": {"type": "revolute", "axis": "Z", "limits": (-10.0, 135.0)},
    "knee_R": {"type": "revolute", "axis": "Z", "limits": (-10.0, 135.0)},
    "ankle_L": {"type": "revolute", "axis": "Z", "limits": (-35.0, 35.0)},
    "ankle_R": {"type": "revolute", "axis": "Z", "limits": (-35.0, 35.0)},
}

DEFAULT_REVOLUTE_DRIVE = {
    "max_force": 150.0,
    "max_velocity": 360.0,
    "stiffness": 40.0,
    "damping": 4.0,
}

# 合計70.0 kg。
LINK_MASSES_KG = {
    "pelvis": 10.0, "torso": 20.0, "head": 5.0,
    "upper_arm_L": 2.5, "upper_arm_R": 2.5,
    "forearm_L": 1.8, "forearm_R": 1.8,
    "hand_L": 0.7, "hand_R": 0.7,
    "thigh_L": 7.0, "thigh_R": 7.0,
    "shin_L": 4.5, "shin_R": 4.5,
    "foot_L": 1.0, "foot_R": 1.0,
}


@dataclass(frozen=True)
class JointGuide:
    name: str
    prim: object
    parent_link: str
    child_link: str
    source_type: str
    source_axis: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    # Asset生成には描画もCUDAも不要。
    args.viz = "none"
    args.device = "cpu"
    return args


def _custom_value(prim, name: str):
    """通常名とBlenderのuserProperties名前空間の両方に対応する。"""
    for candidate in (name, f"userProperties:{name}"):
        attribute = prim.GetAttribute(candidate)
        if attribute and attribute.HasAuthoredValueOpinion():
            return attribute.Get()
    raise RuntimeError(f"{prim.GetPath()}: custom attribute {name!r} is missing")


def _link_name(source_name: str) -> str:
    return source_name.removeprefix("VIS_")


def _float_quaternion(quaternion, Gf):
    imaginary = quaternion.GetImaginary()
    return Gf.Quatf(
        float(quaternion.GetReal()),
        Gf.Vec3f(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    )


def _joint_local_pose(joint_world, body_world, Gf):
    relative = joint_world * body_world.GetInverse()
    transform = Gf.Transform(relative)
    translation = transform.GetTranslation()
    return (
        Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2])),
        _float_quaternion(transform.GetRotation().GetQuat(), Gf),
    )


def _discover_source(source_stage):
    links = {}
    guides = {}
    for prim in source_stage.Traverse():
        if prim.GetTypeName() == "Xform" and prim.GetName().startswith("VIS_"):
            links[_link_name(prim.GetName())] = prim
        elif prim.GetTypeName() == "Xform" and prim.GetName().startswith("JNT_"):
            name = prim.GetName().removeprefix("JNT_")
            guides[name] = JointGuide(
                name=name,
                prim=prim,
                parent_link=_link_name(str(_custom_value(prim, "parent_link"))),
                child_link=_link_name(str(_custom_value(prim, "child_link"))),
                source_type=str(_custom_value(prim, "joint_type")),
                source_axis=str(_custom_value(prim, "joint_axis")),
            )
    return links, guides


def _validate_input(source_stage, links, guides, UsdGeom) -> None:
    default_prim = source_stage.GetDefaultPrim()
    if not default_prim or str(default_prim.GetPath()) != "/Pathfinder":
        raise RuntimeError("Expected the source Default Prim to be /Pathfinder")
    if UsdGeom.GetStageUpAxis(source_stage) != UsdGeom.Tokens.z:
        raise RuntimeError("Expected a Z-up source stage")
    if abs(UsdGeom.GetStageMetersPerUnit(source_stage) - 1.0) > 1.0e-9:
        raise RuntimeError("Expected source metersPerUnit=1.0")
    if set(links) != set(LINK_MASSES_KG):
        raise RuntimeError(
            f"Link mismatch. Missing={sorted(set(LINK_MASSES_KG) - set(links))}, "
            f"extra={sorted(set(links) - set(LINK_MASSES_KG))}"
        )
    if set(guides) != set(JOINT_SETTINGS):
        raise RuntimeError(
            f"Guide mismatch. Missing={sorted(set(JOINT_SETTINGS) - set(guides))}, "
            f"extra={sorted(set(guides) - set(JOINT_SETTINGS))}"
        )


def build(source_path: Path, output_path: Path) -> None:
    # PhysxSchemaはSimulationApp起動後に登録される。
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if source_path == output_path:
        raise ValueError("Source and output paths must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_stage = Usd.Stage.Open(str(source_path), Usd.Stage.LoadAll)
    if source_stage is None:
        raise RuntimeError(f"Could not open source USD: {source_path}")
    source_layer = source_stage.Flatten()
    links, guides = _discover_source(source_stage)
    _validate_input(source_stage, links, guides, UsdGeom)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stage = Usd.Stage.CreateNew(str(output_path))
    UsdGeom.SetStageUpAxis(output_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(output_stage, 1.0)
    output_stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())

    root = UsdGeom.Xform.Define(output_stage, "/Pathfinder").GetPrim()
    output_stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    UsdGeom.Scope.Define(output_stage, "/Pathfinder/Links")
    UsdGeom.Scope.Define(output_stage, "/Pathfinder/Joints")

    # 元の絶対Material Bindingパスを有効なまま保つ。
    material_path = Sdf.Path("/Pathfinder/_materials")
    if source_layer.GetPrimAtPath(material_path):
        Sdf.CopySpec(
            source_layer, material_path, output_stage.GetRootLayer(), material_path
        )

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    body_paths = {}
    body_world = {}
    collision_mesh_count = 0

    # 入れ子Rigid Bodyを避けるため、全リンクをLinks直下へフラット化する。
    for link_name, source_link in sorted(links.items()):
        body_path = Sdf.Path(f"/Pathfinder/Links/{link_name}")
        body_paths[link_name] = body_path
        world_matrix = xform_cache.GetLocalToWorldTransform(source_link)
        body_world[link_name] = world_matrix

        body_xform = UsdGeom.Xform.Define(output_stage, body_path)
        body_xform.AddTransformOp().Set(world_matrix)
        body_prim = body_xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body_prim)
        UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(LINK_MASSES_KG[link_name])
        # inertia/COMは未設定にし、PhysXにCOL形状と質量から推定させる。

        visual_scope = UsdGeom.Scope.Define(
            output_stage, body_path.AppendChild("Visual")
        )
        collision_scope = UsdGeom.Scope.Define(
            output_stage, body_path.AppendChild("Collisions")
        )
        for child in source_link.GetChildren():
            child_name = child.GetName()
            if child.GetTypeName() == "Mesh" and child_name.startswith("VIS_"):
                destination = visual_scope.GetPath().AppendChild(child_name)
                Sdf.CopySpec(
                    source_layer, child.GetPath(), output_stage.GetRootLayer(), destination
                )
            elif child.GetTypeName() == "Xform" and child_name.startswith("COL_"):
                destination = collision_scope.GetPath().AppendChild(child_name)
                Sdf.CopySpec(
                    source_layer, child.GetPath(), output_stage.GetRootLayer(), destination
                )

        collision_meshes = [
            prim
            for prim in Usd.PrimRange(collision_scope.GetPrim())
            if prim.GetTypeName() == "Mesh" and prim.GetName().startswith("COL_")
        ]
        if not collision_meshes:
            raise RuntimeError(f"No COL mesh found for link {link_name}")
        for collision_mesh in collision_meshes:
            UsdPhysics.CollisionAPI.Apply(collision_mesh)
            UsdPhysics.MeshCollisionAPI.Apply(
                collision_mesh
            ).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            PhysxSchema.PhysxCollisionAPI.Apply(collision_mesh)
            collision_mesh_count += 1

    joint_rows = []
    for joint_name, guide in sorted(guides.items()):
        setting = JOINT_SETTINGS[joint_name]
        joint_path = Sdf.Path(f"/Pathfinder/Joints/{joint_name}")
        parent_path = body_paths[guide.parent_link]
        child_path = body_paths[guide.child_link]
        joint_world = xform_cache.GetLocalToWorldTransform(guide.prim)
        local_pos0, local_rot0 = _joint_local_pose(
            joint_world, body_world[guide.parent_link], Gf
        )
        local_pos1, local_rot1 = _joint_local_pose(
            joint_world, body_world[guide.child_link], Gf
        )

        if setting["type"] == "fixed":
            joint = UsdPhysics.FixedJoint.Define(output_stage, joint_path)
            limits = None
        else:
            joint = UsdPhysics.RevoluteJoint.Define(output_stage, joint_path)
            joint.CreateAxisAttr(setting["axis"])
            lower, upper = setting["limits"]
            joint.CreateLowerLimitAttr(lower)
            joint.CreateUpperLimitAttr(upper)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateTargetPositionAttr(0.0)
            drive.CreateStiffnessAttr(DEFAULT_REVOLUTE_DRIVE["stiffness"])
            drive.CreateDampingAttr(DEFAULT_REVOLUTE_DRIVE["damping"])
            drive.CreateMaxForceAttr(DEFAULT_REVOLUTE_DRIVE["max_force"])
            PhysxSchema.PhysxJointAPI.Apply(
                joint.GetPrim()
            ).CreateMaxJointVelocityAttr(DEFAULT_REVOLUTE_DRIVE["max_velocity"])
            limits = (lower, upper)

        joint.CreateBody0Rel().SetTargets([parent_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint.CreateLocalPos0Attr(local_pos0)
        joint.CreateLocalRot0Attr(local_rot0)
        joint.CreateLocalPos1Attr(local_pos1)
        joint.CreateLocalRot1Attr(local_rot1)
        joint.CreateCollisionEnabledAttr(False)

        # 隣接bodyだけを除外し、非隣接body間のcollisionは有効なままにする。
        UsdPhysics.FilteredPairsAPI.Apply(
            output_stage.GetPrimAtPath(parent_path)
        ).CreateFilteredPairsRel().AddTarget(child_path)
        joint_rows.append(
            (joint_name, setting["type"], guide, setting["axis"], limits)
        )

    output_stage.GetRootLayer().Save()

    report_lines = ["", "Rigid Bodies:"]
    for link_name in sorted(body_paths):
        marker = " (root)" if link_name == "pelvis" else ""
        report_lines.append(
            f"  {body_paths[link_name]}: "
            f"{LINK_MASSES_KG[link_name]:.3f} kg{marker}"
        )
    report_lines.extend(("", "Joints:"))
    for name, joint_type, guide, axis, limits in joint_rows:
        limit_text = (
            "fixed"
            if limits is None
            else f"[{limits[0]:.1f}, {limits[1]:.1f}] deg"
        )
        report_lines.append(
            f"  {name}: {joint_type}, {guide.parent_link} -> {guide.child_link}, "
            f"axis={axis}, range={limit_text} "
            f"(source: type={guide.source_type}, axis={guide.source_axis})"
        )
    report_lines.extend(
        (
            "",
            f"Collision proxies: {collision_mesh_count}",
            f"Total mass: {sum(LINK_MASSES_KG.values()):.3f} kg",
            f"Output: {output_path}",
        )
    )
    os.write(REPORT_FD, ("\n".join(report_lines) + "\n").encode("utf-8"))


def main() -> None:
    args = parse_args()
    # 元USDを変更せず、新規Stageのみを保存する。
    app_launcher = AppLauncher(args)
    app = app_launcher.app
    try:
        build(args.source, args.output)
    finally:
        app.close()


if __name__ == "__main__":
    main()
