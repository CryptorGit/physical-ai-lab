"""Validate the Qmini contract, optionally by launching a headless Isaac Lab smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[2]
SRC_ROOT = EXP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qmini_population_bwm.qmini_asset import (
    OFFICIAL_URDF_PATH,
    QMINI_JOINT_ORDER,
    build_qmini_articulation_cfg,
    canonicalize_isaaclab_joint_name,
    official_to_isaaclab_joint_indices,
    load_qmini_contract,
    official_link_mass_total,
    validate_qmini_contract,
)


def contract_only() -> int:
    contract = load_qmini_contract()
    failures = validate_qmini_contract(contract)
    report = {
        "status": "PASS" if not failures else "FAIL",
        "urdf": str(OFFICIAL_URDF_PATH),
        "joint_count": len(contract.joints),
        "joint_names": list(contract.joint_names),
        "link_count": len(contract.links),
        "total_link_mass_kg": official_link_mass_total(contract),
        "foot_collision_links": list(contract.foot_collision_links),
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 2


def isaac_smoke(args: argparse.Namespace) -> int:
    # AppLauncher must be imported and instantiated before Isaac Lab modules
    # that depend on the simulation application.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    report_path = REPO_ROOT / "results" / "exp_015_qmini_population_bodily_world_model" / "isaaclab_asset_import.json"
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        from isaaclab.sim import SimulationContext
        import numpy as np
        from pxr import Usd, UsdPhysics

        sim = SimulationContext(
            sim_utils.SimulationCfg(
                dt=0.001,
                device=args.device,
                render_interval=1,
            )
        )
        sim.set_camera_view(eye=(1.5, 1.5, 1.0), target=(0.0, 0.0, 0.3))
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        cfg = build_qmini_articulation_cfg(prim_path="/World/Qmini")
        robot = Articulation(cfg)
        stage = sim_utils.get_current_stage()
        rigid_body_paths = [
            prim.GetPath().pathString
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Qmini"))
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        source_contract = load_qmini_contract()
        source_links = {
            link.name: link
            for link in source_contract.links
            if link.inertial is not None
        }

        def as_float_list(value: object) -> list[float] | None:
            if value is None:
                return None
            try:
                return [float(item) for item in value]  # type: ignore[arg-type]
            except TypeError:
                return None

        imported_mass_inertia: dict[str, dict[str, object]] = {}
        for body_name in source_links:
            body_path = next(
                (path for path in rigid_body_paths if path.endswith(f"/{body_name}")),
                None,
            )
            if body_path is None:
                imported_mass_inertia[body_name] = {
                    "path": None,
                    "mass_kg": None,
                    "diagonal_inertia_kg_m2": None,
                    "mass_match": False,
                    "inertia_match": False,
                }
                continue
            prim = stage.GetPrimAtPath(body_path)
            mass_value = prim.GetAttribute("physics:mass").Get()
            inertia_value = prim.GetAttribute("physics:diagonalInertia").Get()
            mass = None if mass_value is None else float(mass_value)
            diagonal = as_float_list(inertia_value)
            source_inertial = source_links[body_name].inertial
            assert source_inertial is not None
            source_tensor = np.array([
                [source_inertial.ixx, source_inertial.ixy, source_inertial.ixz],
                [source_inertial.ixy, source_inertial.iyy, source_inertial.iyz],
                [source_inertial.ixz, source_inertial.iyz, source_inertial.izz],
            ], dtype=float)
            source_principal = sorted(float(value) for value in np.linalg.eigvalsh(source_tensor))
            source_diagonal = [
                source_inertial.ixx,
                source_inertial.iyy,
                source_inertial.izz,
            ]
            imported_mass_inertia[body_name] = {
                "path": body_path,
                "mass_kg": mass,
                "diagonal_inertia_kg_m2": diagonal,
                "source_mass_kg": source_inertial.mass,
                "source_inertia_tensor_kg_m2": source_tensor.tolist(),
                "source_diagonal_inertia_kg_m2": source_diagonal,
                "source_principal_inertia_kg_m2": source_principal,
                "mass_match": mass is not None and abs(mass - source_inertial.mass) <= 1e-5,
                "inertia_match": diagonal is not None
                and len(diagonal) == 3
                and all(
                    abs(a - b) <= 1e-5
                    for a, b in zip(sorted(diagonal), source_principal, strict=True)
                ),
            }
        foot_paths = {
            "LL_ankle": "/World/Qmini/Geometry/base_link/LL_hip_yaw/LL_hip_roll/LL_hip_pitch/LL_knee/LL_ankle",
            "RL_ankle": "/World/Qmini/Geometry/base_link/RL_hip_yaw/RL_hip_roll/RL_hip_pitch/RL_knee/RL_ankle",
        }
        # Apply the reporter to the exact converted rigid bodies before the
        # contact sensors initialize. The asset cfg also requests this.
        for path in foot_paths.values():
            sim_utils.schemas.activate_contact_sensors(path)
        foot_sensors = {
            name: ContactSensor(
                ContactSensorCfg(
                    prim_path=path,
                    update_period=0.0,
                    force_threshold=0.1,
                    track_air_time=True,
                    debug_vis=False,
                )
            )
            for name, path in foot_paths.items()
        }
        sim.reset()
        robot.reset()
        for sensor in foot_sensors.values():
            sensor.reset()
        if args.root_height is not None:
            root_pose = robot.data.default_root_pose.torch.clone()
            root_velocity = robot.data.default_root_vel.torch.clone()
            root_pose[:, 2] = args.root_height
            root_velocity.zero_()
            robot.write_root_pose_to_sim(root_pose)
            robot.write_root_velocity_to_sim(root_velocity)
        default_pose = robot.data.default_joint_pos.clone()
        root_min_z = float("inf")
        foot_force_max = [0.0, 0.0]
        for _ in range(args.steps):
            robot.set_joint_position_target(default_pose)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())
            for sensor in foot_sensors.values():
                sensor.update(sim.get_physics_dt(), force_recompute=True)
            root_min_z = min(root_min_z, float(robot.data.root_pos_w[0, 2].item()))
            for index, foot_name in enumerate(("LL_ankle", "RL_ankle")):
                contact_force = foot_sensors[foot_name].data.net_forces_w.torch[0]
                foot_force_max[index] = max(
                    foot_force_max[index],
                    float(contact_force.norm(dim=-1).max().item()),
                )
        imported_names = list(robot.joint_names)
        canonical_names = [
            canonicalize_isaaclab_joint_name(name) for name in imported_names
        ]
        official_to_imported = official_to_isaaclab_joint_indices(imported_names)
        report = {
            "status": "PASS",
            "joint_count": robot.num_joints,
            "joint_names_imported": imported_names,
            "joint_names_canonicalized": canonical_names,
            "official_joint_order": list(QMINI_JOINT_ORDER),
            "official_to_imported_index": list(official_to_imported),
            "body_count": robot.num_bodies,
            "body_names": list(robot.body_names),
            "mass_inertia": imported_mass_inertia,
            "mass_inertia_match": all(
                item["mass_match"] and item["inertia_match"]
                for item in imported_mass_inertia.values()
            ),
            "simulation_dt_s": sim.get_physics_dt(),
            "root_state_finite": bool(robot.data.root_state_w.isfinite().all().item()),
            "root_min_z": root_min_z,
            "contact_sensor_initialized": {
                name: sensor.is_initialized for name, sensor in foot_sensors.items()
            },
            "contact_sensor_body_names": {
                name: list(sensor.body_names or []) for name, sensor in foot_sensors.items()
            },
            "contact_sensor_paths": foot_paths,
            "rigid_body_paths": rigid_body_paths,
            "foot_contact_force_norm_max_n": foot_force_max,
            "contact_threshold_n": 0.1,
            "root_height_test_m": args.root_height,
            "root_position_w": robot.data.root_pos_w[0].detach().cpu().tolist(),
            "root_orientation_w": robot.data.root_state_w[0, 3:7].detach().cpu().tolist(),
            "default_joint_position_imported": robot.data.default_joint_pos[0].detach().cpu().tolist(),
            "foot_position_w": {
                name: robot.data.body_pos_w[0, robot.body_names.index(name)].detach().cpu().tolist()
                for name in ("LL_ankle", "RL_ankle")
            },
        }
        report["joint_order_match"] = (
            [
                report["joint_names_canonicalized"][index]
                for index in official_to_imported
            ]
            == list(QMINI_JOINT_ORDER)
        )
        report["contact_sensor_match"] = all(
            report["contact_sensor_initialized"].values()
        ) and all(
            report["contact_sensor_body_names"][name] == [name]
            for name in ("LL_ankle", "RL_ankle")
        )
        report["status"] = "PASS" if (
            report["root_state_finite"]
            and report["joint_order_match"]
            and report["contact_sensor_match"]
            and report["mass_inertia_match"]
        ) else "FAIL"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), file=sys.__stderr__, flush=True)
        return 0 if report["status"] == "PASS" else 3
    except Exception as exc:
        report = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), file=sys.__stderr__, flush=True)
        return 3
    finally:
        simulation_app.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac", action="store_true", help="launch the Isaac Lab smoke test")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--root-height", type=float)
    if "--isaac" in sys.argv:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.isaac:
        return contract_only()
    return isaac_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
