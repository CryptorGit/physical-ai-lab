"""Audit the official Unitree Qmini repository and write immutable manifests."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = EXP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qmini_population_bwm.qmini_asset import load_qmini_contract, validate_qmini_contract


OFFICIAL_REPOSITORY = "https://github.com/unitreerobotics/Qmini"
EXPECTED_COMMIT = "f6f3fef723f8bb434f9d2679dfb6053b0aca93a8"
ROBOTAMER_REPOSITORY = "https://github.com/vsislab/RoboTamer4Qmini"
ROBOTAMER_SDK_REPOSITORY = "https://github.com/vsislab/RoboTamerSdk4Qmini"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"
    return result.stdout.strip()


def hash_if_present(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return {"path": relative, "sha256": "UNKNOWN", "bytes": "UNKNOWN"}
    return {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}


def audit_source(source_root: Path) -> dict[str, Any]:
    urdf = source_root / "urdf" / "Qmini.urdf"
    if not urdf.exists():
        raise FileNotFoundError(f"official Qmini URDF not found: {urdf}")
    contract = load_qmini_contract(urdf)
    failures = validate_qmini_contract(contract)
    mesh_hashes = [
        hash_if_present(source_root, f"urdf/meshes/{path.name}")
        for path in sorted((source_root / "urdf" / "meshes").glob("*.STL"))
    ]
    known_files = [
        "README.md",
        "LICENSE",
        "urdf/Qmini.urdf",
        "Qmini_DIY.pdf",
        "STEP_file/Qmini_v1_0.STEP.zip",
        "STEP_file/Qmini_v2_0.STEP.zip",
    ]
    source_files = [hash_if_present(source_root, relative) for relative in known_files]
    source_files.extend(mesh_hashes)
    readme = (source_root / "README.md").read_text(encoding="utf-8", errors="replace") if (source_root / "README.md").exists() else ""
    return {
        "experiment": "exp_015_qmini_population_bodily_world_model",
        "source_type": "UNITREE_OFFICIAL_QMINI",
        "repository_url": OFFICIAL_REPOSITORY,
        "commit": git_value(source_root, "rev-parse", "HEAD"),
        "commit_date": git_value(source_root, "show", "-s", "--format=%cI", "HEAD"),
        "commit_subject": git_value(source_root, "show", "-s", "--format=%s", "HEAD"),
        "expected_audited_commit": EXPECTED_COMMIT,
        "commit_matches_expected": git_value(source_root, "rev-parse", "HEAD") == EXPECTED_COMMIT,
        "source_license": "CC BY-NC-SA 4.0",
        "files": source_files,
        "urdf_sha256": sha256(urdf),
        "qmini_contract_status": "PASS" if not failures else "FAIL",
        "qmini_contract_failures": failures,
        "mechanical_contract": {
            "motor_count_documented": 11,
            "locomotion_motor_count_documented": 10,
            "neck_motor_documented_as": "RESERVED_FOR_CUSTOM_EXPANSIONS",
            "default_reference_board": "Raspberry Pi 4 Model B",
            "imu_documented": "GY-91 MPU9250+BMP280 10DOF",
            "motor_model_documented": "Unitree 8010",
            "exact_joint_count_in_current_urdf": len(contract.joints),
            "exact_joint_order_in_current_urdf": list(contract.joint_names),
            "neck_joint_in_current_urdf": None,
            "transmission_count_in_current_urdf": contract.transmission_count,
            "foot_collision_links": list(contract.foot_collision_links),
            "links_without_collision_geometry": list(contract.commented_collision_links),
            "step_files": [
                "STEP_file/Qmini_v1_0.STEP.zip",
                "STEP_file/Qmini_v2_0.STEP.zip",
            ],
            "bom_and_diy_source": "Qmini_DIY.pdf",
        },
        "source_audit_findings": [
            "Current official URDF has 10 revolute joints and no neck joint.",
            "README/DIY material documents 11 physical 8010 motors, with one neck motor reserved for expansion.",
            "Current official URDF has no transmission tags or actuator/controller definitions.",
            "Current official URDF contains no joint dynamics tags; official PD values are UNKNOWN.",
            "Current official URDF collision meshes are absent for both hip_yaw and hip_roll link pairs.",
            "The official source does not publish nominal 8010 torque, velocity, reduction, or gear-ratio values.",
        ],
        "roboTamer_references": {
            "training_repository": ROBOTAMER_REPOSITORY,
            "training_main_commit": "1aed648115fbaf49b86751edd293d94dbb2b3fc5",
            "deployment_repository": ROBOTAMER_SDK_REPOSITORY,
            "deployment_main_commit": "f4cfb4c96d8649480d0cc7da63a10fb9ff302589",
            "simulator": "Isaac Gym Preview 1.0rc3+",
            "training_framework_status": "README_SAYS_NOT_MAINTAINED",
            "reference_policy_format": "ONNX",
            "reference_deployment": "C++ ONNX Runtime on Linux edge",
            "reference_pd": "5 joint-pattern gains explicitly labelled ROBOTAMER_REFERENCE",
            "reference_schema": "12 actions / 43 observations per stack / 3 stacks = 129",
            "compatibility": "NOT_DIRECTLY_TRANSFERABLE_TO_CURRENT_OFFICIAL_URDF",
        },
        "readme_presence_checks": {
            "mentions_11_motors": "11" in readme and "8010" in readme,
            "mentions_10_locomotion": "10" in readme and "locomotion" in readme.lower(),
            "mentions_RoboTamer4Qmini": "RoboTamer4Qmini" in readme,
        },
    }


def write_manifests(source_root: Path, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = audit_source(source_root)
    urdf = source_root / "urdf" / "Qmini.urdf"
    contract = load_qmini_contract(urdf)
    physics = {
        "experiment": "exp_015_qmini_population_bodily_world_model",
        "status": "PASS" if not validate_qmini_contract(contract) else "INVALID_QMINI_PHYSICS_CONTRACT",
        "source": {
            "repository_url": OFFICIAL_REPOSITORY,
            "commit": source_manifest["commit"],
            "urdf": "urdf/Qmini.urdf",
            "urdf_sha256": sha256(urdf),
            "source_of_truth": True,
        },
        "root": {
            "link": contract.root_link,
            "orientation": "UNKNOWN",
            "simulator_root_height_m": 0.45,
            "simulator_root_height_source": "ROBOTAMER_REFERENCE_PROVISIONAL",
        },
        "joints": [asdict(joint) for joint in contract.joints],
        "joint_order": list(contract.joint_names),
        "joint_count": len(contract.joints),
        "links": [asdict(link) for link in contract.links],
        "visual_meshes": {
            link.name: list(link.visual_meshes)
            for link in contract.links
            if link.visual_meshes
        },
        "collision_meshes": {
            link.name: list(link.collision_meshes)
            for link in contract.links
            if link.collision_meshes
        },
        "foot_collision_links": list(contract.foot_collision_links),
        "links_without_collision_geometry": list(contract.commented_collision_links),
        "transmission_count": contract.transmission_count,
        "actuator_assignment": {
            "source": "UNITREE_OFFICIAL_URDF",
            "joint_effort_limits_are": "URDF_JOINT_LIMITS_NOT_MOTOR_NOMINAL",
            "joint_velocity_limits_are": "URDF_JOINT_LIMITS_NOT_MOTOR_NOMINAL",
            "torque_limit": "UNKNOWN_MOTOR_NOMINAL",
            "velocity_limit": "UNKNOWN_MOTOR_NOMINAL",
            "reduction": "UNKNOWN",
            "motor_model": "UNKNOWN",
        },
        "simulator": {
            "isaac_lab_version": "v3.0.0-beta2.patch1",
            "isaac_lab_commit": "ffff603eafc6b74264a5261cc0183d6a65390d78",
            "simulation_dt_s": 0.001,
            "simulation_dt_source": "ROBOTAMER_REFERENCE",
            "control_dt_s": 0.015,
            "control_dt_source": "ROBOTAMER_REFERENCE",
            "gravity_mps2": [0.0, 0.0, -9.81],
            "friction_baseline": 1.0,
            "friction_baseline_source": "ROBOTAMER_REFERENCE_PROVISIONAL",
            "official_qmini_friction": "UNKNOWN",
            "official_qmini_root_orientation": "UNKNOWN",
        },
        "default_pose": {
            "values": contract.default_joint_pose_midpoint,
            "source": "DERIVED_FROM_OFFICIAL_URDF_LIMIT_MIDPOINT",
            "official_named_standing_pose": "UNKNOWN",
        },
        "isaaclab_import": {
            "version": "v3.0.0-beta2.patch1",
            "observed_imported_joint_order": [
                "tn__LLjoint1_ZC", "tn__RLjoint1_ZC",
                "tn__LLjoint2_ZC", "tn__RLjoint2_ZC",
                "tn__LLjoint3_ZC", "tn__RLjoint3_ZC",
                "tn__LLjoint4_ZC", "tn__RLjoint4_ZC",
                "tn__LLjoint5_ZC", "tn__RLjoint5_ZC"
            ],
            "official_to_imported_index": [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
            "control_schema_order": list(contract.joint_names),
            "mapping_required": True,
            "asset_import_smoke_status": "RECORDED_SEPARATELY",
        },
        "mesh_hashes": [
            hash_if_present(source_root, f"urdf/meshes/{path.name}")
            for path in sorted((source_root / "urdf" / "meshes").glob("*.STL"))
        ],
        "unknowns": [
            "8010 nominal continuous torque",
            "8010 nominal velocity",
            "gear/reduction",
            "motor model",
            "official PD gains",
            "official friction baseline",
            "official root orientation convention",
            "official simulator dt/control dt",
        ],
    }
    source_path = output_dir / "qmini_source.json"
    physics_path = output_dir / "physics_contract.json"
    source_path.write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    physics_path.write_text(json.dumps(physics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return source_path, physics_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=EXP_ROOT / "manifests")
    args = parser.parse_args()
    source_path, physics_path = write_manifests(args.source_root.resolve(), args.output_dir.resolve())
    print(json.dumps({"qmini_source": str(source_path), "physics_contract": str(physics_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
