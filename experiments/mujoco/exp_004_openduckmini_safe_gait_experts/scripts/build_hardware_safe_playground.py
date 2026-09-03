"""Build an immutable OpenDuckMini playground using the measured safe contract.

The older exp_003 builder currently emits the broad learning limits and the
official deep home pose.  This entrypoint deliberately calls its tested XML
and reference transformation helpers with ``SAFE_JOINT_LIMITS`` and
``HARDWARE_SAFE_INIT_POS`` instead.  Outputs live under this experiment and
never overwrite either frozen WSL source tree or exp_003 artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from typing import Any
import zipfile


EXP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = EXP_ROOT.parents[2]
EXP003 = (
    WORKSPACE
    / "experiments"
    / "mujoco"
    / "exp_003_openduckmini_calibrated_walk"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--playground-source",
        type=Path,
        default=WORKSPACE / ".openduck_playground_source_review",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXP_ROOT / "artifacts" / "generated_playground",
    )
    parser.add_argument(
        "--legacy-v22-package",
        type=Path,
        default=(
            EXP003
            / "artifacts"
            / "openduckmini_calibrated_hybrid_v22_20260729.zip"
        ),
        help="Read-only source for the three frozen v22 reverse profiles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playground_source = args.playground_source.resolve()
    output_root = args.output_root.resolve()

    builder_path = EXP003 / "build_calibrated_playground.py"
    builder = _load_module("exp003_calibrated_builder", builder_path)
    calibration = builder.load_calibration(WORKSPACE)

    source_xml_dir = (
        playground_source / "playground" / "open_duck_mini_v2" / "xmls"
    )
    source_data_dir = (
        playground_source / "playground" / "open_duck_mini_v2" / "data"
    )
    output_xml_dir = output_root / "playground" / "open_duck_mini_v2" / "xmls"
    output_data_dir = output_root / "playground" / "open_duck_mini_v2" / "data"
    output_xml_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    source_model = source_xml_dir / "open_duck_mini_v2_backlash.xml"
    source_scene = source_xml_dir / "scene_flat_terrain_backlash.xml"
    source_reference = source_data_dir / "polynomial_coefficients.pkl"
    output_model = (
        output_xml_dir
        / "open_duck_mini_v2_backlash_hardware_safe_calibrated.xml"
    )
    output_scene = (
        output_xml_dir
        / "scene_flat_terrain_backlash_hardware_safe_calibrated.xml"
    )
    # Joystick resolves this path relative to the run cwd.  Keep its expected
    # filename while recording the stronger contract in the manifest.
    output_reference = output_data_dir / "polynomial_coefficients_calibrated.pkl"

    profile_members = {
        name: f".openduck_runtime_source_review/{name}"
        for name in (
            "optimized_backward_gait.json",
            "optimized_backward_left_turn_gait.json",
            "optimized_backward_right_turn_gait.json",
        )
    }
    with zipfile.ZipFile(args.legacy_v22_package.resolve()) as archive:
        for output_name, member in profile_members.items():
            (output_data_dir / output_name).write_bytes(archive.read(member))

    safe_limits = {
        **calibration.SAFE_JOINT_LIMITS,
        **calibration.HEAD_JOINT_LIMITS,
    }
    safe_init = dict(calibration.HARDWARE_SAFE_INIT_POS)
    builder.build_model_xml(source_model, output_model, safe_limits)
    builder.build_scene_xml(
        source_scene,
        source_model,
        output_scene,
        output_model,
        safe_init,
    )
    reference_joints = builder.transform_reference(
        source_reference,
        output_reference,
        safe_init,
        safe_limits,
        source_scene,
    )

    # MuJoCo resolves meshdir="assets" relative to the generated model.
    shutil.copytree(
        source_xml_dir / "assets",
        output_xml_dir / "assets",
        dirs_exist_ok=True,
    )
    validation = builder.validate_scene(output_scene, safe_init, safe_limits)

    key_files = {
        "source_builder": builder_path,
        "source_model": source_model,
        "source_scene": source_scene,
        "source_reference": source_reference,
        "legacy_v22_package": args.legacy_v22_package.resolve(),
        "generated_model": output_model,
        "generated_scene": output_scene,
        "generated_reference": output_reference,
        **{
            f"legacy_v22_{Path(name).stem}": output_data_dir / name
            for name in profile_members
        },
        "runtime_calibration": (
            WORKSPACE
            / ".openduck_runtime_source_review"
            / "mini_bdx_runtime"
            / "mini_bdx_runtime"
            / "calibrated_poses.py"
        ),
        "leg_zero_v2": (
            WORKSPACE
            / ".openduck_runtime_source_review"
            / "leg_zero_pose_authoritative_20260729_v2.json"
        ),
        "head_zero_v2": (
            WORKSPACE
            / ".openduck_runtime_source_review"
            / "head_calibration_authoritative_20260729_v2.json"
        ),
    }
    manifest = {
        "schema_version": 1,
        "experiment": EXP_ROOT.name,
        "contract": "hardware_safe_simulation_only",
        "real_hardware_deployment_allowed": False,
        "hardware_safe_home_scale": float(calibration.HARDWARE_SAFE_HOME_SCALE),
        "init_pose": _jsonable(safe_init),
        "joint_limits": _jsonable(safe_limits),
        "joint_directions": _jsonable(calibration.JOINT_DIRECTIONS),
        "leg_zero_raw": _jsonable(calibration.LEG_ZERO_RAW),
        "head_zero_raw": _jsonable(calibration.HEAD_ZERO_RAW),
        "scene_validation": _jsonable(validation),
        "reference_joints": _jsonable(reference_joints),
        "files": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in key_files.items()
        },
    }
    manifest_path = output_root / "hardware_safe_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
