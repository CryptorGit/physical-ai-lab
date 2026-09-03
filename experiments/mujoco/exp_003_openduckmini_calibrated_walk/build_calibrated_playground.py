"""Build an Open Duck Playground profile constrained by measured hardware.

The authoritative raw calibration is never modified.  This script creates:

* a copy of the backlash MJCF with measured safe model-space joint ranges;
* a scene whose home keyframe preserves the complete official home-pose shape;
* a reference-motion data set that preserves the official gait shape while
  fitting every leg trajectory inside the measured safe ranges.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


LEG_ORDER = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
REFERENCE_POSITION_INDICES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
REFERENCE_VELOCITY_INDICES = [16, 17, 18, 19, 20, 27, 28, 29, 30, 31]
REFERENCE_RANGE_MARGIN = 0.90

HEAD_COLLISION_MESHES = {
    # (mesh, contact_type, contact_affinity)
    "trunk_assembly": {
        "body_front": (4, 0),
        "body_back": (8, 0),
        "body_middle_bottom": (8, 0),
        "body_middle_top": (8, 0),
        "trunk_bottom": (8, 0),
        "trunk_top": (8, 0),
    },
    "neck_pitch_assembly": {
        "neck_left_sheet": (0, 4),
        "neck_right_sheet": (0, 4),
    },
    "head_pitch_to_yaw": {"head_pitch_to_yaw": (0, 8)},
    "neck_yaw_assembly": {"head_yaw_to_roll": (0, 8)},
    "head_assembly": {"head_bot_sheet": (0, 8), "head": (0, 8)},
}


def add_head_frame_collisions(root: ET.Element) -> None:
    """Enable rigid-frame contacts used to learn the coupled 32/33 envelope."""
    option = root.find("option")
    if option is None:
        raise RuntimeError("MJCF option element was not found")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    flag.set("filterparent", "disable")

    added = 0
    for body in root.iter("body"):
        body_name = body.get("name")
        selected_meshes = HEAD_COLLISION_MESHES.get(body_name)
        if not selected_meshes:
            continue
        for geom in list(body.findall("geom")):
            mesh = geom.get("mesh")
            if geom.get("class") != "visual" or mesh not in selected_meshes:
                continue
            contact_type, contact_affinity = selected_meshes[mesh]
            collision = copy.deepcopy(geom)
            collision.set("name", f"head_frame_collision_{added}_{mesh}")
            collision.set("class", "collision")
            # Collision bit assignments are retained as metadata in the
            # builder above, while the training model uses the precomputed
            # coupled envelope to avoid the prohibitive cost of mesh contacts
            # across thousands of MJX environments.
            collision.set("contype", "0")
            collision.set("conaffinity", "0")
            collision.set("group", "3")
            collision.set("rgba", "0.9 0.1 0.1 0.15")
            collision.attrib.pop("material", None)
            body.append(collision)
            added += 1
    if added == 0:
        raise RuntimeError("No head collision meshes were added")


def load_calibration(workspace: Path):
    module_path = (
        workspace
        / ".openduck_runtime_source_review"
        / "mini_bdx_runtime"
        / "mini_bdx_runtime"
        / "calibrated_poses.py"
    )
    spec = importlib.util.spec_from_file_location("calibrated_poses", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import calibration from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_model_xml(
    source_model: Path,
    output_model: Path,
    safe_limits: dict[str, tuple[float, float]],
) -> None:
    tree = ET.parse(source_model)
    root = tree.getroot()
    add_head_frame_collisions(root)
    found = set()
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name in safe_limits:
            lower, upper = safe_limits[name]
            joint.set("range", f"{lower:.9f} {upper:.9f}")
            found.add(name)
    missing = set(safe_limits) - found
    if missing:
        raise RuntimeError(f"Leg joints missing from MJCF: {sorted(missing)}")
    ET.indent(tree, space="  ")
    tree.write(output_model, encoding="utf-8", xml_declaration=True)


def build_scene_xml(
    source_scene: Path,
    source_model: Path,
    output_scene: Path,
    output_model: Path,
    safe_init: dict[str, float],
) -> None:
    source_mj_model = mujoco.MjModel.from_xml_path(str(source_scene))
    home = source_mj_model.keyframe("home")
    qpos = np.asarray(home.qpos, dtype=np.float64).copy()
    for joint_name, value in safe_init.items():
        qpos_address = int(
            np.asarray(source_mj_model.joint(joint_name).qposadr).reshape(-1)[0]
        )
        qpos[qpos_address] = value

    # Preserve the official home's mean sole/floor relationship after changing
    # its joint coordinates.  Root height is derived from kinematics rather
    # than being conflated with either the calibrated zero or the stand pose.
    source_data = mujoco.MjData(source_mj_model)
    source_data.qpos[:] = home.qpos
    mujoco.mj_forward(source_mj_model, source_data)
    target_sole_height = np.mean(
        [
            source_data.geom("left_foot_bottom_tpu").xpos[2],
            source_data.geom("right_foot_bottom_tpu").xpos[2],
        ]
    )
    mapped_data = mujoco.MjData(source_mj_model)
    mapped_data.qpos[:] = qpos
    mujoco.mj_forward(source_mj_model, mapped_data)
    mapped_sole_height = np.mean(
        [
            mapped_data.geom("left_foot_bottom_tpu").xpos[2],
            mapped_data.geom("right_foot_bottom_tpu").xpos[2],
        ]
    )
    qpos[2] += target_sole_height - mapped_sole_height

    actuator_names = [
        source_mj_model.actuator(index).name for index in range(source_mj_model.nu)
    ]
    ctrl = np.array([safe_init[name] for name in actuator_names], dtype=np.float64)

    tree = ET.parse(source_scene)
    root = tree.getroot()
    include = root.find("include")
    if include is None or include.get("file") != source_model.name:
        raise RuntimeError("Expected source model include was not found")
    include.set("file", output_model.name)

    key = root.find("./keyframe/key[@name='home']")
    if key is None:
        raise RuntimeError("Home keyframe was not found")
    key.set("qpos", " ".join(f"{value:.9f}" for value in qpos))
    key.set("ctrl", " ".join(f"{value:.9f}" for value in ctrl))
    ET.indent(tree, space="  ")
    tree.write(output_scene, encoding="utf-8", xml_declaration=True)


def sample_reference_extrema(reference: dict) -> tuple[np.ndarray, np.ndarray]:
    minima = None
    maxima = None
    for motion in reference.values():
        coefficients = list(motion["coefficients"].values())
        steps = int(motion["period"] * motion["fps"])
        phase = np.arange(steps, dtype=np.float64) / steps
        values = np.stack(
            [np.polyval(np.flip(coefficient), phase) for coefficient in coefficients],
            axis=1,
        )
        current_min = np.min(values, axis=0)
        current_max = np.max(values, axis=0)
        minima = current_min if minima is None else np.minimum(minima, current_min)
        maxima = current_max if maxima is None else np.maximum(maxima, current_max)
    if minima is None or maxima is None:
        raise RuntimeError("Reference data is empty")
    return minima, maxima


def calculate_reference_scales(
    minima: np.ndarray,
    maxima: np.ndarray,
    official_center: np.ndarray,
    safe_center: np.ndarray,
    safe_limits: dict[str, tuple[float, float]],
) -> np.ndarray:
    scales = []
    for position_index, joint_name, old_center, new_center in zip(
        REFERENCE_POSITION_INDICES,
        LEG_ORDER,
        official_center,
        safe_center,
        strict=True,
    ):
        lower, upper = safe_limits[joint_name]
        lower_room = (new_center - lower) * REFERENCE_RANGE_MARGIN
        upper_room = (upper - new_center) * REFERENCE_RANGE_MARGIN
        old_lower_excursion = max(0.0, old_center - minima[position_index])
        old_upper_excursion = max(0.0, maxima[position_index] - old_center)
        lower_scale = (
            lower_room / old_lower_excursion
            if old_lower_excursion > 1e-9
            else np.inf
        )
        upper_scale = (
            upper_room / old_upper_excursion
            if old_upper_excursion > 1e-9
            else np.inf
        )
        scales.append(min(lower_scale, upper_scale))
    return np.asarray(scales, dtype=np.float64)


def transform_reference(
    source_reference: Path,
    output_reference: Path,
    safe_init: dict[str, float],
    safe_limits: dict[str, tuple[float, float]],
    original_scene: Path,
) -> dict:
    with source_reference.open("rb") as stream:
        reference = pickle.load(stream)

    original_model = mujoco.MjModel.from_xml_path(str(original_scene))
    original_home = original_model.keyframe("home").ctrl
    actuator_index = {
        original_model.actuator(index).name: index
        for index in range(original_model.nu)
    }
    official_center = np.array(
        [original_home[actuator_index[name]] for name in LEG_ORDER],
        dtype=np.float64,
    )
    safe_center = np.array([safe_init[name] for name in LEG_ORDER], dtype=np.float64)
    minima, maxima = sample_reference_extrema(reference)
    scales = calculate_reference_scales(
        minima, maxima, official_center, safe_center, safe_limits
    )

    transformed = pickle.loads(pickle.dumps(reference))
    for motion in transformed.values():
        coefficient_items = list(motion["coefficients"].items())
        for leg_index, (position_index, velocity_index) in enumerate(
            zip(
                REFERENCE_POSITION_INDICES,
                REFERENCE_VELOCITY_INDICES,
                strict=True,
            )
        ):
            position_name, position_coeff = coefficient_items[position_index]
            velocity_name, velocity_coeff = coefficient_items[velocity_index]
            scale = scales[leg_index]

            position_coeff = np.asarray(position_coeff, dtype=np.float64).copy()
            position_coeff[0] = (
                safe_center[leg_index]
                + scale * (position_coeff[0] - official_center[leg_index])
            )
            position_coeff[1:] *= scale
            velocity_coeff = np.asarray(velocity_coeff, dtype=np.float64) * scale

            motion["coefficients"][position_name] = position_coeff
            motion["coefficients"][velocity_name] = velocity_coeff

    with output_reference.open("wb") as stream:
        pickle.dump(transformed, stream)

    transformed_minima, transformed_maxima = sample_reference_extrema(transformed)
    joints = {}
    for leg_index, (joint_name, position_index) in enumerate(
        zip(LEG_ORDER, REFERENCE_POSITION_INDICES, strict=True)
    ):
        lower, upper = safe_limits[joint_name]
        transformed_lower = float(transformed_minima[position_index])
        transformed_upper = float(transformed_maxima[position_index])
        if transformed_lower < lower - 1e-6 or transformed_upper > upper + 1e-6:
            raise RuntimeError(
                f"{joint_name} transformed reference "
                f"[{transformed_lower}, {transformed_upper}] exceeds [{lower}, {upper}]"
            )
        joints[joint_name] = {
            "official_center": float(official_center[leg_index]),
            "safe_center": float(safe_center[leg_index]),
            "scale": float(scales[leg_index]),
            "source_range": [
                float(minima[position_index]),
                float(maxima[position_index]),
            ],
            "transformed_range": [transformed_lower, transformed_upper],
            "safe_range": [float(lower), float(upper)],
        }
    return joints


def validate_scene(
    calibrated_scene: Path,
    safe_init: dict[str, float],
    safe_limits: dict[str, tuple[float, float]],
) -> dict:
    model = mujoco.MjModel.from_xml_path(str(calibrated_scene))
    home = model.keyframe("home")
    data = mujoco.MjData(model)
    data.qpos[:] = home.qpos
    data.ctrl[:] = home.ctrl
    mujoco.mj_forward(model, data)

    for joint_name, (lower, upper) in safe_limits.items():
        qpos_address = int(
            np.asarray(model.joint(joint_name).qposadr).reshape(-1)[0]
        )
        value = float(home.qpos[qpos_address])
        if not lower <= value <= upper:
            raise RuntimeError(f"{joint_name} home value {value} is out of range")
        if not np.isclose(value, safe_init[joint_name], atol=1e-8):
            raise RuntimeError(f"{joint_name} home value does not match safe init")

    sole_heights = [
        float(data.geom("left_foot_bottom_tpu").xpos[2]),
        float(data.geom("right_foot_bottom_tpu").xpos[2]),
    ]
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "home_root_z": float(home.qpos[2]),
        "sole_geom_center_heights": sole_heights,
    }


def parse_args() -> argparse.Namespace:
    experiment = Path(__file__).resolve().parent
    workspace = experiment.parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument(
        "--playground",
        type=Path,
        default=workspace / ".openduck_playground_source_review",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=experiment / "artifacts" / "calibrated_playground_profile.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    playground = args.playground.resolve()
    calibration = load_calibration(workspace)

    xml_dir = playground / "playground" / "open_duck_mini_v2" / "xmls"
    data_dir = playground / "playground" / "open_duck_mini_v2" / "data"
    source_model = xml_dir / "open_duck_mini_v2_backlash.xml"
    output_model = xml_dir / "open_duck_mini_v2_backlash_calibrated.xml"
    source_scene = xml_dir / "scene_flat_terrain_backlash.xml"
    output_scene = xml_dir / "scene_flat_terrain_backlash_calibrated.xml"
    source_reference = data_dir / "polynomial_coefficients.pkl"
    output_reference = data_dir / "polynomial_coefficients_calibrated.pkl"

    learning_limits = {
        **calibration.LEARNING_JOINT_LIMITS,
        **calibration.HEAD_JOINT_LIMITS,
    }
    build_model_xml(source_model, output_model, learning_limits)
    build_scene_xml(
        source_scene,
        source_model,
        output_scene,
        output_model,
        calibration.STAND_INIT_POS,
    )
    reference_joints = transform_reference(
        source_reference,
        output_reference,
        calibration.STAND_INIT_POS,
        learning_limits,
        source_scene,
    )
    scene_validation = validate_scene(
        output_scene,
        calibration.STAND_INIT_POS,
        learning_limits,
    )

    report = {
        "source_scene": str(source_scene),
        "calibrated_scene": str(output_scene),
        "source_reference": str(source_reference),
        "calibrated_reference": str(output_reference),
        "reference_range_margin": REFERENCE_RANGE_MARGIN,
        "stand_home_scale": calibration.STAND_HOME_SCALE,
        "hardware_safe_home_scale": calibration.HARDWARE_SAFE_HOME_SCALE,
        "official_home_pose": calibration.ORIGINAL_HOME_POS,
        "calibrated_zero_pose": calibration.CALIBRATED_ZERO_POS,
        "stand_init_pose": calibration.STAND_INIT_POS,
        "learning_joint_limits": calibration.LEARNING_JOINT_LIMITS,
        "head_joint_limits": calibration.HEAD_JOINT_LIMITS,
        "head_zero_raw": calibration.HEAD_ZERO_RAW,
        "hardware_safe_init_pose": calibration.HARDWARE_SAFE_INIT_POS,
        "hardware_safe_joint_limits": calibration.SAFE_JOINT_LIMITS,
        "scene_validation": scene_validation,
        "joints": reference_joints,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
