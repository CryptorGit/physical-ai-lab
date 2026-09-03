#!/usr/bin/env python3
"""Audit the relative-joint-angle contract from public policy to measured robot.

This is a read-only, offline kinematic analysis.  It proves (or rejects) the
affine relation between the public model joint coordinate and the local
hardware-model joint coordinate before any hardware target could be considered.
It never opens a serial device or enables torque.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


PUBLIC_PLAYGROUND_COMMIT = "1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f"
ENCODER_STEPS = 4096
KINEMATIC_TOLERANCE_RAD = 1.0e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_public_commit(playground_root: Path) -> str:
    root_text = str(playground_root.resolve())
    commands = [["git", "-C", root_text, "rev-parse", "HEAD"]]
    if root_text.startswith("/mnt/") and len(root_text) > 7 and root_text[6] == "/":
        windows_tail = root_text[7:].replace("/", "\\")
        windows_root = f"{root_text[5].upper()}:\\{windows_tail}"
        commands.append(
            ["powershell.exe", "-NoProfile", "-Command", f"git -C '{windows_root}' rev-parse HEAD"]
        )
    failures: list[str] = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
            if commit != PUBLIC_PLAYGROUND_COMMIT:
                raise RuntimeError(f"unexpected public Playground commit: {commit}")
            return commit
        failures.append(result.stderr.strip())
    raise RuntimeError("could not resolve public Playground commit: " + " | ".join(failures))


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("openduck_calibrated_poses", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load calibration module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def rotation_angle(matrix: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return float(math.acos(cosine))


def body_relative_rotation(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> np.ndarray:
    joint_id = model.joint(joint_name).id
    child_body = int(model.jnt_bodyid[joint_id])
    parent_body = int(model.body_parentid[child_body])
    parent_rotation = data.xmat[parent_body].reshape(3, 3)
    child_rotation = data.xmat[child_body].reshape(3, 3)
    return parent_rotation.T @ child_rotation


def set_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    floating_base_qpos: np.ndarray,
    joint_target: dict[str, float],
) -> None:
    mujoco.mj_resetData(model, data)
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(model.jnt_qposadr[0]) != 0:
        raise RuntimeError("model does not start with a floating base at qpos[0:7]")
    data.qpos[:7] = floating_base_qpos[:7]
    for name, value in joint_target.items():
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def target_by_name(model: mujoco.MjModel, home_qpos: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for index in range(model.nu):
        name = model.actuator(index).name
        joint_id = model.joint(name).id
        result[name] = float(home_qpos[model.jnt_qposadr[joint_id]])
    return result


def zero_by_name(source: Path) -> tuple[dict[str, int], dict[str, int]]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    motors = payload.get("motors")
    if not isinstance(motors, dict):
        raise ValueError("authoritative zero source has no motors object")
    raw_by_name: dict[str, int] = {}
    id_by_name: dict[str, int] = {}
    for motor_id, entry in motors.items():
        if not isinstance(entry, dict):
            raise ValueError("invalid motor entry")
        name = str(entry["name"])
        raw_by_name[name] = int(entry["zero_raw"])
        id_by_name[name] = int(motor_id)
    return raw_by_name, id_by_name


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def report_markdown(rows: list[dict[str, Any]], result: dict[str, Any]) -> str:
    lines = [
        "# 公開policy → 計測済み実機モデル: 相対関節角契約",
        "",
        "## 結論",
        "",
        "公開モデルの相対ヒンジ角と実機モデルの相対ヒンジ角は、全14関節で同じ座標です。",
        "従って運動学上の写像は `q_real_model = q_official + 0 rad` です。",
        "左膝・左足首だけは、モデル角をサーボrawへ変換する際の物理モータ方向が負です。",
        "これは相対リンク角の符号差ではなく、サーボ回転方向の符号差です。",
        "",
        f"公式homeを同じqで実機モデルへ入れたときの親子相対回転の最大差: `{result['max_home_relative_frame_error_rad']:.3e} rad`。",
        f"ゼロ姿勢での親子相対回転の最大差: `{result['max_zero_relative_frame_error_rad']:.3e} rad`。",
        "",
        "## 関節別結果",
        "",
        "| 関節 | 公式 q (rad) | 必要な実機モデル q (rad) | 差 | サーボ軸の増分 (rad) | safe範囲内 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        safe = "head disabled" if row["hardware_safe"] is None else ("yes" if row["hardware_safe"] else "NO")
        values = dict(row)
        for field in (
            "official_q_rad",
            "local_required_q_rad",
            "q_difference_rad",
            "servo_shaft_delta_rad",
        ):
            values[field] = "—" if row[field] is None else fmt(float(row[field]))
        lines.append(
            "| {joint_name} | {official_q_rad} | {local_required_q_rad} | {q_difference_rad} | {servo_shaft_delta_rad} | {safe} |".format(
                safe=safe,
                **values,
            )
        )
    lines.extend(
        [
            "",
            "## 実機への含意",
            "",
            "この表はoffset/相対角の座標変換を解決しますが、可動域を増やしません。",
            f"公式homeのうち `{', '.join(result['unsafe_leg_joints'])}` は、現在の計測済みsafe範囲外です。",
            "従って公開重みを**変換だけで**実機投入することはできません。公開policyはhome中心の目標を出すため、",
            "安全な縮小姿勢に置換すると観測・行動・接触の契約自体が変わります。実機にはいかなる値も送信していません。",
            "",
            "`servo_shaft_delta_rad` は、計測済みゼロからサーボ軸を回す符号付き増分です。",
            "実機モデルの相対リンク角 `q_real_model` と混同しないでください。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playground-root", type=Path, required=True)
    parser.add_argument("--local-model", type=Path, required=True)
    parser.add_argument("--zero-source", type=Path, required=True)
    parser.add_argument("--calibrated-poses", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    playground_root = args.playground_root.resolve()
    local_model_path = args.local_model.resolve()
    zero_source = args.zero_source.resolve()
    calibrated_poses_path = args.calibrated_poses.resolve()
    for path in (local_model_path, zero_source, calibrated_poses_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = require_public_commit(playground_root)
    public_xml = playground_root / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    public_base = playground_root / "playground/open_duck_mini_v2/base.py"
    if not public_xml.is_file() or not public_base.is_file():
        raise FileNotFoundError("public XML/assets missing")
    sys.path.insert(0, str(playground_root))
    from playground.open_duck_mini_v2 import base  # pylint: disable=import-outside-toplevel

    public_model = mujoco.MjModel.from_xml_string(public_xml.read_text(), assets=base.get_assets())
    local_model = mujoco.MjModel.from_xml_path(str(local_model_path))
    public_home = public_model.keyframe("home").qpos.copy()
    names = [public_model.actuator(index).name for index in range(public_model.nu)]
    targets = target_by_name(public_model, public_home)
    if len(names) != 14 or set(names) != set(targets):
        raise RuntimeError("unexpected public 14-axis actuator contract")
    for name in names:
        if mujoco.mj_name2id(local_model, mujoco.mjtObj.mjOBJ_JOINT, name) < 0:
            raise RuntimeError(f"local model missing public joint: {name}")

    public_zero = mujoco.MjData(public_model)
    public_target = mujoco.MjData(public_model)
    local_zero = mujoco.MjData(local_model)
    local_target = mujoco.MjData(local_model)
    zero_target = {name: 0.0 for name in names}
    set_pose(public_model, public_zero, public_home, zero_target)
    set_pose(public_model, public_target, public_home, targets)
    set_pose(local_model, local_zero, public_home, zero_target)
    # The kinematically required real-model coordinate is q_local=q_public.
    set_pose(local_model, local_target, public_home, targets)

    calibration = load_module(calibrated_poses_path)
    raw_zero, servo_id = zero_by_name(zero_source)
    if dict(calibration.LEG_ZERO_RAW) != raw_zero:
        raise RuntimeError("calibration module raw zeros differ from authoritative capture")
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    max_zero_error = 0.0
    max_home_error = 0.0
    for name in names:
        public_joint_id = public_model.joint(name).id
        local_joint_id = local_model.joint(name).id
        axis_public = public_model.jnt_axis[public_joint_id]
        axis_local = local_model.jnt_axis[local_joint_id]
        axis_error = float(np.linalg.norm(axis_public - axis_local))
        if axis_error > 1.0e-12:
            failures.append(f"axis:{name}:{axis_error}")
        public_zero_rotation = body_relative_rotation(public_model, public_zero, name)
        local_zero_rotation = body_relative_rotation(local_model, local_zero, name)
        public_home_rotation = body_relative_rotation(public_model, public_target, name)
        local_home_rotation = body_relative_rotation(local_model, local_target, name)
        zero_error = rotation_angle(public_zero_rotation.T @ local_zero_rotation)
        home_error = rotation_angle(public_home_rotation.T @ local_home_rotation)
        max_zero_error = max(max_zero_error, zero_error)
        max_home_error = max(max_home_error, home_error)
        if zero_error > KINEMATIC_TOLERANCE_RAD or home_error > KINEMATIC_TOLERANCE_RAD:
            failures.append(f"relative_pose:{name}:{zero_error}:{home_error}")

        official_q = targets[name]
        entry: dict[str, Any] = {
            "joint_name": name,
            "official_q_rad": official_q,
            "local_required_q_rad": official_q,
            "q_difference_rad": 0.0,
            "public_axis": axis_public.tolist(),
            "local_axis": axis_local.tolist(),
            "axis_difference_norm": axis_error,
            "zero_relative_frame_error_rad": zero_error,
            "home_relative_frame_error_rad": home_error,
            "public_model_joint_range_rad": public_model.jnt_range[public_joint_id].tolist(),
            "local_model_joint_range_rad": local_model.jnt_range[local_joint_id].tolist(),
            "servo_id": None,
            "servo_direction": None,
            "servo_shaft_delta_rad": None,
            "servo_target_raw_float": None,
            "hardware_safe": None,
            "hardware_safe_range_rad": None,
        }
        if name in raw_zero:
            direction = float(calibration.JOINT_DIRECTIONS[name])
            shaft_delta = direction * official_q
            raw_target = (raw_zero[name] + shaft_delta * ENCODER_STEPS / (2.0 * math.pi)) % ENCODER_STEPS
            lower, upper = (float(value) for value in calibration.SAFE_JOINT_LIMITS[name])
            entry.update(
                {
                    "servo_id": servo_id[name],
                    "servo_direction": direction,
                    "servo_shaft_delta_rad": shaft_delta,
                    "servo_target_raw_float": raw_target,
                    "hardware_safe": lower <= official_q <= upper,
                    "hardware_safe_range_rad": [lower, upper],
                }
            )
        rows.append(entry)

    unsafe = [row["joint_name"] for row in rows if row["hardware_safe"] is False]
    if failures:
        raise RuntimeError("relative coordinate contract failed: " + " | ".join(failures))
    result = {
        "schema_version": 1,
        "purpose": "read-only relative-angle mapping from public policy model to measured local hardware model",
        "not_real_robot_command": True,
        "public_playground_commit": commit,
        "affine_model_joint_mapping": "q_real_model = +1.0 * q_official + 0.0 rad",
        "maximum_axis_difference_norm": max(row["axis_difference_norm"] for row in rows),
        "max_zero_relative_frame_error_rad": max_zero_error,
        "max_home_relative_frame_error_rad": max_home_error,
        "kinematic_tolerance_rad": KINEMATIC_TOLERANCE_RAD,
        "unsafe_leg_joints": unsafe,
        "rows": rows,
        "paths": {
            "public_xml": str(public_xml),
            "public_asset_source": str(public_base),
            "local_model": str(local_model_path),
            "authoritative_zero_source": str(zero_source),
            "calibration_source": str(calibrated_poses_path),
        },
        "sha256": {
            "public_xml": sha256(public_xml),
            "public_asset_source": sha256(public_base),
            "local_model": sha256(local_model_path),
            "authoritative_zero_source": sha256(zero_source),
            "calibration_source": sha256(calibrated_poses_path),
            "analysis_script": sha256(Path(__file__).resolve()),
        },
    }
    report_path = output_dir / "relative_joint_contract.json"
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    csv_path = output_dir / "relative_joint_contract.csv"
    csv_fields = [
        "joint_name", "official_q_rad", "local_required_q_rad", "q_difference_rad",
        "servo_id", "servo_direction", "servo_shaft_delta_rad", "servo_target_raw_float",
        "hardware_safe", "zero_relative_frame_error_rad", "home_relative_frame_error_rad",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in csv_fields})
    markdown_path = output_dir / "README.md"
    markdown_path.write_text(report_markdown(rows, result), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
        "max_home_relative_frame_error_rad": max_home_error,
        "unsafe_leg_joints": unsafe,
    }, indent=2))


if __name__ == "__main__":
    main()
