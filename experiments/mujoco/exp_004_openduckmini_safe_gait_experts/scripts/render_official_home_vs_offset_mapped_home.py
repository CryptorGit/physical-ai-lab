#!/usr/bin/env python3
"""Compare public ``home`` with the same pose mapped through measured offsets.

Both panels use the public ``home`` floating-base position and orientation.
The right panel uses the local detailed hardware model with the public joint
targets and calculates the corresponding measured-servo raw targets offline.
It is a visualization and feasibility audit only: it opens no serial port,
enables no torque, and sends no hardware target.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PUBLIC_PLAYGROUND_COMMIT = "1842c8f46a67cb5d6b74e5aaf08c8702cde6e74f"
PANEL_WIDTH = 640
PANEL_HEIGHT = 480
HEADER_HEIGHT = 70
FOOTER_HEIGHT = 94
BACKGROUND = (22, 28, 36)


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
    errors: list[str] = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
            if commit != PUBLIC_PLAYGROUND_COMMIT:
                raise RuntimeError(f"unexpected public Playground commit: {commit}")
            return commit
        errors.append(result.stderr.strip())
    raise RuntimeError("could not resolve public Playground commit: " + " | ".join(errors))


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location("openduck_calibrated_poses", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load calibration module: {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def render(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.23)
    camera.distance = 0.72
    camera.azimuth = 180.0
    camera.elevation = -12.0
    with mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()


def public_home(playground_root: Path):
    root = playground_root.resolve()
    xml_path = root / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    base_path = root / "playground/open_duck_mini_v2/base.py"
    if not xml_path.is_file() or not base_path.is_file():
        raise FileNotFoundError("missing public Open Duck Playground XML/assets")
    sys.path.insert(0, str(root))
    from playground.open_duck_mini_v2 import base  # pylint: disable=import-outside-toplevel

    model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=base.get_assets())
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:] = home.qpos
    data.ctrl[:] = home.ctrl
    mujoco.mj_forward(model, data)
    names = [model.actuator(index).name for index in range(model.nu)]
    target_by_name = {
        name: float(home.qpos[model.jnt_qposadr[model.joint(name).id]]) for name in names
    }
    return model, data, xml_path, base_path, home.qpos.copy(), target_by_name


def measured_zero_by_name(source: Path) -> dict[str, int]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    motors = payload.get("motors")
    if not isinstance(motors, dict):
        raise ValueError("authoritative zero source has no motors object")
    values: dict[str, int] = {}
    for item in motors.values():
        if not isinstance(item, dict):
            raise ValueError("invalid motor record in authoritative zero source")
        values[str(item["name"])] = int(item["zero_raw"])
    return values


def map_home_to_local(
    local_model_path: Path,
    public_base_qpos: np.ndarray,
    target_by_name: dict[str, float],
) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(local_model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_qposadr[0] != 0:
        raise RuntimeError("local hardware model does not start with a free floating base")
    data.qpos[:7] = public_base_qpos[:7]
    for name, target in target_by_name.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"local hardware model is missing public joint: {name}")
        data.qpos[model.jnt_qposadr[joint_id]] = target
    mujoco.mj_forward(model, data)
    return model, data


def raw_targets_and_violations(target_by_name: dict[str, float], zero_source: Path, calibration_module):
    measured_zero = measured_zero_by_name(zero_source)
    if dict(calibration_module.LEG_ZERO_RAW) != measured_zero:
        raise RuntimeError("calibration module raw zeros do not match authoritative capture")
    raw_targets: dict[str, float] = {}
    violations: dict[str, dict[str, float]] = {}
    for name, target in target_by_name.items():
        if name not in calibration_module.LEG_ZERO_RAW:
            continue  # Head targets are zero in the public home keyframe.
        direction = float(calibration_module.JOINT_DIRECTIONS[name])
        raw = (measured_zero[name] + direction * target * 4096.0 / (2.0 * np.pi)) % 4096.0
        raw_targets[name] = raw
        lower, upper = calibration_module.SAFE_JOINT_LIMITS[name]
        if target < lower or target > upper:
            violations[name] = {
                "target_rad": target,
                "lower_rad": lower,
                "upper_rad": upper,
                "violation_rad": lower - target if target < lower else target - upper,
            }
    return raw_targets, violations


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def compose(left: np.ndarray, right: np.ndarray, font_path: Path, violation_count: int) -> Image.Image:
    canvas = Image.new("RGB", (PANEL_WIDTH * 2, HEADER_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT), BACKGROUND)
    canvas.paste(Image.fromarray(left), (0, HEADER_HEIGHT))
    canvas.paste(Image.fromarray(right), (PANEL_WIDTH, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    title = font(font_path, 26)
    subtitle = font(font_path, 17)
    footer = font(font_path, 17)
    draw.rectangle((0, HEADER_HEIGHT - 1, PANEL_WIDTH * 2, HEADER_HEIGHT), fill=(86, 166, 240))
    draw.rectangle((PANEL_WIDTH - 1, 0, PANEL_WIDTH, HEADER_HEIGHT + PANEL_HEIGHT), fill=(86, 166, 240))
    draw.text((22, 14), "公開公式の home 初期姿勢", font=title, fill=(248, 250, 252))
    draw.text((22, 43), "public home keyframe", font=subtitle, fill=(181, 198, 218))
    draw.text((PANEL_WIDTH + 22, 14), "計測offset経由で公式homeを仮想適用", font=title, fill=(248, 250, 252))
    draw.text((PANEL_WIDTH + 22, 43), "同一胴体xyz・quaternion / offline visualization", font=subtitle, fill=(181, 198, 218))
    footer_y = HEADER_HEIGHT + PANEL_HEIGHT
    draw.text((22, footer_y + 12), "左と右の浮遊ベース座標・向きは同一。右の関節目標は公開homeと同一です。", font=footer, fill=(214, 224, 236))
    warning = f"注意: 公式homeは現在の計測safe範囲を {violation_count} 関節で超過するため、右は実機へ送信不可です。"
    draw.text((22, footer_y + 39), warning, font=footer, fill=(255, 183, 77))
    draw.text((22, footer_y + 66), "この画像はoffset変換の座標比較であり、実機写真・実機コマンドではありません。", font=footer, fill=(151, 171, 196))
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playground-root", type=Path, required=True)
    parser.add_argument("--local-model", type=Path, required=True)
    parser.add_argument("--zero-source", type=Path, required=True)
    parser.add_argument("--calibrated-poses", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/meiryo.ttc"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="official_home_vs_offset_mapped_home.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = {
        "playground_root": args.playground_root.resolve(),
        "local_model": args.local_model.resolve(),
        "zero_source": args.zero_source.resolve(),
        "calibrated_poses": args.calibrated_poses.resolve(),
        "font": args.font.resolve(),
    }
    for name, path in paths.items():
        if name != "playground_root" and not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".png":
        raise ValueError("--output-name must be a simple .png filename")

    commit = require_public_commit(paths["playground_root"])
    public_model, public_data, public_xml, public_base, public_qpos, targets = public_home(paths["playground_root"])
    local_model, local_data = map_home_to_local(paths["local_model"], public_qpos, targets)
    calibration = load_module(paths["calibrated_poses"])
    raw_targets, violations = raw_targets_and_violations(targets, paths["zero_source"], calibration)
    image = compose(render(public_model, public_data), render(local_model, local_data), paths["font"], len(violations))

    output_path = output_dir / output_name
    image.save(output_path)
    script_path = Path(__file__).resolve()
    manifest = {
        "purpose": "same-world-pose comparison of public home and measured-offset-mapped virtual home",
        "not_real_robot_command": True,
        "not_safe_for_live_hardware": bool(violations),
        "public_playground_commit": commit,
        "public_home_floating_base_qpos": public_qpos[:7].tolist(),
        "public_home_joint_target_rad": targets,
        "offline_servo_target_raw": raw_targets,
        "hardware_safe_limit_violations": violations,
        "paths": {
            "public_xml": str(public_xml),
            "public_asset_source": str(public_base),
            "local_hardware_model": str(paths["local_model"]),
            "authoritative_zero_source": str(paths["zero_source"]),
            "calibration_source": str(paths["calibrated_poses"]),
            "renderer_script": str(script_path),
            "image": str(output_path),
        },
        "sha256": {
            "public_xml": sha256(public_xml),
            "public_asset_source": sha256(public_base),
            "local_hardware_model": sha256(paths["local_model"]),
            "authoritative_zero_source": sha256(paths["zero_source"]),
            "calibration_source": sha256(paths["calibrated_poses"]),
            "renderer_script": sha256(script_path),
            "image": sha256(output_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"image={output_path}")
    print(f"manifest={manifest_path}")
    print(json.dumps({"safe_limit_violations": violations}, indent=2))


if __name__ == "__main__":
    main()
