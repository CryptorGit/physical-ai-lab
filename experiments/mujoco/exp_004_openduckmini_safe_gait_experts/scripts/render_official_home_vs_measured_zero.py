#!/usr/bin/env python3
"""Make an auditable side-by-side image of official home and measured zero.

The left panel is rendered directly from the public Open Duck Playground
``home`` keyframe.  The right panel reuses the existing front rendering of the
user-confirmed physical servo-offset zero capture.  No hardware is opened or
controlled by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


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


def public_commit(playground_root: Path) -> str:
    commands = [["git", "-C", str(playground_root), "rev-parse", "HEAD"]]
    root_text = str(playground_root)
    # A Windows-created worktree records an absolute ``C:/...`` gitdir in its
    # .git file.  WSL Git cannot resolve that form, while Windows Git can.
    if root_text.startswith("/mnt/") and len(root_text) > 7 and root_text[6] == "/":
        windows_tail = root_text[7:].replace("/", "\\")
        windows_root = f"{root_text[5].upper()}:\\{windows_tail}"
        commands.append(
            [
                "powershell.exe",
                "-NoProfile",
                "-Command",
                f"git -C '{windows_root}' rev-parse HEAD",
            ]
        )

    commit = ""
    errors: list[str] = []
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            commit = result.stdout.strip()
            break
        errors.append(result.stderr.strip())
    if not commit:
        raise RuntimeError("could not resolve public Playground commit: " + " | ".join(errors))
    if commit != PUBLIC_PLAYGROUND_COMMIT:
        raise RuntimeError(
            "the selected Playground checkout is not the pinned public commit: "
            f"{commit} != {PUBLIC_PLAYGROUND_COMMIT}"
        )
    return commit


def load_font(font_path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path), size=size)


def render_official_home(playground_root: Path) -> tuple[np.ndarray, Path, Path]:
    """Render the public XML's home keyframe, with no policy control step."""
    root = playground_root.resolve()
    xml_path = root / "playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml"
    base_path = root / "playground/open_duck_mini_v2/base.py"
    if not xml_path.is_file() or not base_path.is_file():
        raise FileNotFoundError("public Open Duck Playground XML/assets are missing")

    sys.path.insert(0, str(root))
    from playground.open_duck_mini_v2 import base  # pylint: disable=import-outside-toplevel

    model = mujoco.MjModel.from_xml_string(xml_path.read_text(), assets=base.get_assets())
    data = mujoco.MjData(model)
    home = model.keyframe("home")
    data.qpos[:] = home.qpos
    data.ctrl[:] = home.ctrl
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.23)
    camera.distance = 0.72
    camera.azimuth = 180.0
    camera.elevation = -12.0
    with mujoco.Renderer(model, height=PANEL_HEIGHT, width=PANEL_WIDTH) as renderer:
        renderer.update_scene(data, camera=camera)
        image = renderer.render().copy()
    return image, xml_path, base_path


def fit_panel(image: Image.Image) -> Image.Image:
    fitted = ImageOps.contain(image.convert("RGB"), (PANEL_WIDTH, PANEL_HEIGHT), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (PANEL_WIDTH, PANEL_HEIGHT), (42, 52, 64))
    panel.paste(fitted, ((PANEL_WIDTH - fitted.width) // 2, (PANEL_HEIGHT - fitted.height) // 2))
    return panel


def draw_title(draw: ImageDraw.ImageDraw, origin_x: int, title: str, subtitle: str, title_font, subtitle_font) -> None:
    draw.text((origin_x + 22, 14), title, font=title_font, fill=(248, 250, 252))
    draw.text((origin_x + 22, 43), subtitle, font=subtitle_font, fill=(181, 198, 218))


def draw_footer(draw: ImageDraw.ImageDraw, font, footer_y: int) -> None:
    left = "左: 公開 Open Duck Playground / scene_flat_terrain.xml の home keyframe"
    right = "右: 2026-07-29 計測済みサーボ offset の model-space ゼロ姿勢"
    note = "比較用レンダリング（実機写真ではありません）。同一の姿勢・モデルを主張する画像ではありません。"
    draw.text((22, footer_y + 12), left, font=font, fill=(214, 224, 236))
    draw.text((22, footer_y + 39), right, font=font, fill=(214, 224, 236))
    draw.text((22, footer_y + 66), note, font=font, fill=(151, 171, 196))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--playground-root", type=Path, required=True)
    parser.add_argument("--measured-zero-image", type=Path, required=True)
    parser.add_argument("--zero-source", type=Path, required=True)
    parser.add_argument("--font", type=Path, default=Path("/mnt/c/Windows/Fonts/meiryo.ttc"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="official_home_vs_measured_offset_zero.png")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    playground_root = args.playground_root.resolve()
    measured_zero_image = args.measured_zero_image.resolve()
    zero_source = args.zero_source.resolve()
    output_dir = args.output_dir.resolve()
    output_name = Path(args.output_name)
    if output_name.name != args.output_name or output_name.suffix.lower() != ".png":
        raise ValueError("--output-name must be a simple .png filename")
    for path in (measured_zero_image, zero_source, args.font):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = public_commit(playground_root)
    official_image, xml_path, base_path = render_official_home(playground_root)
    measured_image = Image.open(measured_zero_image)

    title_font = load_font(args.font, 26)
    subtitle_font = load_font(args.font, 17)
    footer_font = load_font(args.font, 17)
    canvas = Image.new(
        "RGB",
        (PANEL_WIDTH * 2, HEADER_HEIGHT + PANEL_HEIGHT + FOOTER_HEIGHT),
        BACKGROUND,
    )
    canvas.paste(fit_panel(Image.fromarray(official_image)), (0, HEADER_HEIGHT))
    canvas.paste(fit_panel(measured_image), (PANEL_WIDTH, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, HEADER_HEIGHT - 1, PANEL_WIDTH * 2, HEADER_HEIGHT), fill=(86, 166, 240))
    draw.rectangle((PANEL_WIDTH - 1, 0, PANEL_WIDTH, HEADER_HEIGHT + PANEL_HEIGHT), fill=(86, 166, 240))
    draw_title(draw, 0, "公開公式の home 初期姿勢", "Open Duck Playground public model", title_font, subtitle_font)
    draw_title(draw, PANEL_WIDTH, "あなたの計測済み offset ゼロ姿勢", "2026-07-29 torque-off capture", title_font, subtitle_font)
    draw_footer(draw, footer_font, HEADER_HEIGHT + PANEL_HEIGHT)

    output_path = output_dir / output_name
    canvas.save(output_path)
    script_path = Path(__file__).resolve()
    manifest = {
        "purpose": "visual comparison of public official home and user-confirmed measured offset zero",
        "not_real_robot_photo": True,
        "not_real_robot_command": True,
        "public_playground_commit": commit,
        "paths": {
            "official_xml": str(xml_path),
            "official_asset_source": str(base_path),
            "measured_zero_image": str(measured_zero_image),
            "authoritative_zero_source": str(zero_source),
            "renderer_script": str(script_path),
            "image": str(output_path),
        },
        "sha256": {
            "official_xml": sha256(xml_path),
            "official_asset_source": sha256(base_path),
            "measured_zero_image": sha256(measured_zero_image),
            "authoritative_zero_source": sha256(zero_source),
            "renderer_script": sha256(script_path),
            "image": sha256(output_path),
        },
        "panels": {
            "left": "public scene_flat_terrain.xml home keyframe",
            "right": "existing front rendering of user-confirmed measured servo-offset zero",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"image={output_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
