"""Render the official and calibrated learning STAND poses for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


def render(scene: Path, azimuth: float) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.keyframe("home").id)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = data.body("base").xpos
    camera.lookat[2] -= 0.02
    camera.distance = 0.47
    camera.azimuth = azimuth
    camera.elevation = -10.0

    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data, camera=camera)
    pixels = renderer.render().copy()
    renderer.close()
    return pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    xml_dir = (
        root
        / ".openduck_playground_source_review"
        / "playground"
        / "open_duck_mini_v2"
        / "xmls"
    )
    original = xml_dir / "scene_flat_terrain_backlash.xml"
    calibrated = xml_dir / "scene_flat_terrain_backlash_calibrated.xml"

    # Left pair: exact side view exposes knee flexion.  Right pair: oblique
    # view confirms the complete asymmetric home vector.
    panels = [
        render(original, 90.0),
        render(calibrated, 90.0),
        render(original, 135.0),
        render(calibrated, 135.0),
    ]
    comparison = np.concatenate(panels, axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(comparison).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
