from pathlib import Path
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    ROOT
    / ".openduck_hardware_source_review"
    / "mini_bdx"
    / "robots"
    / "open_duck_mini_v2"
    / "scene.xml"
)
OUTPUT_DIR = ROOT / "artifacts"
RUNTIME_MODULE_DIR = (
    ROOT
    / ".openduck_runtime_source_review"
    / "mini_bdx_runtime"
    / "mini_bdx_runtime"
)
sys.path.insert(0, str(RUNTIME_MODULE_DIR))

from calibrated_poses import SAFE_INIT_POS  # noqa: E402

# User-recorded physical zero:
# 10=115, 11=259, 12=3137, 13=627, 14=3252,
# 20=3832, 21=365, 22=340, 23=342, 24=2711.
# In model space this is the geometric joint origin, not the walking policy's
# bent-knee initialization pose.
RECORDED_ZERO_MODEL_POSE = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": 0.0,
    "left_hip_pitch": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
    "neck_pitch": 0.0,
    "head_pitch": 0.0,
    "head_yaw": 0.0,
    "head_roll": 0.0,
    "left_antenna": 0.0,
    "right_antenna": 0.0,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.0,
    "right_hip_pitch": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
}

WALK_POLICY_INIT_POSE = dict(SAFE_INIT_POS)


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: dict[str, float]) -> None:
    mujoco.mj_resetData(model, data)
    for joint_name, angle in pose.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        data.qpos[model.jnt_qposadr[joint_id]] = angle
    mujoco.mj_forward(model, data)


def render_view(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    *,
    azimuth: float,
    elevation: float = -12.0,
    distance: float = 0.72,
) -> np.ndarray:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.19)
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def label_panel(image: np.ndarray, label: str) -> Image.Image:
    panel = Image.fromarray(image)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=22)
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    draw.rounded_rectangle(
        (18, 18, 38 + width, 55),
        radius=8,
        fill=(10, 16, 24, 210),
    )
    draw.text((28, 25), label, fill=(255, 255, 255), font=font)
    return panel


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.vis.global_.offwidth = 1024
    model.vis.global_.offheight = 1024
    data = mujoco.MjData(model)

    panel_size = 900
    with mujoco.Renderer(model, height=panel_size, width=panel_size) as renderer:
        set_pose(model, data, RECORDED_ZERO_MODEL_POSE)
        recorded_front = label_panel(
            render_view(renderer, data, azimuth=180),
            "Recorded physical zero - front",
        )
        recorded_side = label_panel(
            render_view(renderer, data, azimuth=90),
            "Recorded physical zero - side",
        )

        set_pose(model, data, WALK_POLICY_INIT_POSE)
        policy_init_front = label_panel(
            render_view(renderer, data, azimuth=180),
            "Safe learning init_pos - front",
        )
        policy_init_side = label_panel(
            render_view(renderer, data, azimuth=90),
            "Safe learning init_pos - side",
        )

    front_path = OUTPUT_DIR / "openduck_zero_pose_front.png"
    side_path = OUTPUT_DIR / "openduck_zero_pose_side.png"
    safe_front_path = OUTPUT_DIR / "openduck_safe_init_pose_front.png"
    safe_side_path = OUTPUT_DIR / "openduck_safe_init_pose_side.png"
    recorded_front.save(front_path)
    recorded_side.save(side_path)
    policy_init_front.save(safe_front_path)
    policy_init_side.save(safe_side_path)

    composite_front = recorded_front.resize((480, 480), Image.Resampling.LANCZOS)
    composite_side = recorded_side.resize((480, 480), Image.Resampling.LANCZOS)
    composite_policy = policy_init_front.resize((480, 480), Image.Resampling.LANCZOS)
    composite_policy_side = policy_init_side.resize(
        (480, 480), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (960, 960), (22, 28, 36))
    canvas.paste(composite_front, (0, 0))
    canvas.paste(composite_side, (480, 0))
    canvas.paste(composite_policy, (0, 480))
    canvas.paste(composite_policy_side, (480, 480))

    output_path = OUTPUT_DIR / "openduck_recorded_zero_pose.png"
    canvas.save(output_path)
    print(front_path)
    print(side_path)
    print(safe_front_path)
    print(safe_side_path)
    print(output_path)


if __name__ == "__main__":
    main()
