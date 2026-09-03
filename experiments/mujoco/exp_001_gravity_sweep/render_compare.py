"""地球・火星・月の重力条件を横並び動画で比較する。"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "videos"


CONDITIONS = {
    "Earth": 9.80665,
    "Mars": 3.71,
    "Moon": 1.62,
}


MODEL_TEMPLATE = """
<mujoco model="gravity_compare">
    <option timestep="{timestep}" gravity="0 0 -{gravity}"/>

    <visual>
        <global offwidth="{width}" offheight="{height}"/>
    </visual>

    <worldbody>
        <light
            pos="0 -2 4"
            dir="0 1 -1"
        />

        <geom
            name="floor"
            type="plane"
            size="5 5 0.1"
            rgba="0.75 0.75 0.75 1"
        />

        <body name="ball" pos="0 0 {initial_height}">
            <freejoint/>
            <geom
                name="ball_geom"
                type="sphere"
                size="0.15"
                mass="1.0"
                rgba="0.2 0.5 0.9 1"
            />
        </body>
    </worldbody>
</mujoco>
"""


def add_label(frame: np.ndarray, text: str) -> np.ndarray:
    """フレーム左上へラベル文字を描画する。"""
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    # 背景付きの簡易ラベル
    draw.rectangle((10, 10, 170, 50), fill=(0, 0, 0, 160))
    draw.text((20, 18), text, fill=(255, 255, 255))

    return np.array(image)


def make_camera() -> mujoco.MjvCamera:
    """画面中央にボールが来やすい固定カメラを作る。

    ここで重要なのは lookat です。
    ボールの初期位置は x=0, y=0, z=2 ですが、
    落下全体が見えるように z=1.0 付近を見る設定にしています。
    """
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    # 画面の注視点。ここを変えると「どこを中央に置くか」が変わる。
    cam.lookat[:] = [0.0, 0.0, 1.0]

    # カメラ距離
    cam.distance = 5.5

    # 方位角（左右の回り込み）
    cam.azimuth = 90.0

    # 仰角（上下の見下ろし）
    cam.elevation = -12.0

    return cam


def simulate_condition(
    name: str,
    gravity: float,
    *,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    timestep: float,
    initial_height: float,
) -> list[np.ndarray]:
    """1つの重力条件をレンダリングし、フレーム列を返す。"""
    xml = MODEL_TEMPLATE.format(
        gravity=gravity,
        timestep=timestep,
        initial_height=initial_height,
        width=width,
        height=height,
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = make_camera()

    frames: list[np.ndarray] = []

    frame_interval = 1.0 / fps
    next_frame_time = 0.0

    while data.time < duration_seconds:
        mujoco.mj_step(model, data)

        if data.time >= next_frame_time:
            renderer.update_scene(data, camera=camera)
            frame = renderer.render()
            frame = add_label(frame, name)
            frames.append(frame)
            next_frame_time += frame_interval

    renderer.close()
    return frames


def save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    """フレーム列をMP4へ保存する。"""
    imageio.mimsave(
        path,
        frames,
        fps=fps,
        codec="libx264",
        quality=8,
    )


def main() -> None:
    """地球・火星・月の比較動画を出力する。"""
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    width = 640
    height = 480
    fps = 60
    duration_seconds = 3.0
    timestep = 0.002
    initial_height = 2.0

    all_frames: dict[str, list[np.ndarray]] = {}

    # 各条件をシミュレーション
    for name, gravity in CONDITIONS.items():
        print(f"Rendering: {name}")
        frames = simulate_condition(
            name=name,
            gravity=gravity,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration_seconds,
            timestep=timestep,
            initial_height=initial_height,
        )
        all_frames[name] = frames

        # 個別動画も保存
        save_video(VIDEO_DIR / f"{name.lower()}.mp4", frames, fps=fps)

    # 3条件のフレーム数をそろえる
    min_frames = min(len(frames) for frames in all_frames.values())

    comparison_frames: list[np.ndarray] = []

    for i in range(min_frames):
        earth = all_frames["Earth"][i]
        mars = all_frames["Mars"][i]
        moon = all_frames["Moon"][i]

        combined = np.concatenate([earth, mars, moon], axis=1)
        comparison_frames.append(combined)

    save_video(VIDEO_DIR / "comparison_earth_mars_moon.mp4", comparison_frames, fps=fps)

    print("Saved individual videos and comparison video.")
    print(f"Output dir: {VIDEO_DIR}")


if __name__ == "__main__":
    main()