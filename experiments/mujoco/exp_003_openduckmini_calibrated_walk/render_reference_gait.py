from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from env import OpenDuckCalibratedWalkEnv


HERE = Path(__file__).resolve().parent


def main() -> None:
    env = OpenDuckCalibratedWalkEnv(
        seed=29,
        episode_steps=600,
        command_velocity=0.10,
        reference_residual=True,
    )
    env.reset()
    env.model.vis.global_.offwidth = 640
    env.model.vis.global_.offheight = 640
    renderer = mujoco.Renderer(env.model, height=640, width=640)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 0.72
    camera.azimuth = 90
    camera.elevation = -12

    output_dir = HERE / "artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibrated_reference_gait_side.mp4"

    zero_residual = np.zeros(10, dtype=np.float32)
    with imageio.get_writer(
        output_path,
        fps=50,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for _ in range(600):
            _, _, terminated, truncated, _ = env.step(zero_residual)
            camera.lookat[:] = (
                float(env.data.qpos[0]),
                float(env.data.qpos[1]),
                0.19,
            )
            renderer.update_scene(env.data, camera=camera)
            writer.append_data(renderer.render())
            if terminated or truncated:
                break

    renderer.close()
    env.close()
    print(output_path)


if __name__ == "__main__":
    main()
