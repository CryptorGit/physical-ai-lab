"""学習済みPPOモデルでUnitree Go2を再生する。"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from env import Go2FlatEnv


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "checkpoints" / "go2_flat_final.zip"


def main() -> None:
    """学習済みモデルを読み込み、MuJoCo Viewerで再生する。"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"学習済みモデルがありません: {MODEL_PATH}"
        )

    env = Go2FlatEnv()
    model = PPO.load(MODEL_PATH, env=env)

    observation, info = env.reset(seed=42)

    control_dt = (
        env.model.opt.timestep
        * env.frame_skip
    )

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
    ) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

            viewer.sync()

            if terminated or truncated:
                print(
                    f"Episode ended: "
                    f"height={info['base_height']:.3f}, "
                    f"vx={info['forward_velocity']:.3f}"
                )

                observation, info = env.reset()

            elapsed = time.perf_counter() - step_start
            sleep_time = control_dt - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()