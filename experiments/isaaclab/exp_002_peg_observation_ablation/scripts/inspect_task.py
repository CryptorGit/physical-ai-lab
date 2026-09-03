from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the official Isaac Lab Factory PegInsert task."
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401,E402
import peg_observation_ablation  # noqa: F401,E402


TASK_ID = "Isaac-PegObservationNoAngvel-Direct-v0"


def main() -> None:
    spec = gym.spec(TASK_ID)

    print(f"Task ID: {TASK_ID}")
    print(f"Entry point: {spec.entry_point}")
    print(f"Registered kwargs: {spec.kwargs}")

    env = gym.make(
        TASK_ID,
        cfg=spec.kwargs.get("env_cfg_entry_point"),
        render_mode=None,
    )

    try:
        print(f"Environment class: {type(env.unwrapped).__module__}.{type(env.unwrapped).__name__}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
        print(f"Environment cfg type: {type(env.unwrapped.cfg)}")

        observations, info = env.reset()
        print(f"Reset observation type: {type(observations)}")

        if isinstance(observations, dict):
            for key, value in observations.items():
                shape = getattr(value, "shape", None)
                print(f"Observation[{key!r}] shape: {shape}")
        else:
            print(f"Observation shape: {getattr(observations, 'shape', None)}")

        print(f"Reset info keys: {list(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()