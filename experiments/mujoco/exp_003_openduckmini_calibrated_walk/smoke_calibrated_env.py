"""Minimal JAX/MJX smoke test for the calibrated Playground task."""

from __future__ import annotations

import jax
import jax.numpy as jp

from playground.open_duck_mini_v2.joystick import Joystick


def main() -> None:
    environment = Joystick(task="flat_terrain_backlash_calibrated")
    state = environment.reset(jax.random.PRNGKey(0))
    print("observation_shape", state.obs["state"].shape)
    print("action_size", environment.action_size)
    print("home", environment._default_actuator)
    print("qpos", environment.get_actuator_joints_qpos(state.data.qpos))
    state = environment.step(state, jp.zeros(environment.action_size))
    state.reward.block_until_ready()
    print("reward", float(state.reward))
    print("done", float(state.done))
    print("root_z", float(state.data.qpos[2]))
    print("devices", jax.devices())


if __name__ == "__main__":
    main()
