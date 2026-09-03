from pathlib import Path
import sys

import jax.numpy as jnp

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.device_metrics import aggregate_rollout


def test_device_reduction_does_not_mutate_inputs():
    sidecar = {
        "command": jnp.zeros((2, 3, 7)),
        "done": jnp.zeros((2, 3)),
        "fall": jnp.zeros((2, 3)),
        "truncation": jnp.zeros((2, 3)),
        "episode_start": jnp.zeros((2, 3)),
        "reward": jnp.ones((2, 3)),
        "reward_terms": {},
    }
    before = sidecar["command"]
    aggregate_rollout(
        sidecar,
        jnp.zeros((19, 3)),
        num_updates_per_batch=4,
        vx_edges=jnp.asarray([-0.1, 0.1]),
        vy_edges=jnp.asarray([-0.1, 0.1]),
        yaw_edges=jnp.asarray([-0.1, 0.1]),
        head_edges=jnp.asarray([-0.1, 0.1]),
    )
    assert jnp.array_equal(before, sidecar["command"])

