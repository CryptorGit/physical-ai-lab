"""Serializable state contract for update-boundary PPO resume."""

from __future__ import annotations

from typing import Any

from flax import struct
import jax


@struct.dataclass
class CounterState:
    """Counters not represented by Brax's learner state."""

    optimizer_update_count: jax.Array
    harness_update_count: jax.Array
    global_environment_interactions: jax.Array
    reset_generation: jax.Array
    episode_index: jax.Array
    episode_step: jax.Array


@struct.dataclass
class ExactTrainingState:
    """Every dynamic value required to continue at a completed update boundary."""

    learner_state: Any
    environment_state: Any
    learner_rng: jax.Array
    rollout_rng: jax.Array
    evaluation_rng: jax.Array
    environment_keys: jax.Array
    counters: CounterState


def make_counters(num_envs: int) -> CounterState:
    """Creates explicit lineage counters without consuming an RNG key."""
    import jax.numpy as jnp

    return CounterState(
        optimizer_update_count=jnp.asarray(0, dtype=jnp.int32),
        harness_update_count=jnp.asarray(0, dtype=jnp.int32),
        global_environment_interactions=jnp.asarray(0, dtype=jnp.int32),
        reset_generation=jnp.zeros((num_envs,), dtype=jnp.int32),
        episode_index=jnp.zeros((num_envs,), dtype=jnp.int32),
        episode_step=jnp.zeros((num_envs,), dtype=jnp.int32),
    )
