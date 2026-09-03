from pathlib import Path
import sys

import jax.numpy as jnp

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.checkpointing import load_checkpoint, save_checkpoint, tree_leaf_digest
from training.training_state import CounterState, ExactTrainingState


def test_training_state_roundtrip(tmp_path):
    counters = CounterState(
        optimizer_update_count=jnp.asarray(4),
        harness_update_count=jnp.asarray(1),
        global_environment_interactions=jnp.asarray(8),
        reset_generation=jnp.asarray([0, 0]),
        episode_index=jnp.asarray([0, 1]),
        episode_step=jnp.asarray([2, 0]),
    )
    state = ExactTrainingState(
        learner_state={"optimizer": jnp.arange(3), "normalizer": jnp.ones(2)},
        environment_state={"command": jnp.zeros((2, 7))},
        learner_rng=jnp.asarray([1, 2], dtype=jnp.uint32),
        rollout_rng=jnp.asarray([3, 4], dtype=jnp.uint32),
        evaluation_rng=jnp.asarray([5, 6], dtype=jnp.uint32),
        environment_keys=jnp.arange(4, dtype=jnp.uint32).reshape(2, 2),
        counters=counters,
    )
    model = {"mass": jnp.asarray([1.0, 2.0])}
    save_checkpoint(tmp_path / "checkpoint", state=state, randomized_model=model, metadata={})
    loaded, loaded_model, _ = load_checkpoint(tmp_path / "checkpoint")
    assert tree_leaf_digest(loaded) == tree_leaf_digest(state)
    assert tree_leaf_digest(loaded_model) == tree_leaf_digest(model)

