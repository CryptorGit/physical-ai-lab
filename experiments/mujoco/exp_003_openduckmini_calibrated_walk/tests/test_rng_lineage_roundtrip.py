from pathlib import Path
import sys

import jax
import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.checkpointing import load_checkpoint, save_checkpoint
from training.training_state import CounterState, ExactTrainingState


def test_rng_keys_are_not_advanced_by_checkpoint(tmp_path):
    key = jax.random.PRNGKey(7)
    learner, rollout, evaluation = jax.random.split(key, 3)
    counters = CounterState(*([np.asarray(0, dtype=np.int32)] * 6))
    state = ExactTrainingState(
        learner_state={},
        environment_state={},
        learner_rng=learner,
        rollout_rng=rollout,
        evaluation_rng=evaluation,
        environment_keys=jax.random.split(key, 2),
        counters=counters,
    )
    save_checkpoint(tmp_path / "c", state=state, randomized_model={}, metadata={})
    loaded, _, _ = load_checkpoint(tmp_path / "c")
    assert np.array_equal(loaded.learner_rng, learner)
    assert np.array_equal(loaded.rollout_rng, rollout)
    assert np.array_equal(loaded.evaluation_rng, evaluation)

