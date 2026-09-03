from pathlib import Path
import sys

import jax.numpy as jnp

EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT))

from training.comparison import compare_trees


def test_bit_comparison_detects_first_changed_leaf():
    reference = {"actor": jnp.asarray([1.0, 2.0]), "rng": jnp.asarray([1, 2])}
    same = {"actor": jnp.asarray([1.0, 2.0]), "rng": jnp.asarray([1, 2])}
    changed = {"actor": jnp.asarray([1.0, 2.1]), "rng": jnp.asarray([1, 2])}
    assert all(row["bit_exact"] for row in compare_trees(reference, same))
    assert not all(row["bit_exact"] for row in compare_trees(reference, changed))

