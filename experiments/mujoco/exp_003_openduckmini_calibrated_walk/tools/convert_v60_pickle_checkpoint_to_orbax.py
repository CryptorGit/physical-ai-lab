"""Convert a completed v60 host checkpoint to evaluator-compatible Orbax."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
from flax.training import orbax_utils
from orbax import checkpoint as ocp


if not hasattr(jax.monitoring, "record_scalar"):
    jax.monitoring.record_scalar = lambda *_args, **_kwargs: None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("rb") as stream:
        params = pickle.load(stream)
    params = jax.tree_util.tree_map(jax.device_put, params)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(
        str(output_path), params, force=True, save_args=save_args
    )
    restored = checkpointer.restore(str(output_path))
    source_leaves = jax.tree_util.tree_leaves(params)
    restored_leaves = jax.tree_util.tree_leaves(restored)
    if len(source_leaves) != len(restored_leaves):
        raise RuntimeError("Orbax round-trip leaf-count mismatch")
    for source, target in zip(source_leaves, restored_leaves):
        np.testing.assert_array_equal(np.asarray(source), np.asarray(target))
    print(output_path)


if __name__ == "__main__":
    main()
