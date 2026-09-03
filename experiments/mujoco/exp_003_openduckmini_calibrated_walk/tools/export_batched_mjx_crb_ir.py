#!/usr/bin/env python3
"""Export narrowly scoped JAXPR/HLO for the first divergent MJX CRB stage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/user/openduck_training_backward_v23_20260729")
sys.path[:0] = [str(EXPERIMENT), str(SOURCE), str(EXPERIMENT / "tools")]

from training.checkpointing import load_checkpoint  # noqa: E402
from run_batched_mjx_reproducibility import slice_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    os.chdir(SOURCE)

    from mujoco.mjx._src import smooth
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    state, saved_model, _ = load_checkpoint(args.checkpoint)
    saved_count = int(np.asarray(state.environment_state.done).shape[0])
    _, in_axes = randomize.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(20260730), saved_count)
    )
    model = slice_model(saved_model, in_axes, args.batch_size)
    data = jax.tree_util.tree_map(
        lambda x: np.asarray(x)[: args.batch_size].copy(),
        state.environment_state.data,
    )

    def before_crb(single_model, single_data):
        single_data = smooth.kinematics(single_model, single_data)
        single_data = smooth.com_pos(single_model, single_data)
        single_data = smooth.camlight(single_model, single_data)
        return smooth.tendon(single_model, single_data)

    before = jax.jit(jax.vmap(before_crb, in_axes=(in_axes, 0)))
    tendon_data = before(jax.device_put(model), jax.device_put(data))
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), tendon_data)

    crb = jax.jit(jax.vmap(smooth.crb, in_axes=(in_axes, 0)))
    lowered = crb.lower(jax.device_put(model), tendon_data)
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    jaxpr = str(jax.make_jaxpr(jax.vmap(smooth.crb, in_axes=(in_axes, 0)))(
        model, jax.device_get(tendon_data)
    ))

    args.output.mkdir(parents=True, exist_ok=True)
    tokens = (
        "scatter",
        "reduce",
        "gather",
        "sort",
        "dynamic-update-slice",
        "atomic",
    )
    (args.output / "batch2_crb_filtered_hlo.txt").write_text(
        "\n".join(line for line in hlo.splitlines() if any(t in line.lower() for t in tokens))
        + "\n"
    )
    (args.output / "batch2_crb_filtered_jaxpr.txt").write_text(
        "\n".join(
            line
            for line in jaxpr.splitlines()
            if any(t in line.lower() for t in tokens)
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
