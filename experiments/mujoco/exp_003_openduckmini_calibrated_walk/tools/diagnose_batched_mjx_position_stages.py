#!/usr/bin/env python3
"""Locate the first divergent operation inside MJX fwd_position."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/user/openduck_training_backward_v23_20260729")
sys.path[:0] = [str(EXPERIMENT), str(SOURCE), str(EXPERIMENT / "tools")]

from training.checkpointing import load_checkpoint, tree_leaf_digest  # noqa: E402
from run_batched_mjx_reproducibility import compare_data, slice_model  # noqa: E402


def _slice_data(data, count):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[:count].copy(), data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    os.chdir(SOURCE)

    from mujoco.mjx._src import collision_driver, constraint, smooth
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    state, _, _ = load_checkpoint(args.checkpoint)
    saved_count = int(np.asarray(state.environment_state.done).shape[0])
    _, in_axes = randomize.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(20260730), saved_count)
    )

    def staged_one(model, data, target):
        data = data.replace(ctrl=target)
        kinematics = smooth.kinematics(model, data)
        com_pos = smooth.com_pos(model, kinematics)
        camlight = smooth.camlight(model, com_pos)
        tendon = smooth.tendon(model, camlight)
        crb = smooth.crb(model, tendon)
        tendon_armature = smooth.tendon_armature(model, crb)
        factor_m = smooth.factor_m(model, tendon_armature)
        collision = collision_driver.collision(model, factor_m)
        constrained = constraint.make_constraint(model, collision)
        transmission = smooth.transmission(model, constrained)
        return (
            kinematics,
            com_pos,
            camlight,
            tendon,
            crb,
            tendon_armature,
            factor_m,
            collision,
            constrained,
            transmission,
        )

    names = (
        "kinematics",
        "com_pos",
        "camlight",
        "tendon",
        "crb",
        "tendon_armature",
        "factor_m",
        "collision",
        "make_constraint",
        "transmission",
    )
    staged = jax.jit(jax.vmap(staged_one, in_axes=(in_axes, 0, 0)))

    def fresh():
        fresh_state, fresh_model, _ = load_checkpoint(args.checkpoint)
        return (
            slice_model(fresh_model, in_axes, args.batch_size),
            _slice_data(fresh_state.environment_state.data, args.batch_size),
            np.asarray(fresh_state.environment_state.info["motor_targets"])[
                : args.batch_size
            ].copy(),
        )

    model, data, target = fresh()
    warm = staged(
        jax.device_put(model), jax.device_put(data), jax.device_put(target)
    )
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warm)

    references = None
    summary_rows = []
    leaf_rows = []
    for repeat in range(args.repeats):
        model, data, target = fresh()
        input_hash = tree_leaf_digest(
            {"model": model, "data": data, "target": target}
        )
        result = staged(
            jax.device_put(model), jax.device_put(data), jax.device_put(target)
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        host = jax.tree_util.tree_map(
            lambda x: np.asarray(x).copy(), jax.device_get(result)
        )
        if references is None:
            references = host
        first_stage = ""
        for stage_index, (stage, reference, candidate) in enumerate(
            zip(names, references, host)
        ):
            differences = compare_data(reference, candidate)
            if differences and not first_stage:
                first_stage = stage
            summary_rows.append(
                {
                    "batch_size": args.batch_size,
                    "repeat_id": repeat,
                    "stage_index": stage_index,
                    "stage": stage,
                    "bit_exact": not differences,
                    "different_leaf_count": len(differences),
                    "max_abs_error": max(
                        (row["max_abs_error"] for row in differences), default=0.0
                    ),
                    "first_divergent_stage_for_repeat": first_stage,
                    "input_hash": input_hash,
                }
            )
            for difference in differences:
                leaf_rows.append(
                    {
                        "batch_size": args.batch_size,
                        "repeat_id": repeat,
                        "stage_index": stage_index,
                        "stage": stage,
                        **difference,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        (
            f"position_stage_results_batch{args.batch_size}.csv",
            summary_rows,
        ),
        (
            f"position_stage_leaf_results_batch{args.batch_size}.csv",
            leaf_rows,
        ),
    ):
        fields = []
        for row in rows:
            fields.extend(key for key in row if key not in fields)
        with (args.output / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
