#!/usr/bin/env python3
"""Locate the first divergent MJX forward-dynamics stage."""

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


def slice_data(data, count):
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[:count].copy(), data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    os.chdir(SOURCE)

    from mujoco.mjx._src import forward, sensor, solver
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    state, model, _ = load_checkpoint(args.checkpoint)
    saved_count = int(np.asarray(state.environment_state.done).shape[0])
    _, in_axes = randomize.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(20260730), saved_count)
    )

    def staged_one(m, d, target):
        d = d.replace(ctrl=target)
        position = forward.fwd_position(m, d)
        sensor_position = sensor.sensor_pos(m, position)
        velocity = forward.fwd_velocity(m, sensor_position)
        sensor_velocity = sensor.sensor_vel(m, velocity)
        actuation = forward.fwd_actuation(m, sensor_velocity)
        acceleration = forward.fwd_acceleration(m, actuation)
        solved = solver.solve(m, acceleration)
        sensor_acceleration = sensor.sensor_acc(m, solved)
        integrated = forward.euler(m, sensor_acceleration)
        return (
            position,
            sensor_position,
            velocity,
            sensor_velocity,
            actuation,
            acceleration,
            solved,
            sensor_acceleration,
            integrated,
        )

    names = (
        "fwd_position",
        "sensor_pos",
        "fwd_velocity",
        "sensor_vel",
        "fwd_actuation",
        "fwd_acceleration",
        "solver",
        "sensor_acc",
        "euler",
    )
    staged = jax.jit(jax.vmap(staged_one, in_axes=(in_axes, 0, 0)))

    def fresh():
        s, m, _ = load_checkpoint(args.checkpoint)
        return (
            slice_model(m, in_axes, args.batch_size),
            slice_data(s.environment_state.data, args.batch_size),
            np.asarray(s.environment_state.info["motor_targets"])[
                : args.batch_size
            ].copy(),
        )

    m, d, t = fresh()
    warm = staged(jax.device_put(m), jax.device_put(d), jax.device_put(t))
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warm)
    references = None
    rows = []
    leaf_rows = []
    for repeat in range(args.repeats):
        m, d, t = fresh()
        input_hash = tree_leaf_digest({"model": m, "data": d, "target": t})
        output = staged(jax.device_put(m), jax.device_put(d), jax.device_put(t))
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
        host = jax.tree_util.tree_map(
            lambda x: np.asarray(x).copy(), jax.device_get(output)
        )
        if references is None:
            references = host
        first_stage = ""
        for stage_index, (name, reference, candidate) in enumerate(
            zip(names, references, host)
        ):
            differences = compare_data(reference, candidate)
            if differences and not first_stage:
                first_stage = name
            rows.append(
                {
                    "batch_size": args.batch_size,
                    "repeat_id": repeat,
                    "stage_index": stage_index,
                    "stage": name,
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
                        "stage": name,
                        **difference,
                    }
                )

    for path, values in (
        (args.output / f"physics_stage_results_batch{args.batch_size}.csv", rows),
        (
            args.output / f"physics_stage_leaf_results_batch{args.batch_size}.csv",
            leaf_rows,
        ),
    ):
        fields = []
        for row in values:
            fields.extend(key for key in row if key not in fields)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    main()
