#!/usr/bin/env python3
"""Ablations for a serialized, fixed-control MJX training batch."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE = Path("/home/user/openduck_training_backward_v23_20260729")
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(SOURCE))

from training.checkpointing import load_checkpoint, tree_leaf_digest  # noqa: E402
from run_batched_mjx_reproducibility import compare_data, slice_model  # noqa: E402


def select_model(model: Any, in_axes: Any, index: int) -> Any:
    return jax.tree_util.tree_map(
        lambda x, axis: x[index] if axis == 0 else x,
        model,
        in_axes,
        is_leaf=lambda x: x is None,
    )


def repeat_model(model: Any, in_axes: Any, index: int, count: int) -> Any:
    return jax.tree_util.tree_map(
        lambda x, axis: None
        if x is None
        else (
            np.repeat(np.asarray(x[index : index + 1]), count, axis=0)
            if axis == 0
            else np.asarray(x).copy()
        ),
        model,
        in_axes,
        is_leaf=lambda x: x is None,
    )


def slice_data(data: Any, indices: np.ndarray) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x)[indices].copy(), data)


def stack_data(items: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *items)


def first_summary(reference: Any, comparison: Any) -> dict[str, Any]:
    rows = compare_data(reference, comparison)
    if not rows:
        return {
            "bit_exact": True,
            "first_field": "",
            "first_env": "",
            "max_abs_error": 0.0,
            "different_leaf_count": 0,
            "different_element_count": 0,
        }
    # PyTree path order is retained separately from physics-stage claims.
    first = rows[0]
    return {
        "bit_exact": False,
        "first_field": first["field_path"],
        "first_env": first["environment_index"],
        "max_abs_error": max(row["max_abs_error"] for row in rows),
        "different_leaf_count": len(rows),
        "different_element_count": sum(
            row["different_element_count"] for row in rows
        ),
    }


def repeat_condition(
    compiled: Any,
    payload_factory: Any,
    repeats: int,
) -> tuple[list[dict[str, Any]], list[Any]]:
    model, data, target = payload_factory()
    warm = compiled(jax.device_put(model), jax.device_put(data), jax.device_put(target))
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warm)
    outputs = []
    rows = []
    for repeat in range(repeats):
        model, data, target = payload_factory()
        before = tree_leaf_digest({"model": model, "data": data, "target": target})
        output = compiled(
            jax.device_put(model), jax.device_put(data), jax.device_put(target)
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
        host = jax.tree_util.tree_map(
            lambda x: np.asarray(x).copy(), jax.device_get(output)
        )
        after = tree_leaf_digest({"model": model, "data": data, "target": target})
        if before != after:
            raise RuntimeError("non-donated host input changed")
        summary = (
            first_summary(outputs[0], host)
            if outputs
            else {
                "bit_exact": True,
                "first_field": "",
                "first_env": "",
                "max_abs_error": 0.0,
                "different_leaf_count": 0,
                "different_element_count": 0,
            }
        )
        rows.append({"repeat_id": repeat, "input_hash": before, **summary})
        outputs.append(host)
    return rows, outputs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(SOURCE)

    from mujoco_playground._src import mjx_env
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    state, saved_model, _ = load_checkpoint(args.checkpoint)
    batch_count = int(np.asarray(state.environment_state.done).shape[0])
    _, in_axes = randomize.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(20260730), batch_count)
    )
    host_data = jax.device_get(state.environment_state.data)
    host_target = np.asarray(
        state.environment_state.info["motor_targets"]
    ).copy()

    def unbatched_step(model, data, target):
        return mjx_env.step(model, data, target, 1)

    single = jax.jit(unbatched_step)
    unbatched_results = []
    unbatched_outputs = []
    for index in range(batch_count):
        def factory(index=index):
            fresh_state, fresh_model, _ = load_checkpoint(args.checkpoint)
            return (
                select_model(fresh_model, in_axes, index),
                jax.tree_util.tree_map(
                    lambda x: np.asarray(x)[index].copy(),
                    fresh_state.environment_state.data,
                ),
                np.asarray(
                    fresh_state.environment_state.info["motor_targets"][index]
                ).copy(),
            )

        rows, outputs = repeat_condition(single, factory, args.repeats)
        for row in rows:
            unbatched_results.append({"environment_index": index, **row})
        unbatched_outputs.append(outputs[0])
    write_csv(args.output / "unbatched_repeatability.csv", unbatched_results)

    unbatched_concat = stack_data(unbatched_outputs)
    batched = jax.jit(
        jax.vmap(unbatched_step, in_axes=(in_axes, 0, 0))
    )

    def original_factory():
        fresh_state, fresh_model, _ = load_checkpoint(args.checkpoint)
        return (
            fresh_model,
            fresh_state.environment_state.data,
            np.asarray(fresh_state.environment_state.info["motor_targets"]).copy(),
        )

    batched_rows, batched_outputs = repeat_condition(
        batched, original_factory, args.repeats
    )
    write_csv(args.output / "batched_repeatability.csv", batched_rows)
    uvb_rows = []
    for repeat, output in enumerate(batched_outputs):
        uvb_rows.append(
            {
                "repeat_id": repeat,
                **first_summary(unbatched_concat, output),
            }
        )
    write_csv(args.output / "unbatched_vs_batched.csv", uvb_rows)

    reference_model = repeat_model(saved_model, in_axes, 0, batch_count)
    randomized_zero = repeat_model(saved_model, in_axes, 0, batch_count)
    randomized_one = repeat_model(saved_model, in_axes, 1, batch_count)
    same_data = slice_data(
        host_data, np.zeros(batch_count, dtype=np.int32)
    )
    same_target = np.repeat(host_target[0:1], batch_count, axis=0)
    ablations = {
        "M0_saved_model_env0_replicated": (
            reference_model,
            host_data,
            host_target,
        ),
        "M1_saved_randomized_models": (
            saved_model,
            host_data,
            host_target,
        ),
        "M2_saved_model_env1_replicated": (
            randomized_one,
            host_data,
            host_target,
        ),
        "M3_models_vary_data0_replicated": (
            saved_model,
            same_data,
            same_target,
        ),
        "M4_model0_data0_all_replicated": (
            randomized_zero,
            same_data,
            same_target,
        ),
    }
    model_rows = []
    for name, fixed in ablations.items():
        def factory(fixed=fixed):
            return jax.tree_util.tree_map(
                lambda x: None if x is None else np.asarray(x).copy(),
                fixed,
                is_leaf=lambda x: x is None,
            )

        rows, _ = repeat_condition(batched, factory, args.repeats)
        for row in rows:
            model_rows.append({"condition": name, **row})
    write_csv(args.output / "model_batch_ablation.csv", model_rows)

    permutations = {
        "identity": np.arange(batch_count),
        "reverse": np.arange(batch_count)[::-1],
        "fixed_A": np.asarray([2, 0, 3, 1], dtype=np.int32)[:batch_count],
        "fixed_B": np.asarray([1, 3, 0, 2], dtype=np.int32)[:batch_count],
        "parity": np.asarray([0, 2, 1, 3], dtype=np.int32)[:batch_count],
    }
    permutation_rows = []
    for name, permutation in permutations.items():
        inverse = np.argsort(permutation)

        def factory(permutation=permutation):
            fresh_state, fresh_model, _ = load_checkpoint(args.checkpoint)
            return (
                jax.tree_util.tree_map(
                    lambda x, axis: None
                    if x is None
                    else (
                        np.asarray(x)[permutation].copy()
                        if axis == 0
                        else np.asarray(x).copy()
                    ),
                    fresh_model,
                    in_axes,
                    is_leaf=lambda x: x is None,
                ),
                slice_data(fresh_state.environment_state.data, permutation),
                np.asarray(
                    fresh_state.environment_state.info["motor_targets"]
                )[permutation].copy(),
            )

        rows, outputs = repeat_condition(batched, factory, args.repeats)
        canonical = [
            slice_data(output, inverse) for output in outputs
        ]
        for row, output in zip(rows, canonical):
            versus_identity = first_summary(batched_outputs[0], output)
            permutation_rows.append(
                {
                    "permutation": name,
                    **row,
                    "vs_identity_first_field": versus_identity["first_field"],
                    "vs_identity_first_env": versus_identity["first_env"],
                    "vs_identity_max_abs_error": versus_identity["max_abs_error"],
                }
            )
    write_csv(args.output / "batch_permutation_results.csv", permutation_rows)

    # Contact/constraint group metadata from the exact input.
    data = state.environment_state.data
    ncon = np.asarray(getattr(data, "ncon", np.zeros(batch_count, dtype=int)))
    nefc = np.asarray(getattr(data, "nefc", np.zeros(batch_count, dtype=int)))
    contact_rows = []
    divergent_repeats_by_env = [set() for _ in range(batch_count)]
    max_by_env = np.zeros(batch_count)
    reference_leaves = jax.tree_util.tree_leaves(batched_outputs[0])
    for repeat_id, output in enumerate(batched_outputs[1:], start=1):
        candidate_leaves = jax.tree_util.tree_leaves(output)
        for reference_leaf, candidate_leaf in zip(
            reference_leaves, candidate_leaves
        ):
            reference_array = np.asarray(reference_leaf)
            candidate_array = np.asarray(candidate_leaf)
            if (
                reference_array.shape != candidate_array.shape
                or reference_array.ndim == 0
                or reference_array.shape[0] != batch_count
            ):
                continue
            unequal = reference_array != candidate_array
            finite_difference = np.abs(
                reference_array.astype(np.float64)
                - candidate_array.astype(np.float64)
            )
            for index in range(batch_count):
                if np.any(unequal[index]):
                    divergent_repeats_by_env[index].add(repeat_id)
                    max_by_env[index] = max(
                        max_by_env[index],
                        float(np.nanmax(finite_difference[index])),
                    )
    for index in range(batch_count):
        contact_rows.append(
            {
                "environment_index": index,
                "input_contact_count": int(ncon[index]) if ncon.ndim else int(ncon),
                "input_constraint_count": int(nefc[index]) if nefc.ndim else int(nefc),
                "divergent_repeat_count": len(divergent_repeats_by_env[index]),
                "divergence_rate": float(
                    len(divergent_repeats_by_env[index])
                    / max(args.repeats - 1, 1)
                ),
                "max_any_data_error": float(max_by_env[index]),
            }
        )
    write_csv(args.output / "contact_group_results.csv", contact_rows)

    # Donation is diagnostic-only and always uses a fresh disposable input.
    donated = jax.jit(
        jax.vmap(unbatched_step, in_axes=(in_axes, 0, 0)),
        donate_argnums=(1,),
    )
    donation_rows, _ = repeat_condition(donated, original_factory, args.repeats)
    write_csv(args.output / "production_donation_results.csv", donation_rows)

    hlo_dir = args.output / "hlo"
    hlo_dir.mkdir(exist_ok=True)
    model2 = slice_model(saved_model, in_axes, 2)
    data2 = slice_data(host_data, np.arange(2))
    target2 = host_target[:2]
    lowered = batched.lower(
        jax.device_put(model2), jax.device_put(data2), jax.device_put(target2)
    )
    hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    keywords = (
        "scatter", "reduce", "sort", "topk", "gather", "dynamic-update-slice",
        "atomic", "fusion",
    )
    excerpts = [
        line for line in hlo.splitlines() if any(key in line.lower() for key in keywords)
    ]
    (hlo_dir / "batch2_fixed_target_filtered_hlo.txt").write_text(
        "\n".join(excerpts) + "\n"
    )
    jaxpr_dir = args.output / "jaxpr"
    jaxpr_dir.mkdir(exist_ok=True)
    jaxpr_text = str(
        jax.make_jaxpr(
            jax.vmap(unbatched_step, in_axes=(in_axes, 0, 0))
        )(model2, data2, target2)
    )
    selected = [
        line for line in jaxpr_text.splitlines()
        if any(key in line.lower() for key in keywords)
    ]
    (jaxpr_dir / "batch2_fixed_target_filtered_jaxpr.txt").write_text(
        "\n".join(selected) + "\n"
    )


if __name__ == "__main__":
    main()
