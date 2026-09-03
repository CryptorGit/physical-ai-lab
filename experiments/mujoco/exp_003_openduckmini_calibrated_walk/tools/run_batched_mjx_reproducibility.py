#!/usr/bin/env python3
"""Fixed-action, fresh-load MJX one-step reproducibility diagnostic.

This tool performs no optimizer update and never modifies production code.
Every measured invocation reloads the serialized host payload before device_put.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[2]
SOURCE = Path("/home/user/openduck_training_backward_v23_20260729")
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(SOURCE))

from training.checkpointing import load_checkpoint, tree_leaf_digest  # noqa: E402


STAGE_ORDER = (
    "ctrl",
    "act",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "xanchor",
    "xaxis",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "subtree_com",
    "cinert",
    "cdof",
    "cvel",
    "cdof_dot",
    "cacc",
    "cfrc_int",
    "cfrc_ext",
    "contact",
    "efc",
    "qacc",
    "qacc_warmstart",
    "qvel",
    "qpos",
    "sensordata",
    "time",
)


def sha_array(value: Any) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_path(path: Any) -> str:
    parts = []
    for key in path:
        value = getattr(key, "key", getattr(key, "idx", getattr(key, "name", "")))
        parts.append(str(value))
    return "/".join(parts)


def host_copy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), jax.device_get(tree))


def slice_tree(tree: Any, count: int) -> Any:
    return jax.tree_util.tree_map(lambda x: x[:count], tree)


def slice_model(model: Any, in_axes: Any, count: int) -> Any:
    return jax.tree_util.tree_map(
        lambda x, axis: x[:count] if axis == 0 else x,
        model,
        in_axes,
        is_leaf=lambda x: x is None,
    )


def compare_data(reference: Any, comparison: Any) -> list[dict[str, Any]]:
    left, left_def = jax.tree_util.tree_flatten_with_path(reference)
    right, right_def = jax.tree_util.tree_flatten_with_path(comparison)
    if left_def != right_def:
        raise ValueError("data PyTree definitions differ")
    rows = []
    for (path_a, value_a), (path_b, value_b) in zip(left, right):
        if path_a != path_b:
            raise ValueError("data paths differ")
        a = np.asarray(value_a)
        b = np.asarray(value_b)
        if not np.issubdtype(a.dtype, np.number):
            continue
        unequal = np.frombuffer(a.tobytes(order="C"), dtype=np.uint8).reshape(
            a.shape + (a.dtype.itemsize,)
        ) != np.frombuffer(b.tobytes(order="C"), dtype=np.uint8).reshape(
            b.shape + (b.dtype.itemsize,)
        )
        element_unequal = np.any(unequal, axis=-1)
        if not np.any(element_unequal):
            continue
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if a.ndim:
            env_mask = np.any(element_unequal.reshape((a.shape[0], -1)), axis=1)
            first_env = int(np.flatnonzero(env_mask)[0])
            local = diff[first_env]
            local_index = np.unravel_index(np.nanargmax(local), local.shape)
            element_index = [int(x) for x in local_index]
        else:
            first_env = -1
            element_index = []
        maximum = np.nanmax(diff)
        max_index = np.unravel_index(np.nanargmax(diff), diff.shape)
        rows.append(
            {
                "field_path": stable_path(path_a),
                "environment_index": first_env,
                "element_index": json.dumps(element_index),
                "max_abs_error": float(maximum),
                "different_element_count": int(np.count_nonzero(element_unequal)),
                "reference_value_at_max": float(a[max_index]),
                "comparison_value_at_max": float(b[max_index]),
                "dtype": str(a.dtype),
            }
        )
    return rows


def field_rank(path: str) -> tuple[int, str]:
    leaf = path.split("/")[-1]
    for index, token in enumerate(STAGE_ORDER):
        if leaf == token or leaf.startswith(token + "."):
            return index, path
    return len(STAGE_ORDER), path


def input_payload(checkpoint: Path, count: int, in_axes: Any):
    state, model, manifest = load_checkpoint(checkpoint)
    data = slice_tree(state.environment_state.data, count)
    model = slice_model(model, in_axes, count)
    target = np.asarray(state.environment_state.info["motor_targets"])[:count].copy()
    return state, model, data, target, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument(
        "--n-substeps",
        type=int,
        default=None,
        help="Override control-step substeps; 1 isolates the first MJX simulator step.",
    )
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    os.chdir(SOURCE)

    from mujoco_playground._src import mjx_env
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    n_substeps = env.n_substeps if args.n_substeps is None else args.n_substeps
    initial_state, saved_model, source_manifest = load_checkpoint(args.checkpoint)
    saved_count = int(np.asarray(initial_state.environment_state.done).shape[0])
    _, in_axes = randomize.domain_randomize(
        env.mjx_model, jax.random.split(jax.random.PRNGKey(20260730), saved_count)
    )

    identity = {
        "source_checkpoint": str(args.checkpoint),
        "source_payload_sha256": source_manifest["payload_sha256"],
        "training_state_hash": tree_leaf_digest(initial_state),
        "mjx_data_hash": tree_leaf_digest(initial_state.environment_state.data),
        "mjx_model_hash": tree_leaf_digest(saved_model),
        "rng_hash": tree_leaf_digest(
            {
                "learner": initial_state.learner_rng,
                "rollout": initial_state.rollout_rng,
                "evaluation": initial_state.evaluation_rng,
                "per_environment": initial_state.environment_state.info["rng"],
            }
        ),
        "command_state_hash": tree_leaf_digest(
            initial_state.environment_state.info["command"]
        ),
        "controller_state_hash": tree_leaf_digest(
            {
                "action_history": initial_state.environment_state.info[
                    "action_history"
                ],
                "motor_targets": initial_state.environment_state.info[
                    "motor_targets"
                ],
                "imitation_i": initial_state.environment_state.info["imitation_i"],
                "imitation_phase": initial_state.environment_state.info[
                    "imitation_phase"
                ],
            }
        ),
        "serialized_environment_count": saved_count,
        "input_host_bytes": int(
            sum(
                np.asarray(x).nbytes
                for x in jax.tree_util.tree_leaves(
                    jax.device_get(
                        {
                            "state": initial_state,
                            "model": saved_model,
                        }
                    )
                )
            )
        ),
    }
    (args.output / "input_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n"
    )

    ladder_rows = []
    leaf_rows = []
    for count in [int(x) for x in args.batch_sizes.split(",")]:
        if count > saved_count:
            ladder_rows.append(
                {
                    "batch_size": count,
                    "repeat_id": "",
                    "status": "NOT_APPLICABLE_EXCEEDS_SERIALIZED_BATCH",
                    "first_divergent_env": "",
                    "first_divergent_field": "",
                    "max_abs_error": "",
                    "different_leaf_count": "",
                    "different_element_count": "",
                }
            )
            continue

        def step(model, data, target):
            return jax.vmap(
                lambda one_model, one_data, one_target: mjx_env.step(
                    one_model, one_data, one_target, env.n_substeps
                    if args.n_substeps is None
                    else n_substeps
                ),
                in_axes=(in_axes, 0, 0),
            )(model, data, target)

        compiled = jax.jit(step)
        # Compilation and autotuning use a disposable fresh disk load.
        _, warm_model, warm_data, warm_target, _ = input_payload(
            args.checkpoint, count, in_axes
        )
        warm = compiled(
            jax.device_put(warm_model),
            jax.device_put(warm_data),
            jax.device_put(warm_target),
        )
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), warm)

        reference = None
        for repeat in range(args.repeats):
            _, model, data, target, _ = input_payload(
                args.checkpoint, count, in_axes
            )
            input_hash_before = tree_leaf_digest(
                {"model": model, "data": data, "target": target}
            )
            output = compiled(
                jax.device_put(model),
                jax.device_put(data),
                jax.device_put(target),
            )
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), output)
            host_output = host_copy(output)
            input_hash_after = tree_leaf_digest(
                {"model": model, "data": data, "target": target}
            )
            if input_hash_before != input_hash_after:
                raise RuntimeError("host input changed across non-donated call")
            if reference is None:
                reference = host_output
                rows = []
            else:
                rows = compare_data(reference, host_output)
            ordered = sorted(rows, key=lambda row: field_rank(row["field_path"]))
            first = ordered[0] if ordered else None
            ladder_rows.append(
                {
                    "batch_size": count,
                    "repeat_id": repeat,
                    "status": "DIVERGED" if first else "BIT_EXACT",
                    "first_divergent_env": (
                        first["environment_index"] if first else ""
                    ),
                    "first_divergent_field": first["field_path"] if first else "",
                    "max_abs_error": max(
                        (row["max_abs_error"] for row in rows), default=0.0
                    ),
                    "different_leaf_count": len(rows),
                    "different_element_count": sum(
                        row["different_element_count"] for row in rows
                    ),
                    "input_hash": input_hash_before,
                    "output_hash": tree_leaf_digest(host_output),
                }
            )
            for row in rows:
                leaf_rows.append(
                    {"batch_size": count, "repeat_id": repeat, **row}
                )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = []
        for row in rows:
            fields.extend(key for key in row if key not in fields)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output / "batch_size_ladder.csv", ladder_rows)
    write_csv(args.output / "first_divergent_leaf.csv", leaf_rows)
    runtime = {
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", "unknown"),
        "device": [str(x) for x in jax.devices()],
        "x64": bool(jax.config.jax_enable_x64),
        "jit_disabled": bool(jax.config.jax_disable_jit),
        "main_commit": subprocess.check_output(
            ["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"], text=True
        ).strip(),
        "main_branch": subprocess.check_output(
            ["git", "-C", str(WORKSPACE), "branch", "--show-current"], text=True
        ).strip(),
        "training_source_commit": subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (args.output / "runtime.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
