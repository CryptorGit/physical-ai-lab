#!/usr/bin/env python3
"""Compare all v59 MJX one-step leaves and downstream controller effects."""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint, networks as ppo_networks
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground.config import locomotion_params

from playground.open_duck_mini_v2 import joystick

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from export_v59_stochastic_trace import (
    observation_noise_vector,
    raw_observation_before_noise,
)
from v59_mjx_diagnostic_common import (
    canonical_tree_sha256,
    load_pickle,
    numeric_leaves,
)
from v59_parity_common import compose_motor_target
from v59_stochastic_common import delay_buffer_step


STAGE_ORDER = {
    "time": 0,
    "ctrl": 0,
    "qfrc_applied": 0,
    "xfrc_applied": 0,
    "xpos": 1,
    "xquat": 1,
    "xmat": 1,
    "xipos": 1,
    "ximat": 1,
    "xanchor": 1,
    "xaxis": 1,
    "geom_xpos": 1,
    "geom_xmat": 1,
    "site_xpos": 1,
    "site_xmat": 1,
    "subtree_com": 1,
    "_impl.cinert": 1,
    "cvel": 1,
    "cdof": 1,
    "cdof_dot": 1,
    "_impl.contact": 2,
    "_impl.efc": 3,
    "_impl.solver_niter": 4,
    "qfrc_constraint": 4,
    "_impl.cfrc": 4,
    "qacc": 5,
    "qacc_warmstart": 5,
    "qvel": 6,
    "qpos": 7,
}
STAGE_NAMES = {
    0: "input/control copy",
    1: "kinematics/velocity output",
    2: "collision/contact output",
    3: "constraint construction output",
    4: "solver/constraint-force output",
    5: "acceleration output",
    6: "velocity integration output",
    7: "position integration output",
    99: "other derived output",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric_row(
    comparison: str,
    case_id: str,
    field: str,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    a = np.asarray(reference)
    b = np.asarray(candidate)
    if a.shape != b.shape:
        return {
            "comparison": comparison,
            "case_id": case_id,
            "field_path": field,
            "shape": str(a.shape),
            "dtype": str(a.dtype),
            "max_abs_error": "shape_mismatch",
            "mean_abs_error": "",
            "rmse": "",
            "max_relative_error": "",
            "nonzero_difference_count": "",
            "nan_count_reference": int(np.isnan(a).sum()) if a.dtype.kind in "fc" else 0,
            "nan_count_comparison": int(np.isnan(b).sum()) if b.dtype.kind in "fc" else 0,
            "inf_count_reference": int(np.isinf(a).sum()) if a.dtype.kind in "fc" else 0,
            "inf_count_comparison": int(np.isinf(b).sum()) if b.dtype.kind in "fc" else 0,
            "first_different_index": "",
            "reference_value": "",
            "comparison_value": "",
            "bit_exact": False,
        }
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        aa = a.astype(np.complex128 if a.dtype.kind == "c" else np.float64)
        bb = b.astype(np.complex128 if b.dtype.kind == "c" else np.float64)
        difference = np.abs(aa - bb)
        finite = np.isfinite(difference)
        safe = difference[finite]
        max_abs = float(safe.max()) if safe.size else 0.0
        mean_abs = float(safe.mean()) if safe.size else 0.0
        rmse = float(np.sqrt(np.mean(np.square(safe)))) if safe.size else 0.0
        denominator = np.maximum(np.abs(aa), np.finfo(np.float64).tiny)
        relative = difference / denominator
        finite_relative = relative[np.isfinite(relative)]
        max_relative = (
            float(finite_relative.max()) if finite_relative.size else 0.0
        )
        bit_exact = bool(np.array_equal(a, b, equal_nan=True))
        value_different = ~np.isclose(a, b, rtol=0.0, atol=0.0, equal_nan=True)
    else:
        difference = (a != b).astype(np.float64)
        max_abs = float(difference.max()) if difference.size else 0.0
        mean_abs = float(difference.mean()) if difference.size else 0.0
        rmse = float(np.sqrt(np.mean(difference))) if difference.size else 0.0
        max_relative = max_abs
        value_different = a != b
        bit_exact = bool(np.array_equal(a, b))
    indices = np.argwhere(value_different)
    first_index = (
        tuple(int(index) for index in indices[0]) if len(indices) else None
    )
    return {
        "comparison": comparison,
        "case_id": case_id,
        "field_path": field,
        "shape": json.dumps(list(a.shape)),
        "dtype": str(a.dtype),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "rmse": rmse,
        "max_relative_error": max_relative,
        "nonzero_difference_count": int(np.count_nonzero(value_different)),
        "nan_count_reference": int(np.isnan(a).sum()) if a.dtype.kind in "fc" else 0,
        "nan_count_comparison": int(np.isnan(b).sum()) if b.dtype.kind in "fc" else 0,
        "inf_count_reference": int(np.isinf(a).sum()) if a.dtype.kind in "fc" else 0,
        "inf_count_comparison": int(np.isinf(b).sum()) if b.dtype.kind in "fc" else 0,
        "first_different_index": json.dumps(first_index),
        "reference_value": (
            np.asarray(a[first_index]).item() if first_index is not None else ""
        ),
        "comparison_value": (
            np.asarray(b[first_index]).item() if first_index is not None else ""
        ),
        "bit_exact": bit_exact,
    }


def compare_trees(
    comparison: str, case_id: str, reference: Any, candidate: Any
) -> list[dict[str, Any]]:
    left = numeric_leaves(reference)
    right = numeric_leaves(candidate)
    rows = []
    for field in sorted(set(left) | set(right)):
        if field not in left or field not in right:
            continue
        rows.append(metric_row(comparison, case_id, field, left[field], right[field]))
    # Static MJX structure counters are intentionally not dynamic PyTree leaves.
    for field in ("ne", "nf", "nl", "nefc", "ncon"):
        rows.append(
            metric_row(
                comparison,
                case_id,
                f"_impl.{field}",
                np.asarray(getattr(reference._impl, field)),
                np.asarray(getattr(candidate._impl, field)),
            )
        )
    return rows


def active_contact_signature(data: Any) -> tuple[list, list, list]:
    contact = data._impl.contact
    dist = np.asarray(contact.dist)
    margin = np.asarray(contact.includemargin)
    active = dist < margin
    pairs = np.asarray(contact.geom)[active]
    entries = sorted(
        (
            int(min(pair)),
            int(max(pair)),
            float(distance),
        )
        for pair, distance in zip(pairs, dist[active])
    )
    pair_only = [(entry[0], entry[1]) for entry in entries]
    return active.tolist(), pair_only, entries


def discrete_row(
    comparison: str,
    case_id: str,
    reference: Any,
    candidate: Any,
    env: Any,
) -> dict[str, Any]:
    left_active, left_pairs, left_entries = active_contact_signature(reference)
    right_active, right_pairs, right_entries = active_contact_signature(candidate)
    left_device = jax.device_put(reference)
    right_device = jax.device_put(candidate)
    left_feet = [
        bool(np.asarray(geoms_colliding(left_device, geom, env._floor_geom_id)))
        for geom in env._feet_geom_id
    ]
    right_feet = [
        bool(np.asarray(geoms_colliding(right_device, geom, env._floor_geom_id)))
        for geom in env._feet_geom_id
    ]
    left_done = bool(np.asarray(env._get_termination(left_device)))
    right_done = bool(np.asarray(env._get_termination(right_device)))
    return {
        "comparison": comparison,
        "case_id": case_id,
        "ncon_reference": int(reference._impl.ncon),
        "ncon_comparison": int(candidate._impl.ncon),
        "nefc_reference": int(reference._impl.nefc),
        "nefc_comparison": int(candidate._impl.nefc),
        "solver_niter_reference": int(np.asarray(reference._impl.solver_niter)),
        "solver_niter_comparison": int(np.asarray(candidate._impl.solver_niter)),
        "active_contact_mask_equal": left_active == right_active,
        "active_contact_pairs_equal": left_pairs == right_pairs,
        "active_contact_pairs_reference": json.dumps(left_pairs),
        "active_contact_pairs_comparison": json.dumps(right_pairs),
        "active_contact_entries_reference": json.dumps(left_entries),
        "active_contact_entries_comparison": json.dumps(right_entries),
        "foot_contact_reference": json.dumps(left_feet),
        "foot_contact_comparison": json.dumps(right_feet),
        "foot_contact_equal": left_feet == right_feet,
        "termination_reference": left_done,
        "termination_comparison": right_done,
        "termination_equal": left_done == right_done,
        "fall_flag_reference": left_done,
        "fall_flag_comparison": right_done,
        "joint_limit_structure_available": False,
    }


def field_stage(field: str) -> int:
    best = 99
    for prefix, stage in STAGE_ORDER.items():
        if field == prefix or field.startswith(prefix + "."):
            best = min(best, stage)
    return best


def first_divergence(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    differing = [
        row
        for row in rows
        if row["bit_exact"] is False
        and isinstance(row["max_abs_error"], (int, float))
        and row["max_abs_error"] != 0
    ]
    if not differing:
        return None
    differing.sort(
        key=lambda row: (
            field_stage(row["field_path"]),
            row["field_path"],
        )
    )
    row = dict(differing[0])
    stage = field_stage(row["field_path"])
    row["stage_index"] = stage
    row["stage"] = STAGE_NAMES[stage]
    return row


def output(path: Path) -> Any:
    return load_pickle(path)["data"]


def next_controller_values(
    env: Any,
    params: Any,
    policy_apply: Any,
    payload: dict[str, Any],
    data: Any,
) -> dict[str, np.ndarray]:
    step0 = payload["trace_step0"]
    step1 = payload["trace_step1"]
    before_info = dict(payload["state"].info)
    after_info = dict(before_info)
    after_info["command"] = jp.asarray(step0["raw_command"])
    after_info["motor_targets"] = jp.asarray(step0["motor_target_after_backlash"])
    after_info["imitation_phase"] = jp.asarray(step0["teacher_phase"])
    data_device = jax.device_put(data)
    contact = jp.asarray(
        [
            geoms_colliding(data_device, geom, env._floor_geom_id)
            for geom in env._feet_geom_id
        ]
    )
    raw = raw_observation_before_noise(
        env, data_device, before_info, after_info, contact
    )
    noisy = raw + np.asarray(step0["noise_sample"], dtype=np.float32)
    obs = {
        "state": jp.asarray(noisy),
        "privileged_state": jp.asarray(noisy),
    }
    logits = policy_apply(params[0], params[1], obs)
    loc, raw_scale = jp.split(logits, 2, axis=-1)
    eps = jp.asarray(step1["actor_standard_normal_sample"])
    actor = jp.tanh(loc + (jax.nn.softplus(raw_scale) + 0.001) * eps)

    history, delayed = delay_buffer_step(
        np.asarray(step0["delay_buffer_state"]),
        np.asarray(actor),
        int(np.asarray(step1["delay_length"])),
    )
    command = np.asarray(step0["raw_command"])
    _, _, phase_rate = env._get_backward_parameters(command[2])
    rate = float(np.asarray(phase_rate)) if command[0] < -0.02 else 1.0
    phase1 = rate % env.PRM.nb_steps_in_period
    phase2 = (phase1 + rate) % env.PRM.nb_steps_in_period
    reference = env._get_optimized_backward_reference(phase2, command[2])
    composed = compose_motor_target(
        delayed,
        command=command,
        default=np.asarray(env._default_actuator),
        lower=np.asarray(env._actuator_lowers),
        upper=np.asarray(env._actuator_uppers),
        action_scale=float(env._config.action_scale),
        previous_target=np.asarray(step0["motor_target_after_backlash"]),
        max_motor_velocity=np.asarray(env._config.max_motor_velocity),
        dt=float(env.dt),
        backward_reference=np.asarray(reference),
        backward_actuator_indices=np.asarray(env._backward_actuator_indices),
        backward_joint_indices=np.asarray(env._backward_joint_indices),
        backward_residual_scale=float(env._backward_residual_scale),
        coupled_slope=float(joystick.HEAD_COUPLED_REAR_SLOPE),
        coupled_intercept=float(joystick.HEAD_COUPLED_REAR_INTERCEPT),
    )
    mean = np.asarray(params[0].mean["state"])
    std = np.asarray(params[0].std["state"])
    normalized = (noisy - mean) / std
    return {
        "next_raw_observation": np.asarray(raw),
        "next_noisy_observation": np.asarray(noisy),
        "next_normalized_observation": normalized,
        "next_teacher_phase": np.asarray(
            [
                np.cos(phase2 / env.PRM.nb_steps_in_period * 2 * np.pi),
                np.sin(phase2 / env.PRM.nb_steps_in_period * 2 * np.pi),
            ],
            dtype=np.float32,
        ),
        "next_teacher_action": np.asarray(composed.teacher_action),
        "next_actor_residual": np.asarray(actor),
        "next_combined_action": np.asarray(composed.combined_pre_limit),
        "next_motor_target": np.asarray(composed.motor_target),
        "next_delay_buffer": np.asarray(history),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    root = Path(args.artifact_root)
    outputs = root / "outputs"
    cases = ("D0", "D1a", "D2")
    comparisons = {
        "A_same_process_gpu": (
            outputs / "gpu_same_process",
            outputs / "gpu_same_process",
            0,
            1,
        ),
        "B_fresh_process_gpu": (
            outputs / "gpu_fresh_process_a",
            outputs / "gpu_fresh_process_b",
            0,
            0,
        ),
        "C_cpu_vs_gpu": (
            outputs / "gpu_fresh_process_a",
            outputs / "cpu_process",
            0,
            0,
        ),
    }
    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    params = checkpoint.load(args.checkpoint)
    config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(config.network_factory),
    )
    policy_apply = jax.jit(networks.policy_network.apply)

    all_rows: list[dict[str, Any]] = []
    discrete_rows: list[dict[str, Any]] = []
    next_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for comparison, (left_root, right_root, left_run, right_run) in comparisons.items():
        comparison_rows = []
        summary[comparison] = {}
        for case_id in cases:
            left = output(left_root / case_id / f"run_{left_run:03d}.pkl")
            right = output(right_root / case_id / f"run_{right_run:03d}.pkl")
            rows = compare_trees(comparison, case_id, left, right)
            comparison_rows.extend(rows)
            discrete = discrete_row(comparison, case_id, left, right, env)
            discrete_rows.append(discrete)
            payload = load_pickle(root / "inputs" / f"{case_id}.pkl")
            left_next = next_controller_values(
                env, params, policy_apply, payload, left
            )
            right_next = next_controller_values(
                env, params, policy_apply, payload, right
            )
            next_metrics = {
                key: float(
                    np.max(
                        np.abs(
                            np.asarray(left_next[key], dtype=np.float64)
                            - np.asarray(right_next[key], dtype=np.float64)
                        )
                    )
                )
                for key in left_next
            }
            next_rows.append(
                {"comparison": comparison, "case_id": case_id, **next_metrics}
            )
            summary[comparison][case_id] = {
                "all_numeric_leaves_bit_exact": all(
                    row["bit_exact"] for row in rows
                ),
                "first_divergence": first_divergence(rows),
                "discrete": discrete,
                "next_controller": next_metrics,
                "max_field_error": max(
                    (
                        float(row["max_abs_error"])
                        for row in rows
                        if isinstance(row["max_abs_error"], (int, float))
                    ),
                    default=0.0,
                ),
            }
        all_rows.extend(comparison_rows)
        write_csv(
            root / f"field_comparison_{comparison}.csv", comparison_rows
        )

    repeated_rows = []
    for case_id in cases:
        runs = [
            output(
                outputs
                / "gpu_same_process"
                / case_id
                / f"run_{index:03d}.pkl"
            )
            for index in range(20)
        ]
        leaves = [numeric_leaves(run) for run in runs]
        for field in sorted(leaves[0]):
            arrays = [np.asarray(tree[field]) for tree in leaves]
            hashes = [
                hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                for array in arrays
            ]
            stack = np.stack([array.astype(np.float64) for array in arrays])
            repeated_rows.append(
                {
                    "case_id": case_id,
                    "field_path": field,
                    "shape": json.dumps(list(arrays[0].shape)),
                    "dtype": str(arrays[0].dtype),
                    "min": float(np.nanmin(stack)) if stack.size else 0.0,
                    "max": float(np.nanmax(stack)) if stack.size else 0.0,
                    "mean": float(np.nanmean(stack)) if stack.size else 0.0,
                    "standard_deviation": float(np.nanstd(stack))
                    if stack.size
                    else 0.0,
                    "max_repeat_span": float(
                        np.nanmax(np.nanmax(stack, axis=0) - np.nanmin(stack, axis=0))
                    )
                    if stack.size
                    else 0.0,
                    "unique_bit_patterns": len(set(hashes)),
                }
            )
    for case_id in cases:
        structures = []
        for index in range(20):
            data = output(
                outputs
                / "gpu_same_process"
                / case_id
                / f"run_{index:03d}.pkl"
            )
            active, pairs, _ = active_contact_signature(data)
            structures.append(json.dumps([active, pairs], sort_keys=True))
        repeated_rows.append(
            {
                "case_id": case_id,
                "field_path": "__contact_structure__",
                "shape": "",
                "dtype": "discrete",
                "min": "",
                "max": "",
                "mean": "",
                "standard_deviation": "",
                "max_repeat_span": "",
                "unique_bit_patterns": len(set(structures)),
            }
        )

    write_csv(root / "field_comparison.csv", all_rows)
    write_csv(root / "discrete_structure_comparison.csv", discrete_rows)
    write_csv(root / "repeated_gpu_one_step.csv", repeated_rows)
    write_csv(root / "next_controller_effect.csv", next_rows)
    (root / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
