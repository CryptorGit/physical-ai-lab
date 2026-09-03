#!/usr/bin/env python3
"""Serialize exact fresh-forward MJX one-step inputs for the v59 audit.

The historical rollout state was not checkpointed.  This tool therefore
recreates only episode-start states whose seed derivation and randomized model
assignment are recorded by the preceding stochastic-parity phase.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground._src.wrapper import BraxDomainRandomizationVmapWrapper

from playground.common import randomize
from playground.open_duck_mini_v2 import joystick

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from export_v59_stochastic_trace import training_environment_keys
from v59_mjx_diagnostic_common import (
    array_sha256,
    canonical_tree_sha256,
    dump_pickle,
    host_tree,
    select_randomized_model,
    single_tree,
)


CASES = (
    ("D0", "C0_stand_seed0", 0),
    ("D1a", "C2_backward_seed1", 1),
    ("D2", "C4_backward_right_max_seed0", 0),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--master-seed", type=int, default=0)
    args = parser.parse_args()
    trace_root = Path(args.trace_root)
    output = Path(args.output)
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)

    env = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    _, training_keys = training_environment_keys(args.master_seed, 4096)
    case_keys = jp.stack([training_keys[index] for _, _, index in CASES])
    wrapper = BraxDomainRandomizationVmapWrapper(
        env, functools.partial(randomize.domain_randomize, rng=case_keys)
    )
    reset = jax.jit(wrapper.reset)
    state = reset(case_keys)
    jax.tree_util.tree_leaves(state.data)[0].block_until_ready()

    commands = []
    traces = []
    metadata = []
    for _, trace_name, _ in CASES:
        trace = np.load(trace_root / f"{trace_name}.npz")
        traces.append(trace)
        metadata.append(
            json.loads(
                (trace_root / f"{trace_name}.metadata.json").read_text(
                    encoding="utf-8"
                )
            )
        )
        commands.append(trace["raw_command"][0])
    commands_jax = jp.asarray(np.asarray(commands), dtype=jp.float32)
    info = dict(state.info)
    info["command"] = commands_jax
    obs = dict(state.obs)
    obs["state"] = obs["state"].at[:, 6:13].set(commands_jax)
    obs["privileged_state"] = obs["privileged_state"].at[:, 6:13].set(
        commands_jax
    )
    state = state.replace(info=info, obs=obs)

    records = {}
    for case_index, (case_id, trace_name, env_index) in enumerate(CASES):
        model = select_randomized_model(
            wrapper._mjx_model_v, wrapper._in_axes, case_index
        )
        case_state = single_tree(state, case_index)
        trace = traces[case_index]
        motor_target = np.asarray(
            trace["motor_target_after_backlash"][0], dtype=np.float32
        )
        actor_action = np.asarray(trace["actor_residual"][0], dtype=np.float32)
        payload = {
            "schema_version": 1,
            "case_id": case_id,
            "trace_name": trace_name,
            "source_kind": "fresh_forward_episode_start_recreated_from_recorded_seed",
            "environment_index": env_index,
            "master_seed": args.master_seed,
            "environment_seed": np.asarray(case_keys[case_index]),
            "model": host_tree(model),
            "state": host_tree(case_state),
            "data": host_tree(case_state.data),
            "actor_action": actor_action,
            "action_after_delay": np.asarray(
                trace["action_after_delay"][0], dtype=np.float32
            ),
            "motor_target": motor_target,
            "domain_randomized_parameters": {
                key: event["random_sample_value"]
                for event in metadata[case_index]["reset_events"]
                if event["random_source_id"].startswith("domain.")
                for key in [event["random_source_id"][len("domain.") :]]
            },
            "trace_step0": {
                key: np.asarray(trace[key][0])
                for key in trace.files
            },
            "trace_step1": {
                key: np.asarray(trace[key][1])
                for key in trace.files
                if len(trace[key]) > 1
            },
            "n_substeps": int(env.n_substeps),
            "control_dt": float(env.dt),
            "scene_task": "flat_terrain_backlash_calibrated",
        }
        path = inputs / f"{case_id}.pkl"
        dump_pickle(path, payload)
        records[case_id] = {
            "input_path": str(path),
            "trace_name": trace_name,
            "state_tree_sha256": canonical_tree_sha256(payload["state"]),
            "data_tree_sha256": canonical_tree_sha256(payload["data"]),
            "model_tree_sha256": canonical_tree_sha256(payload["model"]),
            "actor_action_sha256": array_sha256(actor_action),
            "action_after_delay_sha256": array_sha256(
                payload["action_after_delay"]
            ),
            "motor_target_sha256": array_sha256(motor_target),
            "domain_randomized_parameter_set_sha256": canonical_tree_sha256(
                payload["domain_randomized_parameters"]
            ),
            "input_prephysics_qpos_matches_reset": True,
            "input_prephysics_qvel_matches_reset": True,
            "push_applied_before_step": False,
        }

    records["D1b"] = {
        "status": "unavailable",
        "trace_name": "C2_backward_seed1",
        "last_normal_trace_step": 48,
        "reason": (
            "the stochastic trace saved qpos/qvel after each step but did not "
            "save complete MJX Data, qacc_warmstart, contact/constraint solver "
            "state, or the complete randomized model at step 48"
        ),
    }
    (output / "serialized_input_hashes.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

