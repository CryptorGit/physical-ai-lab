"""Independent matched-control trainer for the v60 bounded-yaw pilot.

This entrypoint does not edit the production environment.  Both arms use this
same subclass and differ only in ``objective_mode`` and artifact names.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params


EXP_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get(
        "OPENDUCK_V60_SOURCE_ROOT",
        "/home/user/openduck_training_backward_v23_20260729",
    )
)
TOOLS = EXP_ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from v60_yaw_objective import (  # noqa: E402
    bounded_yaw_progress,
    old_yaw_progress,
)
from playground.common import randomize  # noqa: E402
from playground.open_duck_mini_v2 import joystick  # noqa: E402


PARENT = Path(
    "/home/user/openduck_training_runs/"
    "coupled_head_original_stand_backward_v45_50m/"
    "2026_07_29_235335_47349760"
)
OUTPUT_ROOT = EXP_ROOT / "artifacts" / "v60_bounded_yaw_pilot"
SEED = 20260730
TASK = "flat_terrain_backlash_calibrated"
NUM_TIMESTEPS = 1_000_000
# 2,500 environments triggered a WSL libcuda SIGSEGV after 500k interactions
# in the pre-pilot control attempt.  1,250 divides the unchanged 2,500-sample
# rollout batch and preserves exactly 50,000 interactions per training step.
NUM_ENVS = 1_250
UNROLL_LENGTH = 20
BATCH_SIZE = 125
NUM_MINIBATCHES = 20
NUM_UPDATES_PER_BATCH = 4
# One host epoch avoids a reproducible WSL libcuda failure at the second host
# boundary.  Prefix checkpoints are regenerated as deterministic single-scan
# runs after the uninterrupted 1M causal comparison.
NUM_EVALS = 2
ENV_STEPS_PER_TRAINING_STEP = (
    BATCH_SIZE * UNROLL_LENGTH * NUM_MINIBATCHES
)


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


class V60YawPilotJoystick(joystick.Joystick):
    """Same controller/environment with a selectable yaw reward expression."""

    def __init__(self, *, objective_mode: str):
        if objective_mode not in {
            "old_unbounded_dot",
            "bounded_command_centered_gaussian",
        }:
            raise ValueError(objective_mode)
        self.objective_mode = objective_mode
        # Historical PolyReferenceMotion paths are deliberately relative to
        # the frozen source root.  Record this cwd in the run manifest.
        os.chdir(SOURCE_ROOT)
        super().__init__(task=TASK)

    def _get_reward(self, *args, **kwargs):
        rewards = super()._get_reward(*args, **kwargs)
        data = args[0]
        info = args[2]
        command = info["command"]
        actual_yaw = self.get_gyro(data)[2]
        if self.objective_mode == "old_unbounded_dot":
            yaw_progress = old_yaw_progress(
                command[2], actual_yaw, xp=jp
            )
        else:
            yaw_progress = bounded_yaw_progress(
                command[2], actual_yaw, xp=jp
            )
        # The linear dot product is byte-for-byte the production expression.
        rewards["command_progress"] = jp.dot(
            self.get_local_linvel(data)[:2], command[:2]
        ) + yaw_progress
        return rewards


def resolved_config(arm: str) -> dict[str, Any]:
    objective = (
        "old_unbounded_dot"
        if arm == "control"
        else "bounded_command_centered_gaussian"
    )
    base = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    ).to_dict()
    base.update(
        {
            "num_timesteps": NUM_TIMESTEPS,
            "num_envs": NUM_ENVS,
            "unroll_length": UNROLL_LENGTH,
            "batch_size": BATCH_SIZE,
            "num_minibatches": NUM_MINIBATCHES,
            "num_updates_per_batch": NUM_UPDATES_PER_BATCH,
            "num_evals": NUM_EVALS,
            "seed": SEED,
            "task": TASK,
            "restore_checkpoint_path": str(PARENT),
            "objective_name": objective,
            "run_name": f"v60_{arm}_1m",
            "output_path": str(OUTPUT_ROOT / "checkpoints" / arm),
            "environment_interaction_definition": (
                "one transition from each parallel environment; simulator "
                "substeps and optimizer updates excluded"
            ),
            "environment_steps_per_training_step": (
                ENV_STEPS_PER_TRAINING_STEP
            ),
            "optimizer_updates_at_1m": (
                NUM_TIMESTEPS
                // ENV_STEPS_PER_TRAINING_STEP
                * NUM_UPDATES_PER_BATCH
                * NUM_MINIBATCHES
            ),
        }
    )
    return base


def arm_identity_view(config: dict[str, Any]) -> dict[str, Any]:
    allowed = {"objective_name", "run_name", "output_path"}
    return {key: value for key, value in config.items() if key not in allowed}


def save_params(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Repeated Orbax writes in the frozen JAX 0.5.3 environment reproducibly
    # segfaulted inside WSL libcuda after 500k interactions.  Synchronize each
    # leaf to host and use a plain immutable pickle during the live run.
    # A separate post-run process converts selected files to Orbax.
    host_params = jax.tree_util.tree_map(np.asarray, params)
    with path.open("wb") as stream:
        pickle.dump(host_params, stream, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument(
        "--wiring-only",
        action="store_true",
        help="Run 2 envs and one 40-interaction training step only.",
    )
    args = parser.parse_args()
    arm = args.arm
    objective = (
        "old_unbounded_dot"
        if arm == "control"
        else "bounded_command_centered_gaussian"
    )
    config = resolved_config(arm)
    config_dir = OUTPUT_ROOT / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{arm}_resolved_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )

    env = V60YawPilotJoystick(objective_mode=objective)
    eval_env = V60YawPilotJoystick(objective_mode=objective)
    ppo_config = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    network_config = ppo_config.network_factory.to_dict()
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_config
    )
    training = ppo_config.to_dict()
    training.pop("network_factory")
    training.update(
        {
            "num_timesteps": 40 if args.wiring_only else NUM_TIMESTEPS,
            "num_envs": 2 if args.wiring_only else NUM_ENVS,
            "unroll_length": 20 if args.wiring_only else UNROLL_LENGTH,
            "batch_size": 1 if args.wiring_only else BATCH_SIZE,
            "num_minibatches": 2 if args.wiring_only else NUM_MINIBATCHES,
            "num_updates_per_batch": (
                1 if args.wiring_only else NUM_UPDATES_PER_BATCH
            ),
            "num_evals": 2 if args.wiring_only else NUM_EVALS,
            "seed": SEED,
            "run_evals": False,
            "log_training_metrics": True,
            "restore_checkpoint_path": str(PARENT),
        }
    )
    if args.wiring_only:
        run_dir = OUTPUT_ROOT / "wiring" / arm
    else:
        run_dir = OUTPUT_ROOT / "checkpoints" / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_path = run_dir / "training_curve.csv"
    curve_rows: list[dict[str, Any]] = []
    saved_steps: list[int] = []

    def progress(step: int, metrics: dict[str, Any]) -> None:
        row = {"environment_interactions": int(step)}
        for key, value in metrics.items():
            array = np.asarray(value)
            if array.size == 1:
                row[key] = float(array.reshape(()))
        curve_rows.append(row)
        keys = sorted({key for item in curve_rows for key in item})
        with curve_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(curve_rows)

    def checkpoint_callback(step: int, _make_policy, params: Any) -> None:
        save_params(run_dir / f"step_{int(step):07d}.pkl", params)
        saved_steps.append(int(step))
        (run_dir / "checkpoint_index.json").write_text(
            json.dumps({"steps": saved_steps}, indent=2) + "\n",
            encoding="utf-8",
        )

    make_policy, params, metrics = ppo.train(
        environment=env,
        eval_env=eval_env,
        network_factory=network_factory,
        randomization_fn=randomize.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        progress_fn=progress,
        policy_params_fn=checkpoint_callback,
        **training,
    )
    del make_policy
    final_steps = 40 if args.wiring_only else NUM_TIMESTEPS
    result = {
        "arm": arm,
        "objective": objective,
        "objective_hash": canonical_json_sha({"objective": objective}),
        "parent": str(PARENT),
        "seed": SEED,
        "requested_environment_interactions": final_steps,
        "saved_steps": saved_steps,
        "optimizer_update_count": (
            2
            if args.wiring_only
            else config["optimizer_updates_at_1m"]
        ),
        "metrics": {
            key: float(np.asarray(value))
            for key, value in metrics.items()
            if np.asarray(value).size == 1
        },
        "final_params_leaf_count": len(jax.tree_util.tree_leaves(params)),
        "resolved_config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
    }
    (run_dir / "run_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    main()
