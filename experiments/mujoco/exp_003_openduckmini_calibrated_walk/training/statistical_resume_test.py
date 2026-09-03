#!/usr/bin/env python3
"""Statistical resume trial runner for the frozen old-objective PPO harness.

Measured segments always compile on a disposable state, reload the serialized
input from disk, and then execute at an update boundary.  No reward, PPO, or
environment behavior is changed by this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import jax
import numpy as np


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(
    os.environ.get(
        "OPENDUCK_TRAINING_SOURCE",
        "/home/user/openduck_training_backward_v23_20260729",
    )
)
PARENT = Path(
    os.environ.get(
        "OPENDUCK_V52_ACTOR_PARENT",
        "/home/user/openduck_training_runs/"
        "coupled_head_original_stand_backward_v45_50m/"
        "2026_07_29_235335_47349760",
    )
)
COMMANDS = (
    EXPERIMENT
    / "artifacts"
    / "v59_evaluation_equivalence"
    / "command_definitions.json"
)
sys.path[:0] = [str(EXPERIMENT), str(SOURCE_ROOT)]

from training.checkpointing import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    tree_leaf_digest,
)
from training.instrumented_ppo_train import (  # noqa: E402
    InstrumentedPPOHarness,
    identity_test_config,
    load_official_commands,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array.tolist()


def _build_harness(seed: int) -> InstrumentedPPOHarness:
    os.chdir(SOURCE_ROOT)
    cache_dir = SOURCE_ROOT / ".tmp" / "statistical_resume_jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    from mujoco_playground import wrapper
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    return InstrumentedPPOHarness(
        environment=joystick.Joystick(
            task="flat_terrain_backlash_calibrated"
        ),
        randomization_fn=randomize.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        parent_checkpoint=PARENT,
        official_commands=load_official_commands(COMMANDS),
        config=identity_test_config(seed),
        instrumented=True,
    )


def _load_into_harness(
    harness: InstrumentedPPOHarness, checkpoint: Path
) -> tuple[Any, Any, dict[str, Any]]:
    state, randomized_model, manifest = load_checkpoint(checkpoint)
    harness.install_randomized_model(randomized_model)
    return jax.device_put(state), randomized_model, manifest


def run_segment(
    *,
    checkpoint: Path,
    output: Path,
    updates: int,
    seed: int,
    trial_id: str,
    mode: str,
) -> None:
    if output.exists():
        raise FileExistsError(output)
    harness = _build_harness(seed)

    # Compile and execute using a disposable copy.  The measured state is then
    # reloaded from disk, including all RNG and randomized-model leaves.
    warm_state, _, _ = _load_into_harness(harness, checkpoint)
    warm_result = harness.update(warm_state)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), warm_result)
    state, randomized_model, source_manifest = _load_into_harness(
        harness, checkpoint
    )
    measured_input_hash = tree_leaf_digest(jax.device_get(state))

    telemetry = []
    update_seconds = []
    for update_index in range(updates):
        started = time.perf_counter()
        state, update_telemetry, _ = harness.update(state)
        update_seconds.append(time.perf_counter() - started)
        telemetry.append(_jsonable(jax.device_get(update_telemetry)))

    metadata = {
        "schema_version": "statistical_resume_trial_v1",
        "trial_id": trial_id,
        "mode": mode,
        "seed": seed,
        "updates_this_process": updates,
        "warmup_discarded": true_value(),
        "input_checkpoint": str(checkpoint),
        "input_payload_sha256": source_manifest["payload_sha256"],
        "measured_input_tree_sha256": measured_input_hash,
        "global_environment_interactions": int(
            np.asarray(
                jax.device_get(
                    state.counters.global_environment_interactions
                )
            )
        ),
        "optimizer_update_count": int(
            np.asarray(
                jax.device_get(state.counters.optimizer_update_count)
            )
        ),
        "update_seconds": update_seconds,
        "objective": "old_unbounded_dot",
        "diagnostic_only": True,
    }
    save_checkpoint(
        output,
        state=state,
        randomized_model=randomized_model,
        metadata=metadata,
    )
    (output / "update_telemetry.json").write_text(
        json.dumps(telemetry, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def true_value() -> bool:
    """Avoid a mutable or backend-derived marker in checkpoint metadata."""
    return True


def roundtrip(checkpoint: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    state, model, source_manifest = load_checkpoint(checkpoint)
    before_state = tree_leaf_digest(state)
    before_model = tree_leaf_digest(model)
    saved = save_checkpoint(
        output,
        state=state,
        randomized_model=model,
        metadata={
            "schema_version": "statistical_resume_roundtrip_v1",
            "source_payload_sha256": source_manifest["payload_sha256"],
        },
    )
    loaded_state, loaded_model, loaded_manifest = load_checkpoint(output)
    after_state = tree_leaf_digest(loaded_state)
    after_model = tree_leaf_digest(loaded_model)
    rows = [
        {
            "component": "complete_training_state",
            "before_sha256": before_state,
            "after_sha256": after_state,
            "bit_exact": before_state == after_state,
        },
        {
            "component": "complete_randomized_mjx_model",
            "before_sha256": before_model,
            "after_sha256": after_model,
            "bit_exact": before_model == after_model,
        },
        {
            "component": "serialized_payload_self_check",
            "before_sha256": saved["payload_sha256"],
            "after_sha256": loaded_manifest["payload_sha256"],
            "bit_exact": saved["payload_sha256"]
            == loaded_manifest["payload_sha256"],
        },
    ]
    with (output / "roundtrip_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    if not all(row["bit_exact"] for row in rows):
        raise RuntimeError("CHECKPOINT_PAYLOAD_BIT_EXACT_FAIL")


def orchestrate(
    *,
    initial: Path,
    raw_root: Path,
    artifact_root: Path,
    trials_per_mode: int,
    seed: int,
) -> None:
    raw_root.mkdir(parents=True, exist_ok=False)
    (raw_root / "logs").mkdir()
    artifact_root.mkdir(parents=True, exist_ok=True)
    rows = []
    script = Path(__file__).resolve()

    def invoke(args: list[str], log_name: str) -> int:
        with (raw_root / "logs" / log_name).open("w") as log:
            completed = subprocess.run(
                [sys.executable, str(script), *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return completed.returncode

    order = []
    for index in range(trials_per_mode):
        order.extend(("U", "R") if index % 2 == 0 else ("R", "U"))

    mode_index = {"U": 0, "R": 0}
    for sequence_index, mode in enumerate(order):
        index = mode_index[mode]
        mode_index[mode] += 1
        trial_id = f"{mode}_{index:02d}"
        started = time.time()
        status = "COMPLETED"
        error_stage = ""
        if mode == "U":
            final = raw_root / trial_id / "final"
            code = invoke(
                [
                    "segment",
                    "--checkpoint",
                    str(initial),
                    "--output",
                    str(final),
                    "--updates",
                    "4",
                    "--seed",
                    str(seed),
                    "--trial-id",
                    trial_id,
                    "--mode",
                    mode,
                ],
                f"{trial_id}.log",
            )
            midpoint = ""
            if code:
                status, error_stage = "CRASHED", "uninterrupted"
        else:
            midpoint_path = raw_root / trial_id / "midpoint"
            final = raw_root / trial_id / "final"
            first_code = invoke(
                [
                    "segment",
                    "--checkpoint",
                    str(initial),
                    "--output",
                    str(midpoint_path),
                    "--updates",
                    "2",
                    "--seed",
                    str(seed),
                    "--trial-id",
                    trial_id,
                    "--mode",
                    "R_FIRST",
                ],
                f"{trial_id}_first.log",
            )
            second_code = 1
            if first_code == 0:
                second_code = invoke(
                    [
                        "segment",
                        "--checkpoint",
                        str(midpoint_path),
                        "--output",
                        str(final),
                        "--updates",
                        "2",
                        "--seed",
                        str(seed),
                        "--trial-id",
                        trial_id,
                        "--mode",
                        "R_SECOND",
                    ],
                    f"{trial_id}_second.log",
                )
            midpoint = str(midpoint_path)
            if first_code:
                status, error_stage = "CRASHED", "resume_first"
            elif second_code:
                status, error_stage = "CRASHED", "resume_second"
        rows.append(
            {
                "sequence_index": sequence_index,
                "trial_id": trial_id,
                "mode": mode,
                "logical_seed": seed,
                "status": status,
                "error_stage": error_stage,
                "initial_checkpoint": str(initial),
                "midpoint_checkpoint": midpoint,
                "final_checkpoint": str(final),
                "wall_seconds": time.time() - started,
            }
        )
        with (artifact_root / "resume_trial_manifest.csv").open(
            "w", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment")
    segment.add_argument("--checkpoint", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--updates", type=int, choices=(2, 4), required=True)
    segment.add_argument("--seed", type=int, required=True)
    segment.add_argument("--trial-id", required=True)
    segment.add_argument("--mode", required=True)

    rt = subparsers.add_parser("roundtrip")
    rt.add_argument("--checkpoint", type=Path, required=True)
    rt.add_argument("--output", type=Path, required=True)

    orchestration = subparsers.add_parser("orchestrate")
    orchestration.add_argument("--initial", type=Path, required=True)
    orchestration.add_argument("--raw-root", type=Path, required=True)
    orchestration.add_argument("--artifact-root", type=Path, required=True)
    orchestration.add_argument("--trials-per-mode", type=int, default=20)
    orchestration.add_argument("--seed", type=int, default=20260730)

    args = parser.parse_args()
    if args.command == "segment":
        run_segment(
            checkpoint=args.checkpoint.resolve(),
            output=args.output.resolve(),
            updates=args.updates,
            seed=args.seed,
            trial_id=args.trial_id,
            mode=args.mode,
        )
    elif args.command == "roundtrip":
        roundtrip(args.checkpoint.resolve(), args.output.resolve())
    else:
        orchestrate(
            initial=args.initial.resolve(),
            raw_root=args.raw_root.resolve(),
            artifact_root=args.artifact_root.resolve(),
            trials_per_mode=args.trials_per_mode,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
