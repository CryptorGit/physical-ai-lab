#!/usr/bin/env python3
"""Runs at most five non-performance PPO updates with exact-resume state."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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
WORKSPACE = EXPERIMENT.parents[2]
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

sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(SOURCE_ROOT))

from training.checkpointing import (  # noqa: E402
    save_checkpoint,
    sha256_file,
    tree_leaf_digest,
)
from training.device_metrics import telemetry_transfer_bytes  # noqa: E402
from training.instrumented_ppo_train import (  # noqa: E402
    InstrumentedPPOHarness,
    HarnessConfig,
    identity_test_config,
    load_official_commands,
)
from training.checkpointing import load_checkpoint  # noqa: E402


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True
    ).strip()


def sha(path: Path) -> str:
    return sha256_file(path)


def metadata(config: HarnessConfig, updates: int, resumed_from: str | None) -> dict:
    source_files = {
        "reward": SOURCE_ROOT / "playground/common/rewards.py",
        "teacher_sampler_environment": (
            SOURCE_ROOT / "playground/open_duck_mini_v2/joystick.py"
        ),
        "domain_randomization": SOURCE_ROOT / "playground/common/randomize.py",
        "scene": (
            SOURCE_ROOT
            / "playground/open_duck_mini_v2/xmls/"
            "scene_flat_terrain_backlash_calibrated.xml"
        ),
        "robot_xml": (
            SOURCE_ROOT
            / "playground/open_duck_mini_v2/xmls/"
            "open_duck_mini_v2_backlash_calibrated.xml"
        ),
        "normalizer_parent": PARENT / "_METADATA",
    }
    return {
        "source_commit": git(WORKSPACE, "rev-parse", "HEAD"),
        "source_tree_hash": git(WORKSPACE, "rev-parse", "HEAD^{tree}"),
        "training_source_commit": git(SOURCE_ROOT, "rev-parse", "HEAD"),
        "training_config": dataclasses.asdict(config),
        "training_config_hash": hashlib.sha256(
            json.dumps(
                dataclasses.asdict(config), sort_keys=True
            ).encode("utf-8")
        ).hexdigest(),
        "objective": "old_unbounded_dot",
        "objective_hash": sha(source_files["reward"]),
        "scene_hash": sha(source_files["scene"]),
        "teacher_hash": sha(source_files["teacher_sampler_environment"]),
        "sampler_hash": sha(source_files["teacher_sampler_environment"]),
        "domain_randomization_hash": sha(source_files["domain_randomization"]),
        "command_definition_hash": sha(COMMANDS),
        "external_calibration_hash": sha(
            SOURCE_ROOT
            / "playground/open_duck_mini_v2/data/"
            "polynomial_coefficients_calibrated.pkl"
        ),
        "requested_harness_updates_this_process": updates,
        "resumed_from": resumed_from,
        "wall_clock_metadata": {
            "unix_time": time.time(),
            "pid": os.getpid(),
        },
        "runtime": {
            "python": sys.version,
            "jax": jax.__version__,
            "devices": [str(x) for x in jax.devices()],
            "x64_enabled": bool(jax.config.jax_enable_x64),
            "jit_disabled": bool(jax.config.jax_disable_jit),
        },
        "hashes": {
            name: sha(path)
            for name, path in source_files.items()
            if path.is_file()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--instrumented", action="store_true")
    parser.add_argument("--production-profile", action="store_true")
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    # Keep caller-relative artifact placement stable across the subsequent
    # source-tree chdir needed by mujoco_playground data loaders.
    args.output = args.output.resolve()
    if args.resume is not None:
        args.resume = args.resume.resolve()
    if not 0 <= args.updates <= 5:
        raise ValueError("--updates must be in [0, 5]")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not SOURCE_ROOT.is_dir() or not PARENT.is_dir():
        raise FileNotFoundError(f"missing source or parent: {SOURCE_ROOT}, {PARENT}")

    os.chdir(SOURCE_ROOT)
    cache_dir = SOURCE_ROOT / ".tmp" / "instrumented_harness_jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    from mujoco_playground import wrapper
    from playground.common import randomize
    from playground.open_duck_mini_v2 import joystick

    config = (
        HarnessConfig(seed=args.seed)
        if args.production_profile
        else identity_test_config(args.seed)
    )
    environment = joystick.Joystick(task="flat_terrain_backlash_calibrated")
    harness = InstrumentedPPOHarness(
        environment=environment,
        randomization_fn=randomize.domain_randomize,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        parent_checkpoint=PARENT,
        official_commands=load_official_commands(COMMANDS),
        config=config,
        instrumented=args.instrumented,
    )
    state = harness.initial_state
    resume_manifest = None
    if args.resume:
        state, randomized_model, resume_manifest = load_checkpoint(args.resume)
        harness.install_randomized_model(randomized_model)
        state = jax.device_put(state)

    telemetry_rows: list[dict[str, Any]] = []
    telemetry_payloads: list[Any] = []
    probe_hashes: list[dict[str, str]] = []
    update_seconds: list[float] = []
    host_transfers = 0
    host_transfer_bytes = 0
    for update_index in range(args.updates):
        started = time.perf_counter()
        state, telemetry, probes = harness.update(state)
        elapsed = time.perf_counter() - started
        update_seconds.append(elapsed)
        if args.instrumented:
            host_telemetry = jax.device_get(telemetry)
            host_transfers += 1
            host_transfer_bytes += telemetry_transfer_bytes(host_telemetry)
            telemetry_rows.append(
                {
                    "update": update_index + 1,
                    "tree_sha256": tree_leaf_digest(host_telemetry),
                    "bytes": telemetry_transfer_bytes(host_telemetry),
                }
            )
            telemetry_payloads.append(host_telemetry)
        probe_hashes.append(
            {
                "update": str(update_index + 1),
                "tree_sha256": tree_leaf_digest(probes),
            }
        )

    current_interactions = int(
        np.asarray(
            jax.device_get(state.counters.global_environment_interactions)
        )
    )
    current_optimizer_updates = int(
        np.asarray(jax.device_get(state.counters.optimizer_update_count))
    )
    run_metadata = metadata(
        config, args.updates, str(args.resume) if args.resume else None
    )
    run_metadata.update(
        {
            "global_environment_interactions": current_interactions,
            "optimizer_update_count": current_optimizer_updates,
            "harness_update_count": int(
                np.asarray(jax.device_get(state.counters.harness_update_count))
            ),
            "host_transfer_count": host_transfers,
            "host_transfer_bytes": host_transfer_bytes,
            "update_seconds": update_seconds,
            "telemetry": telemetry_rows,
            "probe_hashes": probe_hashes,
            "resume_payload_hash": (
                resume_manifest["payload_sha256"] if resume_manifest else None
            ),
            "formal_performance_claim": False,
            "test_profile": not args.production_profile,
        }
    )
    manifest = save_checkpoint(
        args.output,
        state=state,
        randomized_model=harness.randomized_model,
        metadata=run_metadata,
    )
    if telemetry_payloads:
        def jsonable(value):
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [jsonable(item) for item in value]
            array = np.asarray(value)
            return array.item() if array.ndim == 0 else array.tolist()

        (args.output / "update_telemetry.json").write_text(
            json.dumps(jsonable(telemetry_payloads), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
