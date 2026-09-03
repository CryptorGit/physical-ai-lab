#!/usr/bin/env python3
"""Compile, warm up, and measure isolated MJX one-step outputs."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import jax
import jax.numpy as jp
import jaxlib
import mujoco
import numpy as np
from mujoco_playground._src import mjx_env

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
from v59_mjx_diagnostic_common import dump_pickle, host_tree, load_pickle


def block(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
            return


def runtime_manifest(label: str) -> dict:
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        smi = "unavailable"
    return {
        "label": label,
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "mujoco_version": mujoco.__version__,
        "mjx_provenance": "mujoco.mjx from mujoco package",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_count": jax.process_count(),
        "process_index": jax.process_index(),
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "jit_disabled": bool(jax.config.jax_disable_jit),
        "default_matmul_precision": str(jax.config.jax_default_matmul_precision),
        "default_dtype_bits": str(jax.config.jax_default_dtype_bits),
        "platforms": str(jax.config.jax_platforms),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
        "jax_default_matmul_precision_env": os.environ.get(
            "JAX_DEFAULT_MATMUL_PRECISION", ""
        ),
        "xla_python_client_preallocate": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", ""
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi_gpu_driver": smi,
        "float_input_dtypes": ["float32"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest(args.label)
    manifest["started_unix"] = time.time()

    # Model is an explicit dynamic argument, exactly as in the vmap wrapper.
    one_step = jax.jit(
        lambda model, data, ctrl, n: mjx_env.step(model, data, ctrl, n),
        static_argnums=(3,),
    )

    input_paths = sorted(input_root.glob("D*.pkl"))
    if not input_paths:
        raise FileNotFoundError(f"no diagnostic inputs under {input_root}")

    # Compile and warm up.  This output is intentionally discarded.
    warm = load_pickle(input_paths[0])
    warm_output = one_step(
        jax.device_put(warm["model"]),
        jax.device_put(warm["data"]),
        jp.asarray(warm["motor_target"]),
        int(warm["n_substeps"]),
    )
    block(warm_output)
    manifest["warmup_case"] = warm["case_id"]
    manifest["measurement_uses_reloaded_pickle_each_run"] = True

    for input_path in input_paths:
        case_id = input_path.stem
        case_dir = output_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(args.repeats):
            # Reload every measured input so no prior output can be carried in.
            payload = load_pickle(input_path)
            model = jax.device_put(payload["model"])
            data = jax.device_put(payload["data"])
            ctrl = jp.asarray(payload["motor_target"])
            started = time.perf_counter()
            result = one_step(
                model, data, ctrl, int(payload["n_substeps"])
            )
            block(result)
            elapsed = time.perf_counter() - started
            output = {
                "case_id": case_id,
                "label": args.label,
                "repeat": repeat,
                "elapsed_seconds": elapsed,
                "data": host_tree(result),
            }
            dump_pickle(case_dir / f"run_{repeat:03d}.pkl", output)

    manifest["finished_unix"] = time.time()
    (output_root / "runtime_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
