"""Isolated, no-PPO C0 GPU runtime probe for JAX and minimal MJX physics.

This does not import the OpenDuck environment, policy, Brax, or deployment
code.  It establishes whether one fixed stack can synchronously run a tiny
integer JAX workload and a B=1/B=2 MJX contact workload with raw-exact output
agreement.  It must not be used as a locomotion or hardware acceptance test.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from jax import lax
from jax import tree_util
from mujoco import mjx


SCRIPT_PATH = Path(__file__).resolve()
MINIMAL_CONTACT_XML = """
<mujoco model="c0_minimal_contact">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.055">
      <freejoint/>
      <geom type="sphere" size="0.05" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _emit(stage: str, **fields: Any) -> None:
    record = {
        "event": "c0_gpu_stack_probe_stage",
        "stage": stage,
        "monotonic_ns": time.monotonic_ns(),
        **fields,
    }
    sys.stderr.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    sys.stderr.flush()


def _block_tree(value: Any) -> Any:
    for leaf in tree_util.tree_leaves(value):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()
    return value


def _array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise TypeError("raw comparison cannot serialize object arrays")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _raw_tree_report(left: Any, right: Any) -> dict[str, Any]:
    left_items, left_tree = tree_util.tree_flatten_with_path(left)
    right_items, right_tree = tree_util.tree_flatten_with_path(right)
    left_digest = hashlib.sha256()
    right_digest = hashlib.sha256()
    mismatches: list[dict[str, Any]] = []
    if left_tree != right_tree or len(left_items) != len(right_items):
        return {
            "raw_equal": False,
            "reason": "pytree_structure_or_leaf_count_mismatch",
            "left_leaf_count": len(left_items),
            "right_leaf_count": len(right_items),
            "mismatches": [],
        }
    for (left_path, left_leaf), (right_path, right_leaf) in zip(
        left_items, right_items, strict=True
    ):
        left_name = tree_util.keystr(left_path)
        right_name = tree_util.keystr(right_path)
        if left_name != right_name:
            return {
                "raw_equal": False,
                "reason": "pytree_path_order_mismatch",
                "left_path": left_name,
                "right_path": right_name,
                "mismatches": [],
            }
        left_array = np.ascontiguousarray(np.asarray(left_leaf))
        right_array = np.ascontiguousarray(np.asarray(right_leaf))
        left_leaf_digest = _array_digest(left_array)
        right_leaf_digest = _array_digest(right_array)
        left_digest.update(left_name.encode("utf-8"))
        left_digest.update(bytes.fromhex(left_leaf_digest))
        right_digest.update(right_name.encode("utf-8"))
        right_digest.update(bytes.fromhex(right_leaf_digest))
        equal = (
            left_array.dtype == right_array.dtype
            and left_array.shape == right_array.shape
            and left_array.tobytes(order="C") == right_array.tobytes(order="C")
        )
        if not equal and len(mismatches) < 20:
            mismatches.append(
                {
                    "path": left_name,
                    "left_dtype": left_array.dtype.str,
                    "right_dtype": right_array.dtype.str,
                    "left_shape": list(left_array.shape),
                    "right_shape": list(right_array.shape),
                    "left_raw_sha256": left_leaf_digest,
                    "right_raw_sha256": right_leaf_digest,
                }
            )
    return {
        "raw_equal": not mismatches,
        "left_tree_raw_sha256": left_digest.hexdigest(),
        "right_tree_raw_sha256": right_digest.hexdigest(),
        "leaf_count": len(left_items),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def _strict_lane_one(value: Any, *, expected_batch: int, label: str) -> Any:
    """Slice lane one only after proving every dynamic leaf has B as axis zero."""

    path_leaves, _tree = tree_util.tree_flatten_with_path(value)
    invalid = []
    for path, leaf in path_leaves:
        shape = getattr(leaf, "shape", None)
        if shape is None or len(shape) < 1 or int(shape[0]) != expected_batch:
            invalid.append(
                {
                    "path": tree_util.keystr(path),
                    "shape": None if shape is None else list(shape),
                }
            )
    if invalid:
        raise RuntimeError(
            f"{label} cannot safely slice B={expected_batch} lane one: {invalid[:20]}"
        )
    return tree_util.tree_map(lambda leaf: leaf[1:2], value)


def _compile(label: str, function: Any, *args: Any) -> Any:
    _emit(f"{label}.lower.start")
    lowered = function.lower(*args)
    _emit(f"{label}.lower.finish")
    _emit(f"{label}.compile.start")
    executable = lowered.compile()
    _emit(f"{label}.compile.finish")
    return executable


def _execute(label: str, executable: Any, *args: Any) -> Any:
    _emit(f"{label}.execute.call")
    result = executable(*args)
    _emit(f"{label}.dispatch.returned")
    _emit(f"{label}.block.start")
    _block_tree(result)
    _emit(f"{label}.block.finish")
    _emit(f"{label}.device_get.start")
    host_result = jax.device_get(result)
    _emit(f"{label}.device_get.finish")
    return host_result


def _run_gate1() -> dict[str, Any]:
    _emit("gate1.input.start")
    b2 = jp.arange(8192, dtype=jp.int32).reshape(2, 4096)
    b1 = b2[1:2]

    def integer_workload(value: Any) -> Any:
        def body(index: Any, current: Any) -> Any:
            return (current * jp.int32(1664525) + index + jp.int32(1013904223))

        return lax.fori_loop(0, 10, body, value)

    workload = jax.jit(integer_workload)
    b1_executable = _compile("gate1.b1", workload, b1)
    b2_executable = _compile("gate1.b2", workload, b2)
    b1_first = _execute("gate1.b1.first", b1_executable, b1)
    b1_second = _execute("gate1.b1.second", b1_executable, b1)
    b2_first = _execute("gate1.b2.first", b2_executable, b2)
    b2_second = _execute("gate1.b2.second", b2_executable, b2)
    reports = {
        "b1_repeat": _raw_tree_report(b1_first, b1_second),
        "b2_repeat": _raw_tree_report(b2_first, b2_second),
        "b1_vs_b2_lane1": _raw_tree_report(b1_first, b2_first[1:2]),
    }
    return {
        "passed": all(report["raw_equal"] for report in reports.values()),
        "reports": reports,
    }


def _run_gate2() -> dict[str, Any]:
    _emit("gate2.model.start")
    host_model = mujoco.MjModel.from_xml_string(MINIMAL_CONTACT_XML)
    model = mjx.put_model(host_model)
    base_data = mjx.make_data(model)

    def make_initial(height: Any) -> Any:
        return base_data.replace(qpos=base_data.qpos.at[2].set(height))

    data_b2 = jax.vmap(make_initial)(jp.asarray((0.060, 0.055), dtype=jp.float32))
    _block_tree(data_b2)
    data_b1 = _strict_lane_one(data_b2, expected_batch=2, label="gate2 initial data")
    initial_report = _raw_tree_report(
        jax.device_get(data_b1),
        _strict_lane_one(jax.device_get(data_b2), expected_batch=2, label="gate2 host initial data"),
    )
    if not initial_report["raw_equal"]:
        raise RuntimeError("gate2 B=1 initial data is not raw-equal to B=2 lane one")
    ctrl_b2 = jp.zeros((2, int(host_model.nu)), dtype=jp.float32)
    ctrl_b1 = ctrl_b2[1:2]

    def one_lane_rollout(data: Any, ctrl: Any) -> Any:
        def one_substep(_index: Any, state: Any) -> Any:
            return mjx.step(model, state.replace(ctrl=ctrl))

        return lax.fori_loop(0, 10, one_substep, data)

    batched_rollout = jax.jit(jax.vmap(one_lane_rollout))
    b1_executable = _compile("gate2.b1", batched_rollout, data_b1, ctrl_b1)
    b2_executable = _compile("gate2.b2", batched_rollout, data_b2, ctrl_b2)
    b1_first = _execute("gate2.b1.first", b1_executable, data_b1, ctrl_b1)
    b1_second = _execute("gate2.b1.second", b1_executable, data_b1, ctrl_b1)
    b2_first = _execute("gate2.b2.first", b2_executable, data_b2, ctrl_b2)
    b2_second = _execute("gate2.b2.second", b2_executable, data_b2, ctrl_b2)
    reports = {
        "initial_b1_vs_b2_lane1": initial_report,
        "b1_repeat": _raw_tree_report(b1_first, b1_second),
        "b2_repeat": _raw_tree_report(b2_first, b2_second),
        "b1_vs_b2_lane1": _raw_tree_report(
            b1_first,
            _strict_lane_one(b2_first, expected_batch=2, label="gate2 final B=2 data"),
        ),
    }
    return {
        "model": {"nq": int(host_model.nq), "nv": int(host_model.nv), "nu": int(host_model.nu)},
        "passed": all(report["raw_equal"] for report in reports.values()),
        "reports": reports,
    }


def _package_versions() -> dict[str, str]:
    names = ("jax", "jaxlib", "jax-cuda12-plugin", "jax-cuda12-pjrt", "mujoco", "mujoco-mjx", "numpy")
    return {name: importlib.metadata.version(name) for name in names}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable C0 probe result: {output}")
    if output.parent.exists() is False:
        raise FileNotFoundError(f"output parent must already exist: {output.parent}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"C0 probe requires GPU backend, got {jax.default_backend()!r}")
    faulthandler.enable(file=sys.stderr)
    if hasattr(signal, "SIGUSR1"):
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "openduckmini_c0_gpu_stack_probe_no_ppo_no_hardware",
        "hardware_deployment": "PROHIBITED",
        "ppo_execution": "NOT_INVOKED",
        "script_sha256": _sha256_file(SCRIPT_PATH),
        "runtime": {
            "packages": _package_versions(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "ld_library_path": os.environ.get("LD_LIBRARY_PATH", ""),
            "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH", ""),
        },
    }
    try:
        payload["gate1_jax_integer"] = _run_gate1()
        payload["gate2_mjx_minimal_contact"] = _run_gate2()
        payload["status"] = (
            "PASS"
            if payload["gate1_jax_integer"]["passed"]
            and payload["gate2_mjx_minimal_contact"]["passed"]
            else "FAIL_RAW_PARITY"
        )
    except BaseException as exc:
        payload["status"] = "ERROR"
        payload["error_type"] = type(exc).__name__
        payload["error"] = str(exc)
        _emit("probe.exception", error_type=type(exc).__name__, error=str(exc))
    finally:
        faulthandler.cancel_dump_traceback_later()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _emit("probe.result_written", status=payload["status"], output=str(output))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
