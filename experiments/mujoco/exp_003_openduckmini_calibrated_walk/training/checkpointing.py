"""Atomic, hash-verified checkpoint I/O for the instrumented harness."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any, Mapping

import jax
import numpy as np


SCHEMA_VERSION = "instrumented_training_harness_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_leaf_digest(tree: Any) -> str:
    """Hashes PyTree structure, leaf dtypes/shapes, and exact host bytes."""
    path_leaves, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    digest = hashlib.sha256()
    digest.update(str(len(path_leaves)).encode("ascii"))
    for path, leaf in path_leaves:
        stable_path = "/".join(
            (
                f"{type(key).__name__}:"
                f"{getattr(key, 'key', getattr(key, 'idx', getattr(key, 'name', '')))}"
            )
            for key in path
        )
        digest.update(stable_path.encode("utf-8"))
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tree_nbytes(tree: Any) -> int:
    return int(
        sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(jax.device_get(tree)))
    )


def save_checkpoint(
    directory: Path,
    *,
    state: Any,
    randomized_model: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Saves an update-boundary checkpoint using atomic file replacement."""
    directory.mkdir(parents=True, exist_ok=False)
    payload_path = directory / "state.pkl"
    manifest_path = directory / "manifest.json"
    host_payload = {
        "schema_version": SCHEMA_VERSION,
        "state": jax.device_get(state),
        "randomized_model": jax.device_get(randomized_model),
    }
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=directory, delete=False, prefix=".state-", suffix=".tmp"
    ) as handle:
        pickle.dump(host_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(payload_path)
    elapsed = time.perf_counter() - started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **dict(metadata),
        "payload": payload_path.name,
        "payload_sha256": sha256_file(payload_path),
        "state_tree_sha256": tree_leaf_digest(host_payload["state"]),
        "randomized_model_tree_sha256": tree_leaf_digest(
            host_payload["randomized_model"]
        ),
        "payload_bytes": payload_path.stat().st_size,
        "uncompressed_tree_bytes": tree_nbytes(host_payload),
        "save_seconds": elapsed,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_checkpoint(directory: Path) -> tuple[Any, Any, dict[str, Any]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    payload_path = directory / manifest["payload"]
    actual = sha256_file(payload_path)
    if actual != manifest["payload_sha256"]:
        raise ValueError(
            f"checkpoint payload hash mismatch: {actual} != {manifest['payload_sha256']}"
        )
    with payload_path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema: {payload['schema_version']}")
    if tree_leaf_digest(payload["state"]) != manifest["state_tree_sha256"]:
        raise ValueError("state tree hash mismatch")
    if (
        tree_leaf_digest(payload["randomized_model"])
        != manifest["randomized_model_tree_sha256"]
    ):
        raise ValueError("randomized model tree hash mismatch")
    return payload["state"], payload["randomized_model"], manifest


def checkpoint_name(requested_interactions: int, actual_interactions: int) -> str:
    """Names a threshold crossing without hiding update-boundary rounding."""
    return (
        f"requested_{requested_interactions:010d}"
        f"_actual_{actual_interactions:010d}"
    )


def crossed_thresholds(
    previous_interactions: int,
    current_interactions: int,
    thresholds: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        threshold
        for threshold in thresholds
        if previous_interactions < threshold <= current_interactions
    )
