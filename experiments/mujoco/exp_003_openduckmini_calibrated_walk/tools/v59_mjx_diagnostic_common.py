"""Shared, production-independent utilities for the v59 MJX one-step audit."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

import jax
import numpy as np


def host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda value: np.asarray(jax.device_get(value))
        if hasattr(value, "dtype")
        else value,
        tree,
    )


def key_path_name(path: Iterable[Any]) -> str:
    parts = []
    for entry in path:
        if hasattr(entry, "name"):
            parts.append(str(entry.name))
        elif hasattr(entry, "key"):
            parts.append(str(entry.key))
        elif hasattr(entry, "idx"):
            parts.append(str(entry.idx))
        else:
            parts.append(str(entry))
    return ".".join(parts)


def numeric_leaves(tree: Any) -> dict[str, np.ndarray]:
    pairs, _ = jax.tree_util.tree_flatten_with_path(tree)
    result: dict[str, np.ndarray] = {}
    for path, value in pairs:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            continue
        if array.dtype.kind in "biufc":
            result[key_path_name(path)] = array
    return result


def canonical_tree_sha256(tree: Any) -> str:
    digest = hashlib.sha256()
    for path, array in sorted(numeric_leaves(tree).items()):
        contiguous = np.ascontiguousarray(array)
        digest.update(path.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def dump_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as stream:
        return pickle.load(stream)


def single_tree(tree: Any, index: int) -> Any:
    return jax.tree_util.tree_map(lambda value: value[index], tree)


def select_randomized_model(model: Any, in_axes: Any, index: int) -> Any:
    return jax.tree_util.tree_map(
        lambda value, axis: value[index] if axis == 0 else value,
        model,
        in_axes,
        is_leaf=lambda value: value is None,
    )

