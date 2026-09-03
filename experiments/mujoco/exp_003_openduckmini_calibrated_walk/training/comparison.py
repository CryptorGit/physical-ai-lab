"""Leaf-level bit identity comparison for harness checkpoints."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def stable_path(path: tuple[Any, ...]) -> str:
    return "/".join(
        (
            f"{type(key).__name__}:"
            f"{getattr(key, 'key', getattr(key, 'idx', getattr(key, 'name', '')))}"
        )
        for key in path
    )


def compare_trees(reference: Any, comparison: Any) -> list[dict[str, Any]]:
    left, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(reference))
    right, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(comparison))
    if len(left) != len(right):
        raise ValueError(f"leaf count differs: {len(left)} != {len(right)}")
    rows: list[dict[str, Any]] = []
    for (left_path, left_leaf), (right_path, right_leaf) in zip(left, right):
        left_name = stable_path(left_path)
        right_name = stable_path(right_path)
        if left_name != right_name:
            raise ValueError(f"path differs: {left_name} != {right_name}")
        a = np.asarray(left_leaf)
        b = np.asarray(right_leaf)
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append(
                {
                    "path": left_name,
                    "shape": str(a.shape),
                    "dtype": str(a.dtype),
                    "bit_exact": False,
                    "max_abs_error": None,
                    "different_count": None,
                    "reason": f"shape/dtype {a.shape}/{a.dtype} != {b.shape}/{b.dtype}",
                }
            )
            continue
        bit_exact = a.tobytes(order="C") == b.tobytes(order="C")
        if np.issubdtype(a.dtype, np.number):
            difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
            max_abs = float(np.nanmax(difference)) if difference.size else 0.0
            a_bytes = np.frombuffer(a.tobytes(order="C"), dtype=np.uint8)
            b_bytes = np.frombuffer(b.tobytes(order="C"), dtype=np.uint8)
            different = int(np.count_nonzero(a_bytes != b_bytes))
        else:
            max_abs = None
            different = int(not bit_exact)
        rows.append(
            {
                "path": left_name,
                "shape": str(a.shape),
                "dtype": str(a.dtype),
                "bit_exact": bit_exact,
                "max_abs_error": max_abs,
                "different_count": different,
                "reason": "" if bit_exact else "value_bytes",
            }
        )
    return rows
