"""Partitioned parquet dataset utilities."""

from __future__ import annotations

import hashlib


def grouped_split(seed: int, episode_id: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"{seed}:{episode_id}".encode()).digest()[:8], "little") / 2**64
    return "train" if value < 0.70 else "validation" if value < 0.85 else "test"


def observation_columns():
    return [f"obs_{index:03d}" for index in range(123)]


def action_columns(prefix="action"):
    return [f"{prefix}_{index:03d}" for index in range(37)]
