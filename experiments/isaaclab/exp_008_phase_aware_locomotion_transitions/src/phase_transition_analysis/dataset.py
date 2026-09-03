"""Parquet dataset loading, grouped splitting, and history construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_layout import ANALYSIS_PHASE_FIELDS, FEATURE_CONDITIONS


def observation_columns():
    return [f"obs_{index:03d}" for index in range(152)]


def action_columns():
    return [f"action_{index:03d}" for index in range(37)]


def load_dataset(path: str | Path):
    return pd.read_parquet(path)


def assign_grouped_splits(frame, seed: int, fractions=(0.6, 0.2, 0.2)):
    """Assign complete episodes; no step or neighboring trajectory leaks."""
    episodes = frame[["episode_id", "evaluation_seed", "source_speed_mps", "checkpoint"]].drop_duplicates()
    groups = episodes[["evaluation_seed", "source_speed_mps", "checkpoint"]].drop_duplicates()
    keyed = []
    for row in groups.itertuples(index=False):
        key = (row.evaluation_seed, row.source_speed_mps, row.checkpoint)
        digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
        keyed.append((int.from_bytes(digest[:8], "little"), key))
    keyed.sort()
    count = len(keyed)
    train_end, validation_end = round(count * fractions[0]), round(count * (fractions[0] + fractions[1]))
    group_mapping = {}
    for index, (_, key) in enumerate(keyed):
        group_mapping[key] = "train" if index < train_end else "validation" if index < validation_end else "test"
    mapping = {
        row.episode_id: group_mapping[(row.evaluation_seed, row.source_speed_mps, row.checkpoint)]
        for row in episodes.itertuples(index=False)
    }
    result = frame.copy()
    result["split"] = result["episode_id"].map(mapping)
    return result, mapping


def feature_matrix(frame, condition: str):
    aliases = {
        "A": "A_full_152D",
        "B": "B_152D_without_timing",
        "C": "C_legacy_123D",
        "D": "D_legacy_123D_plus_action",
        "E": "E_explicit_phase_upper_bound",
    }
    condition = aliases.get(condition, condition)
    spec = FEATURE_CONDITIONS[condition]
    columns = [f"obs_{index:03d}" for index in spec["observation_indices"]]
    if spec["include_action"]:
        columns += action_columns()
    if spec["include_phase"]:
        columns += list(ANALYSIS_PHASE_FIELDS)
    return frame[columns].to_numpy(dtype=np.float32), columns


def history_matrix(frame, condition: str, history: int):
    """Return flattened histories that never cross episode boundaries."""
    base, columns = feature_matrix(frame, condition)
    rows, indices = [], []
    positions = {index: position for position, index in enumerate(frame.index)}
    for _, group in frame.groupby("episode_id", sort=False):
        loc = [positions[index] for index in group.sort_values("transition_step").index]
        for offset in range(history - 1, len(loc)):
            rows.append(base[loc[offset - history + 1 : offset + 1]].reshape(-1))
            indices.append(group.sort_values("transition_step").index[offset])
    return np.asarray(rows, dtype=np.float32), indices, [f"h{lag}_{name}" for lag in range(history) for name in columns]
