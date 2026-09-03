"""Finalize compact metadata after partitioned physics logging."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"
CHECKPOINT = "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt"
CHECKPOINT_SHA = "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    columns = [
        "episode_id",
        "evaluation_seed",
        "source_speed_mps",
        "source_phase",
        "policy_sampling",
        "transition_step",
        "relative_to_first_contact",
        "relative_to_break",
        "break_reason",
        "episode_success",
        "walk_valid_streak_age",
        "split",
    ]
    frame = pd.read_parquet(OUT / "episodes.parquet", columns=columns)
    grouped = frame.groupby("episode_id", sort=False)
    index = grouped.agg(
        evaluation_seed=("evaluation_seed", "first"),
        source_speed_mps=("source_speed_mps", "first"),
        source_phase=("source_phase", "first"),
        policy_sampling=("policy_sampling", "first"),
        first_walk_contact_step=("relative_to_first_contact", lambda values: int(frame.loc[values.index, "transition_step"].iloc[0] - values.iloc[0])),
        break_step=("relative_to_break", lambda values: int(frame.loc[values.index, "transition_step"].iloc[0] - values.iloc[0])),
        break_reason=("break_reason", "first"),
        maximum_walk_valid_streak=("walk_valid_streak_age", "max"),
        episode_success=("episode_success", "first"),
        window_start=("transition_step", "min"),
        window_end=("transition_step", "max"),
        split=("split", "first"),
    ).reset_index()
    index.to_csv(OUT / "sequence_index.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    split_mapping = dict(zip(index["episode_id"], index["split"]))
    dump(
        "split_manifest.json",
        {
            "unit": "episode grouped by evaluation_seed/source_speed/checkpoint",
            "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "episode_counts": dict(Counter(index["split"])),
            "episode_mapping": split_mapping,
            "step_random_split": False,
            "same_reset_seed_cross_split": False,
        },
    )
    speed_counts = Counter(round(float(value), 1) for value in index["source_speed_mps"])
    phase_counts = Counter(index["source_phase"])
    reason_counts = Counter(index["break_reason"])
    sampling_counts = Counter(index["policy_sampling"])
    successful = int(index["episode_success"].sum())
    dump(
        "dataset_manifest.json",
        {
            "source": "frozen diagnostic replay because exp_007 saved trajectories lacked complete 152D/action sequences",
            "checkpoint": CHECKPOINT,
            "checkpoint_sha256": CHECKPOINT_SHA,
            "episodes": int(len(index)),
            "rows": int(len(frame)),
            "successful_20_step_segments": successful,
            "failed_segments": int(len(index) - successful),
            "success_target": 200,
            "failure_target": 2000,
            "success_target_met": successful >= 200,
            "failure_target_met": len(index) - successful >= 2000,
            "source_speed_counts": {str(key): value for key, value in speed_counts.items()},
            "source_phase_counts": dict(phase_counts),
            "break_reason_counts": dict(reason_counts),
            "policy_sampling_counts": dict(sampling_counts),
            "source_preparation_attempts": 4096,
            "source_ready_selected": 2048,
            "pre_window_steps": 16,
            "post_window_steps": 8,
            "parquet_layout": "partitioned directory",
            "ppo_training": 0,
            "ppo_optimizer_updates": 0,
            "transition_actor_optimizer_updates": 0,
        },
    )
    print(json.dumps({"episodes": len(index), "rows": len(frame), "successful": successful}, indent=2))


if __name__ == "__main__":
    main()
