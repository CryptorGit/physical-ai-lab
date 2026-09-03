"""Export the two bounded Stage 2W TensorBoard runs to one audit CSV."""

from __future__ import annotations

import csv
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parent.parent.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2w_independent_walk/training_curves.csv"
RUNS = {
    "pilot1": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_22-32-29_stage2w_independent_walk_pilot1_1024_150",
    "pilot2": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_22-39-22_stage2w_independent_walk_pilot2_1024_100",
}
TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/success_rate",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Policy/mean_std",
    "Loss/value",
    "Loss/surrogate",
    "Loss/entropy",
)


def main() -> None:
    records = []
    for run, directory in RUNS.items():
        event = next(directory.glob("events.out.*"))
        accumulator = EventAccumulator(str(event))
        accumulator.Reload()
        for tag in TAGS:
            for scalar in accumulator.Scalars(tag):
                records.append(
                    {
                        "run": run,
                        "step": scalar.step,
                        "wall_time": scalar.wall_time,
                        "metric": tag,
                        "value": scalar.value,
                    }
                )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("run", "step", "wall_time", "metric", "value"))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
