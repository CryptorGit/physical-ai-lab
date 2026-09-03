"""Recompute geometric CROUCH depth fields from an existing curve CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, q):
    values = sorted(values)
    return values[min(round((len(values) - 1) * q / 100.0), len(values) - 1)] if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.curve.resolve(strict=True).open(newline="", encoding="utf-8") as stream:
        source = list(csv.DictReader(stream))
    records = []
    for episode in sorted({int(row["episode"]) for row in source}):
        rows = [row for row in source if int(row["episode"]) == episode]
        entry = float(rows[-1]["entry_pelvis_height_m"])
        commanded = float(rows[-1]["commanded_height_drop_m"])
        heights = [float(row["pelvis_height_m"]) for row in rows]
        hold = [float(row["pelvis_height_m"]) for row in rows if int(row["phase"]) == 2]
        returning = [float(row["pelvis_height_m"]) for row in rows if int(row["phase"]) == 3]
        minimum = min(heights)
        actual = entry - minimum
        records.append({
            "episode": episode, "commanded_height_drop_m": commanded,
            "entry_pelvis_height_m": entry, "target_minimum_pelvis_height_m": entry - commanded,
            "actual_minimum_pelvis_height_m": minimum, "actual_height_drop_m": actual,
            "depth_error_m": abs(commanded - actual),
            "hold_start_height_drop_m": entry - hold[0] if hold else 0.0,
            "hold_height_drop_mean_m": entry - mean(hold) if hold else 0.0,
            "hold_height_drop_p95_m": percentile([entry - value for value in hold], 95),
            "return_start_pelvis_height_m": returning[0] if returning else entry,
            "down_reached": actual >= commanded - 0.04,
        })
    args.output.resolve().mkdir(parents=True, exist_ok=True)
    with (args.output.resolve() / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    summary = {
        "source_curve": str(args.curve.resolve()), "episodes": len(records),
        "formula": {"actual_height_drop": "entry_pelvis_height - actual_minimum_pelvis_height",
                    "depth_error": "abs(commanded_height_drop - actual_height_drop)"},
        "commanded_height_drop_m": mean([row["commanded_height_drop_m"] for row in records]),
        "actual_height_drop_m": mean([row["actual_height_drop_m"] for row in records]),
        "depth_error_m": mean([row["depth_error_m"] for row in records]),
        "hold_start_height_drop_m": mean([row["hold_start_height_drop_m"] for row in records]),
        "hold_height_drop_mean_m": mean([row["hold_height_drop_mean_m"] for row in records]),
        "return_start_pelvis_height_m": mean([row["return_start_pelvis_height_m"] for row in records]),
        "down_reached_rate": mean([float(row["down_reached"]) for row in records]),
        "prior_return_success_interpretation": "kinematic entry-height retention without prior DOWN achievement",
    }
    (args.output.resolve() / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
