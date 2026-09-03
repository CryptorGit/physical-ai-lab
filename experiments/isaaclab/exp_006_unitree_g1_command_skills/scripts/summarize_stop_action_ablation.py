"""Combine STOP action-space ablation summaries into CSV and JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CONDITIONS = (
    "A_current",
    "B_yaw_mask",
    "C_yaw_ankle_roll_mask",
    "D_lateral_mask",
    "E_symmetric",
)

SKILL_FIELDS = {
    "episodes": "count",
    "success_rate": "success_rate",
    "position_error_m": "stop_position_error_m",
    "hold_end_speed_mps": "stop_hold_end_speed_mps",
    "hold_success_rate": "stop_hold_success_rate",
    "heading_error_mean_rad": "stop_heading_error_mean_rad",
    "heading_error_p95_rad": "stop_heading_error_p95_rad",
    "heading_error_max_rad": "stop_heading_error_max_rad",
    "stop_fall_rate": "fall_rate",
    "saturation_failure_rate": "saturation_failure_rate",
    "maximum_action_magnitude": "maximum_action_magnitude",
    "actual_yaw_rate_abs_mean_rps": "actual_yaw_rate_abs_mean_rps",
    "actual_yaw_rate_abs_p95_rps": "actual_yaw_rate_abs_p95_rps",
    "actual_yaw_rate_abs_max_rps": "actual_yaw_rate_abs_max_rps",
    "legacy_yaw_rate_command_mean_rps": "legacy_yaw_rate_command_mean_rps",
    "legacy_yaw_rate_command_abs_max_rps": "legacy_yaw_rate_command_abs_max_rps",
    "raw_residual_norm": "stop_raw_residual_norm",
    "masked_residual_norm": "stop_masked_residual_norm",
    "braking_speed_tracking_error_mps": "braking_speed_tracking_error_mps",
    "minimum_speed_mps": "stop_phase_min_speed_mps",
    "braking_end_speed_mps": "stop_braking_end_speed_mps",
    "hold_start_speed_mps": "stop_hold_start_speed_mps",
    "hold_max_speed_mps": "stop_hold_max_speed_mps",
    "hold_speed_rebound_mps": "stop_hold_max_speed_rebound_mps",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        summary_path = args.root / condition / "summary.json"
        with summary_path.open(encoding="utf-8") as stream:
            summary = json.load(stream)
        stop = summary["skills"]["STOP"]
        row: dict[str, object] = {
            "condition": condition,
            "ablation_mode": summary["stop_residual_ablation"],
        }
        row.update({output: stop[source] for output, source in SKILL_FIELDS.items()})
        curve_path = args.root / condition / "stop_curve.csv"
        with curve_path.open(encoding="utf-8", newline="") as stream:
            curve = list(csv.DictReader(stream))
        row["actual_yaw_rate_mean_rps"] = sum(
            float(point["actual_yaw_rate_rps"]) for point in curve
        ) / len(curve)
        row["failure_reason_counts"] = json.dumps(summary["failure_reason_counts"], sort_keys=True)
        rows.append(row)

    json_path = args.root / "comparison.json"
    csv_path = args.root / "comparison.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(rows, stream, indent=2)
        stream.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
