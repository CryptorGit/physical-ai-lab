"""Combine standing-base summaries into a comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = (
    "candidate", "settle_success_rate", "standing_hold_success_rate", "fall_rate",
    "actual_speed_mean_mps", "actual_speed_p95_mps", "actual_speed_max_mps",
    "yaw_rate_abs_mean_rps", "yaw_rate_p95_rps", "yaw_rate_max_rps",
    "pelvis_height_range_mean_m", "vertical_velocity_p95_mps", "vertical_velocity_max_mps",
    "roll_abs_p95_rad", "pitch_abs_p95_rad", "double_support_fraction",
    "single_support_fraction", "flight_fraction", "support_switch_count_mean",
    "prolonged_single_support_rate", "ankle_torque_saturation_failure_rate",
    "joint_velocity_saturation_failure_rate", "standing_height_m", "action_magnitude_mean",
    "action_magnitude_p95", "action_magnitude_max", "left_right_sagittal_asymmetry_rad", "gate_pass",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.resolve().glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        rows.append({field: summary.get(field) for field in FIELDS})
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.output.resolve().open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
