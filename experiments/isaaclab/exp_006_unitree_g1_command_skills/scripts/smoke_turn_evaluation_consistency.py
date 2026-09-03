"""Check TURN CSV, summary, and gate use one accumulated-yaw metric."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1.0e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skills", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    with args.skills.resolve(strict=True).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["skill"] == "TURN"]
    summary = json.loads(args.summary.resolve(strict=True).read_text(encoding="utf-8"))
    gate = json.loads(args.gate.resolve(strict=True).read_text(encoding="utf-8"))
    turn = summary["skills"]["TURN"]
    csv_mean = sum(float(row["final_turn_angle_error_rad"]) for row in rows) / len(rows)
    commanded = sorted(round(float(row["commanded_turn_angle_rad"]), 6) for row in rows)
    expected = sorted(round(value, 6) for value in (-math.pi / 2, -math.pi / 4, math.pi / 4, math.pi / 2))
    report = {
        "turn_rows": len(rows),
        "commanded_angles_rad": commanded,
        "all_four_angle_direction_targets_present": commanded == expected,
        "csv_final_error_mean": csv_mean,
        "summary_final_error": turn["final_turn_angle_error_rad"],
        "summary_heading_error": turn["heading_error_rad"],
        "gate_heading_error": gate["metrics"]["turn"]["heading_error_rad"],
        "gate_final_error": gate["metrics"]["turn"]["final_turn_angle_error_rad"],
        "csv_summary_match": close(csv_mean, turn["final_turn_angle_error_rad"]),
        "summary_heading_match": close(turn["heading_error_rad"], turn["final_turn_angle_error_rad"]),
        "gate_heading_match": close(
            gate["metrics"]["turn"]["heading_error_rad"], turn["final_turn_angle_error_rad"]
        ),
        "gate_straight_recovery_match": close(
            gate["metrics"]["turn"]["straight_recovery_rate"], summary["turn_straight_recovery_rate"]
        ),
    }
    report["passed"] = all(
        report[key]
        for key in (
            "all_four_angle_direction_targets_present",
            "csv_summary_match",
            "summary_heading_match",
            "gate_heading_match",
            "gate_straight_recovery_match",
        )
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("TURN evaluation consistency smoke failed")


if __name__ == "__main__":
    main()
