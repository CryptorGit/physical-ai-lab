"""Apply the mandatory standing-base gate before CROUCH training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CHECKS = (
    ("standing_hold_success_rate", ">=", 0.95),
    ("settle_success_rate", ">=", 0.95),
    ("fall_rate", "<=", 0.05),
    ("actual_speed_mean_mps", "<=", 0.05),
    ("actual_speed_p95_mps", "<=", 0.10),
    ("pelvis_height_range_mean_m", "<=", 0.04),
    ("flight_fraction", "<=", 0.001),
    ("prolonged_single_support_rate", "<=", 0.05),
    ("ankle_torque_saturation_failure_rate", "<=", 0.05),
    ("joint_velocity_saturation_failure_rate", "<=", 0.05),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.resolve(strict=True).read_text(encoding="utf-8"))
    failures = []
    checks = {}
    for metric, operation, threshold in CHECKS:
        value = float(summary[metric])
        passed = value >= threshold if operation == ">=" else value <= threshold
        checks[metric] = {"value": value, "operation": operation, "threshold": threshold, "pass": passed}
        if not passed:
            failures.append(f"{metric} {operation} {threshold} failed (value={value})")
    report = {
        "gate": "standing_base_v1", "candidate": summary["candidate"],
        "checkpoint": summary.get("checkpoint"), "summary": str(args.summary.resolve()),
        "episodes": summary["episodes"], "seed": summary["seed"],
        "eligible_for_crouch": not failures, "failures": failures, "checks": checks,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
