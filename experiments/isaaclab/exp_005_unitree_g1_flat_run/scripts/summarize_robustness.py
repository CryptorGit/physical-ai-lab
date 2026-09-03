"""Aggregate resumable Stage 9 robustness evaluations into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _joint_metrics(condition_dir: Path) -> dict[str, float]:
    with (condition_dir / "joint_diagnostics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    def max_joint_mean(pattern: str, field: str) -> float:
        by_joint: dict[str, list[float]] = {}
        for row in rows:
            if pattern in row["joint"]:
                by_joint.setdefault(row["joint"], []).append(float(row[field]))
        return max((_mean(values) for values in by_joint.values()), default=0.0)

    return {
        "knee_velocity_saturation_fraction": max_joint_mean(
            "knee_joint", "velocity_limit_fraction"
        ),
        "ankle_torque_saturation_fraction": max_joint_mean(
            "ankle", "torque_limit_fraction"
        ),
    }


def _recovery_metrics(condition_dir: Path) -> dict[str, float]:
    with (condition_dir / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    applied = [row for row in rows if row["external_force_applied"].lower() == "true"]
    recovered = [row for row in applied if row["external_force_recovered"].lower() == "true"]
    return {
        "external_force_recovery_rate": len(recovered) / len(applied) if applied else 0.0,
        "external_force_mean_recovery_time_s": _mean(
            [float(row["external_force_recovery_time_s"]) for row in recovered]
        ),
    }


def _failure_labels(row: dict[str, object]) -> tuple[str, str]:
    failures: list[str] = []
    if float(row["fall_rate"]) > 0.05:
        failures.append("fall_failure")
    if (
        float(row["mean_actual_speed_mps"]) < 4.75
        or float(row["steady_mean_abs_forward_error_mps"]) > 0.25
    ):
        failures.append("tracking_failure")
    if float(row["periodic_running_rate"]) < 0.80:
        failures.append("periodicity_failure")
    if (
        float(row["mean_contact_foot_slip_mps"]) > 0.55
        or float(row["left_contact_slip_mps"]) > 0.60
        or float(row["right_contact_slip_mps"]) > 0.60
    ):
        failures.append("slip_failure")
    if (
        float(row["knee_velocity_saturation_fraction"]) > 0.05
        or float(row["ankle_torque_saturation_fraction"]) > 0.20
    ):
        failures.append("saturation_failure")
    if (
        float(row["landing_impact_p95_n"]) > 3500.0
        or float(row["landing_impact_over_3500_rate"]) > 0.05
    ):
        failures.append("impact_failure")

    if not failures:
        if float(row["physical_quality_gate_pass_rate"]) >= 0.80:
            return "robust_pass", ""
        return "degraded_but_stable", ""
    return failures[0], ",".join(failures[1:])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows: list[dict[str, object]] = []

    for summary_path in sorted(root.glob("*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if not payload.get("summaries"):
            continue
        summary = payload["summaries"][0]
        condition_dir = summary_path.parent
        row: dict[str, object] = {
            "condition": payload.get("condition_name", condition_dir.name),
            "episodes": summary["episodes"],
            "mean_actual_speed_mps": summary["mean_actual_speed_mps"],
            "steady_mean_abs_forward_error_mps": summary["steady_mean_abs_forward_error_mps"],
            "fall_rate": summary["fall_rate"],
            "periodic_running_rate": summary["periodic_running_rate"],
            "mean_contact_foot_slip_mps": summary["mean_contact_foot_slip_mps"],
            "left_contact_slip_mps": summary["left_contact_slip_mps"],
            "right_contact_slip_mps": summary["right_contact_slip_mps"],
            "mean_flight_duration_s": summary["mean_flight_duration_s"],
            "max_consecutive_running_cycles": summary["max_consecutive_running_cycles"],
            "landing_impact_p95_n": summary["landing_impact_p95_n"],
            "landing_impact_over_3500_rate": summary["landing_impact_over_3500_rate"],
            "stride_asymmetry": summary["stride_asymmetry"],
            "contact_time_asymmetry": summary["contact_time_asymmetry"],
            "physical_quality_gate_pass_rate": summary["physical_quality_gate_pass_rate"],
            **_joint_metrics(condition_dir),
            **_recovery_metrics(condition_dir),
        }
        primary, secondary = _failure_labels(row)
        row["classification"] = primary
        row["primary_failure"] = "" if primary in ("robust_pass", "degraded_but_stable") else primary
        row["secondary_failures"] = secondary
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No completed condition summaries found below {root}")

    csv_path = root / "robustness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "robustness_summary.json").write_text(
        json.dumps({"conditions_completed": len(rows), "conditions": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} conditions to {csv_path}")


if __name__ == "__main__":
    main()
