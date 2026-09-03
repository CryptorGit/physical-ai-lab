"""Pure Stage 1 metric, failure, gate, and retention helpers."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Iterable


FAILURE_PRIORITY = (
    "action_routing_failure",
    "fall",
    "torso_contact",
    "stand_settle_failure",
    "timeout",
    "stand_hold_failure",
    "flight_failure",
    "dangerous_support_loss",
    "double_support_failure",
    "saturation_failure",
    "joint_limit_failure",
    "posture_failure",
    "excessive_horizontal_motion",
    "excessive_vertical_motion",
    "foot_slip_failure",
)

GATE_THRESHOLDS = {
    "settle_success_rate_min": 0.95,
    "stand_hold_success_rate_min": 0.95,
    "fall_rate_max": 0.02,
    "horizontal_speed_mean_mps_max": 0.05,
    "horizontal_speed_p95_mps_max": 0.10,
    "pelvis_height_range_mean_m_max": 0.04,
    "flight_fraction_exact": 0.0,
    "dangerous_support_failure_rate_max": 0.05,
    "saturation_failure_rate_max": 0.05,
    "final_double_support_rate_min": 0.95,
}


def mean(values: Iterable[float]) -> float:
    data = list(values)
    return sum(data) / len(data) if data else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    return data[min(round((len(data) - 1) * q / 100.0), len(data) - 1)]


def failure_flags(record: dict[str, Any]) -> dict[str, bool]:
    flags = {
        "stand_settle_failure": not bool(record["settle_success"]),
        "stand_hold_failure": not bool(record["stand_hold_success"]),
        "fall": bool(record["fall"]),
        "torso_contact": bool(record["torso_contact"]),
        "excessive_horizontal_motion": (
            float(record["horizontal_speed_mean_mps"]) > 0.05
            or float(record["horizontal_speed_p95_mps"]) > 0.10
        ),
        "excessive_vertical_motion": (
            float(record["vertical_speed_p95_mps"]) > 0.05
            or abs(float(record["final_vertical_speed_mps"])) > 0.05
        ),
        "posture_failure": (
            float(record["roll_abs_p95_rad"]) > 0.10
            or float(record["pitch_abs_p95_rad"]) > 0.10
            or float(record["roll_abs_max_rad"]) > 0.20
            or float(record["pitch_abs_max_rad"]) > 0.20
        ),
        "double_support_failure": not bool(record["final_double_support"]),
        "flight_failure": float(record["flight_fraction"]) != 0.0,
        "dangerous_support_loss": bool(record["dangerous_support_loss"]),
        "foot_slip_failure": float(record["foot_slip_p95_mps"]) > 0.10,
        "saturation_failure": bool(record["saturation_failure"]),
        "joint_limit_failure": bool(record["joint_limit_failure"]),
        "action_routing_failure": bool(record["action_routing_failure"]),
        "timeout": bool(record["timeout"]),
    }
    return flags


def classify_failures(record: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    flags = failure_flags(record)
    primary = next((name for name in FAILURE_PRIORITY if flags[name]), "")
    return primary, flags


def summarize_failures(records: list[dict[str, Any]]) -> dict[str, Any]:
    primary = Counter(record["primary_failure"] or "none" for record in records)
    flags = Counter()
    for record in records:
        for name, value in record["failure_flags"].items():
            if value:
                flags[name] += 1
    return {
        "episodes": len(records),
        "primary_failure_counts": dict(sorted(primary.items())),
        "all_failure_flag_counts": {name: flags.get(name, 0) for name in FAILURE_PRIORITY},
    }


def evaluate_gate(metrics: dict[str, float], static_checks: dict[str, bool]) -> tuple[bool, list[str]]:
    failures = []
    comparisons = {
        "settle_success_rate": metrics["settle_success_rate"] >= GATE_THRESHOLDS["settle_success_rate_min"],
        "stand_hold_success_rate": metrics["stand_hold_success_rate"] >= GATE_THRESHOLDS["stand_hold_success_rate_min"],
        "fall_rate": metrics["fall_rate"] <= GATE_THRESHOLDS["fall_rate_max"],
        "horizontal_speed_mean_mps": metrics["horizontal_speed_mean_mps"] <= GATE_THRESHOLDS["horizontal_speed_mean_mps_max"],
        "horizontal_speed_p95_mps": metrics["horizontal_speed_p95_mps"] <= GATE_THRESHOLDS["horizontal_speed_p95_mps_max"],
        "pelvis_height_range_mean_m": metrics["pelvis_height_range_mean_m"] <= GATE_THRESHOLDS["pelvis_height_range_mean_m_max"],
        "flight_fraction": metrics["flight_fraction"] == GATE_THRESHOLDS["flight_fraction_exact"],
        "dangerous_support_failure_rate": metrics["dangerous_support_failure_rate"] <= GATE_THRESHOLDS["dangerous_support_failure_rate_max"],
        "saturation_failure_rate": metrics["saturation_failure_rate"] <= GATE_THRESHOLDS["saturation_failure_rate_max"],
        "final_double_support_rate": metrics["final_double_support_rate"] >= GATE_THRESHOLDS["final_double_support_rate_min"],
    }
    for name, passed in {**comparisons, **static_checks}.items():
        if not passed:
            failures.append(name)
    return not failures, failures


EXP006_REFERENCE = {
    "settle_success_rate": 0.98,
    "stand_hold_success_rate": 0.98,
    "fall_rate": 0.02,
    "horizontal_speed_mean_mps": 0.006718055695095973,
    "horizontal_speed_p95_mps": 0.01334766335785389,
    "flight_fraction": 0.0,
    "final_double_support_rate": 0.98,
    "saturation_failure_rate": 0.0,
}


def retention_vs_exp006(metrics: dict[str, float]) -> dict[str, Any]:
    rows = {}
    for name, reference in EXP006_REFERENCE.items():
        current = float(metrics[name])
        rows[name] = {
            "exp006": reference,
            "exp007": current,
            "absolute_difference": current - reference,
            "percentage_point_drop": (reference - current) * 100.0 if "rate" in name or "fraction" in name else None,
        }
    checks = {
        "settle_drop_lt_5pp": EXP006_REFERENCE["settle_success_rate"] - metrics["settle_success_rate"] < 0.05,
        "hold_drop_lt_5pp": EXP006_REFERENCE["stand_hold_success_rate"] - metrics["stand_hold_success_rate"] < 0.05,
        "fall_worsening_le_2pp": metrics["fall_rate"] - EXP006_REFERENCE["fall_rate"] <= 0.02,
        "saturation_worsening_lt_5pp": metrics["saturation_failure_rate"] - EXP006_REFERENCE["saturation_failure_rate"] < 0.05,
        "speed_p95_within_gate": metrics["horizontal_speed_p95_mps"] <= GATE_THRESHOLDS["horizontal_speed_p95_mps_max"],
        "flight_remains_zero": metrics["flight_fraction"] == 0.0,
    }
    return {
        "status": "RETAINED" if all(checks.values()) else "DEGRADED",
        "metrics": rows,
        "checks": checks,
        "evaluation_condition_difference": {
            "exp006_hold_s": 6.0,
            "exp007_hold_s": 8.0,
            "otherwise_same_task_seed_reset_and_checkpoint": True,
            "interpretation": "Comparable; exp_007 uses a strictly longer observation window.",
        },
    }
