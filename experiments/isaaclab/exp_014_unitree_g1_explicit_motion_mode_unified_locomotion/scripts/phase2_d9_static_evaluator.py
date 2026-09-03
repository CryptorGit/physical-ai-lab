"""Versioned fail-closed evaluator for Exp014 R4 stop distillation static contract V2."""
from __future__ import annotations

from typing import Any


class StaticThresholds:
    overall_mse_max = 0.001
    overall_cosine_min = 0.98
    boundary_mse_max = 0.001
    deceleration_mse_max = 0.001
    acquisition_mse_max = 0.001
    worst_condition_mse_max = 0.001


REQUIRED_INTEGRITY = (
    "material_label_conflict_zero", "teacher_id_input_leakage_zero",
    "condition_id_input_leakage_zero", "future_leakage_zero",
    "split_overlap_zero", "duplicate_sample_id_zero", "initialization_parity_pass",
)


def evaluate_checkpoint(metrics: dict[str, Any], integrity: dict[str, bool], thresholds: StaticThresholds = StaticThresholds()) -> dict[str, Any]:
    """Evaluate only checkpoint-specific action metrics plus immutable integrity gates.

    Independent phase-classifier metrics are deliberately ignored and are not
    copied into the returned checkpoint capability row.
    """
    required = {
        "overall_mse": metrics.get("mse"),
        "overall_cosine": metrics.get("cosine"),
        "boundary_mse": metrics.get("contexts", {}).get("3", {}).get("mse"),
        "deceleration_mse": metrics.get("contexts", {}).get("4", {}).get("mse"),
        "acquisition_mse": metrics.get("contexts", {}).get("5", {}).get("mse"),
        "worst_condition_mse": metrics.get("worst_condition_mse"),
    }
    missing = [name for name, value in required.items() if value is None]
    missing_integrity = [name for name in REQUIRED_INTEGRITY if name not in integrity]
    gates = {
        "overall_mse": required["overall_mse"] is not None and required["overall_mse"] <= thresholds.overall_mse_max,
        "overall_cosine": required["overall_cosine"] is not None and required["overall_cosine"] >= thresholds.overall_cosine_min,
        "boundary_mse": required["boundary_mse"] is not None and required["boundary_mse"] <= thresholds.boundary_mse_max,
        "deceleration_mse": required["deceleration_mse"] is not None and required["deceleration_mse"] <= thresholds.deceleration_mse_max,
        "acquisition_mse": required["acquisition_mse"] is not None and required["acquisition_mse"] <= thresholds.acquisition_mse_max,
        "worst_condition_mse": required["worst_condition_mse"] is not None and required["worst_condition_mse"] <= thresholds.worst_condition_mse_max,
    }
    integrity_gates = {name: bool(integrity.get(name, False)) for name in REQUIRED_INTEGRITY}
    eligible = not missing and not missing_integrity and all(gates.values()) and all(integrity_gates.values())
    reasons = [name for name, passed in gates.items() if not passed] + [name for name, passed in integrity_gates.items() if not passed]
    if missing: reasons.append("missing_checkpoint_specific_metrics:" + ",".join(missing))
    if missing_integrity: reasons.append("missing_integrity_metrics:" + ",".join(missing_integrity))
    return {"eligible": eligible, "action_metrics": required, "action_gates": gates,
            "integrity_gates": integrity_gates, "failure_reasons": reasons,
            "diagnostic_metrics_used_for_eligibility": [], "fail_closed": True}


def selection_key(row: dict[str, Any]) -> tuple:
    """Validation-only ordering from Static Contract V2; eligible rows only."""
    if not row["v2_eligible"]:
        raise ValueError("selection_key requires an eligible checkpoint")
    return (row["boundary_mse"], row["worst_condition_mse"], row["acquisition_mse"],
            row["overall_mse"], -row["overall_cosine"], row["parameter_movement"], row["training_step"])
