"""Calibrate hidden-physics ranges from a real Qmini baseline metric table.

The script refuses to manufacture a calibration result. A baseline table and
explicit candidate ranges are required before any range can be frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = EXP_ROOT.parents[2] / "results" / "exp_015_qmini_population_bodily_world_model"


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("episodes", payload.get("rows", []))
    if not isinstance(payload, list):
        raise ValueError("baseline table must be a JSON list or an object with episodes/rows")
    return [dict(row) for row in payload]


def factor_effect(rows: list[dict[str, Any]], factor: str) -> dict[str, Any]:
    high = [row for row in rows if row.get("hidden_factor") == factor and row.get("condition") == "high"]
    low = [row for row in rows if row.get("hidden_factor") == factor and row.get("condition") == "low"]
    metric_names = ("forward_velocity", "lateral_velocity", "slip", "torque_rms", "mechanical_power", "body_tilt", "fatigue")
    effects: dict[str, float] = {}
    for metric in metric_names:
        high_values = [float(row[metric]) for row in high if row.get(metric) is not None]
        low_values = [float(row[metric]) for row in low if row.get(metric) is not None]
        if high_values and low_values:
            effects[metric] = mean(high_values) - mean(low_values)
    relevant = any(abs(value) > 0.01 for value in effects.values())
    return {
        "factor": factor,
        "high_rows": len(high),
        "low_rows": len(low),
        "metric_effects_high_minus_low": effects,
        "relevant": relevant,
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    factors = ("floor_friction", "wind_xy", "left_leg_actuator_fatigue", "right_leg_actuator_fatigue")
    if args.baseline_table is None:
        return {
            "status": "NO_GO_QMINI_BASELINE",
            "reason": "50-episode Qmini formal baseline table was not supplied",
            "frozen_ranges": "UNKNOWN",
            "fatigue_parameters": "UNKNOWN",
            "factor_effects": [],
        }
    rows = load_rows(args.baseline_table)
    if len(rows) < 50:
        return {
            "status": "NO_GO_QMINI_BASELINE",
            "reason": f"only {len(rows)} baseline rows; at least 50 required",
            "frozen_ranges": "UNKNOWN",
            "fatigue_parameters": "UNKNOWN",
            "factor_effects": [],
        }
    if args.ranges_json is None:
        raise ValueError("explicit candidate ranges are required; no Qmini ranges are guessed")
    ranges = json.loads(args.ranges_json)
    effects = [factor_effect(rows, factor) for factor in factors]
    worst_fall = max((float(row.get("fell", 0.0)) for row in rows), default=1.0)
    fatigue_parameters = {
        "alpha": args.fatigue_alpha,
        "beta": args.fatigue_beta,
        "effectiveness_coefficient": args.effectiveness_coefficient,
        "power_normalization_reference": args.power_normalization_reference,
    }
    parameter_complete = all(value is not None for value in fatigue_parameters.values())
    relevant = all(item["relevant"] for item in effects)
    safe = worst_fall <= 0.10
    status = "PASS" if relevant and safe and parameter_complete else "NO_GO_HIDDEN_FACTOR_RELEVANCE"
    return {
        "status": status,
        "baseline_rows": len(rows),
        "worst_condition_fall_rate": worst_fall,
        "worst_condition_fall_rate_max": 0.10,
        "candidate_ranges": ranges,
        "frozen_ranges": ranges if status == "PASS" else "NOT_FROZEN",
        "fatigue_parameters": fatigue_parameters if parameter_complete else "UNKNOWN",
        "factor_effects": effects,
        "source": "MEASURED_QMINI_BASELINE_ONLY",
        "g1_values_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-table", type=Path)
    parser.add_argument("--ranges-json", help="explicit JSON object of candidate ranges")
    parser.add_argument("--fatigue-alpha", type=float)
    parser.add_argument("--fatigue-beta", type=float)
    parser.add_argument("--effectiveness-coefficient", type=float)
    parser.add_argument("--power-normalization-reference", type=float)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "hidden_physics_calibration.json")
    args = parser.parse_args()
    report = calibrate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
