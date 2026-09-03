"""Apply the production CROUCH_SHALLOW gate to formal evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--retention", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.resolve(strict=True).read_text(encoding="utf-8"))
    retention = json.loads(args.retention.resolve(strict=True).read_text(encoding="utf-8"))
    metric = summary["skills"]["CROUCH_SHALLOW"]
    run = retention["inherited_metrics"]["run"]
    turn = retention["inherited_metrics"]["turn"]
    angle_results = turn.get("angle_results", {})
    checks = {
        "supported_command_rate": metric["supported_command_rate"] >= 1.0,
        "crouch_success": metric["success_rate"] >= 0.90,
        "settle_success": metric["settle_success_rate"] >= 0.95,
        "down_reached": metric["down_reached_rate"] >= 0.90,
        "depth_error_mean": metric["depth_error_m"] <= 0.02,
        "depth_error_p95": metric["depth_error_p95_m"] <= 0.04,
        "hold_success": metric["hold_success_rate"] >= 0.90,
        "return_success": metric["return_success_rate"] >= 0.90,
        "return_height_error": metric["return_height_error_m"] <= 0.05,
        "stand_hold_success": metric["stand_hold_success_rate"] >= 0.90,
        "fall": metric["fall_rate"] <= 0.05,
        "dangerous_contact": metric["dangerous_contact_failure_rate"] <= 0.05,
        "saturation": metric["saturation_failure_rate"] <= 0.05,
        "run_retention": run["success_rate"] >= 0.95,
        "turn_retention": bool(angle_results) and all(
            value["success_rate"] >= 0.90 for value in angle_results.values()
        ),
        "frozen_action_invariance": bool(retention.get("verified", False)),
        "stop_action_invariance": bool(retention.get("stop_action_immutability_verified", False)),
    }
    result = {
        "skill": "CROUCH_SHALLOW",
        "supported_depth_range_m": [0.08, 0.10],
        "eligible_for_pass": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "metrics": metric,
        "retention_provenance": str(args.retention.resolve()),
        "summary": str(args.summary.resolve()),
        "stop_success_is_required": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["eligible_for_pass"] else 2)


if __name__ == "__main__":
    main()
