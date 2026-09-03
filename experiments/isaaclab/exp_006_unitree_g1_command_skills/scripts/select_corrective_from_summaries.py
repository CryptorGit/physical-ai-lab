"""Select a STOP corrective champion while retaining a non-dominated parent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def metrics(summary: dict) -> dict[str, float]:
    stop = summary["skills"]["STOP"]
    return {
        "fall_rate": float(summary["stop_window_fall_rate"]),
        "heading_error_rad": float(stop["heading_error_rad"]),
        "hold_success_rate": float(stop["stop_hold_success_rate"]),
        "position_error_m": float(stop["stop_position_error_m"]),
        "hold_end_speed_mps": float(stop["stop_hold_end_speed_mps"]),
        "saturation_failure_rate": float(stop["saturation_failure_rate"]),
        "parent_action_deviation_norm": float(stop.get("parent_action_deviation_norm", 0.0)),
    }


def rejection_reasons(candidate: dict, parent: dict, saturation_tolerance: float) -> list[str]:
    reasons = []
    if candidate["heading_error_rad"] >= parent["heading_error_rad"]:
        reasons.append("heading_not_better_than_parent")
    if candidate["fall_rate"] > parent["fall_rate"]:
        reasons.append("fall_worse_than_parent")
    if candidate["hold_success_rate"] < parent["hold_success_rate"]:
        reasons.append("hold_worse_than_parent")
    if candidate["position_error_m"] > 0.50:
        reasons.append("position_outside_gate")
    if candidate["hold_end_speed_mps"] > 0.20:
        reasons.append("hold_end_speed_outside_gate")
    if candidate["saturation_failure_rate"] > parent["saturation_failure_rate"] + saturation_tolerance:
        reasons.append("saturation_materially_worse_than_parent")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", default="model_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--saturation-worsening-tolerance", type=float, default=0.05)
    args = parser.parse_args()

    summaries = {}
    for path in sorted(args.root.resolve().glob("model_*/summary.json")):
        summaries[path.parent.name] = metrics(json.loads(path.read_text(encoding="utf-8")))
    if args.parent not in summaries:
        raise ValueError(f"Missing parent summary: {args.parent}")
    parent = summaries[args.parent]
    accepted = []
    rejected = {}
    for name, candidate in summaries.items():
        if name == args.parent:
            continue
        reasons = rejection_reasons(candidate, parent, args.saturation_worsening_tolerance)
        if reasons:
            rejected[name] = reasons
        else:
            accepted.append(name)
    accepted.sort(key=lambda name: (
        -summaries[name]["fall_rate"],
        -summaries[name]["heading_error_rad"],
        summaries[name]["hold_success_rate"],
        -summaries[name]["position_error_m"],
        -summaries[name]["hold_end_speed_mps"],
        -summaries[name]["saturation_failure_rate"],
        -summaries[name]["parent_action_deviation_norm"],
    ), reverse=True)
    selected = accepted[0] if accepted else args.parent
    report = {
        "parent": args.parent,
        "selected": selected,
        "parent_replaced": selected != args.parent,
        "metrics": summaries,
        "accepted_parent_dominators": accepted,
        "rejected": rejected,
        "rule": "candidate must improve heading, not worsen fall/hold, remain inside position/speed gates, and not materially worsen saturation",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
