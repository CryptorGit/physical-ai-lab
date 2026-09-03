"""Select the strongest eligible checkpoint after all-skill retention gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--evaluations", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corrective-parent", default="")
    parser.add_argument("--saturation-worsening-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    gates = {
        path.parent.name: (path, json.loads(path.read_text(encoding="utf-8")))
        for path in args.evaluations.resolve().glob("model_*/gate.json")
    }
    parent_name = args.corrective_parent if args.stage == "stop" else ""
    parent_entry = gates.get(parent_name) if parent_name else None
    parent_metrics = parent_entry[1]["metrics"]["stop"] if parent_entry else None
    candidates = []
    crouch_ranked = []
    comparison_rows = []
    rejected = {}
    for model_name, (gate_path, gate) in gates.items():
        if parent_name and model_name == parent_name:
            continue
        reasons = []
        if not gate["eligible_for_best"]:
            reasons.append("absolute_gate_failed")
        if parent_metrics is not None:
            metrics = gate["metrics"]["stop"]
            if float(metrics["heading_error_rad"]) >= float(parent_metrics["heading_error_rad"]):
                reasons.append("heading_not_better_than_parent")
            if float(metrics["fall_rate"]) > float(parent_metrics["fall_rate"]):
                reasons.append("fall_worse_than_parent")
            if float(metrics["stop_hold_success_rate"]) < float(parent_metrics["stop_hold_success_rate"]):
                reasons.append("hold_worse_than_parent")
            if float(metrics["stop_position_error_m"]) > 0.50:
                reasons.append("position_outside_gate")
            if float(metrics["stop_hold_end_speed_mps"]) > 0.20:
                reasons.append("hold_end_speed_outside_gate")
            if float(metrics["saturation_failure_rate"]) > (
                float(parent_metrics["saturation_failure_rate"]) + args.saturation_worsening_tolerance
            ):
                reasons.append("saturation_materially_worse_than_parent")
            if reasons:
                rejected[model_name] = reasons
                continue
        elif reasons and args.stage != "crouch":
            rejected[model_name] = reasons
            continue
        elif reasons:
            rejected[model_name] = reasons
        checkpoint = args.run_dir.resolve() / f"{gate_path.parent.name}.pt"
        metrics = gate["metrics"][args.stage]
        if args.stage == "stop":
            score = (
                -float(metrics["fall_rate"]),
                -float(metrics["heading_error_rad"]),
                float(metrics["stop_hold_success_rate"]),
                -float(metrics["stop_position_error_m"]),
                -float(metrics["stop_hold_end_speed_mps"]),
                -float(metrics["saturation_failure_rate"]),
                -float(metrics["parent_action_deviation_norm"]),
            )
        elif args.stage == "crouch":
            score = (
                float(metrics["success_rate"]),
                -float(metrics["fall_rate"]),
                float(metrics["return_success_rate"]),
                float(metrics["stand_hold_success_rate"]),
                -float(metrics["dangerous_contact_failure_rate"]),
                -float(metrics["saturation_failure_rate"]),
                -float(metrics["depth_error_m"]),
                -float(metrics["return_height_error_m"]),
                -float(metrics["residual_action_norm"]),
            )
        else:
            score = (float(metrics["success_rate"]),)
        iteration = int(gate_path.parent.name.removeprefix("model_"))
        if args.stage == "crouch":
            crouch_ranked.append((score, iteration, checkpoint, gate_path, gate))
            turn_angles = gate.get("metrics", {}).get("turn", {}).get("angle_results", {})
            comparison_rows.append({
                "model": model_name, "iteration": iteration,
                "eligible": bool(gate.get("eligible_for_best", False)),
                "crouch_success": metrics["success_rate"],
                "settle_success": metrics["settle_success_rate"],
                "depth_error": metrics["depth_error_m"],
                "hold_success": metrics["hold_success_rate"],
                "return_success": metrics["return_success_rate"],
                "return_height_error": metrics["return_height_error_m"],
                "stand_hold_success": metrics["stand_hold_success_rate"],
                "fall": metrics["fall_rate"],
                "dangerous_contact_failure": metrics["dangerous_contact_failure_rate"],
                "saturation_failure": metrics["saturation_failure_rate"],
                "RUN_retention": gate.get("metrics", {}).get("run", {}).get("success_rate", 0.0),
                "TURN_retention": min(
                    [float(value.get("success_rate", 0.0)) for value in turn_angles.values()] or [0.0]
                ),
                "command_sensitivity": gate.get("command_sensitivity_acceptance", {}).get("command_sensitive", False),
                "failures": " | ".join(gate.get("failures", [])),
            })
        if gate["eligible_for_best"]:
            candidates.append((score, iteration, checkpoint, gate_path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    report = {
        "stage": args.stage,
        "selected": None,
        "eligible_count": len(candidates),
        "corrective_parent": parent_name or None,
        "rejected_against_parent": rejected,
        "selection_rule": (
            "eligible STOP checkpoints ranked by fall, heading, hold, position, final speed, saturation, then parent deviation"
            if args.stage == "stop"
            else "eligible CROUCH checkpoints ranked by success, fall, return, stand-hold, dangerous contact, saturation, depth, return height, then residual norm"
            if args.stage == "crouch"
            else "highest current-stage success among checkpoints passing thresholds, command sensitivity, and <5-point retention drop"
        ),
    }
    if parent_entry:
        parent_path, parent_gate = parent_entry
        report["selected"] = {
            "checkpoint": str(args.run_dir.resolve() / f"{parent_name}.pt"),
            "iteration": int(parent_name.removeprefix("model_")),
            "score": "parent_champion",
            "gate": str(parent_path),
            "replaced_parent": False,
        }
    if candidates:
        score, iteration, checkpoint, gate_path = candidates[0]
        report["selected"] = {
            "checkpoint": str(checkpoint),
            "iteration": iteration,
            "score": list(score),
            "gate": str(gate_path),
            "replaced_parent": bool(parent_entry),
        }
    if args.stage == "crouch":
        crouch_ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        report["best_improved_checkpoint"] = None
        if crouch_ranked:
            score, iteration, checkpoint, gate_path, gate = crouch_ranked[0]
            report["best_improved_checkpoint"] = {
                "checkpoint": str(checkpoint), "iteration": iteration, "score": list(score),
                "gate": str(gate_path), "eligible": bool(gate.get("eligible_for_best", False)),
                "dominant_failures": gate.get("failures", []),
            }
        comparison_rows.sort(key=lambda row: row["iteration"])
        comparison_path = args.output.resolve().with_name("checkpoint_comparison.csv")
        if comparison_rows:
            with comparison_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(comparison_rows[0]))
                writer.writeheader()
                writer.writerows(comparison_rows)
        report["comparison_table"] = str(comparison_path)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not candidates and not parent_entry and args.stage != "crouch":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
