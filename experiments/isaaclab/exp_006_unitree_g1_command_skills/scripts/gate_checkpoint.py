"""Combine all learned-skill evaluations and apply retention/pilot gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LEARNED = {
    "run": ("run",),
    "turn": ("run", "turn"),
    "stop": ("run", "turn", "stop"),
    # RUN/TURN are inherited from a verified frozen-route baseline for CROUCH.
    "crouch": ("crouch",),
    "sequence": ("run", "turn", "stop", "sequence"),
}
PREVIOUS = {"run": (), "turn": ("run",), "stop": ("run", "turn"), "crouch": ("run", "turn"), "sequence": ("run", "turn", "stop")}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(LEARNED), required=True)
    parser.add_argument("--turn-curriculum", choices=("45", "full"), default="full")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--retention-provenance", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    early_provenance = load(args.retention_provenance.resolve()) if args.retention_provenance else None
    actual_retention_fallback = bool(
        args.stage == "crouch" and early_provenance and not early_provenance.get("verified", False)
    )
    skills_to_load = ("run", "turn", "crouch") if actual_retention_fallback else LEARNED[args.stage]
    summaries = {skill: load(root / skill / "normal" / "summary.json") for skill in skills_to_load}
    diagnostic = load(args.diagnostic.resolve())
    failures: list[str] = []

    metrics = {}
    for skill, summary in summaries.items():
        named = summary["skills"]
        primary = "RUN" if skill == "run" else "TURN" if skill == "turn" else "STOP" if skill == "stop" else "CROUCH" if skill == "crouch" else None
        primary_metrics = named.get(primary, {}) if primary else {}
        metrics[skill] = {
            "success_rate": primary_metrics.get("success_rate", summary.get("sequence_completion_rate", 0.0)),
            "fall_rate": (
                summary.get("stop_window_fall_rate", primary_metrics.get("fall_rate", 0.0))
                if skill == "stop" else summary["fall_rate"]
            ),
            "sequence_completion_rate": summary.get("sequence_completion_rate", 0.0),
            "periodic_running_rate": named.get("RUN", {}).get("periodic_running_rate", 0.0),
            "speed_error_mps": primary_metrics.get("speed_error_mps", 0.0),
            "heading_error_rad": primary_metrics.get("heading_error_rad", 0.0),
            "course_deviation_failure_rate": summary.get("course_deviation_failure_rate", 0.0),
            "path_lateral_error_p95": named.get("RUN", {}).get("path_lateral_error_p95", 0.0),
            "path_lateral_error_max": named.get("RUN", {}).get("path_lateral_error_max", 0.0),
        }
        if skill == "turn":
            metrics[skill].update({
                "final_turn_angle_error_rad": primary_metrics.get("final_turn_angle_error_rad", float("inf")),
                "maximum_turn_angle_error_rad": primary_metrics.get("maximum_turn_angle_error_rad", float("inf")),
                "turn_completion_time_s": primary_metrics.get("turn_completion_time_s", 0.0),
                "straight_recovery_rate": summary.get("turn_straight_recovery_rate", 0.0),
                "post_turn_heading_error_rad": primary_metrics.get("post_turn_heading_error_rad", float("inf")),
                "post_turn_path_lateral_error_m": primary_metrics.get(
                    "post_turn_path_lateral_error_m", float("inf")
                ),
                "angle_results": summary.get("turn_angle_results", {}),
            })
        if skill == "run":
            run = named.get("RUN", {})
            if run.get("success_rate", 0.0) < 0.90:
                failures.append("RUN success rate < 90%")
            if args.stage in ("turn", "stop", "crouch", "sequence") and run.get("success_rate", 0.0) < 0.95:
                failures.append(f"RUN retention success rate < 95% during {args.stage.upper()}")
            if summary.get("course_deviation_failure_rate", 1.0) > 0.05:
                failures.append("RUN course deviation failure rate > 5%")
            if run.get("periodic_running_rate", 0.0) < 0.90:
                failures.append("RUN periodic running rate < 90%")
            if run.get("speed_error_mps", float("inf")) > 0.25:
                failures.append("RUN speed error > 0.25 m/s")
            if run.get("heading_error_rad", float("inf")) > 0.12:
                failures.append("RUN path heading error > 0.12 rad")
            if summary["fall_rate"] > 0.05:
                failures.append("RUN fall rate > 5%")
        elif skill == "turn":
            turn = metrics[skill]
            required_buckets = ("left_45", "right_45")
            if args.turn_curriculum == "full":
                required_buckets += ("left_90", "right_90")
            for bucket in required_buckets:
                result = turn["angle_results"].get(bucket, {})
                if result.get("count", 0) == 0:
                    failures.append(f"TURN {bucket} has no evaluation episodes")
                if result.get("success_rate", 0.0) < 0.90:
                    failures.append(f"TURN {bucket} success rate < 90%")
                if result.get("final_turn_angle_error_rad", float("inf")) > 0.12:
                    failures.append(f"TURN {bucket} final angle error > 0.12 rad")
                if result.get("straight_recovery_success_rate", 0.0) < 0.90:
                    failures.append(f"TURN {bucket} straight recovery rate < 90%")
            if turn["final_turn_angle_error_rad"] > 0.12:
                failures.append("TURN final accumulated-angle error > 0.12 rad")
            if summary["fall_rate"] > 0.05:
                failures.append("TURN fall rate > 5%")
        elif skill == "stop":
            stop = named.get("STOP", {})
            # Stage-A acceptance is based only on actual entry speeds within
            # its declared range. Moderate/high tails remain mandatory
            # robustness reports but cannot silently redefine the curriculum.
            if summary.get("stop_curriculum") == "A":
                stop = summary.get("stop_entry_speed_strata", {}).get("in_range_le_1.4", stop)
                metrics[skill]["primary_entry_speed_stratum"] = "in_range_le_1.4"
                metrics[skill]["robustness_entry_speed_strata"] = summary.get("stop_entry_speed_strata", {})
                metrics[skill]["success_rate"] = stop.get("success_rate", 0.0)
                metrics[skill]["fall_rate"] = stop.get("fall_rate", 0.0)
                metrics[skill]["heading_error_rad"] = stop.get("heading_error_rad", float("inf"))
            if stop.get("success_rate", 0.0) < 0.90:
                failures.append("STOP region arrival rate < 90%")
            if stop.get("stop_position_error_m", float("inf")) > 0.50:
                failures.append("STOP position error > 0.5 m")
            if stop.get("stop_speed_mps", float("inf")) > 0.20:
                failures.append("STOP final speed > 0.2 m/s")
            if metrics[skill]["fall_rate"] > 0.05:
                failures.append("STOP fall rate > 5%")
            if stop.get("stop_hold_success_rate", 0.0) < 0.90:
                failures.append("STOP post-stop hold success rate < 90%")
            if stop.get("heading_error_rad", float("inf")) > 0.12:
                failures.append("STOP heading error > 0.12 rad")
            saturation_rate = float(stop.get("saturation_failure_rate", 0.0))
            metrics[skill]["saturation_failure_rate"] = saturation_rate
            metrics[skill]["stop_hold_success_rate"] = stop.get("stop_hold_success_rate", 0.0)
            metrics[skill]["stop_position_error_m"] = stop.get("stop_position_error_m", float("inf"))
            metrics[skill]["stop_hold_end_speed_mps"] = stop.get("stop_hold_end_speed_mps", float("inf"))
            metrics[skill]["parent_action_deviation_norm"] = stop.get(
                "parent_action_deviation_norm", float("inf")
            )
            if saturation_rate > 0.05:
                failures.append("STOP saturation failure rate > 5%")
        elif skill == "crouch":
            crouch = named.get("CROUCH", {})
            metrics[skill].update({
                "depth_error_m": crouch.get("depth_error_m", float("inf")),
                "settle_success_rate": crouch.get("settle_success_rate", 0.0),
                "hold_success_rate": crouch.get("hold_success_rate", 0.0),
                "return_success_rate": crouch.get("return_success_rate", 0.0),
                "return_height_error_m": crouch.get("return_height_error_m", float("inf")),
                "stand_hold_success_rate": crouch.get("stand_hold_success_rate", 0.0),
                "saturation_failure_rate": crouch.get("saturation_failure_rate", 1.0),
                "dangerous_contact_failure_rate": crouch.get("foot_contact_loss_rate", 1.0),
                "foot_contact_loss_rate": crouch.get("foot_contact_loss_rate", 1.0),
                "residual_action_norm": crouch.get("residual_action_norm", float("inf")),
            })
            if crouch.get("success_rate", 0.0) < 0.90:
                failures.append("CROUCH success rate < 90%")
            if crouch.get("settle_success_rate", 0.0) < 0.95:
                failures.append("CROUCH settle success rate < 95%")
            if crouch.get("depth_error_m", float("inf")) > 0.04:
                failures.append("CROUCH depth error > 0.04 m")
            if crouch.get("hold_success_rate", 0.0) < 0.90:
                failures.append("CROUCH hold success rate < 90%")
            if crouch.get("return_success_rate", 0.0) < 0.90:
                failures.append("CROUCH return success rate < 90%")
            if crouch.get("return_height_error_m", float("inf")) > 0.05:
                failures.append("CROUCH return height error > 0.05 m")
            if crouch.get("stand_hold_success_rate", 0.0) < 0.90:
                failures.append("CROUCH stand hold success rate < 90%")
            if crouch.get("fall_rate", summary.get("fall_rate", 1.0)) > 0.05:
                failures.append("CROUCH fall rate > 5%")
            if crouch.get("saturation_failure_rate", 1.0) > 0.05:
                failures.append("CROUCH saturation failure rate > 5%")
            if crouch.get("foot_contact_loss_rate", 1.0) > 0.05:
                failures.append("CROUCH dangerous contact failure rate > 5%")
        else:
            if summary["sequence_completion_rate"] < 0.80:
                failures.append("sequence completion rate < 80%")
            if summary["fall_rate"] > 0.05:
                failures.append("sequence fall rate > 5%")
            for name in ("RUN", "TURN", "STOP"):
                if named.get(name, {}).get("success_rate", 0.0) < 0.90:
                    failures.append(f"sequence {name} success rate < 90%")

    retention = {}
    retention_provenance = None
    if args.stage == "crouch":
        if not args.baseline:
            failures.append("CROUCH inherited retention requires --baseline")
        if not args.retention_provenance:
            failures.append("CROUCH inherited retention requires --retention-provenance")
        else:
            retention_provenance = early_provenance
            required_truths = (
                "tensor_hash_verified", "action_equivalence_verified",
                "stop_action_immutability_verified", "running_base_endpoint_verified",
            )
            if actual_retention_fallback:
                retention_provenance["retention_source"] = "checkpoint_specific_normal_evaluation"
                retention_provenance["retention_basis"] = "frozen_route_proof_failed_fallback"
            else:
                if retention_provenance.get("retention_source") != "inherited_from_baseline_gate":
                    failures.append("CROUCH retention provenance source is invalid")
                if retention_provenance.get("retention_basis") != "bitwise_frozen_actor_route":
                    failures.append("CROUCH retention provenance basis is invalid")
                saved_baseline = Path(retention_provenance.get("baseline_gate", ""))
                if not saved_baseline.is_absolute():
                    saved_baseline = (args.retention_provenance.resolve().parent / saved_baseline).resolve()
                else:
                    saved_baseline = saved_baseline.resolve()
                if not args.baseline or saved_baseline != args.baseline.resolve():
                    failures.append("CROUCH retention provenance baseline does not match --baseline")
                if not all(retention_provenance.get(key, False) for key in required_truths):
                    failures.append("CROUCH frozen-route retention proof failed")
                if retention_provenance.get("standing_or_crouch_route_leakage_detected", True):
                    failures.append("CROUCH route leaked into RUN/TURN/STOP action")
                inherited = retention_provenance.get("inherited_metrics", {})
                if "run" not in inherited or "turn" not in inherited:
                    failures.append("CROUCH retention provenance is missing RUN/TURN metrics")
                else:
                    metrics["run"] = inherited["run"]
                    metrics["turn"] = inherited["turn"]
                    if float(metrics["run"].get("success_rate", 0.0)) < 0.95:
                        failures.append("RUN inherited retention success rate < 95% during CROUCH")
                    required_buckets = ("left_45", "right_45", "left_90", "right_90")
                    for bucket in required_buckets:
                        result = metrics["turn"].get("angle_results", {}).get(bucket, {})
                        if float(result.get("success_rate", 0.0)) < 0.90:
                            failures.append(f"TURN inherited {bucket} success rate < 90%")
    if args.baseline:
        baseline = load(args.baseline.resolve())
        if args.stage == "run" and "run" in baseline.get("metrics", {}):
            old_run = baseline["metrics"]["run"]
            new_run = metrics["run"]
            preservation = {
                "periodic_running_drop_points": 100.0 * (
                    float(old_run.get("periodic_running_rate", 0.0)) - new_run["periodic_running_rate"]
                ),
                "fall_rate_increase_points": 100.0 * (
                    new_run["fall_rate"] - float(old_run.get("fall_rate", 0.0))
                ),
                "speed_error_increase_mps": new_run["speed_error_mps"] - float(old_run.get("speed_error_mps", 0.0)),
            }
            retention["stage4_locomotion"] = preservation
            if preservation["periodic_running_drop_points"] >= 5.0 - 1.0e-9:
                failures.append("Stage-4 periodic running drop >= 5 points")
            if preservation["fall_rate_increase_points"] >= 5.0 - 1.0e-9:
                failures.append("Stage-4 fall rate increase >= 5 points")
            if preservation["speed_error_increase_mps"] > 0.10:
                failures.append("Stage-4 speed error increase > 0.10 m/s")
        for skill in PREVIOUS[args.stage]:
            old = float(baseline["metrics"][skill]["success_rate"])
            new = float(metrics[skill]["success_rate"])
            drop = old - new
            retention[skill] = {"baseline": old, "candidate": new, "drop_points": 100.0 * drop}
            if drop >= 0.05 - 1.0e-9:
                failures.append(f"{skill.upper()} retention drop >= 5 points")
    if not diagnostic["acceptance"]["command_sensitive"]:
        failures.append("counterfactual command-action sensitivity gate failed")

    ablations = {}
    for skill in LEARNED[args.stage]:
        normal = metrics[skill]["success_rate"]
        ablations[skill] = {"normal": normal}
        for ablation in ("new_command_zero", "legacy_command_zero", "all_command_zero", "shuffle"):
            path = root / skill / ablation / "summary.json"
            if path.exists():
                summary = load(path)
                primary = "RUN" if skill == "run" else "TURN" if skill == "turn" else "STOP" if skill == "stop" else "CROUCH" if skill == "crouch" else None
                value = summary["skills"].get(primary, {}).get("success_rate", summary.get("sequence_completion_rate", 0.0))
                label = ablation
                if ablation == "shuffle" and summary.get("command_ablation_schema_version", 1) < 2:
                    label = "legacy_saved_incoherent_shuffle"
                ablations[skill][label] = value
                ablations[skill][f"normal_minus_{label}"] = normal - value
        # Pre-fix sweeps named the incomplete, new-29-only ablation "zero".
        old_zero = root / skill / "zero" / "summary.json"
        if old_zero.exists() and "new_command_zero" not in ablations[skill]:
            summary = load(old_zero)
            primary = "RUN" if skill == "run" else "TURN" if skill == "turn" else "STOP" if skill == "stop" else "CROUCH" if skill == "crouch" else None
            value = summary["skills"].get(primary, {}).get("success_rate", summary.get("sequence_completion_rate", 0.0))
            ablations[skill]["legacy_saved_new_command_zero"] = value
            ablations[skill]["normal_minus_legacy_saved_new_command_zero"] = normal - value

    report = {
        "stage": args.stage,
        "turn_curriculum": args.turn_curriculum if args.stage == "turn" else None,
        "eligible_for_best": not failures,
        "failures": failures,
        "metrics": metrics,
        "retention": retention,
        "retention_provenance": (
            str(args.retention_provenance.resolve()) if args.retention_provenance else None
        ),
        "command_diagnostic": str(args.diagnostic.resolve()),
        "command_sensitivity_acceptance": diagnostic["acceptance"],
        "command_ablations": ablations,
        "retention_rule": "reject when any previously learned skill drops by >= 5 percentage points",
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
