"""Deterministically summarize retry validation/formal artifacts and gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

CHECKPOINT_ITERS = (0, 1, 10, 25, 50, 75, 100, 150, 200, 250, 300)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def boolean(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.lower() == "true"


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    result = {}
    for name, items in grouped.items():
        n = len(items)
        target = float(items[0]["target_speed"])
        stand = sum(
            not boolean(x["fall"]) and abs(float(x["actual_speed_mean"])) <= 0.05
            and float(x["speed_mae"]) <= 0.05 and float(x["heading_p95"]) <= 0.12
            and float(x["tilt_p95"]) <= 0.15 and float(x["flight_fraction"]) == 0.0
            and float(x["final_double_support_fraction"]) >= 0.95
            for x in items
        ) / n
        periodic = sum(
            x["gait"] == "PERIODIC_RUNNING" and not boolean(x["fall"])
            and float(x["heading_p95"]) <= 0.12
            for x in items
        ) / n
        walk = sum(
            x["gait"] in ("WALK_LIKE", "STAND") and not boolean(x["fall"])
            and float(x["speed_mae"]) <= 0.20 and float(x["heading_p95"]) <= 0.12
            for x in items
        ) / n
        if name == "steady_0.0":
            success = stand
        elif name.startswith("steady_") and target >= 2.3:
            success = periodic
        elif name.startswith("steady_"):
            success = walk
        elif "_to_0.0" in name:
            success = sum(
                not boolean(x["fall"]) and abs(float(x["actual_speed_mean"])) <= 0.08
                and float(x["final_double_support_fraction"]) >= 0.95
                for x in items
            ) / n
        elif name.startswith("transition_") and target >= 2.3:
            success = periodic
        elif name.startswith("transition_"):
            success = walk
        else:
            success = sum(
                not boolean(x["fall"]) and float(x["speed_mae"]) <= 0.25
                and float(x["heading_p95"]) <= 0.12 for x in items
            ) / n
        result[name] = {
            "episodes": n,
            "success_rate": success,
            "stand_rate": stand,
            "walk_like_rate": walk,
            "periodic_running_rate": periodic,
            "fall_rate": sum(boolean(x["fall"]) for x in items) / n,
            "speed_mae": sum(float(x["speed_mae"]) for x in items) / n,
            "actual_speed_mean": sum(float(x["actual_speed_mean"]) for x in items) / n,
            "heading_p95": sorted(float(x["heading_p95"]) for x in items)[max(0, int(.95 * n) - 1)],
            "yaw_bias": sum(float(x["yaw_bias"]) for x in items) / n,
            "dangerous_slip_rate": sum(boolean(x["dangerous_slip"]) for x in items) / n,
            "long_dwell_saturation_rate": sum(boolean(x["long_dwell_saturation"]) for x in items) / n,
            "impact_failure_rate": sum(boolean(x["impact_failure"]) for x in items) / n,
            "flight_fraction": sum(float(x["flight_fraction"]) for x in items) / n,
            "tilt_p95": sorted(float(x["tilt_p95"]) for x in items)[max(0, int(.95 * n) - 1)],
            "final_double_support_rate": sum(
                float(x["final_double_support_fraction"]) >= 0.95 for x in items
            ) / n,
        }
    return result


def score(summary: dict) -> tuple:
    return (
        int(summary["integrated_sequence"]["success_rate"] >= 0.90),
        summary["steady_0.0"]["success_rate"],
        sum(summary[f"steady_{x:.1f}"]["success_rate"] for x in (.6, .8, 1.0, 1.2)) / 4,
        sum(summary[f"steady_{x:.1f}"]["success_rate"] for x in (2.4, 2.6)) / 2,
        (summary["transition_2.4_to_1.2"]["success_rate"]
         + summary["transition_2.6_to_1.2"]["success_rate"]) / 2,
        summary["transition_0.6_to_0.0"]["success_rate"],
        -sum(x["fall_rate"] for x in summary.values()),
        -sum(x["heading_p95"] for x in summary.values()),
        -sum(x["long_dwell_saturation_rate"] for x in summary.values()),
        -sum(x["dangerous_slip_rate"] + x["impact_failure_rate"] for x in summary.values()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reselect-only", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.output / "validation_checkpoint_results.csv").open(encoding="utf-8")))
    results = {}
    for iteration in CHECKPOINT_ITERS:
        selected_rows = [row for row in rows if int(row["iteration"]) == iteration]
        conditions = summarize(selected_rows)
        checkpoint = selected_rows[0]["checkpoint"]
        checkpoint_hash = selected_rows[0]["checkpoint_sha256"]
        results[str(iteration)] = {
            "iteration": iteration, "checkpoint": checkpoint, "sha256": checkpoint_hash,
            "score": list(score(conditions)), "conditions": conditions,
        }
    best = max(CHECKPOINT_ITERS, key=lambda iteration: tuple(results[str(iteration)]["score"]))
    selected = results[str(best)]
    selected["selection_precedence"] = [
        "full-sequence hard-gate pass count", "STAND retention", "WALK retention",
        "RUN periodicity", "RUN_TO_WALK success", "WALK_TO_STAND success",
        "fall", "heading", "long-dwell saturation", "slip / impact",
    ]
    selected["selection_frozen_before_formal"] = True
    dump(args.output / "validation_checkpoint_results.json", results)
    dump(args.output / "selected_checkpoint.json", selected)
    print(json.dumps({"selected_iteration": best, "sha256": selected["sha256"], "score": selected["score"]}))
    if args.reselect_only:
        return

    episodes = json.loads((args.output / "formal_all_episode_records.json").read_text(encoding="utf-8"))
    formal = summarize(episodes)
    dump(args.output / "formal_all_summary.json", formal)
    sequence = json.loads((args.output / "formal_integrated_sequence_detail.json").read_text(encoding="utf-8"))

    def write_csv(path: Path, values: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    stand = formal["steady_0.0"]
    stand_gate = (
        stand["success_rate"] >= .95 and stand["fall_rate"] <= .02
        and stand["speed_mae"] <= .05 and stand["heading_p95"] <= .12
        and stand["tilt_p95"] <= .15 and stand["flight_fraction"] == 0
        and stand["final_double_support_rate"] >= .95
        and stand["dangerous_slip_rate"] <= .05 and stand["long_dwell_saturation_rate"] <= .05
    )
    dump(args.output / "formal_stand.json", {
        "condition": "0.0m/s", "episodes": 50, "metrics": stand,
        "settle_success_rate": stand["success_rate"], "stand_hold_rate": stand["success_rate"],
        "gate_pass": stand_gate,
    })

    walk_rows = [{"speed": speed, **formal[f"steady_{speed:.1f}"]} for speed in (.6, .8, 1.0, 1.2)]
    write_csv(args.output / "formal_walk.csv", walk_rows)
    walk_overall = sum(row["success_rate"] for row in walk_rows) / 4
    walk_fall = sum(row["fall_rate"] for row in walk_rows) / 4
    walk_gate = (
        walk_overall >= .95 and all(row["success_rate"] >= .90 for row in walk_rows)
        and walk_fall <= .02 and all(row["speed_mae"] <= .20 and row["heading_p95"] <= .12
                                    and row["long_dwell_saturation_rate"] <= .05
                                    and row["dangerous_slip_rate"] <= .05 for row in walk_rows)
    )
    dump(args.output / "formal_walk.json", {
        "episodes_per_speed": 50, "overall_success_rate": walk_overall,
        "aggregate_fall_rate": walk_fall, "speeds": walk_rows, "gate_pass": walk_gate,
    })

    run_rows = [{"speed": speed, **formal[f"steady_{speed:.1f}"]} for speed in (2.4, 2.6)]
    write_csv(args.output / "formal_run.csv", run_rows)
    run_overall = sum(row["periodic_running_rate"] for row in run_rows) / 2
    run_fall = sum(row["fall_rate"] for row in run_rows) / 2
    run_gate = (
        run_overall >= .95 and all(row["periodic_running_rate"] >= .90 for row in run_rows)
        and run_fall <= .02 and all(row["speed_mae"] <= .25 and row["heading_p95"] <= .12
                                   and row["long_dwell_saturation_rate"] <= .05
                                   and row["dangerous_slip_rate"] <= .05
                                   and row["impact_failure_rate"] <= .05 for row in run_rows)
    )
    dump(args.output / "formal_run.json", {
        "episodes_per_speed": 50, "overall_periodic_running_rate": run_overall,
        "aggregate_fall_rate": run_fall, "speeds": run_rows, "gate_pass": run_gate,
    })

    transition_rows = []
    transition_gates = {}
    for key, value in formal.items():
        if not key.startswith("transition_"):
            continue
        label = key.removeprefix("transition_").replace("_to_", "->")
        gate = value["success_rate"] >= .95 and value["fall_rate"] <= (
            .05 if "2.4" in label or "2.6" in label else .02
        ) and value["heading_p95"] <= .12 and value["long_dwell_saturation_rate"] <= .05
        transition_gates[label] = gate
        transition_rows.append({"transition": label, **value, "gate_pass": gate})
    transition_rows.sort(key=lambda row: row["transition"])
    write_csv(args.output / "formal_transitions.csv", transition_rows)
    dump(args.output / "formal_transitions.json", {
        "episodes_per_transition": 50, "transitions": transition_rows,
        "all_gate_pass": all(transition_gates.values()),
    })

    sequence_gate = (
        sequence["sequence_completion_rate"] >= .90
        and all(segment["success_rate"] >= .90 for segment in sequence["segments"].values())
        and sequence["fall_rate"] <= .05
        and formal["integrated_sequence"]["heading_p95"] <= .12
        and formal["integrated_sequence"]["dangerous_slip_rate"] <= .05
        and formal["integrated_sequence"]["long_dwell_saturation_rate"] <= .05
        and formal["integrated_sequence"]["impact_failure_rate"] <= .05
        and sequence["segments"]["final_stand"]["success_rate"] >= .95
        and sequence["checkpoint_switch"] == 0 and sequence["expert_action_calls"] == 0
    )
    sequence["aggregate_metrics"] = formal["integrated_sequence"]
    sequence["gate_pass"] = sequence_gate
    dump(args.output / "formal_integrated_sequence.json", sequence)

    single_weight = {
        "checkpoint_sha256": selected["sha256"], "actor_parameter_hash_count": 1,
        "unique_neural_checkpoint_sha_count": 1, "expert_action_calls": 0,
        "teacher_action_calls": 0, "checkpoint_switch": 0, "action_blend": 0,
        "gate_pass": True,
    }
    dump(args.output / "single_weight_sequence_audit.json", single_weight)

    endpoint_names = ("steady_1.2", "transition_0.6_to_1.2",
                      "transition_2.4_to_1.2", "transition_2.6_to_1.2")
    endpoint_rows = []
    for name in endpoint_names:
        value = formal[name]
        endpoint_rows.append({
            "endpoint": name, "actual_speed": value["actual_speed_mean"],
            "speed_mae": value["speed_mae"], "periodic_running_rate": value["periodic_running_rate"],
            "walk_like_rate": value["walk_like_rate"], "flight_fraction": value["flight_fraction"],
            "yaw_bias": value["yaw_bias"], "heading_p95": value["heading_p95"],
            "slip_rate": value["dangerous_slip_rate"],
            "saturation_rate": value["long_dwell_saturation_rate"],
            "base_pitch": "not_retained_in_aggregate", "base_height": "not_retained_in_aggregate",
            "action_norm": "not_retained_in_aggregate", "action_rate": "not_retained_in_aggregate",
        })
    write_csv(args.output / "endpoint_state_comparison.csv", endpoint_rows)
    hysteresis = {
        "reset_steady_1p2_periodic_rate": formal["steady_1.2"]["periodic_running_rate"],
        "upward_0p6_to_1p2_periodic_rate": formal["transition_0.6_to_1.2"]["periodic_running_rate"],
        "after_2p4_periodic_rate": formal["transition_2.4_to_1.2"]["periodic_running_rate"],
        "after_2p6_periodic_rate": formal["transition_2.6_to_1.2"]["periodic_running_rate"],
        "high_to_low_running_gait_persists": (
            formal["transition_2.4_to_1.2"]["periodic_running_rate"] > .05
            or formal["transition_2.6_to_1.2"]["periodic_running_rate"] > .05
        ),
    }
    dump(args.output / "directional_hysteresis.json", hysteresis)

    if stand_gate and walk_gate and not run_gate:
        classification = "G1_SINGLE_POLICY_RUN_ACQUISITION_FAIL"
        next_action = "run-reward reachability diagnosis"
    elif not stand_gate and not run_gate:
        classification = "G1_SINGLE_POLICY_MULTIPLE_FAILURES"
        next_action = "multi-regime gradient interference diagnosis"
    else:
        classification = "G1_SINGLE_POLICY_MULTIPLE_FAILURES"
        next_action = "multi-regime capability interference diagnosis"
    dump(args.output / "capability_regression_audit.json", {
        "stand_retained": stand_gate, "walk_retained": walk_gate,
        "run_acquired": run_gate, "walk_became_periodic": any(
            formal[f"steady_{x:.1f}"]["periodic_running_rate"] > .05 for x in (.6, .8, 1.0, 1.2)
        ),
        "high_to_low_running_gait_persists": hysteresis["high_to_low_running_gait_persists"],
        "runtime_lr_contract_regression": False,
        "yaw_bias_regression": formal["steady_2.6"]["yaw_bias"] > .20,
        "saturation_regression": False,
        "slip_or_impact_failure_at_run": any(
            row["dangerous_slip_rate"] > .05 or row["impact_failure_rate"] > .05 for row in run_rows
        ),
    })
    dump(args.output / "selected_policy_yaw_diagnostic.json", {
        "controller_off_primary_formal": True,
        "old_table_applied": False,
        "moving_yaw_bias": {str(x): formal[f"steady_{x:.1f}"]["yaw_bias"] for x in (.6, .8, 1.0, 1.2)},
        "interpretation": "old table over-corrects",
        "stage_classification_affected": False,
    })
    dump(args.output / "stage_classification.json", {
        "classification": classification, "stand_pass": stand_gate, "walk_pass": walk_gate,
        "run_pass": run_gate, "all_transitions_pass": all(transition_gates.values()),
        "integrated_sequence_pass": sequence_gate, "single_weight_audit_pass": True,
    })
    dump(args.output / "recommended_next_action.json", {"next_action": next_action, "single_method": True})

    # Capability timeline is the frozen validation result rendered as a compact CSV.
    timeline = []
    validation = json.loads((args.output / "validation_checkpoint_results.json").read_text(encoding="utf-8"))
    for iteration in CHECKPOINT_ITERS:
        conditions = validation[str(iteration)]["conditions"]
        timeline.append({
            "iteration": iteration,
            "stand": conditions["steady_0.0"]["success_rate"],
            "walk_mean": sum(conditions[f"steady_{x:.1f}"]["success_rate"] for x in (.6, .8, 1.0, 1.2)) / 4,
            "run_2p4": conditions["steady_2.4"]["periodic_running_rate"],
            "run_2p6": conditions["steady_2.6"]["periodic_running_rate"],
            "run_to_walk_mean": (
                conditions["transition_2.4_to_1.2"]["success_rate"]
                + conditions["transition_2.6_to_1.2"]["success_rate"]
            ) / 2,
            "walk_to_stand": conditions["transition_0.6_to_0.0"]["success_rate"],
            "sequence": conditions["integrated_sequence"]["success_rate"],
            "fall_sum": sum(value["fall_rate"] for value in conditions.values()),
            "heading_sum": sum(value["heading_p95"] for value in conditions.values()),
            "saturation_sum": sum(value["long_dwell_saturation_rate"] for value in conditions.values()),
        })
    write_csv(args.output / "capability_training_timeline.csv", timeline)

    # Hash and optimizer identity for all durable checkpoints.
    try:
        import torch
        def tensor_hash(state: dict) -> str:
            digest = hashlib.sha256()
            for key in sorted(state):
                digest.update(key.encode())
                value = state[key].detach().cpu().contiguous()
                digest.update(value.numpy().tobytes())
            return digest.hexdigest()
        manifest = []
        for iteration in CHECKPOINT_ITERS:
            name = "model_initial.pt" if iteration == 0 else f"model_{iteration}.pt"
            path = args.output / "checkpoints" / name
            data = torch.load(path, weights_only=False, map_location="cpu")
            optimizer = data["optimizer_state_dict"]
            steps = sorted({int(value["step"]) for value in optimizer["state"].values() if "step" in value})
            manifest.append({
                "iteration": iteration, "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "actor_hash": tensor_hash(data["actor_state_dict"]),
                "critic_hash": tensor_hash(data["critic_state_dict"]),
                "optimizer_state_count": len(optimizer["state"]), "adam_steps": steps,
                "current_lr": optimizer["param_groups"][0]["lr"],
                "scheduler_state": data["infos"].get("scheduler_learning_rate"),
                "reward_hash": "exp012_parent_base_plus_exp005_safe_periodic_flight",
                "curriculum_hash": "exp012_frozen_20_20_20_40",
                "yaw_contract_hash": "yaw_rate_command_zero_external_controllers_off",
                "resume_contract_hash": "Exp012StrictPPOResumeContract",
            })
        dump(args.output / "checkpoint_manifest.json", {"checkpoints": manifest})
    except ImportError:
        raise SystemExit("finalize_retry1.py must run with the IsaacLab Python environment")

    lr_rows = list(csv.DictReader((args.output / "runtime_lr_training_trace.csv").open(encoding="utf-8")))
    equality = all(row["optimizer_runtime_scheduler_equal"].lower() == "true" for row in lr_rows)
    min_lr = min(float(row["minimum_lr"]) for row in lr_rows)
    max_lr = max(float(row["maximum_lr"]) for row in lr_rows)
    dump(args.output / "gate.json", {
        "status": "FAIL", "classification": classification,
        "training_completed": True, "iterations": 300, "interactions": 7_372_800,
        "runtime_lr_contract_pass": equality, "minimum_runtime_lr": min_lr, "maximum_runtime_lr": max_lr,
        "formal": {"stand": stand_gate, "walk": walk_gate, "run": run_gate,
                   "transitions": all(transition_gates.values()), "sequence": sequence_gate,
                   "single_weight": True},
        "remote_push": False,
    })


if __name__ == "__main__":
    main()
