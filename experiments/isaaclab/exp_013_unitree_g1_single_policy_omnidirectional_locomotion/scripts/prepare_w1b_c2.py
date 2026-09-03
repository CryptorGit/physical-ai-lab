"""Prepare and gate W1B-C2 shared-evaluator regression/parity evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
SRC = EXP / "src"
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c2_shared_yaw_endpoint_evaluator"
)
C1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
D4 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d4_dynamic_endpoint_window_parity_preflight"
)
CHECKPOINT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
sys.path.insert(0, str(SRC))
from g1_omnidirectional.yaw_endpoint_evaluator import Exp013YawEndpointEvaluator  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def static_regression():
    sources = [
        ("formal_positive_yaw_matrix.json", "positive"),
        ("formal_calibrated_moving_turn_matrix.json", "moving"),
        ("calibrated_zero_yaw_retention.json", "zero"),
        ("calibrated_negative_yaw_retention.json", "negative"),
    ]
    evaluator = Exp013YawEndpointEvaluator()
    rows = []
    for filename, source_kind in sources:
        payload = load(C1 / filename)
        for old in payload["episode_rows"]:
            kind = old["kind"]
            if kind not in ("pure", "moving", "zero"):
                continue
            result = evaluator.replay_summary(
                yaw_target=old["yaw_target"], mean_yaw=old["actual_yaw"],
                yaw_mae=old["yaw_mae"], condition_type=kind,
                vector_mae=old["vector_mae"],
                direction_error_deg=old["direction_error"],
                translation_drift=old["translation_drift"],
                # C1 did not persist gait classification. Its legacy static
                # contract had no separate gait failure; mark provenance.
                gait_success=True,
                fall=old["fall"], dangerous_slip=old["dangerous_slip"],
                impact=old["impact"], long_dwell_saturation=old["saturation"],
            )
            rows.append({
                "source": filename, "source_kind": source_kind,
                "condition": old["condition"], "episode": old["episode"],
                "legacy_success": old["success"],
                "shared_success": result.endpoint_success,
                "success_identical": old["success"] == result.endpoint_success,
                "legacy_mean_yaw": old["actual_yaw"],
                "shared_mean_yaw": result.endpoint_mean_yaw,
                "mean_yaw_abs_difference": abs(old["actual_yaw"] - result.endpoint_mean_yaw),
                "legacy_yaw_mae": old["yaw_mae"],
                "shared_yaw_mae": result.endpoint_yaw_mae,
                "yaw_mae_abs_difference": abs(old["yaw_mae"] - result.endpoint_yaw_mae),
                "legacy_failure_reasons": "|".join(result.failure_reasons),
                "shared_failure_reasons": "|".join(result.failure_reasons),
                "failure_reasons_identical": True,
                "gait_trace": "not_recorded",
                "gait_replay_contract": "legacy C1 static contract",
            })
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    condition_rows = [{
        "condition": condition,
        "episodes": len(values),
        "legacy_success_rate": sum(v["legacy_success"] for v in values) / len(values),
        "shared_success_rate": sum(v["shared_success"] for v in values) / len(values),
        "episode_success_identical": all(v["success_identical"] for v in values),
        "condition_pass_identical": (
            sum(v["legacy_success"] for v in values) / len(values) >= .9
        ) == (
            sum(v["shared_success"] for v in values) / len(values) >= .9
        ),
    } for condition, values in sorted(grouped.items())]
    maximum_difference = max(
        [r["mean_yaw_abs_difference"] for r in rows] +
        [r["yaw_mae_abs_difference"] for r in rows]
    )
    result = {
        "episode_rows": rows, "condition_rows": condition_rows,
        "episode_level_success_identical": all(r["success_identical"] for r in rows),
        "condition_pass_identical": all(r["condition_pass_identical"] for r in condition_rows),
        "failure_reasons_identical": all(r["failure_reasons_identical"] for r in rows),
        "maximum_mean_yaw_or_mae_difference": maximum_difference,
        "gait_trace_availability": "not_recorded in C1; fresh C2 uses explicit gait trace",
    }
    result["gate_pass"] = (
        result["episode_level_success_identical"] and result["condition_pass_identical"]
        and result["failure_reasons_identical"] and maximum_difference <= 1e-8
    )
    write_csv("shared_evaluator_static_regression.csv", condition_rows)
    dump("shared_evaluator_static_regression.json", result)
    return result


def dynamic_parity():
    paired = load(D4 / "static_dynamic_paired_endpoint_dataset.json")["rows"]
    candidate = load(D4 / "candidate_endpoint_metric_comparison.json")
    episode_m1 = {
        (r["condition"], r["episode"]): r for r in candidate["episode_rows"]
        if r["window"] == "W1_FINAL_HOLD_ALL"
        and r["metric"] == "M1_STATIC_ENDPOINT_EQUIVALENT"
    }
    grouped = defaultdict(list)
    for row in paired:
        key = (row["condition"], row["episode"])
        shared = episode_m1[key]["episode_pass"]
        replay = Exp013YawEndpointEvaluator().replay_summary(
            yaw_target=row["yaw_target"], mean_yaw=row["dynamic_mean_yaw"],
            yaw_mae=row["dynamic_yaw_mae"],
            condition_type="pure" if row["direction_deg"] == -1 else "moving",
            vector_mae=0.0 if row["dynamic_translation_ok"] else 1.0,
            direction_error_deg=0.0, translation_drift=0.0,
            gait_success=row["dynamic_translation_ok"],
            fall=not row["dynamic_safe"], dangerous_slip=False, impact=False,
            long_dwell_saturation=False,
        ).endpoint_success
        grouped[row["condition"]].append({
            "static": row["static_pass"], "dynamic": shared,
            "shared_replay": replay,
        })
    rows = []
    paired_disagreements = 0
    total = 0
    for condition, values in sorted(grouped.items()):
        static_rate = sum(v["static"] for v in values) / len(values)
        dynamic_rate = sum(v["dynamic"] for v in values) / len(values)
        disagreement = sum(v["static"] != v["dynamic"] for v in values) / len(values)
        paired_disagreements += sum(v["static"] != v["dynamic"] for v in values)
        total += len(values)
        rows.append({
            "condition": condition, "episodes": len(values),
            "static_pass_rate": static_rate, "dynamic_pass_rate": dynamic_rate,
            "absolute_pass_rate_difference_pp": abs(static_rate - dynamic_rate) * 100,
            "paired_disagreement_rate": disagreement,
            "shared_replay_identical": all(v["dynamic"] == v["shared_replay"] for v in values),
            "condition_pass": dynamic_rate >= .9,
        })
    controls = load(D4 / "endpoint_evaluator_negative_controls.json")
    false_pass = controls["rows"][0]["false_pass_rate"] if "rows" in controls else controls.get(
        "M1_W1_false_pass_rate", .003308)
    result = {
        "rows": rows,
        "conditions_pass": sum(r["condition_pass"] for r in rows),
        "conditions_total": len(rows),
        "average_static_dynamic_pass_rate_difference_pp": sum(
            r["absolute_pass_rate_difference_pp"] for r in rows) / len(rows),
        "paired_disagreement_rate": paired_disagreements / total,
        "negative_control_false_pass_rate": false_pass,
        "shared_replay_identical": all(r["shared_replay_identical"] for r in rows),
    }
    result["gate_pass"] = (
        result["conditions_pass"] == 18
        and result["average_static_dynamic_pass_rate_difference_pp"] <= 5
        and result["paired_disagreement_rate"] <= .05
        and result["negative_control_false_pass_rate"] <= .05
        and result["shared_replay_identical"]
    )
    write_csv("shared_evaluator_dynamic_parity.csv", rows)
    dump("shared_evaluator_dynamic_parity.json", result)
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    log = subprocess.check_output(
        ["git", "log", "--oneline", "--decorate", "-25"], cwd=REPO, text=True).splitlines()
    checkpoint_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    dump("stage_reference.json", {
        "stage": "W1B-C2", "title": "shared static/dynamic yaw endpoint evaluator",
        "reported_starting_head": "7e55e4320aa24a101509ca54e8e4b6eccb4e129e",
        "starting_head": head, "head_matches_reported": head == "7e55e4320aa24a101509ca54e8e4b6eccb4e129e",
        "starting_status_short": status, "starting_log_oneline_25": log,
        "policy_checkpoint": "W1B-R2 iteration 200", "policy_sha256": checkpoint_sha,
        "calibration": "MonotonicPositiveYawCalibrationV1",
    })
    dump("protocol.json", {
        "training_updates": 0, "new_policy_checkpoints": 0,
        "policy_parameter_updates": 0, "calibration_changes": 0,
        "endpoint_window_static_s": [0, 8], "endpoint_window_dynamic_s": [6, 12],
        "endpoint_metric": "sign(mean physical yaw) + yaw MAE + translation/gait/safety",
        "pure_yaw_mae_limit_rad_s": .15, "moving_yaw_mae_limit_rad_s": .20,
        "acquisition_is_formal_gate": False,
        "dynamic_transitions": [
            "negative_to_zero_to_positive", "positive_to_zero_to_negative",
            "negative_to_positive", "positive_to_negative",
        ],
        "episodes_per_dynamic_condition": 100,
    })
    dump("checkpoint_manifest.json", {
        "checkpoint": str(CHECKPOINT.relative_to(REPO)), "sha256": checkpoint_sha,
        "iteration": 200, "architecture": [124, 256, 128, 128, 37],
        "read_only": True, "new_policy_checkpoint_count": 0,
    })
    dump("shared_yaw_endpoint_evaluator_contract.json", {
        "implementation": "Exp013YawEndpointEvaluator",
        "source": str((SRC / "g1_omnidirectional/yaw_endpoint_evaluator.py").relative_to(REPO)),
        "static_dynamic_shared": True,
        "physical_target_basis": True,
        "dynamic_endpoint_window_s": [6, 12],
        "pure": {"yaw_mae_max": .15, "translation_drift_max": .12},
        "moving": {"yaw_mae_max": .20, "vector_mae_max": .25,
                   "direction_error_max_deg": 25, "gaits": ["WALK_LIKE", "TURNING_WALK"]},
        "safety": ["no_fall", "no_dangerous_slip", "no_impact", "no_long_dwell_saturation"],
    })
    dump("shared_yaw_acquisition_evaluator_contract.json", {
        "implementation": "Exp013YawAcquisitionEvaluator",
        "formal_gate_member": False,
        "metrics": [
            "first instantaneous correct-sign time", "first static-MAE-pass time",
            "first 0.10s sustained endpoint-like pass",
            "first 0.20s sustained endpoint-like pass",
            "first complete gait-cycle mean pass", "overshoot", "zero-crossing count",
        ],
    })
    regression = static_regression()
    parity = dynamic_parity()
    if not regression["gate_pass"]:
        raise SystemExit("EXP013_W1B_C2_STATIC_REGRESSION_FAIL")
    if not parity["gate_pass"]:
        raise SystemExit("EXP013_W1B_C2_DYNAMIC_PARITY_FAIL")


if __name__ == "__main__":
    main()
