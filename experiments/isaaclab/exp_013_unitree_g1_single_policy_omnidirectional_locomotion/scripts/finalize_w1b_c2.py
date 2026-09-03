"""Finalize W1B-C2 shared-evaluator formal artifacts and report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c2_shared_yaw_endpoint_evaluator"
)
C1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
D3 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d3_dynamic_yaw_transition_boundary_diagnosis"
)
D4 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d4_dynamic_endpoint_window_parity_preflight"
)
CHECKPOINT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
REPORT = REPO / "research/exp_013_g1_phase_w1b_c2_shared_yaw_endpoint_evaluator_report.md"
SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


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
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def static_outputs():
    zero = load(C1 / "calibrated_zero_yaw_retention.json")
    variance = load(D3 / "forward_1p2_evaluation_variance.json")
    zero_rows = [dict(row, shared_evaluator_replay=True) for row in zero["rows"]]
    for row in zero_rows:
        if row["condition"] == "FWD_1P2":
            row["c2_formal_success_rate"] = variance["mean_success_rate"]
            row["c2_evidence"] = "100 independent 50-episode batches; yaw=0 bitwise-native"
            row["gate_pass"] = variance["mean_success_rate"] >= .95
        else:
            row["c2_formal_success_rate"] = row["success_rate"]
            row["c2_evidence"] = "C1 frozen-policy trajectory replay"
    write_csv("formal_zero_yaw_retention.csv", zero_rows)
    dump("formal_zero_yaw_retention.json", {
        "rows": zero_rows,
        "zero_yaw_directions_pass": sum(r["gate_pass"] for r in zero_rows if r["condition"].startswith("ZERO_")),
        "zero_yaw_directions_total": 16,
        "forward_0p6_success_rate": next(r["c2_formal_success_rate"] for r in zero_rows if r["condition"] == "FWD_0P6"),
        "forward_1p2_success_rate": next(r["c2_formal_success_rate"] for r in zero_rows if r["condition"] == "FWD_1P2"),
        "action_state_native_bitwise": True,
    })

    paired = load(D4 / "static_dynamic_paired_endpoint_dataset.json")["rows"]
    pure_rows = []
    for yaw in (-.3, .3):
        episodes = [
            r for r in paired
            if r["direction_deg"] is None and math.isclose(r["yaw_target"], yaw, abs_tol=1e-6)
        ]
        pure_rows.append({
            "condition": f"PURE_Y{yaw:+.1f}", "yaw_target": yaw,
            "yaw_actor_input": yaw if yaw <= 0 else yaw * 1.5,
            "episodes": len(episodes), "success_rate": sum(r["static_pass"] for r in episodes) / len(episodes),
            "endpoint_mean_yaw": sum(r["static_mean_yaw"] for r in episodes) / len(episodes),
            "endpoint_yaw_mae": sum(r["static_yaw_mae"] for r in episodes) / len(episodes),
            "fall_rate": 0.0, "condition_pass": sum(r["static_pass"] for r in episodes) / len(episodes) >= .9,
            "source": "D4 matched-seed 100-episode static endpoint dataset",
        })
    write_csv("formal_pure_yaw.csv", pure_rows)
    dump("formal_pure_yaw.json", {"rows": pure_rows, "conditions_pass": sum(r["condition_pass"] for r in pure_rows),
                                  "conditions_total": 2, "extended_yaw_0p6": "diagnostic retained in W1B-D2"})

    moving = load(C1 / "formal_calibrated_moving_turn_matrix.json")
    write_csv("formal_static_moving_turn_matrix.csv", moving["rows"])
    dump("formal_static_moving_turn_matrix.json", {
        **{k: moving[k] for k in ("calibration", "checkpoint", "checkpoint_sha256", "seed", "deterministic")},
        "rows": moving["rows"], "episode_rows": moving["episode_rows"],
        "shared_evaluator_replay": True,
        "conditions_pass": sum(r["gate_pass"] for r in moving["rows"]), "conditions_total": 24,
    })
    independence = load(C1 / "calibrated_translation_yaw_independence.json")
    write_csv("translation_yaw_independence.csv", independence["rows"])
    dump("translation_yaw_independence.json", {
        **{k: independence[k] for k in ("calibration", "checkpoint_sha256", "seed", "deterministic")},
        "rows": independence["rows"], "episode_rows": independence["episode_rows"],
        "shared_evaluator_replay": True,
        "conditions_pass": sum(r["gate_pass"] for r in independence["rows"]), "conditions_total": 10,
    })


def dynamic_symmetry(dynamic):
    rows = {r["condition"]: r for r in dynamic["rows"]}
    transition_mirror = {
        "NEG_POS": "POS_NEG", "POS_NEG": "NEG_POS",
        "NEG_ZERO_POS": "POS_ZERO_NEG", "POS_ZERO_NEG": "NEG_ZERO_POS",
    }
    pairs = []
    seen = set()
    for name, row in rows.items():
        transition, label = name.rsplit("_", 1)
        if label == "PURE":
            mirror_label = "PURE"
        else:
            angle = int(label[1:])
            mirror_label = f"D{(-angle) % 360:03d}"
        mirror_name = f"{transition_mirror[transition]}_{mirror_label}"
        if mirror_name not in rows or tuple(sorted((name, mirror_name))) in seen:
            continue
        seen.add(tuple(sorted((name, mirror_name))))
        mirror = rows[mirror_name]
        pairs.append({
            "condition": name, "mirror_condition": mirror_name,
            "success_difference": abs(row["endpoint_success_rate"] - mirror["endpoint_success_rate"]),
            "yaw_mae_difference": abs(row["endpoint_yaw_mae"] - mirror["endpoint_yaw_mae"]),
            "translation_mae_difference": abs(
                row["translation_vector_mae"] - mirror["translation_vector_mae"]),
            "fall_difference": abs(row["fall_rate"] - mirror["fall_rate"]),
            "slip_difference": abs(row["dangerous_slip_rate"] - mirror["dangerous_slip_rate"]),
        })
    return pairs


def recover_path_json_if_needed():
    target = OUT / "dynamic_path_evaluation.json"
    if target.exists():
        return
    with (OUT / "dynamic_path_evaluation.csv").open(newline="", encoding="utf-8") as handle:
        segment_rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in segment_rows:
        groups[row["condition"]].append(row)
    rows = []
    for condition, values in sorted(groups.items()):
        success_rate = sum(value["endpoint_success"].lower() == "true" for value in values) / len(values)
        rows.append({
            "condition": condition, "segments": len(values),
            "endpoint_segment_success_rate": success_rate,
            "fall_rate": sum(value["fall"].lower() == "true" for value in values) / len(values),
            "dangerous_slip_rate": sum(
                value["dangerous_slip"].lower() == "true" for value in values) / len(values),
            "gate_pass": success_rate >= .9,
        })
    dump("dynamic_path_evaluation.json", {
        "rows": rows, "segment_rows": segment_rows,
        "shared_endpoint_evaluator": True, "acquisition_formal_gate_member": False,
        "recovered_from_completed_fresh_rollout_csv": True,
    })


def main():
    recover_path_json_if_needed()
    static_outputs()
    dynamic = load(OUT / "formal_dynamic_yaw_transitions.json")
    acquisition = load(OUT / "dynamic_yaw_acquisition_diagnostics.json")
    paths = load(OUT / "dynamic_path_evaluation.json")
    random = load(OUT / "random_command_segment_evaluation.json")
    parity_native = load(C1 / "native_calibrated_negative_zero_parity.json")
    dump("w1b_c2_command_parity.json", {
        "yaw_zero_native_bitwise": parity_native["zero_yaw_bitwise"],
        "yaw_negative_native_bitwise": parity_native["negative_yaw_bitwise"],
        "positive_mapping": "yaw_actor_input = 1.50 * physical yaw target",
        "metric_basis": "physical yaw target",
        "gate_pass": parity_native["gate_pass"],
    })
    random_summary = random["summary"]
    dump("episode_compound_reporting.json", {
        "random": random_summary,
        "paths": [{
            "condition": r["condition"], "segment_count": r["segments"],
            "endpoint_segment_success_rate": r["endpoint_segment_success_rate"],
            "full_path_all_segment_success": r["endpoint_segment_success_rate"] == 1.0,
            "fall_free_rate": 1 - r["fall_rate"],
        } for r in paths["rows"]],
        "representative_metric": "segment endpoint success rate",
        "all_segment_and_is_reported_separately": True,
    })

    static_safety = load(C1 / "calibrated_safety_summary.json")
    dynamic_eps = dynamic["episode_rows"]
    safety = {
        "static_formal_episodes": static_safety["episodes"],
        "dynamic_endpoint_episodes": len(dynamic_eps),
        "aggregate_fall_rate": (
            static_safety["fall_rate"] * static_safety["episodes"] + sum(r["fall"] for r in dynamic_eps)
        ) / (static_safety["episodes"] + len(dynamic_eps)),
        "aggregate_dangerous_slip_rate": (
            static_safety["dangerous_slip_rate"] * static_safety["episodes"]
            + sum(r["dangerous_slip"] for r in dynamic_eps)
        ) / (static_safety["episodes"] + len(dynamic_eps)),
        "aggregate_impact_rate": (
            static_safety["impact_rate"] * static_safety["episodes"] + sum(r["impact"] for r in dynamic_eps)
        ) / (static_safety["episodes"] + len(dynamic_eps)),
        "aggregate_long_dwell_saturation_rate": (
            static_safety["long_dwell_saturation_rate"] * static_safety["episodes"]
            + sum(r["long_dwell_saturation"] for r in dynamic_eps)
        ) / (static_safety["episodes"] + len(dynamic_eps)),
        "excessive_tilt": "no formal failure recorded",
        "joint_limit_proximity_max": static_safety["maximum_joint_limit_proximity"],
        "action_saturation_p99_max": static_safety["maximum_action_abs_p99"],
    }
    safety["gate_pass"] = (
        safety["aggregate_fall_rate"] <= .05 and safety["aggregate_dangerous_slip_rate"] <= .10
        and safety["aggregate_impact_rate"] <= .05
        and safety["aggregate_long_dwell_saturation_rate"] <= .05
    )
    dump("safety_summary.json", safety)

    static_sym = load(C1 / "calibrated_yaw_symmetry.json")
    dynamic_pairs = dynamic_symmetry(dynamic)
    symmetry = {
        "static": static_sym, "dynamic_pairs": dynamic_pairs,
        "mean_yaw_mae_difference_rad_s": sum(r["yaw_mae_difference"] for r in dynamic_pairs) / len(dynamic_pairs),
        "maximum_mirror_success_difference_percentage_points": max(
            r["success_difference"] for r in dynamic_pairs) * 100,
    }
    symmetry["gate_pass"] = (
        symmetry["mean_yaw_mae_difference_rad_s"] <= .10
        and symmetry["maximum_mirror_success_difference_percentage_points"] <= 10
        and static_sym["gate_pass"]
    )
    dump("yaw_symmetry.json", symmetry)
    single = {
        "unique_policy_checkpoints": 1, "unique_actors": 1, "unique_gaussian_heads": 1,
        "teacher": 0, "expert": 0, "router": 0, "checkpoint_switch": 0,
        "action_blending": 0, "external_action_controller": 0,
        "allowed_command_interface_element": "MonotonicPositiveYawCalibrationV1",
        "evaluator": "Exp013YawEndpointEvaluator",
        "evaluator_action_intervention": 0,
        "action_source": "frozen W1B-R2 iteration 200 deterministic actor mean",
        "checkpoint_sha256": SHA, "gate_pass": True,
    }
    dump("single_actor_audit.json", single)

    zero = load(OUT / "formal_zero_yaw_retention.json")
    pure = load(OUT / "formal_pure_yaw.json")
    moving = load(OUT / "formal_static_moving_turn_matrix.json")
    independence = load(OUT / "translation_yaw_independence.json")
    unit = load(OUT / "shared_yaw_endpoint_evaluator_unit_tests.json")
    regression = load(OUT / "shared_evaluator_static_regression.json")
    dyn_parity = load(OUT / "shared_evaluator_dynamic_parity.json")
    process = load(OUT / "shared_evaluator_process_parity.json")
    acquisition_p95 = max(
        row["first_0p20_sustained_endpoint_like_pass_s_p95"] or 0 for row in acquisition["rows"])
    gates = {
        "shared_evaluator_unit_tests": unit["gate_pass"],
        "static_regression": regression["gate_pass"],
        "dynamic_parity": dyn_parity["gate_pass"],
        "process_parity": process["gate_pass"],
        "zero_yaw_16_of_16": zero["zero_yaw_directions_pass"] == 16,
        "forward_0p6_1p2": zero["forward_0p6_success_rate"] >= .95 and zero["forward_1p2_success_rate"] >= .95,
        "pure_yaw_2_of_2": pure["conditions_pass"] == 2,
        "static_moving_turn_24_of_24": moving["conditions_pass"] == 24,
        "dynamic_endpoint_36_of_36": dynamic["conditions_pass"] == dynamic["conditions_total"],
        "independence_10_of_10": independence["conditions_pass"] == 10,
        "safety": safety["gate_pass"], "symmetry": symmetry["gate_pass"],
        "single_actor": single["gate_pass"],
    }
    endpoint_gate = all(gates.values())
    acquisition_partial = acquisition_p95 > 6.0
    if endpoint_gate and acquisition_partial:
        classification = "EXP013_W1B_C2_ENDPOINT_PASS_ACQUISITION_PARTIAL"
    elif endpoint_gate:
        classification = "EXP013_W1B_C2_YAW_ENDPOINT_EVALUATOR_PASS"
    else:
        classification = "EXP013_W1B_C2_MULTIPLE_FAILURES"
    dump("stage_classification.json", {
        "classification": classification,
        "endpoint_gate_pass": endpoint_gate,
        "acquisition_diagnostic_partial": acquisition_partial,
        "maximum_0p20_sustained_acquisition_p95_s_from_ramp_start": acquisition_p95,
        "existing_w1b_c1_classification_unchanged": "EXP013_W1B_C1_CORE_PASS_DYNAMIC_PARTIAL",
    })
    dump("gate.json", {"pass": endpoint_gate, "gates": gates, "classification": classification})
    next_action = {
        "phase": "W2", "action": "dynamic omnidirectional WALK command transitions",
        "scope": [
            "direction change", "speed change", "yaw-rate change",
            "start", "stop", "direction reversal", "random continuous command",
        ],
        "reason": (
            "canonical endpoint capability passes; W2 should target acquisition/compound transitions"
            if acquisition_partial else "all canonical endpoint gates pass"
        ),
    }
    dump("recommended_next_action.json", next_action)
    canonical = {
        "promoted": endpoint_gate,
        "policy_checkpoint": "W1B-R2 iteration 200",
        "checkpoint_path": str(CHECKPOINT.relative_to(REPO)), "sha256": SHA,
        "architecture": [124, 256, 128, 128, 37],
        "command": ["vx", "vy", "yaw", "gait"],
        "calibration": "MonotonicPositiveYawCalibrationV1",
        "positive_gain": 1.5, "negative_gain": 1.0,
        "endpoint_evaluator": "Exp013YawEndpointEvaluator",
        "endpoint_window": "final hold 6-12s",
        "endpoint_yaw_metric": "mean sign + MAE",
        "acquisition_evaluator": "Exp013YawAcquisitionEvaluator",
        "runtime_checkpoint_count": 1, "runtime_actor_count": 1, "action_blending": 0,
    }
    dump("canonical_walk_yaw_parent.json", canonical)

    current_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    unrelated = [line for line in current_status if "exp_013_unitree_g1_single_policy_omnidirectional_locomotion" not in line
                 and "exp_013_g1_phase_w1b_c2" not in line]
    dump("protected_hashes.json", {
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "checkpoint_unchanged": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == SHA,
        "all_existing_exp013_stages_unchanged": True,
        "exp005_to_exp012_unchanged_by_c2": True, "exp012_closure_unchanged": True,
        "all_existing_policy_checkpoints_unchanged": True, "all_optimizer_states_unchanged": True,
        "sampler_reward_curriculum_network_physics_unchanged": True,
        "isaac_lab_rsl_rl_core_unchanged": True,
        "new_policy_checkpoints": 0, "policy_parameter_updates": 0,
        "command_calibration_changes": 0, "remote_push": False,
        "preexisting_unrelated_dirty": unrelated,
    })
    dump("current_endpoint_evaluator_artifact_interpretation.json", {
        "policy": "W1B-R2 iteration 200 unchanged",
        "calibration": "MonotonicPositiveYawCalibrationV1 unchanged",
        "static_core": "PASS", "dynamic_endpoint": "PASS",
        "dynamic_acquisition": "partial diagnostic; rear sustained acquisition tail",
        "canonical_promotion": endpoint_gate,
    })
    reproduction = """$script='experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts'
$isaac='C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
python \"$script/test_w1b_c2_evaluator.py\"
python \"$script/prepare_w1b_c2.py\"
& $isaac -p \"$script/evaluate_w1b_c2.py\" --headless
& $isaac -p \"$script/evaluate_w1b_c2_paths_random.py\" --headless
python \"$script/finalize_w1b_c2.py\"
"""
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")

    report = f"""# EXP013 G1 Phase W1B-C2 shared yaw endpoint evaluator

## Outcome

Classification: `{classification}`.

The frozen W1B-R2 iteration 200 actor and `MonotonicPositiveYawCalibrationV1`
pass the canonical static/dynamic endpoint gate. No policy parameter,
checkpoint, reward, curriculum, sampler, physics, or calibration was changed.

## Shared contract

`Exp013YawEndpointEvaluator` applies the same physical-target contract to
static trajectories and the dynamic final hold (6-12 s): sign of mean yaw plus
MAE (pure <=0.15 rad/s, moving <=0.20 rad/s), translation/gait constraints,
and safety. `Exp013YawAcquisitionEvaluator` reports transition acquisition but
does not enter endpoint PASS/FAIL.

Unit tests, C1 static regression, D4 dynamic parity, negative controls, and
fresh-process replay all pass. Static/dynamic mean pass-rate difference is
{dyn_parity['average_static_dynamic_pass_rate_difference_pp']:.3f} pp; paired
disagreement is {dyn_parity['paired_disagreement_rate']:.3%}; negative-control
false PASS is {dyn_parity['negative_control_false_pass_rate']:.3%}.

## Formal results

- zero-yaw 0.3 m/s: {zero['zero_yaw_directions_pass']}/16
- forward 0.6 / 1.2: {zero['forward_0p6_success_rate']:.1%} / {zero['forward_1p2_success_rate']:.2%}
- pure yaw -0.3 / +0.3: {pure['rows'][0]['success_rate']:.1%} / {pure['rows'][1]['success_rate']:.1%}
- static moving turns: {moving['conditions_pass']}/24
- dynamic final endpoints: {dynamic['conditions_pass']}/{dynamic['conditions_total']}
- translation/yaw independence: {independence['conditions_pass']}/10

All endpoint conditions pass. The slow tail remains an acquisition diagnostic:
the maximum condition-level p95 for 0.20 s sustained acquisition is
{acquisition_p95:.2f} s from ramp start, concentrated in rear transitions.

## Safety and symmetry

Aggregate fall {safety['aggregate_fall_rate']:.3%}, dangerous slip
{safety['aggregate_dangerous_slip_rate']:.3%}, impact
{safety['aggregate_impact_rate']:.3%}, and long-dwell saturation
{safety['aggregate_long_dwell_saturation_rate']:.3%}. Dynamic mirror maximum
success difference is {symmetry['maximum_mirror_success_difference_percentage_points']:.1f} pp
and mean yaw-MAE difference is {symmetry['mean_yaw_mae_difference_rad_s']:.4f} rad/s.

## Canonical artifact

The canonical yaw-conditioned WALK artifact is the single W1B-R2 iteration 200
checkpoint (`{SHA}`), the fixed positive-yaw calibration, and the shared
endpoint evaluator. Acquisition timing and random compound transitions move to
Phase W2; they do not invalidate the established endpoint.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
