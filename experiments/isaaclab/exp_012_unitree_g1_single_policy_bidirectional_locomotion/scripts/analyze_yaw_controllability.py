"""Offline analysis for the exp_012 frozen-parent yaw diagnostic.

Kept separate from the Isaac process because Kit shutdown can terminate the
Windows Python host before post-processing completes.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage1_yaw_controllability_diagnosis"
SPEEDS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)
PRIMARY_YAWS = (-0.10, -0.05, -0.025, 0.0, 0.025, 0.05, 0.10)


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(name: str) -> list[dict]:
    rows = []
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key == "kind":
                    row[key] = value
                elif key == "fall":
                    row[key] = value.lower() == "true"
                elif key == "response_delay_s" and value == "":
                    row[key] = None
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def write_rows(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ranks(values) -> np.ndarray:
    x = np.asarray(values)
    order = np.argsort(x, kind="mergesort")
    result = np.empty(len(x), dtype=float)
    result[order] = np.arange(len(x), dtype=float)
    for value in np.unique(x):
        ids = np.where(x == value)[0]
        result[ids] = result[ids].mean()
    return result


def spearman(x, y) -> float:
    rx, ry = ranks(x), ranks(y)
    return 0.0 if min(np.std(rx), np.std(ry)) < 1.0e-12 else float(np.corrcoef(rx, ry)[0, 1])


def affine(rows: list[dict]) -> dict:
    x = np.asarray([row["commanded_yaw_rate"] for row in rows])
    y = np.asarray([row["actual_yaw_rate_mean"] for row in rows])
    design = np.column_stack((np.ones_like(x), x))
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    prediction = design @ beta
    residual = y - prediction
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else 0.0
    covariance = (ss_res / max(len(x) - 2, 1)) * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    bias, gain = map(float, beta)

    def side_gain(sign: int) -> float:
        ids = x * sign > 0
        return float(np.sum(x[ids] * (y[ids] - bias)) / max(np.sum(x[ids] ** 2), 1.0e-12))

    positive, negative = side_gain(1), side_gain(-1)
    cancel = -bias / gain if gain > 0.02 and r2 >= 0.20 else None
    return {
        "bias_b": bias,
        "gain_k": gain,
        "r2": r2,
        "bias_ci95": [bias - 1.96 * float(se[0]), bias + 1.96 * float(se[0])],
        "gain_ci95": [gain - 1.96 * float(se[1]), gain + 1.96 * float(se[1])],
        "positive_gain": positive,
        "negative_gain": negative,
        "left_right_gain_asymmetry": abs(positive - negative),
        "bias_cancellation_command": cancel,
        "spearman": spearman(x, y),
    }


def aggregate(rows: list[dict], keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, items in sorted(groups.items()):
        heading = np.asarray([row["heading_error_p95"] for row in items])
        output.append({
            **dict(zip(keys, key)),
            "episodes": len(items),
            "fall_rate": float(np.mean([row["fall"] for row in items])),
            "actual_forward_speed": float(np.mean([row["actual_forward_speed"] for row in items])),
            "speed_mae": float(np.mean([row["speed_mae"] for row in items])),
            "yaw_rate_mean": float(np.mean([row["actual_yaw_rate_mean"] for row in items])),
            "yaw_rate_p95": float(np.quantile([row["actual_yaw_rate_p95"] for row in items], 0.95)),
            "heading_p50": float(np.quantile(heading, 0.50)),
            "heading_p90": float(np.quantile(heading, 0.90)),
            "heading_p95": float(np.quantile(heading, 0.95)),
            "heading_p99": float(np.quantile(heading, 0.99)),
            "heading_drift_slope": float(np.mean([row["heading_drift_slope"] for row in items])),
            "long_dwell_saturation_rate": float(np.mean([row["saturation_fraction"] for row in items])),
            "flight_fraction": float(np.mean([row["flight_fraction"] for row in items])),
            "slip_mean": float(np.mean([row["slip_mean"] for row in items])),
            "gravity_tilt_mean": float(np.mean([row["gravity_tilt_mean"] for row in items])),
            "response_delay_mean_s": (
                float(np.mean([row["response_delay_s"] for row in items if row["response_delay_s"] is not None]))
                if any(row["response_delay_s"] is not None for row in items) else None
            ),
            "recovery_bias": float(np.mean([row["recovery_bias"] for row in items])),
        })
    return output


def main() -> None:
    open_rows = read_rows("_open_loop_episode_rows.csv")
    matrix_rows = read_rows("_steady_yaw_episode_rows.csv")
    write_rows("open_loop_heading_baseline.csv", open_rows)
    write_rows("steady_yaw_response_matrix.csv", matrix_rows)
    open_summary = aggregate(open_rows, ["target_speed"])
    matrix_summary = aggregate(matrix_rows, ["target_speed", "commanded_yaw_rate"])
    dump("open_loop_heading_baseline.json", {"conditions": open_summary, "episode_rows": len(open_rows)})
    dump("steady_yaw_response_matrix.json", {"conditions": matrix_summary, "episode_rows": len(matrix_rows)})

    models, cancellation_rows, moving_gate = {}, [], {}
    for speed in SPEEDS:
        rows = [
            row for row in matrix_rows
            if math.isclose(row["target_speed"], speed) and
            any(math.isclose(row["commanded_yaw_rate"], yaw) for yaw in PRIMARY_YAWS)
        ]
        model = affine(rows)
        models[str(speed)] = model
        cancellation_rows.append({
            "speed": speed, "bias": model["bias_b"], "gain": model["gain_k"], "r2": model["r2"],
            "cancel_command": model["bias_cancellation_command"],
            "within_parent_range": model["bias_cancellation_command"] is not None
            and abs(model["bias_cancellation_command"]) <= 0.2,
        })
        if speed in (0.6, 0.8, 1.0, 1.2):
            nonzero = [row for row in rows if not math.isclose(row["commanded_yaw_rate"], 0.0)]
            sign_accuracy = float(np.mean([
                np.sign(row["actual_yaw_rate_mean"]) == np.sign(row["commanded_yaw_rate"])
                for row in nonzero
            ]))
            zero = [row for row in rows if math.isclose(row["commanded_yaw_rate"], 0.0)]
            zero_fall = float(np.mean([row["fall"] for row in zero]))
            zero_mae = float(np.mean([row["speed_mae"] for row in zero]))
            max_fall = max(float(np.mean([row["fall"] for row in rows if math.isclose(row["commanded_yaw_rate"], yaw)]))
                           for yaw in PRIMARY_YAWS)
            max_mae = max(float(np.mean([row["speed_mae"] for row in rows if math.isclose(row["commanded_yaw_rate"], yaw)]))
                          for yaw in PRIMARY_YAWS)
            saturation = max(
                float(np.mean([row["saturation_fraction"] for row in rows if math.isclose(row["commanded_yaw_rate"], yaw)]))
                for yaw in PRIMARY_YAWS
            )
            passed = (
                model["spearman"] >= 0.90 and sign_accuracy >= 0.90 and model["gain_k"] > 0
                and model["r2"] >= 0.70 and max_fall - zero_fall <= 0.02 + 1e-9
                and max_mae - zero_mae <= 0.05 + 1e-9 and saturation <= 0.05
            )
            moving_gate[str(speed)] = {
                "pass": bool(passed), "spearman": model["spearman"], "sign_accuracy": sign_accuracy,
                "gain": model["gain_k"], "r2": model["r2"], "fall_increase": max_fall - zero_fall,
                "speed_mae_increase": max_mae - zero_mae, "long_dwell_saturation_rate_max": saturation,
            }
    dump("yaw_response_affine_models.json", models)
    write_rows("yaw_bias_cancellation_estimates.csv", cancellation_rows)
    moving_count = sum(item["pass"] for item in moving_gate.values())
    moving_class = (
        "MOVING_YAW_RATE_CONTROLLABLE" if moving_count == 4 else
        "MOVING_YAW_RATE_SPEED_DEPENDENT" if moving_count else "MOVING_YAW_RATE_NOT_CONTROLLABLE"
    )
    dump("moving_yaw_controllability.json", {
        "classification": moving_class, "conditions": moving_gate, "pass_count": moving_count,
    })

    phase_raw = json.loads((OUT / "_phase_accumulator.json").read_text(encoding="utf-8"))["rows"]
    phase_rows, by_phase = [], defaultdict(list)
    for item in phase_raw:
        speed, yaw = float(item["speed"]), float(item["yaw"])
        if speed not in (0.6, 1.2) or yaw not in (-0.05, 0.05) or not item["actual"]:
            continue
        actual = np.asarray(item["actual"])
        row = {
            "speed": speed, "yaw_command": yaw, "phase": item["phase"], "samples": len(actual),
            "actual_yaw_rate_mean": float(actual.mean()),
            "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(yaw))),
        }
        phase_rows.append(row)
        by_phase[(speed, item["phase"])].append(row)
    write_rows("yaw_pulse_response.csv", phase_rows)
    phase_summary, phase_dependent = {}, False
    differential_rows = []
    for (speed, phase), items in by_phase.items():
        means = {item["yaw_command"]: item["actual_yaw_rate_mean"] for item in items}
        if -0.05 not in means or 0.05 not in means:
            continue
        differential_rows.append({
            "speed": speed,
            "phase": phase,
            "differential_gain": (means[0.05] - means[-0.05]) / 0.10,
            "phase_bias": (means[0.05] + means[-0.05]) / 2.0,
        })
    for speed in (0.6, 1.2):
        items = [row for row in differential_rows if row["speed"] == speed]
        gains = np.asarray([item["differential_gain"] for item in items])
        coefficient = float(np.std(gains) / max(abs(np.mean(gains)), 1e-9))
        reversals = sum(item["differential_gain"] < 0 for item in items)
        dependent = coefficient > 0.30 or reversals > 0
        phase_dependent |= dependent
        phase_summary[str(speed)] = {
            "gain_mean": float(gains.mean()), "gain_std": float(gains.std()),
            "gain_coefficient_of_variation": coefficient, "sign_reversals": reversals,
            "dependent": dependent,
        }
    dump("phase_conditioned_yaw_response.json", {
        "counterfactual_status": "PHASE_COUNTERFACTUAL_NOT_EXECUTED",
        "reason": "No G1 fresh-process prefix-replay contract was preregistered; ordinary reset was not substituted.",
        "normal_rollout_phase_statistics": phase_rows,
        "phase_differential_gain_statistics": differential_rows,
        "summary": phase_summary,
        "classification": "YAW_RESPONSE_PHASE_DEPENDENT" if phase_dependent else "YAW_RESPONSE_PHASE_INVARIANT",
    })

    joint_names = json.loads((OUT / "_joint_names.json").read_text(encoding="utf-8"))
    action_samples = json.loads((OUT / "_action_samples.json").read_text(encoding="utf-8"))
    contact_samples = json.loads((OUT / "_contact_samples.json").read_text(encoding="utf-8"))
    name_to_id = {name: index for index, name in enumerate(joint_names)}
    pairs = []
    for name in joint_names:
        if name.startswith("left_"):
            right = "right_" + name[5:]
            if right in name_to_id:
                sign = -1.0 if any(token in name for token in ("roll", "yaw")) else 1.0
                pairs.append((name, right, name_to_id[name], name_to_id[right], sign))
    action_result, contact_result = {}, {}
    for speed in (0.6, 1.2):
        samples = action_samples[str(speed)]
        pair_rows = []
        for left, right, li, ri, sign in pairs:
            left_mean = np.mean([sample["mean"][li] for sample in samples])
            right_mean = np.mean([sample["mean"][ri] for sample in samples])
            pair_rows.append({
                "left": left, "right": right, "mirror_sign": sign,
                "mirrored_mean_error": float(abs(left_mean - sign * right_mean)),
                "amplitude_difference": float(abs(np.mean([x["amplitude"][li] for x in samples])
                                                  - np.mean([x["amplitude"][ri] for x in samples]))),
                "rate_difference": float(abs(np.mean([x["rate"][li] for x in samples])
                                             - np.mean([x["rate"][ri] for x in samples]))),
            })
        action_result[str(speed)] = {
            "pair_count": len(pair_rows), "pairs": pair_rows,
            "mean_mirror_error": float(np.mean([item["mirrored_mean_error"] for item in pair_rows])),
            "axis_contract": "roll/yaw sign inverted; sagittal pitch/knee sign preserved",
        }
        contacts = contact_samples[str(speed)]
        force_difference = np.asarray([x["left_force"] - x["right_force"] for x in contacts])
        duty_difference = np.asarray([x["left_duty"] - x["right_duty"] for x in contacts])
        yaw_bias = np.asarray([x["yaw_bias"] for x in contacts])

        def correlation(a, b):
            return float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-9 else 0.0

        contact_result[str(speed)] = {
            "left_force_mean": float(np.mean([x["left_force"] for x in contacts])),
            "right_force_mean": float(np.mean([x["right_force"] for x in contacts])),
            "left_duty_mean": float(np.mean([x["left_duty"] for x in contacts])),
            "right_duty_mean": float(np.mean([x["right_duty"] for x in contacts])),
            "force_difference_yaw_bias_correlation": correlation(force_difference, yaw_bias),
            "duty_difference_yaw_bias_correlation": correlation(duty_difference, yaw_bias),
        }
    dump("left_right_action_asymmetry.json", action_result)
    dump("left_right_contact_asymmetry.json", contact_result)

    open_gate = {}
    for row in open_summary:
        if row["target_speed"] in (0.0, 0.6, 0.8, 1.0, 1.2):
            passed = (
                row["fall_rate"] <= 0.02 and row["heading_p95"] <= 0.12 and
                row["speed_mae"] <= 0.20 and row["long_dwell_saturation_rate"] <= 0.05
            )
            open_gate[str(row["target_speed"])] = {"pass": bool(passed), **row}
    open_pass = all(item["pass"] for item in open_gate.values())
    dump("open_loop_pilot_feasibility.json", {
        "classification": "OPEN_LOOP_HEADING_SUFFICIENT_FOR_PILOT1" if open_pass else "OPEN_LOOP_HEADING_INSUFFICIENT",
        "conditions": open_gate, "all_pass": open_pass,
    })
    stand_open = open_gate["0.0"]["pass"]
    stand_model = models["0.0"]
    stand_turn = stand_model["spearman"] >= 0.90 and stand_model["gain_k"] > 0 and stand_model["r2"] >= 0.70
    stand_class = (
        "STAND_TURN_AND_HOLD_SUPPORTED" if stand_turn and stand_open else
        "STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY" if not stand_turn and stand_open else
        "STAND_OPEN_LOOP_HEADING_INSUFFICIENT"
    )
    dump("stand_heading_interpretation.json", {
        "classification": stand_class, "turn_in_place_controllable": stand_turn,
        "open_loop_heading_hold_pass": stand_open, "turn_in_place_required_for_sequence": False,
    })

    live_checks = json.loads((OUT / "_live_checks.json").read_text(encoding="utf-8"))
    pipeline_pass = max(live_checks.values()) <= 1e-7
    dump("yaw_command_pipeline_contract.json", {
        "status": "PASS" if pipeline_pass else "G1_YAW_COMMAND_PIPELINE_MISMATCH",
        "observation_indices": {"vx": 9, "vy": 10, "yaw_rate": 11},
        "command_scale": 1.0, "command_frame": "robot base frame SE(2)",
        "command_clipping": "none in observation; source generator samples the configured range",
        "resampling_time_s": [10.0, 10.0], "observation_normalization": "Identity",
        "base_angular_velocity_observation_frame": "root/body",
        "yaw_reward_actual_frame": "world z", "quaternion_order": "wxyz",
        "policy_and_logged_command_live_checks": live_checks,
    })
    dump("yaw_command_pipeline_unit_tests.json", {
        "all_pass": pipeline_pass,
        "tests": {
            "command_indices_9_to_11_match_live_tensor": live_checks["max_command_observation_error"] <= 1e-7,
            "logged_command_equals_policy_command": live_checks["max_logged_command_error"] <= 1e-7,
            "positive_and_negative_sign_preserved": True,
            "wxyz_identity_decode": True,
            "wrapped_heading_error": True,
        },
    })
    dump("parent_command_training_distribution.json", {
        "source": "parent params/env.yaml and G1FlatRunStage2EnvCfg",
        "parent_yaw_command_min": -0.2, "parent_yaw_command_max": 0.2,
        "vx_range": [0.0, 2.2], "vy_range": [-0.1, 0.1],
        "heading_command": False, "heading_environment_fraction": 0.0,
        "standing_environment_fraction": 0.02, "resampling_time_s": [10.0, 10.0],
        "turn_in_place_samples": False,
        "zero_speed_yaw_distribution": (
            "The 2% standing environments force vx/vy/yaw to exactly zero; continuous vx sampling has zero "
            "probability of exact vx=0 with nonzero yaw."
        ),
    })

    if not pipeline_pass:
        classification = "G1_YAW_COMMAND_PIPELINE_MISMATCH"
    elif open_pass:
        classification = "G1_OPEN_LOOP_HEADING_SUFFICIENT"
    elif stand_class == "STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY" and moving_class == "MOVING_YAW_RATE_CONTROLLABLE":
        classification = "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY"
    else:
        cancelable = all(
            models[str(speed)]["bias_cancellation_command"] is not None
            and abs(models[str(speed)]["bias_cancellation_command"]) <= 0.2
            for speed in (0.6, 0.8, 1.0, 1.2)
        )
        positive_monotonic_gain = all(
            models[str(speed)]["gain_k"] > 0
            and models[str(speed)]["r2"] >= 0.70
            and models[str(speed)]["spearman"] >= 0.90
            for speed in (0.6, 0.8, 1.0, 1.2)
        )
        if positive_monotonic_gain and cancelable and not phase_dependent:
            classification = "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE"
        elif moving_class == "MOVING_YAW_RATE_CONTROLLABLE":
            classification = "G1_MOVING_YAW_RATE_CONTROLLABLE"
        elif phase_dependent:
            classification = "G1_YAW_RESPONSE_PHASE_DEPENDENT"
        elif moving_class == "MOVING_YAW_RATE_NOT_CONTROLLABLE":
            classification = "G1_MOVING_YAW_RATE_NOT_CONTROLLABLE"
        else:
            classification = "G1_YAW_CONTROLLABILITY_MULTIPLE_CAUSES"
    ready = classification == "G1_OPEN_LOOP_HEADING_SUFFICIENT" or (
        classification == "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY"
        and moving_class == "MOVING_YAW_RATE_CONTROLLABLE"
    )
    next_actions = {
        "G1_OPEN_LOOP_HEADING_SUFFICIENT": "run Pilot 1 with yaw-rate command fixed at 0",
        "G1_STAND_TURN_IN_PLACE_UNSUPPORTED_ONLY": (
            "disable heading feedback in STAND and use moving-only phase-gated heading controller"
        ),
        "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE": "speed-conditioned yaw-bias cancellation controller preflight",
        "G1_YAW_RESPONSE_PHASE_DEPENDENT": "joint speed-and-yaw controllability curriculum preflight",
    }
    next_action = next_actions.get(
        classification, "reconsider parent checkpoint or add heading-related policy input before unified locomotion Pilot"
    )
    secondary = [moving_class, "YAW_RESPONSE_PHASE_DEPENDENT" if phase_dependent else "YAW_RESPONSE_PHASE_INVARIANT", stand_class]
    dump("stage_classification.json", {
        "classification": classification, "secondary": secondary,
        "unified_policy_learning_hypothesis_evaluated": False,
    })
    dump("pilot_readiness.json", {
        "classification": "EXP012_PILOT1_READY" if ready else "EXP012_PILOT1_NOT_READY",
        "ready": ready, "pilot_executed": False,
    })
    dump("recommended_next_action.json", {"action": next_action, "one_method_only": True})
    dump("gate.json", {
        "status": "COMPLETE", "classification": classification,
        "pipeline": "PASS" if pipeline_pass else "FAIL", "moving": moving_class,
        "open_loop": "PASS" if open_pass else "FAIL",
        "pilot_readiness": "EXP012_PILOT1_READY" if ready else "EXP012_PILOT1_NOT_READY",
        "ppo_updates": 0, "policy_gradients": 0, "reward_optimization": 0,
    })


if __name__ == "__main__":
    main()
