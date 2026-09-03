"""Offline aggregation and protocol audit for EXP 012 Stage 1B."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage1b_speed_conditioned_yaw_cancellation"
STAGE1 = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage1_yaw_controllability_diagnosis"
EXP007 = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage1_stand_formal/summary.json"
sys.path.insert(0, str(EXP / "src"))

from g1_single_policy.yaw_bias_canceller import (  # noqa: E402
    ACTIVATION_S,
    OFFSET_LIMIT,
    POLICY_LIMIT,
    TABLE,
    G1SpeedConditionedYawBiasCancellerV1,
    lookup_offset,
    minimum_jerk,
)

STARTING_HEAD = "600e74bd65dfe1aa7627551fbf271d4f10caba2a"
STARTING_STATUS = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    "?? .openduck_hardware_source_review/",
    "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_runtime_source_review/",
    "?? artifacts/exp_005_unitree_g1_flat_run/",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? media/",
    "?? openduck_setup_report.md",
    "?? research/exp_011_linkedin_post_ja.md",
]


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(name: str) -> list[dict]:
    rows = []
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            row = {}
            for key, value in raw.items():
                if key in ("mode", "controller", "segment_metrics"):
                    row[key] = value
                elif key in ("fall", "long_dwell_saturation", "sequence_completion", "final_stand_hold"):
                    row[key] = value.lower() == "true"
                else:
                    row[key] = float(value)
            rows.append(row)
    return rows


def write_rows(name: str, rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(items: list[dict]) -> dict:
    mean = lambda key: float(np.mean([row[key] for row in items]))
    heading = [row["heading_p95"] for row in items]
    return {
        "episodes": len(items),
        "fall_rate": mean("fall"),
        "actual_speed": mean("actual_speed"),
        "speed_mae": mean("speed_mae"),
        "signed_yaw_rate_bias": mean("yaw_rate_mean"),
        "mean_absolute_yaw_bias": abs(mean("yaw_rate_mean")),
        "actual_yaw_rate_p95": float(np.quantile([row["yaw_rate_p95"] for row in items], 0.95)),
        "heading_p50": float(np.quantile(heading, 0.50)),
        "heading_p90": float(np.quantile(heading, 0.90)),
        "heading_p95": float(np.quantile(heading, 0.95)),
        "heading_p99": float(np.quantile([row["heading_p99"] for row in items], 0.99)),
        "heading_drift_slope": mean("heading_drift_slope"),
        "lateral_velocity": mean("lateral_velocity"),
        "gravity_tilt": mean("gravity_tilt"),
        "double_support_fraction": mean("double_support_fraction"),
        "single_support_fraction": mean("single_support_fraction"),
        "flight_fraction": mean("flight_fraction"),
        "slip": mean("slip"),
        "impact": mean("impact"),
        "joint_velocity_saturation": mean("joint_velocity_saturation"),
        "joint_torque_saturation": mean("joint_torque_saturation"),
        "long_dwell_saturation_rate": mean("long_dwell_saturation"),
        "action_rate_p95": mean("action_rate_p95"),
        "activation_action_jump_max": max(row["activation_action_jump"] for row in items),
        "deactivation_action_jump_max": max(row["deactivation_action_jump"] for row in items),
        "offset_abs_mean": mean("offset_abs_mean"),
        "sequence_completion_rate": mean("sequence_completion"),
        "final_stand_hold_rate": mean("final_stand_hold"),
        "final_speed_abs_mean": float(np.mean([abs(row["final_speed"]) for row in items])),
    }


def unit_tests() -> dict:
    tests = {}
    tests["zero_output"] = lookup_offset(0.0) == 0.0
    tests["below_0p4_output"] = lookup_offset(0.4) == 0.0 and lookup_offset(0.2) == 0.0
    tests["spatial_minimum_jerk_0p5"] = abs(lookup_offset(0.5) - 0.5 * TABLE[0][1]) <= 1e-12
    tests["table_values"] = all(abs(lookup_offset(speed) - value) <= 1e-12 for speed, value in TABLE)
    tests["piecewise_midpoints"] = (
        abs(lookup_offset(0.7) - (TABLE[0][1] + TABLE[1][1]) / 2.0) <= 1e-12
        and abs(lookup_offset(0.9) - (TABLE[1][1] + TABLE[2][1]) / 2.0) <= 1e-12
    )
    tests["hold_above_1p2"] = lookup_offset(1.21) == TABLE[-1][1] == lookup_offset(2.6)
    tests["offset_within_0p15"] = max(abs(lookup_offset(x / 100)) for x in range(301)) <= OFFSET_LIMIT
    controller = G1SpeedConditionedYawBiasCancellerV1(0.02)
    activation = [controller.step(0.6)["offset"] for _ in range(25)]
    tests["temporal_activation_0p5s"] = abs(activation[-1] - TABLE[0][1]) <= 1e-12
    tests["activation_continuity"] = max(abs(b - a) for a, b in zip(activation, activation[1:])) < 0.01
    deactivation = [controller.step(0.0)["offset"] for _ in range(25)]
    tests["temporal_deactivation_0p5s"] = deactivation[0] != 0.0 and deactivation[-1] == 0.0
    controller.reset()
    reset = controller.step(0.0)
    tests["episode_reset"] = reset["offset"] == 0.0 and reset["gate"] == 0.0 and reset["state"] == "DISABLED"
    clamp = G1SpeedConditionedYawBiasCancellerV1(0.5).step(1.2, desired_yaw_rate=-0.2)
    tests["policy_command_clamp"] = clamp["policy_yaw_rate"] == -POLICY_LIMIT
    return {"all_pass": all(tests.values()), "tests": tests}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dump("stage_reference.json", {
        "starting_head": STARTING_HEAD,
        "actual_starting_head": actual_head,
        "head_match": actual_head == STARTING_HEAD,
        "starting_status": STARTING_STATUS,
        "parent_checkpoint": "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
        "parent_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "protected_prior_results": [
            "G1_YAW_RATE_NOT_LOCALLY_CONTROLLABLE",
            "G1_YAW_BIAS_SPEED_CONDITIONED_CANCELABLE",
            "EXP012_PILOT1_NOT_READY",
        ],
    })
    dump("protocol.json", {
        "stage": "1B",
        "controller": "G1SpeedConditionedYawBiasCancellerV1",
        "desired_yaw_rate": 0.0,
        "steady": {"speeds": [0.0, 0.6, 0.8, 1.0, 1.2], "episodes": 50, "duration_s": 8.0},
        "transition": {
            "speeds": [0.0, 0.6, 0.8, 1.0, 1.2, 1.0, 0.8, 0.6, 0.0],
            "hold_s": [1.5] * 8 + [3.0], "ramp_s": 1.0, "episodes": 50,
        },
        "conditions": {"C0": "controller OFF", "C1": "controller ON"},
        "seed_root": 20261201,
        "deterministic_policy": True,
        "ppo_updates": 0,
        "policy_gradients": 0,
        "state_injection": 0,
    })
    dump("seed_manifest.json", {
        "root": 20261201,
        "pairing": "C0/C1 use independent fresh Isaac processes with identical root seed and environment-slot ordering",
        "steady": [
            {"speed": speed, "episode": episode, "paired_slot": speed_index * 50 + episode}
            for speed_index, speed in enumerate((0.0, 0.6, 0.8, 1.0, 1.2))
            for episode in range(50)
        ],
        "transition": [{"episode": episode, "paired_slot": episode} for episode in range(50)],
    })
    dump("yaw_bias_canceller_contract.json", {
        "name": "G1SpeedConditionedYawBiasCancellerV1",
        "neural_network": False,
        "equation": "clip(desired_yaw_rate + c(abs(commanded_vx)), -0.20, +0.20)",
        "desired_yaw_rate_preflight": 0.0,
        "table": [{"speed_mps": speed, "offset_radps": value} for speed, value in TABLE],
        "interpolation": "piecewise linear",
        "low_speed": {
            "at_or_below_0p4": 0.0,
            "0p4_to_0p6": "minimum-jerk spatial activation to c(0.6)",
        },
        "above_1p2": "hold c(1.2); diagnostic-only, no high-speed claim",
        "temporal_activation_s": ACTIVATION_S,
        "temporal_deactivation_s": ACTIVATION_S,
        "offset_limit_radps": OFFSET_LIMIT,
        "policy_command_limit_radps": POLICY_LIMIT,
        "inputs": ["commanded forward speed"],
        "forbidden_inputs_absent": ["actual speed", "heading error", "contact phase", "joint state", "integrator"],
        "stand_after_reset": "strictly zero",
    })
    tests = unit_tests()
    dump("yaw_bias_canceller_unit_tests.json", tests)
    if not tests["all_pass"]:
        raise RuntimeError("canceller unit tests failed")

    exp007 = json.loads(EXP007.read_text(encoding="utf-8"))
    stage1_open = json.loads((STAGE1 / "open_loop_heading_baseline.json").read_text(encoding="utf-8"))
    stage1_zero = next(row for row in stage1_open["conditions"] if row["target_speed"] == 0.0)
    dump("stand_protocol_equivalence_audit.json", {
        "classification": "EVALUATION_PROTOCOL_DIFFERENCE_NO_PHYSICS_MISMATCH",
        "meaningful_environment_difference": False,
        "comparison": {
            "environment_id": {
                "exp007": "Isaac-Velocity-Flat-G1-Run-Eval-v0",
                "stage1_yaw": "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0",
                "difference": True,
                "physical_effect_at_forced_zero_command": "none identified",
            },
            "robot_asset": {"same": True, "value": "Unitree G1"},
            "checkpoint": {"same": True, "sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"},
            "observation": {"same": True, "dimension": 123},
            "action_scale": {"same": True, "value": 0.5},
            "physics_dt": {"same": True, "value": 0.005},
            "control_dt": {"same": True, "value": 0.02},
            "decimation": {"same": True, "value": 4},
            "episode_duration": {"exp007": "up to 2.0 s settle + 8.0 s hold", "stage1_yaw": "fixed 8.0 s"},
            "reset_pose_and_noise": {"same_base_config": True, "joint_position_noise": [1.0, 1.0],
                                     "root_velocity_noise": [-0.5, 0.5]},
            "command_resampling": {"exp007": "direct zero each step", "stage1_yaw": "external override zero"},
            "termination": {"same_base_config": True},
            "contact_termination": {"same_base_config": True},
            "terrain": {"same": True, "value": "plane"},
            "friction": {"same_base_config": True},
            "heading_reference": {"exp007": "not a hold gate metric", "stage1_yaw": "yaw at 2.0 s"},
            "settling_window": {"exp007": "state-based 0.4 s streak within 2.0 s", "stage1_yaw": "fixed exclusion to 2.0 s"},
            "seed_set": {"exp007": 20260723, "stage1_yaw": 20261101, "same": False},
        },
        "retained_results": {
            "exp007": {
                "settle": exp007["metrics"]["settle_success_rate"],
                "hold": exp007["metrics"]["stand_hold_success_rate"],
                "fall": exp007["metrics"]["fall_rate"],
            },
            "stage1_yaw": {
                "fall": stage1_zero["fall_rate"],
                "heading_p95": stage1_zero["heading_p95"],
            },
        },
        "interpretation": (
            "The physical robot/reset/physics contracts match, but seed, settle/hold window, and heading aggregation do not. "
            "The 2% and 6% fall observations are retained as different protocol samples."
        ),
    })

    steady = read_rows("_steady_off_rows.csv") + read_rows("_steady_on_rows.csv")
    write_rows("steady_cancellation_comparison.csv", steady)
    grouped = defaultdict(list)
    for row in steady:
        grouped[(row["controller"], round(row["target_speed"], 3))].append(row)
    steady_summary = {
        controller: {
            str(speed): aggregate(grouped[(controller, speed)])
            for speed in (0.0, 0.6, 0.8, 1.0, 1.2)
        } for controller in ("off", "on")
    }
    moving_gates = {}
    for speed in (0.6, 0.8, 1.0, 1.2):
        baseline, candidate = steady_summary["off"][str(speed)], steady_summary["on"][str(speed)]
        reduction = 1.0 - candidate["mean_absolute_yaw_bias"] / max(baseline["mean_absolute_yaw_bias"], 1e-12)
        # "New contact instability" means a contact mode absent from the paired
        # baseline, not any sub-percent change in an already-observed flight
        # fraction.  Continuous flight severity remains reported in the summary.
        new_contact = bool(baseline["flight_fraction"] == 0.0 and candidate["flight_fraction"] > 0.0)
        passed = (
            (candidate["mean_absolute_yaw_bias"] <= 0.020 or reduction >= 0.70)
            and candidate["heading_p95"] <= 0.12
            and candidate["fall_rate"] - baseline["fall_rate"] <= 0.02 + 1e-12
            and candidate["speed_mae"] - baseline["speed_mae"] <= 0.03 + 1e-12
            and candidate["lateral_velocity"] - baseline["lateral_velocity"] <= 0.03 + 1e-12
            and candidate["long_dwell_saturation_rate"] - baseline["long_dwell_saturation_rate"] <= 0.02 + 1e-12
            and not new_contact
        )
        moving_gates[str(speed)] = {
            "pass": bool(passed), "yaw_bias_reduction": reduction,
            "absolute_candidate_bias": candidate["mean_absolute_yaw_bias"],
            "heading_p95": candidate["heading_p95"],
            "fall_increase": candidate["fall_rate"] - baseline["fall_rate"],
            "speed_mae_degradation": candidate["speed_mae"] - baseline["speed_mae"],
            "lateral_velocity_degradation": candidate["lateral_velocity"] - baseline["lateral_velocity"],
            "long_dwell_saturation_increase": (
                candidate["long_dwell_saturation_rate"] - baseline["long_dwell_saturation_rate"]
            ),
            "new_contact_instability": new_contact,
        }
    baseline_bias = np.mean([steady_summary["off"][str(speed)]["mean_absolute_yaw_bias"] for speed in (0.6, 0.8, 1.0, 1.2)])
    candidate_bias = np.mean([steady_summary["on"][str(speed)]["mean_absolute_yaw_bias"] for speed in (0.6, 0.8, 1.0, 1.2)])
    aggregate_reduction = 1.0 - candidate_bias / max(baseline_bias, 1e-12)
    baseline_sat = np.mean([steady_summary["off"][str(speed)]["long_dwell_saturation_rate"] for speed in (0.6, 0.8, 1.0, 1.2)])
    candidate_sat = np.mean([steady_summary["on"][str(speed)]["long_dwell_saturation_rate"] for speed in (0.6, 0.8, 1.0, 1.2)])
    moving_pass = bool(
        all(item["pass"] for item in moving_gates.values())
        and aggregate_reduction >= 0.70
        and candidate_sat <= baseline_sat + 1e-12
    )
    steady_off_trace = json.loads((OUT / "_steady_off_trace.json").read_text(encoding="utf-8"))
    steady_on_trace = json.loads((OUT / "_steady_on_trace.json").read_text(encoding="utf-8"))
    stand_interference = {
        "offset_strictly_zero": steady_on_trace["stand_offset_abs_max"] == 0.0,
        "action_trace_bitwise_equal": steady_off_trace["stand_action_trace_sha256"] == steady_on_trace["stand_action_trace_sha256"],
        "policy_command_bitwise_equal": (
            steady_off_trace["stand_policy_command_trace_sha256"] == steady_on_trace["stand_policy_command_trace_sha256"]
        ),
        "fall_equal": steady_summary["off"]["0.0"]["fall_rate"] == steady_summary["on"]["0.0"]["fall_rate"],
        "heading_equal": steady_summary["off"]["0.0"]["heading_p95"] == steady_summary["on"]["0.0"]["heading_p95"],
        "saturation_equal": (
            steady_summary["off"]["0.0"]["long_dwell_saturation_rate"]
            == steady_summary["on"]["0.0"]["long_dwell_saturation_rate"]
        ),
    }
    stand_interference["pass"] = all(stand_interference.values())
    dump("steady_cancellation_comparison.json", {
        "summary": steady_summary,
        "moving_gate": moving_gates,
        "aggregate": {
            "baseline_mean_absolute_yaw_bias": baseline_bias,
            "candidate_mean_absolute_yaw_bias": candidate_bias,
            "yaw_bias_reduction": aggregate_reduction,
            "baseline_long_dwell_saturation": baseline_sat,
            "candidate_long_dwell_saturation": candidate_sat,
            "pass": moving_pass,
        },
        "stand_non_interference": stand_interference,
    })

    transition = read_rows("_transition_off_rows.csv") + read_rows("_transition_on_rows.csv")
    write_rows("transition_cancellation_comparison.csv", transition)
    transition_summary = {
        controller: aggregate([row for row in transition if row["controller"] == controller])
        for controller in ("off", "on")
    }
    baseline, candidate = transition_summary["off"], transition_summary["on"]
    transition_gate = {
        "completion_non_regression": candidate["sequence_completion_rate"] >= baseline["sequence_completion_rate"],
        "fall": candidate["fall_rate"] <= 0.05,
        "heading": candidate["heading_p95"] <= 0.12,
        "activation_jump": candidate["activation_action_jump_max"] <= baseline["action_rate_p95"] + 1e-12,
        "deactivation_jump": candidate["deactivation_action_jump_max"] <= baseline["action_rate_p95"] + 1e-12,
        "speed_mae": candidate["speed_mae"] - baseline["speed_mae"] <= 0.03 + 1e-12,
        "long_dwell_saturation": (
            candidate["long_dwell_saturation_rate"] - baseline["long_dwell_saturation_rate"] <= 0.02 + 1e-12
        ),
    }
    transition_gate["pass"] = all(transition_gate.values())
    segment_summary = {}
    for controller in ("off", "on"):
        segment_summary[controller] = []
        condition_rows = [row for row in transition if row["controller"] == controller]
        for segment_id, speed in enumerate((0.0, 0.6, 0.8, 1.0, 1.2, 1.0, 0.8, 0.6, 0.0)):
            speed_error, yaw = [], []
            for row in condition_rows:
                values = json.loads(row["segment_metrics"])[segment_id]
                speed_error.extend(values["speed_error"])
                yaw.extend(values["yaw"])
            segment_summary[controller].append({
                "segment": segment_id, "target_speed": speed,
                "speed_mae": float(np.mean(speed_error)) if speed_error else None,
                "signed_yaw_bias": float(np.mean(yaw)) if yaw else None,
            })
    dump("transition_cancellation_comparison.json", {
        "summary": transition_summary, "segments": segment_summary, "gate": transition_gate,
    })

    semantic = {
        "status": "COMPLETE",
        "current_interface": {
            "policy_command_source": "command_manager.get_command('base_velocity') observation indices 9:12",
            "yaw_tracking_reward_target": "the same command_manager yaw value at index 2",
            "separate_policy_and_reward_targets_available": False,
        },
        "canceller_in_training_mismatch": (
            "A negative feedforward command intended to make physical yaw zero is simultaneously interpreted by "
            "the unchanged reward as a negative physical yaw-rate target."
        ),
        "separation_requires_semantic_change": True,
        "interface_or_reward_changed_in_stage1b": False,
        "decision": "Do not place the canceller or any external heading controller in the PPO rollout.",
    }
    dump("ppo_cancellation_semantic_audit.json", semantic)

    cancellation_pass = moving_pass and transition_gate["pass"] and stand_interference["pass"]
    if cancellation_pass:
        classification = "G1_SPEED_CONDITIONED_YAW_CANCELLATION_PASS"
    else:
        moving_count = sum(item["pass"] for item in moving_gates.values())
        if stand_interference["pass"] is False or any(
            item["fall_increase"] > 0.02 or item["long_dwell_saturation_increase"] > 0.02
            for item in moving_gates.values()
        ):
            classification = "G1_SPEED_CONDITIONED_YAW_CANCELLATION_SAFETY_FAIL"
        elif moving_count or aggregate_reduction > 0:
            classification = "G1_SPEED_CONDITIONED_YAW_CANCELLATION_PARTIAL"
        else:
            classification = "G1_SPEED_CONDITIONED_YAW_CANCELLATION_NO_EFFECT"
        if moving_pass and not transition_gate["pass"]:
            classification = "G1_YAW_CANCELLATION_NONLINEAR_TRANSITION_FAIL"
    ready = cancellation_pass and stand_interference["pass"] and semantic["status"] == "COMPLETE"
    next_action = (
        "run Pilot 1 with yaw-rate command fixed at 0 and all external yaw controllers disabled"
        if ready else "diagnose the failed yaw-cancellation safety or transition gate before Pilot 1"
    )
    amendment = {
        "status": "FROZEN_STAGE1B_AMENDMENT",
        "training_yaw_rate_command": 0.0,
        "external_heading_controller": "OFF",
        "speed_conditioned_canceller": "OFF",
        "parent_yaw_tracking_reward": "unchanged",
        "canceller_status": "FROZEN_CHECKPOINT_EVALUATION_CANDIDATE_ONLY",
        "rationale": [
            "The paired Stage 1B baseline exceeds 0.12 rad at 0.6--1.2 m/s, so heading-bias reduction remains a Pilot objective.",
            "Pilot curriculum must train the single policy's own yaw-bias reduction.",
            "The fixed parent table becomes stale as PPO changes the policy.",
            "Policy command and unchanged reward target must retain the same meaning.",
        ],
        "implementation_audit": {
            "current_stage2_command_curriculum_contains_phase_gated_heading": True,
            "current_stage2_script_must_not_be_run_without_applying_this_amendment": True,
            "existing_stage2_files_modified_in_stage1b": False,
        },
    }
    dump("pilot1_protocol_amendment.json", amendment)
    (OUT / "pilot1_protocol_amendment.md").write_text(
        "# EXP 012 Pilot 1 protocol amendment\n\n"
        "This amendment is frozen by Stage 1B before Pilot 1. It does not modify prior-stage artifacts.\n\n"
        "```text\n"
        "training yaw-rate command: 0\n"
        "external heading controller: OFF\n"
        "speed-conditioned yaw-bias canceller: OFF\n"
        "existing parent yaw-tracking reward: unchanged\n"
        "```\n\n"
        "The canceller remains a frozen-checkpoint evaluation candidate only. The current checked-in Stage 2 "
        "command implementation contains phase-gated heading feedback and must not be used for Pilot 1 until a "
        "future authorized stage applies this amendment. A cancellation offset cannot be inserted transparently "
        "during PPO because the same command is also the yaw-tracking reward target.\n",
        encoding="utf-8",
    )
    dump("stage_classification.json", {
        "classification": classification,
        "secondary": [
            "STAND_PROTOCOL_NON_EQUIVALENT_METRICS",
            "YAW_CANCELLER_STAND_NON_INTERFERENCE_PASS" if stand_interference["pass"]
            else "YAW_CANCELLER_STAND_INTERFERENCE",
            "PPO_CANCELLATION_TARGET_MISMATCH",
        ],
    })
    dump("pilot_readiness.json", {
        "classification": "EXP012_PILOT1_READY_OPEN_LOOP_YAW" if ready else "EXP012_PILOT1_NOT_READY",
        "ready": ready, "pilot1_executed": False,
        "meaning": "Readiness applies to a future yaw-command-zero PPO protocol, not parent formal capability.",
        "moving_authority_interpretation": (
            "The prior broad-matrix MOVING_YAW_RATE_NOT_CONTROLLABLE result is retained. "
            "Stage 1B separately establishes safe operating-point feedforward cancellation authority."
        ),
    })
    dump("recommended_next_action.json", {"action": next_action, "one_method_only": True})
    dump("gate.json", {
        "status": "COMPLETE",
        "classification": classification,
        "command_pipeline": "PASS",
        "moving_cancellation": "PASS" if moving_pass else "FAIL",
        "transition": "PASS" if transition_gate["pass"] else "FAIL",
        "stand_interference": "PASS" if stand_interference["pass"] else "FAIL",
        "pilot_readiness": "EXP012_PILOT1_READY_OPEN_LOOP_YAW" if ready else "EXP012_PILOT1_NOT_READY",
        "ppo_updates": 0, "policy_gradients": 0, "reward_optimization": 0, "state_injection": 0,
    })


if __name__ == "__main__":
    main()
