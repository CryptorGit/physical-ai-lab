"""Prepare immutable contracts and source audits for W1B-D4."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d4_dynamic_endpoint_window_parity_preflight"
)
EXP = ROOT / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout


starting_head = git("rev-parse", "HEAD").strip()
starting_status = git("status", "--short").splitlines()
starting_log = git("log", "--oneline", "--decorate", "-25").splitlines()
dump("stage_reference.json", {
    "stage": "W1B-D4",
    "title": "dynamic yaw endpoint-window evaluator parity preflight",
    "starting_head": starting_head,
    "reported_starting_head": "1c9b7c27636e0cf5be0f841390ad3672876320ce",
    "head_matches_reported": starting_head == "1c9b7c27636e0cf5be0f841390ad3672876320ce",
    "starting_status_short": starting_status,
    "starting_log_oneline_25": starting_log,
    "policy_checkpoint": "W1B-R2 iteration 200",
    "policy_sha256": "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",
    "calibration": "MonotonicPositiveYawCalibrationV1",
})
dump("protocol.json", {
    "training_updates": 0,
    "new_policy_checkpoints": 0,
    "production_evaluator_changes": 0,
    "production_command_changes": 0,
    "formal_gate_changes": 0,
    "paired_episodes_per_condition": 100,
    "static_duration_s": 8.0,
    "dynamic_pre_hold_s": 4.0,
    "dynamic_ramp_s": 2.0,
    "dynamic_final_hold_s": 6.0,
    "candidate_windows_preregistered": [
        "W0_CURRENT", "W1_FINAL_HOLD_ALL", "W2_FINAL_HOLD_AFTER_0P25",
        "W3_FINAL_HOLD_AFTER_0P50", "W4_FINAL_HOLD_AFTER_1P00",
        "W5_LAST_1_SECOND", "W6_LAST_2_SECONDS", "W7_LAST_3_SECONDS",
    ],
    "candidate_metrics_preregistered": [
        "M0_INSTANT_SIGN_FRACTION", "M1_STATIC_ENDPOINT_EQUIVALENT",
        "M2_WINDOW_MEAN_SIGN_FRACTION", "M3_GAIT_CYCLE_MEAN_SIGN",
        "M4_LOW_PASS_SIGN_FRACTION_2HZ", "M4_LOW_PASS_SIGN_FRACTION_4HZ",
    ],
})

source = EXP / "scripts"
dump("static_dynamic_evaluator_source_locations.json", {
    "locations": [
        {
            "file": str(source / "evaluate_w1b_c1.py"),
            "class_function": "run_suite / static and transition accumulation",
            "line_range": "220-430",
            "input": "physical yaw target plus calibrated actor yaw",
            "output": "episode and condition metrics",
            "state": "fresh DirectionalBaseline environment",
            "time_origin": "episode reset",
            "window": "whole episode",
            "aggregation": "mean yaw error and aggregate success fractions",
            "threshold": "static pure MAE 0.15; moving MAE 0.20",
        },
        {
            "file": str(source / "evaluate_w1b_d3.py"),
            "class_function": "command_at / main finalization",
            "line_range": "132-173, 326-394",
            "input": "physical minimum-jerk target and calibrated command",
            "output": "whole/final instantaneous sign fractions",
            "state": "fresh DirectionalBaseline environment",
            "time_origin": "episode reset; transition starts at 4 s",
            "window": "whole episode and final hold",
            "aggregation": "per-step instantaneous sign Boolean mean",
            "threshold": "sign fraction >=0.95",
        },
        {
            "file": str(source / "finalize_w1b_d3.py"),
            "class_function": "random segment decomposition",
            "line_range": "218-275",
            "input": "random-command episode traces",
            "output": "segment-level yaw and translation metrics",
            "state": "offline",
            "time_origin": "command segment start",
            "window": "entire segment",
            "aggregation": "segment success then episode interpretation",
            "threshold": "sign fraction >=0.95 and yaw MAE <=0.20",
        },
        {
            "file": str(source / "evaluate_w1b_d4.py"),
            "class_function": "diagnostic paired evaluator",
            "line_range": "diagnostic-only",
            "input": "same final physical command for static/dynamic matched seed",
            "output": "time-step traces and paired endpoint metrics",
            "state": "fresh DirectionalBaseline environment",
            "time_origin": "reset, ramp start, and final-hold start recorded separately",
            "window": "all preregistered candidates",
            "aggregation": "diagnostic-only; no production gate mutation",
            "threshold": "existing static thresholds only",
        },
    ]
})
dump("static_dynamic_evaluator_contract_audit.json", {
    "static": {
        "window": "whole constant-command episode",
        "metric": "mean yaw sign plus yaw-rate MAE",
        "episode_aggregation": "physical tracking and safety conjunction",
    },
    "dynamic": {
        "window": "whole episode in C1; D3 also recorded final hold",
        "metric": "instantaneous target-sign fraction",
        "episode_aggregation": "single sign fraction mixes acquisition and retention",
    },
    "material_differences": [
        "ramp and pre-hold samples enter the C1 dynamic sign fraction",
        "static uses mean/MAE while dynamic uses instantaneous sign",
        "acquisition and endpoint retention are conflated",
        "gait-periodic sign crossings can fail dynamic while mean endpoint tracking passes",
    ],
    "classification": "MATERIAL_DIFFERENCE",
    "production_code_changed": False,
})
dump("dynamic_episode_timeline_contract.json", {
    "time_step_s": 0.02,
    "phases": [
        {"phase": "PRE_HOLD", "start_s": 0.0, "end_s": 4.0, "physical_target": "initial endpoint", "ramp_progress": 0.0},
        {"phase": "RAMP", "start_s": 4.0, "end_s": 6.0, "physical_target": "minimum-jerk initial-to-final", "ramp_progress": "0..1"},
        {"phase": "POST_RAMP_SETTLING", "start_s": 6.0, "end_s": 6.25, "physical_target": "final", "ramp_progress": 1.0},
        {"phase": "FINAL_HOLD_EARLY", "start_s": 6.25, "end_s": 7.0, "physical_target": "final", "ramp_progress": 1.0},
        {"phase": "FINAL_HOLD_STEADY", "start_s": 7.0, "end_s": 12.0, "physical_target": "final", "ramp_progress": 1.0},
        {"phase": "EPISODE_END", "start_s": 12.0, "end_s": 12.0, "physical_target": "final", "ramp_progress": 1.0},
    ],
    "actor_input": "calibrate_yaw(physical_target) after minimum-jerk",
    "gait_cycle_index": "incremented on left-foot contact rising edge",
    "contact_phase": "left/right/double/flight from ankle-roll contact sensors",
    "current_sign_fraction_scope": "C1 entire episode; includes PRE_HOLD, RAMP, zero crossing, and FINAL_HOLD",
})
print(json.dumps({"prepared": str(OUT), "starting_head": starting_head}, indent=2))
