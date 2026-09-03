"""Prepare immutable audits and reconstruct the C1 dynamic failures for W1B-D3."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d3_dynamic_yaw_transition_boundary_diagnosis"
)
C1 = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def lines(path, needles):
    text = path.read_text(encoding="utf-8").splitlines()
    return {needle: [i + 1 for i, line in enumerate(text) if needle in line] for needle in needles}


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()
log = subprocess.check_output(["git", "log", "--oneline", "--decorate", "-25"], cwd=ROOT, text=True).splitlines()
dump("stage_reference.json", {
    "stage": "Phase W1B-D3 dynamic yaw-command transition boundary diagnosis",
    "starting_head_reported": "4eee7bc04491bb7cb55bab3954a6d936ecb001fc",
    "starting_head_actual": head,
    "starting_head_matches": head == "4eee7bc04491bb7cb55bab3954a6d936ecb001fc",
    "starting_status_short": status,
    "starting_log_25": log,
    "checkpoint": "W1B-R2 iteration 200",
    "checkpoint_sha256": "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",
    "policy_updates": 0,
    "new_policy_checkpoints": 0,
    "production_command_calibration_changes": 0,
    "remote_push": False,
})
dump("protocol.json", {
    "policy": "frozen W1B-R2 iteration 200 actor mean",
    "calibration": "MonotonicPositiveYawCalibrationV1",
    "training": False,
    "diagnostic_command_profiles_only": True,
    "formal_gate_changed": False,
    "evaluation_task": "Isaac-Exp013-G1-DirectionalBaseline-v0",
    "observation_corruption": False,
    "push_external_force": False,
    "command_frame": "robot body",
})

c1_script = EXP / "scripts/evaluate_w1b_c1.py"
cal_script = EXP / "src/g1_omnidirectional/yaw_calibration.py"
source = {
    "evaluate_w1b_c1.py": lines(c1_script, [
        "def minjerk", "def command_at", "physical =", "calibrate_yaw(physical)",
        "command.external_override[:, 2]", "yawerr", '"sign":', "overshoot = max",
        "sign_changes =", "crossing_delay",
    ]),
    "yaw_calibration.py": lines(cal_script, ["def calibrate_yaw", "torch.where", "POSITIVE_GAIN"]),
}
dump("dynamic_yaw_command_source_locations.json", source)
dump("dynamic_yaw_command_pipeline_audit.json", {
    "current_pipeline": "P1",
    "sequence": [
        "physical yaw endpoints selected by schedule",
        "physical target minimum-jerk interpolation",
        "MonotonicPositiveYawCalibrationV1 applied independently at each step",
        "actor input clipped to [-1,+1] by calibration helper",
        "actor input copied to external_override and observation command indices",
        "metrics evaluated against physical target",
    ],
    "physical_target_ramp": "minimum-jerk, 2 seconds in C1",
    "positive_calibration_position": "after physical-target interpolation",
    "actor_input_derivative_at_zero": {
        "left_slope": 1.0,
        "right_slope": 1.5,
        "value_continuous": True,
        "first_derivative_continuous": False,
    },
    "command_clipping": [-1.0, 1.0],
    "observation_storage": "base velocity command external_override yaw column",
    "evaluation_target": "physical yaw target, never calibrated actor input",
})
dump("dynamic_yaw_evaluator_source_locations.json", {
    "source": str(c1_script.relative_to(ROOT)),
    "locations": source["evaluate_w1b_c1.py"],
})
dump("dynamic_yaw_evaluator_contract.json", {
    "ramp_start": "segment boundary at 6 s",
    "ramp_end": "2 s after segment boundary",
    "target_hold_start": "ramp end",
    "sign_acquisition_metric_start": "episode step 0",
    "sign_acquisition_window": "entire episode including initial state, ramps, zero targets, and holds",
    "settling_definition": "not explicitly implemented in C1",
    "overshoot": "max(max(|actual|-|instantaneous physical target|,0)) across full episode",
    "zero_target_handling": "counted sign-correct when |target|<1e-6; near-zero nonzero targets are not excluded",
    "episode_success": "safe AND whole-episode sign-correct fraction >=0.95",
    "condition_pass": "episode success >=0.90 and fall <=0.05",
    "gate_changed": False,
})
dump("static_dynamic_metric_difference.json", {
    "static": {
        "sign": "mean actual yaw and endpoint target must have matching sign",
        "yaw_mae": "whole-episode mean with constant target",
        "success": "translation + yaw MAE + sign + safety",
    },
    "dynamic": {
        "sign": "per-step correct-sign fraction over the entire episode",
        "yaw_mae": "whole-episode error to instantaneous ramp target",
        "success": "whole-episode sign fraction + safety only",
        "translation_and_yaw_mae": "recorded but not used by dynamic success",
    },
    "material_difference": True,
    "diagnostic_concern": (
        "whole-episode sign fraction penalizes physically unavoidable response during ramps and "
        "pre-transition history; it is not an endpoint settling-window metric"
    ),
})

existing = json.loads((C1 / "zero_crossing_transition.json").read_text(encoding="utf-8"))
reconstructed = []
for row in existing["episode_rows"]:
    failure = []
    if float(row["transition_sign_acquisition"]) < .95:
        failure.append("whole_episode_sign_acquisition_below_0p95")
    if row["fall"]:
        failure.append("fall")
    if row["dangerous_slip"]:
        failure.append("dangerous_slip")
    if row["impact"]:
        failure.append("impact")
    if row["saturation"]:
        failure.append("saturation")
    reconstructed.append({
        "condition": row["condition"],
        "episode": row["episode"],
        "transition_type": row["condition"].split("_")[1],
        "direction_deg": row["direction_deg"],
        "physical_target_trace": "not_recorded",
        "actor_input_trace": "not_recorded",
        "actual_yaw_trace": "not_recorded",
        "actual_translation_trace": "not_recorded",
        "sign_acquisition_fraction": row["transition_sign_acquisition"],
        "sign_acquisition_time": "not_recorded",
        "settling_time": "not_recorded",
        "zero_crossing_delay": row["zero_crossing_delay"],
        "peak_overshoot": row["overshoot"],
        "integrated_absolute_yaw_error": "not_recorded",
        "yaw_mae": row["yaw_mae"],
        "vector_mae": row["vector_mae"],
        "direction_error": row["direction_error"],
        "fall": row["fall"],
        "slip": row["dangerous_slip"],
        "contact_state": "not_recorded",
        "failure_reason": failure or ["none"],
    })
fields = list(reconstructed[0])
with (OUT / "existing_dynamic_yaw_failure_reconstruction.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fields)
    writer.writeheader()
    writer.writerows(reconstructed)
dump("existing_dynamic_yaw_failure_reconstruction.json", {
    "source": str((C1 / "zero_crossing_transition.json").relative_to(ROOT)),
    "limitations": "C1 stored episode aggregates but not time-series; missing fields are not inferred",
    "rows": reconstructed,
})
print(json.dumps({"head": head, "reconstructed_episodes": len(reconstructed), "pipeline": "P1"}, indent=2))
