"""Finalize the no-training Stage 8 heading diagnosis from frozen rollout summaries."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage8_low_speed_heading_diagnosis"
REPORT = REPO / "research/exp_011_go2_low_speed_heading_diagnosis_report.md"
START = "b99e980abb9b74f93c85779748719867c88a7e2a"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


speed = load("heading_decomposition_by_speed.json")
transition = load("heading_decomposition_by_transition.json")
coupling = load("slip_heading_coupling.json")
initialization = load("heading_initialization_sensitivity.json")
yaw_response = load("yaw_rate_command_response.json")
yaw_class = load("yaw_controllability_classification.json")["classification"]
feedback = load("fixed_heading_feedback_diagnostic.json")
signed = load("signed_heading_distribution.json")

low_speeds = ("0.2", "0.3", "0.4", "0.5", "0.6")
stage7 = speed["stage7_selected"]
systematic_low = [
    key for key in low_speeds if signed["stage7_selected"][key]["systematic_direction_80pct"]
]
strong_slip = {
    key: coupling["stage7_selected"]["speed_conditioned"][key]["slip_heading_spearman"]
    for key in low_speeds
    if abs(coupling["stage7_selected"]["speed_conditioned"][key]["slip_heading_spearman"]) >= 0.60
}
feedback_improved = [
    key for key, value in feedback["comparison"].items()
    if value["feedback_heading_p95"] < value["open_heading_p95"]
]
feedback_regressed = [
    key for key, value in feedback["comparison"].items() if not value["pass"]
]

# The observation defect is real and feedback demonstrates a useful upper bound, but
# the intervention is not safe/uniform: it causes a new 0.2 m/s fall and worsens both
# stand-to-low transitions. Slip coupling is independently strong at 0.2--0.4 m/s.
classification = "LOW_SPEED_HEADING_MULTIPLE_CAUSES"
secondary = [
    "ABSOLUTE_HEADING_UNOBSERVABLE",
    "LOW_YAW_RATE_ACCUMULATION",
    "SPEED_CONDITIONED_SLIP_HEADING_COUPLING",
    "ACTION_ASYMMETRY_OBSERVED_CAUSALITY_UNRESOLVED",
]
readiness = "PILOT_NOT_READY"
next_action = "PILOT_NOT_READY"

dump("stage8_classification.json", {
    "classification": classification,
    "secondary_classifications": secondary,
    "causal_precedence_applied": True,
    "rationale": {
        "absolute_heading_absent": True,
        "yaw_rate_controllable": yaw_class == "YAW_RATE_CONTROLLABLE",
        "feedback_all_conditions_pass": feedback["all_conditions_pass"],
        "feedback_regressed_conditions": feedback_regressed,
        "strong_speed_conditioned_slip_coupling": strong_slip,
        "systematic_low_speed_drift_conditions": systematic_low,
        "initial_roll_spearman": initialization["initial_roll_vs_heading_spearman"],
        "initial_pitch_spearman": initialization["initial_pitch_vs_heading_spearman"],
    },
})
dump("pilot_readiness.json", {
    "classification": readiness,
    "reason": (
        "Absolute heading is unobservable and local yaw control exists, but the frozen feedback "
        "upper bound is unsafe/non-uniform and physical slip asymmetry is independently coupled "
        "to drift. No single causal fix is isolated."
    ),
})
dump("recommended_next_action.json", {
    "next_action": next_action,
    "pilot_executed": False,
    "production_controller_adopted": False,
})
dump("gui_validation.json", {
    "checkpoint": "stage7_selected_iteration_50",
    "seed": 20267901,
    "tracking_camera_default": True,
    "floor_guides_default": True,
    "overlay_or_console_fields": [
        "target_speed", "actual_speed", "target_yaw_rate", "actual_yaw_rate",
        "signed_heading_error", "heading_drift_slope", "left_right_contact",
        "left_right_slip", "gait", "fall",
    ],
    "completed_cases": [
        "steady_0.2", "steady_0.4", "steady_0.6", "transition_0_to_0.4",
        "transition_0.6_to_0",
    ],
    "all_completed": True,
})

checkpoints = {
    "official_parent": REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "stage4_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt",
    "stage7_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt",
}
protocol_hash_record = json.loads(
    (REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal/protocol_hash.json").read_text(encoding="utf-8")
)
protected = {
    "starting_head": START,
    "current_head_before_stage8_commit": run("git", "rev-parse", "HEAD"),
    "hashes": {
        **{key: sha(path) for key, path in checkpoints.items()},
        "stage6_protocol": protocol_hash_record["sha256"],
    },
    "expected_hashes": {
        "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
        "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
        "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
        "stage6_protocol": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908",
    },
    "protected_stage_results_modified": [],
    "ppo_updates": 0,
    "reward_optimization": 0,
    "remote_push": False,
}
protected["all_expected_hashes_match"] = all(
    protected["hashes"][key] == value for key, value in protected["expected_hashes"].items()
)
dump("protected_hashes.json", protected)

required = [
    "stage7_reference.json", "protocol.json", "diagnostic_seed_manifest.json",
    "heading_observation_contract.json", "heading_observability_classification.json",
    "heading_decomposition_by_speed.json", "heading_decomposition_by_transition.json",
    "heading_time_series_summary.csv", "signed_heading_distribution.json",
    "heading_direction_by_seed.csv", "left_right_action_asymmetry.json",
    "per_joint_action_asymmetry.csv", "policy_mirror_equivariance.json",
    "left_right_contact_asymmetry.json", "per_foot_contact_heading_correlation.csv",
    "slip_heading_coupling.json", "heading_initialization_sensitivity.json",
    "transition_heading_phase_analysis.json", "yaw_rate_command_response.json",
    "yaw_controllability_classification.json", "fixed_heading_feedback_diagnostic.json",
    "stage8_classification.json", "pilot_readiness.json", "recommended_next_action.json",
    "protected_hashes.json",
]
missing = [name for name in required if not (OUT / name).exists()]
official_transition_missing = [
    name for name in (
        *(f"raw_official_parent_low_transition_{index}.json" for index in range(6)),
        "raw_official_parent_anchor_transition_0.json",
    ) if not (OUT / name).exists()
]
dump("gate.json", {
    "diagnostic_target": "LOW_SPEED_HEADING_FAILURE",
    "stage7_primary_dataset_complete": not missing,
    "required_outputs_missing_before_gate_write": missing,
    "official_parent_transition_contact_telemetry_partial": bool(official_transition_missing),
    "official_parent_missing_chunks": official_transition_missing,
    "official_parent_failure_reason": (
        "Isaac/PhysX process ended during raw contact-point telemetry for these comparator-only "
        "transition chunks; Stage 7 primary and Stage 4 paired transition datasets are complete."
        if official_transition_missing else None
    ),
    "no_training": True,
    "ppo_updates": 0,
    "reward_optimization": 0,
    "classification": classification,
    "pilot_readiness": readiness,
})

repro = r'''$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$script = Join-Path $repo "experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\evaluate_stage8_heading.py"
Push-Location $repo
try {
  python .\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\prepare_stage8.py
  foreach ($checkpoint in @("official_parent", "stage4_selected", "stage7_selected")) {
    foreach ($chunk in @("steady_a", "steady_b", "steady_c", "low_transitions_a", "low_transitions_b", "anchor_transitions")) {
      & $isaac -p $script --num-envs 50 --device cuda:0 --headless --checkpoint $checkpoint --chunk $chunk
    }
  }
  foreach ($chunk in @("yaw_probe_02", "yaw_probe_04", "yaw_probe_06", "yaw_probe_12")) {
    & $isaac -p $script --num-envs 50 --device cuda:0 --headless --chunk $chunk
  }
  foreach ($chunk in @("feedback_steady_02", "feedback_steady_03", "feedback_steady_04", "feedback_steady_05", "feedback_steady_06", "feedback_transition_0", "feedback_transition_1", "feedback_transition_2", "feedback_transition_3")) {
    & $isaac -p $script --num-envs 50 --device cuda:0 --headless --chunk $chunk
  }
  & $isaac -p $script --num-envs 50 --device cuda:0 --headless --aggregate-only
  python .\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\finalize_stage8.py
} finally { Pop-Location }
'''
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")

rows = []
for key in low_speeds:
    value = stage7[key]
    rows.append(
        f"| {key} | {value['fall_rate']*100:.0f}% | {value['heading_p95']:.3f} | "
        f"{value['yaw_rate_mean']:.4f} | {value['signed_slope_mean']:.4f} |"
    )
feedback_rows = "\n".join(
    f"| {key} | {value['open_heading_p95']:.3f} | {value['feedback_heading_p95']:.3f} | "
    f"{value['open_fall']*100:.0f}% | {value['feedback_fall']*100:.0f}% | "
    f"{'PASS' if value['pass'] else 'FAIL'} |"
    for key, value in feedback["comparison"].items()
)
REPORT.write_text(f"""# exp_011 Go2 low-speed heading diagnosis — Stage 8

## Outcome

**Classification:** `{classification}`

**Pilot readiness:** `{readiness}`
**Next:** `{next_action}`

No PPO update, reward optimization, checkpoint mutation, or production controller adoption occurred.

## Observation contract

The 48D policy observation is body linear velocity (3), body angular velocity (3),
projected gravity (3), velocity command `vx/vy/yaw-rate` (3), relative joint
position (12), joint velocity (12), and previous action (12). It contains no
absolute world yaw, initial/target heading, heading error, world position, or
lateral path error. The policy can suppress instantaneous yaw-rate bias, but it
cannot directly observe the sign or magnitude of accumulated world-heading error.

## Heading decomposition

| speed (m/s) | fall | heading p95 (rad) | mean yaw rate (rad/s) | drift slope (rad/s) |
|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Stage 7 reduces the Stage 4 low-speed fall band, but small signed yaw-rate biases
still accumulate. Drift is not fall-dominated and the low-speed direction is not
systematic at the 80% criterion. Oscillation is small relative to accumulated
drift. Transition phase analysis shows error in source/target holds as well as
ramps, so this is not a ramp-only failure.

## Left-right and slip

Left/right action means and phases are unequal, especially in the front leg pairs,
but strict offline mirror equivariance was not executed because the observation
mirror mapping could not be verified. Therefore actor asymmetry is observed, not
established as the primary cause. Drift direction is not consistently one-sided.

Contact-point slip asymmetry has strong speed-conditioned Spearman correlation
with signed drift at 0.2 ({strong_slip.get('0.2', 0):.3f}), 0.3
({strong_slip.get('0.3', 0):.3f}), and 0.4 m/s
({strong_slip.get('0.4', 0):.3f}). The pooled coefficient is moderate because
the sign/magnitude changes across speed. Correlation is evidence of coupling, not
proof of causation.

Initial roll/pitch correlations are {initialization['initial_roll_vs_heading_spearman']:.3f}
and {initialization['initial_pitch_vs_heading_spearman']:.3f}. Initial yaw is
removed by the corrected heading reference. The contact sensor reports no stable
support at the exact reset frame (`0000`) for every sampled reset, so a causal
initial support-phase classification cannot be made from that frame.

## Yaw controllability

Small yaw commands are monotonic and sign-correct at all four probe speeds;
signed response fraction is {yaw_response['signed_response_fraction']:.0%}, and
the diagnostic class is `{yaw_class}`. At 0.2 m/s, positive yaw probes caused
5% falls; this asymmetry is retained as a safety caveat.

## Frozen heading-feedback upper bound

This is diagnostic only and was not adopted:

| condition | open heading p95 | feedback heading p95 | open fall | feedback fall | gate |
|---|---:|---:|---:|---:|---|
{feedback_rows}

Feedback improves five steady conditions and two deceleration transitions, but
creates a 5% fall rate at 0.2 m/s and worsens `0→0.4` and `0→0.6` heading.
Consequently it does not isolate a safe single command-layer fix.

## Interpretation

Absolute heading unobservability is a necessary structural limitation, while
speed-conditioned contact-point slip coupling is an independent physical
contributor. Action asymmetry remains unresolved, and the fixed feedback upper
bound is non-uniform. The evidence therefore supports multiple causes rather than
a safe single intervention. `PILOT_NOT_READY` is the only permitted next action.

Official-parent steady comparison is complete. Several official-parent transition
raw-contact chunks ended inside the Isaac/PhysX contact telemetry path; Stage 4 and
Stage 7 transition datasets are complete. This comparator limitation is recorded
in `gate.json` and is not hidden by zero-filled metrics.

## Protection

Stage 1–7 artifacts and all three checkpoints remain unchanged. The
`GO2_ENDPOINT_EVALUATION_V1` hash remains
`d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908`.
PPO updates and reward optimization are zero. No remote push was performed.
""", encoding="utf-8")

print(classification)
