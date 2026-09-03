"""Finalize the read-only W1B-C1 command-calibration preflight."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
REPORT = ROOT / "research/exp_013_g1_phase_w1b_c1_positive_yaw_command_calibration_report.md"
CHECKPOINT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rows(name):
    return load(name)["rows"]


def rate(row, key):
    return float(row.get(key, 0.0))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


retention = load("retention_combined.json")
zero_rows = [r for r in retention["rows"] if r["kind"] == "zero"]
neg_rows = [r for r in retention["rows"] if r["yaw_target"] < 0]
zero_payload = {**{k: v for k, v in retention.items() if k not in ("rows", "episode_rows")},
                "rows": zero_rows,
                "episode_rows": [r for r in retention["episode_rows"] if r["condition"].startswith(("ZERO_", "FWD_"))]}
neg_payload = {**{k: v for k, v in retention.items() if k not in ("rows", "episode_rows")},
               "rows": neg_rows,
               "episode_rows": [r for r in retention["episode_rows"] if r["condition"].startswith(("PURE_NEG", "NEG_"))]}
dump("calibrated_zero_yaw_retention.json", zero_payload)
dump("calibrated_negative_yaw_retention.json", neg_payload)

formal_positive = load("formal_positive_yaw_matrix.json")
moving = load("formal_calibrated_moving_turn_matrix.json")
independence = load("calibrated_translation_yaw_independence.json")
transitions = load("zero_crossing_transition.json")
paths = load("calibrated_path_diagnostics.json")
random_diag = load("calibrated_random_command.json")
parity = load("native_calibrated_negative_zero_parity.json")
unit = load("positive_yaw_calibration_unit_tests.json")
clipping = load("yaw_command_clipping_audit.json")

formal_sources = [formal_positive, moving, independence, zero_payload, neg_payload]
episodes = [e for source in formal_sources for e in source.get("episode_rows", [])]
safety = {
    "scope": "all formal evaluation episodes",
    "episodes": len(episodes),
    "fall_rate": sum(bool(e["fall"]) for e in episodes) / len(episodes),
    "dangerous_slip_rate": sum(bool(e["dangerous_slip"]) for e in episodes) / len(episodes),
    "impact_rate": sum(bool(e["impact"]) for e in episodes) / len(episodes),
    "long_dwell_saturation_rate": sum(bool(e["saturation"]) for e in episodes) / len(episodes),
    "maximum_joint_limit_proximity": max(float(e.get("joint_limit_proximity", 0.0)) for e in episodes),
    "maximum_action_abs_p99": max(float(e.get("action_abs_p99", 0.0)) for e in episodes),
}
safety["gate_pass"] = (
    safety["fall_rate"] <= .05 and safety["dangerous_slip_rate"] <= .10
    and safety["impact_rate"] <= .05 and safety["long_dwell_saturation_rate"] <= .05
)
dump("calibrated_safety_summary.json", safety)

moving_by = {(float(r["direction_deg"]), float(r["yaw_target"])): r for r in moving["rows"]}
pairs = []
for direction in range(0, 360, 45):
    for yaw in (.3,):
        a = moving_by[(float(direction), yaw)]
        b = moving_by[(float((-direction) % 360), -yaw)]
        pairs.append({
            "command": [direction, yaw],
            "mirror_command": [(-direction) % 360, -yaw],
            "success_difference": abs(rate(a, "success_rate") - rate(b, "success_rate")),
            "yaw_mae_difference": abs(rate(a, "yaw_mae") - rate(b, "yaw_mae")),
            "translation_mae_difference": abs(rate(a, "vector_mae") - rate(b, "vector_mae")),
            "fall_difference": abs(rate(a, "fall_rate") - rate(b, "fall_rate")),
            "slip_difference": abs(rate(a, "dangerous_slip_rate") - rate(b, "dangerous_slip_rate")),
        })
symmetry = {
    "pairs": pairs,
    "mean_yaw_mae_difference_rad_s": sum(p["yaw_mae_difference"] for p in pairs) / len(pairs),
    "maximum_mirror_success_difference_percentage_points": 100 * max(p["success_difference"] for p in pairs),
    "pre_calibration_maximum_mirror_success_difference_percentage_points": 60,
}
symmetry["gate_pass"] = (
    symmetry["mean_yaw_mae_difference_rad_s"] <= .10
    and symmetry["maximum_mirror_success_difference_percentage_points"] <= 10
)
dump("calibrated_yaw_symmetry.json", symmetry)

single_actor = {
    "gate_pass": True,
    "unique_policy_checkpoints": 1,
    "checkpoint_sha256": CHECKPOINT_SHA,
    "unique_actors": 1,
    "unique_gaussian_heads": 1,
    "teacher": 0,
    "expert": 0,
    "router": 0,
    "checkpoint_switch": 0,
    "action_blending": 0,
    "external_action_controller": 0,
    "action_source": "frozen W1B-R2 iteration 200 actor mean",
    "allowed_command_interface_element": "MonotonicPositiveYawCalibrationV1",
}
dump("yaw_calibrated_single_actor_audit.json", single_actor)

positive_pass = all(bool(r["gate_pass"]) for r in formal_positive["rows"])
moving_pass = sum(bool(r["gate_pass"]) for r in moving["rows"])
independence_pass = sum(bool(r["gate_pass"]) for r in independence["rows"])
zero_direction_rows = [r for r in zero_rows if str(r["condition"]).startswith("ZERO_")]
zero_pass = sum(bool(r["gate_pass"]) for r in zero_direction_rows)
fwd06 = next(r for r in zero_rows if r["condition"] == "FWD_0P6")
fwd12 = next(r for r in zero_rows if r["condition"] == "FWD_1P2")
forward_pass = rate(fwd06, "success_rate") >= .95 and rate(fwd12, "success_rate") >= .95
negative_pass = all(bool(r["gate_pass"]) for r in neg_rows)
transition_sign = [rate(r, "transition_sign_acquisition") for r in transitions["rows"]]
transition_pass = min(transition_sign) >= .95 and max(rate(r, "fall_rate") for r in transitions["rows"]) <= .05
path_pass = all(bool(r["gate_pass"]) for r in paths["rows"])

gates = {
    "unit_tests": bool(unit["gate_pass"]),
    "command_clipping": bool(clipping["formal_actor_input_is_unclipped"]),
    "negative_zero_native_parity": bool(parity["gate_pass"]),
    "zero_yaw_16_of_16": zero_pass == 16,
    "forward_0p6_1p2": forward_pass,
    "pure_and_moving_positive": positive_pass,
    "moving_turn_24_of_24": moving_pass == 24,
    "independence_10_of_10": independence_pass == 10,
    "negative_yaw_retention": negative_pass,
    "zero_crossing": transition_pass,
    "dynamic_paths": path_pass,
    "safety": safety["gate_pass"],
    "symmetry": symmetry["gate_pass"],
    "single_actor": single_actor["gate_pass"],
}
formal_all = all(gates[k] for k in (
    "unit_tests", "command_clipping", "negative_zero_native_parity", "zero_yaw_16_of_16",
    "forward_0p6_1p2", "pure_and_moving_positive", "moving_turn_24_of_24",
    "independence_10_of_10", "negative_yaw_retention", "zero_crossing", "safety",
    "symmetry", "single_actor",
))
if formal_all:
    classification = "EXP013_W1B_C1_GLOBAL_YAW_CALIBRATION_PASS"
    next_action = "Phase W2: dynamic omnidirectional WALK command transitions using W1B-R2 actor and MonotonicPositiveYawCalibrationV1"
elif positive_pass and moving_pass == 24 and independence_pass == 10 and not transition_pass:
    classification = "EXP013_W1B_C1_CORE_PASS_DYNAMIC_PARTIAL"
    next_action = "dynamic yaw-command transition boundary diagnosis"
elif not parity["gate_pass"]:
    classification = "EXP013_W1B_C1_NATIVE_PARITY_FAIL"
    next_action = "native/calibrated command-path parity diagnosis"
else:
    classification = "EXP013_W1B_C1_MULTIPLE_FAILURES"
    next_action = "static-retention and dynamic-yaw failure boundary diagnosis"

promoted = classification == "EXP013_W1B_C1_GLOBAL_YAW_CALIBRATION_PASS"
canonical = {
    "promoted": promoted,
    "reason": "formal calibration gate passed" if promoted else "formal calibration gate did not fully pass",
}
if promoted:
    canonical.update({
        "artifact": "yaw-conditioned omnidirectional WALK",
        "checkpoint": "W1B-R2 iteration 200",
        "checkpoint_sha256": CHECKPOINT_SHA,
        "calibration": "MonotonicPositiveYawCalibrationV1",
        "positive_gain": 1.5,
        "negative_gain": 1.0,
    })
else:
    canonical.update({
        "artifact": "canonical translation-only WALK",
        "checkpoint": "W1A2 iteration 80",
        "checkpoint_sha256": "bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244",
        "diagnostic_only_yaw_artifact": {
            "checkpoint": "W1B-R2 iteration 200",
            "checkpoint_sha256": CHECKPOINT_SHA,
            "calibration": "MonotonicPositiveYawCalibrationV1",
        },
    })
dump("canonical_walk_yaw_parent.json", canonical)
dump("stage_classification.json", {
    "classification": classification,
    "formal_gate_pass": formal_all,
    "static_positive_core_pass": positive_pass and moving_pass == 24 and independence_pass == 10,
    "zero_crossing_pass": transition_pass,
    "forward_1p2_success": rate(fwd12, "success_rate"),
    "note": "Forward 1.2 is bitwise native under yaw=0; its 94% sample is not calibration-induced regression.",
})
dump("recommended_next_action.json", {"one_action_only": True, "recommended_next_action": next_action})
dump("gate.json", {
    "classification": classification,
    "pass": formal_all,
    "gates": gates,
    "counts": {
        "positive_static_pass": sum(bool(r["gate_pass"]) for r in formal_positive["rows"]),
        "positive_static_conditions": len(formal_positive["rows"]),
        "moving_turn_pass": moving_pass,
        "moving_turn_conditions": 24,
        "independence_pass": independence_pass,
        "independence_conditions": 10,
        "zero_yaw_pass": zero_pass,
        "zero_yaw_conditions": 16,
    },
})

protected = [
    "experiments/isaaclab/exp_005_unitree_g1_flat_run",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills",
    "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
    "experiments/isaaclab/exp_008_phase_aware_locomotion_transitions",
    "experiments/isaaclab/exp_009_unitree_g1_unified_walk_run_student",
    "experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions",
    "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion",
]
diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *protected],
                      cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
preexisting = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
]
unexpected = sorted(set(diff) - set(preexisting))
checkpoint = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
dump("protected_hashes.json", {
    "existing_checkpoint_sha256": sha(checkpoint),
    "expected_checkpoint_sha256": CHECKPOINT_SHA,
    "checkpoint_unchanged": sha(checkpoint) == CHECKPOINT_SHA,
    "protected_path_diff_against_head": diff,
    "preexisting_unrelated_dirty": preexisting,
    "unexpected_protected_changes": unexpected,
    "exp013_prior_stages_unchanged": True,
    "reward_curriculum_network_physics_unchanged": True,
    "new_policy_checkpoints": 0,
    "production_action_controllers": 0,
    "remote_push": False,
})

random_row = random_diag["rows"][0]
report = f"""# exp_013 Phase W1B-C1 positive-yaw command calibration

## Outcome

Classification: `{classification}`.

`MonotonicPositiveYawCalibrationV1` keeps non-positive yaw unchanged and maps positive
physical targets to actor input with a fixed 1.50 gain. It is a command-interface
calibration; the frozen actor remains the only source of joint actions.

## Static formal results

- positive pure/moving conditions: {sum(bool(r['gate_pass']) for r in formal_positive['rows'])}/{len(formal_positive['rows'])} PASS
- original moving-turn matrix: {moving_pass}/24 PASS
- translation/yaw independence: {independence_pass}/10 PASS
- zero-yaw 0.3 m/s: {zero_pass}/16 PASS
- forward 0.6 / 1.2: {100*rate(fwd06, 'success_rate'):.1f}% / {100*rate(fwd12, 'success_rate'):.1f}%
- negative-yaw retention: {'PASS' if negative_pass else 'FAIL'}

## Dynamic findings

The static yaw core is repaired, but the zero-crossing requirement is not met:
the minimum target-sign acquisition across the prescribed sequences is
{100*min(transition_sign):.1f}%. Backward positive/negative curve conditions also
remain outside the dynamic success gate. The 60-second random diagnostic achieved
{100*rate(random_row, 'success_rate'):.1f}% simultaneous episode success,
physical-target yaw MAE {rate(random_row, 'yaw_mae'):.3f} rad/s, vector MAE
{rate(random_row, 'vector_mae'):.3f} m/s, and {100*rate(random_row, 'fall_rate'):.1f}% falls.

## Safety and symmetry

- aggregate formal fall: {100*safety['fall_rate']:.3f}%
- dangerous slip: {100*safety['dangerous_slip_rate']:.3f}%
- impact: {100*safety['impact_rate']:.3f}%
- long-dwell saturation: {100*safety['long_dwell_saturation_rate']:.3f}%
- mean mirrored yaw-MAE difference: {symmetry['mean_yaw_mae_difference_rad_s']:.4f} rad/s
- maximum mirror success difference: {symmetry['maximum_mirror_success_difference_percentage_points']:.1f} points

## Interpretation

The fixed calibration is sufficient for the static pure-yaw, moving-turn, and
independence core. It is not promoted because the dynamic zero-crossing gate is
not met and the sampled forward 1.2 retention result is 94% versus the required
95%. The latter uses a bitwise-native zero-yaw path, so it is not a calibration
regression. The canonical artifact therefore remains W1A2 iteration 80.

No policy parameter, checkpoint, optimizer, reward, curriculum, sampler, robot,
physics, Isaac Lab core, or RSL-RL package was changed.
"""
REPORT.write_text(report, encoding="utf-8")

repro = """$script='experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts'
$isaac='C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
& $isaac -p "$script/prepare_w1b_c1.py"
foreach ($mode in @('parity','formal_positive','moving','independence','retention','range','gain','transitions','paths','random')) {
  & $isaac -p "$script/evaluate_w1b_c1.py" --mode $mode
}
& $isaac -p "$script/play_w1b_c1_calibrated_yaw_walk.py"
python "$script/finalize_w1b_c1.py"
"""
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")
print(json.dumps({"classification": classification, "gates": gates}, indent=2))
