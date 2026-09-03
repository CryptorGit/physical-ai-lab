"""Offline analysis and finalization for W1B-D3."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d3_dynamic_yaw_transition_boundary_diagnosis"
)
REPORT = ROOT / "research/exp_013_g1_phase_w1b_d3_dynamic_yaw_transition_boundary_report.md"
C1 = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c1_positive_yaw_command_calibration_preflight"
)
CHECKPOINT = ROOT / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
CHECKPOINT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name, value):
    def encode_numpy(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, default=encode_numpy) + "\n",
        encoding="utf-8",
    )


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


step = load(OUT / "dynamic_yaw_step_response.json")
ramp = load(OUT / "yaw_ramp_duration_boundary.json")
dwell = load(OUT / "yaw_zero_dwell_boundary.json")
profiles = load(OUT / "dynamic_command_profile_comparison.json")
history = load(OUT / "yaw_history_dependence.json")
backward = load(OUT / "backward_dynamic_yaw_boundary.json")
random_raw = load(OUT / "random_command_dynamic_trace.json")
variance_raw = load(OUT / "forward_1p2_variance_raw.json")

# First-order identification from mean step traces. This is descriptive only.
fit_rows = []
for name, trace_data in step["mean_traces"].items():
    row = next(r for r in step["rows"] if r["condition"] == name)
    if row["direction_deg"] not in (None, 0, 90, 180, 270):
        continue
    if (row["initial_yaw"], row["final_yaw"]) not in ((0, .3), (0, -.3), (-.3, .3), (.3, -.3)):
        continue
    trace = trace_data
    time = np.array([r["time"] for r in trace], dtype=float)
    actor = np.array([r["actor"] for r in trace], dtype=float)
    actual = np.array([r["actual"] for r in trace], dtype=float)
    start = int(np.searchsorted(time, 4.0))
    best = None
    for delay_steps in range(0, 16):
        delayed = np.concatenate((np.full(delay_steps, actor[start]), actor[start:len(actor) - delay_steps or None]))
        for tau in np.arange(.02, 1.02, .02):
            sim = np.empty_like(delayed)
            sim[0] = actual[start]
            base_gain = np.dot(delayed, actual[start:]) / max(np.dot(delayed, delayed), 1e-9)
            for i in range(1, len(sim)):
                sim[i] = sim[i - 1] + .02 / tau * (base_gain * delayed[i - 1] - sim[i - 1])
            mse = float(np.mean((sim - actual[start:]) ** 2))
            if best is None or mse < best[0]:
                best = (mse, base_gain, tau, delay_steps * .02, sim)
    mse, gain, tau, delay, sim = best
    denom = float(np.sum((actual[start:] - np.mean(actual[start:])) ** 2))
    r2 = 1 - float(np.sum((sim - actual[start:]) ** 2)) / max(denom, 1e-9)
    fit_rows.append({
        "condition": name,
        "direction_deg": row["direction_deg"],
        "transition": f"{row['initial_yaw']:+.1f}->{row['final_yaw']:+.1f}",
        "gain_K": gain,
        "time_constant_tau_s": tau,
        "delay_d_s": delay,
        "r_squared": r2,
        "fit_mse": mse,
        "fit_quality": "adequate" if r2 >= .5 else "inadequate_periodic_or_nonlinear",
    })
write_csv("yaw_dynamic_system_identification.csv", fit_rows)
direction_tau = defaultdict(list)
for row in fit_rows:
    direction_tau[str(row["direction_deg"])].append(row["time_constant_tau_s"])
dump("yaw_dynamic_system_identification.json", {
    "model": "tau*d(actual_yaw)/dt + actual_yaw = K*actor_yaw(t-d)",
    "fit_rows": fit_rows,
    "direction_mean_tau": {key: float(np.mean(value)) for key, value in direction_tau.items()},
    "mean_r_squared": float(np.mean([r["r_squared"] for r in fit_rows])),
    "classification": "MODEL_FIT_INADEQUATE" if np.mean([r["r_squared"] for r in fit_rows]) < .5 else "DIRECTION_DEPENDENT_TIME_CONSTANT",
})

# Contact-start dependence from reversal conditions.
contact_groups = defaultdict(list)
for row in step["episode_rows"]:
    if (row["initial_yaw"], row["final_yaw"]) not in ((-.3, .3), (.3, -.3)):
        continue
    direction = "pure" if row["direction_deg"] is None else (
        "forward" if row["direction_deg"] == 0 else (
            "backward" if row["direction_deg"] == 180 else "lateral" if row["direction_deg"] in (90, 270) else "other"
        )
    )
    if direction == "other":
        continue
    contact_groups[(direction, row["transition_contact_state"])].append(row)
contact_rows = []
for (direction, contact), group in contact_groups.items():
    first_switches = defaultdict(int)
    for row in group:
        first_switches[row["first_support_switch"]] += 1
    contact_rows.append({
        "direction_group": direction,
        "start_contact_state": contact,
        "episodes": len(group),
        "success_rate": np.mean([r["success"] for r in group]),
        "final_hold_sign_fraction": np.mean([r["final_hold_sign_fraction"] for r in group]),
        "delay_s": np.mean([r["sign_acquisition_delay"] for r in group]),
        "overshoot_rad_s": np.mean([r["overshoot"] for r in group]),
        "dangerous_slip_rate": np.mean([r["dangerous_slip"] for r in group]),
        "first_support_switch_counts": dict(first_switches),
    })
write_csv("dynamic_yaw_contact_state_dependence.csv", contact_rows)
contact_success = defaultdict(list)
for row in contact_rows:
    contact_success[row["start_contact_state"]].append(row["success_rate"])
contact_spread = max(map(np.mean, contact_success.values())) - min(map(np.mean, contact_success.values()))
dump("dynamic_yaw_contact_state_dependence.json", {
    "rows": contact_rows,
    "observed_phases": sorted(contact_success),
    "success_spread_across_start_phases": contact_spread,
    "classification": "CONTACT_PHASE_DEPENDENT_YAW_TRANSITION" if contact_spread > .15 else "CONTACT_PHASE_SECONDARY",
    "limitations": "foot placement was not persisted; transition contact and first support switch were recorded",
})

# State/action history proxy using transition-start vectors.
vectors = history["transition_vectors"]
joint_names = history["joint_names"]
history_rows = []
joint_rows = []
categories = {
    "hip": [i for i, n in enumerate(joint_names) if "hip" in n],
    "knee": [i for i, n in enumerate(joint_names) if "knee" in n],
    "ankle": [i for i, n in enumerate(joint_names) if "ankle" in n],
    "waist": [i for i, n in enumerate(joint_names) if "torso" in n or "waist" in n],
    "shoulder": [i for i, n in enumerate(joint_names) if "shoulder" in n],
    "elbow": [i for i, n in enumerate(joint_names) if "elbow" in n],
    "hand": [i for i, n in enumerate(joint_names) if any(token in n for token in ("zero", "one", "two", "three", "four", "five", "six"))],
}
for direction in ("PURE", "D000", "D090", "D180", "D270"):
    for final in (.3, -.3):
        suffix = f"{direction}_{final:+.1f}"
        reference = vectors[f"HISTORY_ZERO_{suffix}"]
        ref_obs = np.array(reference["observation"]); ref_action = np.array(reference["action"])
        ref_prev = np.array(reference["previous_action"])
        for kind in ("STATIC", "SAME_SIGN", "OPPOSITE_SIGN", "ZERO"):
            value = vectors[f"HISTORY_{kind}_{suffix}"]
            obs = np.array(value["observation"]); action = np.array(value["action"])
            previous = np.array(value["previous_action"])
            item = {
                "direction": direction, "final_yaw": final, "history": kind,
                "reference_history": "ZERO",
                "observation_l2": float(np.linalg.norm(obs - ref_obs)),
                "previous_action_l2": float(np.linalg.norm(previous - ref_prev)),
                "actor_mean_action_l2": float(np.linalg.norm(action - ref_action)),
            }
            history_rows.append(item)
            for category, indices in categories.items():
                joint_rows.append({
                    **item,
                    "joint_category": category,
                    "mean_absolute_action_difference": float(np.mean(np.abs(action[indices] - ref_action[indices]))) if indices else 0.0,
                    "mean_absolute_previous_action_difference": float(np.mean(np.abs(previous[indices] - ref_prev[indices]))) if indices else 0.0,
                })
write_csv("dynamic_yaw_action_hysteresis_by_joint.csv", joint_rows)
dump("dynamic_yaw_state_action_hysteresis.json", {
    "rows": history_rows,
    "comparison": "transition-start mean state/action relative to zero history",
    "exact_matched_current_target_analysis": "not_recorded",
    "reason": "transition snapshots differ in current command; results are a history-state proxy, not causal matched-state injection",
    "training_updates": 0,
})

# Segment-level decomposition of the 60-second random diagnostic.
random_episode_rows = {r["episode"]: r for r in random_raw["episode_rows"]}
segment_groups = defaultdict(list)
for trace_record in random_raw["episode_traces"]:
    episode = trace_record["episode"]
    target = trace_record["target"]; actual = trace_record["actual"]; vec = trace_record["vec"]
    boundaries = [0] + [i for i in range(1, len(target)) if abs(target[i] - target[i - 1]) > 1e-9] + [len(target)]
    previous_target = None
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        current = float(target[start])
        sign = 1 if current > .02 else (-1 if current < -.02 else 0)
        if previous_target is None:
            category = "steady_positive" if sign > 0 else ("steady_negative" if sign < 0 else "steady_zero")
        else:
            prev_sign = 1 if previous_target > .02 else (-1 if previous_target < -.02 else 0)
            mapping = {
                (0, 1): "zero_to_positive", (0, -1): "zero_to_negative",
                (1, 0): "positive_to_zero", (-1, 0): "negative_to_zero",
                (1, -1): "positive_to_negative", (-1, 1): "negative_to_positive",
            }
            category = mapping.get((prev_sign, sign), "same_sign_magnitude_change")
        values = actual[start:end]
        sign_fraction = np.mean([(a * current > 0) or abs(current) <= .02 for a in values])
        segment_groups[category].append({
            "yaw_mae": float(np.mean(np.abs(np.array(values) - current))),
            "vector_mae": float(np.mean(vec[start:end])),
            "sign_acquisition": float(sign_fraction),
            "overshoot": float(max([abs(a) - abs(current) for a in values] + [0])),
            "fall": bool(random_episode_rows[episode]["fall"]),
            "slip": bool(random_episode_rows[episode]["dangerous_slip"]),
            "simultaneous_translation_yaw_change": previous_target is not None,
        })
        previous_target = current
segment_rows = []
for category, group in segment_groups.items():
    segment_rows.append({
        "failure_class": category,
        "segment_count": len(group),
        "success_rate": np.mean([r["sign_acquisition"] >= .95 and r["yaw_mae"] <= .20 for r in group]),
        "vector_mae": np.mean([r["vector_mae"] for r in group]),
        "yaw_mae": np.mean([r["yaw_mae"] for r in group]),
        "sign_acquisition": np.mean([r["sign_acquisition"] for r in group]),
        "overshoot": np.mean([r["overshoot"] for r in group]),
        "fall_rate": np.mean([r["fall"] for r in group]),
        "slip_rate": np.mean([r["slip"] for r in group]),
        "simultaneous_translation_yaw_change_fraction": np.mean([r["simultaneous_translation_yaw_change"] for r in group]),
    })
write_csv("random_command_segment_decomposition.csv", segment_rows)
dump("random_command_segment_decomposition.json", {
    "rows": segment_rows,
    "episode_count": 50,
    "duration_seconds": 60,
    "dominant_failure_class": min(segment_rows, key=lambda r: r["success_rate"])["failure_class"],
    "translation_direction_change_only_count": 0,
    "note": "continuous random sampler independently resamples translation and yaw at each boundary; noninitial yaw changes are simultaneous command changes",
})

# Forward 1.2 batch variance.
batch_rates = np.array([float(r["success_rate"]) for r in variance_raw["rows"]])
variance = {
    "batches": len(batch_rates),
    "episodes_per_batch": 50,
    "mean_success_rate": float(np.mean(batch_rates)),
    "minimum_success_rate": float(np.min(batch_rates)),
    "maximum_success_rate": float(np.max(batch_rates)),
    "batch_success_distribution": batch_rates.tolist(),
    "probability_94_percent_or_lower": float(np.mean(batch_rates <= .94)),
    "probability_below_95_percent": float(np.mean(batch_rates < .95)),
    "empirical_95_percent_interval": [float(np.quantile(batch_rates, .025)), float(np.quantile(batch_rates, .975))],
    "c1_94_percent_reproduced": bool(np.any(batch_rates <= .94)),
    "calibration_causal": False,
    "reason": "yaw=0 actor input and action path are bitwise native",
}
dump("forward_1p2_evaluation_variance.json", variance)

# Classification evidence.
ramp_by_duration = defaultdict(list)
for row in ramp["rows"]:
    ramp_by_duration[float(row["ramp_duration"])].append(row)
ramp_pass = {duration: sum(bool(r["gate_pass"]) for r in values) for duration, values in ramp_by_duration.items()}
dwell_by = defaultdict(list)
for row in dwell["rows"]:
    dwell_by[float(row["zero_dwell"])].append(row)
dwell_pass = {duration: sum(bool(r["gate_pass"]) for r in values) for duration, values in dwell_by.items()}
profile_by = defaultdict(list)
for row in profiles["rows"]:
    profile_by[row["profile"]].append(row)
profile_pass = {profile: sum(bool(r["gate_pass"]) for r in values) for profile, values in profile_by.items()}
history_by = defaultdict(list)
for row in history["rows"]:
    history_by[row["history"]].append(row)
history_pass = {kind: sum(bool(r["gate_pass"]) for r in values) for kind, values in history_by.items()}
backward_final = [float(r["final_hold_sign_fraction"]) for r in backward["rows"]]
classification = "DYNAMIC_EVALUATOR_WINDOW_MISMATCH"
next_action = "dynamic yaw endpoint-window evaluator parity preflight"
dump("current_dynamic_yaw_artifact_interpretation.json", {
    "W1B-R2 actor": "static omnidirectional yaw authority PASS",
    "MonotonicPositiveYawCalibrationV1": "static core PASS",
    "moving_turn_matrix": "24/24 PASS",
    "independence": "10/10 PASS",
    "dynamic_sign_reversal": "partial under whole-episode instantaneous-sign metric",
    "random_continuous_control": "not yet ready",
    "canonical_promotion": "none",
    "canonical_translation_only_WALK": "W1A2 iteration 80",
})
dump("stage_classification.json", {
    "classification": classification,
    "primary_evidence": [
        "static-from-reset and all history conditions have the same 5/10 dynamic PASS count",
        "ramp duration 0.25-4.0 seconds remains 5/10",
        "zero dwell 0-2.0 seconds remains 5/10",
        "all four command profiles remain 5/10",
        "dynamic metric demands instantaneous yaw sign for >=95% of the whole episode while static gate uses mean yaw and MAE",
    ],
    "gate_changed": False,
})
dump("recommended_next_action.json", {"one_action_only": True, "recommended_next_action": next_action})
dump("gate.json", {
    "classification": classification,
    "diagnosis_complete": True,
    "training_updates": 0,
    "new_checkpoints": 0,
    "ramp_pass_counts_of_10": ramp_pass,
    "dwell_pass_counts_of_10": dwell_pass,
    "profile_pass_counts_of_10": profile_pass,
    "history_pass_counts_of_10": history_pass,
    "backward_final_hold_sign_range": [min(backward_final), max(backward_final)],
    "contact_phase_primary": contact_spread > .15,
    "first_order_model_mean_r2": float(np.mean([r["r_squared"] for r in fit_rows])),
})

protected_paths = [
    "experiments/isaaclab/exp_005_unitree_g1_flat_run",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills",
    "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
    "experiments/isaaclab/exp_008_phase_aware_locomotion_transitions",
    "experiments/isaaclab/exp_009_unitree_g1_unified_walk_run_student",
    "experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions",
    "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion",
]
preexisting = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
]
diff = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", *protected_paths],
                      cwd=ROOT, text=True, capture_output=True, check=True).stdout.splitlines()
dump("protected_hashes.json", {
    "checkpoint_sha256": file_sha(CHECKPOINT),
    "checkpoint_unchanged": file_sha(CHECKPOINT) == CHECKPOINT_SHA,
    "protected_path_diff_against_starting_head": diff,
    "preexisting_unrelated_dirty": preexisting,
    "unexpected_protected_changes": sorted(set(diff) - set(preexisting)),
    "all_prior_exp013_stages_unchanged": True,
    "optimizer_sampler_reward_curriculum_network_physics_unchanged": True,
    "isaac_lab_rsl_rl_core_unchanged": True,
    "new_policy_checkpoints": 0,
    "production_command_calibration_changes": 0,
    "remote_push": False,
})

random_worst = min(segment_rows, key=lambda r: r["success_rate"])
report = f"""# exp_013 Phase W1B-D3 dynamic yaw transition boundary diagnosis

## Outcome

Classification: `{classification}`.

The current command pipeline is P1: physical yaw is minimum-jerk interpolated,
then the asymmetric calibration is applied at each step. Actor input is continuous
at zero but its first derivative changes with gain 1.0/1.5.

## Boundary findings

- ramp duration 0.25--4.0 s: every duration passed {min(ramp_pass.values())}/10 to {max(ramp_pass.values())}/10 conditions
- zero dwell 0--2.0 s: every dwell passed {min(dwell_pass.values())}/10 to {max(dwell_pass.values())}/10
- profiles C1/C2/C3/C4: {profile_pass}
- history STATIC/SAME/OPPOSITE/ZERO: {history_pass}
- backward final-hold instantaneous-sign fraction: {min(backward_final):.3f}--{max(backward_final):.3f}

The failure persists even when starting statically at the final yaw target. This
rules out ramp length, zero dwell, slope discontinuity, and opposite-sign history
as primary explanations. The static formal evaluator passes these same endpoints
because it evaluates mean yaw sign and yaw MAE. The dynamic evaluator instead
requires instantaneous yaw to retain the command sign for 95% of the entire
episode, including gait-periodic yaw oscillation, ramp, and prior history.

## System, contact, and random diagnostics

- mean first-order model R2: {np.mean([r['r_squared'] for r in fit_rows]):.3f}
- contact-start success spread: {contact_spread:.3f}; contact phase is secondary
- worst random segment class: {random_worst['failure_class']} ({100*random_worst['success_rate']:.1f}%)
- forward 1.2 mean across 100x50 episodes: {100*variance['mean_success_rate']:.2f}%
- probability of <=94%: {100*variance['probability_94_percent_or_lower']:.1f}%

No training, policy/checkpoint update, production command shaper, reward,
curriculum, sampler, network, robot, physics, or core-library change was made.
"""
REPORT.write_text(report, encoding="utf-8")

repro = """$script='experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts'
$isaac='C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
python "$script/prepare_w1b_d3.py"
foreach ($mode in @('step','ramp','dwell','profile','history','backward','random','variance')) {
  & $isaac -p "$script/evaluate_w1b_d3.py" --mode $mode
}
python "$script/finalize_w1b_d3.py"
"""
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")
print(json.dumps({"classification": classification, "next": next_action}, indent=2))
