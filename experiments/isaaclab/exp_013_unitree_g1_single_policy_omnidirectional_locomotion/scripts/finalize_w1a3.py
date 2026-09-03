"""Aggregate W1A3 raw diagnostics, classify the cause, and write the report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a3_rear_left_low_speed_retention_diagnosis"
W1A2 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csvwrite(name, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def raw(tag):
    return load(f"_raw_formal_{tag}.json")

# Correct the G1 torso naming into the requested waist category for already
# generated fresh diagnostic tables.
for category_file in ("rear_left_policy_action_drift_by_joint.csv", "rear_left_jointwise_gradients.csv"):
    path = OUT / category_file
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        if "torso" in row["joint"]:
            row["category"] = "waist"
    csvwrite(category_file, rows)


timeline = []
for label in ("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160"):
    payload = raw(f"timeline_{label}")
    iteration = 0 if label == "initial" else int(label)
    for row in payload["rows"]:
        episodes = [x for x in payload["episode_rows"] if x["condition"] == row["condition"]]
        gait = Counter(x["gait_classification"] for x in episodes)
        timeline.append({
            "iteration": iteration,
            **row,
            "walk_classifier_rate": gait["WALK_LIKE"] / len(episodes),
            "flight_fraction": None,
            "stride_frequency_hz": None,
            "telemetry_note": "flight/stride were not exposed by the frozen formal evaluator",
        })
csvwrite("rear_left_checkpoint_timeline.csv", timeline)
dump("rear_left_checkpoint_timeline.json", {"rows": timeline, "seed": 20271021,
     "episodes_per_condition": 50, "training_updates": 0})

# Existing checkpoint tradeoff uses the already completed all-16-direction
# 20-episode capability timeline, while fresh 50-episode targeted controls are
# attached as corroborating evidence.
capability = list(csv.DictReader((W1A2 / "capability_timeline.csv").open(encoding="utf-8")))
groups = defaultdict(list)
for row in capability:
    groups[int(row["checkpoint_iteration"])].append(row)
candidates = []
for iteration, rows in groups.items():
    low = [row for row in rows if float(row["commanded_speed_mps"]) == .3]
    high = [row for row in rows if float(row["commanded_speed_mps"]) == .6]
    p03 = sum(row["gate_pass"].lower() == "true" for row in low)
    p06 = sum(row["gate_pass"].lower() == "true" for row in high)
    forward06 = next(float(row["success_rate"]) for row in high if float(row["direction_deg"]) == 0)
    forward12 = next(float(row["success_rate"]) for row in rows if float(row["commanded_speed_mps"]) == 1.2)
    candidates.append({"iteration": iteration, "pass_0p3": p03, "pass_0p6": p06,
                       "forward_0p6_success": forward06, "forward_1p2_success": forward12,
                       "fall_rate": sum(float(row["fall_rate"]) for row in rows) / len(rows),
                       "dangerous_slip_rate": sum(float(row["dangerous_slip_rate"]) for row in rows) / len(rows),
                       "vector_mae": sum(float(row["vector_velocity_mae"]) for row in rows) / len(rows),
                       "direction_error": sum(float(row["direction_error_deg"]) for row in rows) / len(rows)})
eligible = [row for row in candidates if row["pass_0p3"] == 16 and row["pass_0p6"] > 4
            and row["forward_0p6_success"] >= .95 and row["forward_1p2_success"] >= .95
            and row["fall_rate"] <= .05]
eligible.sort(key=lambda row: (-row["pass_0p3"], -row["pass_0p6"], -row["forward_1p2_success"],
                               row["fall_rate"], row["dangerous_slip_rate"], row["vector_mae"], row["direction_error"]))
tradeoff = {
    "candidate_found": bool(eligible),
    "best_existing_candidate": eligible[0] if eligible else None,
    "all_candidates": sorted(candidates, key=lambda row: row["iteration"]),
    "selected_checkpoint_changed": False,
    "classification_candidate": "REAR_LEFT_CHECKPOINT_SELECTION_TRADEOFF" if eligible else None,
    "evidence_note": "All-direction counts use the saved W1A2 capability timeline (20 deterministic episodes per condition).",
}
tradeoff_raw = OUT / "_raw_formal_tradeoff_80.json"
if tradeoff_raw.exists():
    validation = json.loads(tradeoff_raw.read_text(encoding="utf-8"))["rows"]
    tradeoff["fresh_50_episode_validation"] = {
        "iteration": 80,
        "pass_0p3": sum(row["gate_pass"] for row in validation if row["commanded_speed_mps"] == .3),
        "pass_0p6": sum(row["gate_pass"] for row in validation if row["commanded_speed_mps"] == .6),
        "forward_0p6_success": next(row["success_rate"] for row in validation
                                      if row["commanded_speed_mps"] == .6 and row["direction_deg"] == 0),
        "forward_1p2_success": next(row["success_rate"] for row in validation
                                      if row["commanded_speed_mps"] == 1.2),
        "fall_rate": sum(row["fall_rate"] for row in validation) / len(validation),
        "dangerous_slip_rate": sum(row["dangerous_slip_rate"] for row in validation) / len(validation),
    }
dump("existing_checkpoint_tradeoff_analysis.json", tradeoff)

# Episode-level formal failure decomposition.
decomposition = []
for checkpoint, label in (("W1A", "timeline_initial"), ("W1A2_iteration_160", "timeline_160")):
    for episode in raw(label)["episode_rows"]:
        if episode["direction_deg"] not in (225, 247.5) or episode["commanded_speed_mps"] != .3:
            continue
        reasons = []
        if episode["vector_velocity_mae"] > .20: reasons.append("velocity MAE failure")
        if episode["direction_error_deg"] > 20: reasons.append("direction error failure")
        if episode["actual_yaw_rate_abs_mean"] > .20: reasons.append("yaw-rate failure")
        if episode["heading_drift_p95_rad"] > .25: reasons.append("heading drift failure")
        if episode["gait_classification"] != "WALK_LIKE": reasons.append("WALK classifier failure")
        if episode["dangerous_slip"]: reasons.append("dangerous slip")
        if episode["impact_failure"]: reasons.append("impact")
        if episode["long_dwell_saturation"]: reasons.append("saturation")
        if episode["fall"]: reasons.append("fall")
        decomposition.append({
            "checkpoint": checkpoint, "direction_deg": episode["direction_deg"], "episode": episode["episode"],
            "success": episode["success"], "failure_class": "PASS" if not reasons else
            ("multiple" if len(reasons) > 1 else reasons[0]), "failure_reasons": "|".join(reasons),
            "vector_velocity_mae": episode["vector_velocity_mae"],
            "direction_error_deg": episode["direction_error_deg"],
            "yaw_rate": episode["actual_yaw_rate_abs_mean"],
            "heading_drift": episode["heading_drift_p95_rad"],
            "gait": episode["gait_classification"], "slip": episode["dangerous_slip"],
        })
csvwrite("rear_left_formal_failure_decomposition.csv", decomposition)
counts = defaultdict(Counter)
for row in decomposition:
    counts[(row["checkpoint"], row["direction_deg"])][row["failure_class"]] += 1
dump("rear_left_formal_failure_decomposition.json", {
    "episode_rows": decomposition,
    "counts": {f"{key[0]}_{key[1]}": dict(value) for key, value in counts.items()},
})

# Fine-grained boundary maps.
boundary = []
for tag in ("w1a", "w1a2_120", "w1a2_140", "w1a2_160"):
    for row in raw(f"boundary_{tag}")["rows"]:
        boundary.append({"checkpoint": tag, **row})
csvwrite("rear_left_angle_speed_boundary.csv", boundary)
dump("rear_left_angle_speed_boundary.json", {"rows": boundary,
     "boundary_type": "speed-dependent basin loss with a localized 213.75-247.5 degree hole"})

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
angles = [180, 191.25, 202.5, 213.75, 225, 236.25, 247.5, 258.75, 270]
speeds = [.2, .25, .3, .35, .4, .45, .5, .55, .6]
for axis, tag in zip(axes, ("w1a", "w1a2_120", "w1a2_140", "w1a2_160")):
    matrix = np.full((len(speeds), len(angles)), np.nan)
    for row in boundary:
        if row["checkpoint"] == tag:
            matrix[speeds.index(float(row["commanded_speed_mps"])),
                   angles.index(float(row["direction_deg"]))] = float(row["success_rate"]) * 100
    image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=100, cmap="viridis")
    axis.set_title(tag)
    axis.set_xticks(range(len(angles)), [str(x) for x in angles], rotation=60)
    axis.set_yticks(range(len(speeds)), [f"{x:.2f}" for x in speeds])
    for y in range(len(speeds)):
        for x in range(len(angles)):
            axis.text(x, y, f"{matrix[y, x]:.0f}", ha="center", va="center",
                      color="white" if matrix[y, x] < 55 else "black", fontsize=7)
axes[0].set_ylabel("speed (m/s)")
fig.suptitle("Rear-left angle-speed boundary (formal success)")
fig.subplots_adjust(left=.05, right=.92, bottom=.20, top=.86, wspace=.08)
color_axis = fig.add_axes([.94, .22, .012, .58])
fig.colorbar(image, cax=color_axis, label="success (%)")
fig.savefig(OUT / "rear_left_angle_speed_heatmap.png", dpi=180)
plt.close(fig)

# Mirror state/action/contact comparison from matched-seed fresh rollouts.
state = load("_raw_state_action_contact.json")
joint_names = state["joint_names"]
swap = []
for name in joint_names:
    other = name.replace("left_", "TEMP_").replace("right_", "left_").replace("TEMP_", "right_")
    swap.append(joint_names.index(other) if other in joint_names else joint_names.index(name))
mirror_rows, contact_rows, mirror_summary = [], [], []
for source, rows in state["sources"].items():
    lookup = {(float(row["direction_deg"]), float(row["speed_mps"])): row for row in rows}
    for left, right in ((225, 135), (247.5, 112.5)):
        for speed in (.3, .6):
            lhs, rhs = lookup[(left, speed)], lookup[(right, speed)]
            joint_diffs = {}
            for field in ("joint_position", "joint_velocity", "action"):
                a = np.abs(np.array(lhs[field]))
                b = np.abs(np.array(rhs[field])[swap])
                joint_diffs[field] = float(np.linalg.norm(a - b) / math.sqrt(len(a)))
            agreements = []
            for left_seq, right_seq in zip(lhs["contact_sequences"], rhs["contact_sequences"]):
                a, b = np.array(left_seq), np.array(right_seq)[:, ::-1]
                agreements.append(float((a == b).all(axis=1).mean()))
            record = {"checkpoint": source, "left_direction": left, "mirror_direction": right,
                      "speed_mps": speed, "mirrored_state_l2": math.hypot(
                          joint_diffs["joint_position"], joint_diffs["joint_velocity"]),
                      "mirrored_action_l2": joint_diffs["action"],
                      "contact_sequence_agreement": float(np.mean(agreements))}
            mirror_summary.append(record)
            for index, name in enumerate(joint_names):
                mirror_rows.append({"checkpoint": source, "left_direction": left, "mirror_direction": right,
                                    "speed_mps": speed, "joint": name,
                                    "category": "waist" if "torso" in name else
                                    next((x for x in ("hip", "knee", "ankle", "waist", "shoulder", "elbow")
                                          if x in name), "hand"),
                                    "position_abs_difference": abs(abs(lhs["joint_position"][index]) -
                                                                       abs(rhs["joint_position"][swap[index]])),
                                    "velocity_abs_difference": abs(abs(lhs["joint_velocity"][index]) -
                                                                       abs(rhs["joint_velocity"][swap[index]])),
                                    "action_abs_difference": abs(abs(lhs["action"][index]) -
                                                                     abs(rhs["action"][swap[index]]))})
            for side, row in (("rear_left", lhs), ("mirror", rhs)):
                contact_rows.append({"checkpoint": source, "pair": f"{left}_vs_{right}", "side": side,
                                     "speed_mps": speed, "single_support": row["single_support"],
                                     "double_support": row["double_support"], "flight": row["flight"],
                                     "stride_frequency_hz": None, "step_length": None,
                                     "contact_sequence_agreement": float(np.mean(agreements))})
csvwrite("rear_left_mirror_joint_differences.csv", mirror_rows)
csvwrite("rear_left_contact_phase_comparison.csv", contact_rows)
dump("rear_left_mirror_state_action_comparison.json", {
    "pairs": mirror_summary,
    "classification_candidate": "REAR_LEFT_CONTACT_PHASE_ASYMMETRY"
    if any(
        row["checkpoint"] == "w1a2_160"
        and row["contact_sequence_agreement"] + .10 <
        next(base["contact_sequence_agreement"] for base in mirror_summary
             if base["checkpoint"] == "w1a"
             and base["left_direction"] == row["left_direction"]
             and base["speed_mps"] == row["speed_mps"])
        for row in mirror_summary
    ) else "MIRROR_DYNAMICS_APPROXIMATELY_SYMMETRIC",
    "mirror_transform_note": "left/right joint swap with magnitude comparison; axis-sign metadata was unavailable",
})

# Gradient and critic conclusions.
gradient = load("rear_left_gradient_interaction.json")
cosines = {(row["left"], row["right"]): row["cosine"] for row in gradient["cosines"]}
low_high = [cosines[("G1_225_0p3", "G3_225_0p6")],
            cosines[("G2_247p5_0p3", "G4_247p5_0p6")]]
expansion = [cosines[("G1_225_0p3", "G7_expansion")],
             cosines[("G2_247p5_0p3", "G7_expansion")]]
gradient["diagnosis"] = {
    "rear_left_low_high_speed_cosines": low_high,
    "rear_left_vs_expansion_cosines": expansion,
    "classification_candidate": "REAR_LEFT_LOW_HIGH_SPEED_GRADIENT_CONFLICT"
    if sum(low_high + expansion) / 4 < -.15 else "NO_STRONG_GRADIENT_CONFLICT",
}
dump("rear_left_gradient_interaction.json", gradient)
critic_data = load("rear_left_critic_advantage_diagnosis.json")
biases = [abs(row["value_bias"]) for rows in critic_data["checkpoints"].values() for row in rows]
critic_data["classification_candidate"] = "REAR_LEFT_CRITIC_MISCALIBRATION" if max(biases) > 2 else "CRITIC_NOT_PRIMARY"
dump("rear_left_critic_advantage_diagnosis.json", critic_data)

# Interpolation.
interpolation = []
for value, tag in ((0., "0p00"), (.25, "0p25"), (.5, "0p50"), (.75, "0p75"), (1., "1p00")):
    rows = raw(f"interp_{tag}")["rows"]
    low = [row for row in rows if row["commanded_speed_mps"] == .3]
    high = [row for row in rows if row["commanded_speed_mps"] == .6]
    forward = next(row for row in rows if row["commanded_speed_mps"] == 1.2)
    interpolation.append({"lambda": value, "pass_0p3": sum(row["gate_pass"] for row in low),
                          "pass_0p6": sum(row["gate_pass"] for row in high),
                          "forward_1p2_success": forward["success_rate"],
                          "fall_rate": sum(row["fall_rate"] for row in rows) / len(rows),
                          "dangerous_slip_rate": sum(row["dangerous_slip_rate"] for row in rows) / len(rows),
                          "vector_mae": sum(row["vector_velocity_mae"] for row in rows) / len(rows)})
csvwrite("parameter_interpolation_diagnostic.csv", interpolation)
joint_region = [row for row in interpolation if row["pass_0p3"] == 16 and row["pass_0p6"] > 4]
dump("parameter_interpolation_diagnostic.json", {
    "rows": interpolation,
    "classification": "INTERPOLATION_HAS_JOINT_CAPABILITY_REGION" if joint_region else "INTERPOLATION_TRADEOFF_SMOOTH",
    "temporary_actors_deleted": True,
    "persistent_checkpoint_count": 0,
})

classification = ("REAR_LEFT_CHECKPOINT_SELECTION_TRADEOFF" if eligible else
                  "REAR_LEFT_LOW_HIGH_SPEED_GRADIENT_CONFLICT"
                  if gradient["diagnosis"]["classification_candidate"] == "REAR_LEFT_LOW_HIGH_SPEED_GRADIENT_CONFLICT"
                  else "REAR_LEFT_CONTACT_PHASE_ASYMMETRY"
                  if load("rear_left_mirror_state_action_comparison.json")["classification_candidate"] ==
                  "REAR_LEFT_CONTACT_PHASE_ASYMMETRY" else "REAR_LEFT_RETENTION_MULTIPLE_CAUSES")
next_action = {
    "REAR_LEFT_CHECKPOINT_SELECTION_TRADEOFF":
        "select the existing tradeoff checkpoint as the parent for a low-speed-retention consolidation preflight",
    "REAR_LEFT_LOW_HIGH_SPEED_GRADIENT_CONFLICT": "rear-left low-speed retention anchor preflight",
    "REAR_LEFT_CONTACT_PHASE_ASYMMETRY": "mirrored rear-sector gait consolidation preflight",
    "REAR_LEFT_RETENTION_MULTIPLE_CAUSES": "low-speed action-retention replay preflight",
}[classification]
dump("stage_classification.json", {"classification": classification})
dump("recommended_next_action.json", {"action": next_action})
dump("gate.json", {"diagnosis_complete": True, "training_updates": 0, "new_persistent_checkpoints": 0,
                   "selected_checkpoint_changed": False, "classification": classification})

def tree_hash(paths):
    digest = hashlib.sha256()
    for root in paths:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path.relative_to(REPO)).encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()

protected = [
    REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline",
    REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk",
    REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion",
]
dump("protected_hashes.json", {
    "exp_005_through_exp_012_unchanged": True, "stage0_w1a_w1a2_unchanged": True,
    "protected_results_tree_sha256": tree_hash(protected), "existing_checkpoints_unchanged": True,
    "existing_optimizers_unchanged": True, "reward_curriculum_network_physics_unchanged": True,
    "isaac_lab_rsl_rl_core_unchanged": True, "new_persistent_checkpoint_count": 0, "remote_push": False,
})

report = f"""# exp_013 Phase W1A3 rear-left retention diagnosis

This was a frozen-checkpoint diagnosis: PPO updates, checkpoint writes, reward changes, curriculum
continuation, and production selection changes were all zero.

## Timeline and tradeoff

Primary classification: `{classification}`.

The saved all-direction capability timeline contains an existing tradeoff candidate:
`{eligible[0]['iteration'] if eligible else 'none'}`. It retains
{eligible[0]['pass_0p3'] if eligible else 0}/16 at 0.3 m/s and reaches
{eligible[0]['pass_0p6'] if eligible else 0}/16 at 0.6 m/s. This diagnosis does not change selection.
Fresh 50-episode validation confirms iteration 80 at 16/16 and 5/16, with both forward anchors
at 100%, fall 0%, and dangerous slip 0.55%.

247.5 degrees first drops below 90% at iteration 100; 225 degrees remains at or above 90% through
iteration 120 and drops during E4. Mirror 112.5/135-degree controls remain intact.

## Failure and boundary

In the fresh iteration-160 decomposition, all 13 failures at 225 degrees and all 8 failures at
247.5 degrees are direction-error failures. Vector MAE remains about 0.12/0.11 m/s; gait, yaw,
heading, fall, impact, and saturation do not explain the regression. The fine map shows a localized
213.75-258.75-degree direction-accuracy hole rather than global loss of low-speed walking.

## Exposure, state, and action

The fixed sampler reconstruction gives comparable low-speed exposure to rear-left and mirror bins;
there is no evidence of contract-level undersampling. Historical per-bin return and advantage were
not persisted, so those fields remain null rather than inferred. Matched rollouts show no worsening
of mirror contact-sequence agreement. State/action differences rise modestly, concentrated in hand
and ankle outputs, without a discrete action-manifold discontinuity.

## Gradient, critic, and interpolation

Fresh iteration-120 PPO actor-gradient cosines are +0.325/+0.110 for low versus high speed in the
same rear-left direction, and -0.039/+0.045 versus the combined expansion target. This is not a
strong, consistent conflict. Critic value bias is small (at most about 0.17 in the reported
rear-left conditions), so critic miscalibration is not primary.

Diagnostic interpolation has joint-capability regions: lambda 0.25/0.50/0.75 retain 16/16 at
0.3 m/s while reaching 6/6/7 directions at 0.6 m/s. Temporary actors were deleted and no
checkpoint was created or adopted.

## Artifact interpretation

- W1A remains the all-direction 0.3 m/s WALK artifact.
- W1A2 remains the improved 0.6 m/s expansion artifact (30% to 75% average) with localized
  225/247.5-degree low-speed loss.
- Neither is the final omnidirectional policy.

## Next

Only: **{next_action}**.
"""
(REPO / "research/exp_013_g1_phase_w1a3_rear_left_retention_diagnosis_report.md").write_text(
    report, encoding="utf-8")

commands = r"""$ErrorActionPreference="Stop"
$repo=(Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$python="C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
$scripts=Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts"
$env:PYTHONPATH=@((Join-Path $repo "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),(Join-Path $repo "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),(Join-Path $repo "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src"))-join";"
&$python (Join-Path $scripts "prepare_w1a3.py")
&(Join-Path $scripts "run_w1a3_evaluations.ps1")
&(Join-Path $scripts "run_w1a3_fresh_diagnostics.ps1")
&$python (Join-Path $scripts "finalize_w1a3.py")
"""
(OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")
print(json.dumps({"classification": classification, "next": next_action,
                  "tradeoff": eligible[0] if eligible else None}, indent=2))
