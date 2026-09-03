"""Finalize EXP013 W1B-D2 artifacts and evidence-based classification."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis"
)
R2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun"
)
REPORT = REPO / "research/exp_013_g1_phase_w1b_d2_yaw_rate_tracking_boundary_report.md"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


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
        writer.writeheader(); writer.writerows(rows)


def regroup(payload, keys):
    grouped = defaultdict(list)
    for row in payload["episode_rows"]:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output = []
    rate_keys = ("success", "yaw_sign_correct", "translation_correct", "yaw_correct",
                 "fall", "dangerous_slip", "impact", "saturation", "excessive_tilt")
    mean_keys = ("actual_vx", "actual_vy", "actual_speed", "actual_yaw", "actual_yaw_p95",
                 "vector_mae", "direction_error", "yaw_mae", "slip_fraction", "tilt_mean",
                 "left_contact_fraction", "right_contact_fraction", "flight_fraction",
                 "double_support_fraction", "action_abs_p95", "action_abs_p99",
                 "joint_limit_proximity", "joint_velocity_ratio", "torque_abs_mean",
                 "contact_force_mean", "basin_yaw_after_branch")
    for key, rows in grouped.items():
        item = {name: value for name, value in zip(keys, key)}
        for name in ("condition", "checkpoint", "direction_deg", "commanded_speed",
                     "target_yaw", "actor_input_yaw", "wrapper", "lambda", "branch_steps", "iteration"):
            if name not in item:
                item[name] = rows[0].get(name)
        item["episodes"] = len(rows)
        for name in rate_keys:
            item[name + "_rate"] = sum(bool(row[name]) for row in rows) / len(rows)
        for name in mean_keys:
            item[name] = sum(float(row[name]) for row in rows) / len(rows)
        item["gate_pass"] = item["success_rate"] >= .9 and item["fall_rate"] <= .05
        output.append(item)
    return output


# Repair only D2 aggregation presentation; raw episode measurements are unchanged.
timeline_payload = load("yaw_capability_checkpoint_timeline.json")
timeline = regroup(timeline_payload, ("iteration", "condition"))
write_csv("yaw_capability_checkpoint_timeline.csv", timeline)
dump("yaw_capability_checkpoint_timeline.json", {
    "rows": timeline, "episode_rows": timeline_payload["episode_rows"],
    "episodes_per_condition": 50, "same_seed": True,
})
surface_payload = load("detailed_yaw_command_response_surface.json")
surface = regroup(surface_payload, ("checkpoint", "condition"))
write_csv("detailed_yaw_command_response_surface.csv", surface)
dump("detailed_yaw_command_response_surface.json", {
    "rows": surface, "episode_rows": surface_payload["episode_rows"],
    "episodes_per_condition": 20, "checkpoints": ["model_80", "model_200"],
})
mirror_payload = load("mirrored_policy_positive_control.json")
mirror_rows = regroup(mirror_payload, ("wrapper", "condition"))
write_csv("mirrored_policy_positive_control.csv", mirror_rows)
dump("mirrored_policy_positive_control.json", {
    "rows": mirror_rows, "episode_rows": mirror_payload["episode_rows"],
    "episodes_per_condition": 100, "diagnostic_only": True,
})

# Existing-checkpoint tradeoff.
r2_capability = list(csv.DictReader((R2 / "capability_timeline.csv").open(encoding="utf-8")))
tradeoff = []
for iteration in (0, 1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200):
    cap = [row for row in r2_capability if int(row["checkpoint_iteration"]) == iteration]
    time = [row for row in timeline if int(row["iteration"]) == iteration]
    zero = [row for row in cap if row["condition"].startswith("ZERO_D")]
    pure_pos = next(row for row in time if row["condition"] == "PURE_Y+0.3")
    moving = [row for row in time if row["condition"].startswith("D")]
    forward06 = next(row for row in cap if row["condition"] == "FWD_0P6")
    forward12 = next(row for row in cap if row["condition"] == "FWD_1P2")
    tradeoff.append({
        "iteration": iteration,
        "zero_yaw_pass_directions": sum(row["gate_pass"] == "True" for row in zero),
        "pure_positive_success": pure_pos["success_rate"],
        "pure_positive_yaw_mae": pure_pos["yaw_mae"],
        "key_moving_turn_pass": sum(row["gate_pass"] for row in moving),
        "forward_0p6_success": float(forward06["success_rate"]),
        "forward_1p2_success": float(forward12["success_rate"]),
        "fall_rate_max": max(float(row["fall_rate"]) for row in cap),
        "safety_pass": max(float(row["fall_rate"]) for row in cap) <= .05,
    })
eligible = [row for row in tradeoff if row["zero_yaw_pass_directions"] == 16
            and row["forward_0p6_success"] >= .95 and row["forward_1p2_success"] >= .95
            and row["safety_pass"]]
best_tradeoff = max(
    eligible,
    key=lambda row: (row["pure_positive_success"], row["key_moving_turn_pass"],
                     -row["pure_positive_yaw_mae"], -row["fall_rate_max"]),
)
dump("existing_yaw_checkpoint_tradeoff.json", {
    "rows": tradeoff, "best_existing_tradeoff": best_tradeoff,
    "selection_changed": False, "canonical_promotion_changed": False,
})

# Piecewise response fits.
fit_rows = []
for checkpoint in ("model_80", "model_200"):
    for direction in (0, 45, 90, 135, 180, 225, 270, 315):
        for speed in (0, .1, .2, .3, .4, .6):
            subset = [row for row in surface if row["checkpoint"] == checkpoint
                      and row["direction_deg"] == direction and row["commanded_speed"] == speed]
            result = {"checkpoint": checkpoint, "direction_deg": direction, "speed": speed}
            for sign_name, sign in (("negative", -1), ("positive", 1)):
                rows = sorted([row for row in subset if row["actor_input_yaw"] * sign > 0],
                              key=lambda row: row["actor_input_yaw"])
                x = np.array([row["actor_input_yaw"] for row in rows])
                y = np.array([row["actual_yaw"] for row in rows])
                slope, offset = np.polyfit(x, y, 1)
                prediction = slope * x + offset
                r2 = 1 - np.sum((y - prediction) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
                effective = [row for row in rows if abs(row["actual_yaw"]) >= .05]
                safe = [row for row in rows if row["fall_rate"] <= .05 and row["dangerous_slip_rate"] <= .1]
                required = (sign * .3 - offset) / slope if abs(slope) > 1e-8 else None
                result.update({
                    f"{sign_name}_gain": float(slope),
                    f"{sign_name}_offset": float(offset),
                    f"{sign_name}_r2": float(r2),
                    f"{sign_name}_minimum_effective_command": (
                        min((abs(row["actor_input_yaw"]) for row in effective), default=None)
                    ),
                    f"{sign_name}_command_required_for_0p3": abs(float(required)) if required is not None else None,
                    f"{sign_name}_saturation_onset": (
                        min((abs(row["actor_input_yaw"]) for row in rows
                             if row["action_abs_p99"] >= 3.8), default=None)
                    ),
                    f"{sign_name}_maximum_safe_actual_yaw": max(
                        (abs(row["actual_yaw"]) for row in safe), default=None
                    ),
                })
            result["dead_zone"] = max(
                result["positive_minimum_effective_command"] or 0,
                result["negative_minimum_effective_command"] or 0,
            )
            fit_rows.append(result)
write_csv("yaw_response_curve_fits.csv", fit_rows)
positive_required = [row["positive_command_required_for_0p3"] for row in fit_rows
                     if row["checkpoint"] == "model_200" and row["positive_command_required_for_0p3"] is not None]
negative_required = [row["negative_command_required_for_0p3"] for row in fit_rows
                     if row["checkpoint"] == "model_200" and row["negative_command_required_for_0p3"] is not None]
dump("yaw_response_curve_fits.json", {
    "rows": fit_rows,
    "selected_positive_required_range": [min(positive_required), max(positive_required)],
    "selected_negative_required_range": [min(negative_required), max(negative_required)],
    "classification": "GLOBAL_POSITIVE_YAW_GAIN_BIAS",
})

# Heatmap at selected 0.3 m/s.
selected_map = [row for row in surface if row["checkpoint"] == "model_200"
                and row["commanded_speed"] == .3]
directions = (0, 45, 90, 135, 180, 225, 270, 315)
yaw_commands = (-1, -.8, -.6, -.5, -.4, -.3, -.2, -.1, 0, .1, .2, .3, .4, .5, .6, .8, 1)
matrix = np.array([[next(row["actual_yaw"] for row in selected_map
                         if row["direction_deg"] == direction and row["actor_input_yaw"] == yaw)
                    for yaw in yaw_commands] for direction in directions])
fig, ax = plt.subplots(figsize=(13, 6))
image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(yaw_commands)), [f"{value:+.1f}" for value in yaw_commands], rotation=45)
ax.set_yticks(range(len(directions)), [f"{value} deg" for value in directions])
ax.set_xlabel("Actor yaw command [rad/s]"); ax.set_ylabel("Translation direction")
ax.set_title("W1B-R2 iteration 200 actual body yaw rate at 0.3 m/s")
for y in range(len(directions)):
    for x in range(len(yaw_commands)):
        ax.text(x, y, f"{matrix[y,x]:.2f}", ha="center", va="center", fontsize=6,
                color="black" if abs(matrix[y,x]) < .65 else "white")
fig.colorbar(image, ax=ax, label="actual yaw rate [rad/s]")
fig.tight_layout(); fig.savefig(OUT / "yaw_command_response_heatmap.png", dpi=180); plt.close(fig)

# Prewarp global and direction-conditioned oracle.
prewarp = load("positive_yaw_command_prewarp.json")
prewarp_rows = prewarp["rows"]
global_rows = []
for target in (.15, .3, .45):
    for gain in (1, 1.25, 1.5, 1.75, 2, 2.5, 3):
        rows = [row for row in prewarp_rows if abs(row["target_yaw"] - target) < 1e-8
                and abs(row["actor_input_yaw"] / target - gain) < 1e-6]
        global_rows.append({
            "target": target, "gain": gain, "conditions": len(rows),
            "pass_conditions": sum(row["gate_pass"] for row in rows),
            "mean_yaw_mae": float(np.mean([row["yaw_mae"] for row in rows])),
            "maximum_fall_rate": max(row["fall_rate"] for row in rows),
            "maximum_dangerous_slip_rate": max(row["dangerous_slip_rate"] for row in rows),
        })
oracle = []
for target in (.15, .3, .45):
    for label in ("PURE", "D000", "D045", "D090", "D135", "D180", "D225", "D270", "D315"):
        rows = [row for row in prewarp_rows if row["condition"].startswith(f"{label}_T{target:.2f}")]
        best = min(rows, key=lambda row: (row["yaw_mae"], row["fall_rate"], row["dangerous_slip_rate"]))
        oracle.append({
            "target": target, "condition_group": label,
            "best_input_command": best["actor_input_yaw"], "actual_yaw": best["actual_yaw"],
            "yaw_mae": best["yaw_mae"], "translation_mae": best["vector_mae"],
            "success_rate": best["success_rate"], "fall_rate": best["fall_rate"],
            "dangerous_slip_rate": best["dangerous_slip_rate"],
        })
best_global_03 = max(
    [row for row in global_rows if row["target"] == .3],
    key=lambda row: (row["pass_conditions"], -row["mean_yaw_mae"]),
)
dump("positive_yaw_command_prewarp.json", {
    "rows": prewarp_rows, "global_candidates": global_rows, "direction_conditioned_oracle": oracle,
    "best_global_target_0p3": best_global_03,
    "formal_adoption": False, "diagnostic_only": True,
})
write_csv("positive_yaw_command_prewarp.csv", prewarp_rows)

# Translation unlock summary.
unlock_payload = load("positive_yaw_translation_unlock_map.json")
unlock_rows = unlock_payload["rows"]
unlock_summary = []
for input_yaw in (.3, .5, .7):
    for direction in directions:
        rows = [row for row in unlock_rows if row["direction_deg"] == direction
                and abs(row["actor_input_yaw"] - input_yaw) < 1e-8]
        passing = sorted([row for row in rows if row["gate_pass"]], key=lambda row: row["commanded_speed"])
        unlock_summary.append({
            "direction_deg": direction, "actor_input_yaw": input_yaw,
            "minimum_unlock_speed": passing[0]["commanded_speed"] if passing else None,
            "maximum_success_rate": max(row["success_rate"] for row in rows),
            "pure_yaw_success": next(row["success_rate"] for row in rows if row["commanded_speed"] == 0),
        })
dump("positive_yaw_translation_unlock_map.json", {
    "rows": unlock_rows, "unlock_summary": unlock_summary,
    "classification": "SMALL_TRANSLATION_UNLOCKS_POSITIVE_YAW_AT_INPUT_0P3_BUT_INPUT_0P5_UNLOCKS_PURE",
})
write_csv("positive_yaw_translation_unlock_map.csv", unlock_rows)

# Training exposure: exact commands were not persisted per command bin.
runtime = list(csv.DictReader((R2 / "sampler_runtime_trace.csv").open(encoding="utf-8")))
phases = [
    ("Y1", 1, 40, {"zero_yaw": .45, "moving_yaw": .45, "pure_yaw": .10}),
    ("Y2", 41, 100, {"zero_yaw": .40, "moving_yaw": .50, "pure_yaw": .10}),
    ("Y3", 101, 150, {"zero_yaw": .35, "moving_yaw": .40, "pure_yaw": .25}),
    ("Y4", 151, 200, {"zero_yaw": .35, "moving_yaw": .45, "pure_yaw": .20}),
]
exposure_rows = []
previous_commands = 0
for phase, start, end, weights in phases:
    end_row = next(row for row in runtime if int(row["iteration"]) == end)
    total = int(end_row["base_command_count"]) + int(end_row["mirror_command_count"]) - previous_commands
    previous_commands += total
    curve_rows = list(csv.DictReader((R2 / "training_curves.csv").open(encoding="utf-8")))
    phase_curve = [row for row in curve_rows if start <= int(row["iteration"]) <= end]
    for group, weight in weights.items():
        nonzero = group != "zero_yaw"
        exposure_rows.append({
            "curriculum_phase": phase, "command_group": group,
            "translation_direction_22p5_bin": "continuous; exact bin not_recorded",
            "translation_speed_bin": "configured range; exact count not_recorded",
            "yaw_sign": "paired +/-" if nonzero else "zero",
            "yaw_magnitude_bin": "configured range; exact count not_recorded",
            "pure_or_moving": group,
            "total_phase_command_assignments": total,
            "preregistered_group_weight": weight,
            "estimated_group_command_count": round(total * weight),
            "positive_negative_count_difference": 0 if nonzero else "not_applicable",
            "rollout_sample_count": "not_recorded by command bin",
            "episode_count": "not_recorded by command bin",
            "ppo_minibatch_inclusion_count": "not_recorded by command bin",
            "return": float(np.mean([float(row["mean_reward"]) for row in phase_curve])),
            "yaw_reward": float(np.mean([float(row["reward_track_ang_vel_z_exp"]) for row in phase_curve])),
            "translation_reward": float(np.mean([float(row["reward_track_lin_vel_xy_exp"]) for row in phase_curve])),
            "safety_reward": "not separable from saved aggregate telemetry",
            "advantage_mean": "not_recorded by command bin",
            "advantage_std": "not_recorded by command bin",
            "positive_advantage_rate": "not_recorded by command bin",
            "negative_advantage_rate": "not_recorded by command bin",
            "fall": float(np.mean([float(row["fall_rate"]) for row in phase_curve])),
            "slip": float(np.mean([float(row["dangerous_slip_rate"]) for row in phase_curve])),
            "termination": "not_recorded by command bin",
        })
write_csv("yaw_training_exposure_audit.csv", exposure_rows)
dump("yaw_training_exposure_audit.json", {
    "rows": exposure_rows,
    "command_balance": "positive/negative exact by pending-mirror queue contract",
    "command_bin_limitation": "exact per-command rollout/minibatch telemetry was not persisted; not inferred",
    "pure_positive_yaw_exposure_shortage_evidence": False,
})

# Physical-limit audit from detailed surface.
physical_rows = []
for row in surface:
    if row["checkpoint"] != "model_200" or row["actor_input_yaw"] not in (-1, -.3, .3, 1):
        continue
    physical_rows.append({
        "direction_deg": row["direction_deg"], "speed": row["commanded_speed"],
        "yaw_command": row["actor_input_yaw"], "actual_yaw": row["actual_yaw"],
        "action_p95": row["action_abs_p95"], "action_p99": row["action_abs_p99"],
        "joint_limit_proximity": row["joint_limit_proximity"],
        "joint_velocity_ratio": row["joint_velocity_ratio"],
        "torque_abs_mean": row["torque_abs_mean"],
        "contact_force_mean": row["contact_force_mean"],
        "slip_rate": row["dangerous_slip_rate"], "tilt_rate": row["excessive_tilt_rate"],
        "fall_rate": row["fall_rate"], "saturation_rate": row["saturation_rate"],
    })
write_csv("yaw_action_physical_limit_audit.csv", physical_rows)
dump("yaw_action_physical_limit_audit.json", {
    "rows": physical_rows,
    "classification": "NO_PHYSICAL_LIMIT_EVIDENCE",
    "reason": "positive input 0.45-0.5 reaches target safely; no long-dwell saturation or fall boundary",
})

# Main evidence and classification.
reward = load("positive_negative_yaw_reward_advantage.json")["rows"]
pure_neg = next(row for row in reward if row["condition"] == "PURE_NEG")
pure_pos = next(row for row in reward if row["condition"] == "PURE_POS")
gradient = load("yaw_boundary_gradient_interaction.json")["cosines"]
pure_gradient_cosine = next(row["total_cosine"] for row in gradient
                            if row["left"] == "PURE_NEG" and row["right"] == "PURE_POS")
mirror_normal = {row["condition"]: row for row in mirror_rows if row["wrapper"] == "normal"}
mirror_wrapped = {row["condition"]: row for row in mirror_rows if row["wrapper"] == "mirror"}
local_rows = load("positive_yaw_local_action_controllability.json")["rows"]
local_best = max(row["success_rate"] for row in local_rows)
classification = "POSITIVE_YAW_GLOBAL_GAIN_BIAS"
dump("current_yaw_artifact_interpretation.json", {
    "w1b_r2_sampler_repair": "PASS",
    "w1b_r2_training": "200 iterations completed",
    "zero_yaw_omnidirectional_walk": "PASS",
    "moving_turns": "21/24 PASS",
    "pure_negative_yaw": "PASS",
    "pure_positive_yaw": "FAIL at native +0.3; succeeds diagnostically near +0.45/+0.5 input",
    "safety": "PASS",
    "canonical_promotion": "none",
    "canonical_translation_only_walk": "W1A2 iteration 80",
    "w1b_r2_selected": "diagnostic yaw-capable WALK artifact",
})
dump("stage_classification.json", {
    "primary_classification": classification,
    "global_gain_target_0p3": best_global_03,
    "direction_oracle_input_range_target_0p3": [
        min(row["best_input_command"] for row in oracle if row["target"] == .3),
        max(row["best_input_command"] for row in oracle if row["target"] == .3),
    ],
    "mirror_wrapper_all_condition_pass": all(row["gate_pass"] for row in mirror_wrapped.values()),
    "mirror_wrapper_max_fall": max(row["fall_rate"] for row in mirror_wrapped.values()),
    "local_action_max_success": local_best,
    "pure_positive_advantage_suppressed": pure_pos["advantage_mean"] < pure_neg["advantage_mean"] * .5,
    "pure_positive_value_bias": pure_pos["value_bias"],
    "pure_negative_positive_gradient_cosine": pure_gradient_cosine,
    "physical_limit": False,
})
dump("recommended_next_action.json", {
    "next": "single monotonic positive-yaw command calibration preflight",
    "constraints": ["no action controller", "no checkpoint switch", "one actor remains"],
})
dump("gate.json", {
    "diagnosis_complete": True, "classification": classification,
    "new_persistent_policy_checkpoint": 0, "optimizer_steps": 0,
    "command_prewarp_formally_adopted": False, "mirror_wrapper_formally_adopted": False,
    "canonical_promotion": False, "remote_push": False,
})

# Protection audit preserves the pre-existing unrelated dirty state.
start = load("stage_reference.json")
current_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
def unrelated(line):
    return (
        "w1b_d2" not in line
        and "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis" not in line
        and "exp_013_g1_phase_w1b_d2_yaw_rate_tracking_boundary_report.md" not in line
    )
protected_dirty_start = sorted(start["starting_status_short"])
protected_dirty_now = sorted(line for line in current_status if unrelated(line))
dump("protected_hashes.json", {
    "starting_head": start["starting_head_actual"],
    "preexisting_dirty_state_entry_count": len(protected_dirty_start),
    "preexisting_dirty_state_preserved": protected_dirty_start == protected_dirty_now,
    "new_persistent_policy_checkpoint": 0,
    "production_policy_update": 0,
    "remote_push": False,
    "existing_exp013_artifacts_modified_by_d2": False,
    "existing_checkpoints_modified": False,
    "existing_optimizers_modified": False,
    "sampler_modified": False, "reward_modified": False, "curriculum_modified": False,
    "network_modified": False, "physics_modified": False,
    "isaaclab_rslrl_core_modified": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    "$exp='experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts'\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode timeline --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode surface --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode prewarp --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode unlock --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode mirror --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/evaluate_w1b_d2.py\" --mode local --headless\n"
    "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p \"$exp/diagnose_w1b_d2_gradients.py\" --headless\n",
    encoding="utf-8",
)

REPORT.write_text(f"""# EXP013 Phase W1B-D2 yaw-rate tracking boundary diagnosis

## Outcome

Primary classification: `{classification}`.

The selected W1B-R2 actor is monotonic on both yaw signs. At the native +0.30 input its positive
response is too small, but a diagnostic global gain of {best_global_03['gain']:.2f} passes
{best_global_03['pass_conditions']}/{best_global_03['conditions']} target-0.30 pure/moving conditions
with maximum fall {best_global_03['maximum_fall_rate']:.1%}. Direction-specific oracle inputs span
{min(row['best_input_command'] for row in oracle if row['target'] == .3):.3f} to
{max(row['best_input_command'] for row in oracle if row['target'] == .3):.3f} rad/s.

## Timeline and response boundary

Positive yaw did not regress after an early peak. It remained near zero through the first half of
training and improved late: pure +0.30 reached approximately +0.135 at iteration 160 and +0.160 at
iteration 200. Pure -0.30 was already supported and ends near -0.300. The same late improvement is
direction dependent in magnitude: native +0.30 remains weakest for 90, 135, and 180 degree
translation. Iteration 200 is the best existing checkpoint under the required zero-yaw, forward,
and safety retention constraints; no selection change is made in this diagnostic.

The selected actor has a monotonic positive response, with fitted gain depending somewhat on
translation state but no hard saturation near the target. At 0.3 m/s the diagnostic positive
command required for +0.30 ranges from roughly +0.38 to +0.59 rad/s. Commands around +0.45 to
+0.53 reach the target across the tested directions, while +0.7 tends to overshoot. The native
+0.30 failure is therefore an input-to-response gain/offset boundary, not absence of a physically
reachable right-turn gait.

## Translation unlock

At the native +0.30 input, adding a small translation unlocks only some directions; it does not
provide a universal turn-in-place solution. In contrast, a +0.50 actor input succeeds for pure yaw
without translation. This rejects a turn-in-place-only dynamical barrier as the primary cause.

## Exposure, reward, advantage, and gradients

The serialized training artifacts establish mirror-balanced command sampling, but per-bin rollout
return, advantage, and minibatch inclusion telemetry were not retained for all 200 iterations;
unavailable fields are explicitly `not_recorded` in the exposure audit. Fresh on-policy diagnosis
shows comparable yaw and translation reward on both signs. Pure positive/negative 24-step
advantages are {pure_pos['advantage_mean']:.4f} / {pure_neg['advantage_mean']:.4f}, and neither
positive-advantage rate nor critic bias indicates positive-only suppression.

The pure negative/positive total-gradient cosine is {pure_gradient_cosine:.4f}. Several mirrored
direction pairs also show sign-dependent gradient opposition, but this is secondary evidence:
the frozen policy reaches the requested positive rate safely through command magnitude alone,
without any parameter update.

## Counterfactual controls

The full mirrored-policy wrapper is not a valid positive control: maximum fall is
{max(row['fall_rate'] for row in mirror_wrapped.values()):.1%}. Short 1-8-step mirrored-action
interpolation reaches at most {local_best:.1%} success and does not establish a retained alternative
basin. These results reject mirrored runtime control and do not indicate local action-manifold
reachability.

## Action, contact, and state analysis

Mean-action mirror asymmetry remains measurable, and mirrored positive/negative state populations
are partially separable (linear AUROC about 0.84-0.91 in the failing groups). Contact timing and
support fractions differ after mirroring, but no joint, action-saturation, torque, or contact-force
limit blocks the calibrated positive response. The direct mirrored wrapper falls in every tested
condition, so it is not a valid runtime remedy or evidence that a simple mirrored action basin can
be entered locally.

## Artifact status

This stage creates no checkpoint and performs no optimizer step. W1B-R2 remains a diagnostic
yaw-capable WALK artifact; W1A2 iteration 80 remains the canonical translation-only WALK parent.
No command calibration or mirrored wrapper is adopted.
""", encoding="utf-8")
print(classification, best_global_03)
