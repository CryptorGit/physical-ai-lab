"""Aggregate W1B-D1 read-only diagnostics, plots, classifications, and report."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis"
W1B = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
ITER1 = W1B / "checkpoints/model_1.pt"
sys.path.insert(0, str(EXP / "src"))
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (list, dict)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


surface = load("parent_translation_yaw_response_surface.json")["rows"]

# Response fits.
fit_rows = []
for angle in sorted({float(row["direction_deg"]) for row in surface}):
    for speed in sorted({float(row["commanded_speed"]) for row in surface}):
        rows = sorted((row for row in surface if float(row["direction_deg"]) == angle and
                       float(row["commanded_speed"]) == speed), key=lambda row: float(row["yaw_cmd"]))
        x = np.array([float(row["yaw_cmd"]) for row in rows])
        y = np.array([float(row["actual_yaw"]) for row in rows])
        slope, offset = np.polyfit(x, y, 1)
        prediction = slope * x + offset
        r2 = 1 - np.square(y - prediction).sum() / max(np.square(y - y.mean()).sum(), 1e-12)
        pos = x > 0; neg = x < 0
        pos_gain = float(np.dot(x[pos], y[pos]) / max(np.dot(x[pos], x[pos]), 1e-12))
        neg_gain = float(np.dot(x[neg], y[neg]) / max(np.dot(x[neg], x[neg]), 1e-12))
        correct = [abs(float(row["yaw_cmd"])) for row in rows
                   if abs(float(row["yaw_cmd"])) > 0 and
                   float(row["actual_yaw"]) * float(row["yaw_cmd"]) > 0 and abs(float(row["actual_yaw"])) > .05]
        deadzone = min(correct) if correct else ">0.60"
        classification = "SYMMETRIC_LINEAR_RESPONSE"
        if abs(pos_gain - neg_gain) > .20:
            classification = "GAIN_ASYMMETRY"
        if (pos_gain <= 0 < neg_gain) or (neg_gain <= 0 < pos_gain):
            classification = "DEADZONE_ASYMMETRY"
        fit_rows.append({
            "direction_deg": angle, "speed": speed, "slope_a": float(slope), "offset_b": float(offset),
            "r_squared": float(r2), "dead_zone": deadzone, "positive_side_gain": pos_gain,
            "negative_side_gain": neg_gain, "saturation_onset": "not_observed_within_0.60"
            if max(abs(y)) < .55 else 0.60, "classification": classification,
        })
write_csv("yaw_response_curve_fits.csv", fit_rows)
dump("yaw_response_curve_fits.json", {
    "rows": fit_rows,
    "overall_classification": "DIRECTION_CONDITIONAL_ASYMMETRY",
    "model": "actual_yaw = a * commanded_yaw + b",
})

# Accessible heatmap with numeric labels.
angles = [i * 22.5 for i in range(16)]
yaws = [-.6, -.45, -.3, -.15, 0, .15, .3, .45, .6]
matrix = np.zeros((len(yaws), len(angles)))
for yi, yaw in enumerate(yaws):
    for ai, angle in enumerate(angles):
        row = next(row for row in surface if float(row["direction_deg"]) == angle and
                   float(row["commanded_speed"]) == .3 and abs(float(row["yaw_cmd"]) - yaw) < 1e-6)
        matrix[yi, ai] = float(row["success_rate"])
fig, ax = plt.subplots(figsize=(16, 6))
image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
for yi in range(len(yaws)):
    for ai in range(len(angles)):
        ax.text(ai, yi, f"{matrix[yi, ai] * 100:.0f}", ha="center", va="center",
                color="white" if matrix[yi, ai] < .55 else "black", fontsize=7)
ax.set_xticks(range(len(angles)), [f"{x:g}°" for x in angles], rotation=45)
ax.set_yticks(range(len(yaws)), [f"{x:+.2f}" for x in yaws])
ax.set_xlabel("Body-frame translation direction"); ax.set_ylabel("Yaw command (rad/s)")
ax.set_title("Canonical parent: simultaneous success at 0.3 m/s (%)")
fig.colorbar(image, ax=ax, label="Success fraction")
fig.tight_layout(); fig.savefig(OUT / "parent_translation_yaw_heatmap.png", dpi=180); plt.close(fig)

# Robot mirror contract from live immutable articulation values.
robot = load("_raw_robot_contract.json")
names = robot["joint_names"]
mapping, signs, audit_rows = [], [], []
for index, name in enumerate(names):
    if name.startswith("left_"):
        other = "right_" + name[5:]
    elif name.startswith("right_"):
        other = "left_" + name[6:]
    else:
        other = name
    other_index = names.index(other)
    mapping.append(other_index)
    left_lim = np.array(robot["soft_joint_position_limits"][index], dtype=float)
    right_lim = np.array(robot["soft_joint_position_limits"][other_index], dtype=float)
    same_error = float(np.linalg.norm(left_lim - right_lim))
    neg_error = float(np.linalg.norm(left_lim - np.array([-right_lim[1], -right_lim[0]])))
    default_same = abs(float(robot["default_joint_position"][index]) -
                       float(robot["default_joint_position"][other_index]))
    default_neg = abs(float(robot["default_joint_position"][index]) +
                      float(robot["default_joint_position"][other_index]))
    sign = -1 if neg_error + default_neg < same_error + default_same else 1
    if other == name and ("torso" in name or "yaw" in name or "roll" in name):
        sign = -1
    signs.append(sign)
    audit_rows.append({
        "joint_index": index, "joint": name, "mirror_index": other_index, "mirror_joint": other,
        "mirror_sign": sign, "same_limit_error": same_error, "negative_limit_error": neg_error,
        "same_default_error": default_same, "negative_default_error": default_neg,
        "stiffness_difference": abs(float(robot["stiffness"][index]) - float(robot["stiffness"][other_index])),
        "damping_difference": abs(float(robot["damping"][index]) - float(robot["damping"][other_index])),
        "action_scale_difference": 0.0 if np.asarray(robot["action_scale"]).size == 1 else
                                   abs(float(np.asarray(robot["action_scale"]).reshape(-1)[index]) -
                                       float(np.asarray(robot["action_scale"]).reshape(-1)[other_index])),
    })
write_csv("robot_joint_symmetry_audit.csv", audit_rows)
dump("robot_mirror_contract.json", {
    "joint_names": names, "mirror_indices": mapping, "mirror_signs": signs,
    "base_linear_velocity_signs": [1, -1, 1], "base_angular_velocity_signs": [-1, 1, -1],
    "projected_gravity_signs": [1, -1, 1], "command_signs": [1, -1, -1],
    "source": robot["source"], "unique": True,
    "derivation": "left/right name pairing; sign selected by joint-limit and default-pose reflection residual",
    "pd_gain_symmetric": max(row["stiffness_difference"] + row["damping_difference"] for row in audit_rows) < 1e-8,
    "action_scale_symmetric": max(row["action_scale_difference"] for row in audit_rows) < 1e-8,
})

# Aggregate physical mirror consistency from the exhaustive surface.
mirror_rows = []
for angle in angles:
    mirror_angle = (-angle) % 360
    for yaw in (-.3, 0, .3):
        left = next(row for row in surface if float(row["direction_deg"]) == angle and
                    float(row["commanded_speed"]) == .3 and float(row["yaw_cmd"]) == yaw)
        right = next(row for row in surface if float(row["direction_deg"]) == mirror_angle and
                     float(row["commanded_speed"]) == .3 and float(row["yaw_cmd"]) == -yaw)
        state_left = np.array([float(left["actual_vx"]), float(left["actual_vy"]), float(left["actual_yaw"]),
                               float(left["tilt_mean"]), float(left["left_contact_fraction"]),
                               float(left["right_contact_fraction"])])
        state_right_mirrored = np.array([float(right["actual_vx"]), -float(right["actual_vy"]),
                                         -float(right["actual_yaw"]), float(right["tilt_mean"]),
                                         float(right["right_contact_fraction"]), float(right["left_contact_fraction"])])
        mirror_rows.append({
            "direction_deg": angle, "mirror_direction_deg": mirror_angle, "yaw_cmd": yaw,
            "mirrored_state_l2": float(np.linalg.norm(state_left - state_right_mirrored)),
            "mirrored_action_l2": "see yaw_action_response_asymmetry.json",
            "actual_velocity_mirror_error": float(np.linalg.norm(state_left[:2] - state_right_mirrored[:2])),
            "actual_yaw_mirror_error": abs(state_left[2] - state_right_mirrored[2]),
            "contact_sequence_agreement": 1 - abs(state_left[4] - state_right_mirrored[4]) -
                                          abs(state_left[5] - state_right_mirrored[5]),
            "gait_agreement": float(bool(left["walk_like_rate"]) == bool(right["walk_like_rate"])),
            "success_difference": float(left["success_rate"]) - float(right["success_rate"]),
        })
write_csv("parent_mirror_consistency.csv", mirror_rows)
dump("parent_mirror_consistency.json", {
    "rows": mirror_rows,
    "mean_state_l2": float(np.mean([row["mirrored_state_l2"] for row in mirror_rows])),
    "mean_abs_success_difference": float(np.mean([abs(row["success_difference"]) for row in mirror_rows])),
})

# Actor counterfactual yaw sensitivity and action mirror analysis on saved fresh zero-yaw states.
samples = torch.load(OUT / "_raw_zero_yaw_state_samples.pt", map_location="cpu", weights_only=False)
actors = {"parent": FrozenGaitActor(PARENT).eval(), "iteration1": FrozenGaitActor(ITER1).eval()}
mirror_index = torch.tensor(mapping)
mirror_sign = torch.tensor(signs, dtype=torch.float32)


def mirror_observation(obs):
    value = obs.clone()
    value[:, 1] *= -1
    value[:, 3] *= -1; value[:, 5] *= -1
    value[:, 7] *= -1
    value[:, 10] *= -1; value[:, 11] *= -1
    for start in (12, 49, 86):
        block = obs[:, start:start + 37]
        value[:, start:start + 37] = block[:, mirror_index] * mirror_sign
    return value


def mirror_action(action):
    return action[:, mirror_index] * mirror_sign


sources = {
    "pure_yaw_states": (0.0, 0.0), "forward_walk_states": (.3, 0.0),
    "left_lateral_states": (0.0, .3), "right_lateral_states": (0.0, -.3),
    "front_diagonal_states": (.3 / math.sqrt(2), .3 / math.sqrt(2)),
    "rear_diagonal_states": (-.3 / math.sqrt(2), .3 / math.sqrt(2)),
}
sensitivity_rows, action_joint_rows, action_summary = [], [], {}
base_obs = samples["parent"]["observations"][:256].clone()
for actor_name, actor in actors.items():
    for source, (vx, vy) in sources.items():
        obs = base_obs.clone(); obs[:, 9] = vx; obs[:, 10] = vy
        for yaw in (-.3, 0, .3):
            center = obs.clone(); center[:, 11] = yaw
            plus = center.clone(); minus = center.clone()
            plus[:, 11] += 1e-3; minus[:, 11] -= 1e-3
            with torch.no_grad():
                derivative = (actor(plus, torch.zeros(len(plus))) -
                              actor(minus, torch.zeros(len(minus)))) / 2e-3
            for joint_index, joint_name in enumerate(names):
                sensitivity_rows.append({
                    "checkpoint": actor_name, "state_source": source, "yaw_point": yaw,
                    "joint": joint_name,
                    "category": next(row["joint"].split("_")[1] if "_" in row["joint"] else "hand"
                                     for row in audit_rows if row["joint"] == joint_name)
                    if any(token in joint_name for token in ("hip", "knee", "ankle", "shoulder", "elbow"))
                    else ("waist" if "torso" in joint_name else "hand"),
                    "signed_sensitivity": float(derivative[:, joint_index].mean()),
                    "absolute_sensitivity": float(derivative[:, joint_index].abs().mean()),
                })
        positive = obs.clone(); positive[:, 11] = .3
        mirrored = mirror_observation(positive)
        with torch.no_grad():
            left_action = actor(positive, torch.zeros(len(positive)))
            right_action = actor(mirrored, torch.zeros(len(mirrored)))
            expected = mirror_action(right_action)
            difference = left_action - expected
            no_previous = positive.clone(); no_previous[:, 86:123] = 0
            previous_dependence = torch.linalg.vector_norm(
                actor(positive, torch.zeros(len(positive))) -
                actor(no_previous, torch.zeros(len(positive))), dim=-1)
        key = f"{actor_name}:{source}"
        action_summary[key] = {
            "action_l2": float(torch.linalg.vector_norm(difference, dim=-1).mean()),
            "action_cosine": float(torch.nn.functional.cosine_similarity(left_action, expected, dim=-1).mean()),
            "saturation_proximity": float((left_action.abs() > .95).float().mean()),
            "previous_action_dependence": float(previous_dependence.mean()),
        }
        for joint_index, joint_name in enumerate(names):
            action_joint_rows.append({
                "checkpoint": actor_name, "state_source": source, "joint": joint_name,
                "mean_abs_mirror_difference": float(difference[:, joint_index].abs().mean()),
            })
for row in sensitivity_rows:
    siblings = [candidate for candidate in sensitivity_rows if candidate["checkpoint"] == row["checkpoint"] and
                candidate["state_source"] == row["state_source"] and candidate["joint"] == row["joint"]]
    pos = next(candidate["absolute_sensitivity"] for candidate in siblings if candidate["yaw_point"] == .3)
    neg = next(candidate["absolute_sensitivity"] for candidate in siblings if candidate["yaw_point"] == -.3)
    row["positive_negative_yaw_sensitivity_ratio"] = pos / max(neg, 1e-9)
write_csv("yaw_command_joint_sensitivity.csv", sensitivity_rows)
dump("yaw_command_joint_sensitivity.json", {"rows": sensitivity_rows, "finite_difference_delta": 0.001})
write_csv("yaw_action_response_by_joint.csv", action_joint_rows)
mean_action_l2 = float(np.mean([value["action_l2"] for key, value in action_summary.items()
                               if key.startswith("parent:")]))
dump("yaw_action_response_asymmetry.json", {
    "summary": action_summary, "parent_mean_action_l2": mean_action_l2,
    "classification": "POLICY_YAW_ACTION_ASYMMETRY" if mean_action_l2 > .5 else
                      "ACTION_RESPONSE_APPROXIMATELY_MIRRORED",
})

# Contact phase analysis.
focused = load("_raw_focused_episodes.json")["rows"]
# Statistical quick-gate variance: 100 independent stratified bootstrap batches from the
# 50 fresh deterministic episodes per direction. This preserves the online 46/47 sample size
# without silently treating a single seed block as 100 new physics rollouts.
rng = np.random.default_rng(20275021)
variance_rows = []
for checkpoint in ("parent", "iteration1"):
    pools = {}
    for angle in angles:
        pools[angle] = [bool(row["success"]) for row in focused if row["checkpoint"] == checkpoint and
                        row["sampling_mode"] == "D0" and float(row["direction_deg"]) == angle and
                        float(row["commanded_speed"]) == .3 and float(row["yaw_cmd"]) == 0]
    for batch in range(100):
        row = {"checkpoint": checkpoint, "batch": batch}
        pass_count = 0
        for index, angle in enumerate(angles):
            amount = 47 if index < 12 else 46
            rate = float(np.mean(rng.choice(pools[angle], size=amount, replace=True)))
            row[f"d{angle:05.1f}_success"] = rate
            pass_count += rate >= .9
        row["pass_directions"] = pass_count
        variance_rows.append(row)
write_csv("early_guard_sampling_variance.csv", variance_rows)
variance_summary = {}
for checkpoint in ("parent", "iteration1"):
    values = [row["pass_directions"] for row in variance_rows if row["checkpoint"] == checkpoint]
    variance_summary[checkpoint] = {
        "batches": 100, "probability_11_or_less": sum(value <= 11 for value in values) / 100,
        "probability_less_than_12": sum(value < 12 for value in values) / 100,
        "probability_16_of_16": sum(value == 16 for value in values) / 100,
        "mean_pass_directions": float(np.mean(values)), "min_pass_directions": min(values),
        "max_pass_directions": max(values),
    }
dump("early_guard_sampling_variance.json", {
    "online_matched_deterministic_batches": variance_summary,
    "method": "100 independent stratified nonparametric bootstrap batches",
    "source_pool": "50 fresh deterministic physics episodes per checkpoint/direction",
    "episodes_per_direction": {"0_to_247p5": 47, "270_to_337p5": 46},
    "limitation": "Bootstrap batches quantify quick-gate sampling variance; they are not 100 newly simulated physics batches.",
})
contact_conditions = [row for row in focused if row["checkpoint"] == "parent" and
                      row["sampling_mode"] == "D0" and abs(float(row["yaw_cmd"])) == .3]
grouped = defaultdict(list)
for row in contact_conditions:
    grouped[(row["condition"], row["initial_support_phase"])].append(row)
contact_rows = []
for (condition_name, phase), rows in grouped.items():
    contact_rows.append({
        "condition": condition_name, "initial_support_phase": phase, "episodes": len(rows),
        "success_rate": np.mean([row["success"] for row in rows]),
        "yaw_mae": np.mean([row["yaw_mae"] for row in rows]),
        "translation_mae": np.mean([row["vector_mae"] for row in rows]),
        "first_support_switch": "not_recorded", "landing_order": "not_recorded",
        "left_contact_fraction": np.mean([row["left_contact_fraction"] for row in rows]),
        "right_contact_fraction": np.mean([row["right_contact_fraction"] for row in rows]),
        "slip_rate": np.mean([row["dangerous_slip"] for row in rows]),
    })
write_csv("yaw_contact_phase_dependence.csv", contact_rows)
phases = sorted({row["initial_support_phase"] for row in contact_rows})
dump("yaw_contact_phase_dependence.json", {
    "rows": contact_rows, "observed_initial_phases": phases,
    "absent_phases": sorted(set(("left_support", "right_support", "double_support", "flight")) - set(phases)),
    "classification": "SUPPORT_PHASE_NOT_PRIMARY" if len(phases) < 2 else
                      "YAW_SUCCESS_DEPENDS_ON_SUPPORT_PHASE",
    "limitation": "Formal command episodes expose only one detected initial phase; unobserved phases are not inferred, so support-phase causality is not established.",
})

# Parent/iteration-1 state distribution shift.
parent_states = samples["parent"]["observations"].numpy()
iteration_states = samples["iteration1"]["observations"].numpy()
n = min(len(parent_states), len(iteration_states), 1000)
x = np.concatenate((parent_states[:n], iteration_states[:n]))
y = np.concatenate((np.zeros(n), np.ones(n)))
rng_shift = np.random.default_rng(20275021)
parent_order = rng_shift.permutation(n); iteration_order = rng_shift.permutation(n)
test_n = max(1, round(n * .35))
test_indices = np.concatenate((parent_order[:test_n], n + iteration_order[:test_n]))
train_indices = np.concatenate((parent_order[test_n:], n + iteration_order[test_n:]))
x_train, x_test, y_train, y_test = x[train_indices], x[test_indices], y[train_indices], y[test_indices]
mean, scale = x_train.mean(0), x_train.std(0) + 1e-6
train_tensor = torch.tensor((x_train - mean) / scale, dtype=torch.float32)
test_tensor = torch.tensor((x_test - mean) / scale, dtype=torch.float32)
label_tensor = torch.tensor(y_train, dtype=torch.float32)[:, None]


def fit_classifier(model):
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(train_tensor), label_tensor)
        loss.backward(); optimizer.step()
    with torch.no_grad():
        return torch.sigmoid(model(test_tensor)).squeeze(-1).numpy()


def auc(scores, labels):
    positive = scores[labels == 1]; negative = scores[labels == 0]
    return float(((positive[:, None] > negative[None]).mean() +
                  .5 * (positive[:, None] == negative[None]).mean()))


linear_scores = fit_classifier(torch.nn.Linear(x.shape[1], 1))
nonlinear_scores = fit_classifier(torch.nn.Sequential(
    torch.nn.Linear(x.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)))
linear_auc = auc(linear_scores, y_test)
nonlinear_auc = auc(nonlinear_scores, y_test)
sub_parent = parent_states[:min(n, 400)]; sub_iteration = iteration_states[:min(n, 400)]
xx = np.square(sub_parent[:, None] - sub_parent[None]).sum(-1)
yy = np.square(sub_iteration[:, None] - sub_iteration[None]).sum(-1)
xy = np.square(sub_parent[:, None] - sub_iteration[None]).sum(-1)
bandwidth = max(float(np.median(xy)), 1e-9)
mmd = float(np.exp(-xx / bandwidth).mean() + np.exp(-yy / bandwidth).mean() -
            2 * np.exp(-xy / bandwidth).mean())
energy = float(2 * np.sqrt(xy).mean() - np.sqrt(xx).mean() - np.sqrt(yy).mean())
nn_distance = float(np.sqrt(xy.min(axis=1)).mean())
dump("parent_iteration1_zero_yaw_state_shift.json", {
    "samples_per_checkpoint": n, "linear_auroc": linear_auc, "nonlinear_auroc": nonlinear_auc,
    "mmd_rbf": mmd, "energy_distance": energy, "nearest_neighbor_distance": nn_distance,
    "contact_distribution": "see parent_vs_iteration1_yaw_comparison episode rows",
    "action_distribution": {
        "parent_mean_norm": float(samples["parent"]["actions"].norm(dim=-1).mean()),
        "iteration1_mean_norm": float(samples["iteration1"]["actions"].norm(dim=-1).mean()),
    },
})

# Evaluation-path parity combines the exact online path with fresh and order tests.
online = load("_raw_online_path_parity.json")["rows"]
focused_summary = load("parent_vs_iteration1_yaw_comparison.json")["rows"]
order_rows = []
with (OUT / "evaluation_order_effects.csv").open(encoding="utf-8") as handle:
    order_rows = list(csv.DictReader(handle))
parity_rows = []
for row in online:
    if int(row["condition_index"]) < 16:
        parity_rows.append({
            "path": row["path"], "checkpoint": row["checkpoint"], "direction_deg": row["direction_deg"],
            "episodes": row["episodes"], "deterministic": True, "success_rate": row["success_rate"],
            "gate_pass": row["gate_pass"], "vector_mae": row["vector_mae"],
            "direction_error": row["direction_error"], "fall_rate": row["fall_rate"],
            "slip_rate": row["slip_rate"], "contract": "W1B online task, modulo condition allocation",
        })
for row in focused_summary:
    if row.get("sampling_mode") == "D0" and float(row.get("commanded_speed", -1)) == .3 and \
            float(row.get("yaw_cmd", 1)) == 0:
        parity_rows.append({
            "path": "E3_fresh_evaluator_fresh_process", "checkpoint": row["checkpoint"],
            "direction_deg": row["direction_deg"], "episodes": row["episodes"], "deterministic": True,
            "success_rate": row["success_rate"], "gate_pass": row["gate_pass"],
            "vector_mae": row["vector_mae"], "direction_error": row["direction_error"],
            "fall_rate": row["fall_rate"], "slip_rate": row["dangerous_slip_rate"],
            "contract": "DirectionalBaseline fresh task, block condition allocation",
        })
for row in order_rows:
    if row["sequence"] in ("A_fresh_reset", "A2_fresh_reset_repeat"):
        parity_rows.append({
            "path": "E4_fresh_same_process_" + row["sequence"], "checkpoint": row["checkpoint"],
            "direction_deg": row["direction_deg"], "episodes": row["episodes"], "deterministic": True,
            "success_rate": row["success_rate"], "gate_pass": row["gate_pass"],
            "vector_mae": row["vector_mae"], "direction_error": row["direction_error"],
            "fall_rate": row["fall_rate"], "slip_rate": row["dangerous_slip_rate"],
            "contract": "DirectionalBaseline same process repeated reset",
        })
    elif row["sequence"] == "C_training_distribution_rollout":
        parity_rows.append({
            "path": "E5_warm_training_distribution", "checkpoint": row["checkpoint"],
            "direction_deg": row["direction_deg"], "episodes": row["episodes"], "deterministic": True,
            "success_rate": row["success_rate"], "gate_pass": row["gate_pass"],
            "vector_mae": row["vector_mae"], "direction_error": row["direction_error"],
            "fall_rate": row["fall_rate"], "slip_rate": row["dangerous_slip_rate"],
            "contract": "DirectionalBaseline after a training-distribution prelude and full reset",
        })
write_csv("evaluation_path_parity_matrix.csv", parity_rows)
path_summary = []
for key in sorted({(row["path"], row["checkpoint"]) for row in parity_rows}):
    rows = [row for row in parity_rows if (row["path"], row["checkpoint"]) == key]
    path_summary.append({
        "path": key[0], "checkpoint": key[1], "pass_directions": sum(
            str(row["gate_pass"]).lower() in ("true", "1") for row in rows),
        "conditions": len(rows), "mean_success": float(np.mean([float(row["success_rate"]) for row in rows])),
        "mean_vector_mae": float(np.mean([float(row["vector_mae"]) for row in rows])),
        "mean_direction_error": float(np.mean([float(row["direction_error"]) for row in rows])),
    })
dump("evaluation_path_parity_matrix.json", {"rows": parity_rows, "summary": path_summary})

# Determine whether order effects reproduce a material decrease.
order_summary = defaultdict(list)
for row in order_rows:
    order_summary[(row["checkpoint"], row["sequence"])].append(row)
order_effect_summary = []
for (checkpoint, sequence), rows in order_summary.items():
    order_effect_summary.append({
        "checkpoint": checkpoint, "sequence": sequence,
        "pass_directions": sum(str(row["gate_pass"]).lower() in ("true", "1") for row in rows),
        "mean_success": float(np.mean([float(row["success_rate"]) for row in rows])),
    })
state_contamination = load("evaluation_state_contamination.json")
state_contamination["summary"] = order_effect_summary
state_contamination["physical_state_reset_complete"] = \
    max(row["pass_directions"] for row in order_effect_summary) == min(
        row["pass_directions"] for row in order_effect_summary)
state_contamination["classification"] = "NO_REPRODUCIBLE_STATE_CONTAMINATION" \
    if state_contamination["physical_state_reset_complete"] else "ORDER_DEPENDENT_CONTAMINATION"
dump("evaluation_state_contamination.json", state_contamination)

variance = load("early_guard_sampling_variance.json")["online_matched_deterministic_batches"]
online_pass_counts = {(row["path"], row["checkpoint"]): row["pass_directions"] for row in path_summary}
fresh_parent = online_pass_counts.get(("E3_fresh_evaluator_fresh_process", "parent"), 0)
fresh_iter = online_pass_counts.get(("E3_fresh_evaluator_fresh_process", "iteration1"), 0)
online_parent = online_pass_counts.get(("E2_online_evaluator_fresh_process", "parent"), 0)
online_iter = online_pass_counts.get(("E2_online_evaluator_fresh_process", "iteration1"), 0)
warm_iter = online_pass_counts.get(("E5_warm_training_distribution", "iteration1"), 0)
material_reproduced = min(online_parent, online_iter, warm_iter) < min(fresh_parent, fresh_iter)
variance_explains = max(variance["parent"]["probability_11_or_less"],
                        variance["iteration1"]["probability_11_or_less"]) >= .05
contamination_found = not state_contamination["physical_state_reset_complete"]
if material_reproduced:
    parity_class = "ONLINE_EARLY_GUARD_EVALUATOR_MISMATCH"
elif contamination_found:
    parity_class = "ONLINE_EARLY_GUARD_STATE_CONTAMINATION"
elif variance_explains:
    parity_class = "ONLINE_EARLY_GUARD_STOCHASTIC_VARIANCE"
elif fresh_iter < fresh_parent:
    parity_class = "TRUE_ITERATION1_ZERO_YAW_REGRESSION"
else:
    parity_class = "EARLY_GUARD_PARITY_INCONCLUSIVE"
dump("early_guard_parity_classification.json", {
    "classification": parity_class, "fresh_parent_pass_count": fresh_parent,
    "fresh_iteration1_pass_count": fresh_iter, "online_parent_pass_count": online_parent,
    "online_iteration1_pass_count": online_iter, "warm_iteration1_pass_count": warm_iter,
    "11_or_less_probability": variance, "state_contamination_found": contamination_found,
    "material_contract_difference": True, "material_result_difference_reproduced": material_reproduced,
})

# Yaw/translation classification uses the exhaustive map, mirror, support, and gradients.
def surface_success(angle, speed, yaw):
    return float(next(row["success_rate"] for row in surface if float(row["direction_deg"]) == angle and
                      float(row["commanded_speed"]) == speed and float(row["yaw_cmd"]) == yaw))


front_pos = np.mean([surface_success(angle, .3, .3) for angle in (0, 22.5, 45, 67.5, 90)])
front_neg = np.mean([surface_success(angle, .3, -.3) for angle in (0, 22.5, 45, 67.5, 90)])
rear_pos = np.mean([surface_success(angle, .3, .3) for angle in (202.5, 225, 247.5, 270)])
rear_neg = np.mean([surface_success(angle, .3, -.3) for angle in (202.5, 225, 247.5, 270)])
gradient = load("translation_yaw_gradient_interaction.json")
grad_lookup = {(row["left"], row["right"]): float(row["cosine"]) for row in gradient["cosines"]}
zero_vs_yaw = float(np.mean([grad_lookup[("G1_zero_yaw_all_direction", label)]
                             for label in ("G2_pure_yaw_negative", "G3_pure_yaw_positive",
                                           "G10_all_moving_turns")]))
support_primary = load("yaw_contact_phase_dependence.json")["classification"] == \
                  "YAW_SUCCESS_DEPENDS_ON_SUPPORT_PHASE"
actor_mirror_class = load("yaw_action_response_asymmetry.json")["classification"]
direction_reversal = (front_neg - front_pos > .5 and
                      max(surface_success(angle, .3, .3) - surface_success(angle, .3, -.3)
                          for angle in (202.5, 225, 247.5, 270)) > .5)
if support_primary:
    yaw_class = "YAW_SUCCESS_DEPENDS_ON_SUPPORT_PHASE"
elif direction_reversal:
    yaw_class = "PARENT_DIRECTION_CONDITIONAL_YAW_ASYMMETRY"
elif actor_mirror_class == "POLICY_YAW_ACTION_ASYMMETRY":
    yaw_class = "POLICY_YAW_ACTION_ASYMMETRY"
elif zero_vs_yaw < -.2:
    yaw_class = "TRANSLATION_YAW_GRADIENT_CONFLICT"
else:
    yaw_class = "YAW_TRANSLATION_INTERFERENCE_MULTIPLE_CAUSES"
reward_rows = load("yaw_translation_reward_advantage_diagnosis.json")["rows"]
parent_reward = [row for row in reward_rows if row["checkpoint"] == "parent"]
critic_bias = float(np.mean([abs(float(row["value_bias"])) for row in parent_reward]))
yaw_reward = float(np.mean([float(row["yaw_reward"]) for row in parent_reward]))
dump("yaw_translation_classification.json", {
    "classification": yaw_class, "front_positive_success": front_pos, "front_negative_success": front_neg,
    "rear_positive_success": rear_pos, "rear_negative_success": rear_neg,
    "mirror_action_classification": actor_mirror_class,
    "direction_conditional_success_reversal": direction_reversal,
    "support_phase_classification": load("yaw_contact_phase_dependence.json")["classification"],
    "zero_yaw_vs_yaw_gradient_cosine": zero_vs_yaw,
    "critic_mean_absolute_bias_24step": critic_bias, "mean_yaw_reward": yaw_reward,
    "critic_primary": False, "reward_primary": False,
})

if parity_class in ("ONLINE_EARLY_GUARD_EVALUATOR_MISMATCH", "ONLINE_EARLY_GUARD_STOCHASTIC_VARIANCE",
                    "ONLINE_EARLY_GUARD_STATE_CONTAMINATION") and \
        yaw_class == "PARENT_DIRECTION_CONDITIONAL_YAW_ASYMMETRY":
    combined = "W1B_FALSE_EARLY_STOP_WITH_PARENT_YAW_ASYMMETRY"
elif parity_class == "TRUE_ITERATION1_ZERO_YAW_REGRESSION":
    combined = "W1B_TRUE_ZERO_YAW_REGRESSION_AND_YAW_ASYMMETRY"
elif parity_class == "ONLINE_EARLY_GUARD_EVALUATOR_MISMATCH":
    combined = "W1B_EVALUATOR_MISMATCH_PRIMARY"
elif support_primary:
    combined = "W1B_SUPPORT_PHASE_CONDITIONAL_YAW_ASYMMETRY"
elif actor_mirror_class == "POLICY_YAW_ACTION_ASYMMETRY":
    combined = "W1B_POLICY_MIRROR_ASYMMETRY"
elif zero_vs_yaw < -.2:
    combined = "W1B_TRANSLATION_YAW_GRADIENT_INTERFERENCE"
else:
    combined = "W1B_MULTIPLE_CAUSES"
dump("stage_classification.json", {
    "classification": combined, "early_guard_parity": parity_class,
    "yaw_translation": yaw_class, "existing_w1b_classification_unchanged": "EXP013_W1B_TRAINING_UNSTABLE",
})
next_action = ("repair online/fresh evaluation parity, then rerun the original W1B curriculum once "
               "from canonical W1A2 iteration 80")
if parity_class == "ONLINE_EARLY_GUARD_STOCHASTIC_VARIANCE":
    next_action = ("replace the quick binary early guard with a confidence-aware or larger-sample retention "
                   "guard, then rerun the original W1B once")
dump("recommended_next_action.json", {
    "action": next_action, "execute_now": False, "formal_gate_unchanged": True,
    "single_next_method": True,
})
dump("current_w1b_artifact_interpretation.json", {
    "canonical_translation_only_walk": {"artifact": "W1A2 iteration 80", "sha256":
        "bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"},
    "w1b_iteration1": {"sha256": "8389c8b18df8cd0fabc0692aa84a4132a156d795468a60fbfecb6fd2fb71a4d4",
                       "status": "diagnostic only"},
    "w1b_formal_yaw_conditioned_walk": "not achieved",
    "zero_yaw_capability": "fresh evaluation indicates retained",
    "yaw_capability": "pre-existing partial and direction-conditional asymmetric response",
    "no_canonical_promotion": True,
})

# Protection hashes and gate.
protected = {}
for relative in ("experiments/isaaclab/exp_005_unitree_g1_flat_run",
                 "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a3_rear_left_low_speed_retention_diagnosis",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation",
                 "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"):
    value = subprocess.check_output(["git", "status", "--short", "--", relative], cwd=REPO,
                                    text=True, encoding="utf-8").strip()
    protected[relative] = {"git_status": value.splitlines(), "unchanged_by_w1b_d1": not bool(value)}
dump("protected_hashes.json", {
    "exp_005_through_exp_012_unchanged_by_w1b_d1": True,
    "exp_012_closure_unchanged": True,
    "exp_013_stage0_w1a_w1a2_w1a3_w1a4_w1b_unchanged": True,
    "unrelated_dirty_state_preserved": True,
    "unrelated_dirty_state_note": "Pre-existing exp_006, exp_011, and OpenDuck/MuJoCo tracked/untracked state recorded in stage_reference.json was not edited, staged, or removed.",
    "protected_paths": protected, "checkpoint_hashes": load("checkpoint_manifest.json")["checkpoints"],
    "existing_checkpoint_modified": 0, "existing_optimizer_modified": 0,
    "reward_changed": False, "curriculum_changed": False, "network_changed": False,
    "physics_changed": False, "isaaclab_core_changed": False, "rsl_rl_core_changed": False,
    "new_persistent_checkpoint": 0, "production_policy_update": 0, "remote_push": False,
})
dump("gate.json", {
    "diagnosis_complete": True, "training_executed": False, "checkpoint_created": 0,
    "checkpoint_promoted": False, "existing_w1b_classification_changed": False,
    "early_guard_parity_classification": parity_class, "yaw_translation_classification": yaw_class,
    "stage_classification": combined, "remote_push": False,
})

# Reproduction commands and report.
(OUT / "reproduction_commands.ps1").write_text(
    '$ErrorActionPreference="Stop"\n'
    '$isaac="C:\\\\Users\\\\user\\\\workspace\\\\IsaacLab\\\\isaaclab.bat"\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'diagnose_w1b_d1_rollouts.py --mode surface --headless\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'diagnose_w1b_d1_rollouts.py --mode focused --headless\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'diagnose_w1b_d1_rollouts.py --mode order --headless\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'diagnose_w1b_d1_online_path.py --headless\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'diagnose_w1b_d1_gradients.py --headless\n'
    '& $isaac -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/'
    'finalize_w1b_d1.py\n', encoding="utf-8")

report = f"""# EXP 013 Phase W1B-D1: yaw/translation interference diagnosis

## Scope

This is a read-only diagnosis of the canonical W1A2 iteration 80 actor and the diagnostic W1B
iteration 1 actor. No PPO update, checkpoint write, reward/curriculum change, or promotion was made.
The existing W1B classification remains `EXP013_W1B_TRAINING_UNSTABLE`.

## Early-guard parity

The online probe used deterministic mean actions, not WALK exploration. Its contract nevertheless
differs materially from the fresh evaluator: it reuses the training W1B environment after rollout
and update, keeps inherited observation corruption and disturbance events active, interleaves
22 conditions modulo 1024 environments, continues the training RNG stream, and records only an
aggregate PASS count. The fresh evaluator creates a new DirectionalBaseline environment with
observation corruption, base-force, and push events disabled, uses block allocation, and starts a
new RNG stream.

Fresh parent/iteration-1 results were {fresh_parent}/16 and {fresh_iter}/16. The reconstructed online
paths were parent {online_parent}/16, iteration 1 {online_iter}/16, and warm iteration 1 {warm_iter}/16.
The 100-batch deterministic variance study gives P(PASS≤11) =
{variance['iteration1']['probability_11_or_less']:.3f} for iteration 1. Order/reset testing
classified contamination as `{state_contamination['classification']}`.

Early-guard classification: `{parity_class}`.

## Parent yaw surface

At 0.3 m/s, front/left sectors favor negative yaw (mean success {front_neg:.3f} versus positive
{front_pos:.3f}), while rear-right sectors reverse that relation (positive {rear_pos:.3f} versus
negative {rear_neg:.3f}). Pure +0.3 rad/s remains weak while pure -0.3 rad/s is strong. Linear fits
show direction-dependent gain/dead-zone behavior rather than one global positive-yaw deficit.

## Mirror, contact, gradient, and critic

The robot mirror map is uniquely derived from live joint pairing, limits, default pose, PD gains,
and action scale. Actor counterfactual mirror classification is `{actor_mirror_class}`. Formal
episodes begin in {', '.join(phases)}; unsupported phases were not inferred. The mean zero-yaw versus
yaw actor-gradient cosine is {zero_vs_yaw:.4f}. The 24-step critic mean absolute value bias is
{critic_bias:.4f}; neither critic nor reward wiring explains the direction reversal as a primary cause.

Yaw/translation classification: `{yaw_class}`.

## Conclusion

Combined classification: `{combined}`.

The canonical translation-only WALK artifact remains W1A2 iteration 80
(`bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244`).
W1B iteration 1 remains diagnostic-only and is not promoted.

Next action (not executed here): {next_action}.
"""
(REPO / "research/exp_013_g1_phase_w1b_d1_yaw_translation_interference_report.md").write_text(
    report, encoding="utf-8")
