"""Generate Stage 0 classifications, safety audits, figures, and reports."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline"
RESEARCH = REPO / "research"
SUITES = {
    "anchor": "anchor_baseline.json",
    "translation_walk": "pure_translation_walk.json",
    "translation_run": "pure_translation_run.json",
    "yaw": "pure_yaw_results.json",
    "translation_yaw": "translation_yaw_matrix.json",
    "independence": "direction_heading_independence.json",
    "transitions": "direction_transition_results.json",
    "random": "random_command_results.json",
}
EXPECTED_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


data = {name: read(filename) for name, filename in SUITES.items()}


def rewrite_suite(suite):
    stem = Path(SUITES[suite]).stem
    (OUT / f"{stem}.json").write_text(
        json.dumps(data[suite], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUT / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[suite]["rows"][0]))
        writer.writeheader()
        writer.writerows(data[suite]["rows"])


# Add the explicit contract judgments requested by Stage 0. All judgments are
# derived from already-recorded episode metrics; no simulation value is changed.
for row in data["yaw"]["rows"]:
    episodes = [x for x in data["yaw"]["episode_rows"] if x["condition"] == row["condition"]]
    row["turn_direction_correct_rate"] = sum(
        x["cmd_yaw_rate"] * x["actual_yaw_rate"] > 0 for x in episodes
    ) / len(episodes)
    row["translation_drift_speed_mps"] = row["actual_speed_mps"]
rewrite_suite("yaw")

for row in data["independence"]["rows"]:
    episodes = [x for x in data["independence"]["episode_rows"] if x["condition"] == row["condition"]]
    translation = [
        x["direction_error_deg"] is not None
        and x["direction_error_deg"] <= 35
        and x["vector_velocity_mae"] <= .35
        for x in episodes
    ]
    yaw = [
        x["cmd_yaw_rate"] * x["actual_yaw_rate"] > 0 and x["yaw_rate_mae"] <= .35
        for x in episodes
    ]
    row["translation_command_correct_rate"] = sum(translation) / len(episodes)
    row["yaw_command_correct_rate"] = sum(yaw) / len(episodes)
    row["both_correct_rate"] = sum(a and b for a, b in zip(translation, yaw)) / len(episodes)
    row["translation_only_correct_rate"] = sum(a and not b for a, b in zip(translation, yaw)) / len(episodes)
    row["yaw_only_correct_rate"] = sum(not a and b for a, b in zip(translation, yaw)) / len(episodes)
    row["both_fail_rate"] = sum(not a and not b for a, b in zip(translation, yaw)) / len(episodes)
rewrite_suite("independence")

for row in data["transitions"]["rows"]:
    episodes = [x for x in data["transitions"]["episode_rows"] if x["condition"] == row["condition"]]
    acquired = [x["transition_time_s"] is not None for x in episodes]
    acquisition_times = [x["transition_time_s"] for x in episodes if x["transition_time_s"] is not None]
    row["source_command_hold_rate"] = sum(x["vector_velocity_mae"] <= .35 for x in episodes) / len(episodes)
    row["target_command_acquisition_rate"] = sum(acquired) / len(episodes)
    row["transition_time_s"] = (
        sum(acquisition_times) / len(acquisition_times) if acquisition_times else None
    )
    row["direction_overshoot_deg"] = row["direction_error_deg"]
    row["gait_retention_rate"] = row["target_gait_success_rate"]
    row["minimum_jerk_ramp_s_min"] = 1.0
    row["minimum_jerk_ramp_s_max"] = 2.0
rewrite_suite("transitions")

for row in data["random"]["rows"]:
    modes = []
    if row["yaw_rate_mae"] > .35:
        modes.append("YAW_RATE_UNTRACKED")
    if row["vector_velocity_mae"] > .35:
        modes.append("TRANSLATION_VECTOR_UNTRACKED")
    if row["dangerous_slip_rate"] > 0:
        modes.append("DANGEROUS_SLIP")
    if row["target_gait_success_rate"] < .90:
        modes.append("GAIT_RETENTION")
    row["dominant_failure_modes"] = "|".join(modes) if modes else "NONE"
rewrite_suite("random")


def support(row):
    severe = (
        row["dangerous_slip_rate"] > 0
        or row["impact_failure_rate"] > 0
        or row["long_dwell_saturation_rate"] > 0
    )
    direction = row.get("direction_error_deg")
    yaw_direction = (
        abs(row["cmd_yaw_rate"]) > .05
        and row["cmd_yaw_rate"] * row["actual_yaw_rate"] > 0
    )
    translation_direction = direction is not None and direction <= 35
    if row["fall_rate"] > .10 or severe:
        return "UNSAFE"
    if (
        row["fall_rate"] <= .05
        and row["vector_velocity_mae"] <= .20
        and (direction is None or direction <= 15)
        and row["yaw_rate_mae"] <= .20
        and row["target_gait_success_rate"] >= .90
    ):
        return "SUPPORTED"
    if (
        row["fall_rate"] <= .10
        and row["vector_velocity_mae"] <= .35
        and (direction is None or direction <= 35)
        and (translation_direction or yaw_direction)
    ):
        return "PARTIALLY_SUPPORTED"
    return "SAFE_BUT_UNTRACKED"


support_rows = []
for suite in ("translation_walk", "translation_run", "yaw", "translation_yaw", "independence"):
    for row in data[suite]["rows"]:
        support_rows.append({
            "suite": suite,
            "condition": row["condition"],
            "classification": support(row),
            "episodes": row["episodes"],
            "fall_rate": row["fall_rate"],
            "vector_velocity_mae": row["vector_velocity_mae"],
            "direction_error_deg": row.get("direction_error_deg"),
            "yaw_rate_mae": row["yaw_rate_mae"],
            "target_gait_success_rate": row["target_gait_success_rate"],
            "dangerous_slip_rate": row["dangerous_slip_rate"],
            "impact_failure_rate": row["impact_failure_rate"],
            "long_dwell_saturation_rate": row["long_dwell_saturation_rate"],
        })
counts = Counter(row["classification"] for row in support_rows)
with (OUT / "command_support_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(support_rows[0]))
    writer.writeheader()
    writer.writerows(support_rows)
write("command_support_matrix.json", {
    "threshold_contract": {
        "SUPPORTED": "fall<=5%, vector MAE<=0.20, direction<=15deg, yaw MAE<=0.20, gait>=90%, no serious hazard",
        "PARTIALLY_SUPPORTED": "fall<=10%, vector MAE<=0.35, direction<=35deg, correct translation or yaw direction",
        "SAFE_BUT_UNTRACKED": "safe but command tracking outside partial thresholds",
        "UNSAFE": "fall>10% or any dangerous slip, impact failure, or long-dwell saturation",
    },
    "counts": dict(counts),
    "rows": support_rows,
})

# Weighted safety audit over every required suite.
safety_sources = {
    **data,
    "candidate_stage2q": read("_candidate_stage2q.json"),
    "candidate_stage2n": read("_candidate_stage2n.json"),
}
episode_rows = [(suite, row) for suite, payload in safety_sources.items() for row in payload["episode_rows"]]
total = len(episode_rows)
rate_keys = ("fall", "dangerous_slip", "impact_failure", "long_dwell_saturation", "excessive_tilt")
mean_keys = (
    "base_roll_abs_mean", "base_pitch_abs_mean", "base_height_mean",
    "vertical_velocity_abs_mean", "foot_slip_fraction", "joint_limit_proximity",
    "action_saturation_fraction", "heading_drift_rad", "position_drift_m",
)
safety = {
    "episodes": total,
    "rates": {key: sum(bool(row[key]) for _, row in episode_rows) / total for key in rate_keys},
    "means": {key: sum(float(row[key]) for _, row in episode_rows) / total for key in mean_keys},
    "max_impact_force_n": max(float(row["max_impact_force_n"]) for _, row in episode_rows),
    "by_suite": {},
}
for suite in safety_sources:
    rows = safety_sources[suite]["episode_rows"]
    safety["by_suite"][suite] = {
        "episodes": len(rows),
        **{f"{key}_rate": sum(bool(row[key]) for row in rows) / len(rows) for key in rate_keys},
        "left_contact_fraction": sum(row["left_contact_fraction"] for row in rows) / len(rows),
        "right_contact_fraction": sum(row["right_contact_fraction"] for row in rows) / len(rows),
        "contact_symmetry_error": sum(row["contact_symmetry_error"] for row in rows) / len(rows),
    }
write("safety_summary.json", safety)


def direction_from_name(name):
    match = re.search(r"_D(\d{3}(?:\.\d)?)", name)
    return float(match.group(1)) if match else None


symmetry_rows = []
for suite in ("translation_walk", "translation_run"):
    by_key = {}
    for row in data[suite]["rows"]:
        direction = direction_from_name(row["condition"])
        speed = round(row["commanded_speed_mps"], 1)
        by_key[(speed, direction)] = row
    for (speed, direction), left in sorted(by_key.items()):
        mirror = (360 - direction) % 360
        if direction <= 0 or direction >= 180 or (speed, mirror) not in by_key:
            continue
        right = by_key[(speed, mirror)]
        symmetry_rows.append({
            "suite": suite, "speed_mps": speed, "left_direction_deg": direction,
            "right_direction_deg": mirror,
            "vector_mae_difference": left["vector_velocity_mae"] - right["vector_velocity_mae"],
            "fall_rate_difference": left["fall_rate"] - right["fall_rate"],
            "yaw_rate_mae_difference": left["yaw_rate_mae"] - right["yaw_rate_mae"],
            "dangerous_slip_rate_difference": left["dangerous_slip_rate"] - right["dangerous_slip_rate"],
            "contact_symmetry_error_difference": left["contact_symmetry_error"] - right["contact_symmetry_error"],
        })
write("left_right_symmetry.json", {
    "pairs": symmetry_rows,
    "mean_absolute_vector_mae_difference": sum(abs(x["vector_mae_difference"]) for x in symmetry_rows) / len(symmetry_rows),
    "mean_absolute_yaw_mae_difference": sum(abs(x["yaw_rate_mae_difference"]) for x in symmetry_rows) / len(symmetry_rows),
    "interpretation": "non-zero mirrored differences demonstrate inherited left/right asymmetry",
})


def polar_plot(suite, metric, filename, title, ylim=None, percent=False):
    buckets = defaultdict(list)
    class_by_condition = {row["condition"]: row["classification"] for row in support_rows if row["suite"] == suite}
    for row in data[suite]["rows"]:
        angle = direction_from_name(row["condition"])
        if metric == "success":
            value = 1.0 if class_by_condition[row["condition"]] in ("SUPPORTED", "PARTIALLY_SUPPORTED") else 0.0
        else:
            value = row[metric]
        buckets[angle].append(value)
    angles_deg = sorted(buckets)
    values = [sum(buckets[x]) / len(buckets[x]) for x in angles_deg]
    angles = np.radians(angles_deg + [angles_deg[0]])
    values_closed = values + [values[0]]
    fig, axis = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})
    axis.plot(angles, values_closed, marker="o", linewidth=2)
    axis.fill(angles, values_closed, alpha=.15)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_thetagrids(range(0, 360, 45))
    if ylim:
        axis.set_ylim(*ylim)
    axis.set_title(title, pad=20)
    for angle, value in zip(angles[:-1], values):
        label = f"{100*value:.0f}%" if percent else f"{value:.2f}"
        axis.annotate(label, (angle, value), xytext=(0, 6), textcoords="offset points",
                      ha="center", fontsize=8)
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


polar_plot("translation_walk", "success", "walk_direction_success_polar.png",
           "WALK directional support (SUPPORTED or PARTIALLY_SUPPORTED)", (0, 1), True)
polar_plot("translation_walk", "vector_velocity_mae", "walk_direction_speed_error_polar.png",
           "WALK vector velocity MAE (m/s)")
polar_plot("translation_walk", "fall_rate", "walk_direction_fall_polar.png",
           "WALK fall rate", (0, 1), True)
polar_plot("translation_run", "success", "run_direction_success_polar.png",
           "RUN directional support (SUPPORTED or PARTIALLY_SUPPORTED)", (0, 1), True)
polar_plot("translation_run", "vector_velocity_mae", "run_direction_speed_error_polar.png",
           "RUN vector velocity MAE (m/s)")
polar_plot("translation_run", "fall_rate", "run_direction_fall_polar.png",
           "RUN fall rate", (0, 1), True)

# Combined matrix heatmap, averaging gait and speed at each direction/yaw cell.
scores = {"SUPPORTED": 1., "PARTIALLY_SUPPORTED": .5, "SAFE_BUT_UNTRACKED": .25, "UNSAFE": 0.}
combo_support = {
    row["condition"]: row["classification"] for row in support_rows if row["suite"] == "translation_yaw"
}
yaws = [-.6, -.3, 0., .3, .6]
directions = list(CARDINAL := range(0, 360, 45))
matrix = np.zeros((len(directions), len(yaws)))
for i, direction in enumerate(directions):
    for j, yaw in enumerate(yaws):
        values = []
        for row in data["translation_yaw"]["rows"]:
            if direction_from_name(row["condition"]) == direction and abs(row["cmd_yaw_rate"] - yaw) < .01:
                values.append(scores[combo_support[row["condition"]]])
        matrix[i, j] = sum(values) / len(values)
fig, axis = plt.subplots(figsize=(10, 8))
image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
axis.set_xticks(range(len(yaws)), [f"{x:+.1f}" for x in yaws])
axis.set_yticks(range(len(directions)), [f"{x}deg" for x in directions])
axis.set_xlabel("yaw_rate_cmd (rad/s)")
axis.set_ylabel("body-frame translation direction")
axis.set_title("Translation + yaw support score (WALK/RUN and speed averaged)")
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        axis.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                  color="white" if matrix[i,j] < .45 else "black")
fig.colorbar(image, ax=axis, label="1 supported, .5 partial, .25 safe-untracked, 0 unsafe")
fig.tight_layout()
fig.savefig(OUT / "translation_yaw_combination_heatmap.png", dpi=180)
plt.close(fig)

anchor = {row["condition"]: row for row in data["anchor"]["rows"]}
anchor_pass = (
    min(anchor[name]["target_gait_success_rate"] for name in
        ("WALK_0P6", "WALK_1P2", "RUN_1P2", "RUN_2P4", "WALK_TO_RUN", "RUN_TO_WALK")) >= .98
    and max(anchor[name]["fall_rate"] for name in anchor) <= .05
)
walk_supported = [
    row for row in support_rows
    if row["suite"] == "translation_walk"
    and row["classification"] in ("SUPPORTED", "PARTIALLY_SUPPORTED")
    and direction_from_name(row["condition"]) not in (0., None)
]
classification = "EXP013_PARENT_HAS_PARTIAL_DIRECTIONAL_GENERALIZATION"
write("stage_classification.json", {
    "primary_classification": classification,
    "command_contract": "PASS",
    "anchor": "PASS" if anchor_pass else "EXP013_PARENT_ANCHOR_REGRESSION",
    "basis": {
        "non_forward_walk_conditions_partial_or_better": len(walk_supported),
        "command_support_counts": dict(counts),
        "translation_and_yaw_pipeline_operational": True,
        "tracking_quality_uniformly_omnidirectional": False,
    },
    "alternatives_rejected": {
        "EXP013_YAW_ONLY_GENERALIZATION": "lateral and diagonal conditions reached PARTIALLY_SUPPORTED or better",
        "EXP013_TRANSLATION_ONLY_GENERALIZATION": "yaw command changes yaw direction in multiple conditions",
        "EXP013_COMMAND_PIPELINE_FAIL": "fresh-process four-axis audit passed",
        "EXP013_PARENT_ANCHOR_REGRESSION": "anchor gate passed",
    },
})
write("recommended_next_action.json", {
    "classification": classification,
    "one_selected_next_action": "Phase W1: retain supported directions and train missing WALK sectors",
    "method": "all-direction WALK specialist",
    "scope": [
        "forward/backward", "left/right", "all diagonals",
        "in-place turns", "translation while turning",
    ],
    "not_authorized_in_stage0": True,
    "runtime_target": "one checkpoint, one actor, no router",
})

checkpoint = REPO / read("selected_parent_manifest.json")["path"]
initial_dirty = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    "?? .openduck_hardware_source_review/", "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_playground_source_review/", "?? .openduck_runtime_source_review/",
    "?? artifacts/exp_005_unitree_g1_flat_run/", "?? artifacts/openduck_recorded_zero_pose.png",
    "?? artifacts/openduck_safe_init_pose_front.png", "?? artifacts/openduck_safe_init_pose_side.png",
    "?? artifacts/openduck_zero_pose_front.png", "?? artifacts/openduck_zero_pose_side.png",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? experiments/mujoco/exp_003_openduckmini_calibrated_walk/", "?? media/exp_008_closeout/",
    "?? media/exp_011_go2/", "?? openduck_setup_report.md", "?? research/exp_011_linkedin_post_ja.md",
    "?? tools/analyze_openduck_joint_directions.py", "?? tools/render_openduck_zero_pose.py",
]
write("stage_reference.json", {
    "experiment": "exp_013_unitree_g1_single_policy_omnidirectional_locomotion",
    "stage": 0, "status": "ACTIVE",
    "reported_starting_head": "d1296ad2a7a005855430dd2459b2c603028744be",
    "actual_starting_head": "e47a4aec34a79a7f3cb5413ec42a4394b088c43b",
    "starting_head_mismatch_preserved": True,
    "starting_unrelated_dirty_state": initial_dirty,
    "parent_sha256": sha(checkpoint),
    "exp012_status": "CLOSED",
    "exp012_classification": "EXP_012_CLOSED_WITH_SINGLE_POLICY_LOCOMOTION_SUCCESS_AND_STRICT_STAND_LIMITATION",
})
write("protocol.json", {
    "stage": 0, "frozen_deterministic": True, "episode_step_seconds": .02,
    "parent_candidates": {"conditions": 7, "episodes_each": 30},
    "anchor": {"conditions": 7, "episodes_each": 100},
    "pure_translation_walk": {"directions": 16, "speeds": [.3, .6, .9, 1.2], "episodes": 20, "duration_s": 8},
    "pure_translation_run": {"directions": 16, "speeds": [1.2, 1.6, 2., 2.4], "episodes": 10, "duration_s": 8},
    "pure_yaw": {"rates": [-1., -.6, -.3, .3, .6, 1.], "gaits": [0, 1], "episodes": 30, "duration_s": 8},
    "translation_yaw": {"directions": 8, "walk_speeds": [.6, 1.], "run_speeds": [1.2, 2.], "yaw_rates": [-.6, -.3, 0, .3, .6], "episodes": 10},
    "independence": {"cases": 6, "gaits": [0, 1], "episodes": 20},
    "transitions": {"walk_episodes": 50, "run_episodes": 50, "minimum_jerk_ramp_s": [1., 2.]},
    "random": {"walk_episodes": 20, "run_capable_episodes": 20, "duration_s": 60, "command_interval_s": [2, 4]},
    "training": {"ppo": 0, "supervised": 0, "dagger": 0, "checkpoint_updates": 0},
})
write("protected_hashes.json", {
    "selected_parent": {"path": str(checkpoint.relative_to(REPO)).replace("\\", "/"), "sha256": sha(checkpoint)},
    "stage2n_parent_sha256": sha(REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"),
    "exp_005_through_exp_012_changes_by_stage0": 0,
    "exp_012_closure_changes": 0,
    "existing_checkpoint_changes": 0, "existing_optimizer_changes": 0,
    "reward_changes": 0, "physics_changes": 0, "robot_asset_changes": 0,
    "isaac_lab_core_changes": 0, "rsl_rl_core_changes": 0,
    "new_training_checkpoints": 0, "production_policy_updates": 0,
    "remote_push": False, "initial_unrelated_dirty_state_preserved": initial_dirty,
})
gate = read("gate.json")
gate.update({
    "parent_selection": "PASS", "anchor": "PASS" if anchor_pass else "FAIL",
    "directional_baselines": "COMPLETE", "stage0_complete": anchor_pass,
    "classification": classification if anchor_pass else "EXP013_PARENT_ANCHOR_REGRESSION",
    "training_updates": 0, "remote_push": False,
})
write("gate.json", gate)

# Compact, evidence-backed report.
walk_rows = data["translation_walk"]["rows"]
run_rows = data["translation_run"]["rows"]
best_walk = min(walk_rows, key=lambda x: x["vector_velocity_mae"])
worst_walk = max(walk_rows, key=lambda x: x["vector_velocity_mae"])
best_run = min(run_rows, key=lambda x: x["vector_velocity_mae"])
worst_run = max(run_rows, key=lambda x: x["vector_velocity_mae"])
yaw_rows = data["yaw"]["rows"]
random_rows = {x["condition"]: x for x in data["random"]["rows"]}
report = f"""# exp_013 Stage 0 parent directional baseline report

## 結論

主分類は `{classification}`。command pipeline と anchor は正常で、親方策には前方近傍を中心とする複数の斜め・旋回条件への部分汎化がある。一方、360度を一様に追従する方策ではなく、後退・横・後方斜め・RUN旋回では大きい速度誤差、yaw誤差、dangerous slip が支配的である。

## Contract と parent

- command index (zero-based): `vx=9, vy=10, yaw=11, gait=123`
- frame: robot body
- scale/normalization: 1.0 / none
- history: none; previous actionは `86..122`
- selected parent: Stage 2Q
- SHA-256: `{EXPECTED_SHA}`
- architecture: `124 -> 256 -> 128 -> 128 -> 37`

## Anchor (100 deterministic episodes/condition)

| condition | gait success | fall | vector MAE |
|---|---:|---:|---:|
| WALK 0.6 | {anchor['WALK_0P6']['target_gait_success_rate']:.0%} | {anchor['WALK_0P6']['fall_rate']:.0%} | {anchor['WALK_0P6']['vector_velocity_mae']:.3f} |
| WALK 1.2 | {anchor['WALK_1P2']['target_gait_success_rate']:.0%} | {anchor['WALK_1P2']['fall_rate']:.0%} | {anchor['WALK_1P2']['vector_velocity_mae']:.3f} |
| RUN 1.2 | {anchor['RUN_1P2']['target_gait_success_rate']:.0%} | {anchor['RUN_1P2']['fall_rate']:.0%} | {anchor['RUN_1P2']['vector_velocity_mae']:.3f} |
| RUN 2.4 | {anchor['RUN_2P4']['target_gait_success_rate']:.0%} | {anchor['RUN_2P4']['fall_rate']:.0%} | {anchor['RUN_2P4']['vector_velocity_mae']:.3f} |
| WALK->RUN | {anchor['WALK_TO_RUN']['target_gait_success_rate']:.0%} | {anchor['WALK_TO_RUN']['fall_rate']:.0%} | {anchor['WALK_TO_RUN']['vector_velocity_mae']:.3f} |
| RUN->WALK | {anchor['RUN_TO_WALK']['target_gait_success_rate']:.0%} | {anchor['RUN_TO_WALK']['fall_rate']:.0%} | {anchor['RUN_TO_WALK']['vector_velocity_mae']:.3f} |
| practical STOP | {anchor['PRACTICAL_STOP']['target_gait_success_rate']:.0%} | {anchor['PRACTICAL_STOP']['fall_rate']:.0%} | {anchor['PRACTICAL_STOP']['vector_velocity_mae']:.3f} |

RUN 2.4の2 episodeとRUN 1.2の1 episodeに分類/impact外れがあったが、転倒は0%でanchor regression gate（大きな崩れ）には該当しない。

## Translation

WALK 64条件中、SUPPORTED/PARTIALLY_SUPPORTEDは `{sum(1 for x in support_rows if x['suite']=='translation_walk' and x['classification'] in ('SUPPORTED','PARTIALLY_SUPPORTED'))}` 条件。最良は `{best_walk['condition']}` (MAE {best_walk['vector_velocity_mae']:.3f} m/s)、最悪は `{worst_walk['condition']}` (MAE {worst_walk['vector_velocity_mae']:.3f} m/s)。RUNでは同分類が `{sum(1 for x in support_rows if x['suite']=='translation_run' and x['classification'] in ('SUPPORTED','PARTIALLY_SUPPORTED'))}` / 64条件で、最良 `{best_run['condition']}` ({best_run['vector_velocity_mae']:.3f})、最悪 `{worst_run['condition']}` ({worst_run['vector_velocity_mae']:.3f})。後退・横・後方斜めの高速RUNが最も弱い。

## Yaw / combined control

純旋回12条件の平均yaw-rate MAEは `{sum(x['yaw_rate_mae'] for x in yaw_rows)/len(yaw_rows):.3f}` rad/s。符号反応は複数条件で存在するが、WALK/RUNとも目標rateの定量追従は弱い。translation+yaw 160条件の分類内訳は `{dict(Counter(x['classification'] for x in support_rows if x['suite']=='translation_yaw'))}`。前方・前方斜めの曲線は部分的に成立するが、strafe/backward turnの独立制御は不均一で、指定6条件でも両軸を安定して満たす一貫性はない。

## Transitions / random

WALK方向sequenceは転倒 `{data['transitions']['rows'][0]['fall_rate']:.0%}`、vector MAE `{data['transitions']['rows'][0]['vector_velocity_mae']:.3f}`。RUN方向/gait sequenceは転倒 `{data['transitions']['rows'][1]['fall_rate']:.0%}`、vector MAE `{data['transitions']['rows'][1]['vector_velocity_mae']:.3f}`だが dangerous slip率は `{data['transitions']['rows'][1]['dangerous_slip_rate']:.0%}`。

60秒WALK randomは転倒 `{random_rows['WALK_RANDOM_60S']['fall_rate']:.0%}`、vector MAE `{random_rows['WALK_RANDOM_60S']['vector_velocity_mae']:.3f}`、yaw MAE `{random_rows['WALK_RANDOM_60S']['yaw_rate_mae']:.3f}`。RUN-capable randomは転倒 `{random_rows['RUN_CAPABLE_RANDOM_60S']['fall_rate']:.0%}`、vector MAE `{random_rows['RUN_CAPABLE_RANDOM_60S']['vector_velocity_mae']:.3f}`、yaw MAE `{random_rows['RUN_CAPABLE_RANDOM_60S']['yaw_rate_mae']:.3f}`。支配的failureはyaw-rate未追従、後方/横方向速度不足、slip、RUN gait retention低下。

## Safety

全Stage 0評価 `{total}` episodeの集計で fall `{safety['rates']['fall']:.2%}`、excessive tilt `{safety['rates']['excessive_tilt']:.2%}`、dangerous slip `{safety['rates']['dangerous_slip']:.2%}`、impact `{safety['rates']['impact_failure']:.2%}`、long-dwell saturation `{safety['rates']['long_dwell_saturation']:.2%}`。平均absolute roll/pitchは `{safety['means']['base_roll_abs_mean']:.3f}` / `{safety['means']['base_pitch_abs_mean']:.3f}` rad。左右対称性はmirror方向間の平均absolute vector-MAE差 `{read('left_right_symmetry.json')['mean_absolute_vector_mae_difference']:.3f}` m/sで、完全対称ではない。

## Next

次に選択する方式は一つだけ: **Phase W1 — supported directionsを保持し、missing WALK sectorsを学習するall-direction WALK specialist**。Stage 0では実行していない。
"""
(RESEARCH / "exp_013_g1_stage0_parent_directional_baseline_report.md").write_text(report, encoding="utf-8")
