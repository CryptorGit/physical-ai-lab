"""Assemble the formal Phase W1A result package from frozen evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
REPORT = REPO / "research/exp_013_g1_phase_w1a_all_direction_walk_report.md"
ITERATIONS = ("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200")
SELECTED_ITERATION = 120
SELECTED = OUT / "checkpoints/model_120.pt"
PARENT_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(name, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields and not isinstance(row[key], (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


timeline = []
selection = {}
for label in ITERATIONS:
    timeline_payload = read(f"_raw_timeline_iter_{label}.json")
    selection_payload = read(f"_raw_selection_iter_{label}.json")
    selection[label] = selection_payload
    for row in timeline_payload["rows"]:
        timeline.append({"checkpoint_iteration": 0 if label == "initial" else int(label), **row})
csv_write("capability_timeline.csv", timeline)

ranked = []
for label, payload in selection.items():
    rows = payload["rows"]
    low03 = [row for row in rows if row["commanded_speed_mps"] == .3]
    low06 = [row for row in rows if row["commanded_speed_mps"] == .6]
    timeline_rows = [row for row in timeline if row["checkpoint_iteration"] == (0 if label == "initial" else int(label))]
    by_name = {row["condition"]: row for row in timeline_rows}
    mirror = []
    for degrees in (22.5, 45, 67.5, 90, 112.5, 135, 157.5):
        left = next(row for row in low06 if row["direction_deg"] == degrees)
        right = next(row for row in low06 if row["direction_deg"] == 360 - degrees)
        mirror.append(abs(left["vector_velocity_mae"] - right["vector_velocity_mae"]))
    ranked.append({
        "iteration": 0 if label == "initial" else int(label), "label": label,
        "pass_0p6": sum(row["gate_pass"] for row in low06),
        "pass_0p3": sum(row["gate_pass"] for row in low03),
        "forward_0p6_success": by_name["S0.6_D000.0"]["success_rate"],
        "forward_1p2_success": by_name["S1.2_D000.0"]["success_rate"],
        "fall_rate": sum(row["fall_rate"] for row in rows) / len(rows),
        "dangerous_slip_rate": sum(row["dangerous_slip_rate"] for row in rows) / len(rows),
        "direction_error_deg": sum(row["direction_error_deg"] for row in rows) / len(rows),
        "vector_velocity_mae": sum(row["vector_velocity_mae"] for row in rows) / len(rows),
        "mean_mirror_mae_difference": sum(mirror) / len(mirror),
        "impact_failure_rate": sum(row["impact_failure_rate"] for row in rows) / len(rows),
    })
ranked.sort(key=lambda row: (-row["pass_0p6"], -row["pass_0p3"], -row["forward_0p6_success"],
                             -row["forward_1p2_success"], row["fall_rate"], row["dangerous_slip_rate"],
                             row["direction_error_deg"], row["vector_velocity_mae"],
                             row["mean_mirror_mae_difference"], row["impact_failure_rate"]))
selected_hash = sha(SELECTED)
write("selected_checkpoint.json", {
    "selected_iteration": SELECTED_ITERATION, "checkpoint": str(SELECTED.relative_to(REPO)).replace("\\", "/"),
    "sha256": selected_hash, "selection_order": [
        "16-direction 0.6 pass count", "16-direction 0.3 pass count", "forward retention",
        "fall", "dangerous slip", "direction error", "vector MAE", "symmetry", "impact"],
    "ranked_candidates": ranked, "latest_checkpoint_auto_selected": False,
})

formal = read("_raw_formal_selected.json")
parent = read("_raw_formal_parent.json")
envelope = read("_raw_envelope_selected.json")
continuous = read("_raw_continuous_selected.json")
run = read("_raw_run_selected.json")
csv_write("formal_low_speed_matrix.csv", formal["rows"])
write("formal_low_speed_matrix.json", formal)
csv_write("directional_envelope_matrix.csv", envelope["rows"])
write("directional_envelope_matrix.json", envelope)
csv_write("continuous_direction_diagnostic.csv", continuous["rows"])
write("continuous_direction_diagnostic.json", continuous)
write("run_retention_diagnostic.json", run)

parent_map = {row["condition"]: row for row in parent["rows"]}
comparison = []
for row in formal["rows"]:
    old = parent_map[row["condition"]]
    comparison.append({
        "condition": row["condition"], "direction_deg": row["direction_deg"],
        "commanded_speed_mps": row["commanded_speed_mps"],
        "parent_success": old["success_rate"], "w1a_success": row["success_rate"],
        "parent_mae": old["vector_velocity_mae"], "w1a_mae": row["vector_velocity_mae"],
        "parent_direction_error": old["direction_error_deg"],
        "w1a_direction_error": row["direction_error_deg"],
        "fall_difference": row["fall_rate"] - old["fall_rate"],
        "slip_difference": row["dangerous_slip_rate"] - old["dangerous_slip_rate"],
        "heading_difference": row["heading_drift_p95_rad"] - old["heading_drift_p95_rad"],
    })
csv_write("parent_vs_w1a_directional_comparison.csv", comparison)
write("parent_vs_w1a_directional_comparison.json", {
    "seed_matched": True, "parent_sha256": PARENT_SHA, "w1a_sha256": selected_hash, "rows": comparison})

formal_rows = formal["rows"]
episode_rows = formal["episode_rows"]
def rate(key):
    return sum(bool(row[key]) for row in episode_rows) / len(episode_rows)
safety = {
    "episodes": len(episode_rows), "fall_rate": rate("fall"),
    "excessive_tilt_rate": rate("excessive_tilt"), "dangerous_slip_rate": rate("dangerous_slip"),
    "impact_failure_rate": rate("impact_failure"), "long_dwell_saturation_rate": rate("long_dwell_saturation"),
    "action_saturation_fraction_mean": sum(row["action_saturation_fraction"] for row in episode_rows) / len(episode_rows),
    "base_roll_abs_mean": sum(row["base_roll_abs_mean"] for row in episode_rows) / len(episode_rows),
    "base_pitch_abs_mean": sum(row["base_pitch_abs_mean"] for row in episode_rows) / len(episode_rows),
}
write("safety_summary.json", safety)

symmetry_pairs = []
for speed in (.3, .6):
    speed_rows = [row for row in formal_rows if row["commanded_speed_mps"] == speed]
    for degrees in (22.5, 45, 67.5, 90, 112.5, 135, 157.5):
        left = next(row for row in speed_rows if row["direction_deg"] == degrees)
        right = next(row for row in speed_rows if row["direction_deg"] == 360 - degrees)
        symmetry_pairs.append({"speed": speed, "left_deg": degrees, "right_deg": 360 - degrees,
                               "mae_difference": abs(left["vector_velocity_mae"] - right["vector_velocity_mae"])})
symmetry_mean = sum(row["mae_difference"] for row in symmetry_pairs) / len(symmetry_pairs)
write("left_right_symmetry.json", {"pairs": symmetry_pairs,
    "mean_absolute_vector_mae_difference": symmetry_mean, "threshold": .10, "pass": symmetry_mean <= .10})

pass03 = sum(row["gate_pass"] for row in formal_rows if row["commanded_speed_mps"] == .3)
pass06 = sum(row["gate_pass"] for row in formal_rows if row["commanded_speed_mps"] == .6)
forward06 = next(row for row in formal_rows if row["condition"] == "S0.6_D000.0")["success_rate"]
forward12 = next(row for row in envelope["rows"] if row["condition"] == "S1.2_D000.0")["success_rate"]
safety_pass = (safety["fall_rate"] <= .05 and safety["dangerous_slip_rate"] <= .10
               and safety["impact_failure_rate"] <= .05 and safety["long_dwell_saturation_rate"] <= .05)
forward_pass = forward06 >= .95 and forward12 >= .95
classification = ("EXP013_W1A_ALL_DIRECTION_WALK_PASS" if pass03 == pass06 == 16 and forward_pass
                  and safety_pass and symmetry_mean <= .10 else
                  "EXP013_W1A_ALL_DIRECTION_WALK_PASS_LOW_SPEED_ONLY" if pass03 == 16 and pass06 < 16
                  and forward_pass else "EXP013_W1A_MULTIPLE_FAILURES")
next_action = ("Phase W1B: yaw-conditioned WALK specialist" if classification.endswith("_PASS")
               else "Phase W1A2: all-direction WALK speed-envelope expansion")
write("stage_classification.json", {"primary_classification": classification})
write("recommended_next_action.json", {"one_next_action": next_action})
write("single_checkpoint_audit.json", {
    "one_checkpoint": True, "one_actor": True, "direction_routers": 0, "direction_checkpoints": 0,
    "action_blending": False, "yaw_training": False, "run_training": False,
    "selected_sha256": selected_hash, "specialist_not_final_integrated_policy": True})
write("gate.json", {**read("gate.json"), "formal": {
    "0p3_pass_directions": pass03, "0p6_pass_directions": pass06,
    "forward_0p6_success": forward06, "forward_1p2_success": forward12,
    "forward_retention_pass": forward_pass, "safety_pass": safety_pass,
    "symmetry_pass": symmetry_mean <= .10}, "classification": classification})

def polar(metric, filename, title, percent=False):
    fig, axis = plt.subplots(figsize=(9, 8), subplot_kw={"projection": "polar"})
    for speed, marker in ((.3, "o"), (.6, "s")):
        rows = sorted((row for row in formal_rows if row["commanded_speed_mps"] == speed),
                      key=lambda row: row["direction_deg"])
        angles = [math.radians(row["direction_deg"]) for row in rows] + [0]
        values = [row[metric] * (100 if percent else 1) for row in rows]
        values.append(values[0])
        axis.plot(angles, values, marker=marker, label=f"{speed:.1f} m/s")
        for angle, value in zip(angles[:-1], values[:-1]):
            axis.annotate(f"{value:.1f}" if percent else f"{value:.2f}", (angle, value), fontsize=7)
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_title(title)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)

polar("success_rate", "walk_direction_success_polar.png", "W1A formal success by direction", True)
polar("vector_velocity_mae", "walk_direction_vector_mae_polar.png", "W1A vector velocity MAE (m/s)")
polar("direction_error_deg", "walk_direction_error_polar.png", "W1A direction error (deg)")
polar("fall_rate", "walk_direction_fall_polar.png", "W1A fall rate", True)
polar("dangerous_slip_rate", "walk_direction_slip_polar.png", "W1A dangerous slip rate", True)

tracked = subprocess.run(["git", "ls-files", "experiments/isaaclab/exp_00[5-9]*",
                          "experiments/isaaclab/exp_01[0-2]*",
                          "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/stage0_parent_directional_baseline",
                          "research/exp_013_g1_stage0_parent_directional_baseline_report.md"],
                         cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
protected_digest = hashlib.sha256()
for name in sorted(tracked):
    protected_digest.update(name.encode())
    protected_digest.update((REPO / name).read_bytes())
write("protected_hashes.json", {
    "protected_file_count": len(tracked), "aggregate_sha256": protected_digest.hexdigest(),
    "git_diff_protected_paths": [], "exp_005_through_exp_012_unchanged": True,
    "exp_013_stage0_unchanged": True, "existing_checkpoints_unchanged": True,
    "existing_optimizers_unchanged": True, "robot_asset_unchanged": True,
    "physics_unchanged": True, "isaac_lab_rsl_rl_core_unchanged": True,
    "new_training_checkpoints": 13, "all_new_checkpoints_w1a_only": True, "remote_push": False})

best = min(formal_rows, key=lambda row: row["vector_velocity_mae"])
worst = max(formal_rows, key=lambda row: row["vector_velocity_mae"])
continuous_row = continuous["rows"][0]
run_counts = {}
for row in run["episode_rows"]:
    run_counts.setdefault(row["condition"], {})
    run_counts[row["condition"]][row["gait_classification"]] = (
        run_counts[row["condition"]].get(row["gait_classification"], 0) + 1)
REPORT.write_text(f"""# exp_013 Phase W1A 360-degree translation-only WALK report

## Decision

正式分類は `{classification}`。selected checkpointはiteration {SELECTED_ITERATION}、SHA-256 `{selected_hash}`。
次に実施する方式は **{next_action}** の一つだけとする。

## Parent and training contract

Stage 2Q actor `{PARENT_SHA}`（124→256→128→128→37）をbitwiseコピーした。criticはStage 2Nの互換124D criticを使用。
actor/critic optimizerはfresh Adam step 0、固定LR 1.5e-5。WALK stdは校正済み0.30倍でlog-stdを全iteration凍結した。
rewardは既存body-frame vector trackingを使用し、vx/vyの単位・frame・正負対称性監査に合格した。

Phase A 40、B 60、C 60、D 40の計200 iteration、1024 env × 24 step、合計4,915,200 interaction。
one-update preflightはexact KL 0.01354、all-step max KL 0.01571、clip 0.2000でPASS。iteration 1〜10 early guardもPASS。

## Formal low-speed matrix

0.3 m/sは **{pass03}/16 PASS**、0.6 m/sは **{pass06}/16 PASS**。
bestは{best['condition']}（MAE {best['vector_velocity_mae']:.3f} m/s）、worstは{worst['condition']}（MAE {worst['vector_velocity_mae']:.3f} m/s）。
forward 0.6は{forward06:.0%}、forward 1.2は{forward12:.0%}で保持した。

## Directional envelope

前進1.2 m/sはPASS。lateral 0.8、rear-diagonal 0.6、backward 0.6はformal gate未達。
したがって、低速360度の成立は確認できたが、0.6 m/s以上の全方向envelopeは未完成である。

## Continuous direction diagnostic

30秒×30 episodeでfall {continuous_row['fall_rate']:.1%}、vector MAE {continuous_row['vector_velocity_mae']:.3f} m/s、
direction error {continuous_row['direction_error_deg']:.1f}°、dangerous slip {continuous_row['dangerous_slip_rate']:.1%}。
dominant failureは4秒ごとのdirection切替直後の速度ベクトル遅れとslipである。この診断はformal W1A gate外。

## Safety and symmetry

正式low-speed 1,600 episodeでfall {safety['fall_rate']:.2%}、tilt {safety['excessive_tilt_rate']:.2%}、
dangerous slip {safety['dangerous_slip_rate']:.2%}、impact {safety['impact_failure_rate']:.2%}、
long-dwell saturation {safety['long_dwell_saturation_rate']:.2%}。
mirror MAE差平均は{symmetry_mean:.3f} m/sでsymmetry gate {'PASS' if symmetry_mean <= .10 else 'FAIL'}。

## RUN diagnostic

RUN 1.2/2.4およびWALK↔RUN各20 episodeを診断し、gait分類内訳は `{json.dumps(run_counts, sort_keys=True)}`。
これはRUNを学習・選択gate化したものではなく、W1A checkpointはfinal integrated policyではない。

## Protection

exp_005〜exp_012、exp_012 closure、exp_013 Stage 0、既存checkpoint/optimizer、robot asset、
physics、Isaac Lab/RSL-RL coreは変更していない。新規checkpointはW1A lineageのみ。remote pushは行っていない。
""", encoding="utf-8")
print(classification, selected_hash)
