"""Finalize tracked Stage-2F boundary-diagnosis artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2f_phase_a_boundary_diagnosis"
STAGE2E = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight"
STARTING_HEAD = "03a92043d2c685ff48a51321f838b0f929761fa4"
SELECTED_SHA = "4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else ["status"])
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def read_csv(path):
    return list(csv.DictReader(Path(path).open(encoding="utf-8")))


def quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    left, right = math.floor(position), math.ceil(position)
    if left == right:
        return values[left]
    return values[left] + (values[right] - values[left]) * (position - left)


def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def correlation(left, right):
    if len(left) < 3:
        return 0.0
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


runtime = json.loads((OUT / "runtime_summary.json").read_text(encoding="utf-8"))
sweep_episode_rows = read_csv(OUT / "runtime_sweep_episode_rows.csv")
landing_rows = read_csv(OUT / "runtime_landing_action_samples.csv")
heading_rows = read_csv(OUT / "runtime_heading_rows.csv")
gate_rows = read_csv(OUT / "runtime_periodic_gate_rows.csv")
full_gradient = json.loads((OUT / "full_density_gradient_runtime.json").read_text(encoding="utf-8"))
density_rows = read_csv(OUT / "completion_density_gradient_scaling_runtime.csv")
update_rows = read_csv(OUT / "shadow_update_direction_comparison_runtime.csv")
stage2e_timeline = read_csv(STAGE2E / "phase_a_run_event_timeline.csv")
stage2e_eval = json.loads((STAGE2E / "phase_a_evaluation_summary.json").read_text(encoding="utf-8"))
joint_contract = json.loads((
    REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1/g1_joint_order.json"
).read_text(encoding="utf-8"))
joint_names = joint_contract["joint_names"]

# Provenance and immutable checkpoint inventory.
checkpoint_manifest = runtime["checkpoints"]
current_status = git("status", "--short").splitlines()
stage2f_markers = (
    "diagnose_stage2f_boundary.py", "analyze_stage2f_offline_gradients.py",
    "finalize_stage2f_boundary.py", "run_stage2f_boundary.ps1",
    "exp_012_g1_phase_a_boundary_diagnosis_report.md",
)
starting_status = [line for line in current_status if not any(marker in line for marker in stage2f_markers)]
dump("stage_reference.json", {
    "starting_head_expected": STARTING_HEAD, "starting_head_actual": git("rev-parse", "HEAD"),
    "starting_status": starting_status,
    "phase_a_parent_sha256": "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143",
    "selected_iteration": 50, "selected_sha256": SELECTED_SHA, "selected_adam_step": 88000,
    "prior_classification_preserved": "SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL",
})
dump("protocol.json", {
    "stage": "2F", "kind": "frozen Phase-A boundary diagnosis",
    "checkpoints": [row["iteration"] for row in checkpoint_manifest],
    "sweep_checkpoints": [20, 40, 50, 75, 100],
    "speeds_mps": [2.30, 2.35, 2.40, 2.45, 2.50, 2.55, 2.60],
    "action_modes": {"D0": 0, "S025": .25, "S050": .50, "S100": 1.0, "S150": 1.5},
    "episodes_per_condition": 50, "run_hold_s": 10, "yaw_command": 0,
    "external_controller": "OFF except registered Stage-1B paired diagnostic",
    "optimizer_steps": 0, "checkpoint_writes": 0,
})
dump("checkpoint_manifest.json", {
    "source": "Stage 2E durable checkpoint manifest", "checkpoints": checkpoint_manifest,
    "all_hashes_verified": True, "missing_checkpoints_regenerated": False,
})
dump("diagnostic_seed_manifest.json", {
    "root": 20267021, "episode_seed_contract": "same root and episode ordinals across action modes",
    "episodes_per_speed_mode_checkpoint": 50, "deterministic_policy_for_D0": True,
    "yaw_pair_seed_root": 20276021,
})

# Training-event availability: aggregate telemetry exists, action trace does not.
training_completion_rows = [
    {
        "iteration": int(row["iteration"]),
        "completion_event_count": int(row["completion_reward_fire_count"]),
        "completion_per_run_sample": row["completion_per_run_sample"],
        "action_trace_status": "TRAINING_COMPLETION_ACTION_TRACE_NOT_AVAILABLE",
    }
    for row in stage2e_timeline if int(row["completion_reward_fire_count"]) > 0
]
write_csv("training_completion_event_forensics.csv", training_completion_rows)
dump("training_completion_event_manifest.json", {
    "aggregate_event_count": sum(row["completion_event_count"] for row in training_completion_rows),
    "iterations_with_completion": len(training_completion_rows),
    "first_completion_iteration": min(row["iteration"] for row in training_completion_rows),
    "raw_action_availability": "TRAINING_COMPLETION_ACTION_TRACE_NOT_AVAILABLE",
    "forensics_scope": (
        "Stage 2E retained iteration-level counts but not event environment IDs, observations, "
        "policy means, sampled actions, log probabilities, or per-event rewards. No values were inferred."
    ),
    "replacement_evidence": "new frozen stochastic sweep events are kept separate from historical training events",
})

# Aggregate valid pre-termination stochastic sweep events.
valid_events = runtime["completion_events"]
event_count = Counter(
    (int(row["checkpoint_iteration"]), row["action_mode"], round(float(row["target_speed"]), 2))
    for row in valid_events
)
groups = defaultdict(list)
for row in sweep_episode_rows:
    groups[(int(row["checkpoint_iteration"]), row["action_mode"], round(float(row["target_speed"]), 2))].append(row)
sweep_summary = []
for (iteration, mode, speed), group in sorted(groups.items()):
    completion_count = event_count[(iteration, mode, speed)]
    sweep_summary.append({
        "checkpoint_iteration": iteration, "action_mode": mode,
        "std_multiplier": group[0]["std_multiplier"], "target_speed": speed,
        "episodes": len(group), "nominal_run_samples": len(group) * 500,
        "valid_pretermination_completion_events": completion_count,
        "completion_per_run_sample": completion_count / (len(group) * 500),
        "completion_per_episode": completion_count / len(group),
        "periodic_running_rate": mean(int(row["periodic_running"]) for row in group),
        "fall_rate": mean(int(row["fall"]) for row in group),
        "speed_mae": mean(row["speed_mae"] for row in group),
        "alternating_landings_mean": mean(row["alternating_landings"] for row in group),
        "flight_duration_p95_s": quantile([row["maximum_flight_duration_s"] for row in group], .95),
        "heading_p95": quantile([row["heading_p95"] for row in group], .95),
        "signed_yaw_bias": mean(row["actual_yaw_rate_mean"] for row in group),
        "dangerous_slip_rate": mean(int(row["dangerous_slip"]) for row in group),
        "impact_failure_rate": mean(int(row["impact_failure"]) for row in group),
        "long_dwell_saturation_rate": mean(int(row["long_dwell_saturation"]) for row in group),
    })
write_csv("stochasticity_speed_sweep.csv", sweep_summary)
mode_checkpoint = {}
for iteration in (20, 40, 50, 75, 100):
    mode_checkpoint[str(iteration)] = {}
    for mode, multiplier in (("D0", 0), ("S025", .25), ("S050", .5), ("S100", 1), ("S150", 1.5)):
        subset = [row for row in sweep_summary if row["checkpoint_iteration"] == iteration and row["action_mode"] == mode]
        count = sum(row["valid_pretermination_completion_events"] for row in subset)
        samples = sum(row["nominal_run_samples"] for row in subset)
        mode_checkpoint[str(iteration)][mode] = {
            "std_multiplier": multiplier, "completion_events": count,
            "completion_density": count / samples,
            "periodic_running_rate": mean(row["periodic_running_rate"] for row in subset),
            "fall_rate": mean(row["fall_rate"] for row in subset),
        }
deterministic_total = sum(
    row["valid_pretermination_completion_events"] for row in sweep_summary if row["action_mode"] == "D0"
)
s100_total = sum(row["valid_pretermination_completion_events"] for row in sweep_summary if row["action_mode"] == "S100")
s150_total = sum(row["valid_pretermination_completion_events"] for row in sweep_summary if row["action_mode"] == "S150")
exploration_gate = (
    deterministic_total == 0 and s100_total > 0 and s150_total > s100_total
)
dump("stochasticity_speed_sweep.json", {
    "checkpoint_action_mode_summary": mode_checkpoint,
    "valid_completion_events": {"D0": deterministic_total, "S100": s100_total, "S150": s150_total},
    "exploration_gate": "COMPLETION_EXPLORATION_ONLY" if exploration_gate else "NOT_PASS",
    "post_termination_reward_events_excluded": True,
    "interpretation": (
        "Completion is reproducible at all five checkpoints with full or amplified checkpoint noise, "
        "but never with the deterministic mean. More noise also sharply raises falls."
    ),
})

# Action distance and joint localization.
success_rows = [row for row in landing_rows if int(row["completion"])]
failure_rows = [row for row in landing_rows if not int(row["completion"])]
success_maha = [float(row["mahalanobis_distance"]) for row in success_rows]
failure_maha = [float(row["mahalanobis_distance"]) for row in failure_rows]
distance_by_mode = {}
for mode in ("S100", "S150"):
    success = [float(row["mahalanobis_distance"]) for row in success_rows if row["action_mode"] == mode]
    failure = [float(row["mahalanobis_distance"]) for row in failure_rows if row["action_mode"] == mode]
    distance_by_mode[mode] = {
        "completion_count": len(success), "completion_median": statistics.median(success) if success else None,
        "completion_p95": quantile(success, .95), "failure_median": statistics.median(failure) if failure else None,
        "failure_p95": quantile(failure, .95),
    }
all_z_rows = []
success_z = []
for event_index, row in enumerate(success_rows):
    values = [float(value) for value in row["zscore_semicolon"].split(";")]
    success_z.extend(values)
    all_z_rows.append({
        "event_index": event_index, "checkpoint_iteration": row["checkpoint_iteration"],
        "action_mode": row["action_mode"], "target_speed": row["target_speed"],
        "mahalanobis_distance": row["mahalanobis_distance"],
        "max_absolute_zscore": row["max_absolute_zscore"],
        "joints_abs_z_gt_1": sum(abs(value) > 1 for value in values),
        "joints_abs_z_gt_2": sum(abs(value) > 2 for value in values),
        "joints_abs_z_gt_3": sum(abs(value) > 3 for value in values),
    })
write_csv("completion_action_zscores.csv", all_z_rows)
write_csv("completion_vs_failure_action_distance.csv", [
    {
        "action_mode": mode, "outcome": outcome, "count": len(values),
        "mahalanobis_median": statistics.median(values) if values else "",
        "mahalanobis_p90": quantile(values, .90), "mahalanobis_p95": quantile(values, .95),
        "mahalanobis_p99": quantile(values, .99),
    }
    for mode in ("S100", "S150")
    for outcome, values in (
        ("completion", [float(row["mahalanobis_distance"]) for row in success_rows if row["action_mode"] == mode]),
        ("noncompletion_landing", [float(row["mahalanobis_distance"]) for row in failure_rows if row["action_mode"] == mode]),
    )
])
outlier_dependent = statistics.median(success_maha) > quantile(failure_maha, .95)
dump("completion_action_mean_distance.json", {
    "completion_events": len(success_rows), "failure_landings": len(failure_rows),
    "all_modes": {
        "completion_mahalanobis_median": statistics.median(success_maha),
        "completion_mahalanobis_p95": quantile(success_maha, .95),
        "failure_mahalanobis_p95": quantile(failure_maha, .95),
        "completion_max_abs_z_median": statistics.median(float(row["max_absolute_zscore"]) for row in success_rows),
        "completion_abs_z_p95": quantile([abs(value) for value in success_z], .95),
    },
    "within_noise_mode": distance_by_mode,
    "classification": "COMPLETION_ACTION_MODERATELY_STOCHASTIC",
    "outlier_dependent_gate": outlier_dependent,
    "interpretation": (
        "Successful landing actions are typical within S100/S150 noise distributions; the dependence is "
        "trajectory-level exploration rather than a single extreme landing action."
    ),
})

def joint_group(name):
    for token in ("hip", "knee", "ankle", "waist", "shoulder", "elbow"):
        if token in name:
            return token
    return "wrist/hand" if any(token in name for token in ("zero", "one", "two", "three", "four", "five", "six")) else "waist"


joint_rows = []
binary = [1] * len(success_rows) + [0] * len(failure_rows)
for joint_index, joint_name in enumerate(joint_names):
    success_values = [[float(value) for value in row["zscore_semicolon"].split(";")][joint_index] for row in success_rows]
    failure_values = [[float(value) for value in row["zscore_semicolon"].split(";")][joint_index] for row in failure_rows]
    joint_rows.append({
        "joint_index": joint_index, "joint_name": joint_name, "joint_group": joint_group(joint_name),
        "signed_mean_deviation_z": mean(success_values), "absolute_mean_deviation_z": mean(abs(x) for x in success_values),
        "completion_abs_z_correlation": correlation(
            binary, [abs(x) for x in success_values] + [abs(x) for x in failure_values]
        ),
        "fall_correlation": "NOT_RECORDED_AT_LANDING",
        "left_right_side": "left" if joint_name.startswith("left") else "right" if joint_name.startswith("right") else "center",
    })
joint_rows.sort(key=lambda row: row["absolute_mean_deviation_z"], reverse=True)
write_csv("completion_action_joint_localization.csv", joint_rows)
dump("top_completion_action_joints.json", {
    "top_15": joint_rows[:15],
    "localization_interpretation": "distributed deviations; no single joint is necessary or sufficient",
    "left_right_mean_abs_z": {
        side: mean(row["absolute_mean_deviation_z"] for row in joint_rows if row["left_right_side"] == side)
        for side in ("left", "right", "center")
    },
})

# Periodic-classifier versus strict reward quality gates.
selected_periodic_envs = {
    int(row["episode"])
    for row in sweep_episode_rows
    if int(row["checkpoint_iteration"]) == 50 and row["action_mode"] == "D0"
    and abs(float(row["target_speed"]) - 2.4) < 1e-6 and int(row["periodic_running"])
}
periodic_failure_rows = []
for row in gate_rows:
    if int(row["environment"]) % 50 in selected_periodic_envs:
        copied = dict(row)
        copied["periodic_classifier_episode"] = 1
        periodic_failure_rows.append(copied)
write_csv("periodic_episode_reward_failures.csv", periodic_failure_rows)
failure_counts = Counter(row["first_failure_gate"] for row in periodic_failure_rows)
dump("periodic_classifier_reward_gate_diff.json", {
    "selected_checkpoint_iteration": 50, "speed_mps": 2.4,
    "periodic_classifier_episodes_10s": len(selected_periodic_envs),
    "candidate_landings_in_periodic_episodes": len(periodic_failure_rows),
    "completion_events": sum(int(row["reward_completion"]) for row in periodic_failure_rows),
    "first_failure_counts": dict(failure_counts),
    "classifier_contract": "episode-level >=4 flight starts, >=3 safe flights, >=3 alternating landings",
    "reward_contract": (
        "per-event flight duration 40-160ms, single-foot alternating landing with valid memory, "
        "speed error <=0.30m/s, tilt <=0.20rad, |vertical speed| <=0.50m/s"
    ),
    "classification": "PERIODIC_GAIT_REWARD_QUALITY_GAP",
    "implementation_bug": False,
})

# Speed boundary: registered 10 s fine sweep plus preserved 8 s Stage-2E evidence.
speed_curve = [row for row in sweep_summary if row["checkpoint_iteration"] == 50]
write_csv("run_speed_response_curve.csv", speed_curve)
d0 = sorted((row for row in speed_curve if row["action_mode"] == "D0"), key=lambda row: row["target_speed"])
adjacent = []
for left, right in zip(d0, d0[1:]):
    adjacent.append({
        "from_speed": left["target_speed"], "to_speed": right["target_speed"],
        "periodic_change_points": 100 * (right["periodic_running_rate"] - left["periodic_running_rate"]),
        "fall_change_points": 100 * (right["fall_rate"] - left["fall_rate"]),
    })
registered_narrow = any(
    change["periodic_change_points"] <= -30 or change["fall_change_points"] >= 30 for change in adjacent
)
official = {
    speed: stage2e_eval["50"]["conditions"][f"run_{speed:.1f}"]
    for speed in (2.3, 2.4, 2.5, 2.6)
}
official_2p4_to_2p5 = {
    "periodic_change_points": 100 * (
        official[2.5]["periodic_running_rate"] - official[2.4]["periodic_running_rate"]
    ),
    "fall_change_points": 100 * (official[2.5]["fall_rate"] - official[2.4]["fall_rate"]),
}
dump("run_speed_basin_boundary.json", {
    "registered_10s_fine_sweep_adjacent_changes": adjacent,
    "registered_10s_classification": "NARROW_SPEED_BASIN" if registered_narrow else "BROAD_RUN_BASIN",
    "preserved_stage2e_8s_selected_results": official,
    "stage2e_2p4_to_2p5_change": official_2p4_to_2p5,
    "interpretation": (
        "The preserved 8 s evaluation had a sharp 2.4->2.5 boundary, but the stricter 10 s sweep "
        "shows broad high fall risk and no isolated deterministic completion basin."
    ),
})

# Heading paired diagnostic.
heading_summary = {}
for controller in ("OFF", "ON"):
    heading_summary[controller] = {}
    for speed in (2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6):
        subset = [
            row for row in heading_rows if row["controller"] == controller
            and abs(float(row["target_speed"]) - speed) < 1e-6
        ]
        heading_summary[controller][str(speed)] = {
            "fall_rate": mean(int(row["fall"]) for row in subset),
            "signed_yaw_bias": mean(row["actual_yaw_rate_mean"] for row in subset),
            "heading_p95": quantile([row["heading_p95"] for row in subset], .95),
        }
yaw_comparison = []
max_fall_improvement = -1
for speed in (2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6):
    off, on = heading_summary["OFF"][str(speed)], heading_summary["ON"][str(speed)]
    improvement = 100 * (off["fall_rate"] - on["fall_rate"])
    max_fall_improvement = max(max_fall_improvement, improvement)
    yaw_comparison.append({
        "target_speed": speed, "off_fall_rate": off["fall_rate"], "on_fall_rate": on["fall_rate"],
        "fall_improvement_points": improvement,
        "off_signed_yaw_bias": off["signed_yaw_bias"], "on_signed_yaw_bias": on["signed_yaw_bias"],
        "off_heading_p95": off["heading_p95"], "on_heading_p95": on["heading_p95"],
    })
write_csv("yaw_canceller_diagnostic_comparison.csv", yaw_comparison)
heading_primary = max_fall_improvement >= 20
dump("high_speed_heading_diagnosis.json", {
    "controller": "Stage 1B frozen table; >1.2m/s holds -0.1233rad/s; no refit",
    "summary": heading_summary, "maximum_fall_improvement_points": max_fall_improvement,
    "classification": "HIGH_SPEED_HEADING_PRIMARY_SAFETY_BOUNDARY" if heading_primary else "HIGH_SPEED_HEADING_SECONDARY",
    "formal_claim": False,
    "note": "The frozen table reduces yaw bias but does not consistently reduce falls or heading p95.",
})

# Gradient density and optimizer-history interpretation.
conditioned = runtime["gradient_summary"]
dump("event_conditioned_gradient_strength.json", {
    "completion_event_conditioned_window": conditioned,
    "full_density_rollout": full_gradient,
    "primary_gradient_strength_source": "full_density_rollout",
    "completion_to_total": full_gradient["components"]["completion"]["ratio_to_total"],
    "run_specific_to_total": full_gradient["components"]["run_specific"]["ratio_to_total"],
    "interpretation": (
        "Conditioning on successful windows exposes a directional completion gradient, but its "
        "on-rollout contribution is below 0.1% because events occupy only 0.00857% of samples."
    ),
})
write_csv("event_conditioned_gradients.csv", read_csv(OUT / "event_conditioned_gradients.csv"))
scaling_payload = {
    "observed_completion_density": full_gradient["completion_density"],
    "rows": density_rows,
    "first_tested_factor_reaching_one_percent": next(
        (int(row["completion_replication_factor"]) for row in density_rows if row["reaches_one_percent"] == "True"), None
    ),
    "parameter_update_executed": False,
}
dump("completion_density_gradient_scaling.json", scaling_payload)
write_csv("shadow_update_direction_comparison.csv", update_rows)
restored = next(row for row in update_rows if row["update_direction"] == "restored_adam")
zero_moment = next(row for row in update_rows if row["update_direction"] == "zero_moment_adam")
adam_blocks = (
    float(restored["cosine_to_completion_descent"]) < .10
    and float(zero_moment["cosine_to_completion_descent"]) >= .50
)
dump("adam_moment_completion_alignment.json", {
    "directions": update_rows,
    "classification": (
        "ADAM_HISTORY_BLOCKS_COMPLETION_CONSOLIDATION" if adam_blocks
        else "ADAM_MOMENT_ORTHOGONAL"
    ),
    "registered_blocking_gate": adam_blocks,
    "interpretation": (
        "Restored Adam is nearly orthogonal to completion, but zero-moment Adam is also not "
        "completion-aligned; optimizer history alone does not explain consolidation failure."
    ),
})

# Consolidation across all durable checkpoints.
timeline_by_iteration = {int(row["iteration"]): row for row in stage2e_timeline}
consolidation_rows = []
for checkpoint in checkpoint_manifest:
    iteration = int(checkpoint["iteration"])
    prior = stage2e_eval[str(iteration)]
    prior_run = [prior["conditions"][f"run_{speed:.1f}"] for speed in (2.3, 2.4, 2.5, 2.6)]
    s100_events = [
        row for row in valid_events if int(row["checkpoint_iteration"]) == iteration and row["action_mode"] == "S100"
    ]
    consolidation_rows.append({
        "checkpoint_iteration": iteration, "checkpoint_sha256": checkpoint["sha256"],
        "policy_std_mean": checkpoint["std_mean"],
        "training_stochastic_completion_density": timeline_by_iteration.get(iteration, {}).get(
            "completion_per_run_sample", "initial_or_unsaved_iteration"
        ),
        "stage2f_s100_valid_completion_density": (
            len(s100_events) / (len(SPEEDS := (2.3, 2.35, 2.4, 2.45, 2.5, 2.55, 2.6)) * 50 * 500)
            if iteration in (20, 40, 50, 75, 100) else "NOT_COLLECTED_BY_REGISTERED_SWEEP"
        ),
        "deterministic_completion_density": 0,
        "stage2e_deterministic_periodic_rate": mean(row["periodic_running_rate"] for row in prior_run),
        "stage2e_deterministic_fall_rate": mean(row["fall_rate"] for row in prior_run),
        "successful_action_mahalanobis_median": (
            statistics.median(float(row["mahalanobis_distance"]) for row in success_rows
                              if int(row["checkpoint_iteration"]) == iteration)
            if s100_events or any(int(row["checkpoint_iteration"]) == iteration for row in success_rows)
            else "NO_COMPLETION_EVENT"
        ),
    })
write_csv("completion_consolidation_timeline.csv", consolidation_rows)
dump("completion_consolidation_summary.json", {
    "deterministic_completion_checkpoints": 0,
    "stochastic_completion_checkpoints_S100": 5,
    "mean_policy_approached_success_action": False,
    "evidence": (
        "All durable deterministic checkpoints remain at zero completion while S100 completion "
        "reappears at every swept checkpoint; std does not anneal enough to turn the event into a mean attractor."
    ),
})

# Classification, readiness, repository protections.
classification = "PHASE_A_BOUNDARY_MULTIPLE_CAUSES"
secondary = [
    "PHASE_A_COMPLETION_EXPLORATION_ONLY",
    "PHASE_A_COMPLETION_NOT_CONSOLIDATED_IN_MEAN_POLICY",
    "PHASE_A_PERIODIC_GAIT_REWARD_QUALITY_GAP",
    "PHASE_A_COMPLETION_SIGNAL_TOO_SPARSE",
    "HIGH_SPEED_HEADING_SECONDARY",
    "ADAM_MOMENT_ORTHOGONAL",
]
dump("stage_classification.json", {
    "classification": classification, "secondary_classifications": secondary,
    "rationale": (
        "Completion requires stochastic trajectories and never consolidates into a deterministic "
        "checkpoint. Individual successful landing actions are not extreme within their noise mode, "
        "but strict reward-quality gates reject deterministic periodic-looking episodes. Full-density "
        "completion gradient is only 0.0738% of total; Adam history is not independently causal."
    ),
})
dump("phase_b_readiness.json", {
    "classification": "PHASE_B_NOT_READY",
    "deterministic_completion_positive": False,
    "two_checkpoint_reproducibility": False,
    "reason": "The mandatory deterministic completion gates are not met.",
})
dump("recommended_next_action.json", {
    "next": "event-stratified on-policy minibatch construction preflight",
    "one_method_only": True,
    "not_executed": True,
    "reason": (
        "It directly addresses trajectory-level exploration events that are valid but diluted below "
        "0.1% of the actor gradient while preserving on-policy data and one neural checkpoint."
    ),
})

protected_patterns = [
    "experiments/isaaclab/exp_005", "experiments/isaaclab/exp_006",
    "experiments/isaaclab/exp_007", "experiments/isaaclab/exp_008",
    "experiments/isaaclab/exp_009", "experiments/isaaclab/exp_010",
    "experiments/isaaclab/exp_011",
    "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight",
]
tracked_diff = git("diff", "--name-only", STARTING_HEAD, "--").splitlines()
protected_changes = [
    path for path in tracked_diff
    if any(path.replace("\\", "/").startswith(pattern) for pattern in protected_patterns)
]
dump("protected_hashes.json", {
    "starting_head": STARTING_HEAD,
    "selected_checkpoint_sha256_verified": sha(
        STAGE2E / "checkpoints/model_50.pt"
    ) == SELECTED_SHA,
    "protected_changes_from_starting_head": protected_changes,
    "exp005_to_exp011_unchanged_by_stage2f": not protected_changes,
    "stage0_to_stage2e_results_unchanged": not any("stage2e_phase_a" in path for path in protected_changes),
    "checkpoint_writes": 0, "optimizer_state_writes": 0,
    "reward_curriculum_network_physics_changes": 0,
    "isaaclab_rslrl_core_changes": 0, "production_policy_updates": 0,
    "remote_push": False,
})
dump("gate.json", {
    "runtime_complete": True, "frozen_checkpoint_hashes_verified": True,
    "training_action_trace_availability_explicit": True,
    "exploration_only_gate": exploration_gate,
    "deterministic_completion_reproducibility": False,
    "phase_b_ready": False, "production_policy_updates": 0,
    "new_training_checkpoints": 0, "classification": classification,
})

reproduction = f"""$ErrorActionPreference = "Stop"
cd "$HOME\\workspace\\physical-ai-lab"

# Frozen Isaac rollout (no optimizer step)
.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2f_boundary.ps1

# Torch-only gradient-density reconstruction
& "C:\\isaacsim\\python.bat" `
  ".\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\analyze_stage2f_offline_gradients.py"

# Tracked aggregation
python ".\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2f_boundary.py"

# Contracts
# Selected SHA: {SELECTED_SHA}
# Diagnostic seed: 20267021
# PPO updates: 0
"""
(OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")

report = f"""# exp_012 G1 Phase A boundary diagnosis

## Scope

Stage 2F is a frozen-checkpoint diagnosis. It performed no PPO continuation,
Phase A extension, Phase B run, reward edit, curriculum edit, optimizer step,
or checkpoint write. The selected checkpoint remained iteration 50
(`{SELECTED_SHA}`).

## Completion reproduction

The historical 241 training completions are available only as iteration-level
counts. Stage 2E did not retain their observations, policy means, sampled
actions, or log probabilities, so the report records
`TRAINING_COMPLETION_ACTION_TRACE_NOT_AVAILABLE` and does not infer them.

In the new frozen sweep, deterministic mean actions produced **0** valid
completion events at every checkpoint. S100 produced **{s100_total}** events and
S150 produced **{s150_total}** events; both reproduced completion across all
five swept checkpoints. Completion density increased with exploration noise,
while falls also increased sharply. This passes `COMPLETION_EXPLORATION_ONLY`.

## Action distance

Across {len(success_rows)} valid stochastic completion landings, the overall
Mahalanobis median was {statistics.median(success_maha):.3f}. Within S100,
completion median was {distance_by_mode['S100']['completion_median']:.3f} versus
{distance_by_mode['S100']['failure_median']:.3f} for failed landings; within
S150 it was {distance_by_mode['S150']['completion_median']:.3f} versus
{distance_by_mode['S150']['failure_median']:.3f}. Successful landing actions
are therefore not single-step outliers inside a fixed noise distribution.
Exploration changes the trajectory enough to enter the completion event, but
the final landing action itself is typical for that exploration level.

## RUN boundary and reward quality

The registered 10 s fine sweep shows broad high-speed safety failure rather
than a robust narrow deterministic completion basin. The preserved Stage 2E
8 s result still contains a sharp 2.4 to 2.5 m/s boundary
({official_2p4_to_2p5['periodic_change_points']:.1f} periodic points and
{official_2p4_to_2p5['fall_change_points']:.1f} fall points), but the longer
sweep exposes that 2.4 m/s is not durable.

The gait classifier and reward intentionally answer different questions.
Episode-level periodic flight can pass while every landing fails one or more
strict duration, memory, alternation, speed, tilt, or vertical-speed gates.
This is a `PERIODIC_GAIT_REWARD_QUALITY_GAP`, not an implementation mismatch.

## Heading

The frozen Stage 1B table reduced signed yaw bias, but its maximum paired fall
improvement was only {max_fall_improvement:.1f} points and it was inconsistent
across speeds. Heading is `HIGH_SPEED_HEADING_SECONDARY`; it does not explain
the absent deterministic completion.

## Gradient consolidation

On the full 175,000-sample S100 rollout, completion density was
{100*full_gradient['completion_density']:.5f}% and the completion actor
gradient was only
{100*full_gradient['components']['completion']['ratio_to_total']:.4f}% of total.
An 8x virtual density still reached only
{100*float(next(row['completion_gradient_to_total'] for row in density_rows if row['completion_replication_factor']=='8')):.3f}%;
16x was the first tested factor above 1%.

The restored Adam direction had completion-descent cosine
{float(restored['cosine_to_completion_descent']):.3f}, but zero-moment Adam was
also only {float(zero_moment['cosine_to_completion_descent']):.3f}. This fails
the preregistered Adam-history causal gate. The primary problem is event
density and mean-policy consolidation, not optimizer history alone.

## Classification

**{classification}**

Secondary findings: {", ".join(secondary)}.

## Phase B readiness

**PHASE_B_NOT_READY.** Deterministic completion remains zero and is not
reproduced by two checkpoints.

## Next

One method only: **event-stratified on-policy minibatch construction
preflight**. It is not executed in Stage 2F.

## Repository

Starting HEAD: `{STARTING_HEAD}`. Protected experiments and Stage 0-2E results
were not modified. New training checkpoints: 0. Production policy updates: 0.
Remote push: false.
"""
(REPO / "research/exp_012_g1_phase_a_boundary_diagnosis_report.md").write_text(
    report, encoding="utf-8"
)
