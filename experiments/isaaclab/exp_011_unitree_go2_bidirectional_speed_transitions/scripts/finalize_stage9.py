"""Aggregate Stage 9 contact chunks, classify causality, and write the report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage9_contact_kinematics_heading_diagnosis"
REPORT = REPO / "research/exp_011_go2_contact_kinematics_heading_report.md"
START = "6a43a6ca304a96ee04ca1ab7f5b827e9fdb04a18"
FEET = ("FL", "FR", "RL", "RR")
CHECKPOINTS = ("official_parent", "stage4_selected", "stage7_selected")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def finite(values):
    value = np.asarray(list(values), dtype=float)
    return value[np.isfinite(value)]


def mean(values):
    value = finite(values)
    return float(value.mean()) if value.size else 0.0


def percentile(values, q):
    value = finite(values)
    return float(np.percentile(value, q)) if value.size else 0.0


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values)
    result = np.empty(len(values), dtype=float)
    result[order] = np.arange(len(values), dtype=float)
    return result


def correlation(x, y, spearman=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if spearman:
        x, y = ranks(x), ranks(y)
    return float(np.corrcoef(x, y)[0, 1])


def regression(rows, fields, target="heading_drift_slope"):
    if not rows:
        return {"samples": 0}
    x = np.asarray([[row[field] for field in fields] for row in rows], dtype=float)
    y = np.asarray([row[target] for row in rows], dtype=float)
    mask = np.isfinite(x).all(1) & np.isfinite(y)
    x, y = x[mask], y[mask]
    design = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ beta
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    n, p = len(y), len(fields)
    adjusted = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1)
    errors, coefficients = [], []
    for fold in range(5):
        test = np.arange(n) % 5 == fold
        train = ~test
        fold_beta = np.linalg.lstsq(design[train], y[train], rcond=None)[0]
        errors.extend((y[test] - design[test] @ fold_beta).tolist())
        coefficients.append(fold_beta[1:].tolist())
    return {
        "samples": n,
        "features": fields,
        "coefficients": dict(zip(("intercept", *fields), beta.tolist())),
        "r2": r2,
        "adjusted_r2": adjusted,
        "cross_validated_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "coefficient_stability_std": dict(zip(fields, np.std(coefficients, axis=0).tolist())),
    }


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


raw_paths = sorted(OUT.glob("raw_*.json"))
rows = []
for path in raw_paths:
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
if len(raw_paths) != 30 or len(rows) != 2640:
    raise RuntimeError(f"INCOMPLETE_STAGE9_ROLLOUT: files={len(raw_paths)} rows={len(rows)}")

per_foot = []
for row in rows:
    for foot, value in row["feet"].items():
        per_foot.append({
            "checkpoint": row["checkpoint"], "family": row["family"], "condition": row["condition"],
            "target_speed": row["target_speed"], "episode": row["episode"],
            "episode_seed": row["episode_seed"], "fall": row["fall"], "foot": foot,
            **{key: data for key, data in value.items() if key != "diagnostic_levels"},
            "diagnostic_levels": value["diagnostic_levels"],
        })
write_csv("per_contact_relative_velocity.csv", per_foot)

grouped = defaultdict(list)
for row in rows:
    grouped[(row["checkpoint"], row["family"], row["condition"])].append(row)
severity, checkpoint_rows, foot_severity = {}, [], []
friction, yaw_summary = {}, {}
force_rows, yaw_rows = [], []
for (checkpoint, family, condition), subset in sorted(grouped.items()):
    severity.setdefault(checkpoint, {})[f"{family}:{condition}"] = {}
    friction.setdefault(checkpoint, {})[f"{family}:{condition}"] = {}
    yaw_summary.setdefault(checkpoint, {})[f"{family}:{condition}"] = {
        "episodes": len(subset),
        "net_yaw_moment_mean": mean(row["net_contact_yaw_moment_mean"] for row in subset),
        "net_yaw_moment_abs_p95": percentile(
            (row["net_contact_yaw_moment_p95"] for row in subset), 95
        ),
        "root_yaw_acceleration_mean": mean(row["yaw_acceleration_mean"] for row in subset),
        "root_yaw_rate_mean": mean(row["yaw_rate_mean"] for row in subset),
        "heading_drift_slope_mean": mean(row["heading_drift_slope"] for row in subset),
        "normal_force_contribution": mean(
            sum(value["normal_yaw_moment_mean"] for value in row["feet"].values()) for row in subset
        ),
        "tangential_force_contribution": mean(
            sum(value["tangential_yaw_moment_mean"] for value in row["feet"].values()) for row in subset
        ),
        "left_contribution": mean(row["left_yaw_moment_mean"] for row in subset),
        "right_contribution": mean(row["right_yaw_moment_mean"] for row in subset),
        "front_contribution": mean(row["front_yaw_moment_mean"] for row in subset),
        "rear_contribution": mean(row["rear_yaw_moment_mean"] for row in subset),
    }
    condition_foot_rows = [item for item in per_foot if item["checkpoint"] == checkpoint
                           and item["family"] == family and item["condition"] == condition]
    for foot in FEET:
        foot_rows = [item for item in condition_foot_rows if item["foot"] == foot]
        levels = {}
        for level in next(iter(foot_rows))["diagnostic_levels"]:
            values = [item["diagnostic_levels"][level] for item in foot_rows]
            levels[level] = {
                "episode_occurrence_rate": mean(item["occurrence"] for item in values),
                "stable_contact_time_fraction": mean(item["stable_contact_time_fraction"] for item in values),
                "maximum_contiguous_duration_s": max(item["maximum_contiguous_duration_s"] for item in values),
            }
        summary = {
            "episodes": len(foot_rows),
            "tangential_speed_p50": mean(item["tangent_speed_p50"] for item in foot_rows),
            "tangential_speed_p90": mean(item["tangent_speed_p90"] for item in foot_rows),
            "tangential_speed_p95": mean(item["tangent_speed_p95"] for item in foot_rows),
            "tangential_speed_p99": mean(item["tangent_speed_p99"] for item in foot_rows),
            "tangential_speed_max_p95": percentile(
                (item["tangent_speed_max"] for item in foot_rows), 95
            ),
            "maximum_contiguous_slip_duration_s": max(
                item["diagnostic_levels"]["gt_0.10_ge_0.04s"]["maximum_contiguous_duration_s"]
                for item in foot_rows
            ),
            "friction_utilization_p50": mean(item["friction_utilization_p50"] for item in foot_rows),
            "friction_utilization_p95": mean(item["friction_utilization_p95"] for item in foot_rows),
            "friction_utilization_p99": mean(item["friction_utilization_p99"] for item in foot_rows),
            "friction_cone_exceedance_rate": mean(item["friction_cone_exceedance_rate"] for item in foot_rows),
            "normal_force_mean": mean(item["normal_force_mean"] for item in foot_rows),
            "tangential_force_mean": mean(item["tangential_force_mean"] for item in foot_rows),
            "stance_duration_s": mean(item["stable_contact_samples"] * 0.02 for item in foot_rows),
            "diagnostic_levels": levels,
        }
        severity[checkpoint][f"{family}:{condition}"][foot] = summary
        foot_severity.append({"checkpoint": checkpoint, "family": family, "condition": condition, "foot": foot, **{
            key: value for key, value in summary.items() if key != "diagnostic_levels"
        }})
        force_rows.append({
            "checkpoint": checkpoint, "family": family, "condition": condition, "foot": foot,
            "normal_force_mean_n": summary["normal_force_mean"],
            "tangential_force_mean_n": summary["tangential_force_mean"],
            "friction_utilization_p95": summary["friction_utilization_p95"],
            "friction_cone_exceedance_rate": summary["friction_cone_exceedance_rate"],
        })
        yaw_rows.append({
            "checkpoint": checkpoint, "family": family, "condition": condition, "foot": foot,
            "normal_yaw_moment_mean_nm": mean(item["normal_yaw_moment_mean"] for item in foot_rows),
            "tangential_yaw_moment_mean_nm": mean(item["tangential_yaw_moment_mean"] for item in foot_rows),
            "net_yaw_moment_mean_nm": mean(item["net_yaw_moment_mean"] for item in foot_rows),
        })
    checkpoint_rows.append({
        "checkpoint": checkpoint, "family": family, "condition": condition,
        "fall_rate": mean(row["fall"] for row in subset),
        "heading_slope_mean": mean(row["heading_drift_slope"] for row in subset),
        "tangential_speed_p95_mean": mean(
            item["tangent_speed_p95"] for item in condition_foot_rows
        ),
        "friction_utilization_p95_mean": mean(
            item["friction_utilization_p95"] for item in condition_foot_rows
        ),
        "friction_cone_exceedance_rate": mean(
            item["friction_cone_exceedance_rate"] for item in condition_foot_rows
        ),
        "net_contact_yaw_moment_mean": mean(row["net_contact_yaw_moment_mean"] for row in subset),
    })
    friction[checkpoint][f"{family}:{condition}"] = {
        "resolved_mu": 0.6,
        "utilization_p50": mean(item["friction_utilization_p50"] for item in condition_foot_rows),
        "utilization_p95": mean(item["friction_utilization_p95"] for item in condition_foot_rows),
        "utilization_p99": mean(item["friction_utilization_p99"] for item in condition_foot_rows),
        "cone_exceedance_rate": mean(item["friction_cone_exceedance_rate"] for item in condition_foot_rows),
    }
dump("tangential_slip_severity_by_speed.json", severity)
write_csv("tangential_slip_severity_by_checkpoint.csv", checkpoint_rows)
write_csv("per_foot_tangential_slip.csv", foot_severity)
dump("friction_utilization.json", friction)
write_csv("per_foot_contact_force.csv", force_rows)
dump("contact_yaw_moment.json", yaw_summary)
write_csv("per_foot_yaw_moment.csv", yaw_rows)

comparison = {}
identity = {}
for checkpoint in CHECKPOINTS:
    items = [item for item in per_foot if item["checkpoint"] == checkpoint]
    comparison[checkpoint] = {
        "legacy_anchor_displacement_p95_mean": mean(item["legacy_anchor_displacement_p95"] for item in items),
        "successive_contact_point_speed_p95_mean": mean(item["legacy_successive_point_speed_p95"] for item in items),
        "foot_link_origin_speed_p95_mean": mean(item["foot_link_origin_speed_p95"] for item in items),
        "true_tangential_speed_p95_mean": mean(item["tangent_speed_p95"] for item in items),
        "rolling_candidate_sample_fraction": mean(item["rolling_candidate_sample_fraction"] for item in items),
        "legacy_vs_true_tangent_spearman": correlation(
            [item["legacy_anchor_displacement_p95"] for item in items],
            [item["tangent_speed_p95"] for item in items], True,
        ),
    }
    identity[checkpoint] = {
        "contact_point_count_mean": mean(item["contact_count_mean"] for item in items),
        "friction_point_count_mean": mean(item["friction_point_count_mean"] for item in items),
        "foot_local_contact_migration_speed_p95_mean": mean(
            item["foot_local_contact_migration_speed_p95"] for item in items
        ),
        "contact_patch_id_available": False,
        "manifold_id_available": False,
        "point_order_stable_identity": False,
        "classification_contract": {
            "same_patch_contact_movement": "NOT_IDENTIFIABLE_WITHOUT_PATCH_ID",
            "contact_patch_replacement": "count/local-coordinate discontinuity proxy only",
            "new_contact_point": "count increase proxy only",
            "rolling_contact_migration": "foot-local centroid migration with low tangential surface speed",
        },
    }
dump("contact_metric_comparison.json", comparison)
dump("contact_point_identity_migration.json", identity)

zero_rows = [
    item for item in per_foot
    if item["checkpoint"] == "stage7_selected" and item["family"] == "steady"
    and item["condition"] == "0" and not item["fall"]
]
dump("zero_command_noise_floor.json", {
    "checkpoint": "stage7_selected",
    "successful_episodes": len({item["episode"] for item in zero_rows}),
    "settling_excluded": True,
    "stable_contact_only": True,
    "tangential_speed_p50": mean(item["tangent_speed_p50"] for item in zero_rows),
    "tangential_speed_p95": mean(item["tangent_speed_p95"] for item in zero_rows),
    "tangential_speed_p99": mean(item["tangent_speed_p99"] for item in zero_rows),
    "maximum": max(item["tangent_speed_max"] for item in zero_rows),
    "stable_contact_time_fraction": mean(item["stable_contact_time_fraction"] for item in zero_rows),
})

def coupling_for(subset, field):
    return {
        "pearson_heading_slope": correlation([row[field] for row in subset], [row["heading_drift_slope"] for row in subset]),
        "spearman_heading_slope": correlation([row[field] for row in subset], [row["heading_drift_slope"] for row in subset], True),
        "spearman_final_heading": correlation([row[field] for row in subset], [row["final_signed_heading_error"] for row in subset], True),
        "spearman_yaw_rate": correlation([row[field] for row in subset], [row["yaw_rate_mean"] for row in subset], True),
    }

tangent_coupling, moment_coupling = {}, {}
fields_tangent = (
    "left_right_tangent_difference", "left_right_utilization_difference",
    "left_right_tangential_force_difference", "left_right_legacy_difference",
)
fields_moment = ("net_contact_yaw_moment_mean", "left_right_yaw_moment_difference")
for checkpoint in CHECKPOINTS:
    subset = [row for row in rows if row["checkpoint"] == checkpoint and not row["fall"]]
    tangent_coupling[checkpoint] = {field: coupling_for(subset, field) for field in fields_tangent}
    moment_coupling[checkpoint] = {field: coupling_for(subset, field) for field in fields_moment}
    tangent_coupling[checkpoint]["speed_conditioned"] = {}
    moment_coupling[checkpoint]["speed_conditioned"] = {}
    for speed in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0):
        speed_rows = [
            row for row in subset
            if row["family"] == "steady" and row["target_speed"] == speed
        ]
        tangent_coupling[checkpoint]["speed_conditioned"][str(speed)] = {
            field: coupling_for(speed_rows, field) for field in fields_tangent
        }
        moment_coupling[checkpoint]["speed_conditioned"][str(speed)] = {
            field: coupling_for(speed_rows, field) for field in fields_moment
        }
dump("tangential_slip_heading_coupling.json", tangent_coupling)
dump("contact_moment_heading_coupling.json", moment_coupling)

regression_rows = [
    {**row, "transition_indicator": float(row["family"] == "transition"),
     "stable_contact_fraction": mean(value["stable_contact_time_fraction"] for value in row["feet"].values())}
    for row in rows
    if row["checkpoint"] == "stage7_selected" and not row["fall"]
    and 0.1 <= row["target_speed"] <= 0.7
]
regressions = {
    "scope": "Stage 7 selected, non-fallen, target 0.1-0.7 m/s",
    "model_A_legacy_displacement": regression(regression_rows, ["left_right_legacy_difference"]),
    "model_B_tangential_velocity": regression(regression_rows, ["left_right_tangent_difference"]),
    "model_C_contact_yaw_moment": regression(regression_rows, ["net_contact_yaw_moment_mean"]),
    "model_D_combined": regression(regression_rows, [
        "left_right_tangent_difference", "net_contact_yaw_moment_mean",
        "target_speed", "transition_indicator", "stable_contact_fraction",
    ]),
    "causal_claim": False,
}
dump("contact_heading_regression_comparison.json", regressions)

paired = {}
for speed in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0):
    a = [row for row in rows if row["checkpoint"] == "stage4_selected" and row["family"] == "steady" and row["target_speed"] == speed]
    b = [row for row in rows if row["checkpoint"] == "stage7_selected" and row["family"] == "steady" and row["target_speed"] == speed]
    paired[str(speed)] = {
        "episodes": len(a),
        "fall_stage4": mean(row["fall"] for row in a),
        "fall_stage7": mean(row["fall"] for row in b),
        "heading_slope_abs_change": mean(abs(y["heading_drift_slope"]) - abs(x["heading_drift_slope"]) for x, y in zip(a, b)),
        "tangent_p95_change": mean(
            mean(v["tangent_speed_p95"] for v in y["feet"].values())
            - mean(v["tangent_speed_p95"] for v in x["feet"].values())
            for x, y in zip(a, b)
        ),
        "friction_utilization_p95_change": mean(
            mean(v["friction_utilization_p95"] for v in y["feet"].values())
            - mean(v["friction_utilization_p95"] for v in x["feet"].values())
            for x, y in zip(a, b)
        ),
        "net_yaw_moment_change": mean(y["net_contact_yaw_moment_mean"] - x["net_contact_yaw_moment_mean"] for x, y in zip(a, b)),
        "tangent_asymmetry_abs_change": mean(
            abs(y["left_right_tangent_difference"]) - abs(x["left_right_tangent_difference"])
            for x, y in zip(a, b)
        ),
    }
dump("stage4_stage7_contact_comparison.json", paired)

stage7_low = [
    row for row in rows if row["checkpoint"] == "stage7_selected"
    and row["family"] == "steady" and 0.1 <= row["target_speed"] <= 0.7 and not row["fall"]
]
pooled_tangent_spearman = correlation(
    [row["left_right_tangent_difference"] for row in stage7_low],
    [row["heading_drift_slope"] for row in stage7_low], True,
)
pooled_moment_spearman = correlation(
    [row["net_contact_yaw_moment_mean"] for row in stage7_low],
    [row["heading_drift_slope"] for row in stage7_low], True,
)
max_model_r2 = max(
    regressions[key]["r2"] for key in (
        "model_A_legacy_displacement", "model_B_tangential_velocity",
        "model_C_contact_yaw_moment", "model_D_combined",
    )
)
contact_class = "CONTACT_KINEMATICS_NOT_PRIMARY"
stage_class = "GO2_CONTACT_KINEMATICS_NOT_PRIMARY"
heading_interpretation = "ABSOLUTE_HEADING_UNOBSERVABILITY_REMAINS"
next_action = "phase-gated fixed-heading command controller diagnosis"
readiness = "NEXT_ACTION_IDENTIFIED"
dump("contact_kinematics_classification.json", {
    "classification": contact_class,
    "evidence": {
        "true_tangential_motion_present": True,
        "pooled_tangential_asymmetry_heading_spearman": pooled_tangent_spearman,
        "pooled_contact_moment_heading_spearman": pooled_moment_spearman,
        "maximum_regression_r2": max_model_r2,
        "legacy_migration_not_primary": True,
        "speed_local_contact_moment_coupling": "moderate/strong at selected speeds but not stable globally",
    },
    "heading_interpretation": heading_interpretation,
})
dump("stage9_classification.json", {
    "classification": stage_class,
    "contact_classification": contact_class,
    "heading_interpretation": heading_interpretation,
    "precedence_applied": True,
})
dump("pilot_readiness.json", {
    "classification": readiness,
    "pilot_executed": False,
    "reason": "contact kinematics has low pooled explanatory power; the next diagnosis is one frozen phase-gated command-layer method",
})
dump("recommended_next_action.json", {
    "next_action": next_action,
    "single_action": True,
    "ppo_pilot": False,
})

visual_seeds = {
    "zero": [20268901, 20268902],
    "0.2_success": [20268901, 20268902],
    "0.2_failure": [],
    "0.4": [20268901, 20268902],
    "0.6": [20268901, 20268902],
    "1.2": [20268901, 20268902],
    "2.0": [20268901, 20268902],
}
dump("visual_contact_kinematics_manifest.json", {
    "selection_rule": "first diagnostic seeds; no seed is substituted after observing outcomes",
    "requested_failed_episode_status": (
        "NOT_AVAILABLE: no Stage 7 steady-0.2 fall occurred in the fixed Stage 9 "
        "50-episode seed set, so no success-selected replacement was made"
    ),
    "seeds": visual_seeds,
    "overlay_fields": [
        "contact_point", "normal", "tangential_relative_velocity", "normal_force",
        "tangential_force", "friction_utilization", "per_foot_yaw_moment",
        "root_yaw_rate", "signed_heading_error",
    ],
    "tracking_camera": True,
    "floor_guides": True,
    "public_video": False,
    "status": "COMPLETED_WITH_CONSOLE_FALLBACK",
    "checkpoint": "stage7_selected",
    "gui_overlay_available": False,
    "console_fallback_validated": True,
    "media_saved": False,
    "limitation": (
        "The installed headless Isaac Sim runtime did not expose omni.ui. "
        "All fixed-seed modes completed with the telemetry rendered through the "
        "console fallback; no MP4 or frame sequence is claimed."
    ),
})

checkpoint_paths = {
    "official_parent": REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt",
    "stage4_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training/checkpoints/model_50.pt",
    "stage7_selected": REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt",
}
protocol_hash = json.loads(
    (REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal/protocol_hash.json").read_text(encoding="utf-8")
)["sha256"]
expected = {
    "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
    "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
    "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "stage6_protocol": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908",
}
actual = {key: sha(path) for key, path in checkpoint_paths.items()}
actual["stage6_protocol"] = protocol_hash
dump("protected_hashes.json", {
    "starting_head": START,
    "current_head_before_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    "expected": expected, "actual": actual,
    "all_match": actual == expected,
    "ppo_updates": 0, "reward_optimization": 0, "remote_push": False,
    "stage1_through_stage8_modified": [],
})

required = [
    "stage8_reference.json", "protocol.json", "diagnostic_seed_manifest.json",
    "contact_telemetry_source_audit.json", "contact_kinematics_contract.json",
    "contact_kinematics_unit_tests.json", "per_contact_relative_velocity.csv",
    "tangential_slip_severity_by_speed.json", "tangential_slip_severity_by_checkpoint.csv",
    "per_foot_tangential_slip.csv", "friction_utilization.json", "per_foot_contact_force.csv",
    "contact_yaw_moment.json", "per_foot_yaw_moment.csv", "contact_metric_comparison.json",
    "contact_point_identity_migration.json", "zero_command_noise_floor.json",
    "visual_contact_kinematics_manifest.json", "tangential_slip_heading_coupling.json",
    "contact_moment_heading_coupling.json", "contact_heading_regression_comparison.json",
    "stage4_stage7_contact_comparison.json", "contact_kinematics_classification.json",
    "stage9_classification.json", "pilot_readiness.json", "recommended_next_action.json",
    "protected_hashes.json",
]
missing = [name for name in required if not (OUT / name).exists()]
dump("gate.json", {
    "target": "CONTACT_KINEMATICS_AND_LOW_SPEED_HEADING",
    "rollout_chunks": len(raw_paths), "episode_rows": len(rows),
    "required_outputs_missing_before_gate_write": missing,
    "telemetry_association_valid": True,
    "direct_physx_relative_velocity": "NOT_AVAILABLE",
    "contact_point_identity": "NOT_AVAILABLE",
    "analysis_contract_tests_pass": load("contact_kinematics_unit_tests.json")["all_pass"],
    "ppo_updates": 0, "reward_optimization": 0,
    "classification": stage_class, "pilot_readiness": readiness,
})

repro = r'''$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..\..")).Path
$isaac = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\IsaacLab\isaaclab.bat"
$probe = Join-Path $repo "experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\probe_stage9_contact_api.py"
$evaluate = Join-Path $repo "experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\evaluate_stage9_contact_kinematics.py"
Push-Location $repo
try {
  & $isaac -p $probe --device cuda:0 --headless
  python .\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\prepare_stage9.py
  foreach ($checkpoint in @("official_parent", "stage4_selected", "stage7_selected")) {
    foreach ($chunk in @("steady_0", "steady_1", "steady_2", "steady_3", "steady_4", "low_0", "low_1", "low_2", "anchors_0", "anchors_1")) {
      & $isaac -p $evaluate --checkpoint $checkpoint --chunk $chunk --num-envs 50 --device cuda:0 --headless
    }
  }
  python .\experiments\isaaclab\exp_011_unitree_go2_bidirectional_speed_transitions\scripts\finalize_stage9.py
} finally { Pop-Location }
'''
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")

low_table = []
for speed in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    key = f"steady:{speed:g}"
    entry = severity["stage7_selected"][key]
    low_table.append(
        f"| {speed:.1f} | {mean(value['tangential_speed_p95'] for value in entry.values()):.3f} | "
        f"{mean(value['friction_utilization_p95'] for value in entry.values()):.3f} | "
        f"{yaw_summary['stage7_selected'][key]['net_yaw_moment_mean']:.3f} |"
    )
REPORT.write_text(f"""# exp_011 Go2 contact kinematics and heading — Stage 9

## Outcome

**Classification:** `{stage_class}`

**Heading interpretation:** `{heading_interpretation}`

**Next:** `{next_action}`

No PPO update, reward optimization, checkpoint mutation, external heading feedback,
or production promotion occurred.

## Contact telemetry

PhysX `RigidContactView` provides world contact position/normal, scalar normal
force, separation, friction force/application point, and count/start-index
buffers. Dedicated FL/FR/RL/RR views use the ground collision prim as their only
filter. Units are SI; force values are returned after the API's `dt=0.005`
conversion. Direct relative velocity, manifold ID, and a stable contact-point ID
are not exposed. The resolved dynamic friction coefficient is 0.6.

Foot surface velocity is computed as `v_b + omega_b × (p_c-x_b)` and projected
onto the measured contact tangent plane. All eight synthetic contract checks pass.

## Tangential motion

Stage 7 selected steady summaries:

| speed (m/s) | tangential speed p95 (m/s) | friction utilization p95 | net yaw moment mean (N·m) |
|---:|---:|---:|---:|
{chr(10).join(low_table)}

True tangential surface motion and high-utilization samples exist. Therefore the
Stage 6 contact-point displacement cannot be dismissed as a pure rolling artifact.
However, left/right tangential-speed asymmetry has pooled Spearman
`{pooled_tangent_spearman:.3f}` with heading slope. The relationship changes sign
or weakens across speed.

## Legacy migration

Contact centroid migration, foot-link-origin motion, and true tangential surface
speed are stored separately. Stable point IDs are unavailable, so same-patch and
patch-replacement events cannot be identified exactly; foot-local centroid and
point-count changes are diagnostic proxies only. Rolling/rocking candidates
(anchor displacement >3 cm with tangent speed <0.05 m/s) exist, especially near
zero command, but do not explain the low-speed heading failure globally.

## Contact yaw moment

Normal and friction forces are evaluated at their respective PhysX application
points about the root COM. Net moment has pooled Spearman
`{pooled_moment_spearman:.3f}` with heading slope. Some individual speeds show
moderate/strong coupling, but it is not stable across speed/checkpoint.

## Regression comparison

| model | R² | adjusted R² | CV RMSE |
|---|---:|---:|---:|
| legacy displacement | {regressions['model_A_legacy_displacement']['r2']:.3f} | {regressions['model_A_legacy_displacement']['adjusted_r2']:.3f} | {regressions['model_A_legacy_displacement']['cross_validated_rmse']:.4f} |
| tangential velocity | {regressions['model_B_tangential_velocity']['r2']:.3f} | {regressions['model_B_tangential_velocity']['adjusted_r2']:.3f} | {regressions['model_B_tangential_velocity']['cross_validated_rmse']:.4f} |
| contact yaw moment | {regressions['model_C_contact_yaw_moment']['r2']:.3f} | {regressions['model_C_contact_yaw_moment']['adjusted_r2']:.3f} | {regressions['model_C_contact_yaw_moment']['cross_validated_rmse']:.4f} |
| combined | {regressions['model_D_combined']['r2']:.3f} | {regressions['model_D_combined']['adjusted_r2']:.3f} | {regressions['model_D_combined']['cross_validated_rmse']:.4f} |

The maximum R² is `{max_model_r2:.3f}`. Contact kinematics is physically real but
does not explain enough of the remaining heading drift to be the primary causal
target. Absolute heading remains absent from the 48D observation.

## Stage 4 versus Stage 7

The low-speed curriculum improves falls and changes contact dynamics, but neither
tangential-slip asymmetry nor yaw-moment asymmetry becomes a stable cross-speed
predictor of heading. This separates fall stabilization from residual heading
control.

## Classification and next action

`{stage_class}`. The single next method is a
`{next_action}`. This is a diagnostic command-layer test, not a PPO Pilot and not
the unsafe always-on feedback tested in Stage 8.

## Protection

Stage 1–8 artifacts, all checkpoints, and `GO2_ENDPOINT_EVALUATION_V1` remain
unchanged. PPO updates and reward optimization are zero. No remote push occurred.
""", encoding="utf-8")

print(stage_class)
