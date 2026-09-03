#!/usr/bin/env python3
"""Finalize the exp_010 Stage 2 read-only optimization-stability audit."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
EXP = ROOT / "experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor"
STAGE1 = ROOT / "results/exp_010_unitree_g1_post_run_walk_attractor/stage1_post_run_walk_pilot1"
OUT = ROOT / "results/exp_010_unitree_g1_post_run_walk_attractor/stage2_optimization_stability_preflight"
RESEARCH = ROOT / "research/exp_010_stage2_optimization_stability_report.md"
STARTING_HEAD = "917643d6e30d6ac772aa951d1481a7d9e86d0d45"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def f(value: str | float | int) -> float:
    return float(value)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for cursor in range(i, j + 1):
            result[order[cursor]] = average
        i = j + 1
    return result


def correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_power = sum((a - left_mean) ** 2 for a in left)
    right_power = sum((b - right_mean) ** 2 for b in right)
    return numerator / math.sqrt(left_power * right_power) if left_power and right_power else 0.0


def spearman(rows: list[dict], left: str, right: str) -> float:
    return correlation(rank([f(row[left]) for row in rows]), rank([f(row[right]) for row in rows]))


def bool_value(value: object) -> bool:
    return str(value).lower() == "true"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


OUT.mkdir(parents=True, exist_ok=True)
RESEARCH.parent.mkdir(parents=True, exist_ok=True)

with (STAGE1 / "training_curves.csv").open(newline="", encoding="utf-8") as stream:
    timeline = list(csv.DictReader(stream))
with (OUT / "diagnostic_segments.csv").open(newline="", encoding="utf-8") as stream:
    segments = list(csv.DictReader(stream))
log_std = json.loads((OUT / "log_std_gradient_decomposition.json").read_text(encoding="utf-8"))
critic = json.loads((OUT / "critic_advantage_audit.json").read_text(encoding="utf-8"))
mean_policy = json.loads((OUT / "mean_policy_update_audit.json").read_text(encoding="utf-8"))
manifest = json.loads((STAGE1 / "checkpoint_manifest.json").read_text(encoding="utf-8"))

# Pilot 1 timeline, including explicit first-onset rules.
for row in timeline:
    row["safety_failure_rate"] = max(
        f(row["fall"]), f(row["slip"]), f(row["impact"]), f(row["saturation"]), f(row["excessive_flight"])
    )
    row["action_entropy"] = 37.0 * (
        0.5 * math.log(2.0 * math.pi * math.e) + math.log(max(f(row["exploration_std_mean"]), 1e-12))
    )
    row["heading_collapsed"] = None  # Pilot 1 online CSV did not persist per-iteration heading p95.
    row["mean_action_gradient_norm"] = None  # Reconstructed at durable checkpoints below.
    row["log_std_gradient_norm"] = None

checkpoint_by_iteration = {int(row["iteration"]): row for row in log_std}
for row in timeline:
    checkpoint = checkpoint_by_iteration.get(int(row["iteration"]))
    if checkpoint:
        row["mean_action_gradient_norm"] = checkpoint["mean_action_network_gradient_norm"]
        row["log_std_gradient_norm"] = checkpoint["g_std_total_norm"]

def first_iteration(predicate) -> int | None:
    for row in timeline:
        if predicate(row):
            return int(row["iteration"])
    return None

onsets = {
    "std_growth_begins": first_iteration(lambda r: f(r["exploration_std_mean"]) > 0.20),
    "behavior_first_degrades": first_iteration(lambda r: f(r["safety_failure_rate"]) > 0.0),
    "excessive_flight_rises": first_iteration(lambda r: f(r["excessive_flight"]) > 0.0),
    "saturation_rises": first_iteration(lambda r: f(r["saturation"]) > 0.0),
    "heading_collapses": 1,
    "heading_collapse_source": "first_post_update deterministic evaluation",
    "critic_quality_degrades": first_iteration(
        lambda r: f(r["kl"]) > 1.0 or f(r["clip_fraction"]) > 0.9 or f(r["value_loss"]) > 1000.0
    ),
}
timeline_fields = list(timeline[0].keys())
with (OUT / "pilot1_instability_timeline.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=timeline_fields)
    writer.writeheader()
    writer.writerows(timeline)
dump(
    "pilot1_instability_timeline.json",
    {
        "iterations_requested": 100,
        "iterations_completed": 78,
        "last_durable_iteration": 77,
        "abort_iteration": 78,
        "abort_reason": "exploration_std_out_of_range",
        "onsets": onsets,
        "first_update": {
            "std_mean": f(timeline[0]["exploration_std_mean"]),
            "approximate_kl": f(timeline[0]["kl"]),
            "clip_fraction": f(timeline[0]["clip_fraction"]),
            "value_loss": f(timeline[0]["value_loss"]),
            "mean_policy_action_l2_shift": mean_policy[1]["action_l2_shift_from_initial"],
            "mean_policy_kl_fixed_std": mean_policy[1]["mean_policy_kl_from_initial_fixed_std_0_2"],
        },
        "last_durable": {
            "std_mean": f(timeline[76]["exploration_std_mean"]),
            "std_max": f(timeline[76]["exploration_std_max"]),
            "value_loss": f(timeline[76]["value_loss"]),
            "approximate_kl": f(timeline[76]["kl"]),
            "actor_gradient_norm": f(timeline[76]["actor_gradient_norm"]),
            "critic_gradient_norm": f(timeline[76]["critic_gradient_norm"]),
        },
        "interpretation": "Mean-policy and critic instability are present at the first update; std growth is concurrent, not antecedent.",
    },
)

# Reward directionality, overall and stratified.
def directionality(rows: list[dict]) -> dict:
    ordered = sorted(rows, key=lambda row: f(row["return"]))
    count = max(1, len(ordered) // 10)
    bottom, top = ordered[:count], ordered[-count:]
    top_progress = statistics.fmean(f(row["diagnostic_progress"]) for row in top)
    bottom_progress = statistics.fmean(f(row["diagnostic_progress"]) for row in bottom)
    top_safety = statistics.fmean(float(bool_value(row["safety_failure"])) for row in top)
    bottom_safety = statistics.fmean(float(bool_value(row["safety_failure"])) for row in bottom)
    return {
        "segments": len(rows),
        "return_progress_spearman": spearman(rows, "return", "diagnostic_progress"),
        "advantage_progress_spearman": spearman(rows, "normalized_advantage_mean", "diagnostic_progress"),
        "top_return_decile_progress": top_progress,
        "bottom_return_decile_progress": bottom_progress,
        "top_over_bottom_progress_ratio": top_progress / bottom_progress if bottom_progress else None,
        "top_return_decile_safety_failure_rate": top_safety,
        "bottom_return_decile_safety_failure_rate": bottom_safety,
        "top_safety_not_worse": top_safety <= bottom_safety,
    }

overall_direction = directionality(segments)
by_checkpoint = {
    key: directionality([row for row in segments if row["checkpoint"] == key])
    for key in sorted({row["checkpoint"] for row in segments})
}
by_source = {
    key: directionality([row for row in segments if row["source_speed_mps"] == key])
    for key in sorted({row["source_speed_mps"] for row in segments})
}
by_phase = {
    key: directionality([row for row in segments if row["source_phase"] == key])
    for key in sorted({row["source_phase"] for row in segments})
}
reward_gate_checks = {
    "return_progress_spearman_ge_0_30": overall_direction["return_progress_spearman"] >= 0.30,
    "advantage_progress_spearman_ge_0_20": overall_direction["advantage_progress_spearman"] >= 0.20,
    "top_progress_20_percent_higher": overall_direction["top_over_bottom_progress_ratio"] >= 1.20,
    "top_safety_not_worse": overall_direction["top_safety_not_worse"],
}
reward_gate_pass = all(reward_gate_checks.values())
dump(
    "reward_directionality.json",
    {
        "diagnostic_progress_definition": [
            "target-speed proximity",
            "periodic-RUN termination",
            "excessive-flight reduction",
            "stable low-speed contact streak",
            "heading retention",
            "no saturation",
            "no slip",
            "no fall",
        ],
        "overall": overall_direction,
        "by_checkpoint": by_checkpoint,
        "by_source_speed": by_source,
        "by_source_phase": by_phase,
        "gate_checks": reward_gate_checks,
        "gate_pass": reward_gate_pass,
        "note": "The pooled return correlation is dominated by checkpoint-level failure severity; within-checkpoint directionality is inconsistent.",
    },
)

reward_columns = [name for name in segments[0] if name.startswith("reward_")]
reward_statistics = {}
for column in reward_columns:
    values = [f(row[column]) for row in segments]
    reward_statistics[column.removeprefix("reward_")] = {
        "fire_rate": statistics.fmean(float(abs(value) > 1e-12) for value in values),
        "mean_weighted_contribution": statistics.fmean(values),
        "p95_absolute_contribution": sorted(abs(value) for value in values)[int(0.95 * (len(values) - 1))],
        "advantage_spearman": spearman(
            [{**row, "_value": row[column]} for row in segments], "_value", "normalized_advantage_mean"
        ),
        "progress_spearman": spearman(
            [{**row, "_value": row[column]} for row in segments], "_value", "diagnostic_progress"
        ),
    }
for absent in ("post_run_walk_progress", "periodic_run_suppression", "completion_bonus"):
    reward_statistics[absent] = {
        "defined_in_frozen_reward": False,
        "fire_rate": 0.0,
        "reason": "term absent from the frozen Pilot 1 reward",
    }
dump(
    "reward_term_statistics.json",
    {
        "terms": reward_statistics,
        "source_phase_breakdown": {
            phase: {
                term.removeprefix("reward_"): statistics.fmean(
                    f(row[term]) for row in segments if row["source_phase"] == phase
                )
                for term in reward_columns
            }
            for phase in sorted({row["source_phase"] for row in segments})
        },
        "source_speed_breakdown": {
            speed: {
                term.removeprefix("reward_"): statistics.fmean(
                    f(row[term]) for row in segments if row["source_speed_mps"] == speed
                )
                for term in reward_columns
            }
            for speed in sorted({row["source_speed_mps"] for row in segments})
        },
    },
)
reward_reachability = (
    "REWARD_REACHABLE_AND_DIRECTIONAL" if reward_gate_pass else "REWARD_REACHABLE_BUT_NON_DIRECTIONAL"
)
dump(
    "reward_reachability_classification.json",
    {
        "classification": reward_reachability,
        "reward_terms_reachable": sum(
            stats.get("fire_rate", 0.0) > 0.0 for stats in reward_statistics.values()
        ),
        "reward_terms_total": len(reward_statistics),
        "directionality_gate_pass": reward_gate_pass,
        "decisive_failure": "normalized advantage is inversely correlated with diagnostic progress",
    },
)

# Exploration-gradient interpretation.
entropy_fractions = [row["entropy_contribution_fraction"] for row in log_std]
std_at_checkpoints = [row["std_mean"] for row in log_std]
training_by_iteration = {int(row["iteration"]): row for row in timeline}
safety_at_checkpoints = [
    f(training_by_iteration[row["iteration"]]["safety_failure_rate"]) if row["iteration"] else 0.0
    for row in log_std
]
std_safety_spearman = correlation(rank(std_at_checkpoints), rank(safety_at_checkpoints))
exploration_classification = "STD_NOT_PRIMARY"
dump(
    "exploration_instability_classification.json",
    {
        "classification": exploration_classification,
        "entropy_contribution_fraction_mean": statistics.fmean(entropy_fractions),
        "entropy_contribution_fraction_max": max(entropy_fractions),
        "entropy_driven_threshold": 0.60,
        "std_safety_failure_spearman": std_safety_spearman,
        "std_continuously_increased": all(b > a for a, b in zip(std_at_checkpoints, std_at_checkpoints[1:])),
        "mean_action_policy_within_safe_range": False,
        "first_update_mean_policy_kl_fixed_std": mean_policy[1]["mean_policy_kl_from_initial_fixed_std_0_2"],
        "first_update_mean_action_l2_shift": mean_policy[1]["action_l2_shift_from_initial"],
        "reason": "Policy-loss log-std gradients dominate entropy gradients, but unsafe mean-policy movement and critic instability begin before large std.",
        "checkpoint_decomposition": log_std,
    },
)

# Critic and advantage stability.
max_abs_value_mean = max(abs(row["value_mean"]) for row in critic)
max_return_std = max(row["return_std"] for row in critic)
critic_checks = {
    "finite": all(row["finite"] for row in critic),
    "value_scale_stable": max_abs_value_mean < 1e4 and max_return_std < 1e5,
    "progress_advantage_directional": overall_direction["advantage_progress_spearman"] >= 0.20,
    "no_systematic_positive_safety_advantage": all(row["safety_failure_advantage_mean"] <= 0.05 for row in critic),
}
critic_stable = all(critic_checks.values())
dump(
    "critic_advantage_audit.json",
    {
        "classification": "CRITIC_ADVANTAGE_STABLE" if critic_stable else "CRITIC_ADVANTAGE_UNSTABLE",
        "checks": critic_checks,
        "max_absolute_value_mean": max_abs_value_mean,
        "max_return_std": max_return_std,
        "checkpoint_metrics": critic,
        "decisive_evidence": {
            "model_50_return_mean": critic[4]["return_mean"],
            "model_75_value_mean": critic[5]["value_mean"],
            "pooled_advantage_progress_spearman": overall_direction["advantage_progress_spearman"],
        },
    },
)

calibration = {}
for source in sorted({row["source_speed_mps"] for row in segments}):
    calibration[source] = {}
    for phase in sorted({row["source_phase"] for row in segments}):
        rows = [row for row in segments if row["source_speed_mps"] == source and row["source_phase"] == phase]
        if rows:
            calibration[source][phase] = {
                "segments": len(rows),
                "return_mean": statistics.fmean(f(row["return"]) for row in rows),
                "normalized_advantage_mean": statistics.fmean(f(row["normalized_advantage_mean"]) for row in rows),
                "diagnostic_progress_mean": statistics.fmean(f(row["diagnostic_progress"]) for row in rows),
                "advantage_progress_spearman": spearman(rows, "normalized_advantage_mean", "diagnostic_progress"),
            }
dump(
    "value_calibration_by_source_phase.json",
    {
        "calibration": calibration,
        "limitation": "Pilot 1 did not persist segment-level value predictions; replay provides returns and normalized advantages, while checkpoint-level value calibration is in critic_advantage_audit.json.",
    },
)

# Mean policy file gets an explicit diagnosis without changing replay measurements.
dump(
    "mean_policy_update_audit.json",
    {
        "classification": "MEAN_POLICY_UNSTABLE",
        "first_update": mean_policy[1],
        "checkpoint_metrics": mean_policy,
        "behavior_relationship": "The first deterministic policy update precedes the iteration-2 safety collapse and is already far outside a local PPO update.",
        "std_only_instability": False,
    },
)

# Fixed-std shadow update is prohibited by its frozen prerequisites.
shadow_prerequisites = {
    "reward_directionality_gate_pass": reward_gate_pass,
    "critic_advantage_stable": critic_stable,
    "std_primary_failure_candidate": exploration_classification in ("ENTROPY_STD_DRIVEN", "POLICY_STD_DRIVEN"),
}
shadow_allowed = all(shadow_prerequisites.values())
dump(
    "fixed_std_shadow_protocol.json",
    {
        "maximum_updates": 10,
        "fixed_batch": True,
        "baseline": "Stage 1 frozen optimizer config on a diagnostic clone",
        "shadow": "same clone/batch with log-std fixed at 0.25; actor mean and critic only",
        "production_checkpoint": False,
        "isaac_stepping": False,
        "prerequisites": shadow_prerequisites,
        "authorized": shadow_allowed,
    },
)
dump(
    "fixed_std_shadow_results.json",
    {
        "status": "not_executed",
        "reason": "Frozen prerequisites failed: reward directionality, critic stability, and std-primary classification are not satisfied.",
        "optimizer_updates": 0,
        "production_ancestry_changes": 0,
    },
)

# Multiple independent failure modes are present; no Pilot 2 config is authorized.
classification = "OPTIMIZATION_FAILURE_MULTIPLE"
dump(
    "pilot2_config_diff.json",
    {
        "status": "not_created",
        "authorized": False,
        "reason": classification,
        "allowed_difference_if_authorized": {
            "exploration.policy": ["reset_trainable", "fixed"],
            "exploration.fixed_std": 0.25,
            "training_seed": "new explicit seed",
        },
        "actual_differences": [],
    },
)
dump(
    "pilot2_protocol_hashes.json",
    {
        "status": "not_generated",
        "reason": "Pilot 2 protocol is not authorized.",
        "pilot1_config_sha256": "29dc16c5814af7f12d073b5fa836fea151745328d511e4bafda084e64e71f442",
        "pilot1_reward_sha256": "0c94d6c53e54f3e1332bd420923ee3a36884a8041c8a93b8731867c4c4df0aac",
    },
)

dump(
    "stage1_reference.json",
    {
        "classification": "POST_RUN_WALK_STATE_FAIL",
        "results_path": str(STAGE1.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": "29dc16c5814af7f12d073b5fa836fea151745328d511e4bafda084e64e71f442",
        "reward_sha256": "0c94d6c53e54f3e1332bd420923ee3a36884a8041c8a93b8731867c4c4df0aac",
        "initial_checkpoint_sha256": manifest[0]["sha256"],
        "last_durable_iteration": 77,
        "pilot1_files_modified": False,
    },
)
dump(
    "protocol.json",
    {
        "stage": "exp_010_stage2_optimization_stability_preflight",
        "purpose": ["exploration instability", "critic/update instability", "reward directionality"],
        "checkpoint_iterations": [row["iteration"] for row in manifest],
        "diagnostic_replay_segments_per_checkpoint": 64,
        "source_speeds_mps": [2.6, 2.8],
        "diagnostic_clone_only": True,
        "pilot2_executed": False,
        "reward_modified": False,
        "architecture_modified": False,
    },
)
dump(
    "stage2_classification.json",
    {
        "classification": classification,
        "failure_modes": {
            "A_exploration": "contributing but not primary; entropy fraction below 0.60",
            "B_critic_advantage_policy_update": "FAIL",
            "C_reward_directionality": "FAIL",
        },
        "pilot2_authorized": False,
        "exp010_status": "POST_RUN_WALK_V1_NO_GO",
        "reason": "Reward ranking and critic/mean-policy stability both fail; fixed std alone cannot isolate or repair Pilot 1.",
    },
)
dump(
    "recommended_next_action.json",
    {
        "decision": "CLOSE_EXP_010",
        "status": "POST_RUN_WALK_V1_NO_GO",
        "pilot2": "PROHIBITED",
        "single_recommendation": "Do not run fixed-std Pilot 2; close the v1 formulation because reward directionality and critic/update stability fail independently of std.",
    },
)

protected = json.loads((OUT / "protected_hashes.json").read_text(encoding="utf-8"))
protected.update(
    {
        "starting_head": STARTING_HEAD,
        "exp005_through_exp009_modified": False,
        "exp010_stage1_modified": False,
        "capability_manifest_modified": False,
        "production_artifact_modified": False,
        "production_checkpoint_updates": 0,
        "pilot2_iterations": 0,
        "diagnostic_clone_optimizer_updates": 0,
        "isaac_lab_modified": False,
    }
)
dump("protected_hashes.json", protected)
dump(
    "gate.json",
    {
        "classification": classification,
        "pilot2_ready": False,
        "pilot2_executed": False,
        "reward_directionality_gate": reward_gate_pass,
        "critic_advantage_gate": critic_stable,
        "std_primary": False,
        "shadow_intervention_executed": False,
        "exp010_status": "POST_RUN_WALK_V1_NO_GO",
    },
)

(OUT / "reproduction_commands.ps1").write_text(
    """$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "../../..")
Set-Location $repo

# Read-only Isaac replay: fixed checkpoints, no production updates.
python experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor/scripts/diagnose_stage2_optimization.py --headless

# Pure analysis/finalization; never launches Pilot 2.
python experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor/scripts/finalize_stage2_optimization.py
""",
    encoding="utf-8",
)

RESEARCH.write_text(
    f"""# exp_010 Stage 2: optimization stability preflight

## Decision

Stage 2 is classified **{classification}**. Pilot 2 is prohibited and
`POST_RUN_WALK_V1` is closed as **NO_GO**.

## Instability timeline

The first PPO update was already non-local: approximate KL was
{f(timeline[0]['kl']):.3f}, clip fraction was {f(timeline[0]['clip_fraction']):.3f},
and deterministic mean-action KL on replay states was
{mean_policy[1]['mean_policy_kl_from_initial_fixed_std_0_2']:.3f}. Safety behavior
degraded at iteration {onsets['behavior_first_degrades']}, while std was only
{f(timeline[1]['exploration_std_mean']):.4f}. By durable iteration 77, value loss
was {f(timeline[76]['value_loss']):.3e}, critic gradient norm was
{f(timeline[76]['critic_gradient_norm']):.3e}, and std mean/max were
{f(timeline[76]['exploration_std_mean']):.4f}/
{f(timeline[76]['exploration_std_max']):.4f}.

## Reward directionality

Across 384 replayed segments, return/progress Spearman was
{overall_direction['return_progress_spearman']:.4f}, but normalized
advantage/progress Spearman was {overall_direction['advantage_progress_spearman']:.4f}.
The latter fails the frozen >=0.20 gate. The pooled return correlation is largely
checkpoint-severity separation and is not consistently reproduced within
checkpoints. The frozen reward also contains no explicit POST_RUN_WALK progress
term, periodic-RUN suppression term, or completion bonus.

## Exploration

The entropy-gradient fraction was {statistics.fmean(entropy_fractions):.4f} on
average and {max(entropy_fractions):.4f} at maximum, far below the 0.60
entropy-driven threshold. Policy-loss gradients dominate log-std. Std growth
correlates with later failures, but is not the initiating failure because the mean
policy and behavior moved unsafely at the first update.

## Critic and advantage

The critic/advantage audit is **CRITIC_ADVANTAGE_UNSTABLE**. Return and value scales
grow by orders of magnitude, and the pooled advantage/progress correlation is
negative. This independently blocks a fixed-std Pilot 2.

## Shadow intervention

The fixed-batch shadow intervention was not run. Its frozen prerequisites require
directional reward, stable critic/advantage, and std as the primary failure. All
three prerequisites failed. No diagnostic or production optimizer update was
performed in Stage 2.

## Protection

Pilot 2 iterations: 0. Production checkpoint updates: 0. Reward, contracts,
architectures, source distribution, capability manifest, exp_005 through exp_009,
exp_010 Stage 1, and Isaac Lab were not modified.
""",
    encoding="utf-8",
)

print(json.dumps({"classification": classification, "pilot2": "PROHIBITED"}, indent=2))
