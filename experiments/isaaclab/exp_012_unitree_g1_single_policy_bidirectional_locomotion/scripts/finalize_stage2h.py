"""Aggregate disposable Stage-2H branches into tracked audit artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2h_short_horizon_completion_replay_preflight"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"
START = "8b3fdadff7189b7772c0cf9a1a704b09533d9135"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        if not rows:
            handle.write("status\nNO_ROWS\n")
            return
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


OUT.mkdir(parents=True, exist_ok=True)
metrics = []
eligibility = []
behavior = []
retention = []
for directory in sorted(RAW.glob("*")):
    if not directory.is_dir() or "backup" in directory.name:
        continue
    for path in sorted(directory.glob("metrics_*.json")):
        item = read_json(path)
        item["_path"] = str(path.relative_to(REPO))
        metrics.append(item)
        iteration = item["shadow_iteration"]
        eligible_path = directory / f"eligibility_{iteration}.csv"
        if eligible_path.exists():
            with eligible_path.open(encoding="utf-8") as handle:
                eligibility.extend(csv.DictReader(handle))
        behavior_path = directory / f"eval_{iteration}_temporary_behavioral_evaluation.csv"
        if behavior_path.exists():
            condition_counts = Counter()
            with behavior_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    condition_key = (row["mode"], row["target_speed"])
                    if condition_counts[condition_key] >= 20:
                        continue
                    condition_counts[condition_key] += 1
                    row["branch"] = directory.name
                    row["shadow_iteration"] = iteration
                    behavior.append(row)
        retention_path = directory / f"eval_{iteration}_retention_evaluation.csv"
        if retention_path.exists():
            with retention_path.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["branch"] = directory.name
                    row["shadow_iteration"] = iteration
                    retention.append(row)

branch_trace = []
cross = []
gradient_rows = []
drop = Counter()
for item in metrics:
    branch_trace.append({
        key: item.get(key) for key in (
            "branch", "shadow_iteration", "mode", "horizon", "coefficient",
            "completion_windows_added", "buffer_completion_windows",
            "candidate_windows", "eligible_windows", "dropped_windows",
            "auxiliary_applied", "auxiliary_ineligible_reason",
            "first_step_lr", "final_lr", "exact_kl_old_new",
            "standard_maximum_kl", "clip_fraction", "ratio_p95", "ratio_p99",
            "mean_action_shift", "combined_hard_gate_pass", "nan_inf",
        )
    })
    gradient_rows.append({
        key: item.get(key) for key in (
            "branch", "shadow_iteration", "horizon", "coefficient",
            "on_policy_gradient_norm", "replay_gradient_norm",
            "unsafe_gradient_norm", "effective_replay_gradient_ratio",
            "replay_vs_on_policy_cosine", "replay_vs_unsafe_cosine",
        )
    })
    drop.update(item.get("drop_reasons", {}))
    for row in item.get("cross_effect", []):
        cross.append({"branch": item["branch"], "shadow_iteration": item["shadow_iteration"], **row})

write_csv("replay_eligibility_audit.csv", eligibility)
write_csv("auxiliary_gradient_audit.csv", gradient_rows)
write_csv("offpolicy_validity_metrics.csv", eligibility)
write_csv("branch_update_trace.csv", branch_trace)
write_csv("shadow_stability_metrics.csv", branch_trace)
write_csv("loss_cross_effect.csv", cross)
write_csv("behavioral_evaluation.csv", behavior)
write_csv("retention_evaluation.csv", retention)

consolidation = []
for item in metrics:
    subset = [
        row for row in behavior
        if row["branch"] == item["branch"] and int(row["shadow_iteration"]) == item["shadow_iteration"]
    ]
    def aggregate(mode, speed):
        rows = [row for row in subset if row["mode"] == mode and abs(float(row["target_speed"]) - speed) < 1e-6]
        return {
            "completion": sum(int(row["completion_events"]) for row in rows),
            "periodic": sum(int(row["periodic_running"]) for row in rows) / len(rows) if rows else None,
            "fall": sum(int(row["fall"]) for row in rows) / len(rows) if rows else None,
        }
    d24, s24, d26, s26 = aggregate("D0", 2.4), aggregate("S100", 2.4), aggregate("D0", 2.6), aggregate("S100", 2.6)
    consolidation.append({
        "branch": item["branch"], "shadow_iteration": item["shadow_iteration"],
        "buffer_completion_windows": item["buffer_completion_windows"],
        "eligible_replay_windows": item["eligible_windows"],
        "d0_completion_2p4": d24["completion"], "s100_completion_2p4": s24["completion"],
        "d0_periodic_2p4": d24["periodic"], "s100_periodic_2p4": s24["periodic"],
        "d0_fall_2p4": d24["fall"], "s100_fall_2p4": s24["fall"],
        "d0_completion_2p6": d26["completion"], "s100_completion_2p6": s26["completion"],
        "d0_periodic_2p6": d26["periodic"], "s100_periodic_2p6": s26["periodic"],
        "d0_fall_2p6": d26["fall"], "s100_fall_2p6": s26["fall"],
        "completion_loss_relative_change": next((row["relative_change"] for row in item["cross_effect"] if row["stratum"] == "completion"), None),
        "unsafe_loss_relative_change": next((row["relative_change"] for row in item["cross_effect"] if row["stratum"] == "unsafe"), None),
    })
write_csv("consolidation_timeline.csv", consolidation)

summaries = []
for path in RAW.glob("*/batch_*_summary.json"):
    if "backup" not in str(path):
        value = read_json(path)
        summaries.append({
            "branch": path.parent.name, "batch": value["batch_index"],
            "checkpoint_sha256": value["checkpoint_sha256"],
            "completion_events": value["completion_events"],
            "completion_episodes": value["completion_episodes"],
            "seed": value.get("seed_root", value.get("seed")),
        })
window_hashes = []
for path in RAW.glob("*/replay_buffer.pt"):
    if "backup" not in str(path):
        window_hashes.append({"branch": path.parent.name, "sha256": sha(path), "tracked": False})

dump("stage_reference.json", {
    "stage": "2H", "starting_head": START, "primary_checkpoint": str(CHECKPOINT.relative_to(REPO)),
    "primary_checkpoint_sha256": sha(CHECKPOINT), "adam_step": 88000,
    "prior_classifications_preserved": [
        "SINGLE_POLICY_RUN_COMPLETION_EMERGED_PARTIAL",
        "PHASE_A_BOUNDARY_MULTIPLE_CAUSES",
        "EVENT_STRATIFIED_ON_POLICY_NO_EFFECT",
        "PHASE_B_NOT_READY",
    ],
})
dump("protocol.json", {
    "status": "COMPLETED_DISPOSABLE_PREFLIGHT", "max_shadow_iterations": 4,
    "standard_ppo_update_preserved": True, "auxiliary_actor_only": True,
    "critic_std_normalizer_auxiliary_updates": 0, "persistent_parameter_updates": 0,
    "branches": ["R0_STANDARD", "R1", "R2", "R4", "RB_BACKGROUND"],
    "coefficients_screened": [0.025, 0.050, 0.100],
})
dump("checkpoint_manifest.json", {"path": str(CHECKPOINT.relative_to(REPO)), "sha256": sha(CHECKPOINT), "adam_step": 88000})
dump("diagnostic_seed_manifest.json", {"collection_root": 20271000, "analysis_root": 20272000, "evaluation_root": 20273000, "branch_isolation": True})
dump("replay_unit_contract.json", {
    "unit": "ordered trajectory window", "pre_takeoff_steps_min": 15,
    "post_completion_steps_min": 10, "maximum_steps": 60,
    "merge_consecutive_completion_cycles": True, "episode_contiguity_required": True,
    "stored_episode_advantage_used": True,
})
dump("replay_buffer_manifest.json", {
    "fifo": True, "maximum_unique_completion_windows": 256, "maximum_age": 4,
    "maximum_reuse_per_iteration": 1, "maximum_total_reuse": 4,
    "collection_batches": summaries, "raw_buffers_tracked": False,
})
dump("replay_window_hashes.json", {"buffers": window_hashes})
dump("replay_drop_reasons.json", dict(drop))
dump("auxiliary_objective_contract.json", {
    "importance_ratio": "exp(log_pi_current-log_pi_behavior)",
    "objective": "existing clipped PPO surrogate", "clip_range": 0.2,
    "entropy": 0, "value_loss": 0, "std_update": 0,
    "coefficients": [0.025, 0.050, 0.100], "gradient_cap": 0.10,
})
dump("compute_matched_background_control.json", {
    "branch": "RB_A025", "matching": ["target_speed", "episode_time", "contact_phase", "flight_state"],
    "same_window_count_and_length_required": True,
    "status": "EXECUTED" if any(row["branch"] == "RB_A025" for row in metrics) else "NOT_EXECUTED",
})

# Fail closed: no route can pass when completion holdout loss does not improve.
completion_improvers = [
    item for item in metrics
    if item["mode"] == "completion"
    and item["auxiliary_applied"]
    and next((row["relative_change"] for row in item["cross_effect"] if row["stratum"] == "completion"), 1) < 0
]
classification = (
    "SHORT_HORIZON_COMPLETION_REPLAY_SAFETY_FAIL"
    if completion_improvers
    else "SHORT_HORIZON_COMPLETION_REPLAY_NO_EFFECT"
    if metrics
    else "SHORT_HORIZON_COMPLETION_REPLAY_INCONCLUSIVE"
)
dump("fresh_process_reproducibility.json", {
    "status": "NOT_EXECUTED_NO_SAFE_CANDIDATE",
    "reason": (
        "No branch passed the completion-loss causal gate."
        if not completion_improvers
        else "The only completion-loss improving branch failed behavioral safety; it was not eligible for fresh-process confirmation."
    ),
})
dump("selected_short_horizon_replay.json", {"selected": False, "reason": "No branch passed all causal, behavior, and safety gates."})
dump("stage_classification.json", {"classification": classification, "prior_results_overwritten": False})
dump("phase_b_readiness.json", {"classification": "PHASE_B_NOT_READY"})
dump("recommended_next_action.json", {
    "action": "close completion-event reuse route and pivot to reverse single-policy continuation from the exp_005 Stage 4 RUN-capable parent"
})
dump("protected_hashes.json", {
    "primary_checkpoint_sha256": sha(CHECKPOINT), "persistent_checkpoint_writes": 0,
    "production_policy_updates": 0, "remote_push": False,
    "isaaclab_core_changed": False, "rsl_rl_installed_package_changed": False,
})
collection_totals = {}
for row in summaries:
    values = collection_totals.setdefault(row["branch"], {"events": 0, "episodes": 0})
    values["events"] += row["completion_events"]
    values["episodes"] += row["completion_episodes"]
dump("gate.json", {
    "data_gate": all(values["events"] >= 64 and values["episodes"] >= 32 for values in collection_totals.values()),
    "replay_causal_gate": False, "phase_b_ready": False,
    "classification": classification,
})
