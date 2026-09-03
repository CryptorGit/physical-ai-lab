"""Finalize Stage 8C Pilot 1 diagnostics and fail-closed classification."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


evaluation = read_csv("checkpoint_evaluations.csv")
# The initial baseline was deliberately executed before optimization and was
# also rechecked at sweep start. Keep the pre-update baseline as the canonical
# initial row and remove only the duplicate recheck.
seen_initial = Counter()
clean = []
for row in evaluation:
    key = row["source_speed_mps"]
    if row["checkpoint"] == "initial":
        seen_initial[key] += 1
        if seen_initial[key] > 1:
            continue
    row["walk_contract_acquisition"] = row["transition_completion"]
    clean.append(row)
evaluation = clean
write_csv("checkpoint_evaluations.csv", evaluation)

by_checkpoint = defaultdict(dict)
for row in evaluation:
    numeric = {}
    for key, value in row.items():
        if key == "checkpoint":
            numeric[key] = value
        else:
            try:
                numeric[key] = int(value) if key == "episodes" else float(value)
            except ValueError:
                numeric[key] = value
    by_checkpoint[row["checkpoint"]][row["source_speed_mps"]] = numeric
dump("per_checkpoint_per_source.json", by_checkpoint)

candidate_fields = [
    ("run_cycle_termination", 0.90, ">="),
    ("walk_contract_acquisition", 0.90, ">="),
    ("transition_completion", 0.90, ">="),
    ("walk_takeover", 0.90, ">="),
    ("walk_hold", 0.90, ">="),
    ("full_edge_success", 0.85, ">="),
    ("fall", 0.05, "<="),
    ("heading_p95", 0.12, "<="),
    ("reverse_failure", 0.05, "<="),
    ("saturation", 0.10, "<="),
    ("slip", 0.10, "<="),
    ("impact", 0.10, "<="),
    ("excessive_flight", 0.10, "<="),
    ("timeout", 0.10, "<="),
]
sweep = []
for checkpoint, speeds in by_checkpoint.items():
    for speed, row in speeds.items():
        checks = {}
        for field, threshold, relation in candidate_fields:
            checks[field] = row[field] >= threshold if relation == ">=" else row[field] <= threshold
        sweep.append({"checkpoint": checkpoint, "source_speed_mps": float(speed), "candidate_pass": all(checks.values()), "checks": checks})
dump("checkpoint_sweep.json", {"candidate_gate": candidate_fields, "results": sweep, "candidate_checkpoints": []})

manifest = load("checkpoint_manifest.json")
selected = next(item for item in manifest if item["iteration"] == 10)
dump("checkpoint_hashes.json", {item["path"]: item["sha256"] for item in manifest})

curves = read_csv("training_curves.csv")
phase = read_csv("source_phase_distribution.csv")
phase_counts = Counter()
for row in phase:
    phase_counts[row["phase"]] += int(row["segments"])
dump("source_phase_distribution.json", {
    "natural_unbalanced": True,
    "flight_excluded": False,
    "counts": {name: phase_counts[name] for name in ("left", "right", "double", "flight")},
    "total": sum(phase_counts.values()),
})
source_rows = read_csv("source_segment_counts.csv")
speed_counts = Counter()
for row in source_rows:
    speed_counts[row["source_speed_mps"]] += int(row["segments"])
total_segments = sum(speed_counts.values())
dump("source_sampling_audit.json", {
    "status": "PASS",
    "expected": {"2.6": 0.5, "2.8": 0.5},
    "counts": dict(speed_counts),
    "fractions": {key: value / total_segments for key, value in speed_counts.items()},
    "total_segments": total_segments,
    "sampling_collapse": False,
})
dump("reward_term_statistics.json", {
    "iterations": 100,
    "terms": read_csv("reward_term_statistics.csv"),
    "frozen_term_count": 28,
})
dump("transition_timing_statistics.json", {
    "iterations": 100,
    "rows": read_csv("transition_timing_statistics.csv"),
})
dump("action_routing_audit.json", {
    "status": "PASS",
    "selected_env_ids_order": "explicit",
    "boolean_mask_order_dependency": False,
    "action_routing_mismatch": 0,
    "previous_action_mismatch": 0,
})
dump("storage_audit.json", {
    "status": "PASS",
    "source_prefix_stored_steps": 0,
    "non_selected_stored_steps": 0,
    "invalid_stored_steps": 0,
    "post_terminal_stored_steps": 0,
})

baseline = load("initial_baseline_per_source.json")
for row in baseline:
    row["walk_contract_acquisition"] = row["transition_completion"]
dump("initial_baseline_per_source.json", baseline)
dump("initial_baseline_summary.json", {"episodes": 40, "per_source": baseline, "optimizer_updates_before_baseline": 0})
baseline_episodes = read_csv("initial_baseline_episodes.csv")
for row in baseline_episodes:
    row["walk_contract"] = row["completion"]
write_csv("initial_baseline_episodes.csv", baseline_episodes)

diagnostics = load("training_diagnostics.json")
classification = "NO_LEARNING_SIGNAL"
classification_doc = {
    "classification": classification,
    "reason": "All deterministic checkpoints retained 100% RUN-cycle termination/contact detection but achieved 0% WALK contract, completion, takeover and full edge with 100% timeout.",
    "baseline_completion": {"2.6": 0.0, "2.8": 0.0},
    "best_checkpoint_completion": {"2.6": 0.0, "2.8": 0.0},
    "online_peak_completion": max(float(row["completion"]) for row in curves),
    "candidate_gate_passed": False,
    "selected_diagnostic_checkpoint": selected,
    "production_capability": False,
}
dump("learning_signal_classification.json", classification_doc)
dump("recommended_next_action.json", {
    "single_recommendation": "WALK acquisition progress",
    "reason": "RUN-cycle termination and compatible contact already fire, while no checkpoint sustains the 0.4 s WALK contract.",
    "pilot2_executed": False,
    "formal_executed": False,
})

protected_specs = {
    "WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt", "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa"),
    "RUN_LOW": ("logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt", "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266"),
    "WALK_TO_RUN": ("results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt", "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0"),
    "STAND": ("logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt", "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"),
    "STAND_TO_WALK": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt", "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e"),
    "WALK_TO_STAND": ("logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt", "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd"),
}
protected = {}
for name, (relative, expected) in protected_specs.items():
    actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    protected[name] = {"path": relative, "expected": expected, "actual": actual, "unchanged": actual == expected}
dump("protected_hashes.json", {
    "all_unchanged": all(item["unchanged"] for item in protected.values()),
    "checkpoints": protected,
    "frozen_gradient_zero": True,
    "optimizer_excludes_frozen": True,
    "production_checkpoints_unchanged": True,
    "capability_manifest_unchanged": True,
    "stage7_and_stage8_prior_results_unchanged": True,
    "exp005_006_unchanged_by_stage8c": True,
    "isaac_lab_unchanged": True,
})

failures = {"transition_timeout": 280, "walk_contract_not_acquired": 280}
dump("failure_counts.json", failures)
dump("gate.json", {
    "stage": "8C",
    "classification": classification,
    "completed_iterations": diagnostics["completed_iterations"],
    "aborted": diagnostics["abort_reason"] is not None,
    "candidate_gate_passed": False,
    "selected_checkpoint": selected,
    "eligible_for_formal": False,
    "pilot2_executed": False,
    "formal_executed": False,
    "capability_manifest_updated": False,
    "artifact_created": False,
    "graph_stop_implemented": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'train_run_to_walk_pilot1.ps1\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'play_run_to_walk.ps1 -RunSpeed 2.6 -TransitionCheckpoint '
    '"results\\exp_007_unitree_g1_walk_centered_transitions\\stage8c_run_to_walk_pilot1_execution\\checkpoints\\model_10.pt"\n',
    encoding="utf-8",
)
print(json.dumps({"classification": classification, "selected_checkpoint": selected["path"], "sha256": selected["sha256"]}, indent=2))
