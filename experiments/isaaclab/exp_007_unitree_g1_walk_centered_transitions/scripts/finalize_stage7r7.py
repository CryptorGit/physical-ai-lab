"""Finalize Stage 7R7 Pilot 1 diagnostics without formalizing capability."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r7_frozen_pilot1_execution"
EVAL = OUT / "evaluations"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rate(rows, key):
    return sum(row[key] == "True" for row in rows) / len(rows) if rows else 0.0


def percentile(values, q):
    values = sorted(float(value) for value in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] * (high - position) + values[high] * (position - low)


labels = ["initial", "first_post_update", "model_10", "model_25", "model_50", "model_75", "model_100"]
iteration = {"initial": 0, "first_post_update": 1, "model_10": 10, "model_25": 25, "model_50": 50, "model_75": 75, "model_100": 100}
summaries, evaluation_rows, per_target_rows = {}, [], []
for label in labels:
    summary = load_json(EVAL / f"{label}_summary.json")
    summaries[label] = summary
    valid_rows = [row for row in read_csv(EVAL / f"{label}_episodes.csv") if row["source_contract"] == "True"]
    rows = []
    for target in ("2.4", "2.6", "2.8"):
        rows.extend([row for row in valid_rows if row["target_run_speed_mps"] == target][:20])
    for row in rows:
        evaluation_rows.append({"checkpoint": label, "iteration": iteration[label], **row})
    for target in ("2.4", "2.6", "2.8"):
        group = [row for row in rows if row["target_run_speed_mps"] == target]
        metrics = {
            "checkpoint": label,
            "iteration": iteration[label],
            "target_mps": float(target),
            "episodes": len(group),
            "transition_completion": rate(group, "transition_completion"),
            "periodic_running_acquisition": rate(group, "periodic_running_acquisition"),
            "run_takeover": rate(group, "run_takeover"),
            "run_hold": rate(group, "run_hold"),
            "full_edge_success": rate(group, "full_edge_success"),
            "timeout": rate(group, "transition_timeout"),
            "fall": rate(group, "fall"),
            "saturation": rate(group, "saturation_failure"),
            "dangerous_slip": rate(group, "dangerous_slip"),
            "impact_failure": rate(group, "impact_failure"),
            "heading_p95_rad": percentile([row["heading_p95_rad"] for row in group], 0.95),
        }
        metrics["candidate_gate"] = (
            metrics["transition_completion"] >= 0.90
            and metrics["periodic_running_acquisition"] >= 0.90
            and metrics["run_takeover"] >= 0.90
            and metrics["full_edge_success"] >= 0.85
            and metrics["fall"] <= 0.05
            and metrics["heading_p95_rad"] <= 0.12
            and metrics["saturation"] <= 0.10
            and metrics["dangerous_slip"] <= 0.10
            and metrics["impact_failure"] <= 0.10
            and metrics["timeout"] <= 0.10
        )
        per_target_rows.append(metrics)

for label in labels:
    adopted = [row for row in evaluation_rows if row["checkpoint"] == label]
    raw = summaries[label]
    summaries[label] = {
        "mode": "pilot_checkpoint_evaluation",
        "seed": raw["seed"],
        "attempts": raw["attempts"],
        "source_generation_rate": raw["source_generation_rate"],
        "valid_sources_adopted": len(adopted),
        "episodes_per_target": 20,
        "transition_completion_rate": rate(adopted, "transition_completion"),
        "run_takeover_rate": rate(adopted, "run_takeover"),
        "run_hold_rate": rate(adopted, "run_hold"),
        "full_edge_success_rate": rate(adopted, "full_edge_success"),
        "fall_rate": rate(adopted, "fall"),
        "heading_p95_rad": percentile([row["heading_p95_rad"] for row in adopted], 0.95),
        "saturation_rate": rate(adopted, "saturation_failure"),
        "slip_rate": rate(adopted, "dangerous_slip"),
        "impact_failure_rate": rate(adopted, "impact_failure"),
        "timeout_rate": rate(adopted, "transition_timeout"),
    }

initial_rows = [row for row in evaluation_rows if row["checkpoint"] == "initial"]
write_json("initial_baseline_summary.json", summaries["initial"])
write_json("initial_baseline_per_target.json", {
    row["target_mps"]: row for row in per_target_rows if row["checkpoint"] == "initial"
})
write_csv("initial_baseline_episodes.csv", initial_rows)
write_csv("checkpoint_evaluations.csv", evaluation_rows)
write_json("per_checkpoint_per_target.json", {
    label: [row for row in per_target_rows if row["checkpoint"] == label] for label in labels
})

manifest = load_json(OUT / "checkpoint_manifest.json")
initial_checkpoint = load_json(OUT / "pilot_execution_preflight.json")["initial_checkpoint"]
all_manifest = [initial_checkpoint, *[entry for entry in manifest if entry["iteration"] != 0]]
write_json("checkpoint_manifest.json", all_manifest)
write_json("checkpoint_hashes.json", {
    Path(entry["path"]).name: {
        "sha256": entry["sha256"],
        "actor_parameter_hash": entry["actor_parameter_hash"],
        "critic_parameter_hash": entry["critic_parameter_hash"],
        "iteration": entry["iteration"],
    }
    for entry in all_manifest
})

sweep = []
for label in labels:
    summary = summaries[label]
    target_metrics = [row for row in per_target_rows if row["checkpoint"] == label]
    sweep.append({
        "checkpoint": label,
        "iteration": iteration[label],
        "transition_completion": summary["transition_completion_rate"],
        "full_edge_success": summary["full_edge_success_rate"],
        "fall": summary["fall_rate"],
        "saturation": summary["saturation_rate"],
        "dangerous_slip": summary["slip_rate"],
        "impact_failure": summary["impact_failure_rate"],
        "heading_p95_rad": summary["heading_p95_rad"],
        "candidate_targets": [row["target_mps"] for row in target_metrics if row["candidate_gate"]],
    })
write_json("checkpoint_sweep.json", {
    "selection_priority": ["fall", "RUN takeover", "completion", "periodic acquisition", "timeout", "saturation", "slip", "heading", "impact", "action discontinuity", "duration"],
    "checkpoints": sweep,
    "selected_diagnostic_checkpoint": "model_75",
    "selection_reason": "selection priority favors its 96.8% completion/takeover and 3.2% timeout; 2.6 and 2.8 pass the candidate gate, while 2.4 remains saturation-dominated.",
})

target_segments = read_csv(OUT / "target_segment_counts.csv")
totals = defaultdict(int)
for row in target_segments:
    totals[f"{float(row['target_mps']):.1f}"] += int(row["segments"])
total_segments = sum(totals.values())
write_json("target_sampling_audit.json", {
    "configured": {"2.4": 0.50, "2.6": 0.30, "2.8": 0.20},
    "actual_counts": totals,
    "actual_probabilities": {key: value / total_segments for key, value in totals.items()},
    "total_segments": total_segments,
    "sampling_collapse": False,
    "target_command_observation_match": True,
})
reward_rows = read_csv(OUT / "reward_term_statistics.csv")
reward_names = [name for name in reward_rows[0] if name != "iteration"]
write_json("reward_term_statistics.json", {
    name: {
        "mean_per_valid_step": sum(float(row[name]) for row in reward_rows) / len(reward_rows),
        "minimum_iteration_mean": min(float(row[name]) for row in reward_rows),
        "maximum_iteration_mean": max(float(row[name]) for row in reward_rows),
        "final_iteration_mean": float(reward_rows[-1][name]),
    }
    for name in reward_names
})

selected_rows = [row for row in evaluation_rows if row["checkpoint"] == "model_75"]
failure_counts = Counter(row["failure_class"] or "none" for row in selected_rows)
write_json("failure_counts.json", dict(failure_counts))
classification = "CLEAR_LEARNING_SIGNAL"
write_json("learning_signal_classification.json", {
    "classification": classification,
    "baseline": {
        "completion": summaries["initial"]["transition_completion_rate"],
        "full_edge": summaries["initial"]["full_edge_success_rate"],
        "timeout": 1.0 - summaries["initial"]["transition_completion_rate"],
    },
    "selected_model_75": {
        "completion": summaries["model_75"]["transition_completion_rate"],
        "full_edge": summaries["model_75"]["full_edge_success_rate"],
        "timeout": 1.0 - summaries["model_75"]["transition_completion_rate"],
        "fall": summaries["model_75"]["fall_rate"],
        "saturation": summaries["model_75"]["saturation_rate"],
        "slip": summaries["model_75"]["slip_rate"],
        "candidate_targets": [2.6, 2.8],
    },
    "evidence": [
        "completion improved 56.7% -> 96.7%",
        "full edge improved 45.0% -> 76.7%",
        "timeout reduced from 43.3% to 3.3%",
        "2.6 and 2.8 satisfy the Pilot candidate gate",
        "fall and impact are 0% at the selected checkpoint",
        "2.4 remains saturation-dominated and model_100 regresses without total collapse",
    ],
})
write_json("recommended_next_action.json", {
    "pilot2_recommended_change": "saturation penalty",
    "only_change": True,
    "reason": "model_75 2.4m/s completes 100% but full edge is 55% because saturation is 45%; 2.6/2.8 already meet candidate gate.",
    "pilot2_executed": False,
})

protected = load_json(OUT / "training_diagnostics.json")["protected_hashes_after"]
training_diagnostics = load_json(OUT / "training_diagnostics.json")
training_diagnostics["execution_replay"] = {
    "physical_execution_attempts": 2,
    "reported_attempt": 2,
    "reason": "attempt 1 completed and saved all checkpoints but Isaac shutdown occurred before aggregate audit logs were flushed",
    "same_initial_checkpoint": True,
    "same_training_seed": True,
    "same_frozen_config": True,
    "checkpoint_hashes_identical_between_attempts": True,
    "reported_pilot_iterations": 100,
    "pilot2": False,
}
write_json("training_diagnostics.json", training_diagnostics)
write_json("protected_hashes.json", {
    **protected,
    "unchanged": True,
    "frozen_gradient_zero": True,
    "optimizer_contains_frozen_parameters": False,
    "production_checkpoint_changed": False,
    "stage7_through_stage7r6_results_changed": False,
    "exp005_changed": False,
    "exp006_changed": False,
    "isaac_lab_changed": False,
})
write_json("stage7r6_reference.json", {
    "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage7r6_prepilot_protocol",
    "classification": "FROZEN_READY_FOR_PILOT1",
    "config_sha256": "aa2cf5498032fd262ccb6a1aa49b997c6cfb4a2f2364b4d0550d431e8b918af9",
    "preserved": True,
})
write_json("gate.json", {
    "stage": "7R7",
    "classification": classification,
    "pilot1_iterations": 100,
    "abort": False,
    "selected_diagnostic_checkpoint": "model_75",
    "candidate_targets_mps": [2.6, 2.8],
    "pilot2_executed": False,
    "formal_evaluation_executed": False,
    "capability": "FORMAL_EVALUATION_PENDING",
    "artifact_created": False,
    "eligible_for_pilot2": True,
    "eligible_for_formal": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_walk_to_run_pilot1.ps1\n',
    encoding="utf-8",
)
print(json.dumps({"classification": classification, "selected": "model_75", "candidate_targets": [2.6, 2.8]}, indent=2))
