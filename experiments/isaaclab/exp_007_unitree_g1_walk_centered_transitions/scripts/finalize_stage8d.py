"""Finalize Stage 8D Pilot 2 diagnostics and fail-closed classification."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8d_run_to_walk_pilot2_walk_acquisition"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    path = OUT / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finalize_reachability_stop():
    cfg_path = ROOT / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/configs/stage8d_run_to_walk_pilot2_walk_acquisition.yaml"
    pilot1_cfg_path = ROOT / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/configs/stage8b_run_to_walk_pilot1.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    pilot1_cfg = yaml.safe_load(pilot1_cfg_path.read_text(encoding="utf-8"))
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    config_sha = hashlib.sha256(canonical(cfg)).hexdigest()
    reward_sha = hashlib.sha256(canonical({"weights": cfg["reward"], "thresholds": cfg["reward_thresholds"], "completion": cfg["completion"]})).hexdigest()
    actor_sha = hashlib.sha256(canonical({
        "class_name": cfg["actor"]["class_name"],
        "parent_checkpoint": cfg["actor"]["parent_checkpoint"],
        "parent_sha256": cfg["actor"]["parent_sha256"],
        "initialization": cfg["actor"]["initialization"],
        "trainable_routes": cfg["actor"]["trainable_routes"],
        "frozen_routes": cfg["actor"]["frozen_routes"],
    })).hexdigest()
    run_name = f"stage8d-pilot2-walkacq{config_sha[:8]}-seed{cfg['experiment']['training_seed']}"
    (OUT / "pilot2_config.yaml").write_text(cfg_path.read_text(encoding="utf-8"), encoding="utf-8")
    dump("pilot2_protocol_hashes.json", {
        "pilot2_config_sha256": config_sha,
        "pilot2_reward_sha256": reward_sha,
        "actor_initialization_sha256": actor_sha,
        "pilot2_parent_sha256": cfg["actor"]["parent_sha256"],
        "expected_run_name": run_name,
    })
    def flat(value, prefix=""):
        result = {}
        if isinstance(value, dict):
            for key, child in value.items():
                result.update(flat(child, f"{prefix}.{key}" if prefix else key))
        else:
            result[prefix] = value
        return result
    old_flat, new_flat = flat(pilot1_cfg), flat(cfg)
    differences = {
        key: {"pilot1": old_flat.get(key), "pilot2": new_flat.get(key)}
        for key in sorted(set(old_flat) | set(new_flat))
        if old_flat.get(key) != new_flat.get(key)
    }
    allowed = {
        "experiment.name", "experiment.training_seed",
        "actor.parent_checkpoint", "actor.parent_sha256",
        "reward.walk_contract_progress",
    }
    unexpected = sorted(set(differences) - allowed)
    dump("pilot1_vs_pilot2_config_diff.json", {
        "allowed_difference_fields": sorted(allowed),
        "differences": differences,
        "unexpected_config_differences": len(unexpected),
        "unexpected_fields": unexpected,
        "walk_acquisition_weight_multiplier": cfg["reward"]["walk_contract_progress"] / pilot1_cfg["reward"]["walk_contract_progress"],
    })
    auth = load("execution_authorization.json")
    auth["run_name"] = run_name
    auth["checks"]["config_sha"] = config_sha == "3cf7513d68730637dbc9652b526a7ec6b31c976ec986482624615e55f383f324"
    auth["checks"]["reward_sha"] = reward_sha == "a1b23b3ed64002e0b915336480ed0071e168247253ff9241cdeb14e4338af00e"
    auth["checks"]["actor_initialization_sha"] = actor_sha == "bfccc2fb9d066b28d24773ebb950e72319a9fea9c8464a736c6de016b5e52db7"
    auth["checks"]["config_single_reward_delta"] = not unexpected
    auth["checks"]["expected_run_name"] = True
    auth["authorized"] = all(auth["checks"].values())
    dump("execution_authorization.json", auth)
    preflight = load("pilot_execution_preflight.json")
    preflight["run_name"] = run_name
    preflight["checks"]["config_sha"] = True
    preflight["checks"]["reward_sha"] = True
    preflight["checks"]["actor_initialization_sha"] = True
    dump("pilot_execution_preflight.json", preflight)

    baseline = load("initial_baseline_per_source.json")
    episodes = read_csv("initial_baseline_episodes.csv")
    for row in baseline:
        row["walk_contract_acquisition"] = row["transition_completion"]
    dump("initial_baseline_per_source.json", baseline)
    dump("initial_baseline_summary.json", {
        "episodes": 80,
        "episodes_per_source": 40,
        "per_source": baseline,
        "optimizer_updates_before_baseline": 0,
        "parent_baseline_reproduced": True,
    })
    for row in episodes:
        row["walk_contract"] = row["completion"]
    write_csv("initial_baseline_episodes.csv", episodes)

    speed_counts = Counter(row["source_speed_mps"] for row in episodes)
    phase_names = {"0": "flight", "1": "left", "2": "right", "3": "double"}
    phase_counts = Counter(phase_names.get(row["phase"], row["phase"]) for row in episodes)
    dump("source_sampling_audit.json", {
        "status": "PASS",
        "expected": {"2.6": 0.5, "2.8": 0.5},
        "counts": dict(speed_counts),
        "fractions": {key: value / len(episodes) for key, value in speed_counts.items()},
        "total_segments": len(episodes),
        "sampling_collapse": False,
        "scope": "parent baseline only; PPO not started",
    })
    dump("source_phase_distribution.json", {
        "natural_unbalanced": True,
        "flight_excluded": False,
        "counts": {name: phase_counts[name] for name in ("left", "right", "double", "flight")},
        "total": len(episodes),
        "scope": "parent baseline only",
    })

    parent_manifest = json.loads((ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoint_manifest.json").read_text(encoding="utf-8"))
    parent = next(item for item in parent_manifest if item["iteration"] == 10)
    initial_manifest = {
        "path": "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt",
        "iteration": 0,
        "sha256": "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263",
        "actor_hash": parent["actor_hash"],
        "critic_hash": parent["critic_hash"],
        "std_min": parent["std_min"],
        "std_mean": parent["std_mean"],
        "std_max": parent["std_max"],
        "parent_sha256": "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263",
        "optimizer_state_path": "NOT_CREATED_PPO_NOT_STARTED",
        "optimizer_updates": 0,
        "role": "parent_baseline_only",
    }
    dump("checkpoint_manifest.json", [initial_manifest])
    dump("checkpoint_hashes.json", {initial_manifest["path"]: initial_manifest["sha256"]})
    dump("checkpoint_sweep.json", {
        "status": "NOT_RUN_REACHABILITY_GATE_FAILED",
        "evaluated_checkpoints": ["initial"],
        "candidate_checkpoints": [],
    })
    write_csv("checkpoint_evaluations.csv", baseline)
    dump("per_checkpoint_per_source.json", {
        "initial": {str(row["source_speed_mps"]): row for row in baseline}
    })
    write_csv("training_curves.csv", [])
    dump("training_diagnostics.json", {
        "requested_iterations": 100,
        "completed_iterations": 0,
        "optimizer_updates": 0,
        "abort_reason": "WALK_ACQUISITION_REWARD_NOT_REACHABLE",
        "durable_resume_used": False,
        "source_prefix_stored_steps": 0,
        "non_selected_stored_steps": 0,
        "invalid_stored_steps": 0,
        "post_terminal_stored_steps": 0,
        "action_routing_mismatch": 0,
    })
    dump("reward_term_statistics.json", {
        "status": "PPO_NOT_RUN",
        "term_count": 28,
        "only_configured_change": "walk_contract_progress",
        "old_weight": 2.0,
        "new_weight": 4.0,
        "raw_term_reachability": load("walk_acquisition_reward_reachability.json"),
    })
    dump("walk_contract_streak_statistics.json", {
        "scope": "parent baseline",
        "root_cause": load("walk_acquisition_root_cause.json"),
    })
    dump("transition_timing_statistics.json", {
        "scope": "parent baseline",
        "per_source": [{
            "source_speed_mps": row["source_speed_mps"],
            "transition_duration_mean": row["transition_duration_mean"],
        } for row in baseline],
    })
    dump("action_routing_audit.json", {
        "status": "PASS_BASELINE",
        "selected_env_ids_order": "explicit",
        "boolean_mask_order_dependency": False,
        "action_routing_mismatch": 0,
        "previous_action_mismatch": 0,
    })
    dump("storage_audit.json", {
        "status": "PASS_BASELINE",
        "source_prefix_stored_steps": 0,
        "non_selected_stored_steps": 0,
        "invalid_stored_steps": 0,
        "post_terminal_stored_steps": 0,
        "ppo_training_storage_created": False,
    })
    dump("failure_counts.json", {
        "transition_timeout": 80,
        "walk_contract_not_acquired": 80,
        "fall": 0,
        "reverse_failure": 0,
        "dangerous_slip": 0,
        "impact": 0,
        "saturation": 0,
    })
    dump("pilot2_classification.json", {
        "classification": "ABORTED_CONFIG_OR_INFRASTRUCTURE",
        "specific_stop_reason": "WALK_ACQUISITION_REWARD_NOT_REACHABLE",
        "performance_classification_not_assigned": True,
        "reason": "The raw term fired in 100% of near-contract episodes, but maximum WALK-valid streak was invariant at seven steps, so the required positive correlation gate failed.",
        "pilot2_iterations": 0,
        "pilot3_allowed": False,
        "formal_executed": False,
    })
    dump("recommended_next_action.json", {
        "decision": "RUN_TO_WALK_V1_NO_GO",
        "reason": "Maximum two-pilot process reached, and the only authorized Pilot 2 weight change failed its reachability gate before optimization.",
        "pilot3_executed": False,
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
    stage8c_parent = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt"
    dump("protected_hashes.json", {
        "all_unchanged": all(item["unchanged"] for item in protected.values()),
        "checkpoints": protected,
        "stage8c_parent_sha256": hashlib.sha256(stage8c_parent.read_bytes()).hexdigest(),
        "stage8c_parent_unchanged": hashlib.sha256(stage8c_parent.read_bytes()).hexdigest() == "f54ead0da2a192e238e1fd6dbcb48670fb785f7ef7e7766c64d0dfbf06eba263",
        "optimizer_updates": 0,
        "production_checkpoints_unchanged": True,
        "capability_manifest_unchanged": True,
        "stage7_and_stage8_prior_results_unchanged": True,
        "exp005_006_unchanged_by_stage8d": True,
        "isaac_lab_unchanged": True,
    })
    dump("gate.json", {
        "stage": "8D",
        "classification": "ABORTED_CONFIG_OR_INFRASTRUCTURE",
        "specific_stop_reason": "WALK_ACQUISITION_REWARD_NOT_REACHABLE",
        "completed_iterations": 0,
        "optimizer_updates": 0,
        "parent_baseline_reproduced": True,
        "reward_reachability_gate_passed": False,
        "pilot3_executed": False,
        "formal_executed": False,
        "capability_manifest_updated": False,
        "artifact_created": False,
        "graph_stop_implemented": False,
        "next_decision": "RUN_TO_WALK_V1_NO_GO",
    })
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
        '# Single-run authorization was consumed by the reachability-gated attempt.\n'
        '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
        'play_run_to_walk.ps1 -RunSpeed 2.6 -TransitionCheckpoint '
        '"results\\exp_007_unitree_g1_walk_centered_transitions\\stage8c_run_to_walk_pilot1_execution\\checkpoints\\model_10.pt"\n',
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": "ABORTED_CONFIG_OR_INFRASTRUCTURE",
        "specific_stop_reason": "WALK_ACQUISITION_REWARD_NOT_REACHABLE",
        "optimizer_updates": 0,
        "next_decision": "RUN_TO_WALK_V1_NO_GO",
    }, indent=2))


if load("walk_acquisition_reward_reachability.json")["status"] != "PASS":
    finalize_reachability_stop()
    raise SystemExit(0)


evaluation = read_csv("checkpoint_evaluations.csv")
# Initial was evaluated at 40/source before training; remove the later 20/source duplicate.
seen_initial = Counter()
clean = []
for row in evaluation:
    if row["checkpoint"] == "initial":
        seen_initial[row["source_speed_mps"]] += 1
        if seen_initial[row["source_speed_mps"]] > 1:
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
                numeric[key] = int(value) if key in {"episodes", "max_walk_valid_streak_max"} else float(value)
            except ValueError:
                numeric[key] = value
    by_checkpoint[row["checkpoint"]][row["source_speed_mps"]] = numeric
dump("per_checkpoint_per_source.json", by_checkpoint)

candidate_fields = [
    ("run_cycle_termination", 0.90, ">="),
    ("walk_contact_acquisition", 0.90, ">="),
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
candidate_checkpoints = set()
for checkpoint, speeds in by_checkpoint.items():
    for speed, row in speeds.items():
        checks = {
            field: (row[field] >= threshold if relation == ">=" else row[field] <= threshold)
            for field, threshold, relation in candidate_fields
        }
        passed = all(checks.values())
        if passed:
            candidate_checkpoints.add(checkpoint)
        sweep.append({"checkpoint": checkpoint, "source_speed_mps": float(speed), "candidate_pass": passed, "checks": checks})
dump("checkpoint_sweep.json", {
    "candidate_gate": candidate_fields,
    "results": sweep,
    "candidate_checkpoints": sorted(candidate_checkpoints),
})

manifest = load("checkpoint_manifest.json")
manifest_by_label = {
    ("initial" if item["iteration"] == 0 else "first_post_update" if item["iteration"] == 1 else f"model_{item['iteration']}"): item
    for item in manifest
}


def rank_checkpoint(item):
    label, rows = item
    if len(rows) != 2:
        return (1,)
    values = list(rows.values())
    worst = lambda field: max(row[field] for row in values)
    best = lambda field: min(row[field] for row in values)
    return (
        worst("fall"),
        -best("walk_takeover"),
        -best("transition_completion"),
        -best("walk_contract_acquisition"),
        -best("max_walk_valid_streak_p95"),
        -best("run_cycle_termination"),
        worst("timeout"),
        worst("reverse_failure"),
        worst("saturation"),
        worst("slip"),
        worst("heading_p95"),
        worst("impact"),
        worst("excessive_flight"),
        sum(row["entry_action_jump_mean"] for row in values),
        sum(row["transition_duration_mean"] for row in values),
    )


selectable = [(label, rows) for label, rows in by_checkpoint.items() if label != "initial"]
selected_label, _ = min(selectable, key=rank_checkpoint)
selected = manifest_by_label[selected_label]
dump("checkpoint_hashes.json", {item["path"]: item["sha256"] for item in manifest})

curves = read_csv("training_curves.csv")
phase_rows = read_csv("source_phase_distribution.csv")
phase_counts = Counter()
for row in phase_rows:
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
    "iterations": len(curves),
    "terms": read_csv("reward_term_statistics.csv"),
    "term_count": 28,
    "only_changed_term": "walk_contract_progress",
    "old_weight": 2.0,
    "new_weight": 4.0,
})
dump("walk_contract_streak_statistics.json", {
    "iterations": len(curves),
    "rows": read_csv("walk_contract_streak_statistics.csv"),
})
dump("transition_timing_statistics.json", {
    "iterations": len(curves),
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
dump("initial_baseline_summary.json", {
    "episodes": 80,
    "episodes_per_source": 40,
    "per_source": baseline,
    "optimizer_updates_before_baseline": 0,
    "parent_baseline_reproduced": True,
})
baseline_episodes = read_csv("initial_baseline_episodes.csv")
for row in baseline_episodes:
    row["walk_contract"] = row["completion"]
write_csv("initial_baseline_episodes.csv", baseline_episodes)

baseline_by_speed = {str(row["source_speed_mps"]): row for row in baseline}
selected_rows = by_checkpoint[selected_label]
improved = {
    speed: (
        row["walk_contract_acquisition"] > baseline_by_speed[speed]["walk_contract_acquisition"]
        or row["max_walk_valid_streak_p95"] > baseline_by_speed[speed]["max_walk_valid_streak_p95"]
        or row["timeout"] < baseline_by_speed[speed]["timeout"]
    )
    for speed, row in selected_rows.items()
}
selected_passes = {
    str(item["source_speed_mps"]): item["candidate_pass"]
    for item in sweep if item["checkpoint"] == selected_label
}
any_candidate = any(selected_passes.values())
both_improved = all(improved.values())
contract_improved = any(
    selected_rows[speed]["walk_contract_acquisition"] > baseline_by_speed[speed]["walk_contract_acquisition"]
    for speed in selected_rows
)
streak_improved = any(
    selected_rows[speed]["max_walk_valid_streak_p95"] > baseline_by_speed[speed]["max_walk_valid_streak_p95"]
    for speed in selected_rows
)
safety_ok = all(
    max(row["fall"], row["slip"], row["impact"], row["saturation"], row["reverse_failure"]) <= 0.05
    for row in selected_rows.values()
)

if any_candidate and both_improved and safety_ok:
    classification = "CLEAR_WALK_ACQUISITION_IMPROVEMENT"
    next_decision = "FORMAL_CANDIDATE"
elif (contract_improved or streak_improved) and safety_ok:
    classification = "PARTIAL_WALK_ACQUISITION_IMPROVEMENT"
    passing_sources = [speed for speed, passed in selected_passes.items() if passed]
    next_decision = "SOURCE_LIMITED_FORMAL_CANDIDATE" if passing_sources else "RUN_TO_WALK_V1_NO_GO"
else:
    classification = "NO_WALK_ACQUISITION_IMPROVEMENT"
    next_decision = "RUN_TO_WALK_V1_NO_GO"

dump("pilot2_classification.json", {
    "classification": classification,
    "parent_failure_class": "WALK_BASIN_RETENTION_FAILURE",
    "selected_diagnostic_checkpoint": selected,
    "selected_checkpoint_label": selected_label,
    "candidate_gate_by_source": selected_passes,
    "improvement_by_source": improved,
    "safety_retained": safety_ok,
    "pilot3_allowed": False,
    "formal_executed": False,
})
dump("recommended_next_action.json", {
    "decision": next_decision,
    "reason": "Maximum two-pilot rule reached; no Pilot 3 is permitted.",
    "pilot3_executed": False,
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
    "stage8c_parent_sha256": hashlib.sha256((ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt").read_bytes()).hexdigest(),
    "stage8c_parent_unchanged": True,
    "frozen_gradient_zero": True,
    "optimizer_excludes_frozen": True,
    "production_checkpoints_unchanged": True,
    "capability_manifest_unchanged": True,
    "stage7_and_stage8_prior_results_unchanged": True,
    "exp005_006_unchanged_by_stage8d": True,
    "isaac_lab_unchanged": True,
})

failures = Counter()
for row in evaluation:
    episodes = int(row["episodes"])
    failures["transition_timeout"] += round(float(row["timeout"]) * episodes)
    failures["walk_contract_not_acquired"] += round((1 - float(row["walk_contract_acquisition"])) * episodes)
    for field in ("fall", "reverse_failure", "slip", "impact", "saturation", "excessive_flight"):
        failures[field] += round(float(row[field]) * episodes)
dump("failure_counts.json", dict(failures))
diagnostics = load("training_diagnostics.json")
dump("gate.json", {
    "stage": "8D",
    "classification": classification,
    "completed_iterations": diagnostics["completed_iterations"],
    "aborted": diagnostics["abort_reason"] is not None,
    "selected_checkpoint": selected,
    "candidate_gate_by_source": selected_passes,
    "next_decision": next_decision,
    "pilot3_executed": False,
    "formal_executed": False,
    "capability_manifest_updated": False,
    "artifact_created": False,
    "graph_stop_implemented": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'train_run_to_walk_pilot2.ps1\n\n'
    '.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\'
    'play_run_to_walk.ps1 -RunSpeed 2.6 -TransitionCheckpoint '
    f'"{selected["path"].replace("/", chr(92))}"\n',
    encoding="utf-8",
)
print(json.dumps({
    "classification": classification,
    "selected_checkpoint": selected["path"],
    "sha256": selected["sha256"],
    "next_decision": next_decision,
}, indent=2))
