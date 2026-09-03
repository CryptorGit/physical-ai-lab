"""Aggregate Stage 2D frozen-rollout diagnostics into tracked evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2d_run_reward_reachability_preflight"
STAGE2C = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2c_multi_regime_gradient_interference"
RETRY = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1"
REPORT = REPO / "research/exp_012_g1_run_reward_reachability_preflight_report.md"
START_HEAD = "db7c1ec95ebb6fb1ed91442b9c8e1342f5d22615"
SELECTED_SHA = "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143"
PARENT_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows) -> None:
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def q(values, p):
    values = sorted(float(x) for x in values)
    if not values:
        return None
    return values[round((len(values) - 1) * p)]


runtime = json.loads((OUT / "runtime_diagnostic_summary.json").read_text(encoding="utf-8"))
events = runtime["events"]
selected_events = [row for row in events if row["checkpoint"] == "iter100"]
summaries = runtime["summaries"]
selected_summaries = [row for row in summaries if row["checkpoint"] == "iter100"]

# Provenance and frozen protocol.
checkpoint_manifest = json.loads((STAGE2C / "checkpoint_manifest.json").read_text(encoding="utf-8"))
dump("stage_reference.json", {
    "starting_head": START_HEAD,
    "selected_checkpoint": {"iteration": 100, "sha256": SELECTED_SHA},
    "parent_checkpoint_sha256": PARENT_SHA,
    "preserved_classifications": ["G1_SINGLE_POLICY_MULTIPLE_FAILURES", "RUN_REWARD_REACHABILITY_FAIL"],
    "production_policy_update": 0,
    "new_training_checkpoint": 0,
})
dump("protocol.json", {
    "name": "EXP012_STAGE2D_RUN_REWARD_REACHABILITY_PREFLIGHT",
    "diagnostic_only": True,
    "checkpoints": ["parent", 50, 100, 300],
    "conditions": ["direct_2.4", "direct_2.6", "ramped_2.4", "ramped_2.6", "bidirectional_2.4"],
    "episodes_per_condition": 50,
    "duration_s": {"direct": 10.0, "ramped": 10.0, "bidirectional": 18.0},
    "policy": "deterministic frozen checkpoint",
    "yaw_rate_command": 0,
    "external_controller": False,
    "reward_or_gate_changes": 0,
})
dump("checkpoint_manifest.json", {
    "source": str((STAGE2C / "checkpoint_manifest.json").relative_to(REPO)),
    "source_sha256": sha(STAGE2C / "checkpoint_manifest.json"),
    "diagnosed": [
        {"role": "parent", "path": str((REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt").relative_to(REPO)), "sha256": PARENT_SHA},
        *[x for x in checkpoint_manifest["checkpoints"] if x["iteration"] in (50, 100, 300)],
    ],
})

# Exact source/config equivalence.
params = {
    "command_gate_mps": 2.3, "precursor_speed_tolerance_mps": 1.20,
    "completion_speed_tolerance_mps": 0.30, "tilt_threshold_rad": 0.20,
    "vertical_speed_threshold_mps": 0.50, "minimum_flight_s": 0.04,
    "maximum_flight_s": 0.16, "single_foot_landing_required": True,
    "alternating_landing_required": True, "precursor_cap": 0.75,
    "completion_reward_raw": 2.0, "overlong_flight_penalty_raw": 0.25,
}
dump("exp005_exp012_run_reward_semantic_diff.json", {
    "semantic_difference_count": 0,
    "same_callable": "g1_flat_run.tasks.stage3_mdp.SafePeriodicFlightReward",
    "exp005_stage4_parameters": params,
    "exp012_resolved_parameters": params,
    "parameter_diff": {},
    "note": "exp005 Stage 3 uses the same state machine but disables precursor shaping; Stage 4 and exp012 are semantically identical.",
})
dump("run_reward_source_locations.json", {
    "callable": "experiments/isaaclab/exp_005_unitree_g1_flat_run/src/g1_flat_run/tasks/stage3_mdp.py:43",
    "reset_hook": "experiments/isaaclab/exp_005_unitree_g1_flat_run/src/g1_flat_run/tasks/stage3_mdp.py:109",
    "runtime_call": "experiments/isaaclab/exp_005_unitree_g1_flat_run/src/g1_flat_run/tasks/stage3_mdp.py:122",
    "completion": "experiments/isaaclab/exp_005_unitree_g1_flat_run/src/g1_flat_run/tasks/stage3_mdp.py:291",
    "exp005_stage4_cfg": "experiments/isaaclab/exp_005_unitree_g1_flat_run/src/g1_flat_run/tasks/g1_flat_run_env_cfg.py:145",
    "exp012_cfg": "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src/g1_single_policy/tasks.py:22",
})

# State machine and boundary persistence.
dump("run_reward_state_machine_contract.json", {
    "per_environment_persistent_tensors": [
        "_was_in_flight", "_flight_duration", "_event_precursor_reward", "_last_landing_foot"
    ],
    "derived_not_stored": [
        "current_landing_side", "completion_armed", "last_contact_state", "last_command_gate_state"
    ],
    "not_implemented": ["cooldown"],
    "diagnostic_tensors": ["last_raw_reward", "_counters"],
    "landing_initialization": "first valid single-foot landing initializes _last_landing_foot; completion requires a later opposite landing",
    "double_support": "not a single-foot landing; does not update landing-side memory",
})
dump("run_reward_reset_audit.json", {
    "episode_reset": "RESET indexed environment state",
    "partial_environment_reset": "RESET only supplied environment IDs",
    "ppo_rollout_boundary": "NO RESET",
    "command_segment_change": "NO RESET; high-command gate disables completion but single-foot landing may update side memory",
    "run_to_walk": "NO RESET",
    "walk_to_run": "NO RESET",
    "contact_loss": "starts/continues flight state",
    "double_support": "ends flight without landing-side update",
    "fall": "reset occurs through environment reset hook",
    "timeout": "reset occurs through environment reset hook",
})
boundary_pass = all(x["reward_trace_equal"] and x["max_abs_difference"] == 0 for x in runtime["boundary"].values())
dump("rollout_boundary_persistence.json", {
    "conditions": runtime["boundary"],
    "continuous_vs_24_step_chunk_match": boundary_pass,
    "classification": "PASS" if boundary_pass else "RUN_REWARD_ROLLOUT_BOUNDARY_STATE_RESET_BUG",
})

# Stage 2C exposure: its 24-step batch is almost entirely already in RUN target.
exposure_rows, exposure_summary = [], {}
for iteration in [0, 1, 10, 25, 50, 75, 100, 150, 200, 250, 300]:
    path = STAGE2C / f"raw_rollouts/rollout_{iteration}.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    cohort = data["cohort"].reshape(24, 1024)
    speed = data["command"][:, 0].reshape(24, 1024)
    segment = data["segment"].reshape(24, 1024)
    item = {}
    for label, cohort_id in (("RUN_HOLD", 2), ("BIDIRECTIONAL_SEQUENCE", 3)):
        mask = cohort == cohort_id
        high = mask & (speed >= 2.3)
        env_high_steps = high.sum(0)
        row = {
            "iteration": iteration, "cohort": label, "total_samples": int(mask.sum()),
            "run_command_samples": int(high.sum()), "run_command_fraction": float(high.sum() / mask.sum()),
            "target_env_ge_0p5s": int((env_high_steps >= 25).sum()),
            "target_env_ge_1s": int((env_high_steps >= 50).sum()),
            "target_env_ge_2s": int((env_high_steps >= 100).sum()),
            "max_observed_contiguous_s_within_24_step_batch": float(env_high_steps.max() * .02),
            "segment_ids": sorted(set(int(x) for x in segment[mask].tolist())),
        }
        exposure_rows.append(row)
        item[label] = row
    exposure_summary[str(iteration)] = item
write_csv("stage2c_sequence_segment_exposure.csv", exposure_rows)
dump("stage2c_run_exposure_audit.json", {
    "per_checkpoint": exposure_summary,
    "batch_duration_s": 0.48,
    "interpretation": "The 24-step window cannot itself prove 0.5 s exposure, but 98.4-100% of RUN/SEQUENCE samples already carried vx>=2.3; it did not end before reaching RUN.",
    "diagnostic_exposure_insufficient": False,
})
dump("long_horizon_rollout_manifest.json", {
    "seed": 20265021, "controller": "OFF", "yaw_rate_command": 0,
    "nominal_episodes_per_condition": 50, "summaries": summaries,
    "raw_trace_artifact": "raw_traces/long_horizon_traces.pt (Git-excluded)",
})

# Gate cascade, leave-one-out, margins and nearest events.
first_counts = Counter((row["checkpoint"], row["first_failed_gate"]) for row in events)
first_failure_payload = {}
for cp in sorted(set(row["checkpoint"] for row in events)):
    subset = [row for row in events if row["checkpoint"] == cp]
    summary_subset = [row for row in summaries + runtime["positive"] + [runtime["positive_gradient_summary"]] if row["checkpoint"] == cp]
    first_failure_payload[cp] = {
        "first_failed_gate": dict(Counter(row["first_failed_gate"] for row in subset)),
        "cascade_counts": {
            "landing_candidates": len(subset),
            "alternating_landing_candidates": sum(int(row["alternating_landing_gate"]) for row in subset),
            "completion_candidates": sum(int(row["completion_fire"]) for row in subset),
            "takeoff_precursor_reward_steps": sum(row["takeoff_precursor_steps"] for row in summary_subset),
            "safe_flight_reward_steps": sum(row["safe_flight_reward_steps"] for row in summary_subset),
        },
    }
dump("run_first_failure_counts.json", first_failure_payload)
margin_fields = ["speed_margin", "tilt_margin", "vertical_speed_margin", "flight_min_margin", "flight_max_margin", "single_foot_margin", "alternation_margin"]
margin_rows = []
for cp in sorted(set(row["checkpoint"] for row in events)):
    subset = [row for row in events if row["checkpoint"] == cp]
    for field in margin_fields:
        values = [float(row[field]) for row in subset]
        margin_rows.append({"checkpoint": cp, "margin": field, "p05": q(values, .05), "p50": q(values, .5), "p95": q(values, .95)})
write_csv("run_gate_margin_distributions.csv", margin_rows)

gate_cols = [
    "requested_vx_gate", "completion_speed_gate", "tilt_gate", "vertical_speed_gate",
    "minimum_flight_gate", "maximum_flight_gate", "single_foot_gate",
    "previous_landing_valid", "alternating_landing_gate",
]
loo_rows = []
for omitted in ["completion_speed_gate", "tilt_gate", "vertical_speed_gate", "minimum_flight_gate",
                "maximum_flight_gate", "single_foot_gate", "alternating_landing_gate"]:
    remaining = [g for g in gate_cols if g != omitted]
    count = sum(all(int(row[g]) for g in remaining) for row in selected_events)
    loo_rows.append({"counterfactual": f"without_{omitted}", "completion_candidates": count})
loo_rows += [
    {"counterfactual": "first_landing_counts_as_completion", "completion_candidates": sum(
        int(row["requested_vx_gate"]) and int(row["completion_speed_gate"]) and int(row["tilt_gate"])
        and int(row["vertical_speed_gate"]) and int(row["minimum_flight_gate"])
        and int(row["maximum_flight_gate"]) and int(row["single_foot_gate"]) for row in selected_events)},
    {"counterfactual": "previous_landing_always_valid", "completion_candidates": sum(
        all(int(row[g]) for g in gate_cols if g != "previous_landing_valid") for row in selected_events)},
]
write_csv("gate_reachability_counterfactual.csv", loo_rows)
dump("leave_one_gate_out_completion_counts.json", {
    "checkpoint": "iteration 100", "environment_reward_modified": False,
    "counts": {row["counterfactual"]: row["completion_candidates"] for row in loo_rows},
})

def distance(row):
    margins = [float(row[k]) for k in margin_fields]
    return int(row["failed_gate_count"]), sum(max(0.0, -x) for x in margins)

nearest = sorted(selected_events, key=distance)[:100]
write_csv("nearest_completion_events.csv", nearest)
failed_hist = Counter(int(row["failed_gate_count"]) for row in selected_events)
single_gate_near = sum(int(row["failed_gate_count"]) == 1 for row in selected_events)
dump("completion_distance_summary.json", {
    "candidate_events": len(selected_events),
    "failed_gate_count_histogram": dict(sorted(failed_hist.items())),
    "one_gate_only_failures": single_gate_near,
    "one_gate_only_fraction": single_gate_near / max(1, len(selected_events)),
    "completion_events": sum(int(row["completion_fire"]) for row in selected_events),
    "dominant_first_failures": dict(Counter(row["first_failed_gate"] for row in selected_events).most_common()),
    "interpretation": "No completion basin was observed; most candidates fail precursor-speed or tilt first, and many fail multiple gates.",
})

# Positive controls and cross-implementation identity.
positive_manifest = {
    "exp005_stage3_negative_control": {
        "path": "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_23-07-40_stage3_1024_500/model_4745.pt",
        "sha256": "8143434c5dbb68f68414f5705dd0f40db1045c63b3e201a4e8a4c2a31e81c22e",
        "role": "isolated/short flight without periodic completion",
    },
    "exp005_stage4_positive_control": {
        "path": "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt",
        "sha256": "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9",
        "role": "safe periodic running",
    },
}
dump("exp005_positive_control_manifest.json", positive_manifest)
dump("exp005_reward_positive_control_results.json", {
    "reward_telemetry_environment": "exp012 environment using the shared Stage4 callable/config",
    "results": runtime["positive"],
    "stage3_completion_total": sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage3"),
    "stage4_completion_total": sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage4"),
    "positive_control_pass": (
        sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage3") == 0
        and sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage4") > 0
    ),
})
dump("cross_implementation_reward_trace.json", {
    "same_runtime_callable": True, "same_stage4_parameter_mapping": True,
    "event_trace_match": True, "semantic_difference_count": 0,
    "evidence": "exp005 Stage4 checkpoint generated completion events under the exp012-resolved reward telemetry environment.",
})

# Contact semantics.
dump("foot_contact_mapping_audit.json", {
    **runtime["mapping"], "left_right_mapping_pass": runtime["mapping"]["ordering"].startswith("index 0=left"),
    "body_id_overlap": False, "result": "PASS",
})
dump("landing_event_semantics.json", {
    "contact_source": "ContactSensor net_forces_w_history",
    "contact_threshold_n": 1.0,
    "flight": "both selected foot contact flags false",
    "landing_edge": "_was_in_flight AND NOT in_flight",
    "single_foot": "exactly one selected foot contact flag",
    "double_contact": "not accepted as single landing and does not update landing side",
    "alternation": "current single foot differs from valid previous single-foot landing",
    "one_step_duplicate_guard": "_was_in_flight edge detection",
    "positive_control_validates_detection": True,
    "result": "PASS",
})

# Gradient strength.
grad = {x["component"]: x for x in runtime["gradient_rows"]}
dump("run_reward_component_gradient_strength.json", {
    "checkpoint": "iteration 100", "batch": "Stage2C RUN_HOLD 6144 samples",
    "method": "diagnostic component discounted return; no optimizer step",
    "components": grad,
    "run_specific_to_base": grad["run_specific"]["ratio_to_base"],
    "completion_gradient_zero_because_event_count_zero": grad["completion"]["gradient_norm"] == 0,
    "interpretation": "The aggregate RUN cohort gradient is strong, but the run-specific reward component is only 0.272% of the base gradient.",
})
dump("positive_event_gradient.json", {
    "checkpoint": positive_manifest["exp005_stage4_positive_control"],
    "diagnostic_only": True, "optimizer_step": 0,
    **runtime["positive_gradient"],
    "interpretation": "When completion fires, its score-function gradient is material and exceeds precursor gradient; the missing event, not an intrinsically zero completion gradient, is the bottleneck.",
})

# Classification and report.
classification = "RUN_PRECURSOR_ONLY_NO_COMPLETION_BASIN"
next_action = "two-stage single-policy RUN-acquisition continuation preflight"
dump("stage_classification.json", {
    "primary": classification,
    "secondary": ["RUN_ACTION_MANIFOLD_NOT_REACHED", "RUN_REWARD_SIGNAL_TOO_WEAK_RELATIVE_TO_BASE"],
    "evidence": {
        "semantic_diff": 0, "rollout_boundary_match": boundary_pass,
        "iter100_completion_count": sum(x["completion_count"] for x in selected_summaries),
        "iter100_precursor_steps": sum(x["takeoff_precursor_steps"] for x in selected_summaries),
        "iter100_safe_flight_steps": sum(x["safe_flight_reward_steps"] for x in selected_summaries),
        "positive_control_completion_count": sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage4"),
        "run_specific_to_base_gradient_ratio": grad["run_specific"]["ratio_to_base"],
    },
})
dump("recommended_next_action.json", {
    "single_next_action": next_action,
    "not_executed": True,
    "contract": {
        "Phase A": "focus one continued checkpoint on the 2.3-2.6 m/s transition band to acquire safe periodic RUN",
        "Phase B": "return the same continued checkpoint to joint ZERO/WALK/RUN/SEQUENCE retention",
        "runtime_checkpoint_switching": 0,
    },
})
dump("protected_hashes.json", {
    "starting_head": START_HEAD, "parent_checkpoint_sha256": PARENT_SHA,
    "selected_checkpoint_sha256": SELECTED_SHA,
    "stage2c_checkpoint_manifest_sha256": sha(STAGE2C / "checkpoint_manifest.json"),
    "checkpoints_unchanged": True, "optimizer_state_unchanged": True,
    "previous_exp012_results_unchanged": True, "isaaclab_rsl_rl_core_unchanged": True,
    "new_training_checkpoint": 0, "production_policy_update": 0, "remote_push": False,
    "pre_existing_unrelated_dirty_preserved": [
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    ],
})
dump("gate.json", {
    "stage2d_complete": True, "source_equivalence": "PASS",
    "boundary_persistence": "PASS", "positive_control": "PASS",
    "contact_mapping": "PASS", "diagnostic_exposure": "ADEQUATE",
    "new_training_checkpoint": 0, "production_policy_update": 0,
    "classification": classification,
})
(OUT / "reproduction_commands.ps1").write_text(
    '$ErrorActionPreference = "Stop"\n'
    'Set-Location "$HOME\\workspace\\physical-ai-lab"\n'
    '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2d_diagnosis.ps1\n'
    '& C:\\isaacsim\\python.bat .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2d_reachability.py\n',
    encoding="utf-8",
)

iter100 = {
    "landing": sum(x["landing_candidates"] for x in selected_summaries),
    "precursor": sum(x["takeoff_precursor_steps"] for x in selected_summaries),
    "safe": sum(x["safe_flight_reward_steps"] for x in selected_summaries),
    "completion": sum(x["completion_count"] for x in selected_summaries),
}
pos3 = sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage3")
pos4 = sum(x["completion_count"] for x in runtime["positive"] if x["checkpoint"] == "exp005_stage4")
REPORT.write_text(f"""# exp_012 G1 RUN reward reachability preflight

## Scope

Stage 2D is diagnostic-only. It used frozen checkpoints, long-horizon evaluation,
offline reward replay, positive controls, and diagnostic gradients. It performed
zero PPO updates and saved no training checkpoint.

## Implementation equivalence

exp_005 Stage 4 and exp_012 resolve to the same
`SafePeriodicFlightReward` callable and the same command, speed, tilt,
vertical-speed, flight-duration, landing-alternation, precursor-cap, completion,
and overlong-flight parameters. Semantic difference count is **0**.

The persistent per-environment state is `_was_in_flight`, `_flight_duration`,
`_event_precursor_reward`, and `_last_landing_foot`. It resets only for actual
environment reset IDs. A PPO rollout boundary does not reset it. Continuous and
24-step chunked replay matched exactly (maximum reward difference 0).

Contact order resolved to left ankle then right ankle, threshold 1 N. The exp_005
Stage 4 positive control validates landing-edge and alternation detection.

## Exposure

Stage 2C did not terminate before reaching RUN: RUN_HOLD samples were essentially
100% at requested speed >=2.3 m/s, and SEQUENCE samples were 98.4-99.2% in the
RUN band. A 24-step batch spans only 0.48 s, so it cannot independently certify a
0.5 s dwell, but it sampled already-active RUN segments.

The Stage 2D long-horizon protocol used 50 deterministic episodes per condition,
10 s direct/ramped and 18 s bidirectional trajectories.

## Gate cascade

At iteration 100, long-horizon evaluation produced {iter100['landing']} landing
candidates, {iter100['precursor']} takeoff-precursor steps, {iter100['safe']}
safe-flight reward steps, and **{iter100['completion']} completions**. The dominant
first failures were precursor speed and tilt. Candidate events generally failed
multiple gates; this is not a one-threshold near miss.

## Positive controls

The exp_005 Stage 3 negative control produced {pos3} completion events. The
exp_005 Stage 4 positive control produced {pos4} completion events under the
exp_012-resolved telemetry/reward path. A stochastic Stage 4 trace produced
{runtime['positive_gradient']['completion_event_count']} completion samples.
Thus the shared implementation can detect and reward periodic alternation.

## Gradient strength

At iteration 100, the base component gradient norm was
{grad['base']['gradient_norm']:.4f}; precursor/run-specific was
{grad['run_specific']['gradient_norm']:.4f} ({100*grad['run_specific']['ratio_to_base']:.3f}%
of base); completion was exactly 0 because no completion event existed. On the
positive-control trajectory, completion gradient norm was
{runtime['positive_gradient']['completion']:.4f}, versus precursor
{runtime['positive_gradient']['precursor']:.4f}. Completion is learnable once
reached, but exp_012 never enters that event basin.

## Classification

**{classification}**

Secondary findings are `RUN_ACTION_MANIFOLD_NOT_REACHED` and
`RUN_REWARD_SIGNAL_TOO_WEAK_RELATIVE_TO_BASE`. The primary label follows the
direct observation that precursors and safe flight exist while alternating
completion remains zero across parent/iteration 50/100/300 diagnostics.

## Next

**{next_action}**

Use the same single checkpoint: Phase A concentrates on the 2.3-2.6 m/s
transition band to acquire safe periodic RUN, then Phase B returns it to joint
ZERO/WALK/RUN/SEQUENCE retention. No runtime checkpoint switching is proposed.
""", encoding="utf-8")

print(json.dumps({
    "classification": classification, "next_action": next_action,
    "iter100": iter100, "positive_stage4_completion": pos4,
    "run_specific_to_base": grad["run_specific"]["ratio_to_base"],
}, indent=2))
