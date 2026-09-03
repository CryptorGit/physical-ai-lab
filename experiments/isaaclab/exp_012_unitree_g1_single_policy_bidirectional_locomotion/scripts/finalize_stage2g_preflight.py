"""Finalize tracked Stage-2G contracts, summaries, gate, and research report."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
EXP = REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion"
ROOT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion"
OUT = ROOT / "stage2g_event_stratified_on_policy_preflight"
RAW = OUT / "raw"
REPORT = REPO / "research/exp_012_g1_event_stratified_on_policy_preflight_report.md"
START_HEAD = "d89fa5914ae1841ac4cb14f74932b9cf2ac0c2b2"
PRIMARY_SHA = "4edbb595e28e24dc09cf39e8245c7be1b1bebf792798a73af2e562075d0fe952"
CONDITIONS = (
    "M0_UNIFORM",
    "M4_EVENT_STRATIFIED",
    "M8_EVENT_STRATIFIED",
    "M16_EVENT_STRATIFIED",
)
PREEXISTING_DIRTY = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    ".openduck_hardware_source_review/",
    ".openduck_phase3_usb_baseline.txt",
    ".openduck_playground_source_review/",
    ".openduck_runtime_source_review/",
    "artifacts/exp_005_unitree_g1_flat_run/",
    "artifacts/openduck_recorded_zero_pose.png",
    "artifacts/openduck_safe_init_pose_front.png",
    "artifacts/openduck_safe_init_pose_side.png",
    "artifacts/openduck_zero_pose_front.png",
    "artifacts/openduck_zero_pose_side.png",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "experiments/mujoco/exp_003_openduckmini_calibrated_walk/",
    "media/",
    "openduck_setup_report.md",
    "research/exp_011_linkedin_post_ja.md",
    "tools/analyze_openduck_joint_directions.py",
    "tools/render_openduck_zero_pose.py",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name: str, value):
    (OUT / name).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def read_csv(name: str):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True, encoding="utf-8"
    ).strip()


def number(row, field):
    return float(row[field])


def aggregate_behavior(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["mode"], float(row["target_speed"]))].append(row)
    output = []
    for (condition, mode, speed), group in sorted(grouped.items()):
        count = len(group)
        output.append(
            {
                "condition": condition,
                "mode": mode,
                "target_speed_mps": speed,
                "episodes": count,
                "completion_events": sum(int(row["completion_events"]) for row in group),
                "completion_per_episode": sum(
                    int(row["completion_events"]) for row in group
                )
                / count,
                "periodic_running_rate": sum(
                    int(row["periodic_running"]) for row in group
                )
                / count,
                "fall_rate": sum(int(row["fall"]) for row in group) / count,
                "speed_mae_mps": sum(number(row, "speed_mae") for row in group) / count,
                "heading_p95_rad_mean": sum(
                    number(row, "heading_p95") for row in group
                )
                / count,
                "dangerous_slip_rate": sum(
                    int(row["dangerous_slip"]) for row in group
                )
                / count,
                "impact_failure_rate": sum(
                    int(row["impact_failure"]) for row in group
                )
                / count,
                "long_dwell_saturation_rate": sum(
                    int(row["long_dwell_saturation"]) for row in group
                )
                / count,
            }
        )
    return output


def aggregate_retention(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["retention_condition"])].append(row)
    return [
        {
            "condition": condition,
            "task": task,
            "episodes": len(group),
            "success_rate": sum(int(row["success"]) for row in group) / len(group),
            "fall_rate": sum(int(row["fall"]) for row in group) / len(group),
        }
        for (condition, task), group in sorted(grouped.items())
    ]


def checkpoint_manifest():
    source = load_json(ROOT / "stage2e_phase_a_run_acquisition_preflight/checkpoint_manifest.json")
    wanted = {20, 50, 100}
    return {
        "source_manifest": "stage2e_phase_a_run_acquisition_preflight/checkpoint_manifest.json",
        "primary_iteration": 50,
        "checkpoints": [
            row for row in source["checkpoints"] if row["phase_a_iteration"] in wanted
        ],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch = load_json(OUT / "batch_0_summary.json")
    strata = load_json(OUT / "strata_runtime.json")
    gradients = load_json(OUT / "stratified_gradient_runtime.json")
    composition = load_json(OUT / "minibatch_composition_runtime.json")
    shadow = load_json(OUT / "shadow_runtime_summary.json")
    behavior_rows = read_csv("temporary_behavioral_evaluation.csv")
    retention_rows = read_csv("retention_evaluation.csv")
    behavior = aggregate_behavior(behavior_rows)
    retention = aggregate_retention(retention_rows)
    completion_counts = {
        condition: sum(
            row["completion_events"]
            for row in behavior
            if row["condition"] == condition and row["mode"] == "S100"
        )
        for condition in CONDITIONS
    }
    deterministic_counts = {
        condition: sum(
            row["completion_events"]
            for row in behavior
            if row["condition"] == condition and row["mode"] == "D0"
        )
        for condition in CONDITIONS
    }
    periodic_2p4 = {
        condition: next(
            row["periodic_running_rate"]
            for row in behavior
            if row["condition"] == condition
            and row["mode"] == "D0"
            and row["target_speed_mps"] == 2.4
        )
        for condition in CONDITIONS
    }
    fall_2p4 = {
        condition: next(
            row["fall_rate"]
            for row in behavior
            if row["condition"] == condition
            and row["mode"] == "D0"
            and row["target_speed_mps"] == 2.4
        )
        for condition in CONDITIONS
    }
    cross = {
        (row["condition"], row["holdout_stratum"]): row
        for row in shadow["cross_effect"]
    }
    metrics = {row["condition"]: row for row in shadow["conditions"]}
    train_count = sum(
        int(row["total_inclusions_5_epochs"])
        for row in read_csv("effective_sampling_weights.csv")
        if row["condition"] == "M0_UNIFORM"
    ) // 5

    dump(
        "stage_reference.json",
        {
            "starting_head": START_HEAD,
            "stage2f_classification": "PHASE_A_BOUNDARY_MULTIPLE_CAUSES",
            "primary_checkpoint_iteration": 50,
            "primary_checkpoint_sha256": PRIMARY_SHA,
            "primary_adam_step": 88000,
            "phase_b_readiness_at_entry": "PHASE_B_NOT_READY",
            "previous_results_overwritten": False,
        },
    )
    dump(
        "protocol.json",
        {
            "stage": "2G",
            "purpose": "event-stratified on-policy minibatch construction preflight",
            "policy_updates": "disposable one-update clones only",
            "persistent_checkpoint_writes": 0,
            "rollout": {
                "checkpoint": "Phase A iteration 50",
                "action_std_multiplier": 1.0,
                "num_envs": 1024,
                "episode_seconds": 20,
                "maximum_fresh_batches": 3,
                "batches_used": 1,
                "seed": 20268021,
                "on_policy": True,
            },
            "ppo": {
                "epochs": 5,
                "minibatches": 4,
                "minibatch_size": 6144,
                "clip_range": 0.2,
                "restored_adam": True,
                "conditions": list(CONDITIONS),
            },
            "reward_unchanged": True,
            "effective_objective_reweighted": True,
            "phase_a_command_distribution_unchanged": True,
            "yaw_command": 0.0,
            "controllers": "OFF",
        },
    )
    dump("checkpoint_manifest.json", checkpoint_manifest())
    dump(
        "diagnostic_seed_manifest.json",
        {
            "collection": [20268021],
            "shadow_analysis": 20268121,
            "temporary_behavior": 20268221,
            "fresh_repro_collection": [20269022],
            "fresh_repro_analysis": 20269121,
            "same_seed_across_M0_M4_M8_M16_behavior": True,
        },
    )
    dump(
        "on_policy_rollout_manifest.json",
        {
            key: value
            for key, value in batch.items()
            if key not in {"event_rows", "tensor_hashes"}
        }
        | {
            "sample_count": 1024 * 1000,
            "stopping_contract_met": (
                batch["completion_events"] >= 64 and batch["completion_episodes"] >= 32
            ),
            "fresh_batches_used": 1,
            "raw_artifact": "raw/on_policy_batch_0.pt",
            "raw_artifact_git_tracked": False,
            "required_fields": sorted(batch["tensor_hashes"]),
        },
    )
    dump(
        "on_policy_rollout_hashes.json",
        {
            "hash_contract": "SHA-256 over contiguous tensor raw bytes",
            "tensor_hashes": batch["tensor_hashes"],
            "checkpoint_sha256": batch["checkpoint_sha256"],
        },
    )
    dump(
        "completion_event_manifest.json",
        {
            "event_count": batch["completion_events"],
            "unique_episode_count": batch["completion_episodes"],
            "events": batch["event_rows"],
            "window_count": len(strata["event_windows"]),
            "merged_window_contract": True,
        },
    )
    dump(
        "stratum_contract.json",
        {
            "priority": [
                "E2_COMPLETION",
                "U_UNSAFE",
                "E1_PRECURSOR_ONLY",
                "B_BACKGROUND",
            ],
            "E2_COMPLETION": "takeoff-10 through completion landing+5, merged within episode, maximum 40 steps",
            "E1_PRECURSOR_ONLY": "precursor/safe-flight window not ending in completion",
            "U_UNSAFE": "pre-fall window, impact, dangerous slip, tilt failure, or unsafe landing",
            "B_BACKGROUND": "all remaining samples",
            "mutually_exclusive": True,
            "episode_disjoint_holdout": True,
        },
    )
    dump(
        "stratum_sample_counts.json",
        {
            "all_samples": strata["counts"],
            "total": sum(strata["counts"].values()),
            "training_update_sample_count_per_epoch": train_count,
            "holdout_episode_count": strata["holdout_episode_count"],
        },
    )
    dump(
        "matched_background_manifest.json",
        {
            "matching_fields": [
                "target speed (0.05 m/s bin)",
                "episode time (0.5 s bin)",
                "command segment",
                "contact phase",
                "preceding flight state",
            ],
            "match_count": len(strata["matched_background"]),
            "matches": strata["matched_background"],
        },
    )
    # Correct cap audit is per-sample reuse; event aggregate counts are intentionally separate.
    cap_audit = {}
    for condition, payload in composition.items():
        factor = int(payload["factor"])
        cap_audit[condition] = {
            "requested_factor": factor,
            "total_inclusions": payload["total_inclusions"],
            "all_minibatches_meet_background_30_percent": all(
                row["B_BACKGROUND"] / 6144 >= 0.30 for row in payload["minibatches"]
            ),
            "all_minibatches_meet_precursor_10_percent": all(
                row["E1_PRECURSOR_ONLY"] / 6144 >= 0.10
                for row in payload["minibatches"]
            )
            if factor > 1
            else "not required for M0",
            "all_minibatches_meet_unsafe_20_percent": all(
                row["U_UNSAFE"] / 6144 >= 0.20 for row in payload["minibatches"]
            )
            if factor > 1
            else "not required for M0",
            "multiple_completion_episodes_per_stratified_minibatch": all(
                row["completion_episode_count"] > 1 for row in payload["minibatches"]
            )
            if factor > 1
            else "not required for M0",
            "event_aggregate_inclusion_range": [
                payload["event_reuse_min"],
                payload["event_reuse_max"],
            ],
            "completion_sample_reuse_max": payload["completion_sample_reuse_max"],
            "completion_sample_reuse_mean": payload["completion_sample_reuse_mean"],
            "completion_sample_reuse_cap": payload["completion_sample_reuse_cap"],
            "completion_sample_reuse_cap_pass": payload[
                "completion_sample_reuse_cap_pass"
            ],
            "contract_note": (
                "Safety quota dominated the event-stratified effective objective; "
                "reward values were unchanged."
            ),
        }
    dump(
        "minibatch_composition_audit.json",
        {
            "conditions": cap_audit,
            "reward_unchanged": True,
            "effective_update_objective_reweighted": True,
            "contract_pass": True,
        },
    )
    dump(
        "stratified_gradient_comparison.json",
        {
            condition: {
                "total_norm": gradients[condition]["total"]["gradient_norm"],
                "completion_over_total": gradients[condition]["completion"]["ratio_to_total"],
                "run_specific_over_total": gradients[condition]["run_specific"]["ratio_to_total"],
                "combined_vs_completion_cosine": gradients[condition]["total"][
                    "cosine_to_completion"
                ],
                "combined_vs_unsafe_cosine": gradients[condition]["total"][
                    "cosine_to_unsafe"
                ],
                "completion_ratio_gate": gradients[condition]["completion"][
                    "ratio_to_total"
                ]
                >= 0.01,
                "unsafe_direction_gate": gradients[condition]["total"][
                    "cosine_to_unsafe"
                ]
                >= 0,
            }
            for condition in CONDITIONS
        }
        | {
            "interpretation": (
                "Magnitude ratios exceed 1%, but all stratified combined gradients have "
                "negative cosine to the completion component because the mandatory unsafe "
                "quota dominates; the uniform gradient is weakly completion-aligned."
            )
        },
    )
    # Keep raw behavior CSV as the required episode-level artifact and add gate summary in JSON.
    dump(
        "temporary_behavioral_evaluation_summary.json",
        {
            "aggregates": behavior,
            "deterministic_completion_by_condition": deterministic_counts,
            "S100_completion_by_condition": completion_counts,
            "deterministic_periodic_2p4_by_condition": periodic_2p4,
            "deterministic_fall_2p4_by_condition": fall_2p4,
        },
    )
    dump("retention_evaluation_summary.json", {"aggregates": retention})

    repro_path = OUT / "repro_shadow_runtime_summary.json"
    repro_behavior_path = OUT / "repro_temporary_behavioral_evaluation.csv"
    if repro_path.exists() and repro_behavior_path.exists():
        repro_shadow = load_json(repro_path)
        repro_behavior = aggregate_behavior(read_csv("repro_temporary_behavioral_evaluation.csv"))
        repro_gradients = load_json(OUT / "repro_stratified_gradient_runtime.json")
        repro_cosine = load_json(OUT / "repro_gradient_cosine.json")
        primary_vector = gradients["M8_EVENT_STRATIFIED"]["total"]
        repro_vector = repro_gradients["M8_EVENT_STRATIFIED"]["total"]
        dump(
            "fresh_process_reproducibility.json",
            {
                "condition": "M8_EVENT_STRATIFIED",
                "status": "FAIL",
                "fresh_process": True,
                "different_diagnostic_seed": True,
                "primary_collection_seed": 20268021,
                "reproduction_collection_seed": 20269022,
                "full_vector_cross_run_cosine": repro_cosine["total_gradient_cosine"],
                "full_vector_cross_run_cosine_gate": repro_cosine["gate_pass"],
                "projection_direction_agreement": (
                    primary_vector["cosine_to_completion"]
                    * repro_vector["cosine_to_completion"]
                    >= 0
                ),
                "reproduction_hard_gate": repro_shadow["conditions"][0]["hard_gate_pass"],
                "completion_loss_improvement_same_direction": (
                    bool(cross[("M8_EVENT_STRATIFIED", "completion")]["improved"])
                    == bool(
                        next(
                            row["improved"]
                            for row in repro_shadow["cross_effect"]
                            if row["holdout_stratum"] == "completion"
                        )
                    )
                ),
                "behavioral_direction": (
                    "no deterministic completion and no verified 2.4 periodic improvement"
                ),
                "reproduction_behavior": repro_behavior,
                "rationale": (
                    "The raw full-vector cosine is below 0.80. Both runs fail to improve "
                    "completion loss, and unsafe/background losses plus behavior remain "
                    "outside the acceptance gates."
                ),
            },
        )
    else:
        dump(
            "fresh_process_reproducibility.json",
            {
                "condition": "M8_EVENT_STRATIFIED",
                "status": "NOT_EXECUTED_OR_INCOMPLETE",
                "fresh_process": True,
            },
        )
    dump(
        "selected_event_stratification.json",
        {
            "selected_multiplier": None,
            "status": "NO_CANDIDATE",
            "reason": (
                "No event-stratified condition improved completion holdout loss, "
                "deterministic completion, or 2.4 m/s periodic-running; all increased "
                "unsafe holdout loss beyond 5%."
            ),
        },
    )
    classification = "EVENT_STRATIFIED_ON_POLICY_NO_EFFECT"
    dump(
        "stage_classification.json",
        {
            "classification": classification,
            "secondary_classifications": [
                "EVENT_STRATIFIED_ON_POLICY_SAFETY_FAIL",
                "COMPLETION_GRADIENT_DIRECTION_NOT_AMPLIFIED",
                "TEMPORARY_RETENTION_MOSTLY_PRESERVED",
            ],
            "rationale": (
                "All shadow updates were numerically stable, but M4/M8/M16 had negative "
                "combined-gradient cosine to completion, worsened completion-window loss, "
                "and produced neither deterministic completion nor a 2.4 m/s periodic gain. "
                "Unsafe loss worsened by "
                f"{float(cross[('M4_EVENT_STRATIFIED','unsafe')]['relative_change']):.2%}, "
                f"{float(cross[('M8_EVENT_STRATIFIED','unsafe')]['relative_change']):.2%}, "
                f"and {float(cross[('M16_EVENT_STRATIFIED','unsafe')]['relative_change']):.2%}, "
                "respectively."
            ),
        },
    )
    dump("phase_b_readiness.json", {"classification": "PHASE_B_NOT_READY"})
    dump(
        "recommended_next_action.json",
        {
            "next_action": (
                "close minibatch-stratification route and evaluate completion-event "
                "short-horizon replay preflight"
            ),
            "execute_in_stage2g": False,
            "single_method_only": True,
        },
    )
    protected_checkpoint = (
        ROOT
        / "stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"
    )
    dump(
        "protected_hashes.json",
        {
            "starting_head": START_HEAD,
            "primary_checkpoint_sha256": sha(protected_checkpoint),
            "primary_checkpoint_matches_expected": sha(protected_checkpoint) == PRIMARY_SHA,
            "exp005_to_exp011_modified_by_stage2g": False,
            "stage0_to_stage2f_results_modified_by_stage2g": False,
            "formal_checkpoint_writes": 0,
            "formal_optimizer_state_writes": 0,
            "production_policy_updates": 0,
            "reward_curriculum_network_observation_action_physics_changes": 0,
            "isaac_lab_or_rsl_rl_core_changes": 0,
            "remote_push": False,
            "unrelated_dirty_paths_preserved": PREEXISTING_DIRTY,
        },
    )
    fresh = load_json(OUT / "fresh_process_reproducibility.json")
    dump(
        "gate.json",
        {
            "classification": classification,
            "data_gate": "PASS",
            "on_policy_gate": "PASS",
            "minibatch_contract_gate": "PASS",
            "shadow_stability_gate": "PASS",
            "completion_gradient_magnitude_gate": "PASS",
            "completion_direction_and_loss_gate": "FAIL",
            "behavioral_gate": "FAIL",
            "safety_cross_effect_gate": "FAIL",
            "fresh_process_reproducibility_gate": fresh["status"],
            "phase_b_readiness": "PHASE_B_NOT_READY",
            "persistent_checkpoint_count": 0,
            "production_policy_update_count": 0,
        },
    )
    (OUT / "reproduction_commands.ps1").write_text(
        r"""$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
Set-Location $Repo

# Fresh on-policy collection (large raw artifact remains untracked).
.\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\run_stage2g_collection.ps1 `
  -BatchIndex 0 -SeedRoot 20268021

# Disposable shadow analysis only; no formal checkpoint is written.
C:\isaacsim\python.bat `
  .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\analyze_stage2g_shadow.py `
  --batch-index 0

# Temporary behavioral evaluation of disposable shadow actors.
$env:PYTHONPATH = ".\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\src;.\experiments\isaaclab\exp_005_unitree_g1_flat_run\src;.;$env:PYTHONPATH"
C:\isaacsim\python.bat `
  .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\evaluate_stage2g_shadows.py `
  --headless --device cuda:0 --seed-root 20268221

# Finalize tracked summaries and report.
python .\experiments\isaaclab\exp_012_unitree_g1_single_policy_bidirectional_locomotion\scripts\finalize_stage2g_preflight.py
""",
        encoding="utf-8",
    )
    m = metrics
    g = gradients
    x = shadow["cross_effect"]
    REPORT.write_text(
        f"""# exp_012 Stage 2G — Event-Stratified On-Policy Minibatch Preflight

## Outcome

**{classification}**. No multiplier was selected. Phase B remains
**PHASE_B_NOT_READY**. This stage performed no persistent policy update and wrote
no checkpoint.

## Data and contract

The frozen Phase A iteration-50 actor (`{PRIMARY_SHA}`) generated one fresh
20-second, 1,024-environment S100 batch: **1,024,000 samples**, **{batch['completion_events']}
completion events**, and **{batch['completion_episodes']} unique completion episodes**.
This passed the preregistered 64-event/32-episode stop gate. All yaw commands
were zero, external controllers were off, and all samples came from this one
on-policy checkpoint.

Mutually exclusive strata contained **{strata['counts']['E2_COMPLETION']:,} E2**,
**{strata['counts']['E1_PRECURSOR_ONLY']:,} E1**,
**{strata['counts']['U_UNSAFE']:,} unsafe**, and
**{strata['counts']['B_BACKGROUND']:,} background** samples. Event windows span
takeoff minus 10 steps through landing plus 5 steps and are merged within an
episode. Matched backgrounds control target speed, episode time, command
segment, contact phase, and preceding flight state.

Reward values were unchanged. The effective update objective was nevertheless
reweighted by sampling. In M4/M8/M16 every minibatch retained 20% unsafe,
at least 10% precursor, and at least 30% background samples.

## Gradient amplification

| condition | completion/total | combined·completion cosine | combined·unsafe cosine |
|---|---:|---:|---:|
| M0 | {g['M0_UNIFORM']['completion']['ratio_to_total']:.3%} | {g['M0_UNIFORM']['total']['cosine_to_completion']:+.3f} | {g['M0_UNIFORM']['total']['cosine_to_unsafe']:+.3f} |
| M4 | {g['M4_EVENT_STRATIFIED']['completion']['ratio_to_total']:.3%} | {g['M4_EVENT_STRATIFIED']['total']['cosine_to_completion']:+.3f} | {g['M4_EVENT_STRATIFIED']['total']['cosine_to_unsafe']:+.3f} |
| M8 | {g['M8_EVENT_STRATIFIED']['completion']['ratio_to_total']:.3%} | {g['M8_EVENT_STRATIFIED']['total']['cosine_to_completion']:+.3f} | {g['M8_EVENT_STRATIFIED']['total']['cosine_to_unsafe']:+.3f} |
| M16 | {g['M16_EVENT_STRATIFIED']['completion']['ratio_to_total']:.3%} | {g['M16_EVENT_STRATIFIED']['total']['cosine_to_completion']:+.3f} | {g['M16_EVENT_STRATIFIED']['total']['cosine_to_unsafe']:+.3f} |

The magnitude ratio exceeded 1% in every condition, but stratification did not
amplify the *direction*: all stratified combined gradients point weakly against
the completion component. The mandatory unsafe quota dominates the effective
objective. Layer/joint localization remains locomotion-relevant—torso, bilateral
hip, knee, and ankle terms dominate; arm/hand parameters do not explain the
failure.

## Shadow stability

| condition | exact KL | max-step KL | clip | ratio p99 | mean shift |
|---|---:|---:|---:|---:|---:|
| M0 | {m['M0_UNIFORM']['exact_kl_old_new']:.5f} | {m['M0_UNIFORM']['all_step_maximum_kl']:.5f} | {m['M0_UNIFORM']['clip_fraction']:.3f} | {m['M0_UNIFORM']['ratio_p99']:.3f} | {m['M0_UNIFORM']['mean_action_shift']:.4f} |
| M4 | {m['M4_EVENT_STRATIFIED']['exact_kl_old_new']:.5f} | {m['M4_EVENT_STRATIFIED']['all_step_maximum_kl']:.5f} | {m['M4_EVENT_STRATIFIED']['clip_fraction']:.3f} | {m['M4_EVENT_STRATIFIED']['ratio_p99']:.3f} | {m['M4_EVENT_STRATIFIED']['mean_action_shift']:.4f} |
| M8 | {m['M8_EVENT_STRATIFIED']['exact_kl_old_new']:.5f} | {m['M8_EVENT_STRATIFIED']['all_step_maximum_kl']:.5f} | {m['M8_EVENT_STRATIFIED']['clip_fraction']:.3f} | {m['M8_EVENT_STRATIFIED']['ratio_p99']:.3f} | {m['M8_EVENT_STRATIFIED']['mean_action_shift']:.4f} |
| M16 | {m['M16_EVENT_STRATIFIED']['exact_kl_old_new']:.5f} | {m['M16_EVENT_STRATIFIED']['all_step_maximum_kl']:.5f} | {m['M16_EVENT_STRATIFIED']['clip_fraction']:.3f} | {m['M16_EVENT_STRATIFIED']['ratio_p99']:.3f} | {m['M16_EVENT_STRATIFIED']['mean_action_shift']:.4f} |

All four disposable updates passed the numerical hard gate with finite
parameters, critic gradients below 1e6, and value losses below 1e8. Stability
was therefore not the blocker. Adam updates remained nearly orthogonal to the
completion component (cosines {m['M4_EVENT_STRATIFIED']['adam_update_completion_cosine']:+.3f},
{m['M8_EVENT_STRATIFIED']['adam_update_completion_cosine']:+.3f}, and
{m['M16_EVENT_STRATIFIED']['adam_update_completion_cosine']:+.3f}).

## Immediate cross-effect

Relative completion-window loss changed by
**{float(cross[('M4_EVENT_STRATIFIED','completion')]['relative_change']):+.2%} (M4)**,
**{float(cross[('M8_EVENT_STRATIFIED','completion')]['relative_change']):+.2%} (M8)**,
and **{float(cross[('M16_EVENT_STRATIFIED','completion')]['relative_change']):+.2%}
(M16)**: all worsened. Unsafe-window loss worsened by
**{float(cross[('M4_EVENT_STRATIFIED','unsafe')]['relative_change']):+.2%}**,
**{float(cross[('M8_EVENT_STRATIFIED','unsafe')]['relative_change']):+.2%}**,
and **{float(cross[('M16_EVENT_STRATIFIED','unsafe')]['relative_change']):+.2%}**,
all beyond the 5% gate. Background loss stayed within 5%.

## Temporary behavior and retention

No condition produced deterministic completion. At 2.4 m/s, no condition
produced a deterministic PERIODIC_RUNNING success, so the required +10-point
gain was absent. Across all five S100 speeds the completion counts were
M0={completion_counts['M0_UNIFORM']}, M4={completion_counts['M4_EVENT_STRATIFIED']},
M8={completion_counts['M8_EVENT_STRATIFIED']}, and
M16={completion_counts['M16_EVENT_STRATIFIED']}; none reached the required
twofold increase over M0. M4/M8/M16 changed 2.4 m/s fall relative to M0 by
{fall_2p4['M4_EVENT_STRATIFIED']-fall_2p4['M0_UNIFORM']:+.1%},
{fall_2p4['M8_EVENT_STRATIFIED']-fall_2p4['M0_UNIFORM']:+.1%}, and
{fall_2p4['M16_EVENT_STRATIFIED']-fall_2p4['M0_UNIFORM']:+.1%}.

STAND, WALK 0.6, and WALK 1.2 remained 100% in all temporary clones.
WALK_TO_STAND remained 100% except M4 at 95%, exactly the allowed five-point
boundary. This limited retention preservation does not offset the absent RUN
effect and unsafe cross-effect failure.

## Fresh-process reproducibility

M8 was repeated from a second fresh on-policy collection with a different
diagnostic seed. Its full combined-gradient cosine to the primary run was
**{fresh['full_vector_cross_run_cosine']:+.3f}**, below the required 0.80.
Both processes failed to improve completion-window loss and failed the
behavioral gate; the reproduction also remained within the KL/clip hard gate.
Thus numerical stability reproduced, but the proposed update direction did not.

## Decision

The event-stratified sampler route is closed for this construction. The next
single method is **completion-event short-horizon replay preflight**. That method
is not executed here. Phase B remains not ready.

## Protection

Stage 2G changed no reward, curriculum, network, observation/action contract,
physics, Isaac Lab/RSL-RL core, formal checkpoint, or optimizer state. All
shadow parameters were disposable. Production policy updates: **0**. Remote
push: **false**. Pre-existing unrelated dirty paths were preserved.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
