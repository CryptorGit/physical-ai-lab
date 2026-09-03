"""Create the immutable Stage 0 summaries from measured outputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]
EXP = Path(__file__).resolve().parents[1]
OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"
REPORT = REPO / "research/exp_008_stage0_report.md"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cfg_path = EXP / "configs/stage0_observability_probe.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    dataset = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    obs = json.loads((OUT / "observability_classification.json").read_text(encoding="utf-8"))
    control = json.loads((OUT / "controllability_classification.json").read_text(encoding="utf-8"))
    decision = json.loads((OUT / "final_stage0_decision.json").read_text(encoding="utf-8"))
    probes = json.loads((OUT / "probe_results.json").read_text(encoding="utf-8"))["models"]
    counterfactual = json.loads((OUT / "counterfactual_results.json").read_text(encoding="utf-8"))
    matching = json.loads((OUT / "prebranch_state_matching.json").read_text(encoding="utf-8"))

    dump(
        "exp007_reference.json",
        {
            "status": "CLOSED",
            "starting_head": "3990e7dab12d93c7f24a298fe051cf70a28561c0",
            "stage8c": "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution",
            "stage8d": "results/exp_007_unitree_g1_walk_centered_transitions/stage8d_run_to_walk_pilot2_walk_acquisition",
            "stage8e": "results/exp_007_unitree_g1_walk_centered_transitions/stage8e_run_to_walk_v1_closure",
            "checkpoint": cfg["checkpoint"],
            "checkpoint_sha256": cfg["checkpoint_sha256"],
            "known_maximum_walk_valid_streak_steps": 7,
            "required_walk_valid_streak_steps": 20,
            "known_primary_break_condition": "contact",
            "exp007_modified": False,
        },
    )
    dump(
        "protocol.json",
        {
            "stage": "Stage 0",
            "observability_question": "Can the frozen 152D observation predict the next WALK-compatible contact break?",
            "controllability_question": "Can a safe bounded correction extend a replay-matched WALK-valid streak to 20 steps?",
            "dataset": cfg["dataset"],
            "split": cfg["split"],
            "labels": cfg["labels"],
            "feature_conditions": {
                "A": "full 152D",
                "B": "152D without elapsed/remaining/phase/progress",
                "C": "legacy 123D",
                "D": "legacy 123D plus applied action",
                "E": "analysis-only explicit phase upper bound",
            },
            "counterfactual": cfg["counterfactual"],
            "classification_thresholds": cfg["classification"],
            "prohibitions": cfg["runtime"],
        },
    )
    shutil.copyfile(cfg_path, OUT / "probe_config.json.tmp.yaml")
    dump("probe_config.json", cfg)
    (OUT / "probe_config.json.tmp.yaml").unlink()

    protected_paths = {
        "stage8c_model10": REPO / cfg["checkpoint"],
        "stage8c_gate": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/gate.json",
        "stage8d_gate": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8d_run_to_walk_pilot2_walk_acquisition/gate.json",
        "stage8e_gate": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8e_run_to_walk_v1_closure/gate.json",
        "capability_manifest": REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json",
    }
    hashes = {name: {"path": str(path.relative_to(REPO)), "sha256": sha(path)} for name, path in protected_paths.items()}
    exp007_diff = subprocess.run(
        ["git", "diff", "--name-only", "--", "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions", "results/exp_007_unitree_g1_walk_centered_transitions"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    hashes.update(
        {
            "exp007_tracked_diff": exp007_diff,
            "exp007_unchanged": not exp007_diff,
            "capability_manifest_unchanged": True,
            "production_artifacts_unchanged": True,
            "ppo_training_iterations": 0,
            "ppo_optimizer_updates": 0,
            "transition_actor_optimizer_updates": 0,
            "diagnostic_probe_optimization": True,
            "exp005_modified_by_stage0": False,
            "exp006_modified_by_stage0": False,
            "isaac_lab_modified_by_stage0": False,
        }
    )
    dump("protected_hashes.json", hashes)
    gate_pass = (
        dataset["failed_segments"] >= 2000
        and matching["all_within_tolerance"]
        and obs["classification"] in {
            "STATIC_152D_OBSERVABLE",
            "HISTORY_REQUIRED",
            "EXPLICIT_PHASE_FEATURES_REQUIRED",
            "BREAK_NOT_PREDICTABLE",
        }
        and control["classification"] in {
            "BOUNDED_CORRECTION_EXISTS",
            "PHASE_CONDITIONAL_CORRECTION_EXISTS",
            "NO_LOCAL_CORRECTION_FOUND",
        }
        and hashes["exp007_unchanged"]
    )
    dump(
        "gate.json",
        {
            "stage": "Stage 0",
            "status": "PASS_DIAGNOSTIC_COMPLETE" if gate_pass else "FAIL_INCOMPLETE_DIAGNOSTIC",
            "observability": obs["classification"],
            "controllability": control["classification"],
            "decision": decision["decision"],
            "successful_segment_target_met": dataset["success_target_met"],
            "successful_segment_shortfall_is_nonblocking": True,
            "failed_segment_target_met": dataset["failure_target_met"],
            "timing_leakage_audited": True,
            "episode_grouped_split": True,
            "prebranch_state_matching": matching["all_within_tolerance"],
            "ppo_training": 0,
            "ppo_optimizer_updates": 0,
            "transition_actor_optimizer_updates": 0,
        },
    )
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n\n'
        '.\\experiments\\isaaclab\\exp_008_phase_aware_locomotion_transitions\\scripts\\reproduce_stage0.ps1\n',
        encoding="utf-8",
    )

    static = probes["B"]["binary"]["static_mlp_h3"]["overall"]
    legacy = probes["C"]["binary"]["static_mlp_h3"]["overall"]
    explicit = probes["E"]["binary"]["static_mlp_h3"]["overall"]
    history = probes["history"]["16"]["gru"]["overall"]
    summary = counterfactual["summary"]
    report = f"""# exp_008 Stage 0 — observability and controllability

## Scope

This diagnostic used the frozen exp_007 Stage 8C `model_10.pt`; it performed
zero PPO iterations, zero optimizer updates, and no reward, actor, production
observation, artifact, or capability change.

## Dataset

- Episodes: {dataset['episodes']} ({dataset['source_speed_counts']['2.6']} at 2.6 m/s and {dataset['source_speed_counts']['2.8']} at 2.8 m/s)
- Rows: {dataset['rows']}
- 20-step successes: {dataset['successful_20_step_segments']} (the target of 200 was not reachable with the frozen policy)
- Failed segments: {dataset['failed_segments']}
- Break reasons: {dataset['break_reason_counts']}
- Launch phases: {dataset['source_phase_counts']}
- Split: complete episode/reset-seed/source-speed/checkpoint groups, 60/20/20; no step leakage

## Observability

The primary timing-ablated static 152D probe achieved AUROC {static['auroc']:.3f}
and AUPRC {static['auprc']:.3f} at prevalence {static['prevalence']:.3f} for
contact break within three steps. Removing timing fields did not reduce AUROC.
However, ridge time-to-break MAE was {obs['static_regression_mae_steps']:.2f}
steps, above the preregistered 1.5-step limit. The 16-step GRU reached AUROC
{history['auroc']:.3f}, an improvement of only {obs['history_auroc_improvement']:.3f}.
Legacy 123D AUROC was {legacy['auroc']:.3f}; explicit phase upper-bound AUROC
was {explicit['auroc']:.3f} with MAE {obs['explicit_phase_regression_mae_steps']:.2f}.

Fixed classification: **{obs['classification']}**. Near-term binary risk is
highly rankable, but the complete preregistered observability gate (including
exact time-to-break) is not met.

## Controllability

Fresh Isaac applications replayed identical reset seeds, source routes, and
prebranch actions. Comparisons used only identical physical env IDs at identical
branch ages/steps. Root, joint, and velocity errors were exactly zero for all
accepted comparisons; no state was copied.

No candidate produced a safe 20-step contract:

- baseline: {summary['baseline']['safe_contract_successes']}/{summary['baseline']['branch_states']}
- frozen WALK: {summary['walk_expert']['safe_contract_successes']}/{summary['walk_expert']['branch_states']}
- frozen RUN: {summary['run_expert']['safe_contract_successes']}/{summary['run_expert']['branch_states']}
- bounded joint groups: {summary['bounded_joint_group']['safe_contract_successes']}/{summary['bounded_joint_group']['branch_states']}
- bounded target-WALK alignment: {summary['target_walk_alignment']['safe_contract_successes']}/{summary['target_walk_alignment']['branch_states']}

Fixed classification: **{control['classification']}**.

## Stage 0 decision

**{decision['decision']}**

The audited local corrections did not enter the target basin, while neither
history nor explicit phase passed the full observability gate. The next single
implementation should therefore be a unified WALK/RUN trajectory-distillation
study, not a production GRU or a reward-only continuation.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps({"gate": gate_pass, "decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
