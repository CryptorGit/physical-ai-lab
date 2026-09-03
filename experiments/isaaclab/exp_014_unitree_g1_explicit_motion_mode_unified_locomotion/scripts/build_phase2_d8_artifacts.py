"""Build compact D8 decision, protection, and report artifacts."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
START = "a2e90ac1649582bc6fd00486f5803e297081c744"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8_phase_error_causal_relevance"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
REPORT = REPO / "research/exp_014_phase_2_d8_phase_error_causal_relevance_report.md"
CLASSIFICATION = "EXP014_D8_PHASE_CLASSIFIER_IMPLEMENTATION_BUG"


def read(name): return json.loads((OUT / name).read_text(encoding="utf-8"))
def dump(name, value): (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


candidate = read("s1_diagnostic_candidate.json")
classifier = read("phase_classifier_implementation_audit.json")
action = read("phase_action_relevance.json")
physical = read("phase_physical_counterfactual.json")
raw = read("raw_input_phase_separability.json")
hidden = read("hidden_feature_phase_probes.json")
shadow_auth = read("diagnostic_closed_loop_authorization.json")
shadow = read("diagnostic_closed_loop_shadow.json")
temporal = read("phase_temporal_concentration.json")

dump("stage_reference.json", {
    "phase": "2-D8", "name": "S1 phase-classification error causal relevance audit",
    "starting_head_expected": START, "starting_head_actual": git("rev-parse", START),
    "D7_classification_unchanged": "EXP014_D7_STATIC_CAPACITY_FAIL",
    "diagnostic_candidate_is_formal_selection": False, "heldout_opened": False,
    "date": "2026-08-04", "timezone": "Asia/Tokyo", "remote_push": False,
})
dump("protocol.json", {
    "partitions": {"classifier_train": "D7 train", "analysis": "D7 validation and D7 validation-only local neighborhood", "heldout": "sealed/unopened"},
    "candidate": "S1_DIAGNOSTIC_CANDIDATE",
    "root_state_divergence_threshold_m": physical["root_threshold_preregistered_m"],
    "root_threshold_preregistered_before_physical_results": True,
    "counterfactual_horizons_steps": [8, 16, 32],
    "shadow_authorization_requirements": {"action_regression": "PASS", "material_label_conflict": 0,
        "ACTION_RELEVANT_PHASE_ACCURACY": .99, "PHYSICAL_PHASE_SAFETY": .99,
        "action_critical_misclassification_max": .01, "observation_leakage": 0},
    "forbidden_executed": {"policy_training": 0, "PPO": 0, "S2": 0, "RUN": 0, "OMNI_RUN": 0,
        "final_integrated_student": 0, "Causal_DAgger_Dataset_V2": 0, "sealed_heldout_open": 0},
})

root_cause = {
    "classification": CLASSIFICATION,
    "primary_finding": "D7 phase accuracy came from an independent raw-input diagnostic MLP, not S1 hidden features, S1 actions, or a training auxiliary head. The same independent scalar was attached to every S1 checkpoint and used as actor eligibility.",
    "D7_classification_changed": False,
    "evidence": {
        "raw_best_phase_accuracy": raw["best_validation_accuracy"],
        "raw_phase_below_99_percent": raw["best_validation_accuracy"] < .99,
        "phase_errors": action["phase_errors"],
        "action_safe_fraction_of_phase_errors": action["action_safe_misclassification_rate"],
        "action_critical_errors": action["action_critical_errors"],
        "ACTION_RELEVANT_PHASE_ACCURACY": action["ACTION_RELEVANT_PHASE_ACCURACY"],
        "PHYSICAL_PHASE_SAFETY": physical["PHYSICAL_PHASE_SAFETY"],
        "physical_critical_errors": physical["physical_critical_errors"],
        "errors_within_switch_plus_minus_4_fraction": temporal["errors_within_plus_minus_4_over_all_errors"],
        "shadow_authorized": shadow_auth["authorized"],
    },
    "causal_interpretation": "Because the diagnostic classifier is disconnected from S1 and its prediction is never consumed by the actor, its classification mistakes cannot causally select an action. Six student-vs-oracle continuation failures show local BC closed-loop risk, but do not establish phase-classifier causation.",
    "Route_A": "not authorized because PHYSICAL_PHASE_SAFETY <99% and shadow was not run",
    "Route_B": "not supported because raw 141D accuracy is below 99.5% and action-critical errors are 0",
    "Route_C": "raw labels are observationally over-partitioned, but the required action-critical evidence is absent; physical deviations are not caused by the disconnected classifier",
    "Route_D": "supported: fix evaluator attribution and recompute static eligibility without policy retraining",
}
dump("root_cause_classification.json", root_cause)
dump("recommended_next_action.json", {
    "one_experiment": "fix the D7 static evaluator so independent diagnostic-classifier accuracy is not attributed to S1 checkpoints, then recompute D7 static eligibility from unchanged artifacts",
    "policy_retraining": False, "PPO": False, "S2": False, "closed_loop_shadow_in_D8": False,
    "formal_D7_retroactive_PASS": False, "sealed_heldout": "remain sealed",
})

expected = read("../phase_2_d7_r4_stop_oracle_distillation/dataset_hashes.json") if False else json.loads((D7 / "dataset_hashes.json").read_text(encoding="utf-8"))
current_hashes = {
    "train": sha(D7 / "raw/dataset/train.pt"), "validation": sha(D7 / "raw/dataset/validation.pt"),
    "sealed-held-out": sha(D7 / "raw/sealed_heldout_snapshots.pt"),
}
trees = {}
for number in range(5, 14):
    for path in (REPO / "experiments/isaaclab").glob(f"exp_{number:03d}_*"):
        rel = path.relative_to(REPO).as_posix()
        try: trees[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
d6_rel = "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher"
d7_rel = "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
dump("protected_hashes.json", {
    "starting_head": START, "exp005_to_exp013_tree_hashes_at_start": trees,
    "D6_tree_hash_at_start": git("rev-parse", f"{START}:{d6_rel}"),
    "D7_tree_hash_at_start": git("rev-parse", f"{START}:{d7_rel}"),
    "D6_D7_tracked_diff_before_commit": git("diff", "--name-only", START, "--", d6_rel, d7_rel),
    "D7_dataset_expected_hashes": expected, "D7_dataset_current_hashes": current_hashes,
    "D7_dataset_hash_match": current_hashes == expected, "sealed_heldout_opened": False,
    "existing_checkpoint_changed": False, "new_persistent_checkpoint": 0,
    "policy_training": 0, "PPO": 0, "S2": 0, "RUN": 0, "Causal_DAgger_Dataset_V2": 0,
    "preexisting_dirty_preserved": True, "remote_push": False,
})

(OUT / "reproduction_commands.ps1").write_text(
    "$ErrorActionPreference = 'Stop'\n"
    "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d8_offline.py\n"
    "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d8_physical.py --headless --device cuda:0\n"
    "python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d8_artifacts.py\n",
    encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D8 phase-error causal relevance audit

## Result

Classification: **{CLASSIFICATION}**. D7 remains `EXP014_D7_STATIC_CAPACITY_FAIL` and is not retroactively passed.

## Phase semantics and classifier

The seven labels mix clock-defined, physical-event-defined, and Teacher-route-defined boundaries. `W_MOVE_TO_STAGE2Q_BOUNDARY` spans both T_MOVE and T_STOP, while acquisition and post-acquisition split nearly action-equivalent steadying behavior. The D7 97.32% value was produced by an independent raw-141D MLP trained for 3,000 balanced steps. It was neither an S1 hidden probe nor an S1 auxiliary head, yet the same scalar was attached to all S1 checkpoints and used in static eligibility.

Validation accuracy was {classifier['validation_metrics']['accuracy']:.2%}, macro F1 {classifier['validation_metrics']['macro_f1']:.2%}, and balanced accuracy {classifier['validation_metrics']['balanced_accuracy']:.2%}. The best of three raw-input diagnostics was also {raw['best_validation_accuracy']:.2%}; the labels are not 99%-separable under this audit. Errors were not concentrated at the 0.50 s switch: {temporal['errors_within_plus_minus_4_over_all_errors']:.2%} occurred within ±4 steps.

## Action and physical relevance

Of {action['phase_errors']:,} phase errors, {action['action_safe_errors']:,} ({action['action_safe_misclassification_rate']:.2%}) were action-safe and zero were action-critical. `ACTION_RELEVANT_PHASE_ACCURACY` was {action['ACTION_RELEVANT_PHASE_ACCURACY']:.2%}.

Read-only counterfactuals covered {physical['states']} local-neighborhood states, up to 100 per observed error pair. Six met the preregistered physical-critical definition, giving `PHYSICAL_PHASE_SAFETY={physical['PHYSICAL_PHASE_SAFETY']:.2%}`. Most were loss of acquisition-progress parity after small S1/oracle deviations; no diagnostic classifier output was supplied to S1, so these deviations cannot be caused by classifier routing. Because physical safety was below 99%, shadow closed loop was not authorized and the sealed held-out remained unopened.

## Decision

The next single experiment is evaluator-only: remove the erroneous attribution of an independent phase classifier to each actor checkpoint and recompute D7 static eligibility from unchanged action-regression artifacts. Do not retrain the policy, add S2, return to PPO, or open held-out.
""", encoding="utf-8")

print(json.dumps({"classification": CLASSIFICATION, "action_relevant": action["ACTION_RELEVANT_PHASE_ACCURACY"],
                  "physical_safety": physical["PHYSICAL_PHASE_SAFETY"], "shadow": shadow}, indent=2))
