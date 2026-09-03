"""Generate the tracked Stage 2A contracts, disposition, and report."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2a_first_update_ratio_clipping_diagnosis"
REPORT = ROOT / "research/exp_012_g1_first_update_ratio_clipping_diagnosis_report.md"
START = "61941d1cabbc626834cf8df144bb00b3154198bf"
PARENT = ROOT / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
INITIAL = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run/checkpoints/model_initial.pt"
ITER1 = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run/checkpoints/model_1.pt"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start_status = [
        " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        " M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
        "?? .openduck_hardware_source_review/", "?? .openduck_phase3_usb_baseline.txt",
        "?? .openduck_runtime_source_review/", "?? artifacts/exp_005_unitree_g1_flat_run/",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
        "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
        "?? media/", "?? openduck_setup_report.md", "?? research/exp_011_linkedin_post_ja.md",
    ]
    dump("stage_reference.json", {
        "starting_head_reported": START, "starting_head_actual": START,
        "starting_status": start_status, "unrelated_dirty_paths_preserved": True,
        "parent_checkpoint": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "parent_sha256": sha(PARENT), "pilot_initial_sha256": sha(INITIAL),
        "pilot_iteration_1_sha256": sha(ITER1),
        "existing_classification_preserved": "EXP012_FIRST_UPDATE_UNSTABLE",
    })
    dump("protocol.json", {
        "stage": "2A", "name": "first_update_ratio_clipping_diagnosis",
        "new_pilot": False, "additional_iterations": 0, "performance_evaluation": False,
        "production_policy_updates": 0, "new_training_interactions": 0,
        "diagnostic_rollout_recollected": True,
        "reason": "The original immutable 24,576-sample rollout and exact permutation were not retained.",
        "shadow_updates": "diagnostic clone only; no checkpoint saved",
    })
    dump("source_code_audit.json", {
        "implementation": "rsl_rl PPO plus exp_012 train_stage2 instrumentation",
        "audit_result": "runtime learning-rate state was not synchronized by optimizer restore",
        "critical_finding": {
            "optimizer_param_group_after_restore": 2.25e-5,
            "ppo_python_learning_rate_attribute": 0.001,
            "first_shadow_optimizer_step_learning_rate": 0.001,
            "final_learning_rate": 1e-5,
            "mechanism": "adaptive KL branch writes self.learning_rate to optimizer.param_groups before each optimizer step",
        },
    })
    locations = {
        "distribution_log_prob_sum": {"file": "rsl_rl/modules/distribution.py", "line": 217},
        "rollout_logprob_capture": {"file": "rsl_rl/algorithms/ppo.py", "lines": "139-146"},
        "global_advantage_normalization": {"file": "rsl_rl/algorithms/ppo.py", "lines": "208-210"},
        "minibatch_generator": {"file": "rsl_rl/storage/rollout_storage.py", "lines": "222-253"},
        "single_randperm_reused_across_epochs": {"file": "rsl_rl/storage/rollout_storage.py", "line": 228},
        "adaptive_exact_kl": {"file": "rsl_rl/algorithms/ppo.py", "lines": "262-294"},
        "joint_ratio_and_surrogate": {"file": "rsl_rl/algorithms/ppo.py", "lines": "297-302"},
        "storage_clear": {"file": "rsl_rl/algorithms/ppo.py", "line": 403},
        "exp012_initial_observation_metric": {"file": "experiments/.../scripts/train_stage2.py", "lines": "165,173-174,205-217"},
        "exp012_rollout_metric": {"file": "experiments/.../scripts/train_stage2.py", "lines": "181-201"},
    }
    dump("ppo_metric_source_locations.json", locations)
    dump("ppo_metric_semantics.json", {
        "log_probability": "Diagonal Gaussian per-action log probabilities summed over 37 dimensions.",
        "probability_ratio": "exp(new joint log probability - stored old joint log probability).",
        "clip_fraction": "Fraction of rollout samples whose joint ratio is outside [0.8,1.2]. It is not a per-dimension fraction.",
        "surrogate_loss": "mean(max(-A*r, -A*clip(r,0.8,1.2))) per minibatch.",
        "reported_rollout_kl": "Post-update exact diagonal-Gaussian KL(old||new), summed over 37 dimensions and averaged over the entire rollout; implementation iterates four full-rollout minibatches only to reduce memory.",
        "analytical_kl_0p03938": "Post-update exact KL(old||new) on the reset/initial observation batch captured before rollout.",
        "aggregation_difference": "0.03938 and 0.20244 are final-policy metrics on different observation distributions, not competing formulas.",
        "optimizer_metric_during_update": "Exact old||new Gaussian KL on the current minibatch before each optimizer step, used by the adaptive schedule.",
        "old_logprob_mutation": False, "observation_normalization": "Actor input follows runner contract; stored observation is replayed through the same actor path.",
        "std_parameterization": "state-independent scalar-space standard deviation, not log or softplus",
        "action_provenance": "log probability is evaluated on the sampled, unclipped normalized action; action scale is downstream.",
        "metric_accumulator_reset": "exp012 local kls/clips lists are newly allocated after runner.learn and aggregated once.",
        "source_locations": locations,
    })

    old = load("old_logprob_reconstruction.json")
    old.update({
        "strict_gate": "FAIL",
        "strict_tolerance": 1e-6,
        "classification": "PPO_OLD_LOGPROB_PROVENANCE_MISMATCH",
        "interpretation": "Saved old mean/std reproduce storage exactly. Re-forwarding the actor differs by 2.29e-5 in joint log probability, consistent with accumulated floating replay error but above the preregistered 1e-6 gate.",
    })
    dump("old_logprob_reconstruction.json", old)

    dump("gate_disposition.json", {
        "disposition": "IMPLEMENTATION_FIX_REQUIRED",
        "metric_amendment_justified": False,
        "true_update_stabilization_required": False,
        "reasons": [
            "PPO runtime learning_rate attribute (0.001) overwrote strictly restored optimizer LR (2.25e-5).",
            "Official-order shadow replay did not reproduce the official iteration-1 checkpoint.",
            "Original reported metric was not numerically reconstructed to 1e-5 because the original rollout/permutation artifact was absent.",
            "Policy-API old-logprob reconstruction exceeded the preregistered 1e-6 tolerance.",
        ],
        "future_source_of_truth_after_fix": [
            "final exact old||new KL over the immutable rollout",
            "maximum exact old||new KL over optimizer steps",
            "joint clip fraction",
            "ratio p95/p99/p99.9 and lower/upper tails",
        ],
    })
    dump("stage_classification.json", {
        "primary": "PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT",
        "secondary": [
            "PPO_REPORTED_KL_SEMANTIC_MISMATCH",
            "PPO_HIGH_DIMENSIONAL_RATIO_ACCUMULATION",
            "PPO_ACTOR_MEAN_UPDATE_DOMINATED",
            "PPO_IMMEDIATE_RATIO_EXPLOSION",
            "PPO_SHADOW_REPLAY_MISMATCH",
            "PPO_OLD_LOGPROB_PROVENANCE_MISMATCH",
            "PPO_REPORTED_METRIC_NOT_REPRODUCIBLE",
            "PPO_RUNTIME_LEARNING_RATE_RESTORE_MISMATCH",
        ],
        "existing_stage2_classification_preserved": "EXP012_FIRST_UPDATE_UNSTABLE",
        "rationale": "The rollout-state exact KL (~0.202) and 72.4% joint clipping describe a broad, mean-dominated change. The smaller 0.03938 measures only initial observations. Dimensional accumulation amplifies clipping but does not erase the real rollout distribution shift.",
    })
    dump("pilot_readiness.json", {
        "status": "EXP012_PILOT1_RETRY_BLOCKED_BY_IMPLEMENTATION",
        "pilot_executed": False,
        "blocker": "Restore/synchronize PPO runtime learning_rate state, then require exact immutable-rollout shadow equivalence before considering a retry.",
    })
    dump("recommended_next_action.json", {
        "single_next_action": "Fix the PPO resume contract so self.learning_rate is restored from the optimizer param group, then rerun the one-update shadow-equivalence diagnostic only.",
        "pilot_retry_now": False,
    })
    dump("protected_hashes.json", {
        "parent_checkpoint": sha(PARENT), "pilot_initial_checkpoint": sha(INITIAL),
        "pilot_iteration_1_checkpoint": sha(ITER1),
        "expected": {
            "parent_checkpoint": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
            "pilot_initial_checkpoint": "371876f89ebc5a1d3ebac5f57be361745a038ad7b4ca243fe730b852e8e7431b",
            "pilot_iteration_1_checkpoint": "1221c1ea154206941d99f7c009bf4f4cdfa14057706ce036e0e01f50174b8879",
        },
        "exp005_to_exp011_changed_by_stage2a": False,
        "exp012_previous_stages_changed_by_stage2a": False,
        "reward_changed": False, "curriculum_changed": False, "network_changed": False,
        "isaac_lab_core_changed": False, "production_policy_updates": 0,
        "new_training_interactions": 0, "remote_push": False,
    })
    dump("gate.json", {
        "overall": "FAIL_CLOSED",
        "existing_first_update_gate": "EXP012_FIRST_UPDATE_UNSTABLE",
        "old_logprob_strict_gate": "FAIL",
        "shadow_equivalence": "FAIL",
        "reported_metric_reconstruction_1e-5": "FAIL",
        "implementation_integrity": "FAIL",
        "gate_disposition": "IMPLEMENTATION_FIX_REQUIRED",
        "pilot_readiness": "EXP012_PILOT1_RETRY_BLOCKED_BY_IMPLEMENTATION",
    })

    report = """# EXP012 Stage 2A — first-update PPO ratio/clipping diagnosis

## Metric semantics

The two KL values do not measure the same state distribution. `0.03938` is exact
diagonal-Gaussian KL(old||new), summed over 37 actions, on the reset observation
batch captured before rollout. `0.20244` is the same exact KL definition on all
24,576 rollout observations after the 20 optimizer steps. The reported clip
fraction `0.72396` is the fraction of samples whose **joint** probability ratio
`exp(sum_37(new_logp-old_logp))` lies outside `[0.8, 1.2]`.

The smaller reset-state KL therefore cannot invalidate the rollout-state KL.
The high-dimensional sum amplifies joint clipping: many individually modest
joint changes accumulate in log-probability space.

## Rollout integrity

The original immutable rollout and exact minibatch permutation were not saved.
One permitted diagnostic recollection produced 24,576 samples (1,024 envs × 24
steps), with zero nonzero yaw commands and cohort counts 4,944 ZERO, 4,944 WALK,
4,896 RUN, and 9,792 SEQUENCE. Canonical field hashes are in
`immutable_rollout_hashes.json`; the raw tensors remain local and untracked.

Stored actions, means, and standard deviations independently reconstruct the old
joint log probability exactly. A fresh policy forward differs by at most
`2.2888e-5`, above the preregistered `1e-6` tolerance, so the strict provenance
gate is fail-closed even though no action-clipping/storage confusion was found.

## Shadow replay

The diagnostic clone executed the prescribed 20 optimizer steps (Adam
85,000→85,020), but did not match the official iteration-1 actor, critic, or std.
Consequently its per-step trace is diagnostic-only. It nevertheless exposed the
decisive resume defect: optimizer restore set LR to `2.25e-5`, while
`PPO.learning_rate` remained `0.001`; the adaptive-KL block wrote `0.001` back
before the first step. The first shadow step already reached exact KL `0.408`
and clip fraction `0.785`, so this was not solely a late-epoch accumulation.

Applying the official iteration-1 checkpoint to the recollected rollout gives
exact KL `0.20289` and clip `0.72306`, close to but not within `1e-5` of the
recorded `0.20244/0.72396`. The initial-observation KL reproduces as `0.0393815`.
The rollout KL is 99.75% actor-mean contribution; std contributes about 0.00050.

## Ratio diagnosis

For the official checkpoint on the recollected rollout, ratio median is `0.841`,
p95 `2.214`, p99 `3.602`, with lower/upper clip fractions `46.4%/25.9%`. All
cohorts show broad clipping (about 71–75%); WALK has the largest exact KL
(`0.238`), but no isolated cohort explains the result. Alternative minibatch
orders were not run after official-order shadow mismatch because that would
confound rollout/RNG mismatch with order sensitivity.

## Classification

Primary: `PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT`.

Secondary: reported-KL state-scope mismatch, 37D ratio accumulation,
actor-mean-dominated update, immediate ratio explosion, shadow mismatch, strict
old-logprob tolerance failure, and runtime learning-rate restore mismatch.
The prior `EXP012_FIRST_UPDATE_UNSTABLE` result is retained unchanged.

## Gate disposition and readiness

`IMPLEMENTATION_FIX_REQUIRED`

`EXP012_PILOT1_RETRY_BLOCKED_BY_IMPLEMENTATION`

Single next action: fix the resume contract so the PPO runtime learning-rate
attribute is synchronized from the restored optimizer, then rerun only the
immutable one-update shadow-equivalence diagnostic. No Pilot retry is authorized.

## Repository

Starting HEAD: `61941d1cabbc626834cf8df144bb00b3154198bf`.
Only Stage 2A implementation, tracked aggregates, contracts, and this report are
included. Pre-existing unrelated dirty paths were preserved. No remote push was
performed.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
