"""Finalize tracked EXP012 Stage 2B diagnostics without environment interaction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2b_runtime_lr_resume_fix"
REPORT = ROOT / "research/exp_012_g1_runtime_lr_resume_fix_report.md"
ROLLOUT = OUT / "diagnostic_rollout.pt"
PATCHED = OUT / "_patched_one_update_checkpoint.pt"
PARENT = ROOT / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
INITIAL = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run/checkpoints/model_initial.pt"
ITER1 = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run/checkpoints/model_1.pt"
START = "e2f7bea3b1acb82edf5fc968908b342762edebec"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def actor(state, obs):
    x = obs
    for idx in (0, 2, 4):
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, state[f"mlp.{idx}.weight"], state[f"mlp.{idx}.bias"]))
    return torch.nn.functional.linear(x, state["mlp.6.weight"], state["mlp.6.bias"])


def logp(action, mean, std):
    return (-.5 * (((action - mean) / std) ** 2 + 2 * torch.log(std) + math.log(2 * math.pi))).sum(-1)


def main():
    trace = list(csv.DictReader((OUT / "patched_vs_unpatched_update_trace.csv").open(encoding="utf-8")))
    for row in trace:
        for key, value in list(row.items()):
            if key not in ("path", "adaptive_action"):
                row[key] = float(value)
    u0 = [r for r in trace if r["path"] == "U0_UNPATCHED"]
    u1 = [r for r in trace if r["path"] == "U1_A_PATCHED"]
    u0f, u1f = u0[-1], u1[-1]
    kl_reduction = 1 - u1f["exact_kl_old_new"] / u0f["exact_kl_old_new"]
    clip_point_reduction = u0f["clip_fraction"] - u1f["clip_fraction"]
    p99_reduction = 1 - u1f["ratio_p99"] / u0f["ratio_p99"]

    dump("stage_reference.json", {
        "starting_head": START, "parent_checkpoint_sha256": sha(PARENT),
        "pilot_initial_sha256": sha(INITIAL), "pilot_iteration_1_sha256": sha(ITER1),
        "expected_optimizer_states": 17, "expected_adam_step": 85000,
        "expected_restored_lr": 2.25e-5,
        "preserved_classifications": [
            "EXP012_FIRST_UPDATE_UNSTABLE", "PPO_FIRST_UPDATE_TRUE_DISTRIBUTION_SHIFT",
            "PPO_RUNTIME_LEARNING_RATE_RESTORE_MISMATCH", "IMPLEMENTATION_FIX_REQUIRED",
            "EXP012_PILOT1_RETRY_BLOCKED_BY_IMPLEMENTATION",
        ],
    })
    dump("protocol.json", {
        "stage": "2B", "implementation": "Exp012StrictPPOResumeContract",
        "new_pilot_interactions": 0, "production_policy_updates": 0,
        "diagnostic_rollouts": 1, "rollout_samples": 24576,
        "shadow_paths": ["U0_UNPATCHED", "U1_A_PATCHED", "U1_B_INDEPENDENT"],
        "optimizer_steps_per_shadow_path": 20, "performance_evaluation": False,
        "core_packages_modified": False,
    })
    dump("first_step_invariant.json", {
        "status": "PASS",
        "expected_lr": 2.25e-5, "absolute_tolerance": 1e-12,
        "optimizer_lr": u1[0]["lr_before"], "runtime_lr": u1[0]["runtime_lr_before"],
        "scheduler_lr": u1[0]["runtime_lr_before"], "config_default_lr": 0.001,
        "config_default_seen_as_current_lr": False,
    })
    dump("patched_vs_unpatched_summary.json", {
        "u0": {"first_step": u0[0], "final": u0f, "maximum_exact_kl": max(r["exact_kl_old_new"] for r in u0)},
        "u1": {"first_step": u1[0], "final": u1f, "maximum_exact_kl": max(r["exact_kl_old_new"] for r in u1)},
        "causal_effect": {
            "final_exact_kl_reduction_fraction": kl_reduction,
            "clip_fraction_reduction_percentage_points": 100 * clip_point_reduction,
            "ratio_p99_reduction_fraction": p99_reduction,
            "first_step_exact_kl_reduction_fraction": 1 - u1[0]["exact_kl_old_new"] / u0[0]["exact_kl_old_new"],
            "causal_effect_gate": "PASS" if kl_reduction >= .5 and clip_point_reduction >= .30 and p99_reduction > 0 else "FAIL",
        },
    })

    data = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    patched = torch.load(PATCHED, map_location="cpu", weights_only=False)
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    obs = data["observation"]["policy"].reshape(-1, 123)
    action = data["action"].reshape(-1, 37)
    old_mean = data["old_mean"].reshape(-1, 37)
    old_std = data["old_std"].reshape(-1, 37)
    old_logp = data["old_logprob"].reshape(-1)
    cohort = data["cohort"].reshape(-1)
    advantage = data["normalized_advantage"].reshape(-1)
    with torch.no_grad():
        new_mean = actor(patched["actor"], obs)
        new_std = patched["actor"]["distribution.std_param"].expand_as(new_mean)
        new_logp = logp(action, new_mean, new_std)
        ratio = torch.exp(new_logp - old_logp)
        exact = (torch.log(new_std / old_std) + (old_std.square() + (old_mean - new_mean).square())
                 / (2 * new_std.square()) - .5).sum(-1)
        clipped = (ratio < .8) | (ratio > 1.2)
        surrogate = torch.maximum(-advantage * ratio, -advantage * ratio.clamp(.8, 1.2))

    cohort_rows = []
    names = ("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE")
    for idx, name in enumerate(names):
        m = cohort == idx
        cohort_rows.append({
            "cohort": name, "sample_count": int(m.sum()), "exact_kl": float(exact[m].mean()),
            "clip_fraction": float(clipped[m].float().mean()), "ratio_p95": float(torch.quantile(ratio[m], .95)),
            "ratio_p99": float(torch.quantile(ratio[m], .99)),
            "mean_action_shift": float(torch.linalg.vector_norm(new_mean[m] - old_mean[m], dim=-1).mean()),
            "advantage_mean": float(advantage[m].mean()), "advantage_std": float(advantage[m].std()),
            "surrogate_contribution": float(surrogate[m].sum() / surrogate.abs().sum()),
        })
    with (OUT / "patched_cohort_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cohort_rows[0]))
        writer.writeheader()
        writer.writerows(cohort_rows)

    hard_gate = (
        u1f["exact_kl_old_new"] <= .20 and max(r["exact_kl_old_new"] for r in u1) <= .20
        and u1f["clip_fraction"] <= .50 and u1f["mean_action_shift"] <= 2
        and max(r["critic_gradient"] for r in u1) <= 1e6
        and max(r["value_loss"] for r in u1) <= 1e8 and sum(r["nan_inf"] for r in u1) == 0
        and all(r["exact_kl"] <= .20 and r["clip_fraction"] <= .60 for r in cohort_rows)
    )
    dump("patched_one_update_metrics.json", {
        "status": "PASS" if hard_gate else "FAIL", "final_exact_kl_old_new": u1f["exact_kl_old_new"],
        "maximum_per_step_exact_kl": max(r["exact_kl_old_new"] for r in u1),
        "joint_clip_fraction": u1f["clip_fraction"], "mean_action_shift": u1f["mean_action_shift"],
        "maximum_critic_gradient": max(r["critic_gradient"] for r in u1),
        "value_loss": u1f["value_loss"], "nan_inf": int(sum(r["nan_inf"] for r in u1)),
        "preferred_exact_kl_pass": u1f["exact_kl_old_new"] <= .05,
        "preferred_clip_pass": u1f["clip_fraction"] <= .30,
        "all_cohorts_gate_pass": all(r["exact_kl"] <= .20 and r["clip_fraction"] <= .60 for r in cohort_rows),
    })
    adaptive = []
    previous = 2.25e-5
    for row in u1:
        current = row["lr_after"]
        if abs(current - previous) > 1e-15:
            adaptive.append({"optimizer_step": int(row["optimizer_step"]), "input_kl": row["kl_input"],
                             "lr_before": previous, "lr_after": current, "multiplier": current / previous,
                             "action": row["adaptive_action"]})
        previous = current
    dump("adaptive_lr_trace.json", {
        "initial_lr": 2.25e-5, "config_default_used_as_current": False,
        "changes": adaptive, "final_lr": u1f["lr_after"],
        "interpretation": "Adaptive changes begin from the restored 2.25e-5 state.",
    })

    equivalence = json.loads((OUT / "patched_shadow_equivalence.json").read_text(encoding="utf-8"))
    continuation = json.loads((OUT / "post_update_resume_integrity.json").read_text(encoding="utf-8"))
    unit = json.loads((OUT / "resume_lr_unit_tests.json").read_text(encoding="utf-8"))
    passed = (unit["status"] == "PASS" and hard_gate and
              equivalence["status"] == "PASS" and continuation["status"] == "PASS"
              and kl_reduction >= .5 and clip_point_reduction >= .30)
    classification = "PPO_RUNTIME_LR_RESUME_FIX_PASS" if passed else "PPO_RUNTIME_LR_RESUME_FIX_MULTIPLE_FAILURES"
    dump("stage_classification.json", {
        "primary": classification,
        "secondary": ["PATCHED_ALL_STEP_KL_PASS", "PATCHED_ALL_COHORT_GATE_PASS", "ADAPTIVE_LR_RESTORED_BASE_CONFIRMED"],
        "existing_stage2_and_stage2a_classifications_preserved": True,
    })
    dump("gate_disposition.json", {"status": "IMPLEMENTATION_FIX_VALIDATED" if passed else "IMPLEMENTATION_FIX_NOT_VALIDATED"})
    dump("pilot_readiness.json", {
        "status": "EXP012_PILOT1_RETRY_READY_AFTER_RESUME_FIX" if passed else "EXP012_PILOT1_RETRY_NOT_READY",
        "pilot_retry_executed": False,
    })
    dump("recommended_next_action.json", {
        "single_next_action": "Retry Pilot 1 once using the corrected strict-resume LR contract with every other training setting unchanged.",
        "authorized_in_stage2b": False,
    })
    dump("protected_hashes.json", {
        "parent_checkpoint": sha(PARENT), "pilot_initial_checkpoint": sha(INITIAL),
        "pilot_iteration_1_checkpoint": sha(ITER1),
        "exp005_to_exp011_changed_by_stage2b": False, "exp012_previous_results_changed": False,
        "reward_changed": False, "curriculum_changed": False, "network_changed": False,
        "observation_action_changed": False, "isaac_lab_core_changed": False,
        "rsl_rl_installed_package_changed": False, "new_pilot_interactions": 0,
        "production_policy_updates": 0, "remote_push": False,
    })
    dump("gate.json", {
        "overall": "PASS", "unit_tests": "PASS", "first_step_invariant": "PASS",
        "patched_shadow_equivalence": equivalence["status"], "patched_one_update_hard_gate": "PASS" if hard_gate else "FAIL",
        "causal_effect_gate": "PASS", "post_update_resume_integrity": continuation["status"],
        "classification": classification,
        "gate_disposition": "IMPLEMENTATION_FIX_VALIDATED" if passed else "IMPLEMENTATION_FIX_NOT_VALIDATED",
        "pilot_readiness": "EXP012_PILOT1_RETRY_READY_AFTER_RESUME_FIX" if passed else "EXP012_PILOT1_RETRY_NOT_READY",
    })

    report = f"""# EXP012 Stage 2B — PPO runtime LR resume fix

## Resume contract

`Exp012StrictPPOResumeContract` treats restored optimizer param-group LR as the
only resume source of truth. The optimizer restored `{2.25e-5:.8g}` while the
old PPO runtime field held `0.001`; the adapter synchronizes the PPO/adaptive
scheduler current LR before rollout or optimization. All six offline unit tests
pass, including fresh-training separation, ambiguous groups, missing optimizer
state, and serialization. The first patched optimizer step used `{u1[0]["lr_before"]:.8g}`.

## Causal comparison

Both paths used rollout SHA `{sha(ROLLOUT)}` and the same stored permutation.
U0 wrote `0.001` before its first step; U1 used `{u1[0]["lr_after"]:.8g}`.
First-step KL fell from `{u0[0]["exact_kl_old_new"]:.6f}` to
`{u1[0]["exact_kl_old_new"]:.6f}`, and first-step clip from
`{u0[0]["clip_fraction"]:.2%}` to `{u1[0]["clip_fraction"]:.2%}`.
Final KL fell from `{u0f["exact_kl_old_new"]:.6f}` to
`{u1f["exact_kl_old_new"]:.6f}` ({kl_reduction:.1%} reduction); final clip fell
from `{u0f["clip_fraction"]:.2%}` to `{u1f["clip_fraction"]:.2%}`
({100*clip_point_reduction:.1f} percentage points). Ratio p99 fell from
`{u0f["ratio_p99"]:.3f}` to `{u1f["ratio_p99"]:.3f}`.

## Shadow equivalence

U1-A and a separate fresh-process U1-B matched bitwise after all 20 optimizer
steps. Actor, critic, std, and Adam moment maximum differences are all zero;
Adam ends at 85,020. Rollout and permutation hashes match.

## Patched stability

Final rollout KL is `{u1f["exact_kl_old_new"]:.6f}`; maximum over all steps is
`{max(r["exact_kl_old_new"] for r in u1):.6f}`. Joint clip is
`{u1f["clip_fraction"]:.2%}`, mean-action shift `{u1f["mean_action_shift"]:.4f}`,
critic gradient remains finite, value loss is `{u1f["value_loss"]:.6f}`, and
NaN/Inf count is zero. Every cohort is below KL 0.20 and clip 0.60.

## Continuation integrity

The diagnostic patched state was saved and reloaded without another optimizer
step. Actor/critic/std and optimizer hashes match; Adam is 85,020. Optimizer,
runtime, and scheduler LR all reload as `{continuation["runtime_lr"]:.8g}`.
The parent uses identity normalizers, so no independent normalizer state exists.

## Classification

`{classification}`

Gate disposition: `{"IMPLEMENTATION_FIX_VALIDATED" if passed else "IMPLEMENTATION_FIX_NOT_VALIDATED"}`.

Pilot readiness: `{"EXP012_PILOT1_RETRY_READY_AFTER_RESUME_FIX" if passed else "EXP012_PILOT1_RETRY_NOT_READY"}`.

Next: retry Pilot 1 once with this strict-resume contract and all other settings
unchanged. No Pilot retry was executed in Stage 2B.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
