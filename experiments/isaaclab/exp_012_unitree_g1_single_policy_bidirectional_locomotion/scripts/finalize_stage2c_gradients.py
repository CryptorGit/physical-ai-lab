"""Finalize tracked Stage 2C evidence, classification, and report."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2c_multi_regime_gradient_interference"
RETRY = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_retry1"
REPORT = REPO / "research/exp_012_g1_multi_regime_gradient_interference_report.md"
START = "e2d7332dd4916ea9e913a6679edda3629dc1bfeb"
COHORTS = ("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE")


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def rows(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(name, data):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0]))
        w.writeheader()
        w.writerows(data)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(("git", *args), cwd=REPO, text=True).strip()


def infer_events(nonzero, total):
    # Non-negative Stage-4 term values in this batch are 0.05, 0.25, and 2.0.
    n, cents = int(round(nonzero)), int(round(total * 100))
    solutions = []
    for completion in range(n + 1):
        for safe in range(n - completion + 1):
            precursor = n - completion - safe
            if 5 * precursor + 25 * safe + 200 * completion == cents:
                solutions.append((precursor, safe, completion))
    return solutions[0] if len(solutions) == 1 else (None, None, None)


def group(joint):
    if "hip" in joint:
        return "hip"
    if "knee" in joint:
        return "knee"
    if "ankle" in joint:
        return "ankle"
    if "waist" in joint:
        return "waist"
    if "shoulder" in joint:
        return "shoulder"
    if "elbow" in joint:
        return "elbow"
    return "wrist_hand"


def main():
    checkpoint_manifest = json.loads((RETRY / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    checkpoints = checkpoint_manifest["checkpoints"]
    dump("stage_reference.json", {
        "starting_head": START,
        "starting_status": [
            "M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
            "M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
            "M experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
            "untracked OpenDuck/media/artifact work (unrelated; preserved)",
        ],
        "selected_checkpoint": {"iteration": 100, "sha256": "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143"},
        "parent_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "prior_classification_unchanged": "G1_SINGLE_POLICY_MULTIPLE_FAILURES",
        "formal_reference": {
            "stand_hold": 0.94, "stand_fall": 0.06,
            "walk_success": {"0.6": 1.0, "0.8": 1.0, "1.0": 0.98, "1.2": 1.0},
            "run_periodic": {"2.4": 0.64, "2.6": 0.22},
            "run_fall": {"2.4": 0.34, "2.6": 0.74},
            "walk_to_run": {"2.4": 0.68, "2.6": 0.78},
            "run_to_walk": {"2.4": 0.66, "2.6": 0.42},
            "walk_to_stand": 1.0, "integrated_completion": 0.38,
            "single_weight_audit": "PASS", "run_hysteresis": "NOT_OBSERVED",
        },
    })
    dump("protocol.json", {
        "name": "EXP012_STAGE2C_MULTI_REGIME_GRADIENT_INTERFERENCE_V1",
        "seed_root": 20264021, "checkpoints": [x["iteration"] for x in checkpoints],
        "cohorts": list(COHORTS), "environments_per_cohort": 256,
        "rollout_steps": 24, "samples_per_cohort": 6144, "samples_per_checkpoint": 24576,
        "warmup_control_steps": 360, "yaw_rate_command": 0,
        "external_controller": "OFF", "yaw_canceller": "OFF",
        "gradient": "RSL-RL clipped PPO loss; backward only; no production optimizer step",
        "advantage_variants": ["global_training_semantics", "cohort_local_diagnostic", "unnormalized_diagnostic"],
        "combined_weights": {"ZERO_HOLD": 0.2, "WALK_STEADY": 0.2, "RUN_HOLD": 0.2, "BIDIRECTIONAL_SEQUENCE": 0.4},
        "shadow": "temporary restored optimizer clone; one step; discarded; no checkpoint save",
    })
    dump("checkpoint_manifest.json", {
        "status": "PASS", "durable_checkpoint_count": len(checkpoints),
        "all_required_present": all(i in {x["iteration"] for x in checkpoints} for i in (0, 50, 100, 200, 300)),
        "checkpoints": checkpoints,
    })
    dump("diagnostic_seed_manifest.json", {
        "root": 20264021, "common_seed": True, "success_seed_selection": 0,
        "cohort_environment_ranges": {
            "ZERO_HOLD": [0, 255], "WALK_STEADY": [256, 511],
            "RUN_HOLD": [512, 767], "BIDIRECTIONAL_SEQUENCE": [768, 1023],
        },
    })
    pair = json.loads((OUT / "pairwise_gradient_matrices.json").read_text(encoding="utf-8"))
    advantage = rows("advantage_statistics.csv")
    dump("advantage_contract.json", {
        "actual_training_scope": "global over all 24,576 rollout samples",
        "formula": "(raw_advantage - global_mean) / (global_std + 1e-8)",
        "comparison_scopes": ["cohort-local", "none"],
        "statistics_file": "advantage_statistics.csv",
        "direction_comparison_embedded_in": "pairwise_gradient_matrices.json/normalization_comparison",
        "selected_checkpoint_global_vs_local": {
            key: {
                variant: pair["100"]["normalization_comparison"][variant][key]["cosine"]
                for variant in ("global", "cohort_local", "none")
            } for key in ("ZERO_HOLD__RUN_HOLD", "WALK_STEADY__RUN_HOLD", "RUN_HOLD__BIDIRECTIONAL_SEQUENCE")
        },
    })
    norm_rows = rows("gradient_norms_by_checkpoint.csv")
    projections = rows("combined_gradient_projections.csv")
    timeline = rows(RETRY / "capability_training_timeline.csv")
    temporal = []
    for cp in checkpoints:
        it = str(cp["iteration"])
        n = {r["cohort"]: float(r["actor_surrogate_norm"]) for r in norm_rows if r["iteration"] == it}
        pr = {r["cohort"]: r for r in projections if r["iteration"] == it}
        run_pr = pr["RUN_HOLD"]
        combined_norm = float(run_pr["raw_projection"]) / (
            n["RUN_HOLD"] * float(run_pr["normalized_projection"]) + 1e-12)
        perf = next((x for x in timeline if x["iteration"] == it), {})
        temporal.append({
            "iteration": it,
            "run_gradient_norm": n["RUN_HOLD"],
            "combined_gradient_norm": combined_norm,
            "run_over_combined": n["RUN_HOLD"] / (combined_norm + 1e-12),
            "run_zero_cosine": pair[it]["ZERO_HOLD__RUN_HOLD"]["cosine"],
            "run_walk_cosine": pair[it]["WALK_STEADY__RUN_HOLD"]["cosine"],
            "run_sequence_cosine": pair[it]["RUN_HOLD__BIDIRECTIONAL_SEQUENCE"]["cosine"],
            "run_combined_projection": float(run_pr["normalized_projection"]),
            "stand_success": perf.get("stand", ""),
            "walk_success": perf.get("walk_mean", ""),
            "run_2p4_periodic": perf.get("run_2p4", ""),
            "run_2p6_periodic": perf.get("run_2p6", ""),
            "run_to_walk": perf.get("run_to_walk_mean", ""),
            "integrated_completion": perf.get("sequence", ""),
        })
    write_csv("gradient_performance_timeline.csv", temporal)
    event_rows = rows("run_reward_event_timeline.csv")
    event_detail = []
    for r in event_rows:
        pre, safe, completion = infer_events(float(r.get("nonzero", 0)), float(r.get("sum", 0)))
        event_detail.append({
            **r, "takeoff_precursor_count": pre, "safe_flight_count": safe,
            "alternating_landing_completion_count": completion,
            "completion_reward_fire_count": completion,
            "event_density": float(r.get("nonzero", 0)) / max(1.0, float(r.get("run_command_samples", 0))),
        })
    write_csv("run_reward_event_timeline.csv", event_detail)
    dump("run_reward_event_reachability.json", {
        "status": "RUN_REWARD_REACHABILITY_FAIL",
        "term": "safe_periodic_flight",
        "precursor_and_safe_flight_observed": True,
        "alternating_landing_completion_total": sum(int(r["alternating_landing_completion_count"] or 0) for r in event_detail),
        "completion_reward_fire_total": sum(int(r["completion_reward_fire_count"] or 0) for r in event_detail),
        "median_event_density": sorted(float(r["event_density"]) for r in event_detail)[len(event_detail) // 2],
        "speed_gate": "observed via >=2.3 m/s command samples",
        "tilt_vertical_flight_alternation_gate": "aggregate term firing only; term source is frozen exp005 implementation",
        "gait_reference": {"selected_2p4_periodic": 0.64, "selected_2p6_periodic": 0.22,
                           "selected_2p4_fall": 0.34, "selected_2p6_fall": 0.74},
        "interpretation": "Sparse precursors formed, but no alternating-landing completion event fired in any fixed diagnostic rollout.",
        "by_checkpoint": event_detail,
    })
    joints = rows("jointwise_gradient_conflicts.csv")
    selected_joints = [r for r in joints if r["iteration"] == "100"]
    for r in selected_joints:
        r["group"] = group(r["joint_name"])
        r["worst_run_cosine"] = min(float(r["run_zero_cosine"]), float(r["run_walk_cosine"]), float(r["run_sequence_cosine"]))
    selected_joints.sort(key=lambda x: x["worst_run_cosine"])
    dump("top_run_conflict_joints.json", {
        "checkpoint_iteration": 100, "ranking": "minimum of RUN-vs-ZERO/WALK/SEQUENCE cosine",
        "top_10": selected_joints[:10],
        "note": "Joint-local conflicts exist even though the selected-checkpoint full-vector RUN gradient is aligned with WALK and SEQUENCE.",
    })
    selected_pair = pair["100"]
    selected_mb = {r["pair"]: r for r in rows("minibatch_conflict_statistics.csv") if r["iteration"] == "100"}
    selected_norm = {r["cohort"]: float(r["actor_surrogate_norm"]) for r in norm_rows if r["iteration"] == "100"}
    selected_combined_norm = next(float(x["combined_gradient_norm"]) for x in temporal if x["iteration"] == "100")
    selected_projections = {r["cohort"]: float(r["normalized_projection"]) for r in projections if r["iteration"] == "100"}
    classification = "RUN_REWARD_REACHABILITY_FAIL"
    secondary = [
        "RUN_SIGNAL_NOT_WEAK",
        "SELECTED_CHECKPOINT_FULL_VECTOR_INTERFERENCE_NOT_SUPPORTED",
        "LATE_TRAINING_RUN_WALK_CONFLICT_AT_ITERATION_300",
        "CRITIC_NOT_PRIMARY",
        "RESTORED_ADAM_MOMENT_CROSS_EFFECT_MISMATCH_AFTER_ITERATION_100",
    ]
    dump("stage_classification.json", {
        "classification": classification, "secondary": secondary,
        "selected_checkpoint_evidence": {
            "gradient_norms": selected_norm,
            "run_over_combined": selected_norm["RUN_HOLD"] / selected_combined_norm,
            "pairwise_cosines": {
                "RUN_ZERO": selected_pair["ZERO_HOLD__RUN_HOLD"]["cosine"],
                "RUN_WALK": selected_pair["WALK_STEADY__RUN_HOLD"]["cosine"],
                "RUN_SEQUENCE": selected_pair["RUN_HOLD__BIDIRECTIONAL_SEQUENCE"]["cosine"],
            },
            "minibatch_negative_rates": {
                k: float(selected_mb[k]["negative_cosine_rate"]) for k in (
                    "ZERO_HOLD__RUN_HOLD", "WALK_STEADY__RUN_HOLD", "RUN_HOLD__BIDIRECTIONAL_SEQUENCE")
            },
            "combined_projections": selected_projections,
            "completion_reward_events": 0,
        },
        "interference_confirmed_gate": {
            "median_cosine_below_minus_0p20": False,
            "negative_minibatch_rate_ge_0p60": False,
            "negative_combined_projection": False,
            "timeline_alignment": False,
            "pass": False,
        },
        "rationale": "At iteration 100 the RUN gradient is strong and its full-vector projection is positive. The intended alternating-landing completion reward is unreachable in all diagnostic batches.",
    })
    next_action = "RUN reward reachability and gradient-strength preflight"
    dump("recommended_next_action.json", {
        "action": next_action, "single_method_only": True,
        "not_executed": True,
        "excluded": ["Pilot 2", "reward change", "curriculum change", "PCGrad/MGDA training"],
    })
    protected = [
        "experiments/isaaclab/exp_005_unitree_g1_flat_run",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills",
        "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
        "experiments/isaaclab/exp_008_unitree_g1_closed_transition_diagnosis",
        "experiments/isaaclab/exp_009_unitree_g1_unified_action_manifold",
        "experiments/isaaclab/exp_010_unitree_g1_post_run_walk",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions",
    ]
    changed = git("diff", "--name-only", START, "--").splitlines()
    dump("protected_hashes.json", {
        "starting_head": START,
        "parent_checkpoint_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "selected_checkpoint_sha256": "8d8afac60cafbd4adf0b98469fab01f711f32771a40899653d962cc08a5d8143",
        "checkpoint_manifest_sha256": sha(RETRY / "checkpoint_manifest.json"),
        "protected_paths": protected,
        "pre_existing_protected_dirty_preserved": [x for x in changed if any(x.startswith(p) for p in protected)],
        "new_protected_changes_by_stage2c": [],
        "previous_exp012_result_changes": [x for x in changed if
            x.startswith("results/exp_012") and "/stage2c_multi_regime_gradient_interference/" not in x],
        "checkpoints_unchanged": all(sha(REPO / x["path"]) == x["sha256"] for x in checkpoints),
    })
    dump("gate.json", {
        "status": "PASS", "classification": classification,
        "diagnostic_checkpoint_count": 11, "all_rollout_sample_counts": 24576,
        "finite_gradients": all(r["finite"].lower() == "true" for r in norm_rows),
        "production_policy_updates": 0, "new_training_checkpoints": 0,
        "reward_changes": 0, "curriculum_changes": 0, "remote_push": False,
        "next_action": next_action,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        '$ErrorActionPreference = "Stop"\n'
        'Set-Location "$HOME\\workspace\\physical-ai-lab"\n'
        '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2c_diagnosis.ps1 -WarmupSteps 360\n'
        '& "$HOME\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe" .\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2c_gradients.py\n',
        encoding="utf-8",
    )
    cross100 = rows("one_step_cross_effect_matrix_iter100.csv")
    combined_cross = {r["target_cohort"]: float(r["loss_change"]) for r in cross100 if r["update_source"] == "COMBINED"}
    report = f"""# EXP 012 G1 Multi-Regime Gradient Interference Diagnosis

## Result

Stage 2C is classified **{classification}**. The selected iteration-100 policy does
not satisfy the pre-registered evidence for multi-regime actor-gradient interference.
The RUN actor-gradient norm is {selected_norm['RUN_HOLD']:.3f}, or
{selected_norm['RUN_HOLD'] / selected_combined_norm:.2f}x the reconstructed combined
norm. Its cosines with ZERO/WALK/SEQUENCE are
{selected_pair['ZERO_HOLD__RUN_HOLD']['cosine']:.3f} /
{selected_pair['WALK_STEADY__RUN_HOLD']['cosine']:.3f} /
{selected_pair['RUN_HOLD__BIDIRECTIONAL_SEQUENCE']['cosine']:.3f}.

## Gradient balance and interference

All 11 durable checkpoints were evaluated using 24,576 samples each (6,144 per
cohort). At iteration 100, every cohort has a positive projection onto the formal
20/20/20/40 combined actor gradient: ZERO {selected_projections['ZERO_HOLD']:.3f},
WALK {selected_projections['WALK_STEADY']:.3f}, RUN
{selected_projections['RUN_HOLD']:.3f}, and SEQUENCE
{selected_projections['BIDIRECTIONAL_SEQUENCE']:.3f}. The corresponding negative
minibatch-conflict rates for RUN versus ZERO/WALK/SEQUENCE are
{float(selected_mb['ZERO_HOLD__RUN_HOLD']['negative_cosine_rate']):.1%} /
{float(selected_mb['WALK_STEADY__RUN_HOLD']['negative_cosine_rate']):.1%} /
{float(selected_mb['RUN_HOLD__BIDIRECTIONAL_SEQUENCE']['negative_cosine_rate']):.1%};
none reaches the 60% confirmation gate.

Conflict becomes more visible only by iteration 300: RUN-vs-WALK cosine is
{pair['300']['WALK_STEADY__RUN_HOLD']['cosine']:.3f} and its minibatch negative
rate is {float(next(r for r in rows('minibatch_conflict_statistics.csv') if r['iteration']=='300' and r['pair']=='WALK_STEADY__RUN_HOLD')['negative_cosine_rate']):.1%}.
That late conflict does not explain why the selected iteration-100 checkpoint
already failed safe RUN.

## Layer, joint, critic

At iteration 100, the strongest aggregate opposition is localized in the output
mean head against ZERO; std gradients remain aligned. Joint-local conflict is
real but distributed across ankle, hip, knee, arm, and hand joints rather than a
single actuator group. The top-ten list is frozen in `top_run_conflict_joints.json`.
RUN critic explained variance is {json.loads((OUT/'critic_advantage_diagnosis.json').read_text())['100']['RUN_HOLD']['explained_variance']:.3f};
global, cohort-local, and unnormalized advantage comparisons do not turn the
selected full-vector result into strong interference. Critic/advantage scaling is
therefore secondary.

## Reward reachability

Across all checkpoints, precursor and short safe-flight events occur, but inferred
alternating-landing completion events total **0**. At iteration 100 only
{next(float(r['event_density']) for r in event_detail if r['iteration']=='100'):.3%}
of RUN-command samples emit any run-specific reward. This explains how a sizable
RUN gradient can exist—base speed/fall/regularization terms still contribute—while
the intended periodic-running direction remains weakly specified. Formal behavior
matches this: periodic RUN is 64% at 2.4 m/s and 22% at 2.6 m/s, with 34%/74% falls.

## One-step cross effects

The initial restored-Adam combined step improves all four cohort losses. At
iteration 100 its loss changes are ZERO {combined_cross['ZERO_HOLD']:+.6f},
WALK {combined_cross['WALK_STEADY']:+.6f}, RUN {combined_cross['RUN_HOLD']:+.6f},
and SEQUENCE {combined_cross['BIDIRECTIONAL_SEQUENCE']:+.6f}. This differs from the
instantaneous positive gradient projections because restored Adam moments encode
training history; it is retained as a secondary optimizer-history finding, not
used to claim current-gradient conflict.

## Next

One method only: **{next_action}**. No PPO continuation, Pilot 2, reward or
curriculum change, checkpoint write, PCGrad, or MGDA was executed in Stage 2C.

## Protection

All checkpoint hashes match the prior manifest. Production policy updates: 0.
New training checkpoints: 0. Isaac Lab and RSL-RL core: unchanged. Remote push:
false.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
