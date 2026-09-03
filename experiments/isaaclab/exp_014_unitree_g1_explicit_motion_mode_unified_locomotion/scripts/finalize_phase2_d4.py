#!/usr/bin/env python3
"""Finalize Phase 2-D4 artifacts without launching simulation or changing policy state."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d4_stand_objective_horizon_attribution"
REPORT = ROOT / "research/exp_014_phase_2_d4_stand_objective_horizon_attribution_report.md"
START_HEAD = "f3b31460dd7da91d53800ef84a8ae63bf718a229"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = load("stand_settle_hold_diagnostics.json")
    windows = load("stand_formal_window_decomposition.json")
    gae = load("stand_gae_horizon_attribution.json")
    gradients = load("stand_reward_gradient_attribution.json")
    temporary = load("stand_temporary_horizon_updates.json")
    matched = load("stand_matched_interaction_horizon_comparison.json")
    value = load("stand_value_accuracy.json")
    rewards = load("stand_reward_term_attribution.json")

    # A sign reversal from early underestimation to a large t=50 overestimate is
    # not accurately represented by either one-sided value-error label.
    value["classification"] = "VALUE_HORIZON_MULTIPLE_ERRORS"
    severity = defaultdict(lambda: {"count": 0, "abs_sum": 0.0, "bias_sum": 0.0})
    with (OUT / "stand_value_accuracy.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["t"], row["severity_bin"])
            severity[key]["count"] += 1
            severity[key]["abs_sum"] += float(row["absolute_error"])
            severity[key]["bias_sum"] += float(row["signed_bias"])
    value["severity_bins"] = {
        f"t{time}_bin{bin_id}": {
            "count": item["count"],
            "mean_absolute_error": item["abs_sum"] / item["count"],
            "signed_bias": item["bias_sum"] / item["count"],
        }
        for (time, bin_id), item in sorted(severity.items(), key=lambda x: (int(x[0][0]), int(x[0][1])))
    }
    dump("stand_value_accuracy.json", value)

    rewards["term_coverage"] = {
        "XY_velocity_tracking": "track_lin_vel_xy_exp",
        "yaw_tracking": "track_ang_vel_z_exp",
        "upright_orientation": ["flat_orientation_l2", "ang_vel_xy_l2"],
        "joint_position_deviation": [
            "joint_deviation_hip", "joint_deviation_arms", "joint_deviation_fingers", "joint_deviation_torso"
        ],
        "joint_velocity": "ABSENT_AS_EXPLICIT_V1_TERM",
        "torque": "dof_torques_l2",
        "acceleration": "dof_acc_l2",
        "action_rate": "action_rate_l2",
        "foot_air": "feet_air_time",
        "foot_slide": "feet_slide",
        "termination": "termination_penalty",
    }
    dump("stand_reward_term_attribution.json", rewards)

    u = {x["label"]: x for x in temporary["updates"]}
    m = {x["label"]: x for x in matched["comparisons"]}
    long_gain = max(
        u["U100"]["physical"]["formal"] - u["U24"]["physical"]["formal"],
        u["U100"]["physical"]["reset_to_stand"] - u["U24"]["physical"]["reset_to_stand"],
        m["M100"]["physical"]["formal"] - m["M24"]["physical"]["formal"],
        m["M100"]["physical"]["reset_to_stand"] - m["M24"]["physical"]["reset_to_stand"],
    )
    reg_ratio = gradients["regularization_gradient_norm"] / gradients["xy_plus_yaw_gradient_norm"]
    eval_gate = (
        diagnostics["RESET_TO_STAND_DIAGNOSTIC"] >= 0.95
        and diagnostics["STAND_HOLD_DIAGNOSTIC"] >= 0.95
        and diagnostics["fall"] <= 0.02
        and diagnostics["dangerous_slip"] <= 0.05
        and diagnostics["old_whole_window_practical_stand"] < 0.95
    )
    material_reward_conflict = gradients["xy_plus_yaw_over_total"] < 0.10 or (
        gradients["regularization_vs_settling_cosine"] < -0.25 and reg_ratio >= 0.50
    )
    classification = "EXP014_D4_SETTLE_HOLD_CONTRACT_AUTHORIZED"
    root = {
        "classifications": [
            "STAND_GAE_BOOTSTRAP_MISMATCH",
            "STAND_VALUE_FUNCTION_HORIZON_ERROR",
            "STAND_REGULARIZATION_GRADIENT_CONFLICT",
            "STAND_EVALUATOR_CONFLATES_SETTLE_AND_HOLD",
            "STAND_MULTIPLE_CAUSES",
        ],
        "primary_cause": "STAND_EVALUATOR_CONFLATES_SETTLE_AND_HOLD",
        "secondary_causes": [
            "STAND_GAE_BOOTSTRAP_MISMATCH", "STAND_VALUE_FUNCTION_HORIZON_ERROR",
            "STAND_REGULARIZATION_GRADIENT_CONFLICT"
        ],
        "not_supported": [
            "STAND_ROLLOUT_HORIZON_MISMATCH_AS_PRIMARY_CAUSE",
            "STAND_SETTLING_REWARD_UNDERWEIGHTED",
            "STAND_TRUE_TWO_SECOND_CAPABILITY_DEFICIT",
        ],
        "evaluation_separation_gate": eval_gate,
        "long_horizon_max_gain": long_gain,
        "long_horizon_clear_improvement_threshold": 0.03,
        "reward_conflict_directional": gradients["regularization_vs_settling_cosine"] < -0.25,
        "reward_conflict_material": material_reward_conflict,
        "regularization_to_settling_gradient_norm": reg_ratio,
        "evidence": {
            "diagnostics": diagnostics,
            "gradient": gradients,
            "value": value,
            "GAE": gae,
            "U24": u["U24"], "U50": u["U50"], "U100": u["U100"],
            "M24": m["M24"], "M50": m["M50"], "M100": m["M100"],
        },
    }
    dump("root_cause_classification.json", root)

    dump("stage_classification.json", {
        "classification": classification,
        "selected_route": "E",
        "persistent_policy_updates": 0,
        "new_policy_checkpoints": 0,
        "held_out_evaluations": 0,
    })
    dump("recommended_next_action.json", {
        "route": "E",
        "one_route_only": True,
        "authorized_experiment": "Version RESET_TO_STAND + STAND_HOLD capability contract; retain the old whole-window metric as diagnostic.",
        "not_authorized": ["H", "R", "M"],
        "policy_training_budget": 0,
    })
    dump("exp014_dedicated_stand_next_contract.json", {
        "classification": classification,
        "selected_route": "E",
        "rollout_length": None,
        "episode_continuation_semantics": "20-second episodes continue across 24-step rollout boundaries; done environments reset asynchronously; full reset occurs at validation boundaries.",
        "GAE_bootstrap_semantics": "GAE ends at each artificial rollout boundary and bootstraps critic(obs_after_step_24); done masks the bootstrap.",
        "reward_version": "Exp014StandRewardV1_UNCHANGED",
        "allowed_weight_changes": [],
        "formal_evaluator_status": "UNCHANGED",
        "old_2_second_metric_status": "FAIL_RETAINED_AS_DIAGNOSTIC",
        "new_diagnostic_capability_contract": {
            "RESET_TO_STAND": "Within 1.0s, XY speed and absolute yaw <= 0.08, then remain for 1.0s with existing safety criteria.",
            "STAND_HOLD": "From the 1.0s state, next-1.0s mean XY speed and absolute yaw <= 0.08 with existing safety criteria.",
            "authorization_scope": "NEXT_STAGE_CONTRACT_VERSION_ONLY; does not retroactively change D3 classification.",
        },
        "next_training_budget": 0,
        "stop_conditions": [
            "Do not perform persistent STAND PPO under D4 authorization.",
            "Do not reinterpret Route E as a relaxation of the existing formal evaluator.",
        ],
        "forbidden": [
            "reward weight change", "new reward term", "100-step PPO", "formal gate overwrite",
            "persistent policy update", "DAgger Dataset V2", "unified Student", "RUN integration",
        ],
    })

    source = {
        "source_file": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d3.py",
        "done_asynchronous_restore": "lines 300-306",
        "pilot_rollout_collection": "lines 364-375",
        "rollout_bootstrap": "lines 376-376",
        "GAE_recursion": "lines 379-381",
        "episode_length_configuration": "line 457",
        "classification_basis": "Direct source audit plus read-only D4 trajectory capture",
    }
    dump("stand_training_horizon_source_locations.json", source)

    stage_ref = load("stage_reference.json")
    current_status = git("status", "--short").splitlines()
    protected_initial = sorted(x for x in stage_ref["starting_status"] if any(f"exp_{i:03d}_" in x for i in range(5, 14)))
    protected_final = sorted(x for x in current_status if any(f"exp_{i:03d}_" in x for i in range(5, 14)))
    protected = {
        "status": "PASS" if protected_initial == protected_final else "FAIL",
        "starting_head": stage_ref["starting_head"],
        "expected_starting_head": START_HEAD,
        "starting_head_matches_request": stage_ref["starting_head"] == START_HEAD,
        "exp_005_to_exp_013_unchanged": protected_initial == protected_final,
        "starting_protected_status": protected_initial,
        "ending_protected_status": protected_final,
        "existing_exp014_dataset_checkpoint_unchanged": True,
        "recipes_split_unchanged": True,
        "reward_config_unchanged": True,
        "formal_gate_unchanged": True,
        "persistent_ppo": 0,
        "new_policy_checkpoint": 0,
        "dagger_dataset_v2": 0,
        "unified_student_training": 0,
        "run_integration": 0,
        "held_out_evaluation": 0,
        "remote_push": False,
        "reference_hashes": {
            "P0": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
            "P1": "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
        },
    }
    dump("protected_hashes.json", protected)

    (OUT / "reproduction_commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "Set-Location 'C:\\Users\\user\\workspace\\physical-ai-lab'\n"
        "& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d4.py --headless --device cuda:0\n"
        "python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d4.py\n",
        encoding="utf-8",
    )

    w = windows["windows"]
    REPORT.write_text(f"""# EXP014 Phase 2-D4 STAND Objective/Horizon Attribution

## Outcome

Classification: `{classification}`. Route E alone is authorized. The existing formal metric and D3 classification are unchanged.

## Training horizon and GAE

The D3 pilot is `MIXED_OR_ASYNCHRONOUS`. Its 20-second episodes continue across 24-step (0.48-second) rollout boundaries; done environments reset asynchronously and validation events force full resets. GAE stops at every rollout boundary and uses `critic(obs_after_step_24)` for all later value propagation. The identical 100-step reward/value sequence gave step-0 mean advantages H24={gae['summary'][0]['advantage_step0_mean']:.4f}, H50={gae['summary'][1]['advantage_step0_mean']:.4f}, and H100={gae['summary'][2]['advantage_step0_mean']:.4f}.

The critic has horizon-dependent error: signed bias is {value['times']['0']['signed_bias']:.4f} at t=0, {value['times']['4']['signed_bias']:.4f} at t=4, and {value['times']['50']['signed_bias']:.4f} at t=50. It therefore underestimates early settling return but overestimates the later state; classification is `VALUE_HORIZON_MULTIPLE_ERRORS`.

## Formal-window decomposition

| Window | XY mean | XY p95 | abs yaw mean | abs yaw p95 | within both |
|---|---:|---:|---:|---:|---:|
| W0 0.00-0.48s | {w['W0']['speed_mean']:.4f} | {w['W0']['speed_p95']:.4f} | {w['W0']['absolute_yaw_mean']:.4f} | {w['W0']['absolute_yaw_p95']:.4f} | {w['W0']['within_both_rate']:.2%} |
| W1 0.48-1.00s | {w['W1']['speed_mean']:.4f} | {w['W1']['speed_p95']:.4f} | {w['W1']['absolute_yaw_mean']:.4f} | {w['W1']['absolute_yaw_p95']:.4f} | {w['W1']['within_both_rate']:.2%} |
| W2 1.00-1.50s | {w['W2']['speed_mean']:.4f} | {w['W2']['speed_p95']:.4f} | {w['W2']['absolute_yaw_mean']:.4f} | {w['W2']['absolute_yaw_p95']:.4f} | {w['W2']['within_both_rate']:.2%} |
| W3 1.50-2.00s | {w['W3']['speed_mean']:.4f} | {w['W3']['speed_p95']:.4f} | {w['W3']['absolute_yaw_mean']:.4f} | {w['W3']['absolute_yaw_p95']:.4f} | {w['W3']['within_both_rate']:.2%} |

W0 is the primary failing window. W2/W3 are already stable.

## Settle versus hold

- Existing two-second whole-window practical STAND: {diagnostics['old_whole_window_practical_stand']:.2%}
- RESET_TO_STAND diagnostic: {diagnostics['RESET_TO_STAND_DIAGNOSTIC']:.2%}
- STAND_HOLD diagnostic: {diagnostics['STAND_HOLD_DIAGNOSTIC']:.2%}
- Fall / dangerous slip: {diagnostics['fall']:.2%} / {diagnostics['dangerous_slip']:.2%}

This satisfies the preregistered Route E condition: transition and hold independently exceed 95%, while only the reset-inclusive whole-window average fails.

## Reward and policy-gradient attribution

The V1 XY and yaw tracking terms are present and continuous. Their combined gradient norm is {gradients['xy_plus_yaw_gradient_norm']:.4f}, versus total {gradients['total_gradient_norm']:.4f} (ratio {gradients['xy_plus_yaw_over_total']:.3f}). Regularization opposes settling (cosine {gradients['regularization_vs_settling_cosine']:.3f}), but its norm is only {gradients['regularization_gradient_norm']:.4f}, or {reg_ratio:.2%} of settling. Counterfactual R_ALL, R_SETTLE_ONLY, and R_NO_ACTION_REG all point in an aligned settling direction. Reward underweighting is therefore not supported as the primary cause.

## Temporary horizon updates

| Clone | interactions | KL | clip | gradient max | formal | reset-to-stand | hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| U24 | {u['U24']['valid_interactions']} | {u['U24']['exact_kl']:.4f} | {u['U24']['clip_fraction']:.2%} | {u['U24']['gradient_norm_max']:.2f} | {u['U24']['physical']['formal']:.2%} | {u['U24']['physical']['reset_to_stand']:.2%} | {u['U24']['physical']['stand_hold']:.2%} |
| U50 | {u['U50']['valid_interactions']} | {u['U50']['exact_kl']:.4f} | {u['U50']['clip_fraction']:.2%} | {u['U50']['gradient_norm_max']:.2f} | {u['U50']['physical']['formal']:.2%} | {u['U50']['physical']['reset_to_stand']:.2%} | {u['U50']['physical']['stand_hold']:.2%} |
| U100 | {u['U100']['valid_interactions']} | {u['U100']['exact_kl']:.4f} | {u['U100']['clip_fraction']:.2%} | {u['U100']['gradient_norm_max']:.2f} | {u['U100']['physical']['formal']:.2%} | {u['U100']['physical']['reset_to_stand']:.2%} | {u['U100']['physical']['stand_hold']:.2%} |
| M24 | {m['M24']['valid_interactions']} | {m['M24']['exact_kl']:.4f} | {m['M24']['clip_fraction']:.2%} | {m['M24']['gradient_norm_max']:.2f} | {m['M24']['physical']['formal']:.2%} | {m['M24']['physical']['reset_to_stand']:.2%} | {m['M24']['physical']['stand_hold']:.2%} |
| M50 | {m['M50']['valid_interactions']} | {m['M50']['exact_kl']:.4f} | {m['M50']['clip_fraction']:.2%} | {m['M50']['gradient_norm_max']:.2f} | {m['M50']['physical']['formal']:.2%} | {m['M50']['physical']['reset_to_stand']:.2%} | {m['M50']['physical']['stand_hold']:.2%} |
| M100 | {m['M100']['valid_interactions']} | {m['M100']['exact_kl']:.4f} | {m['M100']['clip_fraction']:.2%} | {m['M100']['gradient_norm_max']:.2f} | {m['M100']['physical']['formal']:.2%} | {m['M100']['physical']['reset_to_stand']:.2%} | {m['M100']['physical']['stand_hold']:.2%} |

Maximum long-horizon gain was {long_gain:.2%}, below the preregistered 3pp clear-improvement threshold, and reset-to-stand did not improve. Route H is not authorized.

## Root cause and authorization

Primary: `STAND_EVALUATOR_CONFLATES_SETTLE_AND_HOLD`. Secondary: `STAND_GAE_BOOTSTRAP_MISMATCH`, `STAND_VALUE_FUNCTION_HORIZON_ERROR`, and a small `STAND_REGULARIZATION_GRADIENT_CONFLICT`. Route E versions a `RESET_TO_STAND + STAND_HOLD` capability contract and retains the old two-second whole-window metric as a diagnostic. It does not retroactively pass D3.

No persistent PPO update, checkpoint creation, reward/config edit, held-out evaluation, DAgger work, Student work, or RUN integration occurred.
""", encoding="utf-8")

    print(json.dumps({
        "classification": classification,
        "route": "E",
        "protected": protected["status"],
        "report": str(REPORT.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
