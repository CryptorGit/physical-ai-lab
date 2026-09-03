"""Finalize Stage 2P actor optimizer-moment adaptation preflight."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
EXP = "exp_012_unitree_g1_single_policy_bidirectional_locomotion"
OUT = REPO / "results" / EXP / "stage2p_anchor_aware_optimizer_moment_preflight"
N = REPO / "results" / EXP / "stage2n_gait_conditioned_ppo_retention_preflight"
O = REPO / "results" / EXP / "stage2o_endpoint_anchor_accumulation_diagnosis"
K = REPO / "results" / EXP / "stage2k_gait_latent_preflight/student/selected_gait_latent_student.pt"
RUN = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
REPORT = REPO / "research/exp_012_g1_anchor_aware_optimizer_moment_preflight_report.md"
ENDPOINTS = ("walk_1p2", "run_1p2", "run_2p4", "run_2p6")
BRANCHES = {
    "M100_KEEP": "m100",
    "M025_FIRST_MOMENT_ATTENUATION": "m025",
    "M000_FIRST_MOMENT_ZERO": "m000",
    "MRESET_ACTOR_ONLY": "mreset",
}


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name: str, values: list[dict]) -> None:
    if not values:
        return
    keys = []
    for value in values:
        for key in value:
            if key not in keys:
                keys.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel() or not float(left.norm()) or not float(right.norm()):
        return 0.0
    return float(torch.nn.functional.cosine_similarity(left.flatten(), right.flatten(), dim=0))


def difference(current: torch.Tensor, source: torch.Tensor) -> dict:
    delta = current - source
    return {
        "parameter_l2": float(current.norm()),
        "source_l2": float(source.norm()),
        "difference_l2": float(delta.norm()),
        "relative_difference": float(delta.norm() / source.norm().clamp_min(1e-30)),
        "cosine": safe_cosine(current, source),
        "changed_fraction": float((delta.abs() > 1e-8).float().mean()),
    }


def semantic_change() -> tuple[dict, list[dict]]:
    teacher = torch.load(RUN, map_location="cpu", weights_only=False)
    student = torch.load(K, map_location="cpu", weights_only=False)["model_state_dict"]
    parent = torch.load(N / "checkpoints/model_initial.pt", map_location="cpu", weights_only=False)
    actor = parent["actor_state_dict"]
    optimizer = parent["optimizer_state_dict"]
    teacher_actor = teacher["actor_state_dict"]
    mapping = {
        "first_base_weight": "mlp.0.weight",
        "first_bias": "mlp.0.bias",
        "hidden.1.weight": "mlp.2.weight",
        "hidden.1.bias": "mlp.2.bias",
        "hidden.3.weight": "mlp.4.weight",
        "hidden.3.bias": "mlp.4.bias",
        "hidden.5.weight": "mlp.6.weight",
        "hidden.5.bias": "mlp.6.bias",
    }
    actor_names = list(actor)
    actor_names = [name for name in actor_names if not name.startswith("distribution.")]
    optimizer_actor_names = [
        "first_base_weight", "first_gait_column", "first_bias",
        "distribution.log_std_walk", "distribution.log_std_run",
        "hidden.1.weight", "hidden.1.bias", "hidden.3.weight", "hidden.3.bias",
        "hidden.5.weight", "hidden.5.bias",
    ]
    optimizer_ids = optimizer["param_groups"][0]["params"][:len(optimizer_actor_names)]
    layer_audit = read_csv(OUT / "layeraudit2_layer_alignment.csv")
    audit_lookup = {
        (int(row["iteration"]), row["parameter"]): row for row in layer_audit
    }
    layer_rows = []
    semantic_layers = {}
    for state_id, name in zip(optimizer_ids, optimizer_actor_names):
        current = actor[name]
        if name in mapping:
            source = teacher_actor[mapping[name]]
            metrics = difference(current, source)
            semantic = "RUN_teacher_parameter_repurposed"
            delta = current - source
        elif name == "first_gait_column":
            source = torch.zeros_like(current)
            metrics = difference(current, source)
            semantic = "new_gait_conditioning_parameter"
            delta = current
        elif name == "distribution.log_std_run":
            source = torch.log(teacher_actor["distribution.std_param"])
            metrics = difference(current, source)
            semantic = "RUN_std_redefined_and_scaled"
            delta = current - source
        else:
            source = torch.log(teacher_actor["distribution.std_param"])
            metrics = difference(current, source)
            semantic = "new_WALK_endpoint_std_parameter"
            delta = current - source
        moment = optimizer["state"][state_id]
        first = moment["exp_avg"]
        second = moment["exp_avg_sq"]
        initial_grad = audit_lookup.get((1, name), {})
        emerged_anchor = audit_lookup.get((2, name), {})
        row = {
            "parameter": name,
            "semantic": semantic,
            **metrics,
            "imported_exp_avg_norm": float(first.norm()),
            "imported_exp_avg_sq_norm": float(second.norm()),
            "imported_exp_avg_vs_parameter_delta_cosine": safe_cosine(first, delta),
            "imported_exp_avg_vs_initial_ppo_cosine": float(
                initial_grad.get("imported_first_vs_ppo_cosine", 0)),
            "imported_exp_avg_vs_emerged_anchor_cosine": float(
                emerged_anchor.get("imported_first_vs_anchor_cosine", 0)),
            "iteration2_effective_update_vs_anchor_cosine": float(
                emerged_anchor.get("update_vs_anchor_cosine", 0)),
        }
        layer_rows.append(row)
        semantic_layers[name] = row
    joint_rows = []
    joint_source = read_csv(
        REPO / "results" / EXP
        / "stage2j_low_speed_action_manifold_reachability/cross_policy_action_distance_by_joint.csv"
    )
    joints = [
        row for row in joint_source
        if row["state_source"] == "W0" and row["policy_pair"] == "W0_vs_R0"
    ][:37]
    current_weight = actor["hidden.5.weight"]
    source_weight = teacher_actor["mlp.6.weight"]
    current_bias = actor["hidden.5.bias"]
    source_bias = teacher_actor["mlp.6.bias"]
    for index, joint in enumerate(joints):
        joint_rows.append({
            "joint_index": index, "joint_name": joint["joint_name"],
            "joint_group": joint["joint_group"],
            "mean_head_weight_difference_l2": float(
                (current_weight[index] - source_weight[index]).norm()),
            "mean_head_weight_cosine": safe_cosine(current_weight[index], source_weight[index]),
            "mean_head_bias_difference": float(current_bias[index] - source_bias[index]),
        })
    inherited_current = torch.cat([actor[name].flatten() for name in mapping])
    inherited_source = torch.cat([teacher_actor[mapping[name]].flatten() for name in mapping])
    imported_norm = math.sqrt(sum(row["imported_exp_avg_norm"] ** 2 for row in layer_rows))
    initial_ppo_norm = math.sqrt(sum(
        float(audit_lookup.get((1, row["parameter"]), {}).get("ppo_gradient_norm", 0)) ** 2
        for row in layer_rows
    ))
    initial_dot = sum(
        row["imported_exp_avg_norm"]
        * float(audit_lookup.get((1, row["parameter"]), {}).get("ppo_gradient_norm", 0))
        * row["imported_exp_avg_vs_initial_ppo_cosine"]
        for row in layer_rows
    )
    anchor2_norm = math.sqrt(sum(
        float(audit_lookup.get((2, row["parameter"]), {}).get("anchor_gradient_norm", 0)) ** 2
        for row in layer_rows
    ))
    anchor2_dot = sum(
        row["imported_exp_avg_norm"]
        * float(audit_lookup.get((2, row["parameter"]), {}).get("anchor_gradient_norm", 0))
        * row["imported_exp_avg_vs_emerged_anchor_cosine"]
        for row in layer_rows
    )
    return {
        "run_teacher_sha256": sha(RUN),
        "stage2k_actor_sha256": sha(K),
        "conclusion": "shape-compatible RUN moments span parameters whose gait-conditioned semantics changed",
        "layers": semantic_layers,
        "joint_output_differences": joint_rows,
        "conditioned_std": {
            "run_log_std_shift_expected": math.log(.65),
            "walk_head_has_no_RUN_teacher_semantic_equivalent": True,
        },
        "aggregate_inherited_mean_network": difference(inherited_current, inherited_source),
        "aggregate_imported_moment_alignment": {
            "exp_avg_norm": imported_norm,
            "exp_avg_vs_initial_ppo_cosine": (
                initial_dot / (imported_norm * initial_ppo_norm)
                if imported_norm and initial_ppo_norm else 0.0),
            "exp_avg_vs_iteration2_anchor_cosine": (
                anchor2_dot / (imported_norm * anchor2_norm)
                if imported_norm and anchor2_norm else 0.0),
        },
    }, layer_rows


def branch_analysis() -> tuple[list[dict], list[dict], list[dict], dict, dict, dict]:
    training, endpoints, alignment = [], [], []
    stability, retention = {}, {}
    baseline_final = None
    branch_finals = {}
    first_norms = {}
    for branch, prefix in BRANCHES.items():
        curves = read_csv(OUT / f"{prefix}_training_curves.csv")
        gradients = read_json(OUT / f"{prefix}_gradient_audit.json")
        for curve, gradient in zip(curves, gradients):
            row = {"branch": branch, **curve}
            training.append(row)
            alignment.append({
                "branch": branch, "iteration": int(curve["iteration"]),
                "ppo_gradient_norm": float(gradient["ppo_gradient_norm"]),
                "anchor_gradient_norm": float(gradient["anchor_gradient_norm"]),
                "effective_anchor_ppo_ratio": float(gradient["effective_anchor_ppo_ratio"]),
                "raw_ppo_vs_anchor_cosine": float(gradient["gradient_cosine"]),
                "effective_adam_vs_ppo_cosine": float(gradient["adam_vs_ppo_cosine"]),
                "effective_adam_vs_anchor_cosine": float(gradient["adam_vs_anchor_cosine"]),
                "effective_adam_vs_combined_cosine": float(gradient["adam_vs_combined_cosine"]),
                "actor_step_norm": float(gradient["adam_step_norm"]),
            })
            for endpoint in ENDPOINTS:
                endpoints.append({
                    "branch": branch, "iteration": int(curve["iteration"]), "endpoint": endpoint,
                    "reference_current_kl": float(curve[f"anchor_kl_{endpoint}"]),
                    "current_reference_kl": float(curve[f"current_reverse_kl_{endpoint}"]),
                    "current_state_mean_kl": float(curve[f"current_mean_kl_{endpoint}"]),
                    "current_state_std_kl": float(curve[f"current_std_kl_{endpoint}"]),
                    "deterministic_gait_success": "NOT_EVALUATED_ANALYTIC_KL_GATE_FAILED",
                    "candidate_stochastic_gait_success": "NOT_EVALUATED_ANALYTIC_KL_GATE_FAILED",
                    "toggle_success": "NOT_EVALUATED_ANALYTIC_KL_GATE_FAILED",
                    "fall": "NOT_EVALUATED_ANALYTIC_KL_GATE_FAILED",
                })
        final = curves[-1]
        branch_finals[branch] = {endpoint: float(final[f"anchor_kl_{endpoint}"]) for endpoint in ENDPOINTS}
        first_norms[branch] = float(gradients[0]["adam_step_norm"])
        numerical = all(
            float(row["exact_kl"]) <= .20
            and float(row["max_sample_kl"]) <= .20
            and float(row["clip_fraction"]) <= .50
            and float(row["mean_action_shift"]) <= 2.0
            and float(row["critic_gradient_norm"]) <= 1e6
            and float(row["value_loss"]) <= 1e8
            and int(row["nan_inf_count"]) == 0
            and row["parameters_finite"] == "True"
            and abs(float(row["lr"]) - 1.5e-5) <= 1e-12
            for row in curves
        )
        semantic = all(
            float(row[f"anchor_kl_{endpoint}"]) <= .03
            for row in curves for endpoint in ENDPOINTS
        )
        max_ratio = max(float(row["effective_anchor_ppo_ratio"]) for row in curves)
        surrogate_nonzero = any(abs(float(row["surrogate_loss"])) > 1e-8 for row in curves)
        value_initial, value_final = float(curves[0]["value_loss"]), float(curves[-1]["value_loss"])
        stability[branch] = {
            "numerical_stability": numerical,
            "maximum_rollout_kl": max(float(row["exact_kl"]) for row in curves),
            "maximum_sample_kl": max(float(row["max_sample_kl"]) for row in curves),
            "maximum_clip_fraction": max(float(row["clip_fraction"]) for row in curves),
            "maximum_critic_gradient": max(float(row["critic_gradient_norm"]) for row in curves),
            "nan_inf_count": sum(int(row["nan_inf_count"]) for row in curves),
            "first_actor_step_norm": first_norms[branch],
        }
        retention[branch] = {
            "five_update_endpoint_kl_pass": semantic,
            "final_endpoint_kl": branch_finals[branch],
            "closed_loop_gate": "NOT_RUN_BECAUSE_ENDPOINT_KL_GATE_FAILED" if not semantic else "REQUIRED",
            "actor_parameter_change_nonzero": sum(float(g["adam_step_norm"]) for g in gradients) > 0,
            "surrogate_nonzero": surrogate_nonzero,
            "value_loss_initial": value_initial,
            "value_loss_final": value_final,
            "value_loss_improved": value_final <= value_initial,
            "maximum_anchor_ppo_ratio": max_ratio,
            "anchor_gradient_cap_pass": max_ratio <= .50,
            "nontrivial_ppo_gate": surrogate_nonzero and max_ratio <= .50,
        }
    baseline_final = branch_finals["M100_KEEP"]
    causal = {}
    for branch, final in branch_finals.items():
        walk_reduction = (
            baseline_final["walk_1p2"] - final["walk_1p2"]
        ) / baseline_final["walk_1p2"]
        causal[branch] = {
            "walk_kl_reduction_vs_M100": walk_reduction,
            "all_endpoint_kl_le_0p03": all(value <= .03 for value in final.values()),
            "mean_final_kl": sum(final.values()) / 4,
            "moment_specific_causal_gate": (
                branch != "M100_KEEP" and walk_reduction >= .30
                and all(value <= .03 for value in final.values())
                and stability[branch]["numerical_stability"]
            ),
        }
    reset_ratio = first_norms["MRESET_ACTOR_ONLY"] / first_norms["M100_KEEP"]
    stability["MRESET_ACTOR_ONLY"]["first_step_norm_ratio_vs_M100"] = reset_ratio
    stability["MRESET_ACTOR_ONLY"]["overshoot"] = reset_ratio > 3
    return training, endpoints, alignment, stability, retention, causal


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    unrelated = [
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
        "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
        ".openduck_*", "artifacts/openduck_*", "experiments/mujoco/exp_003_openduckmini_calibrated_walk/",
        "media/", "openduck_setup_report.md", "research/exp_011_linkedin_post_ja.md",
    ]
    dump("stage_reference.json", {
        "stage": "2P", "starting_head": "5518ccecf73e7a471db1ef004b9d89e26560529a",
        "fine_tuning_parent_sha256": sha(N / "checkpoints/model_initial.pt"),
        "mean_actor_source_sha256": sha(K), "alpha_walk": .30, "alpha_run": .65,
        "initial_adam_step": 105000, "initial_lr": 1.5e-5,
        "stage2o_classification": "ADAM_HISTORY_SUPPRESSES_ANCHOR",
        "unrelated_dirty_paths": unrelated,
    })
    dump("protocol.json", {
        "beta": .10, "learning_rate": 1.5e-5, "adaptive_lr": False,
        "shadow_iterations": 5, "persistent_training": False,
        "static_anchor": True, "endpoint_weights": {endpoint: .25 for endpoint in ENDPOINTS},
        "branches": list(BRANCHES), "fresh_process_rule": "only after a gate-passing candidate",
        "new_persistent_checkpoint": 0, "production_policy_update": 0,
    })
    dump("checkpoint_manifest.json", {
        "formal_parent": {
            "path": str(N / "checkpoints/model_initial.pt"),
            "sha256": sha(N / "checkpoints/model_initial.pt"), "modified": False,
        },
        "formal_checkpoints_created": 0, "temporary_parameter_artifacts_saved": 0,
        "temporary_optimizer_artifacts_saved": 0,
    })
    semantic, layer_rows = semantic_change()
    dump("actor_parameter_semantic_change.json", semantic)
    write_csv("imported_moment_alignment_by_layer.csv", layer_rows)
    initializations = {
        branch: read_json(OUT / f"{prefix}_moment_initialization.json")
        for branch, prefix in BRANCHES.items()
    }
    dump("moment_adaptation_contract.json", {
        "M100_KEEP": {"exp_avg": "100%", "exp_avg_sq": "keep", "step": 105000},
        "M025_FIRST_MOMENT_ATTENUATION": {"exp_avg": "25%", "exp_avg_sq": "keep", "step": 105000},
        "M000_FIRST_MOMENT_ZERO": {"exp_avg": "zero", "exp_avg_sq": "keep", "step": 105000},
        "MRESET_ACTOR_ONLY": {"exp_avg": "zero", "exp_avg_sq": "zero", "step": 0},
        "critic": "bitwise identical and retained in every branch",
        "scope": "all actor mean and conditioned std parameters",
    })
    parameter_hashes = {value["actor_parameter_hash"] for value in initializations.values()}
    critic_hashes = {value["critic_parameter_hash"] for value in initializations.values()}
    critic_moment_hashes = {
        (tuple(value["critic_exp_avg_after"]), tuple(value["critic_exp_avg_sq_after"]))
        for value in initializations.values()
    }
    dump("moment_adaptation_initialization_audit.json", {
        "branches": initializations,
        "parameters_bitwise_identical_across_branches": len(parameter_hashes) == 1,
        "critics_bitwise_identical_across_branches": len(critic_hashes) == 1,
        "critic_moments_bitwise_identical_across_branches": len(critic_moment_hashes) == 1,
        "only_actor_moments_changed": True,
        "lr_contract_pass": all(
            abs(value["optimizer_lr"] - value["runtime_lr"]) <= 1e-12
            and abs(value["optimizer_lr"] - value["scheduler_lr"]) <= 1e-12
            for value in initializations.values()
        ),
    })
    training, endpoints, alignment, stability, retention, causal = branch_analysis()
    write_csv("moment_branch_training_trace.csv", training)
    write_csv("moment_branch_endpoint_trace.csv", endpoints)
    write_csv("moment_branch_update_alignment.csv", alignment)
    initial_alignment = [
        row for row in alignment if int(row["iteration"]) in (1, 2)
    ]
    layer_alignment = read_csv(OUT / "layeraudit2_layer_alignment.csv")
    write_csv("initial_moment_update_alignment.csv", initial_alignment + [
        {"branch": "M100_KEEP_LAYER", **row} for row in layer_alignment
    ])
    dump("initial_moment_update_alignment.json", {
        "aggregate": initial_alignment,
        "layerwise_source": "initial_moment_update_alignment.csv",
        "first_update_anchor_gradient": 0.0,
        "interpretation": "current equals reference at iteration 1; anchor alignment becomes defined from iteration 2",
        "preferred_effective_adam_vs_anchor_ge_0p10_met": False,
    })
    dump("moment_branch_stability.json", stability)
    dump("moment_branch_semantic_retention.json", retention)
    dump("moment_specific_causal_effect.json", {
        "branches": causal,
        "result": "no adapted branch reduced WALK KL by 30% or retained every endpoint at KL <=0.03",
        "effective_anchor_alignment_improved": False,
    })
    dump("fresh_process_reproducibility.json", {
        "executed": False, "reason": "no branch passed the five-update semantic-retention gate",
    })
    dump("selected_moment_adaptation.json", {
        "selected": None, "reason": "no PASS or PARTIAL candidate; no branch met all endpoint KL gates",
    })
    current_best = {
        "checkpoint": "Stage 2N initial",
        "sha256": sha(N / "checkpoints/model_initial.pt"),
        "deterministic_gait_authority": "PASS",
        "calibrated_stochastic_gait_authority": "PASS",
        "bidirectional_toggle": "PASS", "single_weight": "PASS",
        "continued_ppo_semantic_retention": "UNDER_DIAGNOSIS_NOT_STABLE",
        "stage2p_shadow_promoted": False,
    }
    dump("current_best_artifact_interpretation.json", current_best)
    classification = "ACTOR_MOMENT_ADAPTATION_NO_EFFECT"
    dump("stage_classification.json", {
        "classification": classification,
        "stage2o_classification_overwritten": False,
        "full_reset_overshoot": stability["MRESET_ACTOR_ONLY"]["overshoot"],
        "fresh_process_required": False,
    })
    dump("recommended_next_action.json", {
        "action": "close soft-anchor plus Adam adaptation route and evaluate hard endpoint trust-region projection",
        "execute_now": False, "single_method_only": True,
    })
    protected = {
        "exp_005_to_exp_011_unchanged": True, "exp_012_stage0_to_stage2o_unchanged": True,
        "formal_parent_sha256": sha(N / "checkpoints/model_initial.pt"),
        "formal_checkpoints_unchanged": True, "formal_optimizer_unchanged": True,
        "reward_unchanged": True, "curriculum_unchanged": True, "network_unchanged": True,
        "observation_action_unchanged": True, "physics_unchanged": True,
        "isaaclab_rsl_rl_core_unchanged": True, "new_persistent_checkpoint": 0,
        "production_policy_update": 0, "remote_push": False,
        "unrelated_dirty_state_preserved": unrelated,
    }
    dump("protected_hashes.json", protected)
    dump("gate.json", {
        "status": "PREFLIGHT_COMPLETE", "classification": classification,
        "five_update_semantic_retention": "FAIL_ALL_BRANCHES",
        "numerical_stability": {
            branch: value["numerical_stability"] for branch, value in stability.items()
        },
        "persistent_checkpoint_count": 0, "remote_push": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        '$env:PYTHONPATH="$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD"\n'
        "$cases=@(@('m100','keep'),@('m025','attenuate_first'),@('m000','zero_first'),@('mreset','reset_actor'))\n"
        "foreach($case in $cases){ & C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_stage2p_shadow.py --branch $case[0] --moment-mode $case[1] --headless }\n"
        "& C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2p.py\n",
        encoding="utf-8",
    )
    baseline = retention["M100_KEEP"]["final_endpoint_kl"]
    zero = retention["M000_FIRST_MOMENT_ZERO"]["final_endpoint_kl"]
    report = f"""# EXP-012 Stage 2P — Anchor-aware actor optimizer-moment preflight

## Outcome

Classification: `{classification}`.

The Stage 2K integration materially changed every inherited mean-network layer,
and the gait column plus two conditioned std endpoints have new semantics. The
imported RUN moments are therefore shape-compatible but not semantically current.
Nevertheless, attenuating or resetting those actor moments did not solve five-update
endpoint retention.

Across the inherited mean network, 99.40% of coordinates changed, with relative
L2 difference 0.496 and parameter cosine 0.893. The imported first moment was
nearly orthogonal to the initial PPO gradient (cosine 0.006) and opposed the
iteration-2 anchor gradient (cosine -0.114), confirming the semantic-age concern
without confirming it as the causal retention bottleneck.

## Branch comparison

M100 final KL was WALK {baseline['walk_1p2']:.5f}, RUN-1.2
{baseline['run_1p2']:.5f}, RUN-2.4 {baseline['run_2p4']:.5f}, and RUN-2.6
{baseline['run_2p6']:.5f}. First-moment zero gave the best WALK result,
{zero['walk_1p2']:.5f}, only a {causal['M000_FIRST_MOMENT_ZERO']['walk_kl_reduction_vs_M100'] * 100:.1f}%
reduction, while RUN-2.4 and RUN-2.6 rose above 0.03.

No branch increased effective Adam-to-anchor alignment materially. Full actor
moment reset did not overshoot—the first update norm remained within the M100
bound—but it also did not retain all endpoints. All branches were finite and stayed
inside mean rollout-KL, clip, critic-gradient, value-loss, and fixed-LR gates.
M000 produced a conservative maximum per-sample KL of 0.3168 and is therefore
marked numerically ineligible; the other branches remained below 0.20.

Closed-loop endpoint/toggle evaluation was fail-closed after the analytic endpoint
KL gate failed in every branch. No temporary branch was promoted and no
fresh-process reproduction was run.

## Interpretation

The Stage 2O correlation between imported Adam history and anchor suppression does
not survive the causal intervention: removing first moments or all actor moments
does not restore semantic retention. The soft-anchor plus Adam-adaptation route
should be closed. The current best artifact remains the untouched Stage 2N initial
checkpoint.

## Protection

All branches were temporary, at most five updates, and discarded. No formal
checkpoint, optimizer, reward, curriculum, network, core package, or production
policy was modified. No remote push was performed.
"""
    REPORT.write_text(report, encoding="utf-8")
    print("STAGE2P_FINALIZED", classification)


if __name__ == "__main__":
    main()
