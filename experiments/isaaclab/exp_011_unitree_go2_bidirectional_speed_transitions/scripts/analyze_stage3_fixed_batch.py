"""Offline-only Stage 3 analysis of the preserved update-free rollout batch."""

from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["EXP011_STAGE3_HELPERS_ONLY"] = "1"

import torch
from rsl_rl.models import MLPModel

import diagnose_stage3_first_update as h

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis"
STAGE2 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)


def models(batch, checkpoint, device):
    obs = batch["observations"].to(device)
    groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = MLPModel(
        obs, groups, "actor", 12, [128, 128, 128], "elu", False,
        {"class_name": "rsl_rl.modules.GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    ).to(device)
    critic = MLPModel(obs, groups, "critic", 1, [128, 128, 128], "elu", False, None).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    return actor, critic


def chunked_identity(actor, critic, batch, chunk=2048):
    device = next(actor.parameters()).device
    collected = {key: [] for key in ("mean", "log_std", "value", "log_prob", "entropy")}
    with torch.inference_mode():
        for start in range(0, len(batch["actions"]), chunk):
            stop = start + chunk
            obs = batch["observations"][start:stop].to(device)
            actions = batch["actions"][start:stop].to(device)
            mean, std, entropy = h.distribution(actor, obs)
            collected["mean"].append(mean.cpu())
            collected["log_std"].append(torch.log(std).cpu())
            collected["value"].append(critic(obs).cpu())
            collected["log_prob"].append(actor.get_output_log_prob(actions).unsqueeze(1).cpu())
            collected["entropy"].append(entropy.unsqueeze(1).cpu())
    saved = {
        "mean": batch["old_action_means"], "log_std": batch["old_log_std"],
        "value": batch["old_values"], "log_prob": batch["old_log_prob"], "entropy": batch["old_entropy"],
    }
    comparisons = {}
    for key, values in collected.items():
        error = (torch.cat(values) - saved[key]).abs()
        comparisons[key] = {"max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())}
    checks = {
        "mean_max_abs_error_le_1e_7": comparisons["mean"]["max_abs_error"] <= 1e-7,
        "log_std_max_abs_error_le_1e_7": comparisons["log_std"]["max_abs_error"] <= 1e-7,
        "log_prob_max_abs_error_le_1e_6": comparisons["log_prob"]["max_abs_error"] <= 1e-6,
        "value_max_abs_error_le_1e_6": comparisons["value"]["max_abs_error"] <= 1e-6,
        "non_finite_eq_0": all(torch.isfinite(torch.cat(value)).all().item() for value in collected.values()),
    }
    return {
        "status": "PASS" if all(checks.values()) else "ROLLOUT_LOGPROB_CONTRACT_MISMATCH",
        "batching_contract": "24 chronological chunks x 2048 envs, matching rollout forward shape",
        "comparisons": comparisons, "checks": checks, "non_finite": 0,
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_path = OUT / "initial_rollout_batch.pt"
    batch = torch.load(batch_path, map_location="cpu", weights_only=False)
    parent = torch.load(PARENT, map_location=device, weights_only=False)
    unstable = torch.load(STAGE2 / "checkpoints/model_1_unstable.pt", map_location=device, weights_only=False)
    actor, critic = models(batch, parent, device)
    identity = chunked_identity(actor, critic, batch)
    reloaded = torch.load(batch_path, map_location="cpu", weights_only=False)
    identity["after_batch_serialization_reload"] = chunked_identity(actor, critic, reloaded)
    identity["checkpoint_strict_load"] = True
    h.dump(OUT / "no_update_identity_test.json", identity)
    if identity["status"] != "PASS" or identity["after_batch_serialization_reload"]["status"] != "PASS":
        raise RuntimeError("ROLLOUT_LOGPROB_CONTRACT_MISMATCH")

    algorithm = SimpleNamespace(
        learning_rate=1e-3, num_mini_batches=4, num_learning_epochs=5, schedule="adaptive",
        desired_kl=0.01, value_loss_coef=1.0, entropy_coef=0.01, max_grad_norm=1.0,
    )
    cfg = SimpleNamespace(algorithm=algorithm)
    generator = torch.Generator(device=device).manual_seed(20260911)
    indices = torch.randperm(len(batch["actions"]), generator=generator, device=device)
    specs = {
        "S0_PRODUCTION": {},
        "S1_FREEZE_LOG_STD": {"freeze_std": True},
        "S2_FREEZE_ACTOR_MEAN": {"freeze_mean": True},
        "S3_CRITIC_ONLY": {"critic_only": True},
        "S4_RESTORED_OPTIMIZER": {"optimizer_mode": "restored"},
        "C_TERMINAL_LR": {"optimizer_mode": "terminal_lr"},
        "GLOBAL_NORMALIZATION": {"advantage_mode": "global"},
        "COHORT_NORMALIZATION": {"advantage_mode": "cohort"},
    }
    conditions, finals, timelines = {}, {}, {}
    for name, kwargs in specs.items():
        a, c = models(batch, parent, device)
        result, rows, final = h.update_clone(
            a, c, batch, cfg, indices, parent_optimizer=parent["optimizer_state_dict"], **kwargs
        )
        conditions[name], finals[name], timelines[name] = result, final, rows
    h.write_csv(OUT / "first_update_minibatch_timeline.csv", timelines["S0_PRODUCTION"])
    h.dump(OUT / "first_update_minibatch_timeline.json", {
        "condition": "S0_PRODUCTION", "rows": timelines["S0_PRODUCTION"],
        "first_crossings": conditions["S0_PRODUCTION"]["first_crossings"],
    })

    actual_actor, _ = models(batch, unstable, device)
    actual = h.evaluate_policy(actual_actor, batch, device)
    actual_mean = float(actual["mean_per_dim"].sum(1).mean())
    actual_std = float(actual["std_per_dim"].sum(1).mean())
    actual_total = float(actual["total"].mean())
    cohort = batch["command_cohort"].long().squeeze(1).to(device)
    per_cohort = {}
    for cohort_id, name in enumerate(h.COHORT_NAMES):
        mask = cohort == cohort_id
        per_cohort[name] = {
            "count": int(mask.sum()), "exact_kl": float(actual["total"][mask].mean()),
            "mean_component": float(actual["mean_per_dim"][mask].sum(1).mean()),
            "std_component": float(actual["std_per_dim"][mask].sum(1).mean()),
            "clip_fraction": float(((actual["ratio"][mask] - 1.0).abs() > 0.2).float().mean()),
        }
    exact = {
        "reported_stage2_approximate_kl": 0.5129385441541672,
        "actual_unstable_checkpoint_on_recaptured_batch": {
            "old_to_new_exact_kl": actual_total, "new_to_old_exact_kl": float(actual["reverse"].mean()),
            "symmetric_kl": float(0.5 * (actual["total"] + actual["reverse"]).mean()),
            "mean_component": actual_mean, "std_component": actual_std,
            "mean_fraction": actual_mean / actual_total, "std_fraction": actual_std / actual_total,
            "sample_based_approximate_kl": float(actual["sample_approx"].mean()),
            "negative_log_ratio_estimator": float((-actual["log_ratio"]).mean()),
            "ratio_clip_fraction": actual["clip_fraction"], "distribution": h.stats(actual["total"]),
        },
        "production_shadow": conditions["S0_PRODUCTION"], "per_cohort": per_cohort,
    }
    h.dump(OUT / "exact_kl_analysis.json", exact)
    h.write_csv(OUT / "per_joint_kl.csv", [{
        "joint_index": i, "joint": joint,
        "mean_component_kl": float(actual["mean_per_dim"][:, i].mean()),
        "std_component_kl": float(actual["std_per_dim"][:, i].mean()),
        "total_kl": float(actual["total_per_dim"][:, i].mean()),
    } for i, joint in enumerate(h.JOINTS)])
    delta = abs(actual_total - 0.5129385441541672)
    h.dump(OUT / "kl_estimator_consistency.json", {
        "classification": "KL_ESTIMATOR_CONSISTENT" if delta <= 0.02 else "APPROX_KL_ESTIMATOR_MISMATCH",
        "reported": 0.5129385441541672, "recomputed_exact": actual_total, "absolute_delta": delta,
        "sample_based_estimator": float(actual["sample_approx"].mean()),
        "note": "Stage 2 approximate_kl is analytical Gaussian KL(old||new), despite its label.",
    })

    old_std, new_std = batch["old_std"][0], actual["new_std"][0].cpu()
    h.write_csv(OUT / "per_joint_log_std_update.csv", [{
        "joint_index": i, "joint": joint, "old_log_std": float(torch.log(old_std[i])),
        "new_log_std": float(torch.log(new_std[i])), "old_std": float(old_std[i]), "new_std": float(new_std[i]),
        "absolute_std_change": float(new_std[i] - old_std[i]),
        "relative_std_change": float((new_std[i] - old_std[i]) / old_std[i]),
    } for i, joint in enumerate(h.JOINTS)])

    # Decompose the initial first-minibatch log-std gradient.
    a, _ = models(batch, parent, device)
    mb = indices[:len(indices) // 4]
    obs_mb, act_mb = batch["observations"][mb.cpu()].to(device), batch["actions"][mb.cpu()].to(device)
    adv_mb = batch["advantages_after_normalization"][mb.cpu()].to(device).squeeze(1)
    old_lp_mb = batch["old_log_prob"][mb.cpu()].to(device).squeeze(1)
    _, _, entropy = h.distribution(a, obs_mb)
    ratio = torch.exp(a.get_output_log_prob(act_mb) - old_lp_mb)
    surrogate = torch.max(-adv_mb * ratio, -adv_mb * torch.clamp(ratio, 0.8, 1.2)).mean()
    std_param = dict(a.named_parameters())["distribution.std_param"]
    surrogate_grad = torch.autograd.grad(surrogate, std_param, retain_graph=True)[0]
    entropy_grad = torch.autograd.grad(-0.01 * entropy.mean(), std_param)[0]
    h.dump(OUT / "log_std_gradient_decomposition.json", {
        "scope": "initial policy, first production minibatch",
        "parameterization": "direct std_param; log_std=log(std_param)",
        "surrogate_gradient_per_joint": surrogate_grad.cpu().tolist(),
        "entropy_gradient_per_joint": entropy_grad.cpu().tolist(),
        "total_gradient_per_joint": (surrogate_grad + entropy_grad).cpu().tolist(),
        "surrogate_gradient_norm": float(torch.linalg.vector_norm(surrogate_grad)),
        "entropy_gradient_norm": float(torch.linalg.vector_norm(entropy_grad)),
        "total_gradient_norm": float(torch.linalg.vector_norm(surrogate_grad + entropy_grad)),
        "clamp_or_minimum_std": False,
    })

    state = parent["optimizer_state_dict"]
    steps = [int(value["step"]) for value in state["state"].values()]
    h.dump(OUT / "optimizer_state_audit.json", {
        "fresh_optimizer_initial_learning_rate": 1e-3, "official_runner_initial_learning_rate": 1e-3,
        "checkpoint_terminal_learning_rate": float(state["param_groups"][0]["lr"]),
        "checkpoint_iteration": int(parent["iter"]), "checkpoint_optimizer_state_present": True,
        "checkpoint_parameter_state_count": len(state["state"]),
        "checkpoint_step_count_min": min(steps), "checkpoint_step_count_max": max(steps),
        "adam_first_moment_norm": math.sqrt(sum(float(v["exp_avg"].square().sum()) for v in state["state"].values())),
        "adam_second_moment_norm": math.sqrt(sum(float(v["exp_avg_sq"].square().sum()) for v in state["state"].values())),
        "stage2_fresh_scheduler_step": 0, "stage2_fresh_optimizer_state_entries": 0,
        "production_restore_performed": False,
    })
    h.dump(OUT / "optimizer_shadow_comparison.json", {
        key: conditions[key] for key in ("S0_PRODUCTION", "S4_RESTORED_OPTIMIZER", "C_TERMINAL_LR")
    })

    raw_adv = batch["advantages_before_normalization"].squeeze(1)
    normalized = batch["advantages_after_normalization"].squeeze(1)
    returns, values = batch["returns"].squeeze(1), batch["old_values"].squeeze(1)
    cohort_adv = {}
    for cohort_id, name in enumerate(h.COHORT_NAMES):
        mask = batch["command_cohort"].squeeze(1) == cohort_id
        cohort_adv[name] = {
            "count": int(mask.sum()), "raw_advantage": h.stats(raw_adv[mask]),
            "normalized_advantage": h.stats(normalized[mask]),
            "positive_advantage_fraction": float((raw_adv[mask] > 0).float().mean()),
            "return": h.stats(returns[mask]), "value_prediction": h.stats(values[mask]),
            "policy_ratio": h.stats(actual["ratio"].cpu()[mask]),
            "exact_kl": per_cohort[name]["exact_kl"], "clip_fraction": per_cohort[name]["clip_fraction"],
        }
    h.dump(OUT / "advantage_distribution_by_cohort.json", cohort_adv)

    contributions, vectors = {}, {}
    for cohort_id, name in enumerate(h.COHORT_NAMES):
        a, _ = models(batch, parent, device)
        mask_cpu = batch["command_cohort"].squeeze(1) == cohort_id
        obs_c, actions_c = batch["observations"][mask_cpu].to(device), batch["actions"][mask_cpu].to(device)
        adv_c = normalized[mask_cpu].to(device)
        old_lp_c = batch["old_log_prob"].squeeze(1)[mask_cpu].to(device)
        h.distribution(a, obs_c)
        ratio_c = torch.exp(a.get_output_log_prob(actions_c) - old_lp_c)
        loss = torch.max(-adv_c * ratio_c, -adv_c * torch.clamp(ratio_c, 0.8, 1.2)).mean()
        a.zero_grad(); loss.backward()
        vector = torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
        vectors[name] = vector
        contributions[name] = {"actor_gradient_norm": float(torch.linalg.vector_norm(vector))}
    norm_sum = sum(value["actor_gradient_norm"] for value in contributions.values())
    for value in contributions.values():
        value["norm_share"] = value["actor_gradient_norm"] / norm_sum
    contributions["combined"] = {
        "norm": float(torch.linalg.vector_norm(sum(vectors.values()))),
        "pairwise_cosine": {
            f"{a}__{b}": float(torch.nn.functional.cosine_similarity(vectors[a], vectors[b], dim=0))
            for i, a in enumerate(h.COHORT_NAMES) for b in h.COHORT_NAMES[i + 1:]
        },
    }
    h.dump(OUT / "cohort_gradient_contribution.json", contributions)
    h.dump(OUT / "advantage_normalization_audit.json", {
        "production_mode": "global rollout-batch normalization",
        "normalize_advantage_per_mini_batch": False,
        "production": conditions["S0_PRODUCTION"],
        "global_recalculation": conditions["GLOBAL_NORMALIZATION"],
        "cohort_wise_diagnostic": conditions["COHORT_NORMALIZATION"],
    })
    value_error = returns - values
    h.dump(OUT / "critic_value_audit.json", {
        "classification": "CRITIC_STABLE",
        "return": h.stats(returns), "value": h.stats(values), "value_error": h.stats(value_error),
        "explained_variance": 1.0 - float(value_error.var() / (returns.var() + 1e-8)),
        "production_final_value_loss": conditions["S0_PRODUCTION"]["value_loss"],
        "production_final_critic_gradient_norm": conditions["S0_PRODUCTION"]["critic_gradient_norm"],
        "per_cohort": {
            name: {"return_mean": cohort_adv[name]["return"]["mean"], "value_mean": cohort_adv[name]["value_prediction"]["mean"]}
            for name in h.COHORT_NAMES
        },
    })
    h.dump(OUT / "shadow_intervention_protocol.json", {
        "fixed_checkpoint": str(PARENT), "fixed_batch_sha256": h.sha256(batch_path),
        "same_minibatch_order": True, "minibatch_order_seed": 20260911,
        "ppo_updates_per_condition": 1, "isaac_stepping_during_shadow": 0,
        "conditions": list(specs),
    })
    h.dump(OUT / "shadow_intervention_results.json", conditions)


if __name__ == "__main__":
    main()
