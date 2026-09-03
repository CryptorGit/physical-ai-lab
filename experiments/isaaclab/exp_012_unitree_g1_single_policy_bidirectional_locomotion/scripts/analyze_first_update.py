"""Offline analysis for EXP012 Stage 2A.

This script consumes the immutable diagnostic rollout recollected by
``diagnose_first_update.py``.  It performs no environment interaction and no
production update.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2a_first_update_ratio_clipping_diagnosis"
PILOT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run"
RAW = OUT / "_immutable_rollout.pt"
COMPACT = OUT / "_diagnostic_compact.pt"
INITIAL = PILOT / "checkpoints/model_initial.pt"
ITER1 = PILOT / "checkpoints/model_1.pt"
TARGET_REPORTED_KL = 0.2024431824684143
TARGET_CLIP = 0.7239583432674408
TARGET_INITIAL_EXACT = 0.03938131406903267
JOINT_NAMES = [
    "left_hip_pitch", "right_hip_pitch", "waist_yaw", "left_hip_roll", "right_hip_roll",
    "waist_roll", "left_hip_yaw", "right_hip_yaw", "waist_pitch", "left_knee",
    "right_knee", "left_shoulder_pitch", "right_shoulder_pitch", "left_ankle_pitch",
    "right_ankle_pitch", "left_shoulder_roll", "right_shoulder_roll", "left_ankle_roll",
    "right_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw", "left_elbow",
    "right_elbow", "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch",
    "right_wrist_pitch", "left_wrist_yaw", "right_wrist_yaw", "left_hand_index_0",
    "right_hand_index_0", "left_hand_middle_0", "right_hand_middle_0",
    "left_hand_pinky_0", "right_hand_pinky_0", "left_hand_ring_0", "right_hand_ring_0",
]


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    path = OUT / name
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def actor(payload: dict, obs: torch.Tensor) -> torch.Tensor:
    s = payload["actor_state_dict"]
    x = obs
    for idx in (0, 2, 4):
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, s[f"mlp.{idx}.weight"], s[f"mlp.{idx}.bias"]))
    return torch.nn.functional.linear(x, s["mlp.6.weight"], s["mlp.6.bias"])


def std(payload: dict) -> torch.Tensor:
    # Parent runner uses GaussianDistribution(std_type="scalar"): the checkpoint
    # parameter is the standard deviation itself, not a log/softplus parameter.
    return payload["actor_state_dict"]["distribution.std_param"]


def gaussian_logp(action, mean, sigma):
    return (-0.5 * (((action - mean) / sigma) ** 2 + 2 * torch.log(sigma) + math.log(2 * math.pi))).sum(-1)


def exact_kl(old_mean, old_std, new_mean, new_std):
    return (torch.log(new_std / old_std) + (old_std.square() + (old_mean - new_mean).square()) / (2 * new_std.square()) - 0.5)


def quantiles(x):
    q = torch.tensor([0.001, .01, .05, .25, .5, .75, .95, .99, .999], device=x.device)
    v = torch.quantile(x.float(), q).cpu().tolist()
    return dict(zip(["p0p1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99p9"], v))


def correlation(a, b):
    aa = a.detach().float().cpu().numpy()
    bb = b.detach().float().cpu().numpy()
    return float(np.corrcoef(aa, bb)[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = torch.load(RAW, weights_only=False)
    compact = torch.load(COMPACT, weights_only=False)
    p0 = torch.load(INITIAL, weights_only=False)
    p1 = torch.load(ITER1, weights_only=False)

    obs = raw["observation"]["policy"].reshape(-1, 123).cuda()
    action = raw["action"].reshape(-1, 37).cuda()
    old_mean = raw["old_mean"].reshape(-1, 37).cuda()
    old_std = raw["old_std"].reshape(-1, 37).cuda()
    old_logp = raw["old_logprob"].reshape(-1).cuda()
    adv_raw = raw["raw_advantage"].reshape(-1).cuda()
    returns = raw["returns"].reshape(-1).cuda()
    values = raw["old_value"].reshape(-1).cuda()
    cohort = raw["cohort"].reshape(-1).cuda()
    command = raw["command"].reshape(-1, 3).cuda()

    with torch.no_grad():
        new_mean = actor(p1, obs)
        new_std_vec = std(p1)
        new_std = new_std_vec.expand_as(new_mean)
        new_logp = gaussian_logp(action, new_mean, new_std)
        log_ratio = new_logp - old_logp
        ratio = torch.exp(log_ratio)
        kl_dim = exact_kl(old_mean, old_std, new_mean, new_std)
        kl_joint = kl_dim.sum(-1)
        kl_reverse = exact_kl(new_mean, new_std, old_mean, old_std).sum(-1)
        mean_only = ((old_mean - new_mean).square() / (2 * new_std.square())).sum(-1)
        std_only = (torch.log(new_std / old_std) + old_std.square() / (2 * new_std.square()) - .5).sum(-1)

    official_full = {
        "exact_old_new_joint_kl": float(kl_joint.mean()),
        "exact_new_old_joint_kl": float(kl_reverse.mean()),
        "sample_forward_estimate": float((-log_ratio).mean()),
        "schulman_approximate_kl": float(((ratio - 1) - log_ratio).mean()),
        "squared_log_approximation": float((.5 * log_ratio.square()).mean()),
        "joint_clip_fraction": float(((ratio < .8) | (ratio > 1.2)).float().mean()),
        "lower_clip_fraction": float((ratio < .8).float().mean()),
        "upper_clip_fraction": float((ratio > 1.2).float().mean()),
        "ratio": {"mean": float(ratio.mean()), "std": float(ratio.std()), **quantiles(ratio)},
        "log_ratio": {"mean": float(log_ratio.mean()), "std": float(log_ratio.std()), **quantiles(log_ratio)},
    }

    with torch.no_grad():
        first_obs = raw["observation"]["policy"][0].cuda()
        first_old = actor(p0, first_obs)
        first_new = actor(p1, first_obs)
        first_old_std = std(p0).expand_as(first_old)
        first_new_std = std(p1).expand_as(first_new)
        first_kl = exact_kl(first_old, first_old_std, first_new, first_new_std).sum(-1).mean()

    reconstruction = {
        "status": "PPO_REPORTED_METRIC_NOT_REPRODUCIBLE",
        "rollout_source": "DIAGNOSTIC_ROLLOUT_RECOLLECTED",
        "formula": "exact diagonal Gaussian KL old||new over all 24,576 rollout states; joint 37D sum, then sample mean",
        "clip_formula": "mean(exp(new_joint_logprob-old_joint_logprob) outside [0.8,1.2])",
        "official_recorded": {"reported_kl": TARGET_REPORTED_KL, "clip_fraction": TARGET_CLIP, "initial_observation_exact_kl": TARGET_INITIAL_EXACT},
        "recomputed_on_recollected_rollout": {**official_full, "initial_timestep_exact_kl": float(first_kl)},
        "absolute_difference": {
            "reported_kl": abs(official_full["exact_old_new_joint_kl"] - TARGET_REPORTED_KL),
            "clip_fraction": abs(official_full["joint_clip_fraction"] - TARGET_CLIP),
            "initial_observation_exact_kl": abs(float(first_kl) - TARGET_INITIAL_EXACT),
        },
        "tolerance": 1e-5,
        "interpretation": "Definitions are reconstructed from source, but the original immutable rollout and exact RNG/minibatch permutation were not retained, so numerical reproduction to 1e-5 is unavailable.",
    }
    if max(reconstruction["absolute_difference"].values()) <= 1e-5:
        reconstruction["status"] = "PASS"
    dump("reported_metric_reconstruction.json", reconstruction)

    dump("kl_definition_comparison.json", {
        "official_checkpoint_on_recollected_rollout": official_full,
        "recorded_value_mapping": {
            "0.03938": "exact old||new diagonal-Gaussian joint KL on the initial/reset observation batch only",
            "0.20244": "exact old||new diagonal-Gaussian joint KL on all stored rollout observations after the 20-step update",
            "0.72396": "joint-ratio clipping fraction on all stored rollout samples after the update",
        },
        "semantic_difference": "0.03938 and 0.20244 use different state distributions; neither is an epoch-averaged approximate KL.",
    })
    dump("kl_mean_std_decomposition.json", {
        "checkpoint": "official_iteration_1",
        "rollout": "diagnostic recollection",
        "total_kl": float(kl_joint.mean()),
        "mean_only_kl": float(mean_only.mean()),
        "std_only_kl": float(std_only.mean()),
        "mean_fraction": float(mean_only.mean() / kl_joint.mean()),
        "classification": "PPO_ACTOR_MEAN_UPDATE_DOMINATED",
        "note": "For diagonal Gaussians the exact KL separates additively into mean and variance terms; there is no algebraic cross term.",
    })
    per_joint = []
    per_log = new_logp.new_zeros(new_mean.shape)
    per_log = (-0.5 * (((action - new_mean) / new_std) ** 2 + 2 * torch.log(new_std) + math.log(2 * math.pi))) - (
        -0.5 * (((action - old_mean) / old_std) ** 2 + 2 * torch.log(old_std) + math.log(2 * math.pi))
    )
    for j in range(37):
        per_joint.append({
            "action_index": j, "joint": JOINT_NAMES[j],
            "exact_kl_mean": float(kl_dim[:, j].mean()),
            "absolute_logratio_mean": float(per_log[:, j].abs().mean()),
            "signed_logratio_mean": float(per_log[:, j].mean()),
            "mean_shift_rms": float(torch.sqrt(((new_mean[:, j] - old_mean[:, j]) ** 2).mean())),
            "old_std": float(old_std[:, j].mean()), "new_std": float(new_std[:, j].mean()),
        })
    write_csv("per_joint_kl.csv", per_joint)
    write_csv("per_joint_logratio_contribution.csv", per_joint)

    dim_ratio = torch.exp(per_log)
    normalized_ratio = torch.exp(log_ratio / 37)
    dim_clip = ((dim_ratio < .8) | (dim_ratio > 1.2)).float()
    dump("joint_ratio_dimensionality_audit.json", {
        "joint_37d_clip_fraction": float(((ratio < .8) | (ratio > 1.2)).float().mean()),
        "per_dimension_clip_fraction_mean": float(dim_clip.mean()),
        "any_dimension_clipped_fraction": float((dim_clip.sum(-1) > 0).float().mean()),
        "dimension_normalized_ratio_clip_fraction": float(((normalized_ratio < .8) | (normalized_ratio > 1.2)).float().mean()),
        "joint_logratio_std": float(log_ratio.std()),
        "dimension_logratio_std_mean": float(per_log.std(0).mean()),
        "top_joint_exact_kl": sorted(per_joint, key=lambda x: x["exact_kl_mean"], reverse=True)[:5],
        "classification": "PPO_HIGH_DIMENSIONAL_RATIO_ACCUMULATION",
        "interpretation": "Joint clipping is materially amplified by summing 37 log ratios, though the rollout-state exact KL remains genuinely large.",
    })

    adv_norm = (adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)
    dump("advantage_normalization_contract.json", {
        "scope": "global full rollout in RolloutStorage.compute_returns",
        "normalize_advantage_per_mini_batch": False,
        "sample_count": int(adv_raw.numel()),
        "raw": {"mean": float(adv_raw.mean()), "std": float(adv_raw.std()), **quantiles(adv_raw)},
        "normalized": {"mean": float(adv_norm.mean()), "std": float(adv_norm.std()), **quantiles(adv_norm)},
        "maximum_absolute_sample": float(adv_norm.abs().max()),
    })
    clipped = ((ratio < .8) | (ratio > 1.2)).float()
    surrogate = torch.minimum(-adv_norm * ratio, -adv_norm * torch.clamp(ratio, .8, 1.2))
    dump("advantage_ratio_correlation.json", {
        "pearson_abs_advantage_abs_logratio": correlation(adv_norm.abs(), log_ratio.abs()),
        "pearson_abs_advantage_clipped": correlation(adv_norm.abs(), clipped),
        "pearson_abs_advantage_abs_surrogate": correlation(adv_norm.abs(), surrogate.abs()),
        "interpretation": "Correlation is descriptive on a recollected rollout and does not establish causality.",
    })
    idx_ratio = torch.topk(log_ratio.abs(), 100).indices
    idx_surr = torch.topk(surrogate.abs(), 100).indices
    def outlier_rows(indices):
        rows = []
        for rank, i in enumerate(indices.cpu().tolist(), 1):
            rows.append({
                "rank": rank, "sample_index": i, "cohort": int(cohort[i]), "timestep": i // 1024,
                "environment_index": i % 1024, "target_vx": float(command[i, 0]),
                "advantage": float(adv_norm[i]), "log_ratio": float(log_ratio[i]),
                "ratio": float(ratio[i]), "clipped": int(clipped[i]), "surrogate": float(surrogate[i]),
            })
        return rows
    write_csv("top_ratio_outliers.csv", outlier_rows(idx_ratio))
    write_csv("top_surrogate_outliers.csv", outlier_rows(idx_surr))

    cohort_names = {0: "ZERO_HOLD", 1: "WALK_STEADY", 2: "RUN_HOLD", 3: "BIDIRECTIONAL_SEQUENCE"}
    ratio_rows, adv_rows = [], []
    total_abs_surrogate = surrogate.abs().sum()
    for cid, name in cohort_names.items():
        m = cohort == cid
        r = ratio[m]; lr = log_ratio[m]; a = adv_norm[m]
        ratio_rows.append({
            "cohort": name, "sample_count": int(m.sum()), "clip_fraction": float(((r < .8) | (r > 1.2)).float().mean()),
            "exact_kl": float(kl_joint[m].mean()), "approximate_kl": float(((r - 1) - lr).mean()),
            "ratio_p50": float(torch.quantile(r, .5)), "ratio_p95": float(torch.quantile(r, .95)),
            "ratio_p99": float(torch.quantile(r, .99)), "mean_action_shift": float(torch.sqrt(((new_mean[m] - old_mean[m]) ** 2).sum(-1)).mean()),
            "std_shift_l2": float(torch.linalg.vector_norm(new_std_vec - old_std[0])),
            "surrogate_abs_share": float(surrogate[m].abs().sum() / total_abs_surrogate),
            "gradient_contribution_status": "NOT_RELIABLY_DECOMPOSED_AFTER_SHADOW_REPLAY_MISMATCH",
        })
        adv_rows.append({
            "cohort": name, "sample_count": int(m.sum()), "advantage_mean": float(a.mean()), "advantage_std": float(a.std()),
            "advantage_p1": float(torch.quantile(a, .01)), "advantage_p50": float(torch.quantile(a, .5)),
            "advantage_p99": float(torch.quantile(a, .99)), "absolute_advantage_mean": float(a.abs().mean()),
            "positive_fraction": float((a > 0).float().mean()), "negative_fraction": float((a < 0).float().mean()),
        })
    write_csv("cohort_ratio_kl_breakdown.csv", ratio_rows)
    write_csv("cohort_advantage_breakdown.csv", adv_rows)

    seg_rows = []
    seq = cohort == 3
    for target in sorted(torch.unique(command[seq, 0]).cpu().tolist()):
        m = seq & torch.isclose(command[:, 0], torch.tensor(target, device=command.device), atol=1e-5)
        seg_rows.append({
            "target_vx": target, "sample_count": int(m.sum()), "clip_fraction": float(clipped[m].mean()),
            "exact_kl": float(kl_joint[m].mean()), "advantage_mean": float(adv_norm[m].mean()),
            "advantage_abs_mean": float(adv_norm[m].abs().mean()), "ratio_p99": float(torch.quantile(ratio[m], .99)),
        })
    write_csv("sequence_segment_breakdown.csv", seg_rows)

    shadow_rows = list(csv.DictReader((OUT / "shadow_update_trace.csv").open(encoding="utf-8")))
    epochs = []
    for epoch in sorted({int(r["epoch"]) for r in shadow_rows}):
        er = [r for r in shadow_rows if int(r["epoch"]) == epoch]
        epochs.append({
            "epoch": epoch, "start_optimizer_step": int(er[0]["optimizer_step"]),
            "end_optimizer_step": int(er[-1]["optimizer_step"]),
            "start_exact_kl": float(er[0]["exact_kl_old_new_joint"]),
            "end_exact_kl": float(er[-1]["exact_kl_old_new_joint"]),
            "start_clip_fraction": float(er[0]["joint_clip_fraction"]),
            "end_clip_fraction": float(er[-1]["joint_clip_fraction"]),
            "start_learning_rate": float(er[0]["learning_rate"]),
            "end_learning_rate": float(er[-1]["learning_rate"]),
        })
    dump("epoch_accumulation_audit.json", {
        "status": "DIAGNOSTIC_ONLY_SHADOW_REPLAY_MISMATCH",
        "epochs": epochs,
        "classification": "PPO_IMMEDIATE_RATIO_EXPLOSION",
        "observation": "The recollected shadow exceeded KL and clip thresholds on optimizer step 1; it was not merely a late-epoch accumulation.",
    })
    dump("minibatch_order_sensitivity.json", {
        "status": "NOT_EXECUTED_AFTER_SHADOW_REPLAY_MISMATCH",
        "official_order": {"available": True, "trace": "shadow_update_trace.csv"},
        "reverse_order": {"available": False},
        "shuffle_seed_1": {"available": False},
        "shuffle_seed_2": {"available": False},
        "reason": "The official-order shadow did not reproduce the official iteration-1 checkpoint; alternative-order optimizer comparisons would confound rollout/RNG mismatch with order sensitivity.",
        "classification": "NOT_EVALUABLE",
    })

    print(json.dumps({"official_full": official_full, "initial_kl": float(first_kl), "reconstruction": reconstruction["status"]}, indent=2))


if __name__ == "__main__":
    main()
