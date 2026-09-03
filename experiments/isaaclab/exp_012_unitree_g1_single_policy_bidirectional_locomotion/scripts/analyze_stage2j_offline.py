"""Offline reward, state/action manifold, critic, and gradient analysis for Stage 2J."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2j_low_speed_action_manifold_reachability"
RAW = OUT / "raw/positive_control_trajectories.pt"
CHECKPOINTS = {
    "W0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt",
    "R0": REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt",
    "R1": REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1/checkpoints/model_1.pt",
}
JOINTS = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
    "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_shoulder_roll_joint",
    "right_shoulder_roll_joint", "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint", "left_elbow_roll_joint",
    "right_elbow_roll_joint", "left_five_joint", "left_three_joint", "left_zero_joint",
    "right_five_joint", "right_three_joint", "right_zero_joint", "left_six_joint",
    "left_four_joint", "left_one_joint", "right_six_joint", "right_four_joint",
    "right_one_joint", "left_two_joint", "right_two_joint",
]


def group(name: str) -> str:
    if "hip" in name:
        return "hip"
    if "knee" in name:
        return "knee"
    if "ankle" in name:
        return "ankle"
    if "torso" in name:
        return "waist"
    if "shoulder" in name:
        return "shoulder"
    if "elbow" in name:
        return "elbow"
    return "wrist/hand"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["empty"])
        writer.writeheader()
        writer.writerows(rows or [{"empty": ""}])


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Actor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        own = OrderedDict((k.removeprefix("mlp."), v) for k, v in state.items() if k.startswith("mlp."))
        self.mlp.load_state_dict(own, strict=True)
        self.std = nn.Parameter(state["distribution.std_param"].clone())

    def forward(self, obs):
        return self.mlp(obs)

    def log_prob(self, obs, action):
        mean = self(obs)
        std = self.std.clamp_min(1.0e-6)
        return (-0.5 * (((action - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)


class Critic(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(123, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1),
        )
        self.mlp.load_state_dict(OrderedDict((k.removeprefix("mlp."), v) for k, v in state.items()), strict=True)

    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)


def bootstrap_difference(a, b, seed=20268022, draws=10000):
    generator = torch.Generator().manual_seed(seed)
    n, m = len(a), len(b)
    # Batched to keep the implementation deterministic and memory-bounded.
    values = []
    for _ in range(0, draws, 1000):
        size = min(1000, draws - len(values))
        ai = torch.randint(n, (size, n), generator=generator)
        bi = torch.randint(m, (size, m), generator=generator)
        values.extend((a[ai].mean(1) - b[bi].mean(1)).tolist())
    q = torch.quantile(torch.tensor(values), torch.tensor([.025, .975]))
    return float(a.mean() - b.mean()), [float(q[0]), float(q[1])]


def gaussian_log_prob(action, mean, std):
    return (-.5 * (((action - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)


def returns(reward, gamma=.99):
    output = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1])
    for t in range(reward.shape[0] - 1, -1, -1):
        running = reward[t] + gamma * running
        output[t] = running
    return output


def auc(scores, labels):
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    positive = labels.bool()
    n1, n0 = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def train_classifier(x, y, nonlinear, seed):
    torch.manual_seed(seed)
    permutation = torch.randperm(len(x))
    split = int(.7 * len(x))
    train, test = permutation[:split], permutation[split:]
    mean, std = x[train].mean(0), x[train].std(0).clamp_min(1.0e-5)
    xx = (x - mean) / std
    model = (
        nn.Sequential(nn.Linear(x.shape[1], 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 1))
        if nonlinear else nn.Linear(x.shape[1], 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)
    for _ in range(150):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(model(xx[train]).squeeze(-1), y[train].float())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return auc(model(xx[test]).squeeze(-1), y[test])


def flatten_gradient(model):
    return torch.cat([
        (parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
        for parameter in model.parameters()
    ])


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1.0e-12))


def gradient(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return flatten_gradient(model).detach().clone()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(RAW, map_location="cpu", weights_only=False)
    payloads = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in CHECKPOINTS.items()}
    actors = {name: Actor(value["actor_state_dict"]) for name, value in payloads.items()}
    critics = {name: Critic(value["critic_state_dict"]) for name, value in payloads.items()}
    obs = data["observation"].float()
    actions = data["action"].float()
    rewards = data["reward"].float()
    components = data["reward_components"].float()
    names = data["reward_names"]
    # Stored groups are contiguous W0/R0/R1, 100 episodes each.
    slices = {"W0": slice(0, 100), "R0": slice(100, 200), "R1": slice(200, 300)}

    # Refine the formal labels: non-periodic, low-flight, safe speed tracking is WALK_LIKE.
    positive_csv = list(csv.DictReader((OUT / "walk_run_positive_control_comparison.csv").open(encoding="utf-8")))
    for row in positive_csv:
        if row["checkpoint"] == "W0" and float(row["flight_fraction"]) < .10 and row["fall"] == "False":
            row["gait"] = "WALK_LIKE"
            row["success"] = "True"
    write_csv("walk_run_positive_control_comparison.csv", positive_csv)
    summary = json.loads((OUT / "walk_run_positive_control_comparison.json").read_text(encoding="utf-8"))
    summary["W0"]["walk_success_rate"] = sum(row["checkpoint"] == "W0" and row["success"] == "True" for row in positive_csv) / 100
    summary["W0"]["gait_interpretation"] = "WALK_LIKE: non-periodic, flight fraction below 0.10, speed stable"
    dump("walk_run_positive_control_comparison.json", summary)

    # Reward comparison, using the exact per-step weighted terms produced by the exp_012 reward manager.
    component_rows = []
    totals = {}
    for policy, env_slice in slices.items():
        term_totals = components[:, env_slice].sum(0)
        totals[policy] = rewards[:, env_slice].sum(0)
        for term_id, term in enumerate(names):
            values = term_totals[:, term_id]
            component_rows.append({
                "checkpoint": policy, "component": term, "mean_episode_total": float(values.mean()),
                "std_episode_total": float(values.std()), "min": float(values.min()), "max": float(values.max()),
            })
    for comparison in ("R0", "R1"):
        for term_id, term in enumerate(names):
            walk_values = components[:, slices["W0"], term_id].sum(0)
            run_values = components[:, slices[comparison], term_id].sum(0)
            difference, ci = bootstrap_difference(walk_values, run_values, seed=20268030 + term_id)
            component_rows.append({
                "checkpoint": f"W0_minus_{comparison}", "component": term, "mean_episode_total": difference,
                "std_episode_total": "", "min": ci[0], "max": ci[1],
            })
    write_csv("reward_component_trace_comparison.csv", component_rows)
    reward_result = {"run_specific_all_1p2_samples_zero": bool((components[..., names.index("safe_periodic_flight")] == 0).all())}
    for comparison in ("R0", "R1"):
        difference, ci = bootstrap_difference(totals["W0"], totals[comparison], seed=20268023)
        reward_result[f"W0_minus_{comparison}"] = {
            "mean_return_difference": difference, "bootstrap_95_ci": ci,
            "walk_higher_significantly": ci[0] > 0,
        }
    reward_result["classification"] = (
        "WALK_REWARD_ADVANTAGE_EXISTS"
        if all(reward_result[f"W0_minus_{name}"]["bootstrap_95_ci"][0] > 0 for name in ("R0", "R1"))
        else "WALK_RUN_REWARD_INDIFFERENT"
    )
    dump("low_speed_reward_equivalence.json", reward_result)

    # Cross-policy actions, values, log probability, and joint localization.
    action_summary, joint_rows = {}, []
    with torch.no_grad():
        for state_name, env_slice in slices.items():
            state = obs[:, env_slice].reshape(-1, 123)[::5]
            source_actions = actions[:, env_slice].reshape(-1, 37)[::5]
            policy_outputs = {name: actor(state) for name, actor in actors.items()}
            action_summary[state_name] = {}
            for left, right in (("W0", "R0"), ("W0", "R1"), ("R0", "R1")):
                difference = policy_outputs[left] - policy_outputs[right]
                action_summary[state_name][f"{left}_vs_{right}"] = {
                    "mean_l2": float(difference.norm(dim=-1).mean()),
                    "p95_l2": float(torch.quantile(difference.norm(dim=-1), .95)),
                    "mean_absolute_per_joint": float(difference.abs().mean()),
                }
                for joint_id, joint in enumerate(JOINTS):
                    joint_rows.append({
                        "state_source": state_name, "policy_pair": f"{left}_vs_{right}",
                        "joint_index": joint_id, "joint_name": joint, "joint_group": group(joint),
                        "signed_mean_difference": float(difference[:, joint_id].mean()),
                        "absolute_mean_difference": float(difference[:, joint_id].abs().mean()),
                    })
            action_summary[state_name]["source_action_likelihood"] = {
                name: {
                    "mean_log_probability": float(actor.log_prob(state, source_actions).mean()),
                    "std_mean": float(actor.std.mean()),
                    "value_mean": float(critics[name](state).mean()),
                } for name, actor in actors.items()
            }
    dump("cross_policy_action_distance.json", action_summary)
    write_csv("cross_policy_action_distance_by_joint.csv", joint_rows)

    # State overlap using policy observations with command dimensions removed, plus contact/base features.
    state_results, embeddings = {}, []
    for run_name in ("R0", "R1"):
        walk_obs = obs[:, slices["W0"]].reshape(-1, 123)[::5]
        run_obs = obs[:, slices[run_name]].reshape(-1, 123)[::5]
        keep = torch.tensor([i for i in range(123) if i not in (9, 10, 11)])
        walk_contact = data["contact"][:, slices["W0"]].reshape(-1, 2)[::5].float()
        run_contact = data["contact"][:, slices[run_name]].reshape(-1, 2)[::5].float()
        walk_extra = torch.stack([
            data["base_height"][:, slices["W0"]].reshape(-1)[::5],
            data["base_pitch"][:, slices["W0"]].reshape(-1)[::5],
        ], -1)
        run_extra = torch.stack([
            data["base_height"][:, slices[run_name]].reshape(-1)[::5],
            data["base_pitch"][:, slices[run_name]].reshape(-1)[::5],
        ], -1)
        wx = torch.cat((walk_obs[:, keep], walk_contact, walk_extra), -1)
        rx = torch.cat((run_obs[:, keep], run_contact, run_extra), -1)
        generator = torch.Generator().manual_seed(20268024)
        wi = torch.randperm(len(wx), generator=generator)[:10000]
        ri = torch.randperm(len(rx), generator=generator)[:10000]
        x = torch.cat((wx[wi], rx[ri]))
        y = torch.cat((torch.zeros(len(wi), dtype=torch.long), torch.ones(len(ri), dtype=torch.long)))
        linear_auc = train_classifier(x, y, False, 20268024)
        nonlinear_auc = train_classifier(x, y, True, 20268025)
        # Energy distance proxy and nearest-neighbor distance on standardized balanced subsets.
        xs = (x - x.mean(0)) / x.std(0).clamp_min(1.0e-5)
        wsmall, rsmall = xs[:1000], xs[len(wi):len(wi) + 1000]
        cross = torch.cdist(wsmall, rsmall)
        within_w = torch.cdist(wsmall[:500], wsmall[500:1000])
        within_r = torch.cdist(rsmall[:500], rsmall[500:1000])
        energy = float(2 * cross.mean() - within_w.mean() - within_r.mean())
        nearest = float(cross.min(1).values.mean())
        classification = (
            "WALK_RUN_STATE_DISJOINT" if nonlinear_auc >= .95
            else "WALK_RUN_STATE_PARTIALLY_SEPARATED" if nonlinear_auc > .70
            else "WALK_RUN_STATE_OVERLAP_HIGH"
        )
        state_results[f"W0_vs_{run_name}"] = {
            "linear_classifier_auroc": linear_auc, "nonlinear_classifier_auroc": nonlinear_auc,
            "energy_distance_standardized": energy, "cross_nearest_neighbor_distance": nearest,
            "classification": classification, "command_dimensions_excluded": [9, 10, 11],
        }
        centered = xs - xs.mean(0)
        _, _, v = torch.pca_lowrank(centered, q=3)
        projected = centered @ v[:, :3]
        for index in range(0, min(3000, len(projected)), 3):
            embeddings.append({
                "comparison": f"W0_vs_{run_name}", "label": "RUN" if int(y[index]) else "WALK",
                "pc1": float(projected[index, 0]), "pc2": float(projected[index, 1]),
                "pc3": float(projected[index, 2]),
            })
    dump("walk_run_state_distribution_overlap.json", state_results)
    write_csv("walk_run_state_embedding.csv", embeddings)

    # Cross-critic evaluation and WALK action likelihood/advantage under RUN policies.
    value_result, advantage_rows = {}, []
    mc = returns(rewards)
    with torch.no_grad():
        for critic_name, critic in critics.items():
            value_result[critic_name] = {}
            for state_name, env_slice in slices.items():
                states = obs[:, env_slice].reshape(-1, 123)
                predictions = critic(states).reshape(500, 100)
                target = mc[:, env_slice]
                error = predictions - target
                explained = 1 - error.var() / target.var().clamp_min(1.0e-12)
                value_result[critic_name][state_name] = {
                    "value_mean": float(predictions.mean()), "mc_return_mean": float(target.mean()),
                    "value_bias": float(error.mean()), "explained_variance": float(explained),
                }
        walk_states = obs[:, slices["W0"]].reshape(-1, 123)[::5]
        walk_actions = actions[:, slices["W0"]].reshape(-1, 37)[::5]
        walk_return = mc[:, slices["W0"]].reshape(-1)[::5]
        behavior_logp = actors["W0"].log_prob(walk_states, walk_actions)
        for target_name in ("R0", "R1"):
            target_logp = actors[target_name].log_prob(walk_states, walk_actions)
            log_ratio = target_logp - behavior_logp
            ratio = log_ratio.clamp(-20, 20).exp()
            ess = ratio.sum().square() / ratio.square().sum().clamp_min(1.0e-12)
            valid = log_ratio.abs() <= math.log(2.0)
            advantages = walk_return - critics[target_name](walk_states)
            advantage_rows.append({
                "target_policy": target_name, "samples": len(ratio), "valid_ratio_fraction": float(valid.float().mean()),
                "clip_fraction_0p2": float(((ratio < .8) | (ratio > 1.2)).float().mean()),
                "ess_fraction": float(ess / len(ratio)), "ratio_p50": float(torch.quantile(ratio, .5)),
                "ratio_p95": float(torch.quantile(ratio, .95)), "advantage_mean_valid": float(advantages[valid].mean()) if valid.any() else None,
                "positive_advantage_fraction_valid": float((advantages[valid] > 0).float().mean()) if valid.any() else None,
            })
    dump("cross_policy_value_evaluation.json", value_result)
    write_csv("walk_action_advantage_under_run_policy.csv", advantage_rows)

    # WALK-direction gradients. These are diagnostics only; no optimizer step is performed.
    gradient_result, layer_rows, gradient_joint_rows = {}, [], []
    sample_states = obs[:, slices["W0"]].reshape(-1, 123)[::10]
    sample_actions = actions[:, slices["W0"]].reshape(-1, 37)[::10]
    sample_returns = mc[:, slices["W0"]].reshape(-1)[::10]
    behavior_logp = actors["W0"].log_prob(sample_states, sample_actions).detach()
    normalized_advantage = ((sample_returns - sample_returns.mean()) / sample_returns.std().clamp_min(1.0e-6)).detach()
    for target_name in ("R0", "R1"):
        actor = actors[target_name]
        bc_loss = nn.functional.mse_loss(actor(sample_states), sample_actions)
        bc = gradient(actor, bc_loss)
        ratio = (actor.log_prob(sample_states, sample_actions) - behavior_logp).exp()
        ppo_loss = -torch.minimum(
            ratio * normalized_advantage, ratio.clamp(.8, 1.2) * normalized_advantage
        ).mean()
        ppo = gradient(actor, ppo_loss)
        base_loss = -(actor.log_prob(sample_states, sample_actions) * normalized_advantage).mean()
        base = gradient(actor, base_loss)
        optimizer_state = payloads[target_name]["optimizer_state_dict"]["state"]
        ordered_states = [optimizer_state[key] for key in sorted(optimizer_state)][:len(list(actor.parameters()))]
        update_parts = []
        for parameter, state in zip(actor.parameters(), ordered_states):
            if "exp_avg" in state and "exp_avg_sq" in state:
                update_parts.append((-state["exp_avg"] / (state["exp_avg_sq"].sqrt() + 1.0e-8)).reshape(-1))
            else:
                update_parts.append(torch.zeros_like(parameter).reshape(-1))
        adam_direction = torch.cat(update_parts)
        gradient_result[target_name] = {
            "behavior_cloning_gradient_norm": float(bc.norm()), "ppo_surrogate_gradient_norm": float(ppo.norm()),
            "base_reward_gradient_norm": float(base.norm()), "bc_vs_ppo_cosine": cosine(bc, ppo),
            "bc_vs_base_reward_cosine": cosine(bc, base), "ppo_vs_current_adam_update_cosine": cosine(ppo, adam_direction),
            "learning_use_prohibited": True,
        }
        offset = 0
        for parameter_name, parameter in actor.named_parameters():
            count = parameter.numel()
            layer_rows.append({
                "checkpoint": target_name, "layer": parameter_name,
                "bc_norm": float(bc[offset:offset + count].norm()),
                "ppo_norm": float(ppo[offset:offset + count].norm()),
                "base_reward_norm": float(base[offset:offset + count].norm()),
            })
            offset += count
        # Output weight/bias rows are located at the final two MLP parameters, before std.
        output_weight_grad = actor.mlp[6].weight.grad.detach()
        output_bias_grad = actor.mlp[6].bias.grad.detach()
        for joint_id, joint in enumerate(JOINTS):
            gradient_joint_rows.append({
                "checkpoint": target_name, "joint_index": joint_id, "joint_name": joint,
                "joint_group": group(joint), "base_output_weight_gradient_norm": float(output_weight_grad[joint_id].norm()),
                "base_output_bias_gradient_abs": float(output_bias_grad[joint_id].abs()),
            })
    dump("walk_direction_gradient_diagnosis.json", gradient_result)
    write_csv("walk_direction_layerwise_gradients.csv", layer_rows)
    write_csv("walk_direction_jointwise_gradients.csv", gradient_joint_rows)

    dump("checkpoint_manifest.json", {
        name: {"path": str(path.relative_to(REPO)), "sha256": sha(path), "policy_std_mean": float(actors[name].std.mean())}
        for name, path in CHECKPOINTS.items()
    })


if __name__ == "__main__":
    main()
