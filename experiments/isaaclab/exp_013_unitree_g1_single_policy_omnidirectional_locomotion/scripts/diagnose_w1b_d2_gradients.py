"""Read-only W1B-D2 gradient, critic, action, mirror, and state diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d2_yaw_rate_tracking_boundary_diagnosis"
)
SELECTED = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
)
D1 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_d1_yaw_translation_interference_diagnosis"
)
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1b  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def critic(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = nn.Sequential(nn.Linear(124, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    model.load_state_dict({key.removeprefix("mlp."): value
                           for key, value in payload["critic_state_dict"].items()}, strict=True)
    return model.eval()


def category(name):
    if "torso" in name or "waist" in name:
        return "waist"
    for token in ("hip", "knee", "ankle", "shoulder", "elbow"):
        if token in name:
            return token
    return "hand"


def main():
    labels = ["PURE_NEG", "PURE_POS"]
    for direction in (0, 45, 90, 135, 180, 225, 270, 315):
        labels.extend((f"D{direction:03d}_NEG", f"D{direction:03d}_POS"))
    labels.append("ZERO_RETENTION")
    per = 48
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-W1B-YawWalk-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = per * len(labels)
    cfg.seed = acfg.seed = 20282021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    with launch_simulation(cfg, args):
        base = RslRlVecEnvWrapper(
            gym.make("Isaac-Exp013-G1-W1B-YawWalk-v0", cfg=cfg),
            clip_actions=acfg.clip_actions,
        )
        env, device = base.unwrapped, base.unwrapped.device
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        ids = torch.arange(env.num_envs, device=device) // per
        local = torch.arange(env.num_envs, device=device) % per
        speed = torch.zeros(env.num_envs, device=device)
        direction = torch.zeros_like(speed)
        yaw = torch.zeros_like(speed)
        yaw[ids == 0] = -.3; yaw[ids == 1] = .3
        cursor = 2
        for angle in (0, 45, 90, 135, 180, 225, 270, 315):
            for sign in (-1, 1):
                mask = ids == cursor
                direction[mask] = angle; speed[mask] = .3; yaw[mask] = .3 * sign
                cursor += 1
        mask = ids == len(labels) - 1
        direction[mask] = (local[mask] % 16) * 22.5
        speed[mask] = .3
        radians = torch.deg2rad(direction)
        vx, vy = speed * torch.cos(radians), speed * torch.sin(radians)
        actor = FrozenGaitActor(SELECTED).to(device)
        parent = FrozenGaitActor(PARENT).to(device).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
        value_model = critic(SELECTED, device)
        payload = torch.load(SELECTED, map_location=device, weights_only=False)
        std = payload["actor_state_dict"]["distribution.log_std_walk"].exp()
        mirror = json.loads((D1 / "robot_mirror_contract.json").read_text(encoding="utf-8"))
        mirror_index = torch.tensor(mirror["mirror_indices"], device=device)
        mirror_sign = torch.tensor(mirror["mirror_signs"], dtype=torch.float32, device=device)

        observations, actions, rewards, dones, values, components = [], [], [], [], [], []
        contact_history = []
        obs, _ = base.reset()
        obs = obs["policy"].to(device)
        generator = torch.Generator(device=device).manual_seed(20282021)
        reward_names = list(env.reward_manager.active_terms)
        sensor = env.scene.sensors["contact_forces"]
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        for step in range(24):
            command.external_override[:, 0] = vx
            command.external_override[:, 1] = vy
            command.external_override[:, 2] = yaw
            if step == 0:
                command._update_command()
                obs = base.get_observations()["policy"].to(device)
            with torch.no_grad():
                mean = actor(obs, torch.zeros(env.num_envs, device=device))
                action = mean + torch.randn(mean.shape, generator=generator, device=device) * std
                value = value_model(torch.cat((obs, torch.zeros(env.num_envs, 1, device=device)), -1)).squeeze(-1)
            observations.append(obs.clone()); actions.append(action); values.append(value)
            obs, reward, done, _ = base.step(action)
            obs = obs["policy"].to(device)
            rewards.append(reward); dones.append(done.float())
            components.append(env.reward_manager._step_reward.clone())
            force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
            contact_history.append((force > 5).float())
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards); dones = torch.stack(dones)
        values = torch.stack(values); components = torch.stack(components)
        contacts = torch.stack(contact_history)
        with torch.no_grad():
            next_value = value_model(torch.cat((obs, torch.zeros(env.num_envs, 1, device=device)), -1)).squeeze(-1)
        advantages = torch.zeros_like(rewards); returns = torch.zeros_like(rewards)
        gae = torch.zeros(env.num_envs, device=device); future_return = next_value
        for step in reversed(range(24)):
            future_value = next_value if step == 23 else values[step + 1]
            delta = rewards[step] + .99 * (1 - dones[step]) * future_value - values[step]
            gae = delta + .99 * .95 * (1 - dones[step]) * gae
            advantages[step] = gae
            future_return = rewards[step] + .99 * (1 - dones[step]) * future_return
            returns[step] = future_return
        norm_adv = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
        trans_index = reward_names.index("track_lin_vel_xy_exp")
        yaw_index = reward_names.index("track_ang_vel_z_exp")

        def gradient(group_mask, weight):
            flat_obs = observations[:, group_mask].reshape(-1, 123)
            flat_action = actions[:, group_mask].reshape(-1, 37)
            mean = actor(flat_obs, torch.zeros(len(flat_obs), device=device))
            logp = torch.distributions.Normal(mean, std).log_prob(flat_action).sum(-1)
            loss = -(logp * weight[:, group_mask].reshape(-1).detach()).mean()
            grads = torch.autograd.grad(loss, [p for _, p in named], retain_graph=False, allow_unused=True)
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for g, (_, p) in zip(grads, named)])
            return vector, grads

        reward_rows, vectors, layer_rows, joint_rows = [], {}, [], []
        trans_vectors, yaw_vectors = {}, {}
        for group_id, label in enumerate(labels):
            group = ids == group_id
            total, grads = gradient(group, norm_adv)
            trans, _ = gradient(group, components[:, :, trans_index])
            yaw_grad, _ = gradient(group, components[:, :, yaw_index])
            safety_weight = components.sum(-1) - components[:, :, trans_index] - components[:, :, yaw_index]
            safety, _ = gradient(group, safety_weight)
            vectors[label] = total; trans_vectors[label] = trans; yaw_vectors[label] = yaw_grad
            reward_rows.append({
                "condition": label,
                "episode_return_24step": float(rewards[:, group].sum(0).mean()),
                "translation_reward": float(components[:, group, trans_index].sum(0).mean()),
                "yaw_reward": float(components[:, group, yaw_index].sum(0).mean()),
                "safety_reward": float(safety_weight[:, group].sum(0).mean()),
                "critic_value": float(values[:, group].mean()),
                "monte_carlo_return": float(returns[:, group].mean()),
                "value_bias": float((values[:, group] - returns[:, group]).mean()),
                "advantage_mean": float(advantages[:, group].mean()),
                "advantage_std": float(advantages[:, group].std()),
                "positive_advantage_rate": float((advantages[:, group] > 0).float().mean()),
                "negative_advantage_rate": float((advantages[:, group] < 0).float().mean()),
                "total_gradient_norm": float(total.norm()),
                "translation_gradient_norm": float(trans.norm()),
                "yaw_gradient_norm": float(yaw_grad.norm()),
                "safety_gradient_norm": float(safety.norm()),
                "translation_yaw_cosine": float(nn.functional.cosine_similarity(trans[None], yaw_grad[None])),
                "negative_minibatch_fraction": float((norm_adv[:, group] < 0).float().mean()),
            })
            for (name, parameter), grad in zip(named, grads):
                layer_rows.append({"condition": label, "layer": name,
                                   "gradient_norm": float(grad.norm()) if grad is not None else 0.0})
            output = dict(zip([name for name, _ in named], grads)).get("hidden.5.weight")
            if output is not None:
                for index, joint_name in enumerate(env.scene["robot"].joint_names):
                    joint_rows.append({"condition": label, "joint": joint_name, "category": category(joint_name),
                                       "mean_head_gradient_norm": float(output[index].norm())})
        cosine_rows = []
        for left in labels:
            for right in labels:
                cosine_rows.append({
                    "left": left, "right": right,
                    "total_cosine": float(nn.functional.cosine_similarity(vectors[left][None], vectors[right][None])),
                    "translation_yaw_cross_cosine": float(
                        nn.functional.cosine_similarity(trans_vectors[left][None], yaw_vectors[right][None])
                    ),
                    "combined_projection": float(
                        torch.dot(vectors[left], vectors[right]) / vectors[right].norm().clamp_min(1e-12)
                    ),
                })

        # Counterfactual action sensitivity and mirror residual on exactly the saved on-policy states.
        action_rows, sensitivity_rows = [], []
        state_groups = {
            "pure": 0, "forward": 2, "d090": 6, "d135": 8, "d180": 10,
            "d225": 12, "d270": 14, "d315": 16,
        }
        for checkpoint_name, model in (("parent", parent), ("selected", actor.eval())):
            for source, group_id in state_groups.items():
                sample = observations[:, ids == group_id].reshape(-1, 123)[::12][:192].detach()
                for yaw_point in (-.6, -.3, 0, .3, .6):
                    center = sample.clone(); center[:, 11] = yaw_point
                    plus = center.clone(); minus = center.clone()
                    plus[:, 11] += 1e-3; minus[:, 11] -= 1e-3
                    with torch.no_grad():
                        derivative = (
                            model(plus, torch.zeros(len(plus), device=device)) -
                            model(minus, torch.zeros(len(minus), device=device))
                        ) / 2e-3
                    for joint_index, joint_name in enumerate(env.scene["robot"].joint_names):
                        sensitivity_rows.append({
                            "checkpoint": checkpoint_name, "state_group": source, "yaw_point": yaw_point,
                            "joint": joint_name, "category": category(joint_name),
                            "signed_sensitivity": float(derivative[:, joint_index].mean()),
                            "absolute_sensitivity": float(derivative[:, joint_index].abs().mean()),
                        })
                positive = sample.clone(); positive[:, 11] = .3
                mirrored = positive.clone()
                mirrored[:, 1] *= -1; mirrored[:, 3] *= -1; mirrored[:, 5] *= -1
                mirrored[:, 7] *= -1; mirrored[:, 10] *= -1; mirrored[:, 11] *= -1
                for start in (12, 49, 86):
                    mirrored[:, start:start+37] = positive[:, start:start+37][:, mirror_index] * mirror_sign
                with torch.no_grad():
                    direct = model(positive, torch.zeros(len(positive), device=device))
                    mirrored_action = model(mirrored, torch.zeros(len(mirrored), device=device))
                    expected = mirrored_action[:, mirror_index] * mirror_sign
                    no_previous = positive.clone(); no_previous[:, 86:123] = 0
                    previous = torch.linalg.vector_norm(
                        direct - model(no_previous, torch.zeros(len(positive), device=device)), dim=-1
                    )
                difference = direct - expected
                for joint_index, joint_name in enumerate(env.scene["robot"].joint_names):
                    action_rows.append({
                        "checkpoint": checkpoint_name, "state_group": source,
                        "joint": joint_name, "category": category(joint_name),
                        "mean_abs_mirror_difference": float(difference[:, joint_index].abs().mean()),
                        "action_l2": float(torch.linalg.vector_norm(difference, dim=-1).mean()),
                        "action_cosine": float(nn.functional.cosine_similarity(direct, expected, dim=-1).mean()),
                        "previous_action_contribution": float(previous.mean()),
                        "direct_action_abs_p99": float(torch.quantile(direct[:, joint_index].abs(), .99)),
                    })

        # Contact and state separability after mirroring paired condition summaries.
        contact_rows = []
        pair_names = (("PURE_POS", "PURE_NEG"), ("D090_POS", "D270_NEG"),
                      ("D135_POS", "D225_NEG"), ("D180_POS", "D180_NEG"))
        for positive, negative in pair_names:
            p_id, n_id = labels.index(positive), labels.index(negative)
            p_contact, n_contact = contacts[:, ids == p_id], contacts[:, ids == n_id]
            contact_rows.append({
                "positive": positive, "negative_mirror": negative,
                "positive_left_contact": float(p_contact[:, :, 0].mean()),
                "positive_right_contact": float(p_contact[:, :, 1].mean()),
                "negative_mirrored_left_contact": float(n_contact[:, :, 1].mean()),
                "negative_mirrored_right_contact": float(n_contact[:, :, 0].mean()),
                "contact_sequence_l1": float((p_contact.mean(1) - n_contact.flip(-1).mean(1)).abs().mean()),
                "positive_flight_fraction": float((p_contact.sum(-1) == 0).float().mean()),
                "negative_flight_fraction": float((n_contact.sum(-1) == 0).float().mean()),
                "positive_double_support": float((p_contact.sum(-1) == 2).float().mean()),
                "negative_double_support": float((n_contact.sum(-1) == 2).float().mean()),
                "landing_order": "not_recorded",
                "step_length": "not_recorded",
            })

        overlap = {}
        def auc(labels, scores):
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(scores) + 1)
            positive = labels == 1
            n_pos, n_neg = int(positive.sum()), int((~positive).sum())
            return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))

        for positive, negative in pair_names:
            p = observations[:, ids == labels.index(positive)].reshape(-1, 123).detach().cpu().numpy()
            n = observations[:, ids == labels.index(negative)].reshape(-1, 123).detach().cpu().numpy()
            # Mirror only the kinematic observation contract.
            nt = torch.tensor(n)
            nt[:, 1] *= -1; nt[:, 3] *= -1; nt[:, 5] *= -1
            nt[:, 7] *= -1; nt[:, 10] *= -1; nt[:, 11] *= -1
            mi = mirror_index.cpu(); ms = mirror_sign.cpu()
            original = nt.clone()
            for start in (12, 49, 86):
                nt[:, start:start+37] = original[:, start:start+37][:, mi] * ms
            n = nt.numpy()
            mean_distance = float(np.linalg.norm(p.mean(0) - n.mean(0)))
            pooled = float(np.linalg.norm(p.std(0) + n.std(0)) + 1e-9)
            energy_proxy = mean_distance / pooled
            p_small, n_small = p[::max(1, len(p)//512)][:512], n[::max(1, len(n)//512)][:512]
            x = np.concatenate((p_small, n_small))
            labels_binary = np.concatenate((np.ones(len(p_small), dtype=int), np.zeros(len(n_small), dtype=int)))
            midpoint = (p_small.mean(0) + n_small.mean(0)) / 2
            direction_vector = p_small.mean(0) - n_small.mean(0)
            linear_score = (x - midpoint) @ direction_vector
            nonlinear_score = np.linalg.norm(x - n_small.mean(0), axis=1) - np.linalg.norm(
                x - p_small.mean(0), axis=1
            )
            nearest = torch.cdist(torch.tensor(p_small), torch.tensor(n_small)).amin(1).mean()
            overlap[positive] = {
                "mirrored_mean_l2": mean_distance,
                "normalized_energy_distance_proxy": energy_proxy,
                "linear_auroc": auc(labels_binary, linear_score),
                "nonlinear_nearest_centroid_auroc": auc(labels_binary, nonlinear_score),
                "mmd": float(np.mean((p.mean(0) - n.mean(0)) ** 2)),
                "nearest_neighbor_distance": float(nearest),
            }

        write_csv("positive_negative_yaw_reward_advantage.csv", reward_rows)
        dump("positive_negative_yaw_reward_advantage.json", {
            "rows": reward_rows, "reward_terms": reward_names,
            "rollout_horizon": 24, "optimizer_steps": 0,
        })
        write_csv("yaw_boundary_gradient_cosines.csv", cosine_rows)
        write_csv("yaw_boundary_layerwise_gradients.csv", layer_rows)
        write_csv("yaw_boundary_jointwise_gradients.csv", joint_rows)
        dump("yaw_boundary_gradient_interaction.json", {
            "groups": reward_rows, "cosines": cosine_rows,
            "estimator": "read-only score-function policy-gradient estimator",
            "optimizer_steps": 0,
        })
        write_csv("selected_policy_yaw_mirror_by_joint.csv", action_rows)
        dump("selected_policy_yaw_mirror_asymmetry.json", {
            "rows": action_rows,
            "selected_mean_action_l2": float(np.mean([
                row["action_l2"] for row in action_rows if row["checkpoint"] == "selected"
            ])),
            "robot_mirror_contract": str(D1 / "robot_mirror_contract.json"),
        })
        write_csv("selected_policy_yaw_command_sensitivity.csv", sensitivity_rows)
        dump("selected_policy_yaw_command_sensitivity.json", {
            "rows": sensitivity_rows, "finite_difference_delta": .001,
        })
        write_csv("yaw_boundary_contact_gait_analysis.csv", contact_rows)
        dump("yaw_boundary_contact_gait_analysis.json", {"rows": contact_rows})
        dump("positive_negative_yaw_state_overlap.json", overlap)
        base.close()


if __name__ == "__main__":
    main()
