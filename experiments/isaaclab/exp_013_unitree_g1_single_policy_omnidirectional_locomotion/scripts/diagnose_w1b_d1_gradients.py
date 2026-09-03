"""Read-only score-function gradient, reward, advantage, and critic diagnosis."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
ITER1 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/checkpoints/model_1.pt"
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"))
sys.path.insert(0, str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"))
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(HERE.parent))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1b  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


class W1AVecEnv:
    def __init__(self, base):
        self.base = base
        for name in ("num_envs", "device", "max_episode_length", "num_actions"):
            setattr(self, name, getattr(base, name))
        self.gait = torch.zeros(self.num_envs, device=self.device)
        self.command = self.base.unwrapped.command_manager.get_term("base_velocity")

    @property
    def unwrapped(self):
        return self.base.unwrapped

    def reset(self):
        obs, extras = self.base.reset()
        return torch.cat((obs["policy"], self.gait[:, None]), -1), extras

    def get_observations(self):
        obs = self.base.get_observations()
        return torch.cat((obs["policy"], self.gait[:, None]), -1)

    def step(self, actions):
        obs, rewards, dones, extras = self.base.step(actions)
        return torch.cat((obs["policy"], self.gait[:, None]), -1), rewards, dones, extras

    def close(self):
        self.base.close()


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


def joint_category(name):
    if "torso" in name or "waist" in name:
        return "waist"
    for value in ("hip", "knee", "ankle", "shoulder", "elbow"):
        if value in name:
            return value
    return "hand"


def main():
    labels = [
        "G1_zero_yaw_all_direction", "G2_pure_yaw_negative", "G3_pure_yaw_positive",
        "G4_forward_negative", "G5_forward_positive", "G6_left_negative",
        "G7_left_positive", "G8_right_negative", "G9_right_positive", "G10_all_moving_turns",
    ]
    per = 96
    cfg, acfg = resolve_task_config("Isaac-Exp013-G1-W1B-YawWalk-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = per * len(labels); cfg.seed = acfg.seed = 20275021
    if args.device:
        cfg.sim.device = acfg.device = args.device
    with launch_simulation(cfg, args):
        base = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-W1B-YawWalk-v0", cfg=cfg),
                                  clip_actions=acfg.clip_actions)
        wrapped = W1AVecEnv(base)
        env, device = wrapped.unwrapped, wrapped.device
        command = wrapped.command; command.external_override_enabled = True
        ids = torch.arange(env.num_envs, device=device) // per
        env_index = torch.arange(env.num_envs, device=device) % per
        direction = torch.zeros(env.num_envs, device=device)
        speed = torch.zeros_like(direction); yaw = torch.zeros_like(direction)
        direction[ids == 0] = (env_index[ids == 0] % 16) * 22.5; speed[ids == 0] = .3
        yaw[ids == 1] = -.3; yaw[ids == 2] = .3
        speed[(ids == 3) | (ids == 4)] = .3; yaw[ids == 3] = -.3; yaw[ids == 4] = .3
        direction[(ids == 5) | (ids == 6)] = 90; speed[(ids == 5) | (ids == 6)] = .3
        yaw[ids == 5] = -.3; yaw[ids == 6] = .3
        direction[(ids == 7) | (ids == 8)] = 270; speed[(ids == 7) | (ids == 8)] = .3
        yaw[ids == 7] = -.3; yaw[ids == 8] = .3
        direction[ids == 9] = (env_index[ids == 9] % 16) * 22.5; speed[ids == 9] = .3
        yaw[ids == 9] = torch.where(env_index[ids == 9] % 2 == 0, -.3, .3)
        radians = torch.deg2rad(direction)
        vx, vy = speed * torch.cos(radians), speed * torch.sin(radians)
        reward_rows, vectors, layer_rows, joint_rows, cosine_rows = [], {}, [], [], []
        actor = FrozenGaitActor(PARENT).to(device)
        for parameter in actor.parameters():
            parameter.requires_grad_(True)
        parent_critic = critic(PARENT, device); iteration_critic = critic(ITER1, device)
        payload = torch.load(PARENT, map_location=device, weights_only=False)
        std = payload["actor_state_dict"]["distribution.log_std_walk"].exp()
        observations, actions, rewards, dones, values, components = [], [], [], [], [], []
        obs, _ = wrapped.reset(); obs = obs.to(device)
        generator = torch.Generator(device=device).manual_seed(20275021)
        reward_names = list(env.reward_manager.active_terms)
        for step in range(24):
            command.external_override[:, 0] = vx; command.external_override[:, 1] = vy
            command.external_override[:, 2] = yaw
            if step == 0:
                command._update_command(); obs = wrapped.get_observations().to(device)
            with torch.no_grad():
                mean = actor(obs[:, :123], obs[:, 123])
                action = mean + torch.randn(mean.shape, generator=generator, device=device) * std
                value = parent_critic(obs).squeeze(-1)
            observations.append(obs.clone()); actions.append(action); values.append(value)
            obs, reward, done, _ = wrapped.step(action); obs = obs.to(device)
            rewards.append(reward); dones.append(done.float())
            components.append(env.reward_manager._step_reward.clone())
        observations = torch.stack(observations); actions = torch.stack(actions)
        rewards = torch.stack(rewards); dones = torch.stack(dones); values = torch.stack(values)
        components = torch.stack(components)
        with torch.no_grad():
            next_value = parent_critic(obs).squeeze(-1)
        advantage = torch.zeros_like(rewards); gae = torch.zeros(env.num_envs, device=device)
        returns = torch.zeros_like(rewards); future_return = next_value
        for step in reversed(range(24)):
            future_value = next_value if step == 23 else values[step + 1]
            delta = rewards[step] + .99 * (1 - dones[step]) * future_value - values[step]
            gae = delta + .99 * .95 * (1 - dones[step]) * gae
            advantage[step] = gae
            future_return = rewards[step] + .99 * (1 - dones[step]) * future_return
            returns[step] = future_return
        norm_adv = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
        named = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
        trans_index = reward_names.index("track_lin_vel_xy_exp")
        yaw_index = reward_names.index("track_ang_vel_z_exp")

        def gradient(group_mask, weight):
            actor.zero_grad(set_to_none=True)
            flat_obs = observations[:, group_mask].reshape(-1, 124)
            flat_action = actions[:, group_mask].reshape(-1, 37)
            mean = actor(flat_obs[:, :123], flat_obs[:, 123])
            logp = torch.distributions.Normal(mean, std).log_prob(flat_action).sum(-1)
            loss = -(logp * weight[:, group_mask].reshape(-1).detach()).mean()
            grads = torch.autograd.grad(loss, [p for _, p in named], allow_unused=True, retain_graph=False)
            return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                              for g, (_, p) in zip(grads, named)]), grads

        for group_id, label in enumerate(labels):
            mask = ids == group_id
            total, grads = gradient(mask, norm_adv)
            trans, _ = gradient(mask, components[:, :, trans_index])
            yaw_grad, _ = gradient(mask, components[:, :, yaw_index])
            safety_weight = components.sum(-1) - components[:, :, trans_index] - components[:, :, yaw_index]
            safety, _ = gradient(mask, safety_weight)
            vectors[label] = total
            reward_rows.append({
                "checkpoint": "parent", "condition": label,
                "episode_return_24step": float(rewards[:, mask].sum(0).mean()),
                "translation_reward": float(components[:, mask, trans_index].sum(0).mean()),
                "yaw_reward": float(components[:, mask, yaw_index].sum(0).mean()),
                "other_reward": float(safety_weight[:, mask].sum(0).mean()),
                "critic_value": float(values[:, mask].mean()),
                "monte_carlo_return": float(returns[:, mask].mean()),
                "value_bias": float((values[:, mask] - returns[:, mask]).mean()),
                "advantage_mean": float(advantage[:, mask].mean()),
                "advantage_std": float(advantage[:, mask].std()),
                "positive_advantage_rate": float((advantage[:, mask] > 0).float().mean()),
                "total_gradient_norm": float(total.norm()), "translation_gradient_norm": float(trans.norm()),
                "yaw_gradient_norm": float(yaw_grad.norm()), "safety_gradient_norm": float(safety.norm()),
                "translation_yaw_cosine": float(nn.functional.cosine_similarity(trans[None], yaw_grad[None])),
                "negative_minibatch_fraction": float((norm_adv[:, mask] < 0).float().mean()),
            })
            for (name, parameter), grad in zip(named, grads):
                layer_rows.append({"group": label, "layer": name,
                                   "gradient_norm": float(grad.norm()) if grad is not None else 0.0})
            output_grad = dict(zip([n for n, _ in named], grads)).get("hidden.5.weight")
            if output_grad is not None:
                for joint_index, joint_name in enumerate(env.scene["robot"].joint_names):
                    joint_rows.append({"group": label, "joint": joint_name,
                                       "category": joint_category(joint_name),
                                       "mean_head_gradient_norm": float(output_grad[joint_index].norm())})
        for left in labels:
            for right in labels:
                cosine_rows.append({
                    "left": left, "right": right,
                    "cosine": float(nn.functional.cosine_similarity(vectors[left][None], vectors[right][None])),
                    "combined_projection": float(torch.dot(vectors[left], vectors[right]) /
                                                 vectors[right].norm().clamp_min(1e-12)),
                })
        # Evaluate the iteration-1 critic on exactly the same parent rollout states.
        with torch.no_grad():
            iter_values = iteration_critic(observations.reshape(-1, 124)).reshape_as(values)
        for group_id, label in enumerate(labels):
            mask = ids == group_id
            reward_rows.append({
                "checkpoint": "iteration1_critic_on_parent_states", "condition": label,
                "episode_return_24step": float(rewards[:, mask].sum(0).mean()),
                "translation_reward": float(components[:, mask, trans_index].sum(0).mean()),
                "yaw_reward": float(components[:, mask, yaw_index].sum(0).mean()),
                "other_reward": float((components[:, mask].sum(-1) -
                                      components[:, mask, trans_index] - components[:, mask, yaw_index]).sum(0).mean()),
                "critic_value": float(iter_values[:, mask].mean()),
                "monte_carlo_return": float(returns[:, mask].mean()),
                "value_bias": float((iter_values[:, mask] - returns[:, mask]).mean()),
                "advantage_mean": float((returns[:, mask] - iter_values[:, mask]).mean()),
                "advantage_std": float((returns[:, mask] - iter_values[:, mask]).std()),
                "positive_advantage_rate": float((returns[:, mask] - iter_values[:, mask] > 0).float().mean()),
            })
        write_csv("translation_yaw_gradient_cosines.csv", cosine_rows)
        write_csv("translation_yaw_layerwise_gradients.csv", layer_rows)
        write_csv("translation_yaw_jointwise_gradients.csv", joint_rows)
        write_csv("yaw_translation_reward_advantage_diagnosis.csv", reward_rows)
        dump("translation_yaw_gradient_interaction.json", {
            "groups": reward_rows[:10], "cosines": cosine_rows, "rollout_horizon": 24,
            "estimator": "read-only score-function policy-gradient estimator",
            "ppo_updates": 0, "optimizer_steps": 0,
        })
        dump("yaw_translation_reward_advantage_diagnosis.json", {
            "rows": reward_rows, "reward_terms": reward_names,
            "note": "24-step fresh rollout diagnosis; iteration-1 critic evaluated counterfactually on identical parent states.",
        })
        wrapped.close()


if __name__ == "__main__":
    main()
