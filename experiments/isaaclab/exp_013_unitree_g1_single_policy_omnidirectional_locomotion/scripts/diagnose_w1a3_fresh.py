"""Fresh read-only state/action/contact and PPO-gradient/critic diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a3_rear_left_low_speed_retention_diagnosis"
W1A = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
CP = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints"

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1a2  # noqa: E402,F401
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("state", "gradient"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actor(path, device, trainable=False):
    model = FrozenGaitActor(path).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    return model


def critic(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = nn.Sequential(nn.Linear(124, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                          nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)).to(device)
    model.load_state_dict({
        key.removeprefix("mlp."): value for key, value in payload["critic_state_dict"].items()
    }, strict=True)
    return model


def category(name):
    if "torso" in name or "waist" in name:
        return "waist"
    for value in ("hip", "knee", "ankle", "waist", "shoulder", "elbow"):
        if value in name:
            return value
    return "hand"


def setup(count):
    cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = count
    cfg.episode_length_s = 9
    cfg.seed = agent_cfg.seed = 20273021
    if args.device:
        cfg.sim.device = agent_cfg.device = args.device
    return cfg, agent_cfg


def state_diagnosis():
    conditions = [(225, .3), (247.5, .3), (135, .3), (112.5, .3),
                  (225, .6), (247.5, .6), (135, .6), (112.5, .6)]
    episodes = 20
    cfg, acfg = setup(len(conditions) * episodes)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0", cfg=cfg),
                                     clip_actions=acfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        robot, sensor = env.scene["robot"], env.scene.sensors["contact_forces"]
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        ids = torch.arange(env.num_envs, device=device) // episodes
        cmd = torch.tensor([[s * math.cos(math.radians(d)), s * math.sin(math.radians(d))]
                            for d, s in conditions], device=device)
        models = {
            "w1a": actor(W1A, device),
            "w1a2_120": actor(CP / "model_120.pt", device),
            "w1a2_140": actor(CP / "model_140.pt", device),
            "w1a2_160": actor(CP / "model_160.pt", device),
        }
        joint_names = list(robot.joint_names)
        feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
        source_results = {}
        drift = defaultdict(lambda: {"l2": [], "cosine": [], "previous_action": [], "joint": []})
        for source_name, source in (("w1a", models["w1a"]), ("w1a2_160", models["w1a2_160"])):
            wrapped.seed(20273021)
            obs, _ = wrapped.reset()
            obs = obs["policy"].to(device)
            steps = round(8 / float(env.step_dt))
            sums = defaultdict(lambda: torch.zeros((len(conditions), len(joint_names)), device=device))
            scalar = defaultdict(lambda: torch.zeros(len(conditions), device=device))
            contact_sequences = [[[] for _ in range(episodes)] for _ in conditions]
            for step in range(steps):
                command.external_override[:, :2] = cmd[ids]
                command.external_override[:, 2] = 0
                if step == 0:
                    command._update_command()
                    obs = wrapped.get_observations()["policy"].to(device)
                gait = torch.zeros(env.num_envs, device=device)
                with torch.inference_mode():
                    action = source(obs, gait)
                    outputs = {name: model(obs, gait) for name, model in models.items()}
                    zero_previous = obs.clone()
                    zero_previous[:, -37:] = 0
                    no_prev = source(zero_previous, gait)
                for name, output in outputs.items():
                    key = (source_name, name)
                    drift[key]["l2"].append(torch.linalg.vector_norm(output - action, dim=-1).cpu())
                    drift[key]["cosine"].append(nn.functional.cosine_similarity(output, action, dim=-1).cpu())
                    drift[key]["joint"].append((output - action).abs().cpu())
                drift[(source_name, source_name)]["previous_action"].append(
                    torch.linalg.vector_norm(no_prev - action, dim=-1).cpu())
                obs, _, _, _ = wrapped.step(action)
                obs = obs["policy"].to(device)
                forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
                contacts = forces > 5
                for ci in range(len(conditions)):
                    mask = ids == ci
                    sums["joint_position"][ci] += robot.data.joint_pos[mask].mean(0)
                    sums["joint_velocity"][ci] += robot.data.joint_vel[mask].mean(0)
                    sums["action"][ci] += action[mask].mean(0)
                    scalar["base_velocity"][ci] += torch.linalg.vector_norm(robot.data.root_lin_vel_b[mask, :2], dim=-1).mean()
                    scalar["base_angular_velocity"][ci] += robot.data.root_ang_vel_b[mask].norm(dim=-1).mean()
                    scalar["roll_pitch"][ci] += robot.data.projected_gravity_b[mask, :2].norm(dim=-1).mean()
                    scalar["base_height"][ci] += robot.data.root_pos_w[mask, 2].mean()
                    scalar["single_support"][ci] += (contacts[mask].sum(-1) == 1).float().mean()
                    scalar["double_support"][ci] += (contacts[mask].sum(-1) == 2).float().mean()
                    scalar["flight"][ci] += (contacts[mask].sum(-1) == 0).float().mean()
                    for episode, env_id in enumerate(torch.where(mask)[0].tolist()):
                        contact_sequences[ci][episode].append([bool(x) for x in contacts[env_id].tolist()])
            rows = []
            for ci, (degrees, speed) in enumerate(conditions):
                row = {"direction_deg": degrees, "speed_mps": speed}
                for name in sums:
                    row[name] = (sums[name][ci] / steps).tolist()
                for name in scalar:
                    row[name] = float(scalar[name][ci] / steps)
                row["contact_sequences"] = contact_sequences[ci]
                rows.append(row)
            source_results[source_name] = rows
        drift_rows = []
        drift_summary = {}
        for (source, target), values in drift.items():
            l2 = torch.cat(values["l2"]) if values["l2"] else torch.tensor([])
            cosine = torch.cat(values["cosine"]) if values["cosine"] else torch.tensor([])
            joints = torch.cat(values["joint"]) if values["joint"] else torch.empty((0, len(joint_names)))
            previous = torch.cat(values["previous_action"]) if values["previous_action"] else torch.tensor([])
            drift_summary[f"{source}_states__{target}_actor"] = {
                "mean_action_l2": float(l2.mean()) if l2.numel() else 0,
                "cosine": float(cosine.mean()) if cosine.numel() else 1,
                "previous_action_dependence_l2": float(previous.mean()) if previous.numel() else None,
            }
            if joints.numel():
                for index, name in enumerate(joint_names):
                    drift_rows.append({"observation_source": source, "actor": target, "joint": name,
                                       "category": category(name), "mean_abs_action_difference": float(joints[:, index].mean())})
        dump("_raw_state_action_contact.json", {"joint_names": joint_names, "sources": source_results})
        dump("rear_left_policy_action_drift.json", {"summary": drift_summary,
             "interpretation": "All actors were forwarded on identical saved in-memory observations; no action was blended."})
        with (OUT / "rear_left_policy_action_drift_by_joint.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(drift_rows[0]))
            writer.writeheader()
            writer.writerows(drift_rows)
        wrapped.close()


def gradient_diagnosis():
    labels = ["G1_225_0p3", "G2_247p5_0p3", "G3_225_0p6", "G4_247p5_0p6",
              "G5_135_0p3", "G6_112p5_0p3", "G7_expansion", "G8_forward"]
    per = 32
    cfg, acfg = setup(len(labels) * per)
    with launch_simulation(cfg, args):
        wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-W1A2-SpeedEnvelope-v0", cfg=cfg),
                                     clip_actions=acfg.clip_actions)
        env, device = wrapped.unwrapped, wrapped.unwrapped.device
        command = env.command_manager.get_term("base_velocity")
        command.external_override_enabled = True
        ids = torch.arange(env.num_envs, device=device) // per
        directions = torch.tensor([225, 247.5, 225, 247.5, 135, 112.5, 0, 0], device=device)
        speeds = torch.tensor([.3, .3, .6, .6, .3, .3, .575, .9], device=device)
        env_directions = directions[ids].clone()
        env_speeds = speeds[ids].clone()
        missing = torch.tensor([67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315], device=device)
        expansion_mask = ids == 6
        env_directions[expansion_mask] = missing[
            torch.arange(int(expansion_mask.sum()), device=device) % len(missing)
        ]
        forward_mask = ids == 7
        env_speeds[forward_mask] = torch.where(
            torch.arange(int(forward_mask.sum()), device=device) % 2 == 0, .6, 1.2
        )
        model = actor(CP / "model_120.pt", device, trainable=True)
        critics = {"iteration_120": critic(CP / "model_120.pt", device),
                   "iteration_160": critic(CP / "model_160.pt", device)}
        std = torch.load(CP / "model_120.pt", map_location=device, weights_only=False)["actor_state_dict"]["distribution.log_std_walk"].exp()
        obs, _ = wrapped.reset()
        obs = obs["policy"].to(device)
        observations, actions, rewards, dones, values = [], [], [], [], []
        generator = torch.Generator(device=device).manual_seed(20273021)
        for step in range(24):
            radians = torch.deg2rad(env_directions)
            command.external_override[:, 0] = env_speeds * torch.cos(radians)
            command.external_override[:, 1] = env_speeds * torch.sin(radians)
            command.external_override[:, 2] = 0
            if step == 0:
                command._update_command()
                obs = wrapped.get_observations()["policy"].to(device)
            gait = torch.zeros(env.num_envs, device=device)
            with torch.no_grad():
                mean = model(obs, gait)
                action = mean + torch.randn(mean.shape, generator=generator, device=device) * std
                value = critics["iteration_120"](torch.cat((obs, gait[:, None]), -1)).squeeze(-1)
            observations.append(obs.clone())
            actions.append(action.clone())
            values.append(value)
            obs, reward, done, _ = wrapped.step(action)
            obs = obs["policy"].to(device)
            rewards.append(reward.to(device))
            dones.append(done.to(device).float())
        observations = torch.stack(observations)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        dones = torch.stack(dones)
        values = torch.stack(values)
        with torch.no_grad():
            next_value = critics["iteration_120"](torch.cat((obs, torch.zeros(env.num_envs, 1, device=device)), -1)).squeeze(-1)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(env.num_envs, device=device)
        for step in reversed(range(24)):
            future = next_value if step == 23 else values[step + 1]
            delta = rewards[step] + .99 * (1 - dones[step]) * future - values[step]
            gae = delta + .99 * .95 * (1 - dones[step]) * gae
            advantages[step] = gae
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
        gradients, layer_rows, joint_rows, group_stats = {}, [], [], []
        named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        for gi, label in enumerate(labels):
            mask = ids == gi
            model.zero_grad(set_to_none=True)
            flat_obs = observations[:, mask].reshape(-1, observations.shape[-1])
            flat_action = actions[:, mask].reshape(-1, actions.shape[-1])
            flat_adv = advantages[:, mask].reshape(-1)
            mean = model(flat_obs, torch.zeros(len(flat_obs), device=device))
            logp = torch.distributions.Normal(mean, std).log_prob(flat_action).sum(-1)
            loss = -(logp * flat_adv).mean()
            loss.backward()
            vector = torch.cat([(parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
                                for _, parameter in named]).detach()
            gradients[label] = vector
            group_stats.append({"group": label, "gradient_norm": float(vector.norm()),
                                "negative_minibatch_fraction": float((flat_adv < 0).float().mean()),
                                "advantage_mean": float(flat_adv.mean()), "advantage_std": float(flat_adv.std()),
                                "positive_advantage_rate": float((flat_adv > 0).float().mean())})
            for name, parameter in named:
                layer_rows.append({"group": label, "layer": name,
                                   "gradient_norm": float(parameter.grad.norm()) if parameter.grad is not None else 0})
            output_weight = dict(named)["hidden.5.weight"].grad
            for index, name in enumerate(env.scene["robot"].joint_names):
                joint_rows.append({"group": label, "joint": name, "category": category(name),
                                   "mean_head_gradient_norm": float(output_weight[index].norm())})
        cosine_rows = []
        for left in labels:
            for right in labels:
                cosine_rows.append({"left": left, "right": right,
                                    "cosine": float(nn.functional.cosine_similarity(
                                        gradients[left][None], gradients[right][None]).item()),
                                    "combined_projection": float(torch.dot(gradients[left], gradients[right]) /
                                                                 gradients[right].norm().clamp_min(1e-12))})
        critic_result = {}
        flat_aug = advantages
        for name, critic_model in critics.items():
            rows = []
            for gi, label in enumerate(labels[:6]):
                mask = ids == gi
                flat_obs = observations[:, mask].reshape(-1, observations.shape[-1])
                with torch.no_grad():
                    predicted = critic_model(torch.cat((flat_obs, torch.zeros(len(flat_obs), 1, device=device)), -1)).squeeze(-1)
                returns = (flat_aug[:, mask] + values[:, mask]).reshape(-1)
                variance = returns.var()
                explained = 1 - (returns - predicted).var() / variance if variance > 0 else torch.tensor(0, device=device)
                rows.append({"condition": label, "mean_value": float(predicted.mean()),
                             "monte_carlo_return": float(returns.mean()), "value_bias": float((predicted - returns).mean()),
                             "explained_variance": float(explained), "advantage_mean": float(flat_aug[:, mask].mean()),
                             "advantage_std": float(flat_aug[:, mask].std()),
                             "positive_advantage_rate": float((flat_aug[:, mask] > 0).float().mean()),
                             "negative_advantage_rate": float((flat_aug[:, mask] < 0).float().mean()),
                             "termination_contribution": float(dones[:, mask].mean())})
            critic_result[name] = rows
        dump("rear_left_gradient_interaction.json", {"groups": group_stats,
             "cosines": cosine_rows, "fresh_on_policy": True, "parameter_updates": 0})
        for filename, rows in (("rear_left_gradient_cosines.csv", cosine_rows),
                               ("rear_left_layerwise_gradients.csv", layer_rows),
                               ("rear_left_jointwise_gradients.csv", joint_rows)):
            with (OUT / filename).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        dump("rear_left_critic_advantage_diagnosis.json", {
            "checkpoints": critic_result,
            "rollout_horizon": 24,
            "return_note": "24-step GAE-derived return; no optimizer step was executed.",
        })
        wrapped.close()


if args.mode == "state":
    state_diagnosis()
else:
    gradient_diagnosis()
