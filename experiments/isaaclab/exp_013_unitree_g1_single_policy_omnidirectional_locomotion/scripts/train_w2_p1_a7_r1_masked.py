"""One formal A7-R1 rear-yaw teacher run with accepted-env masked PPO."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import sys
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r1_rear_yaw_start_teacher_masked_ppo"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
PARENT = BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
TEACHER = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
MASK_SHA = "0e32a2b41eae4996c1ec6acf7ef929c473af76e9685e14c2f12f738e1b9e6fb6"
POOL_SHA = "1397a99c6fb8975c43b6f951ee82432a1d543e13ea94a7991bd7373bf8544853"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: F401,E402
import g1_omnidirectional.tasks  # noqa: F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--updates", type=int, default=150)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

N = 1024
ROLLIN = 150
WINDOW = 24
OFFSETS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 251]
SEED = 20278421
SAVE = {0, 1, 10, 20, 45, 75, 100, 120, 130, 140, 150}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hobj(value) -> str:
    h = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            h.update(repr(item).encode())

    visit(value)
    return h.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def minimum_jerk(value):
    value = value.clamp(0.0, 1.0)
    return 10 * value**3 - 15 * value**4 + 6 * value**5


class Actor(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone())
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone())
        self.first_bias = nn.Parameter(state["first_bias"].clone())
        self.hidden = nn.Sequential(nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37))
        self.hidden.load_state_dict(OrderedDict((key.removeprefix("hidden."), value) for key, value in state.items() if key.startswith("hidden.")))
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"].clone())
        self.register_buffer("log_std_run", state["distribution.log_std_run"].clone())

    def forward(self, obs, gait):
        first = F.linear(obs, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait.reshape(-1, 1) * self.first_gait_column.T)

    def checkpoint_state(self):
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        state["distribution.log_std_walk"] = state.pop("log_std_walk")
        state["distribution.log_std_run"] = state.pop("log_std_run")
        return state


class Critic(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(124, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1))
        self.load_state_dict(state)

    def forward(self, obs):
        return self.mlp(obs).squeeze(-1)


def phase(update):
    if update <= 20:
        return "R1_REAR_0P15", 0.15
    if update <= 45:
        return "R2_REAR_0P20", 0.20
    if update <= 75:
        return "R3_REAR_0P25", 0.25
    if update <= 120:
        return "R4_REAR_0P30", 0.30
    return "R5_CONSOLIDATION", None


def largest_remainder(count, residual):
    weights = torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64)
    raw = count * weights + torch.tensor(residual, dtype=torch.float64)
    allocated = torch.floor(raw).long()
    for index in torch.argsort(raw - allocated, descending=True)[: count - int(allocated.sum())]:
        allocated[index] += 1
    return allocated.tolist(), (raw - allocated).tolist()


def physical_targets(train_ids, rear_speed, update, cursor, residual, mirror, device):
    allocation, new_residual = largest_remainder(len(train_ids), residual)
    targets = torch.zeros(N, 3, device=device)
    category = torch.full((N,), -1, dtype=torch.long, device=device)
    ids = train_ids.clone()
    rear_count, other_count, static_count = allocation
    rear_ids = ids[:rear_count]
    other_ids = ids[rear_count : rear_count + other_count]
    static_ids = ids[rear_count + other_count : rear_count + other_count + static_count]
    if rear_speed is None:
        speeds = [0.15, 0.20, 0.25, 0.30]
        counts = [int(rear_count * x) for x in (0.10, 0.15, 0.25)]
        counts.append(rear_count - sum(counts))
        start = 0
        for speed, count in zip(speeds, counts):
            targets[rear_ids[start : start + count], 0] = -speed
            start += count
    else:
        targets[rear_ids, 0] = -rear_speed
    targets[rear_ids, 2] = -0.3
    category[rear_ids] = 0
    other = []
    for angle in (0, 45, 90, 135, 225, 270, 315):
        rad = math.radians(angle)
        for yaw in (-0.3, 0.0, 0.3):
            other.append((0.3 * math.cos(rad), 0.3 * math.sin(rad), yaw))
    for position, env_id in enumerate(other_ids.tolist()):
        targets[env_id] = torch.tensor(other[(position + cursor) % len(other)], device=device)
    category[other_ids] = 1
    static = []
    for angle in range(0, 360, 22):
        rad = math.radians(angle)
        static.append((0.3 * math.cos(rad), 0.3 * math.sin(rad), 0.0))
    static += [(0.0, 0.0, -0.3), (0.0, 0.0, 0.3), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]
    for angle in range(0, 360, 45):
        rad = math.radians(angle)
        for yaw in (-0.3, 0.0, 0.3):
            static.append((0.3 * math.cos(rad), 0.3 * math.sin(rad), yaw))
    for position, env_id in enumerate(static_ids.tolist()):
        targets[env_id] = torch.tensor(static[(position + cursor) % len(static)], device=device)
    category[static_ids] = 2
    if mirror:
        targets[:, 1] *= -1
        targets[:, 2] *= -1
    return targets, category, allocation, new_residual


def write_checkpoint(update, actor, critic, optimizer, parent, runtime, infos):
    path = OUT / "checkpoints" / f"model_{update:03d}.pt"
    payload = {
        "iter": update,
        "actor_state_dict": actor.checkpoint_state(),
        "critic_state_dict": {key: value.detach().cpu() for key, value in critic.state_dict().items()},
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "normalizer_state": copy.deepcopy(parent["normalizer_state"]),
        "sampler_state_dict": copy.deepcopy(parent["sampler_state_dict"]),
        "a7_r1_runtime_state": copy.deepcopy(runtime),
        "infos": infos,
    }
    torch.save(payload, path)
    return path


OUT.mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints").mkdir(exist_ok=True)
parent = torch.load(PARENT, map_location="cpu", weights_only=False)
masks = json.loads((M0 / "a7_environment_masks.json").read_text())["batches"]
if json.loads((M0 / "a7_environment_mask_hashes.json").read_text())["global_hash"] != MASK_SHA:
    raise RuntimeError("EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL")

cfg, agent_cfg = resolve_task_config("Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = N
cfg.episode_length_s = 12.0
cfg.seed = 20278501
cfg.observations.policy.enable_corruption = False
if args.device:
    cfg.sim.device = agent_cfg.device = args.device

training_rows = []
checkpoint_rows = []
runtime = {"collection_cursor": 0, "quota_residual": [0.0, 0.0, 0.0], "policy_rng_seed": SEED, "pending_mirror_state": None, "ppo_interactions": 0, "teacher_rollin_steps": 0, "prefix_warmup_steps": 0, "total_simulator_env_steps": 0}

with launch_simulation(cfg, args):
    wrapped = RslRlVecEnvWrapper(gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg), clip_actions=agent_cfg.clip_actions)
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    feet = sensor.find_bodies(".*_ankle_roll_link")[0]
    robot_feet = robot.find_bodies(".*_ankle_roll_link")[0]
    command = env.command_manager.get_term("base_velocity")
    command.external_override_enabled = True
    teacher = FrozenGaitActor(TEACHER).to(env.device).eval()
    # S0/M0 construct the teacher and then immediately perform the first reset.
    # Preserve that exact RNG boundary before constructing trainable modules,
    # whose default CPU initializers consume RNG despite subsequent state load.
    baseline_cpu_rng = torch.get_rng_state().clone()
    baseline_cuda_rng = torch.cuda.get_rng_state(env.device).clone()
    baseline_numpy_rng = copy.deepcopy(np.random.get_state())
    baseline_python_rng = random.getstate()
    actor = Actor(parent["actor_state_dict"]).to(env.device)
    critic = Critic(parent["critic_state_dict"]).to(env.device)
    actor_params = [actor.first_base_weight, actor.first_gait_column, actor.first_bias, actor.hidden[1].weight, actor.hidden[1].bias, actor.hidden[3].weight, actor.hidden[3].bias, actor.hidden[5].weight, actor.hidden[5].bias]
    critic_params = list(critic.parameters())
    optimizer = torch.optim.Adam([{"params": actor_params, "lr": 1.5e-5, "name": "actor_mean"}, {"params": critic_params, "lr": 1.5e-5, "name": "critic"}], lr=1.5e-5)
    optimizer.load_state_dict(copy.deepcopy(parent["optimizer_state_dict"]))
    std = actor.log_std_walk.exp()
    gait = torch.zeros(N, device=env.device)
    ids = torch.arange(N, device=env.device)

    def restore_rng():
        torch.set_rng_state(baseline_cpu_rng)
        torch.cuda.set_rng_state(baseline_cuda_rng, env.device)
        np.random.set_state(copy.deepcopy(baseline_numpy_rng))
        random.setstate(baseline_python_rng)

    def replay_to_batch(batch):
        restore_rng()
        selected = None
        for batch_id in range(batch + 1):
            env.reset(env_ids=ids)
            command.external_override.zero_()
            command._update_command()
            obs = wrapped.get_observations().to(env.device)
            speed_sum = torch.zeros(N, device=env.device)
            yaw_sum = torch.zeros(N, device=env.device)
            fall = torch.zeros(N, dtype=torch.bool, device=env.device)
            slip = fall.clone(); impact = fall.clone(); saturation = fall.clone()
            slip_streak = torch.zeros(N, dtype=torch.long, device=env.device)
            saturation_streak = slip_streak.clone()
            limits = robot.data.joint_vel_limits
            limits = limits[..., 1].abs() if limits.ndim == 3 else limits
            for step in range(ROLLIN):
                with torch.inference_mode(): action = teacher(obs["policy"], gait)
                obs, _, done, extras = wrapped.step(action); obs = obs.to(env.device)
                timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); fall |= done.bool() & ~timeout
                force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1); contact = force > 5
                foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
                bad = ((foot_speed > 0.55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5
                impact |= force.amax(1) > 3500
                ratio = robot.data.joint_vel.abs().div(limits.clamp_min(1e-6)).amax(1); saturation_streak = torch.where(ratio > 0.95, saturation_streak + 1, torch.zeros_like(saturation_streak)); saturation |= saturation_streak >= 5
                if step >= 50:
                    speed_sum += torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=-1); yaw_sum += robot.data.root_ang_vel_b[:, 2].abs()
            accepted = (speed_sum / 100 <= 0.08) & (yaw_sum / 100 <= 0.08) & ~fall & ~slip & ~impact & ~saturation
            runtime["teacher_rollin_steps"] += ROLLIN * N
            runtime["total_simulator_env_steps"] += ROLLIN * N
            if batch_id == batch:
                expected = torch.tensor(masks[str(batch)]["accepted_mask"], device=env.device)
                if not torch.equal(accepted, expected):
                    mismatch = torch.nonzero(accepted != expected, as_tuple=False).flatten().cpu().tolist()
                    with (M0 / "raw_source_batch_inventory.csv").open(encoding="utf-8", newline="") as handle:
                        reference_rows = {
                            int(row["environment_index"]): row
                            for row in csv.DictReader(handle)
                            if int(row["source_batch_id"]) == batch
                        }
                    mismatch_details = []
                    for env_id in mismatch[:128]:
                        reference = reference_rows[env_id]
                        mismatch_details.append({
                            "environment_index": env_id,
                            "actual_accepted": bool(accepted[env_id]),
                            "expected_pool_selected": bool(expected[env_id]),
                            "actual_mean_speed": float((speed_sum / 100)[env_id]),
                            "reference_mean_speed": float(reference["mean_speed"]),
                            "actual_mean_abs_yaw": float((yaw_sum / 100)[env_id]),
                            "reference_mean_abs_yaw": float(reference["mean_abs_yaw"]),
                            "actual_fall": bool(fall[env_id]),
                            "actual_slip": bool(slip[env_id]),
                            "actual_impact": bool(impact[env_id]),
                            "actual_saturation": bool(saturation[env_id]),
                            "reference_rejection_reason": reference["rejection_reason"],
                        })
                    dump("a7_full_batch_replay_identity.json", {"status": "FAIL", "source_batch": batch, "actual_accepted": int(accepted.sum()), "expected_accepted": int(expected.sum()), "mismatch_count": len(mismatch), "mismatch_env_ids": mismatch[:128], "mismatch_details": mismatch_details, "optimizer_updates": len(training_rows)})
                    raise RuntimeError("EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL accept/reject")
                selected = (obs, accepted)
        return selected

    def collect_pass(batch, offset, targets, train_mask, noise_seed):
        obs, accepted = replay_to_batch(batch)
        initial_hash = hobj({"root": robot.data.root_state_w, "joint_pos": robot.data.joint_pos, "joint_vel": robot.data.joint_vel, "obs": obs["policy"], "previous_action": env.action_manager.prev_action, "contact": sensor.data.net_forces_w_history[:, -1, feet, :]})
        generator = torch.Generator(device=env.device).manual_seed(noise_seed)
        alive = train_mask.clone()
        data = {key: [] for key in ("observation", "action", "old_logp", "old_value", "reward", "done", "valid")}
        fall = torch.zeros(N, dtype=torch.bool, device=env.device); slip = fall.clone(); slip_streak = torch.zeros(N, dtype=torch.long, device=env.device)
        total_steps = offset + WINDOW
        for step in range(total_steps):
            alpha = minimum_jerk(torch.tensor(step / 75.0, device=env.device))
            physical = targets * alpha
            actor_command = physical.clone(); actor_command[:, 2] = torch.where(actor_command[:, 2] > 0, actor_command[:, 2] * 1.5, actor_command[:, 2])
            command.external_override.zero_(); command.external_override[train_mask] = actor_command[train_mask]; command._update_command()
            obs = wrapped.get_observations().to(env.device); full_obs = torch.cat((obs["policy"], gait[:, None]), dim=1)
            with torch.inference_mode():
                mean = actor(obs["policy"], gait); noise = torch.randn(mean.shape, generator=generator, device=env.device); sampled = mean + noise * std; housekeeping = teacher(obs["policy"], gait); action = torch.where(train_mask[:, None], sampled, housekeeping); value = critic(full_obs); logp = (-0.5 * (((sampled - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)
            in_window = step >= offset
            if in_window:
                for key, value_tensor in (("observation", full_obs), ("action", sampled), ("old_logp", logp), ("old_value", value), ("valid", alive)):
                    data[key].append(value_tensor.detach().cpu())
            obs, reward, done, extras = wrapped.step(action); obs = obs.to(env.device)
            actual_yaw = robot.data.root_ang_vel_b[:, 2]
            old_yaw_reward = torch.exp(-torch.square(actor_command[:, 2] - actual_yaw) / 0.5**2)
            physical_yaw_reward = torch.exp(-torch.square(physical[:, 2] - actual_yaw) / 0.5**2)
            reward = reward + env.step_dt * (physical_yaw_reward - old_yaw_reward)
            timeout = extras.get("time_outs", torch.zeros_like(done)).bool(); fall |= done.bool() & ~timeout
            force = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1); contact = force > 5; foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1); bad = ((foot_speed > 0.55) & contact).any(1); slip_streak = torch.where(bad, slip_streak + 1, torch.zeros_like(slip_streak)); slip |= slip_streak >= 5
            if in_window:
                data["reward"].append(reward.detach().cpu()); data["done"].append(done.detach().cpu())
            alive &= ~done.bool()
        last_obs = torch.cat((obs["policy"], gait[:, None]), dim=1)
        with torch.inference_mode(): last_value = critic(last_obs).cpu()
        data = {key: torch.stack(value) for key, value in data.items()}
        data["last_value"] = last_value; data["initial_hash"] = initial_hash; data["fall"] = fall.cpu(); data["slip"] = slip.cpu()
        runtime["prefix_warmup_steps"] += offset * N
        runtime["total_simulator_env_steps"] += total_steps * N
        return data

    def compact_pair(pair):
        compact = []
        for payload in pair:
            valid = payload["valid"].bool(); done = payload["done"].bool(); values = payload["old_value"]; rewards = payload["reward"]; advantage = torch.zeros_like(values); carry = torch.zeros(N)
            for step in range(WINDOW - 1, -1, -1):
                next_value = payload["last_value"] if step == WINDOW - 1 else values[step + 1]; alive = (~done[step]).float(); delta = rewards[step] + 0.99 * alive * next_value - values[step]; carry = delta + 0.99 * 0.95 * alive * carry; advantage[step] = carry
            index = valid.nonzero(as_tuple=False); flat = index[:, 0] * N + index[:, 1]; raw_advantage = advantage.flatten()[flat]
            compact.append({"observation": payload["observation"].flatten(0, 1)[flat], "action": payload["action"].flatten(0, 1)[flat], "old_logp": payload["old_logp"].flatten()[flat], "old_value": values.flatten()[flat], "advantage": raw_advantage, "return": raw_advantage + values.flatten()[flat], "valid_count": len(flat)})
        return {key: torch.cat([entry[key] for entry in compact]) if key != "valid_count" else sum(entry[key] for entry in compact) for key in compact[0]}

    def ppo_update(storage, update):
        observation = storage["observation"].to(env.device); action = storage["action"].to(env.device); old_logp = storage["old_logp"].to(env.device); old_value = storage["old_value"].to(env.device); returns = storage["return"].to(env.device); advantage = storage["advantage"].to(env.device); advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        with torch.inference_mode(): old_mean = actor(observation[:, :123], observation[:, 123])
        generator = torch.Generator(device=env.device).manual_seed(SEED + update)
        metrics = []
        for epoch in range(5):
            permutation = torch.randperm(len(observation), generator=generator, device=env.device)
            for indices in torch.tensor_split(permutation, 4):
                mean = actor(observation[indices, :123], observation[indices, 123]); value = critic(observation[indices]); logp = (-0.5 * (((action[indices] - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1); ratio = (logp - old_logp[indices]).exp(); surrogate = torch.maximum(-advantage[indices] * ratio, -advantage[indices] * ratio.clamp(0.8, 1.2)).mean(); clipped_value = old_value[indices] + (value - old_value[indices]).clamp(-0.2, 0.2); value_loss = torch.maximum((value - returns[indices]) ** 2, (clipped_value - returns[indices]) ** 2).mean(); entropy = (0.5 * (1 + math.log(2 * math.pi)) + std.log()).sum(); loss = surrogate + value_loss - 0.008 * entropy
                optimizer.zero_grad(); loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(actor_params + critic_params, 1.0); optimizer.step(); metrics.append((float(loss.detach()), float(value_loss.detach()), float(gradient)))
        with torch.inference_mode(): new_mean = actor(observation[:, :123], observation[:, 123]); exact_kl = float((0.5 * torch.square((new_mean - old_mean) / std).sum(-1)).mean()); new_logp = (-0.5 * (((action - new_mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1); ratio = (new_logp - old_logp).exp(); clip_fraction = float(((ratio < 0.8) | (ratio > 1.2)).float().mean()); shift = float((new_mean - old_mean).norm(dim=-1).mean())
        return {"loss": sum(x[0] for x in metrics) / len(metrics), "value_loss": sum(x[1] for x in metrics) / len(metrics), "gradient_norm": max(x[2] for x in metrics), "exact_kl": exact_kl, "clip_fraction": clip_fraction, "mean_action_shift": shift, "ratio_p95": float(torch.quantile(ratio, 0.95)), "ratio_p99": float(torch.quantile(ratio, 0.99)), "nan_inf": int(not all(math.isfinite(v) for row in metrics for v in row))}

    initial_path = write_checkpoint(0, actor, critic, optimizer, parent, runtime, {"phase": "INITIAL", "learning_rate": 1.5e-5})
    checkpoint_rows.append({"update": 0, "phase": "INITIAL", "path": str(initial_path.relative_to(OUT)).replace("\\", "/"), "sha256": sha(initial_path), "actor_hash": hobj(actor.checkpoint_state()), "critic_hash": hobj(critic.state_dict()), "optimizer_hash": hobj(optimizer.state_dict())})
    for update in range(1, args.updates + 1):
        phase_name, rear_speed = phase(update); accumulated = []; effective = 0; units = []
        while effective < 24576:
            cursor = runtime["collection_cursor"]; batch = cursor % 5; offset = OFFSETS[cursor % len(OFFSETS)]; train_mask = torch.tensor(masks[str(batch)]["train_mask"], dtype=torch.bool, device=env.device); train_ids = torch.nonzero(train_mask, as_tuple=False).flatten(); targets_a, categories, allocation, new_residual = physical_targets(train_ids, rear_speed, update, cursor, runtime["quota_residual"], False, env.device); targets_b, _, _, _ = physical_targets(train_ids, rear_speed, update, cursor, runtime["quota_residual"], True, env.device); policy_hash_before = hobj(actor.checkpoint_state()); pass_a = collect_pass(batch, offset, targets_a, train_mask, SEED + cursor * 2); pass_b = collect_pass(batch, offset, targets_b, train_mask, SEED + cursor * 2)
            if pass_a["initial_hash"] != pass_b["initial_hash"] or policy_hash_before != hobj(actor.checkpoint_state()): raise RuntimeError("EXP013_W2_P1_A7_R1_MASK_CONTRACT_IDENTITY_FAIL mirror")
            compact = compact_pair((pass_a, pass_b)); accumulated.append(compact); effective += compact["valid_count"]; units.append({"cursor": cursor, "batch": batch, "offset": offset, "allocation": allocation, "valid_samples": compact["valid_count"], "initial_hash": pass_a["initial_hash"], "policy_hash": policy_hash_before}); runtime["quota_residual"] = new_residual; runtime["collection_cursor"] += 1
        storage = {key: torch.cat([entry[key] for entry in accumulated]) if key != "valid_count" else sum(entry[key] for entry in accumulated) for key in accumulated[0]}; metrics = ppo_update(storage, update); runtime["ppo_interactions"] += storage["valid_count"]
        row = {"update": update, "phase": phase_name, "rear_speed": rear_speed if rear_speed is not None else "MIXED", "collection_units": len(units), "effective_valid_samples": storage["valid_count"], "ppo_interactions_cumulative": runtime["ppo_interactions"], "teacher_rollin_env_steps_cumulative": runtime["teacher_rollin_steps"], "prefix_warmup_env_steps_cumulative": runtime["prefix_warmup_steps"], "total_simulator_env_steps_cumulative": runtime["total_simulator_env_steps"], "collection_cursor": runtime["collection_cursor"], **metrics, "units": json.dumps(units, sort_keys=True)}; training_rows.append(row)
        if update == 1: dump("first_update_stability.json", {**metrics, "effective_valid_samples": storage["valid_count"], "status": "PASS" if metrics["exact_kl"] <= 0.20 and metrics["clip_fraction"] <= 0.50 and metrics["mean_action_shift"] <= 2.0 and metrics["gradient_norm"] <= 1e6 and metrics["value_loss"] <= 1e8 and metrics["nan_inf"] == 0 else "FAIL"})
        if metrics["nan_inf"] or metrics["exact_kl"] > 0.50: raise RuntimeError("EXP013_W2_P1_A7_R1_TRAINING_UNSTABLE numerical")
        if update in SAVE:
            path = write_checkpoint(update, actor, critic, optimizer, parent, runtime, {"phase": phase_name, "learning_rate": 1.5e-5, **metrics}); checkpoint_rows.append({"update": update, "phase": phase_name, "path": str(path.relative_to(OUT)).replace("\\", "/"), "sha256": sha(path), "actor_hash": hobj(actor.checkpoint_state()), "critic_hash": hobj(critic.state_dict()), "optimizer_hash": hobj(optimizer.state_dict())})
        with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(training_rows[0])); writer.writeheader(); writer.writerows(training_rows)
        dump("checkpoint_manifest.json", {"checkpoints": checkpoint_rows, "persistent_runs": 1})
        print(json.dumps({"update": update, "phase": phase_name, "samples": storage["valid_count"], "kl": metrics["exact_kl"], "clip": metrics["clip_fraction"], "cursor": runtime["collection_cursor"]}), flush=True)
    wrapped.close()

dump("a7_simulator_step_accounting.json", runtime)
dump("a7_full_batch_replay_identity.json", {"status": "PASS", "accepted_ids": "6144/6144 inherited M0 plus runtime accept-mask exact on every collection", "pool_semantic_sha256": POOL_SHA, "mask_hash": MASK_SHA, "snapshot_restore": False})
dump("early_guard.json", {"status": "PENDING_CLEAN_VALIDATION_EVALUATOR", "numerical_updates_1_10": "PASS"})
print(json.dumps({"status": "TRAINING_COMPLETE", "updates": args.updates, "runtime": runtime}, indent=2))
