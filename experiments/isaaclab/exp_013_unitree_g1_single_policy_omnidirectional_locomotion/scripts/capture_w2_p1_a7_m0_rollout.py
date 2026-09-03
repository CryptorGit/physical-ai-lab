"""Capture one fresh-process, full-batch A7-M0 masked PPO rollout pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
TEACHER_PATH = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
PARENT_PATH = BASE / "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"

sys.path[:0] = [
    str(REPO / "experiments/isaaclab/exp_005_unitree_g1_flat_run/src"),
    str(REPO / "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion/src"),
    str(EXP / "src"),
]
import isaaclab_tasks  # noqa: F401,E402
import g1_omnidirectional.tasks  # noqa: F401,E402
from g1_omnidirectional.policy import FrozenGaitActor  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import (  # noqa: E402
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser()
parser.add_argument("--yaw", required=True, choices=("negative", "positive"))
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

N = 1024
ROLLIN = 150
T = 24
SEED = 20278501


def semantic(parts):
    h = hashlib.sha256()
    for key, value in sorted(parts.items()):
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def minimum_jerk(x):
    x = x.clamp(0.0, 1.0)
    return 10 * x**3 - 15 * x**4 + 6 * x**5


def critic(state, obs):
    x = F.elu(F.linear(obs, state["mlp.0.weight"], state["mlp.0.bias"]))
    x = F.elu(F.linear(x, state["mlp.2.weight"], state["mlp.2.bias"]))
    x = F.elu(F.linear(x, state["mlp.4.weight"], state["mlp.4.bias"]))
    return F.linear(x, state["mlp.6.weight"], state["mlp.6.bias"]).squeeze(-1)


cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp013-G1-DirectionalBaseline-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = N
cfg.episode_length_s = 12.0
cfg.seed = SEED
cfg.observations.policy.enable_corruption = False
if args.device:
    cfg.sim.device = agent_cfg.device = args.device

masks = json.loads((OUT / "a7_environment_masks.json").read_text())["batches"]["0"]
train_mask_cpu = torch.tensor(masks["train_mask"], dtype=torch.bool)
parent = torch.load(PARENT_PATH, map_location="cpu", weights_only=False)

with launch_simulation(cfg, args):
    wrapped = RslRlVecEnvWrapper(
        gym.make("Isaac-Exp013-G1-DirectionalBaseline-v0", cfg=cfg),
        clip_actions=agent_cfg.clip_actions,
    )
    env = wrapped.unwrapped
    robot = env.scene["robot"]
    sensor = env.scene["contact_forces"]
    feet = sensor.find_bodies(".*_ankle_roll_link")[0]
    command = env.command_manager.get_term("base_velocity")
    command.external_override_enabled = True
    teacher = FrozenGaitActor(TEACHER_PATH).to(env.device).eval()
    policy = FrozenGaitActor(PARENT_PATH).to(env.device).eval()
    critic_state = {k: v.to(env.device) for k, v in parent["critic_state_dict"].items()}
    std = parent["actor_state_dict"]["distribution.log_std_walk"].exp().to(env.device)
    gait = torch.zeros(N, device=env.device)
    train_mask = train_mask_cpu.to(env.device)
    env_ids = torch.arange(N, device=env.device)

    env.reset(env_ids=env_ids)
    command.external_override.zero_()
    command._update_command()
    obs = wrapped.get_observations().to(env.device)
    for _ in range(ROLLIN):
        with torch.inference_mode():
            teacher_action = teacher(obs["policy"], gait)
        obs, _, _, _ = wrapped.step(teacher_action)
        obs = obs.to(env.device)

    contact_force = sensor.data.net_forces_w_history[:, -1, feet, :]
    initial_parts = {
        "root_pos_w": robot.data.root_pos_w,
        "root_quat_w": robot.data.root_quat_w,
        "root_lin_vel_w": robot.data.root_lin_vel_w,
        "root_ang_vel_w": robot.data.root_ang_vel_w,
        "joint_pos": robot.data.joint_pos,
        "joint_vel": robot.data.joint_vel,
        "policy_observation": obs["policy"],
        "contact_force": contact_force,
    }
    initial_hash = semantic(initial_parts)

    generator = torch.Generator(device=env.device).manual_seed(20278611)
    rows = {key: [] for key in ("observation", "action", "reward", "done", "old_logp", "old_value", "valid")}
    alive = train_mask.clone()
    yaw_target = -0.3 if args.yaw == "negative" else 0.3
    for step in range(T):
        alpha = minimum_jerk(torch.tensor(step * env.step_dt / 1.5, device=env.device))
        command.external_override.zero_()
        command.external_override[train_mask, 0] = -0.3 * alpha
        actor_yaw = yaw_target * alpha
        if actor_yaw > 0:
            actor_yaw = actor_yaw * 1.5
        command.external_override[train_mask, 2] = actor_yaw
        command._update_command()
        obs = wrapped.get_observations().to(env.device)
        full_obs = torch.cat((obs["policy"], gait[:, None]), dim=1)
        with torch.inference_mode():
            mean = policy(obs["policy"], gait)
            noise = torch.randn(mean.shape, generator=generator, device=env.device)
            sampled = mean + noise * std
            housekeeping = teacher(obs["policy"], gait)
            action = torch.where(train_mask[:, None], sampled, housekeeping)
            value = critic(critic_state, full_obs)
            logp = (-0.5 * (((sampled - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)
        rows["observation"].append(full_obs.cpu())
        rows["action"].append(sampled.cpu())
        rows["old_logp"].append(logp.cpu())
        rows["old_value"].append(value.cpu())
        rows["valid"].append(alive.cpu())
        obs, reward, done, _ = wrapped.step(action)
        obs = obs.to(env.device)
        rows["reward"].append(reward.cpu())
        rows["done"].append(done.cpu())
        alive &= ~done.bool()

    last_obs = torch.cat((obs["policy"], gait[:, None]), dim=1)
    with torch.inference_mode():
        last_value = critic(critic_state, last_obs).cpu()
    payload = {key: torch.stack(values) for key, values in rows.items()}
    payload.update(
        {
            "last_value": last_value,
            "state_id": torch.arange(N),
            "train_mask": train_mask_cpu,
            "initial_full_batch_semantic_hash": initial_hash,
            "parent_sha256": hashlib.sha256(PARENT_PATH.read_bytes()).hexdigest(),
            "yaw_pass": args.yaw,
            "teacher_rollin_steps": ROLLIN,
            "snapshot_restore": False,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(payload, OUT / f"raw_masked_rollout_{args.yaw}.pt")
    wrapped.close()
    print(json.dumps({"yaw": args.yaw, "initial_hash": initial_hash, "valid_samples": int(payload["valid"].sum())}, indent=2))
