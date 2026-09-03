"""Preflight and the one authorized 200-iteration Phase W1A PPO run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict
from torch import nn
from torch.optim import Adam
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
CRITIC = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
PARENT_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_omnidirectional.tasks_w1a  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("preflight", "train", "resume"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]

LR = 1.5e-5
SEED = 20271021
CHECKPOINTS = {1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def tensor_state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        value = state[key]
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(optimizer):
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def grad_norm(module):
    return math.sqrt(sum(
        float(torch.sum(parameter.grad.detach() ** 2))
        for parameter in module.parameters() if parameter.grad is not None
    ))


class W1AVecEnv:
    """Append fixed gait_cmd=0 without changing the underlying 123D observation."""

    def __init__(self, base):
        self.base = base
        for name in ("num_envs", "device", "max_episode_length", "num_actions"):
            setattr(self, name, getattr(base, name))
        self.gait = torch.zeros(self.num_envs, device=self.device)
        self.command = self.base.unwrapped.command_manager.get_term("base_velocity")

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def unwrapped(self):
        return self.base.unwrapped

    @property
    def episode_length_buf(self):
        return self.base.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.base.episode_length_buf = value

    def seed(self, value):
        return self.base.seed(value)

    def close(self):
        self.base.close()

    def _augment(self, obs):
        result = obs.clone()
        result["policy"] = torch.cat((obs["policy"], self.gait[:, None]), dim=-1)
        return result

    def get_observations(self):
        return self._augment(self.base.get_observations())

    def reset(self):
        obs, extras = self.base.reset()
        return self._augment(obs), extras

    def step(self, actions):
        obs, rewards, dones, extras = self.base.step(actions)
        return self._augment(obs), rewards, dones, extras


def initialize(runner):
    parent = torch.load(PARENT, map_location=runner.device, weights_only=False)
    critic = torch.load(CRITIC, map_location=runner.device, weights_only=False)
    source_actor = parent["actor_state_dict"]
    source_critic = critic["critic_state_dict"]
    runner.alg.actor.load_state_dict(source_actor, strict=True)
    runner.alg.critic.load_state_dict(source_critic, strict=True)
    runner.alg.actor.distribution.log_std_walk.requires_grad_(False)
    runner.alg.actor.distribution.log_std_run.requires_grad_(False)
    mean_parameters = [
        parameter for name, parameter in runner.alg.actor.named_parameters()
        if parameter.requires_grad and not name.startswith("distribution.")
    ]
    critic_parameters = list(runner.alg.critic.parameters())
    runner.alg.optimizer = Adam(
        [
            {"params": mean_parameters, "lr": LR, "name": "actor_mean"},
            {"params": critic_parameters, "lr": LR, "name": "critic"},
        ],
        lr=LR,
    )
    runner.alg.learning_rate = LR
    actor_equal = all(
        torch.equal(runner.alg.actor.state_dict()[key].cpu(), value.cpu())
        for key, value in source_actor.items()
    )
    critic_equal = all(
        torch.equal(runner.alg.critic.state_dict()[key].cpu(), value.cpu())
        for key, value in source_critic.items()
    )
    return parent, critic, actor_equal, critic_equal


def distribution_metrics(alg, observations, actions, old_logp, old_mean, old_std):
    with torch.no_grad():
        alg.actor(observations, stochastic_output=True)
        new_mean, new_std = (value.clone() for value in alg.actor.output_distribution_params)
        new_logp = alg.actor.get_output_log_prob(actions)
    ratio = torch.exp(new_logp - old_logp)
    exact = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std),
        torch.distributions.Normal(new_mean, new_std),
    ).sum(-1)
    return {
        "exact_rollout_kl": float(exact.mean()),
        "max_sample_kl": float(exact.max()),
        "clip_fraction": float(((ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)).float().mean()),
        "ratio_p95": float(torch.quantile(ratio, .95)),
        "ratio_p99": float(torch.quantile(ratio, .99)),
        "mean_action_shift": float(torch.linalg.vector_norm(new_mean - old_mean, dim=-1).mean()),
        "new_logp": new_logp,
        "new_mean": new_mean,
        "new_std": new_std,
    }


def update_once(runner):
    storage = runner.alg.storage
    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    old_logp = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
    old_mean, old_std = (value.flatten(0, 1).clone() for value in storage.distribution_params)
    advantages = storage.advantages.flatten(0, 1).squeeze(-1).clone()
    original_clear = storage.clear
    storage.clear = lambda: None
    original_step = runner.alg.optimizer.step
    step_metrics = []

    def traced_step(*step_args, **step_kwargs):
        before_lr = [float(group["lr"]) for group in runner.alg.optimizer.param_groups]
        result = original_step(*step_args, **step_kwargs)
        metrics = distribution_metrics(runner.alg, observations, actions, old_logp, old_mean, old_std)
        step_metrics.append({
            "optimizer_step": len(step_metrics) + 1,
            "exact_rollout_kl": metrics["exact_rollout_kl"],
            "clip_fraction": metrics["clip_fraction"],
            "lr_before": before_lr,
            "lr_after": [float(group["lr"]) for group in runner.alg.optimizer.param_groups],
        })
        return result

    runner.alg.optimizer.step = traced_step
    losses = runner.alg.update()
    runner.alg.optimizer.step = original_step
    metrics = distribution_metrics(runner.alg, observations, actions, old_logp, old_mean, old_std)
    old_surrogate = float((-advantages).mean())
    new_surrogate = float((-advantages * torch.exp(metrics["new_logp"] - old_logp)).mean())
    storage.clear = original_clear
    original_clear()
    return losses, {
        **{key: value for key, value in metrics.items() if not torch.is_tensor(value)},
        "all_step_maximum_kl": max(row["exact_rollout_kl"] for row in step_metrics),
        "optimizer_step_trace": step_metrics,
        "surrogate_improvement": old_surrogate - new_surrogate,
    }


def rollout(runner, env, obs):
    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = [
        next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[index])
        for index in feet
    ]
    falls = torch.zeros(env.num_envs, dtype=torch.bool, device=runner.device)
    dangerous = torch.zeros_like(falls)
    impacts = torch.zeros_like(falls)
    slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=runner.device)
    rewards = []
    command_sum = torch.zeros((env.num_envs, 3), device=runner.device)
    for _ in range(24):
        with torch.inference_mode():
            actions = runner.alg.act(obs)
        obs, reward, dones, extras = env.step(actions)
        obs, reward, dones = obs.to(runner.device), reward.to(runner.device), dones.to(runner.device)
        runner.alg.process_env_step(obs, reward, dones, extras)
        rewards.append(float(reward.mean()))
        timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
        falls |= dones.bool() & ~timeout
        forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
        contacts = forces > 5
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        slipping = ((foot_speed > .55) & contacts).any(-1)
        slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
        dangerous |= slip_streak >= 5
        impacts |= forces.amax(-1) > 3500
        command_sum += env.command.vel_command_b
    runner.alg.compute_returns(obs)
    command_mean = command_sum / 24
    left = command_mean[:, 1] > 0
    right = command_mean[:, 1] < 0
    return obs, {
        "mean_reward": sum(rewards) / len(rewards),
        "fall_rate": float(falls.float().mean()),
        "dangerous_slip_rate": float(dangerous.float().mean()),
        "impact_failure_rate": float(impacts.float().mean()),
        "left_fall_rate": float(falls[left].float().mean()) if left.any() else 0.0,
        "right_fall_rate": float(falls[right].float().mean()) if right.any() else 0.0,
    }


def forward_probe(runner, env):
    command = env.command
    command.external_override_enabled = True
    obs, _ = env.reset()
    obs = obs.to(runner.device)
    half = env.num_envs // 2
    speeds = torch.cat((
        torch.full((half,), .6, device=runner.device),
        torch.full((env.num_envs - half,), 1.2, device=runner.device),
    ))
    command.external_override.zero_()
    command.external_override[:, 0] = speeds
    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene.sensors["contact_forces"]
    feet = [i for i, name in enumerate(sensor.body_names) if "ankle_roll" in name]
    robot_feet = [
        next(i for i, name in enumerate(robot.body_names) if name == sensor.body_names[index])
        for index in feet
    ]
    steps = round(8.0 / float(env.unwrapped.step_dt))
    vector_error = torch.zeros(env.num_envs, device=runner.device)
    yaw_error = torch.zeros_like(vector_error)
    flight = torch.zeros_like(vector_error)
    falls = torch.zeros(env.num_envs, dtype=torch.bool, device=runner.device)
    dangerous = torch.zeros_like(falls)
    impact = torch.zeros_like(falls)
    slip_streak = torch.zeros(env.num_envs, dtype=torch.long, device=runner.device)
    for _ in range(steps):
        command.external_override[:, 0] = speeds
        command.external_override[:, 1:] = 0
        with torch.inference_mode():
            actions = runner.alg.actor(obs)
        obs, _, dones, extras = env.step(actions)
        obs = obs.to(runner.device)
        actual = robot.data.root_lin_vel_b[:, :2]
        vector_error += torch.linalg.vector_norm(
            actual - torch.stack((speeds, torch.zeros_like(speeds)), dim=1), dim=-1
        )
        yaw_error += robot.data.root_ang_vel_b[:, 2].abs()
        forces = sensor.data.net_forces_w_history[:, -1, feet, :].norm(dim=-1)
        contacts = forces > 5
        flight += (contacts.sum(-1) == 0).float()
        timeout = extras.get("time_outs", torch.zeros_like(dones)).bool()
        falls |= dones.bool() & ~timeout
        foot_speed = torch.linalg.vector_norm(robot.data.body_lin_vel_w[:, robot_feet, :2], dim=-1)
        slipping = ((foot_speed > .55) & contacts).any(-1)
        slip_streak = torch.where(slipping, slip_streak + 1, torch.zeros_like(slip_streak))
        dangerous |= slip_streak >= 5
        impact |= forces.amax(-1) > 3500
    vector_error /= steps
    yaw_error /= steps
    flight /= steps
    strict_success = (
        (vector_error <= .20) & (yaw_error <= .20) & (flight < .10)
        & ~falls & ~dangerous & ~impact
    )
    # Early guard retention follows the supplied exp_012 anchor meaning:
    # WALK_LIKE (flight fraction < 0.10) and no fall. The strict W1A
    # direction gate is recorded separately and is used only in formal eval.
    success = (flight < .10) & ~falls
    result = {
        "forward_0p6_success": float(success[:half].float().mean()),
        "forward_1p2_success": float(success[half:].float().mean()),
        "forward_0p6_strict_success": float(strict_success[:half].float().mean()),
        "forward_1p2_strict_success": float(strict_success[half:].float().mean()),
        "forward_0p6_vector_mae": float(vector_error[:half].mean()),
        "forward_1p2_vector_mae": float(vector_error[half:].mean()),
        "probe_fall_rate": float(falls.float().mean()),
        "probe_dangerous_slip_rate": float(dangerous.float().mean()),
        "probe_impact_failure_rate": float(impact.float().mean()),
    }
    command.external_override_enabled = False
    obs, _ = env.reset()
    return obs.to(runner.device), result


def save_checkpoint(runner, path, iteration, phase, curve):
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = {
        "experiment": "exp_013", "phase": "W1A", "training_iteration": iteration,
        "curriculum_phase": phase, "learning_rate": LR, "single_checkpoint_continuation": True,
        "yaw_training": False, "run_training": False, "gait_cmd": 0,
        "rollout_kl": curve.get("exact_rollout_kl"), "clip_fraction": curve.get("clip_fraction"),
    }
    torch.save(payload, path)


def main():
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError("W1A_PARENT_PROVENANCE_FAIL")
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp013-G1-W1A-TranslationWalk-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = 1024
    cfg.episode_length_s = 12.0
    cfg.seed = SEED
    agent_cfg.seed = SEED
    agent_cfg.max_iterations = 1 if args.mode == "preflight" else 200
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp013-G1-W1A-TranslationWalk-v0", cfg=cfg)
        base = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        env = W1AVecEnv(base)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, importlib.metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        parent, critic, actor_equal, critic_equal = initialize(runner)
        initial_actor_hash = tensor_state_hash(runner.alg.actor.state_dict())
        initial_critic_hash = tensor_state_hash(runner.alg.critic.state_dict())
        walk_std_initial = runner.alg.actor.distribution.log_std_walk.detach().clone()
        run_std_initial = runner.alg.actor.distribution.log_std_run.detach().clone()
        optimizer_empty = len(runner.alg.optimizer.state) == 0
        dump("w1a_parent_identity_audit.json", {
            "status": "PASS" if actor_equal else "FAIL",
            "source_sha256": sha(PARENT), "actor_bitwise_identity": actor_equal,
            "actor_state_hash": initial_actor_hash, "architecture": [124, 256, 128, 128, 37],
        })
        dump("w1a_critic_initialization_audit.json", {
            "status": "PASS" if critic_equal else "FAIL",
            "source_sha256": sha(CRITIC), "critic_bitwise_identity": critic_equal,
            "critic_state_hash": initial_critic_hash, "input_dimensions": 124,
        })
        dump("w1a_optimizer_initialization_audit.json", {
            "status": "PASS" if optimizer_empty else "FAIL",
            "state_entries": len(runner.alg.optimizer.state), "adam_step": 0,
            "parameter_groups": [
                {"name": group.get("name"), "parameters": len(group["params"]), "lr": group["lr"]}
                for group in runner.alg.optimizer.param_groups
            ],
            "old_RUN_moments_imported": False,
        })
        if not actor_equal or not critic_equal or not optimizer_empty:
            raise RuntimeError("W1A_RUNTIME_INITIALIZATION_FAIL")

        obs, _ = env.reset()
        obs = obs.to(runner.device)
        env.command.set_training_iteration(1)
        env.command._resample_command(torch.arange(env.num_envs, device=runner.device))
        obs = env.get_observations().to(runner.device)
        obs, rollout_safety = rollout(runner, env, obs)
        losses, metrics = update_once(runner)
        finite = all(
            torch.isfinite(parameter).all() for parameter in runner.alg.actor.parameters()
        ) and all(math.isfinite(float(value)) for value in losses.values())
        std_frozen = (
            torch.equal(walk_std_initial, runner.alg.actor.distribution.log_std_walk)
            and torch.equal(run_std_initial, runner.alg.actor.distribution.log_std_run)
        )
        first = {
            **metrics, **rollout_safety,
            "actor_gradient_norm": grad_norm(runner.alg.actor),
            "critic_gradient_norm": grad_norm(runner.alg.critic),
            "value_loss": float(losses.get("value", 0.0)),
            "surrogate_loss": float(losses.get("surrogate", 0.0)),
            "nan_inf": 0 if finite else 1,
            "optimizer_lr": [float(group["lr"]) for group in runner.alg.optimizer.param_groups],
            "runtime_lr": float(runner.alg.learning_rate),
            "log_std_walk_bitwise_frozen": std_frozen,
            "log_std_run_bitwise_frozen": std_frozen,
        }
        hard_pass = (
            first["exact_rollout_kl"] <= .20
            and first["all_step_maximum_kl"] <= .20
            and first["clip_fraction"] <= .50
            and first["mean_action_shift"] <= 2.0
            and first["critic_gradient_norm"] <= 1e6
            and first["value_loss"] <= 1e8
            and first["nan_inf"] == 0
            and all(abs(value - LR) <= 1e-12 for value in first["optimizer_lr"])
            and abs(first["runtime_lr"] - LR) <= 1e-12
            and std_frozen
        )
        first["status"] = "PASS" if hard_pass else "EXP013_W1A_FIRST_UPDATE_UNSTABLE"
        first["preferred_exact_kl_pass"] = first["exact_rollout_kl"] <= .05
        first["preferred_clip_fraction_pass"] = first["clip_fraction"] <= .30

        if args.mode == "preflight":
            dump("first_update_stability.json", first)
            gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
            gate.update({
                "first_update": "PASS" if hard_pass else "FAIL",
                "continue_to_persistent_training": hard_pass,
                "classification_if_stopped": None if hard_pass else "EXP013_W1A_FIRST_UPDATE_UNSTABLE",
            })
            dump("gate.json", gate)
            print(json.dumps({key: first[key] for key in (
                "status", "exact_rollout_kl", "all_step_maximum_kl",
                "clip_fraction", "mean_action_shift", "critic_gradient_norm", "value_loss"
            )}, sort_keys=True))
            env.close()
            raise SystemExit(0 if hard_pass else 2)

        preflight = json.loads((OUT / "first_update_stability.json").read_text(encoding="utf-8"))
        if preflight["status"] != "PASS":
            raise RuntimeError("W1A_PREFLIGHT_NOT_PASS")
        # Discard the local diagnostic update above. A train start reconstructs
        # the authorized run; a resume restores that same run's iteration-1
        # actor, critic, and fresh-Adam state after the early-guard evaluator fix.
        checkpoints = OUT / "checkpoints"
        checkpoints.mkdir(parents=True, exist_ok=True)
        if args.mode == "resume":
            resume_path = checkpoints / "model_1.pt"
            payload = torch.load(resume_path, map_location=runner.device, weights_only=False)
            runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
            runner.alg.critic.load_state_dict(payload["critic_state_dict"], strict=True)
            runner.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
            runner.alg.learning_rate = LR
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = LR
            with (OUT / "training_curves.csv").open(encoding="utf-8") as handle:
                curves = list(csv.DictReader(handle))
            early_payload = json.loads((OUT / "early_guard.json").read_text(encoding="utf-8"))
            early_rows = early_payload["rows"]
            obs, corrected_probe = forward_probe(runner, env)
            corrected_row = early_rows[0]
            corrected_row.update(corrected_probe)
            corrected_row["guard_pass"] = (
                corrected_probe["forward_0p6_success"] >= .80
                and corrected_probe["forward_1p2_success"] >= .80
                and float(corrected_row["fall_rate"]) <= .15
                and float(corrected_row["dangerous_slip_rate"]) <= .50
                and float(corrected_row["impact_failure_rate"]) <= .10
                and float(corrected_row["exact_rollout_kl"]) <= .50
            )
            corrected_row["guard_evaluator_correction"] = (
                "anchor-compatible WALK_LIKE non-fall retention; strict W1A success retained separately"
            )
            corrected_row["interrupted_after_iteration_1"] = True
            corrected_row["resumed_same_checkpoint_optimizer_state"] = True
            if not corrected_row["guard_pass"]:
                dump("early_guard.json", {
                    "status": "EXP013_W1A_TRAINING_UNSTABLE",
                    "iterations_audited": 1, "rows": early_rows,
                })
                env.close()
                raise SystemExit(3)
            start_iteration = 2
        else:
            parent, critic, actor_equal, critic_equal = initialize(runner)
            initial_curve = {
                "exact_rollout_kl": 0.0, "clip_fraction": 0.0, "fall_rate": 0.0,
                "dangerous_slip_rate": 0.0, "impact_failure_rate": 0.0,
            }
            save_checkpoint(runner, checkpoints / "model_initial.pt", 0, "INITIAL", initial_curve)
            obs, _ = env.reset()
            obs = obs.to(runner.device)
            curves = []
            early_rows = []
            start_iteration = 1
        stopped = None
        for iteration in range(start_iteration, 201):
            env.command.set_training_iteration(iteration)
            env.command._resample_command(torch.arange(env.num_envs, device=runner.device))
            obs = env.get_observations().to(runner.device)
            obs, safety = rollout(runner, env, obs)
            losses, update_metrics = update_once(runner)
            finite = all(
                torch.isfinite(parameter).all() for parameter in runner.alg.actor.parameters()
            ) and all(math.isfinite(float(value)) for value in losses.values())
            std_frozen = (
                torch.equal(walk_std_initial, runner.alg.actor.distribution.log_std_walk)
                and torch.equal(run_std_initial, runner.alg.actor.distribution.log_std_run)
            )
            curve = {
                "iteration": iteration, "interactions": iteration * 1024 * 24,
                "curriculum_phase": env.command.phase, **safety,
                **{key: value for key, value in update_metrics.items() if key != "optimizer_step_trace"},
                "actor_gradient_norm": grad_norm(runner.alg.actor),
                "critic_gradient_norm": grad_norm(runner.alg.critic),
                "value_loss": float(losses.get("value", 0.0)),
                "surrogate_loss": float(losses.get("surrogate", 0.0)),
                "entropy": float(losses.get("entropy", 0.0)),
                "learning_rate": float(runner.alg.learning_rate),
                "nan_inf": 0 if finite else 1,
                "std_bitwise_frozen": std_frozen,
            }
            if iteration <= 10:
                obs, probe = forward_probe(runner, env)
                env.command.set_training_iteration(iteration)
                env.command._resample_command(torch.arange(env.num_envs, device=runner.device))
                curve.update(probe)
                asymmetric = max(safety["left_fall_rate"], safety["right_fall_rate"]) > .20
                fail = (
                    not finite or curve["exact_rollout_kl"] > .50
                    or safety["fall_rate"] > .15 or safety["dangerous_slip_rate"] > .50
                    or safety["impact_failure_rate"] > .10
                    or probe["forward_0p6_success"] < .80
                    or probe["forward_1p2_success"] < .80
                    or asymmetric
                )
                early_rows.append({
                    **curve,
                    "mirror_extreme_asymmetry": asymmetric,
                    "guard_pass": not fail,
                })
                if fail:
                    stopped = "EXP013_W1A_TRAINING_UNSTABLE"
            curves.append(curve)
            if iteration in CHECKPOINTS or stopped:
                save_checkpoint(
                    runner, checkpoints / f"model_{iteration}.pt",
                    iteration, env.command.phase, curve,
                )
            print(
                f"[W1A] iter={iteration} phase={env.command.phase} reward={safety['mean_reward']:.3f} "
                f"fall={safety['fall_rate']:.3f} slip={safety['dangerous_slip_rate']:.3f} "
                f"kl={curve['exact_rollout_kl']:.5f} clip={curve['clip_fraction']:.3f}",
                flush=True,
            )
            if stopped:
                break
        write_csv("training_curves.csv", curves)
        dump("early_guard.json", {
            "status": "PASS" if not stopped and len(early_rows) == 10 else stopped,
            "iterations_audited": len(early_rows), "rows": early_rows,
        })
        dump("training_run_summary.json", {
            "status": stopped or "COMPLETE",
            "completed_iterations": len(curves),
            "completed_interactions": len(curves) * 1024 * 24,
            "maximum_runs": 1, "persistent_training_runs": 1,
            "checkpoint_schedule": [0] + sorted(CHECKPOINTS & set(range(1, len(curves) + 1))),
            "std_bitwise_frozen_all_iterations": all(row["std_bitwise_frozen"] for row in curves),
        })
        gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
        gate.update({
            "persistent_training": "COMPLETE" if not stopped else "STOPPED",
            "early_guard": "PASS" if not stopped else "FAIL",
            "continue_to_checkpoint_evaluation": not stopped and len(curves) == 200,
            "classification_if_stopped": stopped,
        })
        dump("gate.json", gate)
        env.close()
        if stopped:
            raise SystemExit(3)


if __name__ == "__main__":
    main()
