"""EXP 012 Stage 2A: one-rollout, one-update PPO ratio/KL diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2a_first_update_ratio_clipping_diagnosis"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
OFFICIAL = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run/checkpoints/model_1.pt"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(*values):
    h = hashlib.sha256()
    for value in values:
        array = value.detach().contiguous().cpu().numpy()
        h.update(str(array.dtype).encode())
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        h.update(array.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def quantiles(x):
    q = torch.tensor([.001, .01, .05, .25, .50, .75, .95, .99, .999], device=x.device)
    values = torch.quantile(x.float(), q).cpu().tolist()
    return dict(zip(("p0p1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99p9"), values))


def flatten_obs(storage):
    return storage.observations.flatten(0, 1)


def policy_metrics(alg, storage):
    obs = flatten_obs(storage)
    actions = storage.actions.flatten(0, 1)
    old_logp = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
    old_mean, old_std = (p.flatten(0, 1) for p in storage.distribution_params)
    advantages = storage.advantages.flatten(0, 1).squeeze(-1)
    returns = storage.returns.flatten(0, 1).squeeze(-1)
    old_values = storage.values.flatten(0, 1).squeeze(-1)
    with torch.no_grad():
        alg.actor(obs, stochastic_output=True)
        new_mean, new_std = (x.clone() for x in alg.actor.output_distribution_params)
        new_logp = alg.actor.get_output_log_prob(actions)
        new_values = alg.critic(obs).squeeze(-1)
        entropy = alg.actor.output_entropy.mean()
    per_dim_old = torch.distributions.Normal(old_mean, old_std).log_prob(actions)
    per_dim_new = torch.distributions.Normal(new_mean, new_std).log_prob(actions)
    per_dim_log_ratio = per_dim_new - per_dim_old
    log_ratio = new_logp - old_logp
    ratio = torch.exp(log_ratio)
    exact_old_new_dim = torch.distributions.kl_divergence(
        torch.distributions.Normal(old_mean, old_std), torch.distributions.Normal(new_mean, new_std)
    )
    exact_new_old_dim = torch.distributions.kl_divergence(
        torch.distributions.Normal(new_mean, new_std), torch.distributions.Normal(old_mean, old_std)
    )
    mean_dim = 0.5 * (old_mean - new_mean).pow(2) / new_std.pow(2)
    std_dim = torch.log(new_std / old_std) + old_std.pow(2) / (2 * new_std.pow(2)) - 0.5
    clipped = (ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * ratio.clamp(1 - alg.clip_param, 1 + alg.clip_param)
    surrogate_loss = torch.maximum(surrogate, surrogate_clipped).mean()
    value_clipped = old_values + (new_values - old_values).clamp(-alg.clip_param, alg.clip_param)
    value_loss = torch.maximum((new_values - returns).pow(2), (value_clipped - returns).pow(2)).mean()
    return {
        "old_logp": old_logp, "new_logp": new_logp, "log_ratio": log_ratio, "ratio": ratio,
        "clipped": clipped, "lower": ratio < 1 - alg.clip_param, "upper": ratio > 1 + alg.clip_param,
        "old_mean": old_mean, "old_std": old_std, "new_mean": new_mean, "new_std": new_std,
        "per_dim_log_ratio": per_dim_log_ratio, "exact_old_new_dim": exact_old_new_dim,
        "exact_new_old_dim": exact_new_old_dim, "mean_dim": mean_dim, "std_dim": std_dim,
        "advantages": advantages, "surrogate": torch.maximum(surrogate, surrogate_clipped),
        "surrogate_loss": surrogate_loss, "value_loss": value_loss, "entropy": entropy,
        "sample_forward": (old_logp - new_logp).mean(),
        "schulman": ((ratio - 1) - log_ratio).mean(),
        "squared_log": 0.5 * log_ratio.pow(2).mean(),
    }


def metric_row(step, alg, storage):
    m = policy_metrics(alg, storage)
    exact = m["exact_old_new_dim"].sum(-1)
    reverse = m["exact_new_old_dim"].sum(-1)
    return {
        "optimizer_step": step, "epoch": (step - 1) // alg.num_mini_batches,
        "mini_batch": (step - 1) % alg.num_mini_batches,
        "learning_rate": alg.learning_rate,
        "surrogate_loss_full_rollout": float(m["surrogate_loss"]),
        "value_loss_full_rollout": float(m["value_loss"]),
        "entropy": float(m["entropy"]),
        "exact_kl_old_new_joint": float(exact.mean()),
        "exact_kl_new_old_joint": float(reverse.mean()),
        "exact_kl_per_dimension_mean": float(m["exact_old_new_dim"].mean()),
        "exact_kl_per_dimension_max": float(m["exact_old_new_dim"].mean(0).max()),
        "sample_forward_kl": float(m["sample_forward"]),
        "schulman_approx_kl": float(m["schulman"]),
        "squared_log_approx_kl": float(m["squared_log"]),
        "joint_clip_fraction": float(m["clipped"].float().mean()),
        "lower_clip_fraction": float(m["lower"].float().mean()),
        "upper_clip_fraction": float(m["upper"].float().mean()),
        "ratio_mean": float(m["ratio"].mean()), "ratio_std": float(m["ratio"].std()),
        "mean_action_shift": float(torch.linalg.vector_norm(m["new_mean"] - m["old_mean"], dim=-1).mean()),
        "std_shift_l2": float(torch.linalg.vector_norm(m["new_std"] - m["old_std"], dim=-1).mean()),
        "mean_kl_contribution": float(m["mean_dim"].sum(-1).mean()),
        "std_kl_contribution": float(m["std_dim"].sum(-1).mean()),
        **{f"ratio_{k}": v for k, v in quantiles(m["ratio"]).items()},
        **{f"log_ratio_{k}": v for k, v in quantiles(m["log_ratio"]).items()},
    }


def grad_norm(module):
    return math.sqrt(sum(float(torch.sum(p.grad.detach() ** 2)) for p in module.parameters() if p.grad is not None))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg, agent_cfg = resolve_task_config("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 1024
    cfg.seed = 20261021
    agent_cfg.seed = 20261021
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        parent_payload = torch.load(PARENT, map_location=runner.device, weights_only=False)
        runner.load(str(PARENT), load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
                    strict=True, map_location=runner.device)
        obs = wrapped.get_observations().to(runner.device)
        # Match Pilot-1 pre-update identity calls.
        with torch.inference_mode():
            runner.alg.actor(obs, stochastic_output=False).clone()
            runner.alg.actor(obs, stochastic_output=False).clone()
        command_term = raw.unwrapped.command_manager.get_term("base_velocity")
        cohorts, commands, segments, rewards_components = [], [], [], []
        episode_ids = torch.zeros((agent_cfg.num_steps_per_env, 1024), dtype=torch.int64)
        for _ in range(agent_cfg.num_steps_per_env):
            with torch.inference_mode():
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                runner.alg.process_env_step(obs, rewards, dones, extras)
            cohorts.append(command_term.cohort.detach().cpu().clone())
            commands.append(command_term.vel_command_b.detach().cpu().clone())
            segments.append(command_term.segment_index.detach().cpu().clone())
            step_reward = getattr(raw.unwrapped.reward_manager, "_step_reward", None)
            if step_reward is not None:
                rewards_components.append(step_reward.detach().cpu().clone())
        runner.alg.compute_returns(obs)
        storage = runner.alg.storage
        raw_advantage = (storage.returns - storage.values).detach().cpu()
        immutable = {
            "observation": storage.observations.to("cpu"),
            "action": storage.actions.detach().cpu(),
            "old_logprob": storage.actions_log_prob.detach().cpu(),
            "old_mean": storage.distribution_params[0].detach().cpu(),
            "old_std": storage.distribution_params[1].detach().cpu(),
            "old_value": storage.values.detach().cpu(),
            "returns": storage.returns.detach().cpu(),
            "raw_advantage": raw_advantage,
            "normalized_advantage": storage.advantages.detach().cpu(),
            "reward": storage.rewards.detach().cpu(),
            "dones": storage.dones.detach().cpu(),
            "cohort": torch.stack(cohorts),
            "command": torch.stack(commands),
            "segment": torch.stack(segments),
            "episode_id": episode_ids,
            "reward_components": torch.stack(rewards_components) if rewards_components else None,
        }
        torch.save(immutable, OUT / "_immutable_rollout.pt")
        hashes = {}
        for key, value in immutable.items():
            if value is None:
                hashes[key] = "NOT_AVAILABLE"
            elif hasattr(value, "keys") and not isinstance(value, torch.Tensor):
                hashes[key] = tensor_hash(*[value[k] for k in sorted(value.keys())])
            else:
                hashes[key] = tensor_hash(value)
        dump("immutable_rollout_hashes.json", hashes)
        dump("immutable_rollout_manifest.json", {
            "status": "DIAGNOSTIC_ROLLOUT_RECOLLECTED",
            "sample_count": int(storage.num_envs * storage.num_transitions_per_env),
            "num_envs": storage.num_envs, "timesteps": storage.num_transitions_per_env,
            "yaw_nonzero_samples": int(torch.count_nonzero(immutable["command"][..., 2])),
            "cohort_counts": {
                name: int((immutable["cohort"] == index).sum())
                for index, name in enumerate(("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE"))
            },
            "raw_artifact": "_immutable_rollout.pt (local, not tracked)",
        })

        old_action = storage.actions.flatten(0, 1)
        old_mean = storage.distribution_params[0].flatten(0, 1)
        old_std = storage.distribution_params[1].flatten(0, 1)
        stored_lp = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
        independent_lp = torch.distributions.Normal(old_mean, old_std).log_prob(old_action).sum(-1)
        with torch.no_grad():
            runner.alg.actor(flatten_obs(storage), stochastic_output=True)
            api_lp = runner.alg.actor.get_output_log_prob(old_action)
        dump("old_logprob_reconstruction.json", {
            "status": "PASS" if float((stored_lp - independent_lp).abs().max()) <= 1e-6 else "PPO_OLD_LOGPROB_PROVENANCE_MISMATCH",
            "shape": list(stored_lp.shape), "action_dimensions": 37, "reduction": "sum",
            "storage_vs_independent_max_abs": float((stored_lp - independent_lp).abs().max()),
            "storage_vs_policy_api_max_abs": float((stored_lp - api_lp).abs().max()),
            "finite_fraction": float(torch.isfinite(stored_lp).float().mean()),
        })
        dump("action_logprob_provenance.json", {
            "sample_for_logprob": "unclipped Gaussian action stored by PPO.act",
            "storage_action": "same unclipped action",
            "environment_action": "same action passed to wrapper; action manager applies target scaling/clipping downstream",
            "logprob_action": "storage action before environment-side target processing",
            "provenance_mismatch": False,
        })

        # Preserve storage and instrument the single official-order update.
        rng_before_update = torch.cuda.get_rng_state() if str(runner.device).startswith("cuda") else torch.get_rng_state()
        original_clear = storage.clear
        storage.clear = lambda: None
        trace = []
        original_step = runner.alg.optimizer.step
        step_index = 0

        def traced_step(*a, **kw):
            nonlocal step_index
            actor_grad = grad_norm(runner.alg.actor)
            critic_grad = grad_norm(runner.alg.critic)
            result = original_step(*a, **kw)
            step_index += 1
            row = metric_row(step_index, runner.alg, storage)
            row.update({
                "actor_gradient_norm": actor_grad, "critic_gradient_norm": critic_grad,
                "actor_hash": state_hash(runner.alg.actor.state_dict()),
                "critic_hash": state_hash(runner.alg.critic.state_dict()),
                "std_hash": tensor_hash(runner.alg.actor.state_dict()["distribution.std_param"]),
            })
            trace.append(row)
            return result

        runner.alg.optimizer.step = traced_step
        loss = runner.alg.update()
        runner.alg.optimizer.step = original_step
        storage.clear = original_clear
        final_metrics = policy_metrics(runner.alg, storage)
        official = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
        shadow_actor = {k: v.detach().cpu() for k, v in runner.alg.actor.state_dict().items()}
        shadow_critic = {k: v.detach().cpu() for k, v in runner.alg.critic.state_dict().items()}
        actor_max = max(float((shadow_actor[k] - official["actor_state_dict"][k]).abs().max()) for k in shadow_actor)
        critic_max = max(float((shadow_critic[k] - official["critic_state_dict"][k]).abs().max()) for k in shadow_critic)
        opt_shadow = runner.alg.optimizer.state_dict()
        official_steps = sorted({int(float(x["step"])) for x in official["optimizer_state_dict"]["state"].values()})
        shadow_steps = sorted({int(float(x["step"])) for x in opt_shadow["state"].values()})
        equivalence_pass = actor_max == 0.0 and critic_max == 0.0 and shadow_steps == official_steps
        dump("official_shadow_equivalence.json", {
            "status": "PASS" if equivalence_pass else "PPO_SHADOW_REPLAY_MISMATCH",
            "actor_bitwise": actor_max == 0.0, "actor_max_abs": actor_max,
            "critic_bitwise": critic_max == 0.0, "critic_max_abs": critic_max,
            "std_bitwise": torch.equal(shadow_actor["distribution.std_param"], official["actor_state_dict"]["distribution.std_param"]),
            "adam_steps_shadow": shadow_steps, "adam_steps_official": official_steps,
            "learning_rate_shadow": opt_shadow["param_groups"][0]["lr"],
            "learning_rate_official": official["optimizer_state_dict"]["param_groups"][0]["lr"],
            "optimizer_exact_tensor_comparison": "state steps/LR checked; parameter moments implied by deterministic bitwise replay",
        })
        write_csv("shadow_update_trace.csv", trace)
        dump("shadow_update_manifest.json", {
            "optimizer_steps": len(trace), "epochs": runner.alg.num_learning_epochs,
            "mini_batches": runner.alg.num_mini_batches, "final_loss": loss,
            "official_order": True, "diagnostic_clone_only": True,
        })
        write_csv("ratio_evolution_by_optimizer_step.csv", [{
            k: v for k, v in row.items() if k in (
                "optimizer_step", "epoch", "mini_batch", "joint_clip_fraction", "lower_clip_fraction",
                "upper_clip_fraction", "ratio_mean", "ratio_std", "mean_action_shift", "std_shift_l2"
            )
        } for row in trace])
        write_csv("ratio_quantiles_by_optimizer_step.csv", [{
            k: v for k, v in row.items() if k.startswith("ratio_") or k.startswith("log_ratio_")
            or k in ("optimizer_step", "epoch", "mini_batch")
        } for row in trace])
        write_csv("kl_evolution_by_optimizer_step.csv", [{
            k: v for k, v in row.items() if "kl" in k or k in ("optimizer_step", "epoch", "mini_batch")
        } for row in trace])

        # Persist compact tensors for offline finalization.
        diagnostic = {
            "trace": trace,
            "final": {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in final_metrics.items()},
            "cohort": immutable["cohort"].flatten(),
            "command": immutable["command"].reshape(-1, 3),
            "segment": immutable["segment"].flatten(),
            "raw_advantage": immutable["raw_advantage"].flatten(),
            "normalized_advantage": immutable["normalized_advantage"].flatten(),
            "reward": immutable["reward"].flatten(),
            "dones": immutable["dones"].flatten(),
            "actor_names": list(runner.alg.actor.state_dict()),
            "joint_names": raw.unwrapped.scene["robot"].joint_names,
            "rng_before_update": rng_before_update.cpu(),
            "equivalence_pass": equivalence_pass,
        }
        torch.save(diagnostic, OUT / "_diagnostic_compact.pt")
        wrapped.close()


if __name__ == "__main__":
    main()
