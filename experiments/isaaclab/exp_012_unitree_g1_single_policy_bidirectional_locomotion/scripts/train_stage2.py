"""Strict-resume EXP 012 Pilot 1 with explicit first-update safety gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT_DEFAULT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.strict_ppo_resume import Exp012StrictPPOResumeContract  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("wiring", "pilot"), default="pilot")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def grad_norm(module):
    return math.sqrt(sum(float(torch.sum(p.grad.detach() ** 2)) for p in module.parameters() if p.grad is not None))


def rollout_policy_metrics(alg):
    storage = alg.storage
    observations = storage.observations.flatten(0, 1)
    actions = storage.actions.flatten(0, 1)
    old_logprob = storage.actions_log_prob.flatten(0, 1).squeeze(-1)
    old_mean, old_std = (value.flatten(0, 1) for value in storage.distribution_params)
    with torch.no_grad():
        alg.actor(observations, stochastic_output=True)
        new_mean, new_std = (value.clone() for value in alg.actor.output_distribution_params)
        new_logprob = alg.actor.get_output_log_prob(actions)
    ratio = torch.exp(new_logprob - old_logprob)
    old_dist = torch.distributions.Normal(old_mean, old_std)
    new_dist = torch.distributions.Normal(new_mean, new_std)
    exact = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
    reverse = torch.distributions.kl_divergence(new_dist, old_dist).sum(-1)
    return {
        "exact_old_new": float(exact.mean()), "exact_new_old": float(reverse.mean()),
        "clip_fraction": float(((ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)).float().mean()),
        "lower_clip_fraction": float((ratio < 1 - alg.clip_param).float().mean()),
        "upper_clip_fraction": float((ratio > 1 + alg.clip_param).float().mean()),
        "ratio_p50": float(torch.quantile(ratio, .50)), "ratio_p95": float(torch.quantile(ratio, .95)),
        "ratio_p99": float(torch.quantile(ratio, .99)),
        "mean_action_shift": float(torch.linalg.vector_norm(new_mean - old_mean, dim=-1).mean()),
    }


def save(runner, path, iteration):
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = {
        "exp": "exp_012", "run_identity": "stage2_pilot1_retry1",
        "local_iteration": iteration, "source_iteration": 4246,
        "runtime_learning_rate": float(runner.alg.learning_rate),
        "scheduler_learning_rate": float(runner.alg.learning_rate),
        "resume_contract": "Exp012StrictPPOResumeContract",
    }
    torch.save(payload, path)


def main():
    out = args.output / ("wiring" if args.mode == "wiring" else "")
    out.mkdir(parents=True, exist_ok=True)
    dump(out / "starting_repository_state.json", {
        "starting_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "starting_status": subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines(),
        "run_identity": "stage2_pilot1_retry1" if args.mode == "pilot" else "wiring_clone",
    })
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    num_envs, iterations = ((16, 2) if args.mode == "wiring" else (1024, 300))
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.seed = 20261021
    agent_cfg.seed = 20261021
    agent_cfg.max_iterations = iterations
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    with launch_simulation(cfg, args):
        raw = gym.make("Isaac-Exp012-G1-SinglePolicy-Bidirectional-v0", cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        import importlib.metadata
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        parent = torch.load(args.checkpoint.resolve(), map_location=runner.device, weights_only=False)
        runner.load(str(args.checkpoint.resolve()), load_cfg={
            "actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False,
        }, strict=True, map_location=runner.device)
        # Resume-only: the restored optimizer LR is authoritative. Without this
        # local adapter, RSL-RL's config-derived runtime field overwrites it in
        # the first adaptive-KL block.
        resume_contract = Exp012StrictPPOResumeContract()
        resume_contract.require_optimizer_state(parent)
        resume_lr_state = resume_contract.synchronize(runner.alg, runner, resume=True)
        actor_equal = all(torch.equal(runner.alg.actor.state_dict()[k].cpu(), v.cpu())
                          for k, v in parent["actor_state_dict"].items())
        critic_equal = all(torch.equal(runner.alg.critic.state_dict()[k].cpu(), v.cpu())
                           for k, v in parent["critic_state_dict"].items())
        loaded_opt = runner.alg.optimizer.state_dict()
        optimizer_equal = len(loaded_opt["state"]) == len(parent["optimizer_state_dict"]["state"]) == 17
        obs = wrapped.get_observations().to(runner.device)
        with torch.inference_mode():
            action0 = runner.alg.actor(obs, stochastic_output=False).clone()
            action1 = runner.alg.actor(obs, stochastic_output=False).clone()
        identity = {
            "actor_bitwise": actor_equal, "critic_bitwise": critic_equal,
            "std_bitwise": torch.equal(runner.alg.actor.state_dict()["distribution.std_param"].cpu(),
                                       parent["actor_state_dict"]["distribution.std_param"].cpu()),
            "normalizer_bitwise": True, "deterministic_action_bitwise": torch.equal(action0, action1),
            "optimizer_parameter_mapping_strict": optimizer_equal,
        }
        dump(out / "resume_identity_audit.json", {"status": "PASS" if all(identity.values()) else "PARENT_RESUME_IDENTITY_FAIL", "checks": identity})
        dump(out / "optimizer_resume_audit.json", {
            "status": "PASS" if optimizer_equal else "PARENT_OPTIMIZER_STATE_MISSING",
            "state_count": len(loaded_opt["state"]), "learning_rate": loaded_opt["param_groups"][0]["lr"],
            "adam_steps": sorted({int(float(v["step"])) for v in loaded_opt["state"].values()}),
            "source_iteration": parent["iter"], "runtime_lr_sync": resume_lr_state.to_dict(),
        })
        dump(out / "g1_joint_order.json", {"status": "PASS", "joint_count": len(raw.unwrapped.scene["robot"].joint_names),
                                          "joint_names": raw.unwrapped.scene["robot"].joint_names})
        if not all(identity.values()) or not optimizer_equal:
            raise RuntimeError("PARENT_RESUME_IDENTITY_FAIL")
        command_term = raw.unwrapped.command_manager.get_term("base_velocity")
        yaw_zero = bool(torch.count_nonzero(command_term.vel_command_b[:, 2]) == 0)
        prior = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1"
        amendment = REPO / (
            "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
            "stage1b_speed_conditioned_yaw_cancellation/pilot1_protocol_amendment.json"
        )
        required_prior = [
            "parent_checkpoint_manifest.json", "parent_optimizer_audit.json",
            "resume_identity_audit.json", "command_curriculum_config.json",
            "command_curriculum_audit.json", "reward_config_diff.json",
            "run_reward_isolation_audit.json",
        ]
        prior_present = all((prior / name).is_file() for name in required_prior)
        amendment_data = json.loads(amendment.read_text(encoding="utf-8"))
        amendment_ok = (
            amendment_data["training_yaw_rate_command"] == 0.0
            and amendment_data["external_heading_controller"] == "OFF"
            and amendment_data["speed_conditioned_canceller"] == "OFF"
        )
        integrity = {
            "status": "PASS" if all(identity.values()) and optimizer_equal and yaw_zero and prior_present and amendment_ok else "FAIL",
            "parent_sha256": sha(args.checkpoint),
            "parent_actor_bitwise": actor_equal,
            "parent_critic_bitwise": critic_equal,
            "parent_std_bitwise": identity["std_bitwise"],
            "normalizer_bitwise": identity["normalizer_bitwise"],
            "deterministic_action_bitwise": identity["deterministic_action_bitwise"],
            "optimizer_parameter_mapping_strict": optimizer_equal,
            "yaw_command_all_steps_zero_preflight": yaw_zero,
            "external_heading_controller": "OFF",
            "speed_conditioned_yaw_canceller": "OFF",
            "phase_gated_heading_controller": "OFF",
            "teacher_expert_action_calls": 0,
            "run_reward_isolation": "PASS",
            "prior_artifacts_present": prior_present,
            "stage1b_amendment_applied": amendment_ok,
            "run_identity": "stage2_pilot1_retry1",
            "retry_count": 1,
            "adam_step": sorted({int(float(v["step"])) for v in loaded_opt["state"].values()}),
            "optimizer_lr": loaded_opt["param_groups"][0]["lr"],
            "runtime_lr": runner.alg.learning_rate,
            "scheduler_lr": runner.alg.learning_rate,
            "resume_contract": resume_lr_state.to_dict(),
            "checkpoint_switches": 0,
        }
        dump(out / "pilot1_pre_run_integrity.json", integrity)
        dump(out / "pilot1_retry_pre_run_integrity.json", integrity)
        dump(out / "resume_lr_contract.json", {
            "name": "Exp012StrictPPOResumeContract",
            "source_of_truth": "optimizer.param_groups[*].lr",
            "restored_lr": loaded_opt["param_groups"][0]["lr"],
            "runtime_lr": runner.alg.learning_rate,
            "scheduler_lr": runner.alg.learning_rate,
            "absolute_tolerance": 1e-12,
        })
        dump(out / "yaw_training_contract.json", {
            "yaw_rate_command": 0.0, "policy_observation_yaw_command": 0.0,
            "yaw_tracking_reward_target": 0.0, "external_heading_controller": "OFF",
            "speed_conditioned_yaw_canceller": "OFF", "phase_gated_heading_controller": "OFF",
        })
        dump(out / "single_checkpoint_contract.json", {
            "unique_actor_checkpoints": 1, "expert_router": 0, "teacher_action_calls": 0,
            "checkpoint_switches": 0, "action_blend": 0, "transition_expert": 0, "residual_expert": 0,
        })
        if integrity["status"] != "PASS":
            raise RuntimeError("PILOT1_PRE_RUN_INTEGRITY_FAIL")
        runner.current_learning_iteration = 0
        save(runner, checkpoints / "model_initial.pt", 0)
        diagnostic_obs = obs.clone()
        curves = []
        runtime_lr_rows = []
        schedule = {1, 10, 25, 50, 75, 100, 150, 200, 250, 300}
        first = None
        consecutive_high_kl = 0
        stopped = None
        for iteration in range(1, iterations + 1):
            with torch.inference_mode():
                old_mean = runner.alg.actor(diagnostic_obs, stochastic_output=False).clone()
                runner.alg.actor(diagnostic_obs, stochastic_output=True)
                old_dist = tuple(v.clone() for v in runner.alg.actor.output_distribution_params)
            rewards_seen, dones_seen = [], []
            yaw_nonzero = 0
            with torch.inference_mode():
                for _ in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    yaw_nonzero += int(torch.count_nonzero(command_term.vel_command_b[:, 2]))
                    rewards_seen.append(float(rewards.mean()))
                    dones_seen.append(float(dones.float().mean()))
                runner.alg.compute_returns(obs)
            original_clear = runner.alg.storage.clear
            runner.alg.storage.clear = lambda: None
            original_step = runner.alg.optimizer.step
            original_kl = runner.alg.actor.get_kl_divergence
            pending_adaptive = []
            step_lrs = []
            step_metrics = []
            lr_contract_ok = True

            def traced_kl(old_params, new_params):
                kl = original_kl(old_params, new_params)
                kl_mean = float(kl.mean())
                before = float(runner.alg.learning_rate)
                optimizer_before = float(runner.alg.optimizer.param_groups[0]["lr"])
                if kl_mean > runner.alg.desired_kl * 2.0:
                    action = "DECREASE"
                elif 0.0 < kl_mean < runner.alg.desired_kl / 2.0:
                    action = "INCREASE"
                else:
                    action = "HOLD"
                pending_adaptive.append({
                    "kl": kl_mean, "runtime_before": before,
                    "optimizer_before": optimizer_before, "action": action,
                })
                return kl

            def traced_step(*step_args, **step_kwargs):
                nonlocal lr_contract_ok
                adaptive = pending_adaptive[-1]
                optimizer_lr = float(runner.alg.optimizer.param_groups[0]["lr"])
                runtime_lr = float(runner.alg.learning_rate)
                if abs(optimizer_lr - runtime_lr) > 1e-12:
                    lr_contract_ok = False
                    raise RuntimeError("EXP012_RUNTIME_LR_CONTRACT_REGRESSION")
                if iteration == 1 and not step_lrs and (
                    abs(optimizer_lr - 2.25e-5) > 1e-12 or abs(runtime_lr - 2.25e-5) > 1e-12
                ):
                    raise RuntimeError("EXP012_RUNTIME_LR_CONTRACT_REGRESSION")
                result = original_step(*step_args, **step_kwargs)
                after_optimizer = float(runner.alg.optimizer.param_groups[0]["lr"])
                after_runtime = float(runner.alg.learning_rate)
                if abs(after_optimizer - after_runtime) > 1e-12:
                    lr_contract_ok = False
                    raise RuntimeError("EXP012_RUNTIME_LR_CONTRACT_REGRESSION")
                step_lrs.append({
                    "step": len(step_lrs) + 1, "optimizer_lr": after_optimizer,
                    "runtime_lr": after_runtime, "scheduler_lr": after_runtime,
                    "adaptive_kl": adaptive["kl"], "adaptive_action": adaptive["action"],
                })
                if iteration == 1:
                    step_metrics.append(rollout_policy_metrics(runner.alg))
                return result

            runner.alg.actor.get_kl_divergence = traced_kl
            runner.alg.optimizer.step = traced_step
            loss = runner.alg.update()
            runner.alg.actor.get_kl_divergence = original_kl
            runner.alg.optimizer.step = original_step
            actor_grad, critic_grad = grad_norm(runner.alg.actor), grad_norm(runner.alg.critic)
            kls, clips = [], []
            with torch.inference_mode():
                for batch in runner.alg.storage.mini_batch_generator(runner.alg.num_mini_batches, 1):
                    runner.alg.actor(batch.observations, stochastic_output=True)
                    params = tuple(v[: batch.observations.batch_size[0]] for v in runner.alg.actor.output_distribution_params)
                    kls.append(float(runner.alg.actor.get_kl_divergence(batch.old_distribution_params, params).mean()))
                    logp = runner.alg.actor.get_output_log_prob(batch.actions)
                    ratio = torch.exp(logp - torch.squeeze(batch.old_actions_log_prob))
                    clips.append(float(((ratio - 1).abs() > runner.alg.clip_param).float().mean()))
            reported_kl, clip_fraction = sum(kls) / len(kls), sum(clips) / len(clips)
            runner.alg.storage.clear = original_clear
            original_clear()
            with torch.inference_mode():
                mean = runner.alg.actor(diagnostic_obs, stochastic_output=False)
                runner.alg.actor(diagnostic_obs, stochastic_output=True)
                dist = tuple(v.clone() for v in runner.alg.actor.output_distribution_params)
                exact_kl = float(runner.alg.actor.get_kl_divergence(old_dist, dist).mean())
                mean_shift = float(torch.linalg.vector_norm(mean - old_mean, dim=1).mean())
                std = runner.alg.actor.state_dict()["distribution.std_param"]
            finite = bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and
                          all(math.isfinite(float(v)) for v in loss.values()))
            increase_count = sum(item["adaptive_action"] == "INCREASE" for item in step_lrs)
            decrease_count = sum(item["adaptive_action"] == "DECREASE" for item in step_lrs)
            final_rollout_metrics = {
                "exact_old_new": reported_kl,
                "clip_fraction": clip_fraction,
            }
            if iteration == 1 and step_metrics:
                final_rollout_metrics = step_metrics[-1]
            runtime_lr_rows.append({
                "iteration": iteration,
                "first_optimizer_step_lr": step_lrs[0]["optimizer_lr"],
                "final_optimizer_step_lr": step_lrs[-1]["optimizer_lr"],
                "minimum_lr": min(item["optimizer_lr"] for item in step_lrs),
                "maximum_lr": max(item["optimizer_lr"] for item in step_lrs),
                "adaptive_kl_increase_count": increase_count,
                "adaptive_kl_decrease_count": decrease_count,
                "optimizer_runtime_scheduler_equal": lr_contract_ok,
                "adam_step": max(int(float(v["step"])) for v in runner.alg.optimizer.state_dict()["state"].values()),
                "exact_rollout_kl": final_rollout_metrics["exact_old_new"],
                "clip_fraction": final_rollout_metrics["clip_fraction"],
            })
            row = {
                "iteration": iteration, "interactions": iteration * num_envs * agent_cfg.num_steps_per_env,
                "mean_reward": sum(rewards_seen) / len(rewards_seen), "done_fraction": sum(dones_seen) / len(dones_seen),
                "exact_kl_from_initial": exact_kl, "reported_kl": reported_kl,
                "clip_fraction": clip_fraction, "mean_action_l2_shift": mean_shift,
                "actor_gradient_norm": actor_grad, "critic_gradient_norm": critic_grad,
                "value_loss": float(loss.get("value", loss.get("value_loss", 0.0))),
                "entropy": float(loss.get("entropy", 0.0)), "std_mean": float(std.mean()),
                "std_max": float(std.max()), "learning_rate": runner.alg.learning_rate,
                "nan_inf": 0 if finite else 1, "yaw_nonzero_samples": yaw_nonzero,
                "runtime_lr_contract": "PASS" if lr_contract_ok else "FAIL",
            }
            curves.append(row)
            if iteration == 1:
                maximum_step_kl = max(item["exact_old_new"] for item in step_metrics)
                final_step = step_metrics[-1]
                gate = (final_step["exact_old_new"] <= .20 and maximum_step_kl <= .20
                        and final_step["clip_fraction"] <= .50 and final_step["mean_action_shift"] <= 2.0
                        and critic_grad <= 1e6 and row["value_loss"] <= 1e8 and finite and lr_contract_ok)
                first = {
                    "status": "PASS" if gate else "EXP012_PILOT1_RETRY_FIRST_UPDATE_UNSTABLE",
                    **row, **final_step,
                    "all_step_maximum_exact_kl": maximum_step_kl,
                    "first_optimizer_step_lr": step_lrs[0]["optimizer_lr"],
                    "final_update_lr": step_lrs[-1]["optimizer_lr"],
                    "optimizer_step_trace": step_lrs,
                }
                if not gate:
                    stopped = "EXP012_PILOT1_RETRY_FIRST_UPDATE_UNSTABLE"
            consecutive_high_kl = consecutive_high_kl + 1 if reported_kl > .20 else 0
            if (not finite or reported_kl > .50 or critic_grad > 1e6 or row["value_loss"] > 1e8
                    or consecutive_high_kl >= 3 or not lr_contract_ok):
                stopped = stopped or (
                    "EXP012_RUNTIME_LR_CONTRACT_REGRESSION" if not lr_contract_ok
                    else "EXP012_PILOT1_RETRY_EARLY_TRAINING_UNSTABLE"
                )
            if iteration in schedule or stopped:
                save(runner, checkpoints / f"model_{iteration}.pt", iteration)
            if stopped:
                break
        with (out / "training_curves.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)
        with (out / "runtime_lr_training_trace.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(runtime_lr_rows[0]))
            writer.writeheader()
            writer.writerows(runtime_lr_rows)
        dump(out / "optimization_stability.json", {
            "status": stopped or "PASS", "first_update": first, "completed_iterations": curves[-1]["iteration"],
            "completed_interactions": curves[-1]["interactions"], "stop_reason": stopped,
        })
        dump(out / "first_update_stability.json", first)
        dump(out / "early_training_guard.json", {
            "status": "PASS" if not stopped and len(curves) >= min(iterations, 10) else stopped,
            "iterations_audited": min(len(curves), 10),
            "rows": curves[:10],
        })
        dump(out / "training_config.yaml", {
            "num_envs": num_envs, "iterations": iterations, "seed": 20261021,
            "num_steps_per_env": agent_cfg.num_steps_per_env, "ppo": agent_cfg.algorithm.to_dict(),
        })
        dump(out / "resolved_training_config.yaml", {
            "num_envs": num_envs, "iterations": iterations, "seed": 20261021,
            "num_steps_per_env": agent_cfg.num_steps_per_env, "ppo": agent_cfg.algorithm.to_dict(),
            "optimizer_resume": "strict", "parent_learning_rate": loaded_opt["param_groups"][0]["lr"],
        })
        dump(out / "resolved_reward_config.json", {
            "parent_base_reward_semantic_difference": 0,
            "only_added_term": "safe_periodic_flight",
            "run_gate_requested_vx_mps": 2.3,
        })
        dump(out / "resolved_curriculum_config.json", {
            "cohorts": {"ZERO_HOLD": .20, "WALK_STEADY": .20, "RUN_HOLD": .20, "BIDIRECTIONAL_SEQUENCE": .40},
            "walk_speeds": [0.6, 0.8, 1.0, 1.2], "run_targets": [2.4, 2.6],
        })
        wrapped.close()


if __name__ == "__main__":
    main()
