"""EXP012 Stage 2B: strict-resume LR causal one-update diagnosis."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2b_runtime_lr_resume_fix"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
ROLLOUT = OUT / "diagnostic_rollout.pt"
U1A_STATES = OUT / "_u1a_step_states.pt"
PATCHED_CHECKPOINT = OUT / "_patched_one_update_checkpoint.pt"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import g1_single_policy.tasks  # noqa: E402,F401
from g1_single_policy.strict_ppo_resume import Exp012StrictPPOResumeContract  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("collect", "replay"), required=True)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(value):
    array = value.detach().contiguous().cpu().numpy()
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    h.update(array.tobytes())
    return h.hexdigest()


def object_hash(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(tensor_hash(value).encode())
    return h.hexdigest()


def quantiles(x):
    q = torch.tensor([.01, .05, .50, .95, .99], device=x.device)
    vals = torch.quantile(x.float(), q).cpu().tolist()
    return dict(zip(("p1", "p5", "p50", "p95", "p99"), vals))


def grad_norm(module):
    return math.sqrt(sum(float(torch.sum(p.grad.detach() ** 2)) for p in module.parameters() if p.grad is not None))


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
    ratio = torch.exp(new_logp - old_logp)
    old_dist = torch.distributions.Normal(old_mean, old_std)
    new_dist = torch.distributions.Normal(new_mean, new_std)
    exact = torch.distributions.kl_divergence(old_dist, new_dist).sum(-1)
    reverse = torch.distributions.kl_divergence(new_dist, old_dist).sum(-1)
    clipped = (ratio < 1 - alg.clip_param) | (ratio > 1 + alg.clip_param)
    surrogate = torch.maximum(
        -advantages * ratio,
        -advantages * ratio.clamp(1 - alg.clip_param, 1 + alg.clip_param),
    )
    value_clipped = old_values + (new_values - old_values).clamp(-alg.clip_param, alg.clip_param)
    value_loss = torch.maximum((new_values - returns).pow(2), (value_clipped - returns).pow(2)).mean()
    return {
        "exact": exact, "reverse": reverse, "ratio": ratio, "clipped": clipped,
        "lower": ratio < 1 - alg.clip_param, "upper": ratio > 1 + alg.clip_param,
        "old_mean": old_mean, "new_mean": new_mean, "old_std": old_std, "new_std": new_std,
        "advantages": advantages, "surrogate": surrogate, "value_loss": value_loss, "entropy": entropy,
    }


def metric_row(path, step, alg, storage, actor_grad, critic_grad, surrogate_loss, value_loss, adaptive):
    m = policy_metrics(alg, storage)
    return {
        "path": path, "optimizer_step": step, "epoch": (step - 1) // alg.num_mini_batches,
        "minibatch": (step - 1) % alg.num_mini_batches,
        "lr_before": adaptive["optimizer_lr_before"], "runtime_lr_before": adaptive["runtime_lr_before"],
        "kl_input": adaptive["kl_input"], "adaptive_action": adaptive["action"],
        "lr_after": alg.optimizer.param_groups[0]["lr"], "runtime_lr_after": alg.learning_rate,
        "exact_kl_old_new": float(m["exact"].mean()), "exact_kl_new_old": float(m["reverse"].mean()),
        "clip_fraction": float(m["clipped"].float().mean()),
        "lower_clip": float(m["lower"].float().mean()), "upper_clip": float(m["upper"].float().mean()),
        **{f"ratio_{k}": v for k, v in quantiles(m["ratio"]).items()},
        "mean_action_shift": float(torch.linalg.vector_norm(m["new_mean"] - m["old_mean"], dim=-1).mean()),
        "std_shift": float(torch.linalg.vector_norm(m["new_std"] - m["old_std"], dim=-1).mean()),
        "actor_gradient": actor_grad, "critic_gradient": critic_grad,
        "value_loss": value_loss, "surrogate_loss": surrogate_loss, "entropy": float(m["entropy"]),
        "nan_inf": int(not all(torch.isfinite(x).all() for x in (m["ratio"], m["exact"], m["new_mean"], m["new_std"]))),
    }


def snapshot(alg):
    state = alg.optimizer.state_dict()
    return {
        "actor": {k: v.detach().cpu().clone() for k, v in alg.actor.state_dict().items()},
        "critic": {k: v.detach().cpu().clone() for k, v in alg.critic.state_dict().items()},
        "optimizer": copy.deepcopy(state),
        "runtime_lr": float(alg.learning_rate),
    }


def load_rollout(storage, data, device):
    storage.observations = data["observation"].to(device)
    storage.actions.copy_(data["action"].to(device))
    storage.actions_log_prob.copy_(data["old_logprob"].to(device))
    storage.distribution_params = tuple(x.to(device) for x in (data["old_mean"], data["old_std"]))
    storage.values.copy_(data["old_value"].to(device))
    storage.returns.copy_(data["returns"].to(device))
    storage.advantages.copy_(data["normalized_advantage"].to(device))
    storage.rewards.copy_(data["reward"].to(device))
    storage.dones.copy_(data["dones"].to(device))
    storage.step = storage.num_transitions_per_env


def set_rng(data, device):
    random.setstate(data["python_rng"])
    np.random.set_state(data["numpy_rng"])
    torch.set_rng_state(data["torch_cpu_rng"])
    if str(device).startswith("cuda"):
        torch.cuda.set_rng_state(data["torch_cuda_rng"], device=device)


def run_update(path, runner, data, patched):
    alg = runner.alg
    parent = torch.load(PARENT, map_location=runner.device, weights_only=False)
    alg.load(parent, {"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False}, strict=True)
    load_rollout(alg.storage, data, runner.device)
    set_rng(data, runner.device)
    if patched:
        Exp012StrictPPOResumeContract().synchronize(alg, runner, resume=True)
    else:
        alg.learning_rate = 0.001

    initial = {
        "actor_hash": state_hash(alg.actor.state_dict()), "critic_hash": state_hash(alg.critic.state_dict()),
        "std_hash": tensor_hash(alg.actor.state_dict()["distribution.std_param"]),
        "optimizer_hash": object_hash(alg.optimizer.state_dict()),
        "optimizer_lr": alg.optimizer.param_groups[0]["lr"], "runtime_lr": alg.learning_rate,
    }
    trace, lr_events, snapshots, pending = [], [], [], []
    lr_events.append({"event_index": 0, "event_name": "optimizer_state_load_complete", "epoch": -1, "minibatch": -1,
                      "adam_step": 85000, "optimizer_lr": alg.optimizer.param_groups[0]["lr"],
                      "runtime_self_learning_rate": alg.learning_rate, "scheduler_current_lr": alg.learning_rate,
                      "config_default_lr": 0.001, "kl_input": "", "adaptive_action": ""})
    if patched:
        lr_events.append({"event_index": 1, "event_name": "runtime_lr_sync_complete", "epoch": -1, "minibatch": -1,
                          "adam_step": 85000, "optimizer_lr": alg.optimizer.param_groups[0]["lr"],
                          "runtime_self_learning_rate": alg.learning_rate, "scheduler_current_lr": alg.learning_rate,
                          "config_default_lr": 0.001, "kl_input": "", "adaptive_action": "SYNC_FROM_OPTIMIZER"})

    original_kl = alg.actor.get_kl_divergence
    original_step = alg.optimizer.step
    original_clear = alg.storage.clear
    alg.storage.clear = lambda: None
    step_number = 0

    def traced_kl(old_params, new_params):
        kl = original_kl(old_params, new_params)
        kl_mean = float(kl.mean())
        before = float(alg.learning_rate)
        opt_before = float(alg.optimizer.param_groups[0]["lr"])
        if kl_mean > alg.desired_kl * 2:
            action = "DIVIDE_1P5"
        elif 0 < kl_mean < alg.desired_kl / 2:
            action = "MULTIPLY_1P5"
        else:
            action = "HOLD"
        pending.append({"kl_input": kl_mean, "runtime_lr_before": before, "optimizer_lr_before": opt_before, "action": action})
        return kl

    def traced_step(*a, **kw):
        nonlocal step_number
        actor_grad, critic_grad = grad_norm(alg.actor), grad_norm(alg.critic)
        result = original_step(*a, **kw)
        step_number += 1
        adaptive = pending[-1]
        row = metric_row(path, step_number, alg, alg.storage, actor_grad, critic_grad, float("nan"), float("nan"), adaptive)
        trace.append(row)
        snapshots.append(snapshot(alg))
        lr_events.append({
            "event_index": len(lr_events), "event_name": "optimizer_step_after_adaptive",
            "epoch": row["epoch"], "minibatch": row["minibatch"], "adam_step": 85000 + step_number,
            "optimizer_lr": row["lr_after"], "runtime_self_learning_rate": row["runtime_lr_after"],
            "scheduler_current_lr": row["runtime_lr_after"], "config_default_lr": 0.001,
            "kl_input": row["kl_input"], "adaptive_action": row["adaptive_action"],
        })
        return result

    alg.actor.get_kl_divergence = traced_kl
    alg.optimizer.step = traced_step
    loss = alg.update()
    alg.actor.get_kl_divergence = original_kl
    alg.optimizer.step = original_step
    alg.storage.clear = original_clear
    # RSL returns aggregate losses; attach the aggregate to each trace row.
    for row in trace:
        row["surrogate_loss"] = loss["surrogate"]
        row["value_loss"] = loss["value"]
        row["entropy"] = loss["entropy"]
    return {
        "path": path, "initial": initial, "trace": trace, "lr_events": lr_events,
        "snapshots": snapshots, "final": snapshot(alg), "loss": loss,
    }


def create_runner(cfg, agent_cfg, wrapped):
    import importlib.metadata
    resolved = handle_deprecated_rsl_rl_cfg(agent_cfg, importlib.metadata.version("rsl-rl-lib"))
    return OnPolicyRunner(wrapped, resolved.to_dict(), log_dir=None, device=resolved.device), resolved


def collect_mode(cfg, agent_cfg, raw, wrapped):
    runner, agent_cfg = create_runner(cfg, agent_cfg, wrapped)
    parent = torch.load(PARENT, map_location=runner.device, weights_only=False)
    runner.load(str(PARENT), load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
                strict=True, map_location=runner.device)
    obs = wrapped.get_observations().to(runner.device)
    with torch.inference_mode():
        runner.alg.actor(obs, stochastic_output=False).clone()
        runner.alg.actor(obs, stochastic_output=False).clone()
    command_term = raw.unwrapped.command_manager.get_term("base_velocity")
    cohorts, commands, segments, reward_components = [], [], [], []
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
            reward_components.append(step_reward.detach().cpu().clone())
    runner.alg.compute_returns(obs)
    storage = runner.alg.storage
    data = {
        "observation": storage.observations.to("cpu"), "action": storage.actions.cpu(),
        "old_mean": storage.distribution_params[0].cpu(), "old_std": storage.distribution_params[1].cpu(),
        "old_logprob": storage.actions_log_prob.cpu(), "old_value": storage.values.cpu(),
        "returns": storage.returns.cpu(), "raw_advantage": (storage.returns - storage.values).cpu(),
        "normalized_advantage": storage.advantages.cpu(), "reward": storage.rewards.cpu(),
        "dones": storage.dones.cpu(), "cohort": torch.stack(cohorts), "command": torch.stack(commands),
        "segment": torch.stack(segments),
        "reward_components": torch.stack(reward_components) if reward_components else None,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(device=runner.device) if str(runner.device).startswith("cuda") else None,
        "optimizer_state": copy.deepcopy(runner.alg.optimizer.state_dict()),
        "scheduler_state": {"runtime_learning_rate": runner.alg.learning_rate, "schedule": runner.alg.schedule},
    }
    # Canonical minibatch permutation without consuming the saved update RNG.
    if str(runner.device).startswith("cuda"):
        before = torch.cuda.get_rng_state(device=runner.device)
        permutation = torch.randperm(24576, device=runner.device)
        torch.cuda.set_rng_state(before, device=runner.device)
    else:
        before = torch.get_rng_state()
        permutation = torch.randperm(24576)
        torch.set_rng_state(before)
    data["minibatch_permutation"] = permutation.cpu()
    torch.save(data, ROLLOUT)
    hashes = {
        "rollout_file_sha256": file_hash(ROLLOUT),
        "observation": object_hash(data["observation"]), "action": tensor_hash(data["action"]),
        "old_distribution": tensor_hash(torch.cat((data["old_mean"], data["old_std"]), dim=-1)),
        "old_logprob": tensor_hash(data["old_logprob"]), "advantages": tensor_hash(data["normalized_advantage"]),
        "minibatch_permutation": tensor_hash(data["minibatch_permutation"]),
        "initial_optimizer": object_hash(data["optimizer_state"]),
    }
    dump("diagnostic_rollout_hashes.json", hashes)
    dump("diagnostic_rollout_manifest.json", {
        "status": "DIAGNOSTIC_ROLLOUT_RECOLLECTED_FOR_RESUME_FIX", "sample_count": 24576,
        "num_envs": 1024, "rollout_steps": 24, "seed": 20261021,
        "yaw_nonzero_samples": int(torch.count_nonzero(data["command"][..., 2])),
        "controllers": {"external_heading": False, "yaw_canceller": False, "phase_gated": False},
        "cohort_counts": {name: int((data["cohort"] == idx).sum()) for idx, name in enumerate(
            ("ZERO_HOLD", "WALK_STEADY", "RUN_HOLD", "BIDIRECTIONAL_SEQUENCE"))},
        "artifact": "diagnostic_rollout.pt (local, not tracked)",
    })
    dump("diagnostic_minibatch_order.json", {
        "generator": "one torch.randperm(24576) reused over 5 epochs",
        "permutation_sha256": hashes["minibatch_permutation"],
        "length": 24576, "first_64_indices": permutation[:64].cpu().tolist(),
        "full_order_stored_in": "diagnostic_rollout.pt",
    })

    u0 = run_update("U0_UNPATCHED", runner, data, patched=False)
    u1 = run_update("U1_A_PATCHED", runner, data, patched=True)
    torch.save(u1["snapshots"], U1A_STATES)
    payload = copy.deepcopy(u1["final"])
    payload["exp012_resume_state"] = {"runtime_learning_rate": u1["final"]["runtime_lr"], "scheduler_learning_rate": u1["final"]["runtime_lr"]}
    torch.save(payload, PATCHED_CHECKPOINT)
    write_csv("patched_vs_unpatched_update_trace.csv", u0["trace"] + u1["trace"])
    write_csv("learning_rate_state_trace.csv", [
        {"path": path, **row} for path, result in (("U0", u0), ("U1", u1)) for row in result["lr_events"]
    ])
    dump("_u0_u1_summary.json", {
        "u0": {"initial": u0["initial"], "final_trace": u0["trace"][-1], "max_kl": max(r["exact_kl_old_new"] for r in u0["trace"])},
        "u1": {"initial": u1["initial"], "final_trace": u1["trace"][-1], "max_kl": max(r["exact_kl_old_new"] for r in u1["trace"])},
        "rollout_hashes": hashes,
    })


def replay_mode(cfg, agent_cfg, wrapped):
    runner, _ = create_runner(cfg, agent_cfg, wrapped)
    data = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    u1b = run_update("U1_B_INDEPENDENT", runner, data, patched=True)
    reference = torch.load(U1A_STATES, map_location="cpu", weights_only=False)
    rows, all_equal = [], True
    for index, (a, b) in enumerate(zip(reference, u1b["snapshots"]), 1):
        actor_max = max(float((a["actor"][k] - b["actor"][k]).abs().max()) for k in a["actor"])
        critic_max = max(float((a["critic"][k] - b["critic"][k]).abs().max()) for k in a["critic"])
        std_max = float((a["actor"]["distribution.std_param"] - b["actor"]["distribution.std_param"]).abs().max())
        moment_max = 0.0
        for key, state_a in a["optimizer"]["state"].items():
            state_b = b["optimizer"]["state"][key]
            for field in ("exp_avg", "exp_avg_sq"):
                moment_max = max(moment_max, float((state_a[field].cpu() - state_b[field].cpu()).abs().max()))
        steps_a = sorted({int(float(x["step"])) for x in a["optimizer"]["state"].values()})
        steps_b = sorted({int(float(x["step"])) for x in b["optimizer"]["state"].values()})
        exact = actor_max == critic_max == std_max == moment_max == 0.0 and steps_a == steps_b and a["runtime_lr"] == b["runtime_lr"]
        all_equal &= exact
        rows.append({"optimizer_step": index, "actor_max_abs": actor_max, "critic_max_abs": critic_max,
                     "std_max_abs": std_max, "optimizer_moment_max_abs": moment_max,
                     "adam_step_a": steps_a[0], "adam_step_b": steps_b[0],
                     "lr_a": a["runtime_lr"], "lr_b": b["runtime_lr"], "bitwise_equal": exact})
    write_csv("patched_shadow_equivalence.csv", rows)
    dump("patched_shadow_equivalence.json", {
        "status": "PASS" if all_equal else "PPO_PATCHED_SHADOW_EQUIVALENCE_FAIL",
        "rollout_sha256": file_hash(ROLLOUT),
        "minibatch_permutation_sha256": tensor_hash(data["minibatch_permutation"]),
        "initial_actor_hash": u1b["initial"]["actor_hash"], "initial_critic_hash": u1b["initial"]["critic_hash"],
        "initial_std_hash": u1b["initial"]["std_hash"], "initial_optimizer_hash": u1b["initial"]["optimizer_hash"],
        "optimizer_steps_compared": len(rows), "all_steps_bitwise": all_equal,
        "maximums": {key: max(r[key] for r in rows) for key in
                     ("actor_max_abs", "critic_max_abs", "std_max_abs", "optimizer_moment_max_abs")},
    })
    # Reload the diagnostic checkpoint; do not execute another optimizer step.
    patched = torch.load(PATCHED_CHECKPOINT, map_location=runner.device, weights_only=False)
    runner.alg.actor.load_state_dict(patched["actor"])
    runner.alg.critic.load_state_dict(patched["critic"])
    runner.alg.optimizer.load_state_dict(patched["optimizer"])
    runner.alg.learning_rate = 0.001
    state = Exp012StrictPPOResumeContract().synchronize(runner.alg, runner, resume=True)
    adam_steps = sorted({int(float(x["step"])) for x in runner.alg.optimizer.state_dict()["state"].values()})
    dump("post_update_resume_integrity.json", {
        "status": "PASS" if adam_steps == [85020] and state.restored_lr == patched["runtime_lr"] else "FAIL",
        "actor_hash_before_save": state_hash(patched["actor"]), "actor_hash_after_reload": state_hash(runner.alg.actor.state_dict()),
        "critic_hash_before_save": state_hash(patched["critic"]), "critic_hash_after_reload": state_hash(runner.alg.critic.state_dict()),
        "std_equal": torch.equal(patched["actor"]["distribution.std_param"].cpu(),
                                 runner.alg.actor.state_dict()["distribution.std_param"].cpu()),
        "optimizer_hash_before_save": object_hash(patched["optimizer"]),
        "optimizer_hash_after_reload": object_hash(runner.alg.optimizer.state_dict()),
        "adam_step": adam_steps, "optimizer_lr": runner.alg.optimizer.param_groups[0]["lr"],
        "runtime_lr": runner.alg.learning_rate, "scheduler_lr": runner.alg.learning_rate,
        "next_optimizer_step_executed": False, "normalizer": "DISABLED_NOT_PRESENT_IN_PARENT_CONTRACT",
    })
    dump("_u1b_summary.json", {"initial": u1b["initial"], "trace": u1b["trace"], "final": {
        "runtime_lr": u1b["final"]["runtime_lr"], "optimizer_hash": object_hash(u1b["final"]["optimizer"])}})


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
        if args.mode == "collect":
            collect_mode(cfg, agent_cfg, raw, wrapped)
        else:
            replay_mode(cfg, agent_cfg, wrapped)
        wrapped.close()


if __name__ == "__main__":
    main()
