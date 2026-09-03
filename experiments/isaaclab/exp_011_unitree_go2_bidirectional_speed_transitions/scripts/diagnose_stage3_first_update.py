"""Recapture one update-free rollout and diagnose the Stage 2 PPO update on clones."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis"
STAGE2 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

HELPERS_ONLY = os.environ.get("EXP011_STAGE3_HELPERS_ONLY") == "1"
if not HELPERS_ONLY:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--unstable-checkpoint", type=Path, default=STAGE2 / "checkpoints/model_1_unstable.pt")
    parser.add_argument("--output", type=Path, default=OUT)
    from isaaclab.app import AppLauncher  # noqa: E402

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    sys.argv = [sys.argv[0]]

    from rsl_rl.runners import OnPolicyRunner  # noqa: E402

    sys.path.insert(0, str(EXP / "src"))
    import isaaclab_tasks  # noqa: E402,F401
    import go2_bidirectional.stage2_tasks  # noqa: E402,F401
    from go2_bidirectional.stage2_tasks.command import COHORT_NAMES  # noqa: E402
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
    from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402
else:
    COHORT_NAMES = ("ZERO_HOLD", "STEADY_SPEED", "ACCELERATION", "DECELERATION")

JOINTS = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("status,reason\nNOT_RUN,no rows\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(x: torch.Tensor) -> dict:
    x = x.detach().float().flatten().cpu()
    q = torch.quantile(x, torch.tensor([0.01, 0.05, 0.5, 0.9, 0.95, 0.99]))
    return {
        "mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
        "p01": float(q[0]), "p05": float(q[1]), "p50": float(q[2]),
        "p90": float(q[3]), "p95": float(q[4]), "p99": float(q[5]),
        "max": float(x.max()), "max_abs": float(x.abs().max()),
    }


def norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return math.sqrt(total)


def distribution(actor, observations):
    actor(observations, stochastic_output=True)
    mean, std = actor.output_distribution_params
    return mean, std, actor.output_entropy


def exact_components(old_mean, old_std, new_mean, new_std):
    mean = (old_mean - new_mean).square() / (2.0 * new_std.square())
    std = torch.log(new_std / old_std) + old_std.square() / (2.0 * new_std.square()) - 0.5
    return mean, std, mean + std


def flatten_storage(storage, raw_advantage, timeouts, metadata):
    observations = storage.observations.flatten(0, 1).cpu()
    payload = {
        "observations": observations,
        "actions": storage.actions.flatten(0, 1).cpu(),
        "old_action_means": storage.distribution_params[0].flatten(0, 1).cpu(),
        "old_std": storage.distribution_params[1].flatten(0, 1).cpu(),
        "old_log_std": torch.log(storage.distribution_params[1]).flatten(0, 1).cpu(),
        "old_log_prob": storage.actions_log_prob.flatten(0, 1).cpu(),
        "old_values": storage.values.flatten(0, 1).cpu(),
        "returns": storage.returns.flatten(0, 1).cpu(),
        "advantages_before_normalization": raw_advantage.flatten(0, 1).cpu(),
        "advantages_after_normalization": storage.advantages.flatten(0, 1).cpu(),
        "rewards": storage.rewards.flatten(0, 1).cpu(),
        "dones": storage.dones.flatten(0, 1).cpu(),
        "timeouts": timeouts.flatten(0, 1).cpu(),
    }
    for key, value in metadata.items():
        payload[key] = value.flatten(0, 1).cpu()
    payload["old_entropy"] = (
        0.5 + 0.5 * math.log(2.0 * math.pi) + payload["old_log_std"]
    ).sum(dim=-1, keepdim=True)
    return payload


def identity(actor, critic, batch):
    device = next(actor.parameters()).device
    collected = {key: [] for key in ("mean", "log_std", "value", "log_prob", "entropy")}
    with torch.inference_mode():
        for start in range(0, len(batch["actions"]), 2048):
            stop = start + 2048
            obs = batch["observations"][start:stop].to(device)
            actions = batch["actions"][start:stop].to(device)
            mean, std, entropy = distribution(actor, obs)
            collected["mean"].append(mean.cpu())
            collected["log_std"].append(torch.log(std).cpu())
            collected["value"].append(critic(obs).cpu())
            collected["log_prob"].append(actor.get_output_log_prob(actions).unsqueeze(1).cpu())
            collected["entropy"].append(entropy.unsqueeze(1).cpu())
    current_values = {key: torch.cat(value) for key, value in collected.items()}
    comparisons = {}
    for name, current, saved in (
        ("mean", current_values["mean"], batch["old_action_means"]),
        ("log_std", current_values["log_std"], batch["old_log_std"]),
        ("value", current_values["value"], batch["old_values"]),
        ("log_prob", current_values["log_prob"], batch["old_log_prob"]),
        ("entropy", current_values["entropy"], batch["old_entropy"]),
    ):
        error = (current - saved).abs()
        comparisons[name] = {
            "max_abs_error": float(error.max()), "mean_abs_error": float(error.mean())
        }
    non_finite = sum(int((~torch.isfinite(value)).sum()) for value in current_values.values())
    checks = {
        "mean_max_abs_error_le_1e_7": comparisons["mean"]["max_abs_error"] <= 1e-7,
        "log_std_max_abs_error_le_1e_7": comparisons["log_std"]["max_abs_error"] <= 1e-7,
        "log_prob_max_abs_error_le_1e_6": comparisons["log_prob"]["max_abs_error"] <= 1e-6,
        "value_max_abs_error_le_1e_6": comparisons["value"]["max_abs_error"] <= 1e-6,
        "non_finite_eq_0": non_finite == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "ROLLOUT_LOGPROB_CONTRACT_MISMATCH",
        "comparisons": comparisons, "non_finite": non_finite, "checks": checks,
    }


def make_advantages(batch, mode, device):
    raw = batch["advantages_before_normalization"].to(device).clone()
    if mode in ("production", "global"):
        return (raw - raw.mean()) / (raw.std() + 1e-8)
    cohort = batch["command_cohort"].to(device).long().squeeze(1)
    normalized = torch.empty_like(raw)
    for cohort_id in range(4):
        mask = cohort == cohort_id
        normalized[mask] = (raw[mask] - raw[mask].mean()) / (raw[mask].std() + 1e-8)
    return normalized


def evaluate_policy(actor, batch, device):
    obs = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    old_mean = batch["old_action_means"].to(device)
    old_std = batch["old_std"].to(device)
    old_log_prob = batch["old_log_prob"].to(device).squeeze(1)
    with torch.inference_mode():
        new_mean, new_std, entropy = distribution(actor, obs)
        new_log_prob = actor.get_output_log_prob(actions)
        mean_kl, std_kl, total_per_dim = exact_components(old_mean, old_std, new_mean, new_std)
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        sample_approx = (ratio - 1.0 - log_ratio)
        reverse = exact_components(new_mean, new_std, old_mean, old_std)[2].sum(1)
    total = total_per_dim.sum(1)
    return {
        "old_mean": old_mean, "old_std": old_std, "new_mean": new_mean, "new_std": new_std,
        "mean_per_dim": mean_kl, "std_per_dim": std_kl, "total_per_dim": total_per_dim,
        "total": total, "reverse": reverse, "entropy": entropy,
        "new_log_prob": new_log_prob, "log_ratio": log_ratio, "ratio": ratio,
        "sample_approx": sample_approx,
        "clip_fraction": float(((ratio - 1.0).abs() > 0.2).float().mean()),
        "mean_action_shift": float(torch.linalg.vector_norm(new_mean - old_mean, dim=1).mean()),
    }


def update_clone(
    actor,
    critic,
    batch,
    agent_cfg,
    indices,
    optimizer_mode="fresh",
    advantage_mode="production",
    freeze_mean=False,
    freeze_std=False,
    critic_only=False,
    parent_optimizer=None,
    timeline=False,
):
    device = next(actor.parameters()).device
    std_parameter = dict(actor.named_parameters())["distribution.std_param"]
    for name, parameter in actor.named_parameters():
        if critic_only or (freeze_mean and name != "distribution.std_param") or (freeze_std and name == "distribution.std_param"):
            parameter.requires_grad_(False)
    parameters = [p for p in list(actor.parameters()) + list(critic.parameters()) if p.requires_grad]
    learning_rate = float(agent_cfg.algorithm.learning_rate)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    restored = False
    if optimizer_mode == "restored":
        # Strict restore needs the original complete parameter list.
        for parameter in list(actor.parameters()) + list(critic.parameters()):
            parameter.requires_grad_(True)
        parameters = list(actor.parameters()) + list(critic.parameters())
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        optimizer.load_state_dict(copy.deepcopy(parent_optimizer))
        learning_rate = float(optimizer.param_groups[0]["lr"])
        restored = True
    elif optimizer_mode == "terminal_lr":
        learning_rate = float(parent_optimizer["param_groups"][0]["lr"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    obs = batch["observations"].to(device)
    actions = batch["actions"].to(device)
    old_log_prob = batch["old_log_prob"].to(device).squeeze(1)
    old_values = batch["old_values"].to(device)
    returns = batch["returns"].to(device)
    old_mean = batch["old_action_means"].to(device)
    old_std = batch["old_std"].to(device)
    advantages = make_advantages(batch, advantage_mode, device).squeeze(1)
    mini_size = len(indices) // int(agent_cfg.algorithm.num_mini_batches)
    rows = []
    first_cross = {"kl_gt_0_02": None, "kl_gt_0_10": None, "kl_gt_0_20": None, "clip_gt_0_50": None}
    update_index = 0
    for epoch in range(int(agent_cfg.algorithm.num_learning_epochs)):
        for mini in range(int(agent_cfg.algorithm.num_mini_batches)):
            update_index += 1
            idx = indices[mini * mini_size:(mini + 1) * mini_size]
            mb_obs, mb_actions = obs[idx], actions[idx]
            mean_before, std_before, entropy = distribution(actor, mb_obs)
            mb_log_prob = actor.get_output_log_prob(mb_actions)
            mb_old_mean, mb_old_std = old_mean[idx], old_std[idx]
            pre_kl = exact_components(mb_old_mean, mb_old_std, mean_before, std_before)[2].sum(1).mean()
            if agent_cfg.algorithm.schedule == "adaptive":
                if pre_kl > agent_cfg.algorithm.desired_kl * 2.0:
                    learning_rate = max(1e-5, learning_rate / 1.5)
                elif 0.0 < pre_kl < agent_cfg.algorithm.desired_kl / 2.0:
                    learning_rate = min(1e-2, learning_rate * 1.5)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
            ratio = torch.exp(mb_log_prob - old_log_prob[idx])
            surrogate = -advantages[idx] * ratio
            surrogate_clipped = -advantages[idx] * torch.clamp(ratio, 0.8, 1.2)
            policy_loss = torch.max(surrogate, surrogate_clipped).mean()
            values = critic(mb_obs)
            value_clipped = old_values[idx] + (values - old_values[idx]).clamp(-0.2, 0.2)
            value_loss = torch.max((values - returns[idx]).square(), (value_clipped - returns[idx]).square()).mean()
            loss = policy_loss + float(agent_cfg.algorithm.value_loss_coef) * value_loss
            if not critic_only:
                loss = loss - float(agent_cfg.algorithm.entropy_coef) * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            actor_grad = norm(p for p in actor.parameters() if p.requires_grad)
            critic_grad = norm(critic.parameters())
            std_grad = 0.0 if std_parameter.grad is None else float(torch.linalg.vector_norm(std_parameter.grad))
            mean_grad = norm(p for n, p in actor.named_parameters() if n != "distribution.std_param" and p.requires_grad)
            actor_before = [p.detach().clone() for p in actor.parameters()]
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(agent_cfg.algorithm.max_grad_norm))
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(agent_cfg.algorithm.max_grad_norm))
            optimizer.step()
            effective = math.sqrt(sum(float((p.detach() - q).square().sum()) for p, q in zip(actor.parameters(), actor_before)))
            metrics = evaluate_policy(actor, batch, device)
            mean_total = float(metrics["mean_per_dim"].sum(1).mean())
            std_total = float(metrics["std_per_dim"].sum(1).mean())
            total = float(metrics["total"].mean())
            row = {
                "update_index": update_index, "epoch": epoch + 1, "mini_batch": mini + 1,
                "approximate_kl": total, "exact_kl": total,
                "mean_component_kl": mean_total, "std_component_kl": std_total,
                "clip_fraction": metrics["clip_fraction"], "policy_loss": float(policy_loss),
                "value_loss": float(value_loss), "entropy": float(entropy.mean()),
                "actor_gradient_norm": actor_grad, "critic_gradient_norm": critic_grad,
                "log_std_gradient_norm": std_grad, "actor_mean_gradient_norm": mean_grad,
                "mean_action_shift": metrics["mean_action_shift"],
                "log_std_shift_l2": float(torch.linalg.vector_norm(torch.log(metrics["new_std"][0]) - torch.log(old_std[0]))),
                "learning_rate": learning_rate,
                "adam_step_count": update_index if not restored else int(next(iter(optimizer.state.values()))["step"]),
                "actor_effective_update_l2": effective,
            }
            rows.append(row)
            for key, hit in (
                ("kl_gt_0_02", total > 0.02), ("kl_gt_0_10", total > 0.10),
                ("kl_gt_0_20", total > 0.20), ("clip_gt_0_50", metrics["clip_fraction"] > 0.50),
            ):
                if hit and first_cross[key] is None:
                    first_cross[key] = update_index
    final = evaluate_policy(actor, batch, device)
    total_kl = float(final["total"].mean())
    result = {
        "optimizer_mode": optimizer_mode, "advantage_mode": advantage_mode,
        "freeze_mean": freeze_mean, "freeze_std": freeze_std, "critic_only": critic_only,
        "exact_kl": total_kl, "approximate_kl": total_kl,
        "sample_based_approximate_kl": float(final["sample_approx"].mean()),
        "clip_fraction": final["clip_fraction"], "mean_action_l2_shift": final["mean_action_shift"],
        "mean_component_kl": float(final["mean_per_dim"].sum(1).mean()),
        "std_component_kl": float(final["std_per_dim"].sum(1).mean()),
        "final_learning_rate": learning_rate,
        "non_finite": int(sum((~torch.isfinite(x)).sum() for x in (final["new_mean"], final["new_std"], final["total"]))),
        "value_loss": rows[-1]["value_loss"], "critic_gradient_norm": rows[-1]["critic_gradient_norm"],
        "first_crossings": first_cross,
        "safety_gate": {
            "exact_kl_le_0_20": total_kl <= 0.20,
            "approximate_kl_le_0_20": total_kl <= 0.20,
            "clip_fraction_le_0_50": final["clip_fraction"] <= 0.50,
            "nan_inf_eq_0": torch.isfinite(final["total"]).all().item(),
            "critic_gradient_le_1e6": rows[-1]["critic_gradient_norm"] <= 1e6,
            "value_loss_le_1e8": rows[-1]["value_loss"] <= 1e8,
            "mean_action_shift_le_2": final["mean_action_shift"] <= 2.0,
        },
        "preferred": {"exact_kl_le_0_05": total_kl <= 0.05, "clip_fraction_le_0_30": final["clip_fraction"] <= 0.30},
    }
    result["safety_gate_pass"] = all(result["safety_gate"].values())
    return result, rows, final


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    batch_path = args.output / "initial_rollout_batch.pt"
    task = "Isaac-Exp011-Go2-Bidirectional-0To2-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = 2048
    cfg.seed = 20260911
    agent_cfg.seed = 20260911
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device

    with nullcontext():
        raw = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner_device = torch.device(runner.device)
        parent = torch.load(args.checkpoint.resolve(strict=True), map_location=runner.device, weights_only=False)
        runner.load(str(args.checkpoint), load_cfg={
            "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False,
        }, strict=True, map_location=runner.device)
        obs = wrapped.get_observations().to(runner.device)
        runner.alg.train_mode()
        term = wrapped.unwrapped.command_manager.get_term("base_velocity")
        timeouts = []
        metadata = defaultdict(list)
        with torch.inference_mode():
            for time_index in range(int(agent_cfg.num_steps_per_env)):
                metadata["command_cohort"].append(term.cohort.clone())
                metadata["source_speed"].append(term.source_speed.clone())
                metadata["target_speed"].append(term.target_speed.clone())
                phase = torch.zeros_like(term.cohort)
                transition = (term.cohort == 2) | (term.cohort == 3)
                phase[transition & (term.elapsed_s >= term.source_hold_s)] = 1
                phase[transition & (term.elapsed_s >= term.source_hold_s + term.ramp_duration_s)] = 2
                metadata["command_phase"].append(phase)
                metadata["environment_index"].append(torch.arange(wrapped.num_envs, device=runner.device))
                metadata["time_index"].append(torch.full((wrapped.num_envs,), time_index, device=runner.device))
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                timeout = extras.get("time_outs", torch.zeros_like(dones)).to(runner.device)
                timeouts.append(timeout)
                runner.alg.process_env_step(obs, rewards, dones, extras)
            runner.alg.compute_returns(obs)
        raw_advantage = runner.alg.storage.returns - runner.alg.storage.values
        stacked_metadata = {key: torch.stack(value).unsqueeze(-1) for key, value in metadata.items()}
        batch = flatten_storage(
            runner.alg.storage, raw_advantage, torch.stack(timeouts).unsqueeze(-1), stacked_metadata
        )
        rng_state = torch.cuda.get_rng_state(runner_device) if runner_device.type == "cuda" else torch.random.get_rng_state()
        torch.save(batch, batch_path)
        batch_sha = sha256(batch_path)
        manifest = {
            "status": "DIAGNOSTIC_RECAPTURE",
            "reason": "Stage 2 rollout storage was not serialized; user-authorized one-time pre-update recapture.",
            "path": str(batch_path.resolve()), "sha256": batch_sha,
            "samples": len(batch["actions"]), "num_envs": 2048, "steps_per_env": 24,
            "diagnostic_environment_interactions": 49152, "new_pilot_interactions": 0,
            "production_optimizer_updates": 0, "seed": 20260911,
            "fields": sorted(batch), "tensor_shapes": {
                key: list(value.shape) if hasattr(value, "shape") else "TensorDict"
                for key, value in batch.items()
            },
        }
        dump(args.output / "initial_rollout_batch_manifest.json", manifest)

        identity_result = identity(runner.alg.actor, runner.alg.critic, batch)
        reloaded = torch.load(batch_path, map_location="cpu", weights_only=False)
        identity_result["after_batch_serialization_reload"] = identity(runner.alg.actor, runner.alg.critic, reloaded)
        identity_result["checkpoint_strict_load"] = True
        dump(args.output / "no_update_identity_test.json", identity_result)
        if identity_result["status"] != "PASS" or identity_result["after_batch_serialization_reload"]["status"] != "PASS":
            wrapped.close()
            simulation_app.close()
            raise RuntimeError("ROLLOUT_LOGPROB_CONTRACT_MISMATCH")

        initial_actor = copy.deepcopy(runner.alg.actor)
        initial_critic = copy.deepcopy(runner.alg.critic)
        unstable = torch.load(args.unstable_checkpoint.resolve(strict=True), map_location=runner.device, weights_only=False)
        actual_actor = copy.deepcopy(initial_actor)
        actual_actor.load_state_dict(unstable["actor_state_dict"], strict=True)
        actual = evaluate_policy(actual_actor, batch, runner.device)

        if runner_device.type == "cuda":
            torch.cuda.set_rng_state(rng_state, runner_device)
        else:
            torch.random.set_rng_state(rng_state)
        indices = torch.randperm(len(batch["actions"]), device=runner.device)

        conditions = {}
        finals = {}
        timelines = {}
        specs = {
            "S0_PRODUCTION": {},
            "S1_FREEZE_LOG_STD": {"freeze_std": True},
            "S2_FREEZE_ACTOR_MEAN": {"freeze_mean": True},
            "S3_CRITIC_ONLY": {"critic_only": True},
            "S4_RESTORED_OPTIMIZER": {"optimizer_mode": "restored"},
            "C_TERMINAL_LR": {"optimizer_mode": "terminal_lr"},
            "GLOBAL_NORMALIZATION": {"advantage_mode": "global"},
            "COHORT_NORMALIZATION": {"advantage_mode": "cohort"},
        }
        for name, kwargs in specs.items():
            actor = copy.deepcopy(initial_actor)
            critic = copy.deepcopy(initial_critic)
            result, rows, final = update_clone(
                actor, critic, batch, agent_cfg, indices, parent_optimizer=parent["optimizer_state_dict"], **kwargs
            )
            conditions[name] = result
            finals[name] = final
            timelines[name] = rows

        write_csv(args.output / "first_update_minibatch_timeline.csv", timelines["S0_PRODUCTION"])
        dump(args.output / "first_update_minibatch_timeline.json", {
            "condition": "S0_PRODUCTION", "rows": timelines["S0_PRODUCTION"],
            "first_crossings": conditions["S0_PRODUCTION"]["first_crossings"],
        })

        actual_mean = float(actual["mean_per_dim"].sum(1).mean())
        actual_std = float(actual["std_per_dim"].sum(1).mean())
        actual_total = float(actual["total"].mean())
        cohort = batch["command_cohort"].long().squeeze(1).to(runner.device)
        per_cohort = {}
        for cohort_id, cohort_name in enumerate(COHORT_NAMES):
            mask = cohort == cohort_id
            per_cohort[cohort_name] = {
                "count": int(mask.sum()), "exact_kl": float(actual["total"][mask].mean()),
                "mean_component": float(actual["mean_per_dim"][mask].sum(1).mean()),
                "std_component": float(actual["std_per_dim"][mask].sum(1).mean()),
                "clip_fraction": float(((actual["ratio"][mask] - 1.0).abs() > 0.2).float().mean()),
            }
        exact = {
            "reported_stage2_approximate_kl": 0.5129385441541672,
            "actual_unstable_checkpoint_on_recaptured_batch": {
                "old_to_new_exact_kl": actual_total,
                "new_to_old_exact_kl": float(actual["reverse"].mean()),
                "symmetric_kl": float(0.5 * (actual["total"] + actual["reverse"]).mean()),
                "mean_component": actual_mean, "std_component": actual_std,
                "mean_fraction": actual_mean / actual_total, "std_fraction": actual_std / actual_total,
                "sample_based_approximate_kl": float(actual["sample_approx"].mean()),
                "negative_log_ratio_estimator": float((-actual["log_ratio"]).mean()),
                "ratio_clip_fraction": actual["clip_fraction"],
                "distribution": stats(actual["total"]),
            },
            "production_shadow": conditions["S0_PRODUCTION"],
            "per_cohort": per_cohort,
        }
        dump(args.output / "exact_kl_analysis.json", exact)
        per_joint_rows = []
        for index, joint in enumerate(JOINTS):
            per_joint_rows.append({
                "joint_index": index, "joint": joint,
                "mean_component_kl": float(actual["mean_per_dim"][:, index].mean()),
                "std_component_kl": float(actual["std_per_dim"][:, index].mean()),
                "total_kl": float(actual["total_per_dim"][:, index].mean()),
            })
        write_csv(args.output / "per_joint_kl.csv", per_joint_rows)
        consistency_delta = abs(actual_total - 0.5129385441541672)
        consistency = {
            "classification": "KL_ESTIMATOR_CONSISTENT" if consistency_delta <= 0.02 else "APPROX_KL_ESTIMATOR_MISMATCH",
            "reported": 0.5129385441541672, "recomputed_exact": actual_total,
            "absolute_delta": consistency_delta,
            "note": "Stage 2 label approximate_kl was RSL-RL analytical Gaussian KL(old||new), not a sampled estimator.",
        }
        dump(args.output / "kl_estimator_consistency.json", consistency)

        old_std = batch["old_std"][0]
        new_std = actual["new_std"][0].cpu()
        std_rows = []
        for index, joint in enumerate(JOINTS):
            std_rows.append({
                "joint_index": index, "joint": joint,
                "old_log_std": float(torch.log(old_std[index])), "new_log_std": float(torch.log(new_std[index])),
                "old_std": float(old_std[index]), "new_std": float(new_std[index]),
                "absolute_std_change": float(new_std[index] - old_std[index]),
                "relative_std_change": float((new_std[index] - old_std[index]) / old_std[index]),
            })
        write_csv(args.output / "per_joint_log_std_update.csv", std_rows)

        # First-minibatch log-std gradient decomposition at the initial policy.
        actor = copy.deepcopy(initial_actor)
        mb = indices[:len(indices) // int(agent_cfg.algorithm.num_mini_batches)]
        mb_obs = batch["observations"].to(runner.device)[mb]
        mb_actions = batch["actions"].to(runner.device)[mb]
        mb_adv = make_advantages(batch, "production", runner.device).squeeze(1)[mb]
        mb_old_lp = batch["old_log_prob"].to(runner.device).squeeze(1)[mb]
        mean, std, entropy = distribution(actor, mb_obs)
        ratio = torch.exp(actor.get_output_log_prob(mb_actions) - mb_old_lp)
        surrogate = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
        std_param = dict(actor.named_parameters())["distribution.std_param"]
        surrogate_grad = torch.autograd.grad(surrogate, std_param, retain_graph=True)[0]
        entropy_loss = -float(agent_cfg.algorithm.entropy_coef) * entropy.mean()
        entropy_grad = torch.autograd.grad(entropy_loss, std_param)[0]
        dump(args.output / "log_std_gradient_decomposition.json", {
            "scope": "initial policy, first production minibatch",
            "parameterization": "direct learnable std_param; log_std is derived with log(std)",
            "surrogate_gradient_per_joint": surrogate_grad.detach().cpu().tolist(),
            "entropy_gradient_per_joint": entropy_grad.detach().cpu().tolist(),
            "total_gradient_per_joint": (surrogate_grad + entropy_grad).detach().cpu().tolist(),
            "surrogate_gradient_norm": float(torch.linalg.vector_norm(surrogate_grad)),
            "entropy_gradient_norm": float(torch.linalg.vector_norm(entropy_grad)),
            "total_gradient_norm": float(torch.linalg.vector_norm(surrogate_grad + entropy_grad)),
            "clamp_or_minimum_std": False,
        })

        state = parent["optimizer_state_dict"]
        steps = [int(value["step"]) for value in state["state"].values()]
        dump(args.output / "optimizer_state_audit.json", {
            "fresh_optimizer_initial_learning_rate": float(agent_cfg.algorithm.learning_rate),
            "official_runner_initial_learning_rate": 1e-3,
            "checkpoint_terminal_learning_rate": float(state["param_groups"][0]["lr"]),
            "checkpoint_iteration": int(parent["iter"]), "checkpoint_optimizer_state_present": True,
            "checkpoint_parameter_state_count": len(state["state"]),
            "checkpoint_step_count_min": min(steps), "checkpoint_step_count_max": max(steps),
            "adam_first_moment_norm": math.sqrt(sum(float(v["exp_avg"].square().sum()) for v in state["state"].values())),
            "adam_second_moment_norm": math.sqrt(sum(float(v["exp_avg_sq"].square().sum()) for v in state["state"].values())),
            "stage2_fresh_scheduler_step": 0, "stage2_fresh_optimizer_state_entries": 0,
            "production_restore_performed": False,
        })
        optimizer_comparison = {
            key: conditions[key] for key in ("S0_PRODUCTION", "S4_RESTORED_OPTIMIZER", "C_TERMINAL_LR")
        }
        dump(args.output / "optimizer_shadow_comparison.json", optimizer_comparison)

        cohort_adv = {}
        raw_adv = batch["advantages_before_normalization"].squeeze(1)
        norm_adv = batch["advantages_after_normalization"].squeeze(1)
        returns = batch["returns"].squeeze(1)
        values = batch["old_values"].squeeze(1)
        for cohort_id, name in enumerate(COHORT_NAMES):
            mask = batch["command_cohort"].squeeze(1) == cohort_id
            cohort_adv[name] = {
                "count": int(mask.sum()), "raw_advantage": stats(raw_adv[mask]),
                "normalized_advantage": stats(norm_adv[mask]),
                "positive_advantage_fraction": float((raw_adv[mask] > 0).float().mean()),
                "return": stats(returns[mask]), "value_prediction": stats(values[mask]),
                "policy_ratio": stats(actual["ratio"].cpu()[mask]),
                "exact_kl": per_cohort[name]["exact_kl"], "clip_fraction": per_cohort[name]["clip_fraction"],
            }
        dump(args.output / "advantage_distribution_by_cohort.json", cohort_adv)

        # Initial full-batch actor gradient contribution by cohort.
        grad_contribution = {}
        gradient_vectors = {}
        for cohort_id, name in enumerate(COHORT_NAMES):
            actor = copy.deepcopy(initial_actor)
            mask = (batch["command_cohort"].squeeze(1) == cohort_id).to(runner.device)
            obs_c = batch["observations"].to(runner.device)[mask]
            actions_c = batch["actions"].to(runner.device)[mask]
            adv_c = batch["advantages_after_normalization"].to(runner.device).squeeze(1)[mask]
            old_lp_c = batch["old_log_prob"].to(runner.device).squeeze(1)[mask]
            distribution(actor, obs_c)
            ratio_c = torch.exp(actor.get_output_log_prob(actions_c) - old_lp_c)
            loss_c = torch.max(-adv_c * ratio_c, -adv_c * torch.clamp(ratio_c, 0.8, 1.2)).mean()
            actor.zero_grad()
            loss_c.backward()
            vector = torch.cat([p.grad.flatten() for p in actor.parameters() if p.grad is not None])
            gradient_vectors[name] = vector
            grad_contribution[name] = {"actor_gradient_norm": float(torch.linalg.vector_norm(vector))}
        norm_sum = sum(item["actor_gradient_norm"] for item in grad_contribution.values())
        for item in grad_contribution.values():
            item["norm_share"] = item["actor_gradient_norm"] / norm_sum
        combined = sum(gradient_vectors.values())
        grad_contribution["combined"] = {
            "norm": float(torch.linalg.vector_norm(combined)),
            "pairwise_cosine": {
                f"{a}__{b}": float(torch.nn.functional.cosine_similarity(gradient_vectors[a], gradient_vectors[b], dim=0))
                for i, a in enumerate(COHORT_NAMES) for b in COHORT_NAMES[i + 1:]
            },
        }
        dump(args.output / "cohort_gradient_contribution.json", grad_contribution)
        dump(args.output / "advantage_normalization_audit.json", {
            "production_mode": "global rollout-batch normalization in PPO.compute_returns",
            "normalize_advantage_per_mini_batch": False,
            "production": conditions["S0_PRODUCTION"],
            "global_recalculation": conditions["GLOBAL_NORMALIZATION"],
            "cohort_wise_diagnostic": conditions["COHORT_NORMALIZATION"],
        })

        value_error = returns - values
        explained_variance = 1.0 - float(value_error.var() / (returns.var() + 1e-8))
        dump(args.output / "critic_value_audit.json", {
            "classification": "CRITIC_STABLE" if conditions["S0_PRODUCTION"]["value_loss"] < 1e8 else "CRITIC_CONTRIBUTES_TO_INSTABILITY",
            "return": stats(returns), "value": stats(values), "value_error": stats(value_error),
            "explained_variance": explained_variance,
            "production_final_value_loss": conditions["S0_PRODUCTION"]["value_loss"],
            "production_final_critic_gradient_norm": conditions["S0_PRODUCTION"]["critic_gradient_norm"],
            "per_cohort": {
                name: {"return_mean": cohort_adv[name]["return"]["mean"], "value_mean": cohort_adv[name]["value_prediction"]["mean"]}
                for name in COHORT_NAMES
            },
        })
        dump(args.output / "shadow_intervention_protocol.json", {
            "fixed_checkpoint": str(args.checkpoint.resolve()), "fixed_batch_sha256": batch_sha,
            "same_minibatch_order": True, "ppo_updates_per_condition": 1, "isaac_stepping_during_shadow": 0,
            "conditions": list(specs),
        })
        dump(args.output / "shadow_intervention_results.json", conditions)

        wrapped.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
