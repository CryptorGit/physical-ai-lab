"""Warm-start and train the frozen-contract Stage 2 Go2 policy."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
DEFAULT_OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
sys.path.insert(0, str(EXP / "src"))

import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage2_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=("wiring", "pilot"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
add_launcher_args(parser)
args, hydra = setup_preset_cli(parser)
sys.argv = [sys.argv[0], *hydra]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def grad_norm(module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2))
    return math.sqrt(total)


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_curves(path: Path, curves: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


def save_runner_checkpoint(runner, path: Path, iteration: int, infos: dict | None = None) -> None:
    payload = runner.alg.save()
    payload["iter"] = iteration
    payload["infos"] = infos
    torch.save(payload, path)


def distribution_parameters(actor, observations):
    with torch.inference_mode():
        mean = actor(observations, stochastic_output=False).clone()
        actor(observations, stochastic_output=True)
        params = tuple(value.clone() for value in actor.output_distribution_params)
    return mean, params


def main() -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    run_output = args.output if args.mode == "pilot" else args.output / "wiring"
    checkpoints = run_output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    num_envs, iterations = (2048, 300) if args.mode == "pilot" else (16, 2)
    task = "Isaac-Exp011-Go2-Bidirectional-0To2-v0"
    cfg, agent_cfg = resolve_task_config(task, "rsl_rl_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.seed = 20260911
    agent_cfg.seed = 20260911
    agent_cfg.max_iterations = iterations
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device

    with launch_simulation(cfg, args):
        raw = gym.make(task, cfg=cfg)
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
        agent_cfg = handle_deprecated_rsl_rl_cfg(
            agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
        )
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        optimizer_state_before = len(runner.alg.optimizer.state)
        parent = torch.load(args.checkpoint.resolve(strict=True), map_location=wrapped.unwrapped.device, weights_only=False)
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=wrapped.unwrapped.device,
        )
        optimizer_state_after = len(runner.alg.optimizer.state)
        actor_state = runner.alg.actor.state_dict()
        critic_state = runner.alg.critic.state_dict()
        actor_exact = all(
            key in actor_state and torch.equal(actor_state[key].detach().cpu(), value.detach().cpu())
            for key, value in parent["actor_state_dict"].items()
        )
        critic_exact = all(
            key in critic_state and torch.equal(critic_state[key].detach().cpu(), value.detach().cpu())
            for key, value in parent["critic_state_dict"].items()
        )
        observations = wrapped.get_observations().to(runner.device)
        reference_mean, reference_distribution = distribution_parameters(runner.alg.actor, observations)
        runner.load(
            str(args.checkpoint),
            load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
            strict=True,
            map_location=wrapped.unwrapped.device,
        )
        loaded_mean, loaded_distribution = distribution_parameters(runner.alg.actor, observations)
        std_key = "distribution.std_param"
        std_value = actor_state[std_key].detach().cpu()
        warm_checks = {
            "actor_state_bitwise_equal": actor_exact,
            "critic_state_bitwise_equal": critic_exact,
            "deterministic_action_bitwise_equal": torch.equal(reference_mean, loaded_mean),
            "stochastic_distribution_mean_bitwise_equal": torch.equal(
                reference_distribution[0], loaded_distribution[0]
            ),
            "distribution_parameters_bitwise_equal": all(
                torch.equal(left, right) for left, right in zip(reference_distribution, loaded_distribution)
            ),
            "log_std_bitwise_equal": torch.equal(
                parent["actor_state_dict"][std_key].detach().cpu(), std_value
            ),
            "observation_normalizer_state_equal": all(
                torch.equal(actor_state[key].detach().cpu(), value.detach().cpu())
                for key, value in parent["actor_state_dict"].items() if "normalizer" in key
            ),
            "optimizer_state_fresh_before": optimizer_state_before == 0,
            "optimizer_state_fresh_after": optimizer_state_after == 0,
        }
        warm_audit = {
            "status": "PASS" if all(warm_checks.values()) else "WARMSTART_COMPATIBILITY_FAIL",
            "parent_checkpoint": str(args.checkpoint.resolve()),
            "parent_sha256": sha256(args.checkpoint.resolve()),
            "load_cfg": {"actor": True, "critic": True, "optimizer": False, "iteration": False},
            "checks": warm_checks,
            "actor_parameter_hash": tensor_hash(actor_state),
            "critic_parameter_hash": tensor_hash(critic_state),
            "log_std": {
                "mean": float(torch.log(std_value).mean()),
                "max": float(torch.log(std_value).max()),
                "stored_parameter_mean": float(std_value.mean()),
            },
            "observation_normalizer": "Identity; no tensor state in checkpoint",
        }
        dump(run_output / "warmstart_audit.json", warm_audit)
        if args.mode == "pilot":
            dump(args.output / "warmstart_audit.json", warm_audit)
        if not all(warm_checks.values()):
            wrapped.close()
            raise RuntimeError("WARMSTART_COMPATIBILITY_FAIL")

        runner.current_learning_iteration = 0
        initial_path = checkpoints / "model_initial.pt"
        save_runner_checkpoint(
            runner, initial_path, 0,
            infos={"stage": "pre_update_warmstart", "optimizer_inherited": False},
        )
        fixed_obs = observations
        initial_mean, initial_params = distribution_parameters(runner.alg.actor, fixed_obs)
        curves = []
        first_update = {}
        save_iterations = {25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300}
        obs = wrapped.get_observations().to(runner.device)
        runner.alg.train_mode()
        start_time = time.time()
        unstable = False
        for iteration in range(1, iterations + 1):
            rewards_seen = []
            dones_seen = []
            with torch.inference_mode():
                for _ in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    if not torch.isfinite(rewards).all():
                        unstable = True
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    rewards_seen.append(float(rewards.mean()))
                    dones_seen.append(float(dones.float().mean()))
                runner.alg.compute_returns(obs)

            original_clear = None
            if iteration == 1:
                original_clear = runner.alg.storage.clear
                runner.alg.storage.clear = lambda: None
            loss = runner.alg.update()
            actor_gradient_norm = grad_norm(runner.alg.actor)
            critic_gradient_norm = grad_norm(runner.alg.critic)

            approximate_kl = 0.0
            clip_fraction = 0.0
            if iteration == 1:
                kl_values, clipped_values = [], []
                with torch.inference_mode():
                    for batch in runner.alg.storage.mini_batch_generator(
                        runner.alg.num_mini_batches, 1
                    ):
                        runner.alg.actor(batch.observations, stochastic_output=True)
                        current_params = tuple(
                            value[: batch.observations.batch_size[0]]
                            for value in runner.alg.actor.output_distribution_params
                        )
                        kl_values.append(float(runner.alg.actor.get_kl_divergence(
                            batch.old_distribution_params, current_params
                        ).mean()))
                        log_prob = runner.alg.actor.get_output_log_prob(batch.actions)
                        ratio = torch.exp(log_prob - torch.squeeze(batch.old_actions_log_prob))
                        clipped_values.append(float(((ratio - 1.0).abs() > runner.alg.clip_param).float().mean()))
                approximate_kl = sum(kl_values) / len(kl_values)
                clip_fraction = sum(clipped_values) / len(clipped_values)
                runner.alg.storage.clear = original_clear
                original_clear()

            current_mean, current_params = distribution_parameters(runner.alg.actor, fixed_obs)
            policy_kl_initial = float(
                runner.alg.actor.get_kl_divergence(initial_params, current_params).mean()
            )
            mean_action_shift = float(torch.linalg.vector_norm(current_mean - initial_mean, dim=1).mean())
            current_std = runner.alg.actor.state_dict()[std_key]
            log_std = torch.log(current_std)
            finite = (
                all(math.isfinite(float(value)) for value in loss.values())
                and torch.isfinite(current_mean).all().item()
                and torch.isfinite(log_std).all().item()
            )
            row = {
                "iteration": iteration,
                "interaction_count": iteration * num_envs * agent_cfg.num_steps_per_env,
                "reward_mean": sum(rewards_seen) / len(rewards_seen),
                "done_fraction": sum(dones_seen) / len(dones_seen),
                "value_loss": float(loss["value"]),
                "policy_loss": float(loss["surrogate"]),
                "entropy": float(loss["entropy"]),
                "approximate_kl": approximate_kl if iteration == 1 else "",
                "clip_fraction": clip_fraction if iteration == 1 else "",
                "actor_gradient_norm": actor_gradient_norm,
                "critic_gradient_norm": critic_gradient_norm,
                "log_std_mean": float(log_std.mean()),
                "log_std_max": float(log_std.max()),
                "mean_action_l2_shift_from_initial": mean_action_shift,
                "policy_kl_from_initial": policy_kl_initial,
                "finite": bool(finite),
                "elapsed_s": time.time() - start_time,
            }
            curves.append(row)
            if iteration == 1:
                fail_checks = {
                    "nan_or_inf_eq_0": finite and not unstable,
                    "first_update_approximate_kl_le_0_20": approximate_kl <= 0.20,
                    "first_update_mean_action_shift_le_2": mean_action_shift <= 2.0,
                    "critic_gradient_norm_le_1e6": critic_gradient_norm <= 1.0e6,
                    "value_loss_le_1e8": float(loss["value"]) <= 1.0e8,
                    "log_std_finite": torch.isfinite(log_std).all().item(),
                }
                first_update = {**row, "gate_checks": fail_checks, "pass": all(fail_checks.values())}
                dump(run_output / "optimization_stability.json", {
                    "status": (
                        "PASS" if first_update["pass"]
                        else "GO2_TRAINING_UNSTABLE" if args.mode == "pilot"
                        else "WIRING_ONLY_THRESHOLD_WARNING"
                    ),
                    "first_update": first_update,
                })
                if not first_update["pass"] and args.mode == "pilot":
                    write_curves(run_output / "training_curves.csv", curves)
                    runner.current_learning_iteration = iteration
                    save_runner_checkpoint(
                        runner, checkpoints / f"model_{iteration}_unstable.pt", iteration
                    )
                    wrapped.close()
                    raise RuntimeError("GO2_TRAINING_UNSTABLE")
            if iteration in save_iterations:
                runner.current_learning_iteration = iteration
                save_runner_checkpoint(
                    runner, checkpoints / f"model_{iteration}.pt", iteration,
                    infos={"optimizer_inherited": False},
                )
            if iteration % 10 == 0 or iteration <= 2:
                print(
                    f"STAGE2 iteration={iteration}/{iterations} reward={row['reward_mean']:.4f} "
                    f"value={row['value_loss']:.4f} kl_initial={policy_kl_initial:.5f}"
                )

        write_curves(run_output / "training_curves.csv", curves)
        stability = {
            "status": (
                "PASS" if first_update["pass"] else "WIRING_ONLY_THRESHOLD_WARNING"
            ),
            "first_update": first_update,
            "max_value_loss": max(row["value_loss"] for row in curves),
            "max_actor_gradient_norm": max(row["actor_gradient_norm"] for row in curves),
            "max_critic_gradient_norm": max(row["critic_gradient_norm"] for row in curves),
            "nan_inf_count": sum(not row["finite"] for row in curves),
            "final_log_std_mean": curves[-1]["log_std_mean"],
            "final_log_std_max": curves[-1]["log_std_max"],
            "iterations_completed": iterations,
            "interaction_count": iterations * num_envs * agent_cfg.num_steps_per_env,
            "elapsed_s": time.time() - start_time,
        }
        dump(run_output / "optimization_stability.json", stability)
        if args.mode == "pilot":
            (args.output / "training_curves.csv").write_bytes((run_output / "training_curves.csv").read_bytes())
            dump(args.output / "optimization_stability.json", stability)
        wrapped.close()


if __name__ == "__main__":
    main()
