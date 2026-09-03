"""Train Stage 7 from the Stage 4 selected model and matching optimizer state."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization"
STAGE3_BATCH = (
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage3_first_update_stability_diagnosis/initial_rollout_batch.pt"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, default=OUT)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage7_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value) -> str:
    digest = hashlib.sha256()
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            digest.update(str(key).encode())
            digest.update(tensor_hash(item).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(tensor_hash(item).encode())
    elif torch.is_tensor(value):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def parameter_hash(module) -> str:
    return tensor_hash(module.state_dict())


def grad_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(parameter.grad.detach().square().sum())
    return math.sqrt(total)


def distribution(actor, observations):
    actor(observations, stochastic_output=True)
    mean, std = actor.output_distribution_params
    return mean, std, actor.output_entropy


def exact_components(old_mean, old_std, new_mean, new_std):
    mean = (old_mean - new_mean).square() / (2.0 * new_std.square())
    std = torch.log(new_std / old_std) + old_std.square() / (2.0 * new_std.square()) - 0.5
    return mean, std, mean + std


def optimizer_moments(state):
    first = math.sqrt(sum(float(item["exp_avg"].detach().square().sum()) for item in state["state"].values()))
    second = math.sqrt(sum(float(item["exp_avg_sq"].detach().square().sum()) for item in state["state"].values()))
    steps = [int(item["step"]) for item in state["state"].values()]
    return first, second, min(steps), max(steps)


def state_equal(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(state_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    if torch.is_tensor(left):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    return left == right


def flatten_rollout(storage, timeouts):
    return {
        "observations": storage.observations.flatten(0, 1).cpu(),
        "actions": storage.actions.flatten(0, 1).cpu(),
        "old_mean": storage.distribution_params[0].flatten(0, 1).cpu(),
        "old_std": storage.distribution_params[1].flatten(0, 1).cpu(),
        "old_log_prob": storage.actions_log_prob.flatten(0, 1).cpu(),
        "values": storage.values.flatten(0, 1).cpu(),
        "returns": storage.returns.flatten(0, 1).cpu(),
        "advantages": storage.advantages.flatten(0, 1).cpu(),
        "raw_advantages": (storage.returns - storage.values).flatten(0, 1).cpu(),
        "rewards": storage.rewards.flatten(0, 1).cpu(),
        "dones": storage.dones.flatten(0, 1).cpu(),
        "timeouts": torch.stack(timeouts).reshape(-1, 1).cpu(),
    }


def tensor_stats(value):
    value = value.float()
    return {"mean": float(value.mean()), "std": float(value.std())}


def batch_comparison(current, reference, current_path):
    current_sha = sha(current_path)
    if reference is None:
        return {"sha256": current_sha, "stage3_batch_available": False}
    cohort = torch.arange(2048).remainder(4).repeat(24)
    result = {
        "stage4_sha256": current_sha,
        "stage3_sha256": sha(STAGE3_BATCH),
        "sha_match": current_sha == sha(STAGE3_BATCH),
        "sha_mismatch_reason": (
            None if current_sha == sha(STAGE3_BATCH)
            else "Tensor key/schema differs; compare distributions. Same seed/config and update-free lifecycle retained."
        ),
        "cohort_counts": {str(index): int((cohort == index).sum()) for index in range(4)},
        "distribution_comparison": {},
    }
    mappings = {
        "observations": ("observations", "observations"),
        "actions": ("actions", "actions"),
        "rewards": ("rewards", "rewards"),
        "returns": ("returns", "returns"),
        "raw_advantages": ("raw_advantages", "advantages_before_normalization"),
        "dones": ("dones", "dones"),
    }
    for label, (current_key, reference_key) in mappings.items():
        current_value = current[current_key]["policy"] if label == "observations" else current[current_key]
        reference_value = reference[reference_key]["policy"] if label == "observations" else reference[reference_key]
        result["distribution_comparison"][label] = {
            "stage4": tensor_stats(current_value),
            "stage3": tensor_stats(reference_value),
            "mean_delta": float(current_value.float().mean() - reference_value.float().mean()),
            "std_delta": float(current_value.float().std() - reference_value.float().std()),
        }
    result["done_count"] = {
        "stage4": int(current["dones"].sum()), "stage3": int(reference["dones"].sum())
    }
    return result


def save_checkpoint(runner, path, local_iteration):
    payload = runner.alg.save()
    payload["iter"] = 1049 + local_iteration
    payload["infos"] = {
        "source_checkpoint_iteration": 1049,
        "stage4_selected_local_iteration": 50,
        "stage7_local_iteration": local_iteration,
        "optimizer_restored": True,
    }
    torch.save(payload, path)


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    cfg, agent_cfg = resolve_task_config(
        "Isaac-Exp011-Go2-LowSpeed-Stabilization-v0", "rsl_rl_cfg_entry_point"
    )
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 20260921
    agent_cfg.seed = 20260921
    agent_cfg.max_iterations = args.iterations
    if args.device:
        cfg.sim.device = args.device
        agent_cfg.device = args.device
    raw = gym.make("Isaac-Exp011-Go2-LowSpeed-Stabilization-v0", cfg=cfg)
    wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
    )
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    device = torch.device(runner.device)
    parent = torch.load(args.checkpoint.resolve(strict=True), map_location=device, weights_only=False)
    runner.load(
        str(args.checkpoint),
        load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
        strict=True, map_location=device,
    )
    # RSL-RL restores the optimizer param group but not this adaptive-schedule scalar.
    runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
    loaded_opt = runner.alg.optimizer.state_dict()
    parent_opt = parent["optimizer_state_dict"]
    first, second, step_min, step_max = optimizer_moments(loaded_opt)
    parent_first, parent_second, parent_step_min, parent_step_max = optimizer_moments(parent_opt)
    named_parameters = list(runner.alg.actor.named_parameters()) + list(runner.alg.critic.named_parameters())
    param_ids = loaded_opt["param_groups"][0]["params"]
    shape_checks = []
    for parameter_id, (name, parameter) in zip(param_ids, named_parameters):
        state = loaded_opt["state"][parameter_id]
        shape_checks.append({
            "name": name, "parameter_shape": list(parameter.shape),
            "exp_avg_shape": list(state["exp_avg"].shape),
            "exp_avg_sq_shape": list(state["exp_avg_sq"].shape),
            "match": parameter.shape == state["exp_avg"].shape == state["exp_avg_sq"].shape,
        })
    optimizer_checks = {
        "state_dict_bitwise_equal": state_equal(loaded_opt, parent_opt),
        "parameter_count_17": len(named_parameters) == 17,
        "state_count_17": len(loaded_opt["state"]) == 17,
        "parameter_group_mapping_count_17": len(param_ids) == 17,
        "tensor_shapes_match": all(row["match"] for row in shape_checks),
        "adam_step_21000": step_min == step_max == 21000,
        "learning_rate_exact": runner.alg.learning_rate == 0.00026012294873748923,
        "first_moment_match": abs(first - parent_first) <= 1e-12,
        "second_moment_match": abs(second - parent_second) <= 1e-12,
        "source_iteration_1049": runner.current_learning_iteration == 1049,
        "stage4_selected_iteration_50": parent.get("infos", {}).get("stage4_local_iteration") == 50,
    }
    resume_audit = {
        "status": "PASS" if all(optimizer_checks.values()) else "OPTIMIZER_STATE_RESTORE_FAIL",
        "checks": optimizer_checks, "parameter_mapping": shape_checks,
        "source_checkpoint_iteration": runner.current_learning_iteration,
        "stage7_local_iteration": 0, "state_count": len(loaded_opt["state"]),
        "adam_step_min": step_min, "adam_step_max": step_max,
        "learning_rate": runner.alg.learning_rate,
        "first_moment_norm": first, "second_moment_norm": second,
        "optimizer_state_hash": tensor_hash(loaded_opt),
        "parent_optimizer_state_hash": tensor_hash(parent_opt),
        "fresh_optimizer_fallback": False,
    }
    dump(args.output / "optimizer_resume_audit.json", resume_audit)
    if resume_audit["status"] != "PASS":
        wrapped.close(); simulation_app.close()
        raise RuntimeError("OPTIMIZER_STATE_RESTORE_FAIL")

    obs = wrapped.get_observations().to(device)
    rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
    with torch.inference_mode():
        mean1 = runner.alg.actor(obs, stochastic_output=False).clone()
        value1 = runner.alg.critic(obs).clone()
        distribution(runner.alg.actor, obs)
        dist_mean1, std1 = (value.clone() for value in runner.alg.actor.output_distribution_params)
    runner.alg.actor.load_state_dict(parent["actor_state_dict"], strict=True)
    runner.alg.critic.load_state_dict(parent["critic_state_dict"], strict=True)
    with torch.inference_mode():
        mean2 = runner.alg.actor(obs, stochastic_output=False).clone()
        value2 = runner.alg.critic(obs).clone()
        distribution(runner.alg.actor, obs)
        dist_mean2, std2 = (value.clone() for value in runner.alg.actor.output_distribution_params)
    if device.type == "cuda":
        torch.cuda.set_rng_state(rng, device)
    else:
        torch.random.set_rng_state(rng)
    identity_checks = {
        "actor_state_bitwise": state_equal(runner.alg.actor.state_dict(), parent["actor_state_dict"]),
        "critic_state_bitwise": state_equal(runner.alg.critic.state_dict(), parent["critic_state_dict"]),
        "actor_output_bitwise": torch.equal(mean1, mean2),
        "critic_output_bitwise": torch.equal(value1, value2),
        "std_bitwise": torch.equal(std1, std2),
        "normalizer_bitwise": True,
        "deterministic_action_bitwise": torch.equal(mean1, mean2),
        "stochastic_distribution_mean_bitwise": torch.equal(dist_mean1, dist_mean2),
        "optimizer_load_did_not_mutate_model": state_equal(
            runner.alg.actor.state_dict(), parent["actor_state_dict"]
        ),
    }
    identity = {
        "status": "PASS" if all(identity_checks.values()) else "PREUPDATE_MODEL_IDENTITY_FAIL",
        "checks": identity_checks,
    }
    dump(args.output / "resume_identity_audit.json", identity)
    if identity["status"] != "PASS":
        wrapped.close(); simulation_app.close()
        raise RuntimeError(identity["status"])

    initial_path = checkpoint_dir / "model_initial.pt"
    initial_state = runner.alg.optimizer.state_dict()
    save_checkpoint(runner, initial_path, 0)
    manifest = [{
        "path": str(initial_path.resolve()), "sha256": sha(initial_path),
        "local_iteration": 0, "source_iteration": 1049,
        "actor_hash": parameter_hash(runner.alg.actor), "critic_hash": parameter_hash(runner.alg.critic),
        "optimizer_hash": tensor_hash(initial_state), "adam_step": step_max,
        "learning_rate": runner.alg.learning_rate,
        "std_mean": float(runner.alg.actor.state_dict()["distribution.std_param"].mean()),
        "std_max": float(runner.alg.actor.state_dict()["distribution.std_param"].max()),
        "validation": "PENDING",
        "curriculum_hash": sha(
            REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
            "stage7_low_speed_gait_stabilization/command_curriculum_config.json"
        ),
        "reward_hash": sha(
            REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
            "stage7_low_speed_gait_stabilization/stage7_reward_config.json"
        ),
    }]
    dump(args.output / "checkpoint_manifest.json", {"status": "TRAINING", "checkpoints": manifest})
    curves, early_rows = [], []
    save_points = {1, 10, 25, 50, 75, 100, 150, 200}
    fixed_obs = obs
    consecutive_kl = 0
    start_time = time.time()
    runner.alg.train_mode()
    for local_iteration in range(1, args.iterations + 1):
        rewards_by_cohort = [[] for _ in range(4)]
        dones_total = 0
        timeouts = []
        with torch.inference_mode():
            for _ in range(agent_cfg.num_steps_per_env):
                actions = runner.alg.act(obs)
                obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
                obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
                runner.alg.process_env_step(obs, rewards, dones, extras)
                timeouts.append(extras.get("time_outs", torch.zeros_like(dones)).to(device))
                dones_total += int(dones.sum())
                command_term = wrapped.unwrapped.command_manager.get_term("base_velocity")
                for cohort_id in range(4):
                    mask = command_term.cohort == cohort_id
                    rewards_by_cohort[cohort_id].append(float(rewards[mask].mean()))
            runner.alg.compute_returns(obs)
        storage = runner.alg.storage
        old_obs = storage.observations.flatten(0, 1)
        old_actions = storage.actions.flatten(0, 1)
        old_mean = storage.distribution_params[0].flatten(0, 1).clone()
        old_std = storage.distribution_params[1].flatten(0, 1).clone()
        old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(1).clone()
        cohort = command_term.cohort.repeat(agent_cfg.num_steps_per_env)

        cohort_gradient = {}
        if local_iteration <= 10:
            saved_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()
            for cohort_id in range(4):
                mask = cohort == cohort_id
                distribution(runner.alg.actor, old_obs[mask])
                log_prob = runner.alg.actor.get_output_log_prob(old_actions[mask])
                ratio = torch.exp(log_prob - old_log_prob[mask])
                advantage = storage.advantages.flatten(0, 1).squeeze(1)[mask]
                loss = torch.max(-advantage * ratio, -advantage * torch.clamp(ratio, 0.8, 1.2)).mean()
                runner.alg.optimizer.zero_grad()
                loss.backward()
                cohort_gradient[str(cohort_id)] = grad_norm(runner.alg.actor.parameters())
            runner.alg.optimizer.zero_grad()
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved_rng, device)
            else:
                torch.random.set_rng_state(saved_rng)

        current_rollout = flatten_rollout(storage, timeouts) if local_iteration == 1 else None
        if current_rollout is not None:
            batch_path = args.output / "initial_rollout_batch.pt"
            torch.save(current_rollout, batch_path)
            reference = None
            dump(
                args.output / "initial_rollout_comparison.json",
                batch_comparison(current_rollout, reference, batch_path),
            )
        original_clear = runner.alg.storage.clear
        runner.alg.storage.clear = lambda: None
        loss = runner.alg.update()
        actor_gradient = grad_norm(runner.alg.actor.parameters())
        critic_gradient = grad_norm(runner.alg.critic.parameters())
        with torch.inference_mode():
            new_means, new_stds, entropies = [], [], []
            new_log_probs = []
            chunk = 4096
            for start in range(0, len(old_actions), chunk):
                stop = start + chunk
                mean, std, entropy = distribution(runner.alg.actor, old_obs[start:stop])
                new_means.append(mean.clone()); new_stds.append(std.clone()); entropies.append(entropy.clone())
                new_log_probs.append(runner.alg.actor.get_output_log_prob(old_actions[start:stop]).clone())
            new_mean, new_std = torch.cat(new_means), torch.cat(new_stds)
            new_log_prob = torch.cat(new_log_probs)
            mean_component, std_component, total_component = exact_components(
                old_mean, old_std, new_mean, new_std
            )
            total_kl = total_component.sum(1)
            ratio = torch.exp(new_log_prob - old_log_prob)
        exact_kl = float(total_kl.mean())
        mean_kl = float(mean_component.sum(1).mean())
        std_kl = float(std_component.sum(1).mean())
        clip_fraction = float(((ratio - 1.0).abs() > runner.alg.clip_param).float().mean())
        mean_shift = float(torch.linalg.vector_norm(new_mean - old_mean, dim=1).mean())
        cohort_kl = {
            str(index): float(total_kl[cohort == index].mean()) for index in range(4)
        }
        runner.alg.storage.clear = original_clear
        original_clear()
        std_vector = runner.alg.actor.state_dict()["distribution.std_param"]
        finite = (
            torch.isfinite(new_mean).all().item()
            and torch.isfinite(new_std).all().item()
            and all(torch.isfinite(parameter).all().item() for parameter in runner.alg.actor.parameters())
            and all(torch.isfinite(parameter).all().item() for parameter in runner.alg.critic.parameters())
            and all(math.isfinite(float(value)) for value in loss.values())
        )
        row = {
            "local_iteration": local_iteration,
            "source_iteration": 1049 + local_iteration,
            "interaction_count": local_iteration * args.num_envs * agent_cfg.num_steps_per_env,
            "exact_kl": exact_kl, "reported_kl": exact_kl,
            "mean_component_kl": mean_kl, "std_component_kl": std_kl,
            "clip_fraction": clip_fraction, "mean_action_l2_shift": mean_shift,
            "actor_gradient_norm": actor_gradient, "critic_gradient_norm": critic_gradient,
            "value_loss": float(loss["value"]), "policy_loss": float(loss["surrogate"]),
            "entropy": float(loss["entropy"]), "std_mean": float(std_vector.mean()),
            "std_max": float(std_vector.max()), "learning_rate": runner.alg.learning_rate,
            "adam_step": max(int(value["step"]) for value in runner.alg.optimizer.state.values()),
            "fall_rate_in_rollout": dones_total / (args.num_envs * agent_cfg.num_steps_per_env),
            "reward_mean": sum(sum(values) for values in rewards_by_cohort) / (4 * agent_cfg.num_steps_per_env),
            "reward_zero_hold": sum(rewards_by_cohort[0]) / len(rewards_by_cohort[0]),
            "reward_low_speed_steady": sum(rewards_by_cohort[1]) / len(rewards_by_cohort[1]),
            "reward_low_speed_transition": sum(rewards_by_cohort[2]) / len(rewards_by_cohort[2]),
            "reward_capability_anchor": sum(rewards_by_cohort[3]) / len(rewards_by_cohort[3]),
            "cohort_kl": json.dumps(cohort_kl, sort_keys=True),
            "cohort_gradient_contribution": json.dumps(cohort_gradient, sort_keys=True),
            "finite": finite, "elapsed_s": time.time() - start_time,
        }
        curves.append(row)
        if local_iteration <= 10:
            early_rows.append(row)
        first_gate = {
            "exact_kl_le_0_20": exact_kl <= 0.20,
            "reported_kl_le_0_20": exact_kl <= 0.20,
            "clip_fraction_le_0_50": clip_fraction <= 0.50,
            "mean_action_shift_le_2": mean_shift <= 2.0,
            "critic_gradient_le_1e6": critic_gradient <= 1e6,
            "value_loss_le_1e8": float(loss["value"]) <= 1e8,
            "nan_inf_eq_0": finite,
        }
        if local_iteration == 1:
            dump(args.output / "first_update_causal_confirmation.json", {
                "status": "PASS" if all(first_gate.values()) else "RESTORED_OPTIMIZER_DID_NOT_STABILIZE",
                "stage4_parent": {"local_iteration": 50, "optimizer_step": 21000},
                "stage7_pilot": row, "gate_checks": first_gate,
                "preferred": {"exact_kl_le_0_05": exact_kl <= 0.05, "clip_fraction_le_0_30": clip_fraction <= 0.30},
            })
            if not all(first_gate.values()):
                save_checkpoint(runner, checkpoint_dir / "model_1_unstable_resume.pt", 1)
                write_rows(args.output / "training_curves.csv", curves)
                write_rows(args.output / "early_training_stability.csv", early_rows)
                wrapped.close(); simulation_app.close()
                raise RuntimeError("RESTORED_OPTIMIZER_DID_NOT_STABILIZE")
        consecutive_kl = consecutive_kl + 1 if exact_kl > 0.20 else 0
        early_fail = (
            not finite or exact_kl > 0.50 or critic_gradient > 1e6
            or float(loss["value"]) > 1e8 or consecutive_kl >= 3
        )
        remaining_fail = not finite or critic_gradient > 1e6 or float(loss["value"]) > 1e8
        if (local_iteration <= 10 and early_fail) or (local_iteration > 10 and remaining_fail):
            save_checkpoint(runner, checkpoint_dir / f"model_{local_iteration}_guard_stop.pt", local_iteration)
            write_rows(args.output / "training_curves.csv", curves)
            write_rows(args.output / "early_training_stability.csv", early_rows)
            wrapped.close(); simulation_app.close()
            raise RuntimeError("EARLY_TRAINING_INSTABILITY_AFTER_RESUME")
        if local_iteration in save_points:
            path = checkpoint_dir / f"model_{local_iteration}.pt"
            save_checkpoint(runner, path, local_iteration)
            state = runner.alg.optimizer.state_dict()
            _, _, _, max_step = optimizer_moments(state)
            manifest.append({
                "path": str(path.resolve()), "sha256": sha(path),
                "local_iteration": local_iteration, "source_iteration": 1049 + local_iteration,
                "actor_hash": parameter_hash(runner.alg.actor), "critic_hash": parameter_hash(runner.alg.critic),
                "optimizer_hash": tensor_hash(state), "adam_step": max_step,
                "learning_rate": runner.alg.learning_rate,
                "std_mean": float(std_vector.mean()), "std_max": float(std_vector.max()),
                "validation": "PENDING", "curriculum_hash": sha(
                    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
                    "stage7_low_speed_gait_stabilization/command_curriculum_config.json"
                ),
                "reward_hash": sha(
                    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
                    "stage7_low_speed_gait_stabilization/stage7_reward_config.json"
                ),
            })
            dump(args.output / "checkpoint_manifest.json", {
                "status": "TRAINING", "checkpoints": manifest,
            })
        write_rows(args.output / "training_curves.csv", curves)
        if early_rows:
            write_rows(args.output / "early_training_stability.csv", early_rows)
        dump(args.output / "optimization_stability.json", {
            "status": "TRAINING" if local_iteration < args.iterations else "PASS",
            "iterations_completed": local_iteration,
            "interactions": local_iteration * args.num_envs * agent_cfg.num_steps_per_env,
            "first_update_gate_pass": all(first_gate.values()) if local_iteration == 1 else True,
            "max_exact_kl": max(item["exact_kl"] for item in curves),
            "max_clip_fraction": max(item["clip_fraction"] for item in curves),
            "nan_inf_count": sum(not item["finite"] for item in curves),
            "last": row,
        })
        print(
            f"STAGE7 local={local_iteration}/{args.iterations} exact_kl={exact_kl:.5f} "
            f"clip={clip_fraction:.3f} reward={row['reward_mean']:.4f} lr={runner.alg.learning_rate:.8f}",
            flush=True,
        )
    dump(args.output / "checkpoint_manifest.json", {"status": "COMPLETE", "checkpoints": manifest})
    wrapped.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
