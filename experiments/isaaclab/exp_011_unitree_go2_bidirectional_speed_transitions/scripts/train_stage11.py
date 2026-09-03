"""Train the Stage 11 single-reward Pilot after all frozen preflight gates pass."""

from __future__ import annotations

import argparse
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
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
PARENT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"

preflight = json.loads((OUT / "preflight_gate.json").read_text(encoding="utf-8"))
if preflight["status"] != "PASS":
    raise SystemExit(f"Stage 11 Pilot prohibited: {preflight['status']}")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value):
    digest = hashlib.sha256()
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            digest.update(str(key).encode()); digest.update(tensor_hash(item).encode())
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(tensor_hash(item).encode())
    elif torch.is_tensor(value):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    else:
        digest.update(repr(value).encode())
    return digest.hexdigest()


def grad_norm(parameters):
    return math.sqrt(sum(
        float(parameter.grad.detach().square().sum())
        for parameter in parameters if parameter.grad is not None
    ))


def distribution(actor, observations):
    actor(observations, stochastic_output=True)
    mean, std = actor.output_distribution_params
    return mean, std


def save_checkpoint(runner, path, local_iteration):
    payload = runner.alg.save()
    payload["iter"] = 1099 + local_iteration
    payload["infos"] = {
        "source_checkpoint_iteration": 1099,
        "stage7_selected_iteration": 50,
        "stage11_local_iteration": local_iteration,
        "optimizer_restored": True,
        "single_reward_addition": "go2_contact_tangential_slip",
    }
    torch.save(payload, path)


def write_curves(rows):
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


calibration = json.loads((OUT / "slip_reward_calibration.json").read_text(encoding="utf-8"))
weight = float(calibration["lambda_slip"])
cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = args.num_envs
cfg.seed = 20261001
cfg.rewards.go2_contact_tangential_slip.weight = weight
agent_cfg.seed = 20261001
agent_cfg.max_iterations = args.iterations
if args.device:
    cfg.sim.device = args.device
    agent_cfg.device = args.device
raw = gym.make("Isaac-Exp011-Go2-Tangential-Slip-v0", cfg=cfg)
wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
)
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
device = torch.device(runner.device)
runner.load(
    str(PARENT), load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
    strict=True, map_location=device,
)
runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
term = wrapped.unwrapped.reward_manager.get_term_cfg("go2_contact_tangential_slip").func
command = wrapped.unwrapped.command_manager.get_term("base_velocity")
obs = wrapped.get_observations().to(device)
checkpoint_dir = OUT / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)
manifest, curves = [], []
save_points = {1, 10, 25, 50, 75, 100, 150, 200}
initial = checkpoint_dir / "model_initial.pt"
save_checkpoint(runner, initial, 0)
manifest.append({
    "path": str(initial.resolve()), "sha256": sha(initial), "iteration": 0,
    "actor_hash": tensor_hash(runner.alg.actor.state_dict()),
    "critic_hash": tensor_hash(runner.alg.critic.state_dict()),
    "optimizer_hash": tensor_hash(runner.alg.optimizer.state_dict()),
    "adam_step": max(int(x["step"]) for x in runner.alg.optimizer.state.values()),
    "learning_rate": runner.alg.learning_rate,
    "reward_hash": sha(OUT / "stage11_reward_config.json"),
    "curriculum_hash": sha(OUT / "stage7_vs_stage11_curriculum_diff.json"),
    "heading_controller_hash": json.loads((OUT / "phase_gated_heading_hash.json").read_text())["sha256"],
    "validation": "PENDING",
})
start_time = time.time()
consecutive_kl = 0
runner.alg.train_mode()
for iteration in range(1, args.iterations + 1):
    raw_sum = weighted_sum = tangent_sum = friction_sum = 0.0
    stable_sum = flight_sum = falls = 0.0
    foot_score = torch.zeros(4, device=device)
    speed_error_sum = 0.0
    with torch.inference_mode():
        for _ in range(agent_cfg.num_steps_per_env):
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
            obs, rewards, dones = obs.to(device), rewards.to(device), dones.to(device)
            runner.alg.process_env_step(obs, rewards, dones, extras)
            score = term.last_raw_score
            raw_sum += float(score.mean())
            weighted_sum += float((-weight * score).mean())
            tangent_sum += float(term.last_foot_speed.mean())
            stable = term.last_stable.float()
            stable_sum += float(stable.mean())
            flight_sum += float((stable.sum(1) == 0).float().mean())
            foot_score += (stable * term.last_foot_speed).mean(0)
            robot_speed = wrapped.unwrapped.scene["robot"].data.root_lin_vel_b.torch[:, 0]
            speed_error_sum += float((robot_speed - command.vel_command_b[:, 0]).abs().mean())
            falls += int(dones.sum())
        runner.alg.compute_returns(obs)
    storage = runner.alg.storage
    old_obs = storage.observations.flatten(0, 1)
    old_actions = storage.actions.flatten(0, 1)
    old_mean = storage.distribution_params[0].flatten(0, 1).clone()
    old_std = storage.distribution_params[1].flatten(0, 1).clone()
    old_log_prob = storage.actions_log_prob.flatten(0, 1).squeeze(1).clone()
    original_clear = storage.clear
    storage.clear = lambda: None
    loss = runner.alg.update()
    actor_gradient = grad_norm(runner.alg.actor.parameters())
    critic_gradient = grad_norm(runner.alg.critic.parameters())
    means, stds, log_probs = [], [], []
    with torch.inference_mode():
        for begin in range(0, len(old_obs), 4096):
            mean, std = distribution(runner.alg.actor, old_obs[begin:begin + 4096])
            means.append(mean.clone()); stds.append(std.clone())
            log_probs.append(runner.alg.actor.get_output_log_prob(old_actions[begin:begin + 4096]).clone())
    new_mean, new_std = torch.cat(means), torch.cat(stds)
    new_log_prob = torch.cat(log_probs)
    mean_component = (old_mean - new_mean).square() / (2.0 * new_std.square())
    std_component = torch.log(new_std / old_std) + old_std.square() / (2.0 * new_std.square()) - 0.5
    exact_kl = float((mean_component + std_component).sum(1).mean())
    ratio = torch.exp(new_log_prob - old_log_prob)
    clip_fraction = float(((ratio - 1.0).abs() > runner.alg.clip_param).float().mean())
    mean_shift = float(torch.linalg.vector_norm(new_mean - old_mean, dim=1).mean())
    storage.clear = original_clear
    original_clear()
    finite = all(torch.isfinite(parameter).all() for parameter in runner.alg.actor.parameters()) and all(
        torch.isfinite(parameter).all() for parameter in runner.alg.critic.parameters()
    ) and all(math.isfinite(float(value)) for value in loss.values())
    steps = agent_cfg.num_steps_per_env
    row = {
        "iteration": iteration,
        "interactions": iteration * args.num_envs * steps,
        "exact_kl": exact_kl, "reported_kl": exact_kl,
        "clip_fraction": clip_fraction, "mean_action_shift": mean_shift,
        "actor_gradient": actor_gradient, "critic_gradient": critic_gradient,
        "value_loss": float(loss["value"]), "policy_loss": float(loss["surrogate"]),
        "entropy": float(loss["entropy"]),
        "std_mean": float(runner.alg.actor.state_dict()["distribution.std_param"].mean()),
        "raw_slip_score": raw_sum / steps, "weighted_slip_reward": weighted_sum / steps,
        "tangential_speed": tangent_sum / steps,
        "stable_contact_fraction": stable_sum / steps, "flight_fraction": flight_sum / steps,
        "speed_mae": speed_error_sum / steps,
        "fall_rate": falls / (args.num_envs * steps),
        "foot_score": json.dumps((foot_score / steps).cpu().tolist()),
        "learning_rate": runner.alg.learning_rate,
        "adam_step": max(int(x["step"]) for x in runner.alg.optimizer.state.values()),
        "finite": bool(finite), "elapsed_s": time.time() - start_time,
    }
    curves.append(row)
    first_gate = (
        exact_kl <= 0.20 and clip_fraction <= 0.50 and mean_shift <= 2.0
        and critic_gradient <= 1e6 and float(loss["value"]) <= 1e8 and finite
    )
    if iteration == 1 and not first_gate:
        save_checkpoint(runner, checkpoint_dir / "model_1_unstable.pt", 1)
        write_curves(curves)
        dump("optimization_stability.json", {"status": "STAGE11_OPTIMIZATION_UNSTABLE", "first_update": row})
        wrapped.close(); simulation_app.close()
        raise RuntimeError("STAGE11_OPTIMIZATION_UNSTABLE")
    consecutive_kl = consecutive_kl + 1 if exact_kl > 0.20 else 0
    baseline_row = curves[0]
    exploitation = (
        row["flight_fraction"] > 2.0 * max(baseline_row["flight_fraction"], 1.0e-6)
        or row["speed_mae"] > baseline_row["speed_mae"] + 0.15
        or row["stable_contact_fraction"] < 0.5 * baseline_row["stable_contact_fraction"]
        or float((foot_score / foot_score.sum().clamp_min(1e-12)).max()) > 0.70
    )
    if not finite or critic_gradient > 1e6 or float(loss["value"]) > 1e8 or consecutive_kl >= 3 or exploitation:
        save_checkpoint(runner, checkpoint_dir / f"model_{iteration}_guard_stop.pt", iteration)
        write_curves(curves)
        classification = "TANGENTIAL_SLIP_REWARD_EXPLOITATION" if exploitation else "STAGE11_OPTIMIZATION_UNSTABLE"
        dump("optimization_stability.json", {"status": classification, "last": row})
        wrapped.close(); simulation_app.close()
        raise RuntimeError(classification)
    if iteration in save_points:
        path = checkpoint_dir / f"model_{iteration}.pt"
        save_checkpoint(runner, path, iteration)
        manifest.append({
            "path": str(path.resolve()), "sha256": sha(path), "iteration": iteration,
            "actor_hash": tensor_hash(runner.alg.actor.state_dict()),
            "critic_hash": tensor_hash(runner.alg.critic.state_dict()),
            "optimizer_hash": tensor_hash(runner.alg.optimizer.state_dict()),
            "adam_step": row["adam_step"], "learning_rate": row["learning_rate"],
            "std": row["std_mean"], "validation": "PENDING",
            "reward_hash": sha(OUT / "stage11_reward_config.json"),
            "curriculum_hash": sha(OUT / "stage7_vs_stage11_curriculum_diff.json"),
            "heading_controller_hash": json.loads((OUT / "phase_gated_heading_hash.json").read_text())["sha256"],
        })
        dump("checkpoint_manifest.json", {"status": "TRAINING", "checkpoints": manifest})
    write_curves(curves)
    dump("optimization_stability.json", {
        "status": "TRAINING" if iteration < args.iterations else "PASS",
        "iterations_completed": iteration,
        "first_update_gate_pass": True,
        "max_exact_kl": max(item["exact_kl"] for item in curves),
        "max_clip_fraction": max(item["clip_fraction"] for item in curves),
        "nan_inf_count": sum(not item["finite"] for item in curves),
        "last": row,
    })
    dump("slip_reward_behavior_audit.json", {
        "status": "PASS", "reward_exploitation": False,
        "flight_fraction": row["flight_fraction"], "stable_contact_fraction": row["stable_contact_fraction"],
        "speed_mae": row["speed_mae"], "per_foot_score": json.loads(row["foot_score"]),
    })
    print(f"STAGE11 {iteration}/{args.iterations} KL={exact_kl:.5f} slip={row['raw_slip_score']:.4f}", flush=True)
dump("checkpoint_manifest.json", {"status": "COMPLETE", "checkpoints": manifest})
wrapped.close()
simulation_app.close()
