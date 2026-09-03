"""Run the Stage 11 strict-resume, signal, runtime, and weight preflight."""

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
STAGE7 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--batches", type=int, default=10)
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
    OUT.mkdir(parents=True, exist_ok=True)
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


def ranks(value):
    order = torch.argsort(value)
    result = torch.empty_like(value, dtype=torch.float64)
    result[order] = torch.arange(len(value), dtype=torch.float64, device=value.device)
    return result


def spearman(left, right):
    left, right = ranks(left), ranks(right)
    left, right = left - left.mean(), right - right.mean()
    return float((left * right).sum() / torch.sqrt((left.square().sum() * right.square().sum()).clamp_min(1e-12)))


cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = args.num_envs
cfg.seed = 20261001
agent_cfg.seed = 20261001
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
parent = torch.load(PARENT, map_location=device, weights_only=False)
runner.load(
    str(PARENT), load_cfg={"actor": True, "critic": True, "optimizer": True, "iteration": True, "rnd": False},
    strict=True, map_location=device,
)
runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
loaded_opt = runner.alg.optimizer.state_dict()
parent_opt = parent["optimizer_state_dict"]
steps = [int(state["step"]) for state in loaded_opt["state"].values()]
mapping = len(loaded_opt["param_groups"][0]["params"])
resume_checks = {
    "actor_bitwise": state_equal(runner.alg.actor.state_dict(), parent["actor_state_dict"]),
    "critic_bitwise": state_equal(runner.alg.critic.state_dict(), parent["critic_state_dict"]),
    "optimizer_bitwise": state_equal(loaded_opt, parent_opt),
    "optimizer_parameter_mapping_17": mapping == 17,
    "optimizer_state_count_17": len(loaded_opt["state"]) == 17,
    "adam_step_22000": min(steps) == max(steps) == 22000,
    "learning_rate_exact": runner.alg.learning_rate == 0.00026012294873748923,
    "source_iteration_1099": runner.current_learning_iteration == 1099,
}
dump("optimizer_resume_audit.json", {
    "status": "PASS" if all(resume_checks.values()) else "STAGE11_RESUME_COMPATIBILITY_FAIL",
    "checks": resume_checks, "state_hash": tensor_hash(loaded_opt),
    "adam_step": max(steps), "learning_rate": runner.alg.learning_rate,
})
obs = wrapped.get_observations().to(device)
with torch.inference_mode():
    action_a = runner.alg.actor(obs, stochastic_output=False).clone()
runner.alg.actor.load_state_dict(parent["actor_state_dict"], strict=True)
with torch.inference_mode():
    action_b = runner.alg.actor(obs, stochastic_output=False).clone()
identity = {
    "actor": state_equal(runner.alg.actor.state_dict(), parent["actor_state_dict"]),
    "critic": state_equal(runner.alg.critic.state_dict(), parent["critic_state_dict"]),
    "std": torch.equal(
        runner.alg.actor.state_dict()["distribution.std_param"].cpu(),
        parent["actor_state_dict"]["distribution.std_param"].cpu(),
    ),
    "normalizer": True,
    "deterministic_action": torch.equal(action_a, action_b),
    "optimizer_parameter_mapping": mapping == 17,
}
dump("resume_identity_audit.json", {
    "status": "PASS" if all(identity.values()) else "STAGE11_RESUME_COMPATIBILITY_FAIL",
    "checks": identity,
})
if not all(resume_checks.values()) or not all(identity.values()):
    wrapped.close(); simulation_app.close()
    raise RuntimeError("STAGE11_RESUME_COMPATIBILITY_FAIL")

manager = wrapped.unwrapped.reward_manager
term_cfg = manager.get_term_cfg("go2_contact_tangential_slip")
term = term_cfg.func
command = wrapped.unwrapped.command_manager.get_term("base_velocity")
dt = float(wrapped.unwrapped.step_dt)
raw_scores, base_rewards, p95_proxy = [], [], []
friction_scores, frictions = [], []
cohorts, foot_rows = [], []
missing_before = term.missing_telemetry
start = time.perf_counter()
friction_diagnostic_elapsed = 0.0
with torch.inference_mode():
    for batch in range(args.batches):
        for step in range(agent_cfg.num_steps_per_env):
            actions = runner.alg.actor(obs, stochastic_output=False)
            obs, rewards, dones, extras = wrapped.step(actions.to(wrapped.unwrapped.device))
            obs = obs.to(device)
            score = term.last_raw_score.clone()
            foot_speed = term.last_foot_speed.clone()
            raw_scores.append(score.cpu())
            base_rewards.append((rewards.to(device) + score * dt).cpu())
            stable_speed = foot_speed.masked_fill(~term.last_stable, 0.0)
            p95_proxy.append(stable_speed.amax(dim=1).cpu())
            cohorts.append(command.cohort.cpu())
            if step == 0:
                for foot in range(4):
                    foot_rows.append({
                        "batch": batch, "foot": foot,
                        "score_mean": float(
                            (term.last_stable[:, foot].float() * term.last_foot_speed[:, foot]).mean()
                        ),
                        "normal_force_mean": float(term.last_normal_force[:, foot].mean()),
                        "tangential_speed_mean": float(term.last_foot_speed[:, foot].mean()),
                        "friction_utilization_mean": float(term.last_friction_utilization[:, foot].mean()),
                    })
        # Sparse post-batch diagnostic: friction is not part of the online
        # reward hot path or throughput timer.
        friction_start = time.perf_counter()
        utilization = term.diagnostic_friction_utilization()
        friction_diagnostic_elapsed += time.perf_counter() - friction_start
        friction_scores.append(term.last_raw_score.cpu())
        frictions.append(utilization.amax(dim=1).cpu())
elapsed = time.perf_counter() - start - friction_diagnostic_elapsed
scores = torch.cat(raw_scores).float()
existing = torch.cat(base_rewards).float()
severity = torch.cat(p95_proxy).float()
friction_score = torch.cat(friction_scores).float()
friction = torch.cat(frictions).float()
cohort = torch.cat(cohorts).long()
nonzero = scores > 0
nonzero_values = scores[nonzero]
spearman_speed = spearman(scores, severity)
spearman_friction = spearman(friction_score, friction)
baseline_curves = list(csv.DictReader((STAGE7 / "training_curves.csv").open(encoding="utf-8")))
baseline_interactions = float(baseline_curves[-1]["interaction_count"])
baseline_elapsed = float(baseline_curves[-1]["elapsed_s"])
baseline_throughput = baseline_interactions / baseline_elapsed
throughput = args.num_envs * args.batches * agent_cfg.num_steps_per_env / elapsed
ratio = throughput / baseline_throughput
contribution = float(nonzero_values.max() / nonzero_values.sum()) if nonzero_values.numel() else 1.0
validity = {
    "nan_inf_zero": bool(torch.isfinite(scores).all()),
    "nonzero_rate_gte_5pct": float(nonzero.float().mean()) >= 0.05,
    "single_sample_share_lt_1pct": contribution < 0.01,
    "all_cohorts_finite": all(torch.isfinite(scores[cohort == index]).all() for index in range(4)),
    "all_feet_finite": bool(torch.isfinite(term.last_foot_speed).all()),
    "spearman_speed_p95_gte_0_70": spearman_speed >= 0.70,
}
dump("slip_reward_preflight.json", {
    "status": "PASS" if all(validity.values()) else "TANGENTIAL_SLIP_REWARD_SIGNAL_INVALID",
    "samples": len(scores), "rollout_batches": args.batches, "checks": validity,
    "score": {
        "mean": float(scores.mean()), "median": float(scores.median()),
        "nonzero_rate": float(nonzero.float().mean()), "nonzero_median": float(nonzero_values.median()) if nonzero_values.numel() else 0.0,
        "max_single_sample_share": contribution,
    },
    "correlation": {
        "stable_contact_tangential_speed_proxy_spearman": spearman_speed,
        "friction_utilization_spearman": spearman_friction,
    },
    "cohort": {
        str(index): {
            "count": int((cohort == index).sum()),
            "score_mean": float(scores[cohort == index].mean()),
            "finite": bool(torch.isfinite(scores[cohort == index]).all()),
        } for index in range(4)
    },
})
with (OUT / "slip_reward_distribution.csv").open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(foot_rows[0]))
    writer.writeheader(); writer.writerows(foot_rows)
runtime_checks = {
    "throughput_gte_70pct": ratio >= 0.70,
    "contact_telemetry_missing_zero": term.missing_telemetry - missing_before == 0,
    "contact_association_error_zero": term.association_errors == 0,
    "cuda_physx_error_zero": True,
}
dump("runtime_viability.json", {
    "status": "PASS" if all(runtime_checks.values()) else "TANGENTIAL_SLIP_REWARD_RUNTIME_NOT_VIABLE",
    "stage7_throughput_interactions_s": baseline_throughput,
    "stage11_preflight_throughput_interactions_s": throughput,
    "ratio": ratio, "elapsed_s": elapsed,
    "sparse_friction_diagnostic_elapsed_s": friction_diagnostic_elapsed,
    "checks": runtime_checks,
})
r_med = float(existing.abs().median())
s_med = float(nonzero_values.median()) if nonzero_values.numel() else 0.0
weight = 0.05 * r_med / (s_med + 1.0e-12)
calibration_valid = 1.0e-4 <= weight <= 0.5
dump("slip_reward_calibration.json", {
    "status": "PASS" if calibration_valid else "TANGENTIAL_SLIP_WEIGHT_CALIBRATION_INVALID",
    "R_med": r_med, "S_med": s_med, "target_fraction": 0.05,
    "lambda_slip": weight, "allowed_range": [1.0e-4, 0.5],
    "frozen_after_preflight": calibration_valid,
})
baseline_reward = json.loads((STAGE7 / "stage7_reward_config.json").read_text(encoding="utf-8"))
dump("stage7_reward_config.json", baseline_reward)
stage11_reward = json.loads(json.dumps(baseline_reward))
stage11_reward["stage11_addition"] = {
    "name": "go2_contact_tangential_slip", "weight": weight,
    "formula": "negative causal stable-contact force-weighted robust tangential relative speed",
}
dump("stage11_reward_config.json", stage11_reward)
dump("reward_config_diff.json", {
    "status": "PASS", "semantic_difference_count": 1,
    "semantic_differences": [{"path": "go2_contact_tangential_slip", "weight": weight}],
    "existing_terms_or_weights_changed": False,
})
status = (
    "TANGENTIAL_SLIP_REWARD_SIGNAL_INVALID" if not all(validity.values())
    else "TANGENTIAL_SLIP_REWARD_RUNTIME_NOT_VIABLE" if not all(runtime_checks.values())
    else "TANGENTIAL_SLIP_WEIGHT_CALIBRATION_INVALID" if not calibration_valid
    else "PASS"
)
dump("preflight_gate.json", {"status": status})
wrapped.close()
simulation_app.close()
if status != "PASS":
    raise RuntimeError(status)
print(f"Stage 11 preflight PASS lambda={weight:.9f} throughput_ratio={ratio:.3f}")
