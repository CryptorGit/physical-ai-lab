"""Diagnostic-only actor-gradient and checkpoint-alignment analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
STAGE11 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
RAW = OUT / "raw"
CHECKPOINT = STAGE11 / "checkpoints/model_initial.pt"

parser = __import__("argparse").ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app
sys.argv = [sys.argv[0]]

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from tensordict import TensorDict  # noqa: E402

sys.path.insert(0, str(EXP / "src"))
import isaaclab_tasks  # noqa: E402,F401
import go2_bidirectional.stage11_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402

JOINTS = (
    "FL_hip", "FR_hip", "RL_hip", "RR_hip",
    "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh",
    "FL_calf", "FR_calf", "RL_calf", "RR_calf",
)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine(left, right):
    return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-12))


cfg, agent_cfg = resolve_task_config(
    "Isaac-Exp011-Go2-Tangential-Slip-v0", "rsl_rl_cfg_entry_point"
)
cfg.scene.num_envs = 16
cfg.seed = 20272901
if args.device:
    cfg.sim.device = args.device
    agent_cfg.device = args.device
raw = gym.make("Isaac-Exp011-Go2-Tangential-Slip-v0", cfg=cfg)
wrapped = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
agent_cfg = handle_deprecated_rsl_rl_cfg(
    agent_cfg, __import__("importlib.metadata").metadata.version("rsl-rl-lib")
)
runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(
    str(CHECKPOINT),
    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
    strict=True,
    map_location=runner.device,
)
actor = runner.alg.actor
actor.train()
parameters = [(name, parameter) for name, parameter in actor.named_parameters()]
batch = torch.load(RAW / "gradient_batch.pt", map_location="cpu", weights_only=False)
device = torch.device(runner.device)


def policy_observation(tensor):
    """Reconstruct the frozen runner's named 48-D policy observation group."""
    return TensorDict({"policy": tensor}, batch_size=[len(tensor)], device=tensor.device)


def update_direction(observation, action, advantage):
    actor.zero_grad(set_to_none=True)
    observation, action, advantage = (
        observation.to(device), action.to(device), advantage.to(device)
    )
    actor(policy_observation(observation), stochastic_output=True)
    log_prob = actor.get_output_log_prob(action)
    loss = -(log_prob * advantage).mean()
    loss.backward()
    per_layer = {
        name: float(parameter.grad.norm()) if parameter.grad is not None else 0.0
        for name, parameter in parameters
    }
    vector = torch.cat([
        (-parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)).flatten()
        for _, parameter in parameters
    ]).detach()
    return vector, per_layer, float(loss)


obs, action = batch["observation"], batch["action"]
g_base, layer_base, loss_base = update_direction(obs, action, batch["A_base"])
g_slip, layer_slip, loss_slip = update_direction(obs, action, batch["A_slip"])
g_total, layer_total, loss_total = update_direction(obs, action, batch["A_total"])
q_g = float(g_slip.norm() / g_base.norm().clamp_min(1e-12))
base_slip_cosine = cosine(g_base, g_slip)
slip_total_cosine = cosine(g_slip, g_total)
strength = (
    "SLIP_GRADIENT_TOO_WEAK" if q_g < 0.05
    else "SLIP_GRADIENT_MODERATE" if q_g < 0.25 else "SLIP_GRADIENT_STRONG"
)
conflict = (
    "GRADIENT_ALIGNED" if base_slip_cosine >= 0.20
    else "GRADIENT_ORTHOGONAL" if base_slip_cosine > -0.20 else "GRADIENT_CONFLICT"
)
layer_rows = []
for name, _ in parameters:
    layer_rows.append({
        "parameter": name, "base_gradient_norm": layer_base[name],
        "slip_gradient_norm": layer_slip[name], "total_gradient_norm": layer_total[name],
        "slip_to_base_ratio": layer_slip[name] / max(layer_base[name], 1e-12),
    })

# Per-joint output-head contribution.
output_name, output_parameter = next(
    (name, parameter) for name, parameter in reversed(parameters)
    if parameter.ndim == 2 and parameter.shape[0] == 12
)
actor.zero_grad(set_to_none=True)
actor(policy_observation(obs.to(device)), stochastic_output=True)
mean, std = actor.output_distribution_params
joint_log_prob = torch.distributions.Normal(mean, std).log_prob(action.to(device))
for joint in range(12):
    actor.zero_grad(set_to_none=True)
    loss = -(joint_log_prob[:, joint] * batch["A_slip"].to(device)).mean()
    loss.backward(retain_graph=joint < 11)
    layer_rows.append({
        "parameter": f"{output_name}:joint:{JOINTS[joint]}",
        "base_gradient_norm": "",
        "slip_gradient_norm": float(output_parameter.grad[joint].norm()),
        "total_gradient_norm": "",
        "slip_to_base_ratio": "",
    })
write_csv("per_layer_slip_gradient.csv", layer_rows)


def grouped_gradients(values, groups):
    rows = []
    for group in sorted(groups.unique().tolist()):
        mask = groups == group
        if int(mask.sum()) < 32:
            continue
        vector, _, _ = update_direction(obs[mask], action[mask], values[mask])
        rows.append({"group": float(group), "samples": int(mask.sum()), "norm": float(vector.norm()),
                     "cosine_with_global_slip": cosine(vector, g_slip)})
    return rows


speed_gradients = grouped_gradients(batch["A_slip"], batch["target_speed"])
family_gradients = grouped_gradients(batch["A_slip"], batch["family"])
phase_gradients = grouped_gradients(batch["A_slip"], batch["contact_phase"])
dump("actor_gradient_decomposition.json", {
    "fixed_batch_samples": len(obs),
    "gradient_definition": "PPO ratio=1 policy-ascent direction using additive diagnostic GAE components",
    "norms": {
        "g_base": float(g_base.norm()), "g_slip": float(g_slip.norm()),
        "g_total": float(g_total.norm()), "q_g": q_g,
    },
    "cosines": {
        "base_slip": base_slip_cosine, "slip_total": slip_total_cosine,
        "base_total": cosine(g_base, g_total),
    },
    "strength_classification": strength,
    "conflict_classification": conflict,
    "losses": {"base": loss_base, "slip": loss_slip, "total": loss_total},
    "by_speed": speed_gradients, "by_family": family_gradients,
    "by_contact_phase": phase_gradients,
})

# 100 fixed permutations at the Stage 11 mini-batch size.
num_mini_batches = int(agent_cfg.algorithm.num_mini_batches)
mini_size = len(obs) // num_mini_batches
mini_vectors, mini_rows = [], []
for permutation_index in range(100):
    generator = torch.Generator().manual_seed(20272901 + permutation_index)
    index = torch.randperm(len(obs), generator=generator)[:mini_size]
    vector, _, _ = update_direction(obs[index], action[index], batch["A_slip"][index])
    mini_vectors.append(vector.cpu())
    dominant_foot = int(batch["foot_speed"][index].mean(0).argmax())
    speeds, counts = batch["target_speed"][index].unique(return_counts=True)
    dominant_speed = float(speeds[counts.argmax()])
    mini_rows.append({
        "permutation": permutation_index, "samples": len(index),
        "slip_gradient_norm": float(vector.norm()),
        "cosine_with_full": cosine(vector, g_slip),
        "dominant_foot": JOINTS[dominant_foot].split("_")[0],
        "dominant_speed": dominant_speed,
    })
matrix = torch.stack(mini_vectors)
matrix = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
pairwise = matrix @ matrix.T
upper = pairwise[torch.triu(torch.ones(100, 100, dtype=torch.bool), diagonal=1)]
pairwise_median = float(upper.median())
consistency = (
    "SLIP_GRADIENT_CONSISTENT" if pairwise_median >= 0.30
    else "SLIP_GRADIENT_NOISY" if pairwise_median < 0.10 else "MIXED"
)
dump("minibatch_gradient_consistency.json", {
    "permutations": 100, "mini_batch_size": mini_size,
    "pairwise_slip_gradient_cosine_median": pairwise_median,
    "classification": consistency, "rows": mini_rows,
})

# Checkpoint parameter trajectory and fixed-observation action drift.
manifest = json.loads((STAGE11 / "checkpoint_manifest.json").read_text(encoding="utf-8"))
initial_state = {
    name: parameter.detach().cpu().clone() for name, parameter in actor.named_parameters()
}
initial_vector = torch.cat([initial_state[name].flatten() for name, _ in parameters]).to(device)
with torch.inference_mode():
    initial_obs = obs[:4096].to(device)
    initial_action = actor(policy_observation(initial_obs), stochastic_output=False).cpu()
checkpoint_rows, alignment_rows = [], []
for item in manifest["checkpoints"]:
    runner.load(
        item["path"],
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
        strict=True, map_location=runner.device,
    )
    current_vector = torch.cat([
        dict(actor.named_parameters())[name].detach().flatten() for name, _ in parameters
    ])
    delta = current_vector - initial_vector
    with torch.inference_mode():
        current_obs = obs[:4096].to(device)
        current_action = actor(policy_observation(current_obs), stochastic_output=False).cpu()
    delta_norm = float(delta.norm())
    slip_alignment = cosine(g_slip, delta) if delta_norm > 0 else 0.0
    base_alignment = cosine(g_base, delta) if delta_norm > 0 else 0.0
    validation = item.get("validation", {})
    row = {
        "iteration": item["iteration"], "checkpoint": item["path"],
        "sha256": item["sha256"], "actor_hash": item["actor_hash"],
        "critic_hash": item["critic_hash"], "optimizer_step": item["adam_step"],
        "learning_rate": item["learning_rate"], "std": item.get("std", ""),
        "action_l2_from_initial": float((current_action - initial_action).norm(dim=1).mean()),
        "parameter_delta_norm": delta_norm,
        "slip_gradient_alignment": slip_alignment,
        "base_gradient_alignment": base_alignment,
        "g_slip_dot_delta": float(torch.dot(g_slip, delta)),
        "g_base_dot_delta": float(torch.dot(g_base, delta)),
        "validation_hard_pass_count": validation.get("hard_pass_count", ""),
        "validation_dangerous_slip": validation.get("mean_dangerous_slip_episode_rate", ""),
        "validation_tangential_p95": validation.get("mean_tangential_speed_p95", ""),
        "selected": item.get("selected", False),
    }
    checkpoint_rows.append(row)
    alignment_rows.append({
        key: row[key] for key in (
            "iteration", "parameter_delta_norm", "slip_gradient_alignment",
            "base_gradient_alignment", "g_slip_dot_delta", "g_base_dot_delta",
        )
    })
write_csv("checkpoint_action_drift.csv", checkpoint_rows)
positive = [row for row in alignment_rows if row["iteration"] > 0 and row["slip_gradient_alignment"] > 0]
dump("training_gradient_alignment.json", {
    "definition": "alignment of diagnostic policy-ascent direction at initial with theta_k - theta_0",
    "trajectory": alignment_rows,
    "positive_slip_alignment_checkpoints": len(positive),
    "early_positive_then_cancelled": bool(
        positive and alignment_rows[-1]["slip_gradient_alignment"] <= 0
    ),
    "selected_initial_reason": "trained checkpoints did not improve the pre-registered validation precedence",
})

wrapped.close()
simulation_app.close()
print(json.dumps({
    "q_g": q_g, "strength": strength, "conflict": conflict,
    "pairwise_median": pairwise_median, "consistency": consistency,
}, indent=2))
