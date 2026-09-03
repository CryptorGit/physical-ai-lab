"""Torch-only full-density gradient and Adam-direction audit for Stage 2F."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2f_phase_a_boundary_diagnosis"
TRACE = OUT / "raw/iter50_s100_event_trace.pt"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"


class Actor(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.std = torch.nn.Parameter(state["distribution.std_param"].clone())
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(123, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 37),
        )
        mapped = {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")}
        self.mlp.load_state_dict(mapped, strict=True)

    def log_prob(self, observation, action):
        mean = self.mlp(observation)
        std = torch.clamp(self.std, min=1e-8)
        return (-0.5 * (((action - mean) / std) ** 2 + 2 * torch.log(std) + math_log_2pi())).sum(-1)


def math_log_2pi():
    return 1.8378770664093453


def discounted(reward, gamma=0.99):
    result = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for index in range(reward.shape[0] - 1, -1, -1):
        running = reward[index] + gamma * running
        result[index] = running
    return result


def cosine(left, right):
    return float(torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-12))


def gradient(actor, observations, actions, old_logp, reward):
    advantage = discounted(reward).flatten()
    advantage = advantage - advantage.mean()
    logp = actor.log_prob(observations.flatten(0, 1), actions.flatten(0, 1))
    loss = -(advantage.detach() * torch.exp(logp - old_logp.flatten())).mean()
    values = torch.autograd.grad(loss, list(actor.parameters()), allow_unused=True)
    parts = [torch.zeros_like(parameter) if value is None else value for parameter, value in zip(actor.parameters(), values)]
    return torch.cat([value.reshape(-1) for value in parts]).detach(), parts


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trace = torch.load(TRACE, map_location=device, weights_only=False)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    actor = Actor(checkpoint["actor_state_dict"]).to(device)
    stored_observation = trace["observation"]
    observations = stored_observation["policy"].to(device) if hasattr(stored_observation, "keys") else stored_observation.to(device)
    actions = trace["action"].to(device)
    old_logp = trace["old_logp"].to(device)
    components = trace["reward_components"].to(device)
    run_index = trace["reward_names"].index("safe_periodic_flight")
    run = components[:, :, run_index]
    completion = torch.where(run >= 1.0, run, 0.0)
    precursor = run - completion
    base = components.sum(-1) - run
    rewards = {
        "base": base, "precursor": precursor, "completion": completion,
        "run_specific": run, "total": components.sum(-1),
    }
    vectors, parameter_parts = {}, {}
    for name, reward in rewards.items():
        vectors[name], parameter_parts[name] = gradient(actor, observations, actions, old_logp, reward)
    base_norm = torch.linalg.vector_norm(vectors["base"])
    total_norm = torch.linalg.vector_norm(vectors["total"])
    summary = {
        "scope": "full_500_step_x_350_environment_S100_rollout",
        "samples": int(observations.shape[0] * observations.shape[1]),
        "completion_event_samples": int((completion > 0).sum()),
        "completion_density": float((completion > 0).float().mean()),
        "components": {},
    }
    for name, vector in vectors.items():
        summary["components"][name] = {
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "ratio_to_base": float(torch.linalg.vector_norm(vector) / (base_norm + 1e-12)),
            "ratio_to_total": float(torch.linalg.vector_norm(vector) / (total_norm + 1e-12)),
            "cosine_to_base": cosine(vector, vectors["base"]),
            "cosine_to_total": cosine(vector, vectors["total"]),
        }
    density_rows = []
    for factor in (1, 2, 4, 8, 16):
        virtual_reward = base + precursor + factor * completion
        virtual_vector, _ = gradient(actor, observations, actions, old_logp, virtual_reward)
        scaled_completion = factor * vectors["completion"]
        ratio = float(torch.linalg.vector_norm(scaled_completion) / (torch.linalg.vector_norm(virtual_vector) + 1e-12))
        density_rows.append({
            "completion_replication_factor": factor,
            "observed_completion_density": summary["completion_density"],
            "virtual_completion_density": min(1.0, factor * summary["completion_density"]),
            "completion_gradient_to_total": ratio,
            "completion_direction_projection": cosine(virtual_vector, vectors["completion"]),
            "base_completion_cosine": cosine(vectors["base"], vectors["completion"]),
            "reaches_one_percent": ratio >= .01,
        })
    optimizer_state = checkpoint["optimizer_state_dict"]
    state_values = list(optimizer_state["state"].values())[:len(list(actor.parameters()))]
    restored_parts = []
    eps = optimizer_state["param_groups"][0].get("eps", 1e-8)
    for parameter, state in zip(actor.parameters(), state_values):
        restored_parts.append(-state["exp_avg"] / (state["exp_avg_sq"].sqrt() + eps))
    restored = torch.cat([value.reshape(-1) for value in restored_parts])
    total = vectors["total"]
    update_rows = []
    for name, update in (
        ("restored_adam", restored),
        ("zero_moment_adam", -total / (total.abs() + eps)),
        ("raw_sgd", -total),
    ):
        update_rows.append({
            "update_direction": name, "step_norm": float(torch.linalg.vector_norm(update)),
            "cosine_to_completion_descent": cosine(update, -vectors["completion"]),
            "cosine_to_total_descent": cosine(update, -vectors["total"]),
            "cosine_to_base_descent": cosine(update, -vectors["base"]),
        })
    (OUT / "full_density_gradient_runtime.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for name, rows in (
        ("completion_density_gradient_scaling_runtime.csv", density_rows),
        ("shadow_update_direction_comparison_runtime.csv", update_rows),
    ):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
