"""Build event strata and run disposable M0/M4/M8/M16 PPO shadow updates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2g_event_stratified_on_policy_preflight"
RAW = OUT / "raw"
CHECKPOINT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2e_phase_a_run_acquisition_preflight/checkpoints/model_50.pt"
CONDITIONS = {"M0_UNIFORM": 1, "M4_EVENT_STRATIFIED": 4, "M8_EVENT_STRATIFIED": 8, "M16_EVENT_STRATIFIED": 16}
EPOCHS = 5
MINIBATCHES = 4
MINIBATCH_SIZE = 6144
EPOCH_SAMPLES = MINIBATCHES * MINIBATCH_SIZE
SEED = 20268121
parser = argparse.ArgumentParser()
parser.add_argument("--batch-index", type=int, default=0)
parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
parser.add_argument("--output-prefix", default="")
parser.add_argument("--analysis-seed", type=int, default=SEED)
args = parser.parse_args()
CORPUS_PATH = RAW / f"on_policy_batch_{args.batch_index}.pt"
ACTIVE_CONDITIONS = {name: CONDITIONS[name] for name in args.conditions}


def dump(name, value):
    (OUT / f"{args.output_prefix}{name}").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_csv(name, rows):
    rows = list(rows)
    with (OUT / f"{args.output_prefix}{name}").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows or [{"status": "NO_ROWS"}])


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
        self.mlp.load_state_dict(
            {key.removeprefix("mlp."): value for key, value in state.items() if key.startswith("mlp.")},
            strict=True,
        )

    def mean(self, observation):
        return self.mlp(observation)

    def log_prob(self, observation, action):
        mean = self.mean(observation)
        std = torch.clamp(self.std, min=1e-8)
        return (-.5 * (((action - mean) / std) ** 2 + 2 * torch.log(std) + 1.8378770664093453)).sum(-1)

    def entropy(self):
        return (torch.log(torch.clamp(self.std, min=1e-8)) + 1.4189385332046727).sum()


class Critic(torch.nn.Module):
    def __init__(self, state):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(123, 256), torch.nn.ELU(),
            torch.nn.Linear(256, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 128), torch.nn.ELU(),
            torch.nn.Linear(128, 1),
        )
        self.mlp.load_state_dict(
            {key.removeprefix("mlp."): value for key, value in state.items()}, strict=True
        )

    def forward(self, observation):
        return self.mlp(observation).squeeze(-1)


def cosine(left, right):
    return float(torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right) + 1e-12))


def command_segment(time_value):
    if time_value < 1.0:
        return 0
    if time_value < 2.0:
        return 1
    if time_value < 3.5:
        return 2
    if time_value < 5.0:
        return 3
    return 4


def build_windows(data):
    steps, envs = data["completion"].shape
    episode = data["episode_id"]
    takeoff = data["takeoff"]
    completion = data["completion"]
    precursor_event = data["precursor"] | data["safe_flight"]
    e2 = torch.zeros((steps, envs), dtype=torch.bool)
    e1 = torch.zeros_like(e2)
    unsafe = data["impact"] | data["slip"] | (data["tilt"] > .20)
    event_id = torch.full((steps, envs), -1, dtype=torch.long)
    event_manifest, next_event = [], 0
    for time_index, env_index in torch.nonzero(completion, as_tuple=False).tolist():
        episode_id = int(episode[time_index, env_index])
        takeoff_index = time_index
        for candidate in range(time_index, max(-1, time_index - 20), -1):
            if int(episode[candidate, env_index]) != episode_id:
                break
            if bool(takeoff[candidate, env_index]):
                takeoff_index = candidate
                break
        start = max(0, takeoff_index - 10)
        stop = min(steps, time_index + 6)
        valid = episode[start:stop, env_index] == episode_id
        indices = torch.arange(start, stop)[valid]
        if len(indices) > 40:
            indices = indices[-40:]
        existing = event_id[indices, env_index]
        existing_ids = existing[existing >= 0].unique()
        assigned = int(existing_ids[0]) if len(existing_ids) else next_event
        if not len(existing_ids):
            next_event += 1
        e2[indices, env_index] = True
        event_id[indices, env_index] = assigned
        event_manifest.append({
            "event_id": assigned, "environment_id": env_index, "episode_id": episode_id,
            "completion_timestep": time_index, "takeoff_timestep": takeoff_index,
            "window_start": int(indices.min()), "window_stop_inclusive": int(indices.max()),
            "window_samples": len(indices),
        })
    # Precursor-only event windows, excluding E2 by priority.
    for time_index, env_index in torch.nonzero(precursor_event & ~e2, as_tuple=False).tolist():
        episode_id = int(episode[time_index, env_index])
        start, stop = max(0, time_index - 10), min(steps, time_index + 6)
        valid = (episode[start:stop, env_index] == episode_id)
        indices = torch.arange(start, stop)[valid]
        e1[indices, env_index] = True
    # Unsafe keeps the 10 steps before a fall plus point safety violations.
    for time_index, env_index in torch.nonzero(data["fall"], as_tuple=False).tolist():
        episode_id = int(episode[time_index, env_index])
        start = max(0, time_index - 10)
        valid = episode[start:time_index + 1, env_index] == episode_id
        unsafe[torch.arange(start, time_index + 1)[valid], env_index] = True
    # Strict exclusive priority.
    unsafe &= ~e2
    e1 &= ~e2 & ~unsafe
    background = ~e2 & ~unsafe & ~e1
    strata = torch.full((steps, envs), 3, dtype=torch.long)
    strata[e1] = 1
    strata[unsafe] = 2
    strata[e2] = 0
    return strata, event_id, event_manifest


def matched_background(data, strata, event_id):
    flat_strata = strata.flatten()
    background_indices = torch.where(flat_strata == 3)[0]
    completion_indices = torch.where(flat_strata == 0)[0]
    command = data["command"].reshape(-1, 3)
    episode_time = data["episode_time"].flatten()
    contacts = data["contact_state"].reshape(-1, 2)
    flight_duration = data["flight_duration"].flatten()
    buckets = defaultdict(list)
    for index in background_indices.tolist():
        key = (
            round(float(command[index, 0]) / .05),
            round(float(episode_time[index]) / .5),
            command_segment(float(episode_time[index])),
            int(contacts[index, 0]) * 2 + int(contacts[index, 1]),
            int(flight_duration[index] > 0),
        )
        buckets[key].append(index)
    matches, cursor = [], Counter()
    for index in completion_indices.tolist():
        key = (
            round(float(command[index, 0]) / .05),
            round(float(episode_time[index]) / .5),
            command_segment(float(episode_time[index])),
            int(contacts[index, 0]) * 2 + int(contacts[index, 1]),
            int(flight_duration[index] > 0),
        )
        candidates = buckets.get(key, [])
        if not candidates:
            continue
        selected = candidates[cursor[key] % len(candidates)]
        cursor[key] += 1
        matches.append({
            "completion_flat_index": index, "background_flat_index": selected,
            "event_id": int(event_id.flatten()[index]), "matching_key": list(key),
        })
    return matches


def split_indices(data, strata):
    episode = data["episode_id"].flatten()
    unique = sorted(set(int(value) for value in episode.tolist()))
    holdout_episodes = {
        value
        for value in unique
        if hashlib.sha256(f"{args.analysis_seed}:{value}".encode()).digest()[0] < 51
    }
    holdout = torch.tensor([int(value) in holdout_episodes for value in episode.tolist()], dtype=torch.bool)
    return ~holdout, holdout, holdout_episodes


def sample_pool(pool, count, generator, replacement=True):
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    if not replacement and count <= len(pool):
        return pool[torch.randperm(len(pool), generator=generator)[:count]]
    return pool[torch.randint(len(pool), (count,), generator=generator)]


def make_schedule(strata, train_mask, event_id, factor, seed):
    generator = torch.Generator().manual_seed(seed)
    flat_strata, flat_event = strata.flatten(), event_id.flatten()
    pools = {name: torch.where(train_mask & (flat_strata == code))[0] for name, code in (
        ("E2_COMPLETION", 0), ("E1_PRECURSOR_ONLY", 1), ("U_UNSAFE", 2), ("B_BACKGROUND", 3)
    )}
    raw_total = int(train_mask.sum())
    raw_fraction = {name: len(pool) / raw_total for name, pool in pools.items()}
    schedules, rows, event_reuse = [], [], Counter()
    e2_chunks = None
    if factor > 1:
        e2_fraction = min(.40, raw_fraction["E2_COMPLETION"] * factor)
        unsafe_fraction = max(.20, raw_fraction["U_UNSAFE"])
        precursor_fraction = .10 if len(pools["E1_PRECURSOR_ONLY"]) else 0
        per_batch_counts = {
            "E2_COMPLETION": round(MINIBATCH_SIZE * e2_fraction),
            "U_UNSAFE": math.ceil(MINIBATCH_SIZE * unsafe_fraction),
            "E1_PRECURSOR_ONLY": math.ceil(MINIBATCH_SIZE * precursor_fraction),
        }
        per_batch_counts["B_BACKGROUND"] = MINIBATCH_SIZE - sum(per_batch_counts.values())
        if per_batch_counts["B_BACKGROUND"] / MINIBATCH_SIZE < .30:
            raise RuntimeError("EVENT_STRATIFIED_MINIBATCH_CONTRACT_FAIL")
        total_e2 = per_batch_counts["E2_COMPLETION"] * EPOCHS * MINIBATCHES
        e2_cycles = []
        while sum(len(chunk) for chunk in e2_cycles) < total_e2:
            e2_cycles.append(
                pools["E2_COMPLETION"][
                    torch.randperm(len(pools["E2_COMPLETION"]), generator=generator)
                ]
            )
        e2_sequence = torch.cat(e2_cycles)[:total_e2]
        per_sample_reuse = Counter(int(value) for value in e2_sequence.tolist())
        if per_sample_reuse and max(per_sample_reuse.values()) > factor:
            raise RuntimeError("EVENT_STRATIFIED_MINIBATCH_CONTRACT_FAIL")
        e2_chunks = list(e2_sequence.split(per_batch_counts["E2_COMPLETION"]))
    for epoch in range(EPOCHS):
        if factor == 1:
            epoch_indices = sample_pool(torch.where(train_mask)[0], EPOCH_SAMPLES, generator, replacement=False)
            epoch_batches = list(epoch_indices.reshape(MINIBATCHES, MINIBATCH_SIZE))
        else:
            epoch_batches = []
            for minibatch in range(MINIBATCHES):
                chunk_index = epoch * MINIBATCHES + minibatch
                pieces = [e2_chunks[chunk_index]]
                for name in ("U_UNSAFE", "E1_PRECURSOR_ONLY", "B_BACKGROUND"):
                    count = per_batch_counts[name]
                    pieces.append(
                        sample_pool(
                            pools[name],
                            count,
                            generator,
                            replacement=count > len(pools[name]),
                        )
                    )
                batch = torch.cat(pieces)
                batch = batch[torch.randperm(len(batch), generator=generator)]
                epoch_batches.append(batch)
        schedules.append(torch.stack(epoch_batches))
        for minibatch in range(MINIBATCHES):
            batch = schedules[-1][minibatch]
            counts = Counter(int(code) for code in flat_strata[batch].tolist())
            event_ids = flat_event[batch]
            valid_events = event_ids[event_ids >= 0]
            for value in valid_events.tolist():
                event_reuse[int(value)] += 1
            rows.append({
                "epoch": epoch, "minibatch": minibatch,
                "E2_COMPLETION": counts[0], "E1_PRECURSOR_ONLY": counts[1],
                "U_UNSAFE": counts[2], "B_BACKGROUND": counts[3],
                "completion_episode_count": len(set(
                    int(data_episode) for data_episode in event_ids[event_ids >= 0].tolist()
                )),
            })
    return torch.stack(schedules), raw_fraction, rows, event_reuse


def discounted_component(reward, done, gamma=.99):
    output = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1])
    for step in range(reward.shape[0] - 1, -1, -1):
        running = reward[step] + gamma * running * (~done[step]).float()
        output[step] = running
    return output


def flatten_gradient(parameters, gradients):
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ]).detach()


def actor_gradient(actor, data, indices, advantage):
    device = next(actor.parameters()).device
    observation = data["observation"].reshape(-1, 123)[indices].to(device)
    action = data["action"].reshape(-1, 37)[indices].to(device)
    old_logp = data["old_logp"].flatten()[indices].to(device)
    selected_advantage = advantage.flatten()[indices].to(device)
    # Production total advantage is already globally normalized in the corpus.
    # Diagnostic reward-component returns must only be centered; independently
    # normalizing a sparse component would erase its density/strength.
    selected_advantage = selected_advantage - selected_advantage.mean()
    ratio = torch.exp(actor.log_prob(observation, action) - old_logp)
    loss = -torch.minimum(
        ratio * selected_advantage,
        ratio.clamp(.8, 1.2) * selected_advantage,
    ).mean()
    params = list(actor.parameters())
    gradients = torch.autograd.grad(loss, params, allow_unused=True)
    return flatten_gradient(params, gradients), gradients


def component_gradients(checkpoint, data, strata, schedules, condition):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor = Actor(checkpoint["actor_state_dict"]).to(device)
    indices = schedules[0].flatten()
    run_index = data["reward_names"].index("safe_periodic_flight")
    run = data["reward_components"][:, :, run_index]
    completion_reward = torch.where(run >= 1, run, 0)
    precursor_reward = run - completion_reward
    base_reward = data["reward_components"].sum(-1) - run
    done = data["done"]
    components = {
        "total": data["normalized_advantage"],
        "base": discounted_component(base_reward, done),
        "precursor": discounted_component(precursor_reward, done),
        "completion": discounted_component(completion_reward, done),
        "run_specific": discounted_component(run, done),
        "unsafe": data["normalized_advantage"] * (strata == 2),
    }
    vectors, rows, layer_rows, joint_rows = {}, [], [], []
    named_params = list(actor.named_parameters())
    for name, advantage in components.items():
        vector, gradients = actor_gradient(actor, data, indices, advantage)
        vectors[name] = vector
        for (parameter_name, parameter), gradient in zip(named_params, gradients):
            value = torch.zeros_like(parameter) if gradient is None else gradient
            layer = (
                "std_parameter" if parameter_name == "std"
                else "first_hidden" if parameter_name.startswith("mlp.0")
                else "second_hidden" if parameter_name.startswith("mlp.2")
                else "third_hidden" if parameter_name.startswith("mlp.4")
                else "output_mean_head" if parameter_name.startswith("mlp.6")
                else parameter_name
            )
            if condition in ("M0_UNIFORM", "M8_EVENT_STRATIFIED", "M16_EVENT_STRATIFIED"):
                layer_rows.append({
                    "condition": condition, "component": name, "layer": layer,
                    "gradient_norm": float(torch.linalg.vector_norm(value)),
                })
                if parameter_name == "mlp.6.weight":
                    for joint, joint_name in enumerate(data["joint_names"]):
                        joint_rows.append({
                            "condition": condition, "component": name, "joint": joint_name,
                            "gradient_norm": float(torch.linalg.vector_norm(value[joint])),
                        })
    total_norm = torch.linalg.vector_norm(vectors["total"])
    for name, vector in vectors.items():
        rows.append({
            "condition": condition, "component": name,
            "gradient_norm": float(torch.linalg.vector_norm(vector)),
            "ratio_to_total": float(torch.linalg.vector_norm(vector) / (total_norm + 1e-12)),
            "cosine_to_total": cosine(vector, vectors["total"]),
            "cosine_to_completion": cosine(vector, vectors["completion"]),
            "cosine_to_base": cosine(vector, vectors["base"]),
            "cosine_to_unsafe": cosine(vector, vectors["unsafe"]),
        })
    return vectors, rows, layer_rows, joint_rows


def audit_metrics(actor, data, indices, old_mean, old_std):
    device = next(actor.parameters()).device
    chunks = []
    for chunk in indices.split(4096):
        observation = data["observation"].reshape(-1, 123)[chunk].to(device)
        action = data["action"].reshape(-1, 37)[chunk].to(device)
        old_logp = data["old_logp"].flatten()[chunk].to(device)
        new_mean = actor.mean(observation)
        new_std = torch.clamp(actor.std, min=1e-8).expand_as(new_mean)
        new_logp = actor.log_prob(observation, action)
        ratio = torch.exp(new_logp - old_logp)
        old_m, old_s = old_mean[chunk].to(device), old_std[chunk].to(device)
        old_new = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_m, old_s),
            torch.distributions.Normal(new_mean, new_std),
        ).sum(-1)
        chunks.append((ratio.cpu(), old_new.cpu(), torch.linalg.vector_norm(new_mean - old_m, dim=-1).cpu()))
    ratio = torch.cat([value[0] for value in chunks])
    kl = torch.cat([value[1] for value in chunks])
    shift = torch.cat([value[2] for value in chunks])
    return {
        "exact_kl_old_new": float(kl.mean()), "clip_fraction": float(((ratio < .8) | (ratio > 1.2)).float().mean()),
        "ratio_p95": float(torch.quantile(ratio, .95)), "ratio_p99": float(torch.quantile(ratio, .99)),
        "mean_action_shift": float(shift.mean()),
    }


def shadow_update(checkpoint, data, schedules, condition, completion_vector, unsafe_vector, audit_indices):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor = Actor(checkpoint["actor_state_dict"]).to(device)
    critic = Critic(checkpoint["critic_state_dict"]).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(params, lr=checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr = optimizer.param_groups[0]["lr"]
    old_mean = data["mean"].reshape(-1, 37)
    old_std = data["std"].reshape(-1, 37)
    before_actor = torch.cat([parameter.detach().flatten().cpu() for parameter in actor.parameters()])
    step_rows, max_actor_grad, max_critic_grad, max_value_loss = [], 0., 0., 0.
    max_kl = 0.
    for epoch in range(EPOCHS):
        for minibatch in range(MINIBATCHES):
            indices = schedules[epoch, minibatch]
            observation = data["observation"].reshape(-1, 123)[indices].to(device)
            action = data["action"].reshape(-1, 37)[indices].to(device)
            old_logp = data["old_logp"].flatten()[indices].to(device)
            old_value = data["value"].flatten()[indices].to(device)
            returns = data["return"].flatten()[indices].to(device)
            advantage = data["normalized_advantage"].flatten()[indices].to(device)
            with torch.no_grad():
                current_mean = actor.mean(observation)
                current_std = torch.clamp(actor.std, min=1e-8).expand_as(current_mean)
                kl_input = torch.distributions.kl_divergence(
                    torch.distributions.Normal(old_mean[indices].to(device), old_std[indices].to(device)),
                    torch.distributions.Normal(current_mean, current_std),
                ).sum(-1).mean()
                if kl_input > .02:
                    lr = max(1e-5, lr / 1.5)
                elif 0 < kl_input < .005:
                    lr = min(1e-2, lr * 1.5)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            logp = actor.log_prob(observation, action)
            ratio = torch.exp(logp - old_logp)
            surrogate = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
            value = critic(observation)
            value_clipped = old_value + (value - old_value).clamp(-.2, .2)
            value_loss = torch.maximum((value - returns) ** 2, (value_clipped - returns) ** 2).mean()
            entropy_loss = -actor.entropy()
            loss = surrogate + value_loss + .008 * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            actor_grad = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in actor.parameters() if p.grad is not None))
            critic_grad = math.sqrt(sum(float((p.grad.detach() ** 2).sum()) for p in critic.parameters() if p.grad is not None))
            max_actor_grad, max_critic_grad = max(max_actor_grad, actor_grad), max(max_critic_grad, critic_grad)
            max_value_loss = max(max_value_loss, float(value_loss))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            metrics = audit_metrics(actor, data, audit_indices, old_mean, old_std)
            max_kl = max(max_kl, metrics["exact_kl_old_new"])
            step_rows.append({
                "condition": condition, "epoch": epoch, "minibatch": minibatch,
                "learning_rate": lr, "kl_input": float(kl_input), **metrics,
                "actor_gradient": actor_grad, "critic_gradient": critic_grad,
                "value_loss": float(value_loss), "entropy": float(actor.entropy()),
            })
    final = audit_metrics(actor, data, audit_indices, old_mean, old_std)
    after_actor = torch.cat([parameter.detach().flatten().cpu() for parameter in actor.parameters()])
    update = after_actor - before_actor
    final.update({
        "condition": condition, "all_step_maximum_kl": max_kl,
        "max_actor_gradient": max_actor_grad, "max_critic_gradient": max_critic_grad,
        "max_value_loss": max_value_loss, "entropy": float(actor.entropy()),
        "nan_inf": int(not all(torch.isfinite(parameter).all() for parameter in params)),
        "adam_update_completion_cosine": cosine(update, -completion_vector.cpu()),
        "adam_update_unsafe_cosine": cosine(update, -unsafe_vector.cpu()),
        "final_learning_rate": lr,
    })
    final["hard_gate_pass"] = (
        final["exact_kl_old_new"] <= .20 and max_kl <= .20 and final["clip_fraction"] <= .50
        and final["mean_action_shift"] <= 2 and max_critic_grad <= 1e6 and max_value_loss <= 1e8
        and final["nan_inf"] == 0
    )
    temporary = RAW / f"{args.output_prefix}shadow_{condition}.pt"
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()}, temporary)
    return final, step_rows, actor


def holdout_loss(actor, data, indices):
    device = next(actor.parameters()).device
    if not len(indices):
        return 0.
    losses = []
    for chunk in indices.split(4096):
        observation = data["observation"].reshape(-1, 123)[chunk].to(device)
        action = data["action"].reshape(-1, 37)[chunk].to(device)
        old_logp = data["old_logp"].flatten()[chunk].to(device)
        advantage = data["normalized_advantage"].flatten()[chunk].to(device)
        ratio = torch.exp(actor.log_prob(observation, action) - old_logp)
        loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
        losses.append(loss.detach().cpu())
    return float(torch.cat(losses).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = torch.load(CORPUS_PATH, map_location="cpu", weights_only=False)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    strata, event_id, event_manifest = build_windows(data)
    train_mask, holdout_mask, holdout_episodes = split_indices(data, strata)
    matches = matched_background(data, strata, event_id)
    torch.save({
        "strata": strata, "event_id": event_id, "train_mask": train_mask,
        "holdout_mask": holdout_mask,
    }, RAW / f"{args.output_prefix}strata.pt")
    dump("strata_runtime.json", {
        "counts": {
            name: int((strata == code).sum()) for name, code in (
                ("E2_COMPLETION", 0), ("E1_PRECURSOR_ONLY", 1),
                ("U_UNSAFE", 2), ("B_BACKGROUND", 3)
            )
        },
        "event_windows": event_manifest, "holdout_episode_count": len(holdout_episodes),
        "matched_background": matches,
    })
    all_gradient_rows, all_layer_rows, all_joint_rows = [], [], []
    all_sampling_rows, composition = [], {}
    schedules_by_condition, vectors_by_condition = {}, {}
    for condition, factor in ACTIVE_CONDITIONS.items():
        schedules, raw_fraction, composition_rows, reuse = make_schedule(
            strata, train_mask, event_id, factor, args.analysis_seed + factor
        )
        schedules_by_condition[condition] = schedules
        torch.save(schedules, RAW / f"{args.output_prefix}schedule_{condition}.pt")
        vectors, gradient_rows, layer_rows, joint_rows = component_gradients(
            checkpoint, data, strata, schedules, condition
        )
        vectors_by_condition[condition] = vectors
        all_gradient_rows.extend(gradient_rows)
        all_layer_rows.extend(layer_rows)
        all_joint_rows.extend(joint_rows)
        counts = Counter(int(value) for value in strata.flatten()[schedules.flatten()].tolist())
        flat_schedule = schedules.flatten()
        e2_inclusions = flat_schedule[strata.flatten()[flat_schedule] == 0]
        per_sample_reuse = Counter(int(value) for value in e2_inclusions.tolist())
        composition[condition] = {
            "factor": factor, "raw_fraction": raw_fraction, "minibatches": composition_rows,
            "total_inclusions": {
                "E2_COMPLETION": counts[0], "E1_PRECURSOR_ONLY": counts[1],
                "U_UNSAFE": counts[2], "B_BACKGROUND": counts[3],
            },
            "event_reuse_min": min(reuse.values()) if reuse else 0,
            "event_reuse_max": max(reuse.values()) if reuse else 0,
            "event_reuse_mean": sum(reuse.values()) / len(reuse) if reuse else 0,
            "completion_sample_reuse_max": max(per_sample_reuse.values()) if per_sample_reuse else 0,
            "completion_sample_reuse_mean": (
                sum(per_sample_reuse.values()) / len(per_sample_reuse)
                if per_sample_reuse else 0
            ),
            "completion_sample_reuse_cap": factor if factor > 1 else "uniform baseline",
            "completion_sample_reuse_cap_pass": (
                max(per_sample_reuse.values()) <= factor
                if factor > 1 and per_sample_reuse else True
            ),
        }
        for name, code in (("E2_COMPLETION", 0), ("E1_PRECURSOR_ONLY", 1), ("U_UNSAFE", 2), ("B_BACKGROUND", 3)):
            raw_count = int((train_mask & (strata.flatten() == code)).sum())
            included = counts[code]
            all_sampling_rows.append({
                "condition": condition, "stratum": name, "raw_sample_count": raw_count,
                "raw_fraction": raw_fraction[name], "total_inclusions_5_epochs": included,
                "effective_inclusions_per_epoch": included / EPOCHS,
                "effective_sample_multiplier": (included / EPOCHS) / max(1, raw_count),
                "effective_total_weight": included / (EPOCHS * EPOCH_SAMPLES),
            })
    write_csv("effective_sampling_weights.csv", all_sampling_rows)
    dump("minibatch_composition_runtime.json", composition)
    write_csv("stratified_gradient_components.csv", all_gradient_rows)
    write_csv("stratified_layerwise_gradients.csv", all_layer_rows)
    write_csv("stratified_jointwise_gradients.csv", all_joint_rows)
    gradient_summary = {}
    for condition in ACTIVE_CONDITIONS:
        rows = [row for row in all_gradient_rows if row["condition"] == condition]
        gradient_summary[condition] = {row["component"]: row for row in rows}
    dump("stratified_gradient_runtime.json", gradient_summary)
    torch.save(
        {
            condition: {
                component: vector.detach().cpu()
                for component, vector in vectors.items()
            }
            for condition, vectors in vectors_by_condition.items()
        },
        RAW / f"{args.output_prefix}gradient_vectors.pt",
    )
    # Fixed uniform audit subset, independent of each condition's update schedule.
    generator = torch.Generator().manual_seed(args.analysis_seed + 999)
    audit_indices = sample_pool(torch.where(train_mask)[0], EPOCH_SAMPLES, generator, replacement=False)
    flat_strata = strata.flatten()
    holdout_pools = {
        "completion": torch.where(holdout_mask & (flat_strata == 0))[0],
        "precursor": torch.where(holdout_mask & (flat_strata == 1))[0],
        "unsafe": torch.where(holdout_mask & (flat_strata == 2))[0],
        "background": torch.tensor([
            row["background_flat_index"] for row in matches
            if holdout_mask[row["completion_flat_index"]] and holdout_mask[row["background_flat_index"]]
        ], dtype=torch.long),
        "total": torch.where(holdout_mask)[0],
    }
    # Bound holdout computation while preserving disjoint episodes.
    for name, pool in holdout_pools.items():
        if len(pool) > 24576:
            holdout_pools[name] = sample_pool(pool, 24576, generator, replacement=False)
    shadow_rows, step_rows, cross_rows = [], [], []
    for condition in ACTIVE_CONDITIONS:
        completion_vector = vectors_by_condition[condition]["completion"]
        unsafe_vector = vectors_by_condition[condition]["unsafe"]
        final, steps, updated_actor = shadow_update(
            checkpoint, data, schedules_by_condition[condition], condition,
            completion_vector, unsafe_vector, audit_indices,
        )
        shadow_rows.append(final)
        step_rows.extend(steps)
        initial_actor = Actor(checkpoint["actor_state_dict"]).to(next(updated_actor.parameters()).device)
        for stratum_name, indices in holdout_pools.items():
            before = holdout_loss(initial_actor, data, indices)
            after = holdout_loss(updated_actor, data, indices)
            relative = (after - before) / max(abs(before), 1e-8)
            cross_rows.append({
                "condition": condition, "holdout_stratum": stratum_name,
                "loss_before": before, "loss_after": after, "absolute_change": after - before,
                "relative_change": relative, "improved": after < before,
            })
    write_csv("shadow_update_metrics.csv", shadow_rows)
    write_csv("shadow_update_step_trace.csv", step_rows)
    write_csv("shadow_cross_effect_matrix.csv", cross_rows)
    dump("shadow_runtime_summary.json", {
        "conditions": shadow_rows, "cross_effect": cross_rows,
        "temporary_parameter_artifacts": [
            f"raw/{args.output_prefix}shadow_{name}.pt" for name in ACTIVE_CONDITIONS
        ],
        "persistent_checkpoint_writes": 0, "production_policy_updates": 0,
    })


if __name__ == "__main__":
    main()
