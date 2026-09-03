"""Execute one disposable standard PPO update plus at most one actor replay update."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2h_short_horizon_completion_replay_preflight"
RAW = OUT / "raw"

parser = argparse.ArgumentParser()
parser.add_argument("--branch", required=True)
parser.add_argument("--shadow-iteration", type=int, required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--corpus", required=True)
parser.add_argument("--mode", choices=("standard", "completion", "background"), required=True)
parser.add_argument("--horizon", type=int, default=0)
parser.add_argument("--coefficient", type=float, default=0.0)
parser.add_argument("--analysis-seed", type=int, required=True)
args = parser.parse_args()

# Reuse the already audited Stage-2G Gaussian/PPO/window helpers without running its main.
saved_argv = sys.argv
sys.argv = ["analyze_stage2g_shadow.py"]
sys.path.insert(0, str(SCRIPT.parent))
import analyze_stage2g_shadow as base  # noqa: E402
sys.argv = saved_argv
base.args.analysis_seed = args.analysis_seed

EPOCHS = 5
MINIBATCHES = 4
MINIBATCH_SIZE = 6144
AUDIT_SAMPLES = 24576


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_sha(state) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def actor_export(actor):
    exported = {}
    for name, value in actor.state_dict().items():
        exported["distribution.std_param" if name == "std" else name] = value.detach().cpu()
    return exported


def flat_grad(parameters):
    return torch.cat(
        [
            (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad)
            .detach()
            .flatten()
            .cpu()
            for parameter in parameters
        ]
    )


def gradient(actor, data, indices, advantage):
    device = next(actor.parameters()).device
    mean_parameters = [parameter for name, parameter in actor.named_parameters() if name != "std"]
    observation = data["observation"].reshape(-1, 123)[indices].to(device)
    action = data["action"].reshape(-1, 37)[indices].to(device)
    old_logp = data["old_logp"].flatten()[indices].to(device)
    selected_advantage = advantage.flatten()[indices].to(device)
    ratio = torch.exp(actor.log_prob(observation, action) - old_logp)
    loss = -torch.minimum(
        ratio * selected_advantage, ratio.clamp(.8, 1.2) * selected_advantage
    ).mean()
    gradients = torch.autograd.grad(loss, mean_parameters, allow_unused=True)
    vector = torch.cat(
        [
            (torch.zeros_like(parameter) if value is None else value).flatten()
            for parameter, value in zip(mean_parameters, gradients)
        ]
    ).detach().cpu()
    return vector, float(loss)


def build_stage2h_windows(data):
    """Build the Stage-2H replay unit: >=15 steps pre-takeoff and >=10 post-landing."""
    steps, envs = data["completion"].shape
    episode = data["episode_id"]
    takeoff = data["takeoff"]
    completion = data["completion"]
    precursor_event = data["precursor"] | data["safe_flight"]
    e2 = torch.zeros((steps, envs), dtype=torch.bool)
    e1 = torch.zeros_like(e2)
    unsafe = data["impact"] | data["slip"] | (data["tilt"] > .20)
    event_id = torch.full((steps, envs), -1, dtype=torch.long)
    event_manifest = []
    next_event = 0

    # First form intervals, then merge overlapping completion cycles from one episode.
    intervals = defaultdict(list)
    for completion_step, env_index in torch.nonzero(completion, as_tuple=False).tolist():
        episode_id = int(episode[completion_step, env_index])
        takeoff_step = completion_step
        for candidate in range(completion_step, max(-1, completion_step - 60), -1):
            if int(episode[candidate, env_index]) != episode_id:
                break
            if bool(takeoff[candidate, env_index]):
                takeoff_step = candidate
                break
        start = max(0, takeoff_step - 15)
        stop = min(steps, completion_step + 11)
        while start < completion_step and int(episode[start, env_index]) != episode_id:
            start += 1
        while stop > completion_step + 1 and int(episode[stop - 1, env_index]) != episode_id:
            stop -= 1
        if stop - start > 60:
            start = stop - 60
        intervals[(env_index, episode_id)].append(
            {
                "start": start,
                "stop": stop,
                "takeoff_steps": [takeoff_step],
                "completion_steps": [completion_step],
            }
        )

    for (env_index, episode_id), episode_intervals in sorted(intervals.items()):
        merged = []
        for interval in sorted(episode_intervals, key=lambda value: value["start"]):
            if merged and interval["start"] <= merged[-1]["stop"]:
                merged[-1]["stop"] = min(steps, max(merged[-1]["stop"], interval["stop"]))
                merged[-1]["takeoff_steps"].extend(interval["takeoff_steps"])
                merged[-1]["completion_steps"].extend(interval["completion_steps"])
                if merged[-1]["stop"] - merged[-1]["start"] > 60:
                    merged[-1]["start"] = merged[-1]["stop"] - 60
            else:
                merged.append(copy.deepcopy(interval))
        for interval in merged:
            indices = torch.arange(interval["start"], interval["stop"])
            valid = episode[indices, env_index] == episode_id
            indices = indices[valid]
            assigned = next_event
            next_event += 1
            e2[indices, env_index] = True
            event_id[indices, env_index] = assigned
            event_manifest.append(
                {
                    "event_id": assigned,
                    "environment_id": env_index,
                    "episode_id": episode_id,
                    "completion_timesteps": interval["completion_steps"],
                    "takeoff_timesteps": interval["takeoff_steps"],
                    "window_start": int(indices.min()),
                    "window_stop_inclusive": int(indices.max()),
                    "window_samples": len(indices),
                    "ordered_takeoff_flight_landing": True,
                }
            )

    # Keep Stage-2G's audited exclusive strata semantics around the wider replay unit.
    for time_index, env_index in torch.nonzero(precursor_event & ~e2, as_tuple=False).tolist():
        episode_id = int(episode[time_index, env_index])
        start, stop = max(0, time_index - 10), min(steps, time_index + 6)
        valid = episode[start:stop, env_index] == episode_id
        indices = torch.arange(start, stop)[valid]
        e1[indices, env_index] = True
    for time_index, env_index in torch.nonzero(data["fall"], as_tuple=False).tolist():
        episode_id = int(episode[time_index, env_index])
        start = max(0, time_index - 10)
        valid = episode[start:time_index + 1, env_index] == episode_id
        unsafe[torch.arange(start, time_index + 1)[valid], env_index] = True
    unsafe &= ~e2
    e1 &= ~e2 & ~unsafe
    strata = torch.full((steps, envs), 3, dtype=torch.long)
    strata[e1] = 1
    strata[unsafe] = 2
    strata[e2] = 0
    return strata, event_id, event_manifest


def window_payload(data, indices, window_id, source_iteration, source_policy_hash, kind):
    fields = (
        "observation", "action", "mean", "std", "old_logp", "reward",
        "reward_components", "return", "raw_advantage", "normalized_advantage",
        "command", "actual_speed", "flight_duration", "landing_side",
        "previous_landing_side", "completion", "fall", "tilt", "heading",
        "slip", "impact", "saturation", "episode_id", "episode_time",
        "rollout_timestep", "contact_state",
    )
    flat = {}
    for field in fields:
        value = data[field]
        flat[field] = value.reshape(-1, *value.shape[2:])[indices].clone()
    return {
        "window_id": window_id,
        "kind": kind,
        "source_shadow_iteration": source_iteration,
        "source_policy_hash": source_policy_hash,
        "episode_id": int(flat["episode_id"][0]),
        "reuse_count": 0,
        "sample_count": len(indices),
        "data": flat,
    }


def add_windows(buffer, data, strata, event_id, source_iteration, source_hash):
    matches = base.matched_background(data, strata, event_id)
    matched = defaultdict(list)
    for row in matches:
        matched[row["completion_flat_index"]].append(row["background_flat_index"])
    flat_strata = strata.flatten()
    background_pool = torch.where(flat_strata == 3)[0]
    command = data["command"].reshape(-1, 3)[:, 0]
    episode_time = data["episode_time"].flatten()
    contacts = data["contact_state"].reshape(-1, 2)
    flight = data["flight_duration"].flatten() > 0
    bins = defaultdict(list)
    relaxed_bins = defaultdict(list)
    for index in background_pool.tolist():
        speed_bin = round(float(command[index]) / .05)
        time_bin = round(float(episode_time[index]) / .5)
        contact_code = int(contacts[index, 0]) + 2 * int(contacts[index, 1])
        key = (speed_bin, time_bin, contact_code, int(flight[index]))
        bins[key].append(index)
        relaxed_bins[(speed_bin, time_bin, int(flight[index]))].append(index)
    for event in sorted(set(int(value) for value in event_id[event_id >= 0].tolist())):
        indices = torch.where(event_id.flatten() == event)[0]
        if not len(indices):
            continue
        window_id = f"iter{source_iteration:02d}_event{event:04d}"
        buffer.append(window_payload(data, indices, window_id, source_iteration, source_hash, "completion"))
        background_indices = []
        used = set()
        for offset, index in enumerate(indices.tolist()):
            candidates = matched.get(index, [])
            candidate = next((value for value in candidates if value not in used), None)
            if candidate is None:
                speed_bin = round(float(command[index]) / .05)
                time_bin = round(float(episode_time[index]) / .5)
                contact_code = int(contacts[index, 0]) + 2 * int(contacts[index, 1])
                key = (speed_bin, time_bin, contact_code, int(flight[index]))
                pool = bins.get(key, [])
                candidate = next((value for value in pool if value not in used), None)
            if candidate is None:
                key = (
                    round(float(command[index]) / .05),
                    round(float(episode_time[index]) / .5),
                    int(flight[index]),
                )
                pool = relaxed_bins.get(key, [])
                candidate = next((value for value in pool if value not in used), None)
            if candidate is None:
                speed_bin = round(float(command[index]) / .05)
                time_bin = round(float(episode_time[index]) / .5)
                contact_code = int(contacts[index, 0]) + 2 * int(contacts[index, 1])
                # Deterministic nearest-bin fallback. Contact/flight remain exact;
                # command and episode time are relaxed only as far as necessary.
                for radius in range(1, 13):
                    neighbor_keys = []
                    for speed_delta in range(-radius, radius + 1):
                        time_delta = radius - abs(speed_delta)
                        for signed_time in ({-time_delta, time_delta} if time_delta else {0}):
                            neighbor_keys.append(
                                (
                                    speed_bin + speed_delta,
                                    time_bin + signed_time,
                                    contact_code,
                                    int(flight[index]),
                                )
                            )
                    candidate = next(
                        (
                            value
                            for neighbor_key in neighbor_keys
                            for value in bins.get(neighbor_key, [])
                            if value not in used
                        ),
                        None,
                    )
                    if candidate is not None:
                        break
            if candidate is None:
                speed_bin = round(float(command[index]) / .05)
                time_bin = round(float(episode_time[index]) / .5)
                for radius in range(1, 13):
                    candidate = next(
                        (
                            value
                            for speed_delta in range(-radius, radius + 1)
                            for signed_time in {
                                -(radius - abs(speed_delta)),
                                radius - abs(speed_delta),
                            }
                            for value in relaxed_bins.get(
                                (
                                    speed_bin + speed_delta,
                                    time_bin + signed_time,
                                    int(flight[index]),
                                ),
                                [],
                            )
                            if value not in used
                        ),
                        None,
                    )
                    if candidate is not None:
                        break
            if candidate is not None:
                background_indices.append(candidate)
                used.add(candidate)
        if len(background_indices) == len(indices):
            buffer.append(
                window_payload(
                    data,
                    torch.tensor(background_indices, dtype=torch.long),
                    f"{window_id}_background",
                    source_iteration,
                    source_hash,
                    "background",
                )
            )
    # FIFO is applied independently to completion and matched-background windows.
    retained = []
    for kind in ("completion", "background"):
        subset = [window for window in buffer if window["kind"] == kind][-256:]
        retained.extend(subset)
    return retained


def concat_windows(windows):
    result = {}
    for field in windows[0]["data"]:
        result[field] = torch.cat([window["data"][field] for window in windows])
    return result


def eligibility(actor, windows, iteration):
    device = next(actor.parameters()).device
    eligible, rows, reasons = [], [], Counter()
    for window in windows:
        values = window["data"]
        observation = values["observation"].to(device)
        action = values["action"].to(device)
        old_mean = values["mean"].to(device)
        old_std = values["std"].to(device)
        behavior_logp = values["old_logp"].to(device)
        with torch.no_grad():
            current_mean = actor.mean(observation)
            current_std = torch.clamp(actor.std, min=1e-8).expand_as(current_mean)
            kl = torch.distributions.kl_divergence(
                torch.distributions.Normal(old_mean, old_std),
                torch.distributions.Normal(current_mean, current_std),
            ).sum(-1)
            ratio = torch.exp(actor.log_prob(observation, action) - behavior_logp)
            clip_fraction = float(((ratio < .8) | (ratio > 1.2)).float().mean())
            ratio_p99 = float(torch.quantile(ratio, .99))
            ess = float(ratio.sum().square() / (ratio.square().sum() + 1e-12))
            ess_fraction = ess / len(ratio)
            exact_kl = float(kl.mean())
        failed = []
        if exact_kl > .05:
            failed.append("KL")
        if clip_fraction > .30:
            failed.append("CLIP")
        if ratio_p99 > 2.0:
            failed.append("RATIO_P99")
        if ess_fraction < .50:
            failed.append("ESS")
        age = iteration - window["source_shadow_iteration"]
        row = {
            "branch": args.branch, "shadow_iteration": iteration,
            "window_id": window["window_id"], "kind": window["kind"],
            "age": age, "sample_count": window["sample_count"],
            "exact_kl": exact_kl, "joint_clip_fraction": clip_fraction,
            "ratio_p99": ratio_p99, "ess_fraction": ess_fraction,
            "eligible": not failed, "drop_reasons": "|".join(failed),
        }
        rows.append(row)
        if failed:
            reasons.update(failed)
        else:
            eligible.append(window)
    return eligible, rows, dict(reasons)


def standard_update(checkpoint, data, schedules, audit_indices):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    actor = base.Actor(checkpoint["actor_state_dict"]).to(device)
    critic = base.Critic(checkpoint["critic_state_dict"]).to(device)
    parameters = list(actor.parameters()) + list(critic.parameters())
    optimizer = torch.optim.Adam(
        parameters, lr=checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    lr = optimizer.param_groups[0]["lr"]
    old_mean = data["mean"].reshape(-1, 37)
    old_std = data["std"].reshape(-1, 37)
    actor_before = copy.deepcopy(actor).cpu()
    maximum_kl = maximum_actor = maximum_critic = maximum_value = 0.0
    trace = []
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
                    torch.distributions.Normal(
                        old_mean[indices].to(device), old_std[indices].to(device)
                    ),
                    torch.distributions.Normal(current_mean, current_std),
                ).sum(-1).mean()
                if kl_input > .02:
                    lr = max(1e-5, lr / 1.5)
                elif 0 < kl_input < .005:
                    lr = min(1e-2, lr * 1.5)
                for group in optimizer.param_groups:
                    group["lr"] = lr
            ratio = torch.exp(actor.log_prob(observation, action) - old_logp)
            surrogate = -torch.minimum(
                ratio * advantage, ratio.clamp(.8, 1.2) * advantage
            ).mean()
            value = critic(observation)
            value_clipped = old_value + (value - old_value).clamp(-.2, .2)
            value_loss = torch.maximum(
                (value - returns).square(), (value_clipped - returns).square()
            ).mean()
            loss = surrogate + value_loss - .008 * actor.entropy()
            optimizer.zero_grad()
            loss.backward()
            actor_grad = math.sqrt(
                sum(float(parameter.grad.detach().square().sum()) for parameter in actor.parameters() if parameter.grad is not None)
            )
            critic_grad = math.sqrt(
                sum(float(parameter.grad.detach().square().sum()) for parameter in critic.parameters() if parameter.grad is not None)
            )
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            metrics = base.audit_metrics(actor, data, audit_indices, old_mean, old_std)
            maximum_kl = max(maximum_kl, metrics["exact_kl_old_new"])
            maximum_actor = max(maximum_actor, actor_grad)
            maximum_critic = max(maximum_critic, critic_grad)
            maximum_value = max(maximum_value, float(value_loss))
            trace.append({
                "epoch": epoch, "minibatch": minibatch, "learning_rate": lr,
                "kl_input": float(kl_input), **metrics,
            })
    return {
        "actor": actor, "critic": critic, "optimizer": optimizer, "lr": lr,
        "actor_before": actor_before, "maximum_kl": maximum_kl,
        "maximum_actor_gradient": maximum_actor,
        "maximum_critic_gradient": maximum_critic,
        "maximum_value_loss": maximum_value, "trace": trace,
        "old_mean": old_mean, "old_std": old_std,
    }


def main():
    branch_dir = RAW / args.branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    data = torch.load(args.corpus, map_location="cpu", weights_only=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    strata, event_id, event_manifest = build_stage2h_windows(data)
    train_mask, holdout_mask, _ = base.split_indices(data, strata)
    generator = torch.Generator().manual_seed(args.analysis_seed)
    schedules, _, _, _ = base.make_schedule(strata, train_mask, event_id, 1, args.analysis_seed)
    audit_indices = base.sample_pool(
        torch.where(train_mask)[0], AUDIT_SAMPLES, generator, replacement=False
    )
    update = standard_update(checkpoint, data, schedules, audit_indices)
    actor, critic, optimizer = update["actor"], update["critic"], update["optimizer"]
    policy_hash = tensor_sha(checkpoint["actor_state_dict"])

    buffer_path = branch_dir / "replay_buffer.pt"
    # Iteration one is a branch identity boundary. Never inherit an earlier screen buffer.
    buffer = (
        []
        if args.shadow_iteration == 1
        else torch.load(buffer_path, map_location="cpu", weights_only=False)
        if buffer_path.exists()
        else []
    )
    buffer = add_windows(buffer, data, strata, event_id, args.shadow_iteration, policy_hash)
    buffer = [
        window for window in buffer
        if args.shadow_iteration - window["source_shadow_iteration"] <= 4
        and window["reuse_count"] < 4
    ]

    requested_kind = "background" if args.mode == "background" else "completion"
    candidates = [
        window for window in buffer
        if window["kind"] == requested_kind
        and args.shadow_iteration - window["source_shadow_iteration"] <= args.horizon
        and window["reuse_count"] < 4
    ]
    eligible, eligibility_rows, drop_reasons = eligibility(
        actor, candidates, args.shadow_iteration
    )
    eligible = eligible[:256]

    # Gradient audit is computed after the standard update, before any auxiliary step.
    on_policy_vector, _ = gradient(
        actor, data, schedules[0].flatten(), data["normalized_advantage"]
    )
    unsafe_indices = torch.where(strata.flatten() == 2)[0]
    unsafe_vector, _ = gradient(
        actor,
        data,
        base.sample_pool(unsafe_indices, min(len(unsafe_indices), AUDIT_SAMPLES), generator, replacement=False),
        data["normalized_advantage"],
    )
    replay_vector = torch.zeros_like(on_policy_vector)
    replay_loss = 0.0
    gradient_ratio = 0.0
    applied = False
    ineligible_reason = ""
    replay_samples = 0
    if args.mode != "standard":
        if len(eligible) < 16:
            ineligible_reason = "ELIGIBLE_WINDOWS_LT_16"
        else:
            replay_data = concat_windows(eligible)
            replay_samples = len(replay_data["old_logp"])
            replay_actor_data = {
                "observation": replay_data["observation"].unsqueeze(0),
                "action": replay_data["action"].unsqueeze(0),
                "old_logp": replay_data["old_logp"].unsqueeze(0),
            }
            replay_advantage = replay_data["normalized_advantage"].unsqueeze(0)
            replay_indices = torch.arange(replay_samples)
            replay_vector, replay_loss = gradient(
                actor, replay_actor_data, replay_indices, replay_advantage
            )
            gradient_ratio = float(
                args.coefficient * torch.linalg.vector_norm(replay_vector)
                / (torch.linalg.vector_norm(on_policy_vector) + 1e-12)
            )
            if gradient_ratio > .10:
                ineligible_reason = "AUXILIARY_GRADIENT_CAP"
            else:
                device = next(actor.parameters()).device
                mean_parameters = [
                    parameter for name, parameter in actor.named_parameters() if name != "std"
                ]
                values = replay_data
                observation = values["observation"].to(device)
                action = values["action"].to(device)
                behavior_logp = values["old_logp"].to(device)
                advantage = values["normalized_advantage"].to(device)
                ratio = torch.exp(actor.log_prob(observation, action) - behavior_logp)
                loss = -args.coefficient * torch.minimum(
                    ratio * advantage, ratio.clamp(.8, 1.2) * advantage
                ).mean()
                optimizer.zero_grad()
                loss.backward()
                actor.std.grad = None
                torch.nn.utils.clip_grad_norm_(mean_parameters, 1.0)
                optimizer.step()
                applied = True
                for window in eligible:
                    window["reuse_count"] += 1

    torch.save(buffer, buffer_path)
    final_metrics = base.audit_metrics(
        actor, data, audit_indices, update["old_mean"], update["old_std"]
    )
    combined_hard_gate = (
        final_metrics["exact_kl_old_new"] <= .20
        and update["maximum_kl"] <= .20
        and final_metrics["clip_fraction"] <= .50
        and final_metrics["mean_action_shift"] <= 2.0
        and update["maximum_critic_gradient"] <= 1e6
        and update["maximum_value_loss"] <= 1e8
        and all(torch.isfinite(parameter).all() for parameter in list(actor.parameters()) + list(critic.parameters()))
    )
    # Holdout cross-effect uses episode-disjoint samples.
    flat_strata = strata.flatten()
    matches = base.matched_background(data, strata, event_id)
    pools = {
        "completion": torch.where(holdout_mask & (flat_strata == 0))[0],
        "precursor": torch.where(holdout_mask & (flat_strata == 1))[0],
        "unsafe": torch.where(holdout_mask & (flat_strata == 2))[0],
        "background": torch.tensor(
            [
                row["background_flat_index"] for row in matches
                if holdout_mask[row["completion_flat_index"]]
                and holdout_mask[row["background_flat_index"]]
            ],
            dtype=torch.long,
        ),
        "total_run": torch.where(holdout_mask)[0],
    }
    cross_effect = []
    for name, indices in pools.items():
        if len(indices) > AUDIT_SAMPLES:
            indices = base.sample_pool(indices, AUDIT_SAMPLES, generator, replacement=False)
        before = base.holdout_loss(update["actor_before"], data, indices)
        after = base.holdout_loss(actor, data, indices)
        cross_effect.append({
            "stratum": name, "loss_before": before, "loss_after": after,
            "absolute_change": after - before,
            "relative_change": (after - before) / max(abs(before), 1e-8),
            "improved": after < before,
        })

    output_checkpoint = branch_dir / f"state_{args.shadow_iteration}.pt"
    exported = copy.deepcopy(checkpoint)
    exported["actor_state_dict"] = actor_export(actor)
    exported["critic_state_dict"] = {
        name: value.detach().cpu() for name, value in critic.state_dict().items()
    }
    exported["optimizer_state_dict"] = optimizer.state_dict()
    exported["iter"] = int(checkpoint.get("iter", 0)) + 1
    torch.save(exported, output_checkpoint)
    torch.save(
        {"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict()},
        branch_dir / f"shadow_{args.shadow_iteration}.pt",
    )
    torch.save(
        {
            "on_policy": on_policy_vector,
            "replay": replay_vector,
            "unsafe": unsafe_vector,
        },
        branch_dir / f"gradient_vectors_{args.shadow_iteration}.pt",
    )
    metrics = {
        "branch": args.branch, "shadow_iteration": args.shadow_iteration,
        "mode": args.mode, "horizon": args.horizon, "coefficient": args.coefficient,
        "source_policy_hash": policy_hash,
        "completion_windows_added": len(event_manifest),
        "buffer_completion_windows": sum(window["kind"] == "completion" for window in buffer),
        "buffer_background_windows": sum(window["kind"] == "background" for window in buffer),
        "candidate_windows": len(candidates), "eligible_windows": len(eligible),
        "dropped_windows": len(candidates) - len(eligible),
        "drop_reasons": drop_reasons, "replay_samples": replay_samples,
        "auxiliary_applied": applied, "auxiliary_ineligible_reason": ineligible_reason,
        "on_policy_gradient_norm": float(torch.linalg.vector_norm(on_policy_vector)),
        "replay_gradient_norm": float(torch.linalg.vector_norm(replay_vector)),
        "unsafe_gradient_norm": float(torch.linalg.vector_norm(unsafe_vector)),
        "effective_replay_gradient_ratio": gradient_ratio,
        "replay_vs_on_policy_cosine": base.cosine(replay_vector, on_policy_vector),
        "replay_vs_unsafe_cosine": base.cosine(replay_vector, unsafe_vector),
        "standard_maximum_kl": update["maximum_kl"],
        "standard_actor_gradient_max": update["maximum_actor_gradient"],
        "standard_critic_gradient_max": update["maximum_critic_gradient"],
        "standard_value_loss_max": update["maximum_value_loss"],
        "first_step_lr": update["trace"][0]["learning_rate"],
        "final_lr": update["lr"], **final_metrics,
        "combined_hard_gate_pass": bool(combined_hard_gate),
        "nan_inf": int(not all(torch.isfinite(parameter).all() for parameter in list(actor.parameters()) + list(critic.parameters()))),
        "cross_effect": cross_effect,
        "output_checkpoint": str(output_checkpoint.relative_to(REPO)),
        "persistent_checkpoint": False,
        "production_policy_update": False,
    }
    dump(branch_dir / f"metrics_{args.shadow_iteration}.json", metrics)
    with (branch_dir / f"eligibility_{args.shadow_iteration}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eligibility_rows[0]) if eligibility_rows else ["status"])
        writer.writeheader()
        writer.writerows(eligibility_rows or [{"status": "NO_CANDIDATES"}])
    dump(branch_dir / f"standard_trace_{args.shadow_iteration}.json", update["trace"])
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
