"""Offline reward-ranking, temporal, value, and advantage analysis for Stage 12."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage12_tangential_slip_reward_directionality"
RAW = OUT / "raw"
MAJOR = (0.2, 0.4, 0.6, 1.2, 2.0)
SEGMENT = 10


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


def stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "p05": float(np.percentile(values, 5)) if values.size else 0.0,
        "p50": float(np.percentile(values, 50)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
    }


def correlation(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() == 0 or right[valid].std() == 0:
        return 0.0
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def discounted(reward, valid, gamma):
    result = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1])
    for step in reversed(range(reward.shape[0])):
        running = reward[step] + gamma * running
        running = torch.where(valid[step], running, torch.zeros_like(running))
        result[step] = running
    return result


files = sorted(RAW.glob("steady_*.pt")) + sorted(RAW.glob("transition_*.pt"))
if len(files) != 21:
    raise SystemExit(f"expected 21 rollout files, found {len(files)}")

manifest_files = []
segment_rows = []
return_summary = []
temporal_by_condition = []
lag_accumulator = defaultdict(list)
value_obs, value_target, value_episode = [], [], []
adv_chunks = []
global_episode = 0

for file_index, path in enumerate(files):
    data = torch.load(path, map_location="cpu", weights_only=False)
    family = data["family"]
    source, target = float(data["source_speed"]), float(data["target_speed"])
    valid = data["valid"].bool()
    gamma, lam = float(data["gamma"]), float(data["lam"])
    slip_reward = data["weighted_slip_reward"]
    base_reward = data["base_reward"]
    total_reward = data["total_reward"]
    slip_return = discounted(slip_reward, valid, gamma)
    base_return = discounted(base_reward, valid, gamma)
    total_return = discounted(total_reward, valid, gamma)
    manifest_files.append({
        "path": str(path.resolve()), "sha256": sha(path), "bytes": path.stat().st_size,
        "family": family, "source_speed": source, "target_speed": target,
        "samples": int(valid.numel()), "valid_samples": int(valid.sum()),
    })
    label = f"{source:g}->{target:g}" if family == "transition" else f"{target:g}"
    for name, values in (
        ("slip_only", slip_return[valid]), ("base_only", base_return[valid]),
        ("total", total_return[valid]),
    ):
        return_summary.append({
            "family": family, "condition": label, "component": name,
            **stats(values.numpy()),
        })

    # Episode-disjoint diagnostic value/advantage subset: first 20 episodes per condition.
    episode_count = min(20, valid.shape[1])
    subset = slice(0, episode_count)
    saved_observation = data["observation"]
    # RSL-RL 3.x returns a TensorDict observation group.  The frozen Go2
    # actor consumes the 48-D ``policy`` group; persistently select that
    # group here so the diagnostic regressor and gradient replay receive a
    # plain tensor without changing the collected rollout contract.
    if hasattr(saved_observation, "keys") and "policy" in saved_observation.keys():
        saved_observation = saved_observation["policy"]
    obs = saved_observation[:, subset]
    target_return = slip_return[:, subset]
    valid_subset = valid[:, subset]
    episode_ids = torch.arange(global_episode, global_episode + episode_count)
    ids = episode_ids[None].expand(obs.shape[0], -1)
    value_obs.append(obs[valid_subset])
    value_target.append(target_return[valid_subset])
    value_episode.append(ids[valid_subset])
    adv_chunks.append({
        "file": str(path), "family": family, "source": source, "target": target,
        "observation": obs, "action": data["sampled_action"][:, subset],
        "old_log_prob": data["old_log_prob"][:, subset],
        "production_value": data["value"][:, subset],
        "slip_reward": slip_reward[:, subset], "total_reward": total_reward[:, subset],
        "foot_speed": data["tangential_speed"][:, subset],
        "contact_age": data["contact_age"][:, subset],
        "valid": valid_subset, "episode_ids": episode_ids,
        "gamma": gamma, "lam": lam,
    })
    global_episode += episode_count

    # Fixed 0.2 s segments preserve speed and contact pattern.
    steps, episodes = valid.shape
    for episode in range(episodes):
        for begin in range(0, steps - SEGMENT + 1, SEGMENT):
            end = begin + SEGMENT
            if not bool(valid[begin:end, episode].all()):
                continue
            time_s = begin * float(data["dt"])
            if family == "steady":
                phase = "STEADY"
            elif time_s < 3.0:
                phase = "SOURCE"
            elif time_s < 4.5:
                phase = "RAMP"
            else:
                phase = "TARGET"
            contacts = data["foot_contact"][begin:end, episode]
            bits = (contacts.long() * torch.tensor([1, 2, 4, 8])).sum(1)
            pattern = int(torch.mode(bits).values)
            actual = data["actual_speed"][begin:end, episode]
            normal = data["normal_force"][begin:end, episode]
            segment_rows.append({
                "family": family, "condition": label, "source_speed": source,
                "target_speed": target, "phase": phase, "contact_pattern": pattern,
                "episode": episode, "begin_step": begin,
                "raw_slip_score": float(data["raw_slip_score"][begin:end, episode].mean()),
                "speed_mae": float((actual - target).abs().mean()),
                "heading_p95": float(data["heading_error"][begin:end, episode].abs().quantile(0.95)),
                "fall": bool(data["fall"][begin:end, episode].any()),
                "gravity_tilt_p95": float(data["gravity_tilt"][begin:end, episode].quantile(0.95)),
                "duty_factor": float(contacts.float().mean()),
                "flight_fraction": float(data["flight"][begin:end, episode].float().mean()),
                "contact_duration_steps": float((data["contact_age"][begin:end, episode] > 0).float().sum(0).mean()),
                "normal_force_mean": float(normal.mean()),
                "friction_utilization_p95": float(data["friction_utilization"][begin:end, episode].quantile(0.95)),
                "action_rate": float(data["action_rate"][begin:end, episode].mean()),
                "saturation": bool(data["saturation"][begin:end, episode].any()),
                "base_reward": float(base_reward[begin:end, episode].mean()),
            })

    # Temporal event and action-to-slip analysis.
    raw_score = data["raw_slip_score"]
    action_change = data["action_rate"]
    event_durations, onset_ages, release_distances = [], [], []
    foot_persistence = []
    for episode in range(episodes):
        active = (raw_score[:, episode] > 0) & valid[:, episode]
        start = None
        for step, flag in enumerate(active.tolist() + [False]):
            if flag and start is None:
                start = step
            elif not flag and start is not None:
                event_durations.append((step - start) * float(data["dt"]))
                ages = data["contact_age"][start:step, episode]
                onset_ages.append(float(ages[0].float().mean()))
                release_distances.append(float(ages[-1].float().mean()))
                start = None
        foot_active = (
            (data["tangential_speed"][:, episode] > 0.30)
            & (data["contact_age"][:, episode] >= 3)
        )
        foot_persistence.extend(foot_active.float().mean(0).tolist())
    autocorrelation = {}
    score_flat = raw_score.flatten().numpy()
    for lag in range(1, 11):
        autocorrelation[str(lag)] = correlation(score_flat[:-lag], score_flat[lag:])
    for lag in range(11):
        left = action_change[:-lag or None].flatten().numpy()
        right = raw_score[lag:].flatten().numpy()
        lag_accumulator[lag].append(correlation(left, right))
    ages = data["contact_age"]
    phase_values = {}
    for phase_name, mask in (
        ("ONSET", (ages >= 3) & (ages <= 5)),
        ("MID_STANCE", (ages >= 6) & (ages <= 20)),
        ("LATE_STANCE", ages > 20),
    ):
        expanded_score = raw_score[..., None].expand_as(ages)
        phase_values[phase_name] = stats(expanded_score[mask].numpy())
    temporal_by_condition.append({
        "family": family, "condition": label,
        "event_frequency_per_episode": float((raw_score > 0).float().sum() / episodes),
        "event_duration_s": stats(event_durations),
        "contact_age_at_event_onset": stats(onset_ages),
        "contact_age_at_event_offset": stats(release_distances),
        "per_foot_persistence": stats(foot_persistence),
        "autocorrelation": autocorrelation,
        "contact_phase": phase_values,
    })

# Within-condition/contact-pattern deciles.
groups = defaultdict(list)
for row in segment_rows:
    groups[(row["family"], row["condition"], row["phase"], row["contact_pattern"])].append(row)
decile_rows = []
for key, rows in groups.items():
    if len(rows) < 30:
        continue
    values = np.asarray([row["raw_slip_score"] for row in rows])
    cuts = np.percentile(values, [10, 45, 55, 90])
    selections = {
        "LOWEST_10": [row for row in rows if row["raw_slip_score"] <= cuts[0]],
        "MIDDLE_10": [row for row in rows if cuts[1] <= row["raw_slip_score"] <= cuts[2]],
        "HIGHEST_10": [row for row in rows if row["raw_slip_score"] >= cuts[3]],
    }
    for decile, selected in selections.items():
        if not selected:
            continue
        decile_rows.append({
            "family": key[0], "condition": key[1], "phase": key[2],
            "contact_pattern": key[3], "decile": decile, "segments": len(selected),
            "raw_slip_score": np.mean([row["raw_slip_score"] for row in selected]),
            "speed_mae": np.mean([row["speed_mae"] for row in selected]),
            "heading_p95": np.percentile([row["heading_p95"] for row in selected], 95),
            "fall_rate": np.mean([row["fall"] for row in selected]),
            "gravity_tilt_p95": np.percentile([row["gravity_tilt_p95"] for row in selected], 95),
            "duty_factor": np.mean([row["duty_factor"] for row in selected]),
            "flight_fraction": np.mean([row["flight_fraction"] for row in selected]),
            "contact_duration_steps": np.mean([row["contact_duration_steps"] for row in selected]),
            "normal_force_mean": np.mean([row["normal_force_mean"] for row in selected]),
            "friction_utilization_p95": np.percentile([row["friction_utilization_p95"] for row in selected], 95),
            "action_rate": np.mean([row["action_rate"] for row in selected]),
            "saturation_rate": np.mean([row["saturation"] for row in selected]),
            "base_reward": np.mean([row["base_reward"] for row in selected]),
        })
write_csv("reward_decile_behavior_comparison.csv", decile_rows)
ranking_summary = {}
for decile in ("LOWEST_10", "MIDDLE_10", "HIGHEST_10"):
    rows = [row for row in decile_rows if row["decile"] == decile]
    ranking_summary[decile] = {
        "groups": len(rows),
        "raw_slip_score": stats([row["raw_slip_score"] for row in rows]),
        "speed_mae": stats([row["speed_mae"] for row in rows]),
        "heading_p95": stats([row["heading_p95"] for row in rows]),
        "base_reward": stats([row["base_reward"] for row in rows]),
    }
dump("within_condition_reward_ranking.json", {
    "stratification": "family + speed condition + schedule phase + four-foot contact pattern",
    "segment_duration_s": SEGMENT * 0.02,
    "results": ranking_summary,
    "speed_mixing_prohibited": True,
})

# Safe low-slip availability.
availability = {}
availability_rates = []
for speed in MAJOR:
    rows = [
        row for row in segment_rows
        if row["family"] == "steady" and abs(row["target_speed"] - speed) < 1e-9
        and row["begin_step"] >= 75
    ]
    threshold = np.percentile([row["raw_slip_score"] for row in rows], 20)
    qualifying = [
        row for row in rows
        if row["raw_slip_score"] <= threshold and row["speed_mae"] <= 0.15
        and row["heading_p95"] <= 0.12 and not row["fall"] and not row["saturation"]
        and row["flight_fraction"] <= 0.05
    ]
    rate = len(qualifying) / max(1, len(rows))
    availability_rates.append(rate)
    availability[f"{speed:g}"] = {
        "segments": len(rows), "lower_20_threshold": float(threshold),
        "safe_low_slip_segments": len(qualifying), "safe_rate_all_segments": rate,
    }
if all(rate >= 0.05 for rate in availability_rates):
    availability_class = "LOW_SLIP_BEHAVIOR_EXISTS"
elif any(rate >= 0.01 for rate in availability_rates):
    availability_class = "LOW_SLIP_BEHAVIOR_RARE"
else:
    availability_class = "LOW_SLIP_BEHAVIOR_NOT_OBSERVED"
dump("low_slip_behavior_availability.json", {
    "classification": availability_class, "by_speed": availability,
})

lag_rows = [{
    "lag_steps": lag, "lag_s": lag * 0.02,
    "action_change_to_slip_correlation": float(np.mean(values)),
    "condition_std": float(np.std(values)),
} for lag, values in lag_accumulator.items()]
write_csv("action_slip_lag_correlation.csv", lag_rows)
best_lag = max(lag_rows, key=lambda row: abs(row["action_change_to_slip_correlation"]))
if best_lag["lag_steps"] <= 2:
    credit = "IMMEDIATE_OR_SHORT_LAG"
elif best_lag["lag_steps"] <= 10 and abs(best_lag["action_change_to_slip_correlation"]) >= 0.10:
    credit = "MEDIUM_LAG"
else:
    credit = "TEMPORALLY_NOISY_SIGNAL"
dump("slip_reward_temporal_structure.json", {
    "classification": credit, "best_lag": best_lag,
    "by_condition": temporal_by_condition,
})
dump("slip_return_contribution.json", {
    "gamma": adv_chunks[0]["gamma"], "finite_horizon": True,
    "by_condition": return_summary,
})

# Diagnostic V_slip.
observations = torch.cat(value_obs)
targets = torch.cat(value_target).unsqueeze(1)
episode_ids = torch.cat(value_episode)
rng = np.random.default_rng(20272901)
unique = episode_ids.unique().numpy()
rng.shuffle(unique)
train_end, val_end = int(0.70 * len(unique)), int(0.85 * len(unique))
split_ids = {
    "train": set(unique[:train_end]), "validation": set(unique[train_end:val_end]),
    "test": set(unique[val_end:]),
}
masks = {
    name: torch.tensor([int(value) in ids for value in episode_ids], dtype=torch.bool)
    for name, ids in split_ids.items()
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(
    nn.Linear(48, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 1)
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
target_scale = float(targets[masks["train"]].std().clamp_min(1.0e-8))
target_mean = float(targets[masks["train"]].mean())
train_indices = torch.where(masks["train"])[0]
generator = torch.Generator().manual_seed(20272901)
for epoch in range(20):
    permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
    for begin in range(0, len(permutation), 4096):
        index = permutation[begin:begin + 4096]
        prediction = model(observations[index].to(device))
        normalized_target = (targets[index].to(device) - target_mean) / target_scale
        loss = (prediction - normalized_target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate(mask):
    index = torch.where(mask)[0]
    predictions = []
    with torch.inference_mode():
        for begin in range(0, len(index), 8192):
            predictions.append(model(observations[index[begin:begin + 8192]].to(device)).cpu())
    prediction = torch.cat(predictions) * target_scale + target_mean
    truth = targets[index]
    error = prediction - truth
    r2 = 1.0 - float(error.square().sum() / (truth - truth.mean()).square().sum().clamp_min(1e-12))
    return {
        "samples": len(index), "r2": r2, "mae": float(error.abs().mean()),
        "prediction": prediction.squeeze(1), "truth": truth.squeeze(1),
    }


value_results = {name: evaluate(mask) for name, mask in masks.items()}
dump("diagnostic_slip_value_config.json", {
    "architecture": [48, 128, 128, 1], "activation": "ELU",
    "optimizer": "Adam", "learning_rate": 1.0e-3, "epochs": 20,
    "episode_split": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "target_mean": target_mean, "target_scale": target_scale,
})
dump("diagnostic_slip_value_results.json", {
    name: {key: value for key, value in result.items() if key not in ("prediction", "truth")}
    for name, result in value_results.items()
} | {
    "reliability": "RELIABLE" if value_results["test"]["r2"] >= 0.30 else "LIMITED",
    "r2_reliability_threshold": 0.30,
})

# Predict V_slip and derive additive GAE components on the episode subset.
model.eval()
all_a_slip, all_a_total, all_a_base = [], [], []
gradient_obs, gradient_action, gradient_old_log_prob = [], [], []
gradient_speed, gradient_family = [], []
gradient_foot_speed, gradient_contact_phase = [], []
for chunk in adv_chunks:
    obs = chunk["observation"]
    steps, episodes = obs.shape[:2]
    flat_prediction = []
    with torch.inference_mode():
        flat = obs.flatten(0, 1)
        for begin in range(0, len(flat), 8192):
            flat_prediction.append(model(flat[begin:begin + 8192].to(device)).cpu())
    slip_value = (torch.cat(flat_prediction).reshape(steps, episodes) * target_scale + target_mean)
    total_value = chunk["production_value"]
    a_slip = torch.zeros_like(slip_value)
    a_total = torch.zeros_like(total_value)
    gae_slip = torch.zeros(episodes)
    gae_total = torch.zeros(episodes)
    for step in reversed(range(steps)):
        valid_step = chunk["valid"][step]
        next_slip = slip_value[step + 1] if step + 1 < steps else torch.zeros(episodes)
        next_total = total_value[step + 1] if step + 1 < steps else torch.zeros(episodes)
        delta_slip = chunk["slip_reward"][step] + chunk["gamma"] * next_slip - slip_value[step]
        delta_total = chunk["total_reward"][step] + chunk["gamma"] * next_total - total_value[step]
        gae_slip = delta_slip + chunk["gamma"] * chunk["lam"] * gae_slip
        gae_total = delta_total + chunk["gamma"] * chunk["lam"] * gae_total
        gae_slip = torch.where(valid_step, gae_slip, torch.zeros_like(gae_slip))
        gae_total = torch.where(valid_step, gae_total, torch.zeros_like(gae_total))
        a_slip[step], a_total[step] = gae_slip, gae_total
    a_base = a_total - a_slip
    valid = chunk["valid"]
    all_a_slip.append(a_slip[valid]); all_a_total.append(a_total[valid]); all_a_base.append(a_base[valid])
    gradient_obs.append(obs[valid]); gradient_action.append(chunk["action"][valid])
    gradient_old_log_prob.append(chunk["old_log_prob"][valid])
    gradient_speed.append(torch.full((int(valid.sum()),), chunk["target"]))
    gradient_family.append(torch.full((int(valid.sum()),), 0 if chunk["family"] == "steady" else 1))
    gradient_foot_speed.append(chunk["foot_speed"][valid])
    ages = chunk["contact_age"][valid]
    gradient_contact_phase.append(torch.where(
        (ages >= 3).any(1) & (ages <= 5).any(1), torch.zeros(len(ages), dtype=torch.long),
        torch.where((ages > 20).any(1), torch.full((len(ages),), 2, dtype=torch.long),
                    torch.ones(len(ages), dtype=torch.long)),
    ))

a_slip = torch.cat(all_a_slip)
a_total = torch.cat(all_a_total)
a_base = torch.cat(all_a_base)
scale = a_total.std().clamp_min(1.0e-8)
a_slip_norm = (a_slip - a_slip.mean()) / scale
a_base_norm = (a_base - a_base.mean()) / scale
a_total_norm = (a_total - a_total.mean()) / scale
dump("slip_advantage_decomposition.json", {
    "diagnostic_approximation": True,
    "value_model_test_r2": value_results["test"]["r2"],
    "reliability": "RELIABLE" if value_results["test"]["r2"] >= 0.30 else "LIMITED",
    "A_slip": stats(a_slip.numpy()), "A_base": stats(a_base.numpy()),
    "A_total": stats(a_total.numpy()),
    "correlations": {
        "slip_base": correlation(a_slip.numpy(), a_base.numpy()),
        "slip_total": correlation(a_slip.numpy(), a_total.numpy()),
        "base_total": correlation(a_base.numpy(), a_total.numpy()),
    },
    "sign_agreement": {
        "slip_total": float((torch.sign(a_slip) == torch.sign(a_total)).float().mean()),
        "base_total": float((torch.sign(a_base) == torch.sign(a_total)).float().mean()),
    },
})

# Deterministic stratified fixed actor-gradient batch.
gradient_payload = {
    "observation": torch.cat(gradient_obs),
    "action": torch.cat(gradient_action),
    "old_log_prob": torch.cat(gradient_old_log_prob),
    "A_slip": a_slip_norm, "A_base": a_base_norm, "A_total": a_total_norm,
    "target_speed": torch.cat(gradient_speed), "family": torch.cat(gradient_family),
    "foot_speed": torch.cat(gradient_foot_speed),
    "contact_phase": torch.cat(gradient_contact_phase),
}
total = len(gradient_payload["observation"])
selection = torch.randperm(total, generator=torch.Generator().manual_seed(20272901))[:49152]
gradient_payload = {key: value[selection].contiguous() for key, value in gradient_payload.items()}
gradient_path = RAW / "gradient_batch.pt"
torch.save(gradient_payload, gradient_path)

dump("directionality_rollout_manifest.json", {
    "seed_root": 20272901, "files": manifest_files,
    "raw_files_git_policy": "EXCLUDED_LARGE_DIAGNOSTIC_TENSORS",
    "total_samples": sum(item["samples"] for item in manifest_files),
    "total_valid_samples": sum(item["valid_samples"] for item in manifest_files),
    "gradient_batch": {
        "path": str(gradient_path.resolve()), "sha256": sha(gradient_path),
        "samples": len(selection),
    },
    "stored_fields": list(torch.load(files[0], map_location="cpu", weights_only=False).keys()),
    "production_ppo_update": 0, "reward_optimization": 0,
})
print(json.dumps({
    "availability": availability_class, "credit": credit,
    "value_test_r2": value_results["test"]["r2"], "segments": len(segment_rows),
}, indent=2))
