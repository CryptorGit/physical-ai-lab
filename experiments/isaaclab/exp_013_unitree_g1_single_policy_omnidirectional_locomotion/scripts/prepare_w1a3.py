"""Create immutable-reference manifests and reconstruct W1A2 command exposure."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a3_rear_left_low_speed_retention_diagnosis"
W1A_DIR = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
W1A2_DIR = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True)
log = subprocess.check_output(["git", "log", "--oneline", "--decorate", "-20"], cwd=REPO, text=True)
dump("stage_reference.json", {
    "stage": "Phase W1A3",
    "starting_head": head,
    "w1a_sha256": "b128f6b164d151b411eeaf2caf22edc1ea2a69e68fca9534e7d6a965ae4dbba9",
    "w1a2_sha256": "3cb25a32c24b0ca9d8e70e1f418942d28c46da22dd542a4830e33bf54101ef2b",
    "starting_status_short": status.splitlines(),
    "starting_log": log.splitlines(),
})
dump("protocol.json", {
    "training_updates": 0,
    "checkpoint_updates": 0,
    "reward_changes": 0,
    "curriculum_changes": 0,
    "persistent_parameter_interpolations": 0,
    "yaw_rate_cmd": 0,
    "gait_cmd": 0,
    "formal_evaluation_seed": 20271021,
    "fresh_diagnostic_seed": 20273021,
    "timeline_episodes_per_condition": 50,
    "boundary_episodes_per_condition": 20,
    "interpolation_episodes_per_condition": 20,
})

entries = [{"name": "w1a_selected", "iteration": 120,
            "path": str(W1A_DIR / "checkpoints/model_120.pt"),
            "sha256": sha(W1A_DIR / "checkpoints/model_120.pt")}]
for label in ("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160"):
    path = W1A2_DIR / f"checkpoints/model_{label}.pt"
    entries.append({"name": f"w1a2_{label}", "iteration": 0 if label == "initial" else int(label),
                    "path": str(path), "sha256": sha(path)})
dump("checkpoint_manifest.json", {"entries": entries, "all_existing_read_only": True})

# Contract-level exposure reconstruction. Historical command tensors and per-bin
# advantages were not persisted, so these counts are a deterministic schedule
# reconstruction and are explicitly not represented as exact historical samples.
sector = [67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315]
e1 = [0.45, 0.40, 0.40, 0.45, 0.45, 0.45, 0.45, 0.40, 0.40, 0.40, 0.40, 0.45]
generator = torch.Generator().manual_seed(20272021)
bins = defaultdict(lambda: {"episode_count": 0, "rollout_sample_count": 0})


def rand(n):
    return torch.rand(n, generator=generator)


for iteration in range(1, 161):
    n = 1024
    u = rand(n)
    theta = torch.empty(n)
    speed = torch.empty(n)
    a, b, c, d = u < .20, (u >= .20) & (u < .70), (u >= .70) & (u < .90), u >= .90
    theta[a] = rand(int(a.sum())) * 2 * math.pi - math.pi
    speed[a] = .25 + rand(int(a.sum())) * .10
    if b.any():
        idx = torch.randint(len(sector), (int(b.sum()),), generator=generator)
        centers = torch.tensor(sector)[idx] * math.pi / 180
        theta[b] = centers + (rand(int(b.sum())) - .5) * (math.pi / 8)
        maxima = torch.tensor(e1)[idx] if iteration <= 40 else torch.full((int(b.sum()),), .50 if iteration <= 80 else .60)
        speed[b] = .30 + rand(int(b.sum())) * (maxima - .30)
    if c.any():
        m = int(c.sum())
        anchor = rand(m)
        diagonal = anchor >= .5
        signs = torch.where(rand(m) < .5, -1., 1.)
        theta[c] = torch.where(diagonal, signs * (math.pi / 4) * (.75 + rand(m) * .25), torch.zeros(m))
        speed[c] = torch.where(diagonal, .6 + rand(m) * .4, .6 + rand(m) * .6)
    if d.any():
        ids = torch.where(d)[0]
        pairs = (len(ids) // 2) * 2
        base = rand(pairs // 2) * math.pi
        pair_theta = torch.stack((base, -base), 1).flatten()
        pair_speed = .3 + rand(pairs // 2) * .3
        theta[ids[:pairs]] = pair_theta
        speed[ids[:pairs]] = torch.stack((pair_speed, pair_speed), 1).flatten()
        if pairs < len(ids):
            theta[ids[-1]], speed[ids[-1]] = 0, .6
    if iteration > 130:
        target = rand(n) < .55
        theta[target] = rand(int(target.sum())) * 2 * math.pi - math.pi
        speed[target] = .55 + rand(int(target.sum())) * .05
    degrees = torch.remainder(torch.rad2deg(theta), 360)
    angle_bin = torch.remainder(torch.round(degrees / 22.5) * 22.5, 360)
    for angle, value in zip(angle_bin.tolist(), speed.tolist()):
        speed_bin = ("0.20-0.30" if value < .30 else "0.30-0.40" if value < .40 else
                     "0.40-0.50" if value < .50 else "0.50-0.60" if value < .60 else "0.60+")
        key = (float(angle), speed_bin)
        bins[key]["episode_count"] += 1
        bins[key]["rollout_sample_count"] += 24

rows = []
for (angle, speed_bin), values in sorted(bins.items()):
    rows.append({
        "angle_bin_deg": angle,
        "speed_bin_mps": speed_bin,
        **values,
        "ppo_minibatch_inclusion_count": values["rollout_sample_count"] * 5,
        "mean_return": None,
        "mean_advantage": None,
        "positive_advantage_rate": None,
        "fall": None,
        "slip": None,
        "tilt": None,
        "provenance": "deterministic schedule reconstruction; historical tensors not persisted",
    })
with (OUT / "w1a2_command_exposure_audit.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
dump("w1a2_command_exposure_audit.json", {
    "rows": rows,
    "historical_command_tensor_available": False,
    "historical_return_advantage_by_bin_available": False,
    "classification": "EXPOSURE_BALANCED_BY_FIXED_SAMPLING_CONTRACT",
    "interpretation_limit": "Counts reconstruct the fixed sampler, not the consumed global RNG stream.",
})

dump("current_walk_artifact_interpretation.json", {
    "w1a": {"role": "all-direction 0.3m/s WALK artifact", "pass_0p3": "16/16"},
    "w1a2": {"role": "improved 0.6m/s speed-expansion artifact", "pass_0p6_average": "30% -> 75%",
             "limitation": "localized 225/247.5 degree low-speed retention loss"},
    "final_omnidirectional_policy": False,
})
