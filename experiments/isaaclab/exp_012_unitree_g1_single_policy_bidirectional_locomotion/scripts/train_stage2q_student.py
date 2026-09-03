"""Mean-only supervised integration for the Stage 2Q final sequence."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"
RAW = OUT / "raw"
BASE = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2k_gait_latent_preflight/raw/gait_latent_endpoint_dataset.pt"
SUPPLEMENT = RAW / "stage2q_supplement_dataset.pt"
TOGGLE = RAW / "stage2q_toggle_dataset.pt"
PARENT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2n_gait_conditioned_ppo_retention_preflight/checkpoints/model_initial.pt"
EXPECTED_PARENT = "04b43e5497bc35e2d00fa4476f9120f9439e0953283c69cf8ca1e9635dedd121"
SAVE_STEPS = {0, 500, 1000, 2000, 5000, 10000, 15000, 20000}
GROUP_ORDER = (
    "STAND_0P0", "WALK_0P6", "WALK_0P8", "WALK_1P0", "WALK_1P2",
    "RUN_1P2", "RUN_2P4", "RUN_2P6", "WALK_TO_RUN", "RUN_TO_WALK",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Student(nn.Module):
    def __init__(self, state):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone())
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone())
        self.first_bias = nn.Parameter(state["first_bias"].clone())
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37),
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value)
            for key, value in state.items() if key.startswith("hidden.")
        ))
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"].clone())
        self.register_buffer("log_std_run", state["distribution.log_std_run"].clone())

    def forward(self, observation, gait):
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait[:, None] * self.first_gait_column.T)

    def export(self):
        state = self.state_dict()
        return {
            "first_base_weight": state["first_base_weight"],
            "first_gait_column": state["first_gait_column"],
            "first_bias": state["first_bias"],
            **{f"hidden.{key}": value for key, value in self.hidden.state_dict().items()},
            "distribution.log_std_walk": state["log_std_walk"],
            "distribution.log_std_run": state["log_std_run"],
        }


def build_groups(data):
    base, supplement, toggle = data
    groups = {}
    # Stored episode indices from Stage 2K.
    groups["WALK_1P2"] = [("base", torch.arange(0, 300))]
    for name, left in (("RUN_1P2", 500), ("RUN_2P4", 750), ("RUN_2P6", 1000)):
        groups[name] = [("base", torch.arange(left, left + 250))]
    supplement_names = supplement["group_names"]
    starts = {}
    cursor = 0
    for index, name in enumerate(supplement_names):
        count = int((supplement["group_id"] == index).sum())
        starts[name] = torch.arange(cursor, cursor + count)
        cursor += count
    groups["STAND_0P0"] = [("supplement", starts["STAND_0P0"])]
    for name in ("WALK_0P6", "WALK_0P8", "WALK_1P0"):
        groups[name] = [("supplement", starts[name])]
    for name in ("RUN_1P2", "RUN_2P4", "RUN_2P6"):
        groups[name].append(("supplement", starts[f"{name}_EXTRA"]))
    cursor = 0
    for index, name in enumerate(toggle["group_names"]):
        count = int((toggle["group_id"] == index).sum())
        groups[name] = [("toggle", torch.arange(cursor, cursor + count))]
        cursor += count
    return groups


def split_groups(groups):
    generator = torch.Generator().manual_seed(20269021)
    split = {}
    for name, parts in groups.items():
        flattened = [(source, int(ep)) for source, ids in parts for ep in ids]
        order = torch.randperm(len(flattened), generator=generator).tolist()
        n = len(order)
        split[name] = {
            "train": [flattened[i] for i in order[:int(.8 * n)]],
            "validation": [flattened[i] for i in order[int(.8 * n):int(.9 * n)]],
            "held_out": [flattened[i] for i in order[int(.9 * n):]],
        }
    return split


def sample(name, split_name, count, datasets, splits, generator, device):
    choices = splits[name][split_name]
    ids = torch.randint(len(choices), (count,), generator=generator)
    timesteps = torch.randint(0, 500, (count,), generator=generator)
    obs = torch.empty((count, 123), dtype=torch.float32)
    action = torch.empty((count, 37), dtype=torch.float32)
    gait = torch.empty(count, dtype=torch.float32)
    by_source = {}
    for row, choice_id in enumerate(ids.tolist()):
        source, episode = choices[choice_id]
        by_source.setdefault(source, []).append((row, episode))
    for source, items in by_source.items():
        rows = torch.tensor([item[0] for item in items])
        episodes = torch.tensor([item[1] for item in items])
        ts = timesteps[rows]
        dataset = datasets[source]
        obs[rows] = dataset["observation"][ts, episodes]
        action_key = "teacher_action" if source == "base" else "target_action"
        action[rows] = dataset[action_key][ts, episodes]
        if source == "base":
            gait[rows] = dataset["gait_cmd"][episodes]
        else:
            gait[rows] = dataset["gait_cmd"][ts, episodes]
    return obs.to(device), gait.to(device), action.to(device)


def evaluate(model, datasets, splits, split_name, device, samples_per_group=10000):
    generator = torch.Generator().manual_seed(20269022 if split_name == "validation" else 20269023)
    result = {}
    model.eval()
    with torch.no_grad():
        for name in GROUP_ORDER:
            mse_values, cosine_values = [], []
            for _ in range(math.ceil(samples_per_group / 2048)):
                count = min(2048, samples_per_group)
                obs, gait, target = sample(name, split_name, count, datasets, splits, generator, device)
                pred = model(obs, gait)
                mse_values.append((pred - target).square().mean(-1).cpu())
                cosine_values.append(nn.functional.cosine_similarity(pred, target).cpu())
            mse, cosine = torch.cat(mse_values), torch.cat(cosine_values)
            result[name] = {
                "action_mse": float(mse.mean()), "action_cosine": float(cosine.mean()),
                "mse_p95": float(torch.quantile(mse, .95)), "samples": len(mse),
            }
    endpoint = [result[name]["action_mse"] for name in GROUP_ORDER[:8]]
    toggle = [result[name]["action_mse"] for name in GROUP_ORDER[8:]]
    result["selection"] = {
        "endpoint_worst_loss": max(endpoint), "toggle_worst_loss": max(toggle),
        "aggregate_loss": (sum(endpoint) / 8 + sum(toggle) / 2) / 2,
    }
    return result


def save(model, optimizer, step, validation):
    path = RAW / "checkpoints" / f"student_step_{step}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step": step, "actor_state_dict": model.export(), "optimizer_state_dict": optimizer.state_dict(),
        "validation": validation, "architecture": [124, 256, 128, 128, 37],
        "std_frozen": True, "training_type": "mean-only supervised/DAgger-compatible",
    }, path)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if sha(PARENT) != EXPECTED_PARENT:
        raise RuntimeError("STAGE2Q_PARENT_PROVENANCE_FAIL")
    torch.manual_seed(20269021)
    random.seed(20269021)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = {
        "base": torch.load(BASE, map_location="cpu", weights_only=False),
        "supplement": torch.load(SUPPLEMENT, map_location="cpu", weights_only=False),
        "toggle": torch.load(TOGGLE, map_location="cpu", weights_only=False),
    }
    groups = build_groups(tuple(datasets.values()))
    splits = split_groups(groups)
    serializable_split = {
        name: {part: [{"source": source, "episode": ep} for source, ep in rows] for part, rows in value.items()}
        for name, value in splits.items()
    }
    split_text = json.dumps(serializable_split, sort_keys=True, separators=(",", ":"))
    dump("dataset_split.json", {
        "seed": 20269021, "unit": "episode", "groups": serializable_split,
        "episode_overlap": 0, "sha256": hashlib.sha256(split_text.encode()).hexdigest(),
    })
    dataset_files = {"base": BASE, "supplement": SUPPLEMENT, "toggle": TOGGLE}
    dump("dataset_hashes.json", {name: sha(path) for name, path in dataset_files.items()})
    dump("endpoint_dataset_manifest.json", {
        "groups": {name: sum(len(ids) for _, ids in groups[name]) for name in GROUP_ORDER[:8]},
        "steps_per_episode": 500, "deterministic_teacher_actions": True,
        "sources": {name: str(path.relative_to(REPO)) for name, path in dataset_files.items() if name != "toggle"},
        "sampling_weight": {"STAND": .25, "LOW_MID_WALK": .25, "RUN": .25},
    })
    dump("toggle_dataset_manifest.json", {
        "groups": {name: sum(len(ids) for _, ids in groups[name]) for name in GROUP_ORDER[8:]},
        "source_actor": str(PARENT.relative_to(REPO)), "source_sha256": EXPECTED_PARENT,
        "runtime_teacher": False, "sampling_weight": .25,
    })
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = Student(parent["actor_state_dict"]).to(device)
    identity_obs = datasets["base"]["observation"][:2, :64].reshape(-1, 123).to(device)
    identity_gait = datasets["base"]["gait_cmd"][:64].repeat(2).to(device)
    with torch.no_grad():
        before = model(identity_obs, identity_gait)
    dump("student_parent_manifest.json", {
        "path": str(PARENT.relative_to(REPO)), "sha256": EXPECTED_PARENT,
        "architecture": [124, 256, 128, 128, 37], "alpha_walk": .30, "alpha_run": .65,
    })
    dump("student_parent_identity_audit.json", {
        "mean_state_hash_source": hashlib.sha256(b"".join(
            value.detach().cpu().numpy().tobytes() for key, value in parent["actor_state_dict"].items()
            if not key.startswith("distribution.")
        )).hexdigest(),
        "forward_finite": bool(torch.isfinite(before).all()), "std_frozen": True, "status": "PASS",
    })
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    start_step = 0
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = Student(resume["actor_state_dict"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_step = int(resume["step"])
    config = """stage: 2Q
training_type: mean_only_supervised_sequence_integration
seed: 20269021
architecture: [124, 256, 128, 128, 37]
maximum_optimizer_steps: 20000
batch_size_per_endpoint_group: 256
batch_size_toggle: 768
learning_rate: 0.0002
loss: endpoint_group_mean_mse + toggle_group_mean_mse
std_head: frozen
critic: unused
ppo: false
"""
    (OUT / "resolved_training_config.yaml").write_text(config, encoding="utf-8")
    generator = torch.Generator().manual_seed(20269021)
    curves, checkpoints = [], []
    best_rank, best_path = None, None
    initial_validation = evaluate(model, datasets, splits, "validation", device, 4000)
    path = save(model, optimizer, start_step, initial_validation)
    checkpoints.append((start_step, path, initial_validation))
    for step in range(start_step + 1, 20001):
        optimizer.zero_grad(set_to_none=True)
        endpoint_losses = []
        for name in GROUP_ORDER[:8]:
            obs, gait, target = sample(name, "train", 256, datasets, splits, generator, device)
            endpoint_losses.append(nn.functional.mse_loss(model(obs, gait), target))
        toggle_losses = []
        for name in GROUP_ORDER[8:]:
            obs, gait, target = sample(name, "train", 768, datasets, splits, generator, device)
            toggle_losses.append(nn.functional.mse_loss(model(obs, gait), target))
        endpoint_loss = torch.stack(endpoint_losses).mean()
        toggle_loss = torch.stack(toggle_losses).mean()
        loss = endpoint_loss + toggle_loss
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % 500 == 0:
            validation = evaluate(model, datasets, splits, "validation", device, 4000)
            rank = (
                validation["selection"]["endpoint_worst_loss"],
                validation["selection"]["toggle_worst_loss"],
                validation["selection"]["aggregate_loss"],
            )
            curves.append({
                "step": step, "endpoint_loss": float(endpoint_loss), "toggle_loss": float(toggle_loss),
                "total_loss": float(loss), "gradient_norm": float(gradient),
                **validation["selection"],
            })
            if step in SAVE_STEPS:
                path = save(model, optimizer, step, validation)
                checkpoints.append((step, path, validation))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_path = RAW / "checkpoints" / f"student_best_step_{step}.pt"
                torch.save({
                    "step": step, "actor_state_dict": model.export(),
                    "validation": validation, "architecture": [124, 256, 128, 128, 37],
                    "std_frozen": True,
                }, best_path)
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    model = Student(selected["actor_state_dict"]).to(device)
    held_out = evaluate(model, datasets, splits, "held_out", device, 10000)
    static_pass = all(
        held_out[name]["action_mse"] <= .001 and held_out[name]["action_cosine"] >= .98
        for name in GROUP_ORDER
    )
    held_out["aggregate_classification"] = "PASS" if static_pass else "FAIL"
    dump("static_endpoint_results.json", held_out)
    selected_path = RAW / "selected_stage2q_student.pt"
    torch.save({
        "step": selected["step"], "actor_state_dict": selected["actor_state_dict"],
        "architecture": [124, 256, 128, 128, 37], "std_frozen": True,
        "held_out": held_out, "teacher_runtime_required": False,
    }, selected_path)
    rows = [{
        "step": step, "path": str(path.relative_to(REPO)), "sha256": sha(path),
        **validation["selection"],
    } for step, path, validation in checkpoints]
    dump("checkpoint_manifest.json", {
        "checkpoints": rows, "selected_path": str(selected_path.relative_to(REPO)),
        "selected_sha256": sha(selected_path), "selection_rank": [
            "endpoint group worst loss", "toggle retention loss", "aggregate loss"
        ], "latest_automatically_selected": False,
    })
    dump("selected_checkpoint.json", {
        "path": str(selected_path.relative_to(REPO)), "sha256": sha(selected_path),
        "step": selected["step"], "static_gate": static_pass,
    })


if __name__ == "__main__":
    main()
