"""Mean-only supervised W2-P1 practical-stop integration."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_p1_practical_stop_endpoint_acquisition"
)
RAW = OUT / "raw"
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
EXPECTED_PARENT = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
SAVE_STEPS = {0, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000}
MOVING_GROUPS = (
    "ZERO_YAW_TRANSLATION", "PURE_YAW", "MOVING_TURN", "INDEPENDENCE", "DYNAMIC_YAW_ENDPOINT",
)
TOP_WEIGHTS = {"STOP_RECOVERY": .35, "STEADY_STOP": .25, "MOVING_RETENTION": .30, "START_RETENTION": .10}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class Student(nn.Module):
    def __init__(self, state: dict[str, torch.Tensor]):
        super().__init__()
        self.first_base_weight = nn.Parameter(state["first_base_weight"].clone())
        self.first_gait_column = nn.Parameter(state["first_gait_column"].clone())
        self.first_bias = nn.Parameter(state["first_bias"].clone())
        self.hidden = nn.Sequential(
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 37)
        )
        self.hidden.load_state_dict(OrderedDict(
            (key.removeprefix("hidden."), value) for key, value in state.items() if key.startswith("hidden.")
        ))
        self.register_buffer("log_std_walk", state["distribution.log_std_walk"].clone())
        self.register_buffer("log_std_run", state["distribution.log_std_run"].clone())

    def forward(self, observation: torch.Tensor, gait: torch.Tensor) -> torch.Tensor:
        first = nn.functional.linear(observation, self.first_base_weight, self.first_bias)
        return self.hidden(first + gait[:, None] * self.first_gait_column.T)

    def export(self) -> dict[str, torch.Tensor]:
        state = self.state_dict()
        return {
            "first_base_weight": state["first_base_weight"],
            "first_gait_column": state["first_gait_column"],
            "first_bias": state["first_bias"],
            **{f"hidden.{key}": value for key, value in self.hidden.state_dict().items()},
            "distribution.log_std_walk": state["log_std_walk"],
            "distribution.log_std_run": state["log_std_run"],
        }


def load_datasets() -> tuple[list[dict], dict[str, list[tuple[int, int]]]]:
    datasets, groups = [], defaultdict(list)
    for path in sorted(RAW.glob("*_chunk_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        dataset_index = len(datasets)
        datasets.append(data)
        for episode_index, subgroup in enumerate(data["subgroup"]):
            group = "ZERO_YAW_TRANSLATION" if subgroup == "FORWARD_ANCHOR" else subgroup
            groups[group].append((dataset_index, episode_index))
    required = {"STOP_RECOVERY", "STEADY_STOP", "START_RETENTION", *MOVING_GROUPS}
    missing = sorted(required - set(groups))
    if missing:
        raise RuntimeError(f"W2_P1_DATASET_GROUP_MISSING:{missing}")
    return datasets, groups


def split_groups(datasets: list[dict], groups: dict[str, list[tuple[int, int]]]) -> dict:
    generator = torch.Generator().manual_seed(20276021)
    result = {}
    for group, references in groups.items():
        strata: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for dataset_index, episode_index in references:
            strata[datasets[dataset_index]["condition"][episode_index]].append((dataset_index, episode_index))
        result[group] = {"train": [], "validation": [], "held_out": []}
        for condition in sorted(strata):
            refs = strata[condition]
            order = torch.randperm(len(refs), generator=generator).tolist()
            n = len(refs); a = int(.8 * n); b = int(.9 * n)
            result[group]["train"].extend(refs[i] for i in order[:a])
            result[group]["validation"].extend(refs[i] for i in order[a:b])
            result[group]["held_out"].extend(refs[i] for i in order[b:])
    return result


def sample(group: str, part: str, count: int, datasets: list[dict], splits: dict,
           generator: torch.Generator, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    choices = splits[group][part]
    choice_ids = torch.randint(len(choices), (count,), generator=generator)
    obs = torch.empty(count, 123); gait = torch.empty(count); target = torch.empty(count, 37)
    by_dataset: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row, choice_id in enumerate(choice_ids.tolist()):
        dataset_index, episode_index = choices[choice_id]
        by_dataset[dataset_index].append((row, episode_index))
    for dataset_index, items in by_dataset.items():
        data = datasets[dataset_index]
        rows = torch.tensor([item[0] for item in items]); episodes = torch.tensor([item[1] for item in items])
        times = torch.randint(data["observation"].shape[0], (len(items),), generator=generator)
        obs[rows] = data["observation"][times, episodes]
        gait[rows] = data["gait_cmd"][times, episodes]
        target[rows] = data["target_action"][times, episodes]
    return obs.to(device), gait.to(device), target.to(device)


def evaluate(model: Student, datasets: list[dict], splits: dict, part: str, device: torch.device,
             samples_per_group: int = 5000) -> dict:
    generator = torch.Generator().manual_seed(20276022 if part == "validation" else 20276023)
    result = {}
    model.eval()
    with torch.inference_mode():
        for group in ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION"):
            mse_parts, cosine_parts = [], []
            remaining = samples_per_group
            while remaining:
                count = min(2048, remaining); remaining -= count
                obs, gait, target = sample(group, part, count, datasets, splits, generator, device)
                pred = model(obs, gait)
                mse_parts.append((pred - target).square().mean(-1).cpu())
                cosine_parts.append(nn.functional.cosine_similarity(pred, target).cpu())
            mse, cosine = torch.cat(mse_parts), torch.cat(cosine_parts)
            result[group] = {
                "action_mse": float(mse.mean()), "action_cosine": float(cosine.mean()),
                "mse_p95": float(torch.quantile(mse, .95)), "samples": len(mse),
                "gate_pass": bool(float(mse.mean()) <= .001 and float(cosine.mean()) >= .98),
            }
    moving_worst_mse = max(result[group]["action_mse"] for group in MOVING_GROUPS)
    moving_worst_cosine = min(result[group]["action_cosine"] for group in MOVING_GROUPS)
    result["selection"] = {
        "moving_worst_group_mse": moving_worst_mse,
        "moving_worst_group_cosine": moving_worst_cosine,
        "moving_static_gate_pass": moving_worst_mse <= .001 and moving_worst_cosine >= .98,
        "aggregate_loss": sum(result[group]["action_mse"] for group in result if group != "selection") / 8,
    }
    return result


def save_checkpoint(model: Student, optimizer: torch.optim.Optimizer, step: int, validation: dict) -> Path:
    path = RAW / "checkpoints" / f"student_step_{step}.pt"; path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": step, "actor_state_dict": model.export(), "optimizer_state_dict": optimizer.state_dict(),
                "validation": validation, "architecture": [124, 256, 128, 128, 37], "std_frozen": True,
                "training_type": "mean-only supervised W2-P1"}, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=25000)
    args = parser.parse_args()
    if sha(PARENT) != EXPECTED_PARENT:
        raise RuntimeError("W2_P1_PARENT_PROVENANCE_FAIL")
    torch.manual_seed(20276021); random.seed(20276021)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, groups = load_datasets(); splits = split_groups(datasets, groups)
    split_json = {group: {part: [{"dataset": d, "episode": e} for d, e in refs]
                          for part, refs in parts.items()} for group, parts in splits.items()}
    split_text = json.dumps(split_json, sort_keys=True, separators=(",", ":"))
    dump("w2_p1_dataset_split.json", {"unit": "episode", "seed": 20276021, "stratified": True,
                                      "episode_overlap": 0, "groups": split_json,
                                      "sha256": hashlib.sha256(split_text.encode()).hexdigest()})
    chunk_paths = sorted(RAW.glob("*_chunk_*.pt"))
    dump("w2_p1_dataset_hashes.json", {str(p.relative_to(REPO)).replace("\\", "/"): sha(p) for p in chunk_paths})
    dump("w2_p1_dataset_manifest.json", {
        "groups": {group: len(refs) for group, refs in groups.items()}, "record_stride": 5,
        "top_level_weights": TOP_WEIGHTS, "moving_retention_subgroup_weighting": "equal",
        "split": {"train": .8, "validation": .1, "held_out": .1}, "runtime_teacher": False,
    })
    parent = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = Student(parent["actor_state_dict"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    (OUT / "resolved_w2_p1_training_config.yaml").write_text(
        "stage: W2-P1\ntraining_type: mean_action_supervised\nseed: 20276021\n"
        "architecture: [124, 256, 128, 128, 37]\nmaximum_optimizer_steps: 25000\n"
        "learning_rate: 0.0002\nstd_head: frozen\ncritic: unused\nppo: false\n"
        "group_weights: {stop_recovery: 0.35, steady_stop: 0.25, moving_retention: 0.30, start_retention: 0.10}\n",
        encoding="utf-8",
    )
    generator = torch.Generator().manual_seed(20276021)
    curves, candidates = [], []
    initial_validation = evaluate(model, datasets, splits, "validation", device, 2000)
    candidates.append((0, save_checkpoint(model, optimizer, 0, initial_validation), initial_validation))
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        obs, gait, target = sample("STOP_RECOVERY", "train", 512, datasets, splits, generator, device)
        stop_loss = nn.functional.mse_loss(model(obs, gait), target)
        obs, gait, target = sample("STEADY_STOP", "train", 512, datasets, splits, generator, device)
        steady_loss = nn.functional.mse_loss(model(obs, gait), target)
        moving_losses = []
        for group in MOVING_GROUPS:
            obs, gait, target = sample(group, "train", 128, datasets, splits, generator, device)
            moving_losses.append(nn.functional.mse_loss(model(obs, gait), target))
        moving_loss = torch.stack(moving_losses).mean()
        obs, gait, target = sample("START_RETENTION", "train", 256, datasets, splits, generator, device)
        start_loss = nn.functional.mse_loss(model(obs, gait), target)
        loss = .35 * stop_loss + .25 * steady_loss + .30 * moving_loss + .10 * start_loss
        loss.backward(); gradient = nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
        if step % 500 == 0:
            validation = evaluate(model, datasets, splits, "validation", device, 2000)
            curves.append({"step": step, "total_loss": float(loss), "stop_recovery_loss": float(stop_loss),
                           "steady_stop_loss": float(steady_loss), "moving_retention_loss": float(moving_loss),
                           "start_retention_loss": float(start_loss), "gradient_norm": float(gradient),
                           **validation["selection"]})
            print(json.dumps(curves[-1]), flush=True)
            if step in SAVE_STEPS:
                candidates.append((step, save_checkpoint(model, optimizer, step, validation), validation))
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(curves[0])); writer.writeheader(); writer.writerows(curves)
    def rank(item: tuple) -> tuple:
        _, _, value = item; selection = value["selection"]
        return (not selection["moving_static_gate_pass"],
                value["STEADY_STOP"]["action_mse"], value["STOP_RECOVERY"]["action_mse"],
                value["START_RETENTION"]["action_mse"], selection["aggregate_loss"])
    selected_step, selected_candidate, _ = min(candidates, key=rank)
    selected = torch.load(selected_candidate, map_location="cpu", weights_only=False)
    model = Student(selected["actor_state_dict"]).to(device)
    held_out = evaluate(model, datasets, splits, "held_out", device, 10000)
    static_pass = all(held_out[group]["gate_pass"] for group in
                      ("STOP_RECOVERY", "STEADY_STOP", *MOVING_GROUPS, "START_RETENTION"))
    held_out["aggregate_classification"] = "PASS" if static_pass else "FAIL"
    dump("static_heldout_results.json", held_out)
    selected_path = RAW / "selected_w2_p1_student.pt"
    torch.save({"step": selected_step, "actor_state_dict": selected["actor_state_dict"],
                "validation": selected["validation"], "held_out": held_out, "std_frozen": True,
                "architecture": [124, 256, 128, 128, 37]}, selected_path)
    dump("selected_checkpoint.json", {"step": selected_step,
        "path": str(selected_path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(selected_path),
        "static_heldout_gate": "PASS" if static_pass else "FAIL"})
    dump("checkpoint_manifest.json", {"checkpoints": [
        {"step": step, "path": str(path.relative_to(REPO)).replace("\\", "/"), "sha256": sha(path)}
        for step, path, _ in candidates], "selected_step": selected_step})


if __name__ == "__main__":
    main()
