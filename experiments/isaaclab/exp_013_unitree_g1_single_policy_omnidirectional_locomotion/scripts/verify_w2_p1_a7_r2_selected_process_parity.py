"""Verify the frozen A7-R2 validation-selected checkpoint in fresh collectors."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
COLLECTOR = HERE.parent / "collect_w2_p1_a7_r2_window.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
POLICY = OUT / "checkpoints/model_075.pt"
RAW = OUT / "raw/selected_process_parity"
OFFSETS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 251]
N = 1024
SEED = 20278421


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def largest(count: int, residual: list[float]) -> tuple[list[int], list[float]]:
    raw = torch.tensor([0.6, 0.2, 0.2], dtype=torch.float64) * count + torch.tensor(residual)
    allocation = torch.floor(raw).long()
    for index in torch.argsort(raw - allocation, descending=True)[: count - int(allocation.sum())]:
        allocation[index] += 1
    return allocation.tolist(), (raw - allocation).tolist()


def make_targets(train_mask: torch.Tensor, cursor: int, residual: list[float]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    ids = torch.nonzero(train_mask).flatten()
    allocation, _ = largest(len(ids), residual)
    rear, other, static = torch.split(ids, allocation)
    targets = torch.zeros(N, 3)
    counts = [int(len(rear) * fraction) for fraction in (0.10, 0.15, 0.25)]
    counts.append(len(rear) - sum(counts))
    start = 0
    for speed, count in zip((0.15, 0.20, 0.25, 0.30), counts):
        targets[rear[start : start + count], 0] = -speed
        start += count
    targets[rear, 2] = -0.3
    other_commands = []
    for angle in (0, 45, 90, 135, 225, 270, 315):
        radians = math.radians(angle)
        for yaw in (-0.3, 0.0, 0.3):
            other_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), yaw))
    for index, env_id in enumerate(other.tolist()):
        targets[env_id] = torch.tensor(other_commands[(index + cursor) % len(other_commands)])
    static_commands = []
    for angle in range(0, 360, 22):
        radians = math.radians(angle)
        static_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), 0.0))
    static_commands += [(0.0, 0.0, -0.3), (0.0, 0.0, 0.3), (0.6, 0.0, 0.0), (1.2, 0.0, 0.0)]
    for angle in range(0, 360, 45):
        radians = math.radians(angle)
        for yaw in (-0.3, 0.0, 0.3):
            static_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), yaw))
    for index, env_id in enumerate(static.tolist()):
        targets[env_id] = torch.tensor(static_commands[(index + cursor) % len(static_commands)])
    mirror = targets.clone()
    mirror[:, 1] *= -1
    mirror[:, 2] *= -1
    return targets, mirror, allocation


def collect(targets: Path, destination: Path, batch: int, offset: int, seed: int) -> None:
    command = [
        str(ISAAC), "-p", str(COLLECTOR), "--policy", str(POLICY),
        "--targets", str(targets), "--output", str(destination),
        "--batch", str(batch), "--offset", str(offset), "--noise-seed", str(seed),
        "--headless", "--device", "cuda:0",
    ]
    with destination.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(POLICY, map_location="cpu", weights_only=False)
    runtime = checkpoint["a7_r2_runtime_state"]
    cursor = int(runtime["collection_cursor"])
    batch = cursor % 5
    offset = OFFSETS[cursor % 12]
    masks = json.loads((M0 / "a7_environment_masks.json").read_text(encoding="utf-8"))["batches"]
    train_mask = torch.tensor(masks[str(batch)]["train_mask"], dtype=torch.bool)
    negative, positive, allocation = make_targets(train_mask, cursor, runtime["quota_residual"])
    torch.save(negative, RAW / "targets_negative.pt")
    torch.save(positive, RAW / "targets_positive.pt")
    runs = []
    for run in range(2):
        pair = []
        for name in ("negative", "positive"):
            destination = RAW / f"run_{run}_{name}.pt"
            if not destination.exists():
                collect(RAW / f"targets_{name}.pt", destination, batch, offset, SEED + cursor * 2)
            pair.append(destination)
        runs.append(pair)
    keys = ("observation", "action", "old_logp", "old_value", "reward", "done", "valid", "last_value", "state_id", "train_mask")
    comparisons = {}
    all_equal = True
    for pass_index, name in enumerate(("negative", "positive")):
        first = torch.load(runs[0][pass_index], map_location="cpu", weights_only=False)
        second = torch.load(runs[1][pass_index], map_location="cpu", weights_only=False)
        item = {}
        for key in keys:
            equal = torch.equal(first[key], second[key])
            item[key] = {"equal": equal, "hash": tensor_hash(first[key])}
            all_equal &= equal
        item["inventory_hash_equal"] = first["inventory_schema_hash_before_policy_load"] == second["inventory_schema_hash_before_policy_load"]
        item["capture_hash_equal"] = first["capture_schema_hash_before_policy_load"] == second["capture_schema_hash_before_policy_load"]
        item["valid_samples"] = int(first["valid"].sum())
        all_equal &= item["inventory_hash_equal"] and item["capture_hash_equal"]
        comparisons[name] = item
    result = {
        "status": "PASS" if all_equal else "FAIL",
        "selected_update": 75,
        "checkpoint_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        "next_collection_cursor": cursor,
        "next_source_batch": batch,
        "next_offset": offset,
        "command_allocation": allocation,
        "policy_hash_equal_between_fresh_runs": all_equal,
        "fresh_process_runs": 2,
        "comparisons": comparisons,
    }
    (OUT / "selected_checkpoint_process_parity.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not all_equal:
        raise SystemExit("selected checkpoint process parity failed")


if __name__ == "__main__":
    main()
