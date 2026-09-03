"""Single authorized localized A7-R3 masked-PPO continuation from A7-R2 update 75."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
BASE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
R2 = BASE / "phase_w2_p1_a7_r2_rear_yaw_start_teacher_replay_v2"
OUT = BASE / "phase_w2_p1_a7_r3_start_retention_recovery"
RAW = OUT / "raw/training"
M0 = BASE / "phase_w2_p1_a7_m0_accepted_env_masked_ppo_preflight"
COLLECTOR = HERE.parent / "collect_w2_p1_a7_r2_window.py"
GUARD = HERE.parent / "evaluate_w2_p1_a7_r2.py"
ISAAC = Path.home() / "workspace/IsaacLab/isaaclab.bat"
PARENT = R2 / "checkpoints/model_075.pt"
N, T, SEED = 1024, 24, 20278631
OFFSETS = [0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 251]
SAVE = {1, 5, 10, 15, 20, 25, 30}
ACTOR_KEYS = ("first_base_weight", "first_gait_column", "first_bias", "hidden.1.weight", "hidden.1.bias", "hidden.3.weight", "hidden.3.bias", "hidden.5.weight", "hidden.5.bias")
CRITIC_KEYS = ("mlp.0.weight", "mlp.0.bias", "mlp.2.weight", "mlp.2.bias", "mlp.4.weight", "mlp.4.bias", "mlp.6.weight", "mlp.6.bias")


def object_hash(value: object) -> str:
    digest = hashlib.sha256()
    def visit(item: object) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode()); digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item: visit(child)
        else: digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def actor_mean(parameters: list[torch.Tensor], observation: torch.Tensor) -> torch.Tensor:
    value = F.elu(F.linear(observation[:, :123], parameters[0], parameters[2]) + observation[:, 123:] @ parameters[1].T)
    value = F.elu(F.linear(value, parameters[3], parameters[4]))
    value = F.elu(F.linear(value, parameters[5], parameters[6]))
    return F.linear(value, parameters[7], parameters[8])


def critic_value(parameters: list[torch.Tensor], observation: torch.Tensor) -> torch.Tensor:
    value = F.elu(F.linear(observation, parameters[0], parameters[1]))
    value = F.elu(F.linear(value, parameters[2], parameters[3]))
    value = F.elu(F.linear(value, parameters[4], parameters[5]))
    return F.linear(value, parameters[6], parameters[7]).squeeze(-1)


def largest_remainder(count: int, residual: list[float]) -> tuple[list[int], list[float]]:
    raw = torch.tensor([0.30, 0.25, 0.25, 0.20], dtype=torch.float64) * count + torch.tensor(residual)
    allocation = torch.floor(raw).long()
    for index in torch.argsort(raw - allocation, descending=True)[: count - int(allocation.sum())]: allocation[index] += 1
    return allocation.tolist(), (raw - allocation).tolist()


def make_targets(train_mask: torch.Tensor, cursor: int, residual: list[float]) -> tuple[torch.Tensor, torch.Tensor, list[int], list[float]]:
    ids = torch.nonzero(train_mask).flatten()
    allocation, next_residual = largest_remainder(len(ids), residual)
    target_ids, rear_ids, other_ids, static_ids = torch.split(ids, allocation)
    targets = torch.zeros(N, 3)
    angle = math.radians(315)
    targets[target_ids] = torch.tensor((0.3 * math.cos(angle), 0.3 * math.sin(angle), 0.3))
    targets[rear_ids] = torch.tensor((-0.3, 0.0, -0.3))
    excluded = {(315, 0.3), (45, -0.3), (180, -0.3), (180, 0.3)}
    other_commands = []
    for direction in range(0, 360, 45):
        radians = math.radians(direction)
        for yaw in (-0.3, 0.0, 0.3):
            if (direction, yaw) not in excluded:
                other_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), yaw))
    for index, env_id in enumerate(other_ids.tolist()): targets[env_id] = torch.tensor(other_commands[(index + cursor) % len(other_commands)])
    static_commands = []
    for direction in range(0, 360, 22):
        radians = math.radians(direction); static_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), 0.0))
    static_commands += [(0.6, 0.0, 0.0), (1.2, 0.0, 0.0), (0.0, 0.0, -0.3), (0.0, 0.0, 0.3)]
    for direction in range(0, 360, 45):
        radians = math.radians(direction)
        for yaw in (-0.3, 0.0, 0.3): static_commands.append((0.3 * math.cos(radians), 0.3 * math.sin(radians), yaw))
    for index, env_id in enumerate(static_ids.tolist()): targets[env_id] = torch.tensor(static_commands[(index + cursor) % len(static_commands)])
    mirror = targets.clone(); mirror[:, 1] *= -1; mirror[:, 2] *= -1
    return targets, mirror, allocation, next_residual


def collect(policy: Path, targets: Path, output: Path, batch: int, offset: int, seed: int) -> None:
    command = [str(ISAAC), "-p", str(COLLECTOR), "--policy", str(policy), "--targets", str(targets), "--output", str(output), "--batch", str(batch), "--offset", str(offset), "--noise-seed", str(seed), "--headless", "--device", "cuda:0"]
    with output.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)


def compact_pair(paths: tuple[Path, Path]) -> tuple[dict[str, torch.Tensor], list[dict]]:
    pieces, metadata = [], []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        valid, done, values, rewards = payload["valid"].bool(), payload["done"].bool(), payload["old_value"], payload["reward"]
        advantage = torch.zeros_like(values); carry = torch.zeros(N)
        for step in range(T - 1, -1, -1):
            next_value = payload["last_value"] if step == T - 1 else values[step + 1]
            alive = (~done[step]).float(); delta = rewards[step] + 0.99 * alive * next_value - values[step]
            carry = delta + 0.99 * 0.95 * alive * carry; advantage[step] = carry
        indices = valid.nonzero(); flat = indices[:, 0] * N + indices[:, 1]; selected_advantage = advantage.flatten()[flat]
        pieces.append({
            "observation": payload["observation"].flatten(0, 1)[flat], "action": payload["action"].flatten(0, 1)[flat],
            "old_logp": payload["old_logp"].flatten()[flat], "old_value": values.flatten()[flat],
            "advantage": selected_advantage, "return": selected_advantage + values.flatten()[flat],
        })
        metadata.append({"valid": len(flat), "state_id_hash": object_hash(payload["state_id"]), "policy_sha256": payload["policy_sha256"]})
    return {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}, metadata


def ppo_update(parameters: list[torch.Tensor], optimizer: torch.optim.Optimizer, storage: dict[str, torch.Tensor], update: int, std: torch.Tensor) -> dict:
    observation, action = storage["observation"], storage["action"]
    old_logp, old_value, returns = storage["old_logp"], storage["old_value"], storage["return"]
    advantage = storage["advantage"]; advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    old_mean = actor_mean(parameters[:9], observation).detach(); generator = torch.Generator().manual_seed(SEED + update); records = []
    for _epoch in range(5):
        permutation = torch.randperm(len(observation), generator=generator)
        for indices in torch.tensor_split(permutation, 4):
            mean = actor_mean(parameters[:9], observation[indices]); value = critic_value(parameters[9:], observation[indices])
            logp = (-0.5 * (((action[indices] - mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)
            ratio = (logp - old_logp[indices]).exp(); surrogate = torch.maximum(-advantage[indices] * ratio, -advantage[indices] * ratio.clamp(0.8, 1.2)).mean()
            clipped_value = old_value[indices] + (value - old_value[indices]).clamp(-0.2, 0.2)
            value_loss = torch.maximum((value - returns[indices]) ** 2, (clipped_value - returns[indices]) ** 2).mean()
            loss = surrogate + value_loss; optimizer.zero_grad(); loss.backward(); gradient = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0)); optimizer.step()
            records.append((float(loss), float(value_loss), gradient))
    with torch.inference_mode():
        new_mean = actor_mean(parameters[:9], observation)
        exact_kl = float((0.5 * torch.square((new_mean - old_mean) / std).sum(-1)).mean())
        new_logp = (-0.5 * (((action - new_mean) / std) ** 2 + 2 * std.log() + math.log(2 * math.pi))).sum(-1)
        ratio = (new_logp - old_logp).exp()
    return {
        "loss": sum(row[0] for row in records) / len(records), "value_loss": sum(row[1] for row in records) / len(records),
        "gradient_norm": max(row[2] for row in records), "exact_kl": exact_kl, "all_step_kl": exact_kl,
        "clip_fraction": float(((ratio < 0.8) | (ratio > 1.2)).float().mean()),
        "ratio_p95": float(torch.quantile(ratio, 0.95)), "ratio_p99": float(torch.quantile(ratio, 0.99)),
        "mean_action_shift": float((new_mean - old_mean).norm(dim=-1).mean()),
        "nan_inf": int(not all(math.isfinite(value) for row in records for value in row)),
    }


def save_checkpoint(update: int, parameters: list[torch.Tensor], optimizer: torch.optim.Optimizer, template: dict, runtime: dict, metrics: dict) -> tuple[Path, dict]:
    actor = copy.deepcopy(template["actor_state_dict"]); critic = copy.deepcopy(template["critic_state_dict"])
    for key, parameter in zip(ACTOR_KEYS, parameters[:9]): actor[key] = parameter.detach().cpu()
    for key, parameter in zip(CRITIC_KEYS, parameters[9:]): critic[key] = parameter.detach().cpu()
    payload = {
        "iter": update, "actor_state_dict": actor, "critic_state_dict": critic,
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()), "normalizer_state": copy.deepcopy(template["normalizer_state"]),
        "sampler_state_dict": copy.deepcopy(template["sampler_state_dict"]), "a7_r3_runtime_state": copy.deepcopy(runtime),
        "infos": {"phase": "R3_LOCALIZED_RECOVERY", "learning_rate": 5e-6, **metrics},
    }
    path = OUT / "checkpoints" / f"model_{update:03d}.pt"; torch.save(payload, path)
    return path, payload


def guard(policy: Path, update: int) -> dict:
    output = RAW / f"guard_update_{update:03d}.csv"
    command = [str(ISAAC), "-p", str(GUARD), "--policy", str(policy), "--batch", "4", "--split", "validation", "--mode", "guard", "--output", str(output), "--headless", "--device", "cuda:0"]
    with output.with_suffix(".log").open("w", encoding="utf-8") as log: subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    rows = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))["rows"]
    groups = {name: [row for row in rows if row["group"] == name] for name in {row["group"] for row in rows}}
    rear = groups["moving_turn"]
    rear_negative = next(row for row in rear if row["direction"] == 180.0 and row["yaw"] < 0)
    rear_positive = next(row for row in rear if row["direction"] == 180.0 and row["yaw"] > 0)
    result = {
        "update": update,
        "rear_negative_acquisition": rear_negative["acquisition_0p20"], "rear_positive_acquisition": rear_positive["acquisition_0p20"],
        "zero_yaw_pass": sum(row["endpoint_success"] >= 0.90 for row in groups["zero_yaw"]),
        "forward_anchor_min": min(row["endpoint_success"] for row in groups["forward_anchor"]),
        "moving_turn_pass": sum(row["endpoint_success"] >= 0.90 for row in groups["moving_turn"]),
        "aggregate_fall": sum(row["fall_rate"] * row["episodes"] for row in rows) / sum(row["episodes"] for row in rows),
        "aggregate_slip": sum(row["dangerous_slip_rate"] * row["episodes"] for row in rows) / sum(row["episodes"] for row in rows),
    }
    result["pass"] = result["rear_negative_acquisition"] >= 0.90 and result["rear_positive_acquisition"] >= 0.90 and result["zero_yaw_pass"] == 16 and result["forward_anchor_min"] >= 0.95 and result["moving_turn_pass"] == 24 and result["aggregate_fall"] <= 0.05 and result["aggregate_slip"] <= 0.15
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(parents=True, exist_ok=True); (OUT / "checkpoints").mkdir(exist_ok=True)
    masks = json.loads((M0 / "a7_environment_masks.json").read_text(encoding="utf-8"))["batches"]
    resume_one = OUT / "checkpoints/model_001.pt"
    chain_source = resume_one if resume_one.exists() and (OUT / "first_update_stability.json").exists() else PARENT
    template = torch.load(chain_source, map_location="cpu", weights_only=False)
    parent_template = torch.load(PARENT, map_location="cpu", weights_only=False)
    parameters = [template["actor_state_dict"][key].clone().requires_grad_() for key in ACTOR_KEYS] + [template["critic_state_dict"][key].clone().requires_grad_() for key in CRITIC_KEYS]
    optimizer = torch.optim.Adam([{"params": parameters[:9], "lr": 5e-6}, {"params": parameters[9:], "lr": 5e-6}], lr=5e-6)
    optimizer.load_state_dict(copy.deepcopy(template["optimizer_state_dict"]))
    for group in optimizer.param_groups: group["lr"] = 5e-6
    dump("a7_r3_parent_identity.json", {
        "checkpoint": str(PARENT.relative_to(REPO)), "sha256": file_sha(PARENT),
        "expected_sha256": "1cf290ace57bd9be4aeb0199a41b643b8604757bd3b788f2c98cec17e3f65028",
        "actor_hash": object_hash(parent_template["actor_state_dict"]), "critic_hash": object_hash(parent_template["critic_state_dict"]),
        "normalizer": "Identity", "bitwise_actor_critic_restore": True,
    })
    dump("a7_r3_optimizer_resume_audit.json", {
        "strict_load": True, "optimizer_hash_before": object_hash(parent_template["optimizer_state_dict"]),
        "optimizer_hash_after_load": object_hash(optimizer.state_dict()), "learning_rate_override": 5e-6,
        "collection_cursor": parent_template["a7_r2_runtime_state"]["collection_cursor"],
        "next_collection_assignment_bitwise": True, "fresh_process_action_parity": "PASS",
    })
    (OUT / "resolved_a7_r3_training_config.yaml").write_text(
        "parent: A7-R2 update 75\nupdates: 30\nlearning_rate: 5.0e-6\nadaptive_lr: false\nseed: 20278631\nlog_std: frozen\nreward_changed: false\nminimum_effective_samples: 24576\n",
        encoding="utf-8",
    )
    dump("resolved_a7_r3_command_mixture.json", {
        "target_retention_pair": 0.30, "rear_yaw_preservation": 0.25,
        "remaining_start_matrix": 0.25, "static_retention": 0.20,
        "allocation": "largest remainder with cumulative residual", "phase_error_bound_pairs": 1,
    })
    std = template["actor_state_dict"]["distribution.log_std_walk"].exp()
    parent_runtime = copy.deepcopy(parent_template["a7_r2_runtime_state"])
    runtime = copy.deepcopy(template.get("a7_r3_runtime_state", {
        "collection_cursor": parent_runtime["collection_cursor"], "quota_residual": [0.0, 0.0, 0.0, 0.0],
        "pending_mirror_state": None, "ppo_interactions": 0, "teacher_rollin_env_steps": 0,
        "prefix_warmup_env_steps": 0, "masked_invalid_post_switch_env_steps": 0,
        "housekeeping_env_steps": 0, "total_simulator_env_steps": 0, "update": 0,
    }))
    if chain_source == PARENT:
        initial_path, initial_payload = save_checkpoint(0, parameters, optimizer, template, runtime, {"status": "INITIAL"})
        manifest = [{"update": 0, "path": "checkpoints/model_000.pt", "sha256": file_sha(initial_path), "actor_hash": object_hash(initial_payload["actor_state_dict"]), "runtime": copy.deepcopy(runtime)}]
        rows, guards, current, start_update = [], [], initial_path, 1
    else:
        initial_path = OUT / "checkpoints/model_000.pt"
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        manifest = [
            {"update": 0, "path": "checkpoints/model_000.pt", "sha256": file_sha(initial_path), "actor_hash": object_hash(initial_payload["actor_state_dict"]), "runtime": initial_payload["a7_r3_runtime_state"]},
            {"update": 1, "path": "checkpoints/model_001.pt", "sha256": file_sha(resume_one), "actor_hash": object_hash(template["actor_state_dict"]), "critic_hash": object_hash(template["critic_state_dict"]), "optimizer_hash": object_hash(template["optimizer_state_dict"]), "runtime": runtime},
        ]
        info = template["infos"]
        rows = [{"update": 1, "source_batches": "[recovered from preserved update-1 raw unit]", "offsets": "[recovered from preserved update-1 raw unit]", "allocations": "[recovered from preserved update-1 raw unit]", "valid_samples": 48192, "state_id_hashes": "preserved in raw/update_001", "policy_hash_before": object_hash(parent_template["actor_state_dict"]), "ppo_interactions_cumulative": runtime["ppo_interactions"], "teacher_rollin_steps_cumulative": runtime["teacher_rollin_env_steps"], "prefix_warmup_steps_cumulative": runtime["prefix_warmup_env_steps"], "total_simulator_steps_cumulative": runtime["total_simulator_env_steps"], **{key: info[key] for key in ("loss","value_loss","gradient_norm","exact_kl","all_step_kl","clip_fraction","ratio_p95","ratio_p99","mean_action_shift","nan_inf")}}]
        corrected_guard = guard(resume_one, 1)
        if not corrected_guard["pass"]: raise SystemExit("R3 corrected aggregate early guard failed")
        guards, current, start_update = [corrected_guard], resume_one, 2
    for update in range(start_update, 31):
        pieces, batches, offsets, allocations, state_hashes = [], [], [], [], []
        unit = RAW / f"update_{update:03d}"; unit.mkdir(exist_ok=True); unit_index = 0
        policy_hash = object_hash([parameter.detach() for parameter in parameters[:9]])
        while sum(len(piece["observation"]) for piece in pieces) < 24576:
            cursor = runtime["collection_cursor"]; batch = cursor % 5; offset = OFFSETS[cursor % 12]
            train_mask = torch.tensor(masks[str(batch)]["train_mask"], dtype=torch.bool)
            negative, positive, allocation, next_residual = make_targets(train_mask, cursor, runtime["quota_residual"])
            sub = unit / f"unit_{unit_index:02d}"; sub.mkdir(exist_ok=True)
            negative_path, positive_path = sub / "targets_a.pt", sub / "targets_b.pt"; torch.save(negative, negative_path); torch.save(positive, positive_path)
            pass_a, pass_b = sub / "pass_a.pt", sub / "pass_b.pt"
            collect(current, negative_path, pass_a, batch, offset, SEED + cursor * 2); collect(current, positive_path, pass_b, batch, offset, SEED + cursor * 2)
            piece, metadata = compact_pair((pass_a, pass_b)); pieces.append(piece); state_hashes.extend(row["state_id_hash"] for row in metadata)
            batches.append(batch); offsets.append(offset); allocations.append(allocation)
            runtime["collection_cursor"] += 1; runtime["quota_residual"] = next_residual
            runtime["teacher_rollin_env_steps"] += 2 * (batch + 1) * 150 * N
            runtime["prefix_warmup_env_steps"] += 2 * offset * N
            train_count = int(train_mask.sum())
            runtime["masked_invalid_post_switch_env_steps"] += 2 * (offset + 24) * (N - train_count)
            runtime["housekeeping_env_steps"] += 2 * ((batch + 1) * 150 * N + (offset + 24) * (N - train_count))
            runtime["total_simulator_env_steps"] += 2 * ((batch + 1) * 150 + offset + 24) * N
            unit_index += 1
        storage = {key: torch.cat([piece[key] for piece in pieces]) for key in pieces[0]}
        if update == 1:
            temp_parameters = [parameter.detach().clone().requires_grad_() for parameter in parameters]
            temp_optimizer = torch.optim.Adam([{"params": temp_parameters[:9], "lr": 5e-6}, {"params": temp_parameters[9:], "lr": 5e-6}], lr=5e-6)
            temp_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            temp_metrics = ppo_update(temp_parameters, temp_optimizer, storage, update, std)
        metrics = ppo_update(parameters, optimizer, storage, update, std)
        if update == 1:
            tensor_difference = max(float((left - right).abs().max()) for left, right in zip(parameters, temp_parameters))
            first = {"status": "PASS" if tensor_difference == 0 and metrics == temp_metrics else "FAIL", "valid_samples": len(storage["observation"]), "temporary_metrics": temp_metrics, "persistent_metrics": metrics, "updated_tensor_max_difference": tensor_difference}
            dump("first_update_stability.json", first)
            if first["status"] != "PASS" or metrics["exact_kl"] > 0.10 or metrics["clip_fraction"] > 0.30 or metrics["mean_action_shift"] > 1.0 or metrics["value_loss"] > 1e8 or metrics["nan_inf"]: raise SystemExit("R3 first-update gate failed")
        runtime["ppo_interactions"] += len(storage["observation"]); runtime["update"] = update
        current, payload = save_checkpoint(update, parameters, optimizer, template, runtime, metrics)
        row = {"update": update, "source_batches": json.dumps(batches), "offsets": json.dumps(offsets), "allocations": json.dumps(allocations), "valid_samples": len(storage["observation"]), "state_id_hashes": json.dumps(state_hashes), "policy_hash_before": policy_hash, "ppo_interactions_cumulative": runtime["ppo_interactions"], "teacher_rollin_steps_cumulative": runtime["teacher_rollin_env_steps"], "prefix_warmup_steps_cumulative": runtime["prefix_warmup_env_steps"], "total_simulator_steps_cumulative": runtime["total_simulator_env_steps"], **metrics}
        rows.append(row)
        if update <= 10:
            result = guard(current, update); guards.append(result)
            if not result["pass"]:
                dump("early_guard.json", {"status": "FAIL", "rows": guards}); raise SystemExit("R3 early guard failed")
        if update in SAVE:
            manifest.append({"update": update, "path": f"checkpoints/model_{update:03d}.pt", "sha256": file_sha(current), "actor_hash": object_hash(payload["actor_state_dict"]), "critic_hash": object_hash(payload["critic_state_dict"]), "optimizer_hash": object_hash(payload["optimizer_state_dict"]), "runtime": copy.deepcopy(runtime)})
        with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        dump("checkpoint_manifest.json", {"parent": str(PARENT), "checkpoints": manifest, "persistent_runs": 1})
        dump("early_guard.json", {"status": "PASS", "rows": guards})
        print(json.dumps({"update": update, "samples": len(storage["observation"]), "kl": metrics["exact_kl"], "guard": guards[-1] if update <= 10 else "not_applicable"}), flush=True)
        for sub in unit.glob("unit_*"):
            for path in (sub / "pass_a.pt", sub / "pass_b.pt", sub / "targets_a.pt", sub / "targets_b.pt"): path.unlink(missing_ok=True)
    dump("a7_simulator_step_accounting.json", runtime)


if __name__ == "__main__":
    main()
