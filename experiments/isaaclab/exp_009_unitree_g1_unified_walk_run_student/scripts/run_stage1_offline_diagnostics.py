"""Stage 1 fixed-dataset interference diagnostics (no RL, no reward optimization)."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
CFG = yaml.safe_load((EXP / "configs/stage1_interference_diagnosis.yaml").read_text(encoding="utf-8"))
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.dataset import action_columns, observation_columns
from unified_walk_run.stage1_models import DiagnosticMultiHead, DiagnosticSingleHead

REGIMES = ["walk_steady", "run_steady", "walk_to_run"]
REGIME_ID = {name: index for index, name in enumerate(REGIMES)}
OBS, ACT = observation_columns(), action_columns()


def write_json(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fixed_sample() -> pd.DataFrame:
    columns = OBS + ACT + ["regime", "teacher", "split", "episode_id", "sequence_step", "source_speed_mps", "target_speed_mps", "support_phase"]
    pieces = []
    limit = int(CFG["dataset"]["diagnostic_max_rows_per_regime_split"])
    for part in sorted((STAGE0 / "teacher_dataset.parquet").glob("*.parquet")):
        frame = pd.read_parquet(part, columns=columns)
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True)
    sampled = []
    for (regime, split), group in frame.groupby(["regime", "split"], sort=True):
        if len(group) > limit:
            group = group.sample(limit, random_state=CFG["experiment"]["seed"] + REGIME_ID[regime])
        sampled.append(group)
    result = pd.concat(sampled, ignore_index=True)
    del frame, pieces
    return result


def context(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "A_123D":
        return np.empty((len(frame), 0), np.float32)
    maximum = frame.groupby("episode_id")["sequence_step"].transform("max").to_numpy(np.float32)
    elapsed = frame["sequence_step"].to_numpy(np.float32) * 0.02
    progress = np.divide(frame["sequence_step"].to_numpy(np.float32), np.maximum(maximum, 1))
    remaining = np.maximum(0.0, maximum * 0.02 - elapsed)
    if kind == "B_progress":
        return progress[:, None]
    if kind == "C_elapsed_remaining":
        return np.stack([elapsed, remaining], axis=1)
    if kind == "E_teacher_identity":
        ids = frame["regime"].map(REGIME_ID).to_numpy()
        return np.eye(3, dtype=np.float32)[ids]
    # Reconstruct the exact semantic slots available from the frozen Stage-0
    # trajectory metadata. Unlogged recovery/attitude slots are explicitly zero.
    result = np.zeros((len(frame), 29), np.float32)
    ids = frame["regime"].map(REGIME_ID).to_numpy()
    result[np.arange(len(frame)), ids] = 1.0
    result[:, 3:6] = np.eye(3, dtype=np.float32)[ids]
    result[:, 6] = frame["source_speed_mps"].to_numpy(np.float32)
    result[:, 7] = frame["target_speed_mps"].to_numpy(np.float32)
    result[:, 8] = elapsed
    result[:, 9] = remaining
    result[:, 10] = progress
    result[:, 11] = frame["support_phase"].to_numpy(np.float32)
    return result


def batches(x, y, regime, batch_size, seed):
    order = np.random.default_rng(seed).permutation(len(x))
    for begin in range(0, len(order), batch_size):
        index = order[begin : begin + batch_size]
        yield x[index], y[index], regime[index]


def huber(prediction, target, delta=0.1):
    return torch.nn.functional.huber_loss(prediction, target, delta=delta)


def evaluate(model, frame, device, ctx_kind="A_123D", multihead=False):
    x = np.concatenate([frame[OBS].to_numpy(np.float32), context(frame, ctx_kind)], axis=1)
    y = frame[ACT].to_numpy(np.float32)
    r = frame["regime"].map(REGIME_ID).to_numpy(np.int64)
    sums, counts = defaultdict(float), defaultdict(int)
    joint_sq = np.zeros(37, np.float64)
    joint_abs = np.zeros(37, np.float64)
    model.eval()
    with torch.no_grad():
        for xb, yb, rb in batches(x, y, r, 16384, 0):
            xt, yt, rt = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device), torch.from_numpy(rb).to(device)
            pred = model(xt, rt) if multihead else model(xt)
            err = (pred - yt).cpu().numpy()
            for rid, name in enumerate(REGIMES):
                mask = rb == rid
                if mask.any():
                    sums[name] += float(np.abs(err[mask]).mean()) * int(mask.sum())
                    counts[name] += int(mask.sum())
            joint_sq += np.square(err).sum(0)
            joint_abs += np.abs(err).sum(0)
    return {
        "mae": float(joint_abs.sum() / (len(y) * 37)),
        "rmse": float(math.sqrt(joint_sq.sum() / (len(y) * 37))),
        "teacher_mae": {name: sums[name] / max(counts[name], 1) for name in REGIMES},
        "joint_mae": (joint_abs / len(y)).tolist(),
        "joint_rmse": np.sqrt(joint_sq / len(y)).tolist(),
    }


def train_model(name, train, validation, input_dim, hidden, device, ctx_kind="A_123D", only=None, multihead=False, epochs=None, inherit_walk=False):
    if only is not None:
        train = train[train["regime"].eq(only)]
        validation = validation[validation["regime"].eq(only)]
    model = DiagnosticMultiHead().to(device) if multihead else DiagnosticSingleHead(input_dim, hidden).to(device)
    torch.manual_seed(CFG["experiment"]["seed"])
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(module.bias, -bound, bound)
    # Stage 0's fixed initialization rule is WALK-teacher initialization. The
    # 123/256/128/128 diagnostic controls must preserve that exact ancestry.
    if inherit_walk and not multihead and input_dim == 123 and hidden == [256, 128, 128]:
        initial = torch.load(STAGE0 / "checkpoints/initial.pt", map_location=device, weights_only=False)["student"]
        model.network.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
    x = np.concatenate([train[OBS].to_numpy(np.float32), context(train, ctx_kind)], axis=1)
    y = train[ACT].to_numpy(np.float32)
    r = train["regime"].map(REGIME_ID).to_numpy(np.int64)
    curves = []
    gradient_rows = []
    epochs = epochs or int(CFG["training"]["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = count = 0
        for xb, yb, rb in batches(x, y, r, 8192, CFG["experiment"]["seed"] + epoch):
            xt, yt, rt = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device), torch.from_numpy(rb).to(device)
            pred = model(xt, rt) if multihead else model(xt)
            loss = huber(pred, yt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(xb)
            count += len(xb)
        metrics = evaluate(model, validation, device, ctx_kind, multihead)
        curves.append({"epoch": epoch, "train_loss": loss_sum / count, "validation_mae": metrics["mae"], **{f"{key}_mae": value for key, value in metrics["teacher_mae"].items()}})
        if not multihead and ctx_kind == "A_123D" and only is None and epoch in (1, 5, 10):
            gradient_rows.extend(gradient_cosines(model, train, device, epoch))
        print(f"[stage1] {name} epoch={epoch} train={loss_sum/count:.7f} val_mae={metrics['mae']:.7f}", flush=True)
    path = OUT / "checkpoints" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "name": name, "input_dim": input_dim, "hidden": hidden, "multihead": multihead}, path)
    return model, {"name": name, "checkpoint": str(path.relative_to(REPO)), "sha256": sha(path), "parameters": sum(p.numel() for p in model.parameters()), "curves": curves, "validation": evaluate(model, validation, device, ctx_kind, multihead)}, gradient_rows


def gradient_vector(model, frame, device):
    sample = frame.sample(min(8192, len(frame)), random_state=7)
    x = torch.from_numpy(sample[OBS].to_numpy(np.float32)).to(device)
    y = torch.from_numpy(sample[ACT].to_numpy(np.float32)).to(device)
    model.zero_grad(set_to_none=True)
    huber(model(x), y).backward()
    return {name: parameter.grad.detach().flatten().cpu() for name, parameter in model.named_parameters() if parameter.grad is not None}


def gradient_cosines(model, frame, device, epoch):
    vectors = {name: gradient_vector(model, frame[frame["regime"].eq(name)], device) for name in REGIMES}
    rows = []
    for left, right in (("walk_steady", "run_steady"), ("walk_steady", "walk_to_run"), ("run_steady", "walk_to_run")):
        for layer in ["all", *vectors[left]]:
            a = torch.cat(list(vectors[left].values())) if layer == "all" else vectors[left][layer]
            b = torch.cat(list(vectors[right].values())) if layer == "all" else vectors[right][layer]
            rows.append({"epoch": epoch, "pair": f"{left}__{right}", "layer": layer, "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))})
    return rows


def conflict_audit(frame):
    audit_frame = frame[frame["split"].eq("train")].copy()
    per = int(CFG["dataset"]["nearest_neighbor_rows_per_regime"])
    sample = pd.concat([g.sample(min(per, len(g)), random_state=100 + REGIME_ID[n]) for n, g in audit_frame.groupby("regime")])
    x, y = sample[OBS].to_numpy(np.float32), sample[ACT].to_numpy(np.float32)
    regimes = sample["regime"].to_numpy()
    exact_key = pd.util.hash_pandas_object(sample[OBS], index=False)
    exact_groups = pd.DataFrame({"key": exact_key, "regime": regimes}).groupby("key")["regime"].nunique()
    quantized = {}
    for precision in (1e-5, 1e-4, 1e-3, 1e-2):
        q = np.round(x / precision).astype(np.int64)
        keys = pd.util.hash_pandas_object(pd.DataFrame(q), index=False).to_numpy()
        bins = defaultdict(list)
        for index, key in enumerate(keys):
            bins[int(key)].append(index)
        cross, variance, examples = 0, [], []
        for indices in bins.values():
            unique = set(regimes[indices])
            if len(unique) > 1:
                cross += 1
                variance.append(float(np.mean(np.var(y[indices], axis=0))))
                if len(examples) < 20:
                    examples.append({"teachers": sorted(unique), "count": len(indices), "action_l2_max": float(np.linalg.norm(y[indices] - y[indices][0], axis=1).max())})
        quantized[str(precision)] = {"cross_regime_bins": cross, "mean_action_variance": float(np.mean(variance)) if variance else 0.0, "examples": examples}
    nn_rows = []
    for left, right in (("walk_steady", "run_steady"), ("walk_steady", "walk_to_run"), ("run_steady", "walk_to_run")):
        li, ri = np.where(regimes == left)[0], np.where(regimes == right)[0]
        # Exact 123-D nearest neighbours in bounded chunks; deterministic
        # subsampling keeps this O(N^2) diagnostic finite without changing split.
        li, ri = li[:4000], ri[:12000]
        right_x = torch.from_numpy(x[ri])
        nearest_distance, nearest_index = [], []
        for begin in range(0, len(li), 256):
            distances = torch.cdist(torch.from_numpy(x[li[begin:begin + 256]]), right_x)
            values, indices = distances.min(1)
            nearest_distance.append(values.numpy()); nearest_index.append(indices.numpy())
        distance = np.concatenate(nearest_distance)
        index = np.concatenate(nearest_index)
        action_distance = np.linalg.norm(y[li] - y[ri[index]], axis=1)
        nn_rows.append({"pair": f"{left}__{right}", "samples": len(li), "observation_distance_p01": float(np.quantile(distance, .01)), "observation_distance_p50": float(np.quantile(distance, .5)), "action_distance_at_nearest_p50": float(np.quantile(action_distance, .5)), "action_distance_at_nearest_p95": float(np.quantile(action_distance, .95)), "small_distance_action_l2": float(action_distance[distance <= np.quantile(distance, .05)].mean())})
    exact = {"sample_rows": len(sample), "exact_duplicate_groups": int((exact_groups > 1).sum()), "exact_cross_regime_duplicate_rows": int(exact_groups[exact_groups > 1].size)}
    return exact, quantized, nn_rows


def identity_probe(frame):
    train, test = frame[frame["split"].eq("train")], frame[frame["split"].eq("test")]
    limit = int(CFG["dataset"]["identity_probe_rows_per_regime_split"])
    train = pd.concat([g.sample(min(limit, len(g)), random_state=22 + REGIME_ID[n]) for n, g in train.groupby("regime")])
    test = pd.concat([g.sample(min(limit, len(g)), random_state=32 + REGIME_ID[n]) for n, g in test.groupby("regime")])
    xtrain, xtest = train[OBS].to_numpy(np.float32), test[OBS].to_numpy(np.float32)
    ytrain, ytest = train["regime"].map(REGIME_ID), test["regime"].map(REGIME_ID)
    device = torch.device(CFG["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    results = {}
    for name, model in {
        "linear": torch.nn.Linear(123, 3),
        "small_mlp": torch.nn.Sequential(torch.nn.Linear(123, 128), torch.nn.ELU(), torch.nn.Linear(128, 64), torch.nn.ELU(), torch.nn.Linear(64, 3)),
    }.items():
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        labels = ytrain.to_numpy(np.int64)
        for epoch in range(20):
            for xb, _, rb in batches(xtrain, np.zeros((len(xtrain), 1), np.float32), labels, 4096, 100 + epoch):
                logits = model(torch.from_numpy(xb).to(device))
                loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(rb).to(device))
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            probability = model(torch.from_numpy(xtest).to(device)).softmax(-1).cpu().numpy()
        prediction = probability.argmax(1)
        truth = ytest.to_numpy()
        confidence = probability.max(1)
        correct = prediction == truth
        calibration = float(np.mean(np.abs(confidence - correct)))
        matrix = np.zeros((3, 3), dtype=int)
        for actual, predicted in zip(truth, prediction):
            matrix[actual, predicted] += 1
        recalls = np.diag(matrix) / np.maximum(matrix.sum(1), 1)
        accuracy = float(correct.mean())
        results[name] = {
            "accuracy": accuracy,
            "balanced_accuracy": float(recalls.mean()),
            "confusion_matrix": matrix.tolist(),
            "log_loss": float(-np.log(np.maximum(probability[np.arange(len(truth)), truth], 1e-9)).mean()),
            "calibration_error_proxy": calibration,
            "speed_accuracy": {str(speed): float((truth[test["target_speed_mps"].eq(speed)] == prediction[test["target_speed_mps"].eq(speed)]).mean()) for speed in sorted(test["target_speed_mps"].unique()) if test["target_speed_mps"].eq(speed).any()},
            "support_phase_accuracy": {str(phase): float((truth[test["support_phase"].eq(phase)] == prediction[test["support_phase"].eq(phase)]).mean()) for phase in sorted(test["support_phase"].unique())},
        }
    return results


def sequential_forgetting(train, validation, device):
    results = []
    for order_index, order in enumerate([
        ["walk_steady", "run_steady", "walk_to_run"],
        ["run_steady", "walk_steady", "walk_to_run"],
        ["walk_to_run", "walk_steady", "run_steady"],
    ]):
        model = DiagnosticSingleHead(123, [256, 128, 128]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-5)
        phases = []
        for phase_index, regime in enumerate(order):
            sub = train[train["regime"].eq(regime)]
            x, y = sub[OBS].to_numpy(np.float32), sub[ACT].to_numpy(np.float32)
            r = np.zeros(len(x), np.int64)
            for epoch in range(3):
                for xb, yb, _ in batches(x, y, r, 8192, 900 + order_index * 30 + phase_index * 3 + epoch):
                    xt, yt = torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device)
                    loss = huber(model(xt), yt)
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            phases.append({"trained": regime, "evaluation": evaluate(model, validation, device)})
        results.append({"order": order, "phases": phases})
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(CFG["experiment"]["seed"]); np.random.seed(CFG["experiment"]["seed"]); torch.manual_seed(CFG["experiment"]["seed"])
    device = torch.device(CFG["experiment"]["device"] if torch.cuda.is_available() else "cpu")
    frame = load_fixed_sample()
    train, validation, test = frame[frame["split"].eq("train")], frame[frame["split"].eq("validation")], frame[frame["split"].eq("test")]
    stage0_files = sorted((STAGE0 / "teacher_dataset.parquet").glob("*.parquet"))
    write_json("stage0_reference.json", {"stage0_classification": "DISTILLATION_FAIL_INTERFERENCE", "selected_checkpoint": "epoch_10.pt", "selected_sha256": sha(STAGE0 / "checkpoints/epoch_10.pt"), "dataset_rows": 1880660, "dataset_parts": len(stage0_files), "split_preserved": True, "stage0_files_modified": False})
    write_json("protocol.json", CFG)
    exact, quantized, nearest = conflict_audit(frame)
    write_json("label_conflict_audit.json", exact)
    write_json("quantized_conflict_results.json", quantized)
    write_json("nearest_neighbor_disagreement.json", nearest)
    write_json("teacher_identity_probe.json", identity_probe(frame))

    models, gradient_rows = [], []
    # Stage 1A separate/single-task controls.
    single = {}
    for regime in REGIMES:
        _, result, _ = train_model(f"single_{regime}", train, validation, 123, [256, 128, 128], device, only=regime, epochs=20, inherit_walk=True)
        single[regime] = result
    write_json("single_task_capacity_results.json", single)

    # Stage 1C exact three-model capacity sweep.
    capacity = {}
    for name, hidden in {"small": [256, 128, 128], "medium": [512, 256, 256], "large": [512, 512, 256]}.items():
        _, result, rows = train_model(f"capacity_{name}", train, validation, 123, hidden, device)
        capacity[name] = result; gradient_rows.extend(rows)
    write_json("capacity_sweep.json", capacity)

    # Stage 1B context upper bounds.
    contexts = {}
    for kind, extra in CFG["context_conditions"].items():
        _, result, _ = train_model(f"context_{kind}", train, validation, 123 + int(extra), [256, 128, 128], device, ctx_kind=kind)
        contexts[kind] = result
    contexts["D_full_29D"]["context_reconstruction_note"] = "Stage-0 persisted source/target speed, sequence time/progress, regime and support phase; unavailable recovery/attitude semantic slots are explicit zero, not inferred."
    write_json("transition_context_ablation.json", contexts)

    multi_model, multi, _ = train_model("diagnostic_multihead", train, validation, 123, [256, 128, 128], device, multihead=True)
    write_json("diagnostic_multihead_results.json", multi)
    write_json("separate_network_upper_bound.json", {"models": single, "original_teacher_reference": "teacher closed-loop results retained in Stage 0"})
    write_json("sequential_forgetting.json", sequential_forgetting(train, validation, device))
    with (OUT / "layerwise_gradient_cosines.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "pair", "layer", "cosine"]); writer.writeheader(); writer.writerows(gradient_rows)
    write_json("gradient_interference.json", {
        "rows": len(gradient_rows),
        "overall": [row for row in gradient_rows if row["layer"] == "all"],
        "negative_fraction": float(np.mean([row["cosine"] < 0 for row in gradient_rows])) if gradient_rows else 0.0,
    })

    # Joint-wise errors use the frozen Stage-0 selected student, not a re-trained surrogate.
    from unified_walk_run.student_actor import UnifiedWalkRunStudent123
    selected = UnifiedWalkRunStudent123().to(device)
    payload = torch.load(STAGE0 / "checkpoints/epoch_10.pt", map_location=device, weights_only=False)
    selected.load_state_dict(payload["student"])
    joint_rows = []
    for regime in REGIMES:
        metrics = evaluate(selected, test[test["regime"].eq(regime)], device)
        for joint in range(37):
            joint_rows.append({"regime": regime, "joint_index": joint, "mae": metrics["joint_mae"][joint], "rmse": metrics["joint_rmse"][joint]})
    with (OUT / "jointwise_action_errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(joint_rows[0])); writer.writeheader(); writer.writerows(joint_rows)

    write_json("offline_stage1_summary.json", {
        "fixed_sample_rows": len(frame),
        "sample_counts": frame.groupby(["split", "regime"]).size().to_dict().__str__(),
        "base_dataset_rows": 1880660,
        "dagger_rows": 112500,
        "dagger_analyzed_separately": True,
        "models_trained": 3 + 3 + 5 + 1 + 9,
        "ppo_updates": 0,
        "reward_optimization": 0,
        "production_updates": 0,
    })


if __name__ == "__main__":
    main()
