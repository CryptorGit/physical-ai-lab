"""Train the fixed three-member nonlinear dynamics surrogate ensemble."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision"
CFG_PATH = EXP / "configs/stage3_nonlinear_rollout_supervision.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.nonlinear_surrogate import NonlinearLocomotionDynamicsSurrogate  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classification_weight(labels: torch.Tensor, classes: int, train: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels[train].long(), minlength=classes).float().clamp_min(1)
    weights = counts.sum() / (classes * counts)
    return weights / weights.mean()


@torch.no_grad()
def evaluate(model, tensors, ids, normalization, weights, device, batch_size):
    model.eval()
    totals = {key: 0.0 for key in ("loss", "continuous", "contact", "support", "landing", "gait")}
    count = 0
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        obs = tensors["observation"][batch].to(device=device, dtype=torch.float32)
        action = tensors["action"][batch].to(device=device, dtype=torch.float32)
        target = tensors["physical_delta"][batch].to(device=device, dtype=torch.float32)
        contact = tensors["contact"][batch].to(device=device, dtype=torch.float32)
        support = tensors["support"][batch].to(device=device, dtype=torch.long)
        landing = tensors["landing"][batch].to(device=device, dtype=torch.long)
        gait = tensors["gait"][batch].to(device=device, dtype=torch.long)
        xobs = (obs - normalization["obs_mean"]) / normalization["obs_std"]
        xact = (action - normalization["action_mean"]) / normalization["action_std"]
        normalized_target = (target - normalization["delta_mean"]) / normalization["delta_std"]
        prediction = model(xobs, xact)
        losses = {
            "continuous": F.huber_loss(prediction.physical_residual, normalized_target, delta=1.0),
            "contact": F.binary_cross_entropy_with_logits(prediction.contacts, contact),
            "support": F.cross_entropy(prediction.support_logits, support, weight=weights["support"]),
            "landing": F.cross_entropy(prediction.landing_logits, landing, weight=weights["landing"]),
            "gait": F.cross_entropy(prediction.gait_logits, gait, weight=weights["gait"]),
        }
        loss = losses["continuous"] + losses["contact"] + losses["support"] + 0.25 * losses["landing"] + losses["gait"]
        n = len(batch)
        totals["loss"] += float(loss) * n
        for key, value in losses.items():
            totals[key] += float(value) * n
        count += n
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    scfg = cfg["surrogate"]
    device = torch.device(cfg["experiment"]["device"])
    tensors = torch.load(OUT / "surrogate_pairs.pt", map_location="cpu", weights_only=False)
    split = tensors["split"].long()
    train_ids = torch.nonzero(split == 0).flatten()
    validation_ids = torch.nonzero(split == 1).flatten()
    train_obs = tensors["observation"][train_ids].float()
    train_action = tensors["action"][train_ids].float()
    normalization_cpu = {
        "obs_mean": train_obs.mean(0),
        "obs_std": train_obs.std(0).clamp_min(1e-5),
        "action_mean": train_action.mean(0),
        "action_std": train_action.std(0).clamp_min(1e-5),
        "delta_mean": tensors["physical_delta_mean"].float(),
        "delta_std": tensors["physical_delta_std"].float(),
    }
    normalization = {key: value.to(device) for key, value in normalization_cpu.items()}
    weights_cpu = {
        "support": classification_weight(tensors["support"], 4, split == 0),
        "landing": classification_weight(tensors["landing"], 3, split == 0),
        "gait": classification_weight(tensors["gait"], 3, split == 0),
    }
    weights = {key: value.to(device) for key, value in weights_cpu.items()}
    checkpoint_dir = OUT / "surrogate_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows, ensemble = [], []
    batch_size = int(scfg["batch_size"])
    for member, seed in enumerate(scfg["member_seeds"]):
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        model = NonlinearLocomotionDynamicsSurrogate().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(scfg["learning_rate"]), weight_decay=float(scfg["weight_decay"])
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        best = float("inf")
        best_epoch = 0
        patience = 0
        path = checkpoint_dir / f"member_{member}.pt"
        for epoch in range(1, int(scfg["epochs"]) + 1):
            model.train()
            order = train_ids[torch.randperm(len(train_ids), generator=generator)]
            total = 0.0
            seen = 0
            for start in range(0, len(order), batch_size):
                ids = order[start : start + batch_size]
                obs = tensors["observation"][ids].to(device=device, dtype=torch.float32)
                action = tensors["action"][ids].to(device=device, dtype=torch.float32)
                target = tensors["physical_delta"][ids].to(device=device, dtype=torch.float32)
                contact = tensors["contact"][ids].to(device=device, dtype=torch.float32)
                support = tensors["support"][ids].to(device=device, dtype=torch.long)
                landing = tensors["landing"][ids].to(device=device, dtype=torch.long)
                gait = tensors["gait"][ids].to(device=device, dtype=torch.long)
                prediction = model(
                    (obs - normalization["obs_mean"]) / normalization["obs_std"],
                    (action - normalization["action_mean"]) / normalization["action_std"],
                )
                normalized_target = (target - normalization["delta_mean"]) / normalization["delta_std"]
                continuous = F.huber_loss(prediction.physical_residual, normalized_target, delta=1.0)
                contact_loss = F.binary_cross_entropy_with_logits(prediction.contacts, contact)
                support_loss = F.cross_entropy(prediction.support_logits, support, weight=weights["support"])
                landing_loss = F.cross_entropy(prediction.landing_logits, landing, weight=weights["landing"])
                gait_loss = F.cross_entropy(prediction.gait_logits, gait, weight=weights["gait"])
                loss = continuous + contact_loss + support_loss + 0.25 * landing_loss + gait_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(scfg["gradient_clip_norm"]))
                optimizer.step()
                total += float(loss.detach()) * len(ids)
                seen += len(ids)
            validation = evaluate(model, tensors, validation_ids, normalization, weights, device, batch_size)
            rows.append({"member": member, "seed": seed, "epoch": epoch, "train_loss": total / seen, **{f"validation_{k}": v for k, v in validation.items()}})
            if validation["loss"] < best - 1e-6:
                best, best_epoch, patience = validation["loss"], epoch, 0
                torch.save({
                    "model": model.state_dict(), "member": member, "seed": int(seed), "epoch": epoch,
                    "normalization": normalization_cpu, "class_weights": weights_cpu,
                    "architecture": [160, 512, 512, 256, 96], "dataset_sha256": sha256(OUT / "surrogate_pairs.pt"),
                }, path)
            else:
                patience += 1
            if epoch >= int(scfg["early_stopping_minimum_epoch"]) and patience >= int(scfg["early_stopping_patience_epochs"]):
                break
        ensemble.append({
            "member": member, "seed": int(seed), "best_epoch": best_epoch, "best_validation_loss": best,
            "path": str(path.relative_to(REPO)), "sha256": sha256(path), "parameters": sum(p.numel() for p in model.parameters()),
        })
    with (OUT / "surrogate_training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "surrogate_ensemble_manifest.json").write_text(json.dumps({
        "name": "NonlinearLocomotionDynamicsSurrogate", "members": ensemble,
        "architecture_per_member": [160, 512, 512, 256, 96], "activation": "ELU",
        "architecture_sweep": False, "teacher_gradients": 0, "nan_inf": 0,
    }, indent=2), encoding="utf-8")
    print(json.dumps(ensemble, indent=2))


if __name__ == "__main__":
    main()
