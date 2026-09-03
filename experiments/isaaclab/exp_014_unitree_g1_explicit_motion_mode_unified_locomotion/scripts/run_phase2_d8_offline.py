"""Read-only D8 phase-error, action-relevance, and separability audit."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
RAW7 = D7 / "raw"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8_phase_error_causal_relevance"
RAW = OUT / "raw"
DATA = RAW7 / "dataset"
PHASES = (
    "PRE_STOP_WALK", "STOP_REQUEST_B0", "W_MOVE_BRAKING_PREFIX",
    "W_MOVE_TO_STAGE2Q_BOUNDARY", "STAGE2Q_DECELERATION",
    "STOP_ACQUISITION", "POST_ACQUISITION_CONFIRMATION",
)
GROUPS = {
    "legs": [0, 1, 3, 4, 7, 8, 11, 12, 15, 16, 19, 20],
    "waist": [2], "torso_arms": list(range(5, 23)), "hands": list(range(23, 37)),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s1mod = load_module("d7s1", HERE.parent / "run_phase2_d7_s1_bc.py")
bc = s1mod.bc
d3 = bc.d3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, payload) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scores(y: torch.Tensor, pred: torch.Tensor, classes: int = 7) -> dict:
    cm = torch.zeros(classes, classes, dtype=torch.long)
    for true, guess in zip(y.tolist(), pred.tolist()):
        cm[true, guess] += 1
    per = []
    for i in range(classes):
        tp = int(cm[i, i]); support = int(cm[i].sum()); predicted = int(cm[:, i].sum())
        recall = tp / support if support else 0.0
        precision = tp / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per.append({"class_id": i, "class": PHASES[i] if classes == 7 else str(i), "support": support,
                    "predicted": predicted, "precision": precision, "recall": recall, "f1": f1})
    return {"accuracy": float((y == pred).float().mean()), "macro_f1": sum(x["f1"] for x in per) / classes,
            "balanced_accuracy": sum(x["recall"] for x in per) / classes, "per_class": per,
            "confusion": cm.tolist()}


def train_probe(train_features, train_y, val_features, val_y, device, seed, steps=1800) -> tuple[nn.Module, dict, dict]:
    torch.manual_seed(seed)
    dim = train_features.shape[1]
    head = nn.Linear(dim, 7).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=2e-3)
    pools = [torch.where(train_y == i)[0] for i in range(7)]
    for _ in range(steps):
        idx = torch.cat([pool[torch.randint(len(pool), (256,))] for pool in pools])
        loss = F.cross_entropy(head(train_features[idx].to(device)), train_y[idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    def predict(features):
        chunks = []
        with torch.inference_mode():
            for lo in range(0, len(features), 8192):
                chunks.append(head(features[lo:lo + 8192].to(device)).argmax(1).cpu())
        return torch.cat(chunks)
    return head, scores(train_y, predict(train_features)), scores(val_y, predict(val_features))


def d7_classifier(train, val, device):
    # Exact S1 D7 implementation: an independent raw-input diagnostic MLP.
    torch.manual_seed(20279107)
    head = nn.Sequential(nn.Linear(141, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 7)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    pools = [torch.where(train["context_id"] == i)[0] for i in range(7)]
    for _ in range(3000):
        idx = torch.cat([pool[torch.randint(len(pool), (256,))] for pool in pools])
        loss = F.cross_entropy(head(train["observation_141"][idx].to(device)), train["context_id"][idx].to(device))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    def predict(x):
        out = []
        with torch.inference_mode():
            for lo in range(0, len(x), 8192): out.append(head(x[lo:lo + 8192].to(device)).argmax(1).cpu())
        return torch.cat(out)
    return head, predict(train["observation_141"]), predict(val["observation_141"])


def s1_features(model, x, device):
    result = {"raw_input": [], "hidden_layer_1": [], "hidden_layer_2": [], "hidden_layer_3": [], "action_output": []}
    with torch.inference_mode():
        for lo in range(0, len(x), 8192):
            z = x[lo:lo + 8192].to(device); h1 = F.elu(model.mlp[0](z)); h2 = F.elu(model.mlp[2](h1)); h3 = F.elu(model.mlp[4](h2)); action = model.mlp[6](h3)
            for key, value in (("raw_input", z), ("hidden_layer_1", h1), ("hidden_layer_2", h2), ("hidden_layer_3", h3), ("action_output", action)):
                result[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in result.items()}


def parent_actors(device):
    move = bc.initialize(device)
    stop = d3.initialize("P1_STOP_PARENT", device)[0].eval()
    hold = d3.initialize("P0_STAND_PARENT", device)[0].eval()
    return move.eval(), stop, hold


def teacher_action(actors, phase: torch.Tensor, obs: torch.Tensor, device) -> torch.Tensor:
    out = torch.empty(len(obs), 37)
    role = torch.where(phase <= 3, 0, torch.where(phase <= 5, 1, 2))
    with torch.inference_mode():
        for rid, actor in enumerate(actors):
            idx = torch.where(role == rid)[0]
            if len(idx): out[idx] = actor.mean(obs[idx].to(device)).cpu()
    return out


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train = torch.load(DATA / "train.pt", map_location="cpu", weights_only=False)
    val = torch.load(DATA / "validation.pt", map_location="cpu", weights_only=False)
    s1_results = json.loads((RAW7 / "s1_bc_results.json").read_text(encoding="utf-8"))
    candidates = [x for x in s1_results["timeline"] if x["contexts"]["3"]["mse"] <= .001 and
                  x["contexts"]["4"]["mse"] <= .001 and x["contexts"]["5"]["mse"] <= .001 and
                  x["worst_condition_mse"] <= .001]
    best = min(candidates, key=lambda x: (x["mse"], x["step"]))
    checkpoint = RAW7 / "bc_checkpoints" / f"s1_step_{best['step']:05d}.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = s1mod.S1().to(device).eval(); model.load_state_dict(payload["actor_state_dict"])
    candidate = {"name": "S1_DIAGNOSTIC_CANDIDATE", "formal_selected_checkpoint": False,
                 "checkpoint": checkpoint.relative_to(REPO).as_posix(), "sha256": sha256(checkpoint),
                 "training_step": best["step"], "validation_metrics": best, "selection_partition": "validation only",
                 "heldout_used": False}
    dump("s1_diagnostic_candidate.json", candidate)

    phase_head, train_pred, val_pred = d7_classifier(train, val, device)
    train_score = scores(train["context_id"], train_pred); val_score = scores(val["context_id"], val_pred)
    audit = {"implementation": "separate diagnostic model", "input": "raw 141D observation",
             "not_student_hidden_feature": True, "not_student_action": True, "not_auxiliary_head": True,
             "architecture": [141, 256, 128, 7], "activation": "ELU", "training_split": "D7 train only",
             "validation_split": "D7 validation only", "steps": 3000, "optimizer": "Adam", "learning_rate": .001,
             "loss": "unweighted cross entropy on class-balanced minibatches", "class_weighting": "256 samples/class/batch",
             "normalization": "none", "checkpoint_selection": "none; trained once and copied to every S1 checkpoint metric",
             "train_metrics": train_score, "validation_metrics": val_score,
             "implementation_bug": "independent classifier accuracy was incorrectly treated as a property of every S1 actor checkpoint"}
    dump("phase_classifier_implementation_audit.json", audit)
    cm_rows = []
    for i, row in enumerate(val_score["confusion"]):
        for j, count in enumerate(row): cm_rows.append({"true_phase": PHASES[i], "predicted_phase": PHASES[j], "count": count})
    write_csv("phase_confusion_matrix.csv", cm_rows, ["true_phase", "predicted_phase", "count"])

    obs = val["observation_141"]
    with torch.inference_mode():
        student = torch.cat([model.mean(obs[lo:lo + 8192].to(device)).cpu() for lo in range(0, len(obs), 8192)])
    actors = parent_actors(device)
    true_action = val["action_37"]
    alt_action = teacher_action(actors, val_pred, obs, device)
    student_l2 = (student - true_action).norm(dim=1); student_cos = F.cosine_similarity(student, true_action)
    student_mse = (student - true_action).square().mean(dim=1); student_max = (student - true_action).abs().amax(dim=1)
    teacher_l2 = (alt_action - true_action).norm(dim=1); teacher_cos = F.cosine_similarity(alt_action, true_action)
    wrong = val_pred != val["context_id"]
    phase_material = (teacher_l2 >= .5) | (teacher_cos <= .98)
    student_material = (student_l2 >= .5) | (student_cos <= .98)
    action_safe = wrong & (student_mse <= .001) & (teacher_l2 < .5) & (teacher_cos > .98)
    action_critical = wrong & phase_material & student_material
    critical_count = int(action_critical.sum()); wrong_count = int(wrong.sum()); total = len(obs)

    error_rows = []
    cluster = defaultdict(lambda: {"count": 0, "action_safe": 0, "action_critical": 0, "student_l2": [], "teacher_l2": [], "steps": [], "conditions": Counter(), "yaw": Counter()})
    indices = torch.where(wrong)[0]
    for idx in indices.tolist():
        true = int(val["context_id"][idx]); pred = int(val_pred[idx]); episode = val["episodes"][int(val["episode_index"][idx])]
        step = int(val["control_step"][idx]); key = f"{PHASES[true]}->{PHASES[pred]}"; direction_sector = int(val["condition_id"][idx]) % 16
        row = {"sample_id": idx, "episode_id": episode["episode_id"], "recipe_id": int(val["recipe_id"][idx]),
               "condition": int(val["condition_id"][idx]), "true_phase": PHASES[true], "predicted_phase": PHASES[pred],
               "control_step": step, "time_since_stop_request": float(val["time_since_stop_request"][idx]),
               "distance_from_switch_steps": abs(step - 25), "direction_sector": direction_sector,
               "yaw": float(val["current_command"][idx, 2]), "speed": float(val["current_command"][idx, :2].norm()),
               "support_foot": "not encoded in 141D", "contact_state": "not encoded in D7 dataset",
               "base_xy_speed": float(obs[idx, :2].norm()), "base_yaw_rate": float(obs[idx, 5].abs()),
               "student_l2": float(student_l2[idx]), "student_cosine": float(student_cos[idx]),
               "student_mse": float(student_mse[idx]), "student_max_abs": float(student_max[idx]),
               "teacher_to_teacher_l2": float(teacher_l2[idx]), "teacher_to_teacher_cosine": float(teacher_cos[idx]),
               "action_safe": bool(action_safe[idx]), "action_critical": bool(action_critical[idx]),
               "observation_141": json.dumps(obs[idx].tolist(), separators=(",", ":")),
               "student_action": json.dumps(student[idx].tolist(), separators=(",", ":")),
               "true_teacher_action": json.dumps(true_action[idx].tolist(), separators=(",", ":"))}
        for name, joints in GROUPS.items(): row[f"student_error_{name}_l2"] = float((student[idx, joints] - true_action[idx, joints]).norm())
        error_rows.append(row); c = cluster[key]; c["count"] += 1; c["action_safe"] += int(action_safe[idx]); c["action_critical"] += int(action_critical[idx]); c["student_l2"].append(float(student_l2[idx])); c["teacher_l2"].append(float(teacher_l2[idx])); c["steps"].append(step); c["conditions"][int(val["condition_id"][idx])] += 1; c["yaw"][round(float(val["current_command"][idx, 2]), 3)] += 1
    write_csv("phase_error_samples.csv", error_rows)
    clusters = {}
    for key, value in cluster.items():
        clusters[key] = {"count": value["count"], "action_safe": value["action_safe"], "action_critical": value["action_critical"],
                         "student_l2_mean": sum(value["student_l2"]) / value["count"], "teacher_l2_mean": sum(value["teacher_l2"]) / value["count"],
                         "control_step_min": min(value["steps"]), "control_step_max": max(value["steps"]),
                         "conditions": dict(value["conditions"]), "yaw_values": dict(value["yaw"])}
    dump("phase_error_clusters.json", {"misclassified_samples": wrong_count, "clusters": clusters})

    temporal = {}
    distance = (val["control_step"] - 25).abs()
    for radius in (1, 2, 4, 8):
        region = distance <= radius; count = int((wrong & region).sum())
        temporal[f"switch_plus_minus_{radius}"] = {"samples": int(region.sum()), "errors": count,
                                                    "error_rate": count / max(1, int(region.sum())),
                                                    "fraction_of_all_errors": count / max(1, wrong_count)}
    far = distance > 8; count = int((wrong & far).sum())
    temporal["more_than_8_steps_from_switch"] = {"samples": int(far.sum()), "errors": count,
                                                   "error_rate": count / max(1, int(far.sum())),
                                                   "fraction_of_all_errors": count / max(1, wrong_count)}
    temporal["errors_within_plus_minus_4_over_all_errors"] = temporal["switch_plus_minus_4"]["fraction_of_all_errors"]
    dump("phase_temporal_concentration.json", temporal)

    relevance_rows = []
    for idx in indices.tolist():
        relevance_rows.append({key: row[key] for key in ("sample_id", "true_phase", "predicted_phase", "student_l2", "student_cosine", "student_mse", "student_max_abs", "teacher_to_teacher_l2", "teacher_to_teacher_cosine", "action_safe", "action_critical") for row in [error_rows[len(relevance_rows)]]})
    write_csv("phase_action_relevance.csv", relevance_rows)
    action_relevant_accuracy = 1 - critical_count / total
    action_summary = {"validation_samples": total, "phase_errors": wrong_count, "phase_error_rate": wrong_count / total,
                      "action_safe_errors": int(action_safe.sum()), "action_safe_misclassification_rate": float(action_safe.sum()) / max(1, wrong_count),
                      "action_critical_errors": critical_count, "action_critical_misclassification_rate": critical_count / max(1, wrong_count),
                      "action_critical_rate_all_validation": critical_count / total, "action_critical_accuracy": action_relevant_accuracy,
                      "ACTION_RELEVANT_PHASE_ACCURACY": action_relevant_accuracy,
                      "student_material_definition": "L2 >=0.5 or cosine <=0.98",
                      "teacher_material_definition": "L2 >=0.5 or cosine <=0.98"}
    dump("phase_action_relevance.json", action_summary)

    # Raw-input classifiers and frozen S1 feature linear probes.
    raw_linear, raw_train, raw_val = train_probe(train["observation_141"], train["context_id"], obs, val["context_id"], device, 20279201)
    centroids = torch.stack([train["observation_141"][train["context_id"] == i].mean(0) for i in range(7)])
    scale = train["observation_141"].std(0).clamp_min(1e-4)
    def centroid_predict(x):
        chunks = []
        for lo in range(0, len(x), 4096): chunks.append((((x[lo:lo + 4096, None] - centroids[None]) / scale).square().mean(2)).argmin(1))
        return torch.cat(chunks)
    centroid_train = scores(train["context_id"], centroid_predict(train["observation_141"])); centroid_val = scores(val["context_id"], centroid_predict(obs))
    raw_sep = {"linear_multinomial": {"train": raw_train, "validation": raw_val},
               "three_layer_mlp_D7_exact": {"train": train_score, "validation": val_score},
               "nearest_centroid": {"train": centroid_train, "validation": centroid_val},
               "best_validation_accuracy": max(raw_val["accuracy"], val_score["accuracy"], centroid_val["accuracy"]),
               "interpretation_thresholds": {"identifiable": .995, "ambiguous": .99}}
    dump("raw_input_phase_separability.json", raw_sep)

    # Use a deterministic balanced training subset for memory-bounded frozen probes.
    generator = torch.Generator().manual_seed(20279202)
    subset = torch.cat([pool[torch.randperm(len(pool), generator=generator)[:min(12000, len(pool))]] for pool in [torch.where(train["context_id"] == i)[0] for i in range(7)]])
    train_features = s1_features(model, train["observation_141"][subset], device); val_features = s1_features(model, obs, device)
    probes = {}
    for offset, key in enumerate(train_features):
        _, tr, va = train_probe(train_features[key], train["context_id"][subset], val_features[key], val["context_id"], device, 20279210 + offset, steps=1500)
        probes[key] = {"train": tr, "validation": va, "frozen_feature": True, "runtime_head": False}
    dump("hidden_feature_phase_probes.json", probes)

    merges = [((0, 1), "PRE_STOP_WALK+STOP_REQUEST_B0"), ((2, 3), "W_MOVE_BRAKING_PREFIX+W_MOVE_TO_STAGE2Q_BOUNDARY"), ((5, 6), "STOP_ACQUISITION+POST_ACQUISITION_CONFIRMATION")]
    merge_results = {}
    for pair, name in merges:
        mapping = torch.arange(7); mapping[pair[1]] = pair[0]
        merged_true = mapping[val["context_id"]]; merged_pred = mapping[val_pred]
        idx = torch.where((val["context_id"] == pair[0]) | (val["context_id"] == pair[1]))[0]
        actions = true_action[idx]; variance = float(((actions - actions.mean(0)) ** 2).mean())
        material = int(((teacher_l2[idx] >= .5) | (teacher_cos[idx] <= .98)).sum())
        merge_results[name] = {"merged_class_accuracy": float((merged_true == merged_pred).float().mean()),
                               "within_merged_class_action_variance": variance, "samples_in_merged_class": len(idx),
                               "material_alternative_teacher_action_samples": material}
    dump("phase_merge_counterfactual.json", merge_results)

    dump("action_relevant_phase_metric.json", {**action_summary, "PHYSICAL_PHASE_SAFETY": None,
          "physical_metric_pending": True, "diagnostic_candidate_thresholds": {"ACTION_RELEVANT_PHASE_ACCURACY": .99, "PHYSICAL_PHASE_SAFETY": .99},
          "not_a_D7_gate_replacement": True})
    dump("phase_label_semantics.json", {
        "labels": {
            "PRE_STOP_WALK": {"start": "captured W_MOVE-acquired state before STOP request", "end": "STOP request", "time_range_s": [-.02, -.02], "teacher": "T_MOVE", "type": ["time-defined"], "command": "steady WALK command", "physical": "W_MOVE acquired"},
            "STOP_REQUEST_B0": {"start": "STOP request step 0", "end": "after one sample", "time_range_s": [0, 0], "teacher": "T_MOVE", "type": ["time-defined"], "command": "minimum-jerk ramp begins; target STAND", "physical": "unconstrained"},
            "W_MOVE_BRAKING_PREFIX": {"start": "step 1", "end": "step 20", "time_range_s": [.02, .40], "teacher": "T_MOVE", "type": ["time-defined", "Teacher-route-defined"], "command": "minimum-jerk toward zero", "physical": "unconstrained"},
            "W_MOVE_TO_STAGE2Q_BOUNDARY": {"start": "step 21", "end": "step 29", "time_range_s": [.42, .58], "teacher": "T_MOVE through 24; T_STOP from 25", "type": ["time-defined", "Teacher-route-defined"], "command": "ramp ends at step 25", "physical": "unconstrained"},
            "STAGE2Q_DECELERATION": {"start": "step 30", "end": "25 steps before acquisition confirmation", "time_range_s": [0.60, None], "teacher": "T_STOP", "type": ["Teacher-route-defined", "physical-event-defined end"], "command": "zero", "physical": "not yet in final acquisition window"},
            "STOP_ACQUISITION": {"start": "last 25 T_STOP steps", "end": "acquisition confirmation", "time_range_s": [None, None], "teacher": "T_STOP", "type": ["physical-event-defined"], "command": "zero", "physical": "25-step threshold confirmation window"},
            "POST_ACQUISITION_CONFIRMATION": {"start": "confirmation complete", "end": "25 post-confirmation steps", "time_range_s": [None, None], "teacher": "T_HOLD", "type": ["physical-event-defined", "Teacher-route-defined"], "command": "zero", "physical": "acquired basin"}},
        "overlap_finding": "W_MOVE_TO_STAGE2Q_BOUNDARY intentionally spans two Teacher roles; several labels subdivide the same unconstrained physical braking meaning by clock/route rather than capability.",
        "teacher_phase_is_actor_input": False})
    print(json.dumps({"candidate": candidate, "classifier": val_score, "action": action_summary,
                      "raw_best": raw_sep["best_validation_accuracy"],
                      "hidden": {k: v["validation"]["accuracy"] for k, v in probes.items()}}, indent=2))


if __name__ == "__main__":
    main()
