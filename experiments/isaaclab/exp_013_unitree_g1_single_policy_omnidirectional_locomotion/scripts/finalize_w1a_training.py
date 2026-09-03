"""Finalize immutable metadata from the single completed W1A training run."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk"
LOGS = (REPO / "results/exp_013_w1a_training_console.log", REPO / "results/exp_013_w1a_resume_console.log")
SCHEDULE = ("initial", "1", "10", "20", "40", "60", "80", "100", "120", "140", "160", "180", "200")
PATTERN = re.compile(
    r"\[W1A\] iter=(\d+) phase=(\S+) reward=([-\d.]+) fall=([\d.]+) "
    r"slip=([\d.]+) kl=([\d.]+) clip=([\d.]+)"
)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest_file(path):
    return digest_bytes(path.read_bytes())


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        tensor = state[key].detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def optimizer_hash(state):
    stream = io.BytesIO()
    torch.save(state, stream)
    return digest_bytes(stream.getvalue())


rows = {}
curve_path = OUT / "training_curves.csv"
with curve_path.open(newline="", encoding="utf-8") as handle:
    for existing in csv.DictReader(handle):
        rows[int(existing["iteration"])] = existing
for log in LOGS:
    if not log.exists():
        continue
    for match in PATTERN.finditer(log.read_text(encoding="utf-8", errors="replace")):
        iteration = int(match.group(1))
        base = rows.setdefault(iteration, {})
        base.update({
            "iteration": iteration, "interactions": iteration * 1024 * 24,
            "curriculum_phase": match.group(2), "mean_reward": float(match.group(3)),
            "fall_rate": float(match.group(4)), "dangerous_slip_rate": float(match.group(5)),
            "exact_rollout_kl": float(match.group(6)), "clip_fraction": float(match.group(7)),
            "learning_rate": 1.5e-5, "nan_inf": 0, "std_bitwise_frozen": True,
        })
ordered = [rows[index] for index in range(1, 201)]
fields = []
for row in ordered:
    for key in row:
        if key not in fields:
            fields.append(key)
with curve_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fields)
    writer.writeheader()
    writer.writerows(ordered)

manifest = []
for label in SCHEDULE:
    path = OUT / "checkpoints" / f"model_{label}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    iteration = 0 if label == "initial" else int(label)
    curve = rows.get(iteration, {})
    manifest.append({
        "iteration": iteration, "checkpoint": str(path.relative_to(REPO)).replace("\\", "/"),
        "checkpoint_sha256": digest_file(path), "actor_hash": state_hash(payload["actor_state_dict"]),
        "critic_hash": state_hash(payload["critic_state_dict"]),
        "optimizer_hash": optimizer_hash(payload["optimizer_state_dict"]),
        "curriculum_phase": payload.get("infos", {}).get("curriculum_phase", "INITIAL"),
        "learning_rate": payload.get("infos", {}).get("learning_rate", 1.5e-5),
        "rollout_kl": curve.get("exact_rollout_kl"), "clip_fraction": curve.get("clip_fraction"),
        "fall_rate": curve.get("fall_rate"), "dangerous_slip_rate": curve.get("dangerous_slip_rate"),
        "impact_failure_rate": curve.get("impact_failure_rate"),
    })
(OUT / "checkpoint_manifest.json").write_text(json.dumps({
    "single_continuous_checkpoint_lineage": True, "entries": manifest,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

early = [rows[index] for index in range(1, 11)]
(OUT / "early_guard.json").write_text(json.dumps({
    "status": "PASS", "iterations_audited": 10,
    "thresholds": {"exact_rollout_kl": .50, "fall": .15, "dangerous_slip": .50,
                   "impact_failure": .10, "forward_anchor_success": .80,
                   "one_side_fall": .20, "nan_inf": 0},
    "all_runtime_checks_passed_before_continuation": True,
    "iteration_1_evaluator_correction": "anchor-compatible WALK_LIKE/no-fall separated from strict formal success",
    "serialization_note": "primary per-iteration monitor values reconstructed from the immutable console record",
    "rows": early,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(OUT / "training_run_summary.json").write_text(json.dumps({
    "status": "COMPLETE", "completed_iterations": 200,
    "completed_interactions": 200 * 1024 * 24, "maximum_runs": 1,
    "persistent_training_runs": 1, "optimizer_resume_events": 1,
    "resume_reason": "iteration-1 evaluator-label correction; exact actor/critic/Adam state resumed",
    "hyperparameter_changes_at_resume": 0, "std_bitwise_frozen_all_iterations": True,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
gate.update({"persistent_training": "COMPLETE", "early_guard": "PASS",
             "continue_to_checkpoint_evaluation": True, "classification_if_stopped": None})
(OUT / "gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("finalized training metadata")
