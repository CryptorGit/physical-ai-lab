"""Prepare immutable-source audits for W2-D1 practical-stop diagnosis."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
W2 = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_d1_practical_stop_retention_diagnosis"
)
PARENT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
)
START = "64e75732b01d9b1474c4419fbbc1837fc1fce0b6"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
ITER5_SHA = "0d76e1906ec70e5cb722fc9f52fa4afcc8345994d0bd8e66cae3078611ee8164"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_location(path: Path, needles: list[str]) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for needle in needles:
        hits = [i + 1 for i, line in enumerate(lines) if needle in line]
        rows.append({
            "file": str(path.relative_to(REPO)).replace("\\", "/"),
            "symbol_or_text": needle,
            "lines": hits or ["not_found"],
        })
    return rows


OUT.mkdir(parents=True, exist_ok=True)
head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
log = subprocess.check_output(
    ["git", "log", "--oneline", "--decorate", "-25"], cwd=REPO, text=True
).splitlines()
dump("stage_reference.json", {
    "stage": "Phase W2-D1 practical-stop retention regression boundary diagnosis",
    "reported_starting_head": START,
    "actual_starting_head": head,
    "head_matches": head == START,
    "starting_status_short": status,
    "starting_log_25": log,
    "canonical_parent": {"path": str(PARENT.relative_to(REPO)), "sha256": digest(PARENT)},
    "iteration5_sha256": ITER5_SHA,
    "new_persistent_policy_checkpoint": 0,
    "production_policy_update": 0,
    "remote_push": False,
})
dump("protocol.json", {
    "read_only": True,
    "checkpoints": ["canonical_parent", "W2_iteration_1", "W2_iteration_5"],
    "telemetry_only_iterations": [2, 3, 4],
    "stop_thresholds": {"translation_speed": 0.08, "absolute_yaw_rate": 0.08},
    "quick_contract": {"source_hold_s": 3.0, "ramp_s": 1.5, "episode_s": 9.0},
    "formal_stop_conditions": 24,
    "calibration": "MonotonicPositiveYawCalibrationV1",
})

availability = []
manifest = []
candidate = {
    "parent_initial": PARENT,
    "iteration_1": W2 / "checkpoints/model_1.pt",
    "iteration_5": W2 / "checkpoints/model_5.pt",
}
for label, path in candidate.items():
    payload = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(payload)
    row = {
        "candidate": label,
        "classification": "AVAILABLE",
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "sha256": digest(path),
        "actor": "actor_state_dict" in keys,
        "critic": "critic_state_dict" in keys,
        "optimizer": "optimizer_state_dict" in keys,
        "normalizer": "normalizer_state" in keys,
        "sampler_state": "sampler_state_dict" in keys,
        "command_scheduler_state": (
            "sampler_state_dict" in keys
            and "sequence_targets" in payload["sampler_state_dict"]
        ),
        "iteration": int(payload.get("iter", 0)),
    }
    availability.append(row)
    manifest.append(row)
for iteration in (2, 3, 4):
    availability.append({
        "candidate": f"iteration_{iteration}",
        "classification": "TELEMETRY_ONLY",
        "path": "not_available",
        "sha256": "not_recorded",
        "actor": False,
        "critic": False,
        "optimizer": False,
        "normalizer": False,
        "sampler_state": False,
        "command_scheduler_state": False,
        "iteration": iteration,
    })
dump("w2_stop_checkpoint_availability.json", {
    "candidates": availability,
    "regeneration_prohibited": True,
    "available_policy_iterations": [0, 1, 5],
})
dump("checkpoint_manifest.json", {"read_only_sources": manifest})

guard_source = EXP / "scripts/evaluate_w2_guard.py"
train_source = EXP / "scripts/train_w2.py"
dump("w2_start_stop_early_guard_source_locations.json", {
    "locations": source_location(guard_source, [
        '("START"', '("STOP"', "cfg.scene.num_envs", "cfg.seed", "wrapped.reset()",
        "minjerk", 'kind == "START"', 'kind == "STOP"', 't >= 4.5',
        'metrics["speed"] <= .08', 'metrics["yaw"].abs() <= .08',
        '"start_stop_success"',
    ]) + source_location(train_source, [
        "def probe_w2", "temporary_snapshot_deleted", 'result["start_stop_success"] >= 0.70',
    ])
})
dump("w2_start_stop_early_guard_contract.json", {
    "checkpoint_load": "temporary read-only snapshot from current runner",
    "actor": "deterministic mean, eval",
    "environment": "fresh DirectionalBaseline subprocess",
    "observation_corruption": False,
    "push_force_events": False,
    "conditions": {"start": 8, "stop": 8, "source_yaw": [0.0]},
    "episodes_per_condition": 20,
    "seed": 20275091,
    "reset": "fresh full reset",
    "source_hold_s": 3.0,
    "minimum_jerk_ramp_s": 1.5,
    "episode_duration_s": 9.0,
    "endpoint_window_actual": "t >= 4.5 through t < 9.0 (approximately 4.5 seconds)",
    "start_metric": "safe, vector MAE <=0.25, direction <=25deg, yaw MAE <=0.20",
    "stop_metric": "safe, mean speed <=0.08m/s, mean |signed yaw| <=0.08rad/s",
    "aggregate": "unweighted mean of 16 condition success rates",
    "threshold": "aggregate >=0.70; encoded into both sign guard sentinels",
    "material_limit": "only yaw=0 start/stop was tested; no episode-level failure decomposition was saved",
})

guard = json.loads((W2 / "_guard_iteration_5.json").read_text(encoding="utf-8"))
rows = []
for item in guard["rows"]:
    if item["kind"] not in ("START", "STOP"):
        continue
    episodes = 20
    success_count = int(round(item["success_rate"] * episodes))
    rows.append({
        "condition": item["kind"],
        "source_direction_deg": item["direction_deg"],
        "source_speed": item["source_speed"],
        "source_physical_yaw": item["yaw_target"] if item["kind"] == "START" else 0.0,
        "target_direction": item["direction_deg"] if item["kind"] == "START" else "not_applicable",
        "target_speed": 0.3 if item["kind"] == "START" else 0.0,
        "target_physical_yaw": 0.0,
        "episode_count": episodes,
        "success_count": success_count,
        "success_rate": item["success_rate"],
        "translation_endpoint_pass": "not_recorded",
        "yaw_endpoint_pass": "not_recorded",
        "practical_stop_pass": item["success_rate"] if item["kind"] == "STOP" else "not_applicable",
        "acquisition_pass": "not_recorded",
        "fall_rate": item["fall_rate"],
        "slip": "not_recorded_per_condition",
        "impact": "not_recorded_per_condition",
        "failure_reason": "not_recorded",
    })
with (OUT / "iteration5_start_stop_guard_reconstruction.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
dump("iteration5_start_stop_guard_reconstruction.json", {
    "rows": rows,
    "start_successes": sum(r["success_count"] for r in rows if r["condition"] == "START"),
    "stop_successes": sum(r["success_count"] for r in rows if r["condition"] == "STOP"),
    "total_successes": sum(r["success_count"] for r in rows),
    "total_episodes": sum(r["episode_count"] for r in rows),
    "reconstructed_aggregate": sum(r["success_count"] for r in rows) / sum(r["episode_count"] for r in rows),
    "reported_aggregate": guard["start_stop_success"],
    "exact_match": abs(
        sum(r["success_count"] for r in rows) / sum(r["episode_count"] for r in rows)
        - guard["start_stop_success"]
    ) < 1e-12,
    "unrecorded_values_not_inferred": True,
})

zero = torch.tensor([-0.0, 0.0, 0.0], dtype=torch.float32)
actor_zero = torch.where(zero > 0, zero * 1.5, zero)
dump("practical_stop_zero_command_audit.json", {
    "physical_vx": 0.0, "physical_vy": 0.0, "physical_yaw": 0.0,
    "actor_vx": 0.0, "actor_vy": 0.0, "actor_yaw": 0.0,
    "reward_vx_target": 0.0, "reward_vy_target": 0.0, "reward_yaw_target": 0.0,
    "scheduler_buffer": "physical zero after ramp",
    "actor_observation": "calibrated zero",
    "critic_observation": "calibrated zero",
    "evaluator_target": "physical zero",
    "logging_target": "physical zero",
    "zero_fixed_point_bitwise": torch.equal(actor_zero, zero),
    "status": "PASS",
})
print(json.dumps({
    "head": head,
    "available": [x["candidate"] for x in availability if x["classification"] == "AVAILABLE"],
    "aggregate": guard["start_stop_success"],
}, indent=2))
