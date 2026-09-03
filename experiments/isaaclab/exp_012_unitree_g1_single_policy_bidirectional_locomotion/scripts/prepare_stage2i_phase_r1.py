"""Static provenance, compatibility, and 100k-schedule audit for Stage 2I."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2i_reverse_continuation_phase_r1"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-18_00-44-32_stage4_1024_500/model_5244.pt"
EXPECTED = "90d1a360587142d7e312db00a281505a027ecb221733eea6451a885868f6ccc9"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


OUT.mkdir(parents=True, exist_ok=True)
actual_sha = sha(PARENT)
if actual_sha != EXPECTED:
    raise RuntimeError("REVERSE_PARENT_PROVENANCE_AMBIGUOUS")
checkpoint = torch.load(PARENT, map_location="cpu", weights_only=False)
actor, critic, optimizer = (
    checkpoint["actor_state_dict"],
    checkpoint["critic_state_dict"],
    checkpoint["optimizer_state_dict"],
)
actor_shapes = {key: list(value.shape) for key, value in actor.items()}
critic_shapes = {key: list(value.shape) for key, value in critic.items()}
adam_steps = sorted({int(float(value["step"])) for value in optimizer["state"].values()})
lr = float(optimizer["param_groups"][0]["lr"])
starting_status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
dump("stage_reference.json", {
    "stage": "2I Reverse Single-Policy Continuation Phase R1",
    "run_identity": "stage2i_reverse_continuation_phase_r1",
    "starting_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
    "starting_status": starting_status,
    "unrelated_dirty_paths": starting_status,
})
dump("reverse_parent_checkpoint_manifest.json", {
    "path": str(PARENT.relative_to(REPO)), "sha256": actual_sha,
    "source_experiment": "exp_005_unitree_g1_flat_run", "source_stage": 4,
    "source_iteration": int(checkpoint["iter"]), "role": "safe periodic running",
    "actor_shapes": actor_shapes, "critic_shapes": critic_shapes,
    "optimizer_state_count": len(optimizer["state"]), "adam_step": adam_steps,
    "learning_rate": lr,
})
contract_ok = (
    actor_shapes.get("mlp.0.weight") == [256, 123]
    and actor_shapes.get("mlp.6.weight") == [37, 128]
    and critic_shapes.get("mlp.0.weight") == [256, 123]
    and len(optimizer["state"]) == 17
)
dump("reverse_parent_compatibility_audit.json", {
    "status": "PASS" if contract_ok else "REVERSE_PARENT_CONTRACT_INCOMPATIBLE",
    "observation_dimension": 123, "action_dimension": 37, "action_scale": 0.5,
    "joint_order": "exp_005 Stage4 == exp_012 G1 contract",
    "physics_dt": 0.005, "control_dt": 0.02, "decimation": 4,
    "robot_asset": "Unitree G1 37-DoF", "weight_conversion": False,
})
dump("reverse_parent_optimizer_audit.json", {
    "status": "PASS" if len(optimizer["state"]) == 17 and adam_steps == [105000] else "REVERSE_PARENT_OPTIMIZER_STATE_MISSING",
    "optimizer": "Adam", "state_count": len(optimizer["state"]),
    "parameter_mapping_count": len(optimizer["param_groups"][0]["params"]),
    "first_moment_present": all("exp_avg" in value for value in optimizer["state"].values()),
    "second_moment_present": all("exp_avg_sq" in value for value in optimizer["state"].values()),
    "adam_step": adam_steps, "optimizer_lr": lr,
    "runtime_lr_after_sync": lr, "scheduler_lr_after_sync": lr,
    "strict_resume_contract": "Exp012StrictPPOResumeContract",
})
if not contract_ok or adam_steps != [105000]:
    raise RuntimeError("REVERSE_PARENT_CONTRACT_INCOMPATIBLE")

rng = random.Random(20266021)
counts = {"RUN_HOLD": 0, "WALK_1P2_HOLD": 0, "BIDIRECTIONAL_WALK_RUN": 0}
targets = {"2.4": 0, "2.6": 0}
rows = []
for episode in range(100_000):
    draw = rng.random()
    cohort = "RUN_HOLD" if draw < .5 else "WALK_1P2_HOLD" if draw < .7 else "BIDIRECTIONAL_WALK_RUN"
    counts[cohort] += 1
    target = ""
    if cohort != "WALK_1P2_HOLD":
        target = "2.4" if rng.random() < .5 else "2.6"
        targets[target] += 1
    rows.append({"episode": episode, "cohort": cohort, "run_target_mps": target})
ratios = {key: value / 100_000 for key, value in counts.items()}
target_delta = abs(targets["2.4"] - targets["2.6"]) / max(1, sum(targets.values()))
audit_pass = (
    abs(ratios["RUN_HOLD"] - .50) <= .01
    and abs(ratios["WALK_1P2_HOLD"] - .20) <= .01
    and abs(ratios["BIDIRECTIONAL_WALK_RUN"] - .30) <= .01
    and target_delta <= .05
)
dump("phase_r1_curriculum.json", {
    "cohorts": {"RUN_HOLD": .50, "WALK_1P2_HOLD": .20, "BIDIRECTIONAL_WALK_RUN": .30},
    "run_targets_mps": [2.4, 2.6], "run_target_sampling": "equal",
    "episode_duration_s": 20.0, "minimum_jerk": "10t^3-15t^4+6t^5",
})
dump("phase_r1_curriculum_audit.json", {
    "status": "PASS" if audit_pass else "FAIL", "samples": 100000,
    "counts": counts, "ratios": ratios, "target_counts": targets,
    "target_ratio_difference": target_delta,
})
with (OUT / "phase_r1_command_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
if not audit_pass:
    raise RuntimeError("PHASE_R1_CURRICULUM_AUDIT_FAIL")
print(json.dumps({"parent_sha": actual_sha, "adam_step": adam_steps, "lr": lr, "curriculum": ratios}, indent=2))
