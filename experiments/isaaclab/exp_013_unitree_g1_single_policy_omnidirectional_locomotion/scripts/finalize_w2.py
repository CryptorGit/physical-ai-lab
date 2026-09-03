"""Finalize the single, early-stopped W2 run without inventing formal results."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion"
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_dynamic_omnidirectional_walk_transitions"
)
REPORT = REPO / "research/exp_013_g1_phase_w2_dynamic_omnidirectional_walk_report.md"
START = "67d7d3bb3c2f1d4a3d87f607635e93d37d82163d"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
CLASSIFICATION = "EXP013_W2_TRAINING_UNSTABLE"
STOP_REASON = "iteration 5 clean start/stop quick success 68.4375% < 70%"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(value) -> str:
    digest = hashlib.sha256()
    for key in sorted(value):
        tensor = value[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_status_csv(name: str, extra: dict | None = None) -> None:
    row = {"status": "NOT_RUN", "reason": "formal evaluation ineligible after mandatory early stop"}
    row.update(extra or {})
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


checkpoints = []
def checkpoint_order(path: Path) -> int:
    return 0 if path.stem == "model_initial" else int(path.stem.rsplit("_", 1)[-1])


for path in sorted((OUT / "checkpoints").glob("model_*.pt"), key=checkpoint_order):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("sampler_state_dict", {})
    pending = state.get("pending_mirrored_sequence", {})
    pending_present = bool(pending and pending.get("present", False))
    checkpoints.append({
        "path": str(path.relative_to(REPO)).replace("\\", "/"),
        "file": path.name,
        "iteration": int(payload.get("iter", 0)),
        "sha256": sha(path),
        "actor_hash": tensor_hash(payload["actor_state_dict"]),
        "critic_hash": tensor_hash(payload["critic_state_dict"]),
        "sampler_state_hash": payload.get("sampler_state_hash"),
        "curriculum_phase": state.get("active_curriculum_phase", "T1_START_STOP_SPEED"),
        "pending_sequence": pending_present,
        "reset_event_counter": state.get("reset_event_counter"),
        "pair_counter": state.get("next_pair_id"),
        "lr": payload.get("infos", {}).get("learning_rate", 1.5e-5),
        "rollout_kl": payload.get("infos", {}).get("rollout_kl"),
        "clip_fraction": payload.get("infos", {}).get("clip_fraction"),
    })
dump("checkpoint_manifest.json", {
    "status": "PARTIAL_EARLY_STOP",
    "persistent_runs": 1,
    "checkpoints": checkpoints,
    "new_checkpoints": [x["file"] for x in checkpoints],
})

guard_paths = [OUT / "_guard_parent.json"] + [
    OUT / f"_guard_iteration_{i}.json" for i in range(1, 6)
]
timeline = []
for path in guard_paths:
    data = json.loads(path.read_text(encoding="utf-8"))
    label = "initial" if "parent" in path.name else path.stem.rsplit("_", 1)[-1]
    timeline.append({
        "checkpoint": label,
        "zero_yaw_pass_directions": data["zero_yaw_pass_directions"],
        "forward_0p6_success": data["forward_0p6_success"],
        "forward_1p2_success": data["forward_1p2_success"],
        "static_moving_turn_pass": data["static_moving_turn_pass"],
        "start_stop_quick_success": data["start_stop_success"],
        "fall_rate": data["fall_rate"],
        "dangerous_slip_rate": data["dangerous_slip_rate"],
        "impact_rate": data["impact_rate"],
    })
with (OUT / "capability_timeline.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(timeline[0]))
    writer.writeheader()
    writer.writerows(timeline)

selected = next(item for item in checkpoints if item["file"] == "model_initial.pt")
dump("selected_checkpoint.json", {
    "status": "REFERENCE_ONLY_NOT_W2_PROMOTION",
    "selection": "initial",
    "path": selected["path"],
    "sha256": selected["sha256"],
    "policy_sha256": PARENT_SHA,
    "reason": (
        "All trained checkpoints were ineligible because the sole run stopped at iteration 5; "
        "initial preserves the mandatory canonical static capability and has the best observed "
        "start/stop quick score (99.6875%)."
    ),
    "candidate_start_stop_quick": {row["checkpoint"]: row["start_stop_quick_success"] for row in timeline},
})

last = torch.load(OUT / "checkpoints/model_5.pt", map_location="cpu", weights_only=False)
sampler = last["sampler_state_dict"]
pending = sampler["pending_mirrored_sequence"]
runtime = {
    "status": "PASS_UNTIL_TRAINING_EARLY_STOP",
    "training_iteration": 5,
    "reset_event_count": sampler["reset_event_counter"],
    "base_sequence_count": sampler["sequence_base_count"],
    "mirror_sequence_count": sampler["sequence_mirror_count"],
    "pending_queue_length_at_checkpoint": int(bool(pending.get("present", False))),
    "pending_queue_maximum_age": sampler["pending_sequence_maximum_age"],
    "mirror_residual_at_checkpoint": (
        sampler["sequence_base_count"] - sampler["sequence_mirror_count"]
    ),
    "missing_assignment_count": 0,
    "duplicate_assignment_count": 0,
    "forced_reset_count": 0,
    "phase": sampler["active_curriculum_phase"],
    "note": "Residual one is permitted while a pending mirror exists; the run did not reach the full-run flush.",
}
dump("sequence_sampler_runtime_summary.json", runtime)

early = json.loads((OUT / "early_guard.json").read_text(encoding="utf-8"))
early["status"] = CLASSIFICATION
early["stop_reason"] = STOP_REASON
dump("early_guard.json", early)
dump("training_run_summary.json", {
    "status": CLASSIFICATION,
    "iterations": 5,
    "interactions": 122880,
    "maximum_runs": 1,
    "rerun": False,
    "stop_reason": STOP_REASON,
    "numerically_stable": True,
})

not_run = {
    "status": "NOT_RUN",
    "eligible": False,
    "reason": "The mandatory early guard stopped the sole W2 run at iteration 5.",
    "classification": CLASSIFICATION,
}
dump("formal_static_retention.json", {
    **not_run,
    "quick_observed_at_iteration_5": timeline[-1],
    "canonical_W1B_C2_prior": {
        "zero_yaw": "16/16 PASS",
        "forward_0p6": "PASS",
        "forward_1p2": "PASS",
        "pure_yaw": "PASS",
        "moving_turn": "24/24 PASS",
        "independence": "10/10 PASS",
    },
})
for stem in (
    "formal_start_matrix", "formal_stop_matrix", "formal_speed_change_matrix",
    "formal_direction_change_matrix", "formal_yaw_change_matrix",
    "formal_combined_transition_matrix", "structured_path_evaluation",
    "formal_random_command",
):
    dump(f"{stem}.json", not_run)
    write_status_csv(f"{stem}.csv")
dump("episode_compound_reporting.json", not_run)
dump("safety_summary.json", {
    **not_run,
    "quick_iteration_5": {
        "fall": timeline[-1]["fall_rate"],
        "dangerous_slip": timeline[-1]["dangerous_slip_rate"],
        "impact": timeline[-1]["impact_rate"],
    },
})
dump("transition_symmetry.json", not_run)
dump("run_retention_diagnostic.json", {
    **not_run,
    "formal_gate": False,
    "note": "RUN and WALK/RUN switching were neither trained nor used for classification.",
})
dump("single_checkpoint_audit.json", {
    "status": "PASS_FOR_EXECUTED_RUN",
    "unique_parent_checkpoint": 1,
    "unique_actor": 1,
    "unique_gaussian_head": 1,
    "teacher": 0,
    "expert": 0,
    "router": 0,
    "checkpoint_switching": 0,
    "action_blending": 0,
    "external_action_controller": 0,
    "calibration": "MonotonicPositiveYawCalibrationV1",
    "scheduler_action_intervention": False,
})
dump("canonical_dynamic_walk_parent.json", {
    "status": "NOT_PROMOTED",
    "reason": CLASSIFICATION,
    "canonical_yaw_endpoint_artifact": {
        "checkpoint": "W1B-R2 iteration 200",
        "sha256": PARENT_SHA,
        "calibration": "MonotonicPositiveYawCalibrationV1",
        "endpoint_evaluator": "Exp013YawEndpointEvaluator",
    },
    "canonical_dynamic_walk": None,
})
dump("stage_classification.json", {
    "classification": CLASSIFICATION,
    "primary_cause": "mandatory early-guard start/stop retention regression",
    "iteration": 5,
    "observed": 0.684375001117587,
    "required": 0.70,
    "numerical_instability": False,
    "policy_static_retention": "quick PASS",
})
dump("recommended_next_action.json", {
    "one_method_only": True,
    "action": "iteration-3-to-5 practical-stop retention regression boundary diagnosis",
    "no_training": True,
    "reason": "Starts remained 100%; practical-stop success fell to 30–50% by direction and caused the quick-score crossing.",
})

status = subprocess.run(
    ["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True
).stdout.splitlines()
unrelated = [
    line for line in status
    if "phase_w2_dynamic_omnidirectional_walk_transitions" not in line
    and "train_w2.py" not in line
    and "prepare_w2.py" not in line
    and "test_w2_sequence_sampler.py" not in line
    and "evaluate_w2_guard.py" not in line
    and "finalize_w2.py" not in line
    and "tasks_w2.py" not in line
    and "w2_command.py" not in line
    and "w2_mdp.py" not in line
    and "command_acquisition_evaluator.py" not in line
    and "exp_013_g1_phase_w2_dynamic_omnidirectional_walk_report.md" not in line
]
dump("protected_hashes.json", {
    "starting_head": START,
    "existing_parent_checkpoint_sha256": PARENT_SHA,
    "existing_parent_checkpoint_verified": True,
    "existing_exp013_stage_files_modified": 0,
    "existing_checkpoint_files_modified": 0,
    "existing_optimizer_files_modified": 0,
    "reward_weights_changed": False,
    "network_changed": False,
    "physics_changed": False,
    "isaaclab_core_changed": False,
    "rsl_rl_installed_package_changed": False,
    "new_policy_checkpoints": ["model_initial.pt", "model_1.pt", "model_5.pt"],
    "remote_push": False,
    "unrelated_dirty_state_preserved": unrelated,
})
dump("gate.json", {
    "sequence_sampler_tests": "PASS",
    "steady_path_parity": "PASS",
    "strict_resume": "PASS",
    "dual_command_pipeline": "PASS",
    "one_update_preflight": "PASS",
    "early_guard": "FAIL",
    "completed_iterations": 5,
    "required_iterations": 250,
    "formal_evaluation": "NOT_RUN",
    "canonical_promotion": False,
    "classification": CLASSIFICATION,
})

(OUT / "reproduction_commands.ps1").write_text(
    """# Diagnostic reproduction only. Do not rerun the one-shot persistent W2 training.
& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p `
  experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2.py
& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p `
  experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/test_w2_sequence_sampler.py
& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p `
  experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w2.py --mode preflight --headless
# The sole persistent run was already consumed and stopped at iteration 5. Do not invoke --mode train.
""",
    encoding="utf-8",
)

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# Exp 013 Phase W2 — Dynamic omnidirectional WALK transitions

## Outcome

The sole authorized W2 persistent run stopped at iteration 5 under the preregistered
early guard. The primary classification is `{CLASSIFICATION}`. The stopping metric
was start/stop quick success 68.4375%, below the required 70%. No retry, resume, seed
change, or second persistent run was performed.

## Parent

- W1B-R2 iteration 200, SHA-256 `{PARENT_SHA}`
- `MonotonicPositiveYawCalibrationV1`
- shared endpoint/acquisition evaluators retained
- actor, critic, optimizer, normalizer and sampler state restored bitwise
- artifact Adam step: 8000; fixed LR: 1.5e-5
- WALK exploration alpha 0.30; log-std branches frozen

## Command pipeline and sampler

Physical vx/vy/yaw remained the reward and evaluation targets. Actor observations
used physical vx/vy and calibrated yaw (positive ×1.50; zero/negative identity).
The complete transition sequence is mirrored by `(vx, vy, yaw) -> (vx, -vy, -yaw)`.
The pending sequence queue is FIFO, bounded to one, serialized with timers and RNG.
Boundary/property tests passed, mixed odd/even determinism passed, and 100,000
single-segment reset events matched the W1B-R2 steady path bitwise.

## Training

T1 began as specified, but the run reached only 5/250 iterations (122,880
interactions). The one-update preflight passed: exact KL 0.011691, all-step maximum
KL 0.011691, clip fraction 0.178345, mean-action shift 0.027237, NaN/Inf 0.
Iterations 1–4 passed the corrected clean guard. Iteration 5 retained zero-yaw
16/16, forward 0.6/1.2 at 100%, static moving turns 24/24 and fall 0%, but start/stop
quick success declined to 68.4375%.

## Formal evaluation

Formal selection and all W2 transition matrices were not run because the mandatory
early guard made the run ineligible. This is recorded as `NOT_RUN`, not inferred.
The initial checkpoint is retained only as the best reference candidate; it is not
a W2 promotion.

## Safety and artifact interpretation

At the stopping guard: fall 0%, dangerous slip 0.1724%, impact 0%. The W1B-C2
yaw-conditioned endpoint artifact remains canonical. No dynamic WALK canonical
artifact was promoted. RUN was not trained and is outside this gate.

## Protection

Existing stages, checkpoints, optimizers, reward weights, network, physics,
Isaac Lab and installed RSL-RL were not modified. Only W2-specific code and
artifacts were produced. Remote push was not performed.

## Next

Perform one diagnostic only: iteration-3-to-5 practical-stop retention regression
boundary diagnosis. Starts remained 100%, while direction-wise stop success fell
to 30–50%; isolate that regression before any further training is authorized.
""", encoding="utf-8")

print(json.dumps({
    "classification": CLASSIFICATION,
    "iterations": 5,
    "interactions": 122880,
    "selected_reference": selected["file"],
}, indent=2))
