"""Prepare immutable contracts and reconstruct the W1B online guard record."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_d1_yaw_translation_interference_diagnosis"
W1B = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
ITER1 = W1B / "checkpoints/model_1.pt"
TRAIN = EXP / "scripts/train_w1b.py"
EVAL = EXP / "scripts/evaluate_w1b.py"


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8").strip()


OUT.mkdir(parents=True, exist_ok=True)
head = git("rev-parse", "HEAD")
status = git("status", "--short")
log = git("log", "--oneline", "--decorate", "-20")
dump("stage_reference.json", {
    "phase": "W1B-D1",
    "experiment": "exp_013_unitree_g1_single_policy_omnidirectional_locomotion",
    "starting_head_reported": "cac17ed64cf7d8d5ece84764d1f8593c8c109fe3",
    "starting_head_actual": head,
    "head_match": head == "cac17ed64cf7d8d5ece84764d1f8593c8c109fe3",
    "starting_status_short": status.splitlines(),
    "starting_log_oneline_20": log.splitlines(),
    "read_only_policy_diagnosis": True,
})
dump("protocol.json", {
    "objective": ["resolve online 11/16 versus fresh 16/16 parity", "diagnose parent direction-conditioned yaw asymmetry"],
    "persistent_training_runs": 0,
    "ppo_updates": 0,
    "checkpoint_updates": 0,
    "reward_changes": 0,
    "curriculum_changes": 0,
    "seed": 20275021,
})
checkpoints = []
for label, path, expected in (
    ("canonical_parent_w1a2_iteration80", PARENT, "bd60f339c2ac8b193cc5fd4063fea8e1acf520740eb94979ffee6097c63d0244"),
    ("w1b_iteration1_diagnostic", ITER1, "8389c8b18df8cd0fabc0692aa84a4132a156d795468a60fbfecb6fd2fb71a4d4"),
):
    actual = sha(path)
    checkpoints.append({"label": label, "path": str(path.relative_to(REPO)), "sha256": actual,
                        "expected_sha256": expected, "identity_pass": actual == expected, "read_only": True})
dump("checkpoint_manifest.json", {"checkpoints": checkpoints, "persistent_checkpoint_created": 0})

online = {
    "checkpoint_source": "in-memory actor immediately after PPO iteration 1; model_1.pt saved only after probe",
    "checkpoint_load_timing": "parent strict-restored before training; probe reads updated in-memory actor",
    "actor_mode": "train mode, but architecture contains no dropout or batch normalization",
    "action_sampling": "deterministic mean: runner.alg.actor(obs) with stochastic_output default false",
    "action_std_multiplier": 0.0,
    "evaluation_environment": "Isaac-Exp013-G1-W1B-YawWalk-v0 wrapped by W1AVecEnv",
    "observation_corruption": "inherited training setting (enabled); not disabled by W1A/W1A2/W1B configs",
    "disturbance_events": "inherited training base_external_force_torque and push_robot events remain active",
    "shares_training_environment": True,
    "num_envs": 1024,
    "episodes_per_direction": {"directions_0_to_247p5": 47, "directions_270_to_337p5": 46},
    "episode_duration_s": 8.0,
    "seed": 20274021,
    "seed_stream": "continued training-process RNG stream",
    "reset_method": "env.reset() on the shared training environment",
    "command_buffer_reset": "external override enabled before reset; first-step private _update_command()",
    "controller_state_reset": "delegated to shared env.reset; not independently asserted by probe",
    "gait_classifier_reset": "no stateful classifier; flight fraction proxy only",
    "metric_accumulator_reset": "new tensors allocated by probe",
    "command_ramp": "none",
    "command_hold": "8 seconds",
    "commands": "16 x 0.3 m/s zero yaw plus forward anchors, pure yaw, forward moving yaw",
    "success_gate": {"vector_mae": 0.20, "direction_error_deg": 20, "abs_yaw_rate": 0.20,
                     "flight_fraction_lt": 0.10, "dangerous_slip": False, "impact": False, "fall": False},
    "omitted_gates": ["excessive tilt", "long-dwell saturation", "heading drift p95", "named WALK classifier"],
    "episode_aggregation": "per-environment 8-second time means",
    "direction_pass": "success rate >= 0.90",
    "pass_count": "sum of PASS among first 16 conditions",
}
fresh = {
    "checkpoint_source": "model_initial.pt or model_1.pt loaded from disk",
    "checkpoint_load_timing": "before a new evaluator rollout",
    "actor_mode": "eval mode",
    "action_sampling": "deterministic mean through FrozenGaitActor",
    "action_std_multiplier": 0.0,
    "evaluation_environment": "Isaac-Exp013-G1-DirectionalBaseline-v0 with RslRlVecEnvWrapper",
    "observation_corruption": "explicitly disabled by Exp013DirectionalBaselineEnvCfg",
    "disturbance_events": "base_external_force_torque and push_robot explicitly disabled",
    "shares_training_environment": False,
    "num_envs": 520,
    "episodes_per_direction": 20,
    "episode_duration_s": 8.0,
    "seed": 20274021,
    "seed_stream": "new evaluator-process RNG stream",
    "reset_method": "new environment construction followed by wrapper.reset()",
    "command_buffer_reset": "external override enabled; first-step private _update_command()",
    "controller_state_reset": "new process and new environment",
    "gait_classifier_reset": "no stateful classifier; flight fraction proxy only",
    "metric_accumulator_reset": "new tensors allocated by evaluator",
    "command_ramp": "none",
    "command_hold": "8 seconds",
    "success_gate": {"vector_mae": 0.20, "direction_error_deg": 20, "abs_yaw_rate": 0.20,
                     "flight_fraction_lt": 0.10, "dangerous_slip": False, "impact": False, "fall": False,
                     "long_dwell_saturation": False},
    "episode_aggregation": "per-environment 8-second time means",
    "direction_pass": "success rate >= 0.90",
}
dump("online_early_guard_implementation_audit.json", online)
dump("fresh_evaluator_implementation_audit.json", fresh)
dump("online_early_guard_source_locations.json", {
    "online_probe": {"file": str(TRAIN.relative_to(REPO)), "lines": "43-62"},
    "probe_call_and_guard": {"file": str(TRAIN.relative_to(REPO)), "lines": "87-88"},
    "fresh_evaluator_setup": {"file": str(EVAL.relative_to(REPO)), "lines": "62-67"},
    "fresh_action_and_command": {"file": str(EVAL.relative_to(REPO)), "lines": "82-85"},
    "fresh_success_aggregation": {"file": str(EVAL.relative_to(REPO)), "lines": "99-117"},
})
diffs = [
    ("actor action", "deterministic mean", "deterministic mean", "IDENTICAL"),
    ("checkpoint parameters", "in-memory iteration 1", "disk model_1.pt bitwise save", "IDENTICAL"),
    ("task/environment", online["evaluation_environment"], fresh["evaluation_environment"], "MATERIAL_DIFFERENCE"),
    ("observation corruption", online["observation_corruption"], fresh["observation_corruption"], "MATERIAL_DIFFERENCE"),
    ("disturbance events", online["disturbance_events"], fresh["disturbance_events"], "MATERIAL_DIFFERENCE"),
    ("environment lifetime", "shared post-rollout/post-update", "new process/environment", "MATERIAL_DIFFERENCE"),
    ("condition allocation", "modulo interleaving across 1024 envs", "contiguous blocks across 520 envs", "MATERIAL_DIFFERENCE"),
    ("episodes per direction", "46 or 47", "20", "MATERIAL_DIFFERENCE"),
    ("RNG stream", online["seed_stream"], fresh["seed_stream"], "MATERIAL_DIFFERENCE"),
    ("actor train/eval flag", "train", "eval; no stateful train/eval layers", "NON_MATERIAL_DIFFERENCE"),
    ("tilt/saturation checks", "not in online quick gate", "tilt recorded; saturation gated", "NON_MATERIAL_DIFFERENCE"),
]
dump("online_vs_fresh_evaluator_contract_diff.json", {
    "overall": "MATERIAL_DIFFERENCE",
    "differences": [{"field": a, "online": b, "fresh": c, "classification": d} for a, b, c, d in diffs],
    "key_hypothesis": "shared W1B environment state/condition layout and continued RNG stream, not stochastic actions",
})

rows = []
for i in range(16):
    rows.append({
        "angle": i * 22.5,
        "episode_count": 47 if i < 12 else 46,
        "success_count": "not_recorded",
        "success_rate": "not_recorded",
        "vector_mae": "not_recorded",
        "direction_error": "not_recorded",
        "actual_vx": "not_recorded",
        "actual_vy": "not_recorded",
        "actual_speed": "not_recorded",
        "yaw_rate": "not_recorded",
        "heading_drift": "not_recorded",
        "gait": "flight_fraction gate only; per-direction not_recorded",
        "fall": "not_recorded",
        "slip": "not_recorded",
        "impact": "not_recorded",
        "saturation": "not_recorded",
        "failure_reason": "not_recorded",
    })
with (OUT / "online_guard_11_of_16_reconstruction.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
dump("online_guard_11_of_16_reconstruction.json", {
    "recorded_aggregate": {"zero_yaw_pass_directions": 11, "quick_fall_rate": 0.0,
                           "quick_slip_rate": 0.001953125, "quick_impact_rate": 0.0},
    "rows": rows,
    "limitation": "The online probe persisted only the aggregate PASS count; the five failed direction identities and per-direction metrics cannot be reconstructed without guessing.",
})
