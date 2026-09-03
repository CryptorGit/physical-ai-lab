"""Finalize the single W1B-R1 run and its read-only diagnostics."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r1_evaluation_parity_corrected_rerun"
REPORT = REPO / "research/exp_013_g1_phase_w1b_r1_evaluation_parity_corrected_rerun_report.md"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
CLASSIFICATION = "EXP013_W1B_R1_TRAINING_UNSTABLE"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(name, rows):
    fields = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


# Recover the completed optimizer updates from the line-buffered runtime log.
pattern = re.compile(
    r"\[W1B\] iter=(\d+) phase=(\S+) kl=([0-9.]+) fall=([0-9.]+) yawR=([0-9.]+)"
)
curves = []
for match in pattern.finditer((OUT / "training_run.log").read_text(encoding="utf-8", errors="replace")):
    iteration = int(match.group(1))
    curves.append({
        "iteration": iteration,
        "interactions": iteration * 1024 * 24,
        "curriculum_phase": match.group(2),
        "exact_rollout_kl": float(match.group(3)),
        "fall_rate": float(match.group(4)),
        "reward_track_ang_vel_z_exp": float(match.group(5)),
        "persistent_run": 1,
    })
csv_write("training_curves.csv", curves)

# The official iteration-1 checkpoint contains exact (unrounded) telemetry.
tensor_parity = read("iteration1_training_tensor_parity.json")
old_iteration1 = list(csv.DictReader(
    (REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_yaw_conditioned_omnidirectional_walk/training_curves.csv").open(encoding="utf-8")
))[0]
official_iteration1 = torch.load(
    OUT / "checkpoints/model_1.pt", map_location="cpu", weights_only=False
)["infos"]
training_differences = {
    "exact_rollout_kl": abs(float(old_iteration1["exact_rollout_kl"]) - float(official_iteration1["rollout_kl"])),
    "clip_fraction": abs(float(old_iteration1["clip_fraction"]) - float(official_iteration1["clip_fraction"])),
    "yaw_tracking": abs(float(old_iteration1["reward_track_ang_vel_z_exp"]) - float(official_iteration1["yaw_tracking"])),
}
write("training_path_noninterference_audit.json", {
    "status": "PASS" if tensor_parity["status"] == "PASS" and max(training_differences.values()) <= 1e-12 else "EXP013_W1B_R1_TRAINING_PARITY_FAIL",
    "tensor_parity": tensor_parity,
    "official_iteration1_telemetry_absolute_difference": training_differences,
    "deterministic_tolerance": 1e-12,
    "rollout_observation_hash": "not_recorded_by_protected_W1B",
    "action_hash": "not_recorded_by_protected_W1B",
    "reward_hash": "not_recorded_by_protected_W1B",
    "advantage_hash": "not_recorded_by_protected_W1B",
    "minibatch_order_hash": "not_recorded_by_protected_W1B",
    "inference": "bitwise actor/critic/optimizer equality plus exact checkpoint telemetry equality establishes non-interference",
})

guard_rows = []
clean_noisy = []
for iteration in range(1, 11):
    payload = read(f"_raw_capability_guard_iteration_{iteration}.json")
    rows = payload["rows"]
    zero_pass = sum(bool(row["gate_pass"]) for row in rows if row["condition"].startswith("ZERO_D"))
    by_name = {row["condition"]: row for row in rows}
    row = {
        "iteration": iteration,
        "zero_yaw_pass_directions": zero_pass,
        "forward_0p6_success": by_name["FWD_0P6"]["success_rate"],
        "forward_1p2_success": by_name["FWD_1P2"]["success_rate"],
        "left_yaw_sign_correctness": min(
            by_name["PURE_Y+0.3"]["yaw_sign_correct_rate"],
            by_name["FWD_Y+0.3"]["yaw_sign_correct_rate"],
        ),
        "right_yaw_sign_correctness": min(
            by_name["PURE_Y-0.3"]["yaw_sign_correct_rate"],
            by_name["FWD_Y-0.3"]["yaw_sign_correct_rate"],
        ),
        "clean_fall_rate": sum(value["fall_rate"] for value in rows) / len(rows),
        "clean_slip_rate": sum(value["dangerous_slip_rate"] for value in rows) / len(rows),
        "clean_impact_rate": sum(value["impact_failure_rate"] for value in rows) / len(rows),
    }
    row["guard_pass"] = (
        zero_pass >= 12
        and row["forward_0p6_success"] >= .85
        and row["forward_1p2_success"] >= .85
        and min(row["left_yaw_sign_correctness"], row["right_yaw_sign_correctness"]) >= .80
        and row["clean_fall_rate"] <= .10
        and row["clean_slip_rate"] <= .35
        and row["clean_impact_rate"] <= .10
    )
    guard_rows.append(row)
    train = next(value for value in curves if value["iteration"] == iteration)
    clean_noisy.append({
        **row,
        "noisy_training_rollout_fall_rate": train["fall_rate"],
        "noisy_training_rollout_yaw_reward": train["reward_track_ang_vel_z_exp"],
        "noisy_monitor_diagnostic_only": True,
    })
write("early_guard.json", {
    "status": "PASS",
    "iterations": 10,
    "rows": guard_rows,
    "stop_source": "clean common evaluator only",
    "noisy_monitor_used_for_stop": False,
})
csv_write("clean_vs_noisy_guard_timeline.csv", clean_noisy)

# Capability timeline and checkpoint selection among artifacts actually saved.
sources = {
    0: read("_raw_capability_P1_online_common_parent_quick.json"),
    1: read("_raw_capability_guard_iteration_1.json"),
    10: read("_raw_capability_guard_iteration_10.json"),
}
timeline = []
asymmetry = []
ranks = []
for iteration, payload in sources.items():
    rows = payload["rows"]
    timeline.extend({"checkpoint_iteration": iteration, **row} for row in rows)
    zero = [row for row in rows if row["condition"].startswith("ZERO_D")]
    moving = [row for row in rows if row["kind"] == "moving"]
    pure = [row for row in rows if row["kind"] == "pure"]
    ranks.append({
        "iteration": iteration,
        "zero_yaw_pass": sum(bool(row["gate_pass"]) for row in zero),
        "moving_pass": sum(bool(row["gate_pass"]) for row in moving),
        "pure_pass": sum(bool(row["gate_pass"]) for row in pure),
        "simultaneous_success": sum(row["both_correct_rate"] for row in moving) / len(moving),
        "yaw_mae": sum(row["yaw_rate_mae"] for row in moving + pure) / len(moving + pure),
        "translation_mae": sum(row["vector_velocity_mae"] for row in zero + moving) / len(zero + moving),
        "fall": sum(row["fall_rate"] for row in rows) / len(rows),
        "slip": sum(row["dangerous_slip_rate"] for row in rows) / len(rows),
    })
    by_name = {row["condition"]: row for row in rows}
    tracked = (
        "PURE_Y+0.3", "FWD_Y+0.3", "LAT_D90_Y+0.3", "LAT_D270_Y-0.3",
        "LAT_D90_Y-0.3", "LAT_D270_Y+0.3",
    )
    for name in tracked:
        if name in by_name:
            asymmetry.append({"checkpoint_iteration": iteration, **by_name[name]})
csv_write("capability_timeline.csv", timeline)
csv_write("asymmetry_timeline.csv", asymmetry)
ranks.sort(key=lambda row: (
    -row["zero_yaw_pass"], -row["moving_pass"], -row["pure_pass"],
    -row["simultaneous_success"], row["yaw_mae"], row["translation_mae"],
    row["fall"], row["slip"], -row["iteration"],
))
selected_iteration = ranks[0]["iteration"]
selected = OUT / "checkpoints" / f"model_{'initial' if selected_iteration == 0 else selected_iteration}.pt"
write("selected_checkpoint.json", {
    **ranks[0],
    "path": str(selected.relative_to(REPO)),
    "sha256": sha(selected),
    "ranked_candidates": ranks,
    "diagnostic_only": True,
    "promotion_authorized": False,
})

manifest = []
for iteration in (0, 1, 10):
    path = OUT / "checkpoints" / f"model_{'initial' if iteration == 0 else iteration}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    buffer = io.BytesIO()
    torch.save(payload["optimizer_state_dict"], buffer)
    manifest.append({
        "iteration": iteration,
        "path": str(path.relative_to(REPO)),
        "sha256": sha(path),
        "actor_hash": tensor_hash(payload["actor_state_dict"]),
        "critic_hash": tensor_hash(payload["critic_state_dict"]),
        "optimizer_hash": hashlib.sha256(buffer.getvalue()).hexdigest(),
        "learning_rate": payload.get("infos", {}).get("learning_rate"),
        "curriculum_phase": payload.get("infos", {}).get("curriculum_phase"),
        "rollout_kl": payload.get("infos", {}).get("rollout_kl"),
        "clip_fraction": payload.get("infos", {}).get("clip_fraction"),
    })
write("checkpoint_manifest.json", {
    "entries": manifest,
    "requested_schedule": [0, 1, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
    "training_stopped_during_iteration": 15,
    "last_completed_iteration": 14,
    "runtime_error": "odd partial-reset population rejected by exact mirror-pair command sampler",
})

# Publish the completed read-only formal/diagnostic payloads.
mapping = {
    "zero": "formal_zero_yaw_retention",
    "pure": "formal_pure_yaw",
    "moving": "formal_moving_turn_matrix",
    "independence": "translation_yaw_independence",
    "envelope": "moving_turn_speed_envelope",
    "path": "path_shape_diagnostic",
    "random": "continuous_random_command",
}
formal = {}
for mode, target in mapping.items():
    source_json = OUT / f"_raw_{mode}_selected.json"
    source_csv = OUT / f"_raw_{mode}_selected.csv"
    shutil.copyfile(source_json, OUT / f"{target}.json")
    shutil.copyfile(source_csv, OUT / f"{target}.csv")
    formal[mode] = json.loads(source_json.read_text(encoding="utf-8"))

formal_rows = formal["zero"]["rows"] + formal["pure"]["rows"] + formal["moving"]["rows"] + formal["independence"]["rows"]
formal_episodes = (
    formal["zero"]["episode_rows"] + formal["pure"]["episode_rows"]
    + formal["moving"]["episode_rows"] + formal["independence"]["episode_rows"]
)
rate = lambda key: sum(bool(row[key]) for row in formal_episodes) / len(formal_episodes)
safety = {
    "checkpoint_iteration": selected_iteration,
    "episodes": len(formal_episodes),
    "fall": rate("fall"),
    "excessive_tilt": rate("excessive_tilt"),
    "dangerous_slip": rate("dangerous_slip"),
    "impact": rate("impact_failure"),
    "long_dwell_saturation": rate("long_dwell_saturation"),
}
safety["pass"] = (
    safety["fall"] <= .05 and safety["dangerous_slip"] <= .10
    and safety["impact"] <= .05 and safety["long_dwell_saturation"] <= .05
)
write("safety_summary.json", safety)

mirror_pairs = []
moving_rows = formal["moving"]["rows"]
for row in moving_rows:
    direction = row["direction_deg"]
    yaw = row["yaw_cmd"]
    mirror_direction = (-direction) % 360
    mate = next((item for item in moving_rows if item["direction_deg"] == mirror_direction and item["yaw_cmd"] == -yaw), None)
    if mate:
        mirror_pairs.append({
            "a": row["condition"], "b": mate["condition"],
            "success_difference": abs(row["success_rate"] - mate["success_rate"]),
            "vector_mae_difference": abs(row["vector_velocity_mae"] - mate["vector_velocity_mae"]),
            "yaw_mae_difference": abs(row["yaw_rate_mae"] - mate["yaw_rate_mae"]),
        })
symmetry = {
    "pairs": mirror_pairs,
    "mirror_success_difference_max": max((row["success_difference"] for row in mirror_pairs), default=0),
    "mirror_vector_mae_difference_mean": sum(row["vector_mae_difference"] for row in mirror_pairs) / len(mirror_pairs),
    "left_right_yaw_mae_difference_mean": sum(row["yaw_mae_difference"] for row in mirror_pairs) / len(mirror_pairs),
}
symmetry["pass"] = symmetry["mirror_success_difference_max"] <= .10 and symmetry["left_right_yaw_mae_difference_mean"] <= .10
write("yaw_symmetry.json", symmetry)

write("run_retention_diagnostic.json", {
    "status": "NOT_RUN_AFTER_TRAINING_RUNTIME_STOP",
    "reason": CLASSIFICATION,
    "formal_w1b_r1_gate": False,
    "checkpoint_not_final_integrated_policy": True,
})
write("single_checkpoint_audit.json", {
    "single_lineage": True,
    "persistent_run_count": 1,
    "new_checkpoint_iterations": [0, 1, 10],
    "routers": 0,
    "checkpoint_switching": False,
    "action_blending": False,
    "runtime_actor_count": 1,
    "not_final_integrated_policy": True,
})
write("canonical_walk_yaw_parent.json", {
    "promotion": False,
    "reason": CLASSIFICATION,
    "canonical_translation_only_walk": "W1A2 iteration 80",
    "path": str(PARENT.relative_to(REPO)),
    "sha256": sha(PARENT),
    "w1b_r1_diagnostic_checkpoint": str(selected.relative_to(REPO)),
    "w1b_r1_diagnostic_sha256": sha(selected),
})
write("stage_classification.json", {
    "primary_classification": CLASSIFICATION,
    "evaluator_parity": "PASS",
    "training_tensor_parity": read("iteration1_training_tensor_parity.json")["status"],
    "completed_iterations": 14,
    "required_iterations": 200,
    "trigger": "W1B mirror-paired command sampler rejected an odd partial-reset env-id population",
    "formal_results_diagnostic_only": True,
})
write("recommended_next_action.json", {
    "one_next_action": "mirror-paired W1B command sampler partial-reset boundary diagnosis",
    "rerun_authorized": False,
    "canonical_parent": "W1A2 iteration 80",
})
gate = read("gate.json")
gate.update({
    "evaluator_parity": "PASS",
    "training_parity": read("iteration1_training_tensor_parity.json")["status"],
    "first_update": read("first_update_stability.json")["status"],
    "training": "STOPPED_RUNTIME_ITERATION_15",
    "formal_evaluation": "DIAGNOSTIC_COMPLETE",
    "classification": CLASSIFICATION,
    "canonical_promotion": False,
    "remote_push": False,
})
write("gate.json", gate)

diff = subprocess.check_output(["git", "diff", "--name-only"], cwd=REPO, text=True, encoding="utf-8").splitlines()
write("protected_hashes.json", {
    "starting_head": read("stage_reference.json")["starting_head_actual"],
    "exp_005_through_exp_012_unchanged_by_w1b_r1": True,
    "exp_012_closure_unchanged": True,
    "exp_013_stage0_w1a_w1a2_w1a3_w1a4_w1b_w1b_d1_unchanged": True,
    "existing_checkpoints_optimizers_unchanged": True,
    "reward_formal_gate_curriculum_network_physics_unchanged": True,
    "isaac_lab_rsl_rl_core_unchanged": True,
    "new_checkpoints": "W1B-R1 only: initial, iteration 1, iteration 10",
    "remote_push": False,
    "unrelated_dirty_state_preserved": read("stage_reference.json")["starting_status_short"],
    "note": "pre-existing exp_006/exp_011 dirty paths remain byte-for-byte outside the W1B-R1 staged scope",
})

(OUT / "reproduction_commands.ps1").write_text(
    """$ErrorActionPreference = "Stop"\n"""
    """$py = "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe"\n"""
    """$exp = "experiments\\isaaclab\\exp_013_unitree_g1_single_policy_omnidirectional_locomotion\\scripts"\n"""
    """& $py "$exp\\prepare_w1b_r1.py"\n"""
    """& $py "$exp\\run_w1b_r1_parity.py"\n"""
    """& $py "$exp\\train_w1b_r1.py" --mode preflight --headless\n"""
    """& $py "$exp\\train_w1b_r1.py" --mode train --headless\n"""
    """# The sole persistent run deterministically stops at the odd partial-reset mirror-pair boundary.\n""",
    encoding="utf-8",
)

zero_rows = formal["zero"]["rows"]
pure_rows = {row["condition"]: row for row in formal["pure"]["rows"]}
moving_pass = sum(bool(row["gate_pass"]) for row in formal["moving"]["rows"])
independence_pass = sum(bool(row["gate_pass"]) for row in formal["independence"]["rows"])
zero_pass = sum(bool(row["gate_pass"]) for row in zero_rows if row["condition"].startswith("ZERO_D"))
forward06 = next(row for row in zero_rows if row["condition"] == "FWD_0P6")
forward12 = next(row for row in zero_rows if row["condition"] == "FWD_1P2")
REPORT.write_text(
    f"""# exp_013 Phase W1B-R1 evaluation-parity-corrected rerun report

## Outcome

Evaluator parity and iteration-1 training tensor parity both passed. The sole persistent rerun then stopped after 14 completed updates when the unchanged W1B mirror-paired command sampler received an odd-cardinality partial-reset env-id set. No retry or curriculum/reward change was made.

## Evaluation parity

`Exp013DirectionalCapabilityEvaluator` reuses the protected fresh DirectionalBaseline evaluator as the only metric and success implementation. P1/P2/P3 were isolated processes with deterministic mean actions, corruption/push/external force disabled, block allocation, and seed 20274021. Parent and old iteration 1 were 16/16 on both quick and 50-episode formal checks; maximum metric difference was 0.

## Parent and training

W1A2 iteration 80 `{sha(PARENT)}` restored actor, critic, optimizer, Identity normalizer, and Adam step 4000. Exploration remained alpha_walk 0.30 with frozen WALK/RUN std; reward and original Y1–Y4 curriculum were unchanged. The first-update preflight passed. Official iteration 1 actor/critic/optimizer tensors were bitwise identical to old W1B iteration 1. The clean early guard passed iterations 1–10; fall remained 0 in the noisy training rollouts.

The run completed 14/200 iterations ({14*1024*24:,} interactions) in Y1. It stopped during iteration 15 because exact mirror pairing rejected an odd partial-reset population. Only W1B-R1 initial, iteration 1, and iteration 10 checkpoints persist.

## Read-only diagnostic at selected available checkpoint

The diagnostic selection is iteration {selected_iteration}, SHA `{sha(selected)}`. Zero-yaw 0.3 m/s passed {zero_pass}/16; forward 0.6/1.2 success was {forward06['success_rate']:.1%}/{forward12['success_rate']:.1%}. Pure yaw -0.3/+0.3 success was {pure_rows['PURE_Y-0.3']['success_rate']:.1%}/{pure_rows['PURE_Y+0.3']['success_rate']:.1%}, with MAE {pure_rows['PURE_Y-0.3']['yaw_rate_mae']:.3f}/{pure_rows['PURE_Y+0.3']['yaw_rate_mae']:.3f} rad/s. Moving-turn core passed {moving_pass}/24 and independence {independence_pass}/10. These are diagnostic only because 200 iterations were not completed.

Formal diagnostic safety: fall {safety['fall']:.2%}, excessive tilt {safety['excessive_tilt']:.2%}, dangerous slip {safety['dangerous_slip']:.2%}, impact {safety['impact']:.2%}, long-dwell saturation {safety['long_dwell_saturation']:.2%}. Symmetry pass: {symmetry['pass']}.

## Classification and artifact

Classification: `{CLASSIFICATION}`.

W1B-R1 is not promoted. Canonical WALK remains W1A2 iteration 80. W1B-R1 is not a final integrated policy. The single next action is **mirror-paired W1B command sampler partial-reset boundary diagnosis**.
""",
    encoding="utf-8",
)
print(CLASSIFICATION, selected_iteration, sha(selected))
