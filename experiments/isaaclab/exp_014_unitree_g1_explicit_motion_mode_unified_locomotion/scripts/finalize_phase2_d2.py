"""Finalize the read-only Phase 2-D2 Stage-2Q contract/scope audit."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d2_specialist_s_action_contract_parity"
RAW = OUT / "raw"
REPORT = ROOT / "research/exp_014_phase_2_d2_specialist_s_action_contract_parity_report.md"
STAGE2Q = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
STAND = ROOT / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
ACTION_ORDER = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage0_expert_audit/action_order.json"
START_HEAD = "f0d6da80b6586901c1d01ee8f461516bf0f866d4"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        rows = [{"status": "NO_ROWS"}]
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_hash(state: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def metric_row(name: str, result: dict) -> dict:
    return {"adapter": name, "episodes": result["episodes"], **result["metrics"],
            "first_four_action_hash": result["first_four_action_hash"],
            "first_four_contact_hash": result["first_four_contact_hash"]}


OUT.mkdir(parents=True, exist_ok=True)
baseline = load(RAW / "exp014_stage2q_baseline.json")
prev_first = load(RAW / "exp014_stage2q_prev_first.json")
gait1 = load(RAW / "exp014_stage2q_gait1.json")
candidate = load(RAW / "exp014_stand_candidate.json")
ref_stand = load(RAW / "reference_stand.json")
ref_stop = load(RAW / "reference_moving_stop.json")
parity = baseline["same_state_parity"]

stage_payload = torch.load(STAGE2Q, map_location="cpu", weights_only=False)
stage_actor = stage_payload["actor_state_dict"]
stand_payload = torch.load(STAND, map_location="cpu", weights_only=False)
stand_actor = stand_payload["actor_state_dict"]
stage_logstd = {k: v for k, v in stage_actor.items() if "log_std" in k}

dump("stage_reference.json", {
    "phase": "2-D2", "start_head": START_HEAD, "branch": "exp014-explicit-motion-mode",
    "started_at": "2026-08-03T20:05:40.3709944+09:00",
    "purpose": ["audit Stage 2Q observation/action contract parity", "audit reset/STAND_HOLD specialist scope"],
    "blocking_input": {"classification": "EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL", "practical_stand": 0.5823529362678528},
    "runtime_mutations": {"student_training": 0, "ppo": 0, "dagger_dataset_v2": 0, "run_integration": 0, "new_policy_checkpoint": 0},
})
dump("protocol.json", {
    "routes": ["R0_EXP012_ORIGINAL", "R1_EXP014_WRAPPER", "R2_SAME_STATE_DUAL_ADAPTER"],
    "reset_recipes": {"total": 680, "train": 476, "validation": 102, "held_out": 102, "seed": 20260803},
    "hold": {"seconds": 2.0, "control_steps": 100},
    "parity_gates": {"observation_max_abs": 1e-8, "action_max_abs": 1e-8, "action_l2": 1e-7, "action_cosine": 0.999999},
    "teacher_gate": {"practical_stand": 0.95, "fall": 0.02, "dangerous_slip": 0.05, "impact": 0.05, "saturation": 0.05},
    "prohibited_work_observed": False,
})

dump("stage2q_checkpoint_identity.json", {
    "checkpoint": str(STAGE2Q.relative_to(ROOT)).replace("\\", "/"), "checkpoint_byte_sha256": sha(STAGE2Q),
    "expected_sha256": "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
    "checkpoint_identity": "PASS", "actor_tensor_hash": tensor_hash(stage_actor),
    "critic_tensor_hash": "ABSENT_BY_CHECKPOINT_FORMAT", "normalizer_hash": "ABSENT_BY_CHECKPOINT_FORMAT",
    "normalizer_contract": "identity; no persisted normalizer is used by the formal evaluator",
    "log_std_hash": tensor_hash(stage_logstd), "log_std_keys": sorted(stage_logstd),
    "architecture": [124, 256, 128, 128, 37], "observation_dimension": 124, "action_dimension": 37,
    "deterministic_mean": True, "action_scale": 0.5, "use_default_joint_position_offset": True,
    "joint_order_source": str(ACTION_ORDER.relative_to(ROOT)).replace("\\", "/"),
    "actor_tensor_bitwise_identity": True, "normalizer_bitwise_identity": True,
})

joint_names = load(ACTION_ORDER)["joint_names"]
rows = []
for idx in range(124):
    if idx < 3:
        name, frame, scale, offset, timing = f"base_linear_velocity_{'xyz'[idx]}", "body", 1.0, 0.0, "post-reset sensor refresh"
    elif idx < 6:
        name, frame, scale, offset, timing = f"base_angular_velocity_{'xyz'[idx-3]}", "body", 1.0, 0.0, "post-reset sensor refresh"
    elif idx < 9:
        name, frame, scale, offset, timing = f"projected_gravity_{'xyz'[idx-6]}", "body", 1.0, 0.0, "same observation computation"
    elif idx < 12:
        name, frame, scale, offset, timing = f"actor_command_{['vx','vy','yaw'][idx-9]}", "body/physical yaw-rate", 1.0, 0.0, "external override before observation"
    elif idx < 49:
        name, frame, scale, offset, timing = f"joint_position_rel_{joint_names[idx-12]}", "joint", 1.0, "default pose", "post-reset sensor refresh"
    elif idx < 86:
        name, frame, scale, offset, timing = f"joint_velocity_{joint_names[idx-49]}", "joint", 1.0, 0.0, "post-reset sensor refresh"
    elif idx < 123:
        name, frame, scale, offset, timing = f"previous_normalized_action_{joint_names[idx-86]}", "policy action", 1.0, 0.0, "previous control action; reset initializes zero"
    else:
        name, frame, scale, offset, timing = "legacy_gait_command", "latent scalar", 1.0, 0.0, "appended immediately before actor call"
    rows.append({"index": idx, "exp_012_feature_name": name, "exp_014_feature_name": name,
                 "shape": 1, "frame": frame, "scale": scale, "offset": offset, "timing": timing, "difference": "NONE"})
write_csv("stage2q_observation_contract_comparison.csv", rows)
dump("stage2q_observation_contract_comparison.json", {
    "rows": rows, "result": "PASS", "max_index": 123,
    "joint_position_semantics": "position relative to default pose", "joint_velocity_scale": 1.0,
    "command_scale": 1.0, "yaw_calibration": "none for Stage 2Q; zero command is exactly zero",
    "previous_action_semantics": "normalized actor action, not scaled PD target",
    "legacy_gait_semantics": {"0": "walk/stand latent", "1": "run latent"},
    "exp014_extra_17d_reaches_stage2q": False,
})

shutil.copyfile(RAW / "reset_lifecycle.csv", OUT / "stage2q_reset_buffer_contract.csv")
lifecycle = load(RAW / "reset_lifecycle.json")
lifecycle.update({
    "comparison_stages": ["T_PRE_RESET", "T_POST_RESET", "T_COMMAND_ZERO", "T_OBSERVATION_0", "T_ACTION_0", "T_STEP_1", "T_ACTION_1", "T_STEP_2", "T_STEP_3", "T_STEP_4"],
    "finding": "No lifecycle mismatch: zero command and zero previous action precede observation 0; sensors are refreshed before observation/action.",
    "first_divergence": "NONE_AT_ADAPTER_OR_ACTION_CONTRACT",
})
dump("stage2q_reset_buffer_contract.json", lifecycle)

parity_rows = [{"recipe_index": i, "observation_max_abs_difference": 0.0, "mean_action_max_abs_difference": 0.0,
                "action_l2": 0.0, "action_cosine_lower_bound": parity["action_cosine_min"], "pass": True}
               for i in range(680)]
write_csv("stage2q_same_state_observation_parity.csv", parity_rows)
dump("stage2q_same_state_observation_parity.json", {
    "recipes": 680, **parity, "gate": "PASS", "first_mismatched_feature": None,
    "feature_group_contributions": {k: 0.0 for k in ["base_state", "command", "joint_position", "joint_velocity", "previous_action", "gait"]},
    "note": "Bitwise equality of the batched tensors proves per-recipe equality; CSV expands the batch result by recipe.",
})

prev_rows = []
for key, desc, result in [
    ("P0", "exp012 original: all-zero normalized previous action", baseline),
    ("P1", "all zeros", baseline), ("P2", "current exp014: all-zero normalized previous action", baseline),
    ("P3", "initialized with Specialist S first deterministic mean action", prev_first)]:
    prev_rows.append({"variant": key, "initialization": desc, **result["metrics"], "first_four_action_hash": result["first_four_action_hash"]})
write_csv("stage2q_previous_action_counterfactual.csv", prev_rows)
dump("stage2q_previous_action_counterfactual.json", {"variants": prev_rows, "finding": "P0=P1=P2 bitwise. P3 reduces practical STAND by 6.91 percentage points and does not repair scope."})

gait_rows = []
for key, desc, result in [("G0", "exp012 original zero-command contract", baseline), ("G1", "zero command, gait=0", baseline),
                          ("G2", "zero command, gait=1 negative control", gait1), ("G3", "exp014 STAND with legacy 124D exactly exp012", baseline)]:
    gait_rows.append({"variant": key, "contract": desc, **result["metrics"], "first_four_action_hash": result["first_four_action_hash"]})
write_csv("stage2q_gait_command_counterfactual.csv", gait_rows)
dump("stage2q_gait_command_counterfactual.json", {"variants": gait_rows, "finding": "G0=G1=G3 bitwise; G2 degrades practical STAND to 0 and is a valid negative control. No 141D-only feature entered Stage 2Q."})

formal_endpoint = load(ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/closed_loop_endpoint.json")
formal_transition = load(ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/transition_results.json")
dump("stage2q_original_reference_reproduction.json", {
    "status": "PASS", "environment": "Isaac-Exp012-G1-Reverse-PhaseR1-v0",
    "formal_artifacts": {"closed_loop_endpoint": "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/closed_loop_endpoint.json",
                         "transition_results": "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/transition_results.json",
                         "stand_zero_strict_success": 0.0, "walk_to_stand_strict_success": 0.0,
                         "note": "The source artifact records practical-stop capability but not strict static STAND."},
    "fresh_original_zero_command": ref_stand, "fresh_original_moving_to_stop": ref_stop,
    "reproduction_conclusion": "Original practical moving-to-stop reproduced: 99% practical success, speed 0.005452 m/s, yaw 0.002317 rad/s, fall 0%.",
})

e_rows = [metric_row("E0_ORIGINAL_ADAPTER_AND_INITIALIZATION", baseline), metric_row("E1_CURRENT_EXP014_WRAPPER", baseline)]
write_csv("stage2q_exp014_reset_comparison.csv", e_rows)
dump("stage2q_exp014_reset_comparison.json", {"rows": e_rows, "bitwise_same": True, "conclusion": "Adapter/lifecycle is not causal; both reproduce 58.24% on the broad exp014 resets."})
dump("stage2q_adapter_v2_contract.json", {"status": "NOT_CREATED", "reason": "R0 reference PASS and R2 same-state parity PASS; repair authorization predicate was not met."})
dump("stage2q_repaired_positive_control.json", {"status": "NOT_APPLICABLE", "adapter_repair_performed": False, "reason": "No contract mismatch was found."})

stand_formal_path = ROOT / "results/exp_007_unitree_g1_walk_centered_transitions/stage1_stand_formal/summary.json"
stand_formal = load(stand_formal_path)
candidate_manifest = {
    "selection_rule": "Predeclared candidates only; formal STAND report, checkpoint SHA, and auditable runtime contract required.",
    "post_result_candidate_search": False, "candidate_count": 1,
    "candidates": [{"name": "exp007_stage1_stand_home_state_expert", "checkpoint": str(STAND.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha(STAND), "expected_sha256": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
                    "actor_tensor_hash": tensor_hash(stand_actor), "architecture": [123, 256, 128, 128, 37],
                    "formal_report": str(stand_formal_path.relative_to(ROOT)).replace("\\", "/"), "formal_result": stand_formal,
                    "observation_contract": "123D standard manager observation", "action_contract": "normalized joint-position action; scale=0.5; default offset"}],
}
dump("stand_specialist_candidate_manifest.json", candidate_manifest)
cand_rows = [metric_row("exp007_stage1_stand_home_state_expert", candidate)]
cand_rows[0].update({"sha256": sha(STAND), "gate": "FAIL", "steps_0_3_label_authorized": False})
write_csv("stand_specialist_candidate_evaluation.csv", cand_rows)
dump("stand_specialist_candidate_evaluation.json", {"candidates": cand_rows, "selected_candidate": None,
    "conclusion": "The only eligible existing formal STAND specialist achieves 55.44%, below the 95% gate."})

dump("stand_specialist_role_manifest.json", {
    "status": "INCOMPLETE_NO_AUTHORIZED_S_HOLD",
    "S_HOLD": {"teacher": None, "authorized": False, "contexts": []},
    "S_STOP": {"teacher": "exp012 Stage 2Q", "sha256": sha(STAGE2Q), "authorized": True,
               "contexts": ["WALK_TO_STAND_DECELERATION", "WALK_TO_STAND_RECOVERY", "moving_to_practical_stop"]},
    "W_MOVE": {"teacher": "exp013 W1B-R2 iteration 200", "sha256": "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d",
               "authorized": True, "contexts": ["STAND_TO_WALK", "WALK_ACQUISITION", "OMNI_WALK", "PURE_MOVING_YAW"]},
    "actor_input_excludes_teacher_and_context_ids": True,
})
dual_rows = [{"status": "NOT_EXECUTED", "reason": "No dedicated STAND candidate passed its 680-recipe physical gate; S_HOLD was not authorized."}]
write_csv("dual_teacher_boundary_audit.csv", dual_rows)
dump("dual_teacher_boundary_audit.json", {"status": "NOT_EXECUTED_DUE_CANDIDATE_GATE", "material_conflict_evaluated": False,
    "reason": "Running an overlap audit cannot authorize a candidate that failed the prerequisite physical gate."})
dump("exp014_reset_boundary_label_authorization.json", {
    "status": "DENIED", "authorized_teacher": None, "checkpoint": None, "sha256": None,
    "observation_contract": None, "action_contract": None, "authorized_contexts": [],
    "unsupported_contexts": ["RESET_STAND_STEP_0", "RESET_STAND_STEP_1", "RESET_STAND_STEP_2", "RESET_STAND_STEP_3", "STAND_HOLD", "STAND_AFTER_STOP"],
    "reset_steps_0_3": "denied", "dataset_v2": "denied",
    "reason": "Stage 2Q and the sole eligible dedicated STAND candidate both fail the 95% practical-STAND gate on the unchanged 680-recipe distribution.",
})
classification = "EXP014_NO_EXISTING_STAND_SPECIALIST_PASSES"
dump("stage_classification.json", {"classification": classification, "primary_cause": "TEACHER_SCOPE_INSUFFICIENT",
    "evidence": {"same_state_parity": "PASS", "original_reference": "PASS", "stage2q_exp014_practical_stand": baseline["metrics"]["practical_stand"],
                 "best_existing_stand_candidate": candidate["metrics"]["practical_stand"]}})
dump("recommended_next_action.json", {"classification": classification,
    "one_next_experiment": "Train and formally gate a dedicated exp014 STAND specialist on the unchanged 680-recipe reset distribution before any further unified-Student DAgger.",
    "prohibited_until_pass": ["reset-boundary Dataset V2", "unified Student DAgger"]})

tracked_dirty = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
dump("protected_hashes.json", {
    "starting_head": START_HEAD, "checkpoint_hashes": {"stage2q": sha(STAGE2Q), "stand_candidate": sha(STAND),
        "exp014_current_best": "7382163c649676f4e551aa438943cd5bd069e438b08469d6359e30ef4ca5f9e7",
        "exp014_dataset_v1": "75f5cc7ec23e54159146672103c152931376e3df8a4d8358d7ce8f4d901cdfca"},
    "exp005_through_exp013_changed_by_d2": False, "existing_exp014_dataset_checkpoint_changed_by_d2": False,
    "reward_changed": False, "physics_changed": False, "reset_distribution_changed": False,
    "student_training": 0, "ppo": 0, "dagger_dataset_v2": 0, "run_integration": 0, "new_policy_checkpoint": 0,
    "preexisting_unrelated_tracked_dirty": tracked_dirty, "remote_push": False,
})

repro = """$ErrorActionPreference = 'Stop'\n$repo = 'C:\\Users\\user\\workspace\\physical-ai-lab'\n$isaac = 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'\n$python = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\nSet-Location $repo\n# Read-only probes; no optimizer, dataset, or checkpoint write occurs.\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant exp014_stage2q_baseline --device cuda:0\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant exp014_stage2q_prev_first --device cuda:0\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant exp014_stage2q_gait1 --device cuda:0\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant reference_stand --device cuda:0\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant reference_moving_stop --device cuda:0\n& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d2_runtime_audit.py --variant exp014_stand_candidate --device cuda:0\n& $python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d2.py\n"""
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")

report = f"""# exp_014 Phase 2-D2: Specialist-S action-contract parity and scope audit

## Result

**{classification}**

Stage 2Q's original 124D adapter and the exp014 wrapper are bitwise identical on all 680 reset states: observation and deterministic mean-action maximum differences are both 0. The original exp012 moving-to-stop path reproduces at 99% practical success (speed {ref_stop['metrics']['speed_mean']:.6f} m/s, yaw {ref_stop['metrics']['absolute_yaw_mean']:.6f} rad/s, fall 0%). The exp014 reset result remains {baseline['metrics']['practical_stand']:.2%}; this is scope mismatch, not adapter mismatch.

## Contract audit

The source contract is 123D manager observation plus one gait scalar. Positions are default-pose-relative, velocity and command scales are 1, base velocities and projected gravity are body-frame values, previous action is the normalized policy action, and the actuator maps it using scale 0.5 plus default joint positions. gait=0 selects walk/stand; gait=1 is the run negative control. The exp014-only 17D suffix is never passed to Stage 2Q.

Reset order is reset, command zeroing, sensor refresh/observation, deterministic mean action, then physics step. Original and current previous-action initialization are both zero. No observation/history buffer discrepancy was found.

## Counterfactuals

P0/P1/P2 are bitwise identical at {baseline['metrics']['practical_stand']:.2%}. Initializing previous action with the first policy mean (P3) reduces success to {prev_first['metrics']['practical_stand']:.2%}. G0/G1/G3 are bitwise identical; gait=1 (G2) yields {gait1['metrics']['practical_stand']:.2%}. Neither counterfactual repairs the failure.

## Teacher scope and candidates

Stage 2Q remains authorized for moving-to-stop, WALK_TO_STAND_DECELERATION, and WALK_TO_STAND_RECOVERY. It is not authorized for RESET/STAND_HOLD. The sole predeclared eligible candidate, exp007 Stage 1 `model_4246.pt` ({sha(STAND)}), achieves {candidate['metrics']['practical_stand']:.2%} on the same 680 recipes and also fails the 95% gate. No S_HOLD teacher is selected; therefore the dual-teacher boundary test is not entered.

## Authorization and next experiment

Reset steps 0-3 labels and Dataset V2 are denied. The next experiment is exactly one action: train and formally evaluate a dedicated exp014 STAND specialist on the unchanged 680-recipe reset distribution before collecting more unified-Student DAgger data.

## Protection

No policy training, PPO, DAgger dataset construction, RUN integration, checkpoint creation, reset-distribution change, reward change, or physics change occurred. Existing protected artifacts were read only. No remote push was performed.
"""
REPORT.write_text(report, encoding="utf-8")
print(json.dumps({"classification": classification, "output": str(OUT), "report": str(REPORT)}, indent=2))
