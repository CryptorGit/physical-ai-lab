"""Build fail-closed D11 artifacts after result serialization interruption."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
START = "4153ec7310ac449ed60fcc14574886cda5bf9904"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
REPORT = REPO / "research/exp_014_phase_2_d11_stop_student_heldout_report.md"
CLASSIFICATION = "EXP014_D11_HELDOUT_RUNTIME_INTERRUPTED"


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def write_csv(name, rows, fields):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


ledger = read(OUT / "heldout_access_ledger.json")
actual = ledger["sealed_episode_ids"]
ledger.update({"status": "RUNTIME_INTERRUPTED_AFTER_ALL_EPISODES_BEFORE_RESULT_SERIALIZATION", "unevaluated_episode_ids": [], "resume_permitted": False, "rerun_permitted": False, "interruption_reason": "simulation context terminated the process after batch-2 ledger commit and before aggregate/result serialization", "result_selection_access": False})
dump("heldout_access_ledger.json", ledger)

rows = []
for episode_id in actual:
    condition = int(episode_id.split(":c", 1)[1].split(":", 1)[0])
    rows.append({"episode_id": episode_id, "condition_id": condition, "evaluation_attempts": 1, "physics_completed": True, "result_serialized": False, "formal_outcome": "UNAVAILABLE_RUNTIME_INTERRUPTION", "fallback": False})
write_csv("heldout_episode_results.csv", rows, list(rows[0]))

counts = Counter(row["condition_id"] for row in rows)
def base_condition(condition):
    if condition < 16: return "zero_yaw", 22.5 * condition, .3, 0.
    if condition < 32:
        offset = condition - 16; return "moving_yaw", 45. * (offset // 2), .3, (-.3, .3)[offset % 2]
    return "pure_yaw", 0., 0., (-.3, .3)[condition - 32]


matrix_rows = []
for condition in range(34):
    kind, direction, speed, yaw = base_condition(condition)
    matrix_rows.append({"condition_id": condition, "kind": kind, "formal_direction_deg": direction, "formal_speed": speed, "formal_yaw": yaw, "episode_count": counts[condition], "perturbation_contract": "direction +/-5deg; speed +/-0.05m/s; yaw -0.04..+0.04rad/s; timing 0.45/0.50/0.55s", "moving_start_validity": "UNAVAILABLE", "stop_acquisition": "UNAVAILABLE", "conditional_hold": "UNAVAILABLE", "conditional_joint_success": "UNAVAILABLE", "end_to_end_success": "UNAVAILABLE", "fall": "UNAVAILABLE", "slip": "UNAVAILABLE", "impact": "UNAVAILABLE", "saturation": "UNAVAILABLE", "acquisition_time_median": "UNAVAILABLE", "acquisition_time_p95": "UNAVAILABLE"})
write_csv("heldout_condition_matrix.csv", matrix_rows, list(matrix_rows[0]))
dump("heldout_condition_matrix.json", {"status": "NOT_AVAILABLE", "reason": "result serialization interruption; completed episodes may not be rerun", "condition_groups": 34, "condition_episode_counts": {str(key): value for key, value in sorted(counts.items())}, "worst_condition": None})

unavailable = {"status": "NOT_AVAILABLE", "reason": "result serialization interruption after one-time physical evaluation; no rerun or fallback allowed", "episode_count": len(rows)}
dump("heldout_moving_start_validity.json", {**unavailable, "valid_starts": None, "failed_starts": None})
dump("heldout_stop_acquisition.json", {**unavailable, "conditional_STOP_acquisition": None})
dump("heldout_stand_hold.json", {**unavailable, "conditional_S_HOLD": None})
dump("heldout_joint_success.json", {**unavailable, "conditional_joint_success": None, "minimum_condition_joint_success": None})
dump("heldout_end_to_end.json", {**unavailable, "end_to_end_success": None})
dump("heldout_safety.json", {**unavailable, "fall": None, "dangerous_slip": None, "impact": None, "velocity_saturation": None, "torque_saturation": None, "nan_inf": None})
write_csv("heldout_handoff.csv", [], ["episode_id", "action_l2", "action_cosine", "joint_target_jump", "root_continuity", "contact_continuity", "new_safety_failure"])
dump("heldout_handoff.json", {**unavailable, "physical_gate": "NOT_EVALUABLE"})

identity = read(REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction/selected_checkpoint_identity.json")
dump("selected_candidate_identity.json", {**identity, "DAgger_round": 0, "candidate_count": 1, "fallback_count": 0, "policy_updates": 0})
dump("exp014_omnidirectional_stop_specialist_v1_not_authorized.json", {"status": "NOT_AUTHORIZED", "reason": CLASSIFICATION, "candidate": identity["checkpoint"], "sha256": identity["byte_sha256"], "heldout_episode_attempts": len(rows), "completed_episode_ids": len(actual), "formal_results_available": False, "fallback": False})
dump("stage_classification.json", {"classification": CLASSIFICATION, "integrity": "PASS", "access_ledger": ledger["status"], "authorization": "NOT_AUTHORIZED", "fallback": False})
dump("recommended_next_action.json", {"one_experiment": "audit the D11 lifecycle/result-serialization interruption without reopening or rerunning completed held-out episodes", "heldout_rerun": False, "fallback": False, "training": False})
dump("stage_reference.json", {"phase": "2-D11", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START), "D10_classification": "EXP014_D10_FROZEN_S1_CLOSED_LOOP_PASS", "date": "2026-08-04", "timezone": "Asia/Tokyo", "remote_push": False})
dump("protocol.json", {"candidate_count": 1, "sealed_episode_count": 579, "sealed_batches": 2, "sealed_conditions": 34, "condition_families": {"zero_yaw": 16, "moving_yaw": 16, "pure_yaw": 2}, "perturbations": {"direction_deg": [-5, 0, 5], "speed_mps": [-.05, 0, .05], "yaw_delta_rad_s": [-.04, -.03, -.02, -.01, 0, .01, .02, .03, .04], "stop_timing_s": [.45, .50, .55]}, "seed": 20279103, "runtime": {"actor": 1, "checkpoint": 1, "Teacher": 0, "route_switch": 0, "action_blending": 0, "phase_classifier": 0}, "policy_training": 0, "fallback": 0, "heldout_access_count": 1, "result_contract": "fail-closed on missing serialized outcomes"})

protected = {}
for phase in range(6, 11):
    for path in (REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion").glob(f"phase_2_d{phase}*"):
        rel = path.relative_to(REPO).as_posix()
        try: protected[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
trees = {}
for number in range(5, 14):
    for path in (REPO / "experiments/isaaclab").glob(f"exp_{number:03d}_*"):
        rel = path.relative_to(REPO).as_posix()
        try: trees[rel] = git("rev-parse", f"{START}:{rel}")
        except subprocess.CalledProcessError: pass
dump("protected_hashes.json", {"starting_head": START, "exp005_to_exp013_tree_hashes": trees, "D6_to_D10_tree_hashes": protected, "D6_to_D10_committed_diff": 0, "D7_train_validation_hashes": read(D7 / "dataset_hashes.json"), "S1_sha256": identity["byte_sha256"], "sealed_sha256": sha(D7 / "raw/sealed_heldout_snapshots.pt"), "policy_updates": 0, "PPO": 0, "DAgger": 0, "S2": 0, "RUN": 0, "OMNI_RUN": 0, "Causal_DAgger_Dataset_V2": 0, "candidate_count": 1, "fallback": 0, "remote_push": False})
(OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# The one-time held-out evaluation has already been consumed. Do not rerun it.\nGet-FileHash results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation/raw/sealed_heldout_snapshots.pt -Algorithm SHA256\nGet-Content results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11_stop_student_heldout/heldout_access_ledger.json\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/build_phase2_d11_interrupted_artifacts.py\n", encoding="utf-8")

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text(f"""# exp_014 Phase 2-D11 sealed held-out report

## Result

Classification: **{CLASSIFICATION}**. Pre-open identity, sealed hash, split integrity, and protected-path checks passed. The one-time ledger was written before the sealed payload was deserialized.

All 579 sealed episode IDs across 34 conditions were attempted exactly once and batch 2 was committed to the access ledger. The simulation context then terminated the process before episode metrics were serialized. The ledger therefore has zero unevaluated episodes, but no recoverable formal outcomes. Re-running completed held-out episodes is prohibited, so moving-start, STOP, hold, joint, safety, condition-minimum, and handoff results are reported as unavailable rather than inferred.

No checkpoint was selected or changed, no fallback or training was performed, and `Exp014DistilledOmnidirectionalStopSpecialistV1` is **NOT_AUTHORIZED**. The sealed access count is one. The next permissible work is a lifecycle/result-serialization audit that does not reopen or rerun the held-out split.
""", encoding="utf-8")
print(json.dumps({"classification": CLASSIFICATION, "episodes_attempted_once": len(rows), "unevaluated": 0, "authorization": "NOT_AUTHORIZED"}, indent=2))
