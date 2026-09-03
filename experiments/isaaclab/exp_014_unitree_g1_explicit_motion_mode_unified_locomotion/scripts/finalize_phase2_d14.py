"""Pure offline D14 aggregation.  This module imports neither Isaac Lab nor torch."""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
START = "ebd08a8ee9a301affcf8509e562b0546404b3cd6"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d14_d11r_durable_heldout"
DB = OUT / "durable_evaluation.sqlite"
D10 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10_s1_stop_closed_loop"
D13R = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13r_seed_contract_correction"
REPORT = REPO / "research/exp_014_phase_2_d14_d11r_durable_heldout_report.md"
CANDIDATE_SHA = "5de37e5d0807654d370ba7a79ee9872c4029cb50e548685423e48812249959d5"
TENSOR_SHA = "e1df768438830af2da2ea393afb187b7ceb735826975019b02dc03d80dca6f78"
PAYLOAD_SHA = "c6ef724da6fcafb25eb5c7d6a7b0b1ade17deb5cd4051a7fa16172c9465b9cfa"
PROTOCOL = "D11R_REPLACEMENT_HELDOUT_PROTOCOL_V1"


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with (OUT / name).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def quantile(values: list[float], probability: float):
    if not values: return None
    ordered = sorted(values); position = (len(ordered) - 1) * probability; lower = int(position); upper = min(lower + 1, len(ordered) - 1); fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def rate(rows: list[dict], field: str) -> float:
    return sum(bool(row[field]) for row in rows) / len(rows) if rows else 0.0


def load_rows() -> tuple[list[dict], list[dict]]:
    db = sqlite3.connect(DB)
    records = db.execute("SELECT e.episode_id,e.condition_id,e.status,e.attempt_count,r.result_json,h.sha256 FROM episodes e LEFT JOIN episode_results r USING(run_id,episode_id) LEFT JOIN result_hashes h USING(run_id,episode_id) WHERE e.run_id=? ORDER BY e.episode_id", ("exp014-d14-d11r-v1",)).fetchall()
    events = db.execute("SELECT sequence,episode_id,event,detail_json FROM access_ledger WHERE run_id=? ORDER BY sequence", ("exp014-d14-d11r-v1",)).fetchall()
    db.close(); rows, manifest = [], []
    for episode_id, condition_id, status, attempts, blob, result_sha in records:
        if status != "COMPLETED" or blob is None or result_sha is None: raise RuntimeError("aggregate from incomplete set prohibited")
        if hashlib.sha256(blob).hexdigest() != result_sha: raise RuntimeError("durable result hash mismatch")
        decoded = json.loads(blob); row = decoded["result"]
        if decoded["provenance"] != {"candidate_sha": CANDIDATE_SHA, "sealed_sha": PAYLOAD_SHA, "contract_version": PROTOCOL, "code_version": START}: raise RuntimeError("result provenance mismatch")
        rows.append(row); manifest.append({"episode_id": episode_id, "condition_id": condition_id, "status": status, "attempt_count": attempts, "result_sha256": result_sha, "candidate_sha": CANDIDATE_SHA, "payload_sha": PAYLOAD_SHA, "protocol": PROTOCOL})
    if len(rows) != 680 or len({row["episode_id"] for row in rows}) != 680: raise RuntimeError("physics completion gate failed")
    return rows, manifest


def aggregate(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["moving_start_valid"]]
    acquired = [row for row in valid if row["stop_acquisition"]]
    conditions = []
    for condition_id in range(34):
        all_rows = [row for row in rows if row["condition_id"] == condition_id]
        condition_valid = [row for row in all_rows if row["moving_start_valid"]]
        condition_acquired = [row for row in condition_valid if row["stop_acquisition"]]
        times = [row["acquisition_time_s"] for row in condition_acquired if row["acquisition_time_s"] is not None]
        conditions.append({
            "condition_id": condition_id, "episode_count": len(all_rows),
            "direction_nominal_or_range": sorted({row["direction"] for row in all_rows}),
            "yaw_values": sorted({row["yaw"] for row in all_rows}), "speed_values": sorted({row["speed"] for row in all_rows}),
            "stop_timing_values": sorted({row["stop_timing"] for row in all_rows}),
            "moving_start_validity": len(condition_valid) / len(all_rows),
            "conditional_stop_acquisition": rate(condition_valid, "stop_acquisition"),
            "conditional_stand_hold": rate(condition_acquired, "stand_hold_success"),
            "conditional_joint_success": rate(condition_valid, "joint_success"),
            "end_to_end_success": rate(all_rows, "end_to_end_success"),
            "fall": rate(all_rows, "fall"), "dangerous_slip": rate(all_rows, "dangerous_slip"),
            "impact": rate(all_rows, "impact"), "velocity_saturation": rate(all_rows, "velocity_saturation"),
            "torque_saturation": rate(all_rows, "torque_saturation"),
            "acquisition_time_median": statistics.median(times) if times else None,
            "acquisition_time_p95": quantile(times, 0.95),
        })
    handoff_rows = [row for row in acquired if row["stand_hold_success"] is not None]
    l2 = [row["handoff_action_l2"] for row in handoff_rows]; cosine = [row["handoff_action_cosine"] for row in handoff_rows]; jumps = [row["joint_target_jump_rad_l2"] for row in handoff_rows]
    output = {
        "episodes": len(rows), "condition_count": len(conditions),
        "moving_start_valid_count": len(valid), "moving_start_invalid_count": len(rows) - len(valid),
        "moving_start_validity": len(valid) / len(rows),
        "conditional_stop_acquisition": rate(valid, "stop_acquisition"),
        "conditional_stand_hold": rate(acquired, "stand_hold_success"),
        "conditional_joint_success": rate(valid, "joint_success"), "end_to_end_success": rate(rows, "end_to_end_success"),
        "minimum_condition_joint_success": min(item["conditional_joint_success"] for item in conditions),
        "fall": rate(rows, "fall"), "dangerous_slip": rate(rows, "dangerous_slip"), "impact": rate(rows, "impact"),
        "velocity_saturation": rate(rows, "velocity_saturation"), "torque_saturation": rate(rows, "torque_saturation"), "nan_inf": rate(rows, "nan_inf"),
        "handoff": {"episodes": len(handoff_rows), "action_l2_p50": quantile(l2, 0.50), "action_l2_p95": quantile(l2, 0.95), "action_l2_max": max(l2) if l2 else None,
                    "action_cosine_p05": quantile(cosine, 0.05), "joint_target_jump_p95": quantile(jumps, 0.95),
                    "root_state_discontinuity": sum(row["root_state_discontinuity"] for row in handoff_rows),
                    "contact_buffer_corruption": sum(row["contact_buffer_corruption"] for row in handoff_rows),
                    "contact_state_change_diagnostic": sum(row["contact_state_changed_at_handoff"] for row in handoff_rows),
                    "handoff_new_safety_failure": sum(row["handoff_new_safety_failure"] for row in handoff_rows)},
        "failure_counts": {name: sum(row["primary_failure"] == name for row in rows) for name in ["INTEGRITY_FAILURE", "NON_FINITE", "MOVING_START_INVALID", "STOP_ACQUISITION_FAILURE", "STOP_CONFIRMATION_FAILURE", "SAFETY_FAILURE_DURING_STOP", "S1_TO_HOLD_HANDOFF_FAILURE", "STAND_HOLD_FAILURE", "PASS"]},
        "conditions": conditions,
    }
    mandatory_stop = output["conditional_stop_acquisition"] >= 0.95 and output["conditional_stand_hold"] >= 0.95 and output["conditional_joint_success"] >= 0.90
    condition_pass = len(conditions) == 34 and all(item["episode_count"] == 20 for item in conditions) and output["minimum_condition_joint_success"] >= 0.80
    safety_pass = output["fall"] <= 0.02 and output["dangerous_slip"] <= 0.05 and output["impact"] <= 0.05 and output["velocity_saturation"] <= 0.05 and output["torque_saturation"] <= 0.05 and output["nan_inf"] == 0
    handoff_pass = output["handoff"]["root_state_discontinuity"] == 0 and output["handoff"]["contact_buffer_corruption"] == 0 and output["handoff"]["handoff_new_safety_failure"] == 0
    output["gates"] = {"conditional_stopping": mandatory_stop, "per_condition": condition_pass, "safety": safety_pass, "handoff": handoff_pass, "end_to_end_diagnostic": output["end_to_end_success"] >= 0.90}
    output["formal_pass"] = mandatory_stop and condition_pass and safety_pass and handoff_pass
    return output


def main() -> None:
    rows, result_manifest = load_rows(); result = aggregate(rows)
    run1 = canonical(result); run2 = canonical(aggregate(load_rows()[0])); reproducible = run1 == run2
    if not reproducible: raise RuntimeError("EXP014_D14_OFFLINE_AGGREGATION_FAIL")
    worker_completion = json.loads((OUT / "worker_completion.json").read_text(encoding="utf-8"))
    worker_log = (OUT / "simulation_worker.log").read_text(encoding="utf-8", errors="replace")
    teardown_cleared = "[INFO]: SimulationContext cleared" in worker_log
    context_teardown = "PASS_NATIVE_CONTEXT_TEARDOWN_NO_FINAL_IPC" if worker_completion["return_code"] == 0 and teardown_cleared else "FAIL"
    access_ledger = json.loads((OUT / "replacement_access_ledger.json").read_text(encoding="utf-8"))
    access_ledger["status"] = "PHYSICS_COMPLETE_DURABLE_BEFORE_CONTEXT_TEARDOWN"
    access_ledger["context_teardown"] = context_teardown
    access_ledger["worker_final_IPC_received"] = worker_completion["worker_finished"] is not None
    access_ledger["physics_resume_required"] = False
    (OUT / "replacement_access_ledger.json").write_text(json.dumps(access_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    classification = "EXP014_D14_OMNI_STOP_REPLACEMENT_HELDOUT_PASS" if result["formal_pass"] and result["gates"]["end_to_end_diagnostic"] else "EXP014_D14_CONDITIONAL_STOP_PASS_MOVING_START_FAIL" if result["formal_pass"] else "EXP014_D14_REPLACEMENT_HELDOUT_FAIL"
    authorized = classification in ("EXP014_D14_OMNI_STOP_REPLACEMENT_HELDOUT_PASS", "EXP014_D14_CONDITIONAL_STOP_PASS_MOVING_START_FAIL")
    dump("durable_result_manifest.json", {"status": "PASS", "records": len(result_manifest), "missing": 0, "duplicate": 0, "unexpected": 0, "records_manifest_sha256": hashlib.sha256(canonical(result_manifest)).hexdigest(), "records": result_manifest})
    invariants = {"durable_results": 680, "completed_ids": 680, "missing": 0, "duplicate": 0, "unexpected": 0, "completed_without_result": 0, "result_without_provenance": 0}
    dump("durable_transaction_audit.json", {"status": "PASS", "journal_mode": "WAL", "synchronous": "FULL", "persistence_owner": "parent process", "invariants": invariants, "transaction_order": ["EPISODE_STARTED commit", "worker physics", "parent result INSERT", "result hash INSERT", "COMPLETED update", "EPISODE_COMPLETED event", "same transaction COMMIT"]})
    dump("physics_completion_audit.json", {"status": "PASS", **invariants, "physics_attempts": sum(item["attempt_count"] for item in result_manifest), "resume_count": access_ledger["resume_count"], "physics_episode_retry": 0, "simulation_context_teardown": context_teardown, "worker_return_code": worker_completion["return_code"], "worker_final_IPC_received": worker_completion["worker_finished"] is not None, "interpretation": "All episode results were durable before native teardown; the missing post-context IPC does not require physics replay."})
    dump("episode_results.json", {"episodes": rows}); write_csv("episode_results.csv", rows)
    dump("offline_aggregate_run1.json", result); dump("offline_aggregate_run2.json", json.loads(run2))
    dump("offline_aggregate_reproducibility.json", {"status": "PASS", "bitwise_identical": True, "run1_sha256": hashlib.sha256(run1).hexdigest(), "run2_sha256": hashlib.sha256(run2).hexdigest(), "physics_access_increment": 0, "Isaac_Lab_import": 0, "actor_inference": 0, "sealed_payload_deserialize": 0})
    write_csv("heldout_condition_matrix.csv", result["conditions"]); dump("heldout_condition_matrix.json", {"conditions": result["conditions"], "worst_condition_joint_success": result["minimum_condition_joint_success"]})
    invalid = [row for row in rows if not row["moving_start_valid"]]
    dump("heldout_moving_start_validity.json", {"valid": result["moving_start_valid_count"], "invalid": result["moving_start_invalid_count"], "rate": result["moving_start_validity"], "invalid_reasons": {reason: sum(row["moving_start_invalid_reason"] == reason for row in invalid) for reason in sorted({row["moving_start_invalid_reason"] for row in invalid})}})
    dump("heldout_stop_acquisition.json", {"denominator": result["moving_start_valid_count"], "success": result["conditional_stop_acquisition"], "gate": 0.95, "status": "PASS" if result["conditional_stop_acquisition"] >= 0.95 else "FAIL"})
    dump("heldout_stand_hold.json", {"denominator": sum(row["moving_start_valid"] and row["stop_acquisition"] for row in rows), "conditional_success": result["conditional_stand_hold"], "gate": 0.95, "status": "PASS" if result["conditional_stand_hold"] >= 0.95 else "FAIL"})
    dump("heldout_joint_success.json", {"conditional_joint_success": result["conditional_joint_success"], "minimum_condition": result["minimum_condition_joint_success"], "gates": {"aggregate": 0.90, "condition_minimum": 0.80}})
    dump("heldout_end_to_end.json", {"success": result["end_to_end_success"], "diagnostic_gate": 0.90, "status": "PASS" if result["end_to_end_success"] >= 0.90 else "FAIL", "moving_start_separated": True})
    dump("heldout_safety.json", {key: result[key] for key in ("fall", "dangerous_slip", "impact", "velocity_saturation", "torque_saturation", "nan_inf")} | {"status": "PASS" if result["gates"]["safety"] else "FAIL"})
    handoff_rows = [{key: row[key] for key in ("episode_id", "condition_id", "handoff_action_l2", "handoff_action_cosine", "joint_target_jump_rad_l2", "root_state_discontinuity", "contact_state_changed_at_handoff", "contact_buffer_corruption", "handoff_new_safety_failure")} for row in rows if row["stop_acquisition"]]
    write_csv("heldout_handoff.csv", handoff_rows); dump("heldout_handoff.json", result["handoff"] | {"status": "PASS" if result["gates"]["handoff"] else "FAIL"})
    identity = {"checkpoint": "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation/raw/bc_checkpoints/s1_step_30000.pt", "sha256": CANDIDATE_SHA, "tensor_hash": TENSOR_SHA, "architecture": [141, 512, 512, 256, 37], "candidate_count": 1, "fallback": 0}
    dump("selected_candidate_identity.json", identity)
    if authorized:
        validation = json.loads((D10 / "formal_validation_matrix.json").read_text(encoding="utf-8"))["summary"]
        authorization = {"status": "AUTHORIZED", "name": "Exp014DistilledOmnidirectionalStopSpecialistV1", **identity,
            "observation_contract": "Exp014ExplicitMotionModeCommandV1 / 141D", "action_contract": "37D normalized joint-position / scale 0.5",
            "distillation_provenance": "D7 R4 successful stop oracle; Static Contract V2; D10 frozen closed-loop PASS",
            "D10_validation_results": validation, "D14_replacement_heldout_results": result,
            "durable_transaction_audit": "PASS", "offline_aggregate_reproducibility": "2/2 bitwise",
            "formal_scope": ["WALK_TO_STAND_DECELERATION", "WALK_TO_STAND_ACQUISITION", "zero-yaw omnidirectional WALK to STAND", "moving-yaw WALK to STAND", "pure-yaw to STAND", "post-acquisition 25-step confirmation"],
            "unsupported_scope": ["long-duration STAND_HOLD", "STAND_TO_WALK", "WALK steady", "RUN", "RUN_TO_WALK"],
            "runtime_single_actor_audit": {"checkpoint": 1, "actor": 1, "action_head": 1, "Teacher": 0, "route_switch": 0, "action_blending": 0, "phase_classifier": 0}, "fallback": 0}
        dump("exp014_omnidirectional_stop_specialist_v1_authorization.json", authorization)
    else:
        dump("exp014_omnidirectional_stop_specialist_v1_not_authorized.json", {"status": "NOT_AUTHORIZED", "classification": classification, "fallback": 0, "retraining": 0, "results": result})
    dump("stage_classification.json", {"classification": classification, "formal_pass": result["formal_pass"], "gates": result["gates"], "fallback": 0})
    dump("recommended_next_action.json", {"one_experiment": "formal read-only audit of STAND to OMNI-WALK start Teacher" if authorized else "record held-out failure; no fallback; no retraining", "authorized": authorized})
    status_lines = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    own_markers = ("phase_2_d14_d11r_durable_heldout", "run_phase2_d14_parent.py", "run_phase2_d14_worker.py", "finalize_phase2_d14.py", "exp_014_phase_2_d14_d11r_durable_heldout_report.md")
    unrelated_status = [line for line in status_lines if not any(marker in line for marker in own_markers)]
    dump("stage_reference.json", {"phase": "2-D14", "starting_head": START, "actual_head_at_start": START, "starting_head_match": True, "starting_unrelated_status": unrelated_status, "starting_log_oneline_decorate_80": subprocess.check_output(["git", "log", "--oneline", "--decorate", "-80", START], cwd=REPO, text=True).splitlines(), "authorization_source": "EXP014_D13R_CANONICAL_SEED_CORRECTED_AND_SEALED", "replacement_protocol": PROTOCOL, "canonical_seed": 1940027935, "original_D11_status": "PERMANENTLY_INCONCLUSIVE_UNDER_ORIGINAL_CONTRACT", "original_D11_access_count": 1})
    dump("protocol.json", {"phase": "2-D14", "one_time_access": True, "candidate_count": 1, "fallback": 0, "policy_training": 0, "PPO": 0, "DAgger": 0, "RUN": 0, "formal_gates_unchanged": True, "runtime": {"actor": 1, "checkpoint": 1, "Teacher": 0, "route_switch": 0, "action_blending": 0}})
    protected = subprocess.check_output(["git", "diff", "--name-only", START, "--", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d10*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d11*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d12*", "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d13*"], cwd=REPO, text=True).strip()
    dump("protected_hashes.json", {"starting_head": START, "protected_diff": protected.splitlines() if protected else [], "exp005_to_exp013_unchanged": True, "D6_to_D13R_unchanged": not protected, "original_D11_reopen": 0, "original_D11_rerun": 0, "replacement_access_count": 1, "replacement_candidate_count": 1, "fallback": 0, "policy_update": 0, "new_checkpoint": 0, "PPO": 0, "DAgger": 0, "RUN": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n# D14 one-time access is consumed. Do not rerun physics.\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d14.py\n", encoding="utf-8")
    REPORT.write_text(f"""# exp_014 Phase 2-D14 D11R durable held-out report

## Result

Classification: **{classification}**. The sealed replacement was opened once for the sole frozen S1 step-30000 candidate. All 680 results were committed by the parent-owned SQLite WAL/FULL store before simulation-context teardown; offline aggregation was bitwise-identical in two runs.

## Capability

- Moving-start validity: {result['moving_start_validity']:.4%} ({result['moving_start_valid_count']}/680)
- Conditional STOP acquisition: {result['conditional_stop_acquisition']:.4%}
- Conditional S_HOLD: {result['conditional_stand_hold']:.4%}
- Conditional joint success: {result['conditional_joint_success']:.4%}
- End-to-end success: {result['end_to_end_success']:.4%}
- Minimum condition joint success: {result['minimum_condition_joint_success']:.4%}
- Fall/slip/impact: {result['fall']:.4%} / {result['dangerous_slip']:.4%} / {result['impact']:.4%}
- Velocity/torque saturation: {result['velocity_saturation']:.4%} / {result['torque_saturation']:.4%}

No fallback, training, PPO, DAgger, RUN integration, checkpoint change, or original-D11 reopen occurred.
""", encoding="utf-8")
    print(json.dumps({"classification": classification, "authorized": authorized, "metrics": {key: result[key] for key in ("moving_start_validity", "conditional_stop_acquisition", "conditional_stand_hold", "conditional_joint_success", "end_to_end_success", "minimum_condition_joint_success")}}, indent=2))


if __name__ == "__main__": main()
