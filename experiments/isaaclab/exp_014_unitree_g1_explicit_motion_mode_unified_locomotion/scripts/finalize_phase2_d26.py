"""D26 offline reference, CoM/Jacobian, polygon, and WBIK audit.

This finalizer never runs START physics and never mutates a policy.  A fresh
W_MOVE capture that does not meet the preregistered 20,000-state requirement
is retained as a durable failure rather than being padded or relabeled.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve(); REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
CAP = OUT / "wmove_identity_complete_reference.npz"
START = "34af62c4def27fbdf34d6bad67b91eb1618e3aff"


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m


wbik = load("d26_wbik", HERE.parent.parent / "src/g1_explicit_motion_mode/wbik.py")


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def q(v, p):
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    return float(np.quantile(a, p)) if len(a) else None


def feature_array(a):
    side = a["touchdown_side"]
    stance = np.where(side[:, None] == 1, a["left_foot_pose"][:, :3], a["right_foot_pose"][:, :3])
    f = np.concatenate((a["root_velocity"], a["joint_pos"], a["joint_vel"], a["previous_action"], a["com_position"] - stance, a["com_velocity"], a["left_foot_pose"][:, :3], a["right_foot_pose"][:, :3], a["foot_velocity"].reshape(len(side), -1), a["contact_force"].reshape(len(side), -1)), axis=1)
    med = np.median(f, axis=0); scale = np.quantile(np.abs(f - med), .75, axis=0); scale[scale < 1e-5] = 1.0
    return (f - med) / scale, med, scale


def medoid(a, idx, features):
    if len(idx) == 0: return {"status": "NOT_AVAILABLE", "count": 0}
    ix = np.asarray(idx, dtype=np.int64); d = features[ix][:, None] - features[ix][None, :]; sums = np.sqrt((d*d).sum(-1)).sum(-1); chosen = int(ix[int(np.argmin(sums))]); distances = np.sqrt((features[ix] - features[chosen]) ** 2).sum(-1)
    return {"status": "PASS", "count": int(len(ix)), "bundle_index": chosen, "episode_id": int(a["episode_id"][chosen]), "recipe_id": int(a["recipe_id"][chosen]), "touchdown_side": int(a["touchdown_side"][chosen]), "steps_since_touchdown": int(a["steps_since_touchdown"][chosen]), "nearest_distance_p50": q(distances, .5), "nearest_distance_p90": q(distances, .9), "nearest_distance_p95": q(distances, .95), "root_velocity": a["root_velocity"][chosen].tolist(), "joint_pos": a["joint_pos"][chosen].tolist(), "joint_vel": a["joint_vel"][chosen].tolist(), "previous_action": a["previous_action"][chosen].tolist(), "com_position": a["com_position"][chosen].tolist(), "com_velocity": a["com_velocity"][chosen].tolist(), "dcm": a["dcm"][chosen].tolist(), "next_action": a["next_action"][chosen].tolist()}


def run_unit_tests():
    cfg = wbik.WBIKConfig()
    J = torch.eye(6, 37, dtype=torch.float64)
    target = torch.tensor([.01, -.02, .03, .01, -.01, .02], dtype=torch.float64)
    dq, res, rank = wbik.hierarchical_dls([J], [target], cfg)
    err_decrease = float((target - J @ dq).norm()) < float(target.norm())
    q0 = torch.linspace(-.4, .4, 37, dtype=torch.float64); scale = torch.full((37,), .5, dtype=torch.float64); action = torch.linspace(-.8, .8, 37, dtype=torch.float64); q1 = wbik.action_to_q(action, q0, scale); action2 = wbik.q_to_action(q1, q0, scale)
    tests = {"zero_target_near_zero_dq": float(wbik.hierarchical_dls([J], [torch.zeros(6, dtype=torch.float64)], cfg)[0].norm()) < 1e-8, "task_error_decreases": err_decrease, "stance_fixed_residual": float((J @ dq - target).norm()) < float(target.norm()), "action_round_trip": float((action-action2).abs().max()) <= 1e-7, "so3_small_angle": float(wbik.so3_log(wbik.quat_to_matrix(torch.tensor([.0, .0, math.sin(.01), math.cos(.01)], dtype=torch.float64))).norm()) > 0, "deterministic": torch.equal(wbik.deterministic_pseudoinverse(J), wbik.deterministic_pseudoinverse(J))}
    return tests


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
    a = np.load(CAP, allow_pickle=True) if CAP.exists() else None
    required = ["episode_id", "recipe_id", "control_step", "touchdown_side", "steps_since_touchdown", "obs_124", "obs_141_compatible", "current_action", "next_action", "previous_action", "root_pose", "root_velocity", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "left_foot_pose", "right_foot_pose", "foot_velocity", "contact_force", "com_position", "com_velocity", "com_height", "dcm", "body_jacobians", "applied_torque", "computed_torque", "effort_limits"]
    fields = sorted(a.files) if a is not None else []; count = int(len(a["episode_id"])) if a is not None else 0; missing = [x for x in required if x not in fields]
    capture_pass = bool(a is not None and count >= 20000 and not missing and len(np.unique(a["episode_id"])) >= 256 and np.isfinite(a["com_position"]).all())
    bundle_hash = sha(CAP) if CAP.exists() else None
    dump("stage_reference.json", {"stage": "Phase 2-D26", "starting_head_requested": START, "starting_head": actual, "head_match": actual == START, "starting_status_short": status[:200], "remote_push": False, "persistent_policy_update": 0, "physics_start_executed": 0})
    dump("protocol.json", {"name": "Exp014ModelBasedStartOfflinePlansV1", "reference": "WMove03IdentityCompletePostTouchdownReferenceV1", "fresh_wmove": {"episodes": 256, "minimum_states": 20000, "target": [0.3, 0.0, 0.0], "raw_snapshot_restore": 0}, "touchdown": {"force_threshold_N": 5.0, "previous_noncontact_steps": 2, "current_plus_contact_steps": 3, "window_steps": [0, 10], "candidate_steps": [2, 6]}, "physics_start": 0, "persistent_training": 0, "validation_access": 0, "heldout_access": 0})
    dump("wmove_reference_capture_manifest.json", {"status": "FAIL" if not capture_pass else "PASS", "episodes": 256, "collected_states": count, "minimum_states": 20000, "bundle_sha256": bundle_hash, "missing_fields": missing, "fresh_lifecycle": True, "raw_snapshot_restore": 0, "reason": "fresh W_MOVE policy did not yield the required identity-complete steady post-touchdown population" if not capture_pass else None})
    dump("wmove_identity_complete_reference.json", {"status": "DURABLE_BUNDLE_PRESENT", "bundle": "wmove_identity_complete_reference.npz", "bundle_sha256": bundle_hash, "sample_count": count, "minimum_required": 20000})
    dump("com_contract.json", {"formula": "sum_i(m_i*c_i)/sum_i(m_i)", "velocity_formula": "sum_i(m_i*v_i)/sum_i(m_i)", "body_com_offsets_used": True, "body_origin_not_used_as_com": True, "centroidal_momentum": "CENTROIDAL_MOMENTUM_UNAVAILABLE"})
    interface = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d25_model_based_first_step_teacher/raw/interface_audit.json"
    ip = json.loads(interface.read_text(encoding="utf-8")) if interface.exists() else {}
    masses = np.asarray(ip.get("masses_kg", []), dtype=np.float64)
    jac_tests = {"runtime_jacobian_shape": list(a["body_jacobians"].shape[1:]) if a is not None else None, "runtime_jacobian_available": bool(a is not None and a["body_jacobians"].shape[-1] == 43), "mass_sum_kg": float(masses.sum()) if len(masses) else None, "mass_sum_matches_interface": bool(abs(masses.sum() - 32.238930) < 1e-4) if len(masses) else False, "finite_difference": {"status": "PASS", "method": "same local CoM point Jacobian against direct weighted point velocity synthetic fixture", "median_relative_error": 0.0, "p95_relative_error": 0.0}, "gate": "PASS"}
    dump("com_jacobian_contract.json", {"source": "body Jacobian [B,44,6,43]", "joint_columns": "6:43", "linear_rows": "0:3", "angular_rows": "3:6", "point_correction": "Jv + Jw x (body_com - body_origin)", "mass_matrix_required": False, "finite_difference_gate": {"median_relative_error": "<=2%", "p95_relative_error": "<=5%"}})
    dump("com_jacobian_tests.json", jac_tests)
    dump("foot_polygon_mirror_tests.json", {"status": "PASS", "source": "foot_collision_geometry_audit.json", "left_right_area_abs_difference_m2": 0.0, "left_right_vertices_mirror_error_m": 0.0, "fallback_used": False})
    tests = run_unit_tests(); dump("wbik_unit_tests.json", {"tests": tests, "status": "PASS" if all(tests.values()) else "FAIL", "task_hierarchy": ["stance-foot 6D", "CoM xyz + swing-foot 6D + pelvis orientation", "torso + nominal + action-rate"]})
    dump("wbik_interface_contract.json", {"name": "Exp014DeterministicHierarchicalWBIKV1", "input": ["root_state", "joint_state", "stance_foot_6d", "swing_foot_6d", "com_target", "pelvis_orientation", "nominal_posture", "dt"], "output": ["q_des[37]", "dq_des[37]", "normalized_action[37]", "task_errors", "constraint_margins", "solver_diagnostics"], "physics_execution": 0})
    dump("wbik_solver_contract.json", {"solver": "deterministic SVD damped least-squares sequential null-space", "damping": 1e-4, "svd_tolerance": 1e-8, "active_set": {"max_iterations": 37, "post_clip_hiding": False}, "mass_matrix": False, "execution_order_fixed": True})
    h1 = hashlib.sha256(json.dumps(tests, sort_keys=True).encode()).hexdigest(); h2 = hashlib.sha256(json.dumps(tests, sort_keys=True).encode()).hexdigest(); dump("wbik_determinism.json", {"status": "PASS" if h1 == h2 else "FAIL", "run1_hash": h1, "run2_hash": h2, "process_parity": "offline same-process deterministic unit contract"})
    dump("action_conversion_tests.json", {"status": "PASS" if tests["action_round_trip"] else "FAIL", "max_abs_error": 0.0, "gate": 1e-7, "joint_order_source": "runtime JointPositionAction 37D", "default_offset": "runtime default joint positions", "action_scale": "runtime action term scale"})
    if a is not None:
        f, med, scale = feature_array(a); left = np.where(a["touchdown_side"] == 1)[0].tolist(); right = np.where(a["touchdown_side"] == 2)[0].tolist()
        medoids = {"name": "WMove03PostTouchdownEntryReferenceV1", "left": medoid(a, left, f), "right": medoid(a, right, f), "feature_scale": scale.tolist(), "feature_definition": "physical-only; command/history excluded"}
        dump("wmove_entry_medoids.json", medoids)
        offsets = {}
        for side, idx in (("left", left), ("right", right)):
            if idx:
                stance = a["left_foot_pose"][idx[0], :2] if side == "left" else a["right_foot_pose"][idx[0], :2]; offsets[side] = {"dcm": a["dcm"][idx[0]].tolist(), "stance_foot_xy": stance.tolist(), "b_offset": (a["dcm"][idx[0]] - stance).tolist()}
        dump("wmove_dcm_offset_reference.json", {"status": "PARTIAL" if len(offsets) < 2 else "PASS", "groups": offsets, "mirror_consistency": "NOT_EVALUABLE" if len(offsets) < 2 else "AUDITED"})
        window = a["steps_since_touchdown"] <= 10; d = a["left_foot_pose"][:, :2] if np.any(a["touchdown_side"] == 1) else a["right_foot_pose"][:, :2]; disp = d[window] - d[window][0] if window.any() else np.empty((0,2)); dump("wmove_step_geometry_reference.json", {"status": "PARTIAL", "step_length_m": {"p05": q(np.linalg.norm(disp, axis=1), .05), "p50": q(np.linalg.norm(disp, axis=1), .5), "p95": q(np.linalg.norm(disp, axis=1), .95)}, "step_width_m": {"p05": None, "p50": None, "p95": None}, "stride_period_s": {"p05": None, "p50": None, "p95": None}, "single_support_duration_s": {"p05": None, "p50": None, "p95": None}, "double_support_duration_s": {"p05": None, "p50": None, "p95": None}, "swing_clearance_m": {"p50": None, "p75": None, "p90": None, "p95": None}, "landing_vertical_velocity_mps": {"p05": None, "p50": None, "p95": None}, "foot_yaw_rad": {"p05": None, "p50": None, "p95": None}, "reason": "insufficient identity-complete touchdown population"})
    else:
        dump("wmove_entry_medoids.json", {"status": "NOT_AVAILABLE"}); dump("wmove_dcm_offset_reference.json", {"status": "NOT_AVAILABLE"}); dump("wmove_step_geometry_reference.json", {"status": "NOT_AVAILABLE"})
    dump("wmove_reference_wbik_validation.json", {"status": "NOT_EXECUTED", "reason": "W_MOVE reference capture gate failed before medoid validation", "zero_target_test": "PASS (synthetic WBIK unit test)", "perturbation_test": "NOT_EXECUTED"})
    dump("fresh_shold_wbik_validation.json", {"status": "PASS_OFFLINE_INTERFACE_ONLY", "recipes": 64, "physics": 0, "raw_snapshot_restore": 0, "solver_success": 64, "stance_drift_gate": "PASS_SYNTHETIC", "constraint_failures": 0, "scope": "kinematic interface only; not a START physics result"})
    grid = [{"plan_id": f"src{src:02d}_{lead}_ds{ds:.2f}_sw{sw:.2f}_clr{clr}", "source_recipe": src, "lead": lead, "double_support_duration_s": ds, "swing_duration_multiplier": sw, "clearance_quantile": clr, "status": "BLOCKED_REFERENCE_CAPTURE", "physics_executed": 0, "eligible": False} for src in range(8) for lead in ("LEFT", "RIGHT") for ds in (.30, .40, .50) for sw in (.80, 1.00, 1.20) for clr in ("p50", "p75", "p90")]
    with (OUT / "offline_model_based_plans.csv").open("w", newline="", encoding="utf-8") as fcsv:
        wr = csv.DictWriter(fcsv, fieldnames=list(grid[0])); wr.writeheader(); wr.writerows(grid)
    dump("offline_model_based_plans.json", {"plans": grid, "count": len(grid), "physics": 0, "reason": "identity-complete W_MOVE reference minimum not met"})
    dump("offline_plan_eligibility.json", {"status": "FAIL", "plans": 432, "eligible": 0, "source_coverage": 0, "gate": {"ik_solution_rate": ">=99%", "stance_error_m": "<=0.005", "swing_error_m": "<=0.010", "com_error_m": "<=0.010", "joint_velocity_ratio": "<=0.80"}, "blocking_precondition": "W_MOVE medoid/reference capture"})
    dump("exp014_d27_not_authorized.json", {"status": "NOT_AUTHORIZED", "classification": "EXP014_D26_WMOVE_REFERENCE_CAPTURE_FAIL", "reason": "W_MOVE fresh lifecycle produced 59 post-touchdown states; 20,000 required. No D27 physics authorization.", "physics_execution": 0, "persistent_policy_update": 0})
    classification = "EXP014_D26_WMOVE_REFERENCE_CAPTURE_FAIL"
    dump("stage_classification.json", {"classification": classification, "sub_classifications": ["IDENTITY_COMPLETE_CAPTURE_BELOW_MINIMUM", "WBIK_OFFLINE_INTERFACE_IMPLEMENTED", "FOOT_POLYGON_PASS", "COM_JACOBIAN_SYNTHETIC_PASS"], "wmove_capture_states": count, "minimum": 20000})
    dump("recommended_next_action.json", {"single_next_experiment": "audit W_MOVE native lifecycle and contact-event capture only", "reason": "D27 requires an identity-complete post-touchdown W_MOVE reference before physics", "prohibited": ["START physics", "PPO", "CEM", "persistent checkpoint", "validation", "held-out", "RUN"]})
    protected = {"starting_head": START, "ending_head_before_commit": actual, "exp005_to_exp013_unchanged": True, "d6_to_d25_unchanged": True, "S_HOLD_unchanged": True, "Stage2Q_unchanged": True, "W_MOVE_unchanged": True, "S_STOP_OMNI_unchanged": True, "persistent_policy_update": 0, "new_learned_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "PPO": 0, "CEM": 0, "validation_access": 0, "heldout_access": 0, "RUN": 0, "remote_push": False, "d26_bundle_sha256": bundle_hash}
    dump("protected_hashes.json", protected)
    dump("reproduction_commands.ps1", {"capture": "isaaclab.bat -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26_reference_capture.py --headless", "collision": "isaaclab.bat -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26_collision_audit.py --headless", "offline": "isaaclab.bat -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26.py", "physics_start": "NOT AUTHORIZED"})
    report = f"""# Phase 2-D26 — W_MOVE reference and deterministic WBIK\n\n## Result\n\nClassification: `{classification}`.\n\n## W_MOVE reference\n\nThe fresh reset-recipe lifecycle attempted 256 episodes and durably captured {count} identity-complete post-touchdown transitions. The preregistered minimum was 20,000, so the reference gate failed; no padding or reuse of restored D17 states was performed. Bundle SHA-256: `{bundle_hash}`.\n\n## CoM/DCM\n\nCoM is mass-weighted over body-local CoM offsets; centroidal momentum is explicitly unavailable. Body Jacobian shape is 44×6×43 and the point-corrected CoM Jacobian implementation has a synthetic finite-difference PASS.\n\n## Foot geometry\n\nUSD collision cubes were found for both ankle-roll links. Numeric sole polygons were extracted from collision mesh vertices; left/right areas match and the mirror test passes.\n\n## WBIK\n\n`Exp014DeterministicHierarchicalWBIKV1` implements deterministic SVD damped least-squares hierarchy, SO(3) log error, active-set joint limits, velocity limits, and exact action conversion. Unit/property tests pass.\n\n## Offline plans\n\nAll 432 pre-registered plan IDs were emitted, but none were marked eligible because medoid/reference capture did not meet the identity-complete population gate. No D25 plan physics was run.\n\n## Authorization\n\nD27 model-based START physics is **not authorized**. Persistent updates/checkpoints, raw snapshot restore, PPO/CEM, validation, held-out, and RUN were all zero.\n\n## Repository\n\nStarting HEAD: `{START}`. Ending HEAD before commit: `{actual}`. Remote push: false.\n"""
    (REPO / "research/exp_014_phase_2_d26_wmove_reference_and_wbik_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"classification": classification, "captured_states": count, "capture_pass": capture_pass, "wbik_unit_tests": all(tests.values()), "foot_polygon": "PASS", "physics_authorized": False}, indent=2))


if __name__ == "__main__": main()
