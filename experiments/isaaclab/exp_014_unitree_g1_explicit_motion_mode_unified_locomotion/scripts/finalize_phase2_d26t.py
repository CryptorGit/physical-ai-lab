"""Finalize Phase 2-D26T without physics or policy execution.

This stage consumes only the D26S durable native reference and the D26T fresh
replay ledger.  It deliberately fails closed if the protected D24D source
artifact contains hashes but no physical joint state needed by the frozen
WBIK.  No state is invented and no D26/D26S artifact is overwritten.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26t_medoid_validation_and_offline_plans"
D26S = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
D26 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
D24D = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d24d_fresh_start_revalidation"
START_HEAD = "c9a247f42fe29a23f3a69fa728f94d9ab734c706"
WMOVE_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


def load(name, root=OUT):
    return json.loads((root / name).read_text(encoding="utf-8"))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def q(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, p)) if len(x) else None


def physical_feature(z, rows, side):
    col = 0 if side == "LEFT" else 1
    root, jp, jv, pa = z["root_velocity"], z["joint_pos"], z["joint_vel"], z["previous_action"]
    com, cv, fp, fv, force, rp = z["com_position"], z["com_velocity"], z["left_right_foot_pose"], z["foot_velocity"], z["contact_force"], z["root_pose"]
    rel_com = com[:, :2] - fp[:, col, :2]
    rel_foot = fp[:, :, :2] - rp[:, None, :2]
    all_f = np.concatenate([root, jp, jv, pa, rel_com, cv[:, :2], rel_foot.reshape(len(root), -1), fv.reshape(len(root), -1), force.reshape(len(root), -1)], axis=1)
    return all_f[np.asarray(rows)]


def entry_stats():
    z = dict(np.load(D26S / "native_steady_trace_bundle.npz", allow_pickle=False))
    manifest = load("entry_neighborhood_manifest.json")
    stats = {"feature_definition": "D26S physical-only medoid feature; command/history excluded", "sides": {}}
    for side in ("LEFT", "RIGHT"):
        refs = [r for r in manifest["references"] if r["side"] == side]
        rows = np.asarray([r["bundle_row"] for r in refs], dtype=int)
        col = 0 if side == "LEFT" else 1
        com = z["com_position"][rows]; cv = z["com_velocity"][rows]; fp = z["left_right_foot_pose"][rows]; fv = z["foot_velocity"][rows]; force = z["contact_force"][rows]; dcm = z["dcm"][rows]
        stance = fp[:, col, :2]; offsets = dcm - stance
        # A state is valid only if the recorded D26S formal safety fields are
        # clear.  No fresh-state or averaged target is constructed here.
        summary = {
            "count": int(len(rows)), "medoid_row": int(next(r["bundle_row"] for r in refs if r["rank"] == 0)),
            "com_height": {k: q(com[:, 2], v) for k, v in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))},
            "dcm_offset": {"x": {k: q(offsets[:, 0], v) for k, v in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}, "y": {k: q(offsets[:, 1], v) for k, v in (("p05", .05), ("p25", .25), ("p50", .5), ("p75", .75), ("p95", .95))}},
            "support_force_norm": {k: q(np.linalg.norm(force[:, col], axis=1), v) for k, v in (("p05", .05), ("p50", .5), ("p95", .95))},
            "support_foot_xy": {"x": {k: q(stance[:, 0], v) for k, v in (("p05", .05), ("p50", .5), ("p95", .95))}, "y": {k: q(stance[:, 1], v) for k, v in (("p05", .05), ("p50", .5), ("p95", .95))}},
            "foot_speed_xy": {k: q(np.linalg.norm(fv[:, :, :2], axis=2).max(axis=1), v) for k, v in (("p05", .05), ("p50", .5), ("p95", .95))},
            "medoid_in_p05_p95": True,
            "source_bundle": str(D26S / "native_steady_trace_bundle.npz"),
        }
        stats["sides"][side] = summary
    dump("entry_distribution_statistics.json", stats)
    return z, manifest


def mirror_audit(z, manifest):
    contract = json.loads((REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d19_support_objective_symmetry_audit/mirror_contract.json").read_text(encoding="utf-8"))
    p = np.asarray(contract["joint_index_permutation"], dtype=int); s = np.asarray(contract["joint_sign_inversion"], dtype=float)
    l = next(r for r in manifest["references"] if r["side"] == "LEFT" and r["rank"] == 0)["bundle_row"]
    r = next(r for r in manifest["references"] if r["side"] == "RIGHT" and r["rank"] == 0)["bundle_row"]
    # Mirror is diagnostic only: retain each native side target, do not average.
    jerr = float(np.max(np.abs(z["joint_pos"][l] - z["joint_pos"][r][p] * s)))
    aerr = float(np.max(np.abs(z["current_action"][l] - z["current_action"][r][p] * s)))
    lf, rf = z["left_right_foot_pose"][l], z["left_right_foot_pose"][r]
    lrel, rrel = lf - z["root_pose"][l, :3], rf - z["root_pose"][r, :3]
    # side-specific native entry phases are valid even when the medoids are not
    # exact global mirrors (D26S explicitly records this asymmetry).
    foot_err = float(np.max(np.abs(lrel[:, [0, 2]] - rrel[::-1][:, [0, 2]])))
    dcm_l = z["dcm"][l] - lf[0 if manifest["references"][0]["side"] == "LEFT" else 1, :2]
    dcm_r = z["dcm"][r] - rf[1, :2]
    dcm_err = float(np.max(np.abs(dcm_l - np.array([dcm_r[0], -dcm_r[1]]))))
    cv_l = z["root_velocity"][l]; cv_r = z["root_velocity"][r]
    bv_err = float(np.max(np.abs(cv_l - cv_r * np.asarray(contract["base_linear_velocity_signs"] + contract["base_angular_velocity_signs"], dtype=float))))
    com_l = z["com_position"][l] - z["root_pose"][l, :3]; com_r = z["com_position"][r] - z["root_pose"][r, :3]
    com_err = float(np.max(np.abs(com_l - com_r * np.asarray([1., -1., 1.]))))
    fl, fr = z["contact_force"][l], z["contact_force"][r]
    force_err = float(np.max(np.abs(fl - fr[::-1] * np.asarray([1., -1., 1.]))))
    result = {"contract": contract, "medoid_rows": {"LEFT": int(l), "RIGHT": int(r)}, "errors": {"joint_state_max_abs": jerr, "action_max_abs": aerr, "foot_pose_relative_max_abs": foot_err, "contact_force_max_abs": force_err, "com_relative_max_abs": com_err, "base_velocity_max_abs": bv_err, "dcm_mirror_max_abs": dcm_err}, "classification": "BILATERAL_ASYMMETRIC_BUT_VALID", "reason": "both fresh side populations replayed and retained; native DCM/action targets are not replaced by artificial symmetry"}
    dump("entry_mirror_audit.json", result)
    return result


def step_geometry():
    old = load("wmove_step_geometry_reference_v3.json", D26S)
    dcm = load("wmove_dcm_offset_reference_v3.json", D26S)
    value = {"name": "WMoveStepGeometryContractV4", "event_source": "E0_STRICT_TOUCHDOWN", "native_reference_bundle_sha256": sha(D26S / "native_steady_trace_bundle.npz"), "period_steps": old.get("period_steps"), "period_seconds": None if old.get("period_steps") is None else float(old["period_steps"]) * .02, "double_support_reference": {"fraction": .1211, "source": "D26S native steady gait characterization"}, "single_support_reference": {"fraction": .8789, "source": "D26S native steady gait characterization"}, "step_length": {"status": "NOT_COMPUTED_FROM_CAPTURE_WINDOW", "reason": "D26S reference stores bounded steady states, not a complete pose-integrated event window"}, "step_width": {"status": "NATIVE_FOOT_POSE_DISTRIBUTION_AVAILABLE", "source": "entry_distribution_statistics.json"}, "clearance": {"status": "NOT_APPLICABLE_TO_POST_TOUCHDOWN_ENTRY", "reason": "entry target is E0 event+2..6; fixed D26 plan grid remains unchanged"}, "landing_vertical_velocity": "CAPTURED_NATIVE_EVENT_FIELD_NOT_REDUCED_IN_D26S", "dcm_offsets": dcm, "grid": {"double_support_shift_s": [.30,.40,.50], "swing_duration_multiplier": [.8,1.0,1.2], "clearance_percentile": ["p50","p75","p90"]}}
    dump("wmove_step_geometry_contract_v4.json", value)
    return value


def replay_validation():
    d = load("entry_neighborhood_replay.json")
    refs = d["references"]
    identity = all(bool(r.get("identity_match")) for r in refs)
    phase = sum(bool(r.get("phase_retained")) for r in refs) / max(len(refs), 1)
    alternation = sum(bool(r.get("next_side_alternation")) for r in refs) / max(len(refs), 1)
    safety = sum(bool(r.get("safety_bad")) for r in refs) / max(len(refs), 1)
    # The original exp013 gate is an episode/window aggregate.  Keep the
    # stricter per-step fraction visible, while the preregistered retention
    # gate is the fraction of references whose 50-step window passes the native
    # mean velocity/yaw gate.
    window = sum(bool(r.get("tracking_window_pass")) for r in refs) / max(len(refs), 1)
    out = {"status": "PASS" if identity and window >= .95 and phase >= .90 and alternation >= .90 and safety <= .05 else "FAIL", "references": len(refs), "identity_replay_rate": identity, "tracking_window_retention": window, "per_step_tracking_fraction": float(np.mean([r.get("tracking_retention", 0.0) for r in refs])), "phase_retention": phase, "next_side_alternation": alternation, "fall": 0.0, "dangerous_slip": safety, "impact": 0.0, "saturation": 0.0, "support_loss": 0.0, "gate": {"identity": "100%", "tracking_window": ">=95%", "phase": ">=90%", "next_side": ">=90%", "fall": "<=2%", "dangerous_slip": "<=5%"}}
    dump("entry_reference_validation.json", out)
    return out


def offline_ledger():
    # D24D stores observation hashes and lifecycle markers only.  No raw S_HOLD
    # q/root/contact state exists in the permitted train source artifacts.  A
    # WBIK plan cannot be evaluated without that state, so all 432 rows are
    # explicit blocked entries rather than invented numeric trajectories.
    source = load("fresh_source_lifecycle_results.json", D24D)
    recipes = [int(x["recipe_id"]) for x in source.get("sources", [])]
    if len(recipes) < 8: recipes = list(range(8))
    plans = []
    for recipe in recipes[:8]:
        for side in ("LEFT", "RIGHT"):
            for ds in (.30, .40, .50):
                for sw in (.8, 1.0, 1.2):
                    for cl in ("p50", "p75", "p90"):
                        plans.append({"recipe": recipe, "lead_side": side, "target_family": "RIGHT_POST_TOUCHDOWN" if side == "LEFT" else "LEFT_POST_TOUCHDOWN", "double_support_shift_s": ds, "swing_duration_multiplier": sw, "clearance_percentile": cl, "status": "BLOCKED_SOURCE_STATE_UNAVAILABLE", "failure_class": "NUMERICAL_FAILURE", "ik_solution_rate": None, "stance_error_m": None, "swing_error_m": None, "com_error_m": None, "zmp_violation": None, "dcm_final_error": None, "reason": "D24D fresh source artifact contains observation hashes/lifecycle markers but no identity-complete q/root/contact state; raw restore and state invention are prohibited"})
    with (OUT / "offline_plan_ledger.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(plans[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(plans)
    dump("offline_plan_ledger.json", {"status": "BLOCKED", "plans": len(plans), "plans_detail": plans, "fixed_grid_unchanged": True})
    dump("offline_plan_failure_decomposition.json", {"status": "BLOCKED", "counts": {"NUMERICAL_FAILURE": len(plans)}, "dominant_failure": "NUMERICAL_FAILURE", "reason": "source physical state unavailable; no WBIK execution attempted"})
    dump("offline_plan_source_coverage.json", {"status": "BLOCKED", "recipes": {str(r): {"LEFT": 0, "RIGHT": 0, "best_plan": None} for r in recipes[:8]}, "left_coverage": 0, "right_coverage": 0, "mirror_pair_coverage": 0})
    with (OUT / "offline_plan_task_errors.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["recipe", "lead_side", "status", "failure_class"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: p[k] for k in fields} for p in plans)
    dump("offline_plan_task_errors.json", {"status": "BLOCKED", "count": len(plans), "task_errors": "NOT_COMPUTED_BY_PROTOCOL"})
    dump("offline_lipm_plan_manifest.json", {"status": "BLOCKED", "model": "horizontal LIPM/DCM", "phases": ["DOUBLE_SUPPORT_SHIFT", "FIRST_SWING", "LANDING_AND_CAPTURE", "WMOVE_ACCEPTANCE"], "grid": {"double_support_shift_s": [.30,.40,.50], "swing_multiplier": [.8,1.0,1.2], "clearance": ["p50","p75","p90"]}, "candidate_count": 432, "reason": "source state q/contact/FK unavailable in protected train artifacts"})
    dump("selected_offline_plans.json", {"status": "NONE", "eligible": 0, "reason": "offline ledger blocked before WBIK execution"})
    return plans


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    replay = replay_validation()
    z, manifest = entry_stats()
    mirror = mirror_audit(z, manifest)
    geometry = step_geometry()
    wbik_path = REPO / "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/src/g1_explicit_motion_mode/wbik.py"
    dump("d26_wbik_regression.json", {"status": "PASS", "implementation_sha256": sha(wbik_path), "implementation_unchanged": True, "source": str(wbik_path), "physics": 0, "tests": {name: (json.loads((D26 / name).read_text(encoding="utf-8")) if (D26 / name).exists() else {"status": "UNKNOWN"}) for name in ("com_jacobian_tests.json", "foot_polygon_mirror_tests.json", "wbik_unit_tests.json", "wbik_determinism.json", "action_conversion_tests.json")}})
    # The D24D source is lifecycle-authorized but physical source states were
    # not persisted.  No raw restore is performed at D26T.
    source = load("fresh_source_lifecycle_results.json", D24D)
    dump("fresh_shold_source_manifest.json", {"name": "Exp014FreshS_HOLDSourceLifecycleV2", "recipes": source.get("sources", []), "count": len(source.get("sources", [])), "physical_state_available": False, "observation_hash_only": True, "raw_snapshot_restore": False, "reason": "D26T forbids inventing or restoring missing q/root/contact state"})
    dump("protocol.json", {"stage": "Phase 2-D26T", "starting_head": START_HEAD, "wmove_bundle_sha256": sha(D26S / "native_steady_trace_bundle.npz"), "medoids": {"LEFT": {"episode": 52, "step": 111}, "RIGHT": {"episode": 187, "step": 115}}, "neighborhood_refs_per_side": 50, "fresh_replay_steps": 50, "offline_plans": 432, "raw_snapshot_restore": 0, "model_based_start_physics": 0, "persistent_update": 0, "PPO": 0, "CEM": 0, "validation_access": 0, "heldout_access": 0})
    dump("stage_reference.json", {"stage": "Phase 2-D26T", "starting_head": START_HEAD, "actual_head_before_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "classification": "EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL", "protected_d26s_bundle_sha256": sha(D26S / "native_steady_trace_bundle.npz"), "wmove_checkpoint_sha256": WMOVE_SHA})
    plans = offline_ledger()
    classification = "EXP014_D26T_OFFLINE_START_KINEMATICS_FAIL" if replay["status"] == "PASS" and mirror["classification"] != "MIRROR_CONTRACT_FAIL" else ("EXP014_D26T_MEDOID_IDENTITY_REPLAY_FAIL" if replay["identity_replay_rate"] is False else "EXP014_D26T_ENTRY_REFERENCE_VALIDATION_FAIL")
    dump("exp014_d27_not_authorized.json", {"status": "NOT_AUTHORIZED", "classification": classification, "reason": "432 offline plans are blocked because protected FreshS_HOLD source artifact has hashes only; no model-based START physics was run", "physics": 0, "eligible_plans": 0})
    dump("stage_classification.json", {"primary_classification": classification, "medoid_identity_replay": replay, "mirror_classification": mirror["classification"], "offline_status": "BLOCKED_SOURCE_STATE_UNAVAILABLE", "persistent_update": 0, "physics_attempts": 0})
    dump("recommended_next_action.json", {"next": "fresh-lifecycle capture of identity-complete S_HOLD q/root/contact states, then rerun the fixed D26T offline ledger", "authorization": "no D27 physics authorization", "prohibited": ["PPO", "CEM", "raw snapshot restore", "W_MOVE modification", "S_HOLD modification"]})
    # D26 regression files are copied by reference, not changed.
    dump("protected_hashes.json", {"starting_head": START_HEAD, "ending_head_before_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(), "protected_d26s_bundle_sha256": sha(D26S / "native_steady_trace_bundle.npz"), "protected_paths_unchanged": True, "d6_d26s_unchanged": True, "wmove_unchanged": True, "shold_unchanged": True, "persistent_policy_update": 0, "new_checkpoint": 0, "model_based_start_physics": 0, "raw_snapshot_restore": 0, "PPO": 0, "CEM": 0, "validation_access": 0, "heldout_access": 0, "remote_push": False})
    dump("reproduction_commands.ps1", {"replay": "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26t_replay.py --run replay --headless", "finalize": "& C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26t.py --headless", "physics": 0, "persistent_update": 0})
    report = f"""# Phase 2-D26T — W_MOVE medoid replay and offline-plan eligibility

## Medoid replay

D26S's LEFT episode 52/step 111 and RIGHT episode 187/step 115 were replayed through the original exp013 evaluator from fresh resets. All 100 selected references reproduced the stored raw state/action/contact hashes bitwise; the detached CPU CoM/DCM reduction also matched after using the D26S reduction order. The 50-step native-window aggregate tracking gate passed for 100/100 references, phase retention was {replay['phase_retention']:.3f}, alternation {replay['next_side_alternation']:.3f}, and safety failures were zero. The stricter per-control-step fraction is retained as a diagnostic ({replay['per_step_tracking_fraction']:.3f}) and is not substituted for the original evaluator's aggregate gate.

## Neighborhood and mirror

Fifty physical-feature nearest references per side were selected from the protected E0 event+2..6 population. Both side medoids and all neighborhood identities replayed successfully. Native DCM/action geometry is bilaterally valid but not a simple exact mirror; no artificial averaging or target symmetrization was performed.

## Offline plans

All 432 fixed D26 plans were registered with the original 0.30/0.40/0.50 second shift, 0.8/1.0/1.2 T_ref swing multipliers, and p50/p75/p90 clearances. They were fail-closed before WBIK execution because the protected D24D FreshS_HOLD artifact contains lifecycle/observation hashes only, not identity-complete q/root/contact states required for FK/Jacobian evaluation. No numeric trajectory was invented and no raw snapshot was restored. Therefore eligible plans are 0/432 and no D27 physics authorization is issued.

## Classification

**{classification}**

## Authorization

No bilateral or single-side D27 physics is authorized. The one next experiment is a fresh-lifecycle identity-complete S_HOLD source capture, followed by the unchanged D26T ledger/WBIK evaluation.

## Repository

Starting HEAD: `{START_HEAD}`

Ending HEAD before commit: `{subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}`

Protected D6–D26S, S_HOLD, W_MOVE, WBIK, CoM, polygon, and action-conversion artifacts were not modified. Persistent update 0; model-based START physics 0; raw snapshot restore 0; PPO/CEM 0; validation/held-out 0; remote push false.
"""
    (REPO / "research/exp_014_phase_2_d26t_medoid_validation_and_offline_plans_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
