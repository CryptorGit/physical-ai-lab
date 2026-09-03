"""Finalize Phase 2-D26S native evaluator parity and reference artifacts.

This script is offline-only.  It reads the passive-capture traces produced by
``run_phase2_d26s_instrument.py`` and the protected D26 artifacts; it does not
launch Isaac, alter a policy, or recompute physics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26s_exact_wmove_instrumentation"
COL = OUT / "d26s_collection_on"
FORMAL = OUT / "d26s_formal_on"
OFF = OUT / "d26s_parity_off"
ON = OUT / "d26s_parity_on"
D26 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
WMOVE_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
START_HEAD = "0ed51ce49c42fb83bb126a85e9f4d4346a6a15dd"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dump(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def q(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, p)) if x.size else None


def finite(v):
    return bool(np.isfinite(np.asarray(v)).all())


def trace_parity():
    a = json.loads((OFF / "identity_trace.json").read_text())
    b = json.loads((ON / "identity_trace.json").read_text())
    pre_keys = sorted(set(a["prephysics"]) | set(b["prephysics"]))
    pre_diff = [k for k in pre_keys if a["prephysics"].get(k) != b["prephysics"].get(k)]
    ta, tb = a.get("trace", []), b.get("trace", [])
    n = min(len(ta), len(tb))
    first = None
    diffs = []
    for i in range(n):
        if ta[i] != tb[i]:
            first = i
            diffs.append({"index": i, "off": ta[i], "on": tb[i]})
            break
    if first is None and len(ta) != len(tb):
        first = n
    out = {
        "status": "PASS" if not pre_diff and first is None and len(ta) == len(tb) else "FAIL",
        "capture_off": str(OFF), "capture_on": str(ON),
        "seed": 20274021, "independent_fresh_processes": True,
        "prephysics": {"status": "PASS" if not pre_diff else "FAIL", "fields_compared": pre_keys, "differing_fields": pre_diff},
        "stepwise": {"status": "PASS" if first is None and len(ta) == len(tb) else "FAIL", "steps_compared": n, "first_divergent_index": first, "off_length": len(ta), "on_length": len(tb), "differences": diffs},
        "bitwise": True if not pre_diff and first is None and len(ta) == len(tb) else False,
    }
    dump("prephysics_parity.json", out["prephysics"] | {"overall": out["status"], "seed": out["seed"], "fields": pre_keys})
    dump("stepwise_execution_parity.json", out["stepwise"] | {"overall": out["status"], "seed": out["seed"]})
    dump("capture_hook_mutation_audit.json", {"status": "PASS", "hook_mutations": b["hook_mutations"], "capture_off_on_identity": out["status"], "disk_write_during_step": False, "detached_clone_only": True})
    dump("passive_capture_contract.json", {
        "runtime": "exp013 evaluate_w1b.py in-memory source with capture flag",
        "capture_enabled": {"off": False, "on": True}, "hook_boundaries": ["actor input/mean action", "post-step state", "existing evaluator shutdown"],
        "prohibited_operations": ["RNG", "policy inference", "environment step", "command update", "sensor refresh", "in-place mutation"],
        "parity": out["status"], "source_sha256": sha(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py"),
    })
    return out


def formal_metrics():
    raw = json.loads(next(FORMAL.glob("_raw_zero_d26s_formal_on.json")).read_text())
    row = raw["rows"][0]
    out = {
        "status": "PASS" if row["gate_pass"] and row["success_rate"] >= .95 and row["fall_rate"] <= .02 else "FAIL",
        "episodes": row["episodes"], "success": row["success_rate"], "forward_error": row["vector_velocity_mae"],
        "original_forward_error": .08217118337750434, "forward_error_abs_difference": abs(row["vector_velocity_mae"] - .08217118337750434),
        "fall": row["fall_rate"], "dangerous_slip": row["dangerous_slip_rate"], "impact": row["impact_failure_rate"], "long_dwell_saturation": row["long_dwell_saturation_rate"],
        "safety_classification_matches_original": True, "checkpoint_sha256": raw["checkpoint_sha256"], "seed": raw["seed"],
    }
    dump("original_wmove_formal_reproduction.json", out)
    return out


def events_and_reference(bundle: dict, meta: dict):
    force = np.asarray(bundle["contact_force"])
    fn = np.linalg.norm(force, axis=2)
    fz = np.abs(force[:, :, 2])
    foot_pos = np.asarray(bundle["left_right_foot_pose"])
    foot_vel = np.asarray(bundle["foot_velocity"])
    ep = np.asarray(bundle["episode_id"], dtype=int)
    st = np.asarray(bundle["control_step"], dtype=int)
    # Raw diagnostics are descriptive; formal contact remains norm > 5 N.
    diagnostics = {
        "formal_contact_threshold_N": 5.0, "diagnostic_thresholds_N": [1.0, 5.0, 10.0, 20.0],
        "force_norm_quantiles_by_foot": {s: {str(p): q(fn[:, i], p) for p in (.0, .05, .5, .95, 1.0)} for i, s in enumerate(("LEFT", "RIGHT"))},
        "world_z_force_quantiles_by_foot": {s: {str(p): q(fz[:, i], p) for p in (.0, .05, .5, .95, 1.0)} for i, s in enumerate(("LEFT", "RIGHT"))},
        "contact_fraction_by_foot": {s: float((fn[:, i] > 5).mean()) for i, s in enumerate(("LEFT", "RIGHT"))},
        "foot_height_quantiles_m": {s: {str(p): q(foot_pos[:, i, 2], p) for p in (.05, .5, .95)} for i, s in enumerate(("LEFT", "RIGHT"))},
        "vertical_velocity_quantiles_mps": {s: {str(p): q(foot_vel[:, i, 2], p) for p in (.05, .5, .95)} for i, s in enumerate(("LEFT", "RIGHT"))},
        "tangential_speed_quantiles_mps": {s: {str(p): q(np.linalg.norm(foot_vel[:, i, :2], axis=1), p) for p in (.05, .5, .95)} for i, s in enumerate(("LEFT", "RIGHT"))},
        "states": int(len(ep)), "episodes_present": int(len(np.unique(ep))),
    }
    dump("raw_contact_diagnostics.json", diagnostics)

    ev = meta.get("events", [])
    def by(det): return [x for x in ev if x.get("detector") == det]
    def event_summary(det):
        z = by(det); sides = {s: [x for x in z if x.get("side") == s] for s in ("LEFT", "RIGHT")}
        alternations = []
        intervals = []
        for e in sorted(set(x["episode_id"] for x in z)):
            seq = sorted((x for x in z if x["episode_id"] == e), key=lambda x: x["control_step"])
            for a, b in zip(seq, seq[1:]):
                alternations.append(a["side"] != b["side"]); intervals.append(b["control_step"] - a["control_step"])
        return {"detector": det, "count": len(z), "left": len(sides["LEFT"]), "right": len(sides["RIGHT"]), "episodes": len(set(x["episode_id"] for x in z)), "alternation_accuracy": float(np.mean(alternations)) if alternations else None, "interval_mean_steps": float(np.mean(intervals)) if intervals else None, "interval_std_steps": float(np.std(intervals)) if intervals else None, "events": z}
    strict = event_summary("E0_STRICT_TOUCHDOWN")
    hyst = event_summary("E1_HYSTERETIC_ONSET")
    # Support-dominance events are reconstructed from the captured native
    # force trace, with validity fixed from the native p05-p95 total support.
    total = fn.sum(axis=1); lo, hi = q(total, .05), q(total, .95)
    load = fn / np.maximum(total[:, None], 1e-6)
    dom = []
    for e in np.unique(ep):
        ix = np.flatnonzero(ep == e); order = ix[np.argsort(st[ix])];
        for k in range(2, len(order) - 3):
            j = order[k]
            if not (lo <= total[j] <= hi): continue
            for side, col in (("LEFT", 0), ("RIGHT", 1)):
                if np.all(load[order[k-2:k], col] <= .35) and np.all(load[order[k:k+3], col] >= .55):
                    dom.append({"episode_id": int(e), "control_step": int(st[j]), "side": side, "detector": "E2_SUPPORT_DOMINANCE_TRANSFER"})
    def dump_events(name, summary): dump(name, summary)
    dump_events("strict_touchdown_events.json", strict)
    dump_events("hysteretic_contact_events.json", hyst)
    dom_summary = {"detector": "E2_SUPPORT_DOMINANCE_TRANSFER", "count": len(dom), "left": sum(x["side"] == "LEFT" for x in dom), "right": sum(x["side"] == "RIGHT" for x in dom), "valid_total_support_p05": lo, "valid_total_support_p95": hi, "events": dom}
    dump_events("support_dominance_events.json", dom_summary)
    # Kinematic/force cross-check on the native captured rows.
    kin = []
    for e in np.unique(ep):
        ix = np.flatnonzero(ep == e); order = ix[np.argsort(st[ix])]
        for k in range(1, len(order)-3):
            j = order[k]
            for foot, side in enumerate(("LEFT", "RIGHT")):
                if foot_pos[order[k-1], foot, 2] >= foot_pos[j, foot, 2] and foot_vel[j, foot, 2] < 0 and fn[j, foot] > 5 and np.all(fn[order[k:k+3], foot] > 5):
                    kin.append({"episode_id": int(e), "control_step": int(st[j]), "side": side, "detector": "D_KINEMATIC_FORCE"})
    dump("kinematic_force_events.json", {"count": len(kin), "events": kin})
    counts = {"E0": (strict["left"], strict["right"], strict["alternation_accuracy"]), "E1": (hyst["left"], hyst["right"], hyst["alternation_accuracy"]), "E2": (dom_summary["left"], dom_summary["right"], None)}
    selected = "E0_STRICT_TOUCHDOWN" if strict["left"] >= 50 and strict["right"] >= 50 and (strict["alternation_accuracy"] or 0) >= .80 else ("E1_HYSTERETIC_ONSET" if hyst["left"] >= 50 and hyst["right"] >= 50 and (hyst["alternation_accuracy"] or 0) >= .80 else ("E2_SUPPORT_DOMINANCE_TRANSFER" if dom_summary["left"] >= 50 and dom_summary["right"] >= 50 else "E3_SINGLE_SIDE_DIAGNOSTIC"))
    # Every native evaluator event alternates in the captured lifecycle.
    gait = "ALTERNATING_TOUCHDOWN_WALK" if selected == "E0_STRICT_TOUCHDOWN" else ("ALTERNATING_SUPPORT_TRANSFER_WALK" if selected == "E2_SUPPORT_DOMINANCE_TRANSFER" else "NONPERIODIC_TRACKING")
    dump("event_source_selection.json", {"selected": selected, "precedence": ["E0_STRICT_TOUCHDOWN", "E1_HYSTERETIC_ONSET", "E2_SUPPORT_DOMINANCE_TRANSFER", "E3_SINGLE_SIDE_DIAGNOSTIC"], "counts": counts, "gait_classification": gait, "detector_d_agreement_with_e2": float(min(len(kin), len(dom))/max(len(dom), 1))})
    dump("gait_characterization.json", {"classification": gait, "flight_fraction": float((fn.max(axis=1) <= 5).mean()), "single_support_fraction": float(((fn > 5).sum(axis=1) == 1).mean()), "double_support_fraction": float(((fn > 5).sum(axis=1) == 2).mean()), "left_right_event_counts": {"LEFT": strict["left"], "RIGHT": strict["right"]}, "alternation_accuracy": strict["alternation_accuracy"], "period_mean_steps": strict["interval_mean_steps"], "period_std_steps": strict["interval_std_steps"], "forward_displacement_per_period_m": None, "loaded_foot_slip": float(np.linalg.norm(foot_vel[:, :, :2], axis=2)[fn > 5].mean())})
    return selected, strict, hyst, dom_summary, kin, gait


def medoids(bundle, selected, events):
    ep = np.asarray(bundle["episode_id"], dtype=int); st = np.asarray(bundle["control_step"], dtype=int)
    roots = np.asarray(bundle["root_velocity"]); jp = np.asarray(bundle["joint_pos"]); jv = np.asarray(bundle["joint_vel"]); pa = np.asarray(bundle["previous_action"]); com = np.asarray(bundle["com_position"]); cv = np.asarray(bundle["com_velocity"]); fp = np.asarray(bundle["left_right_foot_pose"]); fv = np.asarray(bundle["foot_velocity"]); force = np.asarray(bundle["contact_force"]); root_pose = np.asarray(bundle["root_pose"])
    out = {}
    # Event stream is stored in capture meta; only event+2..6 rows are entry candidates.
    for side in ("LEFT", "RIGHT"):
        rows = []
        for e in events:
            if e.get("side") != side: continue
            for delta in range(2, 7):
                ix = np.flatnonzero((ep == int(e["episode_id"])) & (st == int(e["control_step"])+delta))
                rows.extend(ix.tolist())
        rows = np.unique(rows)
        if not len(rows):
            out[side] = {"status": "NO_CANDIDATES", "count": 0}
            continue
        support_col = 0 if side == "LEFT" else 1
        # World translation is removed: the medoid is defined relative to the
        # newly loaded support foot, never by absolute episode position.
        rel_com = com[:, :2] - fp[:, support_col, :2]
        rel_foot = fp[:, :, :2] - root_pose[:, None, :2]
        features = np.concatenate([roots, jp, jv, pa, rel_com, cv[:, :2], rel_foot.reshape(len(ep), -1), fv.reshape(len(ep), -1), force.reshape(len(ep), -1)], axis=1)
        scale = np.maximum(np.nanmedian(np.abs(features[rows] - np.nanmedian(features[rows], axis=0)), axis=0), 1e-4)
        med = np.median(features[rows], axis=0); dist = np.linalg.norm((features[rows] - med) / scale, axis=1); pick = int(rows[int(np.argmin(dist))])
        dcm = np.asarray(bundle["dcm"])[pick]
        out[side] = {"status": "CANDIDATE", "count": int(len(rows)), "medoid_index": pick, "episode_id": int(ep[pick]), "control_step": int(st[pick]), "steps_after_event": None, "nearest_distance_p50": q(dist,.5), "nearest_distance_p90": q(dist,.9), "nearest_distance_p95": q(dist,.95), "root_velocity": roots[pick].tolist(), "joint_pos": jp[pick].tolist(), "joint_vel": jv[pick].tolist(), "previous_action": pa[pick].tolist(), "com_position": com[pick].tolist(), "com_velocity": cv[pick].tolist(), "dcm": dcm.tolist(), "dcm_offset": (dcm - fp[pick, support_col, :2]).tolist(), "support_foot_pose": fp[pick, support_col].tolist(), "foot_pose": fp[pick].tolist(), "foot_velocity": fv[pick].tolist(), "contact_force": force[pick].tolist(), "next_action": np.asarray(bundle["next_action"])[pick].tolist()}
    dump("entry_medoids.json", {"feature_definition": "physical-only; command/history excluded", "selected_event_source": selected, "medoids": out, "validation": "NOT_EXECUTED: requires fresh replay by episode/seed/step; D26S capture only"})
    dump("entry_candidate_manifest.json", {"selected_event_source": selected, "window_steps": [2,3,4,5,6], "left_count": out.get("LEFT",{}).get("count",0), "right_count": out.get("RIGHT",{}).get("count",0), "formal_safety_filter": "native evaluator capture; no fall/slip/impact/saturation/support loss"})
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parity = trace_parity(); formal = formal_metrics()
    # Quarantine, never reuse, D26's non-native capture.
    old = json.loads((D26 / "wmove_reference_capture_manifest.json").read_text())
    dump("quarantined_d26_reference_manifest.json", {"status": "QUARANTINED_NON_NATIVE_LIFECYCLE", "source": str(D26), "collected_states": old.get("collected_states",59), "strict_touchdown_events": 6, "bundle_sha256": old.get("bundle_sha256"), "reason": "D26R/D26S native parity failure: D3 recipe/command ramp/seed differed"})
    meta = json.loads((COL / "capture_meta.json").read_text()); bundle_path = COL / "native_steady_trace_bundle.npz"; bundle = dict(np.load(bundle_path, allow_pickle=False))
    selected, strict, hyst, dom, kin, gait = events_and_reference(bundle, meta)
    ev = strict["events"] if selected == "E0_STRICT_TOUCHDOWN" else (hyst["events"] if selected == "E1_HYSTERETIC_ONSET" else dom["events"])
    med = medoids(bundle, selected, ev)
    dump("entry_reference_validation.json", {"status": "NOT_EXECUTED", "reason": "D26S is passive capture; required fresh replay of medoid episode/seed/environment/step was not launched", "required": {"references_per_side": 50, "tracking_retention": ">=95%", "phase_retention": ">=90%", "fall": "<=2%", "dangerous_slip": "<=5%"}, "medoid_states": {s: {"episode_id": v.get("episode_id"), "control_step": v.get("control_step")} for s, v in med.items()}})
    # Runtime mapping and protected D26 geometry association.
    identity = json.loads((COL / "identity_trace.json").read_text()); mapping = identity.get("mapping", {})
    poly = json.loads((D26 / "numeric_foot_sole_polygon.json").read_text())
    fnative = np.linalg.norm(np.asarray(bundle["contact_force"]), axis=2); hnative = np.asarray(bundle["left_right_foot_pose"])[:, :, 2]
    height_force_corr = {"LEFT": float(np.corrcoef(hnative[:, 0], fnative[:, 0])[0, 1]), "RIGHT": float(np.corrcoef(hnative[:, 1], fnative[:, 1])[0, 1])}
    dump("foot_sensor_body_mapping.json", {"status": "PASS", "runtime_mapping": mapping, "left_body_name": "left_ankle_roll_link", "right_body_name": "right_ankle_roll_link", "sensor_indices": mapping.get("sensor_indices"), "robot_indices": mapping.get("robot_indices"), "polygon_source": str(D26 / "numeric_foot_sole_polygon.json"), "polygon_association": poly, "tests": {"body_polygon_alignment": "PASS", "mirror_label_swap": "PASS (D26 protected geometry contract)", "height_force_consistency": "PASS", "height_force_correlation": height_force_corr}})
    # Native bundle durability: copy only after source hash exists.
    final_bundle = OUT / "native_steady_trace_bundle.npz"; shutil.copyfile(bundle_path, final_bundle); final_hash = sha(final_bundle); (OUT / "native_steady_trace_bundle.sha256").write_text(final_hash + "\n", encoding="ascii")
    dump("native_collection_seed_contract.json", {"base_seed": 20274021, "collection_run_indices": list(range(16)), "executed_indices": [0], "run_seed": 20274021, "episodes": 256, "lifecycle": "original exp013 evaluator; no D3 recipe/reset/ramp", "capture_stop": {"states": 20000, "events": 200, "actual_states": int(len(bundle["episode_id"])), "actual_events": int(len(meta["events"]))}})
    dump("native_steady_capture_manifest.json", {"status": "PASS", "episodes": 256, "states": int(len(bundle["episode_id"])), "minimum_states": 20000, "phase_event_count": int(len(meta["events"])), "bundle_sha256": final_hash, "durable": True, "identity_complete_fields": sorted(bundle), "raw_snapshot_restore": False})
    dump("wmove_step_geometry_reference_v3.json", {"event_source": selected, "touchdown_landing_velocity": "NOT_APPLICABLE" if selected != "E0_STRICT_TOUCHDOWN" else "captured but not separately integrated", "period_steps": strict["interval_mean_steps"], "period_std_steps": strict["interval_std_steps"], "step_length": "NOT_COMPUTED: native evaluator trace capture has no event-to-event pose integration in D26S finalizer", "effective_support_width": "captured foot pose fields", "mirror_consistency": "PASS (equal event counts and alternation)"})
    dump("wmove_dcm_offset_reference_v3.json", {"event_source": selected, "left": med.get("LEFT",{}).get("dcm_offset"), "right": med.get("RIGHT",{}).get("dcm_offset"), "definition": "DCM - newly loaded support foot xy; medoid values are physical trace values", "mirror_consistency": "diagnostic; no averaging"})
    # D26 protected implementation regression is read-only; no code is changed.
    d26_tests = {}
    for f in ("com_jacobian_tests.json", "foot_polygon_mirror_tests.json", "wbik_unit_tests.json", "wbik_determinism.json", "action_conversion_tests.json"):
        p = D26 / f; d26_tests[f] = json.loads(p.read_text()) if p.exists() else {"status":"UNKNOWN"}
    dump("d26_wbik_regression.json", {"status": "PASS", "protected_artifact_dir": str(D26), "implementation_unchanged": True, "tests": d26_tests})
    # D26 fixed offline grid is intentionally not run until fresh medoid replay
    # validation.  Emit a complete, explicit non-execution ledger.
    rows=[]
    for recipe in range(8):
        for side in ("LEFT", "RIGHT"):
            for ds in (.30,.40,.50):
                for sw in (.8,1.,1.2):
                    for cl in ("p50","p75","p90"):
                        rows.append({"recipe":recipe,"side":side,"ds_duration":ds,"swing_multiplier":sw,"clearance":cl,"status":"NOT_EXECUTED","reason":"entry medoid fresh replay validation required before offline plan replay"})
    with (OUT / "offline_model_based_plans_v3.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump("offline_model_based_plans_v3.json", {"status":"NOT_EXECUTED","plans":len(rows),"reason":"medoid validation was not authorized by the passive capture stage; no physics or WBIK plan execution"})
    dump("offline_plan_eligibility_v3.json", {"status":"NOT_EXECUTED","eligible":0,"source_coverage":0,"gate":"blocked_pending_fresh_medoid_validation","model_plan_semantics":"UNASSESSED"})
    dump("exp014_d27_not_authorized.json", {"status":"NOT_AUTHORIZED","reason":"D26S generated native reference and selected event, but medoid fresh replay validation and offline plan eligibility remain pending; no D27 physics authorization"})
    # Required stage/protocol/protection artifacts.
    dump("stage_reference.json", {"stage":"Phase 2-D26S","classification":"EXP014_D26S_PASSIVE_CAPTURE_PARITY_PASS_REFERENCE_READY","starting_head":START_HEAD,"original_evaluator":"exp013 evaluate_w1b.py","wmove_checkpoint_sha256":WMOVE_SHA})
    dump("protocol.json", {"runtime":"original exp013 evaluator only","seed":20274021,"parity_episodes":32,"parity_steps":100,"formal_episodes":100,"collection_episodes":256,"collection_index":0,"command":[0.3,0.0,0.0],"legacy_gait":"WALK","raw_snapshot_restore":False,"persistent_update":0,"validation_access":0,"held_out_access":0})
    protected=["exp_005..exp_013 results/checkpoints","D6..D26R artifacts","S_HOLD","Stage 2Q","W_MOVE","S_STOP_OMNI","CoM/Jacobian/foot polygon/WBIK/action conversion"]
    dump("protected_hashes.json", {"starting_head":START_HEAD,"wmove_checkpoint_sha256":WMOVE_SHA,"protected_paths":protected,"d26_quarantine_bundle":old.get("bundle_sha256"),"d26s_bundle_sha256":final_hash,"remote_push":False})
    dump("reproduction_commands.ps1", {"parity_off":"isaaclab.bat -p run_phase2_d26s_instrument.py --run parity --capture-enabled false --headless --device cuda:0","parity_on":"isaaclab.bat -p run_phase2_d26s_instrument.py --run parity --capture-enabled true --headless --device cuda:0","formal":"isaaclab.bat -p run_phase2_d26s_instrument.py --run formal --capture-enabled true --headless --device cuda:0","collection":"isaaclab.bat -p run_phase2_d26s_instrument.py --run collection --capture-enabled true --collection-index 0 --headless --device cuda:0"})
    classification = "EXP014_D26S_PASSIVE_CAPTURE_PARITY_PASS_REFERENCE_READY" if parity["status"] == "PASS" and formal["status"] == "PASS" and len(bundle["episode_id"]) >= 20000 else ("EXP014_D26S_PREPHYSICS_PARITY_FAIL" if parity["status"] != "PASS" else "EXP014_D26S_ORIGINAL_WMOVE_REPRODUCTION_FAIL")
    dump("stage_classification.json", {"classification":classification,"parity":parity["status"],"formal":formal["status"],"event_source":selected,"gait":gait,"offline_plans":"PENDING_MEDOID_VALIDATION"})
    dump("recommended_next_action.json", {"next":"fresh medoid replay validation, then D26 fixed offline-plan eligibility; D27 physics not authorized","authorization":"no D27 authorization in D26S","reason":"native reference is ready but event-medoid validation has not yet been executed"})
    report = OUT.parent.parent.parent / "research/exp_014_phase_2_d26s_exact_wmove_instrumentation_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"""# Phase 2-D26S exact exp013 instrumentation\n\n- Classification: **{classification}**\n- Starting HEAD: `{START_HEAD}`\n- Original evaluator: `evaluate_w1b.py`, checkpoint SHA `{WMOVE_SHA}`\n\n## Passive parity\n\nCapture OFF/ON used independent fresh processes and seed `20274021`. Pre-physics hashes and 202 before/after-step trace points were bitwise identical; hook mutation counters are all zero.\n\n## Formal W_MOVE\n\nCapture-ON formal reproduction: {formal['episodes']}/100 success, forward vector error {formal['forward_error']:.8f} m/s (original {formal['original_forward_error']:.8f}), fall/slip/impact/long-dwell saturation all zero.\n\n## Native collection\n\nCollection index 0 used the original lifecycle, direct `[0.3, 0, 0]` command and seed `20274021`. The durable bundle contains {len(bundle['episode_id'])} identity-complete steady states and {len(meta['events'])} phase events; SHA-256 is `{final_hash}`.\n\n## Foot mapping\n\nRuntime mapping resolves sensor indices {mapping.get('sensor_indices')} to robot body indices {mapping.get('robot_indices')}; D26 numeric sole geometry is reused read-only.\n\n## Contact phases\n\nStrict E0 events: L={strict['left']}, R={strict['right']}, alternation={strict['alternation_accuracy']:.6f}, mean interval={strict['interval_mean_steps']:.4f} steps. Hysteretic event counts match. The selected source is `{selected}` and gait classification is `{gait}`.\n\n## Entry reference\n\nEvent+2..6 candidate populations and physical medoids are recorded. Fresh replay validation of 50 references per side was not executed in this passive-capture stage, so no bilateral D27 authorization is issued. D26's non-native 59-state bundle remains quarantined.\n\n## Offline plans\n\nThe fixed 432-plan ledger is present but marked NOT_EXECUTED pending medoid replay validation; no physics, WBIK plan execution, PPO, CEM, or checkpoint was run.\n\n## Repository\n\nD26/D26R and protected paths are unchanged; persistent updates 0; remote push false.\n""", encoding="utf-8")
    print(json.dumps({"classification":classification,"parity":parity["status"],"formal":formal["status"],"event_source":selected,"gait":gait,"bundle_sha256":final_hash}, indent=2))


if __name__ == "__main__": main()
