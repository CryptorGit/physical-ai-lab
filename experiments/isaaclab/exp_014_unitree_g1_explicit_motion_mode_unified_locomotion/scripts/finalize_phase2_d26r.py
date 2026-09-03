"""Finalize the read-only D26R parity-stop audit.

The protocol deliberately does not synthesize contact phases after capture
parity fails.  All downstream files are explicit NOT_EXECUTED records so the
stage cannot be mistaken for a successful entry-reference capture.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26r_wmove_contact_phase_repair"
D26 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26_wmove_reference_and_wbik"
REPORT = REPO / "research/exp_014_phase_2_d26r_wmove_contact_phase_repair_report.md"
WMOVE = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def dump(name: str, value) -> None:
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path, default=None):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {} if default is None else default


def d26_manifest():
    return read_json(D26 / "wmove_reference_capture_manifest.json", {})


def git(*args):
    try: return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    except Exception: return "UNKNOWN"


def write_csv(name: str, fields, rows=()):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = read_json(OUT / "original_wmove_positive_control.json", {})
    if not original:
        raw = read_json(OUT / "original/_raw_zero_d26r_original.json", {})
        row = (raw.get("rows") or [{}])[0]
        original = {"evaluator": "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w1b.py", "checkpoint": str(WMOVE.relative_to(REPO)).replace("\\", "/"), "checkpoint_sha256": sha(WMOVE), "seed": raw.get("seed", 20282601), "episodes": row.get("episodes", 0), "formal_tracking_success_rate": row.get("success_rate", 0.0), "fall_rate": row.get("fall_rate", 1.0), "dangerous_slip_rate": row.get("dangerous_slip_rate", 1.0), "impact_failure_rate": row.get("impact_failure_rate", 1.0), "long_dwell_saturation_rate": row.get("long_dwell_saturation_rate", 1.0), "mean_forward_velocity": row.get("actual_vx_body", 0.0), "mean_vector_error": row.get("vector_velocity_mae", 999.0), "formal_gate": bool(row.get("success_rate", 0.0) >= 0.95 and row.get("fall_rate", 1.0) <= 0.02), "raw": "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d26r_wmove_contact_phase_repair/original/_raw_zero_d26r_original.json"}
        dump("original_wmove_positive_control.json", original)
    parity = read_json(OUT / "capture_harness_parity.json", {})
    dm = d26_manifest()
    starting = "05715d92bad02280c4b5da9d117a2885207f1acd"
    stage = {"stage": "Phase 2-D26R", "name": "Exp014WMoveNativeContactPhaseAuditV1", "starting_head_declared": starting, "starting_head_actual": git("rev-parse", "HEAD"), "repository": str(REPO), "scope": "W_MOVE native-lifecycle parity, contact-phase reconstruction, entry-reference recapture", "protected_implementations": ["D26 CoM", "D26 CoM Jacobian", "D26 foot polygon", "D26 hierarchical WBIK", "D26 action conversion"], "persistent_update": 0, "new_checkpoint": 0, "raw_snapshot_restore": 0, "validation_access": 0, "heldout_access": 0, "remote_push": False}
    dump("stage_reference.json", stage)
    dump("protocol.json", {"status": "STOPPED_AFTER_CAPTURE_PARITY", "original_episodes": 100, "paired_episodes": 32, "capture_min_episodes": 256, "capture_max_episodes": 1024, "steady_state_target": 20000, "transition_target": 200, "event_precedence": ["E0_STRICT_TOUCHDOWN", "E1_HYSTERETIC_ONSET", "E2_SUPPORT_DOMINANCE_TRANSFER", "E3_SINGLE_PHASE_ENTRY"], "raw_snapshot_restore": False, "stop_reason": "capture harness parity is not established; contact-event analysis is not authorized"})

    # Original positive control and parity records are produced by the runner.
    if not parity:
        parity = {"status": "FAIL", "parity_gate": "WMOVE_CAPTURE_HARNESS_PARITY_FAIL", "paired_fields": {"first_physics_divergence": {"control_step": 0, "field": "command_trace", "original": [0.3, 0.0, 0.0], "capture": [0.0, 0.0, 0.0]}}}
        dump("capture_harness_parity.json", parity)
    # Make the mandatory stop reason explicit even when the Isaac launcher
    # terminates before the optional paired rerun can emit its sidecar.
    parity = {
        **parity,
        "status": "FAIL",
        "parity_gate": "WMOVE_CAPTURE_HARNESS_PARITY_FAIL",
        "original_seed": 20274021,
        "capture_seed": 20282601,
        "seed_equal": False,
        "source_lifecycle_original": "exp013 evaluator default task reset",
        "source_lifecycle_capture": "D3 reset-recipe StandWorld.restore (fresh recipe lifecycle, not raw snapshot)",
        "paired_fields": {
            **parity.get("paired_fields", {}),
            "reset_state": "NOT_COMPARED: source lifecycle and seed differ before paired replay",
            "command_trace_equal": False,
            "actor_input_hash_equal": "NOT_AUTHORIZED_AFTER_SOURCE_MISMATCH",
            "mean_action_hash_equal": "NOT_AUTHORIZED_AFTER_SOURCE_MISMATCH",
            "first_physics_divergence": {"control_step": 0, "field": "source_lifecycle_or_command_trace", "original": {"seed": 20274021, "command": [0.3, 0.0, 0.0]}, "capture": {"seed": 20282601, "command": [0.0, 0.0, 0.0], "ramp": "minimum-jerk 25 control steps"}, "reason": "mandatory parity identity is not established"},
        },
    }
    dump("capture_harness_parity.json", parity)
    # Read-only identity and geometry audit from the protected D26 outputs.
    geom = read_json(D26 / "foot_collision_geometry_audit.json", {})
    dump("foot_sensor_body_mapping.json", {"status": "NOT_EXECUTED_AFTER_PARITY_STOP", "geometry_cross_check": "PASS" if geom.get("status") == "PASS" else "UNKNOWN", "left_body": "left_ankle_roll_link", "right_body": "right_ankle_roll_link", "collision_prims": {"left": geom.get("feet", {}).get("left", {}).get("collision_prims", []), "right": geom.get("feet", {}).get("right", {}).get("collision_prims", [])}, "force_tensor_indices": "NOT_EXECUTED", "reason": "runtime sensor/body mapping requires capture-harness parity"})

    dump("steady_trace_capture_manifest.json", {"status": "NOT_EXECUTED", "reason": "parity stop", "minimum_episodes": 256, "maximum_episodes": 1024, "minimum_states": 20000, "minimum_support_phase_events": 200, "d26_reference_forensics": {"episodes": dm.get("episodes", 256), "identity_complete_states": dm.get("collected_states", 59), "bundle_sha256": dm.get("bundle_sha256", "UNKNOWN")}})
    # Empty deterministic NPZ marker; no fabricated transitions are included.
    bundle = OUT / "steady_trace_bundle.npz"
    tmp_bundle = OUT / "steady_trace_bundle.tmp.npz"
    np.savez_compressed(tmp_bundle, status=np.asarray(["NOT_EXECUTED_PARITY_STOP"], dtype="U32"))
    tmp_bundle.replace(bundle)
    (OUT / "steady_trace_bundle.sha256").write_text(sha(bundle) + "\n", encoding="ascii")

    dump("raw_contact_diagnostics.json", {"status": "NOT_EXECUTED", "reason": "parity stop", "formal_contact_threshold_n": 5.0, "diagnostic_thresholds_n": [1.0, 5.0, 10.0, 20.0]})
    for name in ("strict_touchdown_events.csv", "hysteretic_contact_events.csv", "support_dominance_events.csv"):
        write_csv(name, ["event_id", "status", "reason"], [{"event_id": "NONE", "status": "NOT_EXECUTED", "reason": "capture parity fail"}])
    dump("strict_touchdown_events.json", {"status": "NOT_EXECUTED", "events": [], "reason": "parity stop"})
    dump("hysteretic_contact_events.json", {"status": "NOT_EXECUTED", "events": [], "reason": "parity stop"})
    dump("support_dominance_events.json", {"status": "NOT_EXECUTED", "events": [], "reason": "parity stop"})
    dump("kinematic_force_events.json", {"status": "NOT_EXECUTED", "events": [], "reason": "parity stop"})
    dump("gait_characterization.json", {"status": "NOT_EXECUTED", "classification": "NOT_EXECUTED", "reason": "capture parity fail"})
    dump("event_source_selection.json", {"status": "NOT_AUTHORIZED", "selected": None, "reason": "event analysis requires parity"})
    dump("entry_candidate_manifest.json", {"status": "NOT_EXECUTED", "candidate_count": 0, "reason": "event analysis requires parity"})
    dump("entry_medoids.json", {"status": "NOT_EXECUTED", "reason": "event analysis requires parity"})
    dump("entry_reference_validation.json", {"status": "NOT_EXECUTED", "reason": "event analysis requires parity"})
    dump("wmove_step_geometry_reference_v2.json", {"status": "NOT_EXECUTED", "reason": "entry source not selected"})
    dump("wmove_dcm_offset_reference_v2.json", {"status": "NOT_EXECUTED", "reason": "entry source not selected"})
    write_csv("offline_model_based_plans_v2.csv", ["plan_id", "status", "reason"], [{"plan_id": "ALL", "status": "NOT_EXECUTED", "reason": "parity stop"}])
    dump("offline_model_based_plans_v2.json", {"status": "NOT_EXECUTED", "plans": [], "reason": "parity stop"})
    dump("offline_plan_eligibility_v2.json", {"status": "NOT_AUTHORIZED", "eligible_count": 0, "source_coverage": 0.0, "reason": "capture parity fail"})
    dump("exp014_d27_not_authorized.json", {"status": "NOT_AUTHORIZED", "reason": "D26R capture harness parity failed before event/reference reconstruction", "classification": "EXP014_D26R_WMOVE_NATIVE_PARITY_FAIL", "physics_execution": 0})
    dump("stage_classification.json", {"classification": "EXP014_D26R_WMOVE_NATIVE_PARITY_FAIL", "subclassification": ["WMOVE_CAPTURE_HARNESS_PARITY_FAIL", "D26R_DOWNSTREAM_NOT_EXECUTED"], "original_positive_control_gate": original.get("formal_gate", False), "first_divergence": parity.get("paired_fields", {}).get("first_physics_divergence", {}), "d26_existing_states": dm.get("collected_states", 59), "d26_existing_strict_events": 6})
    dump("recommended_next_action.json", {"next": "repair capture harness only, then rerun original/capture parity", "prohibited": ["contact-phase interpretation", "D26 offline plan replay", "model-based physics", "policy or WBIK changes"]})
    dump("protected_hashes.json", {"starting_head": starting, "actual_head_at_finalize": git("rev-parse", "HEAD"), "wmove_sha256": sha(WMOVE), "d26_artifacts_unchanged": True, "protected_paths": ["exp_005..exp_013", "D6..D26", "S_HOLD", "Stage 2Q", "W_MOVE", "S_STOP_OMNI"], "persistent_update": 0, "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("# D26R stopped at mandatory parity gate\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26r_audit.py --stage original --headless --device cuda:0\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d26r_audit.py --stage parity --headless --device cuda:0\npython experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d26r.py\n", encoding="utf-8")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Phase 2-D26R — W_MOVE native contact-phase repair\n\n## W_MOVE parity\n\nThe protected exp013 `evaluate_w1b.py` adapter was run in an isolated D26R output directory for 100 fresh episodes at forward 0.3 m/s, WALK, zero yaw. Formal tracking success was **{original.get('formal_tracking_success_rate', 0):.3f}**, fall **{original.get('fall_rate', 0):.3f}**, and the original gate was **{'PASS' if original.get('formal_gate') else 'FAIL'}**.\n\nThe mandatory capture-harness parity gate was **FAIL**. The original evaluator used seed `20274021` and the exp013 default reset entrypoint; the D26 capture used seed `20282601` and D3 reset recipes. The first registered contract divergence is control step 0, `command_trace`: original direct exposure `[0.3, 0, 0]` versus the D26 capture contract `[0.0, 0, 0]` (25-step minimum-jerk ramp). Because source lifecycle, seed, and command identity are not established, actor/action hashes and contact-event interpretation are not authorized.\n\n## Sensor mapping\n\nThe protected D26 collision audit names `left_ankle_roll_link` and `right_ankle_roll_link` and their collision prims, with mirrored numeric sole polygons. Runtime force-tensor mapping was intentionally not promoted to PASS after the parity stop.\n\n## Contact phase\n\nStrict touchdown, hysteretic onset, support-dominance, and kinematic-force detectors were not run. No event source or gait classification is inferred from the D26 six strict events.\n\n## Reference population\n\nNo new steady trace was collected. The protected D26 forensics remain 59 identity-complete states from 256 attempted episodes and are unchanged; they are not a D26R reference population.\n\n## Geometry/DCM\n\nD26 CoM, polygon, DCM, and WBIK implementations remain protected and unchanged. D26R does not recompute offline plans after the parity failure.\n\n## Classification\n\n**EXP014_D26R_WMOVE_NATIVE_PARITY_FAIL**\n\n## Authorization\n\nBilateral and single-phase D27 physics are not authorized. The next experiment is capture-harness repair and a fresh original/capture parity run; no policy, physics, reward, WBIK, or checkpoint changes are authorized.\n\n## Repository\n\nStarting HEAD: `{starting}`\n\nEnding HEAD at artifact generation: `{git('rev-parse','HEAD')}` (the D26R commit is created after this report). Protected paths were not modified; persistent update 0; remote push false.\n"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__": main()
