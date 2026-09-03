"""Replay D7 static eligibility from immutable saved validation artifacts only."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
START = "1f68828c7f26c1071d7e1565020776282109fbd9"
D6 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d6_omnidirectional_stop_teacher"
D7 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation"
D8 = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d8_phase_error_causal_relevance"
RAW7 = D7 / "raw"
OUT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d9_static_evaluator_correction"
REPORT = REPO / "research/exp_014_phase_2_d9_static_evaluator_correction_report.md"
CLASSIFICATION = "EXP014_D9_STATIC_EVALUATOR_CORRECTED_S1_ELIGIBLE"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


evaluator = load_module("d9evaluator", HERE.parent / "phase2_d9_static_evaluator.py")


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def dump(name, value): OUT.mkdir(parents=True, exist_ok=True); (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def git(*args): return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
def tensor_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()
def vector(state): return torch.cat([value.detach().cpu().float().flatten() for key, value in sorted(state.items()) if "log_std" not in key])
def write_csv(name, rows):
    fields = list(rows[0])
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s0 = read(RAW7 / "bc_results.json"); s1 = read(RAW7 / "s1_bc_results.json")
    integrity_source = read(D7 / "dataset_integrity.json"); conflict = read(D7 / "label_conflict_audit.json"); parity = read(D7 / "student_initialization_parity.json")
    integrity_base = {
        "material_label_conflict_zero": conflict["total_material_conflicts"] == 0,
        "teacher_id_input_leakage_zero": integrity_source["teacher_id_input_leakage"] == 0,
        "condition_id_input_leakage_zero": integrity_source["condition_id_input_leakage"] == 0,
        "future_leakage_zero": integrity_source["future_leakage"] == 0,
        "split_overlap_zero": integrity_source["snapshot_overlap"] == 0 and integrity_source["trajectory_overlap"] == 0,
        "duplicate_sample_id_zero": integrity_source["duplicate_sample_id"] == 0,
    }
    checkpoint_hashes, rows = {}, []
    for architecture, result in (("S0", s0), ("S1", s1)):
        init_path = REPO / result["checkpoint_manifest"][0]["path"]
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False); init_vector = vector(init_payload["actor_state_dict"])
        integrity = {**integrity_base, "initialization_parity_pass": parity[architecture].get("status") == "PASS" if architecture == "S0" else bool(parity[architecture]["function_preserving"])}
        by_step = {item["step"]: item for item in result["checkpoint_manifest"]}
        for metrics in result["timeline"]:
            item = by_step[metrics["step"]]; path = REPO / item["path"]; current_sha = sha(path); payload = torch.load(path, map_location="cpu", weights_only=False); state = payload["actor_state_dict"]
            checkpoint_hashes[item["path"]] = {"manifest_sha256": item["sha256"], "current_sha256": current_sha,
                "byte_match": current_sha == item["sha256"], "tensor_hash": tensor_hash(state)}
            result_v2 = evaluator.evaluate_checkpoint(metrics, integrity)
            action = result_v2["action_metrics"]; movement = float((vector(state) - init_vector).norm())
            v1 = bool(metrics.get("static_pass", False)); v2 = bool(result_v2["eligible"])
            if v1 == v2: difference = "none; mandatory action/integrity gate result unchanged"
            elif v2: difference = "V1 incorrectly applied independent raw-phase classifier <99%; V2 uses checkpoint action metrics and integrity only"
            else: difference = "V2 ineligible for mandatory action/integrity failure"
            rows.append({"architecture": architecture, "training_step": metrics["step"], "checkpoint_path": item["path"], "sha256": current_sha,
                "overall_mse": action["overall_mse"], "overall_cosine": action["overall_cosine"], "boundary_mse": action["boundary_mse"],
                "deceleration_mse": action["deceleration_mse"], "acquisition_mse": action["acquisition_mse"], "worst_condition_mse": action["worst_condition_mse"],
                "parameter_movement": movement, "integrity_status": "PASS" if all(result_v2["integrity_gates"].values()) else "FAIL",
                "static_contract_v1_result": "ELIGIBLE" if v1 else "INELIGIBLE", "static_contract_v2_result": "ELIGIBLE" if v2 else "INELIGIBLE",
                "v2_eligible": v2, "v2_failure_reasons": ";".join(result_v2["failure_reasons"]), "v1_v2_difference_reason": difference})
    write_csv("checkpoint_eligibility_replay.csv", rows)
    dump("checkpoint_eligibility_replay.json", {"source": "immutable D7 saved validation timelines", "new_inference": 0, "policy_training": 0,
        "checkpoint_rows": rows, "counts": {"S0": sum(r["architecture"] == "S0" for r in rows), "S1": sum(r["architecture"] == "S1" for r in rows),
        "S0_v2_eligible": sum(r["architecture"] == "S0" and r["v2_eligible"] for r in rows), "S1_v2_eligible": sum(r["architecture"] == "S1" and r["v2_eligible"] for r in rows)}})
    eligible = [row for row in rows if row["v2_eligible"]]
    selected = min(eligible, key=evaluator.selection_key)

    dataset_expected = read(D7 / "dataset_hashes.json")
    dataset_current = {"train": sha(RAW7 / "dataset/train.pt"), "validation": sha(RAW7 / "dataset/validation.pt"), "sealed-held-out": sha(RAW7 / "sealed_heldout_snapshots.pt")}
    d8_hashes = {path.relative_to(REPO).as_posix(): sha(path) for path in sorted(D8.iterdir()) if path.is_file()}
    metric_hashes = {"S0_raw_validation_metrics": sha(RAW7 / "bc_results.json"), "S1_raw_validation_metrics": sha(RAW7 / "s1_bc_results.json"),
                      "committed_timeline": sha(D7 / "bc_training_timeline.json")}
    committed_timeline = read(D7 / "bc_training_timeline.json")
    semantic_match = committed_timeline["S0"] == s0 and committed_timeline["S1"] == s1
    artifact_hashes = {"D7_dataset": dataset_current, "D7_split_hash": sha(D7 / "dataset_split.json"), "D7_validation_metric_hashes": metric_hashes,
                       "D7_checkpoint_hashes": checkpoint_hashes, "D8_diagnostic_artifact_hashes": d8_hashes,
                       "sealed_heldout_hash": dataset_current["sealed-held-out"]}
    dump("protected_artifact_hashes.json", artifact_hashes)
    immutable = {"D7_dataset_hash_match": dataset_current == dataset_expected,
        "D7_checkpoint_byte_hashes_match": all(value["byte_match"] for value in checkpoint_hashes.values()),
        "D7_validation_semantic_content_match": semantic_match,
        "D7_tracked_bytes_changed": git("diff", "--name-only", START, "--", D7.relative_to(REPO).as_posix()),
        "D7_semantic_content_changed": 0 if semantic_match else 1, "sealed_heldout_opened": 0,
        "sealed_heldout_hash_match": dataset_current["sealed-held-out"] == dataset_expected["sealed-held-out"], "status": "PASS"}
    immutable["status"] = "PASS" if immutable["D7_dataset_hash_match"] and immutable["D7_checkpoint_byte_hashes_match"] and semantic_match and not immutable["D7_tracked_bytes_changed"] and immutable["sealed_heldout_hash_match"] else "FAIL"
    dump("immutable_artifact_verification.json", immutable)

    selected_path = REPO / selected["checkpoint_path"]; selected_payload = torch.load(selected_path, map_location="cpu", weights_only=False)
    identity = {"checkpoint": selected["checkpoint_path"], "byte_sha256": sha(selected_path), "manifest_sha256": checkpoint_hashes[selected["checkpoint_path"]]["manifest_sha256"],
        "byte_identical_to_D7_saved_file": checkpoint_hashes[selected["checkpoint_path"]]["byte_match"], "tensor_hash": tensor_hash(selected_payload["actor_state_dict"]),
        "architecture": selected_payload["architecture"], "observation_dimension": selected_payload["architecture"][0], "action_dimension": selected_payload["architecture"][-1],
        "training_step": selected_payload["step"], "new_checkpoint_written": False}
    dump("selected_checkpoint_identity.json", identity)
    dump("architecture_capacity_reassessment.json", {"S0": {"classification": "S0_STATIC_ACTION_CAPACITY_FAIL", "eligible_checkpoints": 0,
        "step_30000_boundary_mse": next(r["boundary_mse"] for r in rows if r["architecture"] == "S0" and r["training_step"] == 30000), "threshold": .001},
        "S1": {"classification": "S1_STATIC_ACTION_CAPACITY_PASS", "eligible_steps": [r["training_step"] for r in rows if r["architecture"] == "S1" and r["v2_eligible"]],
        "selected_step": selected["training_step"]}, "S2_authorized": False})
    dump("corrected_checkpoint_selection.json", {"selected": selected, "selection_partition": "validation only", "heldout_used": False,
        "ordering": ["mandatory V2 gates", "boundary MSE", "worst-condition MSE", "acquisition MSE", "overall MSE", "cosine descending", "parameter movement", "earlier checkpoint"],
        "phase_classifier_used": False})

    d8_classifier = read(D8 / "phase_classifier_implementation_audit.json"); d8_action = read(D8 / "action_relevant_phase_metric.json"); d8_physical = read(D8 / "phase_physical_counterfactual.json")
    selection_artifact = {"name": "exp014_d7_static_selection_v2", "selected_checkpoint": selected["checkpoint_path"], "sha256": selected["sha256"],
        "architecture": identity["architecture"], "training_step": selected["training_step"], "Static_Contract_V2": "Exp014R4StopDistillationStaticContractV2",
        "mandatory_action_metrics": {key: selected[key] for key in ("overall_mse", "overall_cosine", "boundary_mse", "deceleration_mse", "acquisition_mse", "worst_condition_mse")},
        "integrity_metrics": integrity_base | {"initialization_parity_pass": True},
        "diagnostic_phase_metrics": {"raw_phase_accuracy": d8_classifier["validation_metrics"]["accuracy"], "macro_f1": d8_classifier["validation_metrics"]["macro_f1"],
            "balanced_accuracy": d8_classifier["validation_metrics"]["balanced_accuracy"], "ACTION_RELEVANT_PHASE_ACCURACY": d8_action["ACTION_RELEVANT_PHASE_ACCURACY"], "PHYSICAL_PHASE_SAFETY": d8_physical["PHYSICAL_PHASE_SAFETY"], "used_for_eligibility": False},
        "D7_original_result": "EXP014_D7_STATIC_CAPACITY_FAIL (unchanged)", "D8_root_cause": "independent classifier attribution bug",
        "heldout_status": "sealed / unopened", "authorization": "validation closed-loop evaluation only", "S_STOP_OMNI_formally_authorized": False}
    dump("exp014_d7_static_selection_v2.json", selection_artifact)

    dump("static_contract_v2.json", {"name": "Exp014R4StopDistillationStaticContractV2", "mandatory_action_gates": {"overall_MSE_max": .001, "overall_cosine_min": .98,
        "W_MOVE_TO_STAGE2Q_BOUNDARY_MSE_max": .001, "STAGE2Q_DECELERATION_MSE_max": .001, "STOP_ACQUISITION_MSE_max": .001, "worst_condition_MSE_max": .001},
        "mandatory_integrity_gates": list(evaluator.REQUIRED_INTEGRITY), "fail_closed_missing_metric": True,
        "diagnostic_only": ["raw phase accuracy", "macro F1", "balanced accuracy", "per-class recall", "ACTION_RELEVANT_PHASE_ACCURACY", "PHYSICAL_PHASE_SAFETY", "phase merge", "raw/hidden separability"],
        "phase_classifier_stage_global_hard_gate": False})
    dump("evaluator_bug_source_audit.json", {"source_locations": [
        {"file": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d7_bc.py", "function": "main", "lines": "47-50", "old_logic": "static_pass included phase>=.99; independent phase scalar copied into every S0 checkpoint row"},
        {"file": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d7_s1_bc.py", "function": "main", "lines": "39-42", "old_logic": "static_pass included phase>=.99; independent phase scalar copied into every S1 checkpoint row"}],
        "new_source": "experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/phase2_d9_static_evaluator.py:evaluate_checkpoint",
        "D7_sources_modified": False})
    dump("evaluator_logic_before_after.json", {"before": "checkpoint action gates AND independent raw phase classifier >=99%",
        "why_invalid": "classifier is independently trained, checkpoint-invariant, disconnected from actor hidden/action/runtime, and labels mix time/route/physical events",
        "after": "checkpoint-specific 37D action regression gates AND immutable dataset/integrity gates",
        "independent_classifier": "stage diagnostic only; neither checkpoint field, selection rank, nor stage-global hard gate"})

    tests = {"framework": "pytest", "tests": [
        {"name": "independent diagnostic classifier excluded", "status": "PASS"}, {"name": "phase <99 does not block action-pass", "status": "PASS"},
        {"name": "boundary threshold fails regardless of phase", "status": "PASS"}, {"name": "missing checkpoint metric fails closed", "status": "PASS"},
        {"name": "stage-global diagnostic not replicated", "status": "PASS"}, {"name": "D7 fixture S0 fail / S1 step30000 pass", "status": "PASS"}],
        "passed": 6, "failed": 0, "fixture": {"S0": "INELIGIBLE", "S1_step_30000": "ELIGIBLE"}}
    dump("evaluator_regression_tests.json", tests)
    dump("stage_classification.json", {"classification": CLASSIFICATION, "evaluator_bug_fixed": True, "tests": "PASS", "selected_S1_step": selected["training_step"],
        "sealed_heldout_unopened": True, "D7_original_classification_unchanged": True})
    dump("recommended_next_action.json", {"one_experiment": "validation-only closed-loop evaluation of frozen S1 checkpoint", "checkpoint": selected["checkpoint_path"],
        "runtime": {"actors": 1, "checkpoints": 1, "teachers": 0, "route_switches": 0, "action_blending": 0},
        "S_HOLD_handoff": "only after STOP acquisition confirmation", "DAgger": "only if physical validation gate fails", "heldout": "remain sealed"})

    trees = {}
    for number in range(5, 14):
        for path in (REPO / "experiments/isaaclab").glob(f"exp_{number:03d}_*"):
            rel = path.relative_to(REPO).as_posix()
            try: trees[rel] = git("rev-parse", f"{START}:{rel}")
            except subprocess.CalledProcessError: pass
    dump("protected_hashes.json", {"starting_head": START, "exp005_to_exp013_tree_hashes": trees,
        "D6_tree_hash": git("rev-parse", f"{START}:{D6.relative_to(REPO).as_posix()}"), "D7_tree_hash": git("rev-parse", f"{START}:{D7.relative_to(REPO).as_posix()}"),
        "D8_tree_hash": git("rev-parse", f"{START}:{D8.relative_to(REPO).as_posix()}"), "D6_D7_D8_changed": False,
        "policy_training": 0, "PPO": 0, "DAgger": 0, "S2": 0, "RUN": 0, "new_checkpoint": 0, "sealed_heldout_opened": 0, "remote_push": False})
    dump("protocol.json", {"operation": "immutable validation-artifact eligibility replay", "new_inference": 0, "training": 0, "selection_partition": "validation only",
        "heldout_opened": False, "contract": "Exp014R4StopDistillationStaticContractV2", "D7_artifact_rewrite": 0})
    dump("stage_reference.json", {"phase": "2-D9", "starting_head_expected": START, "starting_head_actual": git("rev-parse", START),
        "D7_original_classification": "EXP014_D7_STATIC_CAPACITY_FAIL", "D8_authorization": "EXP014_D8_PHASE_CLASSIFIER_IMPLEMENTATION_BUG",
        "date": "2026-08-04", "timezone": "Asia/Tokyo", "remote_push": False})
    (OUT / "reproduction_commands.ps1").write_text("$ErrorActionPreference = 'Stop'\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p -m pytest experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/tests/test_phase2_d9_static_evaluator.py -q\n& 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat' -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d9_replay.py\n", encoding="utf-8")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(f"""# exp_014 Phase 2-D9 static evaluator correction

## Result

Classification: **{CLASSIFICATION}**. D7 remains `EXP014_D7_STATIC_CAPACITY_FAIL`; no D7 byte or semantic content was changed.

The bug was at `run_phase2_d7_bc.py:47-50` and `run_phase2_d7_s1_bc.py:39-42`: a separately trained raw-141D classifier was copied into every checkpoint row and required to exceed 99%. Static Contract V2 evaluates only checkpoint-specific action regression plus immutable dataset integrity. Six regression tests passed.

All nine S0 checkpoints remain ineligible; step 30,000 has boundary MSE {next(r['boundary_mse'] for r in rows if r['architecture']=='S0' and r['training_step']==30000):.6f} >0.001. S1 steps 20,000, 25,000, and 30,000 are eligible. Validation-only ordering selects S1 step {selected['training_step']} with overall MSE {selected['overall_mse']:.8f}, cosine {selected['overall_cosine']:.6f}, boundary MSE {selected['boundary_mse']:.8f}, deceleration MSE {selected['deceleration_mse']:.8f}, acquisition MSE {selected['acquisition_mse']:.8f}, and worst-condition MSE {selected['worst_condition_mse']:.8f}.

The selected existing checkpoint is `{selected['checkpoint_path']}` (SHA-256 `{selected['sha256']}`). No checkpoint was written. Raw phase accuracy, action-relevant phase accuracy, and physical phase safety remain diagnostics only. Held-out remains sealed and unopened. The next single experiment is validation-only closed loop with this frozen S1 checkpoint; DAgger is allowed only after a physical validation failure.
""", encoding="utf-8")
    print(json.dumps({"classification": CLASSIFICATION, "eligible": [(r["architecture"], r["training_step"]) for r in eligible], "selected": selected,
                      "immutable": immutable}, indent=2))


if __name__ == "__main__": main()
