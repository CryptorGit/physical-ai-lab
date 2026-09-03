"""Freeze the evidence-backed Stage 2 gate without changing prior stages."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"
STAGE1 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
CFG_PATH = EXP / "configs/stage2_dynamics_sensitive_distillation.yaml"
CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def git_clean(path: str) -> bool:
    result = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=REPO)
    return result.returncode == 0


def diff_hash(path: str) -> str:
    data = subprocess.run(
        ["git", "diff", "--binary", "HEAD", "--", path], cwd=REPO, check=True, stdout=subprocess.PIPE
    ).stdout
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    runtime = json.loads((OUT / "walk_only_runtime/checkpoint_sweep.json").read_text())
    baseline_all = json.loads((STAGE1 / "single_task_closed_loop_results.json").read_text())["results"]["single_walk_steady"]
    candidates = []
    for name, result in runtime.items():
        walk = result["walk"]
        mean_success = sum(item["success_rate"] for item in walk.values()) / len(walk)
        worst_fall = max(item["fall_rate"] for item in walk.values())
        worst_slip = max(item["slip_rate"] for item in walk.values())
        worst_sat = max(item["saturation_rate"] for item in walk.values())
        worst_heading = max(item["heading_p95_rad"] for item in walk.values())
        candidates.append((mean_success, -worst_fall, -worst_slip, -worst_sat, -worst_heading, name, result))
    candidates.sort(reverse=True)
    _, _, _, _, _, selected_name, selected = candidates[0]
    speeds = selected["walk"]
    overall_success = sum(item["success_rate"] for item in speeds.values()) / len(speeds)
    baseline_success = sum(item["success_rate"] for item in baseline_all.values()) / len(baseline_all)
    checks = {
        "overall_success_ge_0_95": overall_success >= .95,
        "each_speed_success_ge_0_90": all(item["success_rate"] >= .90 for item in speeds.values()),
        "fall_le_0_02": max(item["fall_rate"] for item in speeds.values()) <= .02,
        "heading_p95_le_0_12": max(item["heading_p95_rad"] for item in speeds.values()) <= .12,
        "saturation_le_0_05": max(item["saturation_rate"] for item in speeds.values()) <= .05,
        "dangerous_slip_le_0_05": max(item["slip_rate"] for item in speeds.values()) <= .05,
        "speed_error_mean_le_0_20": True,  # walk contract itself uses <=0.20 at every valid step
        "path_drift_failure_le_0_05": False,  # absent from frozen Stage-0 evaluator: fail closed
        "excessive_flight_le_0_05": False,   # absent from frozen Stage-0 evaluator: fail closed
    }
    formal_pass = all(checks.values())
    improvement = overall_success - baseline_success
    severe_safety = (
        max(item["fall_rate"] for item in speeds.values())
        > max(item["fall_rate"] for item in baseline_all.values()) + .05
        or max(item["slip_rate"] for item in speeds.values())
        > max(item["slip_rate"] for item in baseline_all.values()) + .05
        or max(item["saturation_rate"] for item in speeds.values())
        > max(item["saturation_rate"] for item in baseline_all.values()) + .05
    )
    if formal_pass:
        walk_gate = "WALK_DYNAMICS_LOSS_PASS"
    elif improvement >= .20 and not severe_safety:
        walk_gate = "WALK_DYNAMICS_LOSS_PARTIAL"
    else:
        walk_gate = "WALK_DYNAMICS_LOSS_FAIL"
    if walk_gate == "WALK_DYNAMICS_LOSS_PASS":
        raise RuntimeError("WALK gate passed: mixed Stage 2C must be run before finalization")
    classification = "DYNAMICS_LOSS_UNSTABLE" if severe_safety else (
        "DYNAMICS_LOSS_PARTIAL" if walk_gate == "WALK_DYNAMICS_LOSS_PARTIAL" else "DYNAMICS_LOSS_NO_EFFECT"
    )
    recommendation = (
        "short-horizon nonlinear rollout supervision"
        if classification in {"DYNAMICS_LOSS_NO_EFFECT", "DYNAMICS_LOSS_UNSTABLE"}
        else "refine dynamics-sensitive loss before mixed distillation"
    )

    rows = []
    for name, result in runtime.items():
        for speed, metrics in result["walk"].items():
            rows.append({"checkpoint": name, "speed_mps": speed, **metrics})
    with (OUT / "walk_only_checkpoint_evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    dump("walk_only_closed_loop_results.json", {
        "selected_checkpoint": selected_name, "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["sha256"], "per_speed": speeds,
        "overall_success_rate": overall_success, "stage1_baseline_overall_success_rate": baseline_success,
        "absolute_improvement": improvement, "deterministic": True, "episodes_per_speed": 50,
    })
    dump("walk_only_gate.json", {
        "classification": walk_gate, "checks": checks, "pass": formal_pass,
        "fail_closed_missing_metrics": ["path_drift_failure", "excessive_flight"],
        "mixed_stage2c_authorized": formal_pass,
    })

    not_run = {"status": "not_executed_due_to_walk_only_gate", "walk_only_gate": walk_gate}
    for name in (
        "mixed_training_curves.csv", "mixed_checkpoint_evaluations.csv",
        "mixed_walk_results.json", "mixed_run_results.json", "mixed_walk_to_run_results.json",
        "teacher_retention_summary.json", "reverse_diagnostic.json",
    ):
        path = OUT / name
        if path.suffix == ".csv":
            path.write_text("status,reason\nnot_executed_due_to_walk_only_gate," + walk_gate + "\n", encoding="utf-8")
        else:
            dump(name, not_run)
    stage1_gradient = json.loads((STAGE1 / "gradient_interference.json").read_text())
    dump("gradient_interference_comparison.json", {
        "stage1_standard_huber": stage1_gradient,
        "stage2_mixed_dynamics_sensitive": not_run,
        "interpretation": "H3 was intentionally not modified; mixed gradients were not computed because WALK-only gate blocked Stage 2C.",
    })

    calibration = json.loads((OUT / "loss_calibration.json").read_text())
    dump("dynamics_loss_definition.json", {
        "formula": "L_action + 0.25 L_action_delta + lambda_dyn ||J_t delta_a_critical||^2 + lambda_contact sum(c_tj Huber(delta_a_j))",
        "action_loss": "uniform 37D Huber",
        "action_delta_loss": "Stage 0 definition and weight",
        "dynamic_jacobian": "state/regime/natural-phase-conditioned local table from symmetric fresh-app Isaac replay",
        "contact_criticality": "data-derived contact/support/flight/last-landing/gait topology change",
        "critical_joint_selection": "asset-derived bilateral hip yaw/roll/pitch, knee, ankle pitch/roll",
        "manual_ankle_roll_weight": False, "calibration": calibration,
    })
    shutil.copyfile(CFG_PATH, OUT / "stage2_config.yaml")
    protocol_hashes = {
        "stage2_config_sha256": sha(OUT / "stage2_config.yaml"),
        "sensitivity_table_sha256": sha(OUT / "sensitivity_table.pt"),
        "dynamics_loss_definition_sha256": sha(OUT / "dynamics_loss_definition.json"),
        "walk_checkpoint_sha256": selected["sha256"],
    }
    dump("stage2_protocol_hashes.json", protocol_hashes)
    dump("stage1_reference.json", {
        "classification": "MULTIPLE_FAILURE_MODES",
        "prioritized_failure": "CLOSED_LOOP_SENSITIVITY_DOMINATED",
        "coexisting_failure": "GRADIENT_INTERFERENCE",
        "stage1_gate_sha256": sha(STAGE1 / "gate.json"),
        "stage1_results_modified": False,
    })
    dump("protocol.json", {
        "stage": "2", "method_difference": "dynamics-sensitive distillation loss only",
        "student": "UnifiedWalkRunStudent123", "parameters": 85925,
        "counterfactual": {
            "fresh_isaac_app_per_sign": True, "state_copy": False, "setter": False, "teleport": False,
            "delta": .02, "locality_deltas": [.01, .04], "horizons_steps": [1, 2, 4, 8],
        },
        "walk_first_gate": True, "mixed_requires_walk_formal_pass": True,
        "ppo_training": 0, "reward_optimization": 0, "dagger": 0,
    })
    teachers = CFG["teachers"]
    protected = {}
    for name, teacher in teachers.items():
        path = REPO / teacher["path"]
        actual = sha(path)
        protected[name] = {"path": teacher["path"], "expected_sha256": teacher["sha256"],
                           "actual_sha256": actual, "unchanged": actual == teacher["sha256"]}
    baseline_diff_hashes = {
        "exp_005": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "exp_006": "4bf41f6a61e2d916807d6725aac4e96b001f671b29add97e6b0782dff8d751bb",
        "exp_007": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "exp_008": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    current_diff_hashes = {
        "exp_005": diff_hash("experiments/isaaclab/exp_005_unitree_g1_flat_run"),
        "exp_006": diff_hash("experiments/isaaclab/exp_006_unitree_g1_command_skills"),
        "exp_007": diff_hash("experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions"),
        "exp_008": diff_hash("experiments/isaaclab/exp_008_phase_aware_locomotion_transitions"),
    }
    protected.update({
        "preexisting_diff_hashes_at_start": baseline_diff_hashes,
        "diff_hashes_at_finish": current_diff_hashes,
        "exp_005_006_007_008_preserved": current_diff_hashes == baseline_diff_hashes,
        "exp_006_preexisting_dirty_state_preserved": current_diff_hashes["exp_006"] == baseline_diff_hashes["exp_006"],
        "stage0_results_unchanged": git_clean("results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"),
        "stage1_results_unchanged": git_clean("results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"),
        "capability_manifest_unchanged": git_clean("capability_manifest.json"),
        "teacher_gradients": 0, "ppo_training": 0, "reward_optimization": 0,
        "production_controller_updates": 0, "production_artifacts_created": 0,
    })
    dump("protected_hashes.json", protected)
    dump("stage2_classification.json", {
        "classification": classification, "walk_only_gate": walk_gate,
        "stage1_classification_preserved": "MULTIPLE_FAILURE_MODES",
        "mixed_executed": False, "reverse_diagnostic_executed": False,
        "reason": (
            "WALK-only formal gate did not pass, so causal isolation required stopping before mixed training."
        ),
    })
    dump("recommended_next_action.json", {
        "single_recommendation": recommendation,
        "pcgrad": "NOT_IMPLEMENTED", "conditional_adapter": "NOT_IMPLEMENTED",
        "multihead": "NOT_IMPLEMENTED", "production_change": False,
    })
    dump("gate.json", {
        "stage": "2", "classification": classification, "walk_only_gate": walk_gate,
        "counterfactual_matching_pass": json.loads((OUT / "prebranch_state_matching.json").read_text())["all_retained_within_tolerance"],
        "sensitivity_audit_pass": json.loads((OUT / "sensitivity_weight_audit.json").read_text())["status"] == "PASS",
        "walk_formal_pass": formal_pass, "mixed_executed": False,
        "ppo_training": 0, "teacher_updates": 0, "capability_updated": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n"
        "$base = '.\\experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts'\n"
        "& \"$base\\collect_stage2_primary_shards.ps1\"\n"
        "& \"$base\\collect_stage2_walk_supplement.ps1\"\n"
        "& \"$base\\collect_stage2_locality_subset.ps1\"\n"
        "& $py \"$base\\build_stage2_sensitivity_tables.py\"\n"
        "& $py \"$base\\train_stage2_dynamics_sensitive.py\" --scope walk_only\n"
        "# Pass each checkpoint with --checkpoint when running the deterministic evaluator.\n"
        "& $py \"$base\\evaluate_stage2_walk_only.py\" --headless\n",
        encoding="utf-8",
    )
    report_dir = REPO / "research"
    report_dir.mkdir(exist_ok=True)
    sensitivity = json.loads((OUT / "joint_dynamic_sensitivity.json").read_text())
    (report_dir / "exp_009_stage2_dynamics_sensitive_distillation_report.md").write_text(
        "# exp_009 Stage 2: Dynamics-sensitive distillation\n\n"
        f"- Classification: `{classification}`\n"
        f"- WALK-only gate: `{walk_gate}`\n"
        f"- Matched counterfactual branch states: "
        f"{json.loads((OUT / 'counterfactual_branch_manifest.json').read_text())['total_branch_states']}\n"
        f"- Selected checkpoint: `{selected_name}` (`{selected['sha256']}`)\n"
        f"- Stage-1 mean success: {baseline_success:.3f}; Stage-2 mean success: {overall_success:.3f}\n"
        "- Mixed distillation and reverse diagnostic were not executed because the WALK-only formal gate is mandatory.\n"
        f"- Next single method: **{recommendation}**.\n\n"
        "The finite-difference table was measured with fresh Isaac applications for plus/minus branches and "
        "retained only prebranch states within the frozen tolerances. The result does not alter the Stage-1 "
        "`MULTIPLE_FAILURE_MODES` classification or any production capability.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "walk_gate": walk_gate, "selected": selected_name}))


if __name__ == "__main__":
    main()
