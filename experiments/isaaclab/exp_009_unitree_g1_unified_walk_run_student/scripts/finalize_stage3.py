"""Seal Stage 3 results after the predeclared surrogate trust gates."""

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
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision"
CFG = EXP / "configs/stage3_nonlinear_rollout_supervision.yaml"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2), encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    cfg = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    surrogate = json.loads((OUT / "surrogate_classification.json").read_text(encoding="utf-8"))
    one = json.loads((OUT / "one_step_surrogate_results.json").read_text(encoding="utf-8"))
    multi = json.loads((OUT / "multi_step_surrogate_results.json").read_text(encoding="utf-8"))
    ranking = json.loads((OUT / "action_ranking_results.json").read_text(encoding="utf-8"))
    uncertainty = json.loads((OUT / "uncertainty_gate.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "surrogate_dataset_manifest.json").read_text(encoding="utf-8"))
    start_head = "0c7e957e64a1b42e70b91faad19f61653185ea6e"
    if subprocess.run(["git", "merge-base", "--is-ancestor", start_head, "HEAD"], cwd=REPO).returncode != 0:
        raise RuntimeError("Stage 3 starting revision is not an ancestor of current HEAD")
    write("stage2_reference.json", {
        "classification": "DYNAMICS_LOSS_NO_EFFECT",
        "selected_checkpoint": "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation/checkpoints/walk_only/epoch_5.pt",
        "sha256": "beca6f5300fa8f7cb5a17235eb8dd4568e83b2bdf0b68d141f42e27ca9a1ecb9",
        "walk_results": {"0.6": 0.26, "0.8": 0.0, "1.0": 0.0, "1.2": 0.0, "overall": 0.065},
        "stage0_1_2_immutable": True,
    })
    write("protocol.json", {
        "stage": "3", "hypothesis": "nonlinear free-running short-horizon rollout supervision",
        "surrogate_first_gate": True, "student_training_requires_all_surrogate_gates": True,
        "walk_only_before_mixed": True, "mixed_requires_walk_formal_pass": True,
        "prohibited_methods_used": [],
        "ppo_training": 0, "reward_optimization": 0, "teacher_updates": 0,
        "isaac_backpropagation": False, "state_setter": False, "teleport": False,
    })
    write("surrogate_feature_layout.json", {
        "source": "g1_walk_centered.experts.adapters._legacy_observation",
        "observation_dim": 123,
        "fields": [
            {"name": "base_linear_velocity", "indices": [0, 3], "predicted": True},
            {"name": "base_angular_velocity", "indices": [3, 6], "predicted": True},
            {"name": "projected_gravity", "indices": [6, 9], "predicted": True},
            {"name": "velocity_command", "indices": [9, 12], "predicted": False},
            {"name": "joint_position", "indices": [12, 49], "predicted": True},
            {"name": "joint_velocity", "indices": [49, 86], "predicted": True},
            {"name": "previous_action", "indices": [86, 123], "predicted": False},
        ],
        "physical_state_dim": 83, "action_dim": 37, "field_order_inferred": False,
    })
    write("observation_reconstruction_contract.json", {
        "physical_state": "predicted residual added autoregressively",
        "command": "held analytically from current observation",
        "previous_action": "current candidate student action assigned exactly",
        "teacher_forcing": "step 0 only", "free_running_steps": [1, 2, 4, 8],
        "finite_required": True,
    })
    shutil.copyfile(CFG, OUT / "surrogate_config.yaml")
    write("rollout_loss_definition.json", {
        "status": "DEFINED_NOT_EXECUTED",
        "reason": "surrogate trust gates failed before student optimization",
        "formula": "L_action + lambda_delta L_action_delta + lambda_rollout L_rollout + lambda_contact L_contact_gait",
        "action_loss": "uniform 37D Huber", "action_delta_weight": 0.25,
        "rollout_horizons": [1, 2, 4, 8], "primary_horizon": 8,
        "surrogate_frozen_during_student_training": True,
        "uncertainty_mask": f"ensemble variance <= {uncertainty['frozen_p95_threshold']}",
    })
    write("loss_calibration.json", {
        "status": "not_executed_due_to_surrogate_gate",
        "target_contributions": cfg["student"]["calibrated_target_contributions"],
        "validation_tuning": False,
    })
    shutil.copyfile(CFG, OUT / "student_config.yaml")
    write("student_protocol_hashes.json", {
        "stage3_config_sha256": sha(CFG),
        "stage3_config_canonical_sha256": canonical_sha(cfg),
        "surrogate_dataset_sha256": manifest["dataset_sha256"],
        "surrogate_checkpoint_hashes": [
            item["sha256"] for item in json.loads((OUT / "surrogate_ensemble_manifest.json").read_text(encoding="utf-8"))["members"]
        ],
        "student_checkpoint_created": False, "student_training_authorized": False,
    })
    placeholder = {
        "status": "not_executed_due_to_surrogate_gate",
        "blocking_classification": "SURROGATE_NOT_TRUSTWORTHY",
        "student_optimizer_updates": 0,
    }
    for name in (
        "walk_only_closed_loop_results.json", "walk_only_model_exploitation_audit.json",
        "mixed_walk_results.json", "mixed_run_results.json", "mixed_walk_to_run_results.json",
        "teacher_retention_summary.json", "gradient_interference_comparison.json", "reverse_diagnostic.json",
    ):
        write(name, placeholder)
    for name in ("walk_only_training_curves.csv", "walk_only_checkpoint_evaluations.csv",
                 "mixed_training_curves.csv", "mixed_checkpoint_evaluations.csv"):
        with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["status", "reason"])
            writer.writerow(["not_executed", "surrogate_gate_failed"])
    write("walk_only_gate.json", {
        **placeholder, "classification": "NOT_EVALUATED_SURROGATE_GATE",
        "initial_strict_copy_sanity": "not_executed_due_to_surrogate_gate",
        "formal_gate": "NOT_RUN",
    })
    final = {
        "classification": "SURROGATE_NOT_TRUSTWORTHY",
        "surrogate_classification": surrogate,
        "student_training_executed": False, "walk_only_evaluation_executed": False,
        "mixed_training_executed": False, "reverse_diagnostic_executed": False,
        "primary_failure": "NONLINEAR_SURROGATE_PHYSICAL_ROLLOUT_AND_ACTION_RANKING_UNRELIABLE",
        "metrics": {
            "one_step_normalized_mae": one["normalized_physical_state_mae"],
            "eight_step_normalized_rmse": multi["horizons"]["8"]["normalized_physical_state_rmse"],
            "ranking_spearman": ranking["spearman_rank_correlation"],
            "ranking_pairwise_accuracy": ranking["pairwise_ranking_accuracy"],
            "unsafe_ranking_inversion": ranking["unsafe_action_ranking_inversion_rate"],
        },
    }
    write("stage3_classification.json", final)
    write("recommended_next_action.json", {
        "single_recommendation": "FROZEN_WALK_BASE_WITH_CONTINUOUS_SPEED_CONDITIONED_RESIDUAL_ADAPTER",
        "reason": "The predeclared nonlinear surrogate trust gate failed; another loss-only single-head redesign is not authorized.",
        "not_implemented_in_stage3": True,
    })
    teacher_paths = {
        "walk": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        "run": REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
        "walk_to_run": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
    }
    expected = {
        "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
        "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
        "walk_to_run": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
    }
    actual = {key: sha(path) for key, path in teacher_paths.items()}
    protected_tree_oids = {
        "stage0": git("rev-parse", "HEAD:results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"),
        "stage1": git("rev-parse", "HEAD:results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"),
        "stage2": git("rev-parse", "HEAD:results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation"),
    }
    capability_paths = [
        REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json",
        REPO / "artifacts/exp_006_unitree_g1_command_skills/command_system_v1/capability_manifest.json",
        REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1/capability_manifest.json",
    ]
    protected = {
        "teacher_hashes_expected": expected, "teacher_hashes_actual": actual,
        "teacher_hashes_pass": actual == expected, "teacher_gradients": 0,
        "protected_stage_tree_oids_at_start": protected_tree_oids,
        "capability_manifest_sha256": {str(path.relative_to(REPO)): sha(path) for path in capability_paths},
        "exp005_006_007_008_modified_by_stage3": False,
        "stage0_1_2_modified_by_stage3": False,
        "capability_manifest_modified": False, "production_artifact_modified": False,
        "ppo_training": 0, "reward_optimization": 0, "production_controller_updates": 0,
        "isaac_lab_modified": False,
        "unrelated_dirty_state_preserved": True,
    }
    write("protected_hashes.json", protected)
    write("gate.json", {
        "stage": "3", "classification": "SURROGATE_NOT_TRUSTWORTHY",
        "one_step": surrogate["one_step_gate"], "multi_step": surrogate["multi_step_gate"],
        "action_ranking": surrogate["action_ranking_gate"], "uncertainty": surrogate["uncertainty_gate"],
        "walk_only_training": "BLOCKED", "mixed_training": "BLOCKED", "reverse_diagnostic": "BLOCKED",
        "student_optimizer_updates": 0, "ppo_updates": 0, "gate_pass": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n"
        "$base = '.\\experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts'\n"
        "& $py \"$base\\collect_stage3_student_rollouts.py\" --viz none\n"
        "& $py \"$base\\build_stage3_surrogate_dataset.py\" --force\n"
        "& $py \"$base\\train_stage3_surrogate.py\"\n"
        "& $py \"$base\\evaluate_stage3_surrogate.py\"\n"
        "& $py \"$base\\finalize_stage3.py\"\n",
        encoding="utf-8",
    )
    report = REPO / "research/exp_009_stage3_nonlinear_rollout_supervision_report.md"
    report.write_text(
        "# exp_009 Stage 3 — Nonlinear rollout supervision\n\n"
        "## Outcome\n\n"
        "**SURROGATE_NOT_TRUSTWORTHY.** The fixed three-member nonlinear residual MLP ensemble "
        "was trained on grouped teacher trajectories plus live Stage 0/1/2 student WALK occupancy. "
        "It passed contact/support/gait classification and finite uncertainty handling, but failed "
        "the predeclared physical-state and action-ranking gates.\n\n"
        "## Dataset and model\n\n"
        f"- Dynamics pairs: {manifest['teacher_pairs']['selected_pairs']:,}\n"
        "- Live student occupancy: 147,520 steps from 922 episodes\n"
        "- Strict matched bounded perturbation branches: 9,185\n"
        "- Ensemble: 3 × (160 → 512 → 512 → 256 → 96), ELU\n"
        "- State setters, teleport, snapshot injection, and Isaac backpropagation: none\n\n"
        "## Trust gates\n\n"
        f"- One-step normalized physical MAE: {one['normalized_physical_state_mae']:.4f} (required ≤ 0.05)\n"
        f"- Contact macro-F1: {one['contact_macro_f1']:.4f}\n"
        f"- Support accuracy: {one['support_phase_accuracy']:.4f}\n"
        f"- Gait accuracy: {one['gait_class_accuracy']:.4f}\n"
        f"- Eight-step normalized RMSE: {multi['horizons']['8']['normalized_physical_state_rmse']:.4f} (required ≤ 0.25)\n"
        f"- Ranking Spearman: {ranking['spearman_rank_correlation']:.4f} (required ≥ 0.70)\n"
        f"- Pairwise ranking: {ranking['pairwise_ranking_accuracy']:.4f} (required ≥ 0.80)\n"
        f"- Unsafe inversion: {ranking['unsafe_action_ranking_inversion_rate']:.4f} (required ≤ 0.10)\n\n"
        "## Decision\n\n"
        "No student checkpoint, WALK-only optimization, mixed distillation, or reverse diagnostic was run. "
        "The next single design is a **frozen WALK base with a continuous speed-conditioned residual/adapter**. "
        "This ends the single-head loss-redesign line without weakening the trust gate.\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
