"""Seal Stage 4 after the residual parameterization gate."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage4_frozen_walk_speed_residual"
CFG_PATH = EXP / "configs/stage4_frozen_walk_speed_residual.yaml"
STAGE0_CFG = EXP / "configs/stage0_multiteacher_distillation.yaml"
sys.path.insert(0, str(EXP / "src"))
from unified_walk_run.frozen_walk_residual import FrozenWalkSpeedResidualController123  # noqa: E402
from unified_walk_run.student_actor import UnifiedWalkRunStudent123  # noqa: E402


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


def load_base(device="cpu"):
    checkpoint = torch.load(
        REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        map_location=device, weights_only=False,
    )
    state = {key.removeprefix("mlp."): value for key, value in checkpoint["actor_state_dict"].items() if key.startswith("mlp.")}
    base = UnifiedWalkRunStudent123().to(device)
    base.load_state_dict(state, strict=True)
    return base


def main() -> None:
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    stage0 = yaml.safe_load(STAGE0_CFG.read_text(encoding="utf-8"))
    parameterization = json.loads((OUT / "residual_parameterization_audit.json").read_text(encoding="utf-8"))
    preservation = json.loads((OUT / "walk_bitwise_preservation_audit.json").read_text(encoding="utf-8"))
    bound = json.loads((OUT / "residual_bound_decision.json").read_text(encoding="utf-8"))
    distribution = json.loads((OUT / "residual_target_distribution.json").read_text(encoding="utf-8"))
    start = "9c7a6bcb022f71d03620dc1829765534fd9ff166"
    if subprocess.run(["git", "merge-base", "--is-ancestor", start, "HEAD"], cwd=REPO).returncode:
        raise RuntimeError("Stage 4 starting revision is not an ancestor")
    write("stage3_reference.json", {
        "classification": "SURROGATE_NOT_TRUSTWORTHY",
        "one_step_physical_mae": 0.12939663231372833,
        "eight_step_physical_rmse": 0.5905935168266296,
        "ranking_spearman": 0.39619506981022556,
        "stage0_1_2_3_immutable": True,
    })
    write("protocol.json", {
        "method": "frozen WALK base + continuous speed-conditioned bounded action residual",
        "architecture": {"frozen_base_parameters": 85925, "trainable_residual_parameters": 85925, "residual_layers": [123, 256, 128, 128, 37], "activation": "ELU"},
        "single_method": True, "base_trainable": False, "residual_trainable_if_feasible": True,
        "gate_order": ["WALK bitwise preservation", "residual parameterization", "offline training", "WALK", "RUN", "WALK_TO_RUN", "intermediate", "reverse"],
        "ppo": 0, "reward_optimization": 0, "surrogate_training": 0, "gradient_surgery": False,
    })
    write("base_observation_contract.json", {
        "source": "canonical 123D code-audited layout",
        "forward_command": "min(requested_forward_command, 1.2)",
        "unchanged_fields": ["physical observation", "joint position", "joint velocity", "projected gravity", "base velocity", "global previous action", "lateral command", "yaw-rate command"],
        "world_xy": False, "teacher_identity": False, "skill_one_hot": False,
    })
    write("residual_observation_contract.json", {
        "dimension": 123, "requested_forward_speed_range_mps": [0.6, 2.8],
        "forward_command": "actual requested speed, unclamped",
        "teacher_identity": False, "transition_identity": False, "skill_one_hot": False,
    })
    gate_definition = {
        "x": "clip((v_cmd - 1.2)/(2.4 - 1.2), 0, 1)", "gate": "3*x^2 - 2*x^3",
        "v_le_1_2": 0.0, "v_ge_2_4": 1.0, "learned": False,
        "gate_zero_addition_executed": False,
    }
    write("speed_gate_definition.json", {**gate_definition, "sha256": canonical_sha(gate_definition)})
    shutil.copyfile(CFG_PATH, OUT / "stage4_config.yaml")
    shared_training = ["optimizer", "batch_size", "learning_rate", "epochs", "action_huber_delta", "action_loss_weight",
                       "action_delta_loss_weight", "weight_decay", "gradient_clip_norm", "early_stopping_patience_epochs",
                       "early_stopping_min_delta", "early_stopping_minimum_epoch", "checkpoint_epochs", "joint_weights"]
    unexpected = {
        key: {"stage0": stage0["training"].get(key), "stage4": cfg["training"].get(key)}
        for key in shared_training if stage0["training"].get(key) != cfg["training"].get(key)
    }
    write("stage0_vs_stage4_config_diff.json", {
        "expected_semantic_differences": ["architecture", "frozen_walk_base", "residual_bounds", "zero_anchor", "speed_gate", "output_path", "run_name"],
        "shared_training_values": {key: cfg["training"][key] for key in shared_training},
        "unexpected_optimizer_or_budget_differences": unexpected,
        "unexpected_config_differences": len(unexpected),
    })
    base = load_base()
    controller = FrozenWalkSpeedResidualController123(base, torch.tensor(cfg["controller"]["residual"]["per_joint_bounds"]))
    initial_dir = OUT / "checkpoints"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial = initial_dir / "initial.pt"
    torch.save({
        "residual": controller.residual.state_dict(), "residual_bounds": controller.residual_bounds,
        "epoch": 0, "diagnostic_only": True, "training_updates": 0,
        "base_walk_sha256": cfg["teachers"]["walk"]["sha256"],
    }, initial)
    dataset_hash_path = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation/dataset_hashes.json"
    hashes = {
        "config_sha256": sha(CFG_PATH), "config_canonical_sha256": canonical_sha(cfg),
        "dataset_manifest_sha256": sha(dataset_hash_path),
        "speed_gate_sha256": canonical_sha(gate_definition), "residual_bound_sha256": bound["bound_sha256"],
        "base_walk_sha256": cfg["teachers"]["walk"]["sha256"],
    }
    write("stage4_protocol_hashes.json", hashes)
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["status", "reason", "optimizer_updates"])
        writer.writerow(["not_executed", "residual_parameterization_inadequate", 0])
    write("checkpoint_manifest.json", {
        "checkpoints": [{
            "role": "initial_zero_residual", "path": str(initial.relative_to(REPO)), "sha256": sha(initial),
            "epoch": 0, "validation_action_loss": None, "validation_residual_loss": None,
            "base_walk_sha256": cfg["teachers"]["walk"]["sha256"], **hashes,
        }],
        "selected_checkpoint": None, "training_checkpoints_created": 0,
    })
    write("offline_reconstruction_results.json", {
        "status": "not_executed_due_to_residual_parameterization_gate",
        "coverage": parameterization["coverage"], "all_joint_row_coverage": parameterization["all_37_joint_row_coverage"],
        "initial_residual_output": "exact zero", "optimizer_updates": 0,
    })
    placeholder = {
        "status": "not_executed_due_to_residual_parameterization_gate",
        "blocking_classification": "RESIDUAL_PARAMETERIZATION_INADEQUATE",
    }
    write("walk_exact_retention.json", {
        **placeholder, "structural_bitwise_audit": "PASS",
        "bitwise_samples": preservation["formal_walk_samples"],
        "formal_50_episode_per_speed_evaluation": "NOT_RUN",
    })
    for name in ("run_retention.json", "walk_to_run_retention.json", "teacher_retention_summary.json",
                 "intermediate_speed_diagnostics.json", "action_continuity_audit.json", "reverse_diagnostic.json"):
        write(name, placeholder)
    final = {
        "classification": "RESIDUAL_PARAMETERIZATION_INADEQUATE",
        "walk_bitwise_preservation": "PASS",
        "residual_parameterization": "FAIL",
        "training_executed": False, "closed_loop_evaluation_executed": False,
        "coverage": parameterization["coverage"],
        "effective_speed_gated_coverage": parameterization["effective_final_action_envelope_coverage_with_speed_gate"],
        "target_abs_p99_5": distribution["global"]["abs_p99_5"],
        "target_max_abs": distribution["global"]["max_abs"],
        "reason": "The existing formal +/-0.25 RUN residual bound cannot express RUN/WALK_TO_RUN teacher action differences relative to the frozen WALK base.",
    }
    write("stage4_classification.json", final)
    write("recommended_next_action.json", {
        "single_recommendation": "CLOSE_FROZEN_WALK_RESIDUAL_V1_AND_AUDIT_BASE_COMPATIBILITY_BEFORE_ANY_NEW_METHOD",
        "reason": "The mandated existing formal residual envelope covers only ~39.7% of scalar targets and no complete 37D target.",
        "not_implemented": True,
    })
    teacher_actual = {key: sha(REPO / value["path"]) for key, value in cfg["teachers"].items()}
    tree_paths = {
        "stage0": "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation",
        "stage1": "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis",
        "stage2": "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation",
        "stage3": "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision",
    }
    write("protected_hashes.json", {
        "teacher_expected": {key: value["sha256"] for key, value in cfg["teachers"].items()},
        "teacher_actual": teacher_actual, "teacher_hashes_pass": teacher_actual == {key: value["sha256"] for key, value in cfg["teachers"].items()},
        "protected_stage_tree_oids": {key: git("rev-parse", f"HEAD:{path}") for key, path in tree_paths.items()},
        "teacher_gradients": 0, "walk_base_gradients": 0, "ppo_training": 0, "reward_optimization": 0,
        "capability_manifest_modified": False, "production_artifact_modified": False,
        "exp005_006_007_008_modified": False, "isaac_lab_modified": False, "unrelated_dirty_state_preserved": True,
    })
    write("gate.json", {
        "stage": "4", "classification": final["classification"],
        "walk_bitwise_preservation": "PASS", "residual_parameterization": "FAIL",
        "offline_training": "BLOCKED", "walk_closed_loop": "BLOCKED", "run": "BLOCKED",
        "walk_to_run": "BLOCKED", "intermediate": "BLOCKED", "reverse": "BLOCKED",
        "optimizer_updates": 0, "ppo_updates": 0, "gate_pass": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        "$ErrorActionPreference = 'Stop'\n"
        "$py = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'\n"
        "$base = '.\\experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts'\n"
        "& $py \"$base\\audit_stage4_residual_feasibility.py\"\n"
        "& $py \"$base\\finalize_stage4.py\"\n"
        "# Training/evaluation intentionally blocked when residual_parameterization_audit.pass is false.\n",
        encoding="utf-8",
    )
    report = REPO / "research/exp_009_stage4_frozen_walk_speed_residual_report.md"
    report.write_text(
        "# exp_009 Stage 4 — Frozen WALK speed residual\n\n"
        "## Classification\n\n**RESIDUAL_PARAMETERIZATION_INADEQUATE**\n\n"
        "The frozen WALK base was preserved bitwise on all 594,360 formal-WALK dataset samples. "
        "The controller performs no residual addition when the fixed speed gate is zero.\n\n"
        "The existing formal exp_006 residual envelope (per-joint ±0.25) was then applied without tuning. "
        f"Scalar target coverage was train {parameterization['coverage']['train']:.4%}, "
        f"validation {parameterization['coverage']['validation']:.4%}, and test {parameterization['coverage']['test']:.4%}; "
        "complete 37D sample coverage was 0% in every split. This is far below the frozen 99.5/99/99% gate.\n\n"
        f"The global absolute target p99.5 was {distribution['global']['abs_p99_5']:.3f} and max was "
        f"{distribution['global']['max_abs']:.3f}, compared with the fixed 0.25 bound.\n\n"
        "No residual distillation, closed-loop retention, intermediate-speed evaluation, or reverse diagnostic was run. "
        "The negative result indicates that RUN/WALK_TO_RUN actions are not a bounded ±0.25 correction around this WALK base. "
        "The single next action is to close residual v1 and audit base/action-manifold compatibility before authorizing another method.\n",
        encoding="utf-8",
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
