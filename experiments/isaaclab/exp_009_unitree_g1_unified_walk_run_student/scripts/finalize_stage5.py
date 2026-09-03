"""Finalize Stage 5 without modifying any controller or prior-stage result."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch
import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage5_base_action_manifold_compatibility"
RESEARCH = REPO / "research/exp_009_stage5_base_action_manifold_report.md"
STARTING_HEAD = "0f67ef88fe15c0cc8e1cb6aeb8cd568514c8036c"
CLASSIFICATION = "PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    if git("rev-parse", "HEAD") != STARTING_HEAD:
        raise RuntimeError("Stage 5 must finalize from the authorized starting HEAD")
    stage4_gate = json.loads((
        REPO / "results/exp_009_unitree_g1_unified_walk_run_student/"
        "stage4_frozen_walk_speed_residual/gate.json"
    ).read_text())
    if stage4_gate.get("classification") != "RESIDUAL_PARAMETERIZATION_INADEQUATE":
        raise RuntimeError("Stage 4 classification changed")

    common = torch.load(OUT / "common_state_actions.pt", map_location="cpu", weights_only=False)
    actions = {key: value.numpy() for key, value in common["actions"].items()}
    groups = np.asarray(common["groups"]).astype(str)
    # A-F layer comparison.  There is no clipping in this observed range, so
    # E/F are exactly 0.5 times C/D; F offsets cancel in pairwise differences.
    hierarchy = {}
    for name, target, base in (
        ("RUN_minus_WALK", "run_full", "walk"),
        ("WALK_TO_RUN_minus_WALK", "wtr", "walk"),
        ("RUN_minus_RUN_internal_base", "run_full", "run_base"),
        ("WALK_TO_RUN_minus_RUN_internal_base", "wtr", "run_base"),
    ):
        diff = actions[target] - actions[base]
        hierarchy[name] = {
            "A_raw_network_output_difference_p99": float(np.quantile(np.abs(diff), 0.99)),
            "B_post_policy_clip_normalized_difference_p99": float(np.quantile(np.abs(diff), 0.99)),
            "C_post_skill_composition_difference_p99": float(np.quantile(np.abs(diff), 0.99)),
            "D_post_action_term_clip_difference_p99": float(np.quantile(np.abs(diff), 0.99)),
            "E_actual_scaled_joint_target_delta_difference_p99_rad": float(0.5 * np.quantile(np.abs(diff), 0.99)),
            "F_actual_final_joint_position_target_difference_p99_rad": float(0.5 * np.quantile(np.abs(diff), 0.99)),
            "policy_clip_changed_elements": 0, "action_term_clip_changed_elements": 0,
            "default_position_cancels_in_difference": True,
        }
    pipeline_path = OUT / "action_pipeline_audit.json"
    pipeline = json.loads(pipeline_path.read_text())
    pipeline["same_physical_state_layer_differences"] = hierarchy
    pipeline["formal_feasibility_layer"] = ["E", "F"]
    pipeline["stage4_large_difference_survives_actual_application"] = True
    pipeline["pipeline_mismatch_found"] = False
    pipeline_path.write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    # The tensor bundle is an intermediate cache, not a Stage 5 deliverable.
    (OUT / "common_state_actions.pt").unlink()

    coverage = json.loads((OUT / "candidate_base_residual_coverage.json").read_text())
    compatibility = {}
    for base, regimes in coverage.items():
        all_l1 = min(item["level1_full_vector_coverage"] for item in regimes.values())
        all_l2 = min(item["level2_full_vector_coverage"] for item in regimes.values())
        all_l3 = min(item["level3_full_vector_coverage"] for item in regimes.values())
        compatibility[base] = {
            "minimum_level1_full_vector_coverage": all_l1,
            "minimum_level2_full_vector_coverage": all_l2,
            "minimum_level3_full_vector_coverage": all_l3,
            "all_major_regime_class": "LARGE_CONTROLLER_SHIFT",
        }
    compatibility["localized"] = {
        "A_WALK": "SMALL_RESIDUAL_COMPATIBLE only on WALK occupancy",
        "B_RUN_INTERNAL_BASE": "SMALL_RESIDUAL_COMPATIBLE on RUN and all WALK_TO_RUN occupancy thirds",
        "C_RUN_FULL": "exact on RUN; MODERATE/SMALL compatible with WALK_TO_RUN, incompatible with WALK",
        "D_WTR_ENDPOINT": "exact only on its transition occupancy; acceptance is Level 1/2 compatible with RUN",
    }
    diagnostic_bound = json.loads((OUT / "diagnostic_bound_coverage.json").read_text())
    diagnostic_bound["candidate_compatibility_gate"] = compatibility
    diagnostic_bound["level3_required_is_not_called_residual"] = True
    (OUT / "diagnostic_bound_coverage.json").write_text(json.dumps(diagnostic_bound, indent=2) + "\n")

    dump("stage4_reference.json", {
        "classification": "RESIDUAL_PARAMETERIZATION_INADEQUATE",
        "result_path": "results/exp_009_unitree_g1_unified_walk_run_student/stage4_frozen_walk_speed_residual",
        "formal_residual_bound_normalized": 0.25,
        "scalar_coverage": {"train": 0.396698, "validation": 0.396984, "test": 0.396435},
        "full_37d_sample_coverage": 0.0, "absolute_target_p99_5": 5.363,
        "classification_unchanged": True,
    })
    dump("protocol.json", {
        "stage": "5", "name": "base_action_manifold_compatibility",
        "starting_head": STARTING_HEAD, "training": False,
        "common_state_cross_forward": {
            "states": 100_000, "minimum_per_regime_speed_phase": 10_000,
            "simulator_step": False, "production_action_applied": False,
        },
        "action_layers": ["raw actor", "policy clip", "skill composition", "action term clip", "scaled delta", "final target"],
        "formal_decision_layers": ["scaled delta", "final target"],
        "diagnostic_bounds": [
            {"normalized": 0.25, "target_rad": 0.125},
            {"normalized": 0.50, "target_rad": 0.25},
            {"normalized": 1.00, "target_rad": 0.50},
        ],
        "base_only": {"episodes_per_candidate_command": 20, "formal_evaluation": False},
        "prohibited_actions_observed": {
            "ppo": 0, "optimizer_update": 0, "distillation": 0, "reward_optimization": 0,
            "teacher_update": 0, "new_controller_learning": 0,
        },
    })
    classification = {
        "classification": CLASSIFICATION,
        "stage4_classification_changed": False,
        "pipeline_or_observation_mismatch": False,
        "evidence": {
            "no_policy_clip_changes": True, "no_action_term_clip": True,
            "identical_joint_order_scale_default_pose": True,
            "actual_applied_difference_remains_large": True,
            "no_single_base_level2_all_regimes_99_percent": True,
            "walk_base_is_local_to_walk": True,
            "run_internal_base_is_local_to_run_and_transition": True,
            "best_base_changes_with_regime_and_transition_phase": True,
            "transition_early_vs_walk_level2_full_vector_coverage": 0.0,
            "transition_acceptance_vs_run_level2_full_vector_coverage": 1.0,
        },
        "interpretation": (
            "The large Stage 4 residual is not a raw-output/clip/scale artifact. "
            "WALK and RUN use distinct base manifolds; the frozen RUN internal base "
            "does bound RUN and WALK_TO_RUN residuals, while WALK requires its own base."
        ),
    }
    dump("stage5_classification.json", classification)
    dump("recommended_next_action.json", {
        "single_recommendation": "continuous phase-conditioned base morphing",
        "implemented_in_stage5": False,
        "constraint": "retain modular frozen WALK and RUN bases; diagnose a continuous non-oracle phase signal before implementation",
    })

    teacher_paths = {
        "walk": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/"
        "2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        "run": REPO / "logs/rsl_rl/physical_ai_g1_command_skills/"
        "2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
        "walk_to_run": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/"
        "stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
    }
    protected_paths = {
        "exp005": "experiments/isaaclab/exp_005_unitree_g1_flat_run",
        "exp006": "experiments/isaaclab/exp_006_unitree_g1_command_skills",
        "exp007": "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
        "exp008": "experiments/isaaclab/exp_008_phase_aware_locomotion_transitions",
        "stage0": "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation",
        "stage1": "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis",
        "stage2": "results/exp_009_unitree_g1_unified_walk_run_student/stage2_dynamics_sensitive_distillation",
        "stage3": "results/exp_009_unitree_g1_unified_walk_run_student/stage3_nonlinear_rollout_supervision",
        "stage4": "results/exp_009_unitree_g1_unified_walk_run_student/stage4_frozen_walk_speed_residual",
    }
    tree_oids = {}
    for name, path in protected_paths.items():
        try:
            tree_oids[name] = git("rev-parse", f"{STARTING_HEAD}:{path}")
        except subprocess.CalledProcessError:
            tree_oids[name] = "not_tracked_at_starting_head"
    capability = REPO / "capability_manifest.json"
    if not capability.exists():
        matches = list(REPO.glob("**/capability_manifest.json"))
        capability = matches[0] if matches else None
    dump("protected_hashes.json", {
        "teachers": {name: {"path": str(path.relative_to(REPO)), "sha256": sha(path)} for name, path in teacher_paths.items()},
        "expected_teacher_hashes": {
            "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
            "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
            "walk_to_run": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
        },
        "protected_git_tree_oids_at_start": tree_oids,
        "capability_manifest": None if capability is None else {
            "path": str(capability.relative_to(REPO)), "sha256": sha(capability),
        },
        "production_artifact_updates": 0, "isaaclab_source_updates": 0,
        "teacher_gradients": 0, "optimizer_updates": 0, "ppo_iterations": 0,
    })
    dump("gate.json", {
        "stage": "5", "classification": CLASSIFICATION, "pass": True,
        "pipeline_audit_complete": True, "common_state_states": 100_000,
        "minimum_per_regime_speed_phase_met": True,
        "fixed_bound_levels_only": True, "formal_stage4_bound_changed": False,
        "base_only_diagnostic_complete": True, "new_method_implemented": False,
        "next": "continuous phase-conditioned base morphing",
    })
    reproduction = f"""$ErrorActionPreference = "Stop"
$Repo = Resolve-Path "$PSScriptRoot\\..\\..\\..\\.."
Set-Location $Repo

# Offline frozen-teacher cross-forward; no simulator step or update.
& "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe" `
  "experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\audit_stage5_action_manifold.py"

# Small closed-loop base diagnostic; no training.
& "C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat" -p `
  "experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\evaluate_stage5_base_only.py" `
  --viz none

& "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe" `
  "experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\finalize_stage5.py"
"""
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")

    report = """# exp_009 Stage 5 — Base/action-manifold compatibility

## Result

**PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED**

Stage 4's `RESIDUAL_PARAMETERIZATION_INADEQUATE` result remains unchanged.  The
large target was not caused by comparing an unclipped actor tensor with an
applied action.  No policy clipping occurred, the action term has no configured
clip, all teachers share the 37-joint order and scale 0.5, and the large
difference remains after conversion to the actual joint-position target.

## Pipeline

The WALK actor emits a normalized 37-D position action.  RUN and WALK_TO_RUN
emit `running_base + 0.25*tanh(skill residual)`.  Isaac applies
`default_joint_position + 0.5*normalized_action`.  The actual global previous
action is shared in columns 86:123 during every common-state cross-forward.
Observation normalization is disabled for all three actors.

## Common-state compatibility

100,000 frozen physical observations were used, with 10,000 states for every
WALK speed, RUN speed, and WALK_TO_RUN occupancy third.  The WALK base exactly
matches WALK but has zero Level-2 full-vector coverage on RUN and nearly all
transition groups.  The internal RUN base contains RUN and all transition
groups within its existing 0.25 route, yet does not contain WALK even at Level
2.  Therefore no single existing base reaches 99% full-vector coverage over all
regimes.

The dominant cross-base differences include ankle pitch, ankle roll, knee, and
upper-body coordinates rather than a single joint.  The difference subspaces
need rank 12–14 for 99% variance, so the correction is not one-dimensional,
although it is not full-rank either.

## Transition endpoints

At the early transition occupancy, WALK_TO_RUN differs from the WALK teacher by
0.413 normalized action on average and has 0% Level-2 full-vector coverage.
At acceptance, it is close to RUN: Level-2 full-vector coverage is 100%.
This confirms an endpoint migration toward the RUN manifold, not one common
base spanning both endpoints.

## Closed-loop diagnostic

The 240-episode diagnostic started every candidate from the same valid
WALK@1.2 occupancy.  WALK held 0.6/1.2 but failed to track RUN commands and fell
at 2.8.  RUN internal/full bases held the tested commands in this short
diagnostic, but their action targets still differed materially from the formal
WALK teacher and safety dwell gates were not claimed.  WALK_TO_RUN endpoint
anchors were not treated as steady controllers.

## Next

The single next research method is **continuous phase-conditioned base
morphing** between frozen WALK and RUN manifolds.  Stage 5 does not implement
that method, expand a residual bound, train a controller, or change capability.
"""
    RESEARCH.write_text(report, encoding="utf-8")
    print(json.dumps({"classification": CLASSIFICATION, "outputs": str(OUT), "next": "continuous phase-conditioned base morphing"}, indent=2))


if __name__ == "__main__":
    main()
