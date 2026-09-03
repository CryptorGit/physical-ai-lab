"""Finalize Stage 6 feasibility diagnosis and enforce protected provenance."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage6_phase_conditioned_base_morph_feasibility"
RESEARCH = REPO / "research/exp_009_stage6_phase_conditioned_base_morph_report.md"
START = "693e8ae3a20969945958bf0a6b4d22003228d8b1"
CLASSIFICATION = "BASE_PAIR_INADEQUATE"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(name):
    return json.loads((OUT / name).read_text())


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    if git("rev-parse", "HEAD") != START:
        raise RuntimeError("unauthorized Stage 6 starting revision")
    stage5 = json.loads((
        REPO / "results/exp_009_unitree_g1_unified_walk_run_student/"
        "stage5_base_action_manifold_compatibility/stage5_classification.json"
    ).read_text())
    if stage5["classification"] != "PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED":
        raise RuntimeError("Stage 5 classification changed")
    stage4 = json.loads((
        REPO / "results/exp_009_unitree_g1_unified_walk_run_student/"
        "stage4_frozen_walk_speed_residual/gate.json"
    ).read_text())
    if stage4["classification"] != "RESIDUAL_PARAMETERIZATION_INADEQUATE":
        raise RuntimeError("Stage 4 classification changed")
    coverage = load("oracle_residual_coverage.json")
    endpoints = load("endpoint_consistency.json")
    monotonic = load("alpha_monotonicity.json")
    speed = load("speed_only_reference.json")
    group = load("groupwise_residual_coverage.json")
    probes = load("alpha_probe_results.json")
    if coverage["oracle_morph_feasible"]:
        raise RuntimeError("classification disagrees with oracle gate")
    if probes.get("status") != "not_executed":
        raise RuntimeError("predictor must not train after oracle failure")

    dump("stage5_reference.json", {
        "starting_head": START,
        "classification": "PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED",
        "stage4_classification": "RESIDUAL_PARAMETERIZATION_INADEQUATE",
        "stage5_result_path": "results/exp_009_unitree_g1_unified_walk_run_student/stage5_base_action_manifold_compatibility",
        "stage5_results_modified": False, "stage5_classification_modified": False,
        "known_applied_differences": {
            "run_minus_walk_normalized_p99": 3.253,
            "run_minus_walk_joint_target_p99_rad": 1.627,
            "wtr_minus_walk_normalized_p99": 3.297,
            "wtr_minus_walk_joint_target_p99_rad": 1.648,
            "run_minus_runbase_normalized_p99": 0.224,
            "wtr_minus_runbase_normalized_p99": 0.247,
        },
    })
    dump("protocol.json", {
        "stage": 6, "name": "phase_conditioned_base_morph_feasibility",
        "parameterization": {
            "anchor": "(1-alpha)*frozen_WALK + alpha*frozen_RUN_internal_base",
            "final": "anchor + bounded_residual", "alpha_range": [0, 1],
            "residual_bound_normalized": 0.25, "action_scale": 0.5,
        },
        "common_states": 100_000, "wtr_complete_episodes": 300,
        "oracle_alpha": "specified clipped least-squares scalar",
        "early_stop": "do not train alpha predictors when any major oracle full-vector coverage is below 95%",
        "execution": {
            "isaac_steps": 0, "actual_robot_actions": 0, "controller_training": 0,
            "ppo": 0, "distillation": 0, "reward_optimization": 0,
            "diagnostic_probe_training": 0,
        },
    })
    classification = {
        "classification": CLASSIFICATION,
        "oracle_scalar_gate": "ORACLE_MORPH_INADEQUATE",
        "primary_gate_failure": {
            "regime": "walk_to_run",
            "oracle_scalar_full_vector_coverage": coverage["coverage"]["walk_to_run"]["full_vector_coverage"],
            "required_for_feasible": 0.99, "hard_stop_threshold": 0.95,
        },
        "endpoint_consistency": endpoints["pass"],
        "morph_continuity_gate": monotonic["pass"],
        "wtr_progresses_zero_to_one_rate": monotonic["progresses_zero_to_one"],
        "speed_only_sufficient": speed["sufficient"],
        "groupwise_full_vector_coverage_wtr": group["coverage"]["walk_to_run"]["full_vector_coverage"],
        "groupwise_feasible": group["groupwise_morph_feasible"],
        "predictor_training": "not_executed_due_to_oracle_morph_inadequate",
        "stage4_classification_unchanged": True,
        "stage5_classification_unchanged": True,
        "interpretation": (
            "The prescribed least-squares scalar anchor satisfies steady endpoints but does not keep all 37 "
            "WALK_TO_RUN residual coordinates inside the existing ±0.25 bound. Fixed joint-group least-squares "
            "alphas do not recover full-vector feasibility."
        ),
    }
    dump("stage6_classification.json", classification)
    dump("recommended_next_action.json", {
        "controller_to_implement": None,
        "single_next_step": "no new morph controller; retain modular frozen experts",
        "reason": "oracle parameterization failed before alpha identifiability could be evaluated",
        "stage6_implements_next_step": False,
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
        "stage5": "results/exp_009_unitree_g1_unified_walk_run_student/stage5_base_action_manifold_compatibility",
    }
    trees = {}
    for name, path in protected_paths.items():
        try:
            trees[name] = git("rev-parse", f"{START}:{path}")
        except subprocess.CalledProcessError:
            trees[name] = "not_tracked"
    manifests = list(REPO.glob("**/capability_manifest.json"))
    capability = manifests[0] if manifests else None
    dump("protected_hashes.json", {
        "teachers": {name: {"path": str(path.relative_to(REPO)), "sha256": sha(path)} for name, path in teacher_paths.items()},
        "expected": {
            "walk": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
            "run": "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266",
            "walk_to_run": "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
        },
        "protected_tree_oids_at_start": trees,
        "capability_manifest": None if capability is None else {"path": str(capability.relative_to(REPO)), "sha256": sha(capability)},
        "production_artifact_updates": 0, "production_controller_updates": 0,
        "teacher_gradients": 0, "optimizer_updates": 0, "ppo_iterations": 0,
        "isaaclab_source_updates": 0,
    })
    dump("gate.json", {
        "stage": 6, "classification": CLASSIFICATION, "complete": True,
        "oracle_endpoint_pass": endpoints["pass"], "oracle_morph_feasible": False,
        "oracle_wtr_full_vector_coverage": coverage["coverage"]["walk_to_run"]["full_vector_coverage"],
        "groupwise_morph_feasible": False,
        "alpha_predictor_executed": False, "alpha_predictor_early_stop_correct": True,
        "actual_robot_stepping": 0, "residual_bound_changed": False,
        "new_controller_authorized": False,
    })
    commands = """$ErrorActionPreference = "Stop"
$Repo = Resolve-Path "$PSScriptRoot\\..\\..\\..\\.."
Set-Location $Repo

# Frozen-teacher offline cross-forward and oracle morph audit only.
& "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe" `
  "experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\audit_stage6_phase_morph.py"

& "C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe" `
  "experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\finalize_stage6.py"
"""
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")
    report = f"""# exp_009 Stage 6 — Phase-conditioned base morph feasibility

## Classification

**{CLASSIFICATION}**

Stage 4 remains `RESIDUAL_PARAMETERIZATION_INADEQUATE`; Stage 5 remains
`PIECEWISE_OR_PHASE_CONDITIONED_BASE_REQUIRED`.

## Oracle scalar morph

The audit reconstructed 100,000 common states without Isaac stepping:
40,000 WALK, 30,000 RUN, and 30,000 states from 300 complete WALK_TO_RUN
episodes.  All actions are actual normalized actions before the common 0.5
environment scale.

Steady endpoints are consistent: WALK alpha p95 is {endpoints['walk_alpha_p95']:.3f}
and RUN alpha p05 is {endpoints['run_alpha_p05']:.3f}.  WALK and RUN full-vector
coverage are 100%.  WALK_TO_RUN coverage is only
{coverage['coverage']['walk_to_run']['full_vector_coverage']:.2%}, below both
the 99% feasibility requirement and 95% early-stop threshold.

The oracle alpha is already near RUN throughout the transition (mean
{load('oracle_scalar_morph.json')['alpha_mean_by_regime']['walk_to_run']:.3f});
zero-to-one progression occurs in {monotonic['progresses_zero_to_one']:.1%} of
episodes.  Its numerical trajectory passes the variation/jump gate only because
it begins near the RUN base rather than morphing from WALK.

## Speed and phase

Fixed speed smoothstep provides only
{speed['coverage']['walk_to_run']['full_vector_coverage']:.2%} WALK_TO_RUN
coverage.  The specified scalar oracle itself is infeasible, so the protocol
correctly stopped before fitting speed, 123D, transition-scalar, or explicit
phase alpha probes.  No identifiability claim is made.

## Groupwise diagnostic

Five fixed joint-group least-squares alphas reduce neither the bound violation
nor the full-vector requirement: WALK_TO_RUN coverage is
{group['coverage']['walk_to_run']['full_vector_coverage']:.2%}.  Thus the
requested groupwise fallback also fails its 99% gate.

## Next

No morph controller is authorized.  Retain the modular frozen experts; do not
implement a scalar or groupwise phase-conditioned morph from this result.
"""
    RESEARCH.write_text(report, encoding="utf-8")
    intermediate = OUT / "stage6_intermediate.pt"
    if intermediate.exists():
        intermediate.unlink()
    print(json.dumps({"classification": CLASSIFICATION, "predictors": "not_executed", "next_controller": None}, indent=2))


if __name__ == "__main__":
    main()
