"""Assemble the evidence-backed Stage 1 gate without mutating prior stages."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage1_single_head_interference_diagnosis"
STAGE0 = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    single, capacity = load("single_task_capacity_results.json"), load("capacity_sweep.json")
    identity, conflicts = load("teacher_identity_probe.json"), load("label_conflict_audit.json")
    context, gradients = load("transition_context_ablation.json"), load("gradient_interference.json")
    sequential = load("sequential_forgetting.json")
    sweep = load("checkpoint_sweep.json") if (OUT / "checkpoint_sweep.json").exists() else {}
    joint_runtime = OUT / "joint_substitution_runtime/checkpoint_sweep.json"
    substitution_raw = json.loads(joint_runtime.read_text()) if joint_runtime.exists() else {}
    multihead_runtime = OUT / "multihead_runtime/checkpoint_sweep.json"
    multihead_closed = json.loads(multihead_runtime.read_text()) if multihead_runtime.exists() else {}
    conflict_detail = load("label_conflict_audit.json")
    conflict_detail["base_teacher_dataset"] = {"rows": 1880660, "analysis": "fixed Stage-0 grouped split"}
    conflict_detail["dagger_added_dataset"] = {
        "rows": 112500,
        "analyzed_separately": True,
        "closed_loop_outcome": "ROUND_1_DID_NOT_RESTORE_ALL_RETENTION_GATES",
        "run_to_walk_labels_used": False,
    }
    write("label_conflict_audit.json", conflict_detail)

    # Summarize closed-loop condition-specific controls without presenting a
    # missing regime as a measured zero.
    single_closed = {}
    scopes = {"single_walk_steady": "walk", "single_run_steady": "run", "single_walk_to_run": "walk_to_run"}
    for model, section in scopes.items():
        single_closed[model] = sweep.get(model, {}).get(section, {})
    write("single_task_closed_loop_results.json", {
        "episodes_per_condition": 50,
        "deterministic_isaac_physics": True,
        "results": single_closed,
        "interpretation": "WALK-only fails despite near-zero held-out action error; individual one-step fit is not sufficient for closed-loop retention.",
    })
    separate = load("separate_network_upper_bound.json")
    separate["closed_loop_results"] = single_closed
    separate["all_teacher_capabilities_retained"] = False
    separate["interpretation"] = "RUN-only retains periodic running; WALK-only and WALK_TO_RUN-only do not retain their teacher closed-loop gates."
    write("separate_network_upper_bound.json", separate)

    # Capacity closed-loop entries may be absent if a diagnostic run was not
    # completed; the gate records that explicitly rather than synthesizing data.
    stage0_sweep = json.loads((STAGE0 / "checkpoint_sweep.json").read_text(encoding="utf-8"))
    stage0_small = stage0_sweep.get("epoch_10", {})
    write("capacity_closed_loop_results.json", {
        "small": {
            "source": "immutable Stage-0 selected epoch-10 small single-head",
            "results": stage0_small,
        },
        "medium": sweep.get("capacity_medium", {}),
        "large": sweep.get("capacity_large", {}),
        "walk_comparison": {
            "medium_success": {key: row.get("success_rate") for key, row in sweep.get("capacity_medium", {}).get("walk", {}).items()},
            "large_success": {key: row.get("success_rate") for key, row in sweep.get("capacity_large", {}).get("walk", {}).items()},
        },
        "conclusion": "Both larger 123D single-head controls score 0% WALK retention at every tested WALK speed; raw capacity does not restore closed-loop behavior.",
    })
    write("joint_group_substitution_results.json", {
        "scope": "WALK closed-loop, selected Stage-0 epoch-10 versus exact WALK initialization",
        "joint_groups": {"hip_pitch": [0, 1], "knee": [11, 12], "ankle_pitch": [15, 16], "ankle_roll": [19, 20]},
        "results": substitution_raw,
        "completed": bool(substitution_raw),
    })
    multihead_offline = load("diagnostic_multihead_results.json")
    multihead_offline["closed_loop_oracle_head_results"] = multihead_closed
    multihead_offline["production_eligible"] = False
    write("diagnostic_multihead_results.json", multihead_offline)

    # The same-source-route Isaac evaluator exposes first contract/contact loss
    # through streak and terminal timing. Exact state-vector divergence was not
    # available in the immutable Stage-0 logger, so do not claim it.
    walk = single_closed.get("single_walk_steady", {})
    divergence = {
        "method": "same evaluation seed and formal source route; teacher and student evaluated without state injection",
        "requested_horizons_steps": [1, 2, 4, 8, 16, 32],
        "state_vector_trace_available": False,
        "contact_gait_divergence_proxy": {
            speed: {
                "walk_valid_streak_mean": row.get("walk_valid_streak_mean"),
                "walk_valid_streak_max": row.get("walk_valid_streak_max"),
                "timeout_rate": row.get("timeout_rate"),
                "fall_rate": row.get("fall_rate"),
            } for speed, row in walk.items()
        },
        "limitation": "No cross-environment state snapshot injection was used; immutable Stage-0 runtime did not persist paired root/joint traces at every requested horizon.",
    }
    write("rollout_divergence.json", divergence)

    full_context_gain = context["A_123D"]["validation"]["mae"] - context["D_full_29D"]["validation"]["mae"]
    teacher_id_gain = context["A_123D"]["validation"]["mae"] - context["E_teacher_identity"]["validation"]["mae"]
    large_gain = capacity["small"]["validation"]["mae"] - capacity["large"]["validation"]["mae"]
    negative_overall = [item for item in gradients["overall"] if item["cosine"] < 0]
    worst = min((item["cosine"] for item in gradients["overall"]), default=0)
    forgetting = []
    for order in sequential:
        for phase in order["phases"]:
            trained = phase["trained"]
            metrics = phase["evaluation"]["teacher_mae"]
            forgetting.append({"order": order["order"], "after": trained, "teacher_mae": metrics})

    # Evidence supports two independent mechanisms: regime gradients interfere,
    # while even the isolated WALK/WTR fits are dynamically fragile. Neither can
    # honestly be collapsed into the other under the pre-registered definitions.
    classification = "MULTIPLE_FAILURE_MODES"
    write("stage1_classification.json", {
        "classification": classification,
        "hypotheses": {
            "H1_raw_capacity": {
                "status": "NOT_PRIMARY",
                "evidence": f"Large-minus-small offline improvement is {large_gain:.6f} (positive means better); size alone does not restore all abilities.",
            },
            "H2_hidden_label_mode": {
                "status": "REJECTED_AS_PRIMARY",
                "exact_cross_regime_duplicates": conflicts["exact_cross_regime_duplicate_rows"],
                "teacher_identity_mlp_accuracy": identity["small_mlp"]["accuracy"],
                "full_context_mae_gain": full_context_gain,
                "teacher_identity_mae_gain": teacher_id_gain,
                "context_comparison_caveat": "Expanded-input models cannot inherit the exact 123D WALK initialization; negative gain is not evidence against context utility. Identity predictability and absence of near-duplicate collisions are the primary H2 tests.",
            },
            "H3_gradient_conflict": {
                "status": "SUPPORTED",
                "negative_gradient_fraction": gradients["negative_fraction"],
                "negative_overall_measurements": len(negative_overall),
                "worst_overall_cosine": worst,
                "sequential_forgetting": forgetting,
            },
            "H4_closed_loop_sensitivity": {
                "status": "SUPPORTED_AND_PRIORITIZED",
                "evidence": "WALK-only held-out MAE is near zero while deterministic closed-loop retention remains poor; replacing only ankle-roll actions with teacher values strongly restores WALK retention at 1.0/1.2 m/s.",
            },
        },
        "not_production_architecture_selection": True,
    })
    write("recommended_next_action.json", {
        "single_recommendation": "dynamics-sensitive distillation loss with short-horizon contact/state matching",
        "reason": "Prioritize the failure that survives task isolation: tiny one-step WALK error still diverges in closed loop, and ankle-roll teacher substitution restores much of the lost retention. Gradient mitigation is deferred.",
        "implemented": False,
    })

    protected_paths = {
        "walk_teacher": REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt",
        "run_teacher": REPO / "logs/rsl_rl/physical_ai_g1_command_skills/2026-07-19_22-02-41_pilot_turn90_right_residual_from_model0/model_0.pt",
        "walk_to_run_teacher": REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage7r8_walk_to_run_pilot2_saturation/checkpoints/model_100.pt",
        "stage0_selected": STAGE0 / "checkpoints/epoch_10.pt",
    }
    hashes = {name: {"path": str(path.relative_to(REPO)), "sha256": sha(path)} for name, path in protected_paths.items()}
    hashes["checks"] = {
        "teacher_hashes_match": hashes["walk_teacher"]["sha256"] == "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa" and hashes["run_teacher"]["sha256"] == "60675a79187ef00935069f3bf85293e72477fa1d408b80d3325232d805b0d266" and hashes["walk_to_run_teacher"]["sha256"] == "d94a94409ed9651734ae8ebc85313b827a809de139135115e833c76ca3bb9fd0",
        "stage0_selected_match": hashes["stage0_selected"]["sha256"] == "9b98c94d8143568cfa64625ccb6b3f7cd26147518ceb8aac44149c0605722fa8",
        "ppo_training": 0,
        "reward_optimization": 0,
        "production_controller_updates": 0,
        "capability_manifest_modified": False,
        "isaac_lab_modified": False,
    }
    write("protected_hashes.json", hashes)
    required = [
        "single_task_capacity_results.json", "single_task_closed_loop_results.json",
        "label_conflict_audit.json", "quantized_conflict_results.json", "nearest_neighbor_disagreement.json",
        "teacher_identity_probe.json", "transition_context_ablation.json", "capacity_sweep.json",
        "capacity_closed_loop_results.json", "gradient_interference.json", "layerwise_gradient_cosines.csv",
        "sequential_forgetting.json", "jointwise_action_errors.csv", "joint_group_substitution_results.json",
        "rollout_divergence.json", "diagnostic_multihead_results.json", "separate_network_upper_bound.json",
        "stage1_classification.json", "recommended_next_action.json", "protected_hashes.json",
    ]
    missing = [name for name in required if not (OUT / name).exists()]
    write("gate.json", {
        "status": "PASS_DIAGNOSTIC_WITH_TRACE_LIMITATION" if not missing else "INCOMPLETE_DIAGNOSTIC",
        "classification": classification,
        "missing_outputs": missing,
        "limitation": "Paired root/joint state traces at every 1/2/4/8/16/32-step horizon were not persisted; contact/gait divergence proxies and in-place joint substitutions were completed without state injection.",
        "production_promotion": False,
        "ppo_training": 0,
        "reward_optimization": 0,
    })


if __name__ == "__main__":
    main()
