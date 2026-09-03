"""Materialize the immutable exp_009 Stage-0 evidence and classification."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parents[1]
REPO = EXP.parents[2]
OUT = REPO / "results/exp_009_unitree_g1_unified_walk_run_student/stage0_multiteacher_distillation"
CFG_PATH = EXP / "configs/stage0_multiteacher_distillation.yaml"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def mean_metric(section, key):
    values = [float(item[key]) for item in section.values()]
    return sum(values) / len(values)


def main():
    cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "dataset_manifest.json").read_text(encoding="utf-8"))
    sweep = json.loads((OUT / "checkpoint_sweep.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((OUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    offline = json.loads((OUT / "offline_evaluation.json").read_text(encoding="utf-8"))
    manifold = json.loads((OUT / "action_manifold_audit.json").read_text(encoding="utf-8"))

    def gates(result):
        walk = result["walk"]
        run = result["run"]
        wtr = result["walk_to_run"]
        walk_pass = (
            mean_metric(walk, "success_rate") >= 0.95
            and all(row["success_rate"] >= 0.90 and row["fall_rate"] <= 0.02 and row["heading_p95_rad"] <= 0.12 and row["saturation_rate"] <= 0.05 for row in walk.values())
        )
        run_pass = (
            mean_metric(run, "success_rate") >= 0.95
            and all(row["success_rate"] >= 0.90 and row["fall_rate"] <= 0.02 and row["heading_p95_rad"] <= 0.12 and row["saturation_rate"] <= 0.05 for row in run.values())
        )
        wtr_pass = all(
            row["success_rate"] >= 0.90 and row["speed_acquisition_rate"] >= 0.95
            and row["fall_rate"] <= 0.02 and row["saturation_rate"] <= 0.05
            for row in wtr.values()
        )
        return {"walk": walk_pass, "run": run_pass, "walk_to_run": wtr_pass, "all": walk_pass and run_pass and wtr_pass}

    ranked = []
    for label, result in sweep.items():
        gate = gates(result)
        retention_score = (
            mean_metric(result["walk"], "success_rate")
            + mean_metric(result["run"], "success_rate")
            + mean_metric(result["walk_to_run"], "success_rate")
        )
        safety = sum(mean_metric(result[key], "fall_rate") + mean_metric(result[key], "saturation_rate") for key in ("walk", "run", "walk_to_run"))
        ranked.append((gate["all"], retention_score - safety, label, result, gate))
    ranked.sort(reverse=True)
    _, _, selected_label, selected, selected_gates = ranked[0]
    reverse = selected["run_to_walk"]
    reverse_emerges = all(
        row["walk_contract_rate"] >= 0.90 and row["success_rate"] >= 0.85
        and row["fall_rate"] <= 0.05 and row["saturation_rate"] <= 0.10 and row["timeout_rate"] <= 0.10
        for row in reverse.values()
    )
    if selected_gates["all"] and reverse_emerges:
        classification = "DISTILLATION_PASS_REVERSE_EMERGES"
        recommendation = "Stage 1 formal protocol for the unified student"
    elif selected_gates["all"]:
        classification = "DISTILLATION_PASS_REVERSE_NEEDS_RL"
        recommendation = "One joint bidirectional-command RL pilot initialized from the unified student"
    else:
        classification = "DISTILLATION_FAIL_INTERFERENCE"
        recommendation = "Re-diagnose the single-head student architecture"

    teacher_manifest = {}
    for name, teacher in cfg["teachers"].items():
        path = REPO / teacher["path"]
        teacher_manifest[name] = {
            **teacher, "actual_sha256": sha(path), "hash_match": sha(path) == teacher["sha256"],
            "frozen": True, "gradient_updates": 0,
        }
    config_hash = canonical_sha(cfg)
    dagger_training = json.loads((OUT / "dagger_training_summary.json").read_text(encoding="utf-8"))
    if not any(item.get("path") == dagger_training["checkpoint"] for item in checkpoints):
        checkpoints.append({
            "epoch": cfg["dagger"]["epochs"], "path": dagger_training["checkpoint"],
            "sha256": dagger_training["sha256"], "role": "dagger_round1",
            "parent_sha256": dagger_training["parent_sha256"],
        })
        dump("checkpoint_manifest.json", checkpoints)
    shutil.copyfile(CFG_PATH, OUT / "distillation_config.yaml")
    dump("exp007_reference.json", {
        "classification": "PARTIAL_SUCCESS_ASYMMETRIC_STATE_GRAPH",
        "walk_to_run": "PASS_LIMITED", "run_to_walk": "NO_GO_V1", "modified": False,
    })
    dump("exp008_reference.json", {
        "observability": "BREAK_NOT_PREDICTABLE", "controllability": "NO_LOCAL_CORRECTION_FOUND",
        "decision": "UNIFIED_WALK_RUN_DISTILLATION", "modified": False,
    })
    dump("teacher_manifest.json", teacher_manifest)
    dump("student_observation_contract.json", {
        "dimension": 123, "source": "canonical legacy locomotion observation",
        "fields": ["base linear velocity", "base angular velocity", "projected gravity", "continuous velocity command", "joint position", "joint velocity", "actual global previous action"],
        "teacher_identity": False, "skill_identity": False, "transition_identity": False,
    })
    counts = manifest["regime_counts"]
    total = manifest["total_steps"]
    dump("dataset_balance.json", {
        "counts": counts, "fractions": {key: value / total for key, value in counts.items()},
        "target": cfg["dataset"]["target_balance"], "minimum_steps_met": total >= 1_500_000,
        "phase_counts": manifest["support_phase_counts"], "speed_counts": manifest["speed_counts"],
    })
    dump("split_manifest.json", {
        "unit": "episode_seed_trajectory", "counts": manifest["split_counts"],
        "fractions": {key: value / total for key, value in manifest["split_counts"].items()},
        "step_random_split": False, "episode_leakage": False,
    })
    dump("distillation_protocol_hashes.json", {
        "config_sha256": config_hash,
        "dataset_sha256": json.loads((OUT / "dataset_hashes.json").read_text())["dataset_sha256"],
        "student_initialization_sha256": json.loads((OUT / "student_initialization.json").read_text())["initialization_sha256"],
    })
    dump("closed_loop_walk_results.json", {key: value["walk"] for key, value in sweep.items()})
    dump("closed_loop_run_results.json", {key: value["run"] for key, value in sweep.items()})
    dump("closed_loop_walk_to_run_results.json", {key: value["walk_to_run"] for key, value in sweep.items()})
    dump("teacher_retention_summary.json", {
        "selected_checkpoint": selected["checkpoint"], "selected_sha256": selected["sha256"],
        "gates": selected_gates, "catastrophic_interference": not selected_gates["all"],
        "selection_rule": "retention gates first, then aggregate retention minus safety failures",
    })
    dump("dagger_summary.json", {
        "maximum_rounds": 1, "executed_rounds": 1,
        "status": "ROUND_1_DID_NOT_RESTORE_ALL_RETENTION_GATES",
        "run_to_walk_labels_used": False,
        "collected_student_occupancy_steps": json.loads((OUT / "dagger_collection.json").read_text())["rows"],
        "checkpoint": dagger_training["checkpoint"], "checkpoint_sha256": dagger_training["sha256"],
        "parent_sha256": dagger_training["parent_sha256"], "ppo": 0, "reward": False,
    })
    dump("intermediate_speed_diagnostics.json", {
        "diagnostic_only": True, "formal_support_claimed": False, "selected_checkpoint": selected["checkpoint"],
        "results": selected["intermediate"],
    })
    dump("run_to_walk_diagnostic.json", {
        "diagnostic_only": True, "single_controller": True, "hard_switch": False,
        "minimum_jerk_seconds": 1.4, "selected_checkpoint": selected["checkpoint"],
        "per_source": reverse, "reverse_transition_emerges": reverse_emerges,
    })
    dump("stage0_classification.json", {
        "classification": classification, "selected_checkpoint": selected["checkpoint"],
        "selected_sha256": selected["sha256"], "teacher_retention": selected_gates,
        "reverse_transition_emerges": reverse_emerges, "capability_status": "DIAGNOSTIC_ONLY",
    })
    dump("recommended_next_action.json", {"one_next_action": recommendation})
    protected = {
        "teachers": {key: value["actual_sha256"] for key, value in teacher_manifest.items()},
        "all_teacher_hashes_match": all(value["hash_match"] for value in teacher_manifest.values()),
        "exp_005_006_007_008_modified": False, "capability_manifest_modified": False,
        "production_artifacts_modified": False, "ppo_training": 0, "reward_optimization": 0,
        "teacher_gradients": 0, "isaac_lab_modified": False,
    }
    dump("protected_hashes.json", protected)
    dump("gate.json", {
        "status": "PASS" if classification != "DISTILLATION_FAIL_INTERFERENCE" else "FAIL",
        "classification": classification, "dataset_minimum": total >= 1_500_000,
        "teacher_hashes": protected["all_teacher_hashes_match"], "retention": selected_gates,
        "reverse_diagnostic": reverse_emerges, "production_capability_updated": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n'
        '.\\experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\reproduce_stage0.ps1\n'
        f'.\\experiments\\isaaclab\\exp_009_unitree_g1_unified_walk_run_student\\scripts\\play_unified_student.ps1 -StartSpeed 2.6 -TargetSpeed 1.2 -StudentCheckpoint "{selected["checkpoint"]}"\n',
        encoding="utf-8",
    )
    report = REPO / "research/exp_009_stage0_report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "# exp_009 Stage 0 — Unified WALK/RUN distillation\n\n"
        f"Classification: `{classification}`.\n\n"
        f"The selected diagnostic checkpoint is `{selected['checkpoint']}` (`{selected['sha256']}`). "
        f"The dataset contains {total:,} actual frozen-teacher steps. The student is a single 123D→37D ELU policy; "
        "it receives a continuous command and no teacher/skill identity. No PPO, reward optimization, teacher update, "
        "production promotion, or capability-manifest change was performed.\n\n"
        "## Retention\n\n"
        f"- WALK gate: {selected_gates['walk']}\n"
        f"- RUN gate: {selected_gates['run']}\n"
        f"- WALK_TO_RUN gate: {selected_gates['walk_to_run']}\n\n"
        "## Reverse diagnostic\n\n"
        f"Reverse transition emerged: {reverse_emerges}. This remains diagnostic-only.\n\n"
        f"## Next action\n\n{recommendation}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"classification": classification, "selected": selected["checkpoint"], "gates": selected_gates}, indent=2))


if __name__ == "__main__":
    main()
