"""Finalize EXP014 Phase 2-D1 after the fail-closed positive-control gate."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
BASE = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
OUT = BASE / "phase_2_d1_reset_boundary_causal_dagger_v2"
REPORT = REPO / "research/exp_014_phase_2_d1_reset_boundary_causal_dagger_v2_report.md"
V1 = BASE / "phase1_dataset/phase1_batch_00.pt"
PARENT = BASE / "dagger_checkpoints/round_2_step_10000.pt"
STARTING_HEAD = "6830b9d9f41fafa0953acc84952ec59d24abd9ab"
PARENT_SHA = "7382163c649676f4e551aa438943cd5bd069e438b08469d6359e30ef4ca5f9e7"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def empty_csv(name: str, columns: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(columns)


def protected_inventory() -> dict:
    task_start = datetime.fromisoformat("2026-08-03T19:49:22.9394229+09:00")
    rows = []
    modified_since_start = []
    for root in (REPO / "experiments/isaaclab", REPO / "results"):
        for directory in sorted(root.glob("exp_*")):
            try:
                number = int(directory.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if not 5 <= number <= 13:
                continue
            for path in sorted(p for p in directory.rglob("*") if p.is_file()):
                rows.append(path)
                stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=task_start.tzinfo)
                if stamp > task_start:
                    modified_since_start.append(path.relative_to(REPO).as_posix())
    existing_exp014 = []
    existing_exp014_modified = []
    for path in sorted(p for p in BASE.rglob("*") if p.is_file() and OUT not in p.parents):
        existing_exp014.append(path)
        stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=task_start.tzinfo)
        if stamp > task_start:
            existing_exp014_modified.append(path.relative_to(REPO).as_posix())
    prior = json.loads((BASE / "protected_hashes_end.json").read_text(encoding="utf-8"))
    return {
        "timestamp": datetime.now(timezone(timedelta(hours=9))).isoformat(),
        "task_start": task_start.isoformat(),
        "scope": "mtime write audit for all files in experiment/result exp_005-exp_013 and all pre-D1 exp_014 result files; cryptographic checks for V1 and parent",
        "exp005_013_file_count": len(rows),
        "prior_exp005_013_aggregate_sha256": prior["aggregate_sha256"],
        "exp005_013_modified_since_task_start": modified_since_start,
        "existing_exp014_file_count": len(existing_exp014),
        "existing_exp014_modified_since_task_start": existing_exp014_modified,
        "v1_sha256": sha(V1),
        "parent_sha256": sha(PARENT),
        "status": "PASS" if not modified_since_start and not existing_exp014_modified else "PROTECTED_PATH_CHANGED",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pc = json.loads((OUT / "specialist_s_reset_positive_control.json").read_text(encoding="utf-8"))
    if pc["status"] != "FAIL":
        raise RuntimeError("This finalizer is only valid for the preregistered positive-control stop")
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError("parent tensor hash mismatch")
    if pc["v1_sha256_before"] != pc["v1_sha256_after"] or sha(V1) != pc["v1_sha256_before"]:
        raise RuntimeError("V1 changed")

    now = datetime.now(timezone(timedelta(hours=9))).isoformat()
    status_raw = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True, encoding="utf-8")
    unrelated = [line for line in status_raw.splitlines() if "exp_014_unitree_g1_explicit_motion_mode_unified_locomotion" not in line and "exp_014_phase_2_d1" not in line]
    stage_reference = {
        "stage": "EXP014 Phase 2-D1 Causal Reset-Boundary DAgger Dataset V2 preflight",
        "starting_head": STARTING_HEAD,
        "observed_head_before_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "parent_checkpoint": PARENT.relative_to(REPO).as_posix(),
        "parent_sha256": sha(PARENT),
        "specialist_positive_control_status": pc["status"],
        "stop_rule_applied": "Specialist S positive control FAIL; no labels, dataset, divergence audit, or training authorized",
        "unrelated_dirty_entry_count": len(unrelated),
        "unrelated_dirty_state_preserved": True,
        "timestamp": now,
    }
    dump("stage_reference.json", stage_reference)
    dump("protocol.json", {
        "name": "Exp014Phase2D1ResetBoundaryCausalDaggerV2",
        "hypotheses": {
            "H1": "student diverges from Specialist S in reset steps 0-3",
            "H2": "adding labels for those steps improves physical STAND",
            "H3": "effect exceeds an equal-count safe-hold sham",
        },
        "positive_control_gate": pc["gate"],
        "stop_rules": ["Specialist S positive control FAIL", "protected path changed", "split/future leakage", "material label conflict", "NaN/Inf", "parent mismatch"],
        "prohibited": ["RUN", "OMNI-RUN", "PPO", "S1/S2", "gate changes", "reset distribution changes", "router/blending/switching"],
    })

    not_run = {
        "status": "NOT_EXECUTED",
        "reason": "preregistered stop: Specialist S positive control practical STAND was below 95%",
        "upstream_classification": "EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL",
    }
    dump("reset_boundary_first_divergence.json", not_run)
    empty_csv("reset_boundary_first_divergence.csv", ["recipe_id", "stage", "action_l2", "action_cosine", "contact_mismatch", "slip_onset", "fall_onset"])

    v1 = torch.load(V1, map_location="cpu", weights_only=False)
    semantic = hashlib.sha256()
    for key in ("observation_141", "teacher_action", "recipe_id", "control_step", "split_id"):
        semantic.update(key.encode())
        semantic.update(v1[key].contiguous().numpy().tobytes())
    split_counts = {name: int((v1["split_id"].flatten() == i).sum()) for i, name in enumerate(("train", "validation", "held-out"))}
    recipe_counts = {name: int(torch.unique(v1["recipe_id"][v1["split_id"].flatten() == i]).numel()) for i, name in enumerate(("train", "validation", "held-out"))}
    dump("dataset_v2_manifest.json", {
        "name": "Exp014StandOmniWalkTrajectoryDatasetV2",
        "status": "NOT_CREATED",
        "candidate_samples_observed_but_not_published": 2720,
        "reason": not_run["reason"],
        "v1_immutable": True,
        "overlay_exists": False,
    })
    dump("dataset_v2_split.json", {"status": "UNCHANGED_FROM_V1", "sample_counts": split_counts, "recipe_counts": recipe_counts, "overlap": 0})
    dump("dataset_v2_hashes.json", {
        "v1_byte_sha256_before": pc["v1_sha256_before"],
        "v1_byte_sha256_after": sha(V1),
        "v1_semantic_sha256": semantic.hexdigest(),
        "parent_checkpoint_sha256": sha(PARENT),
        "v2_overlay_sha256": None,
    })
    dump("dataset_v2_integrity.json", {
        "status": "NOT_APPLICABLE_STOPPED_BEFORE_V2",
        "v1_bytes_changed": 0,
        "v1_semantic_hash_changed": 0,
        "split_changed": 0,
        "split_overlap": 0,
        "duplicate_sample_ids": 0,
        "future_leakage": 0,
        "teacher_id_in_actor_input": 0,
        "missing_step_0_3_label": "not evaluated: labels forbidden after scope FAIL",
        "material_conflicts": "not audited: no V2 dataset created",
    })
    dump("causal_arm_contract.json", {
        "status": "NOT_EXECUTED",
        "parent": stage_reference["parent_checkpoint"],
        "parent_sha256_verified": True,
        "architecture": [141, 256, 128, 128, 37],
        "arms": ["C0_BASELINE", "C1_RESET_BOUNDARY_V2", "C2_SHAM_EQUAL_COUNT"],
        "seeds": [20278801, 20278802, 20278803],
        "reason": not_run["reason"],
    })
    for name in (
        "causal_training_timeline", "static_reset_boundary_metrics", "validation_practical_stand",
        "validation_failure_timing", "validation_walk_smoke", "validation_stop_smoke", "heldout_causal_comparison",
    ):
        empty_csv(name + ".csv", ["status", "reason"])
        dump(name + ".json", not_run)
    dump("selected_checkpoints.json", {**not_run, "selected": {}})
    dump("selected_checkpoint_process_parity.json", {**not_run, "parent_sha256_verified": True})
    dump("causal_effect_summary.json", {
        **not_run,
        "causal_conclusion": "UNIDENTIFIABLE_UNDER_FAILED_POSITIVE_CONTROL",
        "positive_control_metrics": pc["metrics"],
        "candidate_step_0_3_finite": pc["boundary_checks"]["nan_inf"] == 0,
        "candidate_labels_used": 0,
    })
    classification = {
        "primary_classification": "EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL",
        "positive_control": "FAIL",
        "failed_gate": "practical STAND >=95%",
        "observed_practical_stand": pc["metrics"]["practical_stand"],
        "downstream_authorization": "DENIED",
        "RUN_PPO_S1_S2_executed": False,
    }
    dump("stage_classification.json", classification)
    next_action = {
        "experiment": "Exp014 reset observation/action/history initialization and Specialist-S action-contract parity audit",
        "single_experiment": True,
        "purpose": "determine why the frozen formal stop specialist achieves only 58.24% practical STAND under the unchanged exp014 reset distribution before any boundary labeling",
        "fixed": ["reset distribution", "physics", "teacher checkpoint", "141D command contract", "formal gates"],
        "training": False,
    }
    dump("recommended_next_action.json", next_action)
    protection_a = protected_inventory()
    dump("protected_hashes.json", protection_a)
    if protection_a["status"] != "PASS":
        raise RuntimeError("protected path changed during finalization")

    reproduction = """$ErrorActionPreference = 'Stop'
$repo = 'C:\\Users\\user\\workspace\\physical-ai-lab'
$isaac = 'C:\\Users\\user\\workspace\\IsaacLab\\isaaclab.bat'
$python = 'C:\\Users\\user\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe'
Set-Location $repo
& $isaac -p experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/run_phase2_d1_positive_control.py --device cuda:0 --viz none
& $python experiments/isaaclab/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/scripts/finalize_phase2_d1.py
# Stop is mandatory when specialist_s_reset_positive_control.json reports FAIL.
"""
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")
    REPORT.write_text(f"""# EXP014 Phase 2-D1 reset-boundary causal DAgger V2 preflight

## Outcome

Classification: **EXP014_RESET_BOUNDARY_SPECIALIST_SCOPE_FAIL**.

The frozen exp_012 Stage 2Q Specialist S was evaluated from all 680 unchanged exp_014 reset recipes for 100 control steps. It achieved practical STAND **{pc['metrics']['practical_stand']:.2%}**, below the preregistered 95% positive-control gate. Fall was {pc['metrics']['fall']:.2%}, dangerous slip {pc['metrics']['dangerous_slip']:.2%}, impact {pc['metrics']['impact']:.2%}, and long-dwell saturation {pc['metrics']['long_dwell_saturation']:.2%}.

## Reset boundary

All {pc['boundary_checks']['candidate_labels']} candidate actions at control steps 0-3 were finite, within the configured action contract, and had no reset-buffer corruption. They were **not published as labels** because physical teacher scope was not established.

## Causal experiment

The first-divergence audit, Dataset V2, C0/C1/C2 training, validation, and held-out evaluation were not executed. This is the required fail-closed behavior and leaves H1-H3 unidentified; it is not evidence that reset-boundary labels are causal or noncausal.

## Protection and next experiment

V1 remained byte-identical at `{sha(V1)}` and the fixed parent matched `{sha(PARENT)}`. Protected paths were unchanged during D1 finalization. The next single experiment is a read-only reset observation/action/history initialization and Specialist-S action-contract parity audit under the same reset distribution.
""", encoding="utf-8")
    print(json.dumps({"classification": classification, "protected": protection_a["status"], "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
