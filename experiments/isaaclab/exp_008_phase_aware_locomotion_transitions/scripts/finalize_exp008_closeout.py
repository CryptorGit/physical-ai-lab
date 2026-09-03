"""Materialize the deterministic exp_008 final-closeout records."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[4]
OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/final_closeout"
STARTING_HEAD = "10a5510661142d3f99929c7d65219ab9928273ec"


def write(name: str, value: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


capability_path = REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json"
model10 = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage8c_run_to_walk_pilot1_execution/checkpoints/model_10.pt"
stage0_gate = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability/gate.json"

write("final_classification.json", {
    "exp_008": "CLOSED_DIAGNOSTIC_COMPLETE",
    "stage_gate": "PASS_DIAGNOSTIC_COMPLETE",
    "observability": "BREAK_NOT_PREDICTABLE",
    "controllability": "NO_LOCAL_CORRECTION_FOUND",
    "local_phase_aware_correction": "NO_GO",
    "new_production_capability": "NONE",
    "next_embodiment": "UNITREE_GO2",
})
write("observability_summary.json", {
    "classification": "BREAK_NOT_PREDICTABLE",
    "metrics": {
        "full_152d": {"auroc": 0.9801251562142579, "auprc": 0.8450789967416826, "mae_steps": 5.184906482696533},
        "timing_removed_152d": {"auroc": 0.9817267489413599, "auprc": 0.8708283554866092, "mae_steps": 5.184906482696533},
        "legacy_123d": {"auroc": 0.978, "auprc": 0.878, "mae_steps": 5.22},
        "legacy_123d_plus_action": {"auroc": 0.980, "auprc": 0.879, "mae_steps": 5.21},
        "explicit_phase_upper_bound": {"auroc": 0.9812483700416349, "auprc": 0.8724006881507469, "mae_steps": 5.155930995941162},
        "gru_history_16": {"auroc": 0.9893590656301791, "auprc": 0.9182006348385066, "auroc_gain": 0.007632316688819141},
    },
    "time_to_break_mae_gate_steps": 1.5,
    "interpretation": "Break proximity is rankable, but timing accuracy is insufficient for corrective control.",
})
write("controllability_summary.json", {
    "classification": "NO_LOCAL_CORRECTION_FOUND",
    "branch_states": 512,
    "candidate_successes": {
        "baseline": 0, "frozen_walk": 0, "frozen_run": 0,
        "bounded_joint_group": 0, "bounded_walk_alignment": 0,
    },
    "safe_20_step_success_total": 0,
    "phase_limited_successes": 0,
})
write("final_capability_graph.json", {
    "source": "exp_007 formal capabilities",
    "classification": "PARTIAL_SUCCESS_ASYMMETRIC_STATE_GRAPH",
    "graph": "STAND <-> WALK --WALK_TO_RUN--> RUN_LOW",
    "states": {
        "STAND": {"status": "PASS"},
        "WALK": {"status": "PASS", "speeds_mps": [0.6, 0.8, 1.0, 1.2]},
        "RUN_LOW": {"status": "PASS", "steady_speeds_mps": [2.4, 2.6, 2.8]},
    },
    "transitions": {
        "STAND_TO_WALK": "PASS", "WALK_TO_STAND": "PASS",
        "WALK_TO_RUN": {"status": "PASS_LIMITED", "targets_mps": [2.6, 2.8]},
        "WALK_TO_RUN_2.4": "NOT_SUPPORTED", "RUN_TO_WALK": "NO_GO_V1",
    },
})
write("blocked_capabilities.json", {
    "RUN_TO_WALK": "NO_GO_V1",
    "GRAPH_BASED_STOP": "BLOCKED",
    "FULL_BIDIRECTIONAL_GRAPH": "NOT_ACHIEVED",
    "showcase_executes_unsupported_transitions": False,
})
write("next_embodiment_decision.json", {
    "decision": "UNITREE_GO2",
    "project_rationale": "Retest bidirectional gait transitions with fewer degrees of freedom and simpler contact structure.",
    "experimental_claim": "exp_008 does not establish that Go2 will succeed.",
})
write("protected_hashes.json", {
    "capability_manifest": {"path": str(capability_path.relative_to(REPO)), "sha256": sha256(capability_path)},
    "stage8c_model10": {"path": str(model10.relative_to(REPO)), "sha256": sha256(model10)},
    "stage0_gate": {"path": str(stage0_gate.relative_to(REPO)), "sha256": sha256(stage0_gate)},
    "exp005_unchanged": True, "exp006_unchanged": True, "exp007_unchanged": True,
    "exp008_stage0_unchanged": True, "exp009_untouched": True,
    "capability_manifest_unchanged": True, "isaac_lab_core_unchanged": True,
})
write("exp008_closeout.json", {
    "classification": "CLOSED_DIAGNOSTIC_COMPLETE",
    "stage_gate": "PASS_DIAGNOSTIC_COMPLETE",
    "observability": "BREAK_NOT_PREDICTABLE",
    "controllability": "NO_LOCAL_CORRECTION_FOUND",
    "created_new_capability": False,
    "showcase_capability_source": "exp_007",
    "next_embodiment": "Unitree Go2",
    "starting_head": STARTING_HEAD,
    "ending_head": "PENDING_SHOWCASE_COMMIT",
    "commit_hashes": {"closeout": "PENDING", "showcase": "PENDING"},
    "remote_push": False,
})
write("gate.json", {
    "classification": "CLOSED_DIAGNOSTIC_COMPLETE",
    "stage_gate": "PASS_DIAGNOSTIC_COMPLETE",
    "diagnostic_complete": True,
    "local_phase_aware_route": "NO_GO",
    "new_capability": False,
    "showcase_scope": "EXP_007_FORMAL_CAPABILITIES_ONLY",
})

(OUT / "reproduction_commands.ps1").write_text(r'''$ErrorActionPreference = "Stop"
$repo = Resolve-Path (Join-Path $PSScriptRoot "../../..")
Set-Location $repo

# Stage 0 diagnostic reproduction
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\reproduce_stage0.ps1

# GUI
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1

# Record all formal-capability scenes
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1 `
  -Scene All `
  -RecordVideo `
  -OutputPath ".\media\exp_008_closeout"

# Individual scenes
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1 -Scene StandWalkStand
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1 -Scene WalkToRun26
.\experiments\isaaclab\exp_008_phase_aware_locomotion_transitions\scripts\play_exp008_closeout_showcase.ps1 -Scene WalkToRun28
''', encoding="utf-8")

print(json.dumps({"classification": "CLOSED_DIAGNOSTIC_COMPLETE", "starting_head": STARTING_HEAD}, indent=2))
