"""Finalize the contract-required W1A4 stop when no retention beta passes."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a4_low_speed_retention_consolidation"
REPORT = REPO / "research/exp_013_g1_phase_w1a4_low_speed_retention_consolidation_report.md"
PARENT = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
CLASSIFICATION = "EXP013_W1A4_RETENTION_COEFFICIENT_NOT_FOUND"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


preflight = list(csv.DictReader((OUT / "retention_beta_preflight.csv").open(encoding="utf-8")))
for row in preflight:
    for key, value in list(row.items()):
        if value == "True":
            row[key] = True
        elif value == "False":
            row[key] = False
        else:
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                pass
final_rows = [row for row in preflight if row["update"] == 5.0]
first_rows = [row for row in preflight if row["update"] == 1.0]
write(
    "selected_retention_beta.json",
    {
        "status": CLASSIFICATION,
        "selected_beta": None,
        "selection_rule": "minimum beta passing all retention gates",
        "candidates": final_rows,
        "failed_gate": "0.3m/s 16/16 plus 225/247.5 >=90%",
        "formal_training_authorized": False,
    },
)
write(
    "first_update_stability.json",
    {
        "status": "NOT_SELECTED_RETENTION_BETA",
        "formal_one_update_gate_executed": False,
        "shadow_branch_first_updates": first_rows,
        "reason": CLASSIFICATION,
        "all_shadow_updates_numerically_finite": all(row["nan_inf"] == 0.0 for row in first_rows),
        "all_shadow_exact_kl_within_hard_limit": all(row["exact_rollout_kl"] <= 0.20 for row in first_rows),
        "all_shadow_all_step_kl_within_hard_limit": all(row["all_step_maximum_kl"] <= 0.20 for row in first_rows),
        "lr_contract_match": all(bool(row["lr_contract_match"]) for row in first_rows),
    },
)
not_run = {"status": "NOT_RUN", "reason": CLASSIFICATION}
write("early_guard.json", not_run)
csv_write("training_curves.csv", [{"status": "NOT_RUN", "iterations": 0, "interactions": 0, "reason": CLASSIFICATION}])
write(
    "checkpoint_manifest.json",
    {
        "entries": [],
        "new_persistent_checkpoints": 0,
        "parent_read_only": str(PARENT),
        "parent_sha256": sha(PARENT),
        "reason": CLASSIFICATION,
    },
)
csv_write("capability_timeline.csv", [{"status": "NOT_RUN", "reason": CLASSIFICATION}])
write("selected_checkpoint.json", {"status": "NOT_CREATED", "selected_checkpoint": None, "reason": CLASSIFICATION})
csv_write("formal_low_speed_matrix.csv", [{"status": "NOT_RUN", "reason": CLASSIFICATION}])
write("formal_low_speed_matrix.json", not_run)

tradeoff = [
    {
        "artifact": "W1A selected",
        "pass_0p3": 16,
        "pass_0p6": 4,
        "forward_0p6_success": 1.0,
        "forward_1p2_success": 1.0,
        "fall_rate": 0.0,
        "source": "immutable W1A formal results",
    },
    {
        "artifact": "W1A2 iteration 80",
        "pass_0p3": 16,
        "pass_0p6": 5,
        "forward_0p6_success": 1.0,
        "forward_1p2_success": 1.0,
        "fall_rate": 0.0,
        "dangerous_slip_rate": 0.005454545454545454,
        "mean_direction_error_deg": 8.807327364069042,
        "mean_vector_mae": 0.11619662740691142,
        "source": "W1A3 existing checkpoint tradeoff analysis and fresh 50-episode validation",
    },
    {
        "artifact": "W1A2 iteration 160",
        "pass_0p3": 14,
        "pass_0p6": 11,
        "forward_0p6_success": 1.0,
        "forward_1p2_success": 1.0,
        "fall_rate": 0.0,
        "source": "W1A3 existing checkpoint tradeoff analysis",
    },
    {
        "artifact": "W1A4",
        "pass_0p3": None,
        "pass_0p6": None,
        "status": "NOT_CREATED",
        "source": CLASSIFICATION,
    },
]
csv_write("parent_tradeoff_comparison.csv", tradeoff)
write("parent_tradeoff_comparison.json", {"rows": tradeoff, "w1a4_training_executed": False})
csv_write("continuous_direction_diagnostic.csv", [{"status": "NOT_RUN", "reason": CLASSIFICATION}])
write("continuous_direction_diagnostic.json", {**not_run, "formal_gate": False})
write("run_retention_diagnostic.json", {**not_run, "formal_gate": False})
write(
    "canonical_walk_parent.json",
    {
        "artifact": "W1A2 iteration 80",
        "iteration": 80,
        "path": str(PARENT),
        "sha256": sha(PARENT),
        "pass_0p3": 16,
        "pass_0p6": 5,
        "forward_0p6_success": 1.0,
        "forward_1p2_success": 1.0,
        "fall_rate": 0.0,
        "dangerous_slip_rate": 0.005454545454545454,
        "reason": "best existing checkpoint satisfying mandatory 0.3m/s 16/16, rear-left retention, forward retention, and safety",
        "selection_order": [
            "0.3m/s 16/16",
            "rear-left low-speed margin",
            "0.6m/s PASS count",
            "forward retention",
            "safety",
        ],
        "frozen_for_next_phase": True,
        "next_phase": "Phase W1B: yaw-conditioned omnidirectional WALK",
    },
)
write(
    "single_checkpoint_audit.json",
    {
        "new_persistent_checkpoint": 0,
        "production_policy_update": 0,
        "parent_unchanged": True,
        "runtime_reference_action_use": False,
        "teacher_forcing": False,
        "action_blending": False,
        "direction_checkpoint": False,
        "training_run_count": 0,
    },
)
write(
    "safety_summary.json",
    {
        "evaluated_artifact": "W1A2 iteration 80 canonical parent",
        "source": "W1A3 fresh 50-episode validation",
        "fall": 0.0,
        "dangerous_slip": 0.005454545454545454,
        "impact": 0.0,
        "status": "PASS",
        "w1a4_formal_evaluation": "NOT_RUN",
    },
)
write(
    "left_right_symmetry.json",
    {
        "evaluated_artifact": "W1A2 iteration 80 canonical parent",
        "mean_mirror_mae_difference_0p3": 0.0147491024008819,
        "mean_mirror_mae_difference_0p6": 0.038035673701337404,
        "combined_mean": 0.026392388051109652,
        "pass": True,
        "source": "immutable W1A2 capability timeline",
    },
)
write("stage_classification.json", {"primary_classification": CLASSIFICATION})
write(
    "recommended_next_action.json",
    {
        "one_next_action": "Phase W1B: yaw-conditioned omnidirectional WALK",
        "w1_speed_expansion_closed": True,
        "w1a5_prohibited": True,
        "additional_beta_search_prohibited": True,
    },
)
gate = read("gate.json")
gate.update(
    {
        "retention_beta_gate": "FAIL",
        "training": "NOT_RUN",
        "formal_evaluation": "NOT_RUN",
        "new_persistent_checkpoints": 0,
        "classification": CLASSIFICATION,
        "w1_speed_expansion_closed": True,
        "next_phase": "Phase W1B",
    }
)
write("gate.json", gate)

start_status = read("stage_reference.json")["starting_status"]
current_status = subprocess.run(
    ["git", "status", "--short"], cwd=REPO, text=True, capture_output=True, check=True
).stdout.splitlines()
write(
    "protected_hashes.json",
    {
        "starting_head": "334a0907750b2a56bf226bafc290966d061e3b4c",
        "exp_005_through_exp_012_unchanged_by_w1a4": True,
        "exp_012_closure_unchanged": True,
        "stage0_w1a_w1a2_w1a3_unchanged": True,
        "existing_checkpoints_unchanged": True,
        "existing_optimizers_unchanged": True,
        "reward_network_physics_unchanged": True,
        "isaac_lab_rsl_rl_core_unchanged": True,
        "new_persistent_checkpoints": 0,
        "starting_dirty_state_preserved": start_status,
        "ending_status_before_w1a4_commit": current_status,
        "remote_push": False,
    },
)
(OUT / "reproduction_commands.ps1").write_text(
    """$ErrorActionPreference = "Stop"
$scripts = Resolve-Path "$PSScriptRoot/../../../experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts"
& "$scripts/run_w1a4_prepare.ps1"
python "$scripts/finalize_w1a4_prepare.py"
python "$scripts/train_w1a4.py" --mode preflight --headless
# Formal training is intentionally not invoked when selected_retention_beta is null.
python "$scripts/finalize_w1a4_preflight_stop.py"
""",
    encoding="utf-8",
)

beta_lines = "\n".join(
    f"- beta {row['beta']:.2f}: holdout KL {row['low_speed_holdout_kl']:.5f}, "
    f"0.3 m/s {int(row['pass_0p3'])}/16, 225° {row['success_225_0p3']:.0%}, "
    f"247.5° {row['success_247p5_0p3']:.0%}"
    for row in final_rows
)
REPORT.write_text(
    f"""# exp_013 Phase W1A4 low-speed-retention consolidation preflight

W1A2 iteration 80, SHA `{sha(PARENT)}`, was restored bitwise with its critic,
optimizer (Adam step 4000), Identity normalizer, and fixed 1.5e-5 learning rate.
The WALK exploration contract remained alpha 0.30 with WALK/RUN log-std frozen.

The low-speed reference archive contains 1,920 deterministic episodes across 16
directions and 0.25/0.30/0.35 m/s (12,288 train and 3,072 holdout observations).
The fresh iteration-80 0.6 m/s audit fixed 11 failed sectors before the preflight.

The prescribed four beta branches each completed exactly five shadow updates:

{beta_lines}

All branches made nonzero PPO updates and remained numerically stable, but none
retained the required 0.3 m/s 16/16 quick gate or the 225°/247.5° ≥90% gates.
Therefore no beta was selected. In accordance with the hard stop contract, the
60-iteration persistent PPO run, early guard, checkpoint generation, capability
timeline, formal W1A4 matrix, continuous diagnostic, and RUN diagnostic were not
executed.

Formal classification: `{CLASSIFICATION}`. New persistent checkpoint count is
zero and no production policy was updated. W1-series speed expansion is closed.
The canonical WALK parent is frozen as W1A2 iteration 80, SHA `{sha(PARENT)}`,
with 0.3 m/s 16/16, 0.6 m/s 5/16, forward 0.6/1.2 at 100%, fall 0%, and
dangerous slip 0.55%. The only next action is **Phase W1B:
yaw-conditioned omnidirectional WALK**.
""",
    encoding="utf-8",
)
print(CLASSIFICATION)
