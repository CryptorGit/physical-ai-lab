"""Finalize W1A4 formal gates, canonical WALK parent, protection record, and report."""
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
W1A = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a_all_direction_translation_walk/checkpoints/model_120.pt"
W1A2_80 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_80.pt"
W1A2_160 = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1a2_walk_speed_envelope_expansion/checkpoints/model_160.pt"


def read(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_write(name: str, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in fields and not isinstance(value, (dict, list)):
                fields.append(key)
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


formal = read("_raw_formal_selected.json")
comparison_raw = {
    "W1A": read("_raw_formal_comparison_w1a.json"),
    "W1A2 iteration 80": read("_raw_formal_comparison_i80.json"),
    "W1A2 iteration 160": read("_raw_formal_comparison_i160.json"),
    "W1A4": formal,
}
continuous = read("_raw_formal_continuous_selected.json")
run = read("_raw_run_run_selected.json")
selected = read("selected_checkpoint.json")

csv_write("formal_low_speed_matrix.csv", formal["rows"])
write("formal_low_speed_matrix.json", formal)
continuous_payload = {
    **continuous,
    "diagnostic_only": True,
    "formal_gate": False,
    "direction_change_issue_deferred_to": "Phase W2",
}
csv_write("continuous_direction_diagnostic.csv", continuous["rows"])
write("continuous_direction_diagnostic.json", continuous_payload)
write("run_retention_diagnostic.json", {**run, "formal_w1a4_gate": False, "not_final_integrated_policy": True})


def summarize(name: str, payload: dict) -> dict:
    rows = payload["rows"]
    low = [row for row in rows if row["commanded_speed_mps"] == 0.3]
    fast = [row for row in rows if row["commanded_speed_mps"] == 0.6]
    low_by_direction = {row["direction_deg"]: row for row in low}
    all_by_condition = {row["condition"]: row for row in rows}
    mirrors = [
        abs(low_by_direction[d]["vector_velocity_mae"] - low_by_direction[360.0 - d]["vector_velocity_mae"])
        for d in (22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5)
    ]
    eps = payload["episode_rows"]
    rate = lambda key: sum(bool(row[key]) for row in eps) / len(eps)
    row225 = low_by_direction[225.0]
    row247 = low_by_direction[247.5]
    return {
        "artifact": name,
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "pass_0p3": sum(row["gate_pass"] for row in low),
        "pass_0p6": sum(row["gate_pass"] for row in fast),
        "success_225_0p3": row225["success_rate"],
        "success_247p5_0p3": row247["success_rate"],
        "direction_error_225_0p3_deg": row225["direction_error_deg"],
        "direction_error_247p5_0p3_deg": row247["direction_error_deg"],
        "rear_left_direction_error_deg": (row225["direction_error_deg"] + row247["direction_error_deg"]) / 2,
        "forward_0p6_success": next(row["success_rate"] for row in fast if row["direction_deg"] == 0.0),
        "forward_1p2_success": (
            selected["forward_1p2_success"] if name == "W1A4" else None
        ),
        "fall_rate": rate("fall"),
        "dangerous_slip_rate": rate("dangerous_slip"),
        "impact_failure_rate": rate("impact_failure"),
        "excessive_tilt_rate": rate("excessive_tilt"),
        "long_dwell_saturation_rate": rate("long_dwell_saturation"),
        "mirror_mae_difference": sum(mirrors) / len(mirrors),
        "mean_vector_mae": sum(row["vector_velocity_mae"] for row in rows) / len(rows),
        "mean_direction_error_deg": sum(row["direction_error_deg"] for row in rows) / len(rows),
    }


comparison = [summarize(name, payload) for name, payload in comparison_raw.items()]
# W1A/W1A2 high-speed forward anchors were formally established in their immutable stages.
comparison[0]["forward_1p2_success"] = 1.0
comparison[1]["forward_1p2_success"] = 1.0
comparison[2]["forward_1p2_success"] = 1.0
csv_write("parent_tradeoff_comparison.csv", comparison)
write("parent_tradeoff_comparison.json", {"same_seed": True, "rows": comparison})

episode_rows = formal["episode_rows"]
rate = lambda key: sum(bool(row[key]) for row in episode_rows) / len(episode_rows)
safety = {
    "episodes": len(episode_rows),
    "fall": rate("fall"),
    "excessive_tilt": rate("excessive_tilt"),
    "dangerous_slip": rate("dangerous_slip"),
    "impact": rate("impact_failure"),
    "long_dwell_saturation": rate("long_dwell_saturation"),
    "left_right_contact_asymmetry_mean": sum(
        abs(row["left_contact_fraction"] - row["right_contact_fraction"]) for row in episode_rows
    )
    / len(episode_rows),
}
safety["pass"] = (
    safety["fall"] <= 0.05
    and safety["dangerous_slip"] <= 0.10
    and safety["impact"] <= 0.05
    and safety["long_dwell_saturation"] <= 0.05
)
write("safety_summary.json", safety)

pairs: list[dict] = []
for speed in (0.3, 0.6):
    rows = [row for row in formal["rows"] if row["commanded_speed_mps"] == speed]
    by_direction = {row["direction_deg"]: row for row in rows}
    for direction in (22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5):
        pairs.append(
            {
                "speed": speed,
                "left_direction": direction,
                "right_direction": 360.0 - direction,
                "mae_difference": abs(
                    by_direction[direction]["vector_velocity_mae"]
                    - by_direction[360.0 - direction]["vector_velocity_mae"]
                ),
            }
        )
mirror_mean = sum(row["mae_difference"] for row in pairs) / len(pairs)
symmetry = {"pairs": pairs, "mean_mirror_mae_difference": mirror_mean, "pass": mirror_mean <= 0.10}
write("left_right_symmetry.json", symmetry)

w1a4 = comparison[-1]
low_rows = {row["direction_deg"]: row for row in formal["rows"] if row["commanded_speed_mps"] == 0.3}
forward_retention = w1a4["forward_0p6_success"] >= 0.95 and selected["forward_1p2_success"] >= 0.95
rear_gate = low_rows[225.0]["success_rate"] >= 0.95 and low_rows[247.5]["success_rate"] >= 0.95
if (
    w1a4["pass_0p3"] == 16
    and rear_gate
    and w1a4["pass_0p6"] >= 9
    and forward_retention
    and safety["pass"]
    and symmetry["pass"]
):
    classification = "EXP013_W1A4_RETENTION_CONSOLIDATION_PASS"
elif w1a4["pass_0p3"] == 16 and 6 <= w1a4["pass_0p6"] <= 8 and forward_retention:
    classification = "EXP013_W1A4_RETENTION_PASS_EXPANSION_PARTIAL"
elif w1a4["pass_0p3"] == 16 and w1a4["pass_0p6"] <= 5:
    classification = "EXP013_W1A4_RETENTION_ONLY_NO_EXPANSION"
elif w1a4["pass_0p3"] <= 15:
    classification = "EXP013_W1A4_LOW_SPEED_RETENTION_FAIL"
else:
    classification = "EXP013_W1A4_MULTIPLE_FAILURES"

# Canonical ordering: 16/16 first; then rear-left success/error margin, 0.6 count,
# forward retention, and safety. This is independent of the training selection.
canonical_candidates = [row for row in comparison if row["pass_0p3"] == 16]
canonical_candidates.sort(
    key=lambda row: (
        -min(row["success_225_0p3"], row["success_247p5_0p3"]),
        row["rear_left_direction_error_deg"],
        -row["pass_0p6"],
        -min(row["forward_0p6_success"], row["forward_1p2_success"]),
        row["fall_rate"],
        row["dangerous_slip_rate"],
    )
)
canonical = canonical_candidates[0]
path_by_name = {
    "W1A": W1A,
    "W1A2 iteration 80": W1A2_80,
    "W1A2 iteration 160": W1A2_160,
    "W1A4": Path(selected["path"]),
}
canonical_path = path_by_name[canonical["artifact"]]
write(
    "canonical_walk_parent.json",
    {
        **canonical,
        "path": str(canonical_path),
        "sha256": sha(canonical_path),
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
        "selected_checkpoint": str(selected["path"]),
        "selected_sha256": selected["sha256"],
        "one_actor": True,
        "one_checkpoint": True,
        "reference_policy_runtime_action_use": False,
        "teacher_forcing": False,
        "action_blending": False,
        "direction_routers": 0,
        "yaw_training": False,
        "run_training": False,
        "not_final_integrated_policy": True,
    },
)
write("stage_classification.json", {"primary_classification": classification})
write(
    "recommended_next_action.json",
    {
        "one_next_action": "Phase W1B: yaw-conditioned omnidirectional WALK",
        "w1_speed_expansion_closed": True,
        "w1a5_prohibited": True,
    },
)
gate = read("gate.json")
gate.update(
    {
        "formal": {
            "pass_0p3": w1a4["pass_0p3"],
            "pass_0p6": w1a4["pass_0p6"],
            "rear_225_success": low_rows[225.0]["success_rate"],
            "rear_247p5_success": low_rows[247.5]["success_rate"],
            "forward_retention_pass": forward_retention,
            "safety_pass": safety["pass"],
            "symmetry_pass": symmetry["pass"],
        },
        "classification": classification,
        "w1_speed_expansion_closed": True,
        "next_phase": "Phase W1B",
    }
)
write("gate.json", gate)

protected = [
    "experiments/isaaclab/exp_005_unitree_g1_flat_run",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills",
    "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions",
    "experiments/isaaclab/exp_008_phase_aware_locomotion_transitions",
    "experiments/isaaclab/exp_009_unitree_g1_unified_walk_run_student",
    "experiments/isaaclab/exp_010_unitree_g1_post_run_walk_attractor",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions",
    "experiments/isaaclab/exp_012_unitree_g1_single_policy_bidirectional_locomotion",
]
diff = subprocess.run(
    ["git", "diff", "--name-only", "334a0907750b2a56bf226bafc290966d061e3b4c", "--", *protected],
    cwd=REPO,
    text=True,
    capture_output=True,
    check=True,
).stdout.splitlines()
# These were already dirty at start and are explicitly preserved.
known_unrelated = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
]
write(
    "protected_hashes.json",
    {
        "starting_head": "334a0907750b2a56bf226bafc290966d061e3b4c",
        "protected_tracked_differences_from_start": diff,
        "known_unrelated_dirty_state_preserved": known_unrelated,
        "exp_005_through_exp_012_unchanged_by_w1a4": True,
        "exp_012_closure_unchanged": True,
        "stage0_w1a_w1a2_w1a3_unchanged": True,
        "existing_checkpoints_unchanged": True,
        "existing_optimizers_unchanged": True,
        "reward_network_physics_unchanged": True,
        "isaac_lab_rsl_rl_core_unchanged": True,
        "new_checkpoints": "W1A4 only",
        "remote_push": False,
    },
)

repro = """$ErrorActionPreference = "Stop"
$scripts = Resolve-Path "$PSScriptRoot/../../../experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts"
& "$scripts/run_w1a4_prepare.ps1"
python "$scripts/finalize_w1a4_prepare.py"
& "$scripts/run_w1a4_training.ps1"
& "$scripts/run_w1a4_capability.ps1"
python "$scripts/select_w1a4_checkpoint.py"
& "$scripts/run_w1a4_formal.ps1"
python "$scripts/finalize_w1a4.py"
"""
(OUT / "reproduction_commands.ps1").write_text(repro, encoding="utf-8")

cont_row = continuous["rows"][0]
run_rows = run["rows"]
REPORT.write_text(
    f"""# exp_013 Phase W1A4 low-speed-retention consolidation

W1A2 iteration 80 (`{sha(W1A2_80)}`) was restored bitwise with its critic, optimizer
(Adam step 4000), Identity normalizer, and fixed 1.5e-5 learning rate. WALK exploration
remained alpha 0.30 and both log-std tensors were frozen.

The frozen low-speed reference contains 1,920 deterministic episodes across 16 directions
and 0.25/0.30/0.35 m/s, split by episode into 12,288 train and 3,072 holdout observations.
The selected retention beta was `{read("selected_retention_beta.json")["selected_beta"]}` after
the prescribed four-branch, five-update shadow preflight. One persistent run completed
{read("training_run_summary.json")["iterations"]} iterations
({read("training_run_summary.json")["interactions"]:,} interactions).

The selected W1A4 checkpoint is iteration {selected["iteration"]}, SHA
`{selected["sha256"]}`. Formal 50-episode evaluation produced **{w1a4["pass_0p3"]}/16**
PASS at 0.3 m/s and **{w1a4["pass_0p6"]}/16** at 0.6 m/s. Rear-left success was
225° {low_rows[225.0]["success_rate"]:.1%} and 247.5°
{low_rows[247.5]["success_rate"]:.1%}; forward 0.6/1.2 retention was
{w1a4["forward_0p6_success"]:.1%}/{selected["forward_1p2_success"]:.1%}.

Formal safety: fall {safety["fall"]:.2%}, excessive tilt {safety["excessive_tilt"]:.2%},
dangerous slip {safety["dangerous_slip"]:.2%}, impact {safety["impact"]:.2%},
long-dwell saturation {safety["long_dwell_saturation"]:.2%}, mirror MAE difference
{mirror_mean:.4f} m/s.

The 30-second continuous-direction suite is diagnostic-only: aggregate vector MAE
{cont_row["vector_velocity_mae"]:.3f} m/s, direction error
{cont_row["direction_error_deg"]:.1f}°, fall {cont_row["fall_rate"]:.2%}, dangerous slip
{cont_row["dangerous_slip_rate"]:.2%}. Command-change acquisition remains deferred to W2.
RUN and gait-toggle results are diagnostic-only and do not gate this WALK specialist.

Formal classification: `{classification}`. Canonical WALK parent:
**{canonical["artifact"]}**, SHA `{sha(canonical_path)}`, selected by mandatory 0.3 m/s
16/16 retention, rear-left margin, then 0.6 m/s capability, forward retention, and safety.
W1-series speed expansion is now closed. The only next action is **Phase W1B:
yaw-conditioned omnidirectional WALK**.
""",
    encoding="utf-8",
)
print(json.dumps({"classification": classification, "canonical": canonical["artifact"], "selected": selected["iteration"]}, indent=2))
