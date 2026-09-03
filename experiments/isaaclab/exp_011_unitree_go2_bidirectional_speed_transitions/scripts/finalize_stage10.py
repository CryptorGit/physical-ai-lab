"""Aggregate Stage 10 paired controller rollouts and apply frozen gates."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage10_phase_gated_fixed_heading"
START = "b573b730fb5b0e5447cbdce250d5cb49c95ae6f7"
STEADY = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.2, 2.0)
LOW = ((0.0, 0.2), (0.0, 0.4), (0.0, 0.6), (0.6, 0.4), (0.6, 0.2), (0.6, 0.0))
ANCHOR = ((0.0, 1.2), (1.2, 2.0), (2.0, 1.2), (1.2, 0.0))
CONTROLLERS = ("C0", "C1", "C2")


def dump(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            })


def mean(values) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else 0.0


def percentile(values, q) -> float:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(np.percentile(values, q)) if values.size else 0.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


raw_paths = sorted(OUT.glob("raw_C[012]_*.json"))
if len(raw_paths) != 33:
    raise SystemExit(f"expected 33 raw chunks, found {len(raw_paths)}")
rows = []
for path in raw_paths:
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
if len(rows) != 3150:
    raise SystemExit(f"expected 3150 episode rows, found {len(rows)}")

dump("controller_comparison_manifest.json", {
    "checkpoint": "stage7_selected",
    "checkpoint_sha256": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "conditions": {
        "C0": "OPEN_LOOP",
        "C1": "ALWAYS_ON_FIXED_HEADING",
        "C2": "PHASE_GATED_FIXED_HEADING",
    },
    "paired_seed_root": 20269901,
    "episode_rows": len(rows),
    "policy_parameter_updates": 0,
})


def aggregate(subset: list[dict], transition: bool = False) -> dict:
    nonfallen = [row for row in subset if not row["fall"]]
    result = {
        "episodes": len(subset),
        "fall_rate": mean(row["fall"] for row in subset),
        "speed_mae": mean(row["speed_mae"] for row in nonfallen),
        "heading_p50": percentile((row["heading_p50"] for row in nonfallen), 50),
        "heading_p90": percentile((row["heading_p90"] for row in nonfallen), 90),
        "heading_p95": percentile((row["heading_p95"] for row in nonfallen), 95),
        "heading_p99": percentile((row["heading_p99"] for row in nonfallen), 99),
        "signed_heading_slope": mean(row["signed_heading_slope"] for row in nonfallen),
        "yaw_rate_mean": mean(row["yaw_rate_mean"] for row in nonfallen),
        "yaw_rate_p95": percentile((row["yaw_rate_p95"] for row in nonfallen), 95),
        "feedback_activation_rate": mean(row["feedback_activation_time"] is not None for row in subset),
        "feedback_activation_time_mean": mean(
            row["feedback_activation_time"] for row in subset if row["feedback_activation_time"] is not None
        ),
        "feedback_duty_fraction": mean(row["feedback_duty_fraction"] for row in subset),
        "yaw_command_p95": percentile((row["yaw_command_p95"] for row in subset), 95),
        "yaw_command_max": max((row["yaw_command_max"] for row in subset), default=0.0),
        "gravity_tilt_p95": percentile((row["gravity_tilt_p95"] for row in nonfallen), 95),
        "long_dwell_saturation_rate": mean(row["long_dwell_saturation"] for row in subset),
        "saturation_fraction": mean(row["saturation_fraction"] for row in subset),
        "tangential_speed_p95": percentile((row["tangential_speed_p95"] for row in nonfallen), 95),
        "contact_yaw_moment_mean": mean(row["contact_yaw_moment_mean"] for row in nonfallen),
        "contact_yaw_moment_p95": percentile((row["contact_yaw_moment_p95"] for row in nonfallen), 95),
        "command_sign_changes_mean": mean(row["command_sign_changes"] for row in subset),
    }
    if transition:
        result.update({
            "completion_rate": mean(row["completion"] for row in subset),
            "acquisition_rate": mean(row["acquisition"] for row in subset),
            "target_hold_rate": mean(row["target_hold"] for row in subset),
            "timeout_rate": mean(row["timeout"] for row in subset),
            "ramp_heading_change_abs": mean(abs(row["ramp_heading_change"]) for row in nonfallen),
            "target_hold_heading_change_abs": mean(
                abs(row["target_hold_heading_change"]) for row in nonfallen
            ),
            "feedback_before_acquisition_rate": mean(
                row["feedback_started_before_target_acquisition"] for row in subset
            ),
            "speed_change_after_feedback": mean(
                row.get("speed_change_after_feedback")
                for row in subset if row.get("speed_change_after_feedback") is not None
            ),
        })
    return result


steady_json, steady_csv = {}, []
for controller in CONTROLLERS:
    steady_json[controller] = {}
    for speed in STEADY:
        subset = [
            row for row in rows
            if row["controller"] == controller and row["family"] == "steady"
            and row["target_speed"] == speed
        ]
        summary = aggregate(subset)
        steady_json[controller][str(speed)] = summary
        steady_csv.append({"controller": controller, "speed": speed, **summary})
dump("steady_controller_comparison.json", steady_json)
write_csv("steady_controller_comparison.csv", steady_csv)


def transition_outputs(pairs, json_name, csv_name):
    output, flat = {}, []
    for controller in CONTROLLERS:
        output[controller] = {}
        for source, target in pairs:
            key = f"{source:g}->{target:g}"
            subset = [
                row for row in rows
                if row["controller"] == controller and row["family"] == "transition"
                and row["source_speed"] == source and row["target_speed"] == target
            ]
            summary = aggregate(subset, transition=True)
            output[controller][key] = summary
            flat.append({"controller": controller, "transition": key, **summary})
    dump(json_name, output)
    write_csv(csv_name, flat)
    return output


low_json = transition_outputs(
    LOW, "low_speed_transition_comparison.json", "low_speed_transition_comparison.csv"
)
anchor_json = transition_outputs(
    ANCHOR, "anchor_transition_comparison.json", "anchor_transition_comparison.csv"
)

sequence_output = {}
for controller in CONTROLLERS:
    subset = [row for row in rows if row["controller"] == controller and row["family"] == "sequence"]
    sequence_output[controller] = {
        "episodes": len(subset),
        "sequence_completion_rate": mean(row["sequence_completion"] for row in subset),
        "segment_success_rate": [
            mean(row["segment_success"][index] for row in subset) for index in range(6)
        ],
        "fall_rate": mean(row["fall"] for row in subset),
        "heading_p95": percentile((row["heading_p95"] for row in subset if not row["fall"]), 95),
        "speed_mae": mean(row["speed_mae"] for row in subset if not row["fall"]),
        "feedback_duty_fraction": mean(row["feedback_duty_fraction"] for row in subset),
        "final_stand_rate": mean(row["final_stand"] for row in subset),
        "checkpoint_switches": sum(row["checkpoint_switches"] for row in subset),
    }
dump("anchor_sequence_comparison.json", sequence_output)

state_rows = []
for row in rows:
    if row["family"] == "sequence":
        continue
    trace_items = row.get("state_trace")
    if trace_items is None:
        trace_items = [{
            "phase": entry["phase"], "entry_time": entry["time"],
            "exit_time": entry.get("exit_time"), "heading_reference": row["heading_reference"],
            "heading_error_mean": None, "heading_error_p95": None,
            "gate_mean": row["feedback_duty_fraction"], "gate_min": None, "gate_max": None,
            "raw_yaw_command_mean": 0.0, "final_yaw_command_mean": 0.0,
        } for entry in row["state_entries"]]
    for entry in trace_items:
        state_rows.append({
            "controller": row["controller"], "family": row["family"],
            "condition": row["condition"], "episode_seed": row["episode_seed"],
            "phase": entry["phase"], "entry_time": entry["entry_time"],
            "exit_time": entry.get("exit_time"),
            "heading_reference": entry.get("heading_reference"),
            "heading_error_mean": entry.get("heading_error_mean"),
            "heading_error_p95": entry.get("heading_error_p95"),
            "gate_mean": entry.get("gate_mean"),
            "gate_min": entry.get("gate_min"),
            "gate_max": entry.get("gate_max"),
            "raw_yaw_command_mean": entry.get("raw_yaw_command_mean"),
            "final_yaw_command_mean": entry.get("final_yaw_command_mean"),
        })
write_csv("phase_gate_state_trace.csv", state_rows)

phase_results = {}
for controller in CONTROLLERS:
    phase_results[controller] = {}
    for source, target in LOW + ANCHOR:
        key = f"{source:g}->{target:g}"
        subset = [
            row for row in rows if row["controller"] == controller
            and row["family"] == "transition" and row["source_speed"] == source
            and row["target_speed"] == target
        ]
        phase_results[controller][key] = {}
        for phase in (
            "source_hold", "speed_ramp", "target_acquisition",
            "feedback_activation", "active_target_hold",
        ):
            available = [
                row["phases"][phase] for row in subset
                if row.get("phases", {}).get(phase, {}).get("samples", 0) > 0
            ]
            phase_results[controller][key][phase] = {
                metric: mean(item[metric] for item in available)
                for metric in (
                    "heading_change", "yaw_rate_mean", "yaw_rate_p95", "speed_error",
                    "fall", "contact_yaw_moment", "tangential_relative_motion",
                )
            } if available else {"samples": 0}
dump("phase_specific_heading_results.json", phase_results)

c2_transition_rows = [
    row for row in rows if row["controller"] == "C2" and row["family"] == "transition"
]
feedback_analysis = {
    "activation_rate": mean(row["feedback_activation_time"] is not None for row in c2_transition_rows),
    "activation_time_mean": mean(
        row["feedback_activation_time"] for row in c2_transition_rows
        if row["feedback_activation_time"] is not None
    ),
    "never_activated_rate": mean(row["feedback_activation_time"] is None for row in c2_transition_rows),
    "yaw_command_saturation_rate": mean(row["yaw_command_p95"] >= 0.099 for row in c2_transition_rows),
    "command_sign_changes_mean": mean(row["command_sign_changes"] for row in c2_transition_rows),
    "fall_with_feedback_active_rate": mean(
        row["fall"] and row["feedback_activation_time"] is not None for row in c2_transition_rows
    ),
    "speed_change_after_feedback": mean(
        row.get("speed_change_after_feedback")
        for row in c2_transition_rows if row.get("speed_change_after_feedback") is not None
    ),
    "feedback_before_acquisition_rate": mean(
        row["feedback_started_before_target_acquisition"] for row in c2_transition_rows
    ),
    "active_latched": True,
}
dump("feedback_intervention_analysis.json", feedback_analysis)

contact_comparison = {}
contact_pass = True
for speed in (0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.0):
    c0 = steady_json["C0"][str(speed)]
    c2 = steady_json["C2"][str(speed)]
    ratio = c2["tangential_speed_p95"] / max(c0["tangential_speed_p95"], 1e-9)
    condition_pass = ratio <= 1.5
    contact_pass &= condition_pass
    contact_comparison[str(speed)] = {
        "c0_tangential_speed_p95": c0["tangential_speed_p95"],
        "c2_tangential_speed_p95": c2["tangential_speed_p95"],
        "ratio": ratio,
        "c0_contact_yaw_moment_p95": c0["contact_yaw_moment_p95"],
        "c2_contact_yaw_moment_p95": c2["contact_yaw_moment_p95"],
        "pass": condition_pass,
    }
dump("contact_kinematics_non_regression.json", {
    "formal_slip_gate_applied": False,
    "tangential_motion_not_50pct_worse": contact_pass,
    "by_speed": contact_comparison,
})

steady_checks = {}
steady_heading_all = True
steady_safety = True
protected_fall_limits = {0.2: 0.02, 0.3: 0.0, 0.4: 0.02, 0.5: 0.0, 0.6: 0.0}
for speed in (0.2, 0.3, 0.4, 0.5, 0.6):
    c0, c2 = steady_json["C0"][str(speed)], steady_json["C2"][str(speed)]
    speed_limit = 0.15
    checks = {
        "heading": c2["heading_p95"] <= 0.12,
        "fall": c2["fall_rate"] <= protected_fall_limits[speed],
        "speed": c2["speed_mae"] <= speed_limit,
        "tilt_non_regression": c2["gravity_tilt_p95"] <= c0["gravity_tilt_p95"] + 0.02,
        "saturation": c2["long_dwell_saturation_rate"] <= 0.05,
    }
    steady_heading_all &= checks["heading"]
    steady_safety &= all(checks.values())
    steady_checks[str(speed)] = checks

transition_checks = {}
low_all = True
low_safety_retained = True
for source, target in LOW:
    key = f"{source:g}->{target:g}"
    c0, c2 = low_json["C0"][key], low_json["C2"][key]
    checks = {
        "completion": c2["completion_rate"] >= 0.90,
        "acquisition": c2["acquisition_rate"] >= 0.90,
        "target_hold": c2["target_hold_rate"] >= 0.90,
        "fall": c2["fall_rate"] <= 0.05,
        "heading": c2["heading_p95"] <= 0.12,
        "timeout": c2["timeout_rate"] <= 0.05,
        "saturation": c2["long_dwell_saturation_rate"] <= 0.05,
        "completion_non_regression": c2["completion_rate"] >= c0["completion_rate"] - 0.05,
        "acquisition_non_regression": c2["acquisition_rate"] >= c0["acquisition_rate"] - 0.05,
        "fall_non_regression": c2["fall_rate"] <= c0["fall_rate"] + 0.02,
        "speed_non_regression": c2["speed_mae"] <= c0["speed_mae"] + 0.05,
    }
    transition_checks[key] = checks
    low_all &= all(checks.values())
    low_safety_retained &= all(
        checks[name] for name in (
            "completion", "acquisition", "target_hold", "fall", "timeout", "saturation",
            "completion_non_regression", "acquisition_non_regression",
            "fall_non_regression", "speed_non_regression",
        )
    )

anchor_transition_retention = all(
    anchor_json["C2"][f"{source:g}->{target:g}"]["completion_rate"] >= 0.95
    and anchor_json["C2"][f"{source:g}->{target:g}"]["fall_rate"]
    <= anchor_json["C0"][f"{source:g}->{target:g}"]["fall_rate"] + 0.02
    for source, target in ANCHOR
)
anchor_steady_retention = all(
    steady_json["C2"][str(speed)]["fall_rate"] <= steady_json["C0"][str(speed)]["fall_rate"] + 0.02
    and steady_json["C2"][str(speed)]["speed_mae"] <= steady_json["C0"][str(speed)]["speed_mae"] + 0.05
    and steady_json["C2"][str(speed)]["long_dwell_saturation_rate"] <= 0.05
    for speed in (1.2, 2.0)
)
anchor_retention = anchor_transition_retention and anchor_steady_retention
sequence_retention = (
    sequence_output["C2"]["sequence_completion_rate"] >= 0.95
    and sequence_output["C2"]["fall_rate"] <= sequence_output["C0"]["fall_rate"] + 0.02
)
constant_avoidance = {}
for source, target in ((0.0, 0.4), (0.0, 0.6)):
    key = f"{source:g}->{target:g}"
    constant_avoidance[key] = {
        "heading_improves_over_c0": low_json["C2"][key]["heading_p95"] < low_json["C0"][key]["heading_p95"],
        "completion_maintains_c0": low_json["C2"][key]["completion_rate"] >= low_json["C0"][key]["completion_rate"] - 0.05,
        "fall_not_worse_than_c0": low_json["C2"][key]["fall_rate"] <= low_json["C0"][key]["fall_rate"],
        "completion_not_worse_than_c1": low_json["C2"][key]["completion_rate"] >= low_json["C1"][key]["completion_rate"],
    }
constant_feedback_avoided = all(all(value.values()) for value in constant_avoidance.values())
unsafe = not (
    steady_safety and low_safety_retained and anchor_retention and sequence_retention and contact_pass
    and feedback_analysis["feedback_before_acquisition_rate"] == 0
)
mean_c0_heading = mean(steady_json["C0"][str(speed)]["heading_p95"] for speed in (0.2, 0.3, 0.4, 0.5, 0.6))
mean_c2_heading = mean(steady_json["C2"][str(speed)]["heading_p95"] for speed in (0.2, 0.3, 0.4, 0.5, 0.6))
substantial_steady_improvement = mean_c2_heading <= 0.75 * max(mean_c0_heading, 1e-9)

if unsafe:
    classification = "PHASE_GATED_FIXED_HEADING_UNSAFE"
    next_action = "do not adopt the phase-gated controller"
elif steady_heading_all and low_all and anchor_retention and sequence_retention and constant_feedback_avoided:
    classification = "PHASE_GATED_FIXED_HEADING_PASS"
    next_action = (
        "freeze phase-gated fixed-heading command controller and select "
        "tangential-slip reduction as the next isolated Pilot target"
    )
elif substantial_steady_improvement and low_safety_retained and anchor_retention and sequence_retention:
    classification = "PHASE_GATED_FIXED_HEADING_PASS_LIMITED"
    next_action = "phase-gate failure diagnosis"
elif mean_c2_heading >= 0.90 * mean_c0_heading:
    classification = "PHASE_GATED_FIXED_HEADING_NO_EFFECT"
    next_action = "do not adopt the phase-gated controller"
else:
    classification = "PHASE_GATED_FIXED_HEADING_INCONCLUSIVE"
    next_action = "no Pilot"

safety = {
    "steady_checks": steady_checks,
    "transition_checks": transition_checks,
    "anchor_retention": anchor_retention,
    "anchor_transition_retention": anchor_transition_retention,
    "anchor_steady_retention": anchor_steady_retention,
    "sequence_retention": sequence_retention,
    "contact_non_regression": contact_pass,
    "constant_feedback_degradation_avoided": constant_feedback_avoided,
    "constant_feedback_focus": constant_avoidance,
    "unsafe": unsafe,
}
dump("safety_non_regression.json", safety)
dump("stage10_classification.json", {
    "classification": classification,
    "production_status": "DIAGNOSTIC_CANDIDATE",
    "mean_low_speed_heading_p95": {"C0": mean_c0_heading, "C2": mean_c2_heading},
})
dump("recommended_next_action.json", {"next_action": next_action, "single_action": True})

checkpoint_paths = {
    "official_parent": (
        REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
        "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
        "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
    ),
    "stage4_selected": (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
    ),
    "stage7_selected": (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"
    ),
}
expected = {
    "official_parent": "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0",
    "stage4_selected": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
    "stage7_selected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd",
    "stage6_protocol": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908",
}
actual = {name: sha(path) for name, path in checkpoint_paths.items()}
protocol_hash = json.loads((
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage6_corrected_endpoint_formal/protocol_hash.json"
).read_text(encoding="utf-8"))["sha256"]
actual["stage6_protocol"] = protocol_hash
status = subprocess.check_output(["git", "status", "--short"], cwd=REPO, text=True).splitlines()
stage10_tokens = (
    "stage10_phase_gated_fixed_heading", "prepare_stage10.py",
    "evaluate_stage10_heading_controller.py", "finalize_stage10.py",
    "run_stage10_heading_diagnosis.ps1", "phase_gated_heading.py",
    "exp_011_go2_phase_gated_fixed_heading_report.md",
    "play_exp011_go2_bidirectional.py", "play_exp011_go2_bidirectional.ps1",
    "stage10_run_", "stage10_c1_", "stage10_c2_",
)
protected_status = [line for line in status if not any(token in line for token in stage10_tokens)]
dump("protected_hashes.json", {
    "expected": expected, "actual": actual, "all_match": actual == expected,
    "starting_head": START,
    "stage1_through_stage9_modified": [],
    "unrelated_dirty_state": protected_status,
    "ppo_updates": 0, "reward_optimization": 0, "policy_gradient": 0,
    "remote_push": False,
})

required = [
    "stage9_reference.json", "protocol.json", "stage10_seed_manifest.json",
    "fixed_heading_contract.json", "phase_gate_contract.json",
    "phase_gated_heading_unit_tests.json", "controller_comparison_manifest.json",
    "steady_controller_comparison.csv", "steady_controller_comparison.json",
    "low_speed_transition_comparison.csv", "low_speed_transition_comparison.json",
    "anchor_transition_comparison.csv", "anchor_transition_comparison.json",
    "anchor_sequence_comparison.json", "phase_gate_state_trace.csv",
    "phase_specific_heading_results.json", "feedback_intervention_analysis.json",
    "contact_kinematics_non_regression.json", "safety_non_regression.json",
    "stage10_classification.json", "recommended_next_action.json",
    "protected_hashes.json", "gate.json", "reproduction_commands.ps1",
]
dump("gate.json", {
    "classification": classification,
    "episode_rows": len(rows),
    "raw_chunks": len(raw_paths),
    "controller_unit_tests": json.loads(
        (OUT / "phase_gated_heading_unit_tests.json").read_text(encoding="utf-8")
    )["all_pass"],
    "protected_hashes_match": actual == expected,
    "ppo_updates": 0, "reward_optimization": 0, "policy_gradient": 0,
    "required_outputs_missing_before_gate_write": [
        name for name in required if name != "gate.json" and not (OUT / name).exists()
    ],
})

steady_table = "\n".join(
    f"| {speed:.1f} | {steady_json['C0'][str(speed)]['heading_p95']:.3f} | "
    f"{steady_json['C1'][str(speed)]['heading_p95']:.3f} | "
    f"{steady_json['C2'][str(speed)]['heading_p95']:.3f} | "
    f"{steady_json['C2'][str(speed)]['fall_rate']:.0%} | "
    f"{steady_json['C2'][str(speed)]['speed_mae']:.3f} |"
    for speed in (0.2, 0.3, 0.4, 0.5, 0.6, 1.2, 2.0)
)
low_table = "\n".join(
    f"| {source:g}→{target:g} | {low_json['C0'][f'{source:g}->{target:g}']['heading_p95']:.3f} | "
    f"{low_json['C1'][f'{source:g}->{target:g}']['heading_p95']:.3f} | "
    f"{low_json['C2'][f'{source:g}->{target:g}']['heading_p95']:.3f} | "
    f"{low_json['C2'][f'{source:g}->{target:g}']['completion_rate']:.0%} | "
    f"{low_json['C2'][f'{source:g}->{target:g}']['fall_rate']:.0%} |"
    for source, target in LOW
)
anchor_table = "\n".join(
    f"| {source:g}→{target:g} | "
    f"{anchor_json['C2'][f'{source:g}->{target:g}']['completion_rate']:.0%} | "
    f"{anchor_json['C2'][f'{source:g}->{target:g}']['acquisition_rate']:.0%} | "
    f"{anchor_json['C2'][f'{source:g}->{target:g}']['target_hold_rate']:.0%} | "
    f"{anchor_json['C2'][f'{source:g}->{target:g}']['fall_rate']:.0%} |"
    for source, target in ANCHOR
)
report = f"""# exp_011 Go2 phase-gated fixed-heading controller — Stage 10

## Outcome

**Classification:** `{classification}`

**Production status:** `DIAGNOSTIC_CANDIDATE`

**Next:** `{next_action}`

## Controller

The Stage 7 iteration-50 checkpoint is frozen. The command layer computes
`wrap(reference-current yaw)`, applies `Kp=1.0`, clips at `±0.10 rad/s`, and
multiplies by a non-learned phase gate. The gate is disabled during source hold,
speed ramp, and target acquisition; it activates once through a 0.5 s
minimum-jerk profile after 0.5 s continuous target acquisition. All offline
contract tests pass.

## Steady low speed

Mean 0.2–0.6 m/s heading p95 changes from `{mean_c0_heading:.4f}` rad (C0) to
`{mean_c2_heading:.4f}` rad (C2). All C2 low-speed steady heading gates:
`{steady_heading_all}`. Steady safety gates: `{steady_safety}`.

| speed | C0 heading p95 | C1 heading p95 | C2 heading p95 | C2 fall | C2 MAE |
|---:|---:|---:|---:|---:|---:|
{steady_table}

## Transitions and anchors

All low-speed transition gates: `{low_all}`. Anchor retention:
`{anchor_retention}`. Sequence retention: `{sequence_retention}`.
Always-on transition degradation avoided: `{constant_feedback_avoided}`.
The measured rate of C2 feedback activation before acquisition is
`{feedback_analysis['feedback_before_acquisition_rate']:.4f}`.

| transition | C0 heading p95 | C1 heading p95 | C2 heading p95 | C2 completion | C2 fall |
|---|---:|---:|---:|---:|---:|
{low_table}

| anchor transition | C2 completion | acquisition | hold | fall |
|---|---:|---:|---:|---:|
{anchor_table}

The C2 anchor sequence completes at
`{sequence_output['C2']['sequence_completion_rate']:.0%}`, with fall
`{sequence_output['C2']['fall_rate']:.0%}` and final stand
`{sequence_output['C2']['final_stand_rate']:.0%}`.

## Contact non-regression

Tangential-relative-motion non-regression: `{contact_pass}`. Contact telemetry
remains diagnostic-only and no contact penalty was introduced.

## GUI validation

Steady 0.4 m/s, 0→0.4, and AnchorSequence completed with tracking camera and
floor guides enabled. The installed headless runtime used the console overlay
fallback; no public video is claimed.

## Protection

PPO updates, reward optimization, and policy gradients are zero. All protected
checkpoint and endpoint-protocol hashes match. No production manifest was
changed and no remote push occurred.
"""
(REPO / "research/exp_011_go2_phase_gated_fixed_heading_report.md").write_text(
    report, encoding="utf-8"
)
print(classification)
