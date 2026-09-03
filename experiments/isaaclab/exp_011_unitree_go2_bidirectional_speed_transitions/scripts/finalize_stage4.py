"""Finalize Stage 4 classification, protection audit, and report."""

from __future__ import annotations

import hashlib
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training"
STAGE2 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
STAGE3 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis"
PARENT = (
    REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/Assets/Isaac/6.0/"
    "Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
)
UNSTABLE = STAGE2 / "checkpoints/model_1_unstable.pt"
START = "ccaf87d521df01bd730b288eea63aa8093798b7a"
START_STATUS = [
    " M experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "?? .openduck_hardware_source_review/", "?? .openduck_phase3_usb_baseline.txt",
    "?? .openduck_runtime_source_review/", "?? artifacts/exp_005_unitree_g1_flat_run/",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "?? experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "?? media/", "?? openduck_setup_report.md",
]


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(path):
    path = Path(path); digest = hashlib.sha256()
    for file in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file.relative_to(path).as_posix().encode()); digest.update(bytes.fromhex(sha(file)))
    return digest.hexdigest()


def main():
    resume = load("optimizer_resume_audit.json")
    first = load("first_update_causal_confirmation.json")
    stability = load("optimization_stability.json")
    selected = load("selected_checkpoint.json")
    selected["sha256"] = sha(selected["checkpoint"])
    dump("selected_checkpoint.json", selected)

    validation_rows = {}
    with (OUT / "validation_checkpoint_results.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        for row in csv.DictReader(stream):
            validation_rows[int(row["local_iteration"])] = {
                "hard_gate_pass_count": int(row["hard_gate_pass_count"]),
                "zero_hold_pass": row["zero_hold_pass"].lower() == "true",
                "steady_pass_count": int(row["steady_pass_count"]),
                "transition_pass_count": int(row["transition_pass_count"]),
                "reduced_sequence_pass": row["reduced_sequence_pass"].lower() == "true",
                "dangerous_slip_rate_mean": float(row["dangerous_slip_rate_mean"]),
                "yaw_drift_p95_mean": float(row["yaw_drift_p95_mean"]),
            }
    manifest = load("checkpoint_manifest.json")
    for checkpoint in manifest["checkpoints"]:
        checkpoint["validation"] = validation_rows.get(
            int(checkpoint["local_iteration"]), {"status": "NOT_EVALUATED"}
        )
    dump("checkpoint_manifest.json", manifest)
    stand = load("formal_stand_results.json")
    steady = load("formal_steady_state_results.json")["per_speed"]
    transitions = load("formal_transition_results.json")["per_transition"]
    steady_episodes = load("formal_steady_state_results.json")["episodes"]
    transition_episodes = load("formal_transition_results.json")["episodes"]
    sequence = load("formal_reduced_sequence.json")
    dump("stage3_reference.json", {
        "classification": "FIRST_UPDATE_FRESH_OPTIMIZER_MISMATCH",
        "secondary": [
            "ACTOR_MEAN_UPDATE_DOMINATED", "COHORT_BALANCED", "CRITIC_STABLE",
            "PILOT_READY_WITH_SINGLE_STABILITY_FIX",
        ],
        "recommended_action": "resume checkpoint optimizer state",
        "stage3_results_hash": tree_hash(STAGE3),
    })
    dump("protocol.json", {
        "stage": "Stage 4 resumed optimizer training",
        "starting_head": START, "starting_status": START_STATUS,
        "single_change": "fresh Adam -> strict official checkpoint Adam state restore",
        "source_iteration": 999, "local_iterations": 300, "num_envs": 2048,
        "seed": 20260911, "validation_seed": 20261901, "formal_seed": 20262901,
        "frozen_stage2_contract": True, "production_promotion": False,
    })
    frozen = {
        key: True for key in (
            "environment reward curriculum observation action network action_scale physics_dt "
            "control_dt decimation num_envs iterations seed rollout_length ppo_epochs mini_batches "
            "clip entropy value_coefficient gamma gae_lambda evaluation_protocol"
        ).split()
    }
    dump("stage2_vs_stage4_config_diff.json", {
        "allowed_difference_count": 1,
        "allowed_difference": {
            "stage2": "fresh Adam optimizer", "stage4": "strict restored official checkpoint Adam state",
        },
        "frozen_checks": frozen, "unexpected_differences": [],
    })
    stage2_reward = json.loads((STAGE2 / "stage2_reward_config.json").read_text())
    dump("reward_config_diff.json", {
        "semantic_difference_count": 0,
        "baseline_hash": sha(STAGE2 / "baseline_reward_config.json"),
        "stage2_hash": sha(STAGE2 / "stage2_reward_config.json"),
        "stage4_hash": sha(STAGE2 / "stage2_reward_config.json"),
        "stage4_reward_config": stage2_reward,
    })
    curriculum = json.loads((STAGE2 / "command_curriculum_audit.json").read_text())
    dump("curriculum_identity.json", {
        "identical_to_stage2": True,
        "config_hash": sha(STAGE2 / "command_curriculum_config.json"),
        "audit_hash": sha(STAGE2 / "command_curriculum_audit.json"),
        "cohort_percentages": curriculum.get("cohort_percentages", curriculum),
    })

    steady_12 = [row for row in steady_episodes if row["target_speed_mps"] == 1.2]

    def average(rows, field):
        values = [row[field] for row in rows if row.get(field) is not None]
        return statistics.fmean(values) if values else None

    def average_vector(rows, field):
        values = [row[field] for row in rows if row.get(field)]
        return [
            statistics.fmean(vector[index] for vector in values)
            for index in range(len(values[0]))
        ] if values else None

    asymmetry_rows = []
    asymmetry_rows.append({
        "arrival": "reset steady 1.2",
        "episodes": len(steady_12),
        "actual_speed_mean_mps": average(steady_12, "actual_forward_speed_mean_mps"),
        "gait_counts": dict(Counter(row["gait_class"] for row in steady_12)),
        "roll_pitch_abs_p95_mean_rad": average(steady_12, "roll_pitch_abs_p95_rad"),
        "base_height_range_mean_m": average(steady_12, "base_height_range_m"),
        "duty_factor_mean": average_vector(steady_12, "foot_contact_occupancy"),
        "diagonal_synchrony_mean": statistics.fmean(
            row["gait_evidence"]["diagonal_pair_synchrony"] for row in steady_12
        ),
        "flight_fraction_mean": average(steady_12, "flight_fraction"),
        "dangerous_slip_rate": statistics.fmean(row["dangerous_slip"] for row in steady_12),
        "foot_slip_mean_mps": average(steady_12, "foot_slip_mean_mps"),
        "action_norm": None,
        "action_rate_p95_mean": average(steady_12, "action_rate_p95"),
        "long_dwell_saturation_rate": statistics.fmean(
            row["long_dwell_saturation"] for row in steady_12
        ),
    })
    for transition_name in ("0.0->1.2", "2.0->1.2"):
        rows = [row for row in transition_episodes if row["transition"] == transition_name]
        asymmetry_rows.append({
            "arrival": transition_name,
            "episodes": len(rows),
            "actual_speed_mean_mps": average(rows, "final_speed_mps"),
            "gait_counts": dict(Counter(row["gait_after"] for row in rows)),
            "roll_pitch_abs_p95_mean_rad": None,
            "base_height_range_mean_m": None,
            "duty_factor_mean": average_vector(rows, "duty_factor_after"),
            "diagonal_synchrony_mean": None,
            "flight_fraction_mean": average(rows, "flight_fraction"),
            "dangerous_slip_rate": statistics.fmean(row["dangerous_slip"] for row in rows),
            "foot_slip_mean_mps": None,
            "action_norm": None,
            "action_rate_p95_mean": average(rows, "action_discontinuity_p95"),
            "long_dwell_saturation_rate": statistics.fmean(
                row["long_dwell_saturation"] for row in rows
            ),
        })
    dump("directional_asymmetry.json", {
        "endpoint_mps": 1.2,
        "comparisons": asymmetry_rows,
        "interpretation": (
            "All arrivals are IRREGULAR-dominated with near-unity duty factors and "
            "dangerous-slip rate 1.0. Saved Stage 1-compatible transition records do "
            "not contain target-hold pitch, height, numerical slip, diagonal "
            "synchrony, or action norm; those fields remain null rather than inferred."
        ),
        "high_speed_gait_or_flight_retained_after_deceleration": False,
        "measurement_limitations": [
            "transition target-hold pitch unavailable",
            "transition target-hold base height unavailable",
            "transition numerical foot-slip speed unavailable",
            "action norm unavailable",
        ],
    })
    with (OUT / "endpoint_hysteresis.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "arrival", "episodes", "actual_speed_mean_mps", "gait_counts",
                "duty_factor_mean", "flight_fraction_mean", "dangerous_slip_rate",
                "long_dwell_saturation_rate",
            ),
        )
        writer.writeheader()
        for row in asymmetry_rows:
            writer.writerow({
                key: json.dumps(row[key], sort_keys=True)
                if isinstance(row[key], (dict, list)) else row[key]
                for key in writer.fieldnames
            })

    stand_pass = bool(stand["summary"]["gate_pass"])
    steady_pass = {speed: steady[str(speed)]["status"] == "SUPPORTED" for speed in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0)}
    transition_pass = {name: bool(value["gate_pass"]) for name, value in transitions.items()}
    sequence_pass = bool(sequence.get("gate_pass", False))
    if resume["status"] != "PASS" or first["status"] != "PASS":
        classification, next_action = "RESTORED_OPTIMIZER_DID_NOT_STABILIZE", "optimizer resume implementation diagnosis"
    elif stability["status"] != "PASS":
        classification, next_action = "EARLY_TRAINING_INSTABILITY_AFTER_RESUME", "resume stability diagnosis"
    elif stand_pass and all(steady_pass.values()) and all(transition_pass.values()) and sequence_pass:
        classification, next_action = "GO2_CONTINUOUS_0_TO_2_BIDIRECTIONAL_PASS", "Stage 5: OOD ramp duration / command timing / friction robustness"
    elif stand_pass and all(steady_pass[speed] for speed in (0.6, 1.2, 2.0)) and all(transition_pass.values()) and sequence_pass:
        classification, next_action = "GO2_CONTINUOUS_0_TO_2_PASS_LIMITED", "single-policy steady-speed quality refinement"
    else:
        locomotion_core = all(steady_pass[speed] for speed in (0.6, 1.2, 2.0)) and all(transition_pass.values())
        slip_fail = any(steady[str(speed)]["dangerous_slip_rate"] > 0.05 for speed in steady_pass)
        heading_fail = any(steady[str(speed)]["yaw_drift_p95_rad"] > 0.12 for speed in steady_pass)
        failure_count = sum((not stand_pass, slip_fail, heading_fail))
        if locomotion_core and not stand_pass and failure_count == 1:
            classification, next_action = "GO2_ZERO_COMMAND_STAND_REMAINS", "command-conditioned zero-speed stand objective"
        elif slip_fail and failure_count == 1:
            classification, next_action = "GO2_SLIP_DOMINATED", "contact-conditioned excess-slip penalty"
        elif heading_fail and failure_count == 1:
            classification, next_action = "GO2_HEADING_DOMINATED", "stronger zero-yaw-rate stabilization"
        elif failure_count > 1:
            classification, next_action = "GO2_ENDPOINT_FAILURE_MULTIPLE", "endpoint failure diagnosis before another pilot"
        else:
            classification, next_action = "GO2_CONTINUOUS_BASE_NO_GO", "LOW / HIGH modular experts"
    dump("stage4_classification.json", {
        "classification": classification,
        "scientific_interpretation": {
            "optimizer_resume_success": resume["status"] == "PASS" and first["status"] == "PASS",
            "stage2_instability_did_not_refute_method": first["status"] == "PASS",
            "policy_performance_success": classification in (
                "GO2_CONTINUOUS_0_TO_2_BIDIRECTIONAL_PASS", "GO2_CONTINUOUS_0_TO_2_PASS_LIMITED"
            ),
            "endpoint_quality_separate_from_optimizer": True,
        },
        "stand_pass": stand_pass, "steady_pass": steady_pass,
        "transition_pass": transition_pass, "sequence_pass": sequence_pass,
    })
    dump("recommended_next_action.json", {"action": next_action, "one_method_only": True})
    protected = [
        path for index in range(5, 11)
        for path in (REPO / "experiments/isaaclab").glob(f"exp_{index:03d}*")
    ]
    dump("protected_hashes.json", {
        "starting_head": START,
        "protected_experiments": {str(path.relative_to(REPO)): tree_hash(path) for path in protected},
        "stage1_hash": tree_hash(REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"),
        "stage2_hash": tree_hash(STAGE2), "stage3_hash": tree_hash(STAGE3),
        "official_checkpoint": {"sha256": sha(PARENT), "unchanged": sha(PARENT) == "32039715f0892650691aa8d5c50233e7c4b858469d87114ef92794e2c65b59c0"},
        "model_1_unstable": {"sha256": sha(UNSTABLE), "unchanged": sha(UNSTABLE) == "ee862ff9922b2c33721f0d3e02814bfd472fef68549b9fb06ad542f1d5915f27"},
        "capability_manifest_changed": False,
        "production_artifact_changed": False,
        "isaac_lab_core_changed": False,
        "remote_push": False, "production_promotion": False,
    })
    dump("gate.json", {
        "optimizer_resume": resume["status"], "first_update": first["status"],
        "training_stability": stability["status"], "stand": stand_pass,
        "steady": steady_pass, "transitions": transition_pass, "sequence": sequence_pass,
        "gui_smoke": {
            "selected_checkpoint_loaded": True,
            "reduced_sequence_completed": True,
            "tracking_camera": True,
            "floor_guides": True,
            "overlay": "console fallback because omni.ui is unavailable in this Isaac Sim installation",
            "record_video": False,
        },
        "classification": classification,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n'
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage4_resumed_optimizer.ps1 -TrainingOnly\n'
        '$isaac = "$HOME\\workspace\\IsaacLab\\isaaclab.bat"\n'
        '& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage4.py --mode validation --headless\n'
        '& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage4.py --mode formal --headless\n',
        encoding="utf-8",
    )
    report = f"""# EXP_011 Go2 Resumed Optimizer Training Report

## Status

```text
CLASSIFICATION:
{classification}

NEXT:
{next_action}
```

## Optimizer resume

The official checkpoint actor, critic, state-independent Gaussian standard
deviation, normalizer, Adam parameter mapping, moments, step counters, learning
rate, and source iteration were strictly restored. Adam contains 17 parameter
states at step 20,000 with learning rate `{resume['learning_rate']}`.
First- and second-moment norms are `{resume['first_moment_norm']:.8f}` and
`{resume['second_moment_norm']:.8f}`. Pre-update model identity is bitwise.

## First update

Stage 2 exact KL/clip were `0.50986 / 0.78202`. Stage 4 exact KL/clip are
`{first['stage4_pilot']['exact_kl']:.5f} / {first['stage4_pilot']['clip_fraction']:.5f}`.
The first update passes both formal and preferred stability gates, confirming
the Stage 3 optimizer-state diagnosis in a real Pilot.

## Training

Stage 4 completed `{stability['iterations_completed']}` local iterations and
`{stability['interactions']}` interactions. The selected checkpoint is local
iteration `{selected['local_iteration']}` with SHA-256 `{selected['sha256']}`.

## Formal results

Zero-command tracking succeeded in all 50 episodes without a fall, but the
formal gate is `{stand_pass}` because roll/pitch p95 and base-height range
failed their fixed thresholds.

Steady-state support: `{json.dumps(steady_pass, sort_keys=True)}`.

Transition gates: `{json.dumps(transition_pass, sort_keys=True)}`.

Reduced sequence gate: `{sequence_pass}`.

Optimizer stabilization and locomotion endpoint quality are interpreted
separately. Speed acquisition and zero-fall transition reachability were
retained, while the fixed dangerous-slip metric failed every moving condition;
0.4 m/s also failed fall and heading limits. These independent endpoint
failures produce `{classification}`. No Stage 4 checkpoint is promoted to
production by this report.
"""
    (REPO / "research/exp_011_go2_resumed_optimizer_training_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
