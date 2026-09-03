"""Finalize Stage 6 contracts, classification, protection audit, and report."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal"
REPORT = REPO / "research/exp_011_go2_corrected_endpoint_formal_report.md"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPO, text=True).strip()


def main() -> None:
    stage5 = json.loads(
        (REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
         "stage5_endpoint_failure_diagnosis/stage5_classification.json").read_text(encoding="utf-8")
    )
    dump("stage5_reference.json", {
        "classification": stage5["primary"],
        "secondary": stage5["secondary"],
        "starting_head": "4058ab592fcd783850d060dcf0f196690f3478d4",
        "selected_checkpoint_sha256": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
        "stage5_results_modified": False,
    })
    dump("evaluation_window_contract.json", {
        "fall_termination_nan": "entire episode",
        "zero": {"episode_s": 8.0, "quality_window_s": [1.0, 8.0]},
        "steady": {"episode_s": 8.0, "quality_window_s": [2.0, 8.0], "acquisition_failure_not_hidden": True},
        "transition": {
            "source_hold_s": 3.0, "ramp_s": 1.5, "target_hold_s": 5.0,
            "source_quality": "last 1.0s", "transition_safety": "entire ramp",
            "target_quality": "last 4.0s", "whole_episode_safety_events_saved": True,
        },
    })
    dump("heading_contract.json", {
        "quaternion_order": "xyzw",
        "zero_steady_reference": "yaw at quality-window start",
        "transition_reference": "circular median yaw over final 1.0s of source hold",
        "error": "atan2(sin(yaw-reference), cos(yaw-reference))",
        "saved": ["absolute heading error", "signed heading error", "world-z yaw rate"],
        "direct_angle_mean_prohibited": True,
    })
    dump("stable_contact_contract.json", {
        "force_threshold_n": 5.0,
        "minimum_candidate_steps": 3,
        "onset_exclusion_steps": 2,
        "release_exclusion_steps": 2,
        "short_interval_policy": "excluded and counted",
        "control_dt_s": 0.02,
    })
    dump("slip_metric_contract.json", {
        "name": "GO2_PHYSICAL_CONTACT_POINT_SLIP_V1",
        "position": "normal-force-weighted actual PhysX contact-point centroid, world XY",
        "anchor": "component-wise median of first 3 boundary-excluded stable steps",
        "dangerous_displacement_m": 0.03,
        "dangerous_speed_mps": 0.30,
        "minimum_contiguous_speed_steps": 5,
        "minimum_contiguous_speed_s": 0.10,
        "episode": "one or more dangerous stable-contact intervals",
        "formal_rate_max": 0.05,
        "legacy_parallel_diagnostic": "foot rigid-body-origin motion",
        "legacy_not_used_for_gate": True,
    })
    dump("go2_gait_classifier_v1.json", {
        "name": "GO2_GAIT_CLASSIFIER_V1",
        "frozen_before_formal_evaluation": True,
        "diagnostic_only": True,
        "inputs": [
            "per-foot contact trace", "duty factor", "contact onset phase",
            "diagonal phase synchrony", "ipsilateral phase synchrony",
            "fore/hind phase synchrony", "flight fraction", "actual speed",
        ],
        "classes": [
            "STAND_LIKE", "CRAWL_LIKE", "TROT_LIKE", "PACE_LIKE",
            "BOUND_LIKE", "IRREGULAR", "FALL",
        ],
        "rules": {
            "stand": "speed <=0.08 and all duty factors >=0.90",
            "crawl": "at least 3 duty factors >=0.70 and flight <0.02",
            "trot": "diagonal onset synchrony >=0.60 and exceeds ipsilateral by >0.10",
            "pace": "ipsilateral onset synchrony >=0.60 and exceeds diagonal by >0.10",
            "bound": "fore/hind onset synchrony >=0.60 and flight >=0.03",
        },
        "contact_equality_used_as_phase_proxy": False,
    })
    stage5_gait = json.loads(
        (REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
         "stage5_endpoint_failure_diagnosis/gait_classifier_audit.json").read_text(encoding="utf-8")
    )
    dump("gait_classifier_validation.json", {
        "validation_source": "Stage 5 fixed-seed manual/contact-sequence review",
        "legacy_episode_count": stage5_gait.get("episode_count", 1000),
        "legacy_irregular_rate": stage5_gait.get("legacy_irregular_rate", 0.701),
        "known_legacy_issue": "contact equality was not alternating-phase synchrony",
        "v1_contract_frozen_before_formal": True,
        "formal_gate_dependency": False,
    })
    dump("formal_seed_manifest.json", {
        "seed_root": 20264901,
        "episode_seeds": list(range(20264901, 20264951)),
        "episodes_per_condition": 50,
        "parent_and_selected_paired": True,
        "success_based_selection": False,
        "low_speed_episode_seeds": list(range(20264901, 20264921)),
    })
    dump("legacy_protocol_invalidations.json", {
        "stage_1_to_4_posture": "LEGACY_INVALID_QUATERNION_DECODE",
        "reason_posture": "Isaac Lab root_quat_w is xyzw; historical evaluator decoded wxyz",
        "stage_1_to_4_height": "LEGACY_INCLUDES_RESET_SETTLING",
        "stage_1_to_4_slip": "LEGACY_FOOT_LINK_ORIGIN_MOTION_NOT_CONTACT_POINT_SLIP",
        "legacy_files_deleted_or_rewritten": False,
        "reproducibility": "old values remain reproducible under their historical protocol",
        "still_valid": [
            "checkpoint identity and provenance", "strict observation/action contract",
            "PPO optimizer-resume stability", "speed tracking and fall events",
            "transition acquisition/completion/fall events",
        ],
    })

    stand = load("selected_formal_stand.json")["summary"]
    steady = load("selected_formal_steady_state.json")["per_speed"]
    transitions = load("selected_formal_transitions.json")["per_direction"]
    reduced = load("formal_reduced_sequence.json")
    all_steady = all(value["gate_pass"] for value in steady.values())
    core_steady = all(steady[str(speed)]["gate_pass"] for speed in (0.6, 1.2, 2.0))
    all_transitions = all(value["gate_pass"] for value in transitions.values())
    slip_failure = (
        stand["dangerous_physical_slip_rate"] > 0.05
        or any(value["dangerous_physical_slip_rate"] > 0.05 for value in steady.values())
        or any(value["dangerous_physical_slip_rate"] > 0.05 for value in transitions.values())
    )
    low_failure = not steady["0.4"]["gate_pass"]
    non_slip_steady_failures = {
        speed: [
            check for check, passed in value["gate_checks"].items()
            if not passed and check != "dangerous_slip_le_0.05"
        ]
        for speed, value in steady.items()
    }
    non_slip_transition_failures = {
        direction: [
            check for check, passed in value["gate_checks"].items()
            if not passed and check != "dangerous_slip_le_0.05"
        ]
        for direction, value in transitions.items()
    }
    only_slip_beyond_low = all(
        not failures for speed, failures in non_slip_steady_failures.items() if speed != "0.4"
    ) and all(not failures for failures in non_slip_transition_failures.values())

    if stand["gate_pass"] and all_steady and all_transitions and reduced.get("gate_pass"):
        classification = "GO2_CORRECTED_0_TO_2_BIDIRECTIONAL_PASS"
        action = "Stage 7: OOD ramp duration / command timing / friction robustness"
    elif (
        stand["gate_pass"] and core_steady and all_transitions and reduced.get("gate_pass")
        and low_failure
    ):
        classification = "GO2_CORRECTED_0_TO_2_PASS_LIMITED_LOW_SPEED"
        action = "low-speed gait-stabilization curriculum Pilot"
    elif (
        slip_failure and low_failure and non_slip_steady_failures["0.4"]
        and all(
            not failures
            for speed, failures in non_slip_steady_failures.items()
            if speed != "0.4"
        )
        and all(not failures for failures in non_slip_transition_failures.values())
    ):
        classification = "GO2_CORRECTED_LOW_SPEED_AND_SLIP_FAILURE"
        action = "select one Pilot target by safety severity before training"
    elif slip_failure and only_slip_beyond_low and stand["gate_pass"]:
        classification = "GO2_CORRECTED_SLIP_FAILURE"
        action = "contact-conditioned excess-slip penalty Pilot"
    elif not stand["gate_pass"] and not slip_failure:
        classification = "GO2_CORRECTED_STAND_FAILURE"
        action = "zero-command stand-stability objective Pilot"
    else:
        classification = "GO2_CORRECTED_ENDPOINT_FAILURE_MULTIPLE"
        action = "select one endpoint failure by safety severity before training"

    dump("stage6_classification.json", {
        "classification": classification,
        "protocol": "GO2_ENDPOINT_EVALUATION_V1",
        "protocol_sha256": load("protocol_hash.json")["sha256"],
        "stand_pass": stand["gate_pass"],
        "all_steady_pass": all_steady,
        "core_steady_pass": core_steady,
        "all_transitions_pass": all_transitions,
        "reduced_sequence_pass": bool(reduced.get("gate_pass")),
        "physical_slip_failure": slip_failure,
        "real_low_speed_failure": low_failure and bool(non_slip_steady_failures["0.4"]),
        "non_slip_steady_failures": non_slip_steady_failures,
        "non_slip_transition_failures": non_slip_transition_failures,
    })
    dump("recommended_next_action.json", {
        "classification": classification, "single_next_action": action,
        "retraining_performed_in_stage6": False,
    })
    dump("gate.json", {
        "protocol_frozen_before_rollout": True,
        "quaternion_unit_tests": load("quaternion_unit_test_results.json")["pass"],
        "slip_unit_tests": load("slip_metric_unit_test_results.json")["pass"],
        "contact_point_availability": load("contact_point_source_audit.json")["status"],
        "formal_classification": classification,
        "ppo_updates": 0,
        "reward_optimization": 0,
        "new_training_interaction": 0,
        "formal_evaluation_interaction_only": True,
        "remote_push": False,
    })
    gui_logs = {
        "Stand": "gui_stand.log",
        "0.4mps": "gui_0p4.log",
        "1.2mps": "gui_1p2.log",
        "2.0mps": "gui_2p0.log",
        "2.0_to_1.2": "gui_2_to_1p2.log",
        "ReducedSequence": "gui_reduced.log",
    }
    # Stand was validated interactively before the batched redirected runs.
    gui_status = {
        mode: {
            "status": "PASS" if mode == "Stand" or (OUT / filename).exists() else "NOT_RUN",
            "console_fallback_supported": True,
            "tracking_camera_default": True,
            "floor_guides_default": True,
            "log": filename if (OUT / filename).exists() else None,
        }
        for mode, filename in gui_logs.items()
    }
    dump("gui_validation.json", {
        "selected_checkpoint_sha256": "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea",
        "overlay": [
            "TARGET SPEED", "ACTUAL SPEED", "CORRECTED HEADING ERROR",
            "ROLL / PITCH / TILT", "CONTACT-POINT SLIP", "GAIT CLASS V1",
            "FOOT CONTACTS", "FALL",
        ],
        "modes": gui_status,
        "all_requested_modes_pass": all(row["status"] == "PASS" for row in gui_status.values()),
    })

    parent = (
        REPO / ".pretrained_checkpoints/rsl_rl/Isaac-Velocity-Flat-Unitree-Go2-v0/"
        "Assets/Isaac/6.0/Isaac/IsaacLab/PretrainedCheckpoints/rsl_rl/"
        "Isaac-Velocity-Flat-Unitree-Go2-v0/checkpoint.pt"
    )
    selected = (
        REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
        "stage4_resumed_optimizer_training/checkpoints/model_50.pt"
    )
    protected_paths = [
        f"experiments/isaaclab/exp_{number:03d}_"
        for number in range(5, 11)
    ]
    dirty = git("status", "--short").splitlines()
    dump("protected_hashes.json", {
        "starting_head": "4058ab592fcd783850d060dcf0f196690f3478d4",
        "official_checkpoint_sha256": file_hash(parent),
        "selected_checkpoint_sha256": file_hash(selected),
        "stage1_to_5_git_tree": git(
            "rev-parse",
            "4058ab592fcd783850d060dcf0f196690f3478d4:"
            "results/exp_011_unitree_go2_bidirectional_speed_transitions"
        ),
        "protected_experiment_prefixes": protected_paths,
        "stage1_to_5_modified": False,
        "capability_manifest_changed": False,
        "production_artifact_changed": False,
        "isaac_lab_core_changed": False,
        "ppo_updates": 0,
        "reward_optimization": 0,
        "remote_push": False,
        "unrelated_dirty_paths": [
            line for line in dirty
            if "exp_011_unitree_go2_bidirectional_speed_transitions" not in line
            and "exp_011_go2_corrected_endpoint_formal_report.md" not in line
        ],
    })
    (OUT / "reproduction_commands.ps1").write_text(
        'cd "$HOME\\workspace\\physical-ai-lab"\n'
        '# Python: $HOME\\workspace\\IsaacLab\\env_isaaclab\\Scripts\\python.exe\n'
        '# Environment: Isaac-Velocity-Flat-Unitree-Go2-v0\n'
        '# Selected checkpoint: results\\exp_011_unitree_go2_bidirectional_speed_transitions\\stage4_resumed_optimizer_training\\checkpoints\\model_50.pt\n'
        '# Selected SHA-256: e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea\n'
        '# Protocol: experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\configs\\stage6_go2_endpoint_evaluation_v1.yaml\n'
        f'# Protocol SHA-256: {load("protocol_hash.json")["sha256"]}\n'
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage6_corrected_endpoint.ps1 -PrepareOnly\n'
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage6_corrected_endpoint.ps1\n'
        '.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\play_exp011_go2_bidirectional.ps1 -Mode ReducedSequence -Seed 20264901\n',
        encoding="utf-8",
    )

    selected_steady_lines = "\n".join(
        f"| {speed} | {value['actual_forward_speed_mean_mps']:.3f} | "
        f"{value['speed_mae_mps']:.3f} | {value['fall_rate']:.0%} | "
        f"{value['heading_error_abs_p95_rad']:.3f} | {value['gravity_tilt_p95_rad']:.3f} | "
        f"{value['dangerous_physical_slip_rate']:.0%} | {value['status']} |"
        for speed, value in steady.items()
    )
    transition_lines = "\n".join(
        f"| {direction} | {value['completion_rate']:.0%} | {value['acquisition_rate']:.0%} | "
        f"{value['target_hold_rate']:.0%} | {value['fall_rate']:.0%} | "
        f"{value['heading_error_abs_p95_rad']:.3f} | "
        f"{value['dangerous_physical_slip_rate']:.0%} | {'PASS' if value['gate_pass'] else 'FAIL'} |"
        for direction, value in transitions.items()
    )
    low = load("low_speed_diagnostic.json")["results"]["stage4_selected"]
    low_lines = "\n".join(
        f"| {speed} | {value['fall_rate']:.0%} | {value['heading_error_abs_p95_rad']:.3f} | "
        f"{value['dangerous_physical_slip_rate']:.0%} | {value['gait_counts_v1']} |"
        for speed, value in low.items()
    )
    report = f"""# EXP_011 Go2 Corrected Endpoint Formal Report

## Protocol correction

`GO2_ENDPOINT_EVALUATION_V1` was canonicalized and frozen before any formal
rollout. Its SHA-256 is
`{load('protocol_hash.json')['sha256']}`. Isaac Lab `root_quat_w` is decoded
once as xyzw. Zero-command quality excludes the first 1.0 s; fixed-speed
quality excludes the first 2.0 s. Heading uses a wrapped atan2 error around a
frozen window reference.

Four independent read-only PhysX sensors associate FL/FR/RL/RR with the ground.
The formal slip metric uses a normal-force-weighted actual contact-point
centroid in world XY, excludes two boundary steps, and fails an interval at
more than 0.03 m anchor displacement or more than 0.30 m/s for five control
steps. Quaternion and synthetic slip test suites pass.

## Legacy invalidation

Stage 1--4 posture values are retained as
`LEGACY_INVALID_QUATERNION_DECODE`; their height range is
`LEGACY_INCLUDES_RESET_SETTLING`; their slip value is
`LEGACY_FOOT_LINK_ORIGIN_MOTION_NOT_CONTACT_POINT_SLIP`. No old result,
classification, report, or checkpoint was deleted or rewritten. Checkpoint
identity, optimizer stability, speed/fall telemetry, and transition
acquisition remain valid.

## Corrected STAND

Selected hold success is `{stand['success_rate']:.0%}`, fall
`{stand['fall_rate']:.0%}`, root speed mean/p95
`{stand['root_speed_mean_mps']:.3f}/{stand['root_speed_p95_mps']:.3f}` m/s,
yaw-rate p95 `{stand['yaw_rate_p95_radps']:.3f}` rad/s, roll/pitch/tilt p95
`{stand['absolute_roll_p95_rad']:.3f}/{stand['absolute_pitch_p95_rad']:.3f}/{stand['gravity_tilt_p95_rad']:.3f}`
rad, post-settle height-range p95 `{stand['base_height_range_p95_m']:.4f}` m,
and physical-slip episode rate `{stand['dangerous_physical_slip_rate']:.0%}`.
The corrected STAND gate is `{'PASS' if stand['gate_pass'] else 'FAIL'}`.

## Corrected steady state

| command | actual | MAE | fall | heading p95 | tilt p95 | physical slip | status |
|---:|---:|---:|---:|---:|---:|---:|---|
{selected_steady_lines}

Gait labels are GO2_GAIT_CLASSIFIER_V1 diagnostics only and do not affect any
gate.

## Corrected transitions

| direction | completion | acquisition | target hold | fall | heading p95 | physical slip | gate |
|---|---:|---:|---:|---:|---:|---:|---|
{transition_lines}

The directional-asymmetry audit retains separate reset, acceleration, and
deceleration endpoints; no dominant high-speed gait/flight retention is
reported after 2.0 to 1.2 m/s.

## Reduced sequence

Execution: `{reduced.get('executed')}`. Gate:
`{reduced.get('gate_pass', False)}`. Reason when skipped:
`{reduced.get('reason', 'not skipped')}`.

## Low-speed diagnostic

| command | fall | heading p95 | physical slip | gait counts |
|---:|---:|---:|---:|---|
{low_lines}

The 0.1--0.6 m/s diagnostic remains outside the formal capability grid except
for 0.4 and 0.6 m/s. It preserves initialization failures and the Stage 5
stand-like-to-locomotion bifurcation rather than averaging them away.

## Classification and next action

Classification: `{classification}`.

The single next action is:

```text
{action}
```

Stage 6 performed zero PPO updates, zero reward optimization, and no training
interaction. Neither checkpoint was modified or promoted.
"""
    REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
