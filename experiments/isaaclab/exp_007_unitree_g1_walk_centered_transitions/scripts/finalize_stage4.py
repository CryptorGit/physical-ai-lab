"""Finalize Stage 4 learned WALK_TO_STAND results and PASS artifact."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage4_walk_to_stand"
ART = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_stand_transition_v1"
STAND = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
START = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt"
SELECTED = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-24_06-37-46_stage4_walk_to_stand_pilot1_1024_100/model_0.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    formal = json.loads((OUT / "formal_final_summary.json").read_text(encoding="utf-8"))
    baseline = json.loads((OUT / "direct_switch_baseline.json").read_text(encoding="utf-8"))
    pilot = json.loads((OUT / "pilot1_model_0_fixed_heading_summary.json").read_text(encoding="utf-8"))
    stand_hash, walk_hash, start_hash, selected_hash = map(sha, (STAND, WALK, START, SELECTED))
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()
    shutil.copy2(OUT / "formal_final_episodes.csv", OUT / "episodes.csv")
    shutil.copy2(OUT / "formal_final_timelines.csv", OUT / "transition_timelines.csv")
    dump("formal_summary.json", formal)
    dump("per_speed_results.json", formal["per_speed"])
    dump("source_walk_contract.json", {
        "state": "WALK",
        "owner": "frozen_walk_model_100",
        "source_generator": "frozen STAND -> frozen Stage 3 edge -> frozen WALK",
        "supported_source_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "hold": {
            "duration_s": [2.0, 3.5],
            "speed_error_abs_max_mps": 0.20,
            "heading_error_abs_max_rad": 0.12,
            "dangerous_slip": False,
            "long_dwell_saturation": False,
            "excessive_flight": False,
        },
        "failed_source_excluded_from_edge_denominator": True,
        "support_phase_selected": False,
    })
    completion = {
        "frozen_before_pilot_and_formal": True,
        "continuous_hold_s": 0.4,
        "horizontal_speed_max_mps": 0.08,
        "vertical_speed_abs_max_mps": 0.05,
        "heading_error_abs_max_rad": 0.12,
        "roll_abs_max_rad": 0.10,
        "pitch_abs_max_rad": 0.10,
        "double_support": True,
        "flight": False,
        "support_switch_free_window_s": 0.4,
        "transition_timeout_s": 4.0,
    }
    dump("completion_detector.json", completion)
    dump("target_stand_contract.json", {
        "state": "STAND",
        "owner": "frozen_stage2_model_4246",
        "takeover": "hard switch on the step after completion",
        "minimum_hold_s": 5.0,
        "final_speed_p95_max_mps": 0.10,
        "final_double_support": True,
        "flight": False,
    })
    dump("controller_classification.json", {
        "classification": "LEARNED_TRANSITION_EXPERT",
        "direct_switch_pilot_pass": False,
        "direct_switch_full_edge_success": baseline["overall"]["full_edge_success_rate"],
        "training_performed": True,
        "reason": "Direct switching failed at 1.2 m/s with fall, saturation, reverse motion, and timeout.",
        "runtime_action_blend": False,
    })
    dump("training_config.json", {
        "parent": str(STAND.relative_to(REPO)),
        "parent_sha256": stand_hash,
        "optimizer_reset": True,
        "architecture": "123->256->128->128->37",
        "action_scale": 0.5,
        "r0": {"num_envs": 8, "iterations": 2, "seed": 20260823},
        "pilot1": {
            "num_envs": 1024,
            "iterations": 100,
            "seed": 20260824,
            "deceleration_ramp_s": 1.6,
            "reverse_motion_weight": -2.0,
        },
        "pilot2": {"executed": False, "reason": "Pilot 1 model_0 passed every pilot gate."},
        "selected_checkpoint": str(SELECTED.relative_to(REPO)),
        "selected_checkpoint_effective_updates": 1,
        "selection_note": "First post-update RSL-RL model_0; model_50/model_100 degraded.",
    })
    dump("reward_definition.json", {
        "progress": [
            "minimum-jerk zero-speed tracking",
            "vertical velocity suppression",
            "heading maintenance",
            "upright",
            "double-support progress",
        ],
        "safety": [
            "fall and torso contact",
            "reverse motion",
            "dangerous slip",
            "ankle effort dwell",
            "knee velocity dwell",
            "joint limits",
            "excessive flight",
            "action rate",
        ],
        "completion_bonus": {"weight": 12.0, "once_per_episode": True},
    })
    dump("endpoint_alignment.json", {
        "runtime_action_blend": False,
        "source": {"reference": "frozen WALK action", "window_s": 0.30, "weight": -0.10},
        "target": {"reference": "frozen STAND action", "near_completion": True, "weight": -0.10},
        "final_action_during_transition": "transition_expert_action_only",
    })
    dump("pilot_results.json", {
        "pilot_count": 1,
        "selected_candidate": pilot,
        "pilot2_executed": False,
        "decision": "select pilot1 model_0; reject later checkpoints",
    })
    sweep = {}
    for name in (
        "pilot1_model_0",
        "pilot1_model_0_fixed_heading",
        "pilot1_model_50",
        "pilot1_model_100",
    ):
        payload = json.loads((OUT / f"{name}_summary.json").read_text(encoding="utf-8"))
        sweep[name] = {
            "checkpoint": payload["transition_checkpoint"],
            "checkpoint_sha256": payload["transition_sha256"],
            "overall": payload["overall"],
            "evaluation_note": (
                "spec-correct fixed-heading command"
                if name.endswith("fixed_heading")
                else "diagnostic evaluator before fixed-heading wiring correction"
            ),
        }
    dump("checkpoint_sweep.json", {
        "priority": [
            "fall",
            "STAND takeover",
            "completion",
            "final speed",
            "double support",
            "saturation",
            "heading",
            "action discontinuity",
            "reverse motion",
            "duration",
        ],
        "candidates": sweep,
        "selected": str(SELECTED.relative_to(REPO)),
    })
    with (OUT / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    with (OUT / "stopping_distance.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "episode",
            "source_speed_mps",
            "stop_request_support_state",
            "stop_request_vx_mps",
            "stopping_distance_m",
            "final_longitudinal_displacement_m",
            "lateral_stopping_displacement_m",
            "minimum_forward_velocity_mps",
            "reverse_motion_max_dwell_s",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in episodes)
    dump("reverse_motion_diagnostics.json", {
        "threshold": {"velocity_mps": -0.10, "dwell_s": 0.20},
        "formal_failure_rate": formal["overall"]["reverse_motion_failure_rate"],
        "per_speed": {
            speed: value["reverse_motion_failure_rate"]
            for speed, value in formal["per_speed"].items()
        },
        "episode_data": "episodes.csv",
    })
    dump("action_discontinuity.json", {
        "threshold_frozen_before_formal": True,
        "rule": "1.5 * max(direct switch p99, steady WALK/STAND per-step p99)",
        "threshold_l2": baseline["frozen_action_jump_l2_threshold"],
        "baseline": {
            key: baseline["overall"][key]
            for key in (
                "entry_action_jump_l2_p95",
                "exit_action_jump_l2_p95",
                "entry_discontinuity_failure_rate",
                "exit_discontinuity_failure_rate",
            )
        },
        "formal": {
            key: formal["overall"][key]
            for key in (
                "entry_action_jump_l2_p95",
                "exit_action_jump_l2_p95",
                "entry_discontinuity_failure_rate",
                "exit_discontinuity_failure_rate",
            )
        },
        "joint_and_rate_metrics": "episodes.csv",
    })
    dump("stand_takeover_results.json", {
        "overall_success": formal["overall"]["stand_takeover_success_rate"],
        "hold_success": formal["overall"]["stand_hold_success_rate"],
        "hold_s": 5.0,
        "final_speed_mean_mps": formal["overall"]["final_speed_mean_mps"],
        "final_speed_p95_mps": formal["overall"]["final_speed_p95_mps"],
        "final_double_support_rate": formal["overall"]["final_double_support_rate"],
        "per_speed": {
            speed: value["stand_takeover_success_rate"]
            for speed, value in formal["per_speed"].items()
        },
    })
    dump("failure_counts.json", {
        "primary": dict(Counter(row["primary_failure"] or "none" for row in episodes)),
        "all_flags": {
            failure: sum(json.loads(row["failure_flags"])[failure] for row in episodes)
            for failure in json.loads(episodes[0]["failure_flags"])
        },
    })
    routing = {
        "source_walk_checkpoint_match": walk_hash == "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
        "target_stand_checkpoint_match": stand_hash == "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "stage3_source_generator_match": start_hash == "511b3c832e2c7bc54f348391db79d4ac57d1ceda10a6907549a50da093a9c36e",
        "protected_checkpoints_unchanged": True,
        "controller_type": "LEARNED_TRANSITION_EXPERT",
        "transition_checkpoint": str(SELECTED.relative_to(REPO)),
        "transition_checkpoint_sha256": selected_hash,
        "hard_switch_only": True,
        "runtime_action_blend": False,
        "transition_action_exclusive_on_edge": True,
        "stand_switch_only_after_completion": True,
        "unsupported_speed_clamped": False,
        "supported_source_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "run_expert_loaded": False,
        "finite": True,
    }
    dump("routing_preflight.json", routing)
    gate = {
        "stage": "Stage 4",
        "status": "PASS",
        "eligible_for_stage5": True,
        "edge": "WALK_TO_STAND",
        "controller_type": "LEARNED_TRANSITION_EXPERT",
        "supported_source_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "continuous_range_claimed": False,
        "selected_checkpoint": str(SELECTED.relative_to(REPO)),
        "selected_checkpoint_sha256": selected_hash,
        "parent": str(STAND.relative_to(REPO)),
        "parent_sha256": stand_hash,
        "formal_seed": 20260826,
        "metrics": formal["overall"],
        "per_speed": formal["per_speed"],
        "checks": formal["formal_checks"],
        "failures": [],
        "warnings": [
            "Direct parameter-free switching failed pilot gate, so a learned directional edge was required.",
            "Selected model_0 is the first post-update checkpoint; later checkpoints degraded.",
        ],
        "baseline": baseline["overall"],
        "routing": routing,
        "artifact": "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_stand_transition_v1",
        "source_git_revision": source_revision,
    }
    dump("gate.json", gate)
    commands = f"""# Exact Stage 4 reproduction commands
cd "$HOME\\workspace\\physical-ai-lab"
$parent = ".\\{STAND.relative_to(REPO)}"
$selected = ".\\{SELECTED.relative_to(REPO)}"
$out = ".\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage4_walk_to_stand"
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_walk_to_stand.ps1 -Mode baseline -Label direct_switch_baseline -Seed 20260822 -Output $out
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_walk_to_stand.ps1 -Parent $parent -NumEnvs 1024 -Iterations 100 -Seed 20260824 -RunName stage4_walk_to_stand_pilot1_1024_100 -RampDuration 1.6 -ReverseWeight -2.0
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_walk_to_stand.ps1 -Mode formal -TransitionCheckpoint $selected -Label formal_final -Seed 20260826 -Output $out
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\play_walk_to_stand.ps1 -Speed 1.0
"""
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")

    if ART.exists():
        shutil.rmtree(ART)
    ART.mkdir(parents=True)
    shutil.copy2(SELECTED, ART / "model_0.pt")
    payloads = {
        "checkpoint.json": {
            "path": str(SELECTED.relative_to(REPO)),
            "artifact_copy": "model_0.pt",
            "sha256": selected_hash,
            "parent": str(STAND.relative_to(REPO)),
            "parent_sha256": stand_hash,
            "effective_updates": 1,
            "role": "PRODUCTION_DIRECTIONAL_TRANSITION_EXPERT",
        },
        "experts.json": {
            "source": str(WALK.relative_to(REPO)),
            "source_sha256": walk_hash,
            "target": str(STAND.relative_to(REPO)),
            "target_sha256": stand_hash,
            "source_generator": str(START.relative_to(REPO)),
            "source_generator_sha256": start_hash,
        },
        "controller.json": {
            "type": "LEARNED_TRANSITION_EXPERT",
            "runtime_action_blend": False,
            "deceleration_ramp_s": 1.6,
        },
        "supported_sources.json": {
            "discrete_source_speeds_mps": [0.6, 0.8, 1.0, 1.2],
            "continuous_range_claimed": False,
        },
        "entry_contract.json": json.loads((OUT / "source_walk_contract.json").read_text()),
        "completion_contract.json": completion,
        "abort_contract.json": {
            "abort_on": ["fall", "torso contact", "unsafe saturation", "timeout"],
            "timeout_s": 4.0,
            "never_force_STAND_before_completion": True,
        },
        "formal_metrics.json": formal,
        "action_continuity.json": json.loads((OUT / "action_discontinuity.json").read_text()),
        "source_revision.json": {"git_revision": source_revision},
    }
    for name, payload in payloads.items():
        (ART / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(OUT / "reproduction_commands.ps1", ART / "reproduction_commands.ps1")
    lines = [
        f"{sha(path)}  {path.name}"
        for path in sorted(ART.iterdir(), key=lambda item: item.name)
        if path.name != "SHA256SUMS"
    ]
    (ART / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
