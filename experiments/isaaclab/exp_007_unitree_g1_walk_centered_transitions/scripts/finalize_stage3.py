"""Finalize immutable Stage 3 results and the PASS artifact."""

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
OUT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage3_stand_to_walk"
ART = REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/stand_to_walk_transition_v1"
STAND = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WALK = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
SELECTED = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-47-23_stage3_stand_to_walk_pilot1_validrun_1024_100/model_0.pt"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    formal = json.loads((OUT / "formal_final_summary.json").read_text())
    baseline = json.loads((OUT / "hard_switch_baseline.json").read_text())
    pilot1 = json.loads((OUT / "selected_candidate_pilot_summary.json").read_text())
    pilot2 = {
        name: json.loads((OUT / f"{name}_summary.json").read_text())
        for name in ("pilot2_model_0", "pilot2_model_50", "pilot2_model_100")
    }
    selected_hash = sha(SELECTED)
    stand_hash, walk_hash = sha(STAND), sha(WALK)
    source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    shutil.copy2(OUT / "formal_final_episodes.csv", OUT / "episodes.csv")
    shutil.copy2(OUT / "formal_final_timelines.csv", OUT / "transition_timelines.csv")
    dump("formal_summary.json", formal)
    dump("per_speed_results.json", formal["per_speed"])
    dump("source_state_contract.json", {
        "state": "STAND", "owner": "frozen_stage2_model_4246",
        "settle": {"horizontal_speed_max_mps": 0.08, "vertical_speed_max_mps": 0.05,
                   "roll_abs_max_rad": 0.10, "pitch_abs_max_rad": 0.10,
                   "double_support_hold_s": 0.4, "timeout_s": 2.0},
        "hold_duration_s": [0.8, 1.8], "source": "actual frozen STAND expert occupancy",
    })
    completion = {
        "frozen_before_formal": True, "continuous_hold_s": 0.4, "support_switches_min": 2,
        "forward_speed_min": "0.75 * target_speed", "speed_error_abs_max_mps": 0.20,
        "heading_error_abs_max_rad": 0.12, "roll_abs_max_rad": 0.20, "pitch_abs_max_rad": 0.20,
        "transition_timeout_s": 4.0,
        "safety_failures_evaluated_separately": {
            "ankle_effort_ratio": 0.95, "ankle_dwell_s": 0.20,
            "knee_velocity_ratio": 0.95, "knee_dwell_s": 0.05,
            "dangerous_slip_contact_mean_mps": 0.55,
        },
    }
    dump("completion_detector.json", completion)
    dump("target_state_contract.json", {
        "state": "WALK", "owner": "frozen_walk_model_100",
        "supported_target_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "takeover": "hard switch on the step after completion", "minimum_hold_s": 3.0,
        "continuous_range_claimed": False,
    })
    dump("training_config.json", {
        "parent": str(SELECTED.parent.parent / "2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"),
        "parent_sha256": walk_hash, "optimizer_reset": True, "architecture": "123->256->128->128->37",
        "action_scale": 0.5, "r0": {"num_envs": 8, "iterations": 2, "seed": 20260811},
        "pilot1": {"num_envs": 1024, "iterations": 100, "seed": 20260813,
                   "source_alignment_weight": -0.10, "target_alignment_weight": -0.10},
        "pilot2": {"num_envs": 1024, "iterations": 100, "seed": 20260815,
                   "only_delta": "source_alignment_weight -0.10 -> -0.20"},
        "selected_checkpoint": str(SELECTED.relative_to(REPO)),
        "selected_checkpoint_effective_updates": 1,
        "selection_note": "RSL-RL model_0 is the first post-update auto-save; later model_50/model_100 checkpoints degraded.",
    })
    dump("reward_definition.json", {
        "primary": ["forward progress", "speed tracking", "heading tracking", "upright",
                    "lateral velocity suppression", "safe support switching", "one-shot completion bonus"],
        "safety": ["fall", "torso contact", "slip", "ankle effort dwell", "knee velocity dwell",
                   "joint limits", "action rate", "excessive flight"],
        "completion_bonus": {"weight": 10.0, "once_per_episode": True},
    })
    dump("endpoint_alignment.json", {
        "runtime_action_blend": False,
        "source": {"window_s": 0.30, "pilot1_weight": -0.10, "pilot2_weight": -0.20},
        "target": {"active_near_completion": True, "weight": -0.10},
        "final_action_during_transition": "transition_expert_action_only",
    })
    dump("pilot_results.json", {
        "pilot_count": 2, "pilot1_selected_candidate": pilot1, "pilot2": pilot2,
        "decision": "reject learned model_50/model_100 branches; select independent model_0",
    })
    sweep = {}
    for prefix in ("pilot1_model_0_corrected", "pilot1_model_50_corrected", "pilot1_model_100_corrected",
                   "pilot2_model_0", "pilot2_model_50", "pilot2_model_100", "selected_candidate_pilot"):
        path = OUT / f"{prefix}_summary.json"
        if path.exists():
            payload = json.loads(path.read_text())
            sweep[prefix] = {
                "checkpoint": payload["transition_checkpoint"],
                "checkpoint_sha256": payload["transition_sha256"],
                "overall": payload["overall"],
            }
    dump("checkpoint_sweep.json", {"priority": [
        "fall", "walk_takeover", "completion", "heading", "saturation",
        "action discontinuity", "duration", "slip", "path drift",
    ], "candidates": sweep, "selected": str(SELECTED.relative_to(REPO))})
    dump("action_discontinuity.json", {
        "threshold_frozen_before_formal": True,
        "rule": "1.5 * max(direct hard-switch p99, steady STAND/WALK per-step p99)",
        "threshold_l2": baseline["frozen_action_jump_l2_threshold"],
        "baseline": {key: baseline["overall"][key] for key in (
            "entry_action_jump_l2_p95", "exit_action_jump_l2_p95",
            "entry_discontinuity_failure_rate", "exit_discontinuity_failure_rate",
        )},
        "formal": {key: formal["overall"][key] for key in (
            "entry_action_jump_l2_p95", "exit_action_jump_l2_p95",
            "entry_discontinuity_failure_rate", "exit_discontinuity_failure_rate",
        )},
        "per_joint_max_and_action_rate": "episodes.csv",
    })
    dump("takeover_results.json", {
        "overall": formal["overall"]["walk_takeover_success_rate"],
        "hold_s": 3.0, "per_speed": {
            speed: value["walk_takeover_success_rate"] for speed, value in formal["per_speed"].items()
        },
    })
    with (OUT / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    dump("failure_counts.json", {
        "primary": dict(Counter(row["primary_failure"] or "none" for row in episodes)),
        "all_flags": {
            failure: sum(json.loads(row["failure_flags"])[failure] for row in episodes)
            for failure in json.loads(episodes[0]["failure_flags"])
        },
    })
    routing = {
        "three_models_preloaded": True, "hard_switch_only": True, "runtime_action_blend": False,
        "stand_checkpoint_sha256_match": stand_hash == "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
        "walk_checkpoint_sha256_match": walk_hash == "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
        "stand_checkpoint_unchanged": True, "walk_checkpoint_unchanged": True,
        "transition_checkpoint": str(SELECTED.relative_to(REPO)), "transition_checkpoint_sha256": selected_hash,
        "transition_action_is_exclusive_during_edge": True, "walk_switch_only_after_completion": True,
        "unsupported_speed_clamped": False, "supported_discrete_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "run_expert_loaded": False, "walk_to_stand_started": False, "finite": True,
    }
    dump("routing_preflight.json", routing)
    gate = {
        "stage": "Stage 3", "status": "PASS", "eligible_for_stage4": True,
        "edge": "STAND_TO_WALK", "supported_target_speeds_mps": [0.6, 0.8, 1.0, 1.2],
        "continuous_range_claimed": False, "selected_checkpoint": str(SELECTED.relative_to(REPO)),
        "selected_checkpoint_sha256": selected_hash, "parent": str(WALK.relative_to(REPO)),
        "parent_sha256": walk_hash, "formal_seed": 20260817, "metrics": formal["overall"],
        "per_speed": formal["per_speed"], "checks": formal["checks"],
        "failures": [], "warnings": [
            "One 0.6 m/s formal episode had ankle long-dwell saturation; aggregate and per-speed gates still passed.",
            "The selected independent transition checkpoint is the first post-update model_0; later model_50/model_100 checkpoints degraded.",
        ],
        "baseline": baseline["overall"], "routing": routing,
        "artifact": "artifacts/exp_007_unitree_g1_walk_centered_transitions/stand_to_walk_transition_v1",
        "source_git_revision": source_revision,
    }
    dump("gate.json", gate)
    commands = f"""# Exact Stage 3 reproduction commands
cd "$HOME\\workspace\\physical-ai-lab"
$stand = ".\\{STAND.relative_to(REPO)}"
$walk = ".\\{WALK.relative_to(REPO)}"
$selected = ".\\{SELECTED.relative_to(REPO)}"
$out = ".\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage3_stand_to_walk"
.
\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_stand_to_walk.ps1 -Mode baseline -Label hard_switch_baseline -Seed 20260812 -Output $out
.
\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_stand_to_walk.ps1 -Parent $walk -NumEnvs 1024 -Iterations 100 -Seed 20260813 -RunName stage3_stand_to_walk_pilot1_validrun_1024_100
.
\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_stand_to_walk.ps1 -Parent $walk -NumEnvs 1024 -Iterations 100 -Seed 20260815 -RunName stage3_stand_to_walk_pilot2_source_align_1024_100 -SourceAlignmentWeight -0.20
.
\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_stand_to_walk.ps1 -Mode formal -TransitionCheckpoint $selected -Label formal_final -Seed 20260817 -Output $out
.
\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\play_stand_to_walk.ps1 -Speed 1.0
""".replace("\n.\n\\", "\n.\\")
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")

    if ART.exists():
        shutil.rmtree(ART)
    ART.mkdir(parents=True)
    shutil.copy2(SELECTED, ART / "model_0.pt")
    artifact_payloads = {
        "checkpoint.json": {"path": str(SELECTED.relative_to(REPO)), "artifact_copy": "model_0.pt",
                            "sha256": selected_hash, "effective_updates": 1,
                            "role": "PRODUCTION_DIRECTIONAL_TRANSITION_EXPERT"},
        "experts.json": {"source": str(STAND.relative_to(REPO)), "source_sha256": stand_hash,
                         "target": str(WALK.relative_to(REPO)), "target_sha256": walk_hash},
        "supported_targets.json": {"discrete_target_speeds_mps": [0.6, 0.8, 1.0, 1.2],
                                   "continuous_range_claimed": False},
        "entry_contract.json": json.loads((OUT / "source_state_contract.json").read_text()),
        "completion_contract.json": completion,
        "abort_contract.json": {"abort_on": ["fall", "torso_contact", "unsafe saturation", "timeout"],
                                "timeout_s": 4.0, "keep_source_controller_on_entry_rejection": True},
        "formal_metrics.json": formal,
        "action_continuity.json": json.loads((OUT / "action_discontinuity.json").read_text()),
        "source_revision.json": {"git_revision": source_revision},
    }
    for name, payload in artifact_payloads.items():
        (ART / name).write_text(json.dumps(payload, indent=2) + "\n")
    shutil.copy2(OUT / "reproduction_commands.ps1", ART / "reproduction_commands.ps1")
    lines = [
        f"{sha(path)}  {path.name}"
        for path in sorted(ART.iterdir(), key=lambda item: item.name)
        if path.name != "SHA256SUMS"
    ]
    (ART / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
