"""Assemble immutable Stage 2R NO_GO results from completed bounded pilots."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
RESULT = REPO / "results/exp_007_unitree_g1_walk_centered_transitions/stage2r_unified_stand_walk"
LOG_ROOT = REPO / "logs/rsl_rl/physical_ai_g1_walk_centered"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
PARENT_SHA = "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621"


def dump(name: str, payload) -> None:
    (RESULT / name).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(pattern: str) -> Path:
    matches = sorted(LOG_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return matches[-1]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def correlation(rows: list[dict], x: str, y: str) -> float | None:
    pairs = [(float(row[x]), float(row[y])) for row in rows if row.get(x, "") not in ("", None) and row.get(y, "") not in ("", None)]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx = [value - mx for value in xs]
    dy = [value - my for value in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else None


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    r0 = one("*stage2r_r0_verified")
    p1 = one("*stage2r_r1_pilot1_1024_100")
    p2 = one("*stage2r_r1_pilot2_1024_100")
    selected_dir = RESULT / "evaluations/selected_diagnostic"
    selected_summary = load(selected_dir / "model_50_summary.json")
    selected_csv = selected_dir / "model_50_episodes.csv"
    shutil.copyfile(selected_csv, RESULT / "episodes.csv")

    p0 = load(RESULT / "model_0_summary.json")
    p1m50 = load(RESULT / "evaluations/pilot1_model50/model_50_summary.json")
    p1m100 = load(RESULT / "evaluations/pilot1_model100/model_100_summary.json")
    p2m50 = load(RESULT / "model_50_summary.json")
    p2m100 = load(RESULT / "model_100_summary.json")
    checkpoints = [
        ("pilot1", "model_0", p0),
        ("pilot1", "model_50", p1m50),
        ("pilot1", "model_100", p1m100),
        ("pilot2", "model_50", p2m50),
        ("pilot2", "model_100", p2m100),
    ]
    sweep = []
    for pilot, label, summary in checkpoints:
        stand = summary["stand"]
        walk = summary["steady_walk"]
        retention_reject = (
            stand["hold_success_rate"] < 0.95
            or stand["fall_rate"] > 0.04
            or stand["flight_fraction"] > 0.0
            or stand["saturation_failure_rate"] > 0.05
            or stand["final_double_support_rate"] < 0.93
        )
        sweep.append(
            {
                "pilot": pilot,
                "checkpoint_label": label,
                "checkpoint": summary["checkpoint"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "stand": stand,
                "steady_walk": walk,
                "per_speed": summary["per_speed"],
                "numeric_r1_gate_pass": summary["r1_gate_pass"],
                "stage1_retention_reject": retention_reject,
                "adoptable": summary["r1_gate_pass"] and not retention_reject,
                "failure_counts": summary["failure_counts"],
            }
        )
    dump(
        "checkpoint_sweep.json",
        {
            "selection_priority": [
                "STAND retention",
                "fall",
                "saturation",
                "sustained WALK",
                "heading",
                "speed tracking",
                "transition success",
            ],
            "checkpoints": sweep,
            "selected_diagnostic_only": selected_summary["checkpoint"],
            "selected_checkpoint_sha256": selected_summary["checkpoint_sha256"],
            "production_checkpoint_selected": False,
            "reason": "No checkpoint combined zero-flight Stage 1 retention with the R1 WALK gate.",
        },
    )

    run_infos = [load(run / "stage2r_run.json") for run in (r0, p1, p2)]
    for info, run in zip(run_infos, (r0, p1, p2)):
        event_files = sorted(run.glob("events.out.tfevents.*"))
        info["training_curve_files"] = [
            {"path": str(path.relative_to(REPO)), "sha256": sha(path), "bytes": path.stat().st_size}
            for path in event_files
        ]
        info["checkpoint_files"] = [
            {"path": str(path.relative_to(REPO)), "sha256": sha(path)}
            for path in sorted(run.glob("model_*.pt"))
        ]
    dump(
        "training_config.json",
        {
            "architecture": {"observation_dim": 123, "action_dim": 37, "hidden_dims": [256, 128, 128], "activation": "elu"},
            "action": {"scale": 0.5, "semantics": "default_joint_position + 0.5 * normalized_position_action"},
            "ppo": {
                "optimizer": "Adam reset at each warm-start branch",
                "learning_rate": 0.0003,
                "entropy_coef": 0.002,
                "gamma": 0.99,
                "lam": 0.95,
                "epochs": 5,
                "mini_batches": 4,
            },
            "exploration_std": {
                "parent_min": 0.07002533972263336,
                "parent_mean": 0.967749297618866,
                "parent_max": 1.421368956565857,
                "frozen_pre_pilot_rule": "strict load, then set all trainable std parameters to 0.25 because parent max > 0.5",
            },
            "num_envs": {"R0": 8, "R1": 1024},
            "runs": run_infos,
            "R2_R3_R4": "NOT_RUN_R1_GATE_BLOCKED",
        },
    )
    dump(
        "parent_provenance.json",
        {
            "checkpoint": str(PARENT.relative_to(REPO)),
            "sha256_expected": PARENT_SHA,
            "sha256_after": sha(PARENT),
            "unchanged": sha(PARENT) == PARENT_SHA,
            "actor_ancestry": "strict loaded",
            "critic_ancestry": "strict loaded",
            "optimizer_ancestry": "reset; parent optimizer not loaded",
            "exploration_std_ancestry": "strict loaded, audited, then reset to frozen safe 0.25 before R1",
            "observation_normalizer": "Identity; no learned normalizer",
            "source_git_revision": "aa04b5734f0bd14828e1f40a027990d230650971",
        },
    )
    heading = load(RESULT / "heading_preflight.json")
    dump(
        "heading_controller.json",
        {
            "selected": heading["selected"]["config"],
            "selection": heading,
            "contract": {
                "target_heading_state": "world controller only",
                "policy_input": "legacy yaw-rate command only",
                "world_xy_in_policy": False,
                "turn_command": False,
            },
            "frozen_before_training": True,
        },
    )
    dump(
        "reward_definition.json",
        {
            "parent_terms_preserved": [
                "linear velocity tracking", "yaw-rate tracking", "vertical velocity",
                "roll/pitch angular velocity", "torque", "action-rate", "joint acceleration",
                "joint limits", "foot slide", "termination",
            ],
            "pilot1_additions": {
                "stand_horizontal_speed_l2": -2.0,
                "stand_yaw_rate_l2": -0.25,
                "stand_flight": -1.0,
                "stand_double_support": 0.25,
                "ankle_pitch_effort_squared_hinge_above_0.95": -0.25,
            },
            "pilot2_only_changes": {
                "stand_horizontal_speed_l2": -3.0,
                "stand_flight": -2.0,
                "stand_double_support": 0.5,
            },
            "gating": {"stand": "abs(command_vx)<=0.05", "walk": "target_vx>=0.6"},
            "fixed_gait_phase_reward": False,
        },
    )
    dump(
        "curriculum_definition.json",
        {
            "R0": {"status": "COMPLETE", "envs": 8, "updates": 2, "purpose": "wiring only"},
            "R1": {
                "status": "FAILED_RETENTION_AFTER_TWO_PILOTS",
                "command_sampling": {"exact_zero_STAND": 0.40, "steady_WALK": 0.60, "walk_speeds_mps": [0.6, 0.8, 1.0, 1.2]},
                "episode_command_switch": False,
                "pilots": [{"seed": 20260725, "iterations": 100}, {"seed": 20260727, "iterations": 100}],
            },
            "R2": {"status": "NOT_RUN_R1_GATE_BLOCKED"},
            "R3": {"status": "NOT_RUN_R1_GATE_BLOCKED"},
            "R4": {"status": "NOT_RUN_R1_GATE_BLOCKED"},
            "formal": {"status": "NOT_RUN_NO_R4_CANDIDATE"},
        },
    )
    dump(
        "phase_results.json",
        {
            "R0": run_infos[0],
            "R1": {
                "pilot1_best": p1m50,
                "pilot2_best": p2m50,
                "gate": "FAIL",
                "blocking_condition": "Every WALK-capable checkpoint produced STAND flight; pilot2 also exceeded STAND fall limits.",
            },
            "R2": "NOT_RUN",
            "R3": "NOT_RUN",
            "R4": "NOT_RUN",
        },
    )
    stand_retention = {
        "stage1_reference": {
            "settle": 0.98, "hold": 0.98, "fall": 0.02, "speed_mean_mps": 0.006718,
            "speed_p95_mps": 0.013348, "pelvis_height_range_mean_m": 0.001803,
            "flight": 0.0, "saturation": 0.0, "final_double_support": 0.98,
        },
        "selected_diagnostic": selected_summary["stand"],
        "retained": False,
        "reject_reasons": ["flight occurrence (>0)", "no checkpoint simultaneously passed strict retention and R1 WALK"],
    }
    dump("stand_retention.json", stand_retention)
    dump(
        "steady_walk_results.json",
        {
            "selected_diagnostic_checkpoint": selected_summary["checkpoint"],
            "aggregate": selected_summary["steady_walk"],
            "per_speed": selected_summary["per_speed"],
            "continuous_formal_supported_range": None,
            "formal_claim": False,
        },
    )
    dump(
        "transition_results.json",
        {
            "STAND_TO_WALK": "NOT_RUN_R1_GATE_BLOCKED",
            "WALK_TO_STAND": "NOT_RUN_R1_GATE_BLOCKED",
            "full_sequence": "NOT_RUN_R1_GATE_BLOCKED",
            "reason": "Curriculum rule forbids R2 until an R1 checkpoint passes strict STAND retention and steady WALK.",
        },
    )
    dump(
        "formal_summary.json",
        {
            "status": "NOT_RUN",
            "episodes": 0,
            "reason": "No R4 formal candidate; R1 was terminally blocked after two bounded pilots.",
            "performance_claim": False,
        },
    )
    with selected_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    walk_rows = [row for row in rows if row["stand_case"].lower() == "false"]
    target = "ankle_saturation_fraction"
    correlation_fields = [
        "command_speed_mps", "speed_error_mean_mps", "pelvis_pitch_p95_rad", "foot_slip_mean_mps",
        "left_contact_fraction", "right_contact_fraction", "action_magnitude_mean",
        "ankle_target_abs_p95_rad", "ankle_position_abs_p95_rad", "ankle_position_error_p95_rad",
        "contact_force_p95_n", "yaw_command_p95_radps", "heading_error_p95_rad",
    ]
    dump(
        "saturation_diagnostics.json",
        {
            "target": target,
            "checkpoint": selected_summary["checkpoint"],
            "walk_episode_correlations": {field: correlation(walk_rows, field, target) for field in correlation_fields},
            "per_speed": {
                speed: {
                    "episodes": len(group := [row for row in walk_rows if float(row["command_speed_mps"]) == speed]),
                    "ankle_saturation_fraction_mean": sum(float(row[target]) for row in group) / len(group),
                    "left_ankle_effort_p95_mean": sum(float(row["ankle_pitch_effort_p95_left"]) for row in group) / len(group),
                    "right_ankle_effort_p95_mean": sum(float(row["ankle_pitch_effort_p95_right"]) for row in group) / len(group),
                }
                for speed in (0.6, 0.8, 1.0, 1.2)
            },
            "dominant_joints": ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            "failure_threshold": "continuous utilization >=0.95 for >=0.20 s",
            "interpretation": "Long dwell failures fell to zero in the selected diagnostic checkpoint, but STAND retention failed; this is not a supported skill result.",
        },
    )
    primary = Counter(row["primary_failure"] or "none" for row in rows)
    all_flags = Counter()
    for row in rows:
        for name, active in json.loads(row["failure_flags"]).items():
            if active:
                all_flags[name] += 1
    dump("failure_counts.json", {"selected_diagnostic_primary": dict(primary), "selected_diagnostic_all_flags": dict(all_flags)})
    routing = selected_summary["routing"]
    routing.update(
        {
            "parent_checkpoint_unchanged": sha(PARENT) == PARENT_SHA,
            "actor_observation_dim": 123,
            "actor_action_dim": 37,
            "world_xy_policy_input": False,
            "capability_manifest_updated": False,
        }
    )
    dump("routing_preflight.json", routing)
    dump(
        "gate.json",
        {
            "stage": "Stage 2R",
            "status": "NO_GO_RETRAIN",
            "eligible_for_R2": False,
            "eligible_for_R3": False,
            "eligible_for_stage3_walk_turn": False,
            "formal_evaluation_run": False,
            "supported_walk_speed_range_mps": None,
            "failures": [
                "No R1 checkpoint combined strict Stage 1 STAND retention with steady WALK.",
                "Pilot 1 WALK-capable checkpoints had non-zero STAND flight.",
                "Pilot 2 model_50 had STAND fall=4% and non-zero flight; model_100 degraded further.",
                "Two-pilot limit reached.",
            ],
            "warnings": [
                "pilot1 model_50 is diagnostic only: WALK 96.25%, STAND hold 98%, but flight > 0",
                "path drift flags remained frequent and are not a production claim",
            ],
            "best_diagnostic_checkpoint": selected_summary["checkpoint"],
            "best_diagnostic_checkpoint_sha256": selected_summary["checkpoint_sha256"],
            "stand_retention": stand_retention,
            "steady_walk": selected_summary["steady_walk"],
            "per_speed": selected_summary["per_speed"],
            "transitions": "NOT_RUN",
            "full_sequence": "NOT_RUN",
            "parent_checkpoint": str(PARENT.relative_to(REPO)),
            "parent_sha256": sha(PARENT),
            "run_expert_used": False,
            "transition_bridge_output_bitwise_zero": True,
            "artifact_created": False,
            "capability_manifest_updated": False,
            "source_git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "next_decision": "Stop existing-expert reuse; pivot to scratch unified locomotion design or distillation.",
        },
    )
    reproduction = f"""# Stage 2R bounded reproduction commands (PowerShell)
cd "$HOME\\workspace\\physical-ai-lab"
$parent = ".\\logs\\rsl_rl\\physical_ai_g1_flat_run\\2026-07-17_21-40-39_stage2_1024_750\\model_4246.pt"

.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_unified_stand_walk.ps1 -Phase R0 -Checkpoint $parent -NumEnvs 8 -Iterations 2 -Seed 20260725 -RunName stage2r_r0_verified
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_unified_stand_walk.ps1 -Phase R1 -Checkpoint $parent -NumEnvs 1024 -Iterations 100 -Seed 20260725 -RunName stage2r_r1_pilot1_1024_100
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\train_unified_stand_walk.ps1 -Phase R1 -Checkpoint $parent -NumEnvs 1024 -Iterations 100 -Seed 20260727 -RunName stage2r_r1_pilot2_1024_100

# Exact selected diagnostic re-evaluation (not a formal skill claim)
.\\experiments\\isaaclab\\exp_007_unitree_g1_walk_centered_transitions\\scripts\\evaluate_unified_stand_walk.ps1 -Checkpoint ".\\{selected_summary['checkpoint']}" -Phase R1 -Output ".\\results\\exp_007_unitree_g1_walk_centered_transitions\\stage2r_unified_stand_walk\\evaluations\\selected_diagnostic" -Seed 20260726 -WalkEpisodesPerSpeed 20 -StandEpisodes 50

# R2/R3/R4/formal intentionally omitted: R1 retention gate failed after two pilots.
"""
    (RESULT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")


if __name__ == "__main__":
    main()
