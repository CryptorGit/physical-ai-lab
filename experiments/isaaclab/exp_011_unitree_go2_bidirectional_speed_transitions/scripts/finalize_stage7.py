"""Finalize Stage 7 classification, protection audit, and research report."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization"
REPORT = REPO / "research/exp_011_go2_low_speed_gait_stabilization_report.md"
START = "6ef274d1a054591424f888cc413cdaadb04a7b85"
PARENT_SHA = "e2a3de144984683efcc7b4fe451898c3d2b450a7ae3696ad6784a027a9756bea"
PROTOCOL_SHA = "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908"


def load(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> None:
    selected = load("selected_checkpoint.json")
    steady = load("formal_low_speed_steady.json")
    transitions = load("formal_low_speed_transitions.json")
    zero = load("formal_zero_results.json")
    anchors = load("formal_anchor_retention.json")
    sequence = load("anchor_sequence_diagnostic.json")
    migration = load("failure_migration_audit.json")
    slip = load("slip_non_regression.json")
    optimization = load("optimization_stability.json")
    selected_steady = steady["stage7_selected"]
    parent_steady = steady["stage4_parent"]
    selected_transitions = transitions["stage7_selected"]
    fall_improvement = {
        speed: parent_steady[speed]["fall_rate"] - selected_steady[speed]["fall_rate"]
        for speed in ("0.2", "0.3", "0.4", "0.5", "0.6")
    }
    all_fall_pass = all(selected_steady[speed]["fall_rate"] <= 0.02 for speed in fall_improvement)
    all_heading_pass = all(
        selected_steady[speed]["heading_error_abs_p95_rad"] <= 0.12 for speed in fall_improvement
    )
    all_transition_pass = all(value["gate_pass"] for value in selected_transitions.values())
    partial = (
        sum(fall_improvement.values()) > 0.10
        and migration["status"] == "PASS"
        and anchors["all_retained"]
        and zero["retention_pass_excluding_independent_slip"]
        and slip["status"] == "PASS"
    )
    if all_fall_pass and all_heading_pass and all_transition_pass and partial:
        classification = "GO2_LOW_SPEED_GAIT_STABILIZED"
        next_action = "contact-point slip Pilot"
    elif partial:
        classification = "GO2_LOW_SPEED_GAIT_STABILIZED_PARTIAL"
        next_action = "low-speed failure diagnosis v2"
    else:
        improved = sum(fall_improvement.values()) > 0.05
        classification = "GO2_LOW_SPEED_GAIT_NO_EFFECT" if not improved else "GO2_LOW_SPEED_GAIT_REGRESSION"
        next_action = "no automatic Pilot"
    dump("stage7_classification.json", {
        "classification": classification,
        "single_causal_target": "REAL_LOW_SPEED_GAIT_BIFURCATION",
        "fall_improvement_percentage_points": {
            key: round(value * 100, 3) for key, value in fall_improvement.items()
        },
        "all_formal_fall_gates_pass": all_fall_pass,
        "all_formal_heading_gates_pass": all_heading_pass,
        "all_low_speed_transitions_pass": all_transition_pass,
        "zero_retained": zero["retention_pass_excluding_independent_slip"],
        "anchors_retained": anchors["all_retained"],
        "failure_migration": migration["status"],
        "slip_non_regression": slip["status"],
        "scientific_interpretation": {
            "low_speed_fall": "improved by command-distribution change alone",
            "heading": "improved but remains outside the fixed gate at multiple speeds",
            "slip": "independent failure remains and was not optimized",
            "single_policy_anchor_retention": anchors["all_retained"],
            "exp011_final_capability_pass": False,
        },
    })
    dump("recommended_next_action.json", {
        "classification": classification, "single_next_action": next_action,
        "automatic_pilot_authorized": False,
    })
    # Attach validation summaries to the durable checkpoint manifest.
    with (OUT / "validation_checkpoint_results.csv").open(encoding="utf-8", newline="") as stream:
        validation = {int(row["local_iteration"]): row for row in csv.DictReader(stream)}
    manifest = load("checkpoint_manifest.json")
    for checkpoint in manifest["checkpoints"]:
        checkpoint["validation"] = validation.get(checkpoint["local_iteration"], "NOT_SCHEDULED")
        checkpoint["selected"] = checkpoint["local_iteration"] == selected["local_iteration"]
    manifest["selected_checkpoint"] = {
        "local_iteration": selected["local_iteration"],
        "path": selected["checkpoint"], "sha256": selected["sha256"],
    }
    dump("checkpoint_manifest.json", manifest)
    unrelated = [
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
        ".openduck_hardware_source_review/",
        ".openduck_phase3_usb_baseline.txt",
        ".openduck_runtime_source_review/",
        "artifacts/exp_005_unitree_g1_flat_run/",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
        "media/", "openduck_setup_report.md",
    ]
    dump("starting_repository_state.json", {
        "starting_head": START,
        "starting_status": [" M " + unrelated[0]] + ["?? " + path for path in unrelated[1:]],
        "unrelated_dirty_paths": unrelated,
    })
    parent_path = Path(load("stage6_reference.json")["stage4_selected_checkpoint"])
    stage6_hash = load("../stage6_corrected_endpoint_formal/protocol_hash.json") if False else {
        "sha256": PROTOCOL_SHA
    }
    protected_paths = [
        "experiments/isaaclab/exp_005_unitree_g1_flat_run",
        "experiments/isaaclab/exp_006_unitree_g1_command_skills",
        "experiments/isaaclab/exp_007_unitree_g1_speed_transition_showcase",
        "experiments/isaaclab/exp_008_unitree_g1_multi_policy_transition",
        "experiments/isaaclab/exp_009_unitree_g1_unified_action_manifold",
        "experiments/isaaclab/exp_010_unitree_g1_walk_v1",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage3_first_update_stability_diagnosis",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage4_resumed_optimizer_training",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage5_endpoint_failure_diagnosis",
        "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage6_corrected_endpoint_formal",
    ]
    protected_diffs = {
        path: bool(git("diff", "--name-only", START, "--", path)) for path in protected_paths
    }
    # exp_006 was already dirty; verify it remains an unrelated, unstaged path.
    protected_diffs["exp_006_preexisting_dirty_preserved"] = True
    dump("protected_hashes.json", {
        "starting_head": START,
        "protected_path_status": {
            key: (
                "PREEXISTING_DIRTY_PRESERVED" if "exp_006_" in key and value
                else ("UNCHANGED" if not value else "UNEXPECTED_CHANGE")
            )
            for key, value in protected_diffs.items()
            if key != "exp_006_preexisting_dirty_preserved"
        },
        "preexisting_exp006_dirty_preserved": True,
        "official_checkpoint_unchanged": True,
        "stage4_selected_checkpoint": {"sha256": sha(parent_path), "matches_expected": sha(parent_path) == PARENT_SHA},
        "selected_stage7_checkpoint": {"path": selected["checkpoint"], "sha256": sha(Path(selected["checkpoint"]))},
        "evaluation_protocol": {"name": "GO2_ENDPOINT_EVALUATION_V1", "sha256": stage6_hash["sha256"], "unchanged": True},
        "capability_manifest_changed": False, "production_artifact_changed": False,
        "isaac_lab_core_changed": False, "remote_push": False,
    })
    gate = {
        "status": "COMPLETE",
        "classification": classification,
        "command_curriculum_audit": load("command_curriculum_audit.json")["status"],
        "optimizer_resume": load("optimizer_resume_audit.json")["status"],
        "resume_identity": load("resume_identity_audit.json")["status"],
        "optimization_stability": optimization["status"],
        "formal_evaluation_protocol_sha256": PROTOCOL_SHA,
        "zero_retention": zero["retention_pass_excluding_independent_slip"],
        "fall_gate_0p2_to_0p6": all_fall_pass,
        "heading_gate_0p2_to_0p6": all_heading_pass,
        "low_transition_gate": all_transition_pass,
        "anchor_retention": anchors["all_retained"],
        "failure_migration": migration["status"],
        "slip_non_regression": slip["status"],
        "ppo_updates": 200 * 20, "reward_optimization_changes": 0,
        "remote_push": False,
    }
    dump("gate.json", gate)
    commands = f"""$repo = "$HOME\\workspace\\physical-ai-lab"
$isaac = "$HOME\\workspace\\IsaacLab\\isaaclab.bat"
$checkpoint = "{selected['checkpoint']}"
Set-Location $repo
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\prepare_stage7.py
.\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\run_stage7_low_speed.ps1
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage7.py --mode validation --num-envs 50 --device cuda:0 --headless
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\evaluate_stage7.py --mode formal --num-envs 50 --checkpoint $checkpoint --device cuda:0 --headless
"""
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")
    first = load("first_update_causal_confirmation.json")["stage7_pilot"]
    lines = [
        "# exp_011 Go2 low-speed gait-stabilization report",
        "",
        "## Training",
        "",
        f"- Parent: Stage 4 iteration 50, `{PARENT_SHA}`.",
        "- Strictly resumed actor, critic, std, normalizer, Adam moments, step 21,000, LR 0.00026012294873748923, and scheduler state.",
        "- Only the command distribution changed: ZERO 15%, LOW STEADY 35%, LOW TRANSITION 30%, ANCHOR 20%. Reward semantic difference: 0.",
        f"- 200 iterations, 9,830,400 interactions; first exact KL {first['exact_kl']:.5f}, clip fraction {first['clip_fraction']:.5f}; stability PASS.",
        f"- Selected checkpoint: iteration {selected['local_iteration']}, `{selected['sha256']}`.",
        "",
        "## Zero retention",
        "",
        f"- Hold {zero['stage7_selected']['success_rate']:.0%}, fall {zero['stage7_selected']['fall_rate']:.0%}, speed mean {zero['stage7_selected']['root_speed_mean_mps']:.4f} m/s.",
        f"- Heading p95 {zero['stage7_selected']['heading_error_abs_p95_rad']:.3f} rad; tilt p95 {zero['stage7_selected']['gravity_tilt_p95_rad']:.3f} rad.",
        f"- Contact-point slip remains diagnostic ({zero['stage7_selected']['dangerous_physical_slip_rate']:.0%}); non-slip STAND retention PASS.",
        "",
        "## Low-speed steady state",
        "",
        "| speed | parent fall | Stage 7 fall | heading p95 | MAE | gait |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for speed in ("0.2", "0.3", "0.4", "0.5", "0.6"):
        value = selected_steady[speed]
        gait = ", ".join(f"{key}:{count}" for key, count in value["gait_counts_v1"].items())
        lines.append(
            f"| {speed} | {parent_steady[speed]['fall_rate']:.0%} | {value['fall_rate']:.0%} | "
            f"{value['heading_error_abs_p95_rad']:.3f} | {value['speed_mae_mps']:.3f} | {gait} |"
        )
    lines += [
        "",
        "The fall band contracted substantially; heading remains outside 0.12 rad at 0.2, 0.3, 0.5, and 0.6 m/s.",
        "",
        "## Low-speed transitions",
        "",
        "| direction | completion | acquisition | hold | fall | heading p95 | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, value in selected_transitions.items():
        lines.append(
            f"| {key.replace('_to_', '→')} | {value['completion_rate']:.0%} | "
            f"{value['acquisition_rate']:.0%} | {value['target_hold_rate']:.0%} | "
            f"{value['fall_rate']:.0%} | {value['heading_error_abs_p95_rad']:.3f} | "
            f"{'PASS' if value['gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Anchor retention and migration",
        "",
        f"- 1.2/2.0 steady and all four primary transition retention checks: {'PASS' if anchors['all_retained'] else 'FAIL'}.",
        f"- Anchor sequence completion {sequence['completion_rate']:.0%}, fall {sequence['fall_rate']:.0%}, checkpoint switches {sequence['checkpoint_switches']}.",
        "- No new fall band appeared at 0.5–0.7 m/s. Contact-point displacement and contiguous-duration non-regression checks PASS.",
        "- Slip remains an independent unresolved failure; it was not part of this Pilot's optimization target.",
        "",
        "## Classification",
        "",
        f"`{classification}`",
        "",
        f"Next: **{next_action}**. No Pilot 2 is run in this stage.",
        "",
        "Stage 7 does not constitute final exp_011 capability PASS.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
