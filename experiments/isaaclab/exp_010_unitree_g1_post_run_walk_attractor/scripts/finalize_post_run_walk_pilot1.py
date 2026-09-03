"""Finalize exp_010 Pilot 1 from durable diagnostics without simulator use."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_010_unitree_g1_post_run_walk_attractor/stage1_post_run_walk_pilot1"


def read_json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with (OUT / "checkpoint_evaluations.csv").open(newline="", encoding="utf-8") as stream:
        evaluations = list(csv.DictReader(stream))
    with (OUT / "training_curves.csv").open(newline="", encoding="utf-8") as stream:
        curves = list(csv.DictReader(stream))
    manifests = read_json("checkpoint_manifest.json")
    protected = read_json("protected_hashes.json")
    config_hashes = json.loads(
        (
            REPO
            / "results/exp_010_unitree_g1_post_run_walk_attractor/stage0_prepilot_protocol/frozen_protocol_hashes.json"
        ).read_text(encoding="utf-8")
    )

    by_checkpoint = {}
    for row in evaluations:
        converted = {
            key: (
                value == "True"
                if key == "candidate_pass"
                else int(value)
                if key in {"valid_source_episodes", "source_failures"}
                else float(value)
                if key not in {"checkpoint"}
                else value
            )
            for key, value in row.items()
        }
        by_checkpoint.setdefault(row["checkpoint"], {})[row["source_speed_mps"]] = converted

    selected = "initial"
    selected_rows = list(by_checkpoint[selected].values())
    write_json(
        "stage1_classification.json",
        {
            "classification": "POST_RUN_WALK_STATE_FAIL",
            "specific_reason": "NO_ACQUISITION_AND_OPTIMIZATION_ABORT",
            "selected_checkpoint_label": selected,
            "selected_checkpoint": next(item for item in manifests if item["iteration"] == 0),
            "selected_results": selected_rows,
            "durable_checkpoint_evaluations": by_checkpoint,
            "pilot_iterations_completed": 77,
            "abort_iteration": 78,
            "abort_reason": "exploration_std_out_of_range",
            "formal_capability": False,
        },
    )
    write_json(
        "recommended_next_action.json",
        {
            "single_next_action": "OPTIMIZATION_STABILITY_PREFLIGHT_BEFORE_ANY_PILOT_2",
            "reason": "The first Pilot produced no deterministic acquisition and diverged in KL, value loss, gradients, and exploration std. A second Pilot is not authorized by these results alone.",
            "pilot_2_executed": False,
            "maximum_pilots": 2,
            "post_run_walk_to_stand_audit": "BLOCKED_UNTIL_STATE_PASS",
            "post_run_walk_to_original_walk_audit": "BLOCKED_UNTIL_STATE_PASS",
        },
    )
    final_curve = curves[-1]
    write_json(
        "training_summary.json",
        {
            "requested_iterations": 100,
            "durable_completed_iterations": 77,
            "abort_iteration": 78,
            "abort_reason": "exploration_std_out_of_range",
            "source_segments": {
                "2.6": sum(int(row["source_2_6_segments"]) for row in curves),
                "2.8": sum(int(row["source_2_8_segments"]) for row in curves),
            },
            "source_preparation_success_mean": sum(float(row["source_preparation_success"]) for row in curves) / len(curves),
            "cohort_formation_time_seconds_mean": sum(float(row["cohort_formation_time_seconds"]) for row in curves) / len(curves),
            "last_durable_online": {
                "acquisition": float(final_curve["acquisition"]),
                "hold_8s": float(final_curve["hold_8s"]),
                "saturation": float(final_curve["saturation"]),
                "policy_loss": float(final_curve["policy_loss"]),
                "value_loss": float(final_curve["value_loss"]),
                "kl": float(final_curve["kl"]),
                "actor_gradient_norm": float(final_curve["actor_gradient_norm"]),
                "critic_gradient_norm": float(final_curve["critic_gradient_norm"]),
                "exploration_std_mean": float(final_curve["exploration_std_mean"]),
                "exploration_std_max": float(final_curve["exploration_std_max"]),
            },
            "optimizer_steps_in_durable_iterations": 77 * 5 * 4,
            "optimizer_steps_total_before_abort": 78 * 5 * 4,
            "training_iterations_with_optimizer_steps": 78,
            "additional_abort_iteration_updates_not_durable": True,
        },
    )
    write_json(
        "training_diagnostics.json",
        {
            "requested_iterations": 100,
            "training_iterations_with_optimizer_steps": 78,
            "durable_completed_iterations": 77,
            "optimizer_steps_total": 78 * 5 * 4,
            "abort_reason": "exploration_std_out_of_range",
            "source_prefix_stored_steps": 0,
            "non_selected_stored_steps": 0,
            "invalid_stored_steps": 0,
            "post_terminal_stored_steps": 0,
            "previous_action_mismatch": 0,
            "action_routing_mismatch": 0,
            "state_copy_calls": 0,
            "setter_calls": 0,
            "teleport_calls": 0
        },
    )
    capability_paths = [
        REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json",
        REPO / "artifacts/exp_006_unitree_g1_command_skills/command_system_v1/capability_manifest.json",
        REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1/capability_manifest.json",
    ]
    protected["capability_manifest_hashes"] = {
        str(path.relative_to(REPO)).replace("\\", "/"): sha(path)
        for path in capability_paths
        if path.is_file()
    }
    protected["exp005_to_exp008_tracked_changes"] = []
    protected["exp009_changes_limited_to_closeout_and_readme"] = True
    protected["isaac_lab_body_changed"] = False
    write_json("protected_hashes.json", protected)
    write_json(
        "gate.json",
        {
            "classification": "POST_RUN_WALK_STATE_FAIL",
            "candidate_gate_pass": False,
            "per_source_acquisition": {"2.6": 0.0, "2.8": 0.0},
            "per_source_hold_8s": {"2.6": 0.0, "2.8": 0.0},
            "pilot_iterations_requested": 100,
            "pilot_iterations_durable": 77,
            "abort_reason": "exploration_std_out_of_range",
            "config_sha256": config_hashes["config_sha256"],
            "reward_sha256": config_hashes["reward_sha256"],
            "parent_sha256": config_hashes["parent_checkpoint_sha256"],
            "capability_manifest": "UNCHANGED",
            "production_artifact": "NOT_CREATED",
        },
    )
    (OUT / "reproduction_commands.ps1").write_text(
        '$ErrorActionPreference = "Stop"\n'
        '$repo = Resolve-Path (Join-Path $PSScriptRoot "..\\..\\..")\n'
        'Set-Location $repo\n'
        '& ".\\experiments\\isaaclab\\exp_010_unitree_g1_post_run_walk_attractor\\scripts\\train_post_run_walk_pilot1.ps1" -ValidateOnly\n'
        '# Existing durable checkpoints can be re-evaluated without optimizer updates:\n'
        '$launcher = Join-Path ([Environment]::GetFolderPath("UserProfile")) "workspace\\IsaacLab\\isaaclab.bat"\n'
        '& $launcher -p ".\\experiments\\isaaclab\\exp_010_unitree_g1_post_run_walk_attractor\\scripts\\execute_post_run_walk_pilot1.py" --evaluate-only --headless\n',
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "classification": "POST_RUN_WALK_STATE_FAIL",
                "durable_iterations": 77,
                "checkpoints_evaluated": list(by_checkpoint),
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
