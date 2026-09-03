"""Close Stage 2 after its mandatory first-update stability gate fails."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage2_continuous_0_to_2_training"
STAGE1 = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage1_single_policy_baseline"


def read(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        for item in sorted(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
            digest.update(str(item.relative_to(path)).replace("\\", "/").encode())
            digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        if torch.is_tensor(value):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def git(*parts, cwd=REPO):
    return subprocess.run(["git", *parts], cwd=cwd, text=True, capture_output=True, check=True).stdout.strip()


def checkpoint_entry(path: Path, iteration, validation_status, policy_kl):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    std = payload["actor_state_dict"]["distribution.std_param"]
    return {
        "checkpoint_path": str(path.resolve()),
        "sha256": sha(path),
        "iteration": iteration,
        "actor_parameter_hash": state_hash(payload["actor_state_dict"]),
        "critic_parameter_hash": state_hash(payload["critic_state_dict"]),
        "validation_metrics": {"status": validation_status},
        "log_std": {"mean": float(torch.log(std).mean()), "max": float(torch.log(std).max())},
        "policy_kl_from_initial": policy_kl,
        "command_curriculum_hash": read("command_curriculum_config.json")["sha256"],
        "reward_hash": read("stage2_reward_config.json")["sha256"],
        "optimizer_inherited": False,
    }


def not_run(name, reason):
    dump(name, {"status": "NOT_RUN", "reason": reason, "fail_closed": True})


def main():
    stability = read("optimization_stability.json")
    first = stability["first_update"]
    # The interrupted runner exits before its normal CSV flush; preserve the one completed update.
    curve_fields = [
        "iteration", "interaction_count", "reward_mean", "done_fraction", "value_loss",
        "policy_loss", "entropy", "approximate_kl", "clip_fraction",
        "actor_gradient_norm", "critic_gradient_norm", "log_std_mean", "log_std_max",
        "mean_action_l2_shift_from_initial", "policy_kl_from_initial", "finite", "elapsed_s",
    ]
    with (OUT / "training_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=curve_fields)
        writer.writeheader()
        writer.writerow({key: first[key] for key in curve_fields})

    initial = OUT / "checkpoints/model_initial.pt"
    unstable = OUT / "checkpoints/model_1_unstable.pt"
    missing_iterations = [25, 50, 75, 100, 150, 200, 250, 300]
    manifest = {
        "status": "STOPPED_FIRST_UPDATE_STABILITY_GATE",
        "entries": [
            checkpoint_entry(initial, 0, "NOT_VALIDATED_PARENT_WARMSTART", 0.0),
            checkpoint_entry(unstable, 1, "REJECTED_GO2_TRAINING_UNSTABLE", first["policy_kl_from_initial"]),
        ],
        "required_but_not_created": [
            {"iteration": value, "reason": "fail-closed after first-update approximate KL > 0.20"}
            for value in missing_iterations
        ],
        "automatic_final_checkpoint_selection": False,
    }
    dump("checkpoint_manifest.json", manifest)
    dump("selected_checkpoint.json", {
        "status": "NOT_SELECTED",
        "reason": "GO2_TRAINING_UNSTABLE at first update; unstable checkpoint is ineligible",
        "validation_seed_root": 20261901,
        "selection_changed_after_formal": False,
    })
    reason = "Mandatory first-update stability gate failed: approximate KL 0.51294 > 0.20."
    with (OUT / "validation_checkpoint_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": reason})
    with (OUT / "formal_steady_state_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": reason})
    with (OUT / "formal_transition_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": reason})
    with (OUT / "endpoint_hysteresis.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "reason"])
        writer.writeheader()
        writer.writerow({"status": "NOT_RUN", "reason": reason})
    for name in (
        "formal_stand_results.json",
        "formal_steady_state_results.json",
        "formal_transition_results.json",
        "formal_reduced_sequence.json",
        "directional_asymmetry.json",
        "diagnostic_2p5_results.json",
    ):
        not_run(name, reason)

    classification = "GO2_TRAINING_UNSTABLE"
    dump("stage2_classification.json", {
        "classification": classification,
        "trigger": {
            "metric": "first_update_approximate_kl",
            "value": first["approximate_kl"],
            "maximum": 0.20,
        },
        "supporting_metrics": {
            "clip_fraction": first["clip_fraction"],
            "policy_kl_from_initial": first["policy_kl_from_initial"],
            "mean_action_l2_shift": first["mean_action_l2_shift_from_initial"],
            "actor_gradient_norm": first["actor_gradient_norm"],
            "critic_gradient_norm": first["critic_gradient_norm"],
            "value_loss": first["value_loss"],
            "nan_inf": not first["finite"],
        },
        "iterations_completed": 1,
        "iterations_requested": 300,
        "formal_evaluation_executed": False,
    })
    next_action = "first-update PPO stability diagnosis with the frozen Stage 2 contract"
    dump("recommended_next_action.json", {
        "classification": classification,
        "next": next_action,
        "single_recommendation": True,
        "pilot2_not_authorized": True,
    })
    dump("gate.json", {
        "overall_pass": False,
        "classification": classification,
        "warmstart_pass": read("warmstart_audit.json")["status"] == "PASS",
        "command_curriculum_audit_pass": read("command_curriculum_audit.json")["pass"],
        "reward_semantic_difference": 0,
        "optimization_stability_pass": False,
        "formal_evaluation_allowed": False,
        "threshold_relaxed": False,
    })
    selected_parent = json.loads((STAGE1 / "stage0_selected_baseline.json").read_text())["selected"]
    protected_dirs = []
    for index in range(5, 11):
        protected_dirs.extend((REPO / "experiments/isaaclab").glob(f"exp_{index:03d}_*"))
    capability = [
        REPO / "experiments/isaaclab/exp_007_unitree_g1_walk_centered_transitions/capability_manifest.json",
        REPO / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_run_transition_v1",
        REPO / "artifacts/exp_006_unitree_g1_command_skills/command_system_v1",
    ]
    stage1_hashes = {
        path.name: sha(path) for path in STAGE1.iterdir() if path.is_file()
    }
    original_stage1 = read("stage1_reference.json")["stage1_directory_hashes"]
    isaac = REPO.parent / "IsaacLab"
    parent_path = Path(selected_parent["checkpoint_path"])
    dump("protected_hashes.json", {
        "protected_paths": {
            str(path.relative_to(REPO)): sha(path) for path in [*protected_dirs, *capability]
        },
        "stage1": {
            "current_hashes": stage1_hashes,
            "starting_hashes": original_stage1,
            "unchanged": stage1_hashes == original_stage1,
        },
        "official_checkpoint": {
            "path": str(parent_path),
            "expected_sha256": selected_parent["sha256"],
            "current_sha256": sha(parent_path),
            "unchanged": sha(parent_path) == selected_parent["sha256"],
        },
        "isaaclab": {
            "head": git("rev-parse", "HEAD", cwd=isaac),
            "tracked_diff": git("diff", "--name-only", cwd=isaac).splitlines(),
            "core_unchanged": not bool(git("diff", "--name-only", cwd=isaac)),
        },
        "remote_push": False,
    })
    reproduction = f'''cd "$HOME\\workspace\\physical-ai-lab"
$isaac = "$HOME\\workspace\\IsaacLab\\isaaclab.bat"
$checkpoint = "{selected_parent["checkpoint_path"]}"
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\audit_stage2_protocol.py
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\train_stage2_continuous.py --mode wiring --checkpoint $checkpoint --headless
# Pilot stops fail-closed after update 1 with GO2_TRAINING_UNSTABLE under the frozen protocol.
& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\train_stage2_continuous.py --mode pilot --checkpoint $checkpoint --headless
'''
    (OUT / "reproduction_commands.ps1").write_text(reproduction, encoding="utf-8")
    dump("gui_diagnostic.json", {
        "status": "NOT_RUN",
        "reason": "No validation-selected checkpoint exists after GO2_TRAINING_UNSTABLE.",
        "unstable_checkpoint_playback_prohibited": True,
    })


if __name__ == "__main__":
    main()
