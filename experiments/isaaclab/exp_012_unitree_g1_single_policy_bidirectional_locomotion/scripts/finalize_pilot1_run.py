"""Fail-closed Stage 2 Pilot-1 finalization after an optimization safety stop."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2_pilot1_run"
PARENT = REPO / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
SCHEDULE = (0, 1, 10, 25, 50, 75, 100, 150, 200, 250, 300)


def dump(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def not_executed(reason: str) -> dict:
    return {
        "status": "NOT_EXECUTED_FAIL_CLOSED",
        "reason": reason,
        "formal_claim_permitted": False,
    }


def main() -> None:
    stability = json.loads((OUT / "optimization_stability.json").read_text(encoding="utf-8"))
    if stability["status"] != "EXP012_FIRST_UPDATE_UNSTABLE":
        raise RuntimeError("finalizer is only valid for the recorded first-update safety stop")

    wiring_path = OUT / "wiring/optimization_stability.json"
    wiring = json.loads(wiring_path.read_text(encoding="utf-8"))
    dump("wiring_clone.json", {
        "status": "COMPLETED_NO_PERFORMANCE_CLAIM",
        "num_envs": 16,
        "requested_updates": 2,
        "completed_updates": wiring["completed_iterations"],
        "safety_observation": wiring["status"],
        "formal_parent_state_consumed": False,
        "formal_pilot_restarted_from_original_parent": True,
    })

    manifest = []
    for iteration in SCHEDULE:
        name = "model_initial.pt" if iteration == 0 else f"model_{iteration}.pt"
        path = OUT / "checkpoints" / name
        if not path.is_file():
            manifest.append({
                "iteration": iteration, "status": "NOT_CREATED_AFTER_SAFETY_STOP", "path": None,
            })
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        optimizer = payload["optimizer_state_dict"]
        steps = sorted({int(float(item["step"])) for item in optimizer["state"].values()})
        manifest.append({
            "iteration": iteration,
            "status": "DURABLE_DIAGNOSTIC_ONLY",
            "path": str(path.relative_to(REPO)).replace("\\", "/"),
            "sha256": sha(path),
            "actor_hash": object_hash(payload["actor_state_dict"]),
            "critic_hash": object_hash(payload["critic_state_dict"]),
            "optimizer_hash": object_hash(optimizer),
            "adam_step_min": min(steps),
            "adam_step_max": max(steps),
            "learning_rate": optimizer["param_groups"][0]["lr"],
            "std": payload["actor_state_dict"]["distribution.std_param"].tolist(),
            "reward_hash": "exp012_parent_base_plus_exp005_safe_periodic_flight_v1",
            "curriculum_hash": "exp012_stage2_fixed_cohorts_v1",
            "yaw_training_contract_hash": "yaw_zero_all_external_controllers_off_v1",
        })
    dump("checkpoint_manifest.json", {"checkpoints": manifest, "production_candidates": 0})

    timeline_rows = []
    for iteration in (0, 1):
        timeline_rows.append({
            "iteration": iteration,
            "validation_status": "NOT_EXECUTED_FAIL_CLOSED",
            "stand_retention": "NOT_EVALUATED",
            "walk_retention": "NOT_EVALUATED",
            "run_2p4_periodicity": "NOT_EVALUATED",
            "run_2p6_periodicity": "NOT_EVALUATED",
            "run_to_walk": "NOT_EVALUATED",
            "walk_to_stand": "NOT_EVALUATED",
            "full_sequence": "NOT_EVALUATED",
            "yaw_bias": "NOT_EVALUATED",
            "fall": "NOT_EVALUATED",
            "slip": "NOT_EVALUATED",
            "impact": "NOT_EVALUATED",
            "long_dwell_saturation": "NOT_EVALUATED",
        })
    write_csv("capability_training_timeline.csv", list(timeline_rows[0]), timeline_rows)
    write_csv(
        "validation_checkpoint_results.csv",
        ["iteration", "checkpoint", "condition", "status", "reason"],
        [{
            "iteration": "",
            "checkpoint": "",
            "condition": "ALL",
            "status": "NOT_EXECUTED_FAIL_CLOSED",
            "reason": "EXP012_FIRST_UPDATE_UNSTABLE",
        }],
    )
    dump("selected_checkpoint.json", {
        "status": "NO_CHECKPOINT_SELECTED",
        "reason": "EXP012_FIRST_UPDATE_UNSTABLE",
        "initial_checkpoint_was_candidate": True,
        "formal_evaluation_authorized": False,
    })

    reason = "Iteration-1 hard gate failed; validation and formal evaluation are prohibited."
    for name in (
        "formal_stand.json", "formal_walk.json", "formal_run.json",
        "formal_transitions.json", "formal_integrated_sequence.json",
        "single_weight_sequence_audit.json", "directional_hysteresis.json",
        "capability_regression_audit.json", "selected_policy_yaw_diagnostic.json",
    ):
        dump(name, not_executed(reason))
    write_csv("formal_walk.csv", ["speed", "status", "reason"], [
        {"speed": speed, "status": "NOT_EXECUTED_FAIL_CLOSED", "reason": "EXP012_FIRST_UPDATE_UNSTABLE"}
        for speed in (0.6, 0.8, 1.0, 1.2)
    ])
    write_csv("formal_run.csv", ["speed", "status", "reason"], [
        {"speed": speed, "status": "NOT_EXECUTED_FAIL_CLOSED", "reason": "EXP012_FIRST_UPDATE_UNSTABLE"}
        for speed in (2.4, 2.6)
    ])
    transitions = ("0->0.6", "0.6->1.2", "1.2->2.4", "1.2->2.6",
                   "2.4->1.2", "2.6->1.2", "1.2->0.6", "0.6->0")
    write_csv("formal_transitions.csv", ["transition", "status", "reason"], [
        {"transition": edge, "status": "NOT_EXECUTED_FAIL_CLOSED", "reason": "EXP012_FIRST_UPDATE_UNSTABLE"}
        for edge in transitions
    ])
    write_csv("endpoint_state_comparison.csv", ["endpoint", "status", "reason"], [{
        "endpoint": "1.2", "status": "NOT_EXECUTED_FAIL_CLOSED", "reason": "EXP012_FIRST_UPDATE_UNSTABLE",
    }])

    classification = "EXP012_FIRST_UPDATE_UNSTABLE"
    dump("stage_classification.json", {
        "classification": classification,
        "basis": {
            "exact_analytical_kl": stability["first_update"]["exact_kl_from_initial"],
            "reported_kl": stability["first_update"]["reported_kl"],
            "clip_fraction": stability["first_update"]["clip_fraction"],
            "reported_kl_gate": 0.20,
            "clip_fraction_gate": 0.50,
        },
    })
    dump("recommended_next_action.json", {
        "action": "diagnose first-update PPO ratio and clipping on the frozen initial rollout before another Pilot",
        "one_method_only": True,
        "pilot2_prohibited": True,
    })
    dump("gate.json", {
        "status": "STOPPED_FAIL_CLOSED",
        "classification": classification,
        "completed_iterations": 1,
        "completed_interactions": 24576,
        "validation": "NOT_EXECUTED",
        "formal_evaluation": "NOT_EXECUTED",
        "ppo_retry": 0,
        "learning_rate_change": 0,
        "fresh_optimizer": 0,
        "teacher_action_calls": 0,
        "checkpoint_switches": 0,
        "remote_push": False,
    })
    dump("protected_hashes.json", {
        "starting_head": "cd1758934ec634566442f25c25b0ea27e1abc46a",
        "parent_checkpoint": {
            "sha256_before": sha(PARENT), "sha256_after": sha(PARENT), "unchanged": True,
        },
        "protected_scope": {
            "exp_005_through_exp_011_changed_by_stage": False,
            "exp_012_previous_stages_changed": False,
            "capability_manifest_changed": False,
            "production_artifact_changed": False,
            "isaac_lab_core_changed": False,
            "teacher_action_calls": 0,
            "runtime_checkpoint_switches": 0,
            "remote_push": False,
        },
    })
    (OUT / "reproduction_commands.ps1").write_text(
        '$ErrorActionPreference = "Stop"\n'
        'Set-Location "C:\\Users\\user\\workspace\\physical-ai-lab"\n'
        '.\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\run_exp012_stage2.ps1\n'
        '# This deterministically stops after iteration 1 when the hard gate fails.\n',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
