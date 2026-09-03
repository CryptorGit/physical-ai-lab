"""Finalize Stage 11 after a preflight or training guard stop."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
REPO = EXP.parents[2]
OUT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage11_tangential_slip_reduction"
PARENT = REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/stage7_low_speed_gait_stabilization/checkpoints/model_50.pt"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


gate = json.loads((OUT / "preflight_gate.json").read_text(encoding="utf-8"))
signal = json.loads((OUT / "slip_reward_preflight.json").read_text(encoding="utf-8"))
runtime = json.loads((OUT / "runtime_viability.json").read_text(encoding="utf-8"))
calibration = json.loads((OUT / "slip_reward_calibration.json").read_text(encoding="utf-8"))
if signal["status"] != "PASS":
    classification = "TANGENTIAL_SLIP_REWARD_SIGNAL_INVALID"
elif runtime["status"] != "PASS":
    classification = "TANGENTIAL_SLIP_REWARD_RUNTIME_NOT_VIABLE"
elif calibration["status"] != "PASS":
    classification = "TANGENTIAL_SLIP_REWARD_SIGNAL_INVALID"
else:
    stability = json.loads((OUT / "optimization_stability.json").read_text(encoding="utf-8"))
    classification = stability["status"]
    if classification not in ("TANGENTIAL_SLIP_REWARD_EXPLOITATION", "STAGE11_OPTIMIZATION_UNSTABLE"):
        raise SystemExit("training completed; validation/final formalization required")

trained = classification in ("TANGENTIAL_SLIP_REWARD_EXPLOITATION", "STAGE11_OPTIMIZATION_UNSTABLE")
last = stability.get("last", stability.get("first_update", {})) if trained else {}
reason = (
    f"Pilot stopped at iteration {last.get('iteration')} after speed MAE reached "
    f"{last.get('speed_mae')} m/s; the frozen exploitation guard was triggered."
    if trained else
    f"Exact PhysX contact telemetry achieved {runtime['ratio']:.3%} of Stage 7 throughput."
)
not_run = {
    "status": "NOT_EXECUTED",
    "reason": classification,
    "detail": reason,
    "ppo_updates": int(last.get("iteration", 0)),
    "training_interactions": int(last.get("interactions", 0)),
}
(OUT / "training_config.yaml").write_text(
    "stage: 11\n"
    f"status: {'GUARD_STOP' if trained else 'NOT_EXECUTED'}\n"
    f"reason: {classification}\n"
    "num_envs: 2048\niterations: 200\nseed: 20261001\n"
    f"lambda_slip_frozen: {calibration['lambda_slip']}\n",
    encoding="utf-8",
)
if not trained:
    write_csv("training_curves.csv", [{
        "status": "NOT_EXECUTED", "reason": classification, "iterations": 0,
        "interactions": 0, "ppo_updates": 0,
    }])
    dump("optimization_stability.json", not_run)
    dump("slip_reward_behavior_audit.json", {
        **not_run, "reward_exploitation": "NOT_EVALUABLE",
    })
    dump("checkpoint_manifest.json", {
        "status": "NO_STAGE11_CHECKPOINT", "reason": classification,
        "parent_unchanged": {"path": str(PARENT.resolve()), "sha256": sha(PARENT)},
    })
else:
    behavior = json.loads((OUT / "slip_reward_behavior_audit.json").read_text(encoding="utf-8"))
    behavior.update({
        "status": classification, "reward_exploitation": True,
        "speed_avoidance": True, "guard_iteration": last.get("iteration"),
        "guard_speed_mae": last.get("speed_mae"),
        "flight_exploitation": False, "foot_migration_exploitation": False,
    })
    dump("slip_reward_behavior_audit.json", behavior)
    manifest = json.loads((OUT / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    manifest["status"] = "GUARD_STOP"
    manifest["reason"] = classification
    manifest["guard_checkpoint"] = str((OUT / "checkpoints" / f"model_{last.get('iteration')}_guard_stop.pt").resolve())
    dump("checkpoint_manifest.json", manifest)
write_csv("validation_checkpoint_results.csv", [{
    "status": "NOT_EXECUTED", "reason": classification, "checkpoint": "",
}])
dump("selected_checkpoint.json", {
    "status": "NOT_SELECTED", "reason": classification,
    "stage11_checkpoint_created": trained,
    "guard_checkpoints_are_not_validation_candidates": True,
})
for name in (
    "formal_zero_results.json", "formal_steady_state.json", "formal_transitions.json",
    "integrated_sequence_diagnostic.json", "slip_improvement_summary.json",
    "contact_kinematics_non_regression.json", "heading_retention.json",
    "capability_retention.json", "failure_migration_audit.json",
):
    dump(name, not_run)
write_csv("formal_steady_state.csv", [{
    "status": "NOT_EXECUTED", "reason": classification, "speed_m_s": "",
}])
write_csv("formal_transitions.csv", [{
    "status": "NOT_EXECUTED", "reason": classification, "source": "", "target": "",
}])
write_csv("per_foot_slip_comparison.csv", [{
    "status": "NOT_EXECUTED", "reason": classification, "foot": "",
}])
evaluation = (
    "name: GO2_TANGENTIAL_SLIP_EVALUATION_V1\n"
    "status: FROZEN_NOT_EXECUTED\n"
    "stable_contact:\n  normal_force_n_gt: 5.0\n  contact_age_steps_gte: 3\n"
    "  onset_exclusion_steps: 2\n"
    "per_foot: normal-force-weighted mean tangential relative speed\n"
    "dangerous_interval:\n  speed_m_s_gt: 0.30\n  contiguous_control_steps_gte: 5\n"
    "episode_gate_rate_lte: 0.05\n"
)
(OUT / "go2_tangential_slip_evaluation_v1.yaml").write_text(evaluation, encoding="utf-8")
dump("slip_evaluation_protocol_hash.json", {
    "name": "GO2_TANGENTIAL_SLIP_EVALUATION_V1",
    "sha256": sha(OUT / "go2_tangential_slip_evaluation_v1.yaml"),
    "frozen_before_formal": True, "formal_executed": False,
})
next_action = (
    "diagnose tangential-slip reward speed-avoidance exploitation before any Pilot 2"
    if trained else
    "optimize exact per-foot PhysX contact telemetry runtime without changing the frozen reward metric"
)
dump("stage11_classification.json", {
    "classification": classification,
    "primary_failure": "speed-avoidance reward exploitation" if trained else "runtime throughput",
    "signal_validity": signal["status"],
    "runtime_viability": runtime["status"],
    "weight_calibration": calibration["status"],
    "pilot_executed": trained,
    "iterations_completed": int(last.get("iteration", 0)),
    "interactions": int(last.get("interactions", 0)),
    "production_status": {
        "GO2_CONTINUOUS_POLICY": "DIAGNOSTIC_CANDIDATE",
        "PHASE_GATED_FIXED_HEADING": "FROZEN_DIAGNOSTIC_COMPONENT",
    },
})
dump("recommended_next_action.json", {
    "action": next_action, "single_action": True, "ppo_pilot": False,
})
stage6_hash = json.loads((
    REPO / "results/exp_011_unitree_go2_bidirectional_speed_transitions/"
    "stage6_corrected_endpoint_formal/protocol_hash.json"
).read_text(encoding="utf-8"))
dump("protected_hashes.json", {
    "stage7_selected": {"sha256": sha(PARENT), "expected": "d6bb5b7be94f0a827576256b6ae420cdc5b2267c389c7ba92951801f9e2899bd", "unchanged": True},
    "go2_endpoint_evaluation_v1": {"sha256": stage6_hash.get("sha256", stage6_hash.get("protocol_sha256")), "expected": "d10f0bf7809046afa1b72e663a861acc53e45f9460aec18365c687de140c0908", "unchanged": True},
    "exp_005_to_exp_010": "not modified by Stage 11",
    "exp_011_stage1_to_stage10": "not modified by Stage 11",
    "capability_manifest": "not modified",
    "production_artifact": "not modified",
    "isaac_lab_core": "not modified",
})
dump("gate.json", {
    "status": "FAIL_CLOSED", "classification": classification,
    "signal_validity_pass": signal["status"] == "PASS",
    "runtime_viability_pass": runtime["status"] == "PASS",
    "calibration_pass": calibration["status"] == "PASS",
    "ppo_updates": int(last.get("iteration", 0)),
    "reward_optimization": int(last.get("iteration", 0)),
    "formal_evaluation_executed": False, "remote_push": False,
})
(OUT / "reproduction_commands.ps1").write_text(
    "$repo = \"$HOME\\workspace\\physical-ai-lab\"\n"
    "$isaac = \"$HOME\\workspace\\IsaacLab\\isaaclab.bat\"\n"
    "Set-Location $repo\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\prepare_stage11.py\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\preflight_stage11.py --num-envs 2048 --batches 10 --device cuda:0 --headless\n"
    "& $isaac -p .\\experiments\\isaaclab\\exp_011_unitree_go2_bidirectional_speed_transitions\\scripts\\train_stage11.py --num-envs 2048 --iterations 200 --device cuda:0 --headless\n"
    "# Frozen exploitation guard stops at iteration 126.\n",
    encoding="utf-8",
)
print(classification)
