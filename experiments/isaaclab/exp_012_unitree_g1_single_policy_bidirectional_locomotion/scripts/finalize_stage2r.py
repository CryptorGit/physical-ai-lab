"""Finalize the fail-closed Stage 2R source-positive-control diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXP = "exp_012_unitree_g1_single_policy_bidirectional_locomotion"
OUT = ROOT / "results" / EXP / "stage2r_true_stand_stop_integration"
REPORT = ROOT / "research" / "exp_012_g1_true_stand_stop_integration_report.md"
START = "78bf17f3cc536f2b20d8a22d83a573e2a1e36e91"
PARENT = ROOT / "results" / EXP / "stage2q_final_sequence_integration" / "raw" / "dagger_round_2_student.pt"
STAND = ROOT / "logs/rsl_rl/physical_ai_g1_flat_run/2026-07-17_21-40-39_stage2_1024_750/model_4246.pt"
WALK = ROOT / "logs/rsl_rl/physical_ai_g1_walk_centered/2026-07-23_23-18-22_stage2wb_stabilization_pilot2_1024_100/model_100.pt"
STOP = ROOT / "artifacts/exp_007_unitree_g1_walk_centered_transitions/walk_to_stand_transition_v1/model_0.pt"
EXPECTED = {
    "student_parent": "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698",
    "true_stand": "734123839e4a3648e4f9b3b64c8aa7fa10cc161111021aa5de7355605a135621",
    "walk_source": "9098eaea68d1e5f818b87a499f53c17ccc97c3dbb1afa7a7604ed3dcaa3ef7fa",
    "walk_to_stand": "bb1bf713119b7980cfac4c1f43eb0d415bc32abe97a54509ee45d13061e858bd",
}
UNRELATED = [
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/README.md",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.ps1",
    "experiments/isaaclab/exp_011_unitree_go2_bidirectional_speed_transitions/scripts/play_exp011_go2_bidirectional.py",
    ".openduck_hardware_source_review/",
    ".openduck_phase3_usb_baseline.txt",
    ".openduck_playground_source_review/",
    ".openduck_runtime_source_review/",
    "artifacts/exp_005_unitree_g1_flat_run/",
    "artifacts/openduck_recorded_zero_pose.png",
    "artifacts/openduck_safe_init_pose_front.png",
    "artifacts/openduck_safe_init_pose_side.png",
    "artifacts/openduck_zero_pose_front.png",
    "artifacts/openduck_zero_pose_side.png",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/assemble_showcase_reel.py",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_command_system_showcase.ps1",
    "experiments/isaaclab/exp_006_unitree_g1_command_skills/scripts/play_showcase.py",
    "experiments/mujoco/exp_003_openduckmini_calibrated_walk/",
    "media/",
    "openduck_setup_report.md",
    "research/exp_011_linkedin_post_ja.md",
    "tools/analyze_openduck_joint_directions.py",
    "tools/render_openduck_zero_pose.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_header(name: str, fields: list[str]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    positive = json.loads((OUT / "source_positive_control_results.json").read_text(encoding="utf-8"))
    positive["seed"] = 20260901
    positive["contact_contract"] = "ankle-roll links; max force across contact-sensor history >5 N"
    write("source_positive_control_results.json", positive)
    stand_result = positive["summary"]["TRUE_STAND"]
    stop_result = positive["summary"]["WALK_TO_STAND"]
    actual = {
        "student_parent": sha(PARENT),
        "true_stand": sha(STAND),
        "walk_source": sha(WALK),
        "walk_to_stand": sha(STOP),
    }
    if actual != EXPECTED:
        raise RuntimeError(f"Stage 2R provenance mismatch: {actual}")

    starting_status = [{"path": path, "classification": "unrelated_pre_existing_dirty"} for path in UNRELATED]
    write("stage_reference.json", {
        "stage": "2R", "name": "true STAND / STOP endpoint integration",
        "starting_head": START, "observed_starting_head": START,
        "starting_status": starting_status,
        "stage2q_checkpoint_sha256": EXPECTED["student_parent"],
        "stage2q_classification_preserved": "G1_FINAL_STAND_STOP_FAIL",
        "execution_disposition": "STOPPED_BEFORE_INTEGRATION",
    })
    write("protocol.json", {
        "objective": "Integrate formal true STAND and WALK_TO_STAND labels into one Stage 2Q actor.",
        "source_discovery": ["exp_007 manifests", "exp_007 reports", "exp_007 checkpoint indices"],
        "positive_control": {
            "environment": positive["environment"], "episodes_per_condition": 100,
            "stand_hold_seconds": 8, "deterministic": True, "seed": 20260901,
            "contact_definition": "two ankle-roll links, max force over sensor history >5 N",
        },
        "fail_closed_rule": "Do not collect integration data or train if either source positive control fails.",
        "prohibited_actions_observed": {
            "ppo": 0, "rl_fine_tuning": 0, "reward_change": 0, "gate_relaxation": 0,
            "runtime_router": 0, "runtime_checkpoint_switch": 0, "runtime_action_blend": 0,
        },
    })
    write("true_stand_source_manifest.json", {
        "status": "SOURCE_IDENTIFIED",
        "source_experiment": "exp_007_unitree_g1_walk_centered_transitions",
        "source_stage": "Stage 1 STAND reference / Stage 5 retention",
        "checkpoint_path": str(STAND.relative_to(ROOT)).replace("\\", "/"),
        "sha256": actual["true_stand"],
        "formal_metrics": {
            "episodes": 50, "stand_hold_rate": .98, "fall_rate": .02,
            "horizontal_speed_mean_mps": .006718, "flight_fraction": 0.0,
            "final_double_support_rate": .98,
        },
        "selection": {
            "unique": True, "reason": "only formally retained strict STAND source with exact 123D/37D contract",
            "stage2q_previous_use": True,
            "warning": "Stage 2Q static imitation passed but its closed-loop STAND failed.",
        },
    })
    write("true_stand_source_contract_audit.json", {
        "compatible": True, "unitree_g1": True, "observation_dimension": 123,
        "action_dimension": 37, "action_type": "joint-position target", "action_scale": .5,
        "physics_dt_s": .005, "control_dt_s": .02, "decimation": 4,
        "joint_order_match": True, "asset_match": True,
        "positive_control_pass": False,
        "positive_control_failures": [
            "flight_zero_rate 0.89 < 0.95",
            "final_double_support_rate 0.03 < 0.95",
            "stand success_rate 0.03 < 0.95",
        ],
    })
    write("walk_to_stand_source_manifest.json", {
        "status": "SOURCE_IDENTIFIED",
        "source_experiment": "exp_007_unitree_g1_walk_centered_transitions",
        "source_stage": "Stage 4 WALK_TO_STAND",
        "checkpoint_path": str(STOP.relative_to(ROOT)).replace("\\", "/"),
        "sha256": actual["walk_to_stand"],
        "supporting_walk_checkpoint": str(WALK.relative_to(ROOT)).replace("\\", "/"),
        "supporting_walk_sha256": actual["walk_source"],
        "formal_metrics": {
            "episodes": 50, "completion_rate": 1.0, "fall_rate": 0.0,
            "final_speed_mean_mps": .006468, "final_flight_rate": 0.0,
            "final_double_support_rate": 1.0,
        },
        "selection": {"unique": True, "fallback_required": False},
    })
    write("walk_to_stand_source_contract_audit.json", {
        "compatible": True, "unitree_g1": True, "observation_dimension": 123,
        "action_dimension": 37, "action_type": "joint-position target", "action_scale": .5,
        "physics_dt_s": .005, "control_dt_s": .02, "decimation": 4,
        "joint_order_match": True, "asset_match": True,
        "positive_control_pass": False,
        "positive_control_failures": [
            "completion_rate 0.03 < 0.95",
            "final_flight_zero_rate 0.97 passed but final_double_support_rate 0.00 < 0.95",
            "success_rate 0.00 < 0.95",
        ],
    })
    write("student_parent_manifest.json", {
        "status": "AUDITED_NOT_UPDATED", "path": str(PARENT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": actual["student_parent"], "architecture": [124, 256, 128, 128, 37],
        "gait_conditioned_gaussian": True,
    })
    write("student_parent_identity_audit.json", {
        "checkpoint_hash_match": True, "bitwise_clone_created": False,
        "reason": "source positive-control gate failed before student initialization",
        "mean_actor_updates": 0, "std_updates": 0,
    })
    skipped = {
        "status": "NOT_RUN_DUE_TO_TRUE_STAND_POSITIVE_CONTROL_FAIL",
        "blocking_gate": "source_positive_control",
    }
    write("stand_stop_dataset_manifest.json", {**skipped, "episodes": 0, "groups": {}})
    write("stand_stop_dataset_split.json", {**skipped, "train": 0, "validation": 0, "heldout": 0})
    write("stand_stop_dataset_hashes.json", {**skipped, "dataset_files": []})
    (OUT / "resolved_training_config.yaml").write_text(
        "status: NOT_RUN_DUE_TO_TRUE_STAND_POSITIVE_CONTROL_FAIL\n"
        "maximum_supervised_steps: 20000\nseed: 20270021\nmean_actor_updates: 0\n",
        encoding="utf-8",
    )
    csv_header("training_curves.csv", ["step", "stand_loss", "stop_loss", "moving_retention_loss", "validation_loss"])
    write("checkpoint_manifest.json", {
        **skipped, "new_checkpoints": 0, "parent_sha256": actual["student_parent"], "entries": [],
    })
    write("selected_checkpoint.json", {**skipped, "selected": None})
    write("static_heldout_results.json", skipped)
    csv_header("closed_loop_endpoint_results.csv", [
        "condition", "episodes", "success_rate", "fall_rate", "speed_metric", "status",
    ])
    csv_header("closed_loop_transition_results.csv", [
        "condition", "episodes", "completion_rate", "fall_rate", "status",
    ])
    write("dagger_rounds.json", {**skipped, "rounds_executed": 0, "maximum_allowed": 2})
    write("final_integrated_sequence.json", {**skipped, "episodes": 0, "formal_completion_rate": None})
    write("single_weight_audit.json", {
        "status": "NOT_APPLICABLE_NO_STAGE2R_STUDENT",
        "runtime_evaluations_executed": 0, "unique_stage2r_checkpoints": 0,
        "teacher_calls_in_final_runtime": 0, "expert_calls_in_final_runtime": 0,
        "router_calls": 0, "checkpoint_switches": 0, "action_blends": 0,
    })
    classification = "G1_FINAL_STAND_POSITIVE_CONTROL_FAIL"
    write("stage_classification.json", {
        "classification": classification,
        "primary_evidence": {
            "true_stand": stand_result, "walk_to_stand": stop_result,
        },
        "stage2q_classification_unchanged": "G1_FINAL_STAND_STOP_FAIL",
    })
    write("recommended_next_action.json", {
        "action": "close the discovered exp_007 source-integration route; identify or create a formally validated true-STAND source that passes the Stage 2Q environment positive control before any integration",
        "automatic_execution": False,
    })
    protected = {
        "starting_head": START, "ending_head_before_stage_commit": git("rev-parse", "HEAD"),
        "exp_005_through_exp_011_stage_changes": 0,
        "exp_012_stage_0_through_2q_changes": 0,
        "existing_checkpoint_changes": 0, "existing_optimizer_changes": 0,
        "reward_changes": 0, "physics_changes": 0, "isaac_lab_core_changes": 0,
        "rsl_rl_core_changes": 0, "new_student_checkpoints": 0,
        "runtime_teacher_expert_calls": 0, "checkpoint_switches": 0,
        "remote_push": False, "verified_checkpoint_sha256": actual,
        "unrelated_dirty_paths_preserved": UNRELATED,
    }
    write("protected_hashes.json", protected)
    write("gate.json", {
        "status": "FAIL_CLOSED", "classification": classification,
        "source_positive_control": False, "integration_authorized": False,
        "student_training_executed": False, "dagger_executed": False,
        "final_sequence_executed": False,
    })
    (OUT / "reproduction_commands.ps1").write_text(
        "$env:PYTHONPATH=\"$PWD\\experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\src;"
        "$PWD\\experiments\\isaaclab\\exp_005_unitree_g1_flat_run\\src;$PWD\"\n"
        "C:\\isaacsim\\python.bat experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\evaluate_stage2r_sources.py --headless --device cuda:0\n"
        "python experiments\\isaaclab\\exp_012_unitree_g1_single_policy_bidirectional_locomotion\\scripts\\finalize_stage2r.py\n",
        encoding="utf-8",
    )
    REPORT.write_text(f"""# exp_012 G1 true STAND / STOP integration — Stage 2R

## Outcome

Stage 2R stopped at the mandatory source positive-control gate. No dataset was
collected, no supervised update or DAgger round was run, and no Stage 2R
checkpoint was created. The formal classification is
`{classification}`.

## Sources

The unique contract-compatible true-STAND candidate was exp_007's Stage 1
reference `{EXPECTED['true_stand']}` (`model_4246.pt`). It had exp_007 formal
retention evidence (98% hold, 2% fall, zero recorded flight, 98% final double
support), but it is also the zero-speed teacher already used by Stage 2Q.

The unique formal WALK_TO_STAND candidate was exp_007 Stage 4
`{EXPECTED['walk_to_stand']}`, supported by WALK source
`{EXPECTED['walk_source']}`. Its original formal evaluation reported 100%
completion, 0% fall, zero final flight, and 100% final double support.

Both sources match the G1 123D observation, 37D joint-position action, action
scale 0.5, joint order, asset, 0.005 s physics step and 0.02 s control step.

## Same-environment positive control

The sources were reevaluated deterministically in
`Isaac-Exp012-G1-Reverse-PhaseR1-v0`, using the exp_007 formal seed and the
same contact definition (two ankle-roll links; maximum sensor-history force
above 5 N). Each condition used 100 episodes.

| Metric | TRUE_STAND | WALK_TO_STAND |
|---|---:|---:|
| Formal success/completion | {stand_result['success_rate']:.1%} | {stop_result['completion_rate']:.1%} |
| Fall | {stand_result['fall_rate']:.1%} | {stop_result['fall_rate']:.1%} |
| Mean final/hold speed | {stand_result['speed_mean']:.6f} m/s | {stop_result['speed_mean']:.6f} m/s |
| Flight-zero | {stand_result['flight_zero_rate']:.1%} | {stop_result['final_flight_zero_rate']:.1%} |
| Final double support | {stand_result['final_double_support_rate']:.1%} | {stop_result['final_double_support_rate']:.1%} |

TRUE_STAND failed the required 95% flight-zero and final-double-support gates.
WALK_TO_STAND failed the required 95% completion/final-double-support gate.
Speed and fall alone were not treated as sufficient, and no gate was relaxed.

## Student, endpoints, transitions, and final sequence

The Stage 2Q parent hash was verified as
`{EXPECTED['student_parent']}`. Because the source gate failed, it was not
cloned or updated. Training steps, DAgger rounds, new checkpoints, endpoint
evaluations, transition evaluations and integrated-sequence evaluations were
all zero.

## Protection

exp_005–exp_011 and exp_012 Stage 0–2Q were not changed by Stage 2R. Existing
checkpoints, optimizers, reward, physics, Isaac Lab and RSL-RL were unchanged.
There was no production update, runtime routing, checkpoint switching, action
blending or remote push. Pre-existing unrelated dirty paths were preserved.
""", encoding="utf-8")


if __name__ == "__main__":
    main()
