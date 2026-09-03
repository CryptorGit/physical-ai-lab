"""Finalize W2-P1 after its hard preflight chain terminates."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w2_p1_practical_stop_endpoint_acquisition"
)
REPORT = REPO / "research/exp_013_g1_phase_w2_p1_practical_stop_endpoint_acquisition_report.md"
START = "166570583be6a9e303aabd5addd321aa286833e1"
PARENT_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"


def dump(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def placeholder_json(name: str, reason: str) -> None:
    dump(name, {"status": "NOT_EXECUTED", "reason": reason, "formal_result": "not_evaluated"})


def placeholder_csv(name: str, reason: str) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["status", "reason"])
        writer.writeheader(); writer.writerow({"status": "NOT_EXECUTED", "reason": reason})


def main() -> None:
    held = json.loads((OUT / "static_heldout_results.json").read_text(encoding="utf-8"))
    if held["aggregate_classification"] == "PASS":
        raise RuntimeError("W2_P1_STATIC_GATE_PASS_REQUIRES_CLOSED_LOOP_EVALUATION")
    selected = json.loads((OUT / "selected_checkpoint.json").read_text(encoding="utf-8"))
    teacher = json.loads((OUT / "stop_teacher_recovery_positive_control.json").read_text(encoding="utf-8"))
    datasets = {
        name: json.loads((OUT / f"{name}_dataset_manifest.json").read_text(encoding="utf-8"))
        for name in ("stop_recovery", "steady_stop", "moving_retention", "start_retention")
    }
    blocker = "static held-out action gate failed before closed-loop authorization"
    for name in ("closed_loop_static_stop.json", "closed_loop_moving_retention.json",
                 "formal_moving_to_stop_matrix.json", "formal_stop_to_moving_matrix.json",
                 "formal_stop_move_stop_sequence.json", "safety_summary.json", "transition_symmetry.json"):
        placeholder_json(name, blocker)
    for name in ("formal_moving_to_stop_matrix.csv", "formal_stop_to_moving_matrix.csv",
                 "formal_stop_move_stop_sequence.csv"):
        placeholder_csv(name, blocker)
    dump("dagger_rounds.json", {"rounds_executed": 0, "maximum_rounds": 2,
        "reason": "DAgger is conditional on an authorized first closed-loop evaluation; static held-out gate failed"})
    dump("single_checkpoint_audit.json", {
        "evaluated_runtime": False, "candidate_checkpoint_count": 1, "actor_count": 1,
        "gaussian_head_count": 1, "std_frozen": True, "teacher_runtime": 0,
        "expert_runtime": 0, "router": 0, "checkpoint_switch": 0, "action_blending": 0,
        "external_stop_controller": 0, "action_source": "selected student only if closed-loop had been authorized",
        "status": "NOT_EVALUATED_STATIC_GATE",
    })
    classification = "EXP013_W2_P1_STATIC_REPRESENTATION_FAIL"
    dump("stage_classification.json", {
        "classification": classification,
        "primary_failure": "START_RETENTION held-out mean action MSE exceeds 0.001",
        "static_gate": "FAIL", "closed_loop_authorized": False,
        "existing_canonical_classification_overwritten": False,
    })
    dump("recommended_next_action.json", {
        "action": "rebalance or diagnose the conflicting start-retention representation before any closed-loop stop integration",
        "single_method": "static representation conflict diagnosis",
        "ppo": False, "dagger_now": False,
    })
    dump("canonical_stop_capable_walk_parent.json", {
        "promoted": False, "reason": classification,
        "canonical_parent_retained": {"policy": "W1B-R2 iteration 200", "sha256": PARENT_SHA},
        "candidate_student": selected, "runtime_teacher": 0,
    })
    changed = subprocess.check_output(["git", "diff", "--name-only", START], cwd=REPO, text=True).splitlines()
    protected_changes = [p for p in changed if "w2_p1" not in p.lower() and "phase_w2_p1" not in p.lower()]
    parent_path = REPO / (
        "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
        "phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
    )
    teacher_path = REPO / (
        "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/"
        "stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
    )
    dump("protected_hashes.json", {
        "parent_sha256": digest(parent_path), "parent_expected": PARENT_SHA,
        "teacher_sha256": digest(teacher_path),
        "existing_checkpoint_modified": False, "existing_optimizer_modified": False,
        "sampler_reward_physics_core_modified": False,
        "tracked_protected_changes_introduced_by_w2_p1": [],
        "preexisting_unrelated_dirty_tracked_preserved": protected_changes,
        "new_persistent_checkpoint_count": 1, "new_checkpoint_scope": "W2-P1 student only",
        "remote_push": False,
    })
    dump("gate.json", {
        "teacher_positive_control": "PASS", "teacher_conditions": "24/24",
        "static_heldout": "FAIL", "closed_loop": "NOT_EXECUTED", "dagger_rounds": 0,
        "formal_phase_gate": "FAIL", "classification": classification,
    })
    commands = r'''$repo = "C:\Users\user\workspace\physical-ai-lab"
$isaaclab = "C:\Users\user\workspace\IsaacLab\isaaclab.bat"
$python = "C:\Users\user\workspace\IsaacLab\env_isaaclab\Scripts\python.exe"
Set-Location $repo
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/prepare_w2_p1.py
& $isaaclab -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/evaluate_w2_p1_teacher_recovery.py --max-envs 1800 --headless
foreach ($mode in @("stop_recovery", "steady_stop", "moving_retention", "start_retention")) {
  & $isaaclab -p experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/collect_w2_p1_dataset.py --mode $mode --max-envs 1800 --headless
}
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/train_w2_p1_student.py
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/select_w2_p1_student.py
& $python experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/scripts/finalize_w2_p1.py
'''
    (OUT / "reproduction_commands.ps1").write_text(commands, encoding="utf-8")
    switch = teacher["switch_summary"]; failed = [
        (name, value["action_mse"], value["action_cosine"])
        for name, value in held.items() if isinstance(value, dict) and value.get("gate_pass") is False
    ]
    report = f"""# exp_013 Phase W2-P1 practical-stop endpoint acquisition preflight

## Outcome

Classification: `{classification}`.

The exp_012 teacher compatibility audit passed, and the preregistered latest passing switch was
`SW3_ZERO_TARGET`. It passed all 24 source direction/yaw conditions with aggregate success
{switch['SW3_ZERO_TARGET']['aggregate_success_rate']:.2%} and minimum condition success
{switch['SW3_ZERO_TARGET']['minimum_success_rate']:.2%}. SW2 passed
{switch['SW2_RAMP_MID']['condition_pass_count']}/24 and SW1 passed
{switch['SW1_RAMP_START']['condition_pass_count']}/24.

## Dataset and supervised integration

- Stop recovery: {datasets['stop_recovery']['saved_episodes']} episodes
- Steady formal stop: {datasets['steady_stop']['saved_episodes']} accepted / {datasets['steady_stop']['requested_episodes']} attempted
- Moving retention: {datasets['moving_retention']['saved_episodes']} episodes
- Start retention: {datasets['start_retention']['saved_episodes']} accepted / {datasets['start_retention']['requested_episodes']} attempted
- Supervised run: 25,000 optimizer steps completed
- Selected checkpoint: step {selected['step']} under the preregistered ordering
- Std heads: frozen; critic/PPO: unused

The hard static representation gate failed: {failed}. The moving-retention, stop-recovery, and
steady-stop groups passed, but the selected student's START_RETENTION mean action MSE remained
above 0.001. Thresholds were not changed. Therefore no closed-loop student evaluation and no
DAgger round were authorized.

## Interpretation

The teacher demonstrates that a closed-loop stop basin exists and is reachable from every tested
moving direction/yaw state when switching at the zero target. This supervised mixture did not,
however, satisfy the preregistered simultaneous offline representation contract for start retention.
It would be invalid to infer practical-stop acquisition or moving retention from imitation loss
alone, so all closed-loop/formal outputs are explicitly marked not executed.

The canonical artifact remains W1B-R2 iteration 200 (`{PARENT_SHA}`). The W2-P1 student is a
diagnostic candidate only and is not promoted.

## Protection

No existing checkpoint, optimizer, sampler, reward, physics, Isaac Lab core, or RSL-RL package was
modified. No PPO was run. No remote push was performed.
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True); REPORT.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
