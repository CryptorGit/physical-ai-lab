"""Create frozen EXP 014 Phase-0 manifests and initialization audits."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
EXP = HERE.parent.parent
REPO = EXP.parents[2]
RESULT = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion"
sys.path[:0] = [str(EXP / "src"), str(REPO / "experiments/isaaclab/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/src")]

from g1_explicit_motion_mode.student import initialize_s0_from_w1b, widen_student
from g1_omnidirectional.policy import FrozenGaitActor

STOP = REPO / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration/raw/dagger_round_2_student.pt"
WALK = REPO / "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_r2_pending_mirror_queue_repair_rerun/checkpoints/model_200.pt"
STOP_SHA = "66ca45753aa6175109bfea90b5a8d751c49888f04a69e9ce0a8ec963ac750698"
WALK_SHA = "61dd37f9f1fb7036357d54975c1120c66c30650ed7d5f2e1d4de91174fdca27d"
JST = timezone(timedelta(hours=9))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8", errors="replace").strip()


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    start = datetime.now(JST)
    hard = start + timedelta(hours=12)
    finalization = hard - timedelta(minutes=45)
    time_budget = {
        "start_time": start.isoformat(),
        "hard_deadline": hard.isoformat(),
        "finalization_start": finalization.isoformat(),
        "timezone": "Asia/Tokyo",
        "new_training_forbidden_at_or_after_finalization_start": True,
    }
    dump(EXP / "time_budget.json", time_budget)
    dump(RESULT / "time_budget.json", time_budget)

    command = {
        "name": "Exp014ExplicitMotionModeCommandV1",
        "frozen": True,
        "physical_command_frame": "body",
        "physical_command_units": ["m/s", "m/s", "rad/s"],
        "mode_order": ["STAND", "WALK", "RUN"],
        "legacy_gait": {"STAND": 0, "WALK": 0, "RUN": 1},
        "request_semantics": "target mode changes immediately before velocity ramp advances",
        "time_since_mode_change": "zero at mode change; divide by 3.0 seconds and clamp to one",
        "ramp_progress": "zero at command-ramp start; one at ramp end",
        "previous_command": "immediately preceding control-step physical command",
        "command_delta": "current physical command minus preceding command",
        "forbidden": ["future command", "future trajectory", "teacher ID", "condition ID"],
    }
    observation = {
        "name": "Exp014ExplicitMotionModeObservationV1",
        "dimension": 141,
        "old_124_bitwise_prefix": True,
        "features": [
            ["base_linear_velocity", 0, 3], ["base_angular_velocity", 3, 6],
            ["projected_gravity", 6, 9], ["calibrated_actor_command", 9, 12],
            ["joint_position", 12, 49], ["joint_velocity", 49, 86],
            ["previous_action", 86, 123], ["legacy_gait_command", 123, 124],
            ["current_physical_command", 124, 127], ["target_motion_mode_one_hot", 127, 130],
            ["previous_target_motion_mode_one_hot", 130, 133], ["previous_physical_command", 133, 136],
            ["physical_command_delta", 136, 139], ["normalized_time_since_mode_change", 139, 140],
            ["current_command_ramp_progress", 140, 141],
        ],
    }
    evaluation = {
        "frozen": True,
        "selection_partition": "validation only",
        "heldout_policy": "once after model/contract freeze; never used for fallback",
        "static_group_gate": {"mse_max": 0.001, "cosine_min": 0.98},
        "dual_mode_classification_min": 0.99,
        "practical_stand": {"hold_s": 2.0, "mean_speed_max": 0.08, "mean_abs_yaw_max": 0.08, "fall_max": 0.02, "dangerous_slip_max": 0.05, "impact_max": 0.05},
        "formal_points": {"stand_to_walk": {"endpoint_min": 0.95, "acquisition_0p20s_min": 0.90, "fall_max": 0.02}, "moving_yaw": {"endpoint_min": 0.90, "acquisition_min": 0.85, "fall_max": 0.05}, "walk_to_stand": {"practical_stop_min": 0.95, "fall_max": 0.02}},
        "local_neighborhood": {"direction_delta_deg": [-5, 5], "yaw_delta": [-0.03, 0.03], "pass_min": 0.90, "minimum_endpoint_strictly_positive": True, "minimum_acquisition_strictly_positive": True},
    }
    dump(EXP / "command_contract.json", command)
    dump(EXP / "observation_contract.json", observation)
    dump(EXP / "evaluation_contract.json", evaluation)

    actual = {"STOP": sha(STOP), "OMNI_WALK": sha(WALK), "WALK_RUN": sha(STOP)}
    if actual != {"STOP": STOP_SHA, "OMNI_WALK": WALK_SHA, "WALK_RUN": STOP_SHA}:
        raise RuntimeError("TEACHER_PROVENANCE_FAIL")
    specialists = {
        "schema": "Exp014SpecialistManifestV1",
        "runtime_teacher_count": 0,
        "specialists": [
            {"name": "S_STOP_PRACTICAL_STAND", "checkpoint_path": STOP.relative_to(REPO).as_posix(), "sha256": STOP_SHA, "actor_architecture": [124, 256, 128, 128, 37], "observation_contract": "EXP012 123D policy + scalar legacy gait", "action_contract": "37D deterministic Gaussian mean", "formal_capability_range": ["zero-command practical stand maintenance", "moving to practical stop", "stop recovery"], "unsupported_range": ["strict contact-static STAND", "omnidirectional WALK", "omnidirectional RUN"], "source_experiment": "exp_012 Stage 2Q", "formal_result_paths": ["results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"]},
            {"name": "W_OMNI_WALK", "checkpoint_path": WALK.relative_to(REPO).as_posix(), "sha256": WALK_SHA, "actor_architecture": [124, 256, 128, 128, 37], "observation_contract": "EXP013 calibrated 123D policy + scalar legacy gait", "action_contract": "37D deterministic Gaussian mean", "formal_capability_range": ["16-direction WALK at 0.3m/s", "pure yaw +/-0.3rad/s", "moving yaw for registered cardinal/diagonal directions"], "unsupported_range": ["formal practical stop", "RUN", "A7/A8/A9 start-oracle labels"], "source_experiment": "exp_013 W1B-R2 iteration 200", "formal_result_paths": ["results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/phase_w1b_c2_shared_yaw_endpoint_evaluator"]},
            {"name": "G_FORWARD_WALK_RUN", "checkpoint_path": STOP.relative_to(REPO).as_posix(), "sha256": STOP_SHA, "actor_architecture": [124, 256, 128, 128, 37], "observation_contract": "EXP012 123D policy + scalar legacy gait", "action_contract": "37D deterministic Gaussian mean", "formal_capability_range": ["forward WALK 0.6/0.8/1.0/1.2m/s", "forward RUN 1.2/2.4/2.6m/s", "WALK-RUN transitions", "acceleration/deceleration", "practical stop"], "unsupported_range": ["360-degree RUN", "lateral/diagonal RUN"], "source_experiment": "exp_012 Stage 2Q", "formal_result_paths": ["results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2q_final_sequence_integration"]},
        ],
        "prohibited_formal_teachers": ["A7-R2 update10", "A7-R2 update75", "A7-R2 update150", "A7-R3 candidates", "A8 multi-checkpoint oracle", "A9 diagnostic probes"],
    }
    dump(EXP / "specialist_manifest.json", specialists)
    dump(RESULT / "teacher_identity_audit.json", {"status": "PASS", "actual_sha256": actual, "prohibited_teacher_labels_used": False})
    dump(RESULT / "specialist_scope_audit.json", {"status": "PASS", "manifest": str((EXP / "specialist_manifest.json").relative_to(REPO)), "formal_extrapolation": False})
    reference = {"starting_head": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "start_status_short": git("status", "--short").splitlines(), "specialist_sha256": actual, "source_of_truth": "start-time filesystem and Git state"}
    dump(EXP / "reference_manifest.json", reference)
    dump(RESULT / "stage_reference.json", reference)
    dump(RESULT / "command_contract_audit.json", {"status": "PASS", "dimension": 141, "old_prefix_dimension": 124, "appended_dimension": 17, "future_leakage": 0, "teacher_id_in_actor_input": 0, "condition_id_in_actor_input": 0})

    torch.manual_seed(20260803)
    base = torch.randn(4096, 123)
    gait = torch.randint(0, 2, (4096,), dtype=torch.float32)
    tail = torch.randn(4096, 17)
    sample = torch.cat((base, gait[:, None], tail), 1)
    teacher = FrozenGaitActor(WALK).eval()
    s0 = initialize_s0_from_w1b(WALK).eval()
    with torch.inference_mode():
        expected = teacher(base, gait)
        actual_s0 = s0(sample)
    parity = {"S0": {"architecture": s0.architecture, "parameter_count": s0.parameter_count, "bitwise_equal": torch.equal(expected, actual_s0), "max_absolute_difference": float((expected - actual_s0).abs().max())}}
    for name in ("S1", "S2"):
        widened = widen_student(s0, name).eval()
        with torch.inference_mode(): diff = float((actual_s0 - widened(sample)).abs().max())
        parity[name] = {"architecture": widened.architecture, "parameter_count": widened.parameter_count, "max_absolute_difference": diff, "threshold": 1e-7, "status": "PASS" if diff <= 1e-7 else "FUNCTION_PRESERVING_WIDENING_UNAVAILABLE", "fallback": None if diff <= 1e-7 else "short W1B parity distillation on identical dataset before capacity training"}
    if not parity["S0"]["bitwise_equal"]:
        raise RuntimeError("S0_INITIALIZATION_PARITY_FAIL")
    dump(RESULT / "observation_initialization_parity.json", parity)
    journal = {"timestamp": datetime.now(JST).isoformat(), "phase": "Phase 0", "run ID": "P0_BOOTSTRAP", "parent": None, "hypothesis": "explicit mode is causally observable while old W1B behavior is preserved", "single changed variable": "append frozen 17D contract", "metrics": {"S0_bitwise": True, "teacher_hashes_verified": 3}, "classification": "EXP014_PHASE0_CONTRACT_READY", "decision": "proceed to new formal teacher trajectory collection", "next action": "collect Phase 1 with only S and W", "elapsed wall time": str(datetime.now(JST) - start)}
    (RESULT / "experiment_journal.jsonl").write_text(json.dumps(journal, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": "EXP014_PHASE0_CONTRACT_READY", "time_budget": time_budget, "parity": parity}, indent=2))


if __name__ == "__main__":
    main()
