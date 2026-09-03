"""Run pure Stage 0-1 checks and emit an honest Stage 1 gate report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP_ROOT.parents[2]
SRC_ROOT = EXP_ROOT / "src"
RESULTS_ROOT = REPO_ROOT / "results" / "exp_015_qmini_population_bodily_world_model"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qmini_population_bwm.data_schema import (
    ACTION_DIM,
    OBSERVATION_DIM,
    CanonicalBodilyObservation,
    TransitionRecord,
    observation_field_names,
    validate_reward_separation,
)
from qmini_population_bwm.evaluation import FAILURE_TAXONOMY, classify_stage1_gate
from qmini_population_bwm.executor import ActionExecutor
from qmini_population_bwm.fatigue import FatigueLedger
from qmini_population_bwm.qmini_asset import (
    OFFICIAL_URDF_PATH,
    QMINI_JOINT_ORDER,
    load_qmini_contract,
    validate_qmini_contract,
)
from qmini_population_bwm.snapshot_clone import QminiSnapshot, deterministic_branch_replay


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def protected_source_gate() -> tuple[bool, dict[str, Any]]:
    isaaclab = REPO_ROOT.parent / "IsaacLab"
    if not (isaaclab / ".git").exists():
        return False, {"status": "NOT_FOUND", "path": str(isaaclab)}
    result = subprocess.run(
        ["git", "-C", str(isaaclab), "diff", "--name-only"],
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    protected = [
        path
        for path in tracked
        if path.startswith(("source/", "apps/", "tools/", "docs/"))
    ]
    return not protected, {
        "status": "PASS" if not protected else "FAIL",
        "tracked_diff_paths": tracked,
        "protected_source_diff_paths": protected,
        "untracked_upstream_files_ignored_as_preexisting": True,
    }


def snapshot_gate() -> tuple[bool, dict[str, Any]]:
    rng = random.Random(15015)
    snapshot = QminiSnapshot(
        root_pose=(0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0),
        root_velocity=(0.0,) * 6,
        joint_q=(0.0,) * 10,
        joint_dq=(0.0,) * 10,
        actuator_controller_state={"integrator": [0.0] * 10},
        previous_action=(0.0,) * 10,
        current_command=(0.2, 0.0, 0.0),
        contact_related_state={"left": False, "right": False},
        friction=1.0,
        wind_xy=(0.0, 0.0),
        fatigue_left=(0.0,) * 5,
        fatigue_right=(0.0,) * 5,
        rng_state=rng.getstate(),
        episode_time=0.0,
    )

    def step(current: QminiSnapshot, action: tuple[float, ...]) -> QminiSnapshot:
        local_rng = random.Random()
        local_rng.setstate(current.rng_state)
        q = tuple(value + 0.001 * command for value, command in zip(current.joint_q, action, strict=True))
        local_rng.random()
        current.joint_q = q
        current.previous_action = action
        current.episode_time += 0.015
        current.rng_state = local_rng.getstate()
        return current

    first, second = deterministic_branch_replay(snapshot, (0.1,) * 10, step_fn=step)
    exact = first.to_jsonable() == second.to_jsonable()
    return exact, {
        "status": "PASS" if exact else "FAIL",
        "same_snapshot_same_action_same_rng": exact,
        "snapshot_fields_present": [
            "root_pose", "root_velocity", "joint_q", "joint_dq",
            "actuator_controller_state", "previous_action", "current_command",
            "contact_related_state", "friction", "wind_xy", "fatigue_left",
            "fatigue_right", "rng_state", "episode_time", "recurrent_state",
        ],
    }


def schema_gate() -> tuple[bool, dict[str, Any]]:
    observation = CanonicalBodilyObservation(
        base_linear_velocity_b=(0.0, 0.0, 0.0),
        base_angular_velocity_b=(0.0, 0.0, 0.0),
        projected_gravity_b=(0.0, 0.0, -1.0),
        joint_position=(0.0,) * 10,
        joint_velocity=(0.0,) * 10,
        previous_actually_applied_action=(0.0,) * 10,
        left_foot_contact=0.0,
        right_foot_contact=0.0,
    )
    record = TransitionRecord(
        episode_id="schema-smoke",
        source_snapshot_id="schema-snapshot",
        t=0,
        observation=observation,
        next_observation=observation,
        action_proposed=(0.0,) * 10,
        action_applied=(0.0,) * 10,
        command=(0.2, 0.0, 0.0),
        reward_vector={"velocity_tracking": 0.0},
        hidden_state_for_analysis={"friction": 1.0},
    )
    failures = validate_reward_separation(record)
    ok = (
        OBSERVATION_DIM == 41
        and ACTION_DIM == 10
        and len(observation.flatten()) == OBSERVATION_DIM
        and not failures
        and "teacher_id" not in observation_field_names()
    )
    return ok, {
        "status": "PASS" if ok else "FAIL",
        "observation_dim": OBSERVATION_DIM,
        "action_dim": ACTION_DIM,
        "field_names": list(observation_field_names()),
        "reward_separation_failures": failures,
    }


def fatigue_gate() -> tuple[bool, dict[str, Any]]:
    ledger = FatigueLedger(alpha=0.2, beta=0.1, effectiveness_coefficient=0.5)
    step = ledger.step((1.0,) * 5, (0.0,) * 5, dt=0.015)
    expected_left = 0.2
    ok = all(abs(value - expected_left) < 1e-12 for value in step.after.left) and all(
        value == 0.0 for value in step.after.right
    )
    return ok, {
        "status": "PASS" if ok else "FAIL",
        "expected_left_after_one_step": expected_left,
        "left_after": list(step.after.left),
        "right_after": list(step.after.right),
        "effectiveness_after": list(step.effectiveness),
    }


def action_logging_gate() -> tuple[bool, dict[str, Any]]:
    executor = ActionExecutor(lower=(-1.0,) * 10, upper=(1.0,) * 10)
    applied = executor.apply((2.0,) + (0.0,) * 9)
    ok = applied.action_proposed[0] == 2.0 and applied.action_applied[0] == 1.0 and applied.saturation_mask[0]
    return ok, {
        "status": "PASS" if ok else "FAIL",
        "action_proposed_present": applied.action_proposed is not None,
        "action_applied_present": applied.action_applied is not None,
        "saturation_logged": applied.saturation_mask[0],
    }


def run_pytest() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=EXP_ROOT,
        env={**dict(__import__("os").environ), "PYTHONPATH": str(SRC_ROOT)},
        capture_output=True,
        text=True,
    )
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--isaac-import-report", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "stage1_gate.json")
    args = parser.parse_args()

    contract = load_qmini_contract()
    contract_failures = validate_qmini_contract(contract)
    source_manifest = read_json(EXP_ROOT / "manifests" / "qmini_source.json")
    baseline_manifest = read_json(EXP_ROOT / "manifests" / "baseline_policy_contract.json") or {}
    split_manifest = read_json(EXP_ROOT / "manifests" / "data_splits.json") or {}
    test_manifest = read_json(EXP_ROOT / "manifests" / "test_generation_contract.json") or {}

    source_hash_ok = (
        source_manifest is not None
        and source_manifest.get("commit") == "f6f3fef723f8bb434f9d2679dfb6053b0aca93a8"
        and source_manifest.get("urdf_sha256") == "4d1454510bf403fb0740a7a682fc1883ada0ecbdced844530dd98d484a618215"
        and sha256(OFFICIAL_URDF_PATH) == source_manifest.get("urdf_sha256")
    )
    protected_ok, protected_detail = protected_source_gate()
    snapshot_ok, snapshot_detail = snapshot_gate()
    fatigue_ok, fatigue_detail = fatigue_gate()
    action_ok, action_detail = action_logging_gate()
    schema_ok, schema_detail = schema_gate()
    isaac_detail = read_json(args.isaac_import_report) if args.isaac_import_report else None
    isaac_ok = (
        isaac_detail is not None
        and isaac_detail.get("status") == "PASS"
        and isaac_detail.get("joint_count") == 10
        and isaac_detail.get("joint_order_match") is True
        and isaac_detail.get("contact_sensor_match") is True
        and isaac_detail.get("mass_inertia_match") is True
    )

    calibration = read_json(RESULTS_ROOT / "hidden_physics_calibration.json")
    memory = read_json(RESULTS_ROOT / "memory_necessity.json")
    crossed = read_json(RESULTS_ROOT / "crossed_interventions.json")
    baseline_result = read_json(RESULTS_ROOT / "baseline_formal.json")
    baseline_ok = baseline_result is not None and baseline_result.get("status") == "PASS"
    hidden_ok = calibration is not None and calibration.get("status") == "PASS"
    memory_ok = memory is not None and memory.get("status") == "PASS"
    crossed_ok = (
        crossed is not None
        and crossed.get("status") == "PASS"
        and crossed.get("separated_action_pairs", 0) >= 3
    )
    worst_fall_ok = (
        calibration is not None
        and calibration.get("status") == "PASS"
        and float(calibration.get("worst_condition_fall_rate", 1.0)) <= 0.10
    )
    split_ok = (
        split_manifest.get("split_method") == "stable_sha256_group_hash"
        and sum(float(value) for value in split_manifest.get("ratios", {}).values()) == 1.0
        and "teacher_id" not in split_manifest
    )
    test_contract_ok = (
        test_manifest.get("observation_dim") == OBSERVATION_DIM
        and test_manifest.get("action_dim") == ACTION_DIM
        and test_manifest.get("required_separate_fields") == ["action_proposed", "action_applied"]
    )
    failure_taxonomy_ok = tuple(FAILURE_TAXONOMY) == (
        "EXP015_QMINI_STAGE1_PASS",
        "NO_GO_QMINI_BASELINE",
        "NO_GO_MEMORY_NECESSITY",
        "NO_GO_HIDDEN_FACTOR_RELEVANCE",
        "NO_GO_SNAPSHOT_REPRODUCIBILITY",
        "NO_GO_ACTION_EFFECT_SEPARATION",
        "INVALID_QMINI_PHYSICS_CONTRACT",
        "INVALID_SOURCE_MUTATION",
        "INVALID_DATA_CONTRACT",
    )
    gates = {
        "qmini_source_hash": source_hash_ok,
        "qmini_physics_contract": not contract_failures,
        "qmini_joint_contract": not contract_failures,
        "isaaclab_asset_import": isaac_ok if args.isaac_import_report else None,
        "baseline_walk_formal": baseline_ok,
        "source_protected_write": protected_ok,
        "snapshot_deterministic_replay": snapshot_ok,
        "hidden_factor_relevance": hidden_ok,
        "memory_necessity": memory_ok,
        "crossed_action_separation": crossed_ok,
        "worst_hidden_fall": worst_fall_ok,
        "fatigue_ledger": fatigue_ok,
        "action_proposed_applied_logging": action_ok,
        "canonical_schema": schema_ok,
        "train_dev_test_contract": split_ok,
        "failure_taxonomy": failure_taxonomy_ok,
        "data_contract": test_contract_ok and split_ok,
    }
    classification = classify_stage1_gate(gates)
    report = {
        "experiment": "exp_015_qmini_population_bodily_world_model",
        "scope": "STAGE_0_STAGE_1_ONLY",
        "training_executed": False,
        "deployment_executed": False,
        "gates": classification.to_dict(),
        "details": {
            "qmini_contract_failures": contract_failures,
            "source_hash": {
                "status": "PASS" if source_hash_ok else "FAIL",
                "vendored_urdf_sha256": sha256(OFFICIAL_URDF_PATH),
                "manifest": source_manifest,
            },
            "isaaclab_asset_import": isaac_detail,
            "protected_source": protected_detail,
            "snapshot": snapshot_detail,
            "fatigue": fatigue_detail,
            "action_logging": action_detail,
            "schema": schema_detail,
            "baseline_manifest": baseline_manifest,
            "baseline_result": baseline_result,
            "calibration": calibration,
            "memory": memory,
            "crossed_interventions": crossed,
        },
        "tests": None if args.skip_tests else run_pytest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (RESULTS_ROOT / "protected_source_reaudit.json").write_text(
        json.dumps(protected_detail, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "classification": classification.final_classification,
        "output": str(args.output),
        "pytest": report["tests"],
    }, indent=2))
    return 0 if classification.final_classification == "EXP015_QMINI_STAGE1_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
