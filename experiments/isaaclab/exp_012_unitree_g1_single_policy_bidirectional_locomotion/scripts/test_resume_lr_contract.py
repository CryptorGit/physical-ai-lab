"""Offline unit tests for the EXP012 strict PPO resume LR contract."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
EXP = SCRIPT.parent.parent
ROOT = EXP.parents[2]
OUT = ROOT / "results/exp_012_unitree_g1_single_policy_bidirectional_locomotion/stage2b_runtime_lr_resume_fix"
sys.path.insert(0, str(EXP / "src"))

from g1_single_policy.strict_ppo_resume import (  # noqa: E402
    Exp012StrictPPOResumeContract,
    ResumeContractError,
)


class Optimizer:
    def __init__(self, lrs):
        self.param_groups = [{"lr": lr} for lr in lrs]


class Algorithm:
    def __init__(self, lrs, runtime=0.001):
        self.optimizer = Optimizer(lrs)
        self.learning_rate = runtime


def run_test(name, fn):
    try:
        detail = fn()
        return {"name": name, "status": "PASS", "detail": detail}
    except Exception as exc:  # tests report rather than hiding fail-closed outcomes
        return {"name": name, "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    contract = Exp012StrictPPOResumeContract()

    def test_a():
        alg = Algorithm([2.25e-5])
        state = contract.synchronize(alg, resume=True)
        assert alg.optimizer.param_groups[0]["lr"] == 2.25e-5
        assert alg.learning_rate == 2.25e-5
        return state.to_dict()

    def test_b():
        alg = Algorithm([2.25e-5])
        contract.synchronize(alg, resume=True)
        contract.assert_first_step_invariant(alg, expected=2.25e-5)
        assert alg.optimizer.param_groups[0]["lr"] != 0.001
        return "config default was not written back"

    def test_c():
        alg = Algorithm([0.001], runtime=0.001)
        state = contract.synchronize(alg, resume=False)
        assert alg.learning_rate == 0.001
        return state.to_dict()

    def test_d():
        try:
            contract.synchronize(Algorithm([2.25e-5, 3e-5]), resume=True)
        except ResumeContractError as exc:
            assert str(exc) == "PPO_RESTORED_OPTIMIZER_LR_AMBIGUOUS"
            return str(exc)
        raise AssertionError("mismatched groups did not fail")

    def test_e():
        try:
            contract.require_optimizer_state({})
        except ResumeContractError as exc:
            assert str(exc) == "PPO_RESUME_OPTIMIZER_STATE_MISSING"
            return str(exc)
        raise AssertionError("missing optimizer state did not fail")

    def test_f():
        alg = Algorithm([2.25e-5])
        state = contract.synchronize(alg, resume=True)
        restored = pickle.loads(pickle.dumps(state))
        assert restored.restored_lr == state.restored_lr
        assert restored.algorithm_lr == state.algorithm_lr
        return restored.to_dict()

    tests = [
        run_test("A_strict_resume_lr_synchronization", test_a),
        run_test("B_default_lr_overwrite_prevention", test_b),
        run_test("C_fresh_training_contract", test_c),
        run_test("D_multiple_param_group_mismatch", test_d),
        run_test("E_missing_optimizer_state", test_e),
        run_test("F_serialization", test_f),
    ]
    result = {"status": "PASS" if all(t["status"] == "PASS" for t in tests) else "FAIL", "tests": tests}
    (OUT / "resume_lr_unit_tests.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "resume_lr_contract.json").write_text(json.dumps({
        "name": "Exp012StrictPPOResumeContract",
        "source_of_truth": "optimizer.param_groups[*].lr",
        "expected_resume_lr": 2.25e-5,
        "config_default_lr": 0.001,
        "absolute_tolerance": 1e-12,
        "multiple_lr_behavior": "PPO_RESTORED_OPTIMIZER_LR_AMBIGUOUS",
        "missing_state_behavior": "PPO_RESUME_OPTIMIZER_STATE_MISSING",
        "fresh_training": "config learning rate remains authoritative",
        "resume_training": "checkpoint optimizer learning rate is synchronized to PPO runtime state",
        "production_integration": "train_stage2.py immediately after runner.load and before get_observations/rollout/update",
        "runner_lr_field": "NOT_PRESENT in current OnPolicyRunner",
        "adaptive_scheduler_state": "PPO.learning_rate",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
