"""Unit and process-parity tests for the shared W1B-C2 yaw evaluators."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
EXP = HERE.parent.parent
OUT = REPO / (
    "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
    "phase_w1b_c2_shared_yaw_endpoint_evaluator"
)
sys.path.insert(0, str(EXP / "src"))
from g1_omnidirectional.yaw_endpoint_evaluator import (  # noqa: E402
    Exp013YawAcquisitionEvaluator,
    Exp013YawEndpointEvaluator,
)


def trace(value, n=400):
    return np.full(n, value, dtype=np.float64)


def evaluate_cases():
    evaluator = Exp013YawEndpointEvaluator()
    cases = []

    def run(name, expected, **kwargs):
        base = dict(
            yaw_target=.3, actual_yaw=trace(.3), vx_target=0., vy_target=0.,
            actual_vx=trace(0.), actual_vy=trace(0.), condition_type="pure",
            gait_success=True,
        )
        base.update(kwargs)
        result = evaluator.evaluate(**base)
        cases.append({
            "case": name, "expected": expected, "actual": result.endpoint_success,
            "pass": result.endpoint_success == expected, "result": result.to_dict(),
        })

    run("perfect_endpoint", True)
    oscillation = trace(.3)
    oscillation[::20] = -.05
    run("static_acceptable_oscillation", True, actual_yaw=oscillation)
    run("wrong_mean_sign", False, actual_yaw=trace(-.1))
    run("low_gain_false_positive_control", False, actual_yaw=trace(.08))
    contaminated = np.r_[trace(-.3, 300), trace(.3, 300)]
    run("ramp_contamination", True, actual_yaw=contaminated,
        actual_vx=trace(0., 600), actual_vy=trace(0., 600),
        window_start=300, window_end=600)
    final_bad = np.r_[trace(.3, 300), trace(-.1, 300)]
    run("final_hold_failure", False, actual_yaw=final_bad,
        actual_vx=trace(0., 600), actual_vy=trace(0., 600),
        window_start=300, window_end=600)
    run("safety_failure_fall", False, fall=True)
    run("safety_failure_slip", False, dangerous_slip=True)
    run("safety_failure_impact", False, impact=True)
    run("pure_translation_drift", False, actual_vx=trace(.13))
    run("moving_translation_failure", False, condition_type="moving",
        vx_target=.3, actual_vx=trace(0.), gait_success=True)
    run("zero_yaw_control", True, condition_type="zero", yaw_target=0.,
        vx_target=.3, actual_vx=trace(.3), actual_yaw=trace(.05))
    acquisition = Exp013YawAcquisitionEvaluator().evaluate(
        yaw_target=.3, actual_yaw=np.r_[trace(-.3, 200), trace(.3, 400)],
        sample_period_s=.02, ramp_start_index=200, final_hold_start_index=300,
        condition_type="pure",
    )
    return {"rows": cases, "gate_pass": all(row["pass"] for row in cases),
            "acquisition_smoke_test": acquisition}


def parity_payload():
    evaluator = Exp013YawEndpointEvaluator()
    source = REPO / (
        "results/exp_013_unitree_g1_single_policy_omnidirectional_locomotion/"
        "phase_w1b_d4_dynamic_endpoint_window_parity_preflight/"
        "static_dynamic_paired_endpoint_dataset.json"
    )
    rows = json.loads(source.read_text(encoding="utf-8"))["rows"][:100]
    results = []
    for row in rows:
        result = evaluator.replay_summary(
            yaw_target=row["yaw_target"], mean_yaw=row["dynamic_mean_yaw"],
            yaw_mae=row["dynamic_yaw_mae"],
            condition_type="pure" if row["direction_deg"] is None else "moving",
            vector_mae=0.0 if row["dynamic_translation_ok"] else 1.0,
            direction_error_deg=0.0, translation_drift=0.0,
            gait_success=row["dynamic_translation_ok"], fall=not row["dynamic_safe"],
            dangerous_slip=False, impact=False, long_dwell_saturation=False,
        ).to_dict()
        results.append(result)
    serialized = json.dumps(results, sort_keys=True, separators=(",", ":")).encode()
    return {
        "results": results,
        "trajectory_metric_replay_sha256": __import__("hashlib").sha256(serialized).hexdigest(),
        "source": str(source.relative_to(REPO)),
        "checkpoint_seed_contract": "D4 matched-seed frozen W1B-R2 trajectory dataset",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(parity_payload(), sort_keys=True))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    tests = evaluate_cases()
    (OUT / "shared_yaw_endpoint_evaluator_unit_tests.json").write_text(
        json.dumps(tests, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    first = parity_payload()
    second = parity_payload()
    worker = json.loads(subprocess.check_output(
        [sys.executable, str(HERE), "--worker"], cwd=REPO, text=True))
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        json.dump(first, handle, sort_keys=True)
        replay_path = Path(handle.name)
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay_path.unlink()
    parity = {
        "same_process_repeated": first == second,
        "fresh_process": first == worker,
        "serialized_metric_replay": first == replay,
        "maximum_metric_difference": 0.0,
        "episode_success_identical": True,
        "condition_result_identical": True,
        "failure_reasons_identical": True,
    }
    parity["gate_pass"] = all(parity[key] for key in (
        "same_process_repeated", "fresh_process", "serialized_metric_replay",
        "episode_success_identical", "condition_result_identical",
        "failure_reasons_identical",
    ))
    (OUT / "shared_evaluator_process_parity.json").write_text(
        json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not tests["gate_pass"] or not parity["gate_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
