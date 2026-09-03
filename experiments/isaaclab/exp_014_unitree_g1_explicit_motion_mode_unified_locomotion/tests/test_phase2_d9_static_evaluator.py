"""Regression tests for the versioned D9 static evaluator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
SCRIPT = HERE.parent.parent / "scripts/phase2_d9_static_evaluator.py"
spec = importlib.util.spec_from_file_location("d9evaluator", SCRIPT)
evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
RAW = REPO / "results/exp_014_unitree_g1_explicit_motion_mode_unified_locomotion/phase_2_d7_r4_stop_oracle_distillation/raw"
INTEGRITY = {name: True for name in evaluator.REQUIRED_INTEGRITY}


def good_metrics():
    return {"mse": 1e-4, "cosine": .999, "contexts": {str(i): {"mse": 1e-4} for i in (3, 4, 5)}, "worst_condition_mse": 1e-4, "phase_classification": 0.0}


def test_independent_diagnostic_classifier_is_not_used():
    low = evaluator.evaluate_checkpoint(good_metrics(), INTEGRITY)
    high_metrics = good_metrics(); high_metrics["phase_classification"] = 1.0
    high = evaluator.evaluate_checkpoint(high_metrics, INTEGRITY)
    assert low == high and low["eligible"]


def test_phase_below_99_does_not_block_action_passing_checkpoint():
    metrics = good_metrics(); metrics["phase_classification"] = .5
    assert evaluator.evaluate_checkpoint(metrics, INTEGRITY)["eligible"]


def test_boundary_failure_is_ineligible_regardless_of_phase():
    metrics = good_metrics(); metrics["contexts"]["3"]["mse"] = .00101; metrics["phase_classification"] = 1.0
    assert not evaluator.evaluate_checkpoint(metrics, INTEGRITY)["eligible"]


def test_missing_checkpoint_metric_fails_closed():
    metrics = good_metrics(); del metrics["cosine"]
    result = evaluator.evaluate_checkpoint(metrics, INTEGRITY)
    assert not result["eligible"] and result["fail_closed"]


def test_stage_global_diagnostic_is_not_replicated_into_result():
    result = evaluator.evaluate_checkpoint(good_metrics(), INTEGRITY)
    assert "phase_classification" not in result["action_metrics"]
    assert result["diagnostic_metrics_used_for_eligibility"] == []


def test_d7_saved_fixture_s0_fails_and_s1_step_30000_passes():
    s0 = json.loads((RAW / "bc_results.json").read_text(encoding="utf-8"))["timeline"][-1]
    s1 = json.loads((RAW / "s1_bc_results.json").read_text(encoding="utf-8"))["timeline"][-1]
    assert not evaluator.evaluate_checkpoint(s0, INTEGRITY)["eligible"]
    assert evaluator.evaluate_checkpoint(s1, INTEGRITY)["eligible"]
