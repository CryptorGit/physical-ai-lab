"""Recompute the specified ridge time-to-break probe without retraining binary probes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch  # Load CUDA/OpenMP DLLs before pandas/NumPy on Windows.
import numpy as np
import yaml

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
REPO = EXP.parents[2]
sys.path.insert(0, str(EXP / "src"))

from phase_transition_analysis.dataset import feature_matrix, load_dataset
from phase_transition_analysis.evaluation import regression_metrics
from phase_transition_analysis.probes import Standardizer, ridge_predict

OUT = REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability"


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    config = yaml.safe_load((EXP / "configs/stage0_observability_probe.yaml").read_text(encoding="utf-8"))
    full = load_dataset(OUT / "episodes.parquet")
    frame = full[
        (full["relative_to_first_contact"] >= 0)
        & (full["relative_to_break"] <= 0)
        & (full["steps_until_break"] >= 0)
    ].copy()
    results = json.loads((OUT / "probe_results.json").read_text(encoding="utf-8"))
    target = frame["steps_until_break"].clip(0, 32).to_numpy(np.float32)
    train_mask = frame["split"].eq("train").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    for condition in ("A", "B", "C", "D", "E"):
        x, _ = feature_matrix(frame, condition)
        scaler = Standardizer().fit(x[train_mask])
        sx = scaler.transform(x)
        prediction = ridge_predict(sx[train_mask], target[train_mask], sx[test_mask], alpha=1.0)
        results["models"][condition]["regression"] = regression_metrics(target[test_mask], prediction)
    dump("probe_results.json", results)
    dump("per_feature_condition_results.json", {key: value for key, value in results["models"].items() if key != "history"})

    models = results["models"]
    b = models["B"]["binary"]["static_mlp_h3"]["overall"]
    b_reg = models["B"]["regression"]["mae_steps"]
    threshold = config["classification"]
    static_pass = (
        b["auroc"] >= threshold["static_auroc_min"]
        and b["auprc"] - b["prevalence"] >= threshold["static_auprc_lift_min"]
        and b_reg <= threshold["regression_mae_steps_max"]
    )
    history_results = models["history"]
    best_history = max(
        (
            item.get("gru", item["history_mlp"])["overall"]["auroc"],
            int(history),
            "gru" if "gru" in item else "history_mlp",
        )
        for history, item in history_results.items()
    )
    history_improvement = best_history[0] - b["auroc"]
    phase = models["E"]["binary"]["static_mlp_h3"]["overall"]
    phase_reg = models["E"]["regression"]["mae_steps"]
    phase_pass = (
        phase["auroc"] >= threshold["static_auroc_min"]
        and phase["auprc"] - phase["prevalence"] >= threshold["static_auprc_lift_min"]
        and phase_reg <= threshold["regression_mae_steps_max"]
    )
    if static_pass:
        classification = "STATIC_152D_OBSERVABLE"
    elif history_improvement >= threshold["history_auroc_improvement_min"]:
        classification = "HISTORY_REQUIRED"
    elif phase_pass:
        classification = "EXPLICIT_PHASE_FEATURES_REQUIRED"
    else:
        classification = "BREAK_NOT_PREDICTABLE"
    dump(
        "observability_classification.json",
        {
            "classification": classification,
            "static_timing_ablated": b,
            "static_regression_mae_steps": b_reg,
            "best_history": {"auroc": best_history[0], "steps": best_history[1], "model": best_history[2]},
            "history_auroc_improvement": history_improvement,
            "explicit_phase": phase,
            "explicit_phase_regression_mae_steps": phase_reg,
            "age_matched_results_path": "age_matched_results.json",
            "thresholds": threshold,
            "regression_probe": "closed_form_ridge_alpha_1.0",
        },
    )
    print(json.dumps({"classification": classification, "B_mae": b_reg, "E_mae": phase_reg}, indent=2))


if __name__ == "__main__":
    main()
