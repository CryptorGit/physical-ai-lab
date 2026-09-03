"""Train diagnostic probes on the frozen Stage 0 dataset.

These models are analysis artifacts only.  They are never loaded by a
production controller.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve()
EXP = HERE.parents[1]
REPO = EXP.parents[2]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from phase_transition_analysis.dataset import feature_matrix, history_matrix, load_dataset
from phase_transition_analysis.evaluation import binary_metrics, regression_metrics
from phase_transition_analysis.probes import Standardizer, predict, ridge_predict, train_linear, train_mlp
from phase_transition_analysis.temporal_probe import predict_gru, train_gru, train_history_mlp


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics_by_slice(frame: pd.DataFrame, probabilities: np.ndarray, label: str) -> dict:
    values = {"overall": binary_metrics(frame[label], probabilities)}
    for speed in (2.6, 2.8):
        mask = np.isclose(frame["source_speed_mps"], speed)
        values[f"source_{speed:.1f}"] = binary_metrics(frame.loc[mask, label], probabilities[mask])
    for phase, code in (("flight", 0), ("left_support", 1), ("right_support", 2), ("double_support", 3)):
        mask = frame["support_phase"].eq(code).to_numpy()
        if mask.any():
            values[f"phase_{phase}"] = binary_metrics(frame.loc[mask, label], probabilities[mask])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP / "configs/stage0_observability_probe.yaml")
    parser.add_argument("--results", type=Path, default=REPO / "results/exp_008_phase_aware_locomotion_transitions/stage0_observability_and_controllability")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    full_frame = load_dataset(args.results / "episodes.parquet")
    # Prediction is only meaningful from first WALK-compatible contact through
    # the first break.  Post-break rows are retained in parquet for auditing.
    frame = full_frame[
        (full_frame["relative_to_first_contact"] >= 0)
        & (full_frame["relative_to_break"] <= 0)
        & (full_frame["steps_until_break"] >= 0)
    ].copy()
    frame.sort_values(["episode_id", "transition_step"], inplace=True)
    probe_cfg = {
        "epochs": int(config["probes"]["epochs"]),
        "batch_size": int(config["probes"]["batch_size"]),
        "learning_rate": float(config["probes"]["learning_rate"]),
        "weight_decay": float(config["probes"]["weight_decay"]),
    }
    seed = int(config["seed"])
    results: dict[str, dict] = {}
    saved_models: dict[str, str] = {}
    model_dir = args.results / "diagnostic_probes"
    model_dir.mkdir(parents=True, exist_ok=True)

    for condition in ("A", "B", "C", "D", "E"):
        x, columns = feature_matrix(frame, condition)
        train_mask = frame["split"].eq("train").to_numpy()
        test_mask = frame["split"].eq("test").to_numpy()
        scaler = Standardizer().fit(x[train_mask])
        sx = scaler.transform(x)
        condition_result = {"input_dimensions": len(columns), "binary": {}}
        for horizon in (1, 3, 5):
            label = f"contact_break_within_{horizon}"
            y = frame[label].to_numpy(np.float32)
            linear = train_linear(sx[train_mask], y[train_mask], probe_cfg, seed=seed + horizon)
            linear_prob = predict(linear, sx[test_mask])
            condition_result["binary"][f"linear_h{horizon}"] = metrics_by_slice(
                frame.loc[test_mask].reset_index(drop=True), linear_prob, label
            )
            if horizon == 3:
                mlp = train_mlp(
                    sx[train_mask],
                    y[train_mask],
                    probe_cfg,
                    hidden=tuple(config["probes"]["mlp_hidden"]),
                    seed=seed + 100 + ord(condition),
                )
                mlp_prob = predict(mlp, sx[test_mask])
                condition_result["binary"]["static_mlp_h3"] = metrics_by_slice(
                    frame.loc[test_mask].reset_index(drop=True), mlp_prob, label
                )
                model_path = model_dir / f"static_mlp_{condition}.pt"
                torch.save({"state_dict": mlp.cpu().state_dict(), "columns": columns}, model_path)
                saved_models[f"static_mlp_{condition}"] = str(model_path.relative_to(REPO))

        target = frame["steps_until_break"].clip(0, 32).to_numpy(np.float32)
        prediction = ridge_predict(sx[train_mask], target[train_mask], sx[test_mask], alpha=1.0)
        condition_result["regression"] = regression_metrics(target[test_mask], prediction)
        results[condition] = condition_result

    # History probes use timing-ablated condition B.  Every window stays within
    # a single episode by construction.
    history_results = {}
    for history in config["features"]["history_steps"]:
        hx, indices, columns = history_matrix(full_frame, "B", int(history))
        hframe = full_frame.loc[indices].copy()
        endpoint = (hframe["relative_to_first_contact"] >= 0) & (hframe["relative_to_break"] <= 0)
        hx = hx[endpoint.to_numpy()]
        hframe = hframe.loc[endpoint].copy()
        train_mask = hframe["split"].eq("train").to_numpy()
        test_mask = hframe["split"].eq("test").to_numpy()
        scaler = Standardizer().fit(hx[train_mask])
        shx = scaler.transform(hx)
        y = hframe["contact_break_within_3"].to_numpy(np.float32)
        history_model = train_history_mlp(shx[train_mask], y[train_mask], probe_cfg, seed=seed + int(history))
        probability = predict(history_model, shx[test_mask])
        item = {
            "history_mlp": metrics_by_slice(hframe.loc[test_mask].reset_index(drop=True), probability, "contact_break_within_3")
        }
        if int(history) == 16:
            input_dim = len(columns) // int(history)
            gru = train_gru(
                shx[train_mask],
                y[train_mask],
                probe_cfg,
                input_dim=input_dim,
                hidden_dim=int(config["probes"]["gru_hidden"]),
                seed=seed + 900,
            )
            gru_probability = predict_gru(gru, shx[test_mask], input_dim)
            item["gru"] = metrics_by_slice(hframe.loc[test_mask].reset_index(drop=True), gru_probability, "contact_break_within_3")
            model_path = model_dir / "gru_history16_B.pt"
            torch.save({"state_dict": gru.cpu().state_dict(), "input_dim": input_dim}, model_path)
            saved_models["gru_history16_B"] = str(model_path.relative_to(REPO))
        history_results[str(history)] = item
    results["history"] = history_results

    # Age matching: score examples only against examples of identical streak
    # age, using the timing-ablated static MLP probabilities.
    x, _ = feature_matrix(frame, "B")
    train_mask = frame["split"].eq("train").to_numpy()
    test_mask = frame["split"].eq("test").to_numpy()
    scaler = Standardizer().fit(x[train_mask])
    sx = scaler.transform(x)
    y = frame["contact_break_within_3"].to_numpy(np.float32)
    age_model = train_mlp(sx[train_mask], y[train_mask], probe_cfg, seed=seed + 777)
    probabilities = predict(age_model, sx[test_mask])
    test = frame.loc[test_mask].reset_index(drop=True)
    age_results = {}
    for age in sorted(test["walk_valid_streak_age"].unique()):
        mask = test["walk_valid_streak_age"].eq(age).to_numpy()
        if mask.sum() >= 10:
            age_results[str(int(age))] = binary_metrics(test.loc[mask, "contact_break_within_3"], probabilities[mask])

    dump(args.results / "probe_results.json", {"models": results, "saved_models": saved_models})
    dump(args.results / "per_feature_condition_results.json", {key: value for key, value in results.items() if key != "history"})
    dump(args.results / "age_matched_results.json", age_results)

    a = results["A"]["binary"]["static_mlp_h3"]["overall"]
    b = results["B"]["binary"]["static_mlp_h3"]["overall"]
    timing = {
        "timing_absolute_indices": config["features"]["timing_absolute_indices"],
        "full_152_auroc": a["auroc"],
        "timing_ablated_auroc": b["auroc"],
        "auroc_drop": a["auroc"] - b["auroc"],
        "timing_only_success_not_accepted_as_state_observability": True,
        "age_matched_evaluation_performed": True,
    }
    dump(args.results / "timing_leakage_audit.json", timing)

    thresholds = config["classification"]
    b_reg = results["B"]["regression"]["mae_steps"]
    static_pass = (
        b["auroc"] >= thresholds["static_auroc_min"]
        and b["auprc"] - b["prevalence"] >= thresholds["static_auprc_lift_min"]
        and b_reg <= thresholds["regression_mae_steps_max"]
    )
    best_history = max(
        (
            item.get("gru", item["history_mlp"])["overall"]["auroc"],
            int(history),
            "gru" if "gru" in item else "history_mlp",
        )
        for history, item in history_results.items()
    )
    history_improvement = best_history[0] - b["auroc"]
    phase = results["E"]["binary"]["static_mlp_h3"]["overall"]
    phase_reg = results["E"]["regression"]["mae_steps"]
    phase_pass = (
        phase["auroc"] >= thresholds["static_auroc_min"]
        and phase["auprc"] - phase["prevalence"] >= thresholds["static_auprc_lift_min"]
        and phase_reg <= thresholds["regression_mae_steps_max"]
    )
    if static_pass:
        classification = "STATIC_152D_OBSERVABLE"
    elif history_improvement >= thresholds["history_auroc_improvement_min"]:
        classification = "HISTORY_REQUIRED"
    elif phase_pass:
        classification = "EXPLICIT_PHASE_FEATURES_REQUIRED"
    else:
        classification = "BREAK_NOT_PREDICTABLE"
    dump(
        args.results / "observability_classification.json",
        {
            "classification": classification,
            "static_timing_ablated": b,
            "static_regression_mae_steps": b_reg,
            "best_history": {"auroc": best_history[0], "steps": best_history[1], "model": best_history[2]},
            "history_auroc_improvement": history_improvement,
            "explicit_phase": phase,
            "explicit_phase_regression_mae_steps": phase_reg,
            "age_matched_results_path": "age_matched_results.json",
            "thresholds": thresholds,
        },
    )
    print(json.dumps({"classification": classification, "timing_ablated": b, "best_history": best_history}, indent=2))


if __name__ == "__main__":
    main()
