"""Dependency-light diagnostic metrics."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def binary_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    if positives and negatives:
        ranks = rankdata(p)
        auroc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        order = np.argsort(-p)
        sorted_y = y[order]
        precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
        auprc = float((precision * sorted_y).sum() / positives)
    else:
        auroc = 0.5
        auprc = float(positives / max(len(y), 1))
    predicted = p >= 0.5
    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())
    tpr, tnr = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    ece = 0.0
    for lower in np.linspace(0, 0.9, 10):
        mask = (p >= lower) & (p < lower + 0.1)
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "prevalence": float(y.mean()) if len(y) else 0.0,
        "balanced_accuracy": float((tpr + tnr) / 2),
        "f1": float(f1),
        "calibration_error": float(ece),
        "count": int(len(y)),
        "positive_count": positives,
    }


def regression_metrics(target, prediction):
    error = np.abs(np.asarray(target) - np.asarray(prediction))
    return {
        "mae_steps": float(error.mean()),
        "p95_absolute_error_steps": float(np.quantile(error, 0.95)),
        "within_one_step_accuracy": float((error <= 1).mean()),
        "count": int(len(error)),
    }
