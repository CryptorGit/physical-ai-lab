"""Common diagnostic metric helpers."""

from __future__ import annotations

import numpy as np


def rate(values):
    values = np.asarray(values, dtype=bool)
    return float(values.mean()) if len(values) else 0.0


def percentile(values, q):
    values = np.asarray(values, dtype=float)
    return float(np.quantile(values, q)) if len(values) else 0.0
