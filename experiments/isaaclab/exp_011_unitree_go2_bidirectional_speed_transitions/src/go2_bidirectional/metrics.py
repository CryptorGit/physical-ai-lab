"""Numerically small, dependency-free metric helpers."""

from __future__ import annotations

import math


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, q: float) -> float:
    values = sorted(float(v) for v in values)
    if not values:
        return 0.0
    position = (len(values) - 1) * q / 100.0
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - position) + values[hi] * (position - lo)


def max_true_dwell(values, dt: float) -> float:
    dwell = best = 0
    for value in values:
        dwell = dwell + 1 if value else 0
        best = max(best, dwell)
    return best * dt


def wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))

