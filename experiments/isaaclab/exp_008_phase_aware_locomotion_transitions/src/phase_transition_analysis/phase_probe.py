"""Explicit phase upper-bound helpers."""

from __future__ import annotations

from .dataset import feature_matrix


def explicit_phase_matrix(frame):
    return feature_matrix(frame, "E_explicit_phase_upper_bound")
