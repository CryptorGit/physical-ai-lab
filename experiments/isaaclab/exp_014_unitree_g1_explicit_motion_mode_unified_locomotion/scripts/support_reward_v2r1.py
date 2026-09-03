"""Pure support-transfer reward correction for exp_014 Phase 2-D20.

This module intentionally has no Isaac Lab or policy dependencies.  It versions
the two D20 implementation corrections without changing the D18 artifacts.
"""

from __future__ import annotations

import math


TARGET_PEAK = 0.7
CANONICAL_CONTACT_FORCE_THRESHOLD_N = 5.0


def minimum_jerk(u: float) -> float:
    """Quintic minimum-jerk interpolation on [0, 1]."""
    u = min(1.0, max(0.0, float(u)))
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def corrected_target(t_s: float, target_peak: float = TARGET_PEAK) -> float:
    """V2R1 target: ramp to peak, then retain it until the envelope is zero."""
    t_s = float(t_s)
    if t_s < 0.35:
        return minimum_jerk(t_s / 0.35) * target_peak
    if t_s < 0.75:
        return target_peak
    return 0.0


def corrected_weight_envelope(t_s: float) -> float:
    """V2R1 support-family weight envelope."""
    t_s = float(t_s)
    if t_s < 0.50:
        return 1.0
    if t_s < 0.75:
        return 1.0 - minimum_jerk((t_s - 0.50) / 0.25)
    return 0.0


def old_target(t_s: float, target_peak: float = TARGET_PEAK) -> float:
    """Exact D18 target behavior retained solely for source-diff diagnostics."""
    target = minimum_jerk(float(t_s) / 0.35) * target_peak
    return target if float(t_s) <= 0.50 else 0.0


def support_valid(
    left_force_norm_n: float,
    right_force_norm_n: float,
    threshold_n: float = CANONICAL_CONTACT_FORCE_THRESHOLD_N,
) -> bool:
    """Canonical G1 foot-contact validity: either foot force norm exceeds 5 N."""
    return max(float(left_force_norm_n), float(right_force_norm_n)) > threshold_n


def unsigned_load_imbalance(left_vertical_n: float, right_vertical_n: float) -> float:
    left = max(float(left_vertical_n), 0.0)
    right = max(float(right_vertical_n), 0.0)
    return abs(left - right) / (left + right + 1.0e-6)


def corrected_load_reward(
    *,
    left_vertical_n: float,
    right_vertical_n: float,
    left_force_norm_n: float,
    right_force_norm_n: float,
    t_s: float,
    sigma_load: float,
    target_peak: float = TARGET_PEAK,
) -> float:
    """Masked load-transfer term; zero support is neutral, never rewarded."""
    if not support_valid(left_force_norm_n, right_force_norm_n):
        return 0.0
    target = corrected_target(t_s, target_peak)
    imbalance = unsigned_load_imbalance(left_vertical_n, right_vertical_n)
    return (
        math.exp(-((imbalance - target) / float(sigma_load)) ** 2)
        * corrected_weight_envelope(t_s)
    )
