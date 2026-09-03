"""Versioned RESET_TO_STAND and STAND_HOLD capability evaluators.

These evaluators are policy-agnostic and operate on captured physical
trajectories.  They intentionally leave the legacy two-second average metric
unchanged and separate from the V2 capability classifications.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


CONTROL_DT = 0.02
SPEED_LIMIT = 0.08
YAW_LIMIT = 0.08
ACQUISITION_STEPS = 50
ACQUISITION_HOLD_STEPS = 50
STAND_HOLD_STEPS = 100


@dataclass(frozen=True)
class ResetToStandResult:
    passed: bool
    acquisition_step: int | None
    acquisition_time_s: float | None
    hold_completion_step: int | None
    continuous_hold_duration_s: float
    re_entry_count: int
    re_exit_count: int
    safety_pass: bool
    failure_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StandHoldResult:
    eligible: bool
    passed: bool | None
    start_step: int | None
    end_step: int | None
    speed_mean: float | None
    speed_p95: float | None
    absolute_yaw_mean: float | None
    absolute_yaw_p95: float | None
    safety_pass: bool | None
    failure_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def _bool_array(value, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.ndim == 0:
        array = np.full(length, bool(array), dtype=bool)
    if array.shape != (length,):
        raise ValueError(f"expected ({length},), got {array.shape}")
    return array


class Exp014ResetToStandEvaluatorV2:
    """Acquire both per-step thresholds by 1s and hold them for 50 steps."""

    def evaluate(self, speed, absolute_yaw, *, fall=False, dangerous_slip=False,
                 impact_failure=False, long_dwell_saturation=False) -> ResetToStandResult:
        speed = np.asarray(speed, dtype=float)
        yaw = np.asarray(absolute_yaw, dtype=float)
        if speed.ndim != 1 or yaw.shape != speed.shape or len(speed) < 100:
            raise ValueError("RESET_TO_STAND requires aligned trajectories with at least 100 steps")
        finite = np.isfinite(speed) & np.isfinite(yaw)
        within = finite & (speed <= SPEED_LIMIT) & (yaw <= YAW_LIMIT)
        entries = int(within[0]) + int(np.sum((~within[:-1]) & within[1:]))
        exits = int(np.sum(within[:-1] & (~within[1:])))
        acquisition = None
        for step in range(ACQUISITION_STEPS):
            if within[step:step + ACQUISITION_HOLD_STEPS].all():
                acquisition = step
                break
        safety = ~(
            _bool_array(fall, len(speed)) | _bool_array(dangerous_slip, len(speed))
            | _bool_array(impact_failure, len(speed)) | _bool_array(long_dwell_saturation, len(speed))
        )
        completion = None if acquisition is None else acquisition + ACQUISITION_HOLD_STEPS - 1
        safety_pass = completion is not None and bool(safety[:completion + 1].all())
        passed = acquisition is not None and safety_pass
        if not finite.all():
            reason = "NON_FINITE"
        elif acquisition is None:
            reason = "ACQUISITION_OR_CONTINUOUS_HOLD_FAILURE"
        elif not safety_pass:
            reason = "SAFETY_FAILURE"
        else:
            reason = None
        return ResetToStandResult(
            passed=passed,
            acquisition_step=acquisition,
            acquisition_time_s=None if acquisition is None else (acquisition + 1) * CONTROL_DT,
            hold_completion_step=completion,
            continuous_hold_duration_s=0.0 if acquisition is None else ACQUISITION_HOLD_STEPS * CONTROL_DT,
            re_entry_count=entries,
            re_exit_count=exits,
            safety_pass=safety_pass,
            failure_reason=reason,
        )


class Exp014StandHoldEvaluatorV2:
    """Evaluate 100 steps after the policy completes RESET_TO_STAND hold."""

    def evaluate(self, speed, absolute_yaw, reset_result: ResetToStandResult, *,
                 fall=False, dangerous_slip=False, impact_failure=False,
                 long_dwell_saturation=False) -> StandHoldResult:
        speed = np.asarray(speed, dtype=float)
        yaw = np.asarray(absolute_yaw, dtype=float)
        if speed.ndim != 1 or yaw.shape != speed.shape:
            raise ValueError("STAND_HOLD requires aligned one-dimensional trajectories")
        if not reset_result.passed:
            return StandHoldResult(False, None, None, None, None, None, None, None, None, "RESET_TO_STAND_FAILED")
        start = int(reset_result.hold_completion_step) + 1
        end = start + STAND_HOLD_STEPS
        if len(speed) < end:
            raise ValueError(f"STAND_HOLD requires {end} captured steps, got {len(speed)}")
        s, y = speed[start:end], yaw[start:end]
        safety = ~(
            _bool_array(fall, len(speed))[start:end]
            | _bool_array(dangerous_slip, len(speed))[start:end]
            | _bool_array(impact_failure, len(speed))[start:end]
            | _bool_array(long_dwell_saturation, len(speed))[start:end]
        )
        values = {
            "speed_mean": float(np.mean(s)), "speed_p95": float(np.quantile(s, 0.95)),
            "absolute_yaw_mean": float(np.mean(y)), "absolute_yaw_p95": float(np.quantile(y, 0.95)),
        }
        finite = np.isfinite(s).all() and np.isfinite(y).all()
        thresholds = (
            values["speed_mean"] <= SPEED_LIMIT and values["absolute_yaw_mean"] <= YAW_LIMIT
            and values["speed_p95"] <= 0.12 and values["absolute_yaw_p95"] <= 0.12
        )
        safety_pass = bool(safety.all())
        passed = finite and thresholds and safety_pass
        reason = None if passed else "NON_FINITE" if not finite else "PHYSICAL_THRESHOLD_FAILURE" if not thresholds else "SAFETY_FAILURE"
        return StandHoldResult(True, passed, start, end - 1, **values, safety_pass=safety_pass, failure_reason=reason)


def legacy_whole_window_2s_average(speed, absolute_yaw, *, fall=False,
                                   dangerous_slip=False, impact_failure=False) -> dict:
    """Historical diagnostic only; this is not a V2 capability gate."""
    speed = np.asarray(speed, dtype=float)[:100]
    yaw = np.asarray(absolute_yaw, dtype=float)[:100]
    if len(speed) != 100 or yaw.shape != speed.shape:
        raise ValueError("legacy metric requires 100 aligned steps")
    safety = not (
        _bool_array(fall, 100).any() or _bool_array(dangerous_slip, 100).any()
        or _bool_array(impact_failure, 100).any()
    )
    speed_mean, yaw_mean = float(speed.mean()), float(yaw.mean())
    return {
        "name": "LEGACY_WHOLE_WINDOW_2S_AVERAGE",
        "diagnostic_only": True,
        "passed": bool(speed_mean <= SPEED_LIMIT and yaw_mean <= YAW_LIMIT and safety),
        "speed_mean": speed_mean,
        "absolute_yaw_mean": yaw_mean,
    }
