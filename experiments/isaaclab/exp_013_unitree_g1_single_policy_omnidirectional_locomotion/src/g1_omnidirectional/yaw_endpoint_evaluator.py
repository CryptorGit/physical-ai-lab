"""Shared physical yaw endpoint and acquisition evaluators for EXP013.

The endpoint evaluator deliberately knows nothing about how an actor command was
calibrated.  It evaluates the physical target and measured traces inside one
explicit endpoint window.  Static and dynamic callers therefore share exactly
the same success implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class YawEndpointResult:
    endpoint_mean_yaw: float
    endpoint_yaw_mae: float
    endpoint_yaw_sign: int
    target_sign_correct: bool
    translation_vector_mae: float
    translation_direction_error_deg: float
    translation_drift: float
    gait_success: bool
    fall: bool
    dangerous_slip: bool
    impact: bool
    long_dwell_saturation: bool
    endpoint_success: bool
    failure_reasons: tuple[str, ...]
    sample_count: int

    def to_dict(self) -> dict:
        result = asdict(self)
        result["failure_reasons"] = list(self.failure_reasons)
        return result


def _array(values: Iterable[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional trace")
    if not np.isfinite(result.astype(np.float64)).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return result


def _sign(value: float, epsilon: float = 1e-12) -> int:
    return 1 if value > epsilon else (-1 if value < -epsilon else 0)


class Exp013YawEndpointEvaluator:
    """Evaluate a static or dynamic endpoint with one physical contract."""

    PURE_YAW_MAE_LIMIT = 0.15
    MOVING_YAW_MAE_LIMIT = 0.20
    PURE_TRANSLATION_DRIFT_LIMIT = 0.12
    MOVING_VECTOR_MAE_LIMIT = 0.25
    MOVING_DIRECTION_ERROR_LIMIT_DEG = 25.0

    def evaluate(
        self,
        *,
        yaw_target: float,
        actual_yaw: Iterable[float] | np.ndarray,
        vx_target: float,
        vy_target: float,
        actual_vx: Iterable[float] | np.ndarray,
        actual_vy: Iterable[float] | np.ndarray,
        condition_type: str,
        gait_success: bool | Iterable[bool] | np.ndarray,
        fall: bool | Iterable[bool] | np.ndarray = False,
        dangerous_slip: bool | Iterable[bool] | np.ndarray = False,
        impact: bool | Iterable[bool] | np.ndarray = False,
        long_dwell_saturation: bool | Iterable[bool] | np.ndarray = False,
        window_start: int = 0,
        window_end: int | None = None,
    ) -> YawEndpointResult:
        yaw = _array(actual_yaw, "actual_yaw")
        vx = _array(actual_vx, "actual_vx")
        vy = _array(actual_vy, "actual_vy")
        if not (yaw.size == vx.size == vy.size):
            raise ValueError("yaw/vx/vy trace lengths differ")
        end = yaw.size if window_end is None else int(window_end)
        start = int(window_start)
        if start < 0 or end > yaw.size or start >= end:
            raise ValueError("invalid endpoint window")
        sl = slice(start, end)
        yaw_w, vx_w, vy_w = yaw[sl], vx[sl], vy[sl]
        target_sign = _sign(float(yaw_target))
        mean_yaw = float(np.mean(yaw_w, dtype=np.float64))
        yaw_mae = float(np.mean(np.abs(yaw_w - float(yaw_target)), dtype=np.float64))
        endpoint_sign = _sign(mean_yaw)
        sign_ok = (
            abs(mean_yaw) <= self.MOVING_YAW_MAE_LIMIT
            if target_sign == 0 else endpoint_sign == target_sign
        )
        vx_error = vx_w - float(vx_target)
        vy_error = vy_w - float(vy_target)
        vector_mae = float(np.mean(np.hypot(vx_error, vy_error), dtype=np.float64))
        drift = float(np.mean(np.hypot(vx_w, vy_w), dtype=np.float64))
        commanded_speed = float(np.hypot(vx_target, vy_target))
        if commanded_speed > 1e-12:
            actual_angle = np.arctan2(vy_w, vx_w)
            target_angle = np.arctan2(vy_target, vx_target)
            angle_error = np.abs(np.arctan2(
                np.sin(actual_angle - target_angle), np.cos(actual_angle - target_angle)
            ))
            direction_error = float(np.mean(angle_error, dtype=np.float64) * 180.0 / np.pi)
        else:
            direction_error = 0.0

        def any_in_window(value: bool | Iterable[bool] | np.ndarray) -> bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            trace = _array(value, "safety trace").astype(bool)
            if trace.size != yaw.size:
                raise ValueError("safety trace length differs")
            return bool(np.any(trace[sl]))

        def all_in_window(value: bool | Iterable[bool] | np.ndarray) -> bool:
            if isinstance(value, (bool, np.bool_)):
                return bool(value)
            trace = _array(value, "gait trace").astype(bool)
            if trace.size != yaw.size:
                raise ValueError("gait trace length differs")
            return bool(np.all(trace[sl]))

        gait_ok = all_in_window(gait_success)
        fell = any_in_window(fall)
        slipped = any_in_window(dangerous_slip)
        impacted = any_in_window(impact)
        saturated = any_in_window(long_dwell_saturation)
        reasons: list[str] = []
        if not sign_ok:
            reasons.append("yaw_mean_sign")
        yaw_limit = self.PURE_YAW_MAE_LIMIT if condition_type == "pure" else self.MOVING_YAW_MAE_LIMIT
        if condition_type != "zero" and yaw_mae > yaw_limit:
            reasons.append("yaw_mae")
        if condition_type == "pure":
            if drift > self.PURE_TRANSLATION_DRIFT_LIMIT:
                reasons.append("translation_drift")
        elif condition_type in ("moving", "zero"):
            vector_limit = .20 if condition_type == "zero" else self.MOVING_VECTOR_MAE_LIMIT
            direction_limit = 20.0 if condition_type == "zero" else self.MOVING_DIRECTION_ERROR_LIMIT_DEG
            if vector_mae > vector_limit:
                reasons.append("vector_mae")
            if direction_error > direction_limit:
                reasons.append("translation_direction")
            if not gait_ok:
                reasons.append("gait")
        else:
            raise ValueError("condition_type must be 'pure', 'moving', or 'zero'")
        if fell:
            reasons.append("fall")
        if slipped:
            reasons.append("dangerous_slip")
        if impacted:
            reasons.append("impact")
        if saturated:
            reasons.append("long_dwell_saturation")
        return YawEndpointResult(
            endpoint_mean_yaw=mean_yaw,
            endpoint_yaw_mae=yaw_mae,
            endpoint_yaw_sign=endpoint_sign,
            target_sign_correct=sign_ok,
            translation_vector_mae=vector_mae,
            translation_direction_error_deg=direction_error,
            translation_drift=drift,
            gait_success=gait_ok,
            fall=fell,
            dangerous_slip=slipped,
            impact=impacted,
            long_dwell_saturation=saturated,
            endpoint_success=not reasons,
            failure_reasons=tuple(reasons),
            sample_count=end - start,
        )

    def replay_summary(
        self,
        *,
        yaw_target: float,
        mean_yaw: float,
        yaw_mae: float,
        condition_type: str,
        vector_mae: float,
        direction_error_deg: float,
        translation_drift: float,
        gait_success: bool,
        fall: bool,
        dangerous_slip: bool,
        impact: bool,
        long_dwell_saturation: bool,
    ) -> YawEndpointResult:
        """Replay already-recorded endpoint metrics without inventing traces."""
        reasons: list[str] = []
        sign_ok = (
            abs(mean_yaw) <= self.MOVING_YAW_MAE_LIMIT
            if _sign(yaw_target) == 0 else _sign(mean_yaw) == _sign(yaw_target)
        )
        if not sign_ok:
            reasons.append("yaw_mean_sign")
        limit = self.PURE_YAW_MAE_LIMIT if condition_type == "pure" else self.MOVING_YAW_MAE_LIMIT
        if condition_type != "zero" and yaw_mae > limit:
            reasons.append("yaw_mae")
        if condition_type == "pure":
            if translation_drift > self.PURE_TRANSLATION_DRIFT_LIMIT:
                reasons.append("translation_drift")
        elif condition_type in ("moving", "zero"):
            vector_limit = .20 if condition_type == "zero" else self.MOVING_VECTOR_MAE_LIMIT
            direction_limit = 20.0 if condition_type == "zero" else self.MOVING_DIRECTION_ERROR_LIMIT_DEG
            if vector_mae > vector_limit:
                reasons.append("vector_mae")
            if direction_error_deg > direction_limit:
                reasons.append("translation_direction")
            if not gait_success:
                reasons.append("gait")
        else:
            raise ValueError("condition_type must be 'pure', 'moving', or 'zero'")
        for failed, reason in (
            (fall, "fall"), (dangerous_slip, "dangerous_slip"),
            (impact, "impact"), (long_dwell_saturation, "long_dwell_saturation"),
        ):
            if failed:
                reasons.append(reason)
        return YawEndpointResult(
            endpoint_mean_yaw=float(mean_yaw),
            endpoint_yaw_mae=float(yaw_mae),
            endpoint_yaw_sign=_sign(float(mean_yaw)),
            target_sign_correct=sign_ok,
            translation_vector_mae=float(vector_mae),
            translation_direction_error_deg=float(direction_error_deg),
            translation_drift=float(translation_drift),
            gait_success=bool(gait_success),
            fall=bool(fall),
            dangerous_slip=bool(dangerous_slip),
            impact=bool(impact),
            long_dwell_saturation=bool(long_dwell_saturation),
            endpoint_success=not reasons,
            failure_reasons=tuple(reasons),
            sample_count=0,
        )


class Exp013YawAcquisitionEvaluator:
    """Diagnostic-only transition acquisition timing."""

    def evaluate(
        self,
        *,
        yaw_target: float,
        actual_yaw: Iterable[float] | np.ndarray,
        sample_period_s: float,
        ramp_start_index: int,
        final_hold_start_index: int,
        condition_type: str,
        gait_cycle_id: Iterable[int] | np.ndarray | None = None,
    ) -> dict:
        yaw = _array(actual_yaw, "actual_yaw").astype(np.float64)
        target_sign = _sign(float(yaw_target))
        mae_limit = (
            Exp013YawEndpointEvaluator.PURE_YAW_MAE_LIMIT
            if condition_type == "pure"
            else Exp013YawEndpointEvaluator.MOVING_YAW_MAE_LIMIT
        )
        sign_ok = np.sign(yaw) == target_sign
        mae_ok = np.abs(yaw - float(yaw_target)) <= mae_limit

        def first(mask: np.ndarray) -> float | None:
            hits = np.flatnonzero(mask[ramp_start_index:])
            return None if not hits.size else float(hits[0] * sample_period_s)

        def sustained(mask: np.ndarray, duration: float) -> float | None:
            length = max(1, int(round(duration / sample_period_s)))
            kernel = np.ones(length, dtype=np.int32)
            valid = np.convolve(mask[ramp_start_index:].astype(np.int32), kernel, mode="valid")
            hits = np.flatnonzero(valid == length)
            return None if not hits.size else float(hits[0] * sample_period_s)

        first_cycle = None
        if gait_cycle_id is not None:
            cycles = _array(gait_cycle_id, "gait_cycle_id").astype(np.int64)
            if cycles.size != yaw.size:
                raise ValueError("gait cycle trace length differs")
            for cycle in np.unique(cycles[ramp_start_index:]):
                indices = np.flatnonzero((cycles == cycle) & (np.arange(yaw.size) >= ramp_start_index))
                if indices.size and _sign(float(np.mean(yaw[indices]))) == target_sign:
                    first_cycle = float((indices[0] - ramp_start_index) * sample_period_s)
                    break
        endpoint = yaw[final_hold_start_index:]
        overshoot = float(max(0.0, np.max(np.abs(endpoint)) - abs(float(yaw_target))))
        return {
            "first_instantaneous_correct_sign_s": first(sign_ok),
            "first_static_mae_pass_s": first(mae_ok),
            "first_0p10_sustained_endpoint_like_pass_s": sustained(sign_ok & mae_ok, 0.10),
            "first_0p20_sustained_endpoint_like_pass_s": sustained(sign_ok & mae_ok, 0.20),
            "first_complete_gait_cycle_mean_pass_s": first_cycle,
            "ramp_end_to_first_sign_s": (
                None if first(sign_ok) is None
                else first(sign_ok) - (final_hold_start_index - ramp_start_index) * sample_period_s
            ),
            "overshoot_rad_s": overshoot,
            "zero_crossing_count": int(np.count_nonzero(np.diff(np.signbit(yaw)))),
            "never_acquired_target_sign": not bool(np.any(sign_ok[final_hold_start_index:])),
            "formal_gate_member": False,
        }
