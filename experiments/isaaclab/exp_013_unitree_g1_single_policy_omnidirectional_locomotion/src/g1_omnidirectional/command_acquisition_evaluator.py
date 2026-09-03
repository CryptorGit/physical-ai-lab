"""Translation+yaw acquisition diagnostics for EXP013 Phase W2."""
from __future__ import annotations

from collections.abc import Iterable
import numpy as np


class Exp013CommandAcquisitionEvaluator:
    """Measure acquisition without changing endpoint success semantics."""

    def evaluate(
        self,
        *,
        vx_target: float,
        vy_target: float,
        yaw_target: float,
        actual_vx: Iterable[float],
        actual_vy: Iterable[float],
        actual_yaw: Iterable[float],
        gait_success: Iterable[bool],
        safety_success: Iterable[bool],
        sample_period_s: float,
        ramp_end_index: int,
        gait_cycle_id: Iterable[int] | None = None,
    ) -> dict:
        vx = np.asarray(actual_vx, dtype=np.float64)
        vy = np.asarray(actual_vy, dtype=np.float64)
        yaw = np.asarray(actual_yaw, dtype=np.float64)
        gait = np.asarray(gait_success, dtype=bool)
        safe = np.asarray(safety_success, dtype=bool)
        if not (vx.shape == vy.shape == yaw.shape == gait.shape == safe.shape):
            raise ValueError("W2 acquisition traces must have identical shapes")
        speed_target = float(np.hypot(vx_target, vy_target))
        vector_mae = np.hypot(vx - vx_target, vy - vy_target)
        if speed_target <= 1e-8:
            translation_ok = np.hypot(vx, vy) <= 0.08
            direction_ok = np.ones_like(translation_ok)
        else:
            actual_angle = np.arctan2(vy, vx)
            target_angle = np.arctan2(vy_target, vx_target)
            direction_error = np.abs(np.rad2deg(
                np.arctan2(np.sin(actual_angle - target_angle),
                           np.cos(actual_angle - target_angle))
            ))
            translation_ok = vector_mae <= 0.25
            direction_ok = direction_error <= 25.0
        near_stop = speed_target <= 0.08
        yaw_limit = 0.15 if near_stop else 0.20
        if abs(yaw_target) <= 1e-8:
            yaw_ok = np.abs(yaw) <= yaw_limit
        else:
            yaw_ok = (np.sign(yaw) == np.sign(yaw_target)) & (
                np.abs(yaw - yaw_target) <= yaw_limit
            )
        endpoint_like = translation_ok & direction_ok & yaw_ok & gait & safe

        def first(mask: np.ndarray) -> float | None:
            hits = np.flatnonzero(mask)
            return None if hits.size == 0 else float(
                (hits[0] - ramp_end_index) * sample_period_s
            )

        def sustained(duration: float) -> float | None:
            length = max(1, int(round(duration / sample_period_s)))
            sums = np.convolve(endpoint_like.astype(np.int32),
                               np.ones(length, dtype=np.int32), mode="valid")
            hits = np.flatnonzero(sums == length)
            return None if hits.size == 0 else float(
                (hits[0] - ramp_end_index) * sample_period_s
            )

        first_cycle = None
        if gait_cycle_id is not None:
            cycles = np.asarray(gait_cycle_id, dtype=np.int64)
            for cycle in np.unique(cycles[ramp_end_index:]):
                idx = np.flatnonzero(cycles == cycle)
                if idx.size and bool(np.all(endpoint_like[idx])):
                    first_cycle = float((idx[0] - ramp_end_index) * sample_period_s)
                    break
        return {
            "first_endpoint_like_sample_s": first(endpoint_like),
            "first_0p10_sustained_pass_s": sustained(0.10),
            "first_0p20_sustained_pass_s": sustained(0.20),
            "first_complete_gait_cycle_mean_pass_s": first_cycle,
            "ramp_end_index": int(ramp_end_index),
            "formal_acquisition_metric": "first_0p20_sustained_pass_s",
        }
