"""Same-snapshot macro-action branching for causal intervention data."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Iterable, Mapping, Any

from .snapshot_clone import QminiSnapshot


POLICY_HORIZONS = (1, 5, 10, 25)


@dataclass(frozen=True)
class MacroAction:
    macro_id: str
    action: tuple[float, ...]
    command_velocity_mps: float | None = None
    provenance: str = "MEASURED_SAFE_RANGE_REQUIRED"


@dataclass(frozen=True)
class CrossedInterventionRecord:
    source_snapshot_id: str
    macro_id: str
    horizon_policy_steps: int
    state_delta: dict[str, Any]
    progress: float | None
    velocity: float | None
    velocity_error: float | None
    energy: float | None
    mechanical_work: float | None
    torque_rms: float | None
    torque_peak: float | None
    saturation_dwell: float | None
    impact: float | None
    slip: float | None
    contact: dict[str, Any] | None
    fatigue_delta: dict[str, Any] | None
    stable: bool | None
    fell: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_macros_from_safe_speed_range(
    safe_velocity_range: tuple[float, float] | None,
    *,
    action_by_velocity: Mapping[float, Iterable[float]] | None = None,
) -> tuple[MacroAction, ...]:
    """Create low/nominal/high macros only after an empirical sweep.

    No speed value is embedded here. Passing None is an intentional stop.
    """

    if safe_velocity_range is None:
        raise ValueError("formal Qmini speed sweep is required before macro actions")
    low, high = (float(value) for value in safe_velocity_range)
    if low > high:
        raise ValueError("safe velocity range must be ordered")
    nominal = (low + high) / 2.0
    speeds = (low, nominal, high)
    if action_by_velocity is None:
        raise ValueError("macro action vectors must come from the measured baseline policy")
    macros: list[MacroAction] = []
    for index, speed in enumerate(speeds):
        if speed not in action_by_velocity:
            raise ValueError(f"missing measured action for safe velocity {speed}")
        macros.append(
            MacroAction(
                macro_id=("slow_walk", "nominal_walk", "fast_walk")[index],
                action=tuple(float(value) for value in action_by_velocity[speed]),
                command_velocity_mps=speed,
                provenance="FORMAL_QMINI_BASELINE_SWEEP",
            )
        )
    return tuple(macros)


def _state_delta(start: QminiSnapshot, end: QminiSnapshot) -> dict[str, Any]:
    return {
        "root_pose": [b - a for a, b in zip(start.root_pose, end.root_pose, strict=True)],
        "root_velocity": [b - a for a, b in zip(start.root_velocity, end.root_velocity, strict=True)],
        "joint_q": [b - a for a, b in zip(start.joint_q, end.joint_q, strict=True)],
        "joint_dq": [b - a for a, b in zip(start.joint_dq, end.joint_dq, strict=True)],
    }


def cross_snapshot(
    snapshot: QminiSnapshot,
    macros: Iterable[MacroAction],
    *,
    step_fn: Callable[[QminiSnapshot, tuple[float, ...]], tuple[QminiSnapshot, Mapping[str, Any]]],
    horizons: Iterable[int] = POLICY_HORIZONS,
    source_snapshot_id: str = "UNSET",
) -> tuple[CrossedInterventionRecord, ...]:
    """Branch each macro from an independent clone of one source snapshot."""

    horizon_values = tuple(int(value) for value in horizons)
    if any(value <= 0 for value in horizon_values):
        raise ValueError("intervention horizons must be positive")
    results: list[CrossedInterventionRecord] = []
    for macro in macros:
        for horizon in horizon_values:
            current = snapshot.clone()
            metrics: Mapping[str, Any] = {}
            for _ in range(horizon):
                current, metrics = step_fn(current, macro.action)
            results.append(
                CrossedInterventionRecord(
                    source_snapshot_id=source_snapshot_id,
                    macro_id=macro.macro_id,
                    horizon_policy_steps=horizon,
                    state_delta=_state_delta(snapshot, current),
                    progress=_optional_float(metrics.get("progress")),
                    velocity=_optional_float(metrics.get("velocity")),
                    velocity_error=_optional_float(metrics.get("velocity_error")),
                    energy=_optional_float(metrics.get("energy")),
                    mechanical_work=_optional_float(metrics.get("mechanical_work")),
                    torque_rms=_optional_float(metrics.get("torque_rms")),
                    torque_peak=_optional_float(metrics.get("torque_peak")),
                    saturation_dwell=_optional_float(metrics.get("saturation_dwell")),
                    impact=_optional_float(metrics.get("impact")),
                    slip=_optional_float(metrics.get("slip")),
                    contact=metrics.get("contact"),
                    fatigue_delta=metrics.get("fatigue_delta"),
                    stable=metrics.get("stable"),
                    fell=metrics.get("fell"),
                )
            )
    return tuple(results)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
