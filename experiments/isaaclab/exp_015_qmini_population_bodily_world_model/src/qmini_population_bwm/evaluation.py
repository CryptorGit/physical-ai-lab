"""Stage 0-1 formal gates and failure taxonomy."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Iterable, Mapping


FAILURE_TAXONOMY = (
    "EXP015_QMINI_STAGE1_PASS",
    "NO_GO_QMINI_BASELINE",
    "NO_GO_MEMORY_NECESSITY",
    "NO_GO_HIDDEN_FACTOR_RELEVANCE",
    "NO_GO_SNAPSHOT_REPRODUCIBILITY",
    "NO_GO_ACTION_EFFECT_SEPARATION",
    "INVALID_QMINI_PHYSICS_CONTRACT",
    "INVALID_SOURCE_MUTATION",
    "INVALID_DATA_CONTRACT",
)


@dataclass(frozen=True)
class BaselineFormalResult:
    episodes: int
    falls: int
    fall_rate: float | None
    finite_state_fraction: float | None
    measured_safe_velocity_range_mps: tuple[float, float] | None
    velocity_tracking_error: dict[str, float] | None
    slip_distribution: dict[str, float] | None
    torque_distribution: dict[str, float] | None
    saturation_dwell: dict[str, float] | None
    contact_pattern: dict[str, Any] | None
    orientation: dict[str, float] | None
    power_mechanical_work: dict[str, float] | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stats(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "n": float(len(values)),
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p50": ordered[len(ordered) // 2],
    }


def formal_baseline_gate(episodes: Iterable[Mapping[str, Any]]) -> BaselineFormalResult:
    """Evaluate the required 50-episode Qmini WALK formal gate.

    The input is intentionally a metric table emitted by a real simulator
    evaluation. Missing data never becomes a synthetic PASS.
    """

    rows = list(episodes)
    falls = sum(bool(row.get("fell", False)) for row in rows)
    finite_flags = [bool(row.get("finite_state", False)) for row in rows]
    velocities = [
        float(row["command_velocity_mps"])
        for row in rows
        if row.get("stable", False) and row.get("command_velocity_mps") is not None
    ]
    tracking = [float(row["velocity_tracking_error"]) for row in rows if row.get("velocity_tracking_error") is not None]
    slip = [float(row["slip"]) for row in rows if row.get("slip") is not None]
    torque = [float(row["torque_rms"]) for row in rows if row.get("torque_rms") is not None]
    dwell = [float(row["saturation_dwell"]) for row in rows if row.get("saturation_dwell") is not None]
    roll = [abs(float(row["roll_rad"])) for row in rows if row.get("roll_rad") is not None]
    pitch = [abs(float(row["pitch_rad"])) for row in rows if row.get("pitch_rad") is not None]
    power = [float(row["mechanical_work"]) for row in rows if row.get("mechanical_work") is not None]
    finite_fraction = sum(finite_flags) / len(finite_flags) if finite_flags else None
    fall_rate = falls / len(rows) if rows else None
    valid = (
        len(rows) >= 50
        and fall_rate is not None
        and fall_rate <= 0.02
        and finite_fraction == 1.0
        and bool(velocities)
    )
    return BaselineFormalResult(
        episodes=len(rows),
        falls=falls,
        fall_rate=fall_rate,
        finite_state_fraction=finite_fraction,
        measured_safe_velocity_range_mps=(min(velocities), max(velocities)) if velocities else None,
        velocity_tracking_error=_stats(tracking),
        slip_distribution=_stats(slip),
        torque_distribution=_stats(torque),
        saturation_dwell=_stats(dwell),
        contact_pattern=None,
        orientation={"roll_abs_rad": _stats(roll)["mean"], "pitch_abs_rad": _stats(pitch)["mean"]}
        if roll and pitch
        else None,
        power_mechanical_work=_stats(power),
        status="PASS" if valid else "NO_GO_QMINI_BASELINE",
    )


@dataclass(frozen=True)
class MemoryNecessityResult:
    factor: str
    current_observation_score: float | None
    history_score: float | None
    relative_improvement: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def memory_necessity_gate(
    factor_scores: Mapping[str, tuple[float, float]],
    *,
    minimum_relative_improvement: float = 0.05,
) -> tuple[MemoryNecessityResult, ...]:
    """Compare A=current observation only against B=short history."""

    results: list[MemoryNecessityResult] = []
    for factor, (current_score, history_score) in factor_scores.items():
        current = float(current_score)
        history = float(history_score)
        if current == 0.0:
            relative = math.inf if history > 0.0 else 0.0
        else:
            relative = (history - current) / abs(current)
        results.append(
            MemoryNecessityResult(
                factor=factor,
                current_observation_score=current,
                history_score=history,
                relative_improvement=relative,
                status="PASS" if relative >= minimum_relative_improvement else "NO_GO_MEMORY_NECESSITY",
            )
        )
    return tuple(results)


@dataclass(frozen=True)
class Stage1GateResult:
    gates: dict[str, str]
    final_classification: str
    stop_after_stage1: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_stage1_gate(gates: Mapping[str, bool | None]) -> Stage1GateResult:
    """Map explicit gate statuses to the requested final classification."""

    normalized = {name: ("PASS" if value is True else "FAIL" if value is False else "NOT_RUN") for name, value in gates.items()}
    if gates.get("source_protected_write") is False:
        classification = "INVALID_SOURCE_MUTATION"
    elif gates.get("qmini_physics_contract") is False:
        classification = "INVALID_QMINI_PHYSICS_CONTRACT"
    elif gates.get("data_contract") is False:
        classification = "INVALID_DATA_CONTRACT"
    elif gates.get("baseline_walk_formal") is not True:
        classification = "NO_GO_QMINI_BASELINE"
    elif gates.get("snapshot_deterministic_replay") is not True:
        classification = "NO_GO_SNAPSHOT_REPRODUCIBILITY"
    elif gates.get("hidden_factor_relevance") is not True:
        classification = "NO_GO_HIDDEN_FACTOR_RELEVANCE"
    elif gates.get("memory_necessity") is not True:
        classification = "NO_GO_MEMORY_NECESSITY"
    elif gates.get("crossed_action_separation") is not True:
        classification = "NO_GO_ACTION_EFFECT_SEPARATION"
    else:
        classification = "EXP015_QMINI_STAGE1_PASS"
    return Stage1GateResult(normalized, classification)
