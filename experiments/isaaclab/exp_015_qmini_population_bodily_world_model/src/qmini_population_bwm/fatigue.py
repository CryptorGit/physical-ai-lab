"""Left/right actuator fatigue ledger for hidden-physics experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


ACTUATORS_PER_LEG = 5


def _values(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class FatigueState:
    left: tuple[float, ...] = (0.0,) * ACTUATORS_PER_LEG
    right: tuple[float, ...] = (0.0,) * ACTUATORS_PER_LEG

    def __post_init__(self) -> None:
        left = _values(self.left, ACTUATORS_PER_LEG, "left fatigue")
        right = _values(self.right, ACTUATORS_PER_LEG, "right fatigue")
        if any(value < 0.0 or value > 1.0 for value in (*left, *right)):
            raise ValueError("fatigue must be in [0, 1]")

    @property
    def flat(self) -> tuple[float, ...]:
        return self.left + self.right


@dataclass(frozen=True)
class FatigueStep:
    before: FatigueState
    power_normalized_left: tuple[float, ...]
    power_normalized_right: tuple[float, ...]
    after: FatigueState
    effectiveness: tuple[float, ...]
    mechanical_work_left: float
    mechanical_work_right: float


@dataclass
class FatigueLedger:
    """Implement the frozen Stage-1 equation with explicit parameters.

    power_normalized is a measured, non-negative, dimensionless power input
    supplied by the simulator adapter. Its normalization reference is part of
    calibration and is never silently inferred from G1 or 8010 data.
    """

    alpha: float
    beta: float
    effectiveness_coefficient: float
    state: FatigueState = field(default_factory=FatigueState)
    _mechanical_work_left: float = field(default=0.0, init=False, repr=False)
    _mechanical_work_right: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, value in (
            ("alpha", self.alpha),
            ("beta", self.beta),
            ("effectiveness_coefficient", self.effectiveness_coefficient),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.effectiveness_coefficient > 1.0:
            raise ValueError("effectiveness_coefficient must be <= 1")

    def effectiveness(self, state: FatigueState | None = None) -> tuple[float, ...]:
        current = state or self.state
        return tuple(1.0 - self.effectiveness_coefficient * value for value in current.flat)

    def step(
        self,
        power_normalized_left: Iterable[float],
        power_normalized_right: Iterable[float],
        *,
        dt: float,
    ) -> FatigueStep:
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        left = _values(power_normalized_left, ACTUATORS_PER_LEG, "left normalized power")
        right = _values(power_normalized_right, ACTUATORS_PER_LEG, "right normalized power")
        if any(value < 0.0 for value in (*left, *right)):
            raise ValueError("normalized power must be non-negative")
        before = self.state

        # User-specified model:
        # f[t+1] = clip(f[t] + alpha*P[t] - beta*f[t], 0, 1).
        next_left = tuple(
            min(max(value + self.alpha * power - self.beta * value, 0.0), 1.0)
            for value, power in zip(before.left, left, strict=True)
        )
        next_right = tuple(
            min(max(value + self.alpha * power - self.beta * value, 0.0), 1.0)
            for value, power in zip(before.right, right, strict=True)
        )
        after = FatigueState(next_left, next_right)
        self.state = after
        self._mechanical_work_left += sum(left) * dt
        self._mechanical_work_right += sum(right) * dt
        return FatigueStep(
            before=before,
            power_normalized_left=left,
            power_normalized_right=right,
            after=after,
            effectiveness=self.effectiveness(after),
            mechanical_work_left=self._mechanical_work_left,
            mechanical_work_right=self._mechanical_work_right,
        )

    @property
    def mechanical_work(self) -> tuple[float, float]:
        return self._mechanical_work_left, self._mechanical_work_right

    def reset(self) -> None:
        self.state = FatigueState()
        self._mechanical_work_left = 0.0
        self._mechanical_work_right = 0.0
