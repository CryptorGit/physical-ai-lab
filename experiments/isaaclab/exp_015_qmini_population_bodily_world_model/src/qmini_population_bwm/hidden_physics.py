"""Hidden physical factors and policy-observation firewall."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .fatigue import FatigueLedger, FatigueState


@dataclass(frozen=True)
class HiddenPhysicsState:
    friction: float
    wind_xy: tuple[float, float]
    fatigue_left: tuple[float, ...]
    fatigue_right: tuple[float, ...]

    @property
    def fatigue(self) -> FatigueState:
        return FatigueState(self.fatigue_left, self.fatigue_right)

    def analysis_dict(self) -> dict[str, object]:
        return {
            "friction": self.friction,
            "wind_xy": list(self.wind_xy),
            "fatigue_left": list(self.fatigue_left),
            "fatigue_right": list(self.fatigue_right),
        }


@dataclass(frozen=True)
class HiddenPhysicsRanges:
    """Calibration output, not an unverified Qmini default."""

    friction: tuple[float, float]
    wind_x: tuple[float, float]
    wind_y: tuple[float, float]

    def __post_init__(self) -> None:
        for name, pair in (("friction", self.friction), ("wind_x", self.wind_x), ("wind_y", self.wind_y)):
            if len(pair) != 2 or pair[0] > pair[1]:
                raise ValueError(f"{name} must be an ordered [min, max] pair")


@dataclass
class HiddenPhysics:
    """Sample hidden state using a private deterministic RNG.

    The returned state is for simulator/controller internals and analysis;
    policy_visible_observation intentionally excludes every hidden field.
    """

    ranges: HiddenPhysicsRanges
    fatigue_alpha: float
    fatigue_beta: float
    fatigue_effectiveness_coefficient: float
    seed: int
    _rng: random.Random | None = None

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def sample(self) -> HiddenPhysicsState:
        assert self._rng is not None
        return HiddenPhysicsState(
            friction=self._rng.uniform(*self.ranges.friction),
            wind_xy=(
                self._rng.uniform(*self.ranges.wind_x),
                self._rng.uniform(*self.ranges.wind_y),
            ),
            fatigue_left=(0.0,) * 5,
            fatigue_right=(0.0,) * 5,
        )

    def new_ledger(self, state: HiddenPhysicsState | None = None) -> FatigueLedger:
        return FatigueLedger(
            alpha=self.fatigue_alpha,
            beta=self.fatigue_beta,
            effectiveness_coefficient=self.fatigue_effectiveness_coefficient,
            state=(state or self.sample()).fatigue,
        )

    def policy_visible_observation(self, observation: Iterable[float]) -> tuple[float, ...]:
        """Return the observation unchanged, with no hidden-factor append."""

        return tuple(float(value) for value in observation)

    def effective_torque(self, torque: Iterable[float], state: HiddenPhysicsState) -> tuple[float, ...]:
        values = tuple(float(value) for value in torque)
        eta = tuple(1.0 - self.fatigue_effectiveness_coefficient * value for value in state.fatigue.flat)
        if len(values) != len(eta):
            raise ValueError("torque must have 10 locomotion entries")
        return tuple(value * factor for value, factor in zip(values, eta, strict=True))
