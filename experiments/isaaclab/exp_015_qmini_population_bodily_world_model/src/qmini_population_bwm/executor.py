"""Policy-independent action application and proposed/applied logging."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable


def _finite_tuple(values: Iterable[float], *, size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} has length {len(result)}, expected {size}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class AppliedAction:
    action_proposed: tuple[float, ...]
    action_applied: tuple[float, ...]
    saturation_mask: tuple[bool, ...]
    saturation_dwell: tuple[int, ...]


@dataclass
class ActionExecutor:
    """Clip a proposed joint-target action against the fixed Qmini contract.

    Actions are joint targets in radians. This avoids an unverified scalar
    action scale; normalized policy encodings can be added later using the
    URDF-derived midpoint/half-range map in qmini_asset.py.
    """

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    _saturation_dwell: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper) or not self.lower:
            raise ValueError("lower and upper must have the same non-zero length")
        if any(not math.isfinite(v) for v in (*self.lower, *self.upper)):
            raise ValueError("action bounds must be finite")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper, strict=True)):
            raise ValueError("every lower bound must be below its upper bound")
        self._saturation_dwell = [0] * len(self.lower)

    @property
    def saturation_dwell(self) -> tuple[int, ...]:
        return tuple(self._saturation_dwell)

    def apply(self, action_proposed: Iterable[float]) -> AppliedAction:
        proposed = _finite_tuple(action_proposed, size=len(self.lower), name="action_proposed")
        applied = tuple(
            min(max(value, lower), upper)
            for value, lower, upper in zip(proposed, self.lower, self.upper, strict=True)
        )
        mask = tuple(value != clipped for value, clipped in zip(proposed, applied, strict=True))
        for index, saturated in enumerate(mask):
            self._saturation_dwell[index] = self._saturation_dwell[index] + 1 if saturated else 0
        return AppliedAction(proposed, applied, mask, self.saturation_dwell)

    def reset(self) -> None:
        self._saturation_dwell[:] = [0] * len(self.lower)
