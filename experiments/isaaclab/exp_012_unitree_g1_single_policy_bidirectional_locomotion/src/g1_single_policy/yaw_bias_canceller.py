"""Frozen command-space yaw-bias cancellation for parent-policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass

TABLE = ((0.6, -0.0248), (0.8, -0.0742), (1.0, -0.0920), (1.2, -0.1233))
POLICY_LIMIT = 0.20
OFFSET_LIMIT = 0.15
ACTIVATION_S = 0.50


def minimum_jerk(value: float) -> float:
    tau = min(max(float(value), 0.0), 1.0)
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def lookup_offset(commanded_forward_speed: float) -> float:
    """Pure speed lookup before temporal activation/deactivation."""
    speed = abs(float(commanded_forward_speed))
    if speed <= 0.40:
        return 0.0
    if speed < 0.60:
        return minimum_jerk((speed - 0.40) / 0.20) * TABLE[0][1]
    if speed >= TABLE[-1][0]:
        return TABLE[-1][1]
    for (left_speed, left_value), (right_speed, right_value) in zip(TABLE, TABLE[1:]):
        if left_speed <= speed <= right_speed:
            fraction = (speed - left_speed) / (right_speed - left_speed)
            return left_value + fraction * (right_value - left_value)
    raise AssertionError("unreachable cancellation lookup")


@dataclass
class G1SpeedConditionedYawBiasCancellerV1:
    """Stateful 0.5 s activation/deactivation around a fixed speed lookup."""

    dt: float
    gate: float = 0.0
    _mode: str = "DISABLED"
    _elapsed: float = 0.0
    _deactivation_start: float = 0.0
    _current_offset: float = 0.0

    def reset(self) -> None:
        self.gate = 0.0
        self._mode = "DISABLED"
        self._elapsed = 0.0
        self._deactivation_start = 0.0
        self._current_offset = 0.0

    def step(self, commanded_forward_speed: float, desired_yaw_rate: float = 0.0) -> dict[str, float | str]:
        speed = abs(float(commanded_forward_speed))
        target = max(-OFFSET_LIMIT, min(OFFSET_LIMIT, lookup_offset(speed)))
        if speed > 0.40:
            if self._mode in ("DISABLED", "DEACTIVATING"):
                self._mode = "ACTIVATING"
                self._elapsed = 0.0
            if self._mode == "ACTIVATING":
                self._elapsed += self.dt
                self.gate = minimum_jerk(self._elapsed / ACTIVATION_S)
                if self._elapsed >= ACTIVATION_S:
                    self._mode = "ACTIVE"
                    self.gate = 1.0
            else:
                self.gate = 1.0
            offset = self.gate * target
        else:
            if self._mode in ("ACTIVE", "ACTIVATING"):
                self._mode = "DEACTIVATING"
                self._elapsed = 0.0
                self._deactivation_start = self._current_offset
            if self._mode == "DEACTIVATING":
                self._elapsed += self.dt
                progress = minimum_jerk(self._elapsed / ACTIVATION_S)
                offset = (1.0 - progress) * self._deactivation_start
                self.gate = 1.0 - progress
                if self._elapsed >= ACTIVATION_S:
                    self._mode = "DISABLED"
                    self.gate = 0.0
                    offset = 0.0
            else:
                self._mode = "DISABLED"
                self.gate = 0.0
                offset = 0.0
        self._current_offset = float(offset)
        policy_yaw = max(-POLICY_LIMIT, min(POLICY_LIMIT, float(desired_yaw_rate) + offset))
        return {
            "desired_yaw_rate": float(desired_yaw_rate),
            "offset": float(offset),
            "policy_yaw_rate": float(policy_yaw),
            "target_offset": float(target),
            "gate": float(self.gate),
            "state": self._mode,
        }
