"""Frozen command profiles. Commands are the only transition mechanism."""

from __future__ import annotations


def minimum_jerk(t: float, start: float, target: float, duration: float) -> float:
    if duration <= 0:
        raise ValueError("duration must be positive")
    tau = min(1.0, max(0.0, t / duration))
    p = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return start + (target - start) * p


def transition_command(t: float, source: float, target: float, ramp_duration: float = 1.5) -> tuple[float, str]:
    if t < 3.0:
        return source, "source_hold"
    if t < 3.0 + ramp_duration:
        return minimum_jerk(t - 3.0, source, target, ramp_duration), "ramp"
    return target, "target_hold"


FULL_SEQUENCE = (0.0, 0.6, 1.2, 2.0, 2.5, 2.0, 1.2, 0.6, 0.0)
LIMITED_SEQUENCE = (0.0, 0.6, 1.2, 2.0, 1.2, 0.6, 0.0)


def sequence_command(t: float, speeds: tuple[float, ...], ramp_duration: float = 1.5) -> tuple[float, int, str]:
    hold = 3.0
    if t < hold:
        return speeds[0], 0, "hold"
    block = hold + ramp_duration
    index = min(int((t - hold) // block), len(speeds) - 2)
    local = (t - hold) - index * block
    if local < ramp_duration:
        return minimum_jerk(local, speeds[index], speeds[index + 1], ramp_duration), index + 1, "ramp"
    return speeds[index + 1], index + 1, "hold"

