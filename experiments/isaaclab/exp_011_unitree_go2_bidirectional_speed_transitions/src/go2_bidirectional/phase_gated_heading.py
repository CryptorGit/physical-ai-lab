"""Frozen command-layer heading controller used by exp_011 Stage 10."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .stage6_endpoint_protocol import circular_median


class ControllerPhase(str, Enum):
    DISABLED_SOURCE = "DISABLED_SOURCE"
    DISABLED_RAMP = "DISABLED_RAMP"
    WAIT_TARGET_ACQUISITION = "WAIT_TARGET_ACQUISITION"
    ACTIVATING = "ACTIVATING"
    ACTIVE = "ACTIVE"
    TERMINATED = "TERMINATED"


def wrapped_correction_error(reference: float, current: float) -> float:
    """Return target-minus-current yaw error in [-pi, pi]."""
    return math.atan2(math.sin(reference - current), math.cos(reference - current))


def minimum_jerk(progress: float) -> float:
    tau = min(1.0, max(0.0, float(progress)))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def target_tolerance(target_speed: float) -> float:
    target = abs(float(target_speed))
    if target <= 1e-9:
        return 0.08
    if target <= 0.6:
        return 0.15
    if target <= 1.2:
        return 0.20
    return 0.25


@dataclass
class ControllerOutput:
    phase: ControllerPhase
    reference: float | None
    error: float
    raw_command: float
    gate: float
    command: float
    acquired: bool


class PhaseGatedFixedHeadingController:
    """One-shot phase gate around the frozen policy's yaw-rate command."""

    kp = 1.0
    omega_max = 0.10
    acquisition_duration = 0.50
    activation_duration = 0.50

    def __init__(self, mode: str, kind: str, target_speed: float, dt: float = 0.02):
        if mode not in {"OPEN_LOOP", "ALWAYS_ON_FIXED_HEADING", "PHASE_GATED_FIXED_HEADING"}:
            raise ValueError(f"unsupported controller mode: {mode}")
        if kind not in {"steady", "transition"}:
            raise ValueError(f"unsupported schedule kind: {kind}")
        self.mode = mode
        self.kind = kind
        self.target_speed = float(target_speed)
        self.dt = float(dt)
        self.reset()

    def reset(self) -> None:
        self.phase = ControllerPhase.DISABLED_SOURCE
        self.reference: float | None = None
        self.reference_samples: list[float] = []
        self.acquisition_elapsed = 0.0
        self.activation_elapsed = 0.0
        self.acquired = False
        self.entries: list[dict] = [{"phase": self.phase.value, "time": 0.0}]

    def _enter(self, phase: ControllerPhase, time_s: float) -> None:
        if phase != self.phase:
            self.entries[-1]["exit_time"] = float(time_s)
            self.phase = phase
            self.entries.append({"phase": phase.value, "time": float(time_s)})

    def _set_reference(self, values: Sequence[float], fallback: float) -> None:
        if self.reference is None:
            self.reference = circular_median(list(values)) if values else float(fallback)

    def update(
        self,
        time_s: float,
        yaw: float,
        actual_speed: float,
        schedule_phase: str,
    ) -> ControllerOutput:
        time_s = float(time_s)
        yaw = float(yaw)
        if self.mode == "OPEN_LOOP":
            if self.reference is None:
                self.reference = yaw
            return ControllerOutput(
                self.phase, self.reference, wrapped_correction_error(self.reference, yaw),
                0.0, 0.0, 0.0, False,
            )
        if self.mode == "ALWAYS_ON_FIXED_HEADING":
            if self.reference is None:
                self.reference = yaw
                self._enter(ControllerPhase.ACTIVE, time_s)
            error = wrapped_correction_error(self.reference, yaw)
            raw = max(-self.omega_max, min(self.omega_max, self.kp * error))
            return ControllerOutput(self.phase, self.reference, error, raw, 1.0, raw, True)

        gate = 0.0
        if self.kind == "steady":
            if 0.5 <= time_s <= 1.0:
                self.reference_samples.append(yaw)
            if time_s < 1.0:
                self._enter(ControllerPhase.DISABLED_SOURCE, time_s)
            elif time_s < 1.5:
                self._set_reference(self.reference_samples, yaw)
                self._enter(ControllerPhase.ACTIVATING, time_s)
                gate = minimum_jerk((time_s - 1.0) / self.activation_duration)
            else:
                self._set_reference(self.reference_samples, yaw)
                self._enter(ControllerPhase.ACTIVE, time_s)
                gate = 1.0
                self.acquired = True
        else:
            if 2.5 <= time_s <= 3.0:
                self.reference_samples.append(yaw)
            if schedule_phase == "source":
                self._enter(ControllerPhase.DISABLED_SOURCE, time_s)
            elif schedule_phase == "ramp":
                self._set_reference(self.reference_samples, yaw)
                self._enter(ControllerPhase.DISABLED_RAMP, time_s)
            elif not self.acquired:
                self._set_reference(self.reference_samples, yaw)
                self._enter(ControllerPhase.WAIT_TARGET_ACQUISITION, time_s)
                within = (
                    abs(actual_speed) <= target_tolerance(0.0)
                    if abs(self.target_speed) <= 1e-9
                    else abs(actual_speed - self.target_speed) <= target_tolerance(self.target_speed)
                )
                self.acquisition_elapsed = self.acquisition_elapsed + self.dt if within else 0.0
                if self.acquisition_elapsed + 1e-12 >= self.acquisition_duration:
                    self.acquired = True
                    self.activation_elapsed = 0.0
                    self._enter(ControllerPhase.ACTIVATING, time_s)
            elif self.phase == ControllerPhase.ACTIVATING:
                self.activation_elapsed += self.dt
                gate = minimum_jerk(self.activation_elapsed / self.activation_duration)
                if self.activation_elapsed + 1e-12 >= self.activation_duration:
                    self._enter(ControllerPhase.ACTIVE, time_s)
                    gate = 1.0
            else:
                self._enter(ControllerPhase.ACTIVE, time_s)
                gate = 1.0

        error = wrapped_correction_error(self.reference, yaw) if self.reference is not None else 0.0
        raw = max(-self.omega_max, min(self.omega_max, self.kp * error))
        return ControllerOutput(self.phase, self.reference, error, raw, gate, gate * raw, self.acquired)

    def terminate(self, time_s: float) -> None:
        self._enter(ControllerPhase.TERMINATED, time_s)


def run_unit_tests() -> dict:
    tests: dict[str, dict] = {}

    def record(name: str, passed: bool, **evidence) -> None:
        tests[name] = {"pass": bool(passed), **evidence}

    positive = wrapped_correction_error(0.0, -0.2)
    negative = wrapped_correction_error(0.0, 0.2)
    record("heading_sign_positive", positive > 0, error=positive)
    record("heading_sign_negative", negative < 0, error=negative)
    wrapped = wrapped_correction_error(math.radians(-179), math.radians(179))
    record("angle_wrap_179", abs(wrapped - math.radians(2)) < 1e-12, error=wrapped)
    gates = [minimum_jerk(value) for value in (0.0, 0.5, 1.0)]
    record("minimum_jerk_progression", gates == [0.0, 0.5, 1.0], gates=gates)

    transition = PhaseGatedFixedHeadingController(
        "PHASE_GATED_FIXED_HEADING", "transition", 0.4, 0.02
    )
    outputs = [
        transition.update(0.0, 0.0, 0.0, "source"),
        transition.update(3.1, 0.1, 0.1, "ramp"),
        transition.update(4.6, 0.1, 0.0, "target"),
    ]
    for index in range(25):
        outputs.append(transition.update(4.62 + 0.02 * index, 0.1, 0.4, "target"))
    record(
        "transition_feedback_off_before_acquisition",
        all(item.gate == 0.0 for item in outputs[:3]),
        phases=[item.phase.value for item in outputs[:3]],
    )
    for index in range(25):
        outputs.append(transition.update(5.12 + 0.02 * index, 0.1, 0.4, "target"))
    record(
        "post_acquisition_activation",
        outputs[-1].phase == ControllerPhase.ACTIVE and outputs[-1].gate == 1.0,
        final_phase=outputs[-1].phase.value,
        final_gate=outputs[-1].gate,
    )

    failure = PhaseGatedFixedHeadingController(
        "PHASE_GATED_FIXED_HEADING", "transition", 0.4, 0.02
    )
    failure_outputs = [
        failure.update(4.5 + 0.02 * index, 0.1, 0.0, "target") for index in range(100)
    ]
    record(
        "acquisition_failure_stays_disabled",
        all(item.gate == 0.0 for item in failure_outputs) and not failure.acquired,
    )
    transition.reset()
    record(
        "reset_clears_episode_state",
        transition.reference is None and transition.phase == ControllerPhase.DISABLED_SOURCE,
    )
    return {"tests": tests, "all_pass": all(item["pass"] for item in tests.values())}
