"""Fail-closed external controller router for command_system_v1.

This module deliberately does not add a policy skill, one-hot entry, or actor
observation.  It selects already-frozen controller endpoints outside the
152-dimensional learned policy interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class ControllerState(StrEnum):
    RUN = "RUN"
    TURN = "TURN"
    STOP = "STOP"
    STAND = "STAND"
    CROUCH_SHALLOW = "CROUCH_SHALLOW"


class ControllerFamily(StrEnum):
    RUNNING_FAMILY = "RUNNING_FAMILY"
    STANDING_FAMILY = "STANDING_FAMILY"
    PROTOTYPE = "PROTOTYPE"
    UNSUPPORTED = "UNSUPPORTED"


FAMILY = {
    ControllerState.RUN: ControllerFamily.RUNNING_FAMILY,
    ControllerState.TURN: ControllerFamily.RUNNING_FAMILY,
    ControllerState.STAND: ControllerFamily.STANDING_FAMILY,
    ControllerState.CROUCH_SHALLOW: ControllerFamily.STANDING_FAMILY,
    ControllerState.STOP: ControllerFamily.PROTOTYPE,
}

SUPPORTED_TRANSITIONS = frozenset({
    (ControllerState.RUN, ControllerState.RUN),
    (ControllerState.RUN, ControllerState.TURN),
    (ControllerState.TURN, ControllerState.RUN),
    (ControllerState.TURN, ControllerState.TURN),
    (ControllerState.STAND, ControllerState.STAND),
    (ControllerState.STAND, ControllerState.CROUCH_SHALLOW),
    (ControllerState.CROUCH_SHALLOW, ControllerState.STAND),
    (ControllerState.CROUCH_SHALLOW, ControllerState.CROUCH_SHALLOW),
})

UNSUPPORTED_SKILLS = {
    "CROUCH_DEEP": "DEEP_CROUCH_RETURN_UNRESOLVED",
    "STEP_OVER": "OPTIMIZATION_FAILURE",
    "LAND": "POSITION_OFFSET_LANDING_CONTROLLER_FAILED",
}

BASE_OPTION = {
    ControllerState.RUN: "stage4_running_base",
    ControllerState.TURN: "stage4_running_base",
    ControllerState.STOP: "stage4_running_base_stop_prototype_overlay",
    ControllerState.STAND: "stage2_standing_base_model_4246",
    ControllerState.CROUCH_SHALLOW: "stage2_standing_base_model_4246",
}

SCRIPTED_CONTROLLER = {
    ControllerState.RUN: "none",
    ControllerState.TURN: "none",
    ControllerState.STOP: "stop_prototype_overlay",
    ControllerState.STAND: "none",
    ControllerState.CROUCH_SHALLOW: "scripted_shallow_v1",
}


@dataclass(frozen=True)
class CommandDecision:
    current_controller_state: str
    requested_controller_state: str
    current_family: str
    requested_family: str
    transition_supported: bool
    transition_started: bool
    active_controller_state: str
    rejection_reason: str
    command_sequence_id: int
    transition_progress: float
    base_option_id: str
    scripted_controller_id: str
    skill_supported: bool = True
    primitive_started: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class CommandSystemRouter:
    """Stateful router with atomic rejection and no implicit STOP insertion."""

    def __init__(self, initial_state: ControllerState):
        self.current = ControllerState(initial_state)
        self.active = self.current
        self.pending: ControllerState | None = None
        self.sequence_id = 0
        self.progress = 1.0

    def request(self, requested: str | ControllerState) -> CommandDecision:
        self.sequence_id += 1
        requested_name = str(requested)
        if requested_name in UNSUPPORTED_SKILLS:
            return self._reject(
                requested_name, ControllerFamily.UNSUPPORTED,
                UNSUPPORTED_SKILLS[requested_name], skill_supported=False,
            )
        try:
            target = ControllerState(requested_name)
        except ValueError:
            return self._reject(
                requested_name, ControllerFamily.UNSUPPORTED,
                "UNKNOWN_CONTROLLER_STATE", skill_supported=False,
            )
        edge = (self.current, target)
        if edge not in SUPPORTED_TRANSITIONS:
            reason = (
                "PROTOTYPE_TRANSITION_NOT_FORMAL" if ControllerState.STOP in edge
                else "CROSS_BASE_FAMILY_TRANSITION_UNRESOLVED"
            )
            return self._reject(target.value, FAMILY[target], reason)
        if target == self.current:
            self.active = self.current; self.pending = None; self.progress = 1.0
            return self._decision(target, supported=True, started=False)
        self.pending = target; self.active = target; self.progress = 0.0
        return self._decision(target, supported=True, started=True)

    def set_transition_progress(self, progress: float) -> CommandDecision:
        if self.pending is None:
            self.progress = 1.0
            return self._decision(self.current, supported=True, started=False)
        self.progress = min(max(float(progress), 0.0), 1.0)
        target = self.pending
        if self.progress >= 1.0:
            self.current = target; self.active = target; self.pending = None
        return self._decision(target, supported=True, started=self.pending is not None)

    def _reject(
        self, requested: str, requested_family: ControllerFamily, reason: str,
        *, skill_supported: bool = True,
    ) -> CommandDecision:
        # The active state, base endpoint, current command, and progress are
        # intentionally untouched by a rejected request.
        return CommandDecision(
            current_controller_state=self.current.value,
            requested_controller_state=requested,
            current_family=FAMILY[self.current].value,
            requested_family=requested_family.value,
            transition_supported=False,
            transition_started=False,
            active_controller_state=self.active.value,
            rejection_reason=reason,
            command_sequence_id=self.sequence_id,
            transition_progress=self.progress,
            base_option_id=BASE_OPTION[self.active],
            scripted_controller_id=SCRIPTED_CONTROLLER[self.active],
            skill_supported=skill_supported,
            primitive_started=False,
        )

    def _decision(self, target: ControllerState, *, supported: bool, started: bool) -> CommandDecision:
        active = self.active
        return CommandDecision(
            current_controller_state=self.current.value,
            requested_controller_state=target.value,
            current_family=FAMILY[self.current].value,
            requested_family=FAMILY[target].value,
            transition_supported=supported,
            transition_started=started,
            active_controller_state=active.value,
            rejection_reason="",
            command_sequence_id=self.sequence_id,
            transition_progress=self.progress,
            base_option_id=BASE_OPTION[active],
            scripted_controller_id=SCRIPTED_CONTROLLER[active],
            primitive_started=started and target == ControllerState.CROUCH_SHALLOW,
        )


def select_controller_action(
    state: ControllerState,
    *, running_family_action, standing_base_action, stop_prototype_action,
    crouch_shallow_offset,
):
    """Select exactly one base endpoint and the allowed frozen overlay."""
    state = ControllerState(state)
    if state in (ControllerState.RUN, ControllerState.TURN):
        return running_family_action
    if state == ControllerState.STOP:
        return stop_prototype_action
    if state == ControllerState.STAND:
        return standing_base_action
    if state == ControllerState.CROUCH_SHALLOW:
        return standing_base_action + crouch_shallow_offset
    raise ValueError(f"No action route for {state}")
