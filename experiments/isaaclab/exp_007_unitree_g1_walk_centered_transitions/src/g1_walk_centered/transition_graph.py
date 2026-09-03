"""Manifest-driven locomotion state graph with Stage 0 compatibility aliases."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ControllerState(StrEnum):
    UNINITIALIZED = "UNINITIALIZED"
    STAND = "STAND"
    WALK = "WALK"
    RUN = "RUN"
    RUN_LOW = "RUN"
    CROUCH_HOLD = "CROUCH_HOLD"


class Transition(StrEnum):
    RESET_TO_STAND = "RESET_TO_STAND"
    STAND_TO_WALK = "STAND_TO_WALK"
    WALK_TO_STAND = "WALK_TO_STAND"
    WALK_TO_RUN = "WALK_TO_RUN"
    RUN_TO_WALK = "RUN_TO_WALK"
    STAND_TO_CROUCH = "STAND_TO_CROUCH"
    CROUCH_TO_STAND = "CROUCH_TO_STAND"
    CROUCH_DOWN = "STAND_TO_CROUCH"
    CROUCH_UP = "CROUCH_TO_STAND"


@dataclass(frozen=True)
class StateSpec:
    name: str
    controller: str
    status: str
    kind: str = "STEADY_STATE"
    supported_command_points_mps: list[float] | None = None


@dataclass(frozen=True)
class TransitionSpec:
    name: str
    source: str
    target: str
    controller: str
    status: str
    supported_target_speeds_mps: list[float] | None = None
    supported_source_speeds_mps: list[float] | None = None
    unsupported_target_speeds_mps: list[float] | None = None
    unsupported_reason_2_4: str | None = None
    source_command_mps: float | None = None
    controller_type: str | None = None

    @property
    def executable(self) -> bool:
        return self.status == "PASS"


@dataclass(frozen=True)
class StateGraph:
    home_state: str
    states: dict[str, StateSpec]
    transitions: dict[str, TransitionSpec]
    fail_closed: bool = True

    @classmethod
    def from_manifest(cls, path: str | Path) -> "StateGraph":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        states = {
            name: StateSpec(name=name, **definition)
            for name, definition in payload["states"].items()
        }
        transitions = {
            name: TransitionSpec(name=name, **definition)
            for name, definition in payload["transitions"].items()
        }
        for edge in transitions.values():
            if edge.source not in states or edge.target not in states:
                raise ValueError(f"transition {edge.name} references an unknown state")
        if payload["home_state"] not in states:
            raise ValueError("home_state is not registered")
        return cls(payload["home_state"], states, transitions, bool(payload.get("fail_closed", True)))

    def outgoing(self, state: str, *, executable_only: bool = True) -> tuple[TransitionSpec, ...]:
        edges = tuple(edge for edge in self.transitions.values() if edge.source == state)
        return tuple(edge for edge in edges if edge.executable) if executable_only else edges


FORMAL_SUPPORTED_TRANSITIONS = frozenset({
    Transition.STAND_TO_WALK,
    Transition.WALK_TO_STAND,
    Transition.WALK_TO_RUN,
})
TRANSITION_STATUS = {
    transition: (
        "PASS"
        if transition in (Transition.STAND_TO_WALK, Transition.WALK_TO_STAND, Transition.WALK_TO_RUN)
        else "NOT_IMPLEMENTED"
    )
    for transition in (
        Transition.RESET_TO_STAND,
        Transition.STAND_TO_WALK,
        Transition.WALK_TO_STAND,
        Transition.WALK_TO_RUN,
        Transition.RUN_TO_WALK,
        Transition.STAND_TO_CROUCH,
        Transition.CROUCH_TO_STAND,
    )
}
