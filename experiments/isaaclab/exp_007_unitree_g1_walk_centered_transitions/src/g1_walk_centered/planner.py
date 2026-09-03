"""Fail-closed command decomposition over registered transition edges."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from .transition_graph import StateGraph


@dataclass(frozen=True)
class PathPlan:
    supported: bool
    current_state: str
    target_state: str
    states: tuple[str, ...]
    transitions: tuple[str, ...]
    rejection_reason: str = ""


class ExternalCommandKind(StrEnum):
    WALK = "WALK"
    STOP = "STOP"
    RUN = "RUN"
    CROUCH = "CROUCH"


@dataclass(frozen=True)
class ExternalCommand:
    kind: ExternalCommandKind
    speed_mps: float | None = None


@dataclass(frozen=True)
class CommandPlan:
    supported: bool
    command: ExternalCommand
    target_state: str
    path: PathPlan
    rejection_reason: str = ""


class CommandPlanner:
    def __init__(self, graph: StateGraph):
        self.graph = graph

    def plan(self, current_state: str, target_state: str) -> PathPlan:
        if current_state not in self.graph.states or target_state not in self.graph.states:
            return PathPlan(False, current_state, target_state, (), (), "UNKNOWN_STATE")
        if current_state == target_state:
            return PathPlan(True, current_state, target_state, (current_state,), ())
        queue = deque([(current_state, (current_state,), ())])
        visited = {current_state}
        while queue:
            state, states, transitions = queue.popleft()
            for edge in self.graph.outgoing(state, executable_only=True):
                if edge.target in visited:
                    continue
                next_states = (*states, edge.target)
                next_transitions = (*transitions, edge.name)
                if edge.target == target_state:
                    return PathPlan(True, current_state, target_state, next_states, next_transitions)
                visited.add(edge.target)
                queue.append((edge.target, next_states, next_transitions))
        return PathPlan(False, current_state, target_state, (), (), "NO_FORMALLY_SUPPORTED_PATH")

    def plan_command(self, current_state: str, command: ExternalCommand) -> CommandPlan:
        """Decompose a user command into a registered graph path.

        The WALK contract is intentionally discrete. Unsupported values are
        rejected rather than clamped, and STOP while already at STAND is a
        supported no-op.
        """
        if command.kind is ExternalCommandKind.WALK:
            supported = (0.6, 0.8, 1.0, 1.2)
            if command.speed_mps not in supported:
                empty = PathPlan(False, current_state, "WALK", (), (), "UNSUPPORTED_WALK_SPEED")
                return CommandPlan(False, command, "WALK", empty, "UNSUPPORTED_WALK_SPEED")
            target = "WALK"
        elif command.kind is ExternalCommandKind.STOP:
            target = "STAND"
        elif command.kind is ExternalCommandKind.RUN:
            if command.speed_mps not in (2.6, 2.8):
                empty = PathPlan(False, current_state, "RUN_LOW", (), (), "UNSUPPORTED_WALK_TO_RUN_TARGET")
                return CommandPlan(False, command, "RUN_LOW", empty, "UNSUPPORTED_WALK_TO_RUN_TARGET")
            target = "RUN_LOW"
        else:
            empty = PathPlan(False, current_state, "", (), (), "UNSUPPORTED_COMMAND")
            return CommandPlan(False, command, "", empty, "UNSUPPORTED_COMMAND")
        path = self.plan(current_state, target)
        return CommandPlan(path.supported, command, target, path, path.rejection_reason)
