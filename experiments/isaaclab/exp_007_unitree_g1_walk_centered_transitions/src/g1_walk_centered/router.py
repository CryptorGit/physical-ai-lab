"""Manifest-driven hard-switch expert router with fail-closed transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .planner import CommandPlan
from .transition_graph import ControllerState, StateGraph


@dataclass(frozen=True)
class RouteDecision:
    current_state: str
    requested_state: str
    transition_supported: bool
    transition_started: bool
    rejection_reason: str


@dataclass(frozen=True)
class ActiveController:
    graph_node: str
    controller: str
    kind: str


class ExpertRouter:
    """Resolve registered controllers; never mix model parameters."""

    def __init__(self, graph: StateGraph, controllers: dict[str, Callable], initial_state: str | None = None):
        self.graph = graph
        self.controllers = dict(controllers)
        self.current_state = initial_state or graph.home_state
        self.active_transition: str | None = None
        self.route: tuple[str, ...] = ()
        self.route_cursor = 0
        self.duplicate_commands = 0
        if self.current_state not in graph.states:
            raise ValueError("initial state is not registered")

    def active(self) -> ActiveController:
        if self.active_transition is not None:
            edge = self.graph.transitions[self.active_transition]
            return ActiveController(edge.name, edge.controller, "TRANSITION")
        state = self.graph.states[self.current_state]
        return ActiveController(state.name, state.controller, state.kind)

    def controller(self) -> Callable:
        active = self.active()
        if active.controller not in self.controllers:
            raise RuntimeError(f"controller is not registered: {active.controller}")
        return self.controllers[active.controller]

    def start_transition(self, transition_name: str, *, entry_preconditions_pass: bool) -> RouteDecision:
        if self.active_transition is not None:
            return RouteDecision(self.current_state, transition_name, False, False, "TRANSITION_ALREADY_ACTIVE")
        edge = self.graph.transitions.get(transition_name)
        if edge is None:
            return RouteDecision(self.current_state, transition_name, False, False, "UNKNOWN_TRANSITION")
        if edge.source != self.current_state:
            return RouteDecision(self.current_state, edge.target, False, False, "SOURCE_STATE_MISMATCH")
        if not edge.executable:
            return RouteDecision(self.current_state, edge.target, False, False, "TRANSITION_NOT_FORMALLY_SUPPORTED")
        if not entry_preconditions_pass:
            return RouteDecision(self.current_state, edge.target, False, False, "ENTRY_PRECONDITION_FAILED")
        self.active_transition = edge.name
        return RouteDecision(self.current_state, edge.target, True, True, "")

    def accept_plan(self, plan: CommandPlan, *, entry_preconditions_pass: bool = True) -> RouteDecision:
        """Install a planned path without restarting an active edge."""
        if self.active_transition is not None:
            self.duplicate_commands += 1
            return RouteDecision(
                self.current_state,
                plan.target_state,
                plan.supported,
                False,
                "ACTIVE_TRANSITION_CONTINUED",
            )
        if not plan.supported:
            return RouteDecision(self.current_state, plan.target_state, False, False, plan.rejection_reason)
        self.route = plan.path.transitions
        self.route_cursor = 0
        if not self.route:
            return RouteDecision(self.current_state, plan.target_state, True, False, "")
        return self.start_transition(self.route[0], entry_preconditions_pass=entry_preconditions_pass)

    def complete_transition(self, *, completion_condition_pass: bool) -> bool:
        if self.active_transition is None or not completion_condition_pass:
            return False
        edge = self.graph.transitions[self.active_transition]
        self.current_state = edge.target
        self.active_transition = None
        self.route_cursor += 1
        if self.route_cursor < len(self.route):
            next_edge = self.route[self.route_cursor]
            decision = self.start_transition(next_edge, entry_preconditions_pass=True)
            if not decision.transition_started:
                raise RuntimeError(f"planned route became invalid at {next_edge}: {decision.rejection_reason}")
        return True

    def abort_transition(self) -> None:
        self.active_transition = None
        self.route = ()
        self.route_cursor = 0


class Stage0Router:
    """Compatibility facade retaining the original Stage 0 fail-closed behavior."""

    def __init__(self, state: ControllerState):
        self.state = ControllerState(state)

    def request(self, requested: ControllerState) -> RouteDecision:
        requested = ControllerState(requested)
        if requested == self.state:
            return RouteDecision(self.state, requested, True, False, "")
        return RouteDecision(self.state, requested, False, False, "STAGE_0_BRIDGE_NOT_AUDITED")
