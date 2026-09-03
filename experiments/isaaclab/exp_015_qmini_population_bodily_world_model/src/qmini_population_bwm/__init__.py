"""Auditable Stage 0-1 building blocks for the Unitree Qmini experiment.

This package deliberately contains contracts and deterministic data-generation
utilities only. It does not train a policy, world model, selector, MoE, or
distillation student.
"""

from .data_schema import (
    ACTION_DIM,
    OBSERVATION_DIM,
    CanonicalBodilyObservation,
    TransitionRecord,
)
from .fatigue import FatigueLedger, FatigueState
from .hidden_physics import HiddenPhysics, HiddenPhysicsState
from .qmini_asset import (
    QMINI_JOINT_ORDER,
    QminiContract,
    load_qmini_contract,
    validate_qmini_contract,
)

__all__ = [
    "ACTION_DIM",
    "OBSERVATION_DIM",
    "CanonicalBodilyObservation",
    "FatigueLedger",
    "FatigueState",
    "HiddenPhysics",
    "HiddenPhysicsState",
    "QMINI_JOINT_ORDER",
    "QminiContract",
    "TransitionRecord",
    "load_qmini_contract",
    "validate_qmini_contract",
]
