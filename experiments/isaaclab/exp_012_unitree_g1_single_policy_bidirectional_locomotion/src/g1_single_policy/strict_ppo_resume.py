"""EXP012-local strict PPO resume learning-rate contract.

This adapter intentionally does not modify RSL-RL.  The restored optimizer
state is the sole source of truth for current learning rate on resume.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


class ResumeContractError(RuntimeError):
    """Fail-closed error raised by the strict resume contract."""


@dataclass(frozen=True)
class ResumeLRState:
    mode: str
    optimizer_param_groups: int
    restored_lr: float
    algorithm_lr: float
    scheduler_lr: float
    runner_lr_field: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Exp012StrictPPOResumeContract:
    """Synchronize runtime LR state from a strictly restored optimizer."""

    def __init__(self, absolute_tolerance: float = 1e-12):
        self.absolute_tolerance = float(absolute_tolerance)

    @staticmethod
    def _optimizer_lr(optimizer: Any) -> tuple[float, int]:
        groups = getattr(optimizer, "param_groups", None)
        if not groups:
            raise ResumeContractError("PPO_RESUME_OPTIMIZER_STATE_MISSING")
        values = [float(group["lr"]) for group in groups]
        if not all(math.isfinite(value) for value in values):
            raise ResumeContractError("PPO_RESTORED_OPTIMIZER_LR_AMBIGUOUS")
        if any(value != values[0] for value in values[1:]):
            raise ResumeContractError("PPO_RESTORED_OPTIMIZER_LR_AMBIGUOUS")
        return values[0], len(values)

    def synchronize(self, algorithm: Any, runner: Any | None = None, *, resume: bool = True) -> ResumeLRState:
        if not resume:
            current, count = self._optimizer_lr(algorithm.optimizer)
            return ResumeLRState("fresh", count, current, float(algorithm.learning_rate), float(algorithm.learning_rate), "NOT_PRESENT")

        restored, count = self._optimizer_lr(algorithm.optimizer)
        algorithm.learning_rate = restored

        runner_field = "NOT_PRESENT"
        for name in ("learning_rate", "current_learning_rate"):
            if runner is not None and hasattr(runner, name):
                setattr(runner, name, restored)
                runner_field = name

        # RSL-RL PPO's adaptive scheduler current LR is self.learning_rate.
        scheduler_lr = float(algorithm.learning_rate)
        state = ResumeLRState("resume", count, restored, float(algorithm.learning_rate), scheduler_lr, runner_field)
        self.assert_first_step_invariant(algorithm, expected=restored)
        return state

    def assert_first_step_invariant(self, algorithm: Any, *, expected: float) -> None:
        optimizer_lr, _ = self._optimizer_lr(algorithm.optimizer)
        runtime_lr = float(algorithm.learning_rate)
        if abs(optimizer_lr - expected) > self.absolute_tolerance or abs(runtime_lr - expected) > self.absolute_tolerance:
            raise ResumeContractError("PPO_RUNTIME_LR_SYNC_FAIL")

    @staticmethod
    def require_optimizer_state(payload: dict[str, Any]) -> None:
        state = payload.get("optimizer_state_dict")
        if not state or not state.get("state"):
            raise ResumeContractError("PPO_RESUME_OPTIMIZER_STATE_MISSING")
