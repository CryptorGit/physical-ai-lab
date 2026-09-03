"""Canonical bodily transition schema for the later shared world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import math
from typing import Iterable


LOCOMOTION_JOINT_COUNT = 10
ACTION_DIM = 10
OBSERVATION_FIELD_SIZES: tuple[tuple[str, int], ...] = (
    ("base_linear_velocity_b", 3),
    ("base_angular_velocity_b", 3),
    ("projected_gravity_b", 3),
    ("joint_position", LOCOMOTION_JOINT_COUNT),
    ("joint_velocity", LOCOMOTION_JOINT_COUNT),
    ("previous_actually_applied_action", ACTION_DIM),
    ("left_foot_contact", 1),
    ("right_foot_contact", 1),
)
OBSERVATION_DIM = sum(size for _, size in OBSERVATION_FIELD_SIZES)


def _finite(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} has length {len(result)}, expected {size}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} contains a non-finite value")
    return result


@dataclass(frozen=True)
class CanonicalBodilyObservation:
    """Only body/controller-visible values; hidden state is intentionally absent."""

    base_linear_velocity_b: tuple[float, float, float]
    base_angular_velocity_b: tuple[float, float, float]
    projected_gravity_b: tuple[float, float, float]
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    previous_actually_applied_action: tuple[float, ...]
    left_foot_contact: float
    right_foot_contact: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_linear_velocity_b", _finite(self.base_linear_velocity_b, 3, "base linear velocity"))
        object.__setattr__(self, "base_angular_velocity_b", _finite(self.base_angular_velocity_b, 3, "base angular velocity"))
        object.__setattr__(self, "projected_gravity_b", _finite(self.projected_gravity_b, 3, "projected gravity"))
        object.__setattr__(self, "joint_position", _finite(self.joint_position, 10, "joint position"))
        object.__setattr__(self, "joint_velocity", _finite(self.joint_velocity, 10, "joint velocity"))
        object.__setattr__(
            self,
            "previous_actually_applied_action",
            _finite(self.previous_actually_applied_action, 10, "previous applied action"),
        )
        for name in ("left_foot_contact", "right_foot_contact"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)

    def flatten(self) -> tuple[float, ...]:
        return (
            *self.base_linear_velocity_b,
            *self.base_angular_velocity_b,
            *self.projected_gravity_b,
            *self.joint_position,
            *self.joint_velocity,
            *self.previous_actually_applied_action,
            self.left_foot_contact,
            self.right_foot_contact,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TransitionRecord:
    """One causal transition; reward is analysis metadata, not state input."""

    episode_id: str
    source_snapshot_id: str
    t: int
    observation: CanonicalBodilyObservation
    next_observation: CanonicalBodilyObservation
    action_proposed: tuple[float, ...]
    action_applied: tuple[float, ...]
    command: tuple[float, ...]
    reward_vector: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    hidden_state_for_analysis: dict[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_proposed", _finite(self.action_proposed, ACTION_DIM, "action_proposed"))
        object.__setattr__(self, "action_applied", _finite(self.action_applied, ACTION_DIM, "action_applied"))
        object.__setattr__(self, "command", tuple(float(value) for value in self.command))
        if any(not math.isfinite(value) for value in self.command):
            raise ValueError("command contains a non-finite value")
        for name, values in (("reward_vector", self.reward_vector), ("metrics", self.metrics)):
            if any(not isinstance(key, str) or not math.isfinite(float(value)) for key, value in values.items()):
                raise ValueError(f"{name} must map string names to finite numeric values")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def observation_field_names() -> tuple[str, ...]:
    names: list[str] = []
    for field_name, size in OBSERVATION_FIELD_SIZES:
        names.extend(f"{field_name}[{index}]" for index in range(size))
    return tuple(names)


def assign_split(
    source_snapshot_id: str,
    *,
    seed: int = 15015,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> str:
    """Assign every branch of one source snapshot to one stable split."""

    if abs(sum(ratios) - 1.0) > 1e-12 or any(value <= 0.0 for value in ratios):
        raise ValueError("split ratios must be positive and sum to one")
    digest = hashlib.sha256(f"{seed}:{source_snapshot_id}".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < ratios[0]:
        return "train"
    if unit < ratios[0] + ratios[1]:
        return "dev"
    return "test"


def validate_reward_separation(record: TransitionRecord) -> list[str]:
    """Reject attempts to inject reward/teacher labels into canonical state."""

    failures: list[str] = []
    forbidden = {"reward", "reward_vector", "teacher_id", "teacher", "preference", "hidden_state"}
    names = set(observation_field_names())
    if any(token in " ".join(names).lower() for token in forbidden):
        failures.append("canonical observation names contain forbidden metadata")
    if not isinstance(record.reward_vector, dict):
        failures.append("reward_vector must be a separate dict")
    if "teacher_id" in record.to_dict():
        failures.append("teacher_id must not be a transition field")
    return failures
