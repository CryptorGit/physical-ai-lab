"""Serializable, branchable Qmini snapshots for counterfactual rollouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import base64
import copy
import pickle
from typing import Any, Callable, Iterable


_PICKLE_PREFIX = "pickle-base64:"


def _pack_state(value: Any) -> str:
    try:
        encoded = base64.b64encode(pickle.dumps(value)).decode("ascii")
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise TypeError("snapshot state must be pickle-serializable") from exc
    return _PICKLE_PREFIX + encoded


def _unpack_state(value: Any) -> Any:
    if not isinstance(value, str) or not value.startswith(_PICKLE_PREFIX):
        return value
    return pickle.loads(base64.b64decode(value[len(_PICKLE_PREFIX):]))


def _tuple(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must have length {size}")
    return result


@dataclass
class QminiSnapshot:
    root_pose: tuple[float, ...]
    root_velocity: tuple[float, ...]
    joint_q: tuple[float, ...]
    joint_dq: tuple[float, ...]
    actuator_controller_state: dict[str, Any]
    previous_action: tuple[float, ...]
    current_command: tuple[float, ...]
    contact_related_state: dict[str, Any]
    friction: float
    wind_xy: tuple[float, float]
    fatigue_left: tuple[float, ...]
    fatigue_right: tuple[float, ...]
    rng_state: object
    episode_time: float
    recurrent_state: Any = None
    backend_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.root_pose = tuple(float(value) for value in self.root_pose)
        self.root_velocity = tuple(float(value) for value in self.root_velocity)
        self.joint_q = _tuple(self.joint_q, 10, "joint_q")
        self.joint_dq = _tuple(self.joint_dq, 10, "joint_dq")
        self.previous_action = _tuple(self.previous_action, 10, "previous_action")
        self.current_command = tuple(float(value) for value in self.current_command)
        self.wind_xy = _tuple(self.wind_xy, 2, "wind_xy")
        self.fatigue_left = _tuple(self.fatigue_left, 5, "fatigue_left")
        self.fatigue_right = _tuple(self.fatigue_right, 5, "fatigue_right")

    def clone(self) -> "QminiSnapshot":
        return copy.deepcopy(self)

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rng_state"] = _pack_state(self.rng_state)
        if self.recurrent_state is not None:
            payload["recurrent_state"] = _pack_state(self.recurrent_state)
        if self.backend_state is not None:
            payload["backend_state"] = _pack_state(self.backend_state)
        return payload

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "QminiSnapshot":
        values = dict(payload)
        values["rng_state"] = _unpack_state(values["rng_state"])
        values["recurrent_state"] = _unpack_state(values.get("recurrent_state"))
        values["backend_state"] = _unpack_state(values.get("backend_state"))
        return cls(**values)


def deterministic_branch_replay(
    snapshot: QminiSnapshot,
    action: Iterable[float],
    *,
    step_fn: Callable[[QminiSnapshot, tuple[float, ...]], QminiSnapshot],
) -> tuple[QminiSnapshot, QminiSnapshot]:
    """Run two independent clones and assert exact deterministic equality."""

    action_tuple = tuple(float(value) for value in action)
    first = step_fn(snapshot.clone(), action_tuple)
    second = step_fn(snapshot.clone(), action_tuple)
    if first.to_jsonable() != second.to_jsonable():
        raise AssertionError("same snapshot + same action + same RNG did not replay exactly")
    return first, second
