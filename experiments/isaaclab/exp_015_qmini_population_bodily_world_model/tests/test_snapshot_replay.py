from __future__ import annotations

import random

from qmini_population_bwm.snapshot_clone import QminiSnapshot, deterministic_branch_replay


def make_snapshot() -> QminiSnapshot:
    rng = random.Random(4)
    return QminiSnapshot(
        root_pose=(0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0),
        root_velocity=(0.0,) * 6,
        joint_q=(0.0,) * 10,
        joint_dq=(0.0,) * 10,
        actuator_controller_state={"last": [0.0] * 10},
        previous_action=(0.0,) * 10,
        current_command=(0.2, 0.0, 0.0),
        contact_related_state={"left": 0, "right": 0},
        friction=0.9,
        wind_xy=(0.1, -0.2),
        fatigue_left=(0.0,) * 5,
        fatigue_right=(0.0,) * 5,
        rng_state=rng.getstate(),
        episode_time=0.0,
        recurrent_state={"h": [0.0]},
    )


def test_same_snapshot_action_rng_replays_exactly() -> None:
    def step(snapshot: QminiSnapshot, action: tuple[float, ...]) -> QminiSnapshot:
        local = random.Random()
        local.setstate(snapshot.rng_state)
        noise = local.random()
        snapshot.joint_q = tuple(value + command + noise for value, command in zip(snapshot.joint_q, action, strict=True))
        snapshot.rng_state = local.getstate()
        snapshot.episode_time += 0.015
        return snapshot

    first, second = deterministic_branch_replay(make_snapshot(), (0.01,) * 10, step_fn=step)
    assert first.to_jsonable() == second.to_jsonable()


def test_snapshot_json_roundtrip_keeps_controller_and_hidden_state() -> None:
    snapshot = make_snapshot()
    restored = QminiSnapshot.from_jsonable(snapshot.to_jsonable())
    assert restored.to_jsonable() == snapshot.to_jsonable()
    assert restored.recurrent_state == {"h": [0.0]}
    assert restored.friction == snapshot.friction
