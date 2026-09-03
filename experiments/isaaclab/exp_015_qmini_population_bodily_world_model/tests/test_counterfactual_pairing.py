from __future__ import annotations

from qmini_population_bwm.crossed_interventions import MacroAction, cross_snapshot
from qmini_population_bwm.snapshot_clone import QminiSnapshot


def make_snapshot() -> QminiSnapshot:
    return QminiSnapshot(
        root_pose=(0.0, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0),
        root_velocity=(0.0,) * 6,
        joint_q=(0.0,) * 10,
        joint_dq=(0.0,) * 10,
        actuator_controller_state={},
        previous_action=(0.0,) * 10,
        current_command=(0.2, 0.0, 0.0),
        contact_related_state={"left": False, "right": False},
        friction=0.8,
        wind_xy=(0.0, 0.0),
        fatigue_left=(0.0,) * 5,
        fatigue_right=(0.0,) * 5,
        rng_state=(3, (1,) * 624, None),
        episode_time=0.0,
    )


def test_crossed_interventions_share_one_snapshot_and_are_paired() -> None:
    macros = (
        MacroAction("a", (0.0,) * 10, provenance="TEST"),
        MacroAction("b", (0.1,) * 10, provenance="TEST"),
        MacroAction("c", (-0.1,) * 10, provenance="TEST"),
    )

    def step(snapshot: QminiSnapshot, action: tuple[float, ...]):
        snapshot.joint_q = tuple(value + command for value, command in zip(snapshot.joint_q, action, strict=True))
        return snapshot, {
            "progress": sum(snapshot.joint_q),
            "velocity": action[0],
            "velocity_error": abs(action[0]),
            "stable": True,
            "fell": False,
        }

    rows = cross_snapshot(make_snapshot(), macros, step_fn=step, horizons=(1, 5), source_snapshot_id="source-1")
    assert len(rows) == 6
    assert {row.source_snapshot_id for row in rows} == {"source-1"}
    assert {row.macro_id for row in rows} == {"a", "b", "c"}
    assert {row.horizon_policy_steps for row in rows} == {1, 5}
